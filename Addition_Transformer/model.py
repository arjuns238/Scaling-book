import jax
from typing_extensions import final
from flax import struct
from jax.nn.initializers import he_normal, normal
import jax.numpy as jnp
from config import Config

@struct.dataclass
class Layer:
    q: jax.Array
    k: jax.Array
    v: jax.Array
    proj: jax.Array
    w1: jax.Array
    w2: jax.Array
    wout: jax.Array
    gamma1: jax.Array # for rms norm
    gamma2: jax.Array # for rms norm

    @classmethod
    def init(cls, cfg: Config, rng):
        q_rng, k_rng, v_rng, proj_rng, w1_rng, w2_rng, wout_rng = jax.random.split(rng, 7)
        return cls(
            q = he_normal(in_axis=0, out_axis=(1, 2))(q_rng, (cfg.d_model, cfg.query_heads, cfg.key_dim), dtype=cfg.dtype),
            k = he_normal(in_axis=0, out_axis=(1, 2))(k_rng, (cfg.d_model, cfg.kv_heads, cfg.key_dim), dtype=cfg.dtype),
            v = he_normal(in_axis=0, out_axis=(1, 2))(v_rng, (cfg.d_model, cfg.kv_heads, cfg.key_dim), dtype=cfg.dtype),
            proj = he_normal(in_axis=0, out_axis=(1, 2))(proj_rng, (cfg.d_model, cfg.query_heads, cfg.key_dim), dtype=cfg.dtype),
            w1 = he_normal(in_axis=0, out_axis=1)(w1_rng, (cfg.d_model, int(cfg.d_model * cfg.ffw_multiplier)), dtype=cfg.dtype),
            w2 = he_normal(in_axis=0, out_axis=1)(w2_rng, (cfg.d_model, int(cfg.d_model * cfg.ffw_multiplier)), dtype=cfg.dtype),
            wout = he_normal(in_axis=0, out_axis=1)(wout_rng, (int(cfg.d_model * cfg.ffw_multiplier), cfg.d_model), dtype=cfg.dtype),
            gamma1 = jnp.ones(cfg.d_model, dtype=cfg.dtype),
            gamma2 = jnp.ones(cfg.d_model, dtype=cfg.dtype)
        )

@struct.dataclass
class Weights:
    layers: list[Layer]
    embedding: jax.Array
    pos_embed: jax.Array
    final_gamma: jax.Array

    @classmethod
    def init(cls, cfg: Config, rng):
        layer_rngs = jax.random.split(rng, cfg.num_layers + 2) # plus two for vocab and pos embeddings
        embed_rng = layer_rngs[0]
        pos_embed_rng = layer_rngs[1]
        return cls(
            layers = [Layer.init(cfg, rng) for r in layer_rngs[2:]],
            embedding = normal()(embed_rng, (cfg.vocab_size, cfg.d_model), dtype=cfg.dtype),
            pos_embed = normal()(pos_embed_rng, (cfg.max_seq_len, cfg.d_model), dtype=cfg.dtype),
            final_gamma = jnp.ones((cfg.d_model,), dtype=cfg.dtype)

        )


def rms_norm(x: jax.Array, gamma: jax.Array, eps = 1e-6) -> jax.Array:
    rms = jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)
    return gamma * rms * x

def layer_forward(x: jax.Array, layer: Layer) -> jax.Array:
    # pre attn rms norm
    attn_in = rms_norm(x, layer.gamma1)

    # qkv linear proj
    q_proj = jnp.einsum("btd,dnh->btnh", attn_in, layer.q)
    k_proj = jnp.einsum("btd,dkh->btkh", attn_in, layer.k)
    v_proj = jnp.einsum("btd,dkh->btkh", attn_in, layer.v)

    # scaled dot product attn - standard mha so n == k. also s == t since self attn
    scores = jnp.einsum("btnh,bsnh->bnts", q_proj, k_proj) / jnp.sqrt(q_proj.shape[-1])

    # mask + softmax for causal attn
    mask = jnp.tril(jnp.ones((scores.shape[-2], scores.shape[-1]), dtype = bool)) # need to mask to -inf to ensure softmax is 0. shape = (T, T)
    masked_scores = jnp.where(mask[None, :], scores, -float('inf')) # broadcast to (B, N, T, T)
    attn_weights = jax.nn.softmax(masked_scores, axis=-1)

    # multiply with V
    attn_out = jnp.einsum("bnts,bsnh->btnh", attn_weights, v_proj)

    # attn projection
    attn_out = jnp.einsum("btnh,dnh->btd", attn_out, layer.proj)

    # residual connection
    attn_out = attn_out + x

    # pre ffw rms norm
    ffw_in = rms_norm(attn_out, layer.gamma2) # (B, T, D)

    # ffw - SwiGLU
    up_proj = jnp.einsum("btd,df->btf", ffw_in, layer.w1)
    silu_out = jax.nn.silu(up_proj)
    up_proj2 = jnp.einsum("btd,df->btf", ffw_in, layer.w2)

    ffw_out = up_proj2 * silu_out
    down_proj = jnp.einsum("btf,fd->btd", ffw_out, layer.wout)

    # residual connection
    return down_proj + attn_out

def forward(token_ids: jax.Array, weights: Weights) -> jax.Array:
    # embed tokens - [B, T] -> [B, T, D]
    positions = jnp.arange(token_ids.shape[1], dtype=jnp.int32)
    x = weights.embedding[token_ids, :] + weights.pos_embed[positions] # [B, T, D] + [T, D] = [B, T, D]
    for layer in weights.layers:
        x = layer_forward(x, layer)
    x = rms_norm(x, weights.final_gamma)
    # final embedding
    return jnp.einsum("btd,vd->btv", x, weights.embedding)
