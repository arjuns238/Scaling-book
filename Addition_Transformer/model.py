import jax
from typing_extensions import final
from flax import struct
from jax.nn.initializers import he_normal, normal
import jax.numpy as jnp
from config import Config
from data import stoi, PAD

@struct.dataclass
class Layer:
    q: jax.Array
    k: jax.Array
    v: jax.Array
    proj: jax.Array
    w1: jax.Array | None      # dense FFW, None when cfg.num_experts > 0
    w2: jax.Array | None      # dense FFW, None when cfg.num_experts > 0
    router: jax.Array | None  # MoE, None when cfg.num_experts == 0
    win: jax.Array | None     # MoE, None when cfg.num_experts == 0
    wout: jax.Array           # (F,D) dense, (E,F,D) MoE
    gamma1: jax.Array # for rms norm
    gamma2: jax.Array # for rms norm

    @classmethod
    def init(cls, cfg: Config, rng):
        q_rng, k_rng, v_rng, proj_rng, w1_rng, w2_rng, wout_rng = jax.random.split(rng, 7)
        d_ff = int(cfg.d_model * cfg.ffw_multiplier)
        moe = cfg.num_experts > 0
        return cls(
            q = he_normal(in_axis=0, out_axis=(1, 2))(q_rng, (cfg.d_model, cfg.query_heads, cfg.key_dim), dtype=cfg.dtype),
            k = he_normal(in_axis=0, out_axis=(1, 2))(k_rng, (cfg.d_model, cfg.kv_heads, cfg.key_dim), dtype=cfg.dtype),
            v = he_normal(in_axis=0, out_axis=(1, 2))(v_rng, (cfg.d_model, cfg.kv_heads, cfg.key_dim), dtype=cfg.dtype),
            proj = he_normal(in_axis=0, out_axis=(1, 2))(proj_rng, (cfg.d_model, cfg.query_heads, cfg.key_dim), dtype=cfg.dtype),
            w1 = None if moe else he_normal(in_axis=0, out_axis=1)(w1_rng, (cfg.d_model, d_ff), dtype=cfg.dtype),
            w2 = None if moe else he_normal(in_axis=0, out_axis=1)(w2_rng, (cfg.d_model, d_ff), dtype=cfg.dtype),
            # w1_rng/w2_rng are reused for the MoE params so the split count stays at 7 either way
            router = normal(stddev = 0.02)(w1_rng, (cfg.d_model, cfg.num_experts), dtype=jnp.float32) if moe else None,
            win = normal(stddev = 0.02)(w2_rng, (cfg.num_experts, cfg.d_model, 2 * d_ff), dtype=cfg.dtype) if moe else None, # (E,D,2F) coupling both gate and up projection for swiglu
            wout = (he_normal(in_axis=1, out_axis=2, batch_axis=0)(wout_rng, (cfg.num_experts, d_ff, cfg.d_model), dtype=cfg.dtype) if moe
                    else he_normal(in_axis=0, out_axis=1)(wout_rng, (d_ff, cfg.d_model), dtype=cfg.dtype)),
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
            layers = [Layer.init(cfg, r) for r in layer_rngs[2:]],
            embedding = normal()(embed_rng, (cfg.vocab_size, cfg.d_model), dtype=cfg.dtype),
            pos_embed = normal()(pos_embed_rng, (cfg.max_seq_len, cfg.d_model), dtype=cfg.dtype),
            final_gamma = jnp.ones((cfg.d_model,), dtype=cfg.dtype)

        )


def rms_norm(x: jax.Array, gamma: jax.Array, eps = 1e-6) -> jax.Array:
    rms = jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)
    return gamma * rms * x

def router_aux(logits, expert_ids, valid, E, K):
    m = logits.shape[0]
    # valid represents the valid tokens (anything that is not a pad token is excluded - masked out)
    vf = valid.reshape(m).astype(jnp.float32)
    n = vf.sum()
    # P is the mean routing probability allocated to expert i across all tokens in the batch
    # So P is what the router intended - mean of softmax of router logits
    P = (jax.nn.softmax(logits, -1) * vf[:, None]).sum(0) / (n * K)
    vk = jnp.repeat(vf, K)
    # f is the fraction of tokens dispatched to expert i.
    # f is what actually happened - mean of the tokens dispatched to each expert
    f = (jax.nn.one_hot(expert_ids, E, dtype = jnp.float32) * vk[:, None]).sum(0) / (n * K)
    return E * jnp.sum(f * P)


def moe(ffw_in: jax.Array, valid, cfg: Config, layer: Layer):
    '''
    rough moe algorithm:
        moe
        router - (D, E)
        in - (B,T,D)
        logits - (B,T,E)
        top k - experts, indices = B,T,K and B,T,K
        softmax - B,T,K, B,T,K
        flatten experts - (B*T*K)
        flatten indices - (B*T*K)
        sort experts
        token_ids = jnp.repeat(jnp.arange(B*T), K) repeats B*T, k times one for each expert
        token_idx = token_ids[sort_idx] B*T*K
        X_flat = flatten(x, B*T, D)
        X sorted = X_flat[token_idx] -> (B*T*K, D)
        group sizes = jnp bincount (expert ids, length = E) (E,)
        ragged moe with X_sorted (B*T*K, D), Win(E,D,2F), group size (E) -> (B*T*K, 2F)
        gate, up = split(up projs, 2, axis=-1)             [B*T*K, F] each
        a = silu(gate) * up                      [B*T*K, F]
        out_sorted = ragged_dot(a[B*T*K, F], wout[E,F,D], group_sizes)   [B*T*K, D]
        multiply with router weight
        add k expert weight
    '''
    B, T, D = ffw_in.shape
    E, K = cfg.num_experts, cfg.top_k
    m = B * T

    x_flat = ffw_in.reshape(m, D)
    logits = jnp.einsum("md,de->me", x_flat, layer.router)
    top_logits, idx = jax.lax.top_k(logits, k=K) # (m, K)
    combine = jax.nn.softmax(top_logits, axis=-1) # (m, K), sums to 1 over K

    combine_flat = combine.reshape(m * K) # (m * K)
    expert_ids = idx.reshape(m * K)       # (m * K)

    sort_idx = jnp.argsort(expert_ids)
    token_ids = jnp.repeat(jnp.arange(m), K) # (m * K) repeats m tokens for each K
    token_idx = token_ids[sort_idx]
    w_sorted = combine_flat[sort_idx] # router weight - (m*K)
    x_sorted = x_flat[token_idx]             # (m*K, D) -> duplicate x as well
    group_sizes = jnp.bincount(expert_ids, length = E).astype(jnp.int32)

    h = jax.lax.ragged_dot(x_sorted, layer.win, group_sizes) # (m*k, D) @ (E,D,2F) -> (m*k, 2F)
    gate, up = jnp.split(h, 2, axis=-1) # splitting because the two gates were together in win
    a = jax.nn.silu(gate) * up
    out_sorted = jax.lax.ragged_dot(a, layer.wout, group_sizes) # (m*k, F) @ (E,F,D) -> (m*k, D)

    out_sorted = out_sorted * w_sorted[:, None] # (m*K, D) * (m*K, ) -> (m*K, D) - multiplying with router weights
    # accumulate the k contributions in fp32 (scatter-add of K terms per token), then hand the
    # residual stream back in cfg.dtype so the moe path matches the dense one
    out_flat = jnp.zeros((m, D), jnp.float32).at[token_idx].add(out_sorted.astype(jnp.float32))
    out_flat = out_flat.astype(cfg.dtype)

    # return out flat reshape correctly and the aux loss.
    aux_loss = router_aux(logits, expert_ids, valid, E, K)
    return out_flat.reshape(B, T, D), aux_loss


def layer_forward(x: jax.Array, valid: jax.Array, cfg: Config, layer: Layer) -> jax.Array:
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

    # ffw - routed experts when cfg.num_experts > 0, else dense SwiGLU.
    # layer.router is None for a dense layer, so this branch resolves at trace time and is free under jit
    if layer.router is None:
        # ffw - SwiGLU
        up_proj = jnp.einsum("btd,df->btf", ffw_in, layer.w1)
        silu_out = jax.nn.silu(up_proj)
        up_proj2 = jnp.einsum("btd,df->btf", ffw_in, layer.w2)

        ffw_out = up_proj2 * silu_out
        down_proj = jnp.einsum("btf,fd->btd", ffw_out, layer.wout)
        aux_loss = jnp.zeros((), dtype=jnp.float32) # no router to balance
    else:
        # moe
        down_proj, aux_loss = moe(ffw_in, valid, cfg, layer)

    # residual connection
    return down_proj + attn_out, aux_loss

def forward(token_ids: jax.Array, weights: Weights, cfg: Config) -> jax.Array:
    # valid mask - required for moe loss calculation - anything except the pad ids
    valid = jnp.not_equal(token_ids, stoi[PAD])
    # embed tokens - [B, T] -> [B, T, D]
    positions = jnp.arange(token_ids.shape[1], dtype=jnp.int32)
    x = weights.embedding[token_ids, :] + weights.pos_embed[positions] # [B, T, D] + [T, D] = [B, T, D]
    aux_losses = []
    for layer in weights.layers:
        x, aux_loss = layer_forward(x, valid, cfg, layer)
        aux_losses.append(aux_loss)
    # final rms norm
    x = rms_norm(x, weights.final_gamma)
    # final embedding
    return jnp.einsum("btd,vd->btv", x, weights.embedding), jnp.stack(aux_losses).mean()
