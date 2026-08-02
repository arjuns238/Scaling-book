from flax import struct
import jax.numpy as jnp
@struct.dataclass
class Config:
    d_model: int
    ffw_multiplier: int
    num_layers: int
    query_heads: int
    kv_heads: int
    key_dim: int
    vocab_size: int
    batch_size: int
    num_experts: int
    top_k: int
    dtype: jnp.dtype = jnp.bfloat16
    lr: float = 1e-3
    num_epochs: int = 10
    max_seq_len: int = 16
    lb_factor: float = 0.01