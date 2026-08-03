import jax

def _num_digits(n):
    return len(str(n))

def _count_carries(a, b):
    carries = 0
    carry = 0
    while a > 0 or b > 0:
        s = (a % 10) + (b % 10) + carry
        carry = 1 if s >= 10 else 0
        carries += carry
        a //= 10
        b //= 10
    return carries

def count_params(weights):
    return sum(x.size for x in jax.tree_util.tree_leaves(weights))


def active_params(cfg):
    '''
    N for the 6ND compute model: params touched by a single token.

        ffw    3LDFK    3 matrices per swiglu expert (gate, up, down), K experts active
        attn   4DHL     q, k, v, proj - H is heads * key_dim
        embed  2DV      embedding tied with the output head

    Dense is the K=1 case: every token uses the one ffw. For moe only top_k of the
    num_experts experts run per token, so the expert weights enter N multiplied by
    top_k, not by num_experts - the idle experts cost memory, not flops.

    Excluded, as in the dense accounting: rms norm gammas (2DL + D) and the moe
    router (LDE). Both are <0.1% here and neither scales with the ladder.
    '''
    D = cfg.d_model
    F = int(cfg.d_model * cfg.ffw_multiplier)
    L = cfg.num_layers
    H = cfg.query_heads * cfg.key_dim
    V = cfg.vocab_size
    K = cfg.top_k if cfg.num_experts > 0 else 1

    ffw   = 3 * L * D * F * K
    attn  = 4 * D * H * L
    embed = 2 * D * V
    return ffw + attn + embed


def check_param_formula(cfg, rng=None):
    '''
    Sanity check on active_params: for a dense cfg every param is active, so the
    formula should land on the real pytree count up to the excluded norm/embed terms.
    Returns (formula, actual, delta) so drift shows up instead of being assumed away.
    '''
    from model import Weights
    if cfg.num_experts > 0:
        raise ValueError("only meaningful for a dense cfg - moe total != active by design")
    w = Weights.init(cfg, rng if rng is not None else jax.random.key(0))
    actual = count_params(w)
    formula = active_params(cfg)
    return formula, actual, actual - formula

def make_cfg(base, d_model, num_layers):
    heads = max(1, d_model // 64)
    return base.replace(d_model=d_model, num_layers=num_layers,
                        query_heads=heads, kv_heads=heads)

