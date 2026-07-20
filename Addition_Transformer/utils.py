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
    return sum(x.size for x in jax.tree_util.leaves(weights))

def make_cfg(base, d_model, num_layers):
    heads = max(1, d_model // 64)
    return base.replace(d_model=d_model, num_layers=num_layers,
                        query_heads=heads, kv_heads=heads)

