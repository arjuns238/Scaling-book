from collections import defaultdict
from inference import generate
from data import *

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

def evaluate_accuracy(tokens, masks, weights, cfg, verbose=True, collect_wrongs = False):
    """
    Measure exact-match addition accuracy over a tokenized dataset.

    tokens, masks: [N, max_seq_len] int arrays (as returned by build_dataset)
    Returns overall accuracy (float) and a per-bucket dict.
    Buckets key on (max operand digit-length, carry count).
    """
    correct = defaultdict(int)
    total   = defaultdict(int)
    n_correct = 0
    N = len(tokens)
    wrongs = []

    for row in tokens:
        # 1. Decode the row back to a string, strip BOS/EOS/PAD.
        chars = []
        for tok in row:
            tok = int(tok)
            s = itos[tok]
            if s == BOS or s == PAD:
                continue
            if s == EOS:
                break
            chars.append(s)
        text = "".join(chars)          # e.g. "7+9=61"  (answer is reversed)

        # 2. Split into prompt (a+b=) and the stored answer.
        prompt_part, ans_rev = text.split("=")
        a_str, b_str = prompt_part.split("+")
        a, b = int(a_str), int(b_str)
        true_answer = ''.join([c for c in f"{a+b}"])   # reversed, as generate emits

        # 3. Regenerate from the prompt and compare.
        pred = generate(f"{a}+{b}", weights, max_new_tokens=cfg.max_seq_len, cfg=cfg)
        is_correct = (pred == true_answer)

        if collect_wrongs and not is_correct:
            wrongs.append({
                "a": a, "b": b,
                "true": true_answer, "pred": pred,
            })

        n_correct += int(is_correct)
        key = (max(len(a_str), len(b_str)), _count_carries(a, b))
        total[key]   += 1
        correct[key] += int(is_correct)

    overall = n_correct / N

    if verbose:
        print(f"Overall: {n_correct}/{N} = {overall:.4f}\n")
        print(f"{'digits':>6} {'carries':>7} {'acc':>8}  count")
        for key in sorted(total):
            d, c = key
            acc = correct[key] / total[key]
            print(f"{d:>6} {c:>7} {acc:>8.3f}  {total[key]}")

    if collect_wrongs:
        return overall, {k: correct[k] / total[k] for k in total}, wrongs

    return overall, {k: correct[k] / total[k] for k in total}

def count_params(weights):
    return sum(x.size for x in jax.tree_util.leaves(weights))

def make_cfg(base, d_model, num_layers):
    heads = max(1, d_model // 64)
    return base.replace(d_model=d_model, num_layers=num_layers,
                        query_heads=heads, kv_heads=heads)

