import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from model import forward
from data import stoi, itos, BOS, PAD, EOS

from utils import _count_carries
from collections import defaultdict


def make_loop_body(cfg):
    # cfg is closed over so it stays static under jit - moe needs num_experts/top_k
    # as python ints (jnp.bincount(length=E), lax.top_k(k=K))
    def loop_body(i, carrys):
        buf, curr_len, weights = carrys
        x = jnp.expand_dims(buf, axis=0)
        logits, _aux, _stats = forward(x, weights, cfg) # aux/stats are training-only
        next_token_logits = jax.lax.dynamic_slice_in_dim(logits[0], curr_len - 1, 1, axis=0)
        next_token = jnp.argmax(next_token_logits)
        new_buf = buf.at[curr_len].set(next_token)
        return (new_buf, curr_len + 1, weights)
    return loop_body


@partial(jax.jit, static_argnames=("max_new_tokens", "cfg"))
def generate_core(buf, start_answer_idx, max_new_tokens, weights, cfg):
    final_buf, final_curr_len, _ = jax.lax.fori_loop(
        0, max_new_tokens, make_loop_body(cfg), (buf, start_answer_idx, weights))
    return final_buf, final_curr_len


def generate(prompt, weights, max_new_tokens, cfg):
    prompt_tokens = [stoi[BOS]] + [stoi[c] for c in prompt] + [stoi['=']]
    curr_len = len(prompt_tokens)
    start_answer_idx = len(prompt_tokens)

    buf = np.asarray([stoi[PAD]] * cfg.max_seq_len)
    buf[:curr_len] = prompt_tokens

    final_buf, final_curr_len = generate_core(buf, start_answer_idx, max_new_tokens, weights, cfg)
    generation = np.asarray(final_buf)

    # Finding right end idx since jax loops cannot be stopped with an if statement
    ans_end_idx = start_answer_idx
    for token in generation[start_answer_idx:final_curr_len]:
        if token == stoi['<EOS>']:
            break
        ans_end_idx += 1

    return ''.join([itos[i] for i in generation[start_answer_idx:ans_end_idx]])[::-1]  # reversing cuz training objective is reversed

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

