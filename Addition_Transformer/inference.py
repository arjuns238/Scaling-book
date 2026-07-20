import jax
import jax.numpy as jnp
import numpy as np

from model import forward
from data import stoi, itos, BOS, PAD, EOS


def loop_body(i, carrys):
    buf, curr_len, weights = carrys
    x = jnp.expand_dims(buf, axis=0)
    logits = forward(x, weights)
    next_token_logits = jax.lax.dynamic_slice_in_dim(logits[0], curr_len - 1, 1, axis=0)
    next_token = jnp.argmax(next_token_logits)
    new_buf = buf.at[curr_len].set(next_token)
    return (new_buf, curr_len + 1, weights)


@jax.jit
def generate_core(buf, start_answer_idx, max_new_tokens, weights):
    final_buf, final_curr_len, _ = jax.lax.fori_loop(0, max_new_tokens, loop_body, (buf, start_answer_idx, weights))
    return final_buf, final_curr_len


def generate(prompt, weights, max_new_tokens, cfg):
    prompt_tokens = [stoi[BOS]] + [stoi[c] for c in prompt] + [stoi['=']]
    curr_len = len(prompt_tokens)
    start_answer_idx = len(prompt_tokens)

    buf = np.asarray([stoi[PAD]] * cfg.max_seq_len)
    buf[:curr_len] = prompt_tokens

    final_buf, final_curr_len = generate_core(buf, start_answer_idx, max_new_tokens, weights)
    generation = np.asarray(final_buf)

    # Finding right end idx since jax loops cannot be stopped with an if statement
    ans_end_idx = start_answer_idx
    for token in generation[start_answer_idx:final_curr_len]:
        if token == stoi['<EOS>']:
            break
        ans_end_idx += 1

    return ''.join([itos[i] for i in generation[start_answer_idx:ans_end_idx]])[::-1]  # reversing cuz training objective is reversed
