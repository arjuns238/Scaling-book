from config import Config
import jax.numpy as jnp
from utils import _num_digits, _count_carries

chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '=', ' ', '<BOS>', '<EOS>', '<PAD>']
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
PAD, BOS, EOS = '<PAD>', '<BOS>', '<EOS>'

def create_example(a, b, max_len):
    prompt = f"{a}+{b}="
    answer = f"{a+b}"[::-1]
    full = prompt + answer
    loss_mask = [0] * (len(prompt) + 1) + [1] * (len(answer) + 1) # adding 1 to prompt for bos token and 1 to answer for eos token
    tokens = [stoi[BOS]] + [stoi[c] for c in full] + [stoi[EOS]]
    pad_len = max_len - len(tokens)
    tokens    += [stoi[PAD]] * pad_len
    loss_mask += [0]      * pad_len
    return tokens, loss_mask

import numpy as np
MAX_N = 1000 # digits 0 -> 999
max_len = 14
def build_dataset(cfg: Config, max_n=MAX_N):
    N = max_n * max_n
    tokens = np.empty((N, max_len), dtype = np.int32)
    masks = np.empty((N, max_len), dtype = np.int32)

    idx = 0
    for i in range(max_n):
        for j in range(max_n):
            t, m = create_example(i, j, max_len)
            tokens[idx] = t
            masks[idx] = m
            idx += 1

    return tokens, masks

def generate_split(tokens_all, masks_all, split = 0.9, seed = 0):
    rng = np.random.default_rng(seed=seed)
    indices = rng.permutation(len(tokens_all))
    # 2. Calculate split cut-off points (e.g., 70% train, 15% val, 15% test)
    train_end = int(split * len(tokens_all))

    # 3. Split the indices into 3 parts
    train_idx, val_idx = np.split(indices, [train_end])

    train_dataset, val_dataset = tokens_all[train_idx], tokens_all[val_idx]
    train_masks, val_masks = masks_all[train_idx], masks_all[val_idx]

    return train_dataset, val_dataset, train_masks, val_masks

# Generate a lite dataloader for this small dataset for easy data handling
class Dataloader:
    def __init__(self, dataset, masks, batch_size, shuffle=True):
        self.dataset = dataset
        self.masks = masks
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __iter__(self):
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indices)
        for i in range(0, len(self.dataset) - self.batch_size + 1, self.batch_size):
            batch_indices = indices[i:i+self.batch_size]
            batch_dataset = self.dataset[batch_indices]
            batch_masks = self.masks[batch_indices]
            yield jnp.asarray(batch_dataset), jnp.asarray(batch_masks)

def build_dataset_research_backed(cfg, max_n=MAX_N, val_frac=0.1, target_3to2_ratio=10, seed=0):
    '''
    Strategy (following Lee et al. 2023 / ENTP):
    1. All both 1-digit pairs go entirely to train (fundamentals so no holdout)
    2. The 3-digit dominant bucket is downsampled toward target_3to2_ratio so it doesnt drown everything else
    3. Everything remaining is split train/val, stratified by (max operand digit length, carry count)
    '''
    rng = np.random.default_rng(seed=seed)

    # 1. Enumerate every pair once, bucketed by a difficulty key
    # key = (max_digit_len, carry_count)
    fundamentals = []
    buckets = {}

    for a in range(max_n):
        for b in range(max_n):
            da, db = _num_digits(a), _num_digits(b)
            max_len = max(da, db)
            if da == 1 and db == 1:
                fundamentals.append((a, b))
                continue
            key = (max_len, _count_carries(a, b))
            buckets.setdefault(key, []).append((a, b))

    # 2. downsamaple the 3 digit to target ratio
    n_3digit = sum(len(v) for k, v in buckets.items() if k[0] == 3)
    n_2digit = sum(len(v) for k, v in buckets.items() if k[0] == 2)
    keep_frac_3 = min(1.0, (target_3to2_ratio * n_2digit) / max(n_3digit, 1))

    for key in list(buckets.keys()):
        if key[0] == 3 and keep_frac_3 < 1.0:
            pairs = buckets[key]
            k = max(1, int(round(len(pairs) * keep_frac_3)))
            idx = rng.choice(len(pairs), k, replace=False)
            buckets[key] = [pairs[i] for i in idx]

    # 3. stratified train/val split within each bucket

    train_pairs = list(fundamentals)
    val_pairs = []
    for key, pairs in buckets.items():
        pairs = list(pairs)
        rng.shuffle(pairs)
        n_val = int(round(len(pairs) * val_frac))
        val_pairs.extend(pairs[:n_val])
        train_pairs.extend(pairs[n_val:])

    rng.shuffle(train_pairs)
    rng.shuffle(val_pairs)

    # 4. Convert to tokens

    def _materialize(pairs):
        n = len(pairs)
        tokens = np.empty((n, cfg.max_seq_len), dtype = np.int32)
        masks = np.empty((n, cfg.max_seq_len), dtype = np.int32)

        for i, (a, b) in enumerate(pairs):
            t, m = create_example(a, b, cfg.max_seq_len)
            tokens[i] = t
            masks[i] = m
        return tokens, masks

    train_dataset, train_masks = _materialize(train_pairs)
    val_dataset, val_masks = _materialize(val_pairs)

    return train_dataset, train_masks, val_dataset, val_masks