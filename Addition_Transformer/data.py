from config import Config
import jax.numpy as jnp
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

