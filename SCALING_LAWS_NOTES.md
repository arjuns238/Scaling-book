# Scaling laws on 3-digit addition: what we found and how we got there

A record of the dense and MoE scaling-law experiments in this repo — the result,
the reasoning that produced it, and the wrong turns, which were instructive.

**Headline:** 3-digit addition does not have a Chinchilla-style scaling law in the
range we can train. It has a *capability threshold* at roughly 600k active
parameters. Below it, models fail regardless of how much data they get; above it,
they solve the task almost immediately. The compute-optimal frontier we set out to
measure does not exist here, and we can say why with data rather than assertion.

---

## 1. Setup

**Task.** Reversed-answer 3-digit addition (`123+456=975`), ~999,900 non-trivial
operand pairs, ~4.5 supervised tokens per example. Both-single-digit pairs are
excluded as fundamentals. Train pool after the val carve: **4,394,147 supervised
tokens**.

**Model.** Decoder-only transformer: MHA, SwiGLU FFW, RMSNorm, learned position
embeddings, tied input/output embedding. The MoE variant replaces the dense FFW
with top-k routed experts (`jax.lax.ragged_dot`, sorted dispatch, load-balance aux
loss). Dense and MoE are the same code path, selected by `cfg.num_experts == 0`.

**N is active parameters**, not total:

```
N = 3·L·D·F·K   +   4·D·H·L   +   2·D·V
    ffw (K=top_k)    attn (H=heads·key_dim)   tied embed
```

Dense `(d_model=384, L=6)` = **10,629,120**. This matters: at `E=8, k=2` the MoE
has 60,197,760 total parameters but 17,707,008 active — **5.7×** apart. Using total
params in `6ND` would have made the MoE and dense arms incomparable.

---

## 2. The result

### 2.1 There is a capability threshold near 600k active params

The decisive measurement is an isoFLOP profile at `C = 1.8e13`, where two models
were given the *same compute*, split differently:

| N | D (sup tokens) | epochs | val loss |
|--:|--:|--:|--:|
| 298,656 | 10,045,002 | 2.3 | ~1.5 |
| **594,432** | **4,300,000** | 1.0 | **0.027** |
| 1,777,344 | 1,850,000 | 0.4 | 0.042 |
| 3,940,608 | 795,000 | 0.2 | 0.095 |

The 298k model got **2.3× more data** than the 594k model and still landed at the
guessing plateau — 55× worse loss. That is not a model slightly short on capacity
trading against data. It is a model that cannot learn the algorithm at all.

Reference points for reading those numbers: uniform over the vocabulary is 2.77
nats, a unigram predictor is 2.27, and a *position-conditional* unigram — a model
that has learned the output format and digit frequencies but no arithmetic — is
**1.7883**. Everything at ~1.5 has learned formatting, not addition.

### 2.2 The three compute regimes

| band | losses | verdict |
|---|---|---|
| C = 1.8e12 | 0.58 → 1.6 | guessing plateau; `edge-low` |
| **C = 1.8e13** | **0.027 → 0.095** | **the only usable band; `interior` minimum at N=594,432** |
| C = 1.8e14 | 4e-4 → 6e-3 | solved; `edge-low` |

The entire interesting range of this task spans **about one decade of compute**. A
Chinchilla fit needs several decades of smooth power-law behaviour.

### 2.3 Below the threshold, data always wins

At `C = 1.8e12`, the *smallest* model (N=33,888, 8.85M tokens, 2 epochs) beat every
larger model in its band — 0.58 vs 1.2–1.6. At every budget below the threshold,
spending compute on data dominates spending it on parameters. Measured `D/N` ratios
across our grid ran from 16.3 down to 0.1; Chinchilla-optimal is ~20 tokens/param,
so most of the grid was badly data-starved.

---

## 3. Why the scaling law does not fit — the evidence, in order

### 3.1 The loss surface has two regimes in N, not one

Regressing `log L` on `log D` at fixed N across all 8 data rungs:

| group | weighted β | χ²/dof |
|---|--:|--:|
| N ≤ 298,656 | **0.15 ± 0.05** | 0.70 (consistent) |
| N ≥ 594,432 | **1.34 ± 0.25** | 0.24 (consistent) |

Each group is internally consistent with a single β, and the two groups differ by
9×. Below the threshold data barely helps (β≈0); above it, it helps enormously.
`E + A/N^α + B/D^β` describes one smooth regime — it cannot represent a switch.

Note also that β ≈ 1.34 is far steeper than the ~0.3 typical of language models.
The loss falls too fast for a power law to hold over any useful range.

### 3.2 Run-to-run noise exceeds the effect being measured

Residual scatter around those per-N regressions:

| N | resid sd (log-loss) | loss varies by |
|--:|--:|--:|
| 298,656 | 0.118 | ±1.13× |
| 594,432 | 0.847 | ±2.33× |
| 1,777,344 | 0.918 | ±2.50× |
| 3,940,608 | 0.508 | ±1.66× |

Identical configurations differ by up to 2.5× in loss. The cause is the same
threshold behaviour: whether a run escapes the plateau within its step budget is
partly stochastic, so cells near the threshold are bimodal (≈1.55 *or* ≈0.7, with
nothing between).

### 3.3 At that noise level the parameters are not recoverable — proven

We generated synthetic data from a **known-perfect Chinchilla surface**
(α=0.34, β=0.37), on our exact grid (4 N × 8 D), with our measured noise (sd 0.5),
and fitted it:

| parameterisation | α CI (true 0.34) |
|---|---|
| as written | [0.56, 16.73] |
| centered predictors | [−4.66, 18.08] |
| centered, E fixed to 0 | [−3.36, 18.63] |

No regime edges, no plateau, no saturation — and α is still unrecoverable. **The
problem is not the task, the ladders, the filter, or the fitter.** At this noise and
grid size these five parameters cannot be determined by anything.

For completeness, the fit on the real data returned
`A = 6.58e13 [17.8, 1.35e90]`, `α = 2.434 [0.499, 15.64]`,
`β = 3.585 [1.156, 16.2]` — a degenerate ridge where only the combination
`A·N^(−α)` is constrained, not `A` and `α` separately.

---

## 4. Bugs found along the way

### 4.1 The fitter was returning its own starting values (critical)

`fit_surface` builds `_predict_logL` in JAX and hands it to
`scipy.optimize.least_squares`, which has no analytic jacobian and finite-differences
it. Under JAX's **default float32**, scipy's ~1.5e-8 relative nudge falls below
float32 resolution for the amplitude parameters. Their computed gradient is exactly
zero, and `least_squares` returns them unchanged.

The original dense run reported `A = 0.5517`, `B = 135`. Both are **restart #3's
random starting draws**, matching to the printed precision. `α`, `β` and `E` moved
partially (they multiply `log N ≈ 12–17`, just large enough to register), so the
published `N_opt ~ C^0.4788` was two half-fitted exponents conditioned on two random
numbers. The `α` CI of `[0.2004, 6.996]` printed beside it was the tell.

**Fixed** by porting `_predict_logL` to numpy — float64 natively, so the trap cannot
recur. Identical results to 5 significant figures and **4.2× faster** (20.3s → 4.8s
per fit), since no autodiff was ever used and JAX was contributing only dispatch
overhead (~2,430 residual evaluations per fit at ~9.5 ms each, for arithmetic on 32
numbers).

The training runs were never affected. Only the curve fit on top of them.

### 4.2 Warmup could kill short runs

`warmup_steps = max(50, 0.05·total_steps)` made `decay_steps` negative below 50
steps, crashing those cells outright, and put 58% of an 86-step run into warmup.
Now `max(1, min(50, 0.05·total_steps))`.

### 4.3 Rows were never persisted

The original sweep's `(N, D, val_loss)` table existed only in memory, so re-fitting
required retraining. Sweeps now save to a timestamped JSON and copy to Drive.

---

## 5. Wrong turns

Worth recording, because each cost a sweep or a rebuild.

**Warmup as the explanation for the 1.55 plateau.** Warmup fraction correlated
beautifully with loss across the D ladder (58%, 23%, 10%, 5%, 5%, 5% vs 1.63, 1.55,
0.62, 0.14, 0.045, 0.048). But warmup fraction and loss *both* track D, so the
correlation was confounded. Fixing warmup changed the D=100k loss from 1.6321 to
1.6307 — nothing. The fix was still worth keeping for the crash, but the hypothesis
was wrong.

**Adding smaller models to fix α.** The reasoning was that models near the floor
give no N-signal, so smaller ones would help. They landed in the *guessing* regime
instead — β = 0.15, capacity-limited — and had to be filtered back out of the
surface fit. (They turned out to be essential for the isoFLOP profiles, as the left
arm of the U, but that was luck, not the plan.)

**A diagnostic too noisy to diagnose.** `local_beta` compared adjacent D rungs. With
1.32× spacing, `ln(D₂/D₁) = 0.278`, so noise of 0.2 in log-loss gives β error of
±1.0. It reported "NOT constant" everywhere, including on data that was fine.
Densifying the D ladder made it *worse*, because the signal per step shrank while
noise didn't. Regressing across all 8 rungs at once (±0.12 instead of ±1.0) is what
finally revealed the two-regime structure.

**Dismissing the 4-epoch idea.** When repeats were first proposed, we evaluated them
as "extend D at fixed N", which pushes into saturation and loses points. At **fixed
C** they do the opposite: more D reaches *smaller* N, which is exactly where the
missing left arm was. That reframing is what produced the interior minimum in §2.1.

**Estimates quoted from the wrong context.** A "~100× speedup" predicted from
dispatch-overhead reasoning was 4.2× when measured; a "~5 minute" bootstrap
benchmarked on clean data on a laptop ran 20+ minutes on noisy data on a Colab host.

---

## 6. Method notes worth keeping

**The three-regime structure is standard.** Loss goes random-guessing → power law →
irreducible floor, and the Chinchilla form describes only the middle. Standard
practice is to exclude both ends and document what was dropped. `select_fit_range`
does this: `guessing` (loss ≥ 90% of the position-unigram baseline), `saturated`
(loss ≤ 3× the floor), `undertrained` (< 500 optimizer steps), `capacity-limited`
(N below a threshold).

**A rectangular (N, D) grid makes diagonal isoFLOP lines.** The extreme compute
bands therefore always contain one or two models, by construction. At C=1.8e14 only
the 10.6M model could participate — a 594k model would need 11.5 epochs, a 100k
model 68. This is geometry, not a data problem, and it is why the direct isoFLOP
sweep (choose C, derive `D = C/(6N)` per model) is a better design than a grid.

**Multi-epoch infill is cheap.** Filling the thin bands took 4 runs and 3.8e14
FLOPs — **23%** of the original 8×8 grid — and turned the top band from one point
into three. Capped at 4 epochs, past which repeated tokens are worth measurably less
(Muennighoff et al.).

**Compute is concentrated.** The largest N rung is 61% of a grid sweep's compute;
the smallest four together are 2.8%. Run *count* is a bad proxy for cost — cutting
small models saves nothing.

**Fitting cost is CPU-bound and does not move to the TPU.** `scipy` finite-
differencing over 30 numbers is dispatch overhead; an accelerator has nothing to
accelerate.

---

## 7. What this means

**For this task.** 3-digit addition is learned discretely. There is a threshold near
600k active parameters; below it more data does not help, above it the task is
solved within a decade of compute. "Compute-optimal model size" collapses to "the
smallest model that can represent the algorithm," which is a capability threshold,
not a params-vs-data trade-off. That is a legitimate negative result about when
scaling laws apply, and it is visible in a single plot.

**For the MoE question.** The original plan — fit per expert count, watch `A` decline
with E while α, β, B hold steady — assumed an identifiable surface. There isn't one.
Two options remain:

1. **Iso-FLOP comparison.** At matched compute, compare best loss achieved by dense
   vs each E. No parameters to estimate, so noise hurts far less. Answers "does MoE
   buy anything at matched compute" directly.
2. **Does routing move the threshold?** Given the threshold framing, the sharper
   question is whether an MoE reaches competence at fewer *active* parameters than
   594k. That is a cleaner claim than comparing frontier intercepts, and this setup
   can test it.

Worth noting the prior: since smaller-model-more-data wins at every budget below the
threshold, this task is close to a worst case for MoE, whose proposition is buying
capacity cheaply.

---

## 8. Open questions

- **Threshold or trade-off?** We have one located minimum (N_opt = 594,432 at
  C=1.8e13). If it's a threshold, N_opt stays put across budgets; if it's a real
  trade-off, it marches right as C grows. Infilling C=5e12 and C=5e13 would bracket
  it. This is the single highest-value follow-up.
- **Is the threshold representational or optimisation-limited?** A model that *can*
  represent addition but doesn't find it within its step budget looks identical to
  one that can't. Multiple seeds at N=298,656 would distinguish them: if it learns
  in 1 of 3 runs, the threshold is soft.
- **Is the LR rule contributing?** `peak_lr = 1.5e-3·√(64/d_model)` replaced a
  hand-tuned table and was never validated. It gives smaller models the highest LR,
  which is a live suspect for their bimodality. Kept unchanged so far because
  changing it would make new cells incomparable to existing ones.
- **Would seeds make the surface fittable?** §3.3 says not at sd 0.5. The synthetic
  probe could be re-run at sd 0.2 and 0.1 to find what noise level makes α
  recoverable — i.e. how many seeds would be needed — before spending the compute.

---

## 9. Reproducing this

**Files.** `model.py` (dense + MoE, flag is `cfg.num_experts`), `config.py`,
`utils.py` (`active_params`, `check_param_formula`),
`chinchilla_laws_dense.ipynb`, `chinchilla_laws_moe.ipynb`.

**Dense notebook cell order:** `0,1,2` setup → `12–16` Part 2 definitions → `18`
sweep + save → `19` reload (skip if `rows` is in memory) → `20` fit range → `21`
isoFLOP → `22` infill → `23–26` surface fit. Part 1 (cells 3–10) is a superseded
isoFLOP approach and references an undefined `cfg`.

**Current settings.** `SEEDS = (0,)`, `MIN_N = 500_000` (surface fit only — isoFLOP
deliberately uses all rungs), `MAX_EPOCHS = 4.0`, `E_LADDER = [0, 2, 4, 8, 16]`,
`MODEL_LADDER` 8 rungs spanning 33,728 → 10,629,120 active params,
`DATA_LADDER` 8 rungs spanning 600k → 4.3M supervised tokens.

**Sanity checks that should hold.** `check_param_formula` on a dense config returns a
delta of 4,992 (norm gammas + pos_embed, both excluded from the formula by design).
Dense `(64, 2)` = 100,096 active params. `fit_surface` on a clean synthetic surface
recovers α=0.3593, β=0.3734 against true 0.34/0.37 with CIs of roughly ±0.04.
