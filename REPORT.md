# Post-bootcamp research report — chasing σMAD 0.01, image-only

Independent continuation of the CNN redshift model from **Macrocosm** (Le Wagon batch 2301
final project). Goal: push photometric-redshift accuracy on SDSS toward the published
image-only benchmark of **σMAD ≈ 0.01** (Pasquet et al. 2019: 0.0091).

**Scope: image-only.** Every model here predicts redshift from the 24×24×5 ugriz cutout
alone — no catalog features at inference. (Catalog values appear only as *training targets*
of the pretext task in experiment 3.)

*Last updated: 2026-07-24.*

## Evaluation protocol

Fixed for every experiment — numbers are directly comparable:

- **Data**: `catalog_v4` (quality-cleaned 600k SDSS galaxies) + `sample_v4.5` cutouts
  (24×24×5 ugriz, registered), `preproc='p99'`.
- **Split**: `splits/v4-5-train.csv` (554,461) / `splits/v4-5-validate.csv` (45,357).
- **Metric**: Δz = (z_pred − z_true)/(1 + z_true);
  σMAD = 1.4826·median(|Δz − median(Δz)|); outlier = mean(|Δz| > 0.05). See `eval.py`.

## Leaderboard (image-only)

| # | Model | σMAD | outlier | Notes |
|---|-------|------|---------|-------|
| 1 | frozen v4 CNN, its own MDN head | 0.01263 | 1.15% | bootcamp model, baseline |
| 2 | HGB head on frozen MDN embedding | 0.01227 | 1.19% | head swap, same features |
| 3 | **bins-head CNN** (`arch='bins'`, +TTA) | **0.01192** | 1.07% | run `bins180-v1` — **current best** |
| 4 | HGB [bins emb + MDN emb] | 0.01204 | 1.11% | |
| 5 | HGB [bins + MDN + tab-pretext emb] | 0.01202 | **1.04%** | |
| 6 | HGB [bins + MDN + tab-pretext emb + bins-PDF] | 0.01201 | 1.08% | PDF adds nothing |
| 7 | HGB [hard + easy + bins + MDN emb] | 0.01198 | **1.04%** | best embedding-head combo |

Progress: **0.01263 → 0.01192** (−5.6%). Remaining gap to Pasquet: ~24%.

## Experiments

### 1. Head swap: MDN → HistGradientBoosting (frozen CNN)

The serving fusion model once produced z ≈ 5×10¹⁴ on out-of-distribution inputs — the MDN
head extrapolates its μ in log1p(z) space and `expm1` amplifies it. Replacing the head with
a gradient-boosted tree over the same frozen 64-d embedding:

- σMAD 0.01263 → 0.01227 (~3% better), identical inputs;
- tree output is bounded by construction — the extrapolation failure mode is structurally gone;
- fit time ~10 s on CPU, so head iteration became free.

### 2. Bin-classification head (`arch='bins'`)

Pasquet-style reformulation, implemented in `photoz_cnn.py` (`arch='bins'`) and trained with
`train_photoz_cnn_bins.ipynb` (Colab, run `bins180-v1`):

- 180 softmax bins uniform in log1p(z) over the catalog range z ∈ [0.02, 0.35];
- cross-entropy against Gaussian-soft labels (σ = 1 bin width); point estimate = E[z] over
  the softmax — bounded by construction;
- 8-view dihedral TTA at eval.

Result: **0.01263 → 0.01192**, first sub-0.012 image-only model, out of the box (no
tuning). Its embedding is also better food for the HGB head than the MDN one (0.01216 vs
0.01227), and the two embeddings are complementary (dual: 0.01204).

### 3. Tab-feature pretext CNN (`tab_cnn.py`)

A CNN that predicts the **16 tabular features from the image** (per-feature 2-Gaussian MDN
heads, NaN-masked NLL, targets z-scored and winsorized) — distilling catalog
photometry/morphology into the same 64-d embedding interface, while keeping inference
image-only. Run `tab-mdn-v1`, `train_tab_cnn.ipynb`.

| Embedding combination (HGB head) | σMAD | outlier |
|---|---|---|
| tab-pretext emb alone | 0.01690 | 2.75% |
| bins + MDN emb | 0.01204 | 1.11% |
| bins + MDN + tab-pretext emb | 0.01202 | 1.04% |
| bins + MDN + tab-pretext emb + bins-PDF(180) | 0.01201 | 1.08% |

Alone it is weak (it learns photometry, not z — expected for a pretext task). In
combination it trims the outlier rate (1.11% → 1.04%) but moves σMAD only marginally, and
adding the bins model's 180-d softmax PDF adds nothing the embeddings didn't already carry.

**Per-feature diagnosis and the hard/easy split.** Evaluating the joint model per feature
exposed a clear pattern: magnitudes/sizes are well predicted (R² 0.83–0.97) while the most
z-sensitive colours are weak — u−g R² 0.59, i−z 0.73, conc_r 0.71. Hypothesis: the 5
magnitudes dominate the shared loss. Confirmed by training two dedicated models
(`train_tab_cnn_split.ipynb`, runs `tab-mdn-hard-v1` / `tab-mdn-easy-v1`):

| Feature (hard group) | joint R² | dedicated R² | Δ |
|---|---|---|---|
| u−g | 0.588 | **0.826** | +0.238 |
| i−z | 0.728 | **0.835** | +0.107 |
| r−i | 0.861 | 0.932 | +0.071 |
| conc_r | 0.713 | 0.787 | +0.074 |
| petroRad / petroR90 | 0.83 | 0.85 | +0.01 |

(The easy group stays flat when trained alone — it was never capacity-limited.) So the
24 px image does carry the z-sensitive colour information; joint training was burying it.

Stacking the new embeddings into the HGB head, however, converts those big R² gains into
only a marginal σMAD move: [hard+easy+bins+MDN] = **0.01198** (best embedding-head combo;
ablations: bins+mdn+hard 0.01198, bins+hard+easy 0.01207, adding the old joint tab
embedding is fully redundant at 0.01200). The colour information the hard model recovered
is largely already inside the bins embedding.

**Split v2 closes the line.** Splitting the hard group once more into dedicated colour
(`u-g, i-z, r-i` → runs `tab-mdn-color-v1`) and morphology (`conc_r, petroR90` →
`tab-mdn-morph-v1`) models shows the dedication lever saturating — R² gains vs the hard-6
model shrink to +0.008…+0.039 (only i−z still moves) — and every resulting HGB stack,
up to all six embeddings (384-d), lands on exactly **0.01198**. Feature-prediction quality
is no longer the constraint; the 0.0120 plateau is the hard ceiling of a single trunk's
information.

### Where the embedding-head line plateaus

All embedding-head combinations converge to σMAD ≈ 0.0120 (best: 0.01198 with four
embeddings) — and none of them beats the bins CNN's own head (0.01192). Even a +0.24 R²
jump on the most z-sensitive colour (u−g, via the dedicated hard model) buys only ~0.3% of
σMAD, because the bins embedding already encodes most of that colour signal. Conclusion:
with a single trunk's information, the end-to-end trained head already extracts what there
is; stacked frozen-embedding heads mainly help the outlier tail. The information bottleneck
is the trunk/embedding, not the head.

## Findings

1. **Bounded heads beat parametric density heads**: bins-CE beats MDN as a CNN head
   (−5.6%), and HGB beats MDN over frozen embeddings (−3%). Both also eliminate the
   expm1-extrapolation failure mode that once produced z ~ 10¹⁴.
2. **Pretext embeddings are complementary but small**: embeddings trained toward different
   targets (z-regression, z-classification, tabular reconstruction) combine to the best
   outlier rate (1.04%), but σMAD gains are marginal once the bins embedding is present —
   even after the hard/easy split fixed the pretext model's weak features (u−g R²
   0.59→0.83), σMAD only moved 0.01202→0.01198. Joint multi-task losses do bury
   hard-but-valuable targets, though — a lesson that transfers beyond this experiment.
3. **The head is no longer the bottleneck** — every combination plateaus at ~0.0120 while
   the bins CNN sits at 0.01192. Further gains must come from the trunk: capacity, training
   recipe, ensembling, and cutout size.

## Next steps

1. **Multi-seed bins ensemble** — retrain `arch='bins'` with 2–4 seeds, average predictions
   (and/or concatenate embeddings). The standard next ~3–5%.
2. **Bins-head hyperparameters** — `bins`, `bins_smooth`, longer training, `arch='extend'`
   trunk; entirely untuned so far.
3. **Larger cutouts (32/64 px)** — the strongest literature lever (Pasquet used 64 px);
   requires re-cutting shards and a HiRAM runtime.
4. More pretext targets (raw magnitudes, shape parameters) if the ensemble line stalls.

## Infrastructure

- Experiment tracking: self-hosted MLflow (docker on the home machine, SSH-tunneled and
  reverse-proxied at `https://hangman-lab.top/ex-miscs/mlflow`, token-authenticated).
  Runs `bins180-v1`, `tab-mdn-v1` carry the models, per-galaxy predictions and recipes.
- Training: Google Colab GPU (`train_photoz_cnn_bins.ipynb`, `train_tab_cnn.ipynb`);
  HGB-head experiments run locally on CPU in seconds–minutes.
- The GCP MLflow VM from the bootcamp is retired; data/models remain on
  `gs://macrocosm-lewagon`.
