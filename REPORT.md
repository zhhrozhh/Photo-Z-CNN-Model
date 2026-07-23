# Post-bootcamp research report — chasing σMAD 0.01

Independent continuation of the CNN redshift model from **Macrocosm** (Le Wagon batch 2301
final project). Goal: push photometric-redshift accuracy on SDSS from the bootcamp result
(σMAD 0.0118) to the published benchmark of **σMAD ≈ 0.01** (Pasquet et al. 2019: 0.0091).

*Last updated: 2026-07-23.*

## Evaluation protocol

Fixed for every experiment — numbers are directly comparable:

- **Data**: `catalog_v4` (quality-cleaned 600k SDSS galaxies) + `sample_v4.5` cutouts
  (24×24×5 ugriz, registered), `preproc='p99'`.
- **Split**: `splits/v4-5-train.csv` (554,461) / `splits/v4-5-validate.csv` (45,357).
- **Metric**: Δz = (z_pred − z_true)/(1 + z_true);
  σMAD = 1.4826·median(|Δz − median(Δz)|); outlier = mean(|Δz| > 0.05). See `eval.py`.

## Leaderboard

`tab16` = the 16 engineered catalog features fed directly to the head (NaN-native, no
imputation) — those rows need catalog data at inference. "image" rows need the cutout only.

| # | Model | Inputs | σMAD | outlier | Notes |
|---|-------|--------|------|---------|-------|
| | *bootcamp tabular baseline (3-base stack)* | catalog | 0.0127 | 1.35% | reference |
| | *bootcamp fusion (MLP+MDN over base+mask+emb)* | image+catalog | 0.0118 | ~1.1% | reference |
| 1 | frozen v4 CNN, its own MDN head | image | 0.01263 | 1.15% | baseline being improved |
| 2 | HGB head on frozen MDN embedding | image | 0.01227 | 1.19% | head swap, same features |
| 3 | HGB [MDN emb + tab16] | image+catalog | 0.01132 | 0.97% | first sub-fusion result |
| 4 | **bins-head CNN** (`arch='bins'`, +TTA) | image | 0.01192 | 1.07% | run `bins180-v1` |
| 5 | HGB [bins emb + MDN emb] | image | 0.01204 | 1.11% | |
| 6 | **HGB [bins + MDN + tab-pretext emb]** | image | **0.01202** | 1.04% | **best image-only** |
| 7 | HGB [bins emb + tab16] | image+catalog | 0.01130 | 0.99% | |
| 8 | **HGB [bins + MDN + tab-pretext emb + tab16]** | image+catalog | **0.01116** | **0.89%** | **current best** |

Progress: 0.0118 → **0.01116** (−5.4%); image-only: 0.01263 → 0.01202 (−4.8%).
Remaining gap to 0.01: ~12%. NB the pure-image Pasquet benchmark (0.0091) is properly
compared against the "image" rows.

## Experiments

### 1. Head swap: MDN → HistGradientBoosting (frozen CNN)

The serving fusion model once produced z ≈ 5×10¹⁴ on out-of-distribution inputs — the MDN
head extrapolates its μ in log1p(z) space and `expm1` amplifies it. Replacing the head with
a gradient-boosted tree over the same frozen 64-d embedding:

- σMAD 0.01263 → 0.01227 (~3% better), identical inputs;
- tree output is bounded by construction — the extrapolation failure mode is structurally gone;
- fit time ~10 s on CPU, so head iteration became free.

Adding the 16 raw tabular features (HGB handles NaN natively — no imputation, no presence
mask, no 700 MB base-model stack) reached 0.01132, already beating the bootcamp fusion with
a far simpler serving path: `CNN embedding + one HGB`.

### 2. Bin-classification head (`arch='bins'`)

Pasquet-style reformulation, implemented in `photoz_cnn.py` (`arch='bins'`) and trained with
`train_photoz_cnn_bins.ipynb` (Colab, run `bins180-v1`):

- 180 softmax bins uniform in log1p(z) over the catalog range z ∈ [0.02, 0.35];
- cross-entropy against Gaussian-soft labels (σ = 1 bin width); point estimate = E[z] over
  the softmax — bounded by construction;
- 8-view dihedral TTA at eval.

Image-only result: **0.01263 → 0.01192**, first sub-0.012 pure-image model, out of the box
(no tuning). Its embedding is also better food for the HGB head than the MDN one
(0.01216 vs 0.01227), and the two embeddings are complementary (dual: 0.01204).

### 3. Tab-feature pretext CNN (`tab_cnn.py`)

A CNN that predicts the **16 tabular features from the image** (per-feature 2-Gaussian MDN
heads, NaN-masked NLL, targets z-scored and winsorized) — distilling catalog
photometry/morphology into the same 64-d embedding interface. Run `tab-mdn-v1`,
`train_tab_cnn.ipynb`.

As predicted for a pretext task: nearly useless alone (0.0169), redundant next to the real
tab16 (0.01271 ≈ tabular baseline) — but **worth ~1% inside combinations**, acting as
image-derived photometry free of catalog measurement noise:

| Combination | Inputs | σMAD | outlier |
|---|---|---|---|
| bins + MDN + tab-pretext emb | image | 0.01202 | 1.04% |
| bins emb + tab16 | image+catalog | 0.01130 | 0.99% |
| bins + tab-pretext emb + tab16 | image+catalog | 0.01120 | 0.89% |
| bins + MDN + tab-pretext emb + tab16 | image+catalog | **0.01116** | **0.89%** |

The MDN embedding is nearly retired: dropping it costs only 0.00004.

## Findings

1. **Bounded heads beat parametric density heads** on this problem, twice over: bins-CE
   beats MDN as a CNN head, and HGB beats both as an embedding head. Both also eliminate
   the expm1-extrapolation failure mode.
2. **Embedding diversity pays**: embeddings trained toward different targets (z-regression,
   z-classification, tabular reconstruction) are complementary; concatenating them into one
   HGB is the cheapest ensemble available (~1 min CPU per experiment).
3. **The head is no longer the bottleneck**: every +tab16 combination saturates at
   σMAD ≈ 0.0112 regardless of which embeddings enter. Further gains must come from
   embedding quality (bigger/longer-trained trunks, multi-seed ensembles, larger cutouts)
   rather than head engineering.

## Next steps

1. **Multi-seed bins ensemble** — retrain `arch='bins'` with 2–4 seeds, concatenate
   embeddings into the HGB stack (expected: break 0.011).
2. More pretext targets (raw magnitudes, shape parameters) to keep stacking complementary
   embeddings.
3. Larger cutouts (32/64 px) — the strongest literature lever; requires re-cutting shards
   and a HiRAM runtime.
4. Bins-head hyperparameters (`bins`, `bins_smooth`, `arch='extend'` trunk) — untouched so far.

## Infrastructure

- Experiment tracking: self-hosted MLflow (docker on the home machine, SSH-tunneled and
  reverse-proxied at `https://hangman-lab.top/ex-miscs/mlflow`, token-authenticated).
  Runs `bins180-v1`, `tab-mdn-v1` carry the models, per-galaxy predictions and recipes.
- Training: Google Colab GPU (`train_photoz_cnn_bins.ipynb`, `train_tab_cnn.ipynb`);
  HGB-head experiments run locally on CPU in seconds–minutes.
- The GCP MLflow VM from the bootcamp is retired; data/models remain on
  `gs://macrocosm-lewagon`.
