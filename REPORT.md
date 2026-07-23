# Post-bootcamp research report — chasing σMAD 0.01, image-only

Independent continuation of the CNN redshift model from **Macrocosm** (Le Wagon batch 2301
final project). Goal: push photometric-redshift accuracy on SDSS toward the published
image-only benchmark of **σMAD ≈ 0.01** (Pasquet et al. 2019: 0.0091).

**Scope: image-only.** Every model here predicts redshift from the 24×24×5 ugriz cutout
alone — no catalog features at inference. (Catalog values appear only as *training targets*
of the pretext task in experiment 3.)

*Last updated: 2026-07-23.*

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
| 5 | HGB [bins + MDN + tab-pretext emb] | 0.01202 | **1.04%** | best embedding-head combo |
| 6 | HGB [bins + MDN + tab-pretext emb + bins-PDF] | 0.01201 | 1.08% | PDF adds nothing |

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

### Where the embedding-head line plateaus

All embedding-head combinations converge to σMAD ≈ 0.0120 — and none of them beats the
bins CNN's own head (0.01192). Conclusion: with a single trunk's information, the
end-to-end trained head already extracts what there is; stacked frozen-embedding heads
mainly help the outlier tail. The information bottleneck is the trunk/embedding, not the
head.

## Findings

1. **Bounded heads beat parametric density heads**: bins-CE beats MDN as a CNN head
   (−5.6%), and HGB beats MDN over frozen embeddings (−3%). Both also eliminate the
   expm1-extrapolation failure mode that once produced z ~ 10¹⁴.
2. **Pretext embeddings are complementary but small**: embeddings trained toward different
   targets (z-regression, z-classification, tabular reconstruction) combine to the best
   outlier rate (1.04%), but σMAD gains are marginal once the bins embedding is present.
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
