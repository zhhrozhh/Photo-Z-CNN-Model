# Macrocosm — CNN photo-z (example + your playground)

**One self-contained Colab notebook** — [`train_photoz_cnn.ipynb`](train_photoz_cnn.ipynb). Everything
(data loading, the **model architecture**, the pipeline, metrics, training) is **inline**, so you edit
it right there in Colab — no package to clone or `.py` files to touch.

Predicts a galaxy's **redshift** from a **64×64×5 ugriz cutout**. The included architecture is a
*starting point*: **fork the model cell, redesign it, train your own version, and compare every run in
the shared MLflow UI.**

## Use it
1. Open **`train_photoz_cnn.ipynb`** in Colab → **Runtime → Change runtime type → GPU**.
2. Run top to bottom. Section 0 installs deps, logs into Google, and pulls the image shards.
3. **Make your own version**: edit the **MODEL cell (section 2)** — more/fewer Inception blocks, a
   deeper stem, a classification head over redshift bins (Pasquet-style), transfer learning, etc.
   Name your run in section 4, train, and compare in the MLflow UI.

To log to MLflow, paste the bearer token in section 4 (ask the team — not in git) and start the server
first (`make mlflow-start` from the Macrocosm repo). **Without a token it just trains and skips logging.**

## The example architecture
VGG stem (cheap 64→16 downsample) + 3 lightweight **Inception** modules (multi-scale galaxy features)
→ GlobalAveragePooling → a 64-d **`embedding`** (this is what feeds the fusion head later) → `z`.
~288K params. See KB *CNN architectures — VGG vs Inception* (MCM-A-18).

## The bar to beat
| | σ_MAD | notes |
|---|---|---|
| tabular baseline | **0.0133** | the number images must beat to add value |
| example CNN (~288K params) | ~0.027 @ 10k | small-data sanity run; improves with more data |
| Pasquet 2019 (27M params, ~500k galaxies) | 0.0091 | the published benchmark |

Image-only at small N usually trails the tabular baseline — expected. The endgame is the **fusion
model** (this CNN's `embedding` + the tabular base models → an MLP head), where the image branch adds
complementary morphology/neighbour information on top of colors.

## Pointers (Macrocosm KB)
- **CNN architectures — VGG vs Inception** (MCM-A-18) — the design rationale.
- **Loading the dataset for training** (MCM-A-13) — how the in-RAM data loading works.
- **MLflow tracking server** (MCM-A-14) — server URL, token, lifecycle.
- **Modeling routes** (MCM-A-6) / **Architecture** (MCM-A-5).
