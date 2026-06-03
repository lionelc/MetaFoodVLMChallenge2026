# Dishcovery Mission II — Challenge in Metafood Workshop @ CVPR 2026

by Lei Jiang (lionelchange [at] gmail.com)

## Approach

This solution uses a **4-model embedding ensemble** with **VLM description-based refinement** and **multi-layer rule optimization** to solve both the multi-label (Test1) and single-label (Test2) food recognition tasks. Other than the main models, (ad-hoc) heuristics such as **embedding adaptive base** (in Test1 candidate selection) and **Sinkhorn algorithm** (for Test2 assignment), are helpful in practice (more details below).

It's worth mentioning that there are several other methods that were attempted but did **NOT** show positive effect in testing results:

- Tried fine-tuning Qwen3-2B and 8B with the provided training data and food101 on Huggingface: it helped with Test1 precision but also introduced more noise; F1 didn't improve (admittedly, I joined this competition late so may not dig deep enough in this direction)
- Tried other base models like GME: didn't make enough diversification in the ensemble to help boosting overall effectiveness


### Embedding Models (4-model ensemble)

Four vision-language embedding models produce image–caption similarity matrices:

| Model | Dim | Weight (T1) | Weight (T2) |
|---|---|---|---|
| SigLIP2 ViT-B-16-SigLIP2-256 (webli) | 768 | 0.25 | 0.20 |
| Qwen3-VL-Embedding-8B | 4096 | 0.35 | 0.40 |
| CLIP ViT-H-14 (dfn5b) | 1024 | 0.30 | 0.15 |
| Qwen3-VL-Embedding-2B | 2048 | 0.10 | 0.25 |

Each model encodes all test images and all captions independently. The z-normalized dot-product score matrices are combined with the above weights to form a fused score matrix.

The embedding-only pipeline (without description refinement) achieves a Kaggle score of 0.68154 on the latest test (with other heuristics on). 

### Description Models (pre-computed)

Two VLMs generate free-text food descriptions for each image, boosting the score significantly beyond embedding-only:

- **Qwen3-VL-8B-Instruct** — describes food items per category (protein, starch, vegetable, etc.)
- **Gemma-4-E4B-it** — same prompt, independent descriptions

These descriptions are matched against caption labels using regex patterns to produce `desc_qwen`, `desc_gemma`, and `desc_and` (intersection) signal sets used in layers 2–4 below.

Description caches are pre-computed and shipped with the submission. Generating them from scratch takes hours. If unavailable for new data, the pipeline falls back to embedding-only mode automatically. 

Further, to avoid issues in comparing embeddings only (e.g. both starts with "a delicious plate" but the food items are different), extracting food entities out of text description and adding Jaccard entity-overlap to assignment criteria also proves to be beneficial. By bringing it to the text world, it also leads to the margin gates and clustered  

### Test1 Pipeline (multi-label)

Eight sequential layers refine predictions:

1. **Embedding adaptive base** — Per-image adaptive top-k selection using a gap-based elbow method on the fused score row, plus seasoning/container threshold boosts.
2. **Description ensemble** — Adds labels confirmed by Qwen3 and/or Gemma4 descriptions, filtered by a precision/TP threshold manifest. *(skipped if description caches unavailable)*
3. **Triggered selected** — Adds labels where description signals fire for specific label–gate combinations from a tuned manifest of 35 rules. *(skipped if description caches unavailable)*
4. **Margin gates** — After applying adaptive top-k in embedding, some correct labels get ranked just outside the cutoff (a lower rank). Description models can detect them, but need a stronger signal to trigger an override. With margin gates, 108 rules that add labels when description signals fire AND the embedding score is within a configurable margin of the decision boundary. *(skipped if description caches unavailable)*
5. **Disambiguation** — Removes systematically confused label pairs (e.g., "noodles" vs "instant noodles") identified by clustering analysis (100 rules).
6. **Aggressive clustered** — Second round of confusion pair removal (142 rules).
7. **Second-pass clustered** — Final cleanup of remaining confusion pairs (142 rules).
8. **Recall addbacks** — Re-adds high-confidence labels that were removed by earlier layers, using source predictions from intermediate layers.

### Test2 Pipeline (assignment)

1. Compute the fused 4-model score matrix (same weights as above).
2. Add a Jaccard entity-overlap bonus between image descriptions and captions *(skipped if description caches unavailable)*.
3. Apply entity confusion corrections from three clustered A/B manifests (e.g. "").
4. Sinkhorn optimal transport assignment (τ=0.15, 50 iterations) for 1:1 image–caption matching.

## Repository Structure

```text
run_all_pipeline.sh                  # Main end-to-end runner
cache_final_4models_auto.py          # Builds the four embedding caches (parallel workers)
generate_description_cache.py        # Builds Qwen3/Gemma4 description caches (parallel)
infer_with_rule_final_submission.py  # Generates submission from caches + rule/heuristics
download_challenge_data.py           # Kaggle competition data downloader
requirements.txt                     # Python dependency pins

optimization/                        # Runtime helper scripts + frozen rule manifests
  experiment_results/                # Frozen JSON rule manifests (12 files, still applicable even for unseen data under the same item distribution)

siglip2_cache/                       # Pre-computed embedding caches (4 .npy each)
qwen3_cache/
clip_cache/vit_h_14_dfn5b/
qwen3vl2b_cache/
description_cache/                   # Pre-computed VLM description caches
  qwen3_de_cache/                    #   Qwen3 test1 + test2 descriptions
  gemma4_cache/                      #   Gemma4 test1 descriptions
```

## Quick Start

### Full pipeline (builds embedding caches + generates submission):

```bash
INSTALL_DEPS=0 ./run_all_pipeline.sh
```

This runs three steps:
1. Install dependencies from `requirements.txt`
2. Build or validate the four embedding caches (3 waves, ~42 min on H100 80GB)
3. Generate `submission.csv` from caches + rule manifests (~20 sec)

Below are the break-down steps if you want more control on each stage. 

### Embedding cache build sequence

The pipeline builds caches in three sequential waves to stay within H100 80GB VRAM and avoid thread exhaustion:

| Wave | Models | Concurrency | Time (H100) | VRAM |
|---|---|---|---|---|
| 1 | qwen3_8b + qwen3_2b | 2 (parallel) | ~35 min | ~61 GB |
| 2 | siglip2 | 1 | ~3 min | ~3 GB |
| 3 | clip_h | 1 | ~4 min | ~4 GB |

If all 16 embedding cache files already exist, the cache build step is skipped entirely.

Batch sizes (`qwen3_8b=192 qwen3_2b=192 siglip2=1024 clip_h=512`) affect numerical results due to padding in the SentenceTransformer models and must be preserved for exact reproduction.

**Note**: Considering resource limitation, the following sequence of model running is tested to be working. Clip_h could crash when running with other models, although itself is a relatively lightweighted model. 

#### Wave 1: Qwen3 models (parallel)
```bash
python cache_final_4models_auto.py --models qwen3_8b qwen3_2b --concurrency 2 \ --batch_override qwen3_8b=192 qwen3_2b=192
```

#### Wave 2: SigLIP2
```bash
python cache_final_4models_auto.py --models siglip2 --concurrency 1\
--batch-override siglip2=1024
```

#### Wave 3: CLIP-H (run alone to avoid thread exhaustion)
```bash
python cache_final_4models_auto.py --models clip_h --concurrency 1\
--batch-override clip_h=512
```

### Description cache generation (pre-computed, ships with submission)

```bash
python generate_description_cache.py
```

Runs Qwen3-VL-8B-Instruct and Gemma-4-E4B-it in parallel as separate processes (~38 GB VRAM combined). Resume-safe — saves every 50 images. It could take several hours total.

### Submission generation only (with pre-built caches for public data)

```bash
python infer_with_rule_final_submission.py --output submission.csv
```

Runs in ~20 seconds. Uses pre-computed embedding and description caches.

## Runtime on 1x RTX PRO 6000 S

| Step | Time | VRAM Peak |
|---|---|---|
| Embedding cache build (3 waves) | ~42 min | ~61 GB |
| Submission generation | ~20 sec | ~2 GB |
| **Total (with pre-built description caches)** | **~42 min** | **61 GB** |

Description caches are pre-computed and shipped. For new/private test data, the pipeline automatically detects missing image entries and regenerates descriptions only for those images (resume-safe). If GPU is unavailable, the pipeline gracefully degrades to embedding-only mode.

## Settings/credentials for data access

### Hugging Face

The embedding cache step downloads models from Hugging Face. Provide a token with access to:

- `Qwen/Qwen3-VL-Embedding-8B`
- `Qwen/Qwen3-VL-Embedding-2B`

```bash
export HF_TOKEN='hf_...'
```

### Kaggle (for data download only)

```bash
export KAGGLE_USERNAME='your_username'
export KAGGLE_KEY='your_api_key'
python download_challenge_data.py
```

## Run Options

| Variable | Default | Description |
|---|---|---|
| `BATCH_OVERRIDE` | `qwen3_8b=192 qwen3_2b=192 siglip2=1024 clip_h=512` | Per-model image batch sizes |
| `SKIP_DESCRIPTION_CACHE_BUILD` | 0 | Set to 1 to skip description generation (uses pre-built caches only, degrades gracefully for missing images) |
| `INSTALL_DEPS` | 1 | Auto-install from requirements.txt |
| `FORCE_CACHE` | 0 | Rebuild embedding caches even if complete |
| `PIPELINE_DRY_RUN` | 0 | Print commands without executing |

## References

- **SigLIP2**: Tschannen, M., Gritsenko, A., Wang, X., Naeem, M., Alabdulmohsin, I., Parthasarathy, N., Müller, R., Xiong, Y., Keysers, D., Beyer, L., & Zhai, X. (2025). *SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features*. arXiv:2502.14786.
- **CLIP**: Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I. (2021). *Learning Transferable Visual Models From Natural Language Supervision*. ICML 2021. arXiv:2103.00020.
- **Qwen3-VL**: Bai, S. et al. (2025). *Qwen2.5-VL Technical Report*. arXiv:2502.13923. Model weights: [Qwen/Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B), [Qwen/Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B), [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct).
- **Gemma 4**: Google DeepMind (2025). *Gemma 4*. Model weights: [google/gemma-4-e4b-it](https://huggingface.co/google/gemma-4-e4b-it).
- **Sinkhorn algorithm**: Cuturi, M. (2013). *Sinkhorn Distances: Lightspeed Computation of Optimal Transport*. NeurIPS 2013.
