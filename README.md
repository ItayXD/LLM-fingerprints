# Model Fingerprinting & Data Generation Suite

This repository houses scripts and notebooks for fingerprinting language models on out-of-distribution (OOD) prompts, generating synthetic training corpora, and running follow-up analyses (e.g., clustering via latent embeddings). It supports both open-source Hugging Face models and OpenAI-hosted models via synchronous or batch APIs.

Reference dataset: https://huggingface.co/datasets/royrin/model-fingerprints-data/upload/main

## Repository Layout

- `code/fingerprint.py` – main fingerprinting runner against Hugging Face models.
- `code/generate_training_data_oai.py` – OpenAI-backed generator with batch submit/fetch support.
- `code/generate_training_data.py`, `generate_training_data_anthropic.py`, `generate_training_data_oai.py` – variants for different providers.
- `code/batch_*.py` – helpers to submit/aggregate distillation jobs in bulk.
- `utils/` – shared helpers (API telemetry, model catalogs, etc.).
- `notebooks/` & `embeddings.ipynb` – exploratory analysis and visualization notebooks.
- `fingerprints_output/` – default location for generated fingerprints.

## Getting Started

```bash
# Install dependencies (uv manages both venv + lockfile)
uv sync

# Or create the environment manually
uv venv
uv sync
```

## Fingerprinting Hugging Face Models

### Core Commands

- Default run (Llama-3.2-1B):
  ```bash
  uv run python code/fingerprint.py
  ```
- Specific model:
  ```bash
  uv run python code/fingerprint.py meta-llama/Llama-3.2-3B
  ```
- All configured models:
  ```bash
  uv run python code/fingerprint.py --all
  ```
- List available models:
  ```bash
  uv run python code/fingerprint.py --list
  ```

### Advanced Options

```bash
# Sampling instead of greedy
uv run python code/fingerprint.py --all --temp 1.0

# Higher entropy and explicit seed
uv run python code/fingerprint.py --all --temp 1.5 --seed 123

# Generate more tokens per prompt
uv run python code/fingerprint.py --all --num-tokens 10
```

Important arguments:
- `--temperature/--temp/-t` – decoding temperature (default 0.0).
- `--seed/-s` – RNG seed (default 42).
- `--num-tokens/-n` – number of continuation tokens (default 5).
- `--prompts-file/-p` – JSON file with custom prompts (defaults to `ood_prompts.json`).
- `--all`, `--list` – bulk execution helpers.

### Provider Integrations

The repo fingerprints both local Hugging Face checkpoints and hosted APIs:
- **OpenAI** – via `utils/llm_api.py` and `code/generate_training_data_oai.py`, supporting chat completions, batch submissions, and fine-tuning formats.
- **Anthropic** – through `code/generate_training_data_anthropic.py` and shared helpers so Claude generations can be compared side-by-side with OpenAI and HF baselines.

Regardless of source, results land in `fingerprints_output/{timestamp}/` with one JSON per model plus `all_fingerprints.json`, recording the prompt, continuation, token IDs, and decoding metadata.

## Prompt Generation

Fingerprints rely on high-entropy, out-of-distribution prompts (random strings, nonsensical phrases, pattern repeats, hex/base64-like payloads, etc.). Bring your own prompt list and pass it to `fingerprint.py`:

```bash
uv run python code/fingerprint.py --all --prompts-file my_prompts.json
```

Expected JSON schema:

```json
{
  "description": "My custom prompts",
  "prompts": ["prompt1", "prompt2"]
}
```

## Synthetic Training Data (OpenAI)

`code/generate_training_data_oai.py` produces user/assistant conversations for knowledge distillation. Highlights:

- Supports synchronous calls or batch API submissions with `--use-batch`.
- `--batch-submit-only` uploads a job and exits immediately (safe to close your laptop).
- `--batch-fetch <BATCH_ID>` downloads finished results later and converts them into JSONL/metadata.
- `--prompts-file` or `--prompts-jsonl` seed generation with curated prompts instead of random templates.

Example synchronous run:

```bash
uv run python code/generate_training_data_oai.py \
  --model gpt-4o-mini \
  --min-num-sequences 200 \
  --temperature 0.8
```

Batch submit-only + fetch:

```bash
# Submit
uv run python code/generate_training_data_oai.py \
  --model gpt-4o \
  --use-batch --batch-submit-only \
  --output-name gpt4o_batch_2025

# Later, fetch results using the printed batch_id
uv run python code/generate_training_data_oai.py \
  --model gpt-4o \
  --batch-fetch BATCH_ID \
  --output-name gpt4o_batch_2025
```

Outputs live in `training_data/` (JSONL plus metadata). Batch metadata records input/output file IDs so you can audit submissions via the OpenAI dashboard.

## Knowledge Distillation & Analysis

- `code/knowledge_distillation.py` and related scripts consume the generated datasets to fine-tune student models.
- `batch_distill_pairwise.py`, `batch_generate_distillation_data.py`, and `generate_training_data.py` handle larger sweeps or non-OpenAI providers.
- `embeddings.ipynb` and other notebooks explore latent representations of fingerprints for clustering/visualization.

## Embeddings Notebook (Primary Results)

`embeddings.ipynb` is the main report for downstream fingerprint analysis:

- **Dataset assembly:** loads `encoding_model_analysis/encoding_data/*_embeddings.npz` for GPT-3.5/4.1 variants, Claude 3.5/4.5 family, GPT-4o (⭐), and open models like Llama-3.2; merges them into a shared tensor dataset with per-model labels.
- **Classical baselines:** evaluates Gaussian Naive Bayes, logistic regression, linear SVM, LDA/QDA, and scikit-learn pipelines with standardized features; confusion matrices highlight which providers are easiest/hardest to separate.
- **Neural classifier:** trains a multi-layer SiLU network (384→1000→500→100→classes) hitting high test accuracy and macro-F1, plus aggregated confusion plots mapping predictions to normalized probabilities across GPT/Claude families.
- **Autoencoder + center loss:** builds a joint reconstruction/classification model that learns compact embeddings (`z`) while enforcing per-model cluster centers to stabilize fingerprint geometry.
- **Dimensionality reduction:** generates PCA, t-SNE, MDS, and UMAP scatter plots (color-coded by provider) to visualize how embeddings cluster by model family and whether OOD prompts collapse subclusters.
- **Unsupervised clustering:** applies KMeans, Agglomerative, DBSCAN, and HDBSCAN on latent vectors to test blind identification; silhouette/confusion summaries show models remain distinguishable even without labels.

Use this notebook when communicating results—its figures (classification tables, confusion matrices, latent plots) are the authoritative artifacts for the project.

## Notes & Tips

- All scripts rely on environment variables or CLI flags for API keys (`OPENAI_API_KEY`, Anthropic keys, etc.).
- `uv` manages dependencies via `pyproject.toml` and `uv.lock`—stick with it to keep versions aligned.
- Fingerprinting is deterministic when `--temp 0.0`; raise the temperature plus seed if you want stochastic fingerprints.
- Generated fingerprints and training corpora can grow quickly—periodically prune `fingerprints_output/` and `training_data/` to save disk space.
