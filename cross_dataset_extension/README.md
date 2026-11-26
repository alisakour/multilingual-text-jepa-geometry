# Cross-Dataset Extension: Complete NTREX-128 & MASSIVE v1.1

Extends the cross-dataset evaluation in two directions using the same extraction and geometry protocol (`float32`, `batch_size=32`, `k=10` mutual nearest neighbors at L9 and L12 with both `CLS` and attention-masked `mean` pooling):

1. **NTREX-128 Completion (`max_length = 256`, `N = 1,997` news sentences):**
   Evaluates the remaining 9 models/checkpoints omitted from `../ntrex_validation/` (`sigreg_01_e10`, `infonce_01_e5`, `infonce_1_e1`, the four 15-language InfoNCE encoders, `xlmr_base`, and `me5_base`) and merges them with the 5 archived NTREX checkpoints into a unified 14-model comparison.
2. **Amazon MASSIVE v1.1 (`test` split, `max_length = 128`, `N = 2,974` parallel utterances):**
   Evaluates all 14 models/checkpoints on short, natural everyday assistant utterances across `en-US`, `ar-SA`, `zh-CN`, and `fr-FR` (60 intents, 18 scenarios; zero overlap with `FLORES` or `OPUS100`) to separate sequence-length effects from `mLAMA`'s repetitive templates, including a `within-intent` vs. `across-intent` neighbor decomposition.

## Datasets & Provenance

- **NTREX-128:** Pinned MicrosoftTranslator revision `468c6b69c7f6a75d31d4743d9daba2af566cc18d` (`CC BY-SA 4.0`), verified via Git blob SHA-1 hashes (`1,997` line-aligned news sentences).
- **MASSIVE v1.1:** Official tarball (`amazon-massive-dataset-1.1.tar.gz`, `CC BY 4.0`), verified via SHA-256 `4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577` (`2,974` aligned `test` utterances across `en-US`, `ar-SA`, `zh-CN`, `fr-FR`).

## Usage

From `cross_dataset_extension/` (with a CUDA-enabled PyTorch environment):

```sh
python -m pip install -r ../requirements.txt huggingface_hub
python scripts/evaluate_ntrex_remaining.py
python scripts/evaluate_massive.py
```

By default, results are written under `results/ntrex_full/` and `results/massive/` (pass `--out_dir` to write directly to Google Drive or another output path). Completed checkpoints are skipped automatically if re-run.
