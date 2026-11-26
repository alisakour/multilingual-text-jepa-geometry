# Cross-Dataset Extension: Complete NTREX-128 & MASSIVE v1.1

Extends the cross-dataset evaluation in two directions using the same extraction and geometry protocol (`float32`, `batch_size=32`, `k=10` mutual nearest neighbors at L9 and L12 with both `CLS` and attention-masked `mean` pooling):

1. **NTREX-128 Completion (`max_length = 256`, `N = 1,997` news sentences):**
   Evaluates the remaining 9 models/checkpoints omitted from `../ntrex_validation/` (`sigreg_01_e10`, `infonce_01_e5`, `infonce_1_e1`, the four 15-language InfoNCE encoders, `xlmr_base`, and `me5_base`) and merges them with the 5 archived NTREX checkpoints into a unified 14-model comparison (`results/ntrex_full/`).
2. **Amazon MASSIVE v1.1 (`test` split, `max_length = 128`, `N = 2,974` parallel utterances):**
   Evaluates all 14 models/checkpoints on short, natural everyday assistant utterances across `en-US`, `ar-SA`, `zh-CN`, and `fr-FR` (`59` intents, `18` scenarios; zero overlap with `FLORES` or `OPUS100`) to separate sequence-length effects from `mLAMA`'s repetitive templates, including a `within-intent` vs. `across-intent` neighbor decomposition (`results/massive/`).

## Summary Results

### 1. NTREX-128 Completion (`max_length = 256`, `results/ntrex_full/ntrex_all_layer_shifts.csv`)
- **Final `SIGReg` (`sigreg_01_e10`, Epoch 10):** `CLS` L9→L12 overlap shifts remain small (`+3.08 / +3.40 / +2.13 pp` for AR/ZH/FR), leaving final L12 alignment at `13.72% / 14.14% / 18.10%` — essentially matching unaligned `xlmr_base` (`13.96% / 14.37% / 19.63%`) — while `CLS` effective rank drops `-138.30` (AR) and `-140.37` (FR) (`+7.90` in ZH). Mean-pooling shifts remain negative (`-6.94 / -3.69 / -9.54 pp`).
- **Final `InfoNCE` (`infonce_01_e5` & `infonce_1_e1`):** `infonce_01_e5` reaches `21.04% / 23.17% / 28.96%` (`+13.74 / +13.89 / +17.81 pp`, rank `+101.77` to `+112.35`), and `infonce_1_e1` jumps `+14.63 / +17.77 / +22.22 pp` (`rank +138.46` to `+150.59`).
- **15-Language Encoders:** All four 15-language InfoNCE encoders achieve strong L12 alignment (`44.06%–55.74%` CLS; `43.29%–56.59%` MeanPool, competitive with `me5_base` at `45.08%–53.90%` CLS). Under MeanPool, bilingual packing compresses source effective rank far more severely (`-262.07` to `-318.33`) than monolingual packing (`-182.37` to `-204.32`).

### 2. MASSIVE v1.1 (`N = 2,974`, `59` intents, `results/massive/`)
- **Separating Length vs. Repetitive Templates:** Unlike `mLAMA`, `SIGReg` does **not** turn negative in `CLS` overlap on short non-templated `MASSIVE` utterances (`+6.96 / +8.14 / +9.85 pp` at E1; `+3.09 / +6.07 / +4.35 pp` at E10), confirming that `mLAMA`'s negative `CLS` drop is driven by its repetitive templates rather than short sequence length.
- **Persistent Spectral Collapse & `XLM-R` Ceiling in `SIGReg`:** Across `SIGReg` training (`E1 → E4 → E10`), `CLS` effective-rank loss intensifies monotonically (`-112/-75/-150` → `-200/-170/-185` → `-212.9/-190.4/-195.5`), L12 `CLS` alignment stagnates near `xlmr_base` (`9.14% / 15.46% / 18.13%` vs. `8.88% / 12.89% / 15.18%`), and MeanPool drops worsen from E1 (`-2.73 / -5.03 / -7.02 pp`) to E10 (`-6.88 / -8.35 / -11.55 pp`, ~2× larger than `infonce_01_e5` at `-2.75 / -3.58 / -7.26 pp`).
- **`InfoNCE` & Intent Decomposition (`massive_intent_decomposition.csv`):** `InfoNCE (0.1)` (`L12 = 14.96% / 22.60% / 26.51%` at E4; `15.49% / 23.41% / 28.25%` at E5) consistently outperforms both `SIGReg` and `InfoNCE (1.0)` (`13.52% / 22.35% / 24.12%`). Decomposition shows 5-lang `InfoNCE (0.1, E4)` gains concentrate in `within-intent` overlap (`+6.80 / +10.50 / +10.89 pp` CLS) plus positive `across-intent` gains (`+1.00 / +2.25 / +1.25 pp`), whereas `SIGReg E10` lags in both (`within`: `+2.86 / +5.67 / +5.54 pp`; `across`: `+0.23 / +0.40 / -1.18 pp` CLS). Among 15-lang out-of-domain runs, `opus_mono15` and `opus_biling15` achieve strong `across-intent` gains (`+6.40` to `+9.20 pp` in ZH/FR CLS) while `opus_mono15` retains roughly twice the spectral rank of `opus_biling15`.

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
