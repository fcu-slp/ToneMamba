# Validation

## Environment

| Item | Value |
| --- | --- |
| Platform | Linux x86_64 |
| GPU / driver | NVIDIA GeForce RTX 5090 / 580.173.02 |
| Python / uv | 3.12.10 / 0.8.16 |
| PyTorch / CUDA | 2.11.0+cu128 / 12.8 |
| Mamba / causal-conv1d | 2.3.2.post1 / 1.6.2.post1 |
| Precision | BF16 autocast, TF32 enabled |
| Model | `ToneNetResetBiMamba2WithRhythm`, 5,447,239 parameters |
| Checkpoint | `checkpoints/model.pt`, strict state-dict loading passed |

Checkpoint SHA-256:
`7c8187b21d3a123efcb4d1be604805bb253b4b6e937c9fccb84e03509a6447af`.

## Dataset

[poyu39/FCU_Voice_47_EX](https://huggingface.co/datasets/poyu39/FCU_Voice_47_EX),
`main` resolved to `dcaa03e8b0b3241739cf1a27495e26a8e3664172`.

| Split | Usable IPUs | F0 frames | Syllables | Evaluated syllables |
| --- | ---: | ---: | ---: | ---: |
| train | 1,132 | 130,848 | 5,467 | 5,328 |
| val | 123 | 13,561 | 579 | 568 |
| test | 314 | 33,642 | 1,376 | 1,327 |

All splits passed input loading, finite-feature and reset-length consistency checks.
Model accuracy was evaluated on the test split.

## Inference

| Mode | Frame budget | Batches | Correct / evaluated | Accuracy | Wall time |
| --- | ---: | ---: | ---: | ---: | ---: |
| packing | 80,000 | 1 | 1,273 / 1,327 | 95.93% | 10.89 s |
| packing | 20,000 | 2 | 1,274 / 1,327 | 96.01% | 10.85 s |
| single | — | 314 | 1,274 / 1,327 | 96.01% | 29.95 s |

These runs used the downloaded dataset with `HF_HUB_OFFLINE=1`.
Wall time includes uv/Python startup, loading, inference, output and validation
recording. Dependency and Triton caches were reused. Each configuration was run once.

All modes produced 1,376 syllable results, including 49 blank predictions for
syllables with fewer than 5 valid F0 frames.

### Packing versus single

| Frame budget | Different predicted tones | Maximum absolute probability difference |
| --- | ---: | ---: |
| 80,000 | 1 | 0.017440617 |
| 20,000 | 0 | 0.026373088 |

The differing syllable was `Core00070_0478_CI_2`, index `0`:
packing 80,000 predicted tone **1**; single and the reference were tone **2**.
BF16 predictions can vary with batch layout.

## Checks

| Check | Result |
| --- | --- |
| Unit tests | 17 / 17 passed |
| isort / Ruff / uv lockfile | Passed |
| Fresh Hugging Face cache | Downloaded the three filtered CSVs and `f0/f0_praat.csv`, `f0/f0_praat.npy` |
| Fresh-download inference | 1,273 / 1,327 correct; 17.91 s including download |
| Offline cache reuse | All three inference configurations passed |
| Fresh versus offline packing | Identical predicted tones |
| Sample IDs, syllable indices and reference labels | Exact match to dataset order in all modes |
| CSV file and stdout | Correct columns; probabilities preserved exactly when parsed back |
| Single terminal output | All rows checked; probabilities rounded to four decimals; missing values shown as `-`; no CSV file |
| Accuracy summary | Correct counts and percentage on stderr |

## IPU boundary check

Tested the first two test IPUs: `Core00223_0781_S1_2` and `Core00217_0963_CI_2`.
Changed either IPU's F0 and duration features, then compared the untouched IPU's
logits in the packed pair.

| Precision | First IPU maximum logit change | Second IPU maximum logit change | Untouched IPU tones |
| --- | ---: | ---: | --- |
| BF16 | 0.00390625 | 0.00390625 | Unchanged |
| FP32, TF32 disabled | 0.0000112057 | 0.0000171661 | Unchanged |

FP32 comparisons passed with `rtol=1e-4`, `atol=1e-4`. Repeating identical input
produced zero logit difference in both precisions. These are two-IPU spot checks;
BF16 logits were not bitwise identical after changing the neighboring IPU.

## Commands

```bash
uv lock --check --offline
uv run --locked isort --check-only .
uv run --locked infer.py --output predictions.csv
uv run --locked infer.py --token-batch-size 20000
uv run --locked infer.py --mode single
```

For offline runs, set `HF_HUB_CACHE` to the downloaded cache and `HF_HUB_OFFLINE=1`.
`main` may resolve to a different dataset revision in future runs.
