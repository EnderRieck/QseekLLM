# Model Benchmark Comparison

Main rows compare Qwen3-1.7B Base, Qwen3-1.7B Instruct, and Ours (S4-1140). The third row reports Ours RL (v2-gs300) separately.

- Combined figure: `model_benchmark_comparison.png`
- Pass@1 figure: `model_benchmark_pass1.png`
- Pass@8 figure: `model_benchmark_pass8.png`
- Raw data: `model_benchmark_scores.csv`

## Pass@1 (%)

| benchmark | Qwen3-1.7B Base | Qwen3-1.7B Instruct | Ours (S4-1140) | Ours RL (v2-gs300) |
|---|---:|---:|---:|
| SVAMP | 28.7 | 84.0 | 40.7 | 40.3 |
| GSM8K | 48.1 | 76.9 | 18.7 | 19.1 |
| GSM-Plus | 34.7 | 54.6 | 9.7 | 9.5 |
| CMATH | 34.5 | 63.7 | 40.5 | 45.0 |
| MATH500 | 52.6 | 34.0 | 4.0 | 5.8 |
| CC-reserved | 50.4 | 66.3 | 73.3 | 66.9 |

## Pass@8 (%)

| benchmark | Qwen3-1.7B Base | Qwen3-1.7B Instruct | Ours (S4-1140) | Ours RL (v2-gs300) |
|---|---:|---:|---:|
| SVAMP | 90.7 | 91.0 | 74.3 | 62.7 |
| GSM8K | 92.0 | 89.8 | 44.8 | 38.7 |
| GSM-Plus | 74.1 | 69.2 | 28.3 | 23.8 |
| CMATH | 86.0 | 91.8 | 61.9 | 55.5 |
| MATH500 | 78.6 | 49.6 | 23.0 | 17.2 |
| CC-reserved | 82.1 | — | 84.4 | 74.2 |

Note: Qwen Instruct cc-reserved Pass@8 was not run; its Pass@1 is recomputed from the per-sample dump. Ours RL uses the v2-gs300 final-eval checkpoint.
