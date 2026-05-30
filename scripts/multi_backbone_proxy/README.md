# Multi-Backbone Zero-Cost Proxy Experiments

This folder is a standalone scaffold for the new long-term forecasting project:
sample candidate models across multiple backbones, split them into proxy-train and
proxy-eval pools, then full-train/test selected candidates through `run.py`.

## Target Scope

- Task: `long_term_forecast`
- Datasets: `ECL`, `ETTh1`, `Exchange`, `ILI`, `Traffic`, `Weather`
- Backbones: `Autoformer`, `Crossformer`, `FiLM`, `MICN`, `Mamba`, `PatchTST`,
  `TimesNet`, `Transformer`, `DLinear`
- Default split: 10 candidates per backbone, 5 `proxy_train` and 5 `proxy_eval`

## 1. Sample Candidates

Run from the `Time-Series-Library` repo root:

```bash
python scripts/multi_backbone_proxy/sample_candidates.py \
  --output candidates/multi_backbone_proxy_90.json \
  --seed 2026 \
  --num-per-backbone 10
```

To keep the common training/search knobs fixed for every candidate:

```bash
python scripts/multi_backbone_proxy/sample_candidates.py \
  --output candidates/multi_backbone_proxy_90_fixed_common.json \
  --seed 2026 \
  --num-per-backbone 10 \
  --seq-len 96 \
  --dropout 0.1 \
  --learning-rate 0.0003
```

The JSON stores dataset-independent candidate definitions:

```json
{
  "candidate_id": "PatchTST_003",
  "backbone": "PatchTST",
  "split": "proxy_train",
  "run_args": {
    "seq_len": 96,
    "d_model": 256,
    "e_layers": 2,
    "d_ff": 512,
    "learning_rate": 0.0003
  }
}
```

For every backbone, candidate index `000` is reserved for the representative
default configuration and is marked with `"is_default": true`. Random sampling
fills the remaining candidates while avoiding duplicates.

## 2. Dry-Run Expanded Commands

Print the concrete `run.py` commands without launching training:

```bash
python scripts/multi_backbone_proxy/run_candidates.py \
  --candidates candidates/multi_backbone_proxy_90.json \
  --datasets ECL ETTh1 Exchange ILI Traffic Weather \
  --splits proxy_train \
  --pred-lens 96 \
  --limit 5
```

By default, if `--pred-lens` is omitted, each dataset uses its canonical horizons:
`96 192 336 720`, except `ILI`, which uses `24 36 48 60`.

Dataset-specific configuration is applied when expanding sampled candidates into
`run.py` commands. For example, this keeps most datasets at the sampled/default
`seq_len`, but runs `ILI` with `seq_len=36`, limits `Exchange` to `pred_len=96`,
and uses two ILI horizons:

```bash
python scripts/multi_backbone_proxy/run_candidates.py \
  --candidates candidates/multi_backbone_proxy_90.json \
  --datasets ECL ETTh1 Exchange ILI Traffic Weather \
  --splits proxy_train \
  --dataset-seq-len ILI=36 \
  --dataset-pred-lens Exchange=96 \
  --dataset-pred-lens ILI=24,36 \
  --limit 10
```

You can also force a global sequence length and override only the exceptional
datasets:

```bash
python scripts/multi_backbone_proxy/run_candidates.py \
  --candidates candidates/multi_backbone_proxy_90.json \
  --datasets ECL ETTh1 Exchange ILI Traffic Weather \
  --splits proxy_eval \
  --fixed-seq-len 96 \
  --dataset-seq-len ILI=36 \
  --dataset-pred-lens ILI=24,36,48,60
```

## 3. Execute Full Train/Test

Launch jobs and write one log per candidate run:

```bash
  --dataset-seq-len ILI=36 \
  --dataset-pred-lens ILI=24,36,48,60
```

To run multiple candidates concurrently, increase `--n_jobs`. Each worker thread
launches one independent `run.py` process, then takes the next candidate when it
finishes:

```bash
python scripts/multi_backbone_proxy/run_candidates.py \
  --candidates candidates/autoformer_100_sl96.json \
  --datasets ECL ETTh1 Exchange ILI Traffic Weather \
  --pred-lens 96 \
  --gpu 0 \
  --n_jobs 8 \
  --manifest logs/multi_backbone_proxy/ecl_train_manifest.jsonl \
  --execute \
  --dataset-seq-len ILI=36 \
  --dataset-pred-lens ILI=36
```

Be careful with `--n_jobs` on GPU runs: all workers use the same `--gpu` value
unless you launch separate runner processes with different filters/GPU ids.

Useful filters:

```bash
--backbones PatchTST DLinear
--candidate-ids PatchTST_003 DLinear_007
--splits proxy_eval
--skip-existing
--keep-going
```

For a conda environment:

```bash
--python-cmd conda run -n tslib python
```

The runner reuses the existing `run.py`, `Exp_Long_Term_Forecast`, data loaders,
checkpointing, and result saving. Existing long-term shell scripts are only used as
reference for dataset defaults; commands are generated directly from candidate JSON.

Note: DLinear candidates keep the repo default shared linear layers instead of
`--individual`, because `individual=True` can become very large on Traffic/ECL.

## 4. Export Result Metrics

Collect result folders whose names start with a prefix into one CSV:

```bash
python scripts/multi_backbone_proxy/export_results_csv.py \
  long_term_forecast_mbproxy_ECL_Autoformer \
  --candidates candidates/autoformer_100_sl96.json
```

Default output:

```text
results/long_term_forecast_mbproxy_ECL_Autoformer_summary.csv
```

The CSV has one row per candidate and includes parsed identifiers, candidate
hyperparameters from the JSON, and metrics loaded from each `metrics.npy`:
`mae`, `mse`, `rmse`, `mape`, `mspe`.

To choose the output path explicitly:

```bash
python scripts/multi_backbone_proxy/export_results_csv.py \
  long_term_forecast_mbproxy_ECL_Autoformer \
  --candidates candidates/autoformer_100_sl96.json \
  --output results/autoformer_100_ecl_sl96_pl96_summary.csv
```
