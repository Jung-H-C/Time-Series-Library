# DCSPG

Data-Conditioned Symbolic Proxy Generator for the 53-dataset fixed-split
experiment.

The default setup uses:

- 39 training, 8 validation, and 6 test datasets;
- 7 non-empty training clusters from the K=8 catch22 clustering, with a
  randomly permuted `[4, 4, 4, 5, 5, 5, 5]` episode allocation per batch,
  giving every active cluster an expected contribution of `32 / 7` episodes;
- dataset sampling with replacement and 16 support samples without replacement;
- one `[22, 2]` `[mean, std]` catch22 statistic tensor per episode;
- 16 teacher formulas sampled without replacement per episode;
- percentile-weighted teacher cross-entropy;
- 100 optimizer iterations per epoch;
- terminal summaries every 20 iterations with per-dataset selection counts;
- validation on the 8 datasets after every epoch, using 5 fixed-support
  greedy episodes per dataset and the mean of their individual Spearman scores;
- dataset-wise weighted validation CE on the same fixed episodes, evaluated
  with teacher forcing against every teacher formula and macro-averaged across
  the 8 datasets;
- early stopping with patience 10 using mean `proxy_test` Spearman against
  negative MSE;
- 10 fixed-support generated formulas per test dataset, with support indices
  independent of the training seed.
- test-time `Avg. CE` for each generated formula, computed by teacher-forcing
  that formula against all 10 fixed support episodes from the same dataset
  (10 formulas x 10 episodes = 100 CE evaluations per dataset).

The encoder input token for catch22 feature `j` is:

```text
ValueEmbedding([mean_j, log1p(std_j)]) + FeatureEmbedding(j)
```

The default validation datasets are:

```text
electricity_hourly_dataset
current_velocity__15T
Coastal_T_S__15T
Finland_Traffic__15T
MetroPT-3__5T
Port_Activity__D
SG_Carpark__15T
solar_4_seconds_dataset
```

The default test datasets are `ECL`, `Exchange`, `Illness`, `Traffic`,
`Weather`, and `ETTh1`. CLI names are resolved to the corresponding TS feature,
GroundTruth, and proxy-score dataset names.

## Training

```bash
conda run -n tslib_nightly python train_dcspg_framework.py --gpu-id 0
```

Override validation and test datasets with comma-separated lists:

```bash
conda run -n tslib_nightly python train_dcspg_framework.py \
  --validation-datasets "dataset_a,dataset_b,dataset_c,dataset_d,dataset_e,dataset_f,dataset_g,dataset_h" \
  --test-datasets "ECL,Exchange,Illness,Traffic,Weather,ETTh1" \
  --gpu-id 0
```

Exactly 8 validation and 6 test datasets are required, with one validation
dataset from each cluster. Every remaining training dataset must have a cluster
assignment in
`catch22/dataset_centroid_clusters_47_pca90_k8/cluster_summary_k8.csv`.

The default symbolic limits are `--max-formula-len 12` and
`--max-stack-depth 4`, with `--max-unary-chain 2`. The unary-chain limit is
enforced by the shared RPN grammar during teacher-forcing training, greedy or
sampled generation, and beam search. Across the 53 GroundTruth files, all 51,579 formulas
fit within 10 RPN tokens including EOS and a maximum execution stack depth of
4, and no formula exceeds two consecutive unary operators. The length limit
also leaves room for sequence-boundary handling.

Each run writes:

- validation-ranked `best_checkpoint.pth`, `second_best_checkpoint.pth`,
  `third_best_checkpoint.pth`, `fourth_best_checkpoint.pth`, and
  `fifth_best_checkpoint.pth` (additional ranks use
  `rank_006_checkpoint.pth`, etc.);
- `averaged_checkpoint.pth`, produced by uniform weight averaging of the
  available top validation checkpoints and used for the final test stage.
  The number retained and averaged is configured with
  `--averaged-checkpoint-count` (default `3`);
- `top_checkpoints.csv` with checkpoint ranks, epochs, and validation scores;
- `last.pt`;
- `checkpoints/epoch_XXXX.pth` for only the most recent five epochs;
- `log/epoch_metrics.csv` and `log/train_validation_curve.png`;
- `log/validation_weighted_ce.csv` and
  `log/validation_weighted_ce_curve.png`;
- `train_history.csv`;
- `validation_support_samples.csv`, `validation_results.csv`, and
  `validation_summary.csv`;
- `test_support_samples.csv` containing the seed-independent fixed support
  indices shared by both test evaluations;
- `test_results_best_checkpoint.csv` and
  `test_summary_best_checkpoint.csv` for the best single checkpoint;
- `test_results_averaged_checkpoint.csv` and
  `test_summary_averaged_checkpoint.csv` for the averaged checkpoint;
- `run_config.json` and `summary.json`.

Invalid generated formulas receive the configurable validation/test averaging
penalty `--invalid-spearman-penalty` (default `-1.0`) while retaining their
error reason in the result CSV.

Use `--early-stopping-criterion celoss` (or the equivalent
`--validation-criterion celoss`) to minimize the dataset-macro weighted
validation CE for early stopping and top-checkpoint ranking. The default is
`spearman_corr`. Spearman generation, evaluation, logging, and terminal output
remain enabled in both modes.
