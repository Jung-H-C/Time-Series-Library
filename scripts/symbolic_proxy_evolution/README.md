# Symbolic Proxy Evolution

Evolutionary search framework for data-conditioned symbolic proxy formulas over
the `GroundTruth/<backbone>/*.csv` benchmark files.

## Search Space

Formulas are searched as trees and saved in the archive as postfix/RPN tokens
ending with `<EOS>`.

- Proxy tokens: `MParams`, `L2-Norm`, `GFLOPs`, `Grad_Norm`, `ZiCo`, `Fisher`,
  `GraSP`, `Jacov`, `Jacob_fro`, `plain`, `Snip`, `GSynFlow`
- Unary tokens: `Log`, `Sqrt`, `Square`, `Identity`, `Negative`
- Binary tokens: `Mul`, `Add`, `Sub`, `Div`
- Constraints: `max_binary_ops=3`, `max_unary_chain=2`, `max_tokens=15`
- Redundant unary chains are rejected during tree validation:
  `Negative(Negative(.))`, `Square(Square(.))`, `Sqrt(Sqrt(.))`, and
  `Identity(Identity(.))`.

For standard RPN, the formula
`sqrt(log(MParams) + Snip) * Fisher^2 / -Grad_Norm` is:

```text
MParams Log Snip Identity Add Sqrt Fisher Square Grad_Norm Negative Div Mul Identity <EOS>
```

## Example

Run from the repository root in the intended environment:

```bash
conda run -n tslib_nightly python scripts/symbolic_proxy_evolution/evolve_symbolic_proxy.py \
  --backbone autoformer \
  --dataset ETTh1 \
  --seed 2026 \
  --num-workers 0 \
  --max-generations 100 \
  --visualize-archive \
  --visualize-top-k 20
```

Default outputs are written to:

```text
archive/symbolic_proxy_evolution/<backbone>/<dataset>/seed_<seed>/run_<YYYYmmdd_HHMMSS>/
```

Main outputs:

- `generation_best.csv`: top-1 formula per generation with rounded fitness.
- `generation_population.csv`: every evaluated formula in every generation.
- `archive.csv`: deduplicated archive, sorted by fitness, capped by
  `--archive-size`.
- `visualizations/tree_svg/`: dependency-free tree diagrams for ranked archive
  entries.
- `visualizations/tree_png/`: PNG diagrams when `matplotlib` is available.
- `visualizations/archive_latex.tex`: ranked LaTeX formula table.

## Important CLI Knobs

```bash
--backbone autoformer
--dataset ETTh1
--seed 2026
--max-generations 100
--population-size 200
--archive-size 1000
--archive-threshold-margin 0.05
--target-metric mse
--target-direction minimize
--max-binary-ops 3
--max-unary-chain 2
--max-tokens 15
--num-workers 0
--proxy-score-decimals 12
--max-abs-proxy-score 1e12
```

`--num-workers 0` automatically uses CPU worker processes up to the population
size. Use `--num-workers 1` for sequential evaluation, or set an explicit value
such as `--num-workers 8`.

## Monash + TIME: 47 datasets and adaptive seeds

`ea_on_different_seeds.sh` discovers the 47 Autoformer CSVs under
`proxy_scores/monash_time/`. Each CSV must contain 300 candidates split into
`proxy_train`, `proxy_valid`, and `proxy_test`, together with the 12 zero-cost
proxy columns and `mse`.

The launcher first runs seeds 2027 through 2226 for every dataset. It then reads
the top 10 rows from each seed's `visualizations/archive_latex.tex`, normalizes
formula whitespace, and counts unique formulas per dataset. A dataset whose
count is at most 1,000 receives seed 2227, then 2228, and so on until its count
is strictly greater than 1,000. Datasets that already exceeded the target are
not scheduled again. A dataset is capped at 300 total seeds by default (seeds
2027 through 2326). If it still has at most 1,000 unique formulas at that point,
its summary status becomes `seed_cap_reached`; that dataset stops without
blocking or aborting the remaining datasets.

```bash
conda activate tslib_nightly
nohup bash scripts/symbolic_proxy_evolution/ea_on_different_seeds.sh \
  > log/monash_time_ea.out 2>&1 &
```

The stable default run directory is `run_monash_time_unique1000`. Successful
runs contain `.ea_complete`; rerunning the launcher skips those seeds and safely
resumes the collection. Set a different `RUN_TAG` whenever material EA
hyperparameters change, so outputs from different configurations are not mixed.
The per-dataset counts and last completed seed are written to
`archive/symbolic_proxy_evolution/autoformer/formula_collection_<RUN_TAG>.csv`.

Useful environment variables:

```bash
NUM_WORKERS=1 MAX_PARALLEL=32 RUN_TAG=monash_time_unique1000 \
  bash scripts/symbolic_proxy_evolution/ea_on_different_seeds.sh
```

- `MAX_PARALLEL` is the maximum number of independent dataset/seed EA runs that
  execute simultaneously.
- `NUM_WORKERS` is the number of formula-evaluation processes inside each EA
  run. `1` disables the inner process pool; `0` auto-selects workers and should
  generally not be combined with a large `MAX_PARALLEL`.
- The approximate CPU process demand is `MAX_PARALLEL * NUM_WORKERS`, plus the
  EA parent processes. With many independent seeds, a larger `MAX_PARALLEL` and
  a small `NUM_WORKERS` usually avoids nested-process overhead.
- `PLAN_ONLY=1` validates and prints the configuration without launching EA.
- `DATASETS="Monash__weather_dataset TIME__Crypto__D"` limits a run for testing.
- `MAX_SEEDS_PER_DATASET=300` limits the total number of seeds for each dataset.
  With the default start seed, the final allowed seed is 2326.

Proxy score guards:

- `--proxy-score-decimals`: rounds each computed formula score before Spearman
  correlation. Use `-1` to disable rounding.
- `--max-abs-proxy-score`: marks a formula as invalid if any candidate model's
  computed proxy score exceeds this absolute value. Invalid formulas are written
  to `generation_population.csv` with `status=invalid` and are not added to
  `archive.csv`.

Fitness uses two fixed proxy-train folds of 25 rows each. The score for a
formula is the minimum Spearman correlation across the two folds.

The initial generation contains all single proxies plus exhaustive pairwise
binary formulas over the active binary operators. With `Div` enabled this is
`12 + 66 * 4 = 276` formulas.

The default next-generation recipe is fixed to 200 candidates:

- 5 elites
- 45 mutations sampled from the top-30 formulas
- 25 archive-based mutations, with random-tree fallback when the archive is small
- 25 crossover children sampled from top-30 parents
- 10 one-hop neighbors of the top-1 formula
- 90 random exploration trees under the default population size, using binary-op
  quotas `0:5`, `1:10`, `2:25`, `3:50`
