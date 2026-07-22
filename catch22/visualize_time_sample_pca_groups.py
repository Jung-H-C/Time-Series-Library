from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tslib_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.ipc as ipc
import pycatch22


REPO_ROOT = Path(__file__).resolve().parents[1]

META_COLUMNS = [
    "group_id",
    "group_rank",
    "dataset_id",
    "dataset_name",
    "frequency",
    "dataset_dir",
    "arrow_path",
    "sample_rank",
    "candidate_index",
    "row_index",
    "variate_index",
    "item_id",
    "variate_name",
    "series_length_original",
    "series_length_used",
    "downsampled",
    "sampled_with_replacement",
]


@dataclass(frozen=True)
class DatasetUnit:
    dataset_name: str
    frequency: str
    dataset_dir: Path
    arrow_paths: tuple[Path, ...]

    @property
    def dataset_id(self) -> str:
        return f"{self.dataset_name}/{self.frequency}"


@dataclass(frozen=True)
class SeriesCandidate:
    candidate_index: int
    arrow_path: Path
    row_index: int
    variate_index: int
    item_id: str
    variate_name: str
    values: list[float | int | None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample 1D time series from each TIME dataset/frequency unit, extract catch22 "
            "features, shuffle dataset units, and save one 2D PCA visualization per group."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root. Default: inferred from this script path.",
    )
    parser.add_argument(
        "--time-root",
        type=Path,
        default=Path("TIME"),
        help="TIME dataset root. Relative paths are resolved under --repo-root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("catch22/time_sample_pca_groups"),
        help="Output directory. Relative paths are resolved under --repo-root.",
    )
    parser.add_argument("--seed", type=int, default=20260710, help="Random seed for dataset shuffling and sampling.")
    parser.add_argument(
        "--datasets-per-group",
        type=int,
        default=5,
        help="Number of dataset/frequency units per PCA plot.",
    )
    parser.add_argument(
        "--samples-per-dataset",
        type=int,
        default=5,
        help="Number of sampled 1D time series per dataset/frequency unit.",
    )
    parser.add_argument(
        "--max-series-length",
        type=int,
        default=50000,
        help=(
            "Maximum series length passed to pycatch22 after uniform downsampling. "
            "Use 0 or a negative value to disable downsampling."
        ),
    )
    parser.add_argument("--dpi", type=int, default=220, help="Output figure DPI.")
    parser.add_argument(
        "--max-groups",
        type=int,
        default=0,
        help="Debug option: process only the first N shuffled groups. 0 means all groups.",
    )
    return parser.parse_args()


def resolve_path(path: Path, base: Path) -> Path:
    return path if path.is_absolute() else base / path


def relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def discover_dataset_units(time_root: Path) -> list[DatasetUnit]:
    units: list[DatasetUnit] = []
    for dataset_dir in sorted(time_root.glob("*/*")):
        if not dataset_dir.is_dir():
            continue
        arrow_paths = tuple(sorted(dataset_dir.glob("data-*.arrow")))
        if not arrow_paths:
            continue
        units.append(
            DatasetUnit(
                dataset_name=dataset_dir.parent.name,
                frequency=dataset_dir.name,
                dataset_dir=dataset_dir,
                arrow_paths=arrow_paths,
            )
        )
    return units


def shuffled_groups(
    units: list[DatasetUnit],
    datasets_per_group: int,
    rng: np.random.Generator,
) -> list[list[DatasetUnit]]:
    if datasets_per_group <= 0:
        raise ValueError(f"datasets-per-group must be positive, got {datasets_per_group}.")
    order = rng.permutation(len(units))
    shuffled = [units[int(index)] for index in order]
    return [shuffled[start : start + datasets_per_group] for start in range(0, len(shuffled), datasets_per_group)]


def is_multivariate_target(target: object) -> bool:
    return isinstance(target, list) and len(target) > 0 and isinstance(target[0], list)


def iter_arrow_candidates(unit: DatasetUnit) -> Iterable[SeriesCandidate]:
    candidate_index = 0
    row_offset = 0

    for arrow_path in unit.arrow_paths:
        with arrow_path.open("rb") as handle:
            reader = ipc.open_stream(handle)
            for batch in reader:
                column_names = set(batch.schema.names)
                targets = batch.column(batch.schema.get_field_index("target")).to_pylist()
                item_ids = (
                    batch.column(batch.schema.get_field_index("item_id")).to_pylist()
                    if "item_id" in column_names
                    else ["" for _ in range(batch.num_rows)]
                )
                variate_name_rows = (
                    batch.column(batch.schema.get_field_index("variate_names")).to_pylist()
                    if "variate_names" in column_names
                    else [None for _ in range(batch.num_rows)]
                )

                for local_row_index, target in enumerate(targets):
                    row_index = row_offset + local_row_index
                    item_id = "" if item_ids[local_row_index] is None else str(item_ids[local_row_index])
                    variate_names = variate_name_rows[local_row_index]

                    if is_multivariate_target(target):
                        for variate_index, values in enumerate(target):
                            variate_name = f"variate_{variate_index}"
                            if isinstance(variate_names, list) and variate_index < len(variate_names):
                                variate_name = str(variate_names[variate_index])
                            yield SeriesCandidate(
                                candidate_index=candidate_index,
                                arrow_path=arrow_path,
                                row_index=row_index,
                                variate_index=variate_index,
                                item_id=item_id,
                                variate_name=variate_name,
                                values=values,
                            )
                            candidate_index += 1
                    elif isinstance(target, list):
                        yield SeriesCandidate(
                            candidate_index=candidate_index,
                            arrow_path=arrow_path,
                            row_index=row_index,
                            variate_index=0,
                            item_id=item_id,
                            variate_name="target",
                            values=target,
                        )
                        candidate_index += 1

                row_offset += batch.num_rows


def reservoir_sample_candidates(
    unit: DatasetUnit,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[list[SeriesCandidate], int, bool]:
    if n_samples <= 0:
        raise ValueError(f"samples-per-dataset must be positive, got {n_samples}.")

    reservoir: list[SeriesCandidate] = []
    total_candidates = 0
    for candidate in iter_arrow_candidates(unit):
        total_candidates += 1
        if len(reservoir) < n_samples:
            reservoir.append(candidate)
            continue
        replace_at = int(rng.integers(0, total_candidates))
        if replace_at < n_samples:
            reservoir[replace_at] = candidate

    if total_candidates == 0:
        return [], 0, False

    if total_candidates < n_samples:
        selected_indices = rng.integers(0, total_candidates, size=n_samples)
        return [reservoir[int(index)] for index in selected_indices], total_candidates, True

    rng.shuffle(reservoir)
    return reservoir, total_candidates, False


def values_to_array(values: list[float | int | None]) -> np.ndarray:
    return np.asarray([np.nan if value is None else value for value in values], dtype=np.float64)


def interpolate_missing(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    if finite.all():
        return values
    if finite.sum() == 0:
        raise ValueError("time series contains no finite values")
    if finite.sum() == 1:
        return np.full_like(values, float(values[finite][0]), dtype=np.float64)
    indices = np.arange(len(values), dtype=np.float64)
    return np.interp(indices, indices[finite], values[finite])


def maybe_downsample(values: np.ndarray, max_length: int) -> tuple[np.ndarray, bool]:
    if max_length <= 0 or len(values) <= max_length:
        return values, False
    indices = np.linspace(0, len(values) - 1, num=max_length, dtype=np.int64)
    return values[indices], True


def prepare_series(values: list[float | int | None], max_length: int) -> tuple[np.ndarray, int, bool]:
    original = values_to_array(values)
    if len(original) < 3:
        raise ValueError(f"time series is too short for catch22: length={len(original)}")
    cleaned = interpolate_missing(original)
    sampled, downsampled = maybe_downsample(cleaned, max_length=max_length)
    if len(sampled) < 3:
        raise ValueError(f"downsampled time series is too short for catch22: length={len(sampled)}")
    return sampled.astype(np.float64, copy=False), len(original), downsampled


def catch22_features(values: np.ndarray) -> tuple[list[str], np.ndarray]:
    result = pycatch22.catch22_all(values, catch24=False, short_names=True)
    return list(result["short_names"]), np.asarray(result["values"], dtype=np.float64)


def collect_feature_rows(
    groups: list[list[DatasetUnit]],
    args: argparse.Namespace,
    repo_root: Path,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(args.seed + 1)
    rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    assignments: list[dict[str, object]] = []
    feature_names: list[str] | None = None

    for group_index, group in enumerate(groups, start=1):
        group_id = f"group_{group_index:03d}"
        for group_rank, unit in enumerate(group, start=1):
            assignments.append(
                {
                    "group_id": group_id,
                    "group_rank": group_rank,
                    "dataset_id": unit.dataset_id,
                    "dataset_name": unit.dataset_name,
                    "frequency": unit.frequency,
                    "dataset_dir": relative_path(unit.dataset_dir, repo_root),
                    "n_arrow_files": len(unit.arrow_paths),
                }
            )
            selected, total_candidates, sampled_with_replacement = reservoir_sample_candidates(
                unit=unit,
                n_samples=args.samples_per_dataset,
                rng=rng,
            )
            print(
                f"{group_id}: sampled {len(selected)} / {total_candidates} candidates from {unit.dataset_id}"
                + (" with replacement" if sampled_with_replacement else "")
            )

            if not selected:
                errors.append(
                    {
                        "group_id": group_id,
                        "dataset_id": unit.dataset_id,
                        "sample_rank": "",
                        "candidate_index": "",
                        "error": "dataset unit has no target time series candidates",
                    }
                )
                continue

            for sample_rank, candidate in enumerate(selected, start=1):
                try:
                    prepared, original_length, downsampled = prepare_series(
                        candidate.values,
                        max_length=args.max_series_length,
                    )
                    short_names, features = catch22_features(prepared)
                except Exception as exc:
                    errors.append(
                        {
                            "group_id": group_id,
                            "dataset_id": unit.dataset_id,
                            "sample_rank": sample_rank,
                            "candidate_index": candidate.candidate_index,
                            "error": str(exc),
                        }
                    )
                    continue

                if feature_names is None:
                    feature_names = short_names
                elif feature_names != short_names:
                    raise RuntimeError(f"catch22 feature order changed for {unit.dataset_id}")

                row: dict[str, object] = {
                    "group_id": group_id,
                    "group_rank": group_rank,
                    "dataset_id": unit.dataset_id,
                    "dataset_name": unit.dataset_name,
                    "frequency": unit.frequency,
                    "dataset_dir": relative_path(unit.dataset_dir, repo_root),
                    "arrow_path": relative_path(candidate.arrow_path, repo_root),
                    "sample_rank": sample_rank,
                    "candidate_index": candidate.candidate_index,
                    "row_index": candidate.row_index,
                    "variate_index": candidate.variate_index,
                    "item_id": candidate.item_id,
                    "variate_name": candidate.variate_name,
                    "series_length_original": original_length,
                    "series_length_used": len(prepared),
                    "downsampled": downsampled,
                    "sampled_with_replacement": sampled_with_replacement,
                }
                for name, value in zip(short_names, features):
                    row[name] = float(value)
                rows.append(row)

    if feature_names is None:
        raise RuntimeError("No catch22 feature rows were collected.")

    return pd.DataFrame(rows), feature_names, pd.DataFrame(assignments), pd.DataFrame(errors)


def numeric_feature_frame(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    features = df[feature_names].apply(pd.to_numeric, errors="coerce")
    return features.replace([np.inf, -np.inf], np.nan)


def zscore_and_impute(features: pd.DataFrame) -> pd.DataFrame:
    imputed = features.copy()
    means = imputed.mean(axis=0, skipna=True)
    imputed = imputed.fillna(means).fillna(0.0)
    stds = imputed.std(axis=0, ddof=0).replace(0.0, np.nan)
    return ((imputed - imputed.mean(axis=0)) / stds).fillna(0.0)


def pca_2d(z_features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    matrix = z_features.to_numpy(dtype=float)
    if len(matrix) == 0:
        return np.zeros((0, 2), dtype=float), np.zeros(2, dtype=float)

    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    n_components = min(2, vt.shape[0])

    coords = np.zeros((len(centered), 2), dtype=float)
    explained = np.zeros(2, dtype=float)
    if n_components > 0:
        components = vt[:n_components].T
        coords[:, :n_components] = centered @ components
        variances = singular_values**2
        total_variance = variances.sum()
        if total_variance > 0:
            explained[:n_components] = variances[:n_components] / total_variance
    return coords, explained


def add_groupwise_pca(
    feature_df: pd.DataFrame,
    feature_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    z_frames: list[pd.DataFrame] = []
    pca_frames: list[pd.DataFrame] = []

    for _, group in feature_df.groupby("group_id", sort=False):
        features = numeric_feature_frame(group, feature_names)
        z_features = zscore_and_impute(features)
        z_features.columns = feature_names

        z_output = pd.concat(
            [group[META_COLUMNS].reset_index(drop=True), z_features.reset_index(drop=True)],
            axis=1,
        )
        z_frames.append(z_output)

        coords, explained = pca_2d(z_features)
        pca_output = group[META_COLUMNS].copy().reset_index(drop=True)
        pca_output["pc1"] = coords[:, 0]
        pca_output["pc2"] = coords[:, 1]
        pca_output["pc1_explained_ratio"] = float(explained[0])
        pca_output["pc2_explained_ratio"] = float(explained[1])
        pca_frames.append(pca_output)

    return pd.concat(z_frames, ignore_index=True), pd.concat(pca_frames, ignore_index=True)


def save_group_pca_plot(pca_df: pd.DataFrame, output_path: Path, dpi: int) -> None:
    dataset_ids = list(dict.fromkeys(pca_df["dataset_id"].tolist()))
    cmap = plt.get_cmap("tab10" if len(dataset_ids) <= 10 else "nipy_spectral")
    colors = {
        dataset_id: cmap(index / max(1, len(dataset_ids) - 1))
        for index, dataset_id in enumerate(dataset_ids)
    }

    fig, ax = plt.subplots(figsize=(8.0, 6.0), constrained_layout=True)
    for dataset_id in dataset_ids:
        group = pca_df[pca_df["dataset_id"] == dataset_id]
        color = colors[dataset_id]
        ax.scatter(
            group["pc1"],
            group["pc2"],
            s=52,
            alpha=0.84,
            color=color,
            label=dataset_id,
            edgecolors="white",
            linewidths=0.45,
        )
        ax.text(
            float(group["pc1"].mean()),
            float(group["pc2"].mean()),
            dataset_id,
            fontsize=8,
            color=color,
            weight="bold",
        )

    pc1_ratio = float(pca_df["pc1_explained_ratio"].iloc[0])
    pc2_ratio = float(pca_df["pc2_explained_ratio"].iloc[0])
    group_id = str(pca_df["group_id"].iloc[0])
    ax.axhline(0.0, color="#bbbbbb", linewidth=0.8, zorder=0)
    ax.axvline(0.0, color="#bbbbbb", linewidth=0.8, zorder=0)
    ax.set_xlabel(f"PC1 ({pc1_ratio:.2%})")
    ax.set_ylabel(f"PC2 ({pc2_ratio:.2%})")
    ax.set_title(f"TIME catch22 PCA: {group_id}")
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.55)
    ax.legend(loc="best", fontsize=7)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def write_error_csv(errors: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["group_id", "dataset_id", "sample_rank", "candidate_index", "error"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in errors.to_dict("records"):
            writer.writerow({key: record.get(key, "") for key in fieldnames})


def write_summary(
    output_path: Path,
    args: argparse.Namespace,
    units: list[DatasetUnit],
    groups: list[list[DatasetUnit]],
    feature_df: pd.DataFrame,
    error_df: pd.DataFrame,
) -> None:
    lines = [
        "TIME catch22 grouped PCA summary",
        "",
        f"seed: {args.seed}",
        f"dataset_frequency_units: {len(units)}",
        f"groups: {len(groups)}",
        f"datasets_per_group: {args.datasets_per_group}",
        f"samples_per_dataset: {args.samples_per_dataset}",
        f"max_series_length: {args.max_series_length}",
        f"feature_rows: {len(feature_df)}",
        f"skipped_samples: {len(error_df)}",
        "",
        "Files",
        "- group_assignments.csv: shuffled dataset/frequency unit order and group membership.",
        "- sample_catch22_features.csv: raw catch22 feature rows.",
        "- sample_catch22_features_zscored_by_group.csv: group-wise z-scored features used for PCA.",
        "- sample_catch22_pca_2d.csv: group-wise 2D PCA coordinates.",
        "- plots/group_XXX_pca_2d.png: one PCA visualization per shuffled group.",
        "- skipped_samples.csv: samples skipped during extraction, if any.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    time_root = resolve_path(args.time_root, repo_root).resolve()
    output_dir = resolve_path(args.output_dir, repo_root).resolve()

    units = discover_dataset_units(time_root)
    if not units:
        raise RuntimeError(f"No TIME dataset/frequency units with data-*.arrow were found under {time_root}")

    shuffle_rng = np.random.default_rng(args.seed)
    groups = shuffled_groups(units, args.datasets_per_group, shuffle_rng)
    if args.max_groups > 0:
        groups = groups[: args.max_groups]

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_df, feature_names, assignment_df, error_df = collect_feature_rows(groups, args, repo_root)
    z_feature_df, pca_df = add_groupwise_pca(feature_df, feature_names)

    assignment_df.to_csv(output_dir / "group_assignments.csv", index=False)
    feature_df.to_csv(output_dir / "sample_catch22_features.csv", index=False, float_format="%.9f")
    z_feature_df.to_csv(
        output_dir / "sample_catch22_features_zscored_by_group.csv",
        index=False,
        float_format="%.9f",
    )
    pca_df.to_csv(output_dir / "sample_catch22_pca_2d.csv", index=False, float_format="%.9f")
    write_error_csv(error_df, output_dir / "skipped_samples.csv")

    plots_dir = output_dir / "plots"
    for group_id, group_pca_df in pca_df.groupby("group_id", sort=False):
        save_group_pca_plot(group_pca_df, plots_dir / f"{group_id}_pca_2d.png", dpi=args.dpi)

    write_summary(output_dir / "summary.txt", args, units, groups, feature_df, error_df)

    print(f"Discovered {len(units)} TIME dataset/frequency units under {time_root}")
    print(f"Wrote {len(groups)} grouped PCA plots to {plots_dir}")
    print(f"Feature rows: {len(feature_df)}")
    print(f"Skipped samples/errors: {len(error_df)}")
    print(f"Output directory: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
