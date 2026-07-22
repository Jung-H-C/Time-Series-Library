from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Mapping

import numpy as np
import torch

from DCSPG.config import DCSPGConfig
from DCSPG.data import Catch22FeatureStore
from DCSPG.experiment import build_default_grammar
from DCSPG.model import DCSPGModel
from DCSPG.targets import DEFAULT_DATASET_NAME_MAP
from DCSPG.vocabulary import SymbolicVocabulary


REPO_ROOT = Path(__file__).resolve().parents[1]
SYMBOLIC_PROXY_DIR = REPO_ROOT / "scripts" / "symbolic_proxy_evolution"
if str(SYMBOLIC_PROXY_DIR) not in sys.path:
    sys.path.insert(0, str(SYMBOLIC_PROXY_DIR))

from evolve_symbolic_proxy import normalize_proxy_scores, spearman_correlation  # noqa: E402
from symbolic_tree import PROXY_TO_COLUMN, parse_rpn  # noqa: E402


REVERSE_DATASET_NAME_MAP = {value: key for key, value in DEFAULT_DATASET_NAME_MAP.items()}


@dataclass(frozen=True)
class DCSPGTestResult:
    checkpoint_path: str
    dataset: str
    ts_dataset: str
    condition_dataset: str
    condition_ts_dataset: str
    evaluation_dataset: str
    evaluation_ts_dataset: str
    benchmark_dataset: str
    benchmark_csv: str
    split: str
    split_count: int
    support_indices: tuple[int, ...]
    rpn_tokens: str
    infix: str
    latex: str
    spearman_neg_mse: float
    invalid_reason: str
    beam_size: int = 1
    beam_valid_count: int = 0
    beam_rpn_tokens: tuple[str, ...] = ()
    beam_infix: tuple[str, ...] = ()
    beam_latex: tuple[str, ...] = ()
    beam_log_probs: tuple[float, ...] = ()
    beam_spearman_neg_mse: tuple[float, ...] = ()
    beam_invalid_reasons: tuple[str, ...] = ()

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2)


def resolve_dataset_names(dataset_name: str) -> tuple[str, str]:
    if dataset_name in DEFAULT_DATASET_NAME_MAP:
        return dataset_name, DEFAULT_DATASET_NAME_MAP[dataset_name]
    if dataset_name in REVERSE_DATASET_NAME_MAP:
        return REVERSE_DATASET_NAME_MAP[dataset_name], dataset_name
    raise ValueError(
        f"Unknown dataset {dataset_name!r}. Expected one of "
        f"{sorted(DEFAULT_DATASET_NAME_MAP)} or {sorted(REVERSE_DATASET_NAME_MAP)}."
    )


def resolve_benchmark_csv(benchmark_dir: Path | str, benchmark_dataset: str) -> Path:
    benchmark_dir = Path(benchmark_dir)
    matches = sorted(benchmark_dir.glob(f"*_{benchmark_dataset}_proxy_scores_*.csv"))
    if len(matches) != 1:
        joined = ", ".join(str(path) for path in matches) or "none"
        raise FileNotFoundError(
            f"Expected one benchmark CSV for dataset={benchmark_dataset} under {benchmark_dir}; found {joined}"
        )
    return matches[0]


def load_checkpoint_model(
    checkpoint_path: Path | str,
    device: torch.device,
) -> tuple[DCSPGModel, SymbolicVocabulary, DCSPGConfig, Mapping[str, object]]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = DCSPGConfig(**checkpoint["model_config"])
    vocabulary = SymbolicVocabulary(tokens=tuple(checkpoint["vocabulary_tokens"]))
    grammar = build_default_grammar(vocabulary, config)
    model = DCSPGModel(config=config, vocab_size=len(vocabulary), grammar=grammar).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, vocabulary, config, checkpoint


def sample_support_stats(
    ts_feature_dir: Path | str,
    ts_dataset: str,
    k_samples: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    store = Catch22FeatureStore(ts_feature_dir, dataset_names=[ts_dataset])
    features = store[ts_dataset].features
    if k_samples <= 0:
        raise ValueError("k_samples must be positive.")
    replace = k_samples > len(features)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(features), size=k_samples, replace=replace)
    support = features[indices]
    means = np.nanmean(support, axis=0)
    stds = np.nanstd(support, axis=0, ddof=0)
    means = np.where(np.isfinite(means), means, 0.0)
    stds = np.where(np.isfinite(stds), stds, 0.0)
    stats = np.stack([means, stds], axis=-1).astype(np.float32)
    return torch.from_numpy(stats).unsqueeze(0).to(device), tuple(int(index) for index in indices)


def generated_ids_to_rpn(token_ids: torch.Tensor, vocabulary: SymbolicVocabulary) -> str:
    tokens = []
    for token_id in token_ids.tolist():
        token = vocabulary.tokens[int(token_id)]
        if token == "<pad>":
            continue
        if token == "<eos>":
            tokens.append("<EOS>")
            break
        if token in {"<bos>", "<unk>"}:
            continue
        tokens.append(token)
    if not tokens or tokens[-1] != "<EOS>":
        tokens.append("<EOS>")
    return " ".join(tokens)


def generate_beam_candidates(
    model: DCSPGModel,
    stats: torch.Tensor,
    vocabulary: SymbolicVocabulary,
    beam_size: int,
    max_len: int,
) -> list[tuple[torch.Tensor, float]]:
    if beam_size <= 0:
        raise ValueError("beam_size must be positive.")
    if stats.shape[0] != 1:
        raise ValueError("Beam search currently expects a single conditioning sample.")

    model.eval()
    grammar = model.grammar
    encoder_output = model.encode_full(stats)
    beams: list[tuple[tuple[int, ...], float, bool]] = [((), 0.0, False)]

    for step in range(max_len):
        expanded: list[tuple[tuple[int, ...], float, bool]] = []
        for token_ids, log_prob, finished in beams:
            if finished:
                expanded.append((token_ids, log_prob, finished))
                continue

            prefix = torch.tensor([token_ids], dtype=torch.long, device=stats.device)
            logits = model.decoder(
                prefix,
                context=encoder_output.context,
                memory=encoder_output.memory,
            )[:, -1, :]
            if grammar is not None:
                logits = grammar.mask_next_logits(
                    logits,
                    prefix,
                    remaining_steps=max_len - step - 1,
                )
            log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)
            top_count = min(beam_size, log_probs.numel())
            top_log_probs, top_ids = torch.topk(log_probs, k=top_count)
            for next_log_prob, next_id in zip(top_log_probs.tolist(), top_ids.tolist()):
                if not math.isfinite(float(next_log_prob)):
                    continue
                next_tokens = (*token_ids, int(next_id))
                expanded.append(
                    (
                        next_tokens,
                        log_prob + float(next_log_prob),
                        int(next_id) == vocabulary.eos_id,
                    )
                )

        if not expanded:
            break

        deduped: dict[tuple[int, ...], tuple[tuple[int, ...], float, bool]] = {}
        for item in expanded:
            token_ids, log_prob, _finished = item
            previous = deduped.get(token_ids)
            if previous is None or log_prob > previous[1]:
                deduped[token_ids] = item
        beams = sorted(deduped.values(), key=lambda item: item[1], reverse=True)[:beam_size]
        if all(finished for _token_ids, _log_prob, finished in beams):
            break

    return [
        (torch.tensor(token_ids, dtype=torch.long), log_prob)
        for token_ids, log_prob, _finished in beams
    ]


def load_proxy_test_values(
    benchmark_csv: Path,
    split: str = "proxy_test",
    target_metric: str = "mse",
) -> tuple[dict[str, list[float]], list[float], int]:
    with benchmark_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [column for column in PROXY_TO_COLUMN.values() if column not in fieldnames]
        if missing:
            raise ValueError(f"{benchmark_csv} is missing proxy columns: {', '.join(missing)}")
        if target_metric not in fieldnames:
            raise ValueError(f"{benchmark_csv} is missing target metric column: {target_metric}")
        if "split" not in fieldnames:
            raise ValueError(f"{benchmark_csv} is missing split column.")
        rows = [row for row in reader if row.get("split") == split and row.get("status") == "success"]

    if not rows:
        raise ValueError(f"No successful rows found for split={split!r} in {benchmark_csv}")

    values = {
        column: [safe_float(row[column]) for row in rows]
        for column in sorted(set(PROXY_TO_COLUMN.values()))
    }
    raw_target = [safe_float(row[target_metric]) for row in rows]
    directed_target = [-value for value in raw_target]
    return values, directed_target, len(rows)


def safe_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite benchmark value: {value}")
    return parsed


def evaluate_rpn_tokens(
    rpn_tokens: str,
    values: dict[str, list[float]],
    directed_target: list[float],
    proxy_score_decimals: int | None,
    max_abs_proxy_score: float,
) -> tuple[list[float], float, str, str, str]:
    scores: list[float] = []
    spearman_neg_mse = float("nan")
    infix = ""
    latex = ""
    invalid_reason = ""
    try:
        tree = parse_rpn(rpn_tokens.split())
        infix = tree.to_infix()
        latex = tree.to_latex()
        scores, invalid_reason = normalize_proxy_scores(
            tree.evaluate(values),
            proxy_score_decimals=proxy_score_decimals,
            max_abs_proxy_score=max_abs_proxy_score,
        )
        if not invalid_reason:
            spearman_neg_mse = spearman_correlation(scores, directed_target)
            if not math.isfinite(spearman_neg_mse):
                invalid_reason = "nonfinite_spearman_correlation"
    except Exception as exc:
        invalid_reason = f"{type(exc).__name__}: {exc}"
    return scores, spearman_neg_mse, invalid_reason, infix, latex


def test_checkpoint_on_dataset(
    checkpoint_path: Path | str,
    dataset_name: str,
    ts_feature_dir: Path | str,
    benchmark_dir: Path | str,
    device: torch.device,
    condition_dataset_name: str | None = None,
    k_samples: int = 16,
    seed: int = 2026,
    split: str = "proxy_test",
    target_metric: str = "mse",
    proxy_score_decimals: int | None = None,
    max_abs_proxy_score: float = 1e18,
    max_len: int | None = None,
    beam_size: int = 1,
) -> DCSPGTestResult:
    evaluation_ts_dataset, benchmark_dataset = resolve_dataset_names(dataset_name)
    condition_dataset = condition_dataset_name or dataset_name
    condition_ts_dataset, _condition_benchmark_dataset = resolve_dataset_names(condition_dataset)
    benchmark_csv = resolve_benchmark_csv(benchmark_dir, benchmark_dataset)
    model, vocabulary, config, _checkpoint = load_checkpoint_model(checkpoint_path, device)
    stats, support_indices = sample_support_stats(ts_feature_dir, condition_ts_dataset, k_samples, seed, device)

    values, directed_target, split_count = load_proxy_test_values(
        benchmark_csv=benchmark_csv,
        split=split,
        target_metric=target_metric,
    )

    max_generation_len = max_len or config.max_formula_len
    if beam_size <= 1:
        with torch.no_grad():
            generated = model.generate(
                stats,
                bos_id=vocabulary.bos_id,
                eos_id=vocabulary.eos_id,
                pad_id=vocabulary.pad_id,
                max_len=max_generation_len,
                greedy=True,
            )
        rpn_tokens = generated_ids_to_rpn(generated[0].detach().cpu(), vocabulary)
        scores, spearman_neg_mse, invalid_reason, infix, latex = evaluate_rpn_tokens(
            rpn_tokens,
            values,
            directed_target,
            proxy_score_decimals,
            max_abs_proxy_score,
        )
        beam_rpn_tokens = (rpn_tokens,)
        beam_infix = (infix,)
        beam_latex = (latex,)
        beam_log_probs: tuple[float, ...] = ()
        beam_spearman = (spearman_neg_mse,)
        beam_invalid_reasons = (invalid_reason,)
        beam_valid_count = 0 if invalid_reason else 1
    else:
        with torch.no_grad():
            beam_candidates = generate_beam_candidates(
                model=model,
                stats=stats,
                vocabulary=vocabulary,
                beam_size=beam_size,
                max_len=max_generation_len,
            )
        beam_rpn_list = []
        beam_infix_list = []
        beam_latex_list = []
        beam_log_prob_list = []
        beam_spearman_list = []
        beam_invalid_list = []
        valid_score_vectors = []

        for token_ids, log_prob in beam_candidates:
            candidate_rpn = generated_ids_to_rpn(token_ids.cpu(), vocabulary)
            scores, candidate_spearman, candidate_invalid, candidate_infix, candidate_latex = evaluate_rpn_tokens(
                candidate_rpn,
                values,
                directed_target,
                proxy_score_decimals,
                max_abs_proxy_score,
            )
            beam_rpn_list.append(candidate_rpn)
            beam_infix_list.append(candidate_infix)
            beam_latex_list.append(candidate_latex)
            beam_log_prob_list.append(log_prob)
            beam_spearman_list.append(candidate_spearman)
            beam_invalid_list.append(candidate_invalid)
            if scores:
                valid_score_vectors.append(scores)

        rpn_tokens = beam_rpn_list[0] if beam_rpn_list else ""
        infix = beam_infix_list[0] if beam_infix_list else ""
        latex = beam_latex_list[0] if beam_latex_list else ""
        beam_valid_count = len(valid_score_vectors)
        if valid_score_vectors:
            averaged_scores = [
                float(sum(row_scores) / len(row_scores))
                for row_scores in zip(*valid_score_vectors)
            ]
            spearman_neg_mse = spearman_correlation(averaged_scores, directed_target)
            invalid_reason = "" if math.isfinite(spearman_neg_mse) else "nonfinite_ensemble_spearman_correlation"
        else:
            spearman_neg_mse = float("nan")
            invalid_reason = "no_valid_beam_proxy_scores"

        beam_rpn_tokens = tuple(beam_rpn_list)
        beam_infix = tuple(beam_infix_list)
        beam_latex = tuple(beam_latex_list)
        beam_log_probs = tuple(beam_log_prob_list)
        beam_spearman = tuple(beam_spearman_list)
        beam_invalid_reasons = tuple(beam_invalid_list)

    return DCSPGTestResult(
        checkpoint_path=str(Path(checkpoint_path)),
        dataset=dataset_name,
        ts_dataset=evaluation_ts_dataset,
        condition_dataset=condition_dataset,
        condition_ts_dataset=condition_ts_dataset,
        evaluation_dataset=dataset_name,
        evaluation_ts_dataset=evaluation_ts_dataset,
        benchmark_dataset=benchmark_dataset,
        benchmark_csv=str(benchmark_csv),
        split=split,
        split_count=split_count,
        support_indices=support_indices,
        rpn_tokens=rpn_tokens,
        infix=infix,
        latex=latex,
        spearman_neg_mse=spearman_neg_mse,
        invalid_reason=invalid_reason,
        beam_size=beam_size,
        beam_valid_count=beam_valid_count,
        beam_rpn_tokens=beam_rpn_tokens,
        beam_infix=beam_infix,
        beam_latex=beam_latex,
        beam_log_probs=beam_log_probs,
        beam_spearman_neg_mse=beam_spearman,
        beam_invalid_reasons=beam_invalid_reasons,
    )
