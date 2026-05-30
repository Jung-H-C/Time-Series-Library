from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from proxy_experiment_config import CANONICAL_BACKBONES, normalize_backbone


# Match run.py defaults for common knobs that are not part of candidate search.
DEFAULT_SEQ_LEN = 96
DEFAULT_DROPOUT = 0.1
DEFAULT_LEARNING_RATE = 0.0001

FILM_WINDOW_SIZES = [32, 64, 128, 256, 512]
FILM_MULTISCALE_OPTIONS = ["1", "1,2", "1,2,4", "1,2,4,8"]
FILM_MODE_OPTIONS = [8, 16, 24, 32, 40, 48]


def _choice(rng: random.Random, values: list[Any]) -> Any:
    return values[rng.randrange(len(values))]


def _ff_dim(rng: random.Random, d_model: int, multipliers: tuple[int, ...] = (2, 4)) -> int:
    return int(d_model * _choice(rng, list(multipliers)))


def _fixed_seq_len(seq_lens: list[int]) -> int:
    if not seq_lens:
        raise ValueError("seq_lens must contain at least one value.")
    return int(DEFAULT_SEQ_LEN if DEFAULT_SEQ_LEN in seq_lens else seq_lens[0])


def _film_modes_for_seq_len(requested_modes: int, seq_len: int) -> int:
    requested_modes = int(requested_modes)
    if requested_modes <= 0:
        raise ValueError(f"film_modes must be positive, got: {requested_modes}")
    max_modes = max(1, int(seq_len) // 2)
    if requested_modes <= max_modes:
        return requested_modes
    valid_modes = [mode for mode in FILM_MODE_OPTIONS if mode <= requested_modes and mode <= max_modes]
    if valid_modes:
        return max(valid_modes)
    return max_modes


def _common_args(
    _rng: random.Random,
    seq_lens: list[int],
    dropout: float | None,
    learning_rate: float | None,
) -> dict[str, Any]:
    return _default_common_args(seq_lens, dropout, learning_rate)


def _default_common_args(
    seq_lens: list[int],
    dropout: float | None,
    learning_rate: float | None,
) -> dict[str, Any]:
    return {
        "seq_len": _fixed_seq_len(seq_lens),
        "dropout": float(DEFAULT_DROPOUT if dropout is None else dropout),
        "learning_rate": float(DEFAULT_LEARNING_RATE if learning_rate is None else learning_rate),
    }


def _attention_args(
    rng: random.Random,
    seq_lens: list[int],
    dropout: float | None,
    learning_rate: float | None,
    d_models: list[int],
    e_layers: list[int],
    d_layers: list[int],
) -> dict[str, Any]:
    d_model = int(_choice(rng, d_models))
    return {
        **_common_args(rng, seq_lens, dropout, learning_rate),
        "d_model": d_model,
        "n_heads": int(_choice(rng, [2, 4, 8])),
        "e_layers": int(_choice(rng, e_layers)),
        "d_layers": int(_choice(rng, d_layers)),
        "d_ff": _ff_dim(rng, d_model),
        "factor": int(_choice(rng, [1, 3, 5])),
    }


def default_autoformer(seq_lens: list[int], dropout: float | None, learning_rate: float | None) -> dict[str, Any]:
    return {
        **_default_common_args(seq_lens, dropout, learning_rate),
        "d_model": 128,
        "n_heads": 4,
        "e_layers": 2,
        "d_layers": 1,
        "d_ff": 256,
        "factor": 3,
        "moving_avg": 25,
    }


def sample_autoformer(
    rng: random.Random,
    seq_lens: list[int],
    dropout: float | None,
    learning_rate: float | None,
) -> dict[str, Any]:
    args = _attention_args(
        rng,
        seq_lens,
        dropout,
        learning_rate,
        d_models=[64, 128, 256],
        e_layers=[1, 2, 3],
        d_layers=[1, 2],
    )
    args["moving_avg"] = int(_choice(rng, [13, 25, 49]))
    return args


def default_crossformer(seq_lens: list[int], dropout: float | None, learning_rate: float | None) -> dict[str, Any]:
    return {
        **_default_common_args(seq_lens, dropout, learning_rate),
        "d_model": 128,
        "n_heads": 4,
        "e_layers": 2,
        "d_layers": 1,
        "d_ff": 256,
        "factor": 3,
    }


def sample_crossformer(
    rng: random.Random,
    seq_lens: list[int],
    dropout: float | None,
    learning_rate: float | None,
) -> dict[str, Any]:
    d_model = int(_choice(rng, [32, 64, 128, 256]))
    return {
        **_common_args(rng, seq_lens, dropout, learning_rate),
        "d_model": d_model,
        "n_heads": int(_choice(rng, [2, 4, 8])),
        "e_layers": int(_choice(rng, [1, 2, 3])),
        "d_layers": 1,
        "d_ff": _ff_dim(rng, d_model),
        "factor": int(_choice(rng, [3, 5, 10])),
    }


def default_film(seq_lens: list[int], dropout: float | None, learning_rate: float | None) -> dict[str, Any]:
    common_args = _default_common_args(seq_lens, dropout, learning_rate)
    return {
        **common_args,
        "d_layers": 1,
        "film_window_size": 256,
        "film_multiscale": "1,2,4",
        "film_modes": _film_modes_for_seq_len(32, common_args["seq_len"]),
    }


def sample_film(
    rng: random.Random,
    seq_lens: list[int],
    dropout: float | None,
    learning_rate: float | None,
) -> dict[str, Any]:
    # FiLM's e_layers is not used by the forecast implementation in this repo.
    common_args = _common_args(rng, seq_lens, dropout, learning_rate)
    requested_modes = int(_choice(rng, FILM_MODE_OPTIONS))
    return {
        **common_args,
        "d_layers": 1,
        "film_window_size": int(_choice(rng, FILM_WINDOW_SIZES)),
        "film_multiscale": str(_choice(rng, FILM_MULTISCALE_OPTIONS)),
        "film_modes": _film_modes_for_seq_len(requested_modes, common_args["seq_len"]),
    }


def default_micn(seq_lens: list[int], dropout: float | None, learning_rate: float | None) -> dict[str, Any]:
    return {
        **_default_common_args(seq_lens, dropout, learning_rate),
        "d_model": 128,
        "n_heads": 4,
        "e_layers": 2,
        "d_layers": 1,
        "d_ff": 256,
        "factor": 3,
        "moving_avg": 25,
    }


def sample_micn(
    rng: random.Random,
    seq_lens: list[int],
    dropout: float | None,
    learning_rate: float | None,
) -> dict[str, Any]:
    d_model = int(_choice(rng, [32, 64, 128, 256]))
    return {
        **_common_args(rng, seq_lens, dropout, learning_rate),
        "d_model": d_model,
        "n_heads": int(_choice(rng, [2, 4, 8])),
        "e_layers": int(_choice(rng, [1, 2, 3])),
        "d_layers": int(_choice(rng, [1, 2])),
        "d_ff": _ff_dim(rng, d_model, multipliers=(1, 2, 4)),
        "factor": int(_choice(rng, [1, 3, 5])),
        "moving_avg": int(_choice(rng, [13, 25, 49])),
    }


def default_mamba(seq_lens: list[int], dropout: float | None, learning_rate: float | None) -> dict[str, Any]:
    return {
        **_default_common_args(seq_lens, dropout, learning_rate),
        "d_model": 128,
        "d_ff": 16,
        "d_conv": 4,
        "expand": 2,
        "dt_rank": 8,
        "e_layers": 2,
        "d_layers": 1,
    }


def sample_mamba(
    rng: random.Random,
    seq_lens: list[int],
    dropout: float | None,
    learning_rate: float | None,
) -> dict[str, Any]:
    return {
        **_common_args(rng, seq_lens, dropout, learning_rate),
        "d_model": int(_choice(rng, [64, 96, 128])),
        "d_ff": int(_choice(rng, [8, 12, 16])),
        "d_conv": int(_choice(rng, [2, 3, 4])),
        "expand": int(_choice(rng, [1, 2, 3])),
        "dt_rank": int(_choice(rng, [8, 16, 32])),
        "e_layers": 2,
        "d_layers": 1,
    }


def default_patchtst(seq_lens: list[int], dropout: float | None, learning_rate: float | None) -> dict[str, Any]:
    return {
        **_default_common_args(seq_lens, dropout, learning_rate),
        "d_model": 128,
        "n_heads": 4,
        "e_layers": 2,
        "d_layers": 1,
        "d_ff": 256,
        "factor": 3,
    }


def sample_patchtst(
    rng: random.Random,
    seq_lens: list[int],
    dropout: float | None,
    learning_rate: float | None,
) -> dict[str, Any]:
    d_model = int(_choice(rng, [64, 128, 256, 512]))
    return {
        **_common_args(rng, seq_lens, dropout, learning_rate),
        "d_model": d_model,
        "n_heads": int(_choice(rng, [2, 4, 8])),
        "e_layers": int(_choice(rng, [1, 2, 3])),
        "d_layers": 1,
        "d_ff": _ff_dim(rng, d_model, multipliers=(1, 2, 4)),
        "factor": int(_choice(rng, [1, 3, 5])),
    }


def default_timesnet(seq_lens: list[int], dropout: float | None, learning_rate: float | None) -> dict[str, Any]:
    return {
        **_default_common_args(seq_lens, dropout, learning_rate),
        "d_model": 64,
        "e_layers": 2,
        "d_layers": 1,
        "d_ff": 128,
        "top_k": 5,
        "num_kernels": 6,
        "factor": 3,
    }


def sample_timesnet(
    rng: random.Random,
    seq_lens: list[int],
    dropout: float | None,
    learning_rate: float | None,
) -> dict[str, Any]:
    d_model = int(_choice(rng, [32, 64, 128, 256]))
    return {
        **_common_args(rng, seq_lens, dropout, learning_rate),
        "d_model": d_model,
        "e_layers": int(_choice(rng, [1, 2, 3])),
        "d_layers": 1,
        "d_ff": _ff_dim(rng, d_model, multipliers=(1, 2, 4)),
        "top_k": int(_choice(rng, [2, 3, 5])),
        "num_kernels": int(_choice(rng, [3, 5, 6])),
        "factor": int(_choice(rng, [1, 3, 5])),
    }


def sample_transformer(
    rng: random.Random,
    seq_lens: list[int],
    dropout: float | None,
    learning_rate: float | None,
) -> dict[str, Any]:
    return _attention_args(
        rng,
        seq_lens,
        dropout,
        learning_rate,
        d_models=[64, 128, 256],
        e_layers=[1, 2, 3],
        d_layers=[1, 2],
    )


def default_transformer(seq_lens: list[int], dropout: float | None, learning_rate: float | None) -> dict[str, Any]:
    return {
        **_default_common_args(seq_lens, dropout, learning_rate),
        "d_model": 128,
        "n_heads": 4,
        "e_layers": 2,
        "d_layers": 1,
        "d_ff": 256,
        "factor": 3,
    }


def sample_dlinear(
    rng: random.Random,
    seq_lens: list[int],
    dropout: float | None,
    learning_rate: float | None,
) -> dict[str, Any]:
    return {
        **_common_args(rng, seq_lens, dropout, learning_rate),
        "moving_avg": int(_choice(rng, [13, 25, 49])),
    }


SAMPLERS: dict[str, Callable[[random.Random, list[int], float | None, float | None], dict[str, Any]]] = {
    "Autoformer": sample_autoformer,
    "Crossformer": sample_crossformer,
    "FiLM": sample_film,
    "MICN": sample_micn,
    "Mamba": sample_mamba,
    "PatchTST": sample_patchtst,
    "TimesNet": sample_timesnet,
    "Transformer": sample_transformer,
    "DLinear": sample_dlinear,
}


def default_dlinear(seq_lens: list[int], dropout: float | None, learning_rate: float | None) -> dict[str, Any]:
    return {
        **_default_common_args(seq_lens, dropout, learning_rate),
        "moving_avg": 25,
    }


DEFAULTS: dict[str, Callable[[list[int], float | None, float | None], dict[str, Any]]] = {
    "Autoformer": default_autoformer,
    "Crossformer": default_crossformer,
    "FiLM": default_film,
    "MICN": default_micn,
    "Mamba": default_mamba,
    "PatchTST": default_patchtst,
    "TimesNet": default_timesnet,
    "Transformer": default_transformer,
    "DLinear": default_dlinear,
}


def _signature(run_args: dict[str, Any]) -> str:
    return json.dumps(run_args, sort_keys=True, separators=(",", ":"))


def sample_unique_candidates(
    backbone: str,
    count: int,
    rng: random.Random,
    seq_lens: list[int],
    dropout: float | None,
    learning_rate: float | None,
) -> list[dict[str, Any]]:
    if count < 1:
        raise ValueError("count must be at least 1 so each backbone can include its default candidate.")

    sampler = SAMPLERS[backbone]
    default_args = DEFAULTS[backbone](seq_lens, dropout, learning_rate)
    candidates: list[dict[str, Any]] = [default_args]
    seen: set[str] = {_signature(default_args)}
    max_attempts = max(100, count * 100)

    for _ in range(max_attempts):
        if len(candidates) >= count:
            break
        run_args = sampler(rng, seq_lens, dropout, learning_rate)
        signature = _signature(run_args)
        if signature in seen:
            continue
        seen.add(signature)
        candidates.append(run_args)

    if len(candidates) != count:
        raise RuntimeError(
            f"Could only sample {len(candidates)} unique candidates for {backbone}; "
            f"requested {count}."
        )
    return candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample multi-backbone long-term forecasting candidate models."
    )
    parser.add_argument("--output", type=Path, required=True, help="Output candidate JSON path.")
    parser.add_argument("--seed", type=int, default=2026, help="Deterministic sampling seed.")
    parser.add_argument(
        "--backbones",
        nargs="+",
        default=list(CANONICAL_BACKBONES),
        help="Backbones to sample. Default: all enabled target backbones.",
    )
    parser.add_argument(
        "--num-per-backbone",
        type=int,
        default=10,
        help="Number of candidates sampled per backbone.",
    )
    parser.add_argument(
        "--train-per-backbone",
        type=int,
        default=None,
        help="Candidates per backbone assigned to proxy_train. Default: floor(num_per_backbone / 2).",
    )
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=[DEFAULT_SEQ_LEN],
        help=(
            "Backward-compatible fallback list for fixed seq_len. "
            f"If --seq-len is omitted, uses {DEFAULT_SEQ_LEN} when present, otherwise the first value."
        ),
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help=f"Fix seq_len for every candidate. Default: {DEFAULT_SEQ_LEN}.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help=f"Fix dropout for every candidate. Default: {DEFAULT_DROPOUT}.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help=f"Fix learning_rate for every candidate. Default: {DEFAULT_LEARNING_RATE}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backbones = [normalize_backbone(backbone) for backbone in args.backbones]
    train_per_backbone = (
        args.num_per_backbone // 2
        if args.train_per_backbone is None
        else args.train_per_backbone
    )
    if train_per_backbone < 0 or train_per_backbone > args.num_per_backbone:
        raise ValueError("--train-per-backbone must be in [0, num_per_backbone].")

    rng = random.Random(args.seed)
    fixed_seq_len = int(args.seq_len if args.seq_len is not None else _fixed_seq_len(args.seq_lens))
    fixed_dropout = float(DEFAULT_DROPOUT if args.dropout is None else args.dropout)
    fixed_learning_rate = float(DEFAULT_LEARNING_RATE if args.learning_rate is None else args.learning_rate)
    seq_lens = [fixed_seq_len]
    all_candidates: list[dict[str, Any]] = []
    split_counts: dict[str, dict[str, int]] = {}

    for backbone in backbones:
        sampled = sample_unique_candidates(
            backbone,
            args.num_per_backbone,
            rng,
            seq_lens,
            fixed_dropout,
            fixed_learning_rate,
        )
        split_counts[backbone] = {"proxy_train": 0, "proxy_eval": 0}
        for index, run_args in enumerate(sampled):
            split = "proxy_train" if index < train_per_backbone else "proxy_eval"
            split_counts[backbone][split] += 1
            all_candidates.append(
                {
                    "candidate_id": f"{backbone}_{index:03d}",
                    "backbone": backbone,
                    "is_default": index == 0,
                    "split": split,
                    "run_args": run_args,
                }
            )

    payload = {
        "schema_version": 1,
        "task_name": "long_term_forecast",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "sampling": {
            "backbones": backbones,
            "num_per_backbone": args.num_per_backbone,
            "train_per_backbone": train_per_backbone,
            "seq_len": fixed_seq_len,
            "seq_lens": seq_lens,
            "fixed_dropout": fixed_dropout,
            "fixed_learning_rate": fixed_learning_rate,
            "default_candidate_index": 0,
        },
        "split_counts": split_counts,
        "candidates": all_candidates,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(all_candidates)} candidates to {args.output} "
        f"({train_per_backbone} proxy_train per backbone)."
    )


if __name__ == "__main__":
    main()
