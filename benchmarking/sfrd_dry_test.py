from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from benchmarking.candidate_sampler import discover_available_backbones
from benchmarking.proxy_scorer import (
    _build_args,
    _extract_decoder_last_layer_representation,
    _extract_encoder_last_layer_representation,
    _extract_sfrd_sequence_representation_details,
    _prepare_batches,
    _repo_root,
    _select_exp_class,
)


def _iter_example_runs(repo_root: Path):
    examples_dir = repo_root / 'examples'
    for recipe_path in sorted(examples_dir.rglob('*.json')):
        try:
            payload = json.loads(recipe_path.read_text(encoding='utf-8'))
        except Exception:
            continue
        runs = payload.get('runs')
        if not isinstance(runs, list):
            continue
        for index, run in enumerate(runs):
            if not isinstance(run, dict):
                continue
            run_args = run.get('run_args')
            if not isinstance(run_args, dict):
                continue
            model_name = str(run_args.get('model', '') or '').strip()
            task_name = str(run_args.get('task_name', '') or '').strip()
            if not model_name or not task_name:
                continue
            yield {
                'recipe_path': str(recipe_path.relative_to(repo_root)),
                'run_index': index,
                'run_args': dict(run_args),
                'model': model_name,
                'task_name': task_name,
            }


def _path_exists(repo_root: Path, raw_path: Any) -> bool:
    if not raw_path:
        return True
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    return path.exists()


def _run_args_available(repo_root: Path, run_args: dict[str, Any]) -> bool:
    if not _path_exists(repo_root, run_args.get('root_path')):
        return False
    data_path = run_args.get('data_path')
    if not data_path:
        return True
    root_path = Path(str(run_args.get('root_path')))
    if not root_path.is_absolute():
        root_path = (repo_root / root_path).resolve()
    candidate = root_path / str(data_path)
    return candidate.exists()


def _select_representative_runs(repo_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    fallback: dict[tuple[str, str], dict[str, Any]] = {}
    for item in _iter_example_runs(repo_root):
        key = (item['model'], item['task_name'])
        fallback.setdefault(key, item)
        if key in selected:
            continue
        if _run_args_available(repo_root, item['run_args']):
            selected[key] = item
    for key, item in fallback.items():
        selected.setdefault(key, item)
    return selected


def _shape_str(shape: Any) -> str:
    if shape is None:
        return ''
    return 'x'.join(str(int(dim)) for dim in shape)


def _dry_args(run_args: dict[str, Any]) -> dict[str, Any]:
    updated = dict(run_args)
    updated['use_gpu'] = False
    updated['use_multi_gpu'] = False
    updated['num_workers'] = 0
    if 'batch_size' in updated:
        try:
            updated['batch_size'] = max(1, min(int(updated['batch_size']), 4))
        except Exception:
            updated['batch_size'] = 4
    else:
        updated['batch_size'] = 4
    return updated


def _record_base(model_name: str) -> dict[str, Any]:
    return {
        'model': model_name,
        'task_name': '',
        'data': '',
        'status': '',
        'error': '',
        'recipe_path': '',
        'first_source_kind': '',
        'first_module_name': '',
        'first_raw_shape': '',
        'first_canonical_shape': '',
        'second_source_kind': '',
        'second_module_name': '',
        'second_raw_shape': '',
        'second_canonical_shape': '',
        'expected_second_source_kind': '',
        'expected_second_module_name': '',
        'shape_rank_ok': '',
        'batch_dim_ok': '',
        'first_is_input_signal': '',
        'second_position_ok': '',
        'time_interpolation_needed': '',
        'feature_dim_equal': '',
    }


def _bool_str(value: bool) -> str:
    return 'true' if value else 'false'


def _test_combo(repo_root: Path, model_name: str, task_name: str, recipe_item: dict[str, Any]) -> dict[str, Any]:
    row = _record_base(model_name)
    row['task_name'] = task_name
    row['recipe_path'] = recipe_item['recipe_path']
    run_args = _dry_args(recipe_item['run_args'])
    row['data'] = str(run_args.get('data', '') or '')

    try:
        args = _build_args(run_args, gpu_id=None, repo_root=repo_root)
        args.use_gpu = False
        args.use_multi_gpu = False
        args.num_workers = 0
        Exp = _select_exp_class(args.task_name)
        exp = Exp(args)
        prepared_batch = _prepare_batches(exp, 1)[0]

        details = _extract_sfrd_sequence_representation_details(exp, prepared_batch)
        if details is None:
            raise RuntimeError('SFRD representation details could not be extracted.')

        decoder_info = _extract_decoder_last_layer_representation(
            exp,
            prepared_batch,
            batch_size=int(prepared_batch['batch_x'].size(0)),
            expected_time_sizes=details['expected_time_sizes'],
        )
        encoder_info = _extract_encoder_last_layer_representation(
            exp,
            prepared_batch,
            batch_size=int(prepared_batch['batch_x'].size(0)),
            expected_time_sizes=details['expected_time_sizes'],
        )

        first = details['first']
        second = details['second']
        expected = decoder_info or encoder_info
        expected_kind = expected['source_kind'] if expected is not None else 'fallback_generic'
        expected_module_name = expected['module_name'] if expected is not None else ''

        first_shape = tuple(first['canonical_shape'])
        second_shape = tuple(second['canonical_shape'])
        shape_rank_ok = len(first_shape) == 3 and len(second_shape) == 3
        batch_dim_ok = shape_rank_ok and first_shape[0] == second_shape[0]
        first_is_input_signal = first['source_kind'] == 'input_signal' and first['module_name'] == 'batch_x'
        if expected_kind == 'fallback_generic':
            second_position_ok = second['source_kind'] == 'generic_forward_activation'
        else:
            second_position_ok = (
                second['source_kind'] == expected_kind
                and second['module_name'] == expected_module_name
            )
        time_interpolation_needed = shape_rank_ok and first_shape[1] != second_shape[1]
        feature_dim_equal = shape_rank_ok and first_shape[2] == second_shape[2]

        row.update(
            {
                'status': 'pass' if first_is_input_signal and shape_rank_ok and batch_dim_ok and second_position_ok else 'fail',
                'first_source_kind': first['source_kind'],
                'first_module_name': first['module_name'],
                'first_raw_shape': _shape_str(first['raw_shape']),
                'first_canonical_shape': _shape_str(first_shape),
                'second_source_kind': second['source_kind'],
                'second_module_name': second['module_name'],
                'second_raw_shape': _shape_str(second['raw_shape']),
                'second_canonical_shape': _shape_str(second_shape),
                'expected_second_source_kind': expected_kind,
                'expected_second_module_name': expected_module_name,
                'shape_rank_ok': _bool_str(shape_rank_ok),
                'batch_dim_ok': _bool_str(batch_dim_ok),
                'first_is_input_signal': _bool_str(first_is_input_signal),
                'second_position_ok': _bool_str(second_position_ok),
                'time_interpolation_needed': _bool_str(time_interpolation_needed),
                'feature_dim_equal': _bool_str(feature_dim_equal),
            }
        )
    except Exception as exc:
        row['status'] = 'error'
        row['error'] = f'{type(exc).__name__}: {exc}'
    return row


def run_sfrd_dry_test(repo_root: Path, output_path: Path | None = None) -> tuple[list[dict[str, Any]], Path]:
    selected_runs = _select_representative_runs(repo_root)
    rows: list[dict[str, Any]] = []

    for model_name in discover_available_backbones(repo_root):
        matching_items = [
            selected_runs[key]
            for key in sorted(selected_runs)
            if key[0] == model_name
        ]
        if not matching_items:
            row = _record_base(model_name)
            row['status'] = 'no_example'
            row['error'] = 'No runnable example recipe found for this backbone.'
            rows.append(row)
            continue

        for item in matching_items:
            rows.append(_test_combo(repo_root, model_name, item['task_name'], item))

    if output_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = repo_root / 'benchmarking' / f'sfrd_dry_test_report_{timestamp}.csv'

    fieldnames = list(_record_base('model').keys())
    with output_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows, output_path


def main() -> None:
    parser = argparse.ArgumentParser(description='Dry-test SFRD hook selection across backbones.')
    parser.add_argument('--output', type=str, help='Optional CSV path for the dry-test report.')
    args = parser.parse_args()

    repo_root = _repo_root()
    output_path = Path(args.output).resolve() if args.output else None
    rows, csv_path = run_sfrd_dry_test(repo_root, output_path=output_path)

    total = len(rows)
    by_status: dict[str, int] = {}
    for row in rows:
        by_status[row['status']] = by_status.get(row['status'], 0) + 1

    print(f'Wrote {total} rows to {csv_path}')
    for status in sorted(by_status):
        print(f'{status}: {by_status[status]}')


if __name__ == '__main__':
    main()
