from __future__ import annotations

"""Bounded-memory and quantized-summary helpers for long discovery runs."""

import json
from collections import defaultdict
from hashlib import blake2b
from typing import Any


DEFAULT_SCORE_STEP = 0.05
DEFAULT_PARAMETER_STEP = 0.25


def build_compressed_experience_shard(
    *,
    records: list[dict[str, Any]],
    operator_prior_outcomes: list[dict[str, Any]],
    source: str = 'manual_compaction',
    score_step: float = DEFAULT_SCORE_STEP,
    parameter_step: float = DEFAULT_PARAMETER_STEP,
    sample_limit: int = 3,
) -> dict[str, Any]:
    """Compress raw run/outcome rows into fixed-width, quantized summaries."""
    record_signatures = _record_signatures(records, sample_limit=sample_limit)
    operator_signatures = _operator_prior_signatures(
        operator_prior_outcomes,
        score_step=score_step,
        parameter_step=parameter_step,
        sample_limit=sample_limit,
    )
    raw_payload = {
        'records': records,
        'operator_prior_outcomes': operator_prior_outcomes,
    }
    compressed_payload = {
        'record_signatures': record_signatures,
        'operator_prior_signatures': operator_signatures,
    }
    shard = {
        'version': 1,
        'source': source,
        'quantization': {
            'score_step': score_step,
            'parameter_step': parameter_step,
        },
        'source_counts': {
            'records': len(records),
            'operator_prior_outcomes': len(operator_prior_outcomes),
        },
        'record_signatures': record_signatures,
        'operator_prior_signatures': operator_signatures,
        'estimated_raw_bytes': estimate_json_bytes(raw_payload),
        'estimated_compressed_bytes': estimate_json_bytes(compressed_payload),
    }
    shard['compression_ratio'] = _safe_ratio(
        shard['estimated_raw_bytes'],
        shard['estimated_compressed_bytes'],
    )
    shard['shard_id'] = _stable_digest({
        'source': source,
        'quantization': shard['quantization'],
        'source_counts': shard['source_counts'],
        'record_signatures': record_signatures,
        'operator_prior_signatures': operator_signatures,
    })
    return shard


def resource_efficiency_report(
    *,
    records: list[dict[str, Any]],
    operator_prior_outcomes: list[dict[str, Any]],
    compressed_shards: list[dict[str, Any]],
    recommended_record_window: int = 96,
    recommended_operator_window: int = 192,
) -> dict[str, Any]:
    """Estimate long-run growth and whether bounded summaries are active."""
    anchor_outcomes = [
        outcome for outcome in operator_prior_outcomes
        if outcome.get('retention_role') == 'operator_anchor'
    ]
    recent_operator_outcomes = [
        outcome for outcome in operator_prior_outcomes
        if outcome.get('retention_role') != 'operator_anchor'
    ]
    raw_current = {
        'records': records,
        'operator_prior_outcomes': operator_prior_outcomes,
    }
    current_raw_bytes = estimate_json_bytes(raw_current)
    shard_raw_bytes = sum(
        int(shard.get('estimated_raw_bytes', 0) or 0)
        for shard in compressed_shards
    )
    shard_compressed_bytes = sum(
        int(shard.get('estimated_compressed_bytes', 0) or 0)
        for shard in compressed_shards
    )
    compacted_records = sum(
        int((shard.get('source_counts') or {}).get('records', 0) or 0)
        for shard in compressed_shards
    )
    compacted_operator_outcomes = sum(
        int((shard.get('source_counts') or {}).get('operator_prior_outcomes', 0) or 0)
        for shard in compressed_shards
    )
    retained_summary_count = sum(
        len(shard.get('record_signatures') or [])
        + len(shard.get('operator_prior_signatures') or [])
        for shard in compressed_shards
    )
    total_event_count = (
        len(records)
        + len(operator_prior_outcomes)
        + compacted_records
        + compacted_operator_outcomes
    )
    retained_detail_count = (
        len(records)
        + len(operator_prior_outcomes)
        + retained_summary_count
    )
    estimated_uncompressed_bytes = current_raw_bytes + shard_raw_bytes
    estimated_retained_bytes = current_raw_bytes + shard_compressed_bytes
    actions = []
    if (
        len(records) > recommended_record_window
        or len(recent_operator_outcomes) > recommended_operator_window
    ):
        actions.append({
            'action_kind': 'compact_theory_memory',
            'reason': 'raw discovery memory is beyond the recommended recent window',
            'runs_final': False,
        })
    if not compressed_shards:
        actions.append({
            'action_kind': 'enable_quantized_experience_shards',
            'reason': 'long runs need summaries that grow by signature count, not raw event count',
            'runs_final': False,
        })
    return {
        'version': 1,
        'raw_record_count': len(records),
        'raw_operator_prior_outcome_count': len(operator_prior_outcomes),
        'raw_operator_prior_anchor_count': len(anchor_outcomes),
        'raw_operator_prior_recent_count': len(recent_operator_outcomes),
        'compressed_shard_count': len(compressed_shards),
        'compacted_record_count': compacted_records,
        'compacted_operator_prior_outcome_count': compacted_operator_outcomes,
        'estimated_uncompressed_bytes': estimated_uncompressed_bytes,
        'estimated_retained_bytes': estimated_retained_bytes,
        'estimated_compression_ratio': _safe_ratio(
            estimated_uncompressed_bytes,
            estimated_retained_bytes,
        ),
        'detail_reduction_ratio': _safe_ratio(total_event_count, retained_detail_count),
        'bounded_windows': {
            'recommended_record_window': recommended_record_window,
            'recommended_operator_window': recommended_operator_window,
            'records_within_window': len(records) <= recommended_record_window,
            'operator_outcomes_within_window': (
                len(recent_operator_outcomes) <= recommended_operator_window
            ),
            'operator_anchor_count': len(anchor_outcomes),
        },
        'controls': {
            'quantized_scores': True,
            'quantized_parameters': True,
            'canonical_signatures': True,
            'keeps_recent_raw_window': True,
            'keeps_operator_anchor_examples': True,
        },
        'long_run_ready': bool(compressed_shards)
        and len(records) <= recommended_record_window
        and len(recent_operator_outcomes) <= recommended_operator_window,
        'recommended_actions': actions,
    }


def operator_outcome_anchor_indexes(
    outcomes: list[dict[str, Any]],
    *,
    max_per_operator: int = 2,
) -> set[int]:
    """Pick bounded raw exemplars that preserve successes and failures per operator."""
    by_operator: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, outcome in enumerate(outcomes):
        operator_key = str(
            outcome.get('operator_prior_key')
            or outcome.get('operator_key')
            or outcome.get('operator_kind')
            or 'unknown'
        )
        by_operator[operator_key].append((index, outcome))

    anchors: set[int] = set()
    for grouped in by_operator.values():
        ranked = sorted(
            grouped,
            key=lambda item: (
                _outcome_priority(str(item[1].get('outcome', 'unknown'))),
                float(item[1].get('best_score', 0.0) or 0.0),
                int(item[1].get('matching_equation_count', 0) or 0),
                item[0],
            ),
            reverse=True,
        )
        for index, _ in ranked[:max_per_operator]:
            anchors.add(index)
    return anchors


def estimate_json_bytes(value: Any) -> int:
    return len(_stable_json(value).encode('utf-8'))


def _record_signatures(
    records: list[dict[str, Any]],
    *,
    sample_limit: int,
) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for record in records:
        signature = {
            'context': str(record.get('context', 'unknown')),
            'phase': str(record.get('phase', 'unknown')),
            'disagreement_mode': str(record.get('disagreement_mode') or 'none'),
            'theory_count_bin': _count_bin(record.get('theory_count')),
            'operator_count_bin': _count_bin(record.get('operator_count')),
            'proof_check_count_bin': _count_bin(record.get('proof_check_count')),
        }
        key = _stable_json(signature)
        group = groups.setdefault(
            key,
            {
                **signature,
                'count': 0,
                'seed_min': None,
                'seed_max': None,
                'examples': [],
            },
        )
        seed = _safe_int(record.get('seed'))
        group['count'] += 1
        if seed is not None:
            group['seed_min'] = seed if group['seed_min'] is None else min(group['seed_min'], seed)
            group['seed_max'] = seed if group['seed_max'] is None else max(group['seed_max'], seed)
        _append_sample(group['examples'], _record_example(record), sample_limit)
    return sorted(groups.values(), key=lambda item: (-item['count'], item['context'], item['phase']))


def _operator_prior_signatures(
    outcomes: list[dict[str, Any]],
    *,
    score_step: float,
    parameter_step: float,
    sample_limit: int,
) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for outcome in outcomes:
        parameters = dict(
            outcome.get('operator_prior_refined_parameters')
            or outcome.get('refined_parameters')
            or outcome.get('parameters')
            or {}
        )
        signature = {
            'operator_kind': str(outcome.get('operator_kind', 'unknown')),
            'context': str(outcome.get('context', 'unknown')),
            'outcome': str(outcome.get('outcome', 'unknown')),
            'score_bin': _quantize(outcome.get('best_score'), score_step),
            'match_count_bin': _count_bin(outcome.get('matching_equation_count')),
            'relation': str(parameters.get('relation', 'unknown')),
            'distance_exponent_bin': _quantize(parameters.get('distance_exponent'), parameter_step),
            'cutoff_radius_bin': _quantize(parameters.get('cutoff_radius'), parameter_step),
            'source_context': str(parameters.get('source_context', 'unknown')),
        }
        key = _stable_json(signature)
        group = groups.setdefault(
            key,
            {
                **signature,
                'count': 0,
                'seed_min': None,
                'seed_max': None,
                'best_score': 0.0,
                'examples': [],
            },
        )
        seed = _safe_int(outcome.get('seed'))
        score = float(outcome.get('best_score', 0.0) or 0.0)
        group['count'] += 1
        group['best_score'] = round(max(float(group['best_score']), score), 3)
        if seed is not None:
            group['seed_min'] = seed if group['seed_min'] is None else min(group['seed_min'], seed)
            group['seed_max'] = seed if group['seed_max'] is None else max(group['seed_max'], seed)
        _append_sample(group['examples'], _operator_example(outcome), sample_limit)
    return sorted(
        groups.values(),
        key=lambda item: (
            -item['count'],
            item['operator_kind'],
            item['context'],
            item['outcome'],
        ),
    )


def _record_example(record: dict[str, Any]) -> dict[str, Any]:
    return {
        'context': record.get('context'),
        'seed': record.get('seed'),
        'phase': record.get('phase'),
        'theory_count': record.get('theory_count', 0),
        'proof_check_count': record.get('proof_check_count', 0),
    }


def _operator_example(outcome: dict[str, Any]) -> dict[str, Any]:
    return {
        'context': outcome.get('context'),
        'seed': outcome.get('seed'),
        'operator_kind': outcome.get('operator_kind'),
        'outcome': outcome.get('outcome'),
        'best_score': round(float(outcome.get('best_score', 0.0) or 0.0), 3),
    }


def _append_sample(samples: list[dict[str, Any]], sample: dict[str, Any], limit: int):
    if len(samples) >= limit:
        return
    samples.append(sample)


def _outcome_priority(outcome: str) -> int:
    return {
        'confirmed': 5,
        'operator_prior_repair_confirmed': 5,
        'weak': 4,
        'unmatched': 3,
        'missing': 2,
    }.get(outcome, 1)


def _count_bin(value: Any) -> str:
    count = _safe_int(value) or 0
    if count <= 0:
        return '0'
    if count == 1:
        return '1'
    if count <= 3:
        return '2-3'
    if count <= 7:
        return '4-7'
    return '8+'


def _quantize(value: Any, step: float) -> str:
    if value is None:
        return 'none'
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 'none'
    if step <= 0:
        return str(round(numeric, 3))
    quantized = round(round(numeric / step) * step, 3)
    return f'{quantized:g}'


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    denominator = float(denominator or 0)
    if denominator <= 0:
        return 0.0
    return round(float(numerator or 0) / denominator, 3)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), default=str)


def _stable_digest(value: Any) -> str:
    return blake2b(_stable_json(value).encode('utf-8'), digest_size=8).hexdigest()
