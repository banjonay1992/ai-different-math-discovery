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


def build_canonical_law_shard(
    *,
    domain_world_records: list[dict[str, Any]],
    autonomous_scientist_records: list[dict[str, Any]],
    arithmetic_rediscovery_records: list[dict[str, Any]] | None = None,
    operator_prior_outcomes: list[dict[str, Any]] | None = None,
    source: str = 'canonical_law_compaction',
    sample_limit: int = 3,
) -> dict[str, Any]:
    """Distill repeated equations/invariants into compact canonical law rows."""
    arithmetic_rediscovery_records = arithmetic_rediscovery_records or []
    operator_prior_outcomes = operator_prior_outcomes or []
    law_events = [
        *_law_events_from_domain_worlds(domain_world_records),
        *_law_events_from_scientist_records(autonomous_scientist_records),
        *_law_events_from_arithmetic_records(arithmetic_rediscovery_records),
        *_law_events_from_operator_outcomes(operator_prior_outcomes),
    ]
    laws = _canonical_law_summaries(law_events, sample_limit=sample_limit)
    raw_payload = {
        'domain_world_records': domain_world_records,
        'autonomous_scientist_records': autonomous_scientist_records,
        'arithmetic_rediscovery_records': arithmetic_rediscovery_records,
        'operator_prior_outcomes': operator_prior_outcomes,
    }
    compressed_payload = {'canonical_laws': laws}
    shard = {
        'version': 1,
        'source': source,
        'source_counts': {
            'domain_world_records': len(domain_world_records),
            'autonomous_scientist_records': len(autonomous_scientist_records),
            'arithmetic_rediscovery_records': len(arithmetic_rediscovery_records),
            'operator_prior_outcomes': len(operator_prior_outcomes),
            'law_events': len(law_events),
        },
        'canonical_laws': laws,
        'estimated_raw_bytes': estimate_json_bytes(raw_payload),
        'estimated_canonical_bytes': estimate_json_bytes(compressed_payload),
    }
    shard['compression_ratio'] = _safe_ratio(
        shard['estimated_raw_bytes'],
        shard['estimated_canonical_bytes'],
    )
    shard['shard_id'] = _stable_digest({
        'source_counts': shard['source_counts'],
        'canonical_laws': laws,
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


def canonical_law_compression_report(
    *,
    canonical_law_shards: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize whether canonical law compression is active and useful."""
    canonical_law_count = sum(
        len(shard.get('canonical_laws') or [])
        for shard in canonical_law_shards
    )
    robust_law_count = sum(
        1
        for shard in canonical_law_shards
        for law in shard.get('canonical_laws') or []
        if law.get('status') in {'robust_law', 'arithmetic_ready', 'confirmed_operator'}
    )
    raw_bytes = sum(
        int(shard.get('estimated_raw_bytes', 0) or 0)
        for shard in canonical_law_shards
    )
    canonical_bytes = sum(
        int(shard.get('estimated_canonical_bytes', 0) or 0)
        for shard in canonical_law_shards
    )
    actions = []
    if not canonical_law_shards:
        actions.append({
            'action_kind': 'compact_canonical_laws',
            'reason': 'long runs need reusable law summaries in addition to raw event shards',
            'runs_final': False,
        })
    return {
        'version': 1,
        'canonical_law_shard_count': len(canonical_law_shards),
        'canonical_law_count': canonical_law_count,
        'robust_law_count': robust_law_count,
        'estimated_raw_law_bytes': raw_bytes,
        'estimated_canonical_law_bytes': canonical_bytes,
        'estimated_law_compression_ratio': _safe_ratio(raw_bytes, canonical_bytes),
        'long_run_law_ready': bool(canonical_law_shards) and canonical_law_count > 0,
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


def _law_events_from_domain_worlds(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    events = []
    for record in records:
        domain_key = str(record.get('domain_key', 'unknown'))
        for equation in list(record.get('self_authored_equations') or []):
            expression = str(equation.get('expression') or '')
            if not expression:
                continue
            events.append({
                'law_family': _law_family_from_expression(expression),
                'status': 'domain_world_candidate',
                'expression': expression,
                'context': domain_key,
                'score': float(equation.get('confidence', 0.0) or 0.0),
                'source_kind': 'domain_world',
                'falsification_tests': list(equation.get('falsification_tests') or []),
            })
    return events


def _law_events_from_scientist_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    events = []
    for record in records:
        variants = ','.join(str(item) for item in record.get('variants') or [])
        context = f"scientist:{record.get('seed_start')}:{variants}"
        for invariant in list(record.get('invariant_consolidations') or []):
            expression = str(
                invariant.get('dominant_expression')
                or invariant.get('expression')
                or invariant.get('law_family')
                or invariant.get('domain_key')
                or ''
            )
            if not expression:
                continue
            events.append({
                'law_family': str(
                    invariant.get('law_family')
                    or invariant.get('relation_kind')
                    or _law_family_from_expression(expression)
                ),
                'status': str(invariant.get('status') or 'scientist_candidate'),
                'expression': expression,
                'context': context,
                'score': float(invariant.get('confidence', 0.0) or 0.0),
                'source_kind': 'autonomous_scientist',
                'falsification_tests': list(invariant.get('falsification_tests') or []),
            })
        for equation in list(record.get('authored_equation_extensions') or []):
            expression = str(equation.get('expression') or '')
            if not expression:
                continue
            events.append({
                'law_family': str(
                    equation.get('equation_kind')
                    or equation.get('grammar_key')
                    or _law_family_from_expression(expression)
                ),
                'status': 'authored_equation_extension',
                'expression': expression,
                'context': context,
                'score': float(equation.get('confidence', 0.0) or 0.0),
                'source_kind': 'autonomous_scientist',
                'falsification_tests': list(equation.get('falsification_tests') or []),
            })
    return events


def _law_events_from_arithmetic_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    events = []
    for record in records:
        context = f"arithmetic:{record.get('seed_start')}:{','.join(str(v) for v in record.get('variants') or [])}"
        for equation in list(record.get('self_authored_equations') or []):
            expression = str(equation.get('expression') or '')
            if not expression:
                continue
            events.append({
                'law_family': str(equation.get('target') or _law_family_from_expression(expression)),
                'status': str(record.get('status') or 'arithmetic_candidate'),
                'expression': expression,
                'context': context,
                'score': float(equation.get('confidence', 0.0) or 0.0),
                'source_kind': 'arithmetic_rediscovery',
                'falsification_tests': list(equation.get('falsification_tests') or []),
            })
    return events


def _law_events_from_operator_outcomes(
    outcomes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    events = []
    for outcome in outcomes:
        outcome_status = str(outcome.get('outcome') or '')
        if outcome_status not in {'confirmed', 'operator_prior_repair_confirmed', 'weak'}:
            continue
        parameters = dict(
            outcome.get('operator_prior_refined_parameters')
            or outcome.get('refined_parameters')
            or outcome.get('parameters')
            or {}
        )
        operator_kind = str(outcome.get('operator_kind') or outcome.get('operator_key') or 'operator')
        relation = str(parameters.get('relation') or 'unknown_relation')
        exponent = parameters.get('distance_exponent')
        expression = f"{operator_kind}:{relation}"
        if exponent is not None:
            expression = f"{expression}:distance_exponent~{_quantize(exponent, DEFAULT_PARAMETER_STEP)}"
        events.append({
            'law_family': operator_kind,
            'status': 'confirmed_operator' if 'confirmed' in outcome_status else outcome_status,
            'expression': expression,
            'context': str(outcome.get('context') or 'unknown'),
            'score': float(outcome.get('best_score', 0.0) or 0.0),
            'source_kind': 'operator_prior',
            'falsification_tests': [],
        })
    return events


def _canonical_law_summaries(
    events: list[dict[str, Any]],
    *,
    sample_limit: int,
) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for event in events:
        signature = {
            'law_family': str(event.get('law_family', 'unknown')),
            'expression_signature': _expression_signature(str(event.get('expression', ''))),
        }
        key = _stable_json(signature)
        group = groups.setdefault(
            key,
            {
                **signature,
                'support_count': 0,
                'contexts': set(),
                'source_kinds': set(),
                'statuses': set(),
                'best_score': 0.0,
                'falsification_tests': [],
                'examples': [],
            },
        )
        group['support_count'] += 1
        group['contexts'].add(str(event.get('context') or 'unknown'))
        group['source_kinds'].add(str(event.get('source_kind') or 'unknown'))
        group['statuses'].add(str(event.get('status') or 'unknown'))
        group['best_score'] = max(
            float(group['best_score']),
            float(event.get('score', 0.0) or 0.0),
        )
        for test in event.get('falsification_tests') or []:
            _append_unique(group['falsification_tests'], str(test), sample_limit)
        _append_sample(
            group['examples'],
            {
                'expression': event.get('expression'),
                'context': event.get('context'),
                'status': event.get('status'),
            },
            sample_limit,
        )

    summaries = []
    for group in groups.values():
        statuses = set(group.pop('statuses'))
        source_kinds = set(group.pop('source_kinds'))
        contexts = set(group.pop('contexts'))
        status = _canonical_status(statuses)
        summaries.append({
            **group,
            'contexts': sorted(contexts),
            'source_kinds': sorted(source_kinds),
            'status': status,
            'best_score': round(float(group['best_score']), 3),
            'source_statuses': sorted(statuses),
        })
    return sorted(
        summaries,
        key=lambda item: (
            item['status'] in {'robust_law', 'arithmetic_ready', 'confirmed_operator'},
            int(item['support_count']),
            float(item['best_score']),
            item['law_family'],
        ),
        reverse=True,
    )


def _canonical_status(statuses: set[str]) -> str:
    if 'robust_law' in statuses:
        return 'robust_law'
    if 'arithmetic_ready' in statuses:
        return 'arithmetic_ready'
    if 'confirmed_operator' in statuses:
        return 'confirmed_operator'
    if 'authored_equation_extension' in statuses:
        return 'authored_equation_extension'
    if statuses:
        return sorted(statuses)[0]
    return 'candidate'


def _law_family_from_expression(expression: str) -> str:
    text = expression.lower()
    if 'extent' in text or 'count' in text:
        return 'quantity_cardinality'
    if 'distance' in text or 'separation' in text:
        return 'distance_relation'
    if 'phase' in text or 'sin' in text or 'cos' in text:
        return 'periodic_relation'
    if 'state' in text or 'next' in text:
        return 'state_update'
    if 'transform' in text or 'coordinate' in text:
        return 'invariance_transform'
    return 'general_equation'


def _expression_signature(expression: str) -> str:
    lowered = expression.lower()
    pieces = []
    token = []
    for char in lowered:
        if char.isalnum() or char == '_':
            token.append(char)
        else:
            if token:
                pieces.append(''.join(token))
                token = []
            if char in {'+', '-', '*', '/', '=', '<', '>'}:
                pieces.append(char)
    if token:
        pieces.append(''.join(token))
    return ' '.join(pieces[:24]) or 'empty'


def _append_unique(values: list[str], value: str, limit: int):
    if value in values or len(values) >= limit:
        return
    values.append(value)


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
