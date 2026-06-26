"""Validation for bounded abstraction-transfer replay artifacts."""

from __future__ import annotations

import hashlib
import json
from typing import Any


ABSTRACTION_REPLAY_ARTIFACT_SCHEMA_VERSION = 'abstraction_replay_artifact.v1'


class AbstractionReplayArtifactValidationError(ValueError):
    """Raised when an abstraction replay artifact violates the evidence contract."""


_KIND_TO_EVIDENCE_TYPE = {
    'bounded_abstraction_transfer_replay_pack': 'replay_candidate_benefit',
    'bounded_abstraction_transfer_replay_matrix': 'candidate_replay_matrix_evidence',
    'bounded_abstraction_transfer_negative_control_matrix': (
        'candidate_replay_negative_control_evidence'
    ),
}


def artifact_content_hash(artifact: dict[str, Any]) -> str:
    """Hash an artifact payload without its self-referential content hash."""
    payload = {
        key: value
        for key, value in artifact.items()
        if key != 'artifact_content_hash'
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode('utf-8')
    ).hexdigest()


def validate_abstraction_replay_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    """Validate an abstraction replay artifact and return a compact summary."""
    if not isinstance(artifact, dict):
        _fail('artifact', 'artifact must be a JSON object')
    run_kind = _required_str(artifact, 'run_kind')
    expected_evidence_type = _KIND_TO_EVIDENCE_TYPE.get(run_kind)
    if expected_evidence_type is None:
        _fail('run_kind', f'unsupported abstraction replay artifact kind: {run_kind}')
    schema_version = _required_str(artifact, 'artifact_schema_version')
    if schema_version != ABSTRACTION_REPLAY_ARTIFACT_SCHEMA_VERSION:
        _fail(
            'artifact_schema_version',
            f'expected {ABSTRACTION_REPLAY_ARTIFACT_SCHEMA_VERSION}',
        )
    evidence_type = _required_str(artifact, 'evidence_type')
    if evidence_type != expected_evidence_type:
        _fail(
            'evidence_type',
            f'wrong evidence type for {run_kind}: expected {expected_evidence_type}',
        )
    _require_bool(artifact, 'runs_final', False)
    _require_bool(artifact, 'mutates_runtime_theory_memory', False)
    _require_bool(artifact, 'candidate_not_causal', True)
    _require_bool(artifact, 'project_owned_checkpoint_claimed', False)
    _require_bool(artifact, 'third_party_checkpoint_used')
    _require_bool(artifact, 'hf_validation_used')
    wording = _required_str(artifact, 'candidate_not_causal_wording')
    lowered_wording = wording.lower()
    if 'causal proof' not in lowered_wording or 'benchmark proof' not in lowered_wording:
        _fail(
            'candidate_not_causal_wording',
            'missing no-overclaim wording for causal proof and benchmark proof',
        )
    if 'project-owned' not in lowered_wording and 'project owned' not in lowered_wording:
        _fail(
            'candidate_not_causal_wording',
            'missing project-owned checkpoint/model boundary wording',
        )
    config = _required_dict(artifact, 'config')
    if 'seed_start' not in config:
        _fail('config.seed_start', 'missing seed_start in artifact config')
    comparisons = _required_list(artifact, 'comparisons')
    if not comparisons:
        _fail('comparisons', 'artifact must include at least one comparison row')
    _validate_common_rows(comparisons)
    content_hash = _required_str(artifact, 'artifact_content_hash')
    if len(content_hash) != 64 or any(ch not in '0123456789abcdef' for ch in content_hash):
        _fail('artifact_content_hash', 'artifact content hash must be lowercase sha256 hex')
    expected_hash = artifact_content_hash(artifact)
    if content_hash != expected_hash:
        _fail('artifact_content_hash', 'artifact content hash does not match payload')
    if run_kind == 'bounded_abstraction_transfer_replay_pack':
        aggregate = _required_dict(artifact, 'evidence_counts')
        _require_int_keys(
            aggregate,
            ['candidate_win', 'control_no_gain', 'weak_or_absent_bridge', 'control_win'],
            'evidence_counts',
        )
    elif run_kind == 'bounded_abstraction_transfer_replay_matrix':
        _required_str(artifact, 'matrix_id')
        aggregate = _validate_matrix_aggregate(artifact)
        if aggregate['comparison_count'] != len(comparisons):
            _fail('aggregate_counts.comparison_count', 'comparison count does not match rows')
    else:
        _required_str(artifact, 'matrix_id')
        aggregate = _validate_matrix_aggregate(artifact)
        decision = _required_dict(artifact, 'decision')
        decision_label = _required_str(decision, 'decision')
        _require_bool(decision, 'promote_bridge')
        if decision_label == 'hold_for_more_evidence' and decision.get('promote_bridge') is not False:
            _fail('decision.promote_bridge', 'hold_for_more_evidence must not promote')
        _require_int_keys(
            aggregate,
            [
                'candidate_survives_negative_controls_count',
                'shuffled_or_mismatched_control_win_count',
            ],
            'aggregate_counts',
        )
        if 'candidate_survives_negative_controls_rate' not in aggregate:
            _fail(
                'aggregate_counts.candidate_survives_negative_controls_rate',
                'missing negative-control survival rate',
            )
        for row in comparisons:
            _required_str(row, 'negative_control_type')
            _required_str(row, 'shuffled_control_type')
            _required_str(row, 'negative_control_class')
            _required_dict(row, 'shuffled_outcome')
    return {
        'valid': True,
        'schema_version': schema_version,
        'run_kind': run_kind,
        'evidence_type': evidence_type,
        'comparison_count': len(comparisons),
        'content_hash': content_hash,
    }


def _validate_common_rows(comparisons: list[Any]) -> None:
    for index, row in enumerate(comparisons):
        if not isinstance(row, dict):
            _fail(f'comparisons[{index}]', 'comparison row must be an object')
        _required_str(row, 'scenario_id')
        _required_str(row, 'selected_replay_class')
        _required_dict(row, 'candidate_outcome')
        _required_dict(row, 'control_outcome')
        _required_str(row['candidate_outcome'], 'outcome')
        _required_str(row['control_outcome'], 'outcome')


def _validate_matrix_aggregate(artifact: dict[str, Any]) -> dict[str, Any]:
    aggregate = _required_dict(artifact, 'aggregate_counts')
    _require_int_keys(
        aggregate,
        [
            'comparison_count',
            'candidate_win_count',
            'control_no_gain_count',
            'weak_or_absent_count',
        ],
        'aggregate_counts',
    )
    if 'candidate_win_rate' not in aggregate:
        _fail('aggregate_counts.candidate_win_rate', 'missing candidate win rate')
    if not isinstance(aggregate['candidate_win_rate'], (int, float)):
        _fail('aggregate_counts.candidate_win_rate', 'candidate win rate must be numeric')
    return aggregate


def _require_int_keys(payload: dict[str, Any], keys: list[str], prefix: str) -> None:
    for key in keys:
        if key not in payload:
            _fail(f'{prefix}.{key}', 'missing aggregate count')
        if not isinstance(payload[key], int):
            _fail(f'{prefix}.{key}', 'aggregate count must be an integer')


def _required_str(payload: dict[str, Any], key: str) -> str:
    if key not in payload:
        _fail(key, 'missing required field')
    value = payload[key]
    if not isinstance(value, str) or not value:
        _fail(key, 'field must be a non-empty string')
    return value


def _required_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    if key not in payload:
        _fail(key, 'missing required field')
    value = payload[key]
    if not isinstance(value, dict):
        _fail(key, 'field must be an object')
    return value


def _required_list(payload: dict[str, Any], key: str) -> list[Any]:
    if key not in payload:
        _fail(key, 'missing required field')
    value = payload[key]
    if not isinstance(value, list):
        _fail(key, 'field must be a list')
    return value


def _require_bool(payload: dict[str, Any], key: str, expected: bool | None = None) -> None:
    if key not in payload:
        _fail(key, 'missing required field')
    value = payload[key]
    if not isinstance(value, bool):
        _fail(key, 'field must be a boolean')
    if expected is not None and value is not expected:
        _fail(key, f'expected {expected}')


def _fail(field: str, message: str) -> None:
    raise AbstractionReplayArtifactValidationError(f'{field}: {message}')
