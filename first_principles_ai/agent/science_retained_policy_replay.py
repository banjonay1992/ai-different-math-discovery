"""Science retained-policy replay evaluator.

Replays retained science coordination policy memory against simple controls on
plain symbolic evidence. This produces replay_candidate_benefit evidence only:
it is not causal proof, not a science benchmark, and not a project-owned
checkpoint claim.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .module_chat_adapter import (
    build_module_chat_message,
    label_leak_terms,
    module_chat_message_id,
    stable_digest,
    validate_module_chat_message,
    validate_participant,
)
from .science_coordination_interaction_history import SCIENCE_COORDINATION_INTERACTION_HISTORY_LEDGER_KIND
from .science_history_guided_policy import SCIENCE_HISTORY_GUIDED_POLICY_LEDGER_KIND
from .science_history_guided_policy_outcome import SCIENCE_HISTORY_GUIDED_POLICY_OUTCOME_LEDGER_KIND
from .science_history_guided_policy_retention import SCIENCE_HISTORY_GUIDED_POLICY_RETENTION_LEDGER_KIND


SCIENCE_RETAINED_POLICY_REPLAY_LEDGER_KIND = 'ai_different.science_retained_policy_replay_ledger'


def empty_science_retained_policy_replay_ledger() -> dict[str, Any]:
    return {
        'schema_version': 1,
        'ledger_kind': SCIENCE_RETAINED_POLICY_REPLAY_LEDGER_KIND,
        'processed_message_ids': [],
        'processed_source_hashes': [],
        'replayed_policy_keys': [],
        'replay_records': [],
        'replay_rows': [],
        'outgoing_response_ids': [],
        'latest': {},
    }


def load_science_retained_policy_replay_ledger(path: str | Path | None) -> dict[str, Any]:
    if not path or not Path(path).exists():
        return empty_science_retained_policy_replay_ledger()
    with Path(path).open('r', encoding='utf-8') as handle:
        ledger = json.load(handle)
    return validate_science_retained_policy_replay_ledger(ledger)


def write_science_retained_policy_replay_ledger(path: str | Path, ledger: dict[str, Any]) -> Path:
    validated = validate_science_retained_policy_replay_ledger(ledger)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(validated, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return output


def validate_science_retained_policy_replay_ledger(ledger: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(ledger, dict):
        raise ValueError('science retained-policy replay ledger must be a JSON object')
    if ledger.get('ledger_kind') != SCIENCE_RETAINED_POLICY_REPLAY_LEDGER_KIND:
        raise ValueError('science retained-policy replay ledger has wrong ledger_kind')
    for field in ('processed_message_ids', 'processed_source_hashes', 'replayed_policy_keys', 'outgoing_response_ids'):
        values = ledger.get(field)
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            raise ValueError(f'{field} must be strings')
    for field in ('replay_records', 'replay_rows'):
        values = ledger.get(field)
        if not isinstance(values, list) or not all(isinstance(item, dict) for item in values):
            raise ValueError(f'{field} must be objects')
    latest = ledger.get('latest', {})
    if not isinstance(latest, dict):
        raise ValueError('science retained-policy replay latest must be an object')
    validated = {
        'schema_version': int(ledger.get('schema_version', 1) or 1),
        'ledger_kind': SCIENCE_RETAINED_POLICY_REPLAY_LEDGER_KIND,
        'processed_message_ids': _unique_strings(ledger['processed_message_ids']),
        'processed_source_hashes': _unique_strings(ledger['processed_source_hashes']),
        'replayed_policy_keys': _unique_strings(ledger['replayed_policy_keys']),
        'replay_records': list(ledger['replay_records']),
        'replay_rows': list(ledger['replay_rows']),
        'outgoing_response_ids': _unique_strings(ledger['outgoing_response_ids']),
        'latest': dict(latest),
    }
    if ledger.get('ledger_hash'):
        validated['ledger_hash'] = str(ledger['ledger_hash'])
    if ledger.get('artifact_path'):
        validated['artifact_path'] = str(ledger['artifact_path'])
    return validated


def read_science_retained_policy_replay_transcript(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {'path': None, 'messages': [], 'invalid_messages': []}
    transcript = Path(path)
    if not transcript.exists():
        return {'path': str(transcript), 'messages': [], 'invalid_messages': []}
    messages = []
    invalid = []
    with transcript.open('r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                messages.append(validate_module_chat_message(json.loads(text)))
            except (json.JSONDecodeError, ValueError) as error:
                invalid.append({'line': line_number, 'error': str(error), 'raw': text})
    return {'path': str(transcript), 'messages': messages, 'invalid_messages': invalid}


def load_plain_json(path: str | Path | None) -> dict[str, Any]:
    if not path or not Path(path).exists():
        return {}
    with Path(path).open('r', encoding='utf-8') as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError('plain science retained-policy replay input must be a JSON object')
    return value


def build_science_retained_policy_replay_evaluation(
    *,
    transcript_messages: list[dict[str, Any]],
    replay_ledger: dict[str, Any],
    retention_ledger: dict[str, Any] | None = None,
    policy_outcome_ledger: dict[str, Any] | None = None,
    history_guided_policy_ledger: dict[str, Any] | None = None,
    interaction_history_ledger: dict[str, Any] | None = None,
    theory_memory_ledger: dict[str, Any] | None = None,
    campaign_ledger: dict[str, Any] | None = None,
    hypothesis_ledger: dict[str, Any] | None = None,
    experiment_ledger: dict[str, Any] | None = None,
    module_chat_ledger: dict[str, Any] | None = None,
    prior_replay_ledger: dict[str, Any] | None = None,
    runtime_memory_data: dict[str, Any] | None = None,
    runtime_memory_hash_state: dict[str, Any],
    project_owned_boundary: dict[str, Any],
    hf_use_status: dict[str, Any] | None = None,
    artifact_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    ledger = validate_science_retained_policy_replay_ledger(replay_ledger)
    retention = _valid_kind_or_empty(retention_ledger or {}, SCIENCE_HISTORY_GUIDED_POLICY_RETENTION_LEDGER_KIND, 'retention_records')
    outcome = _valid_kind_or_empty(policy_outcome_ledger or {}, SCIENCE_HISTORY_GUIDED_POLICY_OUTCOME_LEDGER_KIND, 'outcome_records')
    policy = _valid_kind_or_empty(history_guided_policy_ledger or {}, SCIENCE_HISTORY_GUIDED_POLICY_LEDGER_KIND, 'policy_records')
    interaction_history = _valid_kind_or_empty(interaction_history_ledger or {}, SCIENCE_COORDINATION_INTERACTION_HISTORY_LEDGER_KIND, 'scorecard_records')
    theory_memory = _valid_plain_or_empty(theory_memory_ledger or {})
    campaign = _valid_plain_or_empty(campaign_ledger or {})
    hypothesis = _valid_plain_or_empty(hypothesis_ledger or {})
    experiment = _valid_plain_or_empty(experiment_ledger or {})
    module_chat = _valid_plain_or_empty(module_chat_ledger or {})
    prior_replay = _valid_prior_replay_or_empty(prior_replay_ledger or {})
    runtime_memory = dict(runtime_memory_data or {})
    hf_status = dict(hf_use_status or {'hf_validation_used': False})
    source_hash = _source_hash(retention, outcome, policy, interaction_history, theory_memory, campaign, hypothesis, experiment, module_chat, prior_replay, runtime_memory, hf_status)
    source_is_new = source_hash not in set(ledger['processed_source_hashes'])
    processed = set(ledger['processed_message_ids'])
    new_messages = [message for message in transcript_messages if module_chat_message_id(message) not in processed]
    skipped_messages = [message for message in transcript_messages if module_chat_message_id(message) in processed]
    rows = _extract_replay_rows(
        ledger['replay_rows'],
        retention,
        outcome,
        policy,
        interaction_history,
        theory_memory,
        campaign,
        hypothesis,
        experiment,
        module_chat,
        transcript_messages,
    )
    if not new_messages and not source_is_new:
        selected = _noop_replay('no new retained-policy replay evidence or source ledger state')
    else:
        selected = _select_replay(rows=rows, project_owned_boundary=project_owned_boundary)
    replay_id = 'science_retained_policy_replay_' + stable_digest({
        'source_hash': source_hash,
        'new_message_ids': [module_chat_message_id(message) for message in new_messages],
        'class': selected['selected_replay_class'],
        'retained': selected.get('retained_policy_key'),
        'control': selected.get('control_policy_key'),
    })[:16]
    selected['science_retained_policy_replay_id'] = replay_id
    message = export_science_retained_policy_replay_message(
        selected,
        rows=rows,
        runtime_memory_hash_state=runtime_memory_hash_state,
        project_owned_boundary=project_owned_boundary,
        hf_use_status=hf_status,
        source_hashes={
            'science_retained_policy_replay_source_hash': source_hash,
            'retention_ledger_hash': retention.get('ledger_hash'),
            'policy_outcome_ledger_hash': outcome.get('ledger_hash'),
            'history_guided_policy_ledger_hash': policy.get('ledger_hash'),
            'interaction_history_ledger_hash': interaction_history.get('ledger_hash'),
            'theory_memory_ledger_hash': theory_memory.get('ledger_hash'),
            'campaign_ledger_hash': campaign.get('ledger_hash'),
            'hypothesis_ledger_hash': hypothesis.get('ledger_hash'),
            'experiment_ledger_hash': experiment.get('ledger_hash'),
            'module_chat_ledger_hash': module_chat.get('ledger_hash'),
            'prior_replay_ledger_hash': prior_replay.get('ledger_hash'),
        },
    )
    new_ids = [module_chat_message_id(message) for message in new_messages]
    outgoing_id = module_chat_message_id(message) if message else None
    if selected['selected_replay_class'] != 'summarize_noop':
        ledger['replayed_policy_keys'] = _unique_strings(list(ledger['replayed_policy_keys']) + [_replay_key(selected)])
    if new_messages:
        ledger['processed_message_ids'] = _unique_strings(list(ledger['processed_message_ids']) + new_ids)
    if source_is_new:
        ledger['processed_source_hashes'] = _unique_strings(list(ledger['processed_source_hashes']) + [source_hash])
    if outgoing_id:
        ledger['outgoing_response_ids'] = _unique_strings(list(ledger['outgoing_response_ids']) + [outgoing_id])
    state_counts = _state_counts(selected)
    latest = {
        'science_retained_policy_replay_id': replay_id,
        'new_message_count': len(new_messages),
        'skipped_message_count': len(skipped_messages),
        'source_is_new': source_is_new,
        'state_counts': state_counts,
        'selected_replay_class': selected['selected_replay_class'],
        'next_planner_recommendation': selected.get('next_planner_recommendation'),
        'selected_action': selected['selected_action'],
        'chosen_recipient': message.get('recipient') if message else None,
        'outbox_count': 1 if message else 0,
        'label_leaks': selected.get('label_leaks', []),
        'checkpoint_boundary_state': selected.get('checkpoint_boundary_state'),
        'runtime_memory_hash_state': dict(runtime_memory_hash_state),
        'runtime_memory_mutated': not bool(runtime_memory_hash_state.get('unchanged', True)),
        'hf_use_status': hf_status,
    }
    record = {
        'science_retained_policy_replay_id': replay_id,
        'replay_hash': stable_digest({'replay_id': replay_id, 'selected': selected}),
        'processed_message_ids': new_ids,
        'source_hash': source_hash,
        'source_retention_ids': selected.get('source_retention_ids'),
        'source_outcome_ids': selected.get('source_outcome_ids'),
        'source_policy_ids': selected.get('source_policy_ids'),
        'source_interaction_history_ids': selected.get('source_interaction_history_ids'),
        'retained_policy_key': selected.get('retained_policy_key'),
        'control_policy_key': selected.get('control_policy_key'),
        'replay_target_ids': selected.get('replay_target_ids'),
        'candidate_evidence_score': selected.get('candidate_evidence_score'),
        'control_evidence_score': selected.get('control_evidence_score'),
        'evidence_counts': selected.get('evidence_counts'),
        'selected_replay_class': selected['selected_replay_class'],
        'next_planner_recommendation': selected.get('next_planner_recommendation'),
        'checkpoint_boundary_state': selected.get('checkpoint_boundary_state'),
        'candidate_not_causal_wording': selected.get('candidate_not_causal_wording'),
        'no_overclaiming_proof': selected.get('no_overclaiming_proof'),
        'hf_use_status': hf_status,
        'outgoing_response_id': outgoing_id,
    }
    ledger['replay_rows'] = rows
    if new_messages or source_is_new or message is not None:
        ledger['replay_records'] = list(ledger['replay_records']) + [record]
    ledger['latest'] = latest
    if artifact_path is not None:
        ledger['artifact_path'] = str(Path(artifact_path))
    ledger['ledger_hash'] = stable_digest({key: value for key, value in ledger.items() if key != 'ledger_hash'})
    return ledger, message


def export_science_retained_policy_replay_message(
    selected: dict[str, Any],
    *,
    rows: list[dict[str, Any]],
    runtime_memory_hash_state: dict[str, Any],
    project_owned_boundary: dict[str, Any],
    hf_use_status: dict[str, Any],
    source_hashes: dict[str, Any],
    topic: str = 'ai_different.science_retained_policy_replay',
) -> dict[str, Any] | None:
    if selected['selected_replay_class'] == 'summarize_noop':
        return None
    recipient = validate_participant(selected.get('selected_recipient') or 'history')
    row = _find_row(rows, selected.get('retained_policy_key')) or {}
    leak_terms = label_leak_terms({'selected': selected, 'row': row})
    body = {
        'module': 'AI Different',
        'response_kind': 'science_retained_policy_replay',
        'science_retained_policy_replay_id': selected.get('science_retained_policy_replay_id'),
        'selected_replay_class': selected['selected_replay_class'],
        'selected_action': selected['selected_action'],
        'next_planner_recommendation': selected.get('next_planner_recommendation'),
        'selected_recipient': recipient,
        'retained_policy_key': selected.get('retained_policy_key'),
        'control_policy_key': selected.get('control_policy_key'),
        'replay_target_ids': selected.get('replay_target_ids') or [],
        'candidate_evidence_score': selected.get('candidate_evidence_score'),
        'control_evidence_score': selected.get('control_evidence_score'),
        'evidence_counts': selected.get('evidence_counts') or {},
        'source_retention_ids': selected.get('source_retention_ids') or [],
        'source_outcome_ids': selected.get('source_outcome_ids') or [],
        'source_policy_ids': selected.get('source_policy_ids') or [],
        'source_interaction_history_ids': selected.get('source_interaction_history_ids') or [],
        'tested_commits': selected.get('tested_commits') or [],
        'tested_tests': selected.get('tested_tests') or [],
        'candidate_not_causal': True,
        'candidate_not_causal_wording': selected.get('candidate_not_causal_wording'),
        'checkpoint_boundary_state': selected.get('checkpoint_boundary_state'),
        'checkpoint_boundary_notes': selected.get('checkpoint_boundary_notes') or [],
        'no_overclaiming_proof': selected.get('no_overclaiming_proof'),
        'source_ledger_hashes': dict(source_hashes),
        'runtime_memory_hash_state': dict(runtime_memory_hash_state),
        'runtime_memory_mutated': not bool(runtime_memory_hash_state.get('unchanged', True)),
        'project_owned_boundary': dict(project_owned_boundary),
        'third_party_checkpoint_used': bool(project_owned_boundary.get('third_party_checkpoint_used')),
        'project_owned_checkpoint_claimed': bool(project_owned_boundary.get('project_owned_checkpoint_verified')),
        'hf_use_status': dict(hf_use_status),
        'hf_validation_used': bool(hf_use_status.get('hf_validation_used')),
        'no_sibling_imports': True,
        'label_leaks': leak_terms,
        'label_clean': not leak_terms,
    }
    return build_module_chat_message(
        sender='ai_different',
        recipient=recipient,
        topic=topic,
        body=body,
        evidence={
            'science_retained_policy_replay_id': body['science_retained_policy_replay_id'],
            'selected_replay_class': body['selected_replay_class'],
            'retained_policy_key': body['retained_policy_key'],
            'control_policy_key': body['control_policy_key'],
            'replay_candidate_benefit': body['candidate_evidence_score'] > body['control_evidence_score'],
            'hf_validation_used': body['hf_validation_used'],
            'label_clean': body['label_clean'],
            'no_sibling_imports': True,
        },
        tags=['ai_different', 'science_retained_policy_replay', body['selected_replay_class'], 'label_clean' if body['label_clean'] else 'label_review_needed'],
    )


def write_science_retained_policy_replay_outbox_jsonl(path: str | Path, message: dict[str, Any] | None) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text((json.dumps(message, sort_keys=True) + '\n') if message is not None else '', encoding='utf-8')
    return output


def _valid_kind_or_empty(ledger, kind, record_field):
    if not ledger:
        return {'ledger_kind': kind, record_field: [], 'ledger_hash': None}
    if ledger.get('ledger_kind') != kind:
        raise ValueError(f'plain science retained-policy replay source has wrong ledger_kind: {ledger.get("ledger_kind")}')
    return dict(ledger)


def _valid_plain_or_empty(ledger):
    if not ledger:
        return {}
    if not isinstance(ledger, dict):
        raise ValueError('plain science retained-policy replay source must be a JSON object')
    return dict(ledger)


def _valid_prior_replay_or_empty(ledger):
    if not ledger:
        return {}
    if ledger.get('ledger_kind') != SCIENCE_RETAINED_POLICY_REPLAY_LEDGER_KIND:
        raise ValueError('prior science retained-policy replay ledger has wrong ledger_kind')
    return validate_science_retained_policy_replay_ledger(ledger)


def _extract_replay_rows(existing, retention, outcome, policy, interaction_history, theory_memory, campaign, hypothesis, experiment, module_chat, messages):
    rows: dict[str, dict[str, Any]] = {}
    for row in list(existing or []):
        _upsert_row(rows, row)
    for record in list(retention.get('retention_records') or []):
        _upsert_row(rows, _row_from_source(record, 'retention_record'))
    for record in list(retention.get('retention_rows') or []):
        _upsert_row(rows, _row_from_source(record, 'retention_row'))
    for record in list(outcome.get('outcome_records') or []):
        _upsert_row(rows, _row_from_source(record, 'policy_outcome'))
    for record in list(policy.get('policy_records') or []):
        _upsert_row(rows, _row_from_source(record, 'policy'))
    for record in list(interaction_history.get('scorecard_records') or []):
        _upsert_row(rows, _row_from_source(record, 'interaction_history'))
    for row in _plain_rows(theory_memory, ('theory_memory_rows', 'theory_records', 'theories')):
        _upsert_row(rows, _row_from_source(row, 'theory_memory'))
    for row in _plain_rows(campaign, ('campaign_rows', 'campaigns', 'campaign_records')):
        _upsert_row(rows, _row_from_source(row, 'campaign'))
    for row in _plain_rows(hypothesis, ('hypothesis_rows', 'hypotheses', 'hypothesis_records')):
        _upsert_row(rows, _row_from_source(row, 'hypothesis'))
    for row in _plain_rows(experiment, ('experiment_rows', 'simulation_rows', 'simulation_results', 'requests')):
        _upsert_row(rows, _row_from_source(row, 'experiment'))
    for row in _plain_rows(module_chat, ('messages', 'records', 'module_chat_rows')):
        if isinstance(row, dict) and {'sender', 'recipient', 'topic', 'body', 'evidence'}.issubset(row):
            _upsert_row(rows, _row_from_message(row))
    for message in messages:
        _upsert_row(rows, _row_from_message(message))
    return [_finalize_row(row) for row in rows.values()]


def _row_from_message(message):
    body = dict(message.get('body') or {})
    evidence = dict(message.get('evidence') or {})
    source = {**body, **evidence}
    source['source_retention_id'] = body.get('science_history_guided_policy_retention_id') or evidence.get('science_history_guided_policy_retention_id')
    source['source_outcome_id'] = body.get('science_history_guided_policy_outcome_id') or evidence.get('science_history_guided_policy_outcome_id')
    source['source_policy_id'] = body.get('science_history_guided_policy_id') or evidence.get('science_history_guided_policy_id')
    source['lineage'] = [f"message:{message.get('sender')}"]
    source['label_leaks'] = label_leak_terms({'body': body, 'evidence': evidence})
    return _row_from_source(source, 'message')


def _row_from_source(source, lineage):
    source = dict(source or {})
    retained_key = _retained_key_from_source(source)
    selected = source.get('selected_replay_class') or source.get('selected_decision') or source.get('selected_outcome_class') or source.get('selected_action') or source.get('retention_decision') or source.get('status')
    counts = {'candidate_win': 0, 'control_win': 0, 'underpowered': 0, 'no_gain': 0, 'boundary': 0}
    signal = _signal_from_value(selected, source.get('retention_decision'), source)
    if signal in counts:
        counts[signal] += 1
    if bool(source.get('third_party_checkpoint_used')) or source.get('checkpoint_boundary_state') == 'repair':
        counts['boundary'] += 1
    candidate_score = _numeric(source.get('candidate_evidence_score') or source.get('replay_candidate_score'))
    control_score = _numeric(source.get('control_evidence_score') or source.get('replay_control_score'))
    if candidate_score == 0 and signal == 'candidate_win':
        candidate_score = 1
    if control_score == 0 and signal == 'control_win':
        control_score = 1
    return {
        'retained_policy_key': retained_key,
        'control_policy_key': source.get('control_policy_key') or 'science_only_control',
        'replay_target_ids': _unique_strings(list(source.get('replay_target_ids') or []) + list(source.get('target_campaign_ids') or []) + list(source.get('target_hypothesis_ids') or []) + list(source.get('target_simulation_ids') or [])),
        'source_retention_ids': _unique_strings(list(source.get('source_retention_ids') or []) + ([source.get('source_retention_id')] if source.get('source_retention_id') else []) + ([source.get('science_history_guided_policy_retention_id')] if source.get('science_history_guided_policy_retention_id') else [])),
        'source_outcome_ids': _unique_strings(list(source.get('source_outcome_ids') or []) + ([source.get('source_outcome_id')] if source.get('source_outcome_id') else []) + ([source.get('science_history_guided_policy_outcome_id')] if source.get('science_history_guided_policy_outcome_id') else [])),
        'source_policy_ids': _unique_strings(list(source.get('source_policy_ids') or []) + ([source.get('source_policy_id')] if source.get('source_policy_id') else []) + ([source.get('science_history_guided_policy_id')] if source.get('science_history_guided_policy_id') else [])),
        'source_interaction_history_ids': _unique_strings(list(source.get('source_interaction_history_ids') or []) + ([source.get('science_coordination_interaction_history_id')] if source.get('science_coordination_interaction_history_id') else [])),
        'evidence_counts': counts,
        'candidate_evidence_score': candidate_score,
        'control_evidence_score': control_score,
        'tested_commits': list(source.get('tested_commits') or source.get('source_commits') or []),
        'tested_tests': list(source.get('tested_tests') or source.get('source_tests') or []),
        'checkpoint_boundary_state': 'repair' if counts['boundary'] else source.get('checkpoint_boundary_state') or 'clean',
        'checkpoint_boundary_notes': list(source.get('checkpoint_boundary_notes') or (['checkpoint boundary preserved'] if counts['boundary'] else [])),
        'label_leaks': list(source.get('label_leaks') or []),
        'lineage': list(source.get('lineage') or [lineage]),
    }


def _upsert_row(rows, incoming):
    key = incoming.get('retained_policy_key') or stable_digest(incoming)[:12]
    current = rows.setdefault(str(key), {
        'retained_policy_key': incoming.get('retained_policy_key'),
        'control_policy_key': incoming.get('control_policy_key') or 'science_only_control',
        'replay_target_ids': [],
        'source_retention_ids': [],
        'source_outcome_ids': [],
        'source_policy_ids': [],
        'source_interaction_history_ids': [],
        'evidence_counts': {'candidate_win': 0, 'control_win': 0, 'underpowered': 0, 'no_gain': 0, 'boundary': 0},
        'candidate_evidence_score': 0,
        'control_evidence_score': 0,
        'tested_commits': [],
        'tested_tests': [],
        'checkpoint_boundary_state': 'clean',
        'checkpoint_boundary_notes': [],
        'label_leaks': [],
        'lineage': [],
    })
    for field in ('replay_target_ids', 'source_retention_ids', 'source_outcome_ids', 'source_policy_ids', 'source_interaction_history_ids', 'tested_commits', 'tested_tests', 'checkpoint_boundary_notes', 'label_leaks', 'lineage'):
        current[field] = _unique_strings(list(current.get(field) or []) + list(incoming.get(field) or []))
    counts = dict(current.get('evidence_counts') or {})
    for count_key, value in dict(incoming.get('evidence_counts') or {}).items():
        counts[count_key] = int(counts.get(count_key, 0) or 0) + int(value or 0)
    current['evidence_counts'] = counts
    current['candidate_evidence_score'] = _numeric(current.get('candidate_evidence_score')) + _numeric(incoming.get('candidate_evidence_score'))
    current['control_evidence_score'] = _numeric(current.get('control_evidence_score')) + _numeric(incoming.get('control_evidence_score'))
    current['checkpoint_boundary_state'] = 'repair' if current.get('checkpoint_boundary_state') == 'repair' or incoming.get('checkpoint_boundary_state') == 'repair' else 'clean'


def _finalize_row(row):
    if row.get('label_leaks'):
        row['checkpoint_boundary_state'] = 'repair'
        row['evidence_counts']['boundary'] = int(row['evidence_counts'].get('boundary', 0) or 0) + 1
    if row.get('candidate_evidence_score', 0) == 0 and _count(row, 'candidate_win'):
        row['candidate_evidence_score'] = _count(row, 'candidate_win')
    if row.get('control_evidence_score', 0) == 0 and _count(row, 'control_win'):
        row['control_evidence_score'] = _count(row, 'control_win')
    row['replay_row_hash'] = stable_digest({
        'retained': row.get('retained_policy_key'),
        'control': row.get('control_policy_key'),
        'counts': row.get('evidence_counts'),
        'candidate': row.get('candidate_evidence_score'),
        'control_score': row.get('control_evidence_score'),
        'boundary': row.get('checkpoint_boundary_state'),
    })
    return row


def _select_replay(*, rows, project_owned_boundary):
    for row in rows:
        if project_owned_boundary.get('third_party_checkpoint_used') or row.get('checkpoint_boundary_state') == 'repair' or _count(row, 'boundary'):
            boundary_row = dict(row)
            boundary_row['checkpoint_boundary_state'] = 'repair'
            boundary_row['checkpoint_boundary_notes'] = _unique_strings(list(boundary_row.get('checkpoint_boundary_notes') or []) + ['checkpoint boundary preserved'])
            return _replay(boundary_row, 'preserve_checkpoint_boundary_replay', 'preserve checkpoint boundary before replay use')
    for key, replay_class in (
        ('code_simulation_before_science', 'replay_retained_code_simulation_ordering_beats_control'),
        ('proof_before_hypothesis', 'replay_retained_proof_before_hypothesis_ordering_beats_control'),
        ('language_summary_before_science', 'replay_retained_language_summary_ordering_beats_control'),
        ('history_policy_memory_before_science', 'replay_retained_history_memory_ordering_beats_control'),
    ):
        for row in rows:
            if row.get('retained_policy_key') == key and _candidate_beats_control(row):
                return _replay(row, replay_class, f'use retained {key} ordering as candidate replay benefit evidence')
    for row in rows:
        if _control_beats_candidate(row):
            return _replay(row, 'replay_control_or_regression_beats_retained_policy', 'weaken retained policy pending more replay evidence')
    for row in rows:
        if _count(row, 'underpowered'):
            return _replay(row, 'replay_underpowered_waiting_for_more_evidence', 'keep replay waiting for more evidence')
    for row in rows:
        if _count(row, 'no_gain') or _candidate_ties_control(row):
            return _replay(row, 'replay_no_measurable_retained_policy_gain', 'record no measurable retained-policy replay gain')
    return _noop_replay('no retained-policy replay decision selected')


def _replay(row, replay_class, recommendation):
    return {
        'selected_replay_class': replay_class,
        'selected_action': replay_class,
        'selected_recipient': 'history',
        'next_planner_recommendation': recommendation,
        'retained_policy_key': row.get('retained_policy_key'),
        'control_policy_key': row.get('control_policy_key') or 'science_only_control',
        'replay_target_ids': list(row.get('replay_target_ids') or []),
        'candidate_evidence_score': _numeric(row.get('candidate_evidence_score')),
        'control_evidence_score': _numeric(row.get('control_evidence_score')),
        'evidence_counts': dict(row.get('evidence_counts') or {}),
        'source_retention_ids': list(row.get('source_retention_ids') or []),
        'source_outcome_ids': list(row.get('source_outcome_ids') or []),
        'source_policy_ids': list(row.get('source_policy_ids') or []),
        'source_interaction_history_ids': list(row.get('source_interaction_history_ids') or []),
        'tested_commits': list(row.get('tested_commits') or []),
        'tested_tests': list(row.get('tested_tests') or []),
        'checkpoint_boundary_state': row.get('checkpoint_boundary_state') or 'clean',
        'checkpoint_boundary_notes': list(row.get('checkpoint_boundary_notes') or []),
        'candidate_not_causal_wording': 'Replay_candidate_benefit only: retained science policy memory is compared with a symbolic control on replayable evidence, not causal proof.',
        'no_overclaiming_proof': 'replay_candidate_benefit only; not causal proof, not live AGI, not a science benchmark, and no project-owned checkpoint claim without local verification',
        'label_leaks': list(row.get('label_leaks') or []),
    }


def _noop_replay(reason):
    return {
        'selected_replay_class': 'summarize_noop',
        'selected_action': 'summarize_noop',
        'selected_recipient': None,
        'next_planner_recommendation': reason,
        'retained_policy_key': None,
        'control_policy_key': None,
        'replay_target_ids': [],
        'candidate_evidence_score': 0,
        'control_evidence_score': 0,
        'evidence_counts': {},
        'source_retention_ids': [],
        'source_outcome_ids': [],
        'source_policy_ids': [],
        'source_interaction_history_ids': [],
        'tested_commits': [],
        'tested_tests': [],
        'checkpoint_boundary_state': 'clean',
        'checkpoint_boundary_notes': [],
        'candidate_not_causal_wording': reason,
        'no_overclaiming_proof': reason,
        'label_leaks': [],
    }


def _state_counts(selected):
    counts = {'candidate_win': 0, 'control_win': 0, 'waiting': 0, 'boundary': 0, 'no_gain': 0}
    mapping = {
        'preserve_checkpoint_boundary_replay': 'boundary',
        'replay_retained_code_simulation_ordering_beats_control': 'candidate_win',
        'replay_retained_proof_before_hypothesis_ordering_beats_control': 'candidate_win',
        'replay_retained_language_summary_ordering_beats_control': 'candidate_win',
        'replay_retained_history_memory_ordering_beats_control': 'candidate_win',
        'replay_control_or_regression_beats_retained_policy': 'control_win',
        'replay_underpowered_waiting_for_more_evidence': 'waiting',
        'replay_no_measurable_retained_policy_gain': 'no_gain',
        'summarize_noop': 'no_gain',
    }
    key = mapping.get(selected.get('selected_replay_class'))
    if key:
        counts[key] += 1
    return counts


def _retained_key_from_source(source):
    explicit = source.get('retained_policy_key') or source.get('selected_ordering_key') or source.get('interaction_ordering_key') or source.get('ordering_key')
    if explicit:
        return str(explicit)
    selected = f'{source.get("selected_replay_class", "")} {source.get("selected_decision", "")} {source.get("selected_outcome_class", "")}'.lower()
    if 'code' in selected or 'simulation' in selected:
        return 'code_simulation_before_science'
    if 'proof' in selected or 'formal' in selected:
        return 'proof_before_hypothesis'
    if 'language' in selected or 'summary' in selected or 'terminology' in selected:
        return 'language_summary_before_science'
    if 'history' in selected or 'memory' in selected:
        return 'history_policy_memory_before_science'
    return 'unknown_ordering'


def _signal_from_value(selected, retention, source):
    mapping = {
        'replay_retained_code_simulation_ordering_beats_control': 'candidate_win',
        'replay_retained_proof_before_hypothesis_ordering_beats_control': 'candidate_win',
        'replay_retained_language_summary_ordering_beats_control': 'candidate_win',
        'replay_retained_history_memory_ordering_beats_control': 'candidate_win',
        'retain_code_simulation_ordering_when_repeatedly_helpful': 'candidate_win',
        'retain_proof_before_hypothesis_ordering_when_repeatedly_helpful': 'candidate_win',
        'retain_language_summary_ordering_when_repeatedly_helpful': 'candidate_win',
        'retain_history_policy_memory_ordering_when_repeatedly_helpful': 'candidate_win',
        'retained': 'candidate_win',
        'replay_control_or_regression_beats_retained_policy': 'control_win',
        'control_win': 'control_win',
        'regression': 'control_win',
        'retired': 'control_win',
        'weakened': 'control_win',
        'replay_underpowered_waiting_for_more_evidence': 'underpowered',
        'underpowered': 'underpowered',
        'waiting': 'underpowered',
        'replay_no_measurable_retained_policy_gain': 'no_gain',
        'no_gain': 'no_gain',
        'tie': 'no_gain',
        'preserve_checkpoint_boundary_replay': 'boundary',
        'preserve_checkpoint_boundary_memory': 'boundary',
    }
    if selected in mapping:
        return mapping[selected]
    if retention in mapping:
        return mapping[retention]
    gate = str(source.get('evidence_gate') or '').lower()
    if 'control' in gate or 'regression' in gate:
        return 'control_win'
    if 'candidate' in gate or 'retained' in gate or 'benefit' in gate:
        return 'candidate_win'
    if 'waiting' in gate or 'underpowered' in gate:
        return 'underpowered'
    if 'no_gain' in gate or 'tie' in gate:
        return 'no_gain'
    return None


def _candidate_beats_control(row):
    return _numeric(row.get('candidate_evidence_score')) > _numeric(row.get('control_evidence_score')) and _numeric(row.get('candidate_evidence_score')) > 0


def _control_beats_candidate(row):
    return _numeric(row.get('control_evidence_score')) > _numeric(row.get('candidate_evidence_score')) or _count(row, 'control_win') >= 1


def _candidate_ties_control(row):
    candidate = _numeric(row.get('candidate_evidence_score'))
    control = _numeric(row.get('control_evidence_score'))
    return candidate == control and candidate > 0


def _count(row, key):
    return int((row.get('evidence_counts') or {}).get(key, 0) or 0)


def _numeric(value):
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _source_hash(*sources):
    def digest_or_hash(value):
        if not value:
            return None
        if isinstance(value, dict):
            return value.get('ledger_hash') or stable_digest(value)
        return stable_digest(value)
    return stable_digest([digest_or_hash(source) for source in sources])


def _plain_rows(source, keys):
    rows = []
    for key in keys:
        values = source.get(key)
        if isinstance(values, list):
            rows.extend([dict(item) for item in values if isinstance(item, dict)])
    return rows


def _find_row(rows, retained_policy_key):
    for row in rows:
        if retained_policy_key and row.get('retained_policy_key') == retained_policy_key:
            return row
    return None


def _replay_key(selected):
    return stable_digest({
        'retained': selected.get('retained_policy_key'),
        'control': selected.get('control_policy_key'),
        'class': selected.get('selected_replay_class'),
    })


def _unique_strings(values):
    seen = set()
    output = []
    for value in values:
        text = str(value)
        if text not in seen:
            output.append(text)
            seen.add(text)
    return output
