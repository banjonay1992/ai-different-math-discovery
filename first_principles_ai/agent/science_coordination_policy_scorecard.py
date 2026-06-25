"""Science coordination policy scorecard.

Aggregates repeated AI Different-local science coordination policy outcomes
into cautious retain/weaken/retire/probe decisions. This is organizational
memory for symbolic science-side sequencing only: candidate-not-causal, not
empirical science proof, not global orchestration, and not a project-owned
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
from .science_campaign_cycle_strategy_outcome import SCIENCE_CAMPAIGN_CYCLE_STRATEGY_OUTCOME_LEDGER_KIND
from .science_coordination_history import SCIENCE_COORDINATION_HISTORY_LEDGER_KIND
from .science_coordination_policy import SCIENCE_COORDINATION_POLICY_LEDGER_KIND
from .science_coordination_policy_outcome import SCIENCE_COORDINATION_POLICY_OUTCOME_LEDGER_KIND


SCIENCE_COORDINATION_POLICY_SCORECARD_LEDGER_KIND = 'ai_different.science_coordination_policy_scorecard_ledger'


def empty_science_coordination_policy_scorecard_ledger() -> dict[str, Any]:
    return {
        'schema_version': 1,
        'ledger_kind': SCIENCE_COORDINATION_POLICY_SCORECARD_LEDGER_KIND,
        'processed_message_ids': [],
        'processed_source_hashes': [],
        'scored_policy_keys': [],
        'scorecard_records': [],
        'scorecard_rows': [],
        'outgoing_response_ids': [],
        'latest': {},
    }


def load_science_coordination_policy_scorecard_ledger(path: str | Path | None) -> dict[str, Any]:
    if not path or not Path(path).exists():
        return empty_science_coordination_policy_scorecard_ledger()
    with Path(path).open('r', encoding='utf-8') as handle:
        ledger = json.load(handle)
    return validate_science_coordination_policy_scorecard_ledger(ledger)


def write_science_coordination_policy_scorecard_ledger(path: str | Path, ledger: dict[str, Any]) -> Path:
    validated = validate_science_coordination_policy_scorecard_ledger(ledger)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(validated, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return output


def validate_science_coordination_policy_scorecard_ledger(ledger: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(ledger, dict):
        raise ValueError('science coordination policy scorecard ledger must be a JSON object')
    if ledger.get('ledger_kind') != SCIENCE_COORDINATION_POLICY_SCORECARD_LEDGER_KIND:
        raise ValueError('science coordination policy scorecard ledger has wrong ledger_kind')
    for field in ('processed_message_ids', 'processed_source_hashes', 'scored_policy_keys', 'outgoing_response_ids'):
        values = ledger.get(field)
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            raise ValueError(f'{field} must be strings')
    for field in ('scorecard_records', 'scorecard_rows'):
        values = ledger.get(field)
        if not isinstance(values, list) or not all(isinstance(item, dict) for item in values):
            raise ValueError(f'{field} must be objects')
    latest = ledger.get('latest', {})
    if not isinstance(latest, dict):
        raise ValueError('science coordination policy scorecard latest must be an object')
    validated = {
        'schema_version': int(ledger.get('schema_version', 1) or 1),
        'ledger_kind': SCIENCE_COORDINATION_POLICY_SCORECARD_LEDGER_KIND,
        'processed_message_ids': _unique_strings(ledger['processed_message_ids']),
        'processed_source_hashes': _unique_strings(ledger['processed_source_hashes']),
        'scored_policy_keys': _unique_strings(ledger['scored_policy_keys']),
        'scorecard_records': list(ledger['scorecard_records']),
        'scorecard_rows': list(ledger['scorecard_rows']),
        'outgoing_response_ids': _unique_strings(ledger['outgoing_response_ids']),
        'latest': dict(latest),
    }
    if ledger.get('ledger_hash'):
        validated['ledger_hash'] = str(ledger['ledger_hash'])
    if ledger.get('artifact_path'):
        validated['artifact_path'] = str(ledger['artifact_path'])
    return validated


def read_science_coordination_policy_scorecard_transcript(path: str | Path | None) -> dict[str, Any]:
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
        raise ValueError('plain science coordination policy scorecard input must be a JSON object')
    return value


def build_science_coordination_policy_scorecard(
    *,
    transcript_messages: list[dict[str, Any]],
    scorecard_ledger: dict[str, Any],
    policy_outcome_ledger: dict[str, Any] | None = None,
    policy_ledger: dict[str, Any] | None = None,
    history_ledger: dict[str, Any] | None = None,
    cycle_strategy_outcome_ledger: dict[str, Any] | None = None,
    theory_memory_ledger: dict[str, Any] | None = None,
    campaign_cycle_memory_ledger: dict[str, Any] | None = None,
    hypothesis_ledger: dict[str, Any] | None = None,
    experiment_ledger: dict[str, Any] | None = None,
    module_chat_ledger: dict[str, Any] | None = None,
    prior_scorecard_ledger: dict[str, Any] | None = None,
    runtime_memory_data: dict[str, Any] | None = None,
    runtime_memory_hash_state: dict[str, Any],
    project_owned_boundary: dict[str, Any],
    artifact_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    ledger = validate_science_coordination_policy_scorecard_ledger(scorecard_ledger)
    policy_outcome = _valid_kind_or_empty(policy_outcome_ledger or {}, SCIENCE_COORDINATION_POLICY_OUTCOME_LEDGER_KIND, 'outcome_records')
    policy = _valid_kind_or_empty(policy_ledger or {}, SCIENCE_COORDINATION_POLICY_LEDGER_KIND, 'policy_records')
    history = _valid_kind_or_empty(history_ledger or {}, SCIENCE_COORDINATION_HISTORY_LEDGER_KIND, 'event_records')
    cycle_outcome = _valid_kind_or_empty(cycle_strategy_outcome_ledger or {}, SCIENCE_CAMPAIGN_CYCLE_STRATEGY_OUTCOME_LEDGER_KIND, 'outcome_records')
    theory_memory = _valid_plain_or_empty(theory_memory_ledger or {})
    campaign_cycle_memory = _valid_plain_or_empty(campaign_cycle_memory_ledger or {})
    hypothesis = _valid_plain_or_empty(hypothesis_ledger or {})
    experiment = _valid_plain_or_empty(experiment_ledger or {})
    module_chat = _valid_plain_or_empty(module_chat_ledger or {})
    prior_scorecard = _valid_prior_scorecard_or_empty(prior_scorecard_ledger or {})
    runtime_memory = dict(runtime_memory_data or {})
    source_hash = _source_hash(
        policy_outcome,
        policy,
        history,
        cycle_outcome,
        theory_memory,
        campaign_cycle_memory,
        hypothesis,
        experiment,
        module_chat,
        prior_scorecard,
        runtime_memory,
    )
    source_is_new = source_hash not in set(ledger['processed_source_hashes'])
    processed = set(ledger['processed_message_ids'])
    new_messages = [message for message in transcript_messages if module_chat_message_id(message) not in processed]
    skipped_messages = [message for message in transcript_messages if module_chat_message_id(message) in processed]
    rows = _extract_scorecard_rows(
        ledger['scorecard_rows'],
        policy_outcome,
        policy,
        history,
        cycle_outcome,
        theory_memory,
        campaign_cycle_memory,
        hypothesis,
        experiment,
        module_chat,
        transcript_messages,
    )
    if not new_messages and not source_is_new:
        selected = _noop_scorecard('no new science coordination policy scorecard evidence or source ledger state')
    else:
        selected = _select_scorecard(rows=rows, project_owned_boundary=project_owned_boundary)
    scorecard_id = 'science_coordination_policy_scorecard_' + stable_digest({
        'source_hash': source_hash,
        'new_message_ids': [module_chat_message_id(message) for message in new_messages],
        'decision': selected['selected_retention_decision'],
        'policy_class': selected.get('policy_class'),
    })[:16]
    selected['science_coordination_policy_scorecard_id'] = scorecard_id
    message = export_science_coordination_policy_scorecard_message(
        selected,
        rows=rows,
        runtime_memory_hash_state=runtime_memory_hash_state,
        project_owned_boundary=project_owned_boundary,
        source_hashes={
            'science_coordination_policy_scorecard_source_hash': source_hash,
            'policy_outcome_ledger_hash': policy_outcome.get('ledger_hash'),
            'policy_ledger_hash': policy.get('ledger_hash'),
            'history_ledger_hash': history.get('ledger_hash'),
            'cycle_strategy_outcome_ledger_hash': cycle_outcome.get('ledger_hash'),
            'theory_memory_ledger_hash': theory_memory.get('ledger_hash'),
            'campaign_cycle_memory_ledger_hash': campaign_cycle_memory.get('ledger_hash'),
            'hypothesis_ledger_hash': hypothesis.get('ledger_hash'),
            'experiment_ledger_hash': experiment.get('ledger_hash'),
            'module_chat_ledger_hash': module_chat.get('ledger_hash'),
            'prior_scorecard_ledger_hash': prior_scorecard.get('ledger_hash'),
        },
    )
    new_ids = [module_chat_message_id(message) for message in new_messages]
    outgoing_id = module_chat_message_id(message) if message else None
    scorecard_key = _scorecard_key(selected)
    if selected['selected_retention_decision'] != 'summarize_noop':
        ledger['scored_policy_keys'] = _unique_strings(list(ledger['scored_policy_keys']) + [scorecard_key])
    if new_messages:
        ledger['processed_message_ids'] = _unique_strings(list(ledger['processed_message_ids']) + new_ids)
    if source_is_new:
        ledger['processed_source_hashes'] = _unique_strings(list(ledger['processed_source_hashes']) + [source_hash])
    if outgoing_id:
        ledger['outgoing_response_ids'] = _unique_strings(list(ledger['outgoing_response_ids']) + [outgoing_id])
    state_counts = _state_counts(selected)
    latest = {
        'science_coordination_policy_scorecard_id': scorecard_id,
        'new_message_count': len(new_messages),
        'skipped_message_count': len(skipped_messages),
        'source_is_new': source_is_new,
        'state_counts': state_counts,
        'selected_retention_decision': selected['selected_retention_decision'],
        'selected_action': selected['selected_action'],
        'recommendation_strength': selected['recommendation_strength'],
        'chosen_recipient': message.get('recipient') if message else None,
        'outbox_count': 1 if message else 0,
        'label_leaks': selected.get('label_leaks', []),
        'checkpoint_boundary_state': selected.get('checkpoint_boundary_state'),
        'runtime_memory_hash_state': dict(runtime_memory_hash_state),
        'runtime_memory_mutated': not bool(runtime_memory_hash_state.get('unchanged', True)),
    }
    record = {
        'science_coordination_policy_scorecard_id': scorecard_id,
        'scorecard_hash': stable_digest({'scorecard_id': scorecard_id, 'selected': selected}),
        'processed_message_ids': new_ids,
        'source_hash': source_hash,
        'source_policy_ids': selected.get('source_policy_ids'),
        'source_outcome_ids': selected.get('source_outcome_ids'),
        'policy_class': selected.get('policy_class'),
        'observed_sequences': selected.get('observed_sequences'),
        'outcome_class_counts': selected.get('outcome_class_counts'),
        'selected_retention_decision': selected['selected_retention_decision'],
        'selected_action': selected['selected_action'],
        'selected_recipient': latest['chosen_recipient'],
        'recommendation_strength': selected.get('recommendation_strength'),
        'checkpoint_boundary_state': selected.get('checkpoint_boundary_state'),
        'no_overclaiming_proof': selected.get('no_overclaiming_proof'),
        'outgoing_response_id': outgoing_id,
    }
    ledger['scorecard_rows'] = rows
    if new_messages or source_is_new or message is not None:
        ledger['scorecard_records'] = list(ledger['scorecard_records']) + [record]
    ledger['latest'] = latest
    if artifact_path is not None:
        ledger['artifact_path'] = str(Path(artifact_path))
    ledger['ledger_hash'] = stable_digest({key: value for key, value in ledger.items() if key != 'ledger_hash'})
    return ledger, message


def export_science_coordination_policy_scorecard_message(
    selected: dict[str, Any],
    *,
    rows: list[dict[str, Any]],
    runtime_memory_hash_state: dict[str, Any],
    project_owned_boundary: dict[str, Any],
    source_hashes: dict[str, Any],
    topic: str = 'ai_different.science_coordination_policy_scorecard',
) -> dict[str, Any] | None:
    if selected['selected_retention_decision'] == 'summarize_noop':
        return None
    recipient = validate_participant(selected.get('selected_recipient') or 'orchestrator')
    row = _find_row(rows, selected.get('policy_class')) or {}
    leak_terms = label_leak_terms({'selected': selected, 'row': row})
    body = {
        'module': 'AI Different',
        'response_kind': 'science_coordination_policy_scorecard',
        'science_coordination_policy_scorecard_id': selected.get('science_coordination_policy_scorecard_id'),
        'source_policy_ids': selected.get('source_policy_ids') or [],
        'source_outcome_ids': selected.get('source_outcome_ids') or [],
        'policy_class': selected.get('policy_class'),
        'observed_sequences': selected.get('observed_sequences') or [],
        'outcome_class_counts': selected.get('outcome_class_counts') or {},
        'tested_commits': selected.get('tested_commits') or [],
        'tested_tests': selected.get('tested_tests') or [],
        'campaign_evidence_ids': selected.get('campaign_evidence_ids') or [],
        'hypothesis_evidence_ids': selected.get('hypothesis_evidence_ids') or [],
        'selected_retention_decision': selected['selected_retention_decision'],
        'selected_action': selected['selected_action'],
        'selected_recipient': recipient,
        'recommendation_strength': selected['recommendation_strength'],
        'policy_should': selected.get('policy_should'),
        'checkpoint_boundary_state': selected.get('checkpoint_boundary_state'),
        'checkpoint_boundary_notes': selected.get('checkpoint_boundary_notes') or [],
        'candidate_not_causal': True,
        'candidate_not_causal_wording': 'Candidate-not-causal scorecard: repeated local outcomes can guide cautious sequencing, not prove policy causality.',
        'no_overclaiming_proof': selected.get('no_overclaiming_proof'),
        'source_ledger_hashes': dict(source_hashes),
        'runtime_memory_hash_state': dict(runtime_memory_hash_state),
        'runtime_memory_mutated': not bool(runtime_memory_hash_state.get('unchanged', True)),
        'project_owned_boundary': dict(project_owned_boundary),
        'third_party_checkpoint_used': bool(project_owned_boundary.get('third_party_checkpoint_used')),
        'project_owned_checkpoint_claimed': bool(project_owned_boundary.get('project_owned_checkpoint_verified')),
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
            'science_coordination_policy_scorecard_id': body['science_coordination_policy_scorecard_id'],
            'policy_class': body['policy_class'],
            'selected_retention_decision': body['selected_retention_decision'],
            'recommendation_strength': body['recommendation_strength'],
            'candidate_not_causal': True,
            'label_clean': body['label_clean'],
            'no_sibling_imports': True,
        },
        tags=['ai_different', 'science_coordination_policy_scorecard', body['selected_retention_decision'], 'label_clean' if body['label_clean'] else 'label_review_needed'],
    )


def write_science_coordination_policy_scorecard_outbox_jsonl(path: str | Path, message: dict[str, Any] | None) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text((json.dumps(message, sort_keys=True) + '\n') if message is not None else '', encoding='utf-8')
    return output


def _valid_kind_or_empty(ledger, kind, record_field):
    if not ledger:
        return {'ledger_kind': kind, record_field: [], 'ledger_hash': None}
    if ledger.get('ledger_kind') != kind:
        raise ValueError(f'plain science coordination policy scorecard source has wrong ledger_kind: {ledger.get("ledger_kind")}')
    return dict(ledger)


def _valid_plain_or_empty(ledger):
    if not ledger:
        return {}
    if not isinstance(ledger, dict):
        raise ValueError('plain science coordination policy scorecard source must be a JSON object')
    return dict(ledger)


def _valid_prior_scorecard_or_empty(ledger):
    if not ledger:
        return {}
    if ledger.get('ledger_kind') != SCIENCE_COORDINATION_POLICY_SCORECARD_LEDGER_KIND:
        raise ValueError('prior science coordination policy scorecard ledger has wrong ledger_kind')
    return validate_science_coordination_policy_scorecard_ledger(ledger)


def _extract_scorecard_rows(existing, policy_outcome, policy, history, cycle_outcome, theory_memory, campaign_cycle_memory, hypothesis, experiment, module_chat, messages):
    rows: dict[str, dict[str, Any]] = {}
    for row in list(existing or []):
        _upsert_row(rows, row)
    for record in list(policy_outcome.get('outcome_records') or []):
        _upsert_row(rows, _row_from_policy_outcome(record))
    for row in list(policy_outcome.get('policy_rows') or []):
        _upsert_row(rows, _row_from_policy_outcome(row))
    for record in list(policy.get('policy_records') or []):
        _upsert_row(rows, _row_from_policy(record))
    for row in list(policy.get('policy_rows') or []):
        _upsert_row(rows, _row_from_policy(row))
    for record in list(history.get('event_records') or []):
        _upsert_row(rows, _row_from_generic(record, 'history'))
    for record in list(cycle_outcome.get('outcome_records') or []):
        _upsert_row(rows, _row_from_generic(record, 'cycle_strategy_outcome'))
    for row in _plain_rows(theory_memory, ('theory_memory_rows', 'theory_records', 'theories')):
        _upsert_row(rows, _row_from_generic(row, 'theory_memory'))
    for row in _plain_rows(campaign_cycle_memory, ('campaign_cycle_memory_rows', 'cycle_memory_rows')):
        _upsert_row(rows, _row_from_generic(row, 'campaign_cycle_memory'))
    for row in _plain_rows(hypothesis, ('hypothesis_rows', 'hypotheses', 'hypothesis_records')):
        _upsert_row(rows, _row_from_generic(row, 'hypothesis'))
    for row in _plain_rows(experiment, ('experiment_rows', 'simulation_rows', 'simulation_results', 'requests')):
        _upsert_row(rows, _row_from_generic(row, 'experiment'))
    for row in _plain_rows(module_chat, ('messages', 'records', 'module_chat_rows')):
        if isinstance(row, dict) and {'sender', 'recipient', 'topic', 'body', 'evidence'}.issubset(row):
            _upsert_row(rows, _row_from_message(row))
    for message in messages:
        _upsert_row(rows, _row_from_message(message))
    return [_finalize_row(row) for row in rows.values()]


def _base_row(source):
    policy_class = source.get('policy_class') or source.get('selected_policy') or 'unknown_policy'
    return {
        'policy_class': policy_class,
        'source_policy_ids': list(source.get('source_policy_ids') or ([source.get('science_coordination_policy_id')] if source.get('science_coordination_policy_id') else [])),
        'source_outcome_ids': list(source.get('source_outcome_ids') or ([source.get('science_coordination_policy_outcome_id')] if source.get('science_coordination_policy_outcome_id') else [])),
        'observed_sequences': list(source.get('observed_sequences') or source.get('interaction_sequence') or []),
        'outcome_classes': list(source.get('outcome_classes') or ([source.get('selected_outcome')] if source.get('selected_outcome') else [])),
        'tested_commits': list(source.get('tested_commits') or source.get('source_commits') or []),
        'tested_tests': list(source.get('tested_tests') or source.get('source_tests') or []),
        'campaign_evidence_ids': list(source.get('campaign_evidence_ids') or ([source.get('campaign_id')] if source.get('campaign_id') else [])),
        'hypothesis_evidence_ids': list(source.get('hypothesis_evidence_ids') or ([source.get('hypothesis_id')] if source.get('hypothesis_id') else [])),
        'checkpoint_boundary_state': source.get('checkpoint_boundary_state') or 'clean',
        'checkpoint_boundary_notes': list(source.get('checkpoint_boundary_notes') or []),
        'label_leaks': list(source.get('label_leaks') or []),
        'lineage': list(source.get('lineage') or []),
    }


def _row_from_policy_outcome(record):
    row = _base_row(record)
    row['policy_class'] = record.get('selected_policy') or record.get('policy_class') or row['policy_class']
    row['lineage'] = _unique_strings(row['lineage'] + ['policy_outcome'])
    return row


def _row_from_policy(record):
    row = _base_row(record)
    row['policy_class'] = record.get('selected_policy') or row['policy_class']
    row['lineage'] = _unique_strings(row['lineage'] + ['policy'])
    return row


def _row_from_generic(record, lineage):
    source = dict(record)
    value = source.get('selected_outcome') or source.get('selected_action') or source.get('status') or source.get('payoff_class')
    mapped = _class_from_value(value, source)
    if mapped:
        source['outcome_classes'] = list(source.get('outcome_classes') or []) + [mapped]
    source['lineage'] = [lineage]
    return _base_row(source)


def _row_from_message(message):
    body = dict(message.get('body') or {})
    evidence = dict(message.get('evidence') or {})
    sender = message.get('sender')
    value = body.get('selected_outcome') or evidence.get('selected_outcome') or body.get('selected_retention_decision') or evidence.get('selected_retention_decision') or body.get('status') or evidence.get('status')
    mapped = _class_from_value(value, {'sender': sender, **body, **evidence})
    source = {
        'policy_class': body.get('selected_policy') or evidence.get('selected_policy') or body.get('policy_class'),
        'science_coordination_policy_id': body.get('science_coordination_policy_id') or evidence.get('science_coordination_policy_id'),
        'science_coordination_policy_outcome_id': body.get('science_coordination_policy_outcome_id') or evidence.get('science_coordination_policy_outcome_id'),
        'selected_outcome': mapped or value,
        'observed_sequences': body.get('observed_sequences') or [],
        'source_commits': body.get('source_commits') or evidence.get('source_commits') or [],
        'source_tests': body.get('source_tests') or evidence.get('source_tests') or [],
        'campaign_id': body.get('campaign_id') or evidence.get('campaign_id'),
        'hypothesis_id': body.get('hypothesis_id') or evidence.get('hypothesis_id'),
        'checkpoint_boundary_state': body.get('checkpoint_boundary_state') or evidence.get('checkpoint_boundary_state') or 'clean',
        'checkpoint_boundary_notes': body.get('checkpoint_boundary_notes') or [],
        'label_leaks': label_leak_terms({'body': body, 'evidence': evidence}),
        'lineage': [f'message:{sender}'],
    }
    if body.get('third_party_checkpoint_used') or evidence.get('third_party_checkpoint_used'):
        source['checkpoint_boundary_state'] = 'repair'
        source['checkpoint_boundary_notes'] = list(source['checkpoint_boundary_notes']) + ['third-party checkpoint boundary preserved']
    return _base_row(source)


def _upsert_row(rows, incoming):
    key = incoming.get('policy_class') or stable_digest(incoming)[:12]
    current = rows.setdefault(str(key), {
        'policy_class': incoming.get('policy_class'),
        'source_policy_ids': [],
        'source_outcome_ids': [],
        'observed_sequences': [],
        'outcome_classes': [],
        'tested_commits': [],
        'tested_tests': [],
        'campaign_evidence_ids': [],
        'hypothesis_evidence_ids': [],
        'checkpoint_boundary_state': 'clean',
        'checkpoint_boundary_notes': [],
        'label_leaks': [],
        'lineage': [],
    })
    current['policy_class'] = current.get('policy_class') or incoming.get('policy_class')
    for field in ('source_policy_ids', 'source_outcome_ids', 'observed_sequences', 'tested_commits', 'tested_tests', 'campaign_evidence_ids', 'hypothesis_evidence_ids', 'checkpoint_boundary_notes', 'label_leaks', 'lineage'):
        current[field] = _unique_strings(list(current.get(field) or []) + list(incoming.get(field) or []))
    current['outcome_classes'] = list(current.get('outcome_classes') or []) + list(incoming.get('outcome_classes') or [])
    current['checkpoint_boundary_state'] = 'repair' if current.get('checkpoint_boundary_state') == 'repair' or incoming.get('checkpoint_boundary_state') == 'repair' else 'clean'


def _finalize_row(row):
    if row.get('label_leaks'):
        row['checkpoint_boundary_state'] = 'repair'
    row['outcome_class_counts'] = _count_classes(row.get('outcome_classes') or [])
    row['scorecard_row_hash'] = stable_digest({
        'policy_class': row.get('policy_class'),
        'counts': row.get('outcome_class_counts'),
        'boundary': row.get('checkpoint_boundary_state'),
    })
    return row


def _select_scorecard(*, rows, project_owned_boundary):
    for row in rows:
        if project_owned_boundary.get('third_party_checkpoint_used') or row.get('checkpoint_boundary_state') == 'repair':
            boundary_row = dict(row)
            boundary_row['checkpoint_boundary_state'] = 'repair'
            boundary_row['checkpoint_boundary_notes'] = _unique_strings(list(boundary_row.get('checkpoint_boundary_notes') or []) + ['checkpoint boundary preserved'])
            return _scorecard(boundary_row, 'preserve_checkpoint_boundary', 'preserve_checkpoint_boundary', 'retained', 'strong')
    for row in rows:
        counts = row.get('outcome_class_counts') or {}
        if _success_count(counts) >= 2:
            return _scorecard(row, 'retain_policy_after_repeated_science_campaign_improvement', 'retain_science_coordination_policy', 'retained', 'strong')
    for row in rows:
        counts = row.get('outcome_class_counts') or {}
        if _success_count(counts) >= 1 and (_negative_count(counts) >= 1 or counts.get('policy_waiting_for_sibling_evidence', 0) >= 1):
            return _scorecard(row, 'weaken_policy_after_mixed_evidence', 'weaken_science_coordination_policy', 'weakened', 'moderate')
    for row in rows:
        counts = row.get('outcome_class_counts') or {}
        if _negative_count(counts) >= 2 or counts.get('repeated_noop_policy_retired', 0) >= 1:
            return _scorecard(row, 'retire_policy_after_repeated_noop_or_no_gain', 'retire_science_coordination_policy', 'retired', 'moderate')
    for row in rows:
        counts = row.get('outcome_class_counts') or {}
        if counts.get('policy_waiting_for_sibling_evidence', 0) >= 1:
            return _scorecard(row, 'keep_policy_waiting_for_more_evidence', 'keep_policy_waiting_for_more_evidence', 'waiting', 'weak')
    for row in rows:
        counts = row.get('outcome_class_counts') or {}
        if _success_count(counts) == 1:
            return _scorecard(row, 'schedule_science_policy_ab_probe', 'schedule_science_policy_ab_probe', 'probed', 'weak')
    for row in rows:
        counts = row.get('outcome_class_counts') or {}
        if counts.get('no_measurable_policy_gain', 0) >= 1:
            return _scorecard(row, 'record_no_measurable_scorecard_gain', 'record_no_measurable_scorecard_gain', 'weakened', 'weak')
    return _noop_scorecard('no science coordination policy scorecard selected')


def _scorecard(row, decision, action, policy_should, strength):
    return {
        'selected_retention_decision': decision,
        'selected_action': action,
        'selected_recipient': 'orchestrator',
        'recommendation_strength': strength,
        'policy_should': policy_should,
        'policy_class': row.get('policy_class'),
        'source_policy_ids': list(row.get('source_policy_ids') or []),
        'source_outcome_ids': list(row.get('source_outcome_ids') or []),
        'observed_sequences': list(row.get('observed_sequences') or []),
        'outcome_class_counts': dict(row.get('outcome_class_counts') or {}),
        'tested_commits': list(row.get('tested_commits') or []),
        'tested_tests': list(row.get('tested_tests') or []),
        'campaign_evidence_ids': list(row.get('campaign_evidence_ids') or []),
        'hypothesis_evidence_ids': list(row.get('hypothesis_evidence_ids') or []),
        'checkpoint_boundary_state': row.get('checkpoint_boundary_state') or 'clean',
        'checkpoint_boundary_notes': list(row.get('checkpoint_boundary_notes') or []),
        'no_overclaiming_proof': 'candidate-not-causal science coordination policy scorecard only; no causal proof and no project-owned checkpoint claim without local verification',
        'label_leaks': list(row.get('label_leaks') or []),
    }


def _noop_scorecard(reason):
    return {
        'selected_retention_decision': 'summarize_noop',
        'selected_action': 'summarize_noop',
        'selected_recipient': None,
        'recommendation_strength': 'none',
        'policy_should': 'none',
        'policy_class': None,
        'source_policy_ids': [],
        'source_outcome_ids': [],
        'observed_sequences': [],
        'outcome_class_counts': {},
        'tested_commits': [],
        'tested_tests': [],
        'campaign_evidence_ids': [],
        'hypothesis_evidence_ids': [],
        'checkpoint_boundary_state': 'clean',
        'checkpoint_boundary_notes': [],
        'no_overclaiming_proof': reason,
        'label_leaks': [],
    }


def _state_counts(selected):
    counts = {
        'retained': 0,
        'weakened': 0,
        'retired': 0,
        'waiting': 0,
        'boundary': 0,
        'probe': 0,
        'no_gain': 0,
    }
    mapping = {
        'preserve_checkpoint_boundary': 'boundary',
        'retain_policy_after_repeated_science_campaign_improvement': 'retained',
        'weaken_policy_after_mixed_evidence': 'weakened',
        'retire_policy_after_repeated_noop_or_no_gain': 'retired',
        'keep_policy_waiting_for_more_evidence': 'waiting',
        'schedule_science_policy_ab_probe': 'probe',
        'record_no_measurable_scorecard_gain': 'no_gain',
        'summarize_noop': 'no_gain',
    }
    key = mapping.get(selected.get('selected_retention_decision'))
    if key:
        counts[key] += 1
    return counts


def _class_from_value(value, source):
    if value in {
        'code_simulation_policy_improved_science_campaign',
        'funfun_formalization_policy_improved_science_campaign',
        'language_terminology_policy_improved_science_campaign',
        'theory_repair_policy_improved_next_campaign',
        'hypothesis_campaign_cycle_scheduled_or_closed',
    }:
        return value
    if value in {'retain_policy_after_repeated_science_campaign_improvement', 'retained'}:
        return 'code_simulation_policy_improved_science_campaign'
    if value in {'no_measurable_policy_gain', 'record_no_measurable_policy_gain'}:
        return 'no_measurable_policy_gain'
    if value in {'repeated_noop_policy_retired', 'retire_policy_after_repeated_noop_or_no_gain', 'retired'}:
        return 'repeated_noop_policy_retired'
    if value in {'policy_waiting_for_sibling_evidence', 'waiting'}:
        return 'policy_waiting_for_sibling_evidence'
    if value == 'preserve_checkpoint_boundary':
        return 'preserve_checkpoint_boundary'
    sender = str(source.get('sender') or '')
    gate = str(source.get('evidence_gate') or '')
    if sender == 'code_module' or 'code' in gate or 'simulation' in gate:
        return 'code_simulation_policy_improved_science_campaign'
    if sender == 'funfun' or 'proof' in gate or 'formal' in gate:
        return 'funfun_formalization_policy_improved_science_campaign'
    if sender == 'language_model_2' or 'language' in gate or 'protocol' in gate:
        return 'language_terminology_policy_improved_science_campaign'
    return None


def _success_count(counts):
    return sum(
        int(counts.get(key, 0) or 0)
        for key in (
            'code_simulation_policy_improved_science_campaign',
            'funfun_formalization_policy_improved_science_campaign',
            'language_terminology_policy_improved_science_campaign',
            'theory_repair_policy_improved_next_campaign',
            'hypothesis_campaign_cycle_scheduled_or_closed',
        )
    )


def _negative_count(counts):
    return int(counts.get('no_measurable_policy_gain', 0) or 0) + int(counts.get('repeated_noop_policy_retired', 0) or 0)


def _count_classes(values):
    counts: dict[str, int] = {}
    for value in values:
        if not value:
            continue
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


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


def _find_row(rows, policy_class):
    for row in rows:
        if policy_class and row.get('policy_class') == policy_class:
            return row
    return None


def _scorecard_key(selected):
    return stable_digest({
        'policy_class': selected.get('policy_class'),
        'decision': selected.get('selected_retention_decision'),
        'outcomes': selected.get('outcome_class_counts'),
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
