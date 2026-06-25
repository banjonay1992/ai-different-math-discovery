"""Science coordination A/B probe planner.

Turns local science coordination scorecards and outcome evidence into one
bounded comparison plan. This is not a science benchmark and not causal proof:
it only proposes a small candidate-vs-control ordering test for future module
coordination.
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
from .science_coordination_policy_scorecard import SCIENCE_COORDINATION_POLICY_SCORECARD_LEDGER_KIND


SCIENCE_COORDINATION_AB_PROBE_LEDGER_KIND = 'ai_different.science_coordination_ab_probe_ledger'


def empty_science_coordination_ab_probe_ledger() -> dict[str, Any]:
    return {
        'schema_version': 1,
        'ledger_kind': SCIENCE_COORDINATION_AB_PROBE_LEDGER_KIND,
        'processed_message_ids': [],
        'processed_source_hashes': [],
        'planned_probe_keys': [],
        'probe_records': [],
        'probe_rows': [],
        'outgoing_response_ids': [],
        'latest': {},
    }


def load_science_coordination_ab_probe_ledger(path: str | Path | None) -> dict[str, Any]:
    if not path or not Path(path).exists():
        return empty_science_coordination_ab_probe_ledger()
    with Path(path).open('r', encoding='utf-8') as handle:
        ledger = json.load(handle)
    return validate_science_coordination_ab_probe_ledger(ledger)


def write_science_coordination_ab_probe_ledger(path: str | Path, ledger: dict[str, Any]) -> Path:
    validated = validate_science_coordination_ab_probe_ledger(ledger)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(validated, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return output


def validate_science_coordination_ab_probe_ledger(ledger: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(ledger, dict):
        raise ValueError('science coordination A/B probe ledger must be a JSON object')
    if ledger.get('ledger_kind') != SCIENCE_COORDINATION_AB_PROBE_LEDGER_KIND:
        raise ValueError('science coordination A/B probe ledger has wrong ledger_kind')
    for field in ('processed_message_ids', 'processed_source_hashes', 'planned_probe_keys', 'outgoing_response_ids'):
        values = ledger.get(field)
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            raise ValueError(f'{field} must be strings')
    for field in ('probe_records', 'probe_rows'):
        values = ledger.get(field)
        if not isinstance(values, list) or not all(isinstance(item, dict) for item in values):
            raise ValueError(f'{field} must be objects')
    latest = ledger.get('latest', {})
    if not isinstance(latest, dict):
        raise ValueError('science coordination A/B probe latest must be an object')
    validated = {
        'schema_version': int(ledger.get('schema_version', 1) or 1),
        'ledger_kind': SCIENCE_COORDINATION_AB_PROBE_LEDGER_KIND,
        'processed_message_ids': _unique_strings(ledger['processed_message_ids']),
        'processed_source_hashes': _unique_strings(ledger['processed_source_hashes']),
        'planned_probe_keys': _unique_strings(ledger['planned_probe_keys']),
        'probe_records': list(ledger['probe_records']),
        'probe_rows': list(ledger['probe_rows']),
        'outgoing_response_ids': _unique_strings(ledger['outgoing_response_ids']),
        'latest': dict(latest),
    }
    if ledger.get('ledger_hash'):
        validated['ledger_hash'] = str(ledger['ledger_hash'])
    if ledger.get('artifact_path'):
        validated['artifact_path'] = str(ledger['artifact_path'])
    return validated


def read_science_coordination_ab_probe_transcript(path: str | Path | None) -> dict[str, Any]:
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
        raise ValueError('plain science coordination A/B probe input must be a JSON object')
    return value


def build_science_coordination_ab_probe_plan(
    *,
    transcript_messages: list[dict[str, Any]],
    ab_probe_ledger: dict[str, Any],
    scorecard_ledger: dict[str, Any] | None = None,
    policy_outcome_ledger: dict[str, Any] | None = None,
    policy_ledger: dict[str, Any] | None = None,
    history_ledger: dict[str, Any] | None = None,
    cycle_strategy_outcome_ledger: dict[str, Any] | None = None,
    theory_memory_ledger: dict[str, Any] | None = None,
    campaign_cycle_memory_ledger: dict[str, Any] | None = None,
    hypothesis_ledger: dict[str, Any] | None = None,
    experiment_ledger: dict[str, Any] | None = None,
    module_chat_ledger: dict[str, Any] | None = None,
    prior_ab_probe_ledger: dict[str, Any] | None = None,
    runtime_memory_data: dict[str, Any] | None = None,
    runtime_memory_hash_state: dict[str, Any],
    project_owned_boundary: dict[str, Any],
    hf_use_status: dict[str, Any] | None = None,
    artifact_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    ledger = validate_science_coordination_ab_probe_ledger(ab_probe_ledger)
    scorecard = _valid_kind_or_empty(scorecard_ledger or {}, SCIENCE_COORDINATION_POLICY_SCORECARD_LEDGER_KIND, 'scorecard_records')
    policy_outcome = _valid_kind_or_empty(policy_outcome_ledger or {}, SCIENCE_COORDINATION_POLICY_OUTCOME_LEDGER_KIND, 'outcome_records')
    policy = _valid_kind_or_empty(policy_ledger or {}, SCIENCE_COORDINATION_POLICY_LEDGER_KIND, 'policy_records')
    history = _valid_kind_or_empty(history_ledger or {}, SCIENCE_COORDINATION_HISTORY_LEDGER_KIND, 'event_records')
    cycle_outcome = _valid_kind_or_empty(cycle_strategy_outcome_ledger or {}, SCIENCE_CAMPAIGN_CYCLE_STRATEGY_OUTCOME_LEDGER_KIND, 'outcome_records')
    theory_memory = _valid_plain_or_empty(theory_memory_ledger or {})
    campaign_cycle_memory = _valid_plain_or_empty(campaign_cycle_memory_ledger or {})
    hypothesis = _valid_plain_or_empty(hypothesis_ledger or {})
    experiment = _valid_plain_or_empty(experiment_ledger or {})
    module_chat = _valid_plain_or_empty(module_chat_ledger or {})
    prior_probe = _valid_prior_probe_or_empty(prior_ab_probe_ledger or {})
    runtime_memory = dict(runtime_memory_data or {})
    hf_status = dict(hf_use_status or {'hf_validation_used': False})
    source_hash = _source_hash(
        scorecard,
        policy_outcome,
        policy,
        history,
        cycle_outcome,
        theory_memory,
        campaign_cycle_memory,
        hypothesis,
        experiment,
        module_chat,
        prior_probe,
        runtime_memory,
        hf_status,
    )
    source_is_new = source_hash not in set(ledger['processed_source_hashes'])
    processed = set(ledger['processed_message_ids'])
    new_messages = [message for message in transcript_messages if module_chat_message_id(message) not in processed]
    skipped_messages = [message for message in transcript_messages if module_chat_message_id(message) in processed]
    rows = _extract_probe_rows(
        ledger['probe_rows'],
        scorecard,
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
        selected = _noop_probe('no new science coordination A/B probe evidence or source ledger state')
    else:
        selected = _select_probe(rows=rows, project_owned_boundary=project_owned_boundary)
    probe_id = 'science_coordination_ab_probe_' + stable_digest({
        'source_hash': source_hash,
        'new_message_ids': [module_chat_message_id(message) for message in new_messages],
        'decision': selected['selected_probe_decision'],
        'policy_class': selected.get('policy_class'),
    })[:16]
    selected['science_coordination_ab_probe_id'] = probe_id
    message = export_science_coordination_ab_probe_message(
        selected,
        rows=rows,
        runtime_memory_hash_state=runtime_memory_hash_state,
        project_owned_boundary=project_owned_boundary,
        hf_use_status=hf_status,
        source_hashes={
            'science_coordination_ab_probe_source_hash': source_hash,
            'scorecard_ledger_hash': scorecard.get('ledger_hash'),
            'policy_outcome_ledger_hash': policy_outcome.get('ledger_hash'),
            'policy_ledger_hash': policy.get('ledger_hash'),
            'history_ledger_hash': history.get('ledger_hash'),
            'cycle_strategy_outcome_ledger_hash': cycle_outcome.get('ledger_hash'),
            'theory_memory_ledger_hash': theory_memory.get('ledger_hash'),
            'campaign_cycle_memory_ledger_hash': campaign_cycle_memory.get('ledger_hash'),
            'hypothesis_ledger_hash': hypothesis.get('ledger_hash'),
            'experiment_ledger_hash': experiment.get('ledger_hash'),
            'module_chat_ledger_hash': module_chat.get('ledger_hash'),
            'prior_ab_probe_ledger_hash': prior_probe.get('ledger_hash'),
        },
    )
    new_ids = [module_chat_message_id(message) for message in new_messages]
    outgoing_id = module_chat_message_id(message) if message else None
    probe_key = _probe_key(selected)
    if selected['selected_probe_decision'] != 'summarize_noop':
        ledger['planned_probe_keys'] = _unique_strings(list(ledger['planned_probe_keys']) + [probe_key])
    if new_messages:
        ledger['processed_message_ids'] = _unique_strings(list(ledger['processed_message_ids']) + new_ids)
    if source_is_new:
        ledger['processed_source_hashes'] = _unique_strings(list(ledger['processed_source_hashes']) + [source_hash])
    if outgoing_id:
        ledger['outgoing_response_ids'] = _unique_strings(list(ledger['outgoing_response_ids']) + [outgoing_id])
    state_counts = _state_counts(selected)
    latest = {
        'science_coordination_ab_probe_id': probe_id,
        'new_message_count': len(new_messages),
        'skipped_message_count': len(skipped_messages),
        'source_is_new': source_is_new,
        'state_counts': state_counts,
        'selected_probe_decision': selected['selected_probe_decision'],
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
        'science_coordination_ab_probe_id': probe_id,
        'probe_hash': stable_digest({'probe_id': probe_id, 'selected': selected}),
        'processed_message_ids': new_ids,
        'source_hash': source_hash,
        'source_scorecard_ids': selected.get('source_scorecard_ids'),
        'source_outcome_ids': selected.get('source_outcome_ids'),
        'policy_class': selected.get('policy_class'),
        'candidate_sequence': selected.get('candidate_sequence'),
        'control_sequence': selected.get('control_sequence'),
        'target_campaign_ids': selected.get('target_campaign_ids'),
        'target_hypothesis_ids': selected.get('target_hypothesis_ids'),
        'target_simulation_ids': selected.get('target_simulation_ids'),
        'success_metric': selected.get('success_metric'),
        'stop_condition': selected.get('stop_condition'),
        'sample_evidence_requirements': selected.get('sample_evidence_requirements'),
        'checkpoint_boundary_state': selected.get('checkpoint_boundary_state'),
        'selected_probe_decision': selected['selected_probe_decision'],
        'no_overclaiming_proof': selected.get('no_overclaiming_proof'),
        'hf_use_status': hf_status,
        'outgoing_response_id': outgoing_id,
    }
    ledger['probe_rows'] = rows
    if new_messages or source_is_new or message is not None:
        ledger['probe_records'] = list(ledger['probe_records']) + [record]
    ledger['latest'] = latest
    if artifact_path is not None:
        ledger['artifact_path'] = str(Path(artifact_path))
    ledger['ledger_hash'] = stable_digest({key: value for key, value in ledger.items() if key != 'ledger_hash'})
    return ledger, message


def export_science_coordination_ab_probe_message(
    selected: dict[str, Any],
    *,
    rows: list[dict[str, Any]],
    runtime_memory_hash_state: dict[str, Any],
    project_owned_boundary: dict[str, Any],
    hf_use_status: dict[str, Any],
    source_hashes: dict[str, Any],
    topic: str = 'ai_different.science_coordination_ab_probe',
) -> dict[str, Any] | None:
    if selected['selected_probe_decision'] == 'summarize_noop':
        return None
    recipient = validate_participant(selected.get('selected_recipient') or 'orchestrator')
    row = _find_row(rows, selected.get('policy_class')) or {}
    leak_terms = label_leak_terms({'selected': selected, 'row': row})
    body = {
        'module': 'AI Different',
        'response_kind': 'science_coordination_ab_probe',
        'science_coordination_ab_probe_id': selected.get('science_coordination_ab_probe_id'),
        'selected_probe_decision': selected['selected_probe_decision'],
        'selected_action': selected['selected_action'],
        'selected_recipient': recipient,
        'policy_class': selected.get('policy_class'),
        'candidate_sequence': selected.get('candidate_sequence') or [],
        'control_sequence': selected.get('control_sequence') or [],
        'source_scorecard_ids': selected.get('source_scorecard_ids') or [],
        'source_outcome_ids': selected.get('source_outcome_ids') or [],
        'target_campaign_ids': selected.get('target_campaign_ids') or [],
        'target_hypothesis_ids': selected.get('target_hypothesis_ids') or [],
        'target_simulation_ids': selected.get('target_simulation_ids') or [],
        'tested_commits': selected.get('tested_commits') or [],
        'tested_tests': selected.get('tested_tests') or [],
        'success_metric': selected.get('success_metric'),
        'stop_condition': selected.get('stop_condition'),
        'sample_evidence_requirements': selected.get('sample_evidence_requirements') or {},
        'candidate_not_causal': True,
        'candidate_not_causal_wording': 'Candidate-not-causal A/B probe plan: compare orderings in a bounded future cycle, not proof that one ordering causes the outcome.',
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
            'science_coordination_ab_probe_id': body['science_coordination_ab_probe_id'],
            'selected_probe_decision': body['selected_probe_decision'],
            'policy_class': body['policy_class'],
            'candidate_not_causal': True,
            'hf_validation_used': body['hf_validation_used'],
            'label_clean': body['label_clean'],
            'no_sibling_imports': True,
        },
        tags=['ai_different', 'science_coordination_ab_probe', body['selected_probe_decision'], 'label_clean' if body['label_clean'] else 'label_review_needed'],
    )


def write_science_coordination_ab_probe_outbox_jsonl(path: str | Path, message: dict[str, Any] | None) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text((json.dumps(message, sort_keys=True) + '\n') if message is not None else '', encoding='utf-8')
    return output


def _valid_kind_or_empty(ledger, kind, record_field):
    if not ledger:
        return {'ledger_kind': kind, record_field: [], 'ledger_hash': None}
    if ledger.get('ledger_kind') != kind:
        raise ValueError(f'plain science coordination A/B probe source has wrong ledger_kind: {ledger.get("ledger_kind")}')
    return dict(ledger)


def _valid_plain_or_empty(ledger):
    if not ledger:
        return {}
    if not isinstance(ledger, dict):
        raise ValueError('plain science coordination A/B probe source must be a JSON object')
    return dict(ledger)


def _valid_prior_probe_or_empty(ledger):
    if not ledger:
        return {}
    if ledger.get('ledger_kind') != SCIENCE_COORDINATION_AB_PROBE_LEDGER_KIND:
        raise ValueError('prior science coordination A/B probe ledger has wrong ledger_kind')
    return validate_science_coordination_ab_probe_ledger(ledger)


def _extract_probe_rows(existing, scorecard, policy_outcome, policy, history, cycle_outcome, theory_memory, campaign_cycle_memory, hypothesis, experiment, module_chat, messages):
    rows: dict[str, dict[str, Any]] = {}
    for row in list(existing or []):
        _upsert_row(rows, row)
    for record in list(scorecard.get('scorecard_records') or []):
        _upsert_row(rows, _row_from_scorecard(record))
    for row in list(scorecard.get('scorecard_rows') or []):
        _upsert_row(rows, _row_from_scorecard(row))
    for record in list(policy_outcome.get('outcome_records') or []):
        _upsert_row(rows, _row_from_generic(record, 'policy_outcome'))
    for record in list(policy.get('policy_records') or []):
        _upsert_row(rows, _row_from_generic(record, 'policy'))
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
        'source_scorecard_ids': list(source.get('source_scorecard_ids') or ([source.get('science_coordination_policy_scorecard_id')] if source.get('science_coordination_policy_scorecard_id') else [])),
        'source_outcome_ids': list(source.get('source_outcome_ids') or ([source.get('science_coordination_policy_outcome_id')] if source.get('science_coordination_policy_outcome_id') else [])),
        'decisions': list(source.get('decisions') or ([source.get('selected_retention_decision')] if source.get('selected_retention_decision') else [])),
        'target_campaign_ids': list(source.get('target_campaign_ids') or source.get('campaign_evidence_ids') or ([source.get('campaign_id')] if source.get('campaign_id') else [])),
        'target_hypothesis_ids': list(source.get('target_hypothesis_ids') or source.get('hypothesis_evidence_ids') or ([source.get('hypothesis_id')] if source.get('hypothesis_id') else [])),
        'target_simulation_ids': list(source.get('target_simulation_ids') or ([source.get('simulation_id')] if source.get('simulation_id') else [])),
        'tested_commits': list(source.get('tested_commits') or source.get('source_commits') or []),
        'tested_tests': list(source.get('tested_tests') or source.get('source_tests') or []),
        'checkpoint_boundary_state': source.get('checkpoint_boundary_state') or 'clean',
        'checkpoint_boundary_notes': list(source.get('checkpoint_boundary_notes') or []),
        'label_leaks': list(source.get('label_leaks') or []),
        'lineage': list(source.get('lineage') or []),
    }


def _row_from_scorecard(record):
    row = _base_row(record)
    row['policy_class'] = record.get('policy_class') or row['policy_class']
    row['lineage'] = _unique_strings(row['lineage'] + ['scorecard'])
    return row


def _row_from_generic(record, lineage):
    source = dict(record)
    value = source.get('selected_retention_decision') or source.get('selected_outcome') or source.get('selected_policy') or source.get('status')
    mapped = _decision_from_value(value, source)
    if mapped:
        source['decisions'] = list(source.get('decisions') or []) + [mapped]
    source['lineage'] = [lineage]
    return _base_row(source)


def _row_from_message(message):
    body = dict(message.get('body') or {})
    evidence = dict(message.get('evidence') or {})
    sender = message.get('sender')
    value = body.get('selected_retention_decision') or evidence.get('selected_retention_decision') or body.get('selected_outcome') or evidence.get('selected_outcome') or body.get('selected_policy') or evidence.get('selected_policy') or body.get('status')
    mapped = _decision_from_value(value, {'sender': sender, **body, **evidence})
    source = {
        'policy_class': body.get('policy_class') or evidence.get('policy_class') or body.get('selected_policy') or evidence.get('selected_policy'),
        'science_coordination_policy_scorecard_id': body.get('science_coordination_policy_scorecard_id') or evidence.get('science_coordination_policy_scorecard_id'),
        'science_coordination_policy_outcome_id': body.get('science_coordination_policy_outcome_id') or evidence.get('science_coordination_policy_outcome_id'),
        'selected_retention_decision': mapped or value,
        'campaign_id': body.get('campaign_id') or evidence.get('campaign_id'),
        'hypothesis_id': body.get('hypothesis_id') or evidence.get('hypothesis_id'),
        'simulation_id': body.get('simulation_id') or evidence.get('simulation_id'),
        'source_commits': body.get('tested_commits') or body.get('source_commits') or evidence.get('source_commits') or [],
        'source_tests': body.get('tested_tests') or body.get('source_tests') or evidence.get('source_tests') or [],
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
        'source_scorecard_ids': [],
        'source_outcome_ids': [],
        'decisions': [],
        'target_campaign_ids': [],
        'target_hypothesis_ids': [],
        'target_simulation_ids': [],
        'tested_commits': [],
        'tested_tests': [],
        'checkpoint_boundary_state': 'clean',
        'checkpoint_boundary_notes': [],
        'label_leaks': [],
        'lineage': [],
    })
    current['policy_class'] = current.get('policy_class') or incoming.get('policy_class')
    for field in ('source_scorecard_ids', 'source_outcome_ids', 'target_campaign_ids', 'target_hypothesis_ids', 'target_simulation_ids', 'tested_commits', 'tested_tests', 'checkpoint_boundary_notes', 'label_leaks', 'lineage'):
        current[field] = _unique_strings(list(current.get(field) or []) + list(incoming.get(field) or []))
    current['decisions'] = list(current.get('decisions') or []) + list(incoming.get('decisions') or [])
    current['checkpoint_boundary_state'] = 'repair' if current.get('checkpoint_boundary_state') == 'repair' or incoming.get('checkpoint_boundary_state') == 'repair' else 'clean'


def _finalize_row(row):
    if row.get('label_leaks'):
        row['checkpoint_boundary_state'] = 'repair'
    row['decision_counts'] = _count_values(row.get('decisions') or [])
    row['probe_row_hash'] = stable_digest({
        'policy_class': row.get('policy_class'),
        'decisions': row.get('decision_counts'),
        'boundary': row.get('checkpoint_boundary_state'),
    })
    return row


def _select_probe(*, rows, project_owned_boundary):
    for row in rows:
        if project_owned_boundary.get('third_party_checkpoint_used') or row.get('checkpoint_boundary_state') == 'repair':
            boundary_row = dict(row)
            boundary_row['checkpoint_boundary_state'] = 'repair'
            boundary_row['checkpoint_boundary_notes'] = _unique_strings(list(boundary_row.get('checkpoint_boundary_notes') or []) + ['checkpoint boundary preserved'])
            return _probe(boundary_row, 'preserve_checkpoint_boundary')
    for row in rows:
        if _decision_count(row, 'retain_policy_after_repeated_science_campaign_improvement') or _decision_count(row, 'schedule_science_policy_ab_probe'):
            policy_class = row.get('policy_class') or ''
            if 'code' in policy_class or 'simulation' in policy_class:
                return _probe(row, 'schedule_code_simulation_vs_science_only_probe')
    for row in rows:
        if _decision_count(row, 'retain_policy_after_repeated_science_campaign_improvement') or _decision_count(row, 'schedule_science_policy_ab_probe'):
            policy_class = row.get('policy_class') or ''
            if 'funfun' in policy_class or 'proof' in policy_class or 'formal' in policy_class:
                return _probe(row, 'schedule_proof_before_hypothesis_probe')
    for row in rows:
        if _decision_count(row, 'retain_policy_after_repeated_science_campaign_improvement') or _decision_count(row, 'schedule_science_policy_ab_probe'):
            policy_class = row.get('policy_class') or ''
            if 'language' in policy_class or 'terminology' in policy_class or 'summary' in policy_class:
                return _probe(row, 'schedule_language_summary_before_campaign_probe')
    for row in rows:
        if _decision_count(row, 'retain_policy_after_repeated_science_campaign_improvement') or _decision_count(row, 'schedule_science_policy_ab_probe'):
            return _probe(row, 'schedule_history_policy_memory_probe')
    for row in rows:
        if _decision_count(row, 'keep_policy_waiting_for_more_evidence') or _decision_count(row, 'weaken_policy_after_mixed_evidence'):
            return _probe(row, 'keep_probe_waiting_for_more_scorecard_evidence')
    for row in rows:
        if _decision_count(row, 'retire_policy_after_repeated_noop_or_no_gain'):
            return _probe(row, 'retire_probe_for_repeated_no_gain_policy')
    for row in rows:
        if _decision_count(row, 'record_no_measurable_scorecard_gain'):
            return _probe(row, 'record_no_measurable_probe_gain')
    return _noop_probe('no science coordination A/B probe selected')


def _probe(row, decision):
    candidate, control = _candidate_and_control(decision)
    return {
        'selected_probe_decision': decision,
        'selected_action': decision,
        'selected_recipient': 'orchestrator',
        'policy_class': row.get('policy_class'),
        'candidate_sequence': candidate,
        'control_sequence': control,
        'source_scorecard_ids': list(row.get('source_scorecard_ids') or []),
        'source_outcome_ids': list(row.get('source_outcome_ids') or []),
        'target_campaign_ids': list(row.get('target_campaign_ids') or []),
        'target_hypothesis_ids': list(row.get('target_hypothesis_ids') or []),
        'target_simulation_ids': list(row.get('target_simulation_ids') or []),
        'tested_commits': list(row.get('tested_commits') or []),
        'tested_tests': list(row.get('tested_tests') or []),
        'success_metric': _success_metric(decision),
        'stop_condition': 'stop after one matched candidate/control cycle or on boundary/label/checkpoint violation',
        'sample_evidence_requirements': {
            'candidate_cycles': 1,
            'control_cycles': 1,
            'requires_label_clean': True,
            'requires_runtime_memory_preserved': True,
        },
        'checkpoint_boundary_state': row.get('checkpoint_boundary_state') or 'clean',
        'checkpoint_boundary_notes': list(row.get('checkpoint_boundary_notes') or []),
        'no_overclaiming_proof': 'candidate-not-causal science coordination A/B probe only; no causal proof, no benchmark claim, and no project-owned checkpoint claim without local verification',
        'label_leaks': list(row.get('label_leaks') or []),
    }


def _noop_probe(reason):
    return {
        'selected_probe_decision': 'summarize_noop',
        'selected_action': 'summarize_noop',
        'selected_recipient': None,
        'policy_class': None,
        'candidate_sequence': [],
        'control_sequence': [],
        'source_scorecard_ids': [],
        'source_outcome_ids': [],
        'target_campaign_ids': [],
        'target_hypothesis_ids': [],
        'target_simulation_ids': [],
        'tested_commits': [],
        'tested_tests': [],
        'success_metric': None,
        'stop_condition': reason,
        'sample_evidence_requirements': {},
        'checkpoint_boundary_state': 'clean',
        'checkpoint_boundary_notes': [],
        'no_overclaiming_proof': reason,
        'label_leaks': [],
    }


def _candidate_and_control(decision):
    mapping = {
        'preserve_checkpoint_boundary': (['preserve_checkpoint_boundary', 'defer_probe'], ['no_probe']),
        'schedule_code_simulation_vs_science_only_probe': (['code_simulation', 'science_campaign'], ['science_campaign_only']),
        'schedule_proof_before_hypothesis_probe': (['formal_proof', 'hypothesis_campaign'], ['hypothesis_campaign_only']),
        'schedule_language_summary_before_campaign_probe': (['language_summary', 'science_campaign'], ['science_campaign_no_summary']),
        'schedule_history_policy_memory_probe': (['history_policy_memory', 'science_campaign'], ['science_campaign_without_history_policy_memory']),
        'keep_probe_waiting_for_more_scorecard_evidence': (['wait_for_more_scorecard_evidence'], ['no_probe']),
        'retire_probe_for_repeated_no_gain_policy': (['retire_policy_probe'], ['no_probe']),
        'record_no_measurable_probe_gain': (['record_no_measurable_probe_gain'], ['no_probe']),
    }
    return mapping.get(decision, ([], []))


def _success_metric(decision):
    mapping = {
        'preserve_checkpoint_boundary': 'boundary/checkpoint claim remains clean before any probe is scheduled',
        'schedule_code_simulation_vs_science_only_probe': 'candidate yields stronger label-clean campaign outcome than science-only control',
        'schedule_proof_before_hypothesis_probe': 'candidate yields clearer accepted/refuted hypothesis state than hypothesis-only control',
        'schedule_language_summary_before_campaign_probe': 'candidate reduces protocol ambiguity versus no-summary control',
        'schedule_history_policy_memory_probe': 'candidate reduces repeated no-op policy loops versus no-history control',
        'keep_probe_waiting_for_more_scorecard_evidence': 'enough additional scorecard rows arrive to schedule or retire a probe',
        'retire_probe_for_repeated_no_gain_policy': 'retirement prevents another no-gain cycle',
        'record_no_measurable_probe_gain': 'no safe measurable probe gain recorded',
    }
    return mapping.get(decision)


def _decision_from_value(value, source):
    if value in {
        'preserve_checkpoint_boundary',
        'retain_policy_after_repeated_science_campaign_improvement',
        'weaken_policy_after_mixed_evidence',
        'retire_policy_after_repeated_noop_or_no_gain',
        'keep_policy_waiting_for_more_evidence',
        'schedule_science_policy_ab_probe',
        'record_no_measurable_scorecard_gain',
    }:
        return value
    if value in {'code_simulation_policy_improved_science_campaign', 'funfun_formalization_policy_improved_science_campaign', 'language_terminology_policy_improved_science_campaign'}:
        return 'schedule_science_policy_ab_probe'
    if value in {'repeated_noop_policy_retired', 'no_measurable_policy_gain'}:
        return 'record_no_measurable_scorecard_gain'
    if value in {'policy_waiting_for_sibling_evidence', 'waiting'}:
        return 'keep_policy_waiting_for_more_evidence'
    sender = str(source.get('sender') or '')
    topic = str(source.get('topic') or '')
    if 'scorecard' in topic and sender in {'code_module', 'funfun', 'language_model_2'}:
        return 'schedule_science_policy_ab_probe'
    return None


def _state_counts(selected):
    counts = {
        'scheduled': 0,
        'waiting': 0,
        'retired': 0,
        'boundary': 0,
        'no_gain': 0,
    }
    mapping = {
        'preserve_checkpoint_boundary': 'boundary',
        'schedule_code_simulation_vs_science_only_probe': 'scheduled',
        'schedule_proof_before_hypothesis_probe': 'scheduled',
        'schedule_language_summary_before_campaign_probe': 'scheduled',
        'schedule_history_policy_memory_probe': 'scheduled',
        'keep_probe_waiting_for_more_scorecard_evidence': 'waiting',
        'retire_probe_for_repeated_no_gain_policy': 'retired',
        'record_no_measurable_probe_gain': 'no_gain',
        'summarize_noop': 'no_gain',
    }
    key = mapping.get(selected.get('selected_probe_decision'))
    if key:
        counts[key] += 1
    return counts


def _decision_count(row, decision):
    return int((row.get('decision_counts') or {}).get(decision, 0) or 0)


def _count_values(values):
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


def _probe_key(selected):
    return stable_digest({
        'policy_class': selected.get('policy_class'),
        'decision': selected.get('selected_probe_decision'),
        'candidate': selected.get('candidate_sequence'),
        'control': selected.get('control_sequence'),
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
