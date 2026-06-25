"""Science coordination policy outcome assessor.

Checks whether AI Different-local science coordination policy recommendations
were followed by useful symbolic science/campaign evidence. This is a
candidate-not-causal coordination assessment layer only: no empirical science
claim, no global orchestration claim, and no project-owned checkpoint claim.
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
from .science_campaign_cycle_strategy import SCIENCE_CAMPAIGN_CYCLE_STRATEGY_LEDGER_KIND
from .science_campaign_cycle_strategy_outcome import SCIENCE_CAMPAIGN_CYCLE_STRATEGY_OUTCOME_LEDGER_KIND
from .science_campaign_strategy_outcome import SCIENCE_CAMPAIGN_STRATEGY_OUTCOME_LEDGER_KIND
from .science_coordination_history import SCIENCE_COORDINATION_HISTORY_LEDGER_KIND
from .science_coordination_policy import SCIENCE_COORDINATION_POLICY_LEDGER_KIND
from .science_theory_frontier_outcome import SCIENCE_THEORY_FRONTIER_OUTCOME_LEDGER_KIND


SCIENCE_COORDINATION_POLICY_OUTCOME_LEDGER_KIND = 'ai_different.science_coordination_policy_outcome_ledger'


def empty_science_coordination_policy_outcome_ledger() -> dict[str, Any]:
    return {
        'schema_version': 1,
        'ledger_kind': SCIENCE_COORDINATION_POLICY_OUTCOME_LEDGER_KIND,
        'processed_message_ids': [],
        'processed_source_hashes': [],
        'assessed_policy_keys': [],
        'outcome_records': [],
        'policy_rows': [],
        'outgoing_response_ids': [],
        'latest': {},
    }


def load_science_coordination_policy_outcome_ledger(path: str | Path | None) -> dict[str, Any]:
    if not path or not Path(path).exists():
        return empty_science_coordination_policy_outcome_ledger()
    with Path(path).open('r', encoding='utf-8') as handle:
        ledger = json.load(handle)
    return validate_science_coordination_policy_outcome_ledger(ledger)


def write_science_coordination_policy_outcome_ledger(path: str | Path, ledger: dict[str, Any]) -> Path:
    validated = validate_science_coordination_policy_outcome_ledger(ledger)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(validated, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return output


def validate_science_coordination_policy_outcome_ledger(ledger: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(ledger, dict):
        raise ValueError('science coordination policy outcome ledger must be a JSON object')
    if ledger.get('ledger_kind') != SCIENCE_COORDINATION_POLICY_OUTCOME_LEDGER_KIND:
        raise ValueError('science coordination policy outcome ledger has wrong ledger_kind')
    for field in ('processed_message_ids', 'processed_source_hashes', 'assessed_policy_keys', 'outgoing_response_ids'):
        values = ledger.get(field)
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            raise ValueError(f'{field} must be strings')
    for field in ('outcome_records', 'policy_rows'):
        values = ledger.get(field)
        if not isinstance(values, list) or not all(isinstance(item, dict) for item in values):
            raise ValueError(f'{field} must be objects')
    latest = ledger.get('latest', {})
    if not isinstance(latest, dict):
        raise ValueError('science coordination policy outcome latest must be an object')
    validated = {
        'schema_version': int(ledger.get('schema_version', 1) or 1),
        'ledger_kind': SCIENCE_COORDINATION_POLICY_OUTCOME_LEDGER_KIND,
        'processed_message_ids': _unique_strings(ledger['processed_message_ids']),
        'processed_source_hashes': _unique_strings(ledger['processed_source_hashes']),
        'assessed_policy_keys': _unique_strings(ledger['assessed_policy_keys']),
        'outcome_records': list(ledger['outcome_records']),
        'policy_rows': list(ledger['policy_rows']),
        'outgoing_response_ids': _unique_strings(ledger['outgoing_response_ids']),
        'latest': dict(latest),
    }
    if ledger.get('ledger_hash'):
        validated['ledger_hash'] = str(ledger['ledger_hash'])
    if ledger.get('artifact_path'):
        validated['artifact_path'] = str(ledger['artifact_path'])
    return validated


def read_science_coordination_policy_outcome_transcript(path: str | Path | None) -> dict[str, Any]:
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
        raise ValueError('plain science coordination policy outcome input must be a JSON object')
    return value


def build_science_coordination_policy_outcome_assessment(
    *,
    transcript_messages: list[dict[str, Any]],
    policy_outcome_ledger: dict[str, Any],
    policy_ledger: dict[str, Any] | None = None,
    history_ledger: dict[str, Any] | None = None,
    cycle_strategy_outcome_ledger: dict[str, Any] | None = None,
    cycle_strategy_ledger: dict[str, Any] | None = None,
    strategy_outcome_ledger: dict[str, Any] | None = None,
    frontier_outcome_ledger: dict[str, Any] | None = None,
    theory_memory_ledger: dict[str, Any] | None = None,
    campaign_cycle_memory_ledger: dict[str, Any] | None = None,
    hypothesis_ledger: dict[str, Any] | None = None,
    campaign_ledger: dict[str, Any] | None = None,
    experiment_ledger: dict[str, Any] | None = None,
    module_chat_ledger: dict[str, Any] | None = None,
    prior_policy_outcome_ledger: dict[str, Any] | None = None,
    runtime_memory_data: dict[str, Any] | None = None,
    runtime_memory_hash_state: dict[str, Any],
    project_owned_boundary: dict[str, Any],
    artifact_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    ledger = validate_science_coordination_policy_outcome_ledger(policy_outcome_ledger)
    policy = _valid_kind_or_empty(policy_ledger or {}, SCIENCE_COORDINATION_POLICY_LEDGER_KIND, 'policy_records')
    history = _valid_kind_or_empty(history_ledger or {}, SCIENCE_COORDINATION_HISTORY_LEDGER_KIND, 'event_records')
    cycle_outcome = _valid_kind_or_empty(cycle_strategy_outcome_ledger or {}, SCIENCE_CAMPAIGN_CYCLE_STRATEGY_OUTCOME_LEDGER_KIND, 'outcome_records')
    cycle_strategy = _valid_kind_or_empty(cycle_strategy_ledger or {}, SCIENCE_CAMPAIGN_CYCLE_STRATEGY_LEDGER_KIND, 'cycle_strategy_records')
    strategy_outcome = _valid_kind_or_empty(strategy_outcome_ledger or {}, SCIENCE_CAMPAIGN_STRATEGY_OUTCOME_LEDGER_KIND, 'outcome_records')
    frontier_outcome = _valid_kind_or_empty(frontier_outcome_ledger or {}, SCIENCE_THEORY_FRONTIER_OUTCOME_LEDGER_KIND, 'outcome_records')
    theory_memory = _valid_plain_or_empty(theory_memory_ledger or {})
    campaign_cycle_memory = _valid_plain_or_empty(campaign_cycle_memory_ledger or {})
    hypothesis = _valid_plain_or_empty(hypothesis_ledger or {})
    campaign = _valid_plain_or_empty(campaign_ledger or {})
    experiment = _valid_plain_or_empty(experiment_ledger or {})
    module_chat = _valid_plain_or_empty(module_chat_ledger or {})
    prior_outcome = _valid_prior_outcome_or_empty(prior_policy_outcome_ledger or {})
    runtime_memory = dict(runtime_memory_data or {})
    source_hash = _source_hash(
        policy,
        history,
        cycle_outcome,
        cycle_strategy,
        strategy_outcome,
        frontier_outcome,
        theory_memory,
        campaign_cycle_memory,
        hypothesis,
        campaign,
        experiment,
        module_chat,
        prior_outcome,
        runtime_memory,
    )
    source_is_new = source_hash not in set(ledger['processed_source_hashes'])
    processed = set(ledger['processed_message_ids'])
    new_messages = [message for message in transcript_messages if module_chat_message_id(message) not in processed]
    skipped_messages = [message for message in transcript_messages if module_chat_message_id(message) in processed]
    rows = _extract_outcome_rows(
        ledger['policy_rows'],
        policy,
        history,
        cycle_outcome,
        cycle_strategy,
        strategy_outcome,
        frontier_outcome,
        theory_memory,
        campaign_cycle_memory,
        hypothesis,
        campaign,
        experiment,
        module_chat,
        transcript_messages,
    )
    if not new_messages and not source_is_new:
        selected = _noop_outcome('no new science coordination policy outcome evidence or source ledger state')
    else:
        selected = _select_outcome(rows=rows, project_owned_boundary=project_owned_boundary)
    outcome_id = 'science_coordination_policy_outcome_' + stable_digest({
        'source_hash': source_hash,
        'new_message_ids': [module_chat_message_id(message) for message in new_messages],
        'outcome': selected['selected_outcome'],
        'policy_id': selected.get('science_coordination_policy_id'),
        'scenario_id': selected.get('scenario_id'),
    })[:16]
    selected['science_coordination_policy_outcome_id'] = outcome_id
    message = export_science_coordination_policy_outcome_message(
        selected,
        rows=rows,
        runtime_memory_hash_state=runtime_memory_hash_state,
        project_owned_boundary=project_owned_boundary,
        source_hashes={
            'science_coordination_policy_outcome_source_hash': source_hash,
            'policy_ledger_hash': policy.get('ledger_hash'),
            'history_ledger_hash': history.get('ledger_hash'),
            'cycle_strategy_outcome_ledger_hash': cycle_outcome.get('ledger_hash'),
            'cycle_strategy_ledger_hash': cycle_strategy.get('ledger_hash'),
            'strategy_outcome_ledger_hash': strategy_outcome.get('ledger_hash'),
            'frontier_outcome_ledger_hash': frontier_outcome.get('ledger_hash'),
            'theory_memory_ledger_hash': theory_memory.get('ledger_hash'),
            'campaign_cycle_memory_ledger_hash': campaign_cycle_memory.get('ledger_hash'),
            'hypothesis_ledger_hash': hypothesis.get('ledger_hash'),
            'campaign_ledger_hash': campaign.get('ledger_hash'),
            'experiment_ledger_hash': experiment.get('ledger_hash'),
            'module_chat_ledger_hash': module_chat.get('ledger_hash'),
            'prior_policy_outcome_ledger_hash': prior_outcome.get('ledger_hash'),
        },
    )
    new_ids = [module_chat_message_id(message) for message in new_messages]
    outgoing_id = module_chat_message_id(message) if message else None
    assessed_key = _outcome_key(selected)
    if selected['selected_outcome'] != 'summarize_noop':
        ledger['assessed_policy_keys'] = _unique_strings(list(ledger['assessed_policy_keys']) + [assessed_key])
    if new_messages:
        ledger['processed_message_ids'] = _unique_strings(list(ledger['processed_message_ids']) + new_ids)
    if source_is_new:
        ledger['processed_source_hashes'] = _unique_strings(list(ledger['processed_source_hashes']) + [source_hash])
    if outgoing_id:
        ledger['outgoing_response_ids'] = _unique_strings(list(ledger['outgoing_response_ids']) + [outgoing_id])
    state_counts = _state_counts(selected)
    latest = {
        'science_coordination_policy_outcome_id': outcome_id,
        'new_message_count': len(new_messages),
        'skipped_message_count': len(skipped_messages),
        'source_is_new': source_is_new,
        'state_counts': state_counts,
        'selected_outcome': selected['selected_outcome'],
        'selected_action': selected['selected_action'],
        'policy_retention_state': selected.get('policy_retention_state'),
        'chosen_recipient': message.get('recipient') if message else None,
        'outbox_count': 1 if message else 0,
        'label_leaks': selected.get('label_leaks', []),
        'checkpoint_boundary_state': selected.get('checkpoint_boundary_state'),
        'runtime_memory_hash_state': dict(runtime_memory_hash_state),
        'runtime_memory_mutated': not bool(runtime_memory_hash_state.get('unchanged', True)),
    }
    record = {
        'science_coordination_policy_outcome_id': outcome_id,
        'outcome_hash': stable_digest({'outcome_id': outcome_id, 'selected': selected}),
        'processed_message_ids': new_ids,
        'source_hash': source_hash,
        'source_policy_ids': selected.get('source_policy_ids'),
        'source_history_ids': selected.get('source_history_ids'),
        'source_campaign_ids': selected.get('source_campaign_ids'),
        'science_coordination_policy_id': selected.get('science_coordination_policy_id'),
        'selected_policy': selected.get('selected_policy'),
        'selected_policy_action': selected.get('selected_policy_action'),
        'selected_policy_recipient': selected.get('selected_policy_recipient'),
        'selected_outcome': selected['selected_outcome'],
        'selected_action': selected['selected_action'],
        'selected_recipient': latest['chosen_recipient'],
        'policy_retention_state': selected.get('policy_retention_state'),
        'observed_science_side_evidence': selected.get('observed_science_side_evidence'),
        'validated_hypothesis_state': selected.get('validated_hypothesis_state'),
        'campaign_cycle_memory_state': selected.get('campaign_cycle_memory_state'),
        'checkpoint_boundary_state': selected.get('checkpoint_boundary_state'),
        'checkpoint_boundary_notes': selected.get('checkpoint_boundary_notes'),
        'no_overclaiming_proof': selected.get('no_overclaiming_proof'),
        'outgoing_response_id': outgoing_id,
    }
    ledger['policy_rows'] = rows
    if new_messages or source_is_new or message is not None:
        ledger['outcome_records'] = list(ledger['outcome_records']) + [record]
    ledger['latest'] = latest
    if artifact_path is not None:
        ledger['artifact_path'] = str(Path(artifact_path))
    ledger['ledger_hash'] = stable_digest({key: value for key, value in ledger.items() if key != 'ledger_hash'})
    return ledger, message


def export_science_coordination_policy_outcome_message(
    selected: dict[str, Any],
    *,
    rows: list[dict[str, Any]],
    runtime_memory_hash_state: dict[str, Any],
    project_owned_boundary: dict[str, Any],
    source_hashes: dict[str, Any],
    topic: str = 'ai_different.science_coordination_policy_outcome',
) -> dict[str, Any] | None:
    if selected['selected_outcome'] == 'summarize_noop':
        return None
    recipient = validate_participant(selected.get('selected_recipient') or 'orchestrator')
    row = _find_row(rows, selected.get('science_coordination_policy_id'), selected.get('scenario_id'), selected.get('campaign_id'), selected.get('hypothesis_id')) or {}
    leak_terms = label_leak_terms({'selected': selected, 'row': row})
    body = {
        'module': 'AI Different',
        'response_kind': 'science_coordination_policy_outcome',
        'science_coordination_policy_outcome_id': selected.get('science_coordination_policy_outcome_id'),
        'science_coordination_policy_id': selected.get('science_coordination_policy_id'),
        'source_policy_ids': selected.get('source_policy_ids') or [],
        'source_history_ids': selected.get('source_history_ids') or [],
        'source_campaign_ids': selected.get('source_campaign_ids') or [],
        'hypothesis_id': selected.get('hypothesis_id'),
        'campaign_id': selected.get('campaign_id'),
        'family_id': selected.get('family_id'),
        'scenario_id': selected.get('scenario_id'),
        'selected_policy': selected.get('selected_policy'),
        'selected_policy_action': selected.get('selected_policy_action'),
        'selected_policy_recipient': selected.get('selected_policy_recipient'),
        'selected_outcome': selected['selected_outcome'],
        'selected_action': selected['selected_action'],
        'selected_recipient': recipient,
        'policy_retention_state': selected.get('policy_retention_state'),
        'observed_science_side_evidence': selected.get('observed_science_side_evidence') or [],
        'validated_hypothesis_state': selected.get('validated_hypothesis_state'),
        'campaign_cycle_memory_state': selected.get('campaign_cycle_memory_state'),
        'checkpoint_boundary_state': selected.get('checkpoint_boundary_state'),
        'checkpoint_boundary_notes': selected.get('checkpoint_boundary_notes') or [],
        'candidate_not_causal': True,
        'candidate_not_causal_wording': 'Candidate-not-causal policy outcome: useful for cautious local sequencing, not proof that the policy caused the science result.',
        'source_commits': selected.get('source_commits') or [],
        'source_tests': selected.get('source_tests') or [],
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
            'science_coordination_policy_outcome_id': body['science_coordination_policy_outcome_id'],
            'science_coordination_policy_id': body['science_coordination_policy_id'],
            'selected_outcome': body['selected_outcome'],
            'policy_retention_state': body['policy_retention_state'],
            'candidate_not_causal': True,
            'label_clean': body['label_clean'],
            'no_sibling_imports': True,
        },
        tags=['ai_different', 'science_coordination_policy_outcome', body['selected_outcome'], 'label_clean' if body['label_clean'] else 'label_review_needed'],
    )


def write_science_coordination_policy_outcome_outbox_jsonl(path: str | Path, message: dict[str, Any] | None) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text((json.dumps(message, sort_keys=True) + '\n') if message is not None else '', encoding='utf-8')
    return output


def _valid_kind_or_empty(ledger, kind, record_field):
    if not ledger:
        return {'ledger_kind': kind, record_field: [], 'ledger_hash': None}
    if ledger.get('ledger_kind') != kind:
        raise ValueError(f'plain science coordination policy outcome source has wrong ledger_kind: {ledger.get("ledger_kind")}')
    return dict(ledger)


def _valid_plain_or_empty(ledger):
    if not ledger:
        return {}
    if not isinstance(ledger, dict):
        raise ValueError('plain science coordination policy outcome source must be a JSON object')
    return dict(ledger)


def _valid_prior_outcome_or_empty(ledger):
    if not ledger:
        return {}
    if ledger.get('ledger_kind') != SCIENCE_COORDINATION_POLICY_OUTCOME_LEDGER_KIND:
        raise ValueError('prior science coordination policy outcome ledger has wrong ledger_kind')
    return validate_science_coordination_policy_outcome_ledger(ledger)


def _extract_outcome_rows(existing, policy, history, cycle_outcome, cycle_strategy, strategy_outcome, frontier_outcome, theory_memory, campaign_cycle_memory, hypothesis, campaign, experiment, module_chat, messages):
    rows: dict[str, dict[str, Any]] = {}
    for row in list(existing or []):
        _upsert_row(rows, row)
    for record in list(policy.get('policy_records') or []):
        _upsert_row(rows, _row_from_policy(record))
    for row in list(policy.get('policy_rows') or []):
        _upsert_row(rows, _row_from_policy(row))
    for record in list(history.get('event_records') or []):
        _upsert_row(rows, _row_from_history(record))
    for row in list(history.get('history_rows') or []):
        _upsert_row(rows, _row_from_history(row))
    for record in list(cycle_outcome.get('outcome_records') or []):
        _upsert_row(rows, _row_from_generic_evidence(record, 'cycle_strategy_outcome'))
    for record in list(cycle_strategy.get('cycle_strategy_records') or []):
        _upsert_row(rows, _row_from_generic_evidence(record, 'cycle_strategy'))
    for record in list(strategy_outcome.get('outcome_records') or []):
        _upsert_row(rows, _row_from_generic_evidence(record, 'strategy_outcome'))
    for record in list(frontier_outcome.get('outcome_records') or []):
        _upsert_row(rows, _row_from_generic_evidence(record, 'frontier_outcome'))
    for row in _plain_rows(theory_memory, ('theory_memory_rows', 'theory_records', 'theories')):
        _upsert_row(rows, _row_from_generic_evidence(row, 'theory_memory'))
    for row in _plain_rows(campaign_cycle_memory, ('campaign_cycle_memory_rows', 'cycle_memory_rows')):
        _upsert_row(rows, _row_from_generic_evidence(row, 'campaign_cycle_memory'))
    for row in _plain_rows(hypothesis, ('hypothesis_rows', 'hypotheses', 'hypothesis_records')):
        _upsert_row(rows, _row_from_generic_evidence(row, 'hypothesis'))
    for row in _plain_rows(campaign, ('campaign_rows', 'campaigns', 'campaign_records')):
        _upsert_row(rows, _row_from_generic_evidence(row, 'campaign'))
    for row in _plain_rows(experiment, ('experiment_rows', 'simulation_results', 'simulation_requests', 'requests')):
        _upsert_row(rows, _row_from_generic_evidence(row, 'experiment'))
    for row in _plain_rows(module_chat, ('messages', 'records', 'module_chat_rows')):
        if isinstance(row, dict) and {'sender', 'recipient', 'topic', 'body', 'evidence'}.issubset(row):
            _upsert_row(rows, _row_from_message(row))
    for message in messages:
        _upsert_row(rows, _row_from_message(message))
    return [_finalize_row(row) for row in rows.values()]


def _base_row(source):
    return {
        'scenario_id': source.get('scenario_id') or source.get('campaign_id') or source.get('hypothesis_id'),
        'hypothesis_id': source.get('hypothesis_id'),
        'campaign_id': source.get('campaign_id'),
        'family_id': source.get('family_id') or source.get('hypothesis_family_id'),
        'science_coordination_policy_id': source.get('science_coordination_policy_id'),
        'source_policy_ids': list(source.get('source_policy_ids') or ([source.get('science_coordination_policy_id')] if source.get('science_coordination_policy_id') else [])),
        'source_history_ids': list(source.get('source_history_ids') or ([source.get('science_coordination_history_id')] if source.get('science_coordination_history_id') else [])),
        'source_campaign_ids': list(source.get('source_campaign_ids') or ([source.get('campaign_cycle_strategy_outcome_id')] if source.get('campaign_cycle_strategy_outcome_id') else [])),
        'selected_policy': source.get('selected_policy'),
        'selected_policy_action': source.get('selected_action') or source.get('selected_policy_action'),
        'selected_policy_recipient': source.get('selected_recipient') or source.get('selected_policy_recipient'),
        'evidence_classes': list(source.get('evidence_classes') or []),
        'observed_science_side_evidence': list(source.get('observed_science_side_evidence') or []),
        'validated_hypothesis_state': source.get('validated_hypothesis_state') or 'unknown',
        'campaign_cycle_memory_state': source.get('campaign_cycle_memory_state') or 'unknown',
        'checkpoint_boundary_state': source.get('checkpoint_boundary_state') or 'clean',
        'checkpoint_boundary_notes': list(source.get('checkpoint_boundary_notes') or []),
        'source_commits': list(source.get('source_commits') or []),
        'source_tests': list(source.get('source_tests') or []),
        'label_leaks': list(source.get('label_leaks') or []),
        'lineage': list(source.get('lineage') or []),
    }


def _row_from_policy(record):
    row = _base_row(record)
    row['lineage'] = _unique_strings(row['lineage'] + ['policy'])
    return row


def _row_from_history(record):
    row = _base_row(record)
    payoff = record.get('payoff_class') or (record.get('payoff_classes') or [None])[0]
    if payoff:
        row['evidence_classes'] = _unique_strings(row['evidence_classes'] + [payoff])
    row['lineage'] = _unique_strings(row['lineage'] + ['history'])
    return row


def _row_from_generic_evidence(record, lineage):
    source = dict(record)
    outcome = source.get('selected_outcome') or source.get('selected_strategy') or source.get('selected_action') or source.get('status')
    evidence_class = _evidence_class_from_value(outcome, source)
    if evidence_class:
        source['evidence_classes'] = list(source.get('evidence_classes') or []) + [evidence_class]
    if source.get('status') in {'received', 'accepted', 'passed', 'resolved', 'recorded', 'closed', 'scheduled'}:
        source['evidence_classes'] = list(source.get('evidence_classes') or []) + [str(source.get('evidence_gate') or source.get('status'))]
    source['observed_science_side_evidence'] = list(source.get('observed_science_side_evidence') or source.get('source_evidence_used') or source.get('observed_sibling_evidence') or [])
    source['validated_hypothesis_state'] = source.get('validated_hypothesis_state') or ('validated' if source.get('promotion_state') == 'promoted' or source.get('status') == 'validated' else 'unknown')
    source['campaign_cycle_memory_state'] = source.get('campaign_cycle_memory_state') or source.get('after_campaign_cycle_memory_state') or ('recorded' if lineage == 'campaign_cycle_memory' else 'unknown')
    source['lineage'] = [lineage]
    return _base_row(source)


def _row_from_message(message):
    body = dict(message.get('body') or {})
    evidence = dict(message.get('evidence') or {})
    sender = message.get('sender')
    source = {
        'scenario_id': body.get('scenario_id') or evidence.get('scenario_id') or body.get('campaign_id') or evidence.get('campaign_id') or body.get('hypothesis_id') or evidence.get('hypothesis_id'),
        'hypothesis_id': body.get('hypothesis_id') or evidence.get('hypothesis_id'),
        'campaign_id': body.get('campaign_id') or evidence.get('campaign_id'),
        'family_id': body.get('family_id') or evidence.get('family_id'),
        'science_coordination_policy_id': body.get('science_coordination_policy_id') or evidence.get('science_coordination_policy_id'),
        'science_coordination_history_id': body.get('science_coordination_history_id') or evidence.get('science_coordination_history_id'),
        'campaign_cycle_strategy_outcome_id': body.get('campaign_cycle_strategy_outcome_id') or evidence.get('campaign_cycle_strategy_outcome_id'),
        'selected_policy': body.get('selected_policy') or evidence.get('selected_policy'),
        'selected_action': body.get('selected_policy_action') or body.get('selected_action') or evidence.get('selected_action'),
        'selected_recipient': body.get('selected_policy_recipient') or body.get('selected_recipient') or evidence.get('selected_recipient'),
        'source_commits': body.get('source_commits') or evidence.get('source_commits') or [],
        'source_tests': body.get('source_tests') or evidence.get('source_tests') or [],
        'checkpoint_boundary_state': body.get('checkpoint_boundary_state') or evidence.get('checkpoint_boundary_state') or 'clean',
        'checkpoint_boundary_notes': body.get('checkpoint_boundary_notes') or [],
        'label_leaks': label_leak_terms({'body': body, 'evidence': evidence}),
        'lineage': [f'message:{sender}'],
    }
    evidence_values = [
        body.get('selected_outcome'),
        evidence.get('selected_outcome'),
        body.get('selected_strategy'),
        evidence.get('selected_strategy'),
        body.get('payoff_class'),
        evidence.get('payoff_class'),
        body.get('evidence_gate'),
        evidence.get('evidence_gate'),
        body.get('status'),
        evidence.get('status'),
    ]
    source['evidence_classes'] = _unique_strings([
        item
        for value in evidence_values
        for item in ([value, _evidence_class_from_value(value, body)] if value else [])
        if item
    ])
    source['observed_science_side_evidence'] = list(body.get('observed_science_side_evidence') or body.get('source_evidence_used') or body.get('observed_sibling_evidence') or [])
    source['validated_hypothesis_state'] = body.get('validated_hypothesis_state') or evidence.get('validated_hypothesis_state') or 'unknown'
    source['campaign_cycle_memory_state'] = body.get('campaign_cycle_memory_state') or body.get('after_campaign_cycle_memory_state') or evidence.get('campaign_cycle_memory_state') or 'unknown'
    if body.get('third_party_checkpoint_used') or evidence.get('third_party_checkpoint_used'):
        source['checkpoint_boundary_state'] = 'repair'
        source['checkpoint_boundary_notes'] = list(source['checkpoint_boundary_notes']) + ['third-party checkpoint boundary preserved']
    return _base_row(source)


def _upsert_row(rows, incoming):
    key = incoming.get('scenario_id') or incoming.get('campaign_id') or incoming.get('hypothesis_id') or incoming.get('science_coordination_policy_id') or stable_digest(incoming)[:12]
    current = rows.setdefault(str(key), {
        'scenario_id': incoming.get('scenario_id'),
        'hypothesis_id': incoming.get('hypothesis_id'),
        'campaign_id': incoming.get('campaign_id'),
        'family_id': incoming.get('family_id'),
        'science_coordination_policy_id': incoming.get('science_coordination_policy_id'),
        'source_policy_ids': [],
        'source_history_ids': [],
        'source_campaign_ids': [],
        'selected_policy': None,
        'selected_policy_action': None,
        'selected_policy_recipient': None,
        'evidence_classes': [],
        'observed_science_side_evidence': [],
        'validated_hypothesis_state': 'unknown',
        'campaign_cycle_memory_state': 'unknown',
        'checkpoint_boundary_state': 'clean',
        'checkpoint_boundary_notes': [],
        'source_commits': [],
        'source_tests': [],
        'label_leaks': [],
        'lineage': [],
    })
    for field in ('scenario_id', 'hypothesis_id', 'campaign_id', 'family_id', 'science_coordination_policy_id', 'selected_policy', 'selected_policy_action', 'selected_policy_recipient'):
        current[field] = current.get(field) or incoming.get(field)
    for field in ('source_policy_ids', 'source_history_ids', 'source_campaign_ids', 'evidence_classes', 'checkpoint_boundary_notes', 'source_commits', 'source_tests', 'label_leaks', 'lineage'):
        current[field] = _unique_strings(list(current.get(field) or []) + list(incoming.get(field) or []))
    current['observed_science_side_evidence'] = list(current.get('observed_science_side_evidence') or []) + list(incoming.get('observed_science_side_evidence') or [])
    current['validated_hypothesis_state'] = _dominant_state(current.get('validated_hypothesis_state'), incoming.get('validated_hypothesis_state'), ['unknown', 'waiting', 'blocked', 'validated', 'retired'])
    current['campaign_cycle_memory_state'] = _dominant_state(current.get('campaign_cycle_memory_state'), incoming.get('campaign_cycle_memory_state'), ['unknown', 'waiting', 'scheduled', 'closed', 'recorded'])
    current['checkpoint_boundary_state'] = 'repair' if current.get('checkpoint_boundary_state') == 'repair' or incoming.get('checkpoint_boundary_state') == 'repair' else 'clean'


def _finalize_row(row):
    if row.get('label_leaks'):
        row['checkpoint_boundary_state'] = 'repair'
    row['policy_outcome_row_hash'] = stable_digest({
        'scenario_id': row.get('scenario_id'),
        'policy': row.get('selected_policy'),
        'evidence': row.get('evidence_classes'),
        'boundary': row.get('checkpoint_boundary_state'),
    })
    return row


def _select_outcome(*, rows, project_owned_boundary):
    for row in rows:
        if project_owned_boundary.get('third_party_checkpoint_used') or row.get('checkpoint_boundary_state') == 'repair':
            boundary_row = dict(row)
            boundary_row['checkpoint_boundary_state'] = 'repair'
            boundary_row['checkpoint_boundary_notes'] = _unique_strings(list(boundary_row.get('checkpoint_boundary_notes') or []) + ['checkpoint boundary preserved'])
            return _outcome(boundary_row, 'preserve_checkpoint_boundary', 'preserve_checkpoint_boundary', 'retained', 'orchestrator')
    for row in rows:
        if row.get('selected_policy') in {'try_code_simulation_before_science_campaign', 'gather_more_sibling_evidence_before_policy_change', 'record_no_measurable_policy_gain'} and _has_evidence(row, {'code_simulation', 'code_simulation_or_primitive_capability_help_received', 'primitive_simulation_help', 'simulation_result', 'code_counterexample'}):
            return _outcome(row, 'code_simulation_policy_improved_science_campaign', 'retain_code_simulation_before_campaign_policy', 'retained', 'code_module')
    for row in rows:
        if row.get('selected_policy') in {'try_funfun_formalization_before_science_campaign', 'gather_more_sibling_evidence_before_policy_change', 'record_no_measurable_policy_gain'} and _has_evidence(row, {'funfun_formalization', 'funfun_formal_or_proof_help_received', 'formal_proof_certificate', 'proof_certificate'}):
            return _outcome(row, 'funfun_formalization_policy_improved_science_campaign', 'retain_funfun_formalization_before_campaign_policy', 'retained', 'funfun')
    for row in rows:
        if row.get('selected_policy') in {'try_language_terminology_before_science_campaign', 'gather_more_sibling_evidence_before_policy_change', 'record_no_measurable_policy_gain'} and _has_evidence(row, {'language_terminology', 'language_terminology_or_protocol_help_received', 'protocol_clarification', 'terminology_clarification'}):
            return _outcome(row, 'language_terminology_policy_improved_science_campaign', 'retain_language_terminology_before_campaign_policy', 'retained', 'language_model_2')
    for row in rows:
        if row.get('selected_policy') == 'try_theory_repair_before_next_campaign' and _has_evidence(row, {'theory_repair_loop_helped', 'theory_repair_cycle_reopened', 'repair_reopened', 'repair'}):
            return _outcome(row, 'theory_repair_policy_improved_next_campaign', 'retain_theory_repair_before_campaign_policy', 'retained', 'orchestrator')
    for row in rows:
        if _has_evidence(row, {'next_frontier_hypothesis_campaign_scheduled', 'stable_science_campaign_cycle_closed', 'campaign_cycle_memory_promoted', 'campaign_closed', 'scheduled', 'closed'}):
            return _outcome(row, 'hypothesis_campaign_cycle_scheduled_or_closed', 'retain_schedule_or_close_campaign_policy', 'retained', 'orchestrator')
    for row in rows:
        if row.get('selected_policy') == 'avoid_repeated_noop_sequence' or _has_evidence(row, {'repeated_no_gain_or_noop_loop', 'summarize_noop'}):
            return _outcome(row, 'repeated_noop_policy_retired', 'retire_repeated_noop_policy_path', 'retired', 'orchestrator')
    for row in rows:
        if row.get('selected_policy') == 'gather_more_sibling_evidence_before_policy_change' or _has_evidence(row, {'sibling_request_waiting_for_evidence', 'waiting'}):
            return _outcome(row, 'policy_waiting_for_sibling_evidence', 'keep_policy_waiting_for_sibling_evidence', 'waiting', 'broadcast')
    for row in rows:
        if row.get('selected_policy') == 'record_no_measurable_policy_gain' or _has_evidence(row, {'no_measurable_coordination_payoff', 'no_measurable_science_campaign_cycle_strategy_gain', 'no_gain'}):
            return _outcome(row, 'no_measurable_policy_gain', 'weaken_policy_until_more_evidence', 'weakened', 'orchestrator')
    return _noop_outcome('no science coordination policy outcome selected')


def _outcome(row, selected_outcome, selected_action, retention_state, recipient):
    return {
        'selected_outcome': selected_outcome,
        'selected_action': selected_action,
        'selected_recipient': validate_participant(recipient),
        'policy_retention_state': retention_state,
        'science_coordination_policy_id': row.get('science_coordination_policy_id'),
        'source_policy_ids': list(row.get('source_policy_ids') or []),
        'source_history_ids': list(row.get('source_history_ids') or []),
        'source_campaign_ids': list(row.get('source_campaign_ids') or []),
        'selected_policy': row.get('selected_policy'),
        'selected_policy_action': row.get('selected_policy_action'),
        'selected_policy_recipient': row.get('selected_policy_recipient'),
        'scenario_id': row.get('scenario_id'),
        'hypothesis_id': row.get('hypothesis_id'),
        'campaign_id': row.get('campaign_id'),
        'family_id': row.get('family_id'),
        'observed_science_side_evidence': list(row.get('observed_science_side_evidence') or []),
        'validated_hypothesis_state': row.get('validated_hypothesis_state') or 'unknown',
        'campaign_cycle_memory_state': row.get('campaign_cycle_memory_state') or 'unknown',
        'checkpoint_boundary_state': row.get('checkpoint_boundary_state') or 'clean',
        'checkpoint_boundary_notes': list(row.get('checkpoint_boundary_notes') or []),
        'source_commits': list(row.get('source_commits') or []),
        'source_tests': list(row.get('source_tests') or []),
        'no_overclaiming_proof': 'candidate-not-causal science coordination policy outcome only; no causal proof and no project-owned checkpoint claim without local verification',
        'label_leaks': list(row.get('label_leaks') or []),
    }


def _noop_outcome(reason):
    return {
        'selected_outcome': 'summarize_noop',
        'selected_action': 'summarize_noop',
        'selected_recipient': None,
        'policy_retention_state': 'none',
        'science_coordination_policy_id': None,
        'source_policy_ids': [],
        'source_history_ids': [],
        'source_campaign_ids': [],
        'selected_policy': None,
        'selected_policy_action': None,
        'selected_policy_recipient': None,
        'scenario_id': None,
        'hypothesis_id': None,
        'campaign_id': None,
        'family_id': None,
        'observed_science_side_evidence': [],
        'validated_hypothesis_state': 'unknown',
        'campaign_cycle_memory_state': 'unknown',
        'checkpoint_boundary_state': 'clean',
        'checkpoint_boundary_notes': [],
        'source_commits': [],
        'source_tests': [],
        'no_overclaiming_proof': reason,
        'label_leaks': [],
    }


def _state_counts(selected):
    counts = {
        'code': 0,
        'funfun': 0,
        'language': 0,
        'theory': 0,
        'boundary': 0,
        'waiting': 0,
        'no_gain': 0,
        'retired': 0,
    }
    mapping = {
        'preserve_checkpoint_boundary': 'boundary',
        'code_simulation_policy_improved_science_campaign': 'code',
        'funfun_formalization_policy_improved_science_campaign': 'funfun',
        'language_terminology_policy_improved_science_campaign': 'language',
        'theory_repair_policy_improved_next_campaign': 'theory',
        'hypothesis_campaign_cycle_scheduled_or_closed': 'theory',
        'repeated_noop_policy_retired': 'retired',
        'policy_waiting_for_sibling_evidence': 'waiting',
        'no_measurable_policy_gain': 'no_gain',
        'summarize_noop': 'no_gain',
    }
    key = mapping.get(selected.get('selected_outcome'))
    if key:
        counts[key] += 1
    return counts


def _evidence_class_from_value(value, source):
    mapping = {
        'code_simulation_or_primitive_capability_help_received': 'code_simulation',
        'primitive_simulation_help': 'code_simulation',
        'simulation_result': 'code_simulation',
        'code_counterexample': 'code_simulation',
        'funfun_formal_or_proof_help_received': 'funfun_formalization',
        'formal_proof_certificate': 'funfun_formalization',
        'proof_certificate': 'funfun_formalization',
        'language_terminology_or_protocol_help_received': 'language_terminology',
        'protocol_clarification': 'language_terminology',
        'terminology_clarification': 'language_terminology',
        'theory_repair_loop_helped': 'theory_repair',
        'theory_repair_cycle_reopened': 'theory_repair',
        'campaign_cycle_memory_promoted': 'campaign_cycle_scheduled_or_closed',
        'next_frontier_hypothesis_campaign_scheduled': 'campaign_cycle_scheduled_or_closed',
        'stable_science_campaign_cycle_closed': 'campaign_cycle_scheduled_or_closed',
        'closed': 'campaign_cycle_scheduled_or_closed',
        'scheduled': 'campaign_cycle_scheduled_or_closed',
        'repeated_no_gain_or_noop_loop': 'repeated_noop',
        'summarize_noop': 'repeated_noop',
        'sibling_request_waiting_for_evidence': 'waiting',
        'waiting': 'waiting',
        'no_measurable_coordination_payoff': 'no_gain',
        'no_measurable_science_campaign_cycle_strategy_gain': 'no_gain',
        'no_gain': 'no_gain',
        'preserve_checkpoint_boundary': 'boundary',
    }
    mapped = mapping.get(value)
    if mapped:
        return mapped
    sender = str(source.get('sender') or '')
    gate = str(source.get('evidence_gate') or '')
    if sender == 'code_module' or 'code' in gate or 'simulation' in gate:
        return 'code_simulation'
    if sender == 'funfun' or 'proof' in gate or 'formal' in gate:
        return 'funfun_formalization'
    if sender == 'language_model_2' or 'protocol' in gate or 'terminology' in gate:
        return 'language_terminology'
    return None


def _has_evidence(row, values):
    evidence = set(row.get('evidence_classes') or [])
    return bool(evidence.intersection(values))


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


def _find_row(rows, policy_id, scenario_id, campaign_id, hypothesis_id):
    for row in rows:
        if policy_id and row.get('science_coordination_policy_id') == policy_id:
            return row
        if scenario_id and row.get('scenario_id') == scenario_id:
            return row
        if campaign_id and row.get('campaign_id') == campaign_id:
            return row
        if hypothesis_id and row.get('hypothesis_id') == hypothesis_id:
            return row
    return None


def _outcome_key(selected):
    return stable_digest({
        'policy_id': selected.get('science_coordination_policy_id'),
        'scenario_id': selected.get('scenario_id'),
        'campaign_id': selected.get('campaign_id'),
        'hypothesis_id': selected.get('hypothesis_id'),
        'outcome': selected.get('selected_outcome'),
    })


def _dominant_state(current, incoming, order):
    current = current or order[0]
    incoming = incoming or order[0]
    if current not in order:
        current = order[0]
    if incoming not in order:
        incoming = order[0]
    return current if order.index(current) >= order.index(incoming) else incoming


def _unique_strings(values):
    seen = set()
    output = []
    for value in values:
        text = str(value)
        if text not in seen:
            output.append(text)
            seen.add(text)
    return output
