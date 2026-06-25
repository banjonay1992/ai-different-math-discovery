"""Science coordination policy recommender.

Consumes AI Different-local science coordination history/payoff records and
emits cautious science-side sequence recommendations. This is a symbolic
coordination policy layer only: candidate organizational knowledge, not causal
proof, empirical science, or a project-owned checkpoint claim.
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
from .science_campaign_strategy import SCIENCE_CAMPAIGN_STRATEGY_LEDGER_KIND
from .science_campaign_strategy_outcome import SCIENCE_CAMPAIGN_STRATEGY_OUTCOME_LEDGER_KIND
from .science_coordination_history import SCIENCE_COORDINATION_HISTORY_LEDGER_KIND
from .science_theory_frontier_outcome import SCIENCE_THEORY_FRONTIER_OUTCOME_LEDGER_KIND


SCIENCE_COORDINATION_POLICY_LEDGER_KIND = 'ai_different.science_coordination_policy_ledger'


def empty_science_coordination_policy_ledger() -> dict[str, Any]:
    return {
        'schema_version': 1,
        'ledger_kind': SCIENCE_COORDINATION_POLICY_LEDGER_KIND,
        'processed_message_ids': [],
        'processed_source_hashes': [],
        'planned_policy_keys': [],
        'policy_records': [],
        'policy_rows': [],
        'outgoing_response_ids': [],
        'latest': {},
    }


def load_science_coordination_policy_ledger(path: str | Path | None) -> dict[str, Any]:
    if not path or not Path(path).exists():
        return empty_science_coordination_policy_ledger()
    with Path(path).open('r', encoding='utf-8') as handle:
        ledger = json.load(handle)
    return validate_science_coordination_policy_ledger(ledger)


def write_science_coordination_policy_ledger(path: str | Path, ledger: dict[str, Any]) -> Path:
    validated = validate_science_coordination_policy_ledger(ledger)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(validated, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return output


def validate_science_coordination_policy_ledger(ledger: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(ledger, dict):
        raise ValueError('science coordination policy ledger must be a JSON object')
    if ledger.get('ledger_kind') != SCIENCE_COORDINATION_POLICY_LEDGER_KIND:
        raise ValueError('science coordination policy ledger has wrong ledger_kind')
    for field in ('processed_message_ids', 'processed_source_hashes', 'planned_policy_keys', 'outgoing_response_ids'):
        values = ledger.get(field)
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            raise ValueError(f'{field} must be strings')
    for field in ('policy_records', 'policy_rows'):
        values = ledger.get(field)
        if not isinstance(values, list) or not all(isinstance(item, dict) for item in values):
            raise ValueError(f'{field} must be objects')
    latest = ledger.get('latest', {})
    if not isinstance(latest, dict):
        raise ValueError('science coordination policy latest must be an object')
    validated = {
        'schema_version': int(ledger.get('schema_version', 1) or 1),
        'ledger_kind': SCIENCE_COORDINATION_POLICY_LEDGER_KIND,
        'processed_message_ids': _unique_strings(ledger['processed_message_ids']),
        'processed_source_hashes': _unique_strings(ledger['processed_source_hashes']),
        'planned_policy_keys': _unique_strings(ledger['planned_policy_keys']),
        'policy_records': list(ledger['policy_records']),
        'policy_rows': list(ledger['policy_rows']),
        'outgoing_response_ids': _unique_strings(ledger['outgoing_response_ids']),
        'latest': dict(latest),
    }
    if ledger.get('ledger_hash'):
        validated['ledger_hash'] = str(ledger['ledger_hash'])
    if ledger.get('artifact_path'):
        validated['artifact_path'] = str(ledger['artifact_path'])
    return validated


def read_science_coordination_policy_transcript(path: str | Path | None) -> dict[str, Any]:
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
        raise ValueError('plain science coordination policy input must be a JSON object')
    return value


def build_science_coordination_policy_recommendation(
    *,
    transcript_messages: list[dict[str, Any]],
    policy_ledger: dict[str, Any],
    history_ledger: dict[str, Any] | None = None,
    cycle_strategy_outcome_ledger: dict[str, Any] | None = None,
    cycle_strategy_ledger: dict[str, Any] | None = None,
    strategy_outcome_ledger: dict[str, Any] | None = None,
    strategy_ledger: dict[str, Any] | None = None,
    frontier_outcome_ledger: dict[str, Any] | None = None,
    theory_memory_ledger: dict[str, Any] | None = None,
    campaign_cycle_memory_ledger: dict[str, Any] | None = None,
    hypothesis_ledger: dict[str, Any] | None = None,
    campaign_ledger: dict[str, Any] | None = None,
    module_chat_ledger: dict[str, Any] | None = None,
    prior_policy_ledger: dict[str, Any] | None = None,
    runtime_memory_data: dict[str, Any] | None = None,
    runtime_memory_hash_state: dict[str, Any],
    project_owned_boundary: dict[str, Any],
    artifact_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    ledger = validate_science_coordination_policy_ledger(policy_ledger)
    history = _valid_kind_or_empty(history_ledger or {}, SCIENCE_COORDINATION_HISTORY_LEDGER_KIND, 'event_records')
    cycle_outcome = _valid_kind_or_empty(cycle_strategy_outcome_ledger or {}, SCIENCE_CAMPAIGN_CYCLE_STRATEGY_OUTCOME_LEDGER_KIND, 'outcome_records')
    cycle_strategy = _valid_kind_or_empty(cycle_strategy_ledger or {}, SCIENCE_CAMPAIGN_CYCLE_STRATEGY_LEDGER_KIND, 'cycle_strategy_records')
    strategy_outcome = _valid_kind_or_empty(strategy_outcome_ledger or {}, SCIENCE_CAMPAIGN_STRATEGY_OUTCOME_LEDGER_KIND, 'outcome_records')
    strategy = _valid_kind_or_empty(strategy_ledger or {}, SCIENCE_CAMPAIGN_STRATEGY_LEDGER_KIND, 'strategy_records')
    frontier_outcome = _valid_kind_or_empty(frontier_outcome_ledger or {}, SCIENCE_THEORY_FRONTIER_OUTCOME_LEDGER_KIND, 'outcome_records')
    theory_memory = _valid_plain_or_empty(theory_memory_ledger or {})
    campaign_cycle_memory = _valid_plain_or_empty(campaign_cycle_memory_ledger or {})
    hypothesis = _valid_plain_or_empty(hypothesis_ledger or {})
    campaign = _valid_plain_or_empty(campaign_ledger or {})
    module_chat = _valid_plain_or_empty(module_chat_ledger or {})
    prior_policy = _valid_prior_policy_or_empty(prior_policy_ledger or {})
    runtime_memory = dict(runtime_memory_data or {})
    source_hash = _source_hash(
        history,
        cycle_outcome,
        cycle_strategy,
        strategy_outcome,
        strategy,
        frontier_outcome,
        theory_memory,
        campaign_cycle_memory,
        hypothesis,
        campaign,
        module_chat,
        prior_policy,
        runtime_memory,
    )
    source_is_new = source_hash not in set(ledger['processed_source_hashes'])
    processed = set(ledger['processed_message_ids'])
    new_messages = [message for message in transcript_messages if module_chat_message_id(message) not in processed]
    skipped_messages = [message for message in transcript_messages if module_chat_message_id(message) in processed]
    rows = _extract_policy_rows(
        ledger['policy_rows'],
        history,
        cycle_outcome,
        cycle_strategy,
        strategy_outcome,
        strategy,
        frontier_outcome,
        theory_memory,
        campaign_cycle_memory,
        hypothesis,
        campaign,
        module_chat,
        transcript_messages,
    )
    if not new_messages and not source_is_new:
        selected = _noop_policy('no new science coordination policy evidence or source ledger state')
    else:
        selected = _select_policy(
            rows=rows,
            planned_keys=ledger['planned_policy_keys'],
            project_owned_boundary=project_owned_boundary,
        )
    policy_id = 'science_coordination_policy_' + stable_digest({
        'source_hash': source_hash,
        'new_message_ids': [module_chat_message_id(message) for message in new_messages],
        'policy': selected['selected_policy'],
        'scenario_id': selected.get('scenario_id'),
    })[:16]
    selected['science_coordination_policy_id'] = policy_id
    message = export_science_coordination_policy_message(
        selected,
        rows=rows,
        runtime_memory_hash_state=runtime_memory_hash_state,
        project_owned_boundary=project_owned_boundary,
        source_hashes={
            'science_coordination_policy_source_hash': source_hash,
            'history_ledger_hash': history.get('ledger_hash'),
            'cycle_strategy_outcome_ledger_hash': cycle_outcome.get('ledger_hash'),
            'cycle_strategy_ledger_hash': cycle_strategy.get('ledger_hash'),
            'strategy_outcome_ledger_hash': strategy_outcome.get('ledger_hash'),
            'strategy_ledger_hash': strategy.get('ledger_hash'),
            'frontier_outcome_ledger_hash': frontier_outcome.get('ledger_hash'),
            'theory_memory_ledger_hash': theory_memory.get('ledger_hash'),
            'campaign_cycle_memory_ledger_hash': campaign_cycle_memory.get('ledger_hash'),
            'hypothesis_ledger_hash': hypothesis.get('ledger_hash'),
            'campaign_ledger_hash': campaign.get('ledger_hash'),
            'module_chat_ledger_hash': module_chat.get('ledger_hash'),
            'prior_policy_ledger_hash': prior_policy.get('ledger_hash'),
        },
    )
    new_ids = [module_chat_message_id(message) for message in new_messages]
    outgoing_id = module_chat_message_id(message) if message else None
    policy_key = _policy_key(selected)
    if selected['selected_policy'] != 'summarize_noop':
        ledger['planned_policy_keys'] = _unique_strings(list(ledger['planned_policy_keys']) + [policy_key])
    if new_messages:
        ledger['processed_message_ids'] = _unique_strings(list(ledger['processed_message_ids']) + new_ids)
    if source_is_new:
        ledger['processed_source_hashes'] = _unique_strings(list(ledger['processed_source_hashes']) + [source_hash])
    if outgoing_id:
        ledger['outgoing_response_ids'] = _unique_strings(list(ledger['outgoing_response_ids']) + [outgoing_id])
    state_counts = _state_counts(selected)
    latest = {
        'science_coordination_policy_id': policy_id,
        'new_message_count': len(new_messages),
        'skipped_message_count': len(skipped_messages),
        'source_is_new': source_is_new,
        'state_counts': state_counts,
        'selected_policy': selected['selected_policy'],
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
        'science_coordination_policy_id': policy_id,
        'policy_hash': stable_digest({'policy_id': policy_id, 'selected': selected}),
        'processed_message_ids': new_ids,
        'source_hash': source_hash,
        'source_history_ids': selected.get('source_history_ids'),
        'observed_sequences': selected.get('observed_sequences'),
        'payoff_classes': selected.get('payoff_classes'),
        'recommendation_strength': selected.get('recommendation_strength'),
        'selected_policy': selected['selected_policy'],
        'selected_action': selected['selected_action'],
        'selected_recipient': latest['chosen_recipient'],
        'hypothesis_id': selected.get('hypothesis_id'),
        'campaign_id': selected.get('campaign_id'),
        'family_id': selected.get('family_id'),
        'checkpoint_boundary_state': selected.get('checkpoint_boundary_state'),
        'no_overclaiming_proof': selected.get('no_overclaiming_proof'),
        'outgoing_response_id': outgoing_id,
    }
    ledger['policy_rows'] = rows
    if new_messages or source_is_new or message is not None:
        ledger['policy_records'] = list(ledger['policy_records']) + [record]
    ledger['latest'] = latest
    if artifact_path is not None:
        ledger['artifact_path'] = str(Path(artifact_path))
    ledger['ledger_hash'] = stable_digest({key: value for key, value in ledger.items() if key != 'ledger_hash'})
    return ledger, message


def export_science_coordination_policy_message(
    selected: dict[str, Any],
    *,
    rows: list[dict[str, Any]],
    runtime_memory_hash_state: dict[str, Any],
    project_owned_boundary: dict[str, Any],
    source_hashes: dict[str, Any],
    topic: str = 'ai_different.science_coordination_policy',
) -> dict[str, Any] | None:
    if selected['selected_policy'] == 'summarize_noop':
        return None
    recipient = validate_participant(selected.get('selected_recipient') or 'orchestrator')
    row = _find_row(rows, selected.get('scenario_id'), selected.get('campaign_id'), selected.get('hypothesis_id')) or {}
    leak_terms = label_leak_terms({'selected': selected, 'row': row})
    body = {
        'module': 'AI Different',
        'response_kind': 'science_coordination_policy',
        'science_coordination_policy_id': selected.get('science_coordination_policy_id'),
        'selected_policy': selected['selected_policy'],
        'selected_action': selected['selected_action'],
        'selected_recipient': recipient,
        'source_history_ids': selected.get('source_history_ids') or [],
        'observed_sequences': selected.get('observed_sequences') or [],
        'payoff_classes': selected.get('payoff_classes') or [],
        'recommendation_strength': selected['recommendation_strength'],
        'recommendation_text': selected.get('recommendation_text'),
        'candidate_not_causal': True,
        'source_commits': selected.get('source_commits') or [],
        'source_tests': selected.get('source_tests') or [],
        'hypothesis_id': selected.get('hypothesis_id'),
        'campaign_id': selected.get('campaign_id'),
        'family_id': selected.get('family_id'),
        'scenario_id': selected.get('scenario_id'),
        'checkpoint_boundary_state': selected.get('checkpoint_boundary_state'),
        'checkpoint_boundary_notes': selected.get('checkpoint_boundary_notes') or [],
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
            'science_coordination_policy_id': body['science_coordination_policy_id'],
            'selected_policy': body['selected_policy'],
            'recommendation_strength': body['recommendation_strength'],
            'candidate_not_causal': True,
            'label_clean': body['label_clean'],
            'no_sibling_imports': True,
        },
        tags=['ai_different', 'science_coordination_policy', body['selected_policy'], 'label_clean' if body['label_clean'] else 'label_review_needed'],
    )


def write_science_coordination_policy_outbox_jsonl(path: str | Path, message: dict[str, Any] | None) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text((json.dumps(message, sort_keys=True) + '\n') if message is not None else '', encoding='utf-8')
    return output


def _valid_kind_or_empty(ledger, kind, record_field):
    if not ledger:
        return {'ledger_kind': kind, record_field: [], 'ledger_hash': None}
    if ledger.get('ledger_kind') != kind:
        raise ValueError(f'plain science coordination policy source has wrong ledger_kind: {ledger.get("ledger_kind")}')
    return dict(ledger)


def _valid_plain_or_empty(ledger):
    if not ledger:
        return {}
    if not isinstance(ledger, dict):
        raise ValueError('plain science coordination policy source must be a JSON object')
    return dict(ledger)


def _valid_prior_policy_or_empty(ledger):
    if not ledger:
        return {}
    if ledger.get('ledger_kind') != SCIENCE_COORDINATION_POLICY_LEDGER_KIND:
        raise ValueError('prior science coordination policy ledger has wrong ledger_kind')
    return validate_science_coordination_policy_ledger(ledger)


def _extract_policy_rows(existing, history, cycle_outcome, cycle_strategy, strategy_outcome, strategy, frontier_outcome, theory_memory, campaign_cycle_memory, hypothesis, campaign, module_chat, messages):
    rows: dict[str, dict[str, Any]] = {}
    for row in list(existing or []):
        _upsert_row(rows, row)
    for record in list(history.get('event_records') or []):
        _upsert_row(rows, _row_from_history_record(record))
    for row in list(history.get('history_rows') or []):
        _upsert_row(rows, _row_from_history_row(row))
    for record in list(cycle_outcome.get('outcome_records') or []):
        _upsert_row(rows, _row_from_generic(record, 'cycle_strategy_outcome_record'))
    for record in list(cycle_strategy.get('cycle_strategy_records') or []):
        _upsert_row(rows, _row_from_generic(record, 'cycle_strategy_record'))
    for record in list(strategy_outcome.get('outcome_records') or []):
        _upsert_row(rows, _row_from_generic(record, 'strategy_outcome_record'))
    for record in list(strategy.get('strategy_records') or []):
        _upsert_row(rows, _row_from_generic(record, 'strategy_record'))
    for record in list(frontier_outcome.get('outcome_records') or []):
        _upsert_row(rows, _row_from_generic(record, 'frontier_outcome_record'))
    for row in _plain_rows(theory_memory, ('theory_memory_rows', 'theory_records', 'theories')):
        _upsert_row(rows, _row_from_generic(row, 'theory_memory'))
    for row in _plain_rows(campaign_cycle_memory, ('campaign_cycle_memory_rows', 'cycle_memory_rows')):
        _upsert_row(rows, _row_from_generic(row, 'campaign_cycle_memory'))
    for row in _plain_rows(hypothesis, ('hypothesis_rows', 'hypotheses', 'hypothesis_records')):
        _upsert_row(rows, _row_from_generic(row, 'hypothesis'))
    for row in _plain_rows(campaign, ('campaign_rows', 'campaigns', 'campaign_records')):
        _upsert_row(rows, _row_from_generic(row, 'campaign'))
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
        'source_history_ids': list(source.get('source_history_ids') or ([source.get('science_coordination_history_id')] if source.get('science_coordination_history_id') else [])),
        'observed_sequences': list(source.get('observed_sequences') or source.get('interaction_sequence') or []),
        'payoff_classes': list(source.get('payoff_classes') or ([source.get('payoff_class')] if source.get('payoff_class') else [])),
        'recommendation_strength': source.get('recommendation_strength') or 'weak',
        'source_commits': list(source.get('source_commits') or []),
        'source_tests': list(source.get('source_tests') or []),
        'checkpoint_boundary_state': source.get('checkpoint_boundary_state') or 'clean',
        'checkpoint_boundary_notes': list(source.get('checkpoint_boundary_notes') or []),
        'label_leaks': list(source.get('label_leaks') or []),
        'lineage': [],
    }


def _row_from_history_record(record):
    row = _base_row(record)
    row['lineage'] = ['history_record']
    return row


def _row_from_history_row(row):
    value = _base_row(row)
    value['lineage'] = ['history_row']
    return value


def _row_from_generic(record, lineage):
    row = _base_row(record)
    payoff = record.get('payoff_class') or _payoff_from_outcome(record.get('selected_outcome') or record.get('selected_science_outcome') or record.get('selected_strategy'))
    if payoff and not row['payoff_classes']:
        row['payoff_classes'] = [payoff]
    row['lineage'] = [lineage]
    return row


def _row_from_message(message):
    body = dict(message.get('body') or {})
    evidence = dict(message.get('evidence') or {})
    sender = message.get('sender')
    source = {
        'scenario_id': body.get('scenario_id') or evidence.get('scenario_id') or body.get('campaign_id') or evidence.get('campaign_id') or body.get('hypothesis_id') or evidence.get('hypothesis_id'),
        'hypothesis_id': body.get('hypothesis_id') or evidence.get('hypothesis_id'),
        'campaign_id': body.get('campaign_id') or evidence.get('campaign_id'),
        'family_id': body.get('family_id') or evidence.get('family_id'),
        'science_coordination_history_id': body.get('science_coordination_history_id') or evidence.get('science_coordination_history_id'),
        'interaction_sequence': body.get('observed_sequences') or body.get('interaction_sequence') or [],
        'payoff_class': body.get('payoff_class') or evidence.get('payoff_class') or _payoff_from_outcome(body.get('selected_outcome') or body.get('selected_strategy')),
        'recommendation_strength': body.get('recommendation_strength') or evidence.get('recommendation_strength') or 'weak',
        'source_commits': body.get('source_commits') or evidence.get('source_commits') or [],
        'source_tests': body.get('source_tests') or evidence.get('source_tests') or [],
        'checkpoint_boundary_state': body.get('checkpoint_boundary_state') or evidence.get('checkpoint_boundary_state') or 'clean',
        'checkpoint_boundary_notes': body.get('checkpoint_boundary_notes') or [],
        'label_leaks': label_leak_terms({'body': body, 'evidence': evidence}),
    }
    if body.get('third_party_checkpoint_used') or evidence.get('third_party_checkpoint_used'):
        source['checkpoint_boundary_state'] = 'repair'
        source['checkpoint_boundary_notes'] = list(source['checkpoint_boundary_notes']) + ['third-party checkpoint boundary preserved']
    row = _base_row(source)
    row['lineage'] = [f'message:{sender}']
    return row


def _upsert_row(rows, incoming):
    key = incoming.get('scenario_id') or incoming.get('campaign_id') or incoming.get('hypothesis_id') or stable_digest(incoming)[:12]
    current = rows.setdefault(str(key), {
        'scenario_id': incoming.get('scenario_id'),
        'hypothesis_id': incoming.get('hypothesis_id'),
        'campaign_id': incoming.get('campaign_id'),
        'family_id': incoming.get('family_id'),
        'source_history_ids': [],
        'observed_sequences': [],
        'payoff_classes': [],
        'recommendation_strength': 'weak',
        'source_commits': [],
        'source_tests': [],
        'checkpoint_boundary_state': 'clean',
        'checkpoint_boundary_notes': [],
        'label_leaks': [],
        'lineage': [],
    })
    for field in ('scenario_id', 'hypothesis_id', 'campaign_id', 'family_id'):
        current[field] = current.get(field) or incoming.get(field)
    current['source_history_ids'] = _unique_strings(current['source_history_ids'] + list(incoming.get('source_history_ids') or []))
    current['observed_sequences'] = _unique_strings(current['observed_sequences'] + list(incoming.get('observed_sequences') or []))
    current['payoff_classes'] = _dominant_payoffs(current['payoff_classes'] + list(incoming.get('payoff_classes') or []))
    current['recommendation_strength'] = _dominant_strength(current.get('recommendation_strength'), incoming.get('recommendation_strength'))
    current['source_commits'] = _unique_strings(current['source_commits'] + list(incoming.get('source_commits') or []))
    current['source_tests'] = _unique_strings(current['source_tests'] + list(incoming.get('source_tests') or []))
    current['checkpoint_boundary_state'] = 'repair' if current.get('checkpoint_boundary_state') == 'repair' or incoming.get('checkpoint_boundary_state') == 'repair' else 'clean'
    current['checkpoint_boundary_notes'] = _unique_strings(current['checkpoint_boundary_notes'] + list(incoming.get('checkpoint_boundary_notes') or []))
    current['label_leaks'] = _unique_strings(current['label_leaks'] + list(incoming.get('label_leaks') or []))
    current['lineage'] = _unique_strings(current['lineage'] + list(incoming.get('lineage') or []))


def _finalize_row(row):
    if row.get('label_leaks'):
        row['checkpoint_boundary_state'] = 'repair'
    row['policy_row_hash'] = stable_digest({
        'scenario_id': row.get('scenario_id'),
        'payoffs': row.get('payoff_classes'),
        'boundary': row.get('checkpoint_boundary_state'),
    })
    return row


def _select_policy(*, rows, planned_keys, project_owned_boundary):
    for row in rows:
        if project_owned_boundary.get('third_party_checkpoint_used') or row.get('checkpoint_boundary_state') == 'repair':
            boundary_row = dict(row)
            if project_owned_boundary.get('third_party_checkpoint_used'):
                boundary_row['checkpoint_boundary_state'] = 'repair'
                boundary_row['checkpoint_boundary_notes'] = _unique_strings(
                    list(boundary_row.get('checkpoint_boundary_notes') or [])
                    + ['third-party checkpoint boundary preserved']
                )
            return _policy(boundary_row, 'preserve_checkpoint_boundary', 'preserve_checkpoint_boundary', 'code_module')
    for row in rows:
        if _has_payoff(row, 'code_simulation_before_science_campaign_paid_off'):
            return _policy(row, 'try_code_simulation_before_science_campaign', 'request_code_simulation_before_campaign', 'code_module')
    for row in rows:
        if _has_payoff(row, 'funfun_formalization_before_science_campaign_paid_off'):
            return _policy(row, 'try_funfun_formalization_before_science_campaign', 'request_funfun_formalization_before_campaign', 'funfun')
    for row in rows:
        if _has_payoff(row, 'language_terminology_before_science_campaign_paid_off'):
            return _policy(row, 'try_language_terminology_before_science_campaign', 'request_language_terminology_before_campaign', 'language_model_2')
    for row in rows:
        if _has_payoff(row, 'theory_repair_loop_helped'):
            return _policy(row, 'try_theory_repair_before_next_campaign', 'reopen_theory_repair_before_campaign', 'orchestrator')
    for row in rows:
        if _has_payoff(row, 'repeated_no_gain_or_noop_loop'):
            return _policy(row, 'avoid_repeated_noop_sequence', 'avoid_repeated_noop_sequence', 'orchestrator')
    for row in rows:
        if _has_payoff(row, 'sibling_request_waiting_for_evidence'):
            return _policy(row, 'gather_more_sibling_evidence_before_policy_change', 'gather_more_sibling_evidence', 'broadcast')
    for row in rows:
        if _has_payoff(row, 'next_frontier_hypothesis_campaign_helped') or _has_payoff(row, 'stable_science_campaign_cycle_closed') or _has_payoff(row, 'validated_hypothesis_cycle_memory_payoff'):
            return _policy(row, 'schedule_hypothesis_campaign_cycle', 'schedule_hypothesis_campaign_cycle', 'orchestrator')
    for row in rows:
        if _has_payoff(row, 'no_measurable_coordination_payoff'):
            return _policy(row, 'record_no_measurable_policy_gain', 'record_no_measurable_policy_gain', 'orchestrator')
    return _noop_policy('no science coordination policy selected')


def _policy(row, selected_policy, action, recipient):
    return {
        'selected_policy': selected_policy,
        'selected_action': action,
        'selected_recipient': validate_participant(recipient),
        'source_history_ids': list(row.get('source_history_ids') or []),
        'observed_sequences': list(row.get('observed_sequences') or []),
        'payoff_classes': list(row.get('payoff_classes') or []),
        'recommendation_strength': row.get('recommendation_strength') or _strength_for_policy(selected_policy),
        'recommendation_text': _recommendation_text(selected_policy),
        'source_commits': list(row.get('source_commits') or []),
        'source_tests': list(row.get('source_tests') or []),
        'hypothesis_id': row.get('hypothesis_id'),
        'campaign_id': row.get('campaign_id'),
        'family_id': row.get('family_id'),
        'scenario_id': row.get('scenario_id'),
        'checkpoint_boundary_state': row.get('checkpoint_boundary_state') or 'clean',
        'checkpoint_boundary_notes': list(row.get('checkpoint_boundary_notes') or []),
        'no_overclaiming_proof': 'candidate science-side coordination policy only; not causal proof and no local-owned checkpoint claim unless status capsule verifies it',
        'label_leaks': list(row.get('label_leaks') or []),
    }


def _noop_policy(reason):
    return {
        'selected_policy': 'summarize_noop',
        'selected_action': 'summarize_noop',
        'selected_recipient': None,
        'source_history_ids': [],
        'observed_sequences': [],
        'payoff_classes': [],
        'recommendation_strength': 'none',
        'recommendation_text': reason,
        'source_commits': [],
        'source_tests': [],
        'hypothesis_id': None,
        'campaign_id': None,
        'family_id': None,
        'scenario_id': None,
        'checkpoint_boundary_state': 'clean',
        'checkpoint_boundary_notes': [],
        'no_overclaiming_proof': 'no project-owned checkpoint claim made',
        'label_leaks': [],
    }


def _recommendation_text(policy):
    return (
        f'Candidate-not-causal science coordination policy: {policy}. '
        'Use as cautious sequencing guidance, not proof of global orchestration benefit.'
    )


def _state_counts(selected):
    counts = {
        'code': 0,
        'funfun': 0,
        'language': 0,
        'boundary': 0,
        'avoid': 0,
        'no_gain': 0,
        'waiting': 0,
        'schedule': 0,
    }
    mapping = {
        'preserve_checkpoint_boundary': 'boundary',
        'try_code_simulation_before_science_campaign': 'code',
        'try_funfun_formalization_before_science_campaign': 'funfun',
        'try_language_terminology_before_science_campaign': 'language',
        'try_theory_repair_before_next_campaign': 'schedule',
        'avoid_repeated_noop_sequence': 'avoid',
        'gather_more_sibling_evidence_before_policy_change': 'waiting',
        'schedule_hypothesis_campaign_cycle': 'schedule',
        'record_no_measurable_policy_gain': 'no_gain',
        'summarize_noop': 'no_gain',
    }
    key = mapping.get(selected.get('selected_policy'))
    if key:
        counts[key] += 1
    return counts


def _payoff_from_outcome(outcome):
    mapping = {
        'campaign_cycle_memory_promoted': 'validated_hypothesis_cycle_memory_payoff',
        'code_simulation_or_primitive_capability_help_received': 'code_simulation_before_science_campaign_paid_off',
        'funfun_formal_or_proof_help_received': 'funfun_formalization_before_science_campaign_paid_off',
        'language_terminology_or_protocol_help_received': 'language_terminology_before_science_campaign_paid_off',
        'theory_repair_cycle_reopened': 'theory_repair_loop_helped',
        'stable_science_campaign_cycle_closed': 'stable_science_campaign_cycle_closed',
        'next_frontier_hypothesis_campaign_scheduled': 'next_frontier_hypothesis_campaign_helped',
        'planned_science_campaign_cycle_strategy_waiting_for_evidence': 'sibling_request_waiting_for_evidence',
        'no_measurable_science_campaign_cycle_strategy_gain': 'no_measurable_coordination_payoff',
        'summarize_noop': 'repeated_no_gain_or_noop_loop',
        'preserve_checkpoint_boundary': 'preserve_checkpoint_boundary',
    }
    return mapping.get(outcome)


def _has_payoff(row, payoff):
    return payoff in set(row.get('payoff_classes') or [])


def _strength_for_policy(policy):
    if policy in {'preserve_checkpoint_boundary', 'try_code_simulation_before_science_campaign'}:
        return 'strong'
    if policy in {'try_funfun_formalization_before_science_campaign', 'try_language_terminology_before_science_campaign', 'try_theory_repair_before_next_campaign'}:
        return 'moderate'
    if policy == 'summarize_noop':
        return 'none'
    return 'weak'


def _dominant_payoffs(payoffs):
    order = {
        'preserve_checkpoint_boundary': 10,
        'code_simulation_before_science_campaign_paid_off': 9,
        'funfun_formalization_before_science_campaign_paid_off': 8,
        'language_terminology_before_science_campaign_paid_off': 7,
        'theory_repair_loop_helped': 6,
        'repeated_no_gain_or_noop_loop': 5,
        'sibling_request_waiting_for_evidence': 4,
        'next_frontier_hypothesis_campaign_helped': 3,
        'stable_science_campaign_cycle_closed': 3,
        'validated_hypothesis_cycle_memory_payoff': 3,
        'no_measurable_coordination_payoff': 1,
    }
    return sorted(_unique_strings([item for item in payoffs if item]), key=lambda item: -order.get(item, 0))


def _dominant_strength(current, incoming):
    order = {'strong': 3, 'moderate': 2, 'weak': 1, 'none': 0, None: 0}
    return current if order.get(current, 0) >= order.get(incoming, 0) else incoming


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


def _find_row(rows, scenario_id, campaign_id, hypothesis_id):
    for row in rows:
        if scenario_id and row.get('scenario_id') == scenario_id:
            return row
        if campaign_id and row.get('campaign_id') == campaign_id:
            return row
        if hypothesis_id and row.get('hypothesis_id') == hypothesis_id:
            return row
    return None


def _policy_key(selected):
    return stable_digest({
        'scenario_id': selected.get('scenario_id'),
        'hypothesis_id': selected.get('hypothesis_id'),
        'campaign_id': selected.get('campaign_id'),
        'family_id': selected.get('family_id'),
        'selected_policy': selected.get('selected_policy'),
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
