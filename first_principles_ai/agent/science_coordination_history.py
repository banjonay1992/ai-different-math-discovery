"""Science coordination payoff/history exporter.

This adapter converts AI Different-local campaign/cycle outcome evidence into
neutral candidate organizational-learning records for the module-chat
``history`` participant. It does not own global history, does not prove causal
science benefit, and does not claim a project-owned checkpoint unless local
status evidence verifies one.
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
from .science_theory_frontier import SCIENCE_THEORY_FRONTIER_LEDGER_KIND
from .science_theory_frontier_outcome import SCIENCE_THEORY_FRONTIER_OUTCOME_LEDGER_KIND


SCIENCE_COORDINATION_HISTORY_LEDGER_KIND = 'ai_different.science_coordination_history_ledger'


def empty_science_coordination_history_ledger() -> dict[str, Any]:
    return {
        'schema_version': 1,
        'ledger_kind': SCIENCE_COORDINATION_HISTORY_LEDGER_KIND,
        'processed_message_ids': [],
        'processed_source_hashes': [],
        'exported_event_keys': [],
        'event_records': [],
        'history_rows': [],
        'outgoing_response_ids': [],
        'latest': {},
    }


def load_science_coordination_history_ledger(path: str | Path | None) -> dict[str, Any]:
    if not path or not Path(path).exists():
        return empty_science_coordination_history_ledger()
    with Path(path).open('r', encoding='utf-8') as handle:
        ledger = json.load(handle)
    return validate_science_coordination_history_ledger(ledger)


def write_science_coordination_history_ledger(path: str | Path, ledger: dict[str, Any]) -> Path:
    validated = validate_science_coordination_history_ledger(ledger)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(validated, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return output


def validate_science_coordination_history_ledger(ledger: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(ledger, dict):
        raise ValueError('science coordination history ledger must be a JSON object')
    if ledger.get('ledger_kind') != SCIENCE_COORDINATION_HISTORY_LEDGER_KIND:
        raise ValueError('science coordination history ledger has wrong ledger_kind')
    for field in ('processed_message_ids', 'processed_source_hashes', 'exported_event_keys', 'outgoing_response_ids'):
        values = ledger.get(field)
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            raise ValueError(f'{field} must be strings')
    for field in ('event_records', 'history_rows'):
        values = ledger.get(field)
        if not isinstance(values, list) or not all(isinstance(item, dict) for item in values):
            raise ValueError(f'{field} must be objects')
    latest = ledger.get('latest', {})
    if not isinstance(latest, dict):
        raise ValueError('science coordination history latest must be an object')
    validated = {
        'schema_version': int(ledger.get('schema_version', 1) or 1),
        'ledger_kind': SCIENCE_COORDINATION_HISTORY_LEDGER_KIND,
        'processed_message_ids': _unique_strings(ledger['processed_message_ids']),
        'processed_source_hashes': _unique_strings(ledger['processed_source_hashes']),
        'exported_event_keys': _unique_strings(ledger['exported_event_keys']),
        'event_records': list(ledger['event_records']),
        'history_rows': list(ledger['history_rows']),
        'outgoing_response_ids': _unique_strings(ledger['outgoing_response_ids']),
        'latest': dict(latest),
    }
    if ledger.get('ledger_hash'):
        validated['ledger_hash'] = str(ledger['ledger_hash'])
    if ledger.get('artifact_path'):
        validated['artifact_path'] = str(ledger['artifact_path'])
    return validated


def read_science_coordination_history_transcript(path: str | Path | None) -> dict[str, Any]:
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
        raise ValueError('plain science coordination history input must be a JSON object')
    return value


def build_science_coordination_history_export(
    *,
    transcript_messages: list[dict[str, Any]],
    history_ledger: dict[str, Any],
    cycle_strategy_outcome_ledger: dict[str, Any] | None = None,
    cycle_strategy_ledger: dict[str, Any] | None = None,
    strategy_outcome_ledger: dict[str, Any] | None = None,
    strategy_ledger: dict[str, Any] | None = None,
    frontier_outcome_ledger: dict[str, Any] | None = None,
    frontier_ledger: dict[str, Any] | None = None,
    theory_memory_ledger: dict[str, Any] | None = None,
    campaign_cycle_memory_ledger: dict[str, Any] | None = None,
    hypothesis_ledger: dict[str, Any] | None = None,
    campaign_ledger: dict[str, Any] | None = None,
    experiment_ledger: dict[str, Any] | None = None,
    module_chat_ledger: dict[str, Any] | None = None,
    prior_history_ledger: dict[str, Any] | None = None,
    runtime_memory_data: dict[str, Any] | None = None,
    runtime_memory_hash_state: dict[str, Any],
    project_owned_boundary: dict[str, Any],
    artifact_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    ledger = validate_science_coordination_history_ledger(history_ledger)
    cycle_outcome = _valid_kind_or_empty(cycle_strategy_outcome_ledger or {}, SCIENCE_CAMPAIGN_CYCLE_STRATEGY_OUTCOME_LEDGER_KIND, 'outcome_records')
    cycle_strategy = _valid_kind_or_empty(cycle_strategy_ledger or {}, SCIENCE_CAMPAIGN_CYCLE_STRATEGY_LEDGER_KIND, 'cycle_strategy_records')
    strategy_outcome = _valid_kind_or_empty(strategy_outcome_ledger or {}, SCIENCE_CAMPAIGN_STRATEGY_OUTCOME_LEDGER_KIND, 'outcome_records')
    strategy = _valid_kind_or_empty(strategy_ledger or {}, SCIENCE_CAMPAIGN_STRATEGY_LEDGER_KIND, 'strategy_records')
    frontier_outcome = _valid_kind_or_empty(frontier_outcome_ledger or {}, SCIENCE_THEORY_FRONTIER_OUTCOME_LEDGER_KIND, 'outcome_records')
    frontier = _valid_kind_or_empty(frontier_ledger or {}, SCIENCE_THEORY_FRONTIER_LEDGER_KIND, 'frontier_records')
    theory_memory = _valid_plain_or_empty(theory_memory_ledger or {})
    campaign_cycle_memory = _valid_plain_or_empty(campaign_cycle_memory_ledger or {})
    hypothesis = _valid_plain_or_empty(hypothesis_ledger or {})
    campaign = _valid_plain_or_empty(campaign_ledger or {})
    experiment = _valid_plain_or_empty(experiment_ledger or {})
    module_chat = _valid_plain_or_empty(module_chat_ledger or {})
    prior_history = _valid_prior_history_or_empty(prior_history_ledger or {})
    runtime_memory = dict(runtime_memory_data or {})
    source_hash = _source_hash(
        cycle_outcome,
        cycle_strategy,
        strategy_outcome,
        strategy,
        frontier_outcome,
        frontier,
        theory_memory,
        campaign_cycle_memory,
        hypothesis,
        campaign,
        experiment,
        module_chat,
        prior_history,
        runtime_memory,
    )
    source_is_new = source_hash not in set(ledger['processed_source_hashes'])
    processed = set(ledger['processed_message_ids'])
    new_messages = [message for message in transcript_messages if module_chat_message_id(message) not in processed]
    skipped_messages = [message for message in transcript_messages if module_chat_message_id(message) in processed]
    rows = _extract_history_rows(
        ledger['history_rows'],
        cycle_outcome,
        cycle_strategy,
        strategy_outcome,
        strategy,
        frontier_outcome,
        frontier,
        theory_memory,
        campaign_cycle_memory,
        hypothesis,
        campaign,
        experiment,
        module_chat,
        transcript_messages,
    )
    if not new_messages and not source_is_new:
        selected = _noop_event('no new science coordination history evidence or source ledger state')
    else:
        selected = _select_payoff(
            rows=rows,
            exported_keys=ledger['exported_event_keys'],
            project_owned_boundary=project_owned_boundary,
        )
    event_id = 'science_coordination_history_' + stable_digest({
        'source_hash': source_hash,
        'new_message_ids': [module_chat_message_id(message) for message in new_messages],
        'payoff_class': selected['payoff_class'],
        'scenario_id': selected.get('scenario_id'),
    })[:16]
    selected['science_coordination_history_id'] = event_id
    message = export_science_coordination_history_message(
        selected,
        rows=rows,
        runtime_memory_hash_state=runtime_memory_hash_state,
        project_owned_boundary=project_owned_boundary,
        source_hashes={
            'science_coordination_history_source_hash': source_hash,
            'cycle_strategy_outcome_ledger_hash': cycle_outcome.get('ledger_hash'),
            'cycle_strategy_ledger_hash': cycle_strategy.get('ledger_hash'),
            'strategy_outcome_ledger_hash': strategy_outcome.get('ledger_hash'),
            'strategy_ledger_hash': strategy.get('ledger_hash'),
            'frontier_outcome_ledger_hash': frontier_outcome.get('ledger_hash'),
            'frontier_ledger_hash': frontier.get('ledger_hash'),
            'theory_memory_ledger_hash': theory_memory.get('ledger_hash'),
            'campaign_cycle_memory_ledger_hash': campaign_cycle_memory.get('ledger_hash'),
            'hypothesis_ledger_hash': hypothesis.get('ledger_hash'),
            'campaign_ledger_hash': campaign.get('ledger_hash'),
            'experiment_ledger_hash': experiment.get('ledger_hash'),
            'module_chat_ledger_hash': module_chat.get('ledger_hash'),
            'prior_history_ledger_hash': prior_history.get('ledger_hash'),
        },
    )
    new_ids = [module_chat_message_id(message) for message in new_messages]
    outgoing_id = module_chat_message_id(message) if message else None
    event_key = _event_key(selected)
    if selected['payoff_class'] != 'summarize_noop':
        ledger['exported_event_keys'] = _unique_strings(list(ledger['exported_event_keys']) + [event_key])
    if new_messages:
        ledger['processed_message_ids'] = _unique_strings(list(ledger['processed_message_ids']) + new_ids)
    if source_is_new:
        ledger['processed_source_hashes'] = _unique_strings(list(ledger['processed_source_hashes']) + [source_hash])
    if outgoing_id:
        ledger['outgoing_response_ids'] = _unique_strings(list(ledger['outgoing_response_ids']) + [outgoing_id])
    state_counts = _state_counts(selected)
    latest = {
        'science_coordination_history_id': event_id,
        'new_message_count': len(new_messages),
        'skipped_message_count': len(skipped_messages),
        'source_is_new': source_is_new,
        'state_counts': state_counts,
        'payoff_class': selected['payoff_class'],
        'recommendation_strength': selected['recommendation_strength'],
        'chosen_recipient': message.get('recipient') if message else None,
        'outbox_count': 1 if message else 0,
        'label_leaks': selected.get('label_leaks', []),
        'checkpoint_boundary_state': selected.get('checkpoint_boundary_state'),
        'runtime_memory_hash_state': dict(runtime_memory_hash_state),
        'runtime_memory_mutated': not bool(runtime_memory_hash_state.get('unchanged', True)),
    }
    record = {
        'science_coordination_history_id': event_id,
        'event_hash': stable_digest({'event_id': event_id, 'selected': selected}),
        'processed_message_ids': new_ids,
        'source_hash': source_hash,
        'interaction_sequence': selected.get('interaction_sequence'),
        'involved_modules': selected.get('involved_modules'),
        'source_commits': selected.get('source_commits'),
        'source_tests': selected.get('source_tests'),
        'hypothesis_id': selected.get('hypothesis_id'),
        'campaign_id': selected.get('campaign_id'),
        'family_id': selected.get('family_id'),
        'selected_science_outcome': selected.get('selected_science_outcome'),
        'sibling_evidence_used': selected.get('sibling_evidence_used'),
        'checkpoint_boundary_state': selected.get('checkpoint_boundary_state'),
        'payoff_class': selected['payoff_class'],
        'recommendation_strength': selected['recommendation_strength'],
        'no_overclaiming_proof': selected.get('no_overclaiming_proof'),
        'outgoing_response_id': outgoing_id,
    }
    ledger['history_rows'] = rows
    if new_messages or source_is_new or message is not None:
        ledger['event_records'] = list(ledger['event_records']) + [record]
    ledger['latest'] = latest
    if artifact_path is not None:
        ledger['artifact_path'] = str(Path(artifact_path))
    ledger['ledger_hash'] = stable_digest({key: value for key, value in ledger.items() if key != 'ledger_hash'})
    return ledger, message


def export_science_coordination_history_message(
    selected: dict[str, Any],
    *,
    rows: list[dict[str, Any]],
    runtime_memory_hash_state: dict[str, Any],
    project_owned_boundary: dict[str, Any],
    source_hashes: dict[str, Any],
    topic: str = 'ai_different.science_coordination_history',
) -> dict[str, Any] | None:
    if selected['payoff_class'] == 'summarize_noop':
        return None
    recipient = validate_participant(selected.get('selected_recipient') or 'history')
    row = _find_row(rows, selected.get('scenario_id'), selected.get('campaign_id'), selected.get('hypothesis_id')) or {}
    leak_terms = label_leak_terms({'selected': selected, 'row': row})
    body = {
        'module': 'AI Different',
        'response_kind': 'science_coordination_history',
        'science_coordination_history_id': selected.get('science_coordination_history_id'),
        'interaction_sequence': selected.get('interaction_sequence') or [],
        'involved_modules': selected.get('involved_modules') or [],
        'source_commits': selected.get('source_commits') or [],
        'source_tests': selected.get('source_tests') or [],
        'hypothesis_id': selected.get('hypothesis_id'),
        'campaign_id': selected.get('campaign_id'),
        'family_id': selected.get('family_id'),
        'scenario_id': selected.get('scenario_id'),
        'selected_science_outcome': selected.get('selected_science_outcome'),
        'payoff_class': selected['payoff_class'],
        'recommendation_strength': selected['recommendation_strength'],
        'recommendation_text': selected.get('recommendation_text'),
        'sibling_evidence_used': selected.get('sibling_evidence_used') or [],
        'source_evidence_used': selected.get('source_evidence_used') or [],
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
            'science_coordination_history_id': body['science_coordination_history_id'],
            'payoff_class': body['payoff_class'],
            'recommendation_strength': body['recommendation_strength'],
            'hypothesis_id': body['hypothesis_id'],
            'campaign_id': body['campaign_id'],
            'family_id': body['family_id'],
            'label_clean': body['label_clean'],
            'no_sibling_imports': True,
        },
        tags=['ai_different', 'science_coordination_history', body['payoff_class'], 'label_clean' if body['label_clean'] else 'label_review_needed'],
    )


def write_science_coordination_history_outbox_jsonl(path: str | Path, message: dict[str, Any] | None) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text((json.dumps(message, sort_keys=True) + '\n') if message is not None else '', encoding='utf-8')
    return output


def _valid_kind_or_empty(ledger, kind, record_field):
    if not ledger:
        return {'ledger_kind': kind, record_field: [], 'ledger_hash': None}
    if ledger.get('ledger_kind') != kind:
        raise ValueError(f'plain science coordination history source has wrong ledger_kind: {ledger.get("ledger_kind")}')
    return dict(ledger)


def _valid_plain_or_empty(ledger):
    if not ledger:
        return {}
    if not isinstance(ledger, dict):
        raise ValueError('plain science coordination history source must be a JSON object')
    return dict(ledger)


def _valid_prior_history_or_empty(ledger):
    if not ledger:
        return {}
    if ledger.get('ledger_kind') != SCIENCE_COORDINATION_HISTORY_LEDGER_KIND:
        raise ValueError('prior science coordination history ledger has wrong ledger_kind')
    return validate_science_coordination_history_ledger(ledger)


def _extract_history_rows(existing, cycle_outcome, cycle_strategy, strategy_outcome, strategy, frontier_outcome, frontier, theory_memory, campaign_cycle_memory, hypothesis, campaign, experiment, module_chat, messages):
    rows: dict[str, dict[str, Any]] = {}
    for row in list(existing or []):
        _upsert_row(rows, row)
    for record in list(cycle_outcome.get('outcome_records') or []):
        _upsert_row(rows, _row_from_cycle_outcome(record))
    for row in list(cycle_outcome.get('cycle_strategy_rows') or []):
        _upsert_row(rows, _row_from_generic(row, 'cycle_outcome_row'))
    for record in list(cycle_strategy.get('cycle_strategy_records') or []):
        _upsert_row(rows, _row_from_cycle_strategy(record))
    for record in list(strategy_outcome.get('outcome_records') or []):
        _upsert_row(rows, _row_from_generic(record, 'strategy_outcome_record'))
    for record in list(strategy.get('strategy_records') or []):
        _upsert_row(rows, _row_from_generic(record, 'strategy_record'))
    for record in list(frontier_outcome.get('outcome_records') or []):
        _upsert_row(rows, _row_from_generic(record, 'frontier_outcome_record'))
    for record in list(frontier.get('frontier_records') or []):
        _upsert_row(rows, _row_from_generic(record, 'frontier_record'))
    for row in _plain_rows(theory_memory, ('theory_memory_rows', 'theory_records', 'theories')):
        _upsert_row(rows, _row_from_memory(row, 'theory_memory'))
    for row in _plain_rows(campaign_cycle_memory, ('campaign_cycle_memory_rows', 'cycle_memory_rows')):
        _upsert_row(rows, _row_from_memory(row, 'campaign_cycle_memory'))
    for row in _plain_rows(hypothesis, ('hypothesis_rows', 'hypotheses', 'hypothesis_records')):
        _upsert_row(rows, _row_from_generic(row, 'hypothesis'))
    for row in _plain_rows(campaign, ('campaign_rows', 'campaigns', 'campaign_records')):
        _upsert_row(rows, _row_from_campaign(row))
    for row in _plain_rows(experiment, ('experiment_rows', 'simulation_requests', 'requests')):
        _upsert_row(rows, _row_from_experiment(row))
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
        'selected_science_outcome': source.get('selected_outcome') or source.get('payoff_class') or source.get('selected_strategy'),
        'interaction_sequence': list(source.get('interaction_sequence') or []),
        'involved_modules': list(source.get('involved_modules') or []),
        'source_commits': list(source.get('source_commits') or []),
        'source_tests': list(source.get('source_tests') or []),
        'sibling_evidence_used': list(source.get('sibling_evidence_used') or source.get('observed_sibling_evidence') or []),
        'source_evidence_used': list(source.get('source_evidence_used') or []),
        'checkpoint_boundary_state': source.get('checkpoint_boundary_state') or 'clean',
        'checkpoint_boundary_notes': list(source.get('checkpoint_boundary_notes') or []),
        'rework_noop_count': int(source.get('rework_noop_count', 0) or 0),
        'waiting_state': source.get('waiting_blocker_state') or source.get('waiting_state') or 'resolved',
        'lineage': [],
        'label_leaks': list(source.get('label_leaks') or []),
    }


def _row_from_cycle_outcome(record):
    row = _base_row(record)
    row['lineage'] = ['cycle_strategy_outcome_record']
    row['interaction_sequence'] = row['interaction_sequence'] or ['cycle_strategy', 'cycle_strategy_outcome']
    return row


def _row_from_cycle_strategy(record):
    row = _base_row(record)
    row['selected_science_outcome'] = record.get('selected_strategy')
    row['lineage'] = ['cycle_strategy_record']
    row['interaction_sequence'] = row['interaction_sequence'] or ['cycle_strategy']
    return row


def _row_from_generic(record, lineage):
    row = _base_row(record)
    row['lineage'] = [lineage]
    row['interaction_sequence'] = row['interaction_sequence'] or [lineage]
    return row


def _row_from_memory(row, lineage):
    base = _base_row(row)
    base['selected_science_outcome'] = row.get('selected_outcome') or 'campaign_cycle_memory_promoted'
    base['source_evidence_used'] = [{'source': lineage, 'status': row.get('status') or 'recorded'}]
    base['lineage'] = [lineage]
    base['interaction_sequence'] = [lineage]
    return base


def _row_from_campaign(row):
    base = _base_row(row)
    state = str(row.get('campaign_state') or row.get('cycle_state') or row.get('status') or '')
    base['selected_science_outcome'] = row.get('selected_outcome') or state
    base['lineage'] = ['campaign']
    base['interaction_sequence'] = ['campaign']
    return base


def _row_from_experiment(row):
    base = _base_row(row)
    sender = row.get('sender') or 'code_module'
    gate = row.get('request_type') or row.get('kind') or 'simulation'
    status = row.get('status') or 'received'
    base['sibling_evidence_used'] = [{'sender': sender, 'evidence_gate': gate, 'status': status}]
    base['lineage'] = ['experiment']
    base['interaction_sequence'] = ['science_request', sender]
    return base


def _row_from_message(message):
    body = dict(message.get('body') or {})
    evidence = dict(message.get('evidence') or {})
    sender = message.get('sender')
    scenario_id = body.get('scenario_id') or evidence.get('scenario_id') or body.get('campaign_id') or evidence.get('campaign_id') or body.get('hypothesis_id') or evidence.get('hypothesis_id')
    source = {
        'scenario_id': scenario_id,
        'hypothesis_id': body.get('hypothesis_id') or evidence.get('hypothesis_id'),
        'campaign_id': body.get('campaign_id') or evidence.get('campaign_id'),
        'family_id': body.get('family_id') or evidence.get('family_id'),
        'selected_outcome': body.get('selected_outcome') or body.get('selected_strategy') or body.get('payoff_class') or evidence.get('selected_outcome'),
        'interaction_sequence': body.get('interaction_sequence') or [],
        'involved_modules': body.get('involved_modules') or [],
        'source_commits': body.get('source_commits') or evidence.get('source_commits') or [],
        'source_tests': body.get('source_tests') or evidence.get('source_tests') or [],
        'sibling_evidence_used': body.get('sibling_evidence_used') or body.get('observed_sibling_evidence') or [],
        'source_evidence_used': body.get('source_evidence_used') or [],
        'checkpoint_boundary_state': body.get('checkpoint_boundary_state') or evidence.get('checkpoint_boundary_state') or 'clean',
        'checkpoint_boundary_notes': body.get('checkpoint_boundary_notes') or [],
        'rework_noop_count': body.get('rework_noop_count') or evidence.get('rework_noop_count') or 0,
        'waiting_blocker_state': body.get('waiting_blocker_state') or body.get('waiting_state') or evidence.get('waiting_state') or 'resolved',
        'label_leaks': label_leak_terms({'body': body, 'evidence': evidence}),
    }
    gate = body.get('evidence_gate') or evidence.get('evidence_gate')
    status = str(body.get('status') or evidence.get('status') or '').lower()
    if sender in {'funfun', 'code_module', 'language_model_2', 'history', 'orchestrator'} and (gate or status):
        source['sibling_evidence_used'] = list(source['sibling_evidence_used']) + [{'sender': sender, 'evidence_gate': str(gate or 'advisory'), 'status': status or 'received'}]
    if body.get('third_party_checkpoint_used') or evidence.get('third_party_checkpoint_used'):
        source['checkpoint_boundary_state'] = 'repair'
        source['checkpoint_boundary_notes'] = list(source['checkpoint_boundary_notes']) + ['third-party checkpoint boundary preserved']
    row = _base_row(source)
    row['lineage'] = [f'message:{sender}']
    row['interaction_sequence'] = row['interaction_sequence'] or [f'message:{sender}']
    if sender in {'funfun', 'code_module', 'language_model_2'}:
        row['involved_modules'] = _unique_strings(row['involved_modules'] + [sender])
    return row


def _upsert_row(rows, incoming):
    key = incoming.get('scenario_id') or incoming.get('campaign_id') or incoming.get('hypothesis_id') or stable_digest(incoming)[:12]
    current = rows.setdefault(str(key), {
        'scenario_id': incoming.get('scenario_id'),
        'hypothesis_id': incoming.get('hypothesis_id'),
        'campaign_id': incoming.get('campaign_id'),
        'family_id': incoming.get('family_id'),
        'selected_science_outcome': incoming.get('selected_science_outcome'),
        'interaction_sequence': [],
        'involved_modules': [],
        'source_commits': [],
        'source_tests': [],
        'sibling_evidence_used': [],
        'source_evidence_used': [],
        'checkpoint_boundary_state': 'clean',
        'checkpoint_boundary_notes': [],
        'rework_noop_count': 0,
        'waiting_state': 'resolved',
        'lineage': [],
        'label_leaks': [],
    })
    for field in ('scenario_id', 'hypothesis_id', 'campaign_id', 'family_id'):
        current[field] = current.get(field) or incoming.get(field)
    current['selected_science_outcome'] = _dominant_outcome(current.get('selected_science_outcome'), incoming.get('selected_science_outcome'))
    current['interaction_sequence'] = _unique_strings(list(current.get('interaction_sequence') or []) + list(incoming.get('interaction_sequence') or []))
    current['involved_modules'] = _unique_strings(list(current.get('involved_modules') or []) + list(incoming.get('involved_modules') or []))
    current['source_commits'] = _unique_strings(list(current.get('source_commits') or []) + list(incoming.get('source_commits') or []))
    current['source_tests'] = _unique_strings(list(current.get('source_tests') or []) + list(incoming.get('source_tests') or []))
    current['sibling_evidence_used'] = _unique_dicts(list(current.get('sibling_evidence_used') or []) + list(incoming.get('sibling_evidence_used') or []))
    current['source_evidence_used'] = _unique_dicts(list(current.get('source_evidence_used') or []) + list(incoming.get('source_evidence_used') or []))
    current['checkpoint_boundary_state'] = 'repair' if current.get('checkpoint_boundary_state') == 'repair' or incoming.get('checkpoint_boundary_state') == 'repair' else 'clean'
    current['checkpoint_boundary_notes'] = _unique_strings(list(current.get('checkpoint_boundary_notes') or []) + list(incoming.get('checkpoint_boundary_notes') or []))
    current['rework_noop_count'] = max(int(current.get('rework_noop_count', 0) or 0), int(incoming.get('rework_noop_count', 0) or 0))
    current['waiting_state'] = _dominant_waiting(current.get('waiting_state'), incoming.get('waiting_state'))
    current['lineage'] = _unique_strings(list(current.get('lineage') or []) + list(incoming.get('lineage') or []))
    current['label_leaks'] = _unique_strings(list(current.get('label_leaks') or []) + list(incoming.get('label_leaks') or []))


def _finalize_row(row):
    if row.get('label_leaks'):
        row['checkpoint_boundary_state'] = 'repair'
    if not row.get('involved_modules'):
        row['involved_modules'] = _infer_modules(row)
    row['history_row_hash'] = stable_digest({
        'scenario_id': row.get('scenario_id'),
        'outcome': row.get('selected_science_outcome'),
        'modules': row.get('involved_modules'),
        'boundary': row.get('checkpoint_boundary_state'),
    })
    return row


def _select_payoff(*, rows, exported_keys, project_owned_boundary):
    for row in rows:
        if project_owned_boundary.get('third_party_checkpoint_used') or row.get('checkpoint_boundary_state') == 'repair':
            return _event(row, 'preserve_checkpoint_boundary', 'strong')
    for row in rows:
        if row.get('selected_science_outcome') == 'campaign_cycle_memory_promoted':
            return _event(row, 'validated_hypothesis_cycle_memory_payoff', 'strong')
    for row in rows:
        if row.get('selected_science_outcome') == 'code_simulation_or_primitive_capability_help_received' or _has_module(row, 'code_module', {'simulation', 'primitive', 'counterexample'}):
            return _event(row, 'code_simulation_before_science_campaign_paid_off', 'moderate')
    for row in rows:
        if row.get('selected_science_outcome') == 'funfun_formal_or_proof_help_received' or _has_module(row, 'funfun', {'formal', 'proof', 'certificate'}):
            return _event(row, 'funfun_formalization_before_science_campaign_paid_off', 'moderate')
    for row in rows:
        if row.get('selected_science_outcome') == 'language_terminology_or_protocol_help_received' or _has_module(row, 'language_model_2', {'terminology', 'protocol', 'clarification'}):
            return _event(row, 'language_terminology_before_science_campaign_paid_off', 'moderate')
    for row in rows:
        if row.get('selected_science_outcome') == 'planned_science_campaign_cycle_strategy_waiting_for_evidence' or row.get('waiting_state') == 'waiting':
            return _event(row, 'sibling_request_waiting_for_evidence', 'weak')
    for row in rows:
        if row.get('rework_noop_count', 0) >= 2 or row.get('selected_science_outcome') == 'summarize_noop':
            return _event(row, 'repeated_no_gain_or_noop_loop', 'weak')
    for row in rows:
        if row.get('selected_science_outcome') == 'theory_repair_cycle_reopened':
            return _event(row, 'theory_repair_loop_helped', 'moderate')
    for row in rows:
        if row.get('selected_science_outcome') == 'stable_science_campaign_cycle_closed':
            return _event(row, 'stable_science_campaign_cycle_closed', 'moderate')
    for row in rows:
        if row.get('selected_science_outcome') == 'next_frontier_hypothesis_campaign_scheduled':
            return _event(row, 'next_frontier_hypothesis_campaign_helped', 'weak')
    for row in rows:
        if row.get('selected_science_outcome') == 'no_measurable_science_campaign_cycle_strategy_gain':
            return _event(row, 'no_measurable_coordination_payoff', 'weak')
    return _noop_event('no science coordination payoff selected')


def _event(row, payoff_class, strength):
    return {
        'payoff_class': payoff_class,
        'selected_recipient': 'history',
        'recommendation_strength': strength,
        'recommendation_text': _recommendation_text(payoff_class, strength),
        'scenario_id': row.get('scenario_id'),
        'hypothesis_id': row.get('hypothesis_id'),
        'campaign_id': row.get('campaign_id'),
        'family_id': row.get('family_id'),
        'selected_science_outcome': row.get('selected_science_outcome'),
        'interaction_sequence': list(row.get('interaction_sequence') or []),
        'involved_modules': _unique_strings(['ai_different'] + list(row.get('involved_modules') or [])),
        'source_commits': list(row.get('source_commits') or []),
        'source_tests': list(row.get('source_tests') or []),
        'sibling_evidence_used': list(row.get('sibling_evidence_used') or []),
        'source_evidence_used': list(row.get('source_evidence_used') or []),
        'checkpoint_boundary_state': row.get('checkpoint_boundary_state') or 'clean',
        'checkpoint_boundary_notes': list(row.get('checkpoint_boundary_notes') or []),
        'no_overclaiming_proof': 'candidate coordination knowledge only; no causal proof and no local-owned checkpoint claim unless status capsule verifies it',
        'label_leaks': list(row.get('label_leaks') or []),
    }


def _noop_event(reason):
    return {
        'payoff_class': 'summarize_noop',
        'selected_recipient': None,
        'recommendation_strength': 'none',
        'recommendation_text': reason,
        'scenario_id': None,
        'hypothesis_id': None,
        'campaign_id': None,
        'family_id': None,
        'selected_science_outcome': None,
        'interaction_sequence': [],
        'involved_modules': [],
        'source_commits': [],
        'source_tests': [],
        'sibling_evidence_used': [],
        'source_evidence_used': [],
        'checkpoint_boundary_state': 'clean',
        'checkpoint_boundary_notes': [],
        'no_overclaiming_proof': 'no project-owned checkpoint claim made',
        'label_leaks': [],
    }


def _recommendation_text(payoff_class, strength):
    return (
        f'Candidate organizational memory ({strength}): {payoff_class}; '
        'treat as coordination evidence to try again, not causal proof.'
    )


def _state_counts(selected):
    counts = {
        'payoff': 0,
        'no_gain': 0,
        'waiting': 0,
        'boundary': 0,
        'rework': 0,
        'sibling': 0,
        'request': 0,
    }
    payoff = selected.get('payoff_class')
    if payoff == 'preserve_checkpoint_boundary':
        counts['boundary'] += 1
    elif payoff == 'sibling_request_waiting_for_evidence':
        counts['waiting'] += 1
        counts['request'] += 1
    elif payoff == 'repeated_no_gain_or_noop_loop':
        counts['rework'] += 1
    elif payoff in {'no_measurable_coordination_payoff', 'summarize_noop'}:
        counts['no_gain'] += 1
    elif payoff in {
        'code_simulation_before_science_campaign_paid_off',
        'funfun_formalization_before_science_campaign_paid_off',
        'language_terminology_before_science_campaign_paid_off',
    }:
        counts['payoff'] += 1
        counts['sibling'] += 1
    else:
        counts['payoff'] += 1
    return counts


def _dominant_outcome(current, incoming):
    order = {
        'preserve_checkpoint_boundary': 12,
        'campaign_cycle_memory_promoted': 11,
        'code_simulation_or_primitive_capability_help_received': 10,
        'funfun_formal_or_proof_help_received': 9,
        'language_terminology_or_protocol_help_received': 8,
        'planned_science_campaign_cycle_strategy_waiting_for_evidence': 7,
        'summarize_noop': 6,
        'theory_repair_cycle_reopened': 5,
        'stable_science_campaign_cycle_closed': 4,
        'next_frontier_hypothesis_campaign_scheduled': 3,
        'no_measurable_science_campaign_cycle_strategy_gain': 2,
        None: 0,
    }
    return current if order.get(current, 0) >= order.get(incoming, 0) else incoming


def _dominant_waiting(current, incoming):
    order = {'blocked': 3, 'waiting': 2, 'resolved': 1, None: 0}
    return current if order.get(current, 0) >= order.get(incoming, 0) else incoming


def _has_module(row, module, terms):
    for item in row.get('sibling_evidence_used') or []:
        if item.get('sender') != module:
            continue
        gate = str(item.get('evidence_gate') or '').lower()
        status = str(item.get('status') or '').lower()
        if any(term in gate for term in terms) and status not in {'', 'waiting', 'missing'}:
            return True
    return False


def _infer_modules(row):
    modules = []
    for item in row.get('sibling_evidence_used') or []:
        sender = item.get('sender')
        if sender:
            modules.append(str(sender))
    return _unique_strings(modules)


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


def _event_key(selected):
    return stable_digest({
        'scenario_id': selected.get('scenario_id'),
        'hypothesis_id': selected.get('hypothesis_id'),
        'campaign_id': selected.get('campaign_id'),
        'family_id': selected.get('family_id'),
        'payoff_class': selected.get('payoff_class'),
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


def _unique_dicts(values):
    seen = set()
    output = []
    for value in values:
        if not isinstance(value, dict):
            continue
        marker = stable_digest(value)
        if marker not in seen:
            output.append(dict(value))
            seen.add(marker)
    return output
