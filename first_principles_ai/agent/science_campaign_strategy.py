"""Science campaign strategy planner.

This layer turns assessed theory-frontier outcomes into one durable next
science-campaign strategy. It is symbolic orchestration only: no empirical
discovery claim, no sibling project imports, and no model/checkpoint overclaim.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .campaign_planner import CAMPAIGN_LEDGER_KIND
from .module_chat_adapter import (
    build_module_chat_message,
    label_leak_terms,
    module_chat_message_id,
    stable_digest,
    validate_module_chat_message,
    validate_participant,
)
from .science_theory_frontier import SCIENCE_THEORY_FRONTIER_LEDGER_KIND
from .science_theory_frontier_outcome import SCIENCE_THEORY_FRONTIER_OUTCOME_LEDGER_KIND


SCIENCE_CAMPAIGN_STRATEGY_LEDGER_KIND = 'ai_different.science_campaign_strategy_ledger'


def empty_science_campaign_strategy_ledger() -> dict[str, Any]:
    return {
        'schema_version': 1,
        'ledger_kind': SCIENCE_CAMPAIGN_STRATEGY_LEDGER_KIND,
        'processed_message_ids': [],
        'processed_source_hashes': [],
        'planned_strategy_keys': [],
        'strategy_records': [],
        'strategy_rows': [],
        'outgoing_response_ids': [],
        'latest': {},
    }


def load_science_campaign_strategy_ledger(path: str | Path | None) -> dict[str, Any]:
    if not path or not Path(path).exists():
        return empty_science_campaign_strategy_ledger()
    with Path(path).open('r', encoding='utf-8') as handle:
        ledger = json.load(handle)
    return validate_science_campaign_strategy_ledger(ledger)


def write_science_campaign_strategy_ledger(path: str | Path, ledger: dict[str, Any]) -> Path:
    validated = validate_science_campaign_strategy_ledger(ledger)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(validated, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return output


def validate_science_campaign_strategy_ledger(ledger: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(ledger, dict):
        raise ValueError('science campaign strategy ledger must be a JSON object')
    if ledger.get('ledger_kind') != SCIENCE_CAMPAIGN_STRATEGY_LEDGER_KIND:
        raise ValueError('science campaign strategy ledger has wrong ledger_kind')
    for field in (
        'processed_message_ids',
        'processed_source_hashes',
        'planned_strategy_keys',
        'outgoing_response_ids',
    ):
        values = ledger.get(field)
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            raise ValueError(f'{field} must be strings')
    for field in ('strategy_records', 'strategy_rows'):
        values = ledger.get(field)
        if not isinstance(values, list) or not all(isinstance(item, dict) for item in values):
            raise ValueError(f'{field} must be objects')
    latest = ledger.get('latest', {})
    if not isinstance(latest, dict):
        raise ValueError('science campaign strategy latest must be an object')
    validated = {
        'schema_version': int(ledger.get('schema_version', 1) or 1),
        'ledger_kind': SCIENCE_CAMPAIGN_STRATEGY_LEDGER_KIND,
        'processed_message_ids': _unique_strings(ledger['processed_message_ids']),
        'processed_source_hashes': _unique_strings(ledger['processed_source_hashes']),
        'planned_strategy_keys': _unique_strings(ledger['planned_strategy_keys']),
        'strategy_records': list(ledger['strategy_records']),
        'strategy_rows': list(ledger['strategy_rows']),
        'outgoing_response_ids': _unique_strings(ledger['outgoing_response_ids']),
        'latest': dict(latest),
    }
    if ledger.get('ledger_hash'):
        validated['ledger_hash'] = str(ledger['ledger_hash'])
    if ledger.get('artifact_path'):
        validated['artifact_path'] = str(ledger['artifact_path'])
    return validated


def read_science_campaign_strategy_transcript(path: str | Path | None) -> dict[str, Any]:
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
        raise ValueError('plain science campaign strategy input must be a JSON object')
    return value


def build_science_campaign_strategy_plan(
    *,
    transcript_messages: list[dict[str, Any]],
    strategy_ledger: dict[str, Any],
    frontier_outcome_ledger: dict[str, Any] | None = None,
    frontier_ledger: dict[str, Any] | None = None,
    campaign_ledger: dict[str, Any] | None = None,
    theory_memory_ledger: dict[str, Any] | None = None,
    hypothesis_ledger: dict[str, Any] | None = None,
    experiment_ledger: dict[str, Any] | None = None,
    module_chat_ledger: dict[str, Any] | None = None,
    prior_strategy_ledger: dict[str, Any] | None = None,
    runtime_memory_data: dict[str, Any] | None = None,
    runtime_memory_hash_state: dict[str, Any],
    project_owned_boundary: dict[str, Any],
    artifact_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    ledger = validate_science_campaign_strategy_ledger(strategy_ledger)
    frontier_outcome = _valid_frontier_outcome_or_empty(frontier_outcome_ledger or {})
    frontier = _valid_frontier_or_empty(frontier_ledger or {})
    campaign = _valid_campaign_or_empty(campaign_ledger or {})
    theory_memory = _valid_plain_or_empty(theory_memory_ledger or {})
    hypothesis = _valid_plain_or_empty(hypothesis_ledger or {})
    experiment = _valid_plain_or_empty(experiment_ledger or {})
    module_chat = _valid_plain_or_empty(module_chat_ledger or {})
    prior_strategy = _valid_prior_strategy_or_empty(prior_strategy_ledger or {})
    runtime_memory = dict(runtime_memory_data or {})
    source_hash = _source_hash(
        frontier_outcome,
        frontier,
        campaign,
        theory_memory,
        hypothesis,
        experiment,
        module_chat,
        prior_strategy,
        runtime_memory,
    )
    source_is_new = source_hash not in set(ledger['processed_source_hashes'])
    processed = set(ledger['processed_message_ids'])
    new_messages = [
        message for message in transcript_messages
        if module_chat_message_id(message) not in processed
    ]
    skipped_messages = [
        message for message in transcript_messages
        if module_chat_message_id(message) in processed
    ]
    rows = _extract_strategy_rows(
        ledger['strategy_rows'],
        frontier_outcome,
        frontier,
        campaign,
        theory_memory,
        hypothesis,
        experiment,
        module_chat,
        transcript_messages,
    )
    if not new_messages and not source_is_new:
        selected = _noop_strategy('no new science campaign strategy evidence or source ledger state')
    else:
        selected = _select_strategy(
            rows=rows,
            planned_keys=ledger['planned_strategy_keys'],
            project_owned_boundary=project_owned_boundary,
        )
    strategy_id = 'science_strategy_' + stable_digest({
        'source_hash': source_hash,
        'new_message_ids': [module_chat_message_id(message) for message in new_messages],
        'strategy': selected['selected_strategy'],
        'frontier_outcome_id': selected.get('frontier_outcome_id'),
        'scenario_id': selected.get('scenario_id'),
    })[:16]
    selected['strategy_id'] = strategy_id
    message = export_science_campaign_strategy_message(
        selected,
        rows=rows,
        runtime_memory_hash_state=runtime_memory_hash_state,
        project_owned_boundary=project_owned_boundary,
        source_hashes={
            'science_campaign_strategy_source_hash': source_hash,
            'frontier_outcome_ledger_hash': frontier_outcome.get('ledger_hash'),
            'frontier_ledger_hash': frontier.get('ledger_hash'),
            'campaign_ledger_hash': campaign.get('ledger_hash'),
            'theory_memory_ledger_hash': theory_memory.get('ledger_hash'),
            'hypothesis_ledger_hash': hypothesis.get('ledger_hash'),
            'experiment_ledger_hash': experiment.get('ledger_hash'),
            'module_chat_ledger_hash': module_chat.get('ledger_hash'),
            'prior_strategy_ledger_hash': prior_strategy.get('ledger_hash'),
        },
    )
    new_ids = [module_chat_message_id(message) for message in new_messages]
    outgoing_id = module_chat_message_id(message) if message else None
    strategy_key = _strategy_key(selected)
    if selected['selected_strategy'] != 'summarize_noop':
        ledger['planned_strategy_keys'] = _unique_strings(list(ledger['planned_strategy_keys']) + [strategy_key])
    if new_messages:
        ledger['processed_message_ids'] = _unique_strings(list(ledger['processed_message_ids']) + new_ids)
    if source_is_new:
        ledger['processed_source_hashes'] = _unique_strings(list(ledger['processed_source_hashes']) + [source_hash])
    if outgoing_id:
        ledger['outgoing_response_ids'] = _unique_strings(list(ledger['outgoing_response_ids']) + [outgoing_id])
    state_counts = _state_counts(selected)
    latest = {
        'strategy_id': strategy_id,
        'new_message_count': len(new_messages),
        'skipped_message_count': len(skipped_messages),
        'source_is_new': source_is_new,
        'state_counts': state_counts,
        'selected_strategy': selected['selected_strategy'],
        'selected_action': selected['selected_action'],
        'chosen_recipient': message.get('recipient') if message else None,
        'outbox_count': 1 if message else 0,
        'label_leaks': selected.get('label_leaks', []),
        'checkpoint_boundary_state': selected.get('checkpoint_boundary_state'),
        'runtime_memory_hash_state': dict(runtime_memory_hash_state),
        'runtime_memory_mutated': not bool(runtime_memory_hash_state.get('unchanged', True)),
    }
    record = {
        'strategy_id': strategy_id,
        'strategy_hash': stable_digest({'strategy_id': strategy_id, 'selected': selected}),
        'processed_message_ids': new_ids,
        'source_hash': source_hash,
        'frontier_outcome_id': selected.get('frontier_outcome_id'),
        'frontier_id': selected.get('frontier_id'),
        'hypothesis_id': selected.get('hypothesis_id'),
        'campaign_id': selected.get('campaign_id'),
        'selected_strategy': selected['selected_strategy'],
        'selected_action': selected['selected_action'],
        'selected_recipient': latest['chosen_recipient'],
        'promotion_state': selected.get('promotion_state'),
        'retirement_state': selected.get('retirement_state'),
        'repair_state': selected.get('repair_state'),
        'closure_state': selected.get('closure_state'),
        'requested_sibling_evidence': selected.get('requested_sibling_evidence'),
        'before_theory_memory_state': selected.get('before_theory_memory_state'),
        'after_theory_memory_state': selected.get('after_theory_memory_state'),
        'checkpoint_boundary_notes': selected.get('checkpoint_boundary_notes'),
        'no_overclaiming_proof': selected.get('no_overclaiming_proof'),
        'outgoing_response_id': outgoing_id,
    }
    ledger['strategy_rows'] = rows
    if new_messages or source_is_new or message is not None:
        ledger['strategy_records'] = list(ledger['strategy_records']) + [record]
    ledger['latest'] = latest
    if artifact_path is not None:
        ledger['artifact_path'] = str(Path(artifact_path))
    ledger['ledger_hash'] = stable_digest({key: value for key, value in ledger.items() if key != 'ledger_hash'})
    return ledger, message


def export_science_campaign_strategy_message(
    selected: dict[str, Any],
    *,
    rows: list[dict[str, Any]],
    runtime_memory_hash_state: dict[str, Any],
    project_owned_boundary: dict[str, Any],
    source_hashes: dict[str, Any],
    topic: str = 'ai_different.science_campaign_strategy',
) -> dict[str, Any] | None:
    if selected['selected_strategy'] == 'summarize_noop':
        return None
    recipient = validate_participant(selected.get('selected_recipient') or 'orchestrator')
    row = _find_row(rows, selected.get('frontier_outcome_id'), selected.get('scenario_id')) or {}
    leak_terms = label_leak_terms({'selected': selected, 'row': row})
    body = {
        'module': 'AI Different',
        'response_kind': 'science_campaign_strategy',
        'strategy_id': selected.get('strategy_id'),
        'frontier_outcome_id': selected.get('frontier_outcome_id'),
        'frontier_id': selected.get('frontier_id'),
        'source_outcome_id': selected.get('source_outcome_id'),
        'hypothesis_id': selected.get('hypothesis_id'),
        'campaign_id': selected.get('campaign_id'),
        'scenario_id': selected.get('scenario_id'),
        'selected_strategy': selected['selected_strategy'],
        'selected_action': selected['selected_action'],
        'selected_recipient': recipient,
        'target_hypothesis': selected.get('target_hypothesis'),
        'target_campaign': selected.get('target_campaign'),
        'required_followup_evidence': selected.get('required_followup_evidence') or [],
        'source_evidence_used': selected.get('source_evidence_used') or [],
        'sibling_evidence_used': selected.get('sibling_evidence_used') or [],
        'theory_memory_delta': selected.get('theory_memory_delta'),
        'promotion_state': selected.get('promotion_state'),
        'retirement_state': selected.get('retirement_state'),
        'repair_state': selected.get('repair_state'),
        'closure_state': selected.get('closure_state'),
        'before_theory_memory_state': selected.get('before_theory_memory_state'),
        'after_theory_memory_state': selected.get('after_theory_memory_state'),
        'checkpoint_boundary_notes': selected.get('checkpoint_boundary_notes') or [],
        'no_overclaiming_proof': selected.get('no_overclaiming_proof'),
        'source_ledger_hashes': dict(source_hashes),
        'runtime_memory_hash_state': dict(runtime_memory_hash_state),
        'runtime_memory_mutated': not bool(runtime_memory_hash_state.get('unchanged', True)),
        'project_owned_boundary': dict(project_owned_boundary),
        'third_party_checkpoint_used': bool(project_owned_boundary.get('third_party_checkpoint_used')),
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
            'strategy_id': body['strategy_id'],
            'frontier_outcome_id': body['frontier_outcome_id'],
            'hypothesis_id': body['hypothesis_id'],
            'campaign_id': body['campaign_id'],
            'selected_strategy': body['selected_strategy'],
            'selected_action': body['selected_action'],
            'label_clean': body['label_clean'],
            'no_sibling_imports': True,
        },
        tags=['ai_different', 'science_campaign_strategy', body['selected_strategy'], 'label_clean' if body['label_clean'] else 'label_review_needed'],
    )


def write_science_campaign_strategy_outbox_jsonl(path: str | Path, message: dict[str, Any] | None) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text((json.dumps(message, sort_keys=True) + '\n') if message is not None else '', encoding='utf-8')
    return output


def _valid_frontier_outcome_or_empty(ledger):
    if not ledger:
        return {'ledger_kind': SCIENCE_THEORY_FRONTIER_OUTCOME_LEDGER_KIND, 'outcome_records': [], 'frontier_rows': [], 'ledger_hash': None}
    if ledger.get('ledger_kind') != SCIENCE_THEORY_FRONTIER_OUTCOME_LEDGER_KIND:
        raise ValueError('science theory frontier outcome ledger has wrong ledger_kind')
    return dict(ledger)


def _valid_frontier_or_empty(ledger):
    if not ledger:
        return {'ledger_kind': SCIENCE_THEORY_FRONTIER_LEDGER_KIND, 'frontier_records': [], 'frontier_rows': [], 'ledger_hash': None}
    if ledger.get('ledger_kind') != SCIENCE_THEORY_FRONTIER_LEDGER_KIND:
        raise ValueError('science theory frontier ledger has wrong ledger_kind')
    return dict(ledger)


def _valid_campaign_or_empty(ledger):
    if not ledger:
        return {'ledger_kind': CAMPAIGN_LEDGER_KIND, 'campaigns': [], 'campaign_rows': [], 'ledger_hash': None}
    if ledger.get('ledger_kind') != CAMPAIGN_LEDGER_KIND:
        raise ValueError('campaign ledger has wrong ledger_kind')
    return dict(ledger)


def _valid_plain_or_empty(ledger):
    if not ledger:
        return {}
    if not isinstance(ledger, dict):
        raise ValueError('plain science campaign strategy ledger must be a JSON object')
    return dict(ledger)


def _valid_prior_strategy_or_empty(ledger):
    if not ledger:
        return {}
    if ledger.get('ledger_kind') != SCIENCE_CAMPAIGN_STRATEGY_LEDGER_KIND:
        raise ValueError('prior science campaign strategy ledger has wrong ledger_kind')
    return validate_science_campaign_strategy_ledger(ledger)


def _extract_strategy_rows(existing, frontier_outcome, frontier, campaign, theory_memory, hypothesis, experiment, module_chat, messages):
    rows: dict[str, dict[str, Any]] = {}
    for row in list(existing or []):
        _upsert_row(rows, row)
    for record in list(frontier_outcome.get('outcome_records') or []):
        _upsert_row(rows, _row_from_frontier_outcome_record(record))
    for row in list(frontier_outcome.get('frontier_rows') or []):
        _upsert_row(rows, _row_from_frontier_outcome_row(row))
    for record in list(frontier.get('frontier_records') or []):
        _upsert_row(rows, _row_from_frontier_record(record))
    for row in _plain_rows(campaign, ('campaigns', 'campaign_rows', 'campaign_records')):
        _upsert_row(rows, _row_from_campaign(row))
    for row in _plain_rows(theory_memory, ('theory_memory_rows', 'theory_records', 'promoted_hypotheses', 'theories')):
        _upsert_row(rows, _row_from_theory_memory(row))
    for row in _plain_rows(hypothesis, ('hypothesis_rows', 'hypotheses', 'hypothesis_records')):
        _upsert_row(rows, _row_from_hypothesis(row))
    for row in _plain_rows(experiment, ('experiment_rows', 'simulation_requests', 'requests')):
        _upsert_row(rows, _row_from_experiment(row))
    for row in _plain_rows(module_chat, ('messages', 'records', 'module_chat_rows')):
        if isinstance(row, dict) and {'sender', 'recipient', 'topic', 'body', 'evidence'}.issubset(row):
            _upsert_row(rows, _row_from_message(row))
    for message in messages:
        _upsert_row(rows, _row_from_message(message))
    return [_finalize_row(row) for row in rows.values()]


def _row_from_frontier_outcome_record(record):
    return {
        'frontier_outcome_id': record.get('frontier_outcome_id'),
        'frontier_id': record.get('frontier_id') or _first(record.get('frontier_ids') or []),
        'source_outcome_id': record.get('source_outcome_id') or _first(record.get('source_outcome_ids') or []),
        'hypothesis_id': record.get('hypothesis_id') or _first(record.get('hypothesis_ids') or []),
        'campaign_id': record.get('campaign_id') or _first(record.get('campaign_ids') or []),
        'scenario_id': record.get('scenario_id') or _first(record.get('scenario_ids') or []),
        'selected_outcome': record.get('selected_outcome'),
        'planned_theory_move': record.get('planned_theory_move'),
        'source_evidence_used': list(record.get('observed_theory_evidence') or []),
        'sibling_evidence_used': list(record.get('observed_sibling_evidence') or []),
        'promotion_state': record.get('promotion_state') or 'none',
        'retirement_state': record.get('retirement_block_state') or 'none',
        'repair_state': 'open' if record.get('boundary_checkpoint_state') == 'repair' else 'none',
        'closure_state': record.get('closure_state') or 'none',
        'before_theory_memory_state': record.get('theory_memory_state') or 'unknown',
        'after_theory_memory_state': record.get('theory_memory_state') or 'unknown',
        'checkpoint_boundary_state': record.get('boundary_checkpoint_state') or 'clean',
        'checkpoint_boundary_notes': list(record.get('boundary_notes') or []),
        'label_leaks': [],
        'lineage': ['frontier_outcome_record'],
    }


def _row_from_frontier_outcome_row(row):
    return {
        'frontier_outcome_id': row.get('frontier_outcome_id'),
        'frontier_id': row.get('frontier_id'),
        'source_outcome_id': row.get('source_outcome_id'),
        'hypothesis_id': row.get('hypothesis_id'),
        'campaign_id': row.get('campaign_id'),
        'scenario_id': row.get('scenario_id'),
        'selected_outcome': row.get('selected_outcome'),
        'planned_theory_move': row.get('planned_theory_move'),
        'source_evidence_used': list(row.get('observed_theory_evidence') or row.get('source_evidence_used') or []),
        'sibling_evidence_used': list(row.get('observed_sibling_evidence') or row.get('sibling_evidence_used') or []),
        'promotion_state': row.get('promotion_state') or 'none',
        'retirement_state': row.get('retirement_block_state') or row.get('retirement_state') or 'none',
        'repair_state': row.get('repair_state') or ('open' if row.get('boundary_checkpoint_state') == 'repair' else 'none'),
        'closure_state': row.get('closure_state') or 'none',
        'before_theory_memory_state': row.get('before_theory_memory_state') or row.get('theory_memory_state') or 'unknown',
        'after_theory_memory_state': row.get('after_theory_memory_state') or row.get('theory_memory_state') or 'unknown',
        'checkpoint_boundary_state': row.get('boundary_checkpoint_state') or row.get('checkpoint_boundary_state') or 'clean',
        'checkpoint_boundary_notes': list(row.get('boundary_notes') or row.get('checkpoint_boundary_notes') or []),
        'label_leaks': list(row.get('label_leaks') or []),
        'lineage': ['frontier_outcome_row'],
    }


def _row_from_frontier_record(record):
    return {
        'frontier_outcome_id': None,
        'frontier_id': record.get('frontier_id'),
        'source_outcome_id': _first(record.get('outcome_ids') or []),
        'hypothesis_id': _first(record.get('hypothesis_ids') or []),
        'campaign_id': _first(record.get('campaign_ids') or []),
        'scenario_id': _first(record.get('scenario_ids') or []),
        'selected_outcome': None,
        'planned_theory_move': record.get('selected_theory_move'),
        'source_evidence_used': [],
        'sibling_evidence_used': list(record.get('observed_sibling_evidence') or []),
        'promotion_state': 'none',
        'retirement_state': record.get('retirement_block_state') or 'none',
        'repair_state': 'open' if record.get('boundary_checkpoint_state') == 'repair' else 'none',
        'closure_state': 'none',
        'before_theory_memory_state': 'unknown',
        'after_theory_memory_state': 'unknown',
        'checkpoint_boundary_state': record.get('boundary_checkpoint_state') or 'clean',
        'checkpoint_boundary_notes': list(record.get('boundary_notes') or []),
        'label_leaks': [],
        'lineage': ['frontier_record'],
    }


def _row_from_campaign(row):
    return {
        'campaign_id': row.get('campaign_id') or row.get('id'),
        'scenario_id': row.get('scenario_id') or row.get('campaign_id'),
        'hypothesis_id': row.get('hypothesis_id'),
        'selected_outcome': row.get('selected_outcome') or row.get('campaign_state'),
        'planned_theory_move': row.get('planned_theory_move'),
        'source_evidence_used': [],
        'sibling_evidence_used': [],
        'promotion_state': 'none',
        'retirement_state': row.get('retirement_state') or 'none',
        'repair_state': row.get('repair_state') or 'none',
        'closure_state': row.get('closure_state') or ('stable' if row.get('campaign_state') == 'stable' else 'none'),
        'before_theory_memory_state': row.get('before_theory_memory_state') or 'unknown',
        'after_theory_memory_state': row.get('after_theory_memory_state') or 'unknown',
        'checkpoint_boundary_state': row.get('checkpoint_boundary_state') or 'clean',
        'checkpoint_boundary_notes': list(row.get('checkpoint_boundary_notes') or []),
        'label_leaks': list(row.get('label_leaks') or []),
        'lineage': ['campaign'],
    }


def _row_from_theory_memory(row):
    return {
        'frontier_outcome_id': row.get('frontier_outcome_id'),
        'frontier_id': row.get('frontier_id'),
        'source_outcome_id': row.get('source_outcome_id') or row.get('outcome_id'),
        'hypothesis_id': row.get('hypothesis_id'),
        'campaign_id': row.get('campaign_id'),
        'scenario_id': row.get('scenario_id') or row.get('hypothesis_id'),
        'selected_outcome': 'theory_memory_recorded_or_hypothesis_promoted',
        'planned_theory_move': row.get('planned_theory_move') or 'promote_supported_hypothesis_to_theory_memory',
        'source_evidence_used': [{'source': row.get('source') or 'theory_memory', 'status': row.get('status') or 'recorded'}],
        'sibling_evidence_used': [],
        'promotion_state': 'promoted' if row.get('status') in {None, 'recorded', 'promoted', 'accepted'} else 'none',
        'retirement_state': 'none',
        'repair_state': 'none',
        'closure_state': row.get('closure_state') or 'none',
        'before_theory_memory_state': row.get('before_theory_memory_state') or 'not_recorded',
        'after_theory_memory_state': row.get('after_theory_memory_state') or row.get('status') or 'recorded',
        'checkpoint_boundary_state': row.get('checkpoint_boundary_state') or 'clean',
        'checkpoint_boundary_notes': list(row.get('checkpoint_boundary_notes') or []),
        'label_leaks': list(row.get('label_leaks') or []),
        'lineage': ['theory_memory'],
    }


def _row_from_hypothesis(row):
    state = str(row.get('state') or row.get('status') or '')
    return {
        'hypothesis_id': row.get('hypothesis_id') or row.get('id'),
        'campaign_id': row.get('campaign_id'),
        'scenario_id': row.get('scenario_id') or row.get('hypothesis_id') or row.get('id'),
        'selected_outcome': row.get('selected_outcome') or state,
        'planned_theory_move': row.get('planned_theory_move'),
        'source_evidence_used': [],
        'sibling_evidence_used': [],
        'promotion_state': 'promoted' if state in {'validated', 'promoted'} else 'none',
        'retirement_state': 'retired' if state in {'retired', 'weak'} else 'none',
        'repair_state': 'open' if state == 'repair' else 'none',
        'closure_state': 'none',
        'before_theory_memory_state': 'unknown',
        'after_theory_memory_state': 'unknown',
        'checkpoint_boundary_state': row.get('checkpoint_boundary_state') or 'clean',
        'checkpoint_boundary_notes': list(row.get('checkpoint_boundary_notes') or []),
        'label_leaks': list(row.get('label_leaks') or []),
        'lineage': ['hypothesis'],
    }


def _row_from_experiment(row):
    request_type = str(row.get('request_type') or row.get('kind') or '')
    planned_move = row.get('planned_theory_move')
    if 'code' in request_type or 'simulation' in request_type or 'primitive' in request_type:
        planned_move = planned_move or 'request_code_experiment_or_counterexample'
    return {
        'hypothesis_id': row.get('hypothesis_id'),
        'campaign_id': row.get('campaign_id'),
        'scenario_id': row.get('scenario_id') or row.get('request_id'),
        'selected_outcome': row.get('selected_outcome'),
        'planned_theory_move': planned_move,
        'source_evidence_used': [],
        'sibling_evidence_used': [],
        'promotion_state': 'none',
        'retirement_state': 'none',
        'repair_state': 'none',
        'closure_state': 'none',
        'before_theory_memory_state': 'unknown',
        'after_theory_memory_state': 'unknown',
        'checkpoint_boundary_state': row.get('checkpoint_boundary_state') or 'clean',
        'checkpoint_boundary_notes': list(row.get('checkpoint_boundary_notes') or []),
        'label_leaks': list(row.get('label_leaks') or []),
        'lineage': ['experiment'],
    }


def _row_from_message(message):
    body = dict(message.get('body') or {})
    evidence = dict(message.get('evidence') or {})
    sender = message.get('sender')
    scenario_id = (
        body.get('scenario_id')
        or evidence.get('scenario_id')
        or body.get('campaign_id')
        or evidence.get('campaign_id')
        or body.get('hypothesis_id')
        or evidence.get('hypothesis_id')
    )
    selected_outcome = body.get('selected_outcome') or evidence.get('selected_outcome')
    selected_strategy = body.get('selected_strategy') or evidence.get('selected_strategy')
    planned_move = (
        body.get('planned_theory_move')
        or body.get('selected_theory_move')
        or evidence.get('selected_theory_move')
    )
    sibling = list(body.get('sibling_evidence_used') or body.get('observed_sibling_evidence') or [])
    source = list(body.get('source_evidence_used') or body.get('observed_theory_evidence') or [])
    gate = body.get('evidence_gate') or evidence.get('evidence_gate')
    status = str(body.get('status') or evidence.get('status') or '').lower()
    if sender in {'funfun', 'code_module', 'language_model_2'} and (gate or status):
        sibling.append({'sender': sender, 'evidence_gate': str(gate or 'advisory'), 'status': status or 'advisory'})
    leaks = label_leak_terms({'body': body, 'evidence': evidence})
    third_party = bool(body.get('third_party_checkpoint_used') or evidence.get('third_party_checkpoint_used'))
    return {
        'frontier_outcome_id': body.get('frontier_outcome_id') or evidence.get('frontier_outcome_id'),
        'frontier_id': body.get('frontier_id') or evidence.get('frontier_id'),
        'source_outcome_id': body.get('source_outcome_id') or body.get('outcome_id') or evidence.get('outcome_id'),
        'hypothesis_id': body.get('hypothesis_id') or evidence.get('hypothesis_id'),
        'campaign_id': body.get('campaign_id') or evidence.get('campaign_id'),
        'scenario_id': str(scenario_id) if scenario_id else None,
        'selected_outcome': selected_outcome or selected_strategy,
        'planned_theory_move': planned_move,
        'source_evidence_used': source,
        'sibling_evidence_used': sibling,
        'promotion_state': body.get('promotion_state') or ('promoted' if selected_outcome == 'theory_memory_recorded_or_hypothesis_promoted' else 'none'),
        'retirement_state': body.get('retirement_state') or body.get('retirement_block_state') or ('retired' if selected_outcome == 'refuted_hypothesis_retired_or_blocked' else 'none'),
        'repair_state': body.get('repair_state') or ('open' if selected_outcome == 'preserve_boundary_or_checkpoint_repair' else 'none'),
        'closure_state': body.get('closure_state') or ('stable' if selected_outcome in {'close_stable_science_campaign', 'campaign_stable'} else 'none'),
        'before_theory_memory_state': body.get('before_theory_memory_state') or body.get('before_hypothesis_state') or 'unknown',
        'after_theory_memory_state': body.get('after_theory_memory_state') or body.get('after_hypothesis_state') or 'unknown',
        'checkpoint_boundary_state': 'repair' if leaks or third_party or body.get('boundary_checkpoint_state') == 'repair' or body.get('checkpoint_boundary_state') == 'repair' else 'clean',
        'checkpoint_boundary_notes': list(body.get('checkpoint_boundary_notes') or body.get('boundary_notes') or ([] if not leaks else ['label leak detected'])),
        'label_leaks': leaks,
        'lineage': [f'message:{sender}'],
    }


def _upsert_row(rows, incoming):
    key = incoming.get('scenario_id') or incoming.get('hypothesis_id') or incoming.get('frontier_outcome_id') or incoming.get('campaign_id')
    if not key:
        return
    current = rows.setdefault(str(key), {
        'frontier_outcome_id': incoming.get('frontier_outcome_id'),
        'frontier_id': incoming.get('frontier_id'),
        'source_outcome_id': incoming.get('source_outcome_id'),
        'hypothesis_id': incoming.get('hypothesis_id'),
        'campaign_id': incoming.get('campaign_id'),
        'scenario_id': incoming.get('scenario_id'),
        'selected_outcome': incoming.get('selected_outcome'),
        'planned_theory_move': incoming.get('planned_theory_move'),
        'source_evidence_used': [],
        'sibling_evidence_used': [],
        'promotion_state': 'none',
        'retirement_state': 'none',
        'repair_state': 'none',
        'closure_state': 'none',
        'before_theory_memory_state': 'unknown',
        'after_theory_memory_state': 'unknown',
        'checkpoint_boundary_state': 'clean',
        'checkpoint_boundary_notes': [],
        'label_leaks': [],
        'lineage': [],
    })
    for field in ('frontier_outcome_id', 'frontier_id', 'source_outcome_id', 'hypothesis_id', 'campaign_id', 'scenario_id'):
        current[field] = current.get(field) or incoming.get(field)
    current['selected_outcome'] = _dominant_outcome(current.get('selected_outcome'), incoming.get('selected_outcome'))
    current['planned_theory_move'] = current.get('planned_theory_move') or incoming.get('planned_theory_move')
    current['source_evidence_used'] = _unique_dicts(list(current.get('source_evidence_used') or []) + list(incoming.get('source_evidence_used') or []))
    current['sibling_evidence_used'] = _unique_dicts(list(current.get('sibling_evidence_used') or []) + list(incoming.get('sibling_evidence_used') or []))
    current['promotion_state'] = _dominant_state(current.get('promotion_state'), incoming.get('promotion_state'))
    current['retirement_state'] = _dominant_retirement(current.get('retirement_state'), incoming.get('retirement_state'))
    current['repair_state'] = _dominant_state(current.get('repair_state'), incoming.get('repair_state'))
    current['closure_state'] = _dominant_state(current.get('closure_state'), incoming.get('closure_state'))
    current['before_theory_memory_state'] = current.get('before_theory_memory_state') or incoming.get('before_theory_memory_state') or 'unknown'
    current['after_theory_memory_state'] = _dominant_memory_state(current.get('after_theory_memory_state'), incoming.get('after_theory_memory_state'))
    current['checkpoint_boundary_state'] = 'repair' if current.get('checkpoint_boundary_state') == 'repair' or incoming.get('checkpoint_boundary_state') == 'repair' else 'clean'
    current['checkpoint_boundary_notes'] = _unique_strings(list(current.get('checkpoint_boundary_notes') or []) + list(incoming.get('checkpoint_boundary_notes') or []))
    current['label_leaks'] = _unique_strings(list(current.get('label_leaks') or []) + list(incoming.get('label_leaks') or []))
    current['lineage'] = _unique_strings(list(current.get('lineage') or []) + list(incoming.get('lineage') or []))


def _finalize_row(row):
    if row.get('label_leaks'):
        row['checkpoint_boundary_state'] = 'repair'
    outcome = row.get('selected_outcome')
    if outcome == 'theory_memory_recorded_or_hypothesis_promoted':
        row['promotion_state'] = 'promoted'
    if outcome == 'refuted_hypothesis_retired_or_blocked':
        row['retirement_state'] = 'retired'
    if outcome == 'preserve_boundary_or_checkpoint_repair':
        row['repair_state'] = 'open'
        row['checkpoint_boundary_state'] = 'repair'
    row['strategy_row_hash'] = stable_digest({
        'frontier_outcome_id': row.get('frontier_outcome_id'),
        'scenario_id': row.get('scenario_id'),
        'selected_outcome': outcome,
        'planned_theory_move': row.get('planned_theory_move'),
        'boundary': row.get('checkpoint_boundary_state'),
    })
    return row


def _select_strategy(*, rows, planned_keys, project_owned_boundary):
    for row in rows:
        if project_owned_boundary.get('third_party_checkpoint_used') or row.get('checkpoint_boundary_state') == 'repair':
            return _strategy(row, 'preserve_checkpoint_boundary', 'preserve_checkpoint_boundary', 'code_module')
    for row in rows:
        if row.get('selected_outcome') == 'theory_memory_recorded_or_hypothesis_promoted' or row.get('promotion_state') == 'promoted':
            return _strategy(row, 'promote_validated_hypothesis_to_theory_memory', 'promote_validated_hypothesis_to_theory_memory', 'broadcast')
    for row in rows:
        if row.get('selected_outcome') == 'refuted_hypothesis_retired_or_blocked' or row.get('retirement_state') in {'retired', 'blocked'}:
            return _strategy(row, 'retire_weak_hypothesis_route', 'retire_weak_hypothesis_route', 'broadcast')
    for row in rows:
        if _needs_funfun(row):
            return _strategy(row, 'request_funfun_formal_or_proof_clarification', 'request_funfun_formal_or_proof_clarification', 'funfun')
    for row in rows:
        if _needs_code(row):
            return _strategy(row, 'request_code_simulation_or_primitive_capability', 'request_code_simulation_or_primitive_capability', 'code_module')
    for row in rows:
        if _needs_language(row):
            return _strategy(row, 'request_language_terminology_or_protocol_clarification', 'request_language_terminology_or_protocol_clarification', 'language_model_2')
    for row in rows:
        if row.get('repair_state') == 'open' or row.get('selected_outcome') == 'hypothesis_refinement_accepted':
            return _strategy(row, 'reopen_theory_repair_cycle', 'reopen_theory_repair_cycle', 'orchestrator')
    for row in rows:
        if row.get('closure_state') in {'stable', 'closed'} or row.get('selected_outcome') in {'close_stable_science_campaign', 'campaign_stable'}:
            return _strategy(row, 'close_stable_science_campaign', 'close_stable_science_campaign', 'orchestrator')
    for row in rows:
        if row.get('selected_outcome') in {'next_campaign_frontier_scheduled', 'planned_theory_move_waiting_for_evidence'}:
            key = _strategy_key_for(row, 'schedule_next_frontier_hypothesis')
            if key not in set(planned_keys):
                return _strategy(row, 'schedule_next_frontier_hypothesis', 'schedule_next_frontier_hypothesis', 'orchestrator')
    for row in rows:
        if row.get('selected_outcome') in {'no_measurable_theory_frontier_gain', 'record_no_measurable_science_gain'}:
            return _strategy(row, 'record_no_measurable_science_gain', 'record_no_measurable_science_gain', 'orchestrator')
    return _noop_strategy('no science campaign strategy selected')


def _strategy(row, selected_strategy, action, recipient):
    return {
        'selected_strategy': selected_strategy,
        'selected_action': action,
        'selected_recipient': validate_participant(recipient),
        'frontier_outcome_id': row.get('frontier_outcome_id'),
        'frontier_id': row.get('frontier_id'),
        'source_outcome_id': row.get('source_outcome_id'),
        'hypothesis_id': row.get('hypothesis_id'),
        'campaign_id': row.get('campaign_id'),
        'scenario_id': row.get('scenario_id'),
        'target_hypothesis': row.get('hypothesis_id'),
        'target_campaign': row.get('campaign_id') or row.get('scenario_id'),
        'required_followup_evidence': _required_evidence(selected_strategy),
        'source_evidence_used': list(row.get('source_evidence_used') or []),
        'sibling_evidence_used': list(row.get('sibling_evidence_used') or []),
        'theory_memory_delta': _theory_memory_delta(selected_strategy, row),
        'promotion_state': row.get('promotion_state') or 'none',
        'retirement_state': row.get('retirement_state') or 'none',
        'repair_state': row.get('repair_state') or 'none',
        'closure_state': row.get('closure_state') or 'none',
        'requested_sibling_evidence': _required_evidence(selected_strategy),
        'before_theory_memory_state': row.get('before_theory_memory_state') or 'unknown',
        'after_theory_memory_state': row.get('after_theory_memory_state') or 'unknown',
        'checkpoint_boundary_state': row.get('checkpoint_boundary_state') or 'clean',
        'checkpoint_boundary_notes': list(row.get('checkpoint_boundary_notes') or []),
        'no_overclaiming_proof': 'no local-owned checkpoint claim made unless status capsule verifies it',
        'label_leaks': list(row.get('label_leaks') or []),
    }


def _noop_strategy(reason):
    return {
        'selected_strategy': 'summarize_noop',
        'selected_action': 'summarize_noop',
        'selected_recipient': None,
        'frontier_outcome_id': None,
        'frontier_id': None,
        'source_outcome_id': None,
        'hypothesis_id': None,
        'campaign_id': None,
        'scenario_id': None,
        'target_hypothesis': None,
        'target_campaign': None,
        'required_followup_evidence': [],
        'source_evidence_used': [],
        'sibling_evidence_used': [],
        'theory_memory_delta': reason,
        'promotion_state': 'none',
        'retirement_state': 'none',
        'repair_state': 'none',
        'closure_state': 'none',
        'requested_sibling_evidence': [],
        'before_theory_memory_state': 'unknown',
        'after_theory_memory_state': 'unknown',
        'checkpoint_boundary_state': 'clean',
        'checkpoint_boundary_notes': [],
        'no_overclaiming_proof': 'no project-owned checkpoint claim made',
        'label_leaks': [],
    }


def _required_evidence(strategy):
    mapping = {
        'request_funfun_formal_or_proof_clarification': ['formal_proof_or_certificate'],
        'request_code_simulation_or_primitive_capability': ['simulation_or_primitive_capability'],
        'request_language_terminology_or_protocol_clarification': ['terminology_or_protocol_clarification'],
        'reopen_theory_repair_cycle': ['repair_plan'],
        'schedule_next_frontier_hypothesis': ['next_frontier_hypothesis'],
    }
    return mapping.get(strategy, [])


def _theory_memory_delta(strategy, row):
    if strategy == 'promote_validated_hypothesis_to_theory_memory':
        return 'queue validated hypothesis for symbolic theory-memory promotion'
    if strategy == 'retire_weak_hypothesis_route':
        return 'mark weak/refuted hypothesis route retired'
    if strategy == 'close_stable_science_campaign':
        return 'close stable campaign after accepted strategy evidence'
    if strategy == 'record_no_measurable_science_gain':
        return 'record no safe science-strategy gain'
    return row.get('selected_outcome') or strategy


def _needs_funfun(row):
    return row.get('selected_outcome') == 'funfun_certificate_supports_or_blocks_theory_move' or row.get('planned_theory_move') == 'request_funfun_certificate'


def _needs_code(row):
    return row.get('selected_outcome') == 'code_experiment_or_counterexample_changes_theory_move' or row.get('planned_theory_move') == 'request_code_experiment_or_counterexample'


def _needs_language(row):
    return row.get('selected_outcome') == 'language_protocol_clarification_resolves_theory_move' or row.get('planned_theory_move') == 'request_language_protocol_clarification'


def _state_counts(selected):
    counts = {
        'boundary': 0,
        'promote': 0,
        'retire': 0,
        'funfun': 0,
        'code': 0,
        'language': 0,
        'repair': 0,
        'close': 0,
        'schedule': 0,
        'waiting': 0,
        'no_gain': 0,
    }
    mapping = {
        'preserve_checkpoint_boundary': 'boundary',
        'promote_validated_hypothesis_to_theory_memory': 'promote',
        'retire_weak_hypothesis_route': 'retire',
        'request_funfun_formal_or_proof_clarification': 'funfun',
        'request_code_simulation_or_primitive_capability': 'code',
        'request_language_terminology_or_protocol_clarification': 'language',
        'reopen_theory_repair_cycle': 'repair',
        'close_stable_science_campaign': 'close',
        'schedule_next_frontier_hypothesis': 'schedule',
        'record_no_measurable_science_gain': 'no_gain',
    }
    key = mapping.get(selected.get('selected_strategy'))
    if key:
        counts[key] += 1
    if selected.get('selected_strategy') == 'summarize_noop':
        counts['no_gain'] += 1
    if selected.get('selected_strategy') in {
        'request_funfun_formal_or_proof_clarification',
        'request_code_simulation_or_primitive_capability',
        'request_language_terminology_or_protocol_clarification',
    }:
        counts['waiting'] += 1
    return counts


def _source_hash(frontier_outcome, frontier, campaign, theory_memory, hypothesis, experiment, module_chat, prior_strategy, runtime_memory):
    return stable_digest({
        'frontier_outcome': frontier_outcome.get('ledger_hash'),
        'frontier': frontier.get('ledger_hash'),
        'campaign': campaign.get('ledger_hash'),
        'theory_memory': theory_memory.get('ledger_hash') or stable_digest(theory_memory) if theory_memory else None,
        'hypothesis': hypothesis.get('ledger_hash') or stable_digest(hypothesis) if hypothesis else None,
        'experiment': experiment.get('ledger_hash') or stable_digest(experiment) if experiment else None,
        'module_chat': module_chat.get('ledger_hash') or stable_digest(module_chat) if module_chat else None,
        'prior_strategy': prior_strategy.get('ledger_hash'),
        'runtime_keys': sorted(runtime_memory.keys()),
    })


def _plain_rows(source, keys):
    rows = []
    for key in keys:
        values = source.get(key)
        if isinstance(values, list):
            rows.extend([dict(item) for item in values if isinstance(item, dict)])
    return rows


def _find_row(rows, frontier_outcome_id, scenario_id):
    for row in rows:
        if frontier_outcome_id and row.get('frontier_outcome_id') == frontier_outcome_id:
            return row
        if scenario_id and row.get('scenario_id') == scenario_id:
            return row
    return None


def _strategy_key(selected):
    return _strategy_key_for(selected, selected.get('selected_strategy'))


def _strategy_key_for(row, strategy):
    return stable_digest({
        'frontier_outcome_id': row.get('frontier_outcome_id'),
        'scenario_id': row.get('scenario_id'),
        'hypothesis_id': row.get('hypothesis_id'),
        'campaign_id': row.get('campaign_id'),
        'strategy': strategy,
    })


def _dominant_outcome(current, incoming):
    order = {
        'preserve_boundary_or_checkpoint_repair': 11,
        'theory_memory_recorded_or_hypothesis_promoted': 10,
        'refuted_hypothesis_retired_or_blocked': 9,
        'funfun_certificate_supports_or_blocks_theory_move': 8,
        'code_experiment_or_counterexample_changes_theory_move': 7,
        'language_protocol_clarification_resolves_theory_move': 6,
        'hypothesis_refinement_accepted': 5,
        'close_stable_science_campaign': 4,
        'campaign_stable': 4,
        'next_campaign_frontier_scheduled': 3,
        'planned_theory_move_waiting_for_evidence': 3,
        'no_measurable_theory_frontier_gain': 2,
        None: 0,
    }
    return current if order.get(current, 0) >= order.get(incoming, 0) else incoming


def _dominant_state(current, incoming):
    order = {'promoted': 5, 'stable': 5, 'closed': 5, 'open': 4, 'accepted': 4, 'updated': 4, 'scheduled': 3, 'none': 1, None: 0}
    return current if order.get(current, 0) >= order.get(incoming, 0) else incoming


def _dominant_retirement(current, incoming):
    order = {'retired': 5, 'blocked': 5, 'weak': 4, 'none': 1, None: 0}
    return current if order.get(current, 0) >= order.get(incoming, 0) else incoming


def _dominant_memory_state(current, incoming):
    order = {'promoted': 5, 'recorded': 4, 'accepted': 4, 'not_recorded': 1, 'unknown': 0, None: 0}
    return current if order.get(current, 0) >= order.get(incoming, 0) else incoming


def _first(values):
    return values[0] if values else None


def _unique_strings(values):
    seen = set()
    output = []
    for value in values:
        if value is None:
            continue
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def _unique_dicts(values):
    seen = set()
    output = []
    for value in values:
        if not isinstance(value, dict):
            continue
        key = stable_digest(value)
        if key in seen:
            continue
        seen.add(key)
        output.append(dict(value))
    return output
