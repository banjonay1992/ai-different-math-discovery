"""Science campaign cycle strategy planner.

This layer turns assessed science campaign strategy outcomes into one durable
higher-level campaign-cycle move. It is a plain-data orchestration artifact:
no sibling imports, no empirical discovery claim, and no project-owned
checkpoint claim unless the local status capsule verifies one.
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
from .science_campaign_strategy import SCIENCE_CAMPAIGN_STRATEGY_LEDGER_KIND
from .science_campaign_strategy_outcome import SCIENCE_CAMPAIGN_STRATEGY_OUTCOME_LEDGER_KIND
from .science_theory_frontier import SCIENCE_THEORY_FRONTIER_LEDGER_KIND
from .science_theory_frontier_outcome import SCIENCE_THEORY_FRONTIER_OUTCOME_LEDGER_KIND


SCIENCE_CAMPAIGN_CYCLE_STRATEGY_LEDGER_KIND = 'ai_different.science_campaign_cycle_strategy_ledger'


def empty_science_campaign_cycle_strategy_ledger() -> dict[str, Any]:
    return {
        'schema_version': 1,
        'ledger_kind': SCIENCE_CAMPAIGN_CYCLE_STRATEGY_LEDGER_KIND,
        'processed_message_ids': [],
        'processed_source_hashes': [],
        'planned_cycle_strategy_keys': [],
        'cycle_strategy_records': [],
        'cycle_strategy_rows': [],
        'outgoing_response_ids': [],
        'latest': {},
    }


def load_science_campaign_cycle_strategy_ledger(path: str | Path | None) -> dict[str, Any]:
    if not path or not Path(path).exists():
        return empty_science_campaign_cycle_strategy_ledger()
    with Path(path).open('r', encoding='utf-8') as handle:
        ledger = json.load(handle)
    return validate_science_campaign_cycle_strategy_ledger(ledger)


def write_science_campaign_cycle_strategy_ledger(path: str | Path, ledger: dict[str, Any]) -> Path:
    validated = validate_science_campaign_cycle_strategy_ledger(ledger)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(validated, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return output


def validate_science_campaign_cycle_strategy_ledger(ledger: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(ledger, dict):
        raise ValueError('science campaign cycle strategy ledger must be a JSON object')
    if ledger.get('ledger_kind') != SCIENCE_CAMPAIGN_CYCLE_STRATEGY_LEDGER_KIND:
        raise ValueError('science campaign cycle strategy ledger has wrong ledger_kind')
    for field in (
        'processed_message_ids',
        'processed_source_hashes',
        'planned_cycle_strategy_keys',
        'outgoing_response_ids',
    ):
        values = ledger.get(field)
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            raise ValueError(f'{field} must be strings')
    for field in ('cycle_strategy_records', 'cycle_strategy_rows'):
        values = ledger.get(field)
        if not isinstance(values, list) or not all(isinstance(item, dict) for item in values):
            raise ValueError(f'{field} must be objects')
    latest = ledger.get('latest', {})
    if not isinstance(latest, dict):
        raise ValueError('science campaign cycle strategy latest must be an object')
    validated = {
        'schema_version': int(ledger.get('schema_version', 1) or 1),
        'ledger_kind': SCIENCE_CAMPAIGN_CYCLE_STRATEGY_LEDGER_KIND,
        'processed_message_ids': _unique_strings(ledger['processed_message_ids']),
        'processed_source_hashes': _unique_strings(ledger['processed_source_hashes']),
        'planned_cycle_strategy_keys': _unique_strings(ledger['planned_cycle_strategy_keys']),
        'cycle_strategy_records': list(ledger['cycle_strategy_records']),
        'cycle_strategy_rows': list(ledger['cycle_strategy_rows']),
        'outgoing_response_ids': _unique_strings(ledger['outgoing_response_ids']),
        'latest': dict(latest),
    }
    if ledger.get('ledger_hash'):
        validated['ledger_hash'] = str(ledger['ledger_hash'])
    if ledger.get('artifact_path'):
        validated['artifact_path'] = str(ledger['artifact_path'])
    return validated


def read_science_campaign_cycle_strategy_transcript(path: str | Path | None) -> dict[str, Any]:
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
        raise ValueError('plain science campaign cycle strategy input must be a JSON object')
    return value


def build_science_campaign_cycle_strategy_plan(
    *,
    transcript_messages: list[dict[str, Any]],
    cycle_strategy_ledger: dict[str, Any],
    strategy_outcome_ledger: dict[str, Any] | None = None,
    strategy_ledger: dict[str, Any] | None = None,
    frontier_outcome_ledger: dict[str, Any] | None = None,
    frontier_ledger: dict[str, Any] | None = None,
    campaign_ledger: dict[str, Any] | None = None,
    theory_memory_ledger: dict[str, Any] | None = None,
    hypothesis_ledger: dict[str, Any] | None = None,
    experiment_ledger: dict[str, Any] | None = None,
    module_chat_ledger: dict[str, Any] | None = None,
    prior_cycle_strategy_ledger: dict[str, Any] | None = None,
    runtime_memory_data: dict[str, Any] | None = None,
    runtime_memory_hash_state: dict[str, Any],
    project_owned_boundary: dict[str, Any],
    artifact_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    ledger = validate_science_campaign_cycle_strategy_ledger(cycle_strategy_ledger)
    strategy_outcome = _valid_strategy_outcome_or_empty(strategy_outcome_ledger or {})
    strategy_source = _valid_strategy_or_empty(strategy_ledger or {})
    frontier_outcome = _valid_frontier_outcome_or_empty(frontier_outcome_ledger or {})
    frontier = _valid_frontier_or_empty(frontier_ledger or {})
    campaign = _valid_campaign_or_empty(campaign_ledger or {})
    theory_memory = _valid_plain_or_empty(theory_memory_ledger or {})
    hypothesis = _valid_plain_or_empty(hypothesis_ledger or {})
    experiment = _valid_plain_or_empty(experiment_ledger or {})
    module_chat = _valid_plain_or_empty(module_chat_ledger or {})
    prior_cycle = _valid_prior_cycle_or_empty(prior_cycle_strategy_ledger or {})
    runtime_memory = dict(runtime_memory_data or {})
    source_hash = _source_hash(
        strategy_outcome,
        strategy_source,
        frontier_outcome,
        frontier,
        campaign,
        theory_memory,
        hypothesis,
        experiment,
        module_chat,
        prior_cycle,
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
    rows = _extract_cycle_strategy_rows(
        ledger['cycle_strategy_rows'],
        strategy_outcome,
        strategy_source,
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
        selected = _noop_strategy('no new science campaign cycle strategy evidence or source ledger state')
    else:
        selected = _select_cycle_strategy(
            rows=rows,
            planned_keys=ledger['planned_cycle_strategy_keys'],
            project_owned_boundary=project_owned_boundary,
        )
    strategy_id = 'science_cycle_strategy_' + stable_digest({
        'source_hash': source_hash,
        'new_message_ids': [module_chat_message_id(message) for message in new_messages],
        'strategy': selected['selected_strategy'],
        'outcome_id': selected.get('campaign_strategy_outcome_id'),
        'scenario_id': selected.get('scenario_id'),
    })[:16]
    selected['campaign_cycle_strategy_id'] = strategy_id
    message = export_science_campaign_cycle_strategy_message(
        selected,
        rows=rows,
        runtime_memory_hash_state=runtime_memory_hash_state,
        project_owned_boundary=project_owned_boundary,
        source_hashes={
            'science_campaign_cycle_strategy_source_hash': source_hash,
            'strategy_outcome_ledger_hash': strategy_outcome.get('ledger_hash'),
            'strategy_ledger_hash': strategy_source.get('ledger_hash'),
            'frontier_outcome_ledger_hash': frontier_outcome.get('ledger_hash'),
            'frontier_ledger_hash': frontier.get('ledger_hash'),
            'campaign_ledger_hash': campaign.get('ledger_hash'),
            'theory_memory_ledger_hash': theory_memory.get('ledger_hash'),
            'hypothesis_ledger_hash': hypothesis.get('ledger_hash'),
            'experiment_ledger_hash': experiment.get('ledger_hash'),
            'module_chat_ledger_hash': module_chat.get('ledger_hash'),
            'prior_cycle_strategy_ledger_hash': prior_cycle.get('ledger_hash'),
        },
    )
    new_ids = [module_chat_message_id(message) for message in new_messages]
    outgoing_id = module_chat_message_id(message) if message else None
    planned_key = _cycle_strategy_key(selected)
    if selected['selected_strategy'] != 'summarize_noop':
        ledger['planned_cycle_strategy_keys'] = _unique_strings(list(ledger['planned_cycle_strategy_keys']) + [planned_key])
    if new_messages:
        ledger['processed_message_ids'] = _unique_strings(list(ledger['processed_message_ids']) + new_ids)
    if source_is_new:
        ledger['processed_source_hashes'] = _unique_strings(list(ledger['processed_source_hashes']) + [source_hash])
    if outgoing_id:
        ledger['outgoing_response_ids'] = _unique_strings(list(ledger['outgoing_response_ids']) + [outgoing_id])
    state_counts = _state_counts(selected)
    latest = {
        'campaign_cycle_strategy_id': strategy_id,
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
        'campaign_cycle_strategy_id': strategy_id,
        'strategy_hash': stable_digest({'strategy_id': strategy_id, 'selected': selected}),
        'processed_message_ids': new_ids,
        'source_hash': source_hash,
        'campaign_strategy_outcome_id': selected.get('campaign_strategy_outcome_id'),
        'campaign_strategy_id': selected.get('campaign_strategy_id'),
        'frontier_outcome_id': selected.get('frontier_outcome_id'),
        'frontier_id': selected.get('frontier_id'),
        'hypothesis_id': selected.get('hypothesis_id'),
        'campaign_id': selected.get('campaign_id'),
        'family_id': selected.get('family_id'),
        'selected_strategy': selected['selected_strategy'],
        'selected_action': selected['selected_action'],
        'selected_recipient': latest['chosen_recipient'],
        'promotion_state': selected.get('promotion_state'),
        'retirement_state': selected.get('retirement_state'),
        'repair_state': selected.get('repair_state'),
        'closure_state': selected.get('closure_state'),
        'required_sibling_evidence': selected.get('required_sibling_evidence'),
        'before_campaign_cycle_memory_state': selected.get('before_campaign_cycle_memory_state'),
        'after_campaign_cycle_memory_state': selected.get('after_campaign_cycle_memory_state'),
        'checkpoint_boundary_notes': selected.get('checkpoint_boundary_notes'),
        'no_overclaiming_proof': selected.get('no_overclaiming_proof'),
        'outgoing_response_id': outgoing_id,
    }
    ledger['cycle_strategy_rows'] = rows
    if new_messages or source_is_new or message is not None:
        ledger['cycle_strategy_records'] = list(ledger['cycle_strategy_records']) + [record]
    ledger['latest'] = latest
    if artifact_path is not None:
        ledger['artifact_path'] = str(Path(artifact_path))
    ledger['ledger_hash'] = stable_digest({key: value for key, value in ledger.items() if key != 'ledger_hash'})
    return ledger, message


def export_science_campaign_cycle_strategy_message(
    selected: dict[str, Any],
    *,
    rows: list[dict[str, Any]],
    runtime_memory_hash_state: dict[str, Any],
    project_owned_boundary: dict[str, Any],
    source_hashes: dict[str, Any],
    topic: str = 'ai_different.science_campaign_cycle_strategy',
) -> dict[str, Any] | None:
    if selected['selected_strategy'] == 'summarize_noop':
        return None
    recipient = validate_participant(selected.get('selected_recipient') or 'orchestrator')
    row = _find_row(rows, selected.get('campaign_strategy_outcome_id'), selected.get('scenario_id')) or {}
    leak_terms = label_leak_terms({'selected': selected, 'row': row})
    body = {
        'module': 'AI Different',
        'response_kind': 'science_campaign_cycle_strategy',
        'campaign_cycle_strategy_id': selected.get('campaign_cycle_strategy_id'),
        'campaign_strategy_outcome_id': selected.get('campaign_strategy_outcome_id'),
        'campaign_strategy_id': selected.get('campaign_strategy_id'),
        'frontier_outcome_id': selected.get('frontier_outcome_id'),
        'frontier_id': selected.get('frontier_id'),
        'hypothesis_id': selected.get('hypothesis_id'),
        'campaign_id': selected.get('campaign_id'),
        'family_id': selected.get('family_id'),
        'scenario_id': selected.get('scenario_id'),
        'selected_strategy': selected['selected_strategy'],
        'selected_action': selected['selected_action'],
        'selected_recipient': recipient,
        'target_hypothesis': selected.get('target_hypothesis'),
        'target_campaign': selected.get('target_campaign'),
        'target_family': selected.get('target_family'),
        'required_followup_evidence': selected.get('required_followup_evidence') or [],
        'required_sibling_evidence': selected.get('required_sibling_evidence') or [],
        'source_evidence_used': selected.get('source_evidence_used') or [],
        'sibling_evidence_used': selected.get('sibling_evidence_used') or [],
        'campaign_cycle_memory_delta': selected.get('campaign_cycle_memory_delta'),
        'theory_memory_delta': selected.get('theory_memory_delta'),
        'promotion_state': selected.get('promotion_state'),
        'retirement_state': selected.get('retirement_state'),
        'repair_state': selected.get('repair_state'),
        'closure_state': selected.get('closure_state'),
        'waiting_blocker_state': selected.get('waiting_blocker_state'),
        'before_campaign_cycle_memory_state': selected.get('before_campaign_cycle_memory_state'),
        'after_campaign_cycle_memory_state': selected.get('after_campaign_cycle_memory_state'),
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
            'campaign_cycle_strategy_id': body['campaign_cycle_strategy_id'],
            'campaign_strategy_outcome_id': body['campaign_strategy_outcome_id'],
            'hypothesis_id': body['hypothesis_id'],
            'campaign_id': body['campaign_id'],
            'family_id': body['family_id'],
            'selected_strategy': body['selected_strategy'],
            'selected_action': body['selected_action'],
            'label_clean': body['label_clean'],
            'no_sibling_imports': True,
        },
        tags=['ai_different', 'science_campaign_cycle_strategy', body['selected_strategy'], 'label_clean' if body['label_clean'] else 'label_review_needed'],
    )


def write_science_campaign_cycle_strategy_outbox_jsonl(path: str | Path, message: dict[str, Any] | None) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text((json.dumps(message, sort_keys=True) + '\n') if message is not None else '', encoding='utf-8')
    return output


def _valid_strategy_outcome_or_empty(ledger):
    if not ledger:
        return {'ledger_kind': SCIENCE_CAMPAIGN_STRATEGY_OUTCOME_LEDGER_KIND, 'outcome_records': [], 'strategy_rows': [], 'ledger_hash': None}
    if ledger.get('ledger_kind') != SCIENCE_CAMPAIGN_STRATEGY_OUTCOME_LEDGER_KIND:
        raise ValueError('science campaign strategy outcome ledger has wrong ledger_kind')
    return dict(ledger)


def _valid_strategy_or_empty(ledger):
    if not ledger:
        return {'ledger_kind': SCIENCE_CAMPAIGN_STRATEGY_LEDGER_KIND, 'strategy_records': [], 'strategy_rows': [], 'ledger_hash': None}
    if ledger.get('ledger_kind') != SCIENCE_CAMPAIGN_STRATEGY_LEDGER_KIND:
        raise ValueError('science campaign strategy ledger has wrong ledger_kind')
    return dict(ledger)


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
        raise ValueError('plain science campaign cycle strategy ledger must be a JSON object')
    return dict(ledger)


def _valid_prior_cycle_or_empty(ledger):
    if not ledger:
        return {}
    if ledger.get('ledger_kind') != SCIENCE_CAMPAIGN_CYCLE_STRATEGY_LEDGER_KIND:
        raise ValueError('prior science campaign cycle strategy ledger has wrong ledger_kind')
    return validate_science_campaign_cycle_strategy_ledger(ledger)


def _extract_cycle_strategy_rows(existing, strategy_outcome, strategy_ledger, frontier_outcome, frontier, campaign, theory_memory, hypothesis, experiment, module_chat, messages):
    rows: dict[str, dict[str, Any]] = {}
    for row in list(existing or []):
        _upsert_row(rows, row)
    for record in list(strategy_outcome.get('outcome_records') or []):
        _upsert_row(rows, _row_from_strategy_outcome_record(record))
    for row in list(strategy_outcome.get('strategy_rows') or []):
        _upsert_row(rows, _row_from_strategy_outcome_row(row))
    for record in list(strategy_ledger.get('strategy_records') or []):
        _upsert_row(rows, _row_from_strategy_record(record))
    for record in list(frontier_outcome.get('outcome_records') or []):
        _upsert_row(rows, _row_from_frontier_outcome(record))
    for record in list(frontier.get('frontier_records') or []):
        _upsert_row(rows, _row_from_frontier(record))
    for row in _plain_rows(campaign, ('campaigns', 'campaign_rows', 'campaign_records')):
        _upsert_row(rows, _row_from_campaign(row))
    for row in _plain_rows(theory_memory, ('campaign_cycle_memory_rows', 'cycle_strategy_rows', 'theory_memory_rows', 'theories')):
        _upsert_row(rows, _row_from_memory(row))
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


def _row_from_strategy_outcome_record(record):
    return {
        'campaign_strategy_outcome_id': record.get('campaign_strategy_outcome_id') or record.get('outcome_id'),
        'campaign_strategy_id': record.get('strategy_id'),
        'frontier_outcome_id': record.get('frontier_outcome_id'),
        'frontier_id': record.get('frontier_id'),
        'hypothesis_id': record.get('hypothesis_id'),
        'campaign_id': record.get('campaign_id'),
        'family_id': record.get('family_id') or record.get('hypothesis_family_id'),
        'scenario_id': record.get('scenario_id') or record.get('campaign_id') or record.get('hypothesis_id'),
        'selected_outcome': record.get('selected_outcome'),
        'planned_strategy': record.get('planned_strategy'),
        'source_evidence_used': list(record.get('source_evidence_used') or []),
        'sibling_evidence_used': list(record.get('observed_sibling_evidence') or []),
        'promotion_state': record.get('promotion_state') or 'none',
        'retirement_state': record.get('retirement_state') or 'none',
        'repair_state': record.get('repair_state') or 'none',
        'closure_state': record.get('closure_state') or 'none',
        'waiting_blocker_state': record.get('waiting_blocker_state') or 'waiting',
        'before_campaign_cycle_memory_state': record.get('before_campaign_cycle_memory_state') or record.get('before_theory_memory_state') or 'unknown',
        'after_campaign_cycle_memory_state': record.get('after_campaign_cycle_memory_state') or record.get('after_theory_memory_state') or 'unknown',
        'checkpoint_boundary_state': 'repair' if record.get('checkpoint_boundary_notes') else 'clean',
        'checkpoint_boundary_notes': list(record.get('checkpoint_boundary_notes') or []),
        'label_leaks': list(record.get('label_leaks') or []),
        'lineage': ['strategy_outcome_record'],
    }


def _row_from_strategy_outcome_row(row):
    return {
        'campaign_strategy_outcome_id': row.get('campaign_strategy_outcome_id') or row.get('outcome_id'),
        'campaign_strategy_id': row.get('strategy_id') or row.get('campaign_strategy_id'),
        'frontier_outcome_id': row.get('frontier_outcome_id'),
        'frontier_id': row.get('frontier_id'),
        'hypothesis_id': row.get('hypothesis_id'),
        'campaign_id': row.get('campaign_id'),
        'family_id': row.get('family_id') or row.get('hypothesis_family_id'),
        'scenario_id': row.get('scenario_id') or row.get('campaign_id') or row.get('hypothesis_id'),
        'selected_outcome': row.get('selected_outcome'),
        'planned_strategy': row.get('planned_strategy') or row.get('selected_strategy'),
        'source_evidence_used': list(row.get('source_evidence_used') or []),
        'sibling_evidence_used': list(row.get('observed_sibling_evidence') or row.get('sibling_evidence_used') or []),
        'promotion_state': row.get('promotion_state') or 'none',
        'retirement_state': row.get('retirement_state') or 'none',
        'repair_state': row.get('repair_state') or 'none',
        'closure_state': row.get('closure_state') or 'none',
        'waiting_blocker_state': row.get('waiting_blocker_state') or 'waiting',
        'before_campaign_cycle_memory_state': row.get('before_campaign_cycle_memory_state') or row.get('before_theory_memory_state') or 'unknown',
        'after_campaign_cycle_memory_state': row.get('after_campaign_cycle_memory_state') or row.get('after_theory_memory_state') or 'unknown',
        'checkpoint_boundary_state': row.get('checkpoint_boundary_state') or 'clean',
        'checkpoint_boundary_notes': list(row.get('checkpoint_boundary_notes') or []),
        'label_leaks': list(row.get('label_leaks') or []),
        'lineage': ['strategy_outcome_row'],
    }


def _row_from_strategy_record(record):
    return {
        'campaign_strategy_id': record.get('strategy_id'),
        'frontier_outcome_id': record.get('frontier_outcome_id'),
        'frontier_id': record.get('frontier_id'),
        'hypothesis_id': record.get('hypothesis_id'),
        'campaign_id': record.get('campaign_id'),
        'family_id': record.get('family_id') or record.get('hypothesis_family_id'),
        'scenario_id': record.get('scenario_id') or record.get('campaign_id') or record.get('hypothesis_id'),
        'selected_outcome': record.get('selected_outcome'),
        'planned_strategy': record.get('selected_strategy') or record.get('planned_strategy'),
        'source_evidence_used': [],
        'sibling_evidence_used': list(record.get('requested_sibling_evidence') or []),
        'promotion_state': record.get('promotion_state') or 'none',
        'retirement_state': record.get('retirement_state') or 'none',
        'repair_state': record.get('repair_state') or 'none',
        'closure_state': record.get('closure_state') or 'none',
        'waiting_blocker_state': 'waiting' if record.get('requested_sibling_evidence') else 'resolved',
        'before_campaign_cycle_memory_state': record.get('before_campaign_cycle_memory_state') or 'unknown',
        'after_campaign_cycle_memory_state': record.get('after_campaign_cycle_memory_state') or 'unknown',
        'checkpoint_boundary_state': 'repair' if record.get('checkpoint_boundary_notes') else 'clean',
        'checkpoint_boundary_notes': list(record.get('checkpoint_boundary_notes') or []),
        'label_leaks': [],
        'lineage': ['strategy_record'],
    }


def _row_from_frontier_outcome(record):
    return {
        'frontier_outcome_id': record.get('frontier_outcome_id'),
        'frontier_id': record.get('frontier_id'),
        'hypothesis_id': record.get('hypothesis_id'),
        'campaign_id': record.get('campaign_id'),
        'family_id': record.get('family_id'),
        'scenario_id': record.get('scenario_id') or record.get('campaign_id') or record.get('hypothesis_id'),
        'selected_outcome': record.get('selected_outcome'),
        'planned_strategy': record.get('selected_theory_move'),
        'source_evidence_used': list(record.get('observed_theory_evidence') or []),
        'sibling_evidence_used': list(record.get('observed_sibling_evidence') or []),
        'promotion_state': record.get('promotion_state') or 'none',
        'retirement_state': record.get('retirement_block_state') or 'none',
        'repair_state': 'open' if record.get('boundary_checkpoint_state') == 'repair' else 'none',
        'closure_state': record.get('closure_state') or 'none',
        'waiting_blocker_state': record.get('waiting_blocker_state') or 'waiting',
        'before_campaign_cycle_memory_state': 'unknown',
        'after_campaign_cycle_memory_state': 'unknown',
        'checkpoint_boundary_state': record.get('boundary_checkpoint_state') or 'clean',
        'checkpoint_boundary_notes': list(record.get('boundary_notes') or []),
        'label_leaks': [],
        'lineage': ['frontier_outcome_record'],
    }


def _row_from_frontier(record):
    return {
        'frontier_id': record.get('frontier_id'),
        'hypothesis_id': _first(record.get('hypothesis_ids') or []),
        'campaign_id': _first(record.get('campaign_ids') or []),
        'family_id': record.get('family_id'),
        'scenario_id': _first(record.get('scenario_ids') or []),
        'selected_outcome': record.get('selected_theory_move'),
        'planned_strategy': record.get('selected_theory_move'),
        'source_evidence_used': [],
        'sibling_evidence_used': list(record.get('observed_sibling_evidence') or []),
        'promotion_state': 'none',
        'retirement_state': record.get('retirement_block_state') or 'none',
        'repair_state': 'open' if record.get('boundary_checkpoint_state') == 'repair' else 'none',
        'closure_state': 'none',
        'waiting_blocker_state': record.get('waiting_blocker_state') or 'waiting',
        'before_campaign_cycle_memory_state': 'unknown',
        'after_campaign_cycle_memory_state': 'unknown',
        'checkpoint_boundary_state': record.get('boundary_checkpoint_state') or 'clean',
        'checkpoint_boundary_notes': list(record.get('boundary_notes') or []),
        'label_leaks': [],
        'lineage': ['frontier_record'],
    }


def _row_from_campaign(row):
    state = str(row.get('campaign_state') or row.get('cycle_state') or row.get('status') or '')
    return {
        'campaign_id': row.get('campaign_id') or row.get('id'),
        'hypothesis_id': row.get('hypothesis_id'),
        'family_id': row.get('family_id') or row.get('hypothesis_family_id'),
        'scenario_id': row.get('scenario_id') or row.get('campaign_id') or row.get('id'),
        'selected_outcome': row.get('selected_outcome') or state,
        'planned_strategy': row.get('selected_strategy') or row.get('planned_strategy'),
        'source_evidence_used': [],
        'sibling_evidence_used': [],
        'promotion_state': row.get('promotion_state') or 'none',
        'retirement_state': row.get('retirement_state') or 'none',
        'repair_state': row.get('repair_state') or 'none',
        'closure_state': row.get('closure_state') or ('closed' if state in {'stable', 'closed'} else 'none'),
        'waiting_blocker_state': row.get('waiting_blocker_state') or 'resolved',
        'before_campaign_cycle_memory_state': row.get('before_campaign_cycle_memory_state') or 'unknown',
        'after_campaign_cycle_memory_state': row.get('after_campaign_cycle_memory_state') or state or 'unknown',
        'checkpoint_boundary_state': row.get('checkpoint_boundary_state') or 'clean',
        'checkpoint_boundary_notes': list(row.get('checkpoint_boundary_notes') or []),
        'label_leaks': list(row.get('label_leaks') or []),
        'lineage': ['campaign'],
    }


def _row_from_memory(row):
    status = str(row.get('status') or row.get('cycle_state') or 'recorded')
    return {
        'campaign_strategy_outcome_id': row.get('campaign_strategy_outcome_id'),
        'campaign_strategy_id': row.get('campaign_strategy_id') or row.get('strategy_id'),
        'hypothesis_id': row.get('hypothesis_id'),
        'campaign_id': row.get('campaign_id'),
        'family_id': row.get('family_id') or row.get('hypothesis_family_id'),
        'scenario_id': row.get('scenario_id') or row.get('hypothesis_id') or row.get('campaign_id'),
        'selected_outcome': 'validated_hypothesis_promoted_to_theory_memory' if status in {'recorded', 'promoted', 'accepted'} else row.get('selected_outcome'),
        'planned_strategy': row.get('selected_strategy') or 'promote_validated_hypothesis_strategy_to_campaign_cycle_memory',
        'source_evidence_used': [{'source': row.get('source') or 'campaign_cycle_memory', 'status': status}],
        'sibling_evidence_used': [],
        'promotion_state': 'promoted' if status in {'recorded', 'promoted', 'accepted'} else 'none',
        'retirement_state': row.get('retirement_state') or 'none',
        'repair_state': row.get('repair_state') or 'none',
        'closure_state': row.get('closure_state') or 'none',
        'waiting_blocker_state': 'resolved',
        'before_campaign_cycle_memory_state': row.get('before_campaign_cycle_memory_state') or 'not_recorded',
        'after_campaign_cycle_memory_state': row.get('after_campaign_cycle_memory_state') or status,
        'checkpoint_boundary_state': row.get('checkpoint_boundary_state') or 'clean',
        'checkpoint_boundary_notes': list(row.get('checkpoint_boundary_notes') or []),
        'label_leaks': list(row.get('label_leaks') or []),
        'lineage': ['campaign_cycle_memory'],
    }


def _row_from_hypothesis(row):
    state = str(row.get('state') or row.get('status') or '')
    return {
        'hypothesis_id': row.get('hypothesis_id') or row.get('id'),
        'campaign_id': row.get('campaign_id'),
        'family_id': row.get('family_id') or row.get('hypothesis_family_id'),
        'scenario_id': row.get('scenario_id') or row.get('hypothesis_id') or row.get('id'),
        'selected_outcome': row.get('selected_outcome') or state,
        'planned_strategy': row.get('selected_strategy') or row.get('planned_strategy'),
        'source_evidence_used': [],
        'sibling_evidence_used': [],
        'promotion_state': 'promoted' if state in {'validated', 'promoted'} else 'none',
        'retirement_state': 'retired' if state in {'retired', 'weak', 'blocked'} else 'none',
        'repair_state': 'open' if state == 'repair' else 'none',
        'closure_state': 'none',
        'waiting_blocker_state': 'resolved',
        'before_campaign_cycle_memory_state': 'unknown',
        'after_campaign_cycle_memory_state': 'unknown',
        'checkpoint_boundary_state': row.get('checkpoint_boundary_state') or 'clean',
        'checkpoint_boundary_notes': list(row.get('checkpoint_boundary_notes') or []),
        'label_leaks': list(row.get('label_leaks') or []),
        'lineage': ['hypothesis'],
    }


def _row_from_experiment(row):
    request_type = str(row.get('request_type') or row.get('kind') or '')
    sibling = []
    if request_type:
        sibling.append({'sender': row.get('sender') or 'code_module', 'evidence_gate': request_type, 'status': row.get('status') or 'received'})
    return {
        'campaign_strategy_id': row.get('strategy_id'),
        'hypothesis_id': row.get('hypothesis_id'),
        'campaign_id': row.get('campaign_id'),
        'family_id': row.get('family_id') or row.get('hypothesis_family_id'),
        'scenario_id': row.get('scenario_id') or row.get('request_id'),
        'selected_outcome': row.get('selected_outcome'),
        'planned_strategy': row.get('selected_strategy') or row.get('planned_strategy'),
        'source_evidence_used': [],
        'sibling_evidence_used': sibling,
        'promotion_state': 'none',
        'retirement_state': 'none',
        'repair_state': 'none',
        'closure_state': 'none',
        'waiting_blocker_state': 'resolved' if sibling else 'waiting',
        'before_campaign_cycle_memory_state': 'unknown',
        'after_campaign_cycle_memory_state': 'unknown',
        'checkpoint_boundary_state': row.get('checkpoint_boundary_state') or 'clean',
        'checkpoint_boundary_notes': list(row.get('checkpoint_boundary_notes') or []),
        'label_leaks': list(row.get('label_leaks') or []),
        'lineage': ['experiment'],
    }


def _row_from_message(message):
    body = dict(message.get('body') or {})
    evidence = dict(message.get('evidence') or {})
    sender = message.get('sender')
    scenario_id = body.get('scenario_id') or evidence.get('scenario_id') or body.get('campaign_id') or evidence.get('campaign_id') or body.get('hypothesis_id') or evidence.get('hypothesis_id')
    selected_outcome = body.get('selected_outcome') or body.get('selected_strategy') or evidence.get('selected_outcome') or evidence.get('selected_strategy')
    sibling = list(body.get('observed_sibling_evidence') or body.get('sibling_evidence_used') or [])
    gate = body.get('evidence_gate') or evidence.get('evidence_gate')
    status = str(body.get('status') or evidence.get('status') or '').lower()
    if sender in {'funfun', 'code_module', 'language_model_2'} and (gate or status):
        sibling.append({'sender': sender, 'evidence_gate': str(gate or 'advisory'), 'status': status or 'received'})
    leaks = label_leak_terms({'body': body, 'evidence': evidence})
    third_party = bool(body.get('third_party_checkpoint_used') or evidence.get('third_party_checkpoint_used'))
    return {
        'campaign_strategy_outcome_id': body.get('campaign_strategy_outcome_id') or evidence.get('campaign_strategy_outcome_id') or body.get('outcome_id') or evidence.get('outcome_id'),
        'campaign_strategy_id': body.get('strategy_id') or body.get('campaign_strategy_id') or evidence.get('strategy_id'),
        'frontier_outcome_id': body.get('frontier_outcome_id') or evidence.get('frontier_outcome_id'),
        'frontier_id': body.get('frontier_id') or evidence.get('frontier_id'),
        'hypothesis_id': body.get('hypothesis_id') or evidence.get('hypothesis_id'),
        'campaign_id': body.get('campaign_id') or evidence.get('campaign_id'),
        'family_id': body.get('family_id') or body.get('hypothesis_family_id') or evidence.get('family_id'),
        'scenario_id': str(scenario_id) if scenario_id else None,
        'selected_outcome': selected_outcome,
        'planned_strategy': body.get('planned_strategy') or body.get('selected_strategy') or evidence.get('planned_strategy'),
        'source_evidence_used': list(body.get('source_evidence_used') or []),
        'sibling_evidence_used': sibling,
        'promotion_state': body.get('promotion_state') or ('promoted' if selected_outcome in {'validated_hypothesis_promoted_to_theory_memory', 'promote_validated_hypothesis_to_theory_memory'} else 'none'),
        'retirement_state': body.get('retirement_state') or ('retired' if selected_outcome in {'weak_hypothesis_route_retired', 'retire_weak_hypothesis_route'} else 'none'),
        'repair_state': body.get('repair_state') or ('open' if selected_outcome in {'theory_repair_cycle_reopened', 'reopen_theory_repair_cycle'} else 'none'),
        'closure_state': body.get('closure_state') or ('closed' if selected_outcome in {'stable_science_campaign_closed', 'close_stable_science_campaign'} else 'none'),
        'waiting_blocker_state': body.get('waiting_blocker_state') or ('waiting' if selected_outcome and 'waiting' in str(selected_outcome) else 'resolved'),
        'before_campaign_cycle_memory_state': body.get('before_campaign_cycle_memory_state') or body.get('before_theory_memory_state') or 'unknown',
        'after_campaign_cycle_memory_state': body.get('after_campaign_cycle_memory_state') or body.get('after_theory_memory_state') or 'unknown',
        'checkpoint_boundary_state': 'repair' if leaks or third_party or body.get('checkpoint_boundary_state') == 'repair' else 'clean',
        'checkpoint_boundary_notes': list(body.get('checkpoint_boundary_notes') or ([] if not leaks else ['label leak detected'])),
        'label_leaks': leaks,
        'lineage': [f'message:{sender}'],
    }


def _upsert_row(rows, incoming):
    key = incoming.get('scenario_id') or incoming.get('campaign_strategy_outcome_id') or incoming.get('campaign_strategy_id') or incoming.get('hypothesis_id') or incoming.get('campaign_id')
    if not key:
        return
    current = rows.setdefault(str(key), {
        'campaign_strategy_outcome_id': incoming.get('campaign_strategy_outcome_id'),
        'campaign_strategy_id': incoming.get('campaign_strategy_id'),
        'frontier_outcome_id': incoming.get('frontier_outcome_id'),
        'frontier_id': incoming.get('frontier_id'),
        'hypothesis_id': incoming.get('hypothesis_id'),
        'campaign_id': incoming.get('campaign_id'),
        'family_id': incoming.get('family_id'),
        'scenario_id': incoming.get('scenario_id'),
        'selected_outcome': incoming.get('selected_outcome'),
        'planned_strategy': incoming.get('planned_strategy'),
        'source_evidence_used': [],
        'sibling_evidence_used': [],
        'promotion_state': 'none',
        'retirement_state': 'none',
        'repair_state': 'none',
        'closure_state': 'none',
        'waiting_blocker_state': 'waiting',
        'before_campaign_cycle_memory_state': 'unknown',
        'after_campaign_cycle_memory_state': 'unknown',
        'checkpoint_boundary_state': 'clean',
        'checkpoint_boundary_notes': [],
        'label_leaks': [],
        'lineage': [],
    })
    for field in ('campaign_strategy_outcome_id', 'campaign_strategy_id', 'frontier_outcome_id', 'frontier_id', 'hypothesis_id', 'campaign_id', 'family_id', 'scenario_id'):
        current[field] = current.get(field) or incoming.get(field)
    current['selected_outcome'] = _dominant_outcome(current.get('selected_outcome'), incoming.get('selected_outcome'))
    current['planned_strategy'] = current.get('planned_strategy') or incoming.get('planned_strategy')
    current['source_evidence_used'] = _unique_dicts(list(current.get('source_evidence_used') or []) + list(incoming.get('source_evidence_used') or []))
    current['sibling_evidence_used'] = _unique_dicts(list(current.get('sibling_evidence_used') or []) + list(incoming.get('sibling_evidence_used') or []))
    current['promotion_state'] = _dominant_state(current.get('promotion_state'), incoming.get('promotion_state'))
    current['retirement_state'] = _dominant_retirement(current.get('retirement_state'), incoming.get('retirement_state'))
    current['repair_state'] = _dominant_state(current.get('repair_state'), incoming.get('repair_state'))
    current['closure_state'] = _dominant_state(current.get('closure_state'), incoming.get('closure_state'))
    current['waiting_blocker_state'] = _dominant_waiting(current.get('waiting_blocker_state'), incoming.get('waiting_blocker_state'))
    current['before_campaign_cycle_memory_state'] = current.get('before_campaign_cycle_memory_state') or incoming.get('before_campaign_cycle_memory_state') or 'unknown'
    current['after_campaign_cycle_memory_state'] = _dominant_memory_state(current.get('after_campaign_cycle_memory_state'), incoming.get('after_campaign_cycle_memory_state'))
    current['checkpoint_boundary_state'] = 'repair' if current.get('checkpoint_boundary_state') == 'repair' or incoming.get('checkpoint_boundary_state') == 'repair' else 'clean'
    current['checkpoint_boundary_notes'] = _unique_strings(list(current.get('checkpoint_boundary_notes') or []) + list(incoming.get('checkpoint_boundary_notes') or []))
    current['label_leaks'] = _unique_strings(list(current.get('label_leaks') or []) + list(incoming.get('label_leaks') or []))
    current['lineage'] = _unique_strings(list(current.get('lineage') or []) + list(incoming.get('lineage') or []))


def _finalize_row(row):
    if row.get('label_leaks'):
        row['checkpoint_boundary_state'] = 'repair'
    row['cycle_strategy_hash'] = stable_digest({
        'campaign_strategy_outcome_id': row.get('campaign_strategy_outcome_id'),
        'scenario_id': row.get('scenario_id'),
        'outcome': row.get('selected_outcome'),
        'boundary': row.get('checkpoint_boundary_state'),
    })
    return row


def _select_cycle_strategy(*, rows, planned_keys, project_owned_boundary):
    for row in rows:
        if project_owned_boundary.get('third_party_checkpoint_used') or row.get('checkpoint_boundary_state') == 'repair':
            return _strategy(row, 'preserve_checkpoint_boundary', 'preserve_checkpoint_boundary', 'code_module')
    for row in rows:
        if row.get('selected_outcome') == 'validated_hypothesis_promoted_to_theory_memory' or row.get('promotion_state') == 'promoted':
            return _strategy(row, 'promote_validated_hypothesis_strategy_to_campaign_cycle_memory', 'record_campaign_cycle_memory_promotion', 'broadcast')
    for row in rows:
        if row.get('selected_outcome') == 'weak_hypothesis_route_retired' or row.get('retirement_state') in {'retired', 'blocked', 'weak'}:
            return _strategy(row, 'retire_weak_hypothesis_family', 'record_weak_family_retirement', 'broadcast')
    for row in rows:
        if row.get('selected_outcome') == 'funfun_formal_or_proof_clarification_received' or row.get('planned_strategy') == 'request_funfun_formal_or_proof_clarification':
            return _strategy(row, 'request_funfun_formal_or_proof_help', 'request_funfun_formal_or_proof_help', 'funfun')
    for row in rows:
        if row.get('selected_outcome') == 'code_simulation_or_primitive_capability_received' or row.get('planned_strategy') == 'request_code_simulation_or_primitive_capability':
            return _strategy(row, 'request_code_simulation_or_primitive_capability_help', 'request_code_simulation_or_primitive_capability_help', 'code_module')
    for row in rows:
        if row.get('selected_outcome') == 'language_terminology_or_protocol_clarification_received' or row.get('planned_strategy') == 'request_language_terminology_or_protocol_clarification':
            return _strategy(row, 'request_language_terminology_or_protocol_help', 'request_language_terminology_or_protocol_help', 'language_model_2')
    for row in rows:
        if row.get('selected_outcome') == 'theory_repair_cycle_reopened' or row.get('repair_state') == 'open':
            return _strategy(row, 'reopen_theory_repair_cycle', 'reopen_theory_repair_cycle', 'orchestrator')
    for row in rows:
        if row.get('selected_outcome') == 'stable_science_campaign_closed' or row.get('closure_state') in {'closed', 'stable'}:
            return _strategy(row, 'close_stable_science_campaign_cycle', 'close_stable_science_campaign_cycle', 'orchestrator')
    for row in rows:
        if row.get('selected_outcome') == 'next_frontier_hypothesis_scheduled' and _cycle_strategy_key_for(row, 'schedule_next_frontier_hypothesis_campaign') not in set(planned_keys):
            return _strategy(row, 'schedule_next_frontier_hypothesis_campaign', 'schedule_next_frontier_hypothesis_campaign', 'orchestrator')
    for row in rows:
        if row.get('selected_outcome') in {'planned_science_campaign_strategy_waiting_for_evidence', 'planned_science_campaign_strategy_waiting_for_evidence'}:
            return _strategy(row, 'schedule_next_frontier_hypothesis_campaign', 'schedule_next_frontier_hypothesis_campaign', 'orchestrator')
    for row in rows:
        if row.get('selected_outcome') == 'no_measurable_science_strategy_gain':
            return _strategy(row, 'record_no_measurable_science_cycle_gain', 'record_no_measurable_science_cycle_gain', 'orchestrator')
    return _noop_strategy('no science campaign cycle strategy selected')


def _strategy(row, selected_strategy, action, recipient):
    return {
        'selected_strategy': selected_strategy,
        'selected_action': action,
        'selected_recipient': validate_participant(recipient),
        'campaign_strategy_outcome_id': row.get('campaign_strategy_outcome_id'),
        'campaign_strategy_id': row.get('campaign_strategy_id'),
        'frontier_outcome_id': row.get('frontier_outcome_id'),
        'frontier_id': row.get('frontier_id'),
        'hypothesis_id': row.get('hypothesis_id'),
        'campaign_id': row.get('campaign_id'),
        'family_id': row.get('family_id') or row.get('hypothesis_id'),
        'scenario_id': row.get('scenario_id'),
        'target_hypothesis': row.get('hypothesis_id'),
        'target_campaign': row.get('campaign_id') or row.get('scenario_id'),
        'target_family': row.get('family_id') or row.get('hypothesis_id'),
        'required_followup_evidence': _required_followup(selected_strategy),
        'required_sibling_evidence': _required_sibling(selected_strategy),
        'source_evidence_used': list(row.get('source_evidence_used') or []),
        'sibling_evidence_used': list(row.get('sibling_evidence_used') or []),
        'campaign_cycle_memory_delta': _cycle_memory_delta(selected_strategy, row),
        'theory_memory_delta': _theory_memory_delta(selected_strategy, row),
        'promotion_state': row.get('promotion_state') or 'none',
        'retirement_state': row.get('retirement_state') or 'none',
        'repair_state': row.get('repair_state') or 'none',
        'closure_state': row.get('closure_state') or 'none',
        'waiting_blocker_state': row.get('waiting_blocker_state') or 'waiting',
        'before_campaign_cycle_memory_state': row.get('before_campaign_cycle_memory_state') or 'unknown',
        'after_campaign_cycle_memory_state': row.get('after_campaign_cycle_memory_state') or 'unknown',
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
        'campaign_strategy_outcome_id': None,
        'campaign_strategy_id': None,
        'frontier_outcome_id': None,
        'frontier_id': None,
        'hypothesis_id': None,
        'campaign_id': None,
        'family_id': None,
        'scenario_id': None,
        'target_hypothesis': None,
        'target_campaign': None,
        'target_family': None,
        'required_followup_evidence': [],
        'required_sibling_evidence': [],
        'source_evidence_used': [],
        'sibling_evidence_used': [],
        'campaign_cycle_memory_delta': reason,
        'theory_memory_delta': reason,
        'promotion_state': 'none',
        'retirement_state': 'none',
        'repair_state': 'none',
        'closure_state': 'none',
        'waiting_blocker_state': 'waiting',
        'before_campaign_cycle_memory_state': 'unknown',
        'after_campaign_cycle_memory_state': 'unknown',
        'checkpoint_boundary_state': 'clean',
        'checkpoint_boundary_notes': [],
        'no_overclaiming_proof': 'no project-owned checkpoint claim made',
        'label_leaks': [],
    }


def _required_followup(strategy):
    mapping = {
        'request_funfun_formal_or_proof_help': ['formal_or_proof_help'],
        'request_code_simulation_or_primitive_capability_help': ['simulation_or_primitive_capability_help'],
        'request_language_terminology_or_protocol_help': ['terminology_or_protocol_help'],
        'reopen_theory_repair_cycle': ['theory_repair_cycle'],
        'schedule_next_frontier_hypothesis_campaign': ['next_frontier_hypothesis_campaign'],
    }
    return mapping.get(strategy, [])


def _required_sibling(strategy):
    mapping = {
        'request_funfun_formal_or_proof_help': [{'recipient': 'funfun', 'evidence_gate': 'formal_or_proof_help'}],
        'request_code_simulation_or_primitive_capability_help': [{'recipient': 'code_module', 'evidence_gate': 'simulation_or_primitive_capability_help'}],
        'request_language_terminology_or_protocol_help': [{'recipient': 'language_model_2', 'evidence_gate': 'terminology_or_protocol_help'}],
    }
    return mapping.get(strategy, [])


def _cycle_memory_delta(strategy, row):
    if strategy == 'promote_validated_hypothesis_strategy_to_campaign_cycle_memory':
        return 'promote stable validated-hypothesis strategy into campaign-cycle memory row'
    if strategy == 'retire_weak_hypothesis_family':
        return 'retire weak hypothesis family from active campaign-cycle route'
    if strategy == 'close_stable_science_campaign_cycle':
        return 'close stable science campaign cycle'
    if strategy == 'record_no_measurable_science_cycle_gain':
        return 'record no safe science-cycle strategy gain'
    return row.get('selected_outcome') or strategy


def _theory_memory_delta(strategy, row):
    if strategy == 'promote_validated_hypothesis_strategy_to_campaign_cycle_memory':
        return 'theory-memory promotion evidence preserved as cycle strategy memory'
    if strategy == 'retire_weak_hypothesis_family':
        return 'weak hypothesis family marked retired for theory route'
    return row.get('selected_outcome') or strategy


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
        'promote_validated_hypothesis_strategy_to_campaign_cycle_memory': 'promote',
        'retire_weak_hypothesis_family': 'retire',
        'request_funfun_formal_or_proof_help': 'funfun',
        'request_code_simulation_or_primitive_capability_help': 'code',
        'request_language_terminology_or_protocol_help': 'language',
        'reopen_theory_repair_cycle': 'repair',
        'close_stable_science_campaign_cycle': 'close',
        'schedule_next_frontier_hypothesis_campaign': 'schedule',
        'record_no_measurable_science_cycle_gain': 'no_gain',
    }
    key = mapping.get(selected.get('selected_strategy'))
    if key:
        counts[key] += 1
    if selected.get('selected_strategy') == 'summarize_noop':
        counts['no_gain'] += 1
    if selected.get('waiting_blocker_state') == 'waiting' and selected.get('selected_strategy') != 'summarize_noop':
        counts['waiting'] += 1
    return counts


def _source_hash(strategy_outcome, strategy_source, frontier_outcome, frontier, campaign, theory_memory, hypothesis, experiment, module_chat, prior_cycle, runtime_memory):
    def digest_or_hash(value):
        if not value:
            return None
        return value.get('ledger_hash') or stable_digest(value)

    return stable_digest({
        'strategy_outcome': digest_or_hash(strategy_outcome),
        'strategy': digest_or_hash(strategy_source),
        'frontier_outcome': digest_or_hash(frontier_outcome),
        'frontier': digest_or_hash(frontier),
        'campaign': digest_or_hash(campaign),
        'theory_memory': digest_or_hash(theory_memory),
        'hypothesis': digest_or_hash(hypothesis),
        'experiment': digest_or_hash(experiment),
        'module_chat': digest_or_hash(module_chat),
        'prior_cycle': digest_or_hash(prior_cycle),
        'runtime_keys': sorted(runtime_memory.keys()),
    })


def _plain_rows(source, keys):
    rows = []
    for key in keys:
        values = source.get(key)
        if isinstance(values, list):
            rows.extend([dict(item) for item in values if isinstance(item, dict)])
    return rows


def _find_row(rows, outcome_id, scenario_id):
    for row in rows:
        if outcome_id and row.get('campaign_strategy_outcome_id') == outcome_id:
            return row
        if scenario_id and row.get('scenario_id') == scenario_id:
            return row
    return None


def _cycle_strategy_key(selected):
    return _cycle_strategy_key_for(selected, selected.get('selected_strategy'))


def _cycle_strategy_key_for(row, strategy):
    return stable_digest({
        'campaign_strategy_outcome_id': row.get('campaign_strategy_outcome_id'),
        'scenario_id': row.get('scenario_id'),
        'hypothesis_id': row.get('hypothesis_id'),
        'campaign_id': row.get('campaign_id'),
        'family_id': row.get('family_id'),
        'strategy': strategy,
    })


def _dominant_outcome(current, incoming):
    order = {
        'preserve_checkpoint_boundary': 11,
        'validated_hypothesis_promoted_to_theory_memory': 10,
        'weak_hypothesis_route_retired': 9,
        'funfun_formal_or_proof_clarification_received': 8,
        'code_simulation_or_primitive_capability_received': 7,
        'language_terminology_or_protocol_clarification_received': 6,
        'theory_repair_cycle_reopened': 5,
        'stable_science_campaign_closed': 4,
        'next_frontier_hypothesis_scheduled': 3,
        'planned_science_campaign_strategy_waiting_for_evidence': 2,
        'no_measurable_science_strategy_gain': 1,
        None: 0,
    }
    return current if order.get(current, 0) >= order.get(incoming, 0) else incoming


def _dominant_state(current, incoming):
    order = {'promoted': 5, 'closed': 5, 'stable': 5, 'open': 4, 'accepted': 4, 'updated': 4, 'scheduled': 3, 'none': 1, None: 0}
    return current if order.get(current, 0) >= order.get(incoming, 0) else incoming


def _dominant_retirement(current, incoming):
    order = {'retired': 5, 'blocked': 5, 'weak': 4, 'none': 1, None: 0}
    return current if order.get(current, 0) >= order.get(incoming, 0) else incoming


def _dominant_waiting(current, incoming):
    order = {'blocked': 3, 'waiting': 2, 'resolved': 1, None: 0}
    return current if order.get(current, 0) >= order.get(incoming, 0) else incoming


def _dominant_memory_state(current, incoming):
    order = {'promoted': 5, 'recorded': 4, 'accepted': 4, 'closed': 3, 'scheduled': 3, 'not_recorded': 1, 'unknown': 0, None: 0}
    return current if order.get(current, 0) >= order.get(incoming, 0) else incoming


def _first(values):
    return values[0] if values else None


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
