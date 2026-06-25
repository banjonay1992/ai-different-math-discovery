"""Science campaign cycle strategy outcome assessor.

This layer checks whether a planned campaign-cycle strategy actually changed
symbolic science/campaign/theory state. It is a plain-data assessment layer:
no empirical discovery claim, no sibling imports, and no checkpoint ownership
claim unless local status evidence verifies one.
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
from .science_campaign_cycle_strategy import SCIENCE_CAMPAIGN_CYCLE_STRATEGY_LEDGER_KIND
from .science_campaign_strategy import SCIENCE_CAMPAIGN_STRATEGY_LEDGER_KIND
from .science_campaign_strategy_outcome import SCIENCE_CAMPAIGN_STRATEGY_OUTCOME_LEDGER_KIND
from .science_theory_frontier import SCIENCE_THEORY_FRONTIER_LEDGER_KIND
from .science_theory_frontier_outcome import SCIENCE_THEORY_FRONTIER_OUTCOME_LEDGER_KIND


SCIENCE_CAMPAIGN_CYCLE_STRATEGY_OUTCOME_LEDGER_KIND = 'ai_different.science_campaign_cycle_strategy_outcome_ledger'


def empty_science_campaign_cycle_strategy_outcome_ledger() -> dict[str, Any]:
    return {
        'schema_version': 1,
        'ledger_kind': SCIENCE_CAMPAIGN_CYCLE_STRATEGY_OUTCOME_LEDGER_KIND,
        'processed_message_ids': [],
        'processed_source_hashes': [],
        'assessed_cycle_strategy_keys': [],
        'outcome_records': [],
        'cycle_strategy_rows': [],
        'outgoing_response_ids': [],
        'latest': {},
    }


def load_science_campaign_cycle_strategy_outcome_ledger(path: str | Path | None) -> dict[str, Any]:
    if not path or not Path(path).exists():
        return empty_science_campaign_cycle_strategy_outcome_ledger()
    with Path(path).open('r', encoding='utf-8') as handle:
        ledger = json.load(handle)
    return validate_science_campaign_cycle_strategy_outcome_ledger(ledger)


def write_science_campaign_cycle_strategy_outcome_ledger(path: str | Path, ledger: dict[str, Any]) -> Path:
    validated = validate_science_campaign_cycle_strategy_outcome_ledger(ledger)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(validated, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return output


def validate_science_campaign_cycle_strategy_outcome_ledger(ledger: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(ledger, dict):
        raise ValueError('science campaign cycle strategy outcome ledger must be a JSON object')
    if ledger.get('ledger_kind') != SCIENCE_CAMPAIGN_CYCLE_STRATEGY_OUTCOME_LEDGER_KIND:
        raise ValueError('science campaign cycle strategy outcome ledger has wrong ledger_kind')
    for field in (
        'processed_message_ids',
        'processed_source_hashes',
        'assessed_cycle_strategy_keys',
        'outgoing_response_ids',
    ):
        values = ledger.get(field)
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            raise ValueError(f'{field} must be strings')
    for field in ('outcome_records', 'cycle_strategy_rows'):
        values = ledger.get(field)
        if not isinstance(values, list) or not all(isinstance(item, dict) for item in values):
            raise ValueError(f'{field} must be objects')
    latest = ledger.get('latest', {})
    if not isinstance(latest, dict):
        raise ValueError('science campaign cycle strategy outcome latest must be an object')
    validated = {
        'schema_version': int(ledger.get('schema_version', 1) or 1),
        'ledger_kind': SCIENCE_CAMPAIGN_CYCLE_STRATEGY_OUTCOME_LEDGER_KIND,
        'processed_message_ids': _unique_strings(ledger['processed_message_ids']),
        'processed_source_hashes': _unique_strings(ledger['processed_source_hashes']),
        'assessed_cycle_strategy_keys': _unique_strings(ledger['assessed_cycle_strategy_keys']),
        'outcome_records': list(ledger['outcome_records']),
        'cycle_strategy_rows': list(ledger['cycle_strategy_rows']),
        'outgoing_response_ids': _unique_strings(ledger['outgoing_response_ids']),
        'latest': dict(latest),
    }
    if ledger.get('ledger_hash'):
        validated['ledger_hash'] = str(ledger['ledger_hash'])
    if ledger.get('artifact_path'):
        validated['artifact_path'] = str(ledger['artifact_path'])
    return validated


def read_science_campaign_cycle_strategy_outcome_transcript(path: str | Path | None) -> dict[str, Any]:
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
        raise ValueError('plain science campaign cycle strategy outcome input must be a JSON object')
    return value


def build_science_campaign_cycle_strategy_outcome_assessment(
    *,
    transcript_messages: list[dict[str, Any]],
    cycle_strategy_outcome_ledger: dict[str, Any],
    cycle_strategy_ledger: dict[str, Any] | None = None,
    strategy_outcome_ledger: dict[str, Any] | None = None,
    strategy_ledger: dict[str, Any] | None = None,
    frontier_outcome_ledger: dict[str, Any] | None = None,
    frontier_ledger: dict[str, Any] | None = None,
    campaign_ledger: dict[str, Any] | None = None,
    theory_memory_ledger: dict[str, Any] | None = None,
    campaign_cycle_memory_ledger: dict[str, Any] | None = None,
    hypothesis_ledger: dict[str, Any] | None = None,
    experiment_ledger: dict[str, Any] | None = None,
    module_chat_ledger: dict[str, Any] | None = None,
    prior_cycle_strategy_outcome_ledger: dict[str, Any] | None = None,
    runtime_memory_data: dict[str, Any] | None = None,
    runtime_memory_hash_state: dict[str, Any],
    project_owned_boundary: dict[str, Any],
    artifact_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    ledger = validate_science_campaign_cycle_strategy_outcome_ledger(cycle_strategy_outcome_ledger)
    cycle_strategy = _valid_cycle_strategy_or_empty(cycle_strategy_ledger or {})
    strategy_outcome = _valid_strategy_outcome_or_empty(strategy_outcome_ledger or {})
    strategy_source = _valid_strategy_or_empty(strategy_ledger or {})
    frontier_outcome = _valid_frontier_outcome_or_empty(frontier_outcome_ledger or {})
    frontier = _valid_frontier_or_empty(frontier_ledger or {})
    campaign = _valid_campaign_or_empty(campaign_ledger or {})
    theory_memory = _valid_plain_or_empty(theory_memory_ledger or {})
    campaign_cycle_memory = _valid_plain_or_empty(campaign_cycle_memory_ledger or {})
    hypothesis = _valid_plain_or_empty(hypothesis_ledger or {})
    experiment = _valid_plain_or_empty(experiment_ledger or {})
    module_chat = _valid_plain_or_empty(module_chat_ledger or {})
    prior_outcome = _valid_prior_outcome_or_empty(prior_cycle_strategy_outcome_ledger or {})
    runtime_memory = dict(runtime_memory_data or {})
    source_hash = _source_hash(
        cycle_strategy,
        strategy_outcome,
        strategy_source,
        frontier_outcome,
        frontier,
        campaign,
        theory_memory,
        campaign_cycle_memory,
        hypothesis,
        experiment,
        module_chat,
        prior_outcome,
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
    rows = _extract_outcome_rows(
        ledger['cycle_strategy_rows'],
        cycle_strategy,
        strategy_outcome,
        strategy_source,
        frontier_outcome,
        frontier,
        campaign,
        theory_memory,
        campaign_cycle_memory,
        hypothesis,
        experiment,
        module_chat,
        transcript_messages,
    )
    if not new_messages and not source_is_new:
        selected = _noop_outcome('no new science campaign cycle strategy outcome evidence or source ledger state')
    else:
        selected = _select_outcome(
            rows=rows,
            assessed_keys=ledger['assessed_cycle_strategy_keys'],
            project_owned_boundary=project_owned_boundary,
        )
    outcome_id = 'science_cycle_strategy_outcome_' + stable_digest({
        'source_hash': source_hash,
        'new_message_ids': [module_chat_message_id(message) for message in new_messages],
        'outcome': selected['selected_outcome'],
        'cycle_strategy_id': selected.get('campaign_cycle_strategy_id'),
        'scenario_id': selected.get('scenario_id'),
    })[:16]
    selected['campaign_cycle_strategy_outcome_id'] = outcome_id
    message = export_science_campaign_cycle_strategy_outcome_message(
        selected,
        rows=rows,
        runtime_memory_hash_state=runtime_memory_hash_state,
        project_owned_boundary=project_owned_boundary,
        source_hashes={
            'science_campaign_cycle_strategy_outcome_source_hash': source_hash,
            'cycle_strategy_ledger_hash': cycle_strategy.get('ledger_hash'),
            'strategy_outcome_ledger_hash': strategy_outcome.get('ledger_hash'),
            'strategy_ledger_hash': strategy_source.get('ledger_hash'),
            'frontier_outcome_ledger_hash': frontier_outcome.get('ledger_hash'),
            'frontier_ledger_hash': frontier.get('ledger_hash'),
            'campaign_ledger_hash': campaign.get('ledger_hash'),
            'theory_memory_ledger_hash': theory_memory.get('ledger_hash'),
            'campaign_cycle_memory_ledger_hash': campaign_cycle_memory.get('ledger_hash'),
            'hypothesis_ledger_hash': hypothesis.get('ledger_hash'),
            'experiment_ledger_hash': experiment.get('ledger_hash'),
            'module_chat_ledger_hash': module_chat.get('ledger_hash'),
            'prior_cycle_strategy_outcome_ledger_hash': prior_outcome.get('ledger_hash'),
        },
    )
    new_ids = [module_chat_message_id(message) for message in new_messages]
    outgoing_id = module_chat_message_id(message) if message else None
    assessed_key = _outcome_key(selected)
    if selected['selected_outcome'] != 'summarize_noop':
        ledger['assessed_cycle_strategy_keys'] = _unique_strings(list(ledger['assessed_cycle_strategy_keys']) + [assessed_key])
    if new_messages:
        ledger['processed_message_ids'] = _unique_strings(list(ledger['processed_message_ids']) + new_ids)
    if source_is_new:
        ledger['processed_source_hashes'] = _unique_strings(list(ledger['processed_source_hashes']) + [source_hash])
    if outgoing_id:
        ledger['outgoing_response_ids'] = _unique_strings(list(ledger['outgoing_response_ids']) + [outgoing_id])
    state_counts = _state_counts(selected)
    latest = {
        'campaign_cycle_strategy_outcome_id': outcome_id,
        'new_message_count': len(new_messages),
        'skipped_message_count': len(skipped_messages),
        'source_is_new': source_is_new,
        'state_counts': state_counts,
        'selected_outcome': selected['selected_outcome'],
        'selected_action': selected['selected_action'],
        'chosen_recipient': message.get('recipient') if message else None,
        'outbox_count': 1 if message else 0,
        'label_leaks': selected.get('label_leaks', []),
        'checkpoint_boundary_state': selected.get('checkpoint_boundary_state'),
        'runtime_memory_hash_state': dict(runtime_memory_hash_state),
        'runtime_memory_mutated': not bool(runtime_memory_hash_state.get('unchanged', True)),
    }
    record = {
        'campaign_cycle_strategy_outcome_id': outcome_id,
        'outcome_hash': stable_digest({'outcome_id': outcome_id, 'selected': selected}),
        'processed_message_ids': new_ids,
        'source_hash': source_hash,
        'campaign_cycle_strategy_id': selected.get('campaign_cycle_strategy_id'),
        'campaign_strategy_outcome_id': selected.get('campaign_strategy_outcome_id'),
        'campaign_strategy_id': selected.get('campaign_strategy_id'),
        'frontier_outcome_id': selected.get('frontier_outcome_id'),
        'frontier_id': selected.get('frontier_id'),
        'hypothesis_id': selected.get('hypothesis_id'),
        'campaign_id': selected.get('campaign_id'),
        'family_id': selected.get('family_id'),
        'selected_outcome': selected['selected_outcome'],
        'selected_action': selected['selected_action'],
        'selected_recipient': latest['chosen_recipient'],
        'before_campaign_cycle_memory_state': selected.get('before_campaign_cycle_memory_state'),
        'after_campaign_cycle_memory_state': selected.get('after_campaign_cycle_memory_state'),
        'promotion_state': selected.get('promotion_state'),
        'retirement_state': selected.get('retirement_state'),
        'repair_state': selected.get('repair_state'),
        'closure_state': selected.get('closure_state'),
        'schedule_state': selected.get('schedule_state'),
        'waiting_blocker_state': selected.get('waiting_blocker_state'),
        'observed_sibling_evidence': selected.get('observed_sibling_evidence'),
        'checkpoint_boundary_notes': selected.get('checkpoint_boundary_notes'),
        'no_overclaiming_proof': selected.get('no_overclaiming_proof'),
        'outgoing_response_id': outgoing_id,
    }
    ledger['cycle_strategy_rows'] = rows
    if new_messages or source_is_new or message is not None:
        ledger['outcome_records'] = list(ledger['outcome_records']) + [record]
    ledger['latest'] = latest
    if artifact_path is not None:
        ledger['artifact_path'] = str(Path(artifact_path))
    ledger['ledger_hash'] = stable_digest({key: value for key, value in ledger.items() if key != 'ledger_hash'})
    return ledger, message


def export_science_campaign_cycle_strategy_outcome_message(
    selected: dict[str, Any],
    *,
    rows: list[dict[str, Any]],
    runtime_memory_hash_state: dict[str, Any],
    project_owned_boundary: dict[str, Any],
    source_hashes: dict[str, Any],
    topic: str = 'ai_different.science_campaign_cycle_strategy_outcome',
) -> dict[str, Any] | None:
    if selected['selected_outcome'] == 'summarize_noop':
        return None
    recipient = validate_participant(selected.get('selected_recipient') or 'orchestrator')
    row = _find_row(rows, selected.get('campaign_cycle_strategy_id'), selected.get('scenario_id')) or {}
    leak_terms = label_leak_terms({'selected': selected, 'row': row})
    body = {
        'module': 'AI Different',
        'response_kind': 'science_campaign_cycle_strategy_outcome',
        'campaign_cycle_strategy_outcome_id': selected.get('campaign_cycle_strategy_outcome_id'),
        'campaign_cycle_strategy_id': selected.get('campaign_cycle_strategy_id'),
        'campaign_strategy_outcome_id': selected.get('campaign_strategy_outcome_id'),
        'campaign_strategy_id': selected.get('campaign_strategy_id'),
        'frontier_outcome_id': selected.get('frontier_outcome_id'),
        'frontier_id': selected.get('frontier_id'),
        'hypothesis_id': selected.get('hypothesis_id'),
        'campaign_id': selected.get('campaign_id'),
        'family_id': selected.get('family_id'),
        'scenario_id': selected.get('scenario_id'),
        'planned_cycle_strategy': selected.get('planned_cycle_strategy'),
        'selected_outcome': selected['selected_outcome'],
        'selected_action': selected['selected_action'],
        'selected_recipient': recipient,
        'target_hypothesis': selected.get('target_hypothesis'),
        'target_campaign': selected.get('target_campaign'),
        'target_family': selected.get('target_family'),
        'required_followup_evidence': selected.get('required_followup_evidence') or [],
        'source_evidence_used': selected.get('source_evidence_used') or [],
        'observed_sibling_evidence': selected.get('observed_sibling_evidence') or [],
        'campaign_cycle_memory_delta': selected.get('campaign_cycle_memory_delta'),
        'theory_memory_delta': selected.get('theory_memory_delta'),
        'before_campaign_cycle_memory_state': selected.get('before_campaign_cycle_memory_state'),
        'after_campaign_cycle_memory_state': selected.get('after_campaign_cycle_memory_state'),
        'promotion_state': selected.get('promotion_state'),
        'retirement_state': selected.get('retirement_state'),
        'repair_state': selected.get('repair_state'),
        'closure_state': selected.get('closure_state'),
        'schedule_state': selected.get('schedule_state'),
        'waiting_blocker_state': selected.get('waiting_blocker_state'),
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
            'campaign_cycle_strategy_outcome_id': body['campaign_cycle_strategy_outcome_id'],
            'campaign_cycle_strategy_id': body['campaign_cycle_strategy_id'],
            'hypothesis_id': body['hypothesis_id'],
            'campaign_id': body['campaign_id'],
            'family_id': body['family_id'],
            'selected_outcome': body['selected_outcome'],
            'selected_action': body['selected_action'],
            'label_clean': body['label_clean'],
            'no_sibling_imports': True,
        },
        tags=['ai_different', 'science_campaign_cycle_strategy_outcome', body['selected_outcome'], 'label_clean' if body['label_clean'] else 'label_review_needed'],
    )


def write_science_campaign_cycle_strategy_outcome_outbox_jsonl(path: str | Path, message: dict[str, Any] | None) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text((json.dumps(message, sort_keys=True) + '\n') if message is not None else '', encoding='utf-8')
    return output


def _valid_cycle_strategy_or_empty(ledger):
    if not ledger:
        return {'ledger_kind': SCIENCE_CAMPAIGN_CYCLE_STRATEGY_LEDGER_KIND, 'cycle_strategy_records': [], 'cycle_strategy_rows': [], 'ledger_hash': None}
    if ledger.get('ledger_kind') != SCIENCE_CAMPAIGN_CYCLE_STRATEGY_LEDGER_KIND:
        raise ValueError('science campaign cycle strategy ledger has wrong ledger_kind')
    return dict(ledger)


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
        raise ValueError('plain science campaign cycle strategy outcome ledger must be a JSON object')
    return dict(ledger)


def _valid_prior_outcome_or_empty(ledger):
    if not ledger:
        return {}
    if ledger.get('ledger_kind') != SCIENCE_CAMPAIGN_CYCLE_STRATEGY_OUTCOME_LEDGER_KIND:
        raise ValueError('prior science campaign cycle strategy outcome ledger has wrong ledger_kind')
    return validate_science_campaign_cycle_strategy_outcome_ledger(ledger)


def _extract_outcome_rows(existing, cycle_strategy, strategy_outcome, strategy_ledger, frontier_outcome, frontier, campaign, theory_memory, campaign_cycle_memory, hypothesis, experiment, module_chat, messages):
    rows: dict[str, dict[str, Any]] = {}
    for row in list(existing or []):
        _upsert_row(rows, row)
    for record in list(cycle_strategy.get('cycle_strategy_records') or []):
        _upsert_row(rows, _row_from_cycle_strategy_record(record))
    for row in list(cycle_strategy.get('cycle_strategy_rows') or []):
        _upsert_row(rows, _row_from_cycle_strategy_row(row))
    for record in list(strategy_outcome.get('outcome_records') or []):
        _upsert_row(rows, _row_from_strategy_outcome_record(record))
    for record in list(strategy_ledger.get('strategy_records') or []):
        _upsert_row(rows, _row_from_generic_record(record, 'strategy_record'))
    for record in list(frontier_outcome.get('outcome_records') or []):
        _upsert_row(rows, _row_from_generic_record(record, 'frontier_outcome_record'))
    for record in list(frontier.get('frontier_records') or []):
        _upsert_row(rows, _row_from_generic_record(record, 'frontier_record'))
    for row in _plain_rows(campaign, ('campaigns', 'campaign_rows', 'campaign_records')):
        _upsert_row(rows, _row_from_campaign(row))
    for row in _plain_rows(theory_memory, ('theory_memory_rows', 'theory_records', 'theories')):
        _upsert_row(rows, _row_from_memory(row, 'theory_memory'))
    for row in _plain_rows(campaign_cycle_memory, ('campaign_cycle_memory_rows', 'cycle_strategy_rows', 'cycle_memory_rows')):
        _upsert_row(rows, _row_from_memory(row, 'campaign_cycle_memory'))
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


def _row_from_cycle_strategy_record(record):
    return {
        'campaign_cycle_strategy_id': record.get('campaign_cycle_strategy_id'),
        'campaign_strategy_outcome_id': record.get('campaign_strategy_outcome_id'),
        'campaign_strategy_id': record.get('campaign_strategy_id'),
        'frontier_outcome_id': record.get('frontier_outcome_id'),
        'frontier_id': record.get('frontier_id'),
        'hypothesis_id': record.get('hypothesis_id'),
        'campaign_id': record.get('campaign_id'),
        'family_id': record.get('family_id'),
        'scenario_id': record.get('scenario_id') or record.get('campaign_id') or record.get('hypothesis_id'),
        'planned_cycle_strategy': record.get('selected_strategy'),
        'source_evidence_used': list(record.get('source_evidence_used') or []),
        'observed_sibling_evidence': list(record.get('observed_sibling_evidence') or record.get('required_sibling_evidence') or []),
        'promotion_state': record.get('promotion_state') or 'none',
        'retirement_state': record.get('retirement_state') or 'none',
        'repair_state': record.get('repair_state') or 'none',
        'closure_state': record.get('closure_state') or 'none',
        'schedule_state': 'scheduled' if record.get('selected_strategy') == 'schedule_next_frontier_hypothesis_campaign' else 'none',
        'waiting_blocker_state': 'waiting' if record.get('required_sibling_evidence') else record.get('waiting_blocker_state') or 'resolved',
        'before_campaign_cycle_memory_state': record.get('before_campaign_cycle_memory_state') or 'unknown',
        'after_campaign_cycle_memory_state': record.get('after_campaign_cycle_memory_state') or 'unknown',
        'checkpoint_boundary_state': 'repair' if record.get('checkpoint_boundary_notes') else 'clean',
        'checkpoint_boundary_notes': list(record.get('checkpoint_boundary_notes') or []),
        'label_leaks': list(record.get('label_leaks') or []),
        'lineage': ['cycle_strategy_record'],
    }


def _row_from_cycle_strategy_row(row):
    return {
        'campaign_cycle_strategy_id': row.get('campaign_cycle_strategy_id'),
        'campaign_strategy_outcome_id': row.get('campaign_strategy_outcome_id'),
        'campaign_strategy_id': row.get('campaign_strategy_id') or row.get('strategy_id'),
        'frontier_outcome_id': row.get('frontier_outcome_id'),
        'frontier_id': row.get('frontier_id'),
        'hypothesis_id': row.get('hypothesis_id'),
        'campaign_id': row.get('campaign_id'),
        'family_id': row.get('family_id'),
        'scenario_id': row.get('scenario_id') or row.get('campaign_id') or row.get('hypothesis_id'),
        'planned_cycle_strategy': row.get('selected_strategy') or row.get('planned_cycle_strategy'),
        'source_evidence_used': list(row.get('source_evidence_used') or []),
        'observed_sibling_evidence': list(row.get('observed_sibling_evidence') or row.get('sibling_evidence_used') or []),
        'promotion_state': row.get('promotion_state') or 'none',
        'retirement_state': row.get('retirement_state') or 'none',
        'repair_state': row.get('repair_state') or 'none',
        'closure_state': row.get('closure_state') or 'none',
        'schedule_state': row.get('schedule_state') or 'none',
        'waiting_blocker_state': row.get('waiting_blocker_state') or 'waiting',
        'before_campaign_cycle_memory_state': row.get('before_campaign_cycle_memory_state') or 'unknown',
        'after_campaign_cycle_memory_state': row.get('after_campaign_cycle_memory_state') or 'unknown',
        'checkpoint_boundary_state': row.get('checkpoint_boundary_state') or 'clean',
        'checkpoint_boundary_notes': list(row.get('checkpoint_boundary_notes') or []),
        'label_leaks': list(row.get('label_leaks') or []),
        'lineage': ['cycle_strategy_row'],
    }


def _row_from_strategy_outcome_record(record):
    return {
        'campaign_strategy_outcome_id': record.get('campaign_strategy_outcome_id'),
        'campaign_strategy_id': record.get('strategy_id'),
        'frontier_outcome_id': record.get('frontier_outcome_id'),
        'frontier_id': record.get('frontier_id'),
        'hypothesis_id': record.get('hypothesis_id'),
        'campaign_id': record.get('campaign_id'),
        'family_id': record.get('family_id'),
        'scenario_id': record.get('scenario_id') or record.get('campaign_id') or record.get('hypothesis_id'),
        'planned_cycle_strategy': _strategy_from_prior_outcome(record.get('selected_outcome')),
        'source_evidence_used': list(record.get('source_evidence_used') or []),
        'observed_sibling_evidence': list(record.get('observed_sibling_evidence') or []),
        'promotion_state': record.get('promotion_state') or 'none',
        'retirement_state': record.get('retirement_state') or 'none',
        'repair_state': record.get('repair_state') or 'none',
        'closure_state': record.get('closure_state') or 'none',
        'schedule_state': 'scheduled' if record.get('selected_outcome') == 'next_frontier_hypothesis_scheduled' else 'none',
        'waiting_blocker_state': record.get('waiting_blocker_state') or 'waiting',
        'before_campaign_cycle_memory_state': record.get('before_campaign_cycle_memory_state') or record.get('before_theory_memory_state') or 'unknown',
        'after_campaign_cycle_memory_state': record.get('after_campaign_cycle_memory_state') or record.get('after_theory_memory_state') or 'unknown',
        'checkpoint_boundary_state': 'repair' if record.get('checkpoint_boundary_notes') else 'clean',
        'checkpoint_boundary_notes': list(record.get('checkpoint_boundary_notes') or []),
        'label_leaks': list(record.get('label_leaks') or []),
        'lineage': ['strategy_outcome_record'],
    }


def _row_from_generic_record(record, lineage):
    return {
        'campaign_strategy_id': record.get('strategy_id') or record.get('campaign_strategy_id'),
        'frontier_outcome_id': record.get('frontier_outcome_id'),
        'frontier_id': record.get('frontier_id'),
        'hypothesis_id': record.get('hypothesis_id') or _first(record.get('hypothesis_ids') or []),
        'campaign_id': record.get('campaign_id') or _first(record.get('campaign_ids') or []),
        'family_id': record.get('family_id'),
        'scenario_id': record.get('scenario_id') or _first(record.get('scenario_ids') or []),
        'planned_cycle_strategy': record.get('selected_strategy') or record.get('selected_theory_move'),
        'source_evidence_used': [],
        'observed_sibling_evidence': list(record.get('observed_sibling_evidence') or record.get('requested_sibling_evidence') or []),
        'promotion_state': record.get('promotion_state') or 'none',
        'retirement_state': record.get('retirement_state') or record.get('retirement_block_state') or 'none',
        'repair_state': record.get('repair_state') or ('open' if record.get('boundary_checkpoint_state') == 'repair' else 'none'),
        'closure_state': record.get('closure_state') or 'none',
        'schedule_state': record.get('schedule_state') or 'none',
        'waiting_blocker_state': record.get('waiting_blocker_state') or 'waiting',
        'before_campaign_cycle_memory_state': record.get('before_campaign_cycle_memory_state') or 'unknown',
        'after_campaign_cycle_memory_state': record.get('after_campaign_cycle_memory_state') or 'unknown',
        'checkpoint_boundary_state': record.get('checkpoint_boundary_state') or record.get('boundary_checkpoint_state') or 'clean',
        'checkpoint_boundary_notes': list(record.get('checkpoint_boundary_notes') or record.get('boundary_notes') or []),
        'label_leaks': list(record.get('label_leaks') or []),
        'lineage': [lineage],
    }


def _row_from_campaign(row):
    state = str(row.get('campaign_state') or row.get('cycle_state') or row.get('status') or '')
    return {
        'campaign_id': row.get('campaign_id') or row.get('id'),
        'hypothesis_id': row.get('hypothesis_id'),
        'family_id': row.get('family_id'),
        'scenario_id': row.get('scenario_id') or row.get('campaign_id') or row.get('id'),
        'planned_cycle_strategy': row.get('selected_strategy') or row.get('planned_cycle_strategy'),
        'source_evidence_used': [],
        'observed_sibling_evidence': [],
        'promotion_state': row.get('promotion_state') or 'none',
        'retirement_state': row.get('retirement_state') or 'none',
        'repair_state': row.get('repair_state') or 'none',
        'closure_state': row.get('closure_state') or ('closed' if state in {'stable', 'closed'} else 'none'),
        'schedule_state': row.get('schedule_state') or ('scheduled' if state == 'scheduled' else 'none'),
        'waiting_blocker_state': row.get('waiting_blocker_state') or 'resolved',
        'before_campaign_cycle_memory_state': row.get('before_campaign_cycle_memory_state') or 'unknown',
        'after_campaign_cycle_memory_state': row.get('after_campaign_cycle_memory_state') or state or 'unknown',
        'checkpoint_boundary_state': row.get('checkpoint_boundary_state') or 'clean',
        'checkpoint_boundary_notes': list(row.get('checkpoint_boundary_notes') or []),
        'label_leaks': list(row.get('label_leaks') or []),
        'lineage': ['campaign'],
    }


def _row_from_memory(row, lineage):
    status = str(row.get('status') or row.get('cycle_state') or 'recorded')
    return {
        'campaign_cycle_strategy_id': row.get('campaign_cycle_strategy_id'),
        'campaign_strategy_outcome_id': row.get('campaign_strategy_outcome_id'),
        'hypothesis_id': row.get('hypothesis_id'),
        'campaign_id': row.get('campaign_id'),
        'family_id': row.get('family_id'),
        'scenario_id': row.get('scenario_id') or row.get('hypothesis_id') or row.get('campaign_id'),
        'planned_cycle_strategy': row.get('planned_cycle_strategy') or 'promote_validated_hypothesis_strategy_to_campaign_cycle_memory',
        'source_evidence_used': [{'source': lineage, 'status': status}],
        'observed_sibling_evidence': [],
        'promotion_state': 'promoted' if status in {'recorded', 'promoted', 'accepted'} else 'none',
        'retirement_state': row.get('retirement_state') or 'none',
        'repair_state': row.get('repair_state') or 'none',
        'closure_state': row.get('closure_state') or 'none',
        'schedule_state': row.get('schedule_state') or 'none',
        'waiting_blocker_state': 'resolved',
        'before_campaign_cycle_memory_state': row.get('before_campaign_cycle_memory_state') or 'not_recorded',
        'after_campaign_cycle_memory_state': row.get('after_campaign_cycle_memory_state') or status,
        'checkpoint_boundary_state': row.get('checkpoint_boundary_state') or 'clean',
        'checkpoint_boundary_notes': list(row.get('checkpoint_boundary_notes') or []),
        'label_leaks': list(row.get('label_leaks') or []),
        'lineage': [lineage],
    }


def _row_from_hypothesis(row):
    state = str(row.get('state') or row.get('status') or '')
    return {
        'hypothesis_id': row.get('hypothesis_id') or row.get('id'),
        'campaign_id': row.get('campaign_id'),
        'family_id': row.get('family_id'),
        'scenario_id': row.get('scenario_id') or row.get('hypothesis_id') or row.get('id'),
        'planned_cycle_strategy': row.get('planned_cycle_strategy') or row.get('selected_strategy'),
        'source_evidence_used': [],
        'observed_sibling_evidence': [],
        'promotion_state': 'promoted' if state in {'validated', 'promoted'} else 'none',
        'retirement_state': 'retired' if state in {'retired', 'weak', 'blocked'} else 'none',
        'repair_state': 'open' if state == 'repair' else 'none',
        'closure_state': 'none',
        'schedule_state': 'none',
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
        'campaign_cycle_strategy_id': row.get('campaign_cycle_strategy_id'),
        'hypothesis_id': row.get('hypothesis_id'),
        'campaign_id': row.get('campaign_id'),
        'family_id': row.get('family_id'),
        'scenario_id': row.get('scenario_id') or row.get('request_id'),
        'planned_cycle_strategy': row.get('planned_cycle_strategy') or row.get('selected_strategy'),
        'source_evidence_used': [],
        'observed_sibling_evidence': sibling,
        'promotion_state': 'none',
        'retirement_state': 'none',
        'repair_state': 'none',
        'closure_state': 'none',
        'schedule_state': 'none',
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
    strategy = body.get('selected_strategy') or body.get('planned_cycle_strategy') or evidence.get('selected_strategy')
    sibling = list(body.get('observed_sibling_evidence') or body.get('sibling_evidence_used') or [])
    gate = body.get('evidence_gate') or evidence.get('evidence_gate')
    status = str(body.get('status') or evidence.get('status') or '').lower()
    if sender in {'funfun', 'code_module', 'language_model_2'} and (gate or status):
        sibling.append({'sender': sender, 'evidence_gate': str(gate or 'advisory'), 'status': status or 'received'})
    leaks = label_leak_terms({'body': body, 'evidence': evidence})
    third_party = bool(body.get('third_party_checkpoint_used') or evidence.get('third_party_checkpoint_used'))
    return {
        'campaign_cycle_strategy_id': body.get('campaign_cycle_strategy_id') or evidence.get('campaign_cycle_strategy_id'),
        'campaign_strategy_outcome_id': body.get('campaign_strategy_outcome_id') or evidence.get('campaign_strategy_outcome_id'),
        'campaign_strategy_id': body.get('campaign_strategy_id') or body.get('strategy_id') or evidence.get('strategy_id'),
        'frontier_outcome_id': body.get('frontier_outcome_id') or evidence.get('frontier_outcome_id'),
        'frontier_id': body.get('frontier_id') or evidence.get('frontier_id'),
        'hypothesis_id': body.get('hypothesis_id') or evidence.get('hypothesis_id'),
        'campaign_id': body.get('campaign_id') or evidence.get('campaign_id'),
        'family_id': body.get('family_id') or evidence.get('family_id'),
        'scenario_id': str(scenario_id) if scenario_id else None,
        'planned_cycle_strategy': strategy,
        'source_evidence_used': list(body.get('source_evidence_used') or []),
        'observed_sibling_evidence': sibling,
        'promotion_state': body.get('promotion_state') or ('promoted' if strategy == 'promote_validated_hypothesis_strategy_to_campaign_cycle_memory' else 'none'),
        'retirement_state': body.get('retirement_state') or ('retired' if strategy == 'retire_weak_hypothesis_family' else 'none'),
        'repair_state': body.get('repair_state') or ('open' if strategy == 'reopen_theory_repair_cycle' else 'none'),
        'closure_state': body.get('closure_state') or ('closed' if strategy == 'close_stable_science_campaign_cycle' else 'none'),
        'schedule_state': body.get('schedule_state') or ('scheduled' if strategy == 'schedule_next_frontier_hypothesis_campaign' else 'none'),
        'waiting_blocker_state': body.get('waiting_blocker_state') or ('waiting' if strategy and strategy.startswith('request_') and not sibling else 'resolved'),
        'before_campaign_cycle_memory_state': body.get('before_campaign_cycle_memory_state') or 'unknown',
        'after_campaign_cycle_memory_state': body.get('after_campaign_cycle_memory_state') or 'unknown',
        'checkpoint_boundary_state': 'repair' if leaks or third_party or body.get('checkpoint_boundary_state') == 'repair' else 'clean',
        'checkpoint_boundary_notes': list(body.get('checkpoint_boundary_notes') or ([] if not leaks else ['label leak detected'])),
        'label_leaks': leaks,
        'lineage': [f'message:{sender}'],
    }


def _upsert_row(rows, incoming):
    key = incoming.get('scenario_id') or incoming.get('campaign_cycle_strategy_id') or incoming.get('campaign_strategy_outcome_id') or incoming.get('hypothesis_id') or incoming.get('campaign_id')
    if not key:
        return
    current = rows.setdefault(str(key), {
        'campaign_cycle_strategy_id': incoming.get('campaign_cycle_strategy_id'),
        'campaign_strategy_outcome_id': incoming.get('campaign_strategy_outcome_id'),
        'campaign_strategy_id': incoming.get('campaign_strategy_id'),
        'frontier_outcome_id': incoming.get('frontier_outcome_id'),
        'frontier_id': incoming.get('frontier_id'),
        'hypothesis_id': incoming.get('hypothesis_id'),
        'campaign_id': incoming.get('campaign_id'),
        'family_id': incoming.get('family_id'),
        'scenario_id': incoming.get('scenario_id'),
        'planned_cycle_strategy': incoming.get('planned_cycle_strategy'),
        'source_evidence_used': [],
        'observed_sibling_evidence': [],
        'promotion_state': 'none',
        'retirement_state': 'none',
        'repair_state': 'none',
        'closure_state': 'none',
        'schedule_state': 'none',
        'waiting_blocker_state': 'waiting',
        'before_campaign_cycle_memory_state': 'unknown',
        'after_campaign_cycle_memory_state': 'unknown',
        'checkpoint_boundary_state': 'clean',
        'checkpoint_boundary_notes': [],
        'label_leaks': [],
        'lineage': [],
    })
    for field in ('campaign_cycle_strategy_id', 'campaign_strategy_outcome_id', 'campaign_strategy_id', 'frontier_outcome_id', 'frontier_id', 'hypothesis_id', 'campaign_id', 'family_id', 'scenario_id'):
        current[field] = current.get(field) or incoming.get(field)
    current['planned_cycle_strategy'] = current.get('planned_cycle_strategy') or incoming.get('planned_cycle_strategy')
    current['source_evidence_used'] = _unique_dicts(list(current.get('source_evidence_used') or []) + list(incoming.get('source_evidence_used') or []))
    current['observed_sibling_evidence'] = _unique_dicts(list(current.get('observed_sibling_evidence') or []) + list(incoming.get('observed_sibling_evidence') or []))
    current['promotion_state'] = _dominant_state(current.get('promotion_state'), incoming.get('promotion_state'))
    current['retirement_state'] = _dominant_retirement(current.get('retirement_state'), incoming.get('retirement_state'))
    current['repair_state'] = _dominant_state(current.get('repair_state'), incoming.get('repair_state'))
    current['closure_state'] = _dominant_state(current.get('closure_state'), incoming.get('closure_state'))
    current['schedule_state'] = _dominant_state(current.get('schedule_state'), incoming.get('schedule_state'))
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
    if row.get('planned_cycle_strategy') == 'promote_validated_hypothesis_strategy_to_campaign_cycle_memory':
        row['promotion_state'] = 'promoted'
    if row.get('planned_cycle_strategy') == 'retire_weak_hypothesis_family':
        row['retirement_state'] = 'retired'
    if row.get('planned_cycle_strategy') == 'reopen_theory_repair_cycle':
        row['repair_state'] = 'open'
    if row.get('planned_cycle_strategy') == 'close_stable_science_campaign_cycle':
        row['closure_state'] = 'closed'
    if row.get('planned_cycle_strategy') == 'schedule_next_frontier_hypothesis_campaign':
        row['schedule_state'] = 'scheduled'
    row['cycle_strategy_outcome_hash'] = stable_digest({
        'campaign_cycle_strategy_id': row.get('campaign_cycle_strategy_id'),
        'scenario_id': row.get('scenario_id'),
        'planned_cycle_strategy': row.get('planned_cycle_strategy'),
        'boundary': row.get('checkpoint_boundary_state'),
    })
    return row


def _select_outcome(*, rows, assessed_keys, project_owned_boundary):
    for row in rows:
        if project_owned_boundary.get('third_party_checkpoint_used') or row.get('checkpoint_boundary_state') == 'repair':
            return _outcome(row, 'preserve_checkpoint_boundary', 'preserve_checkpoint_boundary', 'code_module')
    for row in rows:
        if row.get('planned_cycle_strategy') == 'promote_validated_hypothesis_strategy_to_campaign_cycle_memory' or row.get('promotion_state') == 'promoted':
            return _outcome(row, 'campaign_cycle_memory_promoted', 'record_campaign_cycle_memory_promotion', 'broadcast')
    for row in rows:
        if row.get('planned_cycle_strategy') == 'retire_weak_hypothesis_family' or row.get('retirement_state') in {'retired', 'blocked', 'weak'}:
            return _outcome(row, 'weak_hypothesis_family_retired', 'record_weak_family_retirement', 'broadcast')
    for row in rows:
        if row.get('planned_cycle_strategy') == 'request_funfun_formal_or_proof_help' and _has_sibling(row, 'funfun'):
            return _outcome(row, 'funfun_formal_or_proof_help_received', 'record_funfun_help_received', 'broadcast')
    for row in rows:
        if row.get('planned_cycle_strategy') == 'request_code_simulation_or_primitive_capability_help' and _has_sibling(row, 'code_module'):
            return _outcome(row, 'code_simulation_or_primitive_capability_help_received', 'record_code_help_received', 'broadcast')
    for row in rows:
        if row.get('planned_cycle_strategy') == 'request_language_terminology_or_protocol_help' and _has_sibling(row, 'language_model_2'):
            return _outcome(row, 'language_terminology_or_protocol_help_received', 'record_language_help_received', 'broadcast')
    for row in rows:
        if row.get('planned_cycle_strategy') == 'reopen_theory_repair_cycle' or row.get('repair_state') == 'open':
            return _outcome(row, 'theory_repair_cycle_reopened', 'record_theory_repair_reopened', 'orchestrator')
    for row in rows:
        if row.get('planned_cycle_strategy') == 'close_stable_science_campaign_cycle' or row.get('closure_state') in {'closed', 'stable'}:
            return _outcome(row, 'stable_science_campaign_cycle_closed', 'record_stable_campaign_cycle_closed', 'orchestrator')
    for row in rows:
        if row.get('planned_cycle_strategy') == 'schedule_next_frontier_hypothesis_campaign' or row.get('schedule_state') == 'scheduled':
            return _outcome(row, 'next_frontier_hypothesis_campaign_scheduled', 'record_next_frontier_hypothesis_campaign_scheduled', 'orchestrator')
    for row in rows:
        if row.get('planned_cycle_strategy') == 'preserve_checkpoint_boundary' and row.get('checkpoint_boundary_state') == 'clean':
            return _outcome(row, 'checkpoint_boundary_policy_strengthened', 'record_checkpoint_boundary_policy_strengthened', 'orchestrator')
    for row in rows:
        if _waiting_for_evidence(row) and _outcome_key_for(row, 'planned_science_campaign_cycle_strategy_waiting_for_evidence') not in set(assessed_keys):
            return _outcome(row, 'planned_science_campaign_cycle_strategy_waiting_for_evidence', 'wait_for_science_campaign_cycle_strategy_evidence', 'orchestrator')
    for row in rows:
        if row.get('planned_cycle_strategy') == 'record_no_measurable_science_cycle_gain':
            return _outcome(row, 'no_measurable_science_campaign_cycle_strategy_gain', 'record_no_measurable_science_campaign_cycle_strategy_gain', 'orchestrator')
    return _noop_outcome('no science campaign cycle strategy outcome selected')


def _outcome(row, selected_outcome, action, recipient):
    return {
        'selected_outcome': selected_outcome,
        'selected_action': action,
        'selected_recipient': validate_participant(recipient),
        'campaign_cycle_strategy_id': row.get('campaign_cycle_strategy_id'),
        'campaign_strategy_outcome_id': row.get('campaign_strategy_outcome_id'),
        'campaign_strategy_id': row.get('campaign_strategy_id'),
        'frontier_outcome_id': row.get('frontier_outcome_id'),
        'frontier_id': row.get('frontier_id'),
        'hypothesis_id': row.get('hypothesis_id'),
        'campaign_id': row.get('campaign_id'),
        'family_id': row.get('family_id') or row.get('hypothesis_id'),
        'scenario_id': row.get('scenario_id'),
        'planned_cycle_strategy': row.get('planned_cycle_strategy'),
        'target_hypothesis': row.get('hypothesis_id'),
        'target_campaign': row.get('campaign_id') or row.get('scenario_id'),
        'target_family': row.get('family_id') or row.get('hypothesis_id'),
        'required_followup_evidence': _required_followup(selected_outcome),
        'source_evidence_used': list(row.get('source_evidence_used') or []),
        'observed_sibling_evidence': list(row.get('observed_sibling_evidence') or []),
        'campaign_cycle_memory_delta': _cycle_memory_delta(selected_outcome, row),
        'theory_memory_delta': _theory_memory_delta(selected_outcome, row),
        'before_campaign_cycle_memory_state': row.get('before_campaign_cycle_memory_state') or 'unknown',
        'after_campaign_cycle_memory_state': row.get('after_campaign_cycle_memory_state') or 'unknown',
        'promotion_state': row.get('promotion_state') or 'none',
        'retirement_state': row.get('retirement_state') or 'none',
        'repair_state': row.get('repair_state') or 'none',
        'closure_state': row.get('closure_state') or 'none',
        'schedule_state': row.get('schedule_state') or 'none',
        'waiting_blocker_state': row.get('waiting_blocker_state') or 'waiting',
        'checkpoint_boundary_state': row.get('checkpoint_boundary_state') or 'clean',
        'checkpoint_boundary_notes': list(row.get('checkpoint_boundary_notes') or []),
        'no_overclaiming_proof': 'no local-owned checkpoint claim made unless status capsule verifies it',
        'label_leaks': list(row.get('label_leaks') or []),
    }


def _noop_outcome(reason):
    return {
        'selected_outcome': 'summarize_noop',
        'selected_action': 'summarize_noop',
        'selected_recipient': None,
        'campaign_cycle_strategy_id': None,
        'campaign_strategy_outcome_id': None,
        'campaign_strategy_id': None,
        'frontier_outcome_id': None,
        'frontier_id': None,
        'hypothesis_id': None,
        'campaign_id': None,
        'family_id': None,
        'scenario_id': None,
        'planned_cycle_strategy': None,
        'target_hypothesis': None,
        'target_campaign': None,
        'target_family': None,
        'required_followup_evidence': [],
        'source_evidence_used': [],
        'observed_sibling_evidence': [],
        'campaign_cycle_memory_delta': reason,
        'theory_memory_delta': reason,
        'before_campaign_cycle_memory_state': 'unknown',
        'after_campaign_cycle_memory_state': 'unknown',
        'promotion_state': 'none',
        'retirement_state': 'none',
        'repair_state': 'none',
        'closure_state': 'none',
        'schedule_state': 'none',
        'waiting_blocker_state': 'waiting',
        'checkpoint_boundary_state': 'clean',
        'checkpoint_boundary_notes': [],
        'no_overclaiming_proof': 'no project-owned checkpoint claim made',
        'label_leaks': [],
    }


def _required_followup(outcome):
    mapping = {
        'planned_science_campaign_cycle_strategy_waiting_for_evidence': ['cycle_strategy_return_evidence'],
        'theory_repair_cycle_reopened': ['repair_cycle_outcome'],
    }
    return mapping.get(outcome, [])


def _cycle_memory_delta(outcome, row):
    if outcome == 'campaign_cycle_memory_promoted':
        return 'campaign-cycle memory promotion observed'
    if outcome == 'weak_hypothesis_family_retired':
        return 'weak hypothesis family retirement observed'
    if outcome == 'stable_science_campaign_cycle_closed':
        return 'stable science campaign cycle closure observed'
    if outcome == 'checkpoint_boundary_policy_strengthened':
        return 'checkpoint boundary policy strengthened without claiming checkpoint ownership'
    if outcome == 'no_measurable_science_campaign_cycle_strategy_gain':
        return 'record no safe science campaign-cycle strategy gain'
    return row.get('planned_cycle_strategy') or outcome


def _theory_memory_delta(outcome, row):
    if outcome == 'campaign_cycle_memory_promoted':
        return 'validated cycle strategy preserved for theory-memory handoff'
    if outcome == 'weak_hypothesis_family_retired':
        return 'weak family marked retired for theory route'
    return row.get('planned_cycle_strategy') or outcome


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
        'strengthen': 0,
        'waiting': 0,
        'no_gain': 0,
    }
    mapping = {
        'preserve_checkpoint_boundary': 'boundary',
        'campaign_cycle_memory_promoted': 'promote',
        'weak_hypothesis_family_retired': 'retire',
        'funfun_formal_or_proof_help_received': 'funfun',
        'code_simulation_or_primitive_capability_help_received': 'code',
        'language_terminology_or_protocol_help_received': 'language',
        'theory_repair_cycle_reopened': 'repair',
        'stable_science_campaign_cycle_closed': 'close',
        'next_frontier_hypothesis_campaign_scheduled': 'schedule',
        'checkpoint_boundary_policy_strengthened': 'strengthen',
        'planned_science_campaign_cycle_strategy_waiting_for_evidence': 'waiting',
        'no_measurable_science_campaign_cycle_strategy_gain': 'no_gain',
    }
    key = mapping.get(selected.get('selected_outcome'))
    if key:
        counts[key] += 1
    if selected.get('selected_outcome') == 'summarize_noop':
        counts['no_gain'] += 1
    return counts


def _strategy_from_prior_outcome(outcome):
    mapping = {
        'validated_hypothesis_promoted_to_theory_memory': 'promote_validated_hypothesis_strategy_to_campaign_cycle_memory',
        'weak_hypothesis_route_retired': 'retire_weak_hypothesis_family',
        'funfun_formal_or_proof_clarification_received': 'request_funfun_formal_or_proof_help',
        'code_simulation_or_primitive_capability_received': 'request_code_simulation_or_primitive_capability_help',
        'language_terminology_or_protocol_clarification_received': 'request_language_terminology_or_protocol_help',
        'theory_repair_cycle_reopened': 'reopen_theory_repair_cycle',
        'stable_science_campaign_closed': 'close_stable_science_campaign_cycle',
        'next_frontier_hypothesis_scheduled': 'schedule_next_frontier_hypothesis_campaign',
        'no_measurable_science_strategy_gain': 'record_no_measurable_science_cycle_gain',
        'preserve_checkpoint_boundary': 'preserve_checkpoint_boundary',
    }
    return mapping.get(outcome)


def _has_sibling(row, sender):
    return any(item.get('sender') == sender and str(item.get('status') or '') for item in row.get('observed_sibling_evidence') or [])


def _waiting_for_evidence(row):
    strategy = row.get('planned_cycle_strategy')
    if strategy in {'record_no_measurable_science_cycle_gain', 'preserve_checkpoint_boundary'}:
        return False
    return (
        strategy in {
            'request_funfun_formal_or_proof_help',
            'request_code_simulation_or_primitive_capability_help',
            'request_language_terminology_or_protocol_help',
        }
        or row.get('waiting_blocker_state') == 'waiting'
    )


def _source_hash(cycle_strategy, strategy_outcome, strategy_source, frontier_outcome, frontier, campaign, theory_memory, campaign_cycle_memory, hypothesis, experiment, module_chat, prior_outcome, runtime_memory):
    def digest_or_hash(value):
        if not value:
            return None
        return value.get('ledger_hash') or stable_digest(value)

    return stable_digest({
        'cycle_strategy': digest_or_hash(cycle_strategy),
        'strategy_outcome': digest_or_hash(strategy_outcome),
        'strategy': digest_or_hash(strategy_source),
        'frontier_outcome': digest_or_hash(frontier_outcome),
        'frontier': digest_or_hash(frontier),
        'campaign': digest_or_hash(campaign),
        'theory_memory': digest_or_hash(theory_memory),
        'campaign_cycle_memory': digest_or_hash(campaign_cycle_memory),
        'hypothesis': digest_or_hash(hypothesis),
        'experiment': digest_or_hash(experiment),
        'module_chat': digest_or_hash(module_chat),
        'prior_outcome': digest_or_hash(prior_outcome),
        'runtime_keys': sorted(runtime_memory.keys()),
    })


def _plain_rows(source, keys):
    rows = []
    for key in keys:
        values = source.get(key)
        if isinstance(values, list):
            rows.extend([dict(item) for item in values if isinstance(item, dict)])
    return rows


def _find_row(rows, cycle_strategy_id, scenario_id):
    for row in rows:
        if cycle_strategy_id and row.get('campaign_cycle_strategy_id') == cycle_strategy_id:
            return row
        if scenario_id and row.get('scenario_id') == scenario_id:
            return row
    return None


def _outcome_key(selected):
    return _outcome_key_for(selected, selected.get('selected_outcome'))


def _outcome_key_for(row, outcome):
    return stable_digest({
        'campaign_cycle_strategy_id': row.get('campaign_cycle_strategy_id'),
        'scenario_id': row.get('scenario_id'),
        'hypothesis_id': row.get('hypothesis_id'),
        'campaign_id': row.get('campaign_id'),
        'family_id': row.get('family_id'),
        'outcome': outcome,
    })


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
