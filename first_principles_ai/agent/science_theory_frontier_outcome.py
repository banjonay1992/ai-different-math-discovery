"""Science theory-frontier outcome assessor.

This layer checks whether a symbolic theory-frontier move actually changed
plain-data theory state. It records management evidence only: no real-world
science proof, no sibling imports, and no project-owned checkpoint claim unless
the status capsule verifies one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .campaign_outcome_assessor import CAMPAIGN_OUTCOME_LEDGER_KIND
from .campaign_planner import CAMPAIGN_LEDGER_KIND
from .cross_module_adjudicator import ADJUDICATOR_LEDGER_KIND
from .evidence_scorecard import SCORECARD_LEDGER_KIND
from .experiment_agenda import AGENDA_LEDGER_KIND
from .experiment_contracts import CONTRACT_LEDGER_KIND, validate_evaluator_ledger
from .family_outcome_evaluator import EVALUATOR_LEDGER_KIND
from .hypothesis_lifecycle import LIFECYCLE_LEDGER_KIND
from .module_chat_adapter import (
    build_module_chat_message,
    label_leak_terms,
    module_chat_message_id,
    stable_digest,
    validate_module_chat_message,
    validate_participant,
)
from .science_action_outcome_assessor import SCIENCE_ACTION_OUTCOME_LEDGER_KIND
from .science_benefit_evaluator import SCIENCE_BENEFIT_LEDGER_KIND
from .science_campaign_action_planner import SCIENCE_ACTION_LEDGER_KIND
from .science_theory_frontier import (
    SCIENCE_THEORY_FRONTIER_LEDGER_KIND,
    validate_science_theory_frontier_ledger,
)


SCIENCE_THEORY_FRONTIER_OUTCOME_LEDGER_KIND = 'ai_different.science_theory_frontier_outcome_ledger'


def empty_science_theory_frontier_outcome_ledger() -> dict[str, Any]:
    return {
        'schema_version': 1,
        'ledger_kind': SCIENCE_THEORY_FRONTIER_OUTCOME_LEDGER_KIND,
        'processed_message_ids': [],
        'processed_source_hashes': [],
        'assessed_frontier_keys': [],
        'outcome_records': [],
        'frontier_rows': [],
        'outgoing_response_ids': [],
        'latest': {},
    }


def load_science_theory_frontier_outcome_ledger(path: str | Path | None) -> dict[str, Any]:
    if not path or not Path(path).exists():
        return empty_science_theory_frontier_outcome_ledger()
    with Path(path).open('r', encoding='utf-8') as handle:
        ledger = json.load(handle)
    return validate_science_theory_frontier_outcome_ledger(ledger)


def write_science_theory_frontier_outcome_ledger(path: str | Path, ledger: dict[str, Any]) -> Path:
    validated = validate_science_theory_frontier_outcome_ledger(ledger)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(validated, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return output


def validate_science_theory_frontier_outcome_ledger(ledger: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(ledger, dict):
        raise ValueError('science theory frontier outcome ledger must be a JSON object')
    if ledger.get('ledger_kind') != SCIENCE_THEORY_FRONTIER_OUTCOME_LEDGER_KIND:
        raise ValueError('science theory frontier outcome ledger has wrong ledger_kind')
    for field in (
        'processed_message_ids',
        'processed_source_hashes',
        'assessed_frontier_keys',
        'outgoing_response_ids',
    ):
        values = ledger.get(field)
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            raise ValueError(f'{field} must be strings')
    for field in ('outcome_records', 'frontier_rows'):
        values = ledger.get(field)
        if not isinstance(values, list) or not all(isinstance(item, dict) for item in values):
            raise ValueError(f'{field} must be objects')
    latest = ledger.get('latest', {})
    if not isinstance(latest, dict):
        raise ValueError('science theory frontier outcome latest must be an object')
    validated = {
        'schema_version': int(ledger.get('schema_version', 1) or 1),
        'ledger_kind': SCIENCE_THEORY_FRONTIER_OUTCOME_LEDGER_KIND,
        'processed_message_ids': _unique_strings(ledger['processed_message_ids']),
        'processed_source_hashes': _unique_strings(ledger['processed_source_hashes']),
        'assessed_frontier_keys': _unique_strings(ledger['assessed_frontier_keys']),
        'outcome_records': list(ledger['outcome_records']),
        'frontier_rows': list(ledger['frontier_rows']),
        'outgoing_response_ids': _unique_strings(ledger['outgoing_response_ids']),
        'latest': dict(latest),
    }
    if ledger.get('ledger_hash'):
        validated['ledger_hash'] = str(ledger['ledger_hash'])
    if ledger.get('artifact_path'):
        validated['artifact_path'] = str(ledger['artifact_path'])
    return validated


def read_science_theory_frontier_outcome_transcript(path: str | Path | None) -> dict[str, Any]:
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
        raise ValueError('plain science theory frontier outcome input must be a JSON object')
    return value


def build_science_theory_frontier_outcome_assessment(
    *,
    transcript_messages: list[dict[str, Any]],
    frontier_outcome_ledger: dict[str, Any],
    frontier_ledger: dict[str, Any] | None = None,
    action_outcome_ledger: dict[str, Any] | None = None,
    action_ledger: dict[str, Any] | None = None,
    benefit_ledger: dict[str, Any] | None = None,
    evaluator_ledger: dict[str, Any] | None = None,
    outcome_ledger: dict[str, Any] | None = None,
    contract_ledger: dict[str, Any] | None = None,
    adjudicator_ledger: dict[str, Any] | None = None,
    agenda_ledger: dict[str, Any] | None = None,
    lifecycle_ledger: dict[str, Any] | None = None,
    scorecard_ledger: dict[str, Any] | None = None,
    campaign_ledger: dict[str, Any] | None = None,
    campaign_outcome_ledger: dict[str, Any] | None = None,
    theory_memory_ledger: dict[str, Any] | None = None,
    prior_frontier_outcome_ledger: dict[str, Any] | None = None,
    sibling_frontier_outcome_ledgers: dict[str, Any] | None = None,
    runtime_memory_data: dict[str, Any] | None = None,
    runtime_memory_hash_state: dict[str, Any],
    project_owned_boundary: dict[str, Any],
    artifact_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    ledger = validate_science_theory_frontier_outcome_ledger(frontier_outcome_ledger)
    frontier = _valid_frontier_or_empty(frontier_ledger or {})
    action_outcome = _valid_action_outcome_or_empty(action_outcome_ledger or {})
    action_source = _valid_action_or_empty(action_ledger or {})
    benefit = _valid_benefit_or_empty(benefit_ledger or {})
    evaluator = _valid_evaluator_or_empty(evaluator_ledger or {})
    prior_outcome = _valid_prior_outcome_or_empty(outcome_ledger or {})
    contracts = _valid_contract_or_empty(contract_ledger or {})
    adjudicator = _valid_adjudicator_or_empty(adjudicator_ledger or {})
    agenda = _valid_agenda_or_empty(agenda_ledger or {})
    lifecycle = _valid_lifecycle_or_empty(lifecycle_ledger or {})
    scorecard = _valid_scorecard_or_empty(scorecard_ledger or {})
    campaign = _valid_campaign_or_empty(campaign_ledger or {})
    campaign_outcome = _valid_campaign_outcome_or_empty(campaign_outcome_ledger or {})
    theory_memory = _valid_plain_or_empty(theory_memory_ledger or {})
    prior_frontier_outcome = _valid_prior_frontier_outcome_or_empty(prior_frontier_outcome_ledger or {})
    sibling = dict(sibling_frontier_outcome_ledgers or {})
    runtime_memory = dict(runtime_memory_data or {})
    source_hash = _source_hash(
        frontier,
        action_outcome,
        action_source,
        benefit,
        evaluator,
        prior_outcome,
        contracts,
        adjudicator,
        agenda,
        lifecycle,
        scorecard,
        campaign,
        campaign_outcome,
        theory_memory,
        prior_frontier_outcome,
        sibling,
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
    rows = _extract_frontier_outcome_rows(
        ledger['frontier_rows'],
        frontier,
        theory_memory,
        runtime_memory,
        transcript_messages,
    )
    if not new_messages and not source_is_new:
        selected = _noop_outcome('no new science theory-frontier outcome evidence or source ledger state')
    else:
        selected = _select_frontier_outcome(
            rows=rows,
            assessed_keys=ledger['assessed_frontier_keys'],
            project_owned_boundary=project_owned_boundary,
        )
    frontier_outcome_id = 'science_frontier_outcome_' + stable_digest({
        'source_hash': source_hash,
        'new_message_ids': [module_chat_message_id(message) for message in new_messages],
        'outcome': selected['selected_outcome'],
        'frontier_id': selected.get('frontier_id'),
        'scenario_id': selected.get('scenario_id'),
    })[:16]
    selected['frontier_outcome_id'] = frontier_outcome_id
    message = export_science_theory_frontier_outcome_message(
        selected,
        rows=rows,
        runtime_memory_hash_state=runtime_memory_hash_state,
        project_owned_boundary=project_owned_boundary,
        source_hashes={
            'science_theory_frontier_outcome_source_hash': source_hash,
            'frontier_ledger_hash': frontier.get('ledger_hash'),
            'action_outcome_ledger_hash': action_outcome.get('ledger_hash'),
            'action_ledger_hash': action_source.get('ledger_hash'),
            'benefit_ledger_hash': benefit.get('ledger_hash'),
            'evaluator_ledger_hash': evaluator.get('ledger_hash'),
            'outcome_ledger_hash': prior_outcome.get('ledger_hash'),
            'contract_ledger_hash': contracts.get('ledger_hash'),
            'adjudicator_ledger_hash': adjudicator.get('ledger_hash'),
            'agenda_ledger_hash': agenda.get('ledger_hash'),
            'lifecycle_ledger_hash': lifecycle.get('ledger_hash'),
            'scorecard_ledger_hash': scorecard.get('ledger_hash'),
            'campaign_ledger_hash': campaign.get('ledger_hash'),
            'campaign_outcome_ledger_hash': campaign_outcome.get('ledger_hash'),
            'theory_memory_ledger_hash': theory_memory.get('ledger_hash'),
            'prior_frontier_outcome_ledger_hash': prior_frontier_outcome.get('ledger_hash'),
        },
    )
    new_ids = [module_chat_message_id(message) for message in new_messages]
    outgoing_id = module_chat_message_id(message) if message else None
    assessed_key = _frontier_outcome_key(selected)
    if selected['selected_outcome'] != 'summarize_noop':
        ledger['assessed_frontier_keys'] = _unique_strings(list(ledger['assessed_frontier_keys']) + [assessed_key])
    if new_messages:
        ledger['processed_message_ids'] = _unique_strings(list(ledger['processed_message_ids']) + new_ids)
    if source_is_new:
        ledger['processed_source_hashes'] = _unique_strings(list(ledger['processed_source_hashes']) + [source_hash])
    if outgoing_id:
        ledger['outgoing_response_ids'] = _unique_strings(list(ledger['outgoing_response_ids']) + [outgoing_id])
    state_counts = _state_counts(selected, rows)
    latest = {
        'frontier_outcome_id': frontier_outcome_id,
        'new_message_count': len(new_messages),
        'skipped_message_count': len(skipped_messages),
        'source_is_new': source_is_new,
        'state_counts': state_counts,
        'selected_outcome': selected['selected_outcome'],
        'selected_action': selected['selected_action'],
        'chosen_recipient': message.get('recipient') if message else None,
        'outbox_count': 1 if message else 0,
        'label_leaks': selected.get('label_leaks', []),
        'runtime_memory_hash_state': dict(runtime_memory_hash_state),
        'runtime_memory_mutated': not bool(runtime_memory_hash_state.get('unchanged', True)),
    }
    record = {
        'frontier_outcome_id': frontier_outcome_id,
        'outcome_hash': stable_digest({'frontier_outcome_id': frontier_outcome_id, 'selected': selected}),
        'processed_message_ids': new_ids,
        'source_hash': source_hash,
        'frontier_ids': _unique_strings([row.get('frontier_id') for row in rows if row.get('frontier_id')]),
        'source_outcome_ids': _unique_strings([row.get('source_outcome_id') for row in rows if row.get('source_outcome_id')]),
        'action_ids': _unique_strings([row.get('action_id') for row in rows if row.get('action_id')]),
        'benefit_ids': _unique_strings([row.get('benefit_id') for row in rows if row.get('benefit_id')]),
        'scenario_ids': _unique_strings([row.get('scenario_id') for row in rows if row.get('scenario_id')]),
        'hypothesis_ids': _unique_strings([row.get('hypothesis_id') for row in rows if row.get('hypothesis_id')]),
        'planned_theory_move': selected.get('planned_theory_move'),
        'selected_outcome': selected['selected_outcome'],
        'selected_recipient': latest['chosen_recipient'],
        'observed_theory_evidence': selected.get('observed_theory_evidence'),
        'observed_sibling_evidence': selected.get('observed_sibling_evidence'),
        'before_hypothesis_state': selected.get('before_hypothesis_state'),
        'after_hypothesis_state': selected.get('after_hypothesis_state'),
        'theory_memory_state': selected.get('theory_memory_state'),
        'promotion_state': selected.get('promotion_state'),
        'refinement_state': selected.get('refinement_state'),
        'retirement_block_state': selected.get('retirement_block_state'),
        'campaign_frontier_state': selected.get('campaign_frontier_state'),
        'waiting_blocker_state': selected.get('waiting_blocker_state'),
        'boundary_checkpoint_state': selected.get('boundary_checkpoint_state'),
        'outgoing_response_id': outgoing_id,
        'runtime_memory_hash_state': dict(runtime_memory_hash_state),
        'no_overclaiming_proof': selected.get('no_overclaiming_proof'),
        'source_ledger_hashes': message['body']['source_ledger_hashes'] if message else {},
    }
    ledger['frontier_rows'] = rows
    if new_messages or source_is_new or message is not None:
        ledger['outcome_records'] = list(ledger['outcome_records']) + [record]
    ledger['latest'] = latest
    if artifact_path is not None:
        ledger['artifact_path'] = str(Path(artifact_path))
    ledger['ledger_hash'] = stable_digest({key: value for key, value in ledger.items() if key != 'ledger_hash'})
    return ledger, message


def export_science_theory_frontier_outcome_message(
    selected: dict[str, Any],
    *,
    rows: list[dict[str, Any]],
    runtime_memory_hash_state: dict[str, Any],
    project_owned_boundary: dict[str, Any],
    source_hashes: dict[str, Any],
    topic: str = 'ai_different.science_theory_frontier_outcome',
) -> dict[str, Any] | None:
    if selected['selected_outcome'] == 'summarize_noop':
        return None
    recipient = validate_participant(selected.get('selected_recipient') or 'orchestrator')
    row = _find_row(rows, selected.get('frontier_id'), selected.get('scenario_id')) or {}
    leak_terms = label_leak_terms({'selected': selected, 'row': row})
    body = {
        'module': 'AI Different',
        'response_kind': 'science_theory_frontier_outcome',
        'frontier_outcome_id': selected.get('frontier_outcome_id'),
        'frontier_id': selected.get('frontier_id'),
        'source_outcome_id': selected.get('source_outcome_id'),
        'action_id': selected.get('action_id'),
        'benefit_id': selected.get('benefit_id'),
        'scenario_id': selected.get('scenario_id'),
        'hypothesis_id': selected.get('hypothesis_id'),
        'planned_theory_move': selected.get('planned_theory_move'),
        'selected_outcome': selected['selected_outcome'],
        'selected_action': selected['selected_action'],
        'selected_recipient': recipient,
        'observed_theory_evidence': selected.get('observed_theory_evidence') or [],
        'observed_sibling_evidence': selected.get('observed_sibling_evidence') or [],
        'before_hypothesis_state': selected.get('before_hypothesis_state'),
        'after_hypothesis_state': selected.get('after_hypothesis_state'),
        'theory_memory_state': selected.get('theory_memory_state'),
        'promotion_state': selected.get('promotion_state'),
        'refinement_state': selected.get('refinement_state'),
        'retirement_block_state': selected.get('retirement_block_state'),
        'campaign_frontier_state': selected.get('campaign_frontier_state'),
        'waiting_blocker_state': selected.get('waiting_blocker_state'),
        'boundary_checkpoint_state': selected.get('boundary_checkpoint_state'),
        'boundary_notes': selected.get('boundary_notes') or [],
        'before_after_delta': selected.get('before_after_delta'),
        'no_overclaiming_proof': selected.get('no_overclaiming_proof'),
        'recommended_action': selected.get('recommended_action') or selected['selected_action'],
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
            'frontier_outcome_id': body['frontier_outcome_id'],
            'frontier_id': body['frontier_id'],
            'scenario_id': body['scenario_id'],
            'hypothesis_id': body['hypothesis_id'],
            'selected_outcome': body['selected_outcome'],
            'selected_action': body['selected_action'],
            'runtime_memory_hash_state': body['runtime_memory_hash_state'],
            'runtime_memory_mutated': body['runtime_memory_mutated'],
            'label_clean': body['label_clean'],
            'no_sibling_imports': True,
        },
        tags=['ai_different', 'science_theory_frontier_outcome', body['selected_outcome'], 'label_clean' if body['label_clean'] else 'label_review_needed'],
    )


def write_science_theory_frontier_outcome_outbox_jsonl(path: str | Path, message: dict[str, Any] | None) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text((json.dumps(message, sort_keys=True) + '\n') if message is not None else '', encoding='utf-8')
    return output


def _valid_frontier_or_empty(ledger):
    if not ledger:
        return {'ledger_kind': SCIENCE_THEORY_FRONTIER_LEDGER_KIND, 'frontier_records': [], 'frontier_rows': [], 'ledger_hash': None}
    if ledger.get('ledger_kind') != SCIENCE_THEORY_FRONTIER_LEDGER_KIND:
        raise ValueError('science theory frontier ledger has wrong ledger_kind')
    return validate_science_theory_frontier_ledger(ledger)


def _valid_action_outcome_or_empty(ledger):
    if not ledger:
        return {'ledger_kind': SCIENCE_ACTION_OUTCOME_LEDGER_KIND, 'outcome_records': [], 'action_rows': [], 'ledger_hash': None}
    if ledger.get('ledger_kind') != SCIENCE_ACTION_OUTCOME_LEDGER_KIND:
        raise ValueError('science action outcome ledger has wrong ledger_kind')
    return dict(ledger)


def _valid_action_or_empty(ledger):
    if not ledger:
        return {'ledger_kind': SCIENCE_ACTION_LEDGER_KIND, 'action_records': [], 'scenario_rows': [], 'ledger_hash': None}
    if ledger.get('ledger_kind') != SCIENCE_ACTION_LEDGER_KIND:
        raise ValueError('science action ledger has wrong ledger_kind')
    return dict(ledger)


def _valid_benefit_or_empty(ledger):
    if not ledger:
        return {'ledger_kind': SCIENCE_BENEFIT_LEDGER_KIND, 'benefit_records': [], 'scenario_records': [], 'ledger_hash': None}
    if ledger.get('ledger_kind') != SCIENCE_BENEFIT_LEDGER_KIND:
        raise ValueError('benefit ledger has wrong ledger_kind')
    return dict(ledger)


def _valid_evaluator_or_empty(ledger):
    if not ledger:
        return {}
    if ledger.get('ledger_kind') != EVALUATOR_LEDGER_KIND:
        raise ValueError('evaluator ledger has wrong ledger_kind')
    return validate_evaluator_ledger(ledger)


def _valid_prior_outcome_or_empty(ledger):
    if not ledger:
        return {}
    if ledger.get('ledger_kind') not in {EVALUATOR_LEDGER_KIND, 'ai_different.outcome_ledger'}:
        raise ValueError('outcome ledger has wrong ledger_kind')
    return dict(ledger)


def _valid_contract_or_empty(ledger):
    if not ledger:
        return {'ledger_kind': CONTRACT_LEDGER_KIND, 'contracts': [], 'ledger_hash': None}
    if ledger.get('ledger_kind') != CONTRACT_LEDGER_KIND:
        raise ValueError('contract ledger has wrong ledger_kind')
    return dict(ledger)


def _valid_adjudicator_or_empty(ledger):
    if not ledger:
        return {'ledger_kind': ADJUDICATOR_LEDGER_KIND, 'contract_states': [], 'ledger_hash': None}
    if ledger.get('ledger_kind') != ADJUDICATOR_LEDGER_KIND:
        raise ValueError('adjudicator ledger has wrong ledger_kind')
    return dict(ledger)


def _valid_agenda_or_empty(ledger):
    if not ledger:
        return {'ledger_kind': AGENDA_LEDGER_KIND, 'hypotheses': [], 'ledger_hash': None}
    if ledger.get('ledger_kind') != AGENDA_LEDGER_KIND:
        raise ValueError('agenda ledger has wrong ledger_kind')
    return dict(ledger)


def _valid_lifecycle_or_empty(ledger):
    if not ledger:
        return {'ledger_kind': LIFECYCLE_LEDGER_KIND, 'hypotheses': [], 'ledger_hash': None}
    if ledger.get('ledger_kind') != LIFECYCLE_LEDGER_KIND:
        raise ValueError('lifecycle ledger has wrong ledger_kind')
    return dict(ledger)


def _valid_scorecard_or_empty(ledger):
    if not ledger:
        return {'ledger_kind': SCORECARD_LEDGER_KIND, 'scorecards': [], 'ledger_hash': None}
    if ledger.get('ledger_kind') != SCORECARD_LEDGER_KIND:
        raise ValueError('scorecard ledger has wrong ledger_kind')
    return dict(ledger)


def _valid_campaign_or_empty(ledger):
    if not ledger:
        return {'ledger_kind': CAMPAIGN_LEDGER_KIND, 'campaigns': [], 'ledger_hash': None}
    if ledger.get('ledger_kind') != CAMPAIGN_LEDGER_KIND:
        raise ValueError('campaign ledger has wrong ledger_kind')
    return dict(ledger)


def _valid_campaign_outcome_or_empty(ledger):
    if not ledger:
        return {'ledger_kind': CAMPAIGN_OUTCOME_LEDGER_KIND, 'outcomes': [], 'ledger_hash': None}
    if ledger.get('ledger_kind') != CAMPAIGN_OUTCOME_LEDGER_KIND:
        raise ValueError('campaign outcome ledger has wrong ledger_kind')
    return dict(ledger)


def _valid_plain_or_empty(ledger):
    if not ledger:
        return {}
    if not isinstance(ledger, dict):
        raise ValueError('theory memory ledger must be a JSON object')
    return dict(ledger)


def _valid_prior_frontier_outcome_or_empty(ledger):
    if not ledger:
        return {}
    if ledger.get('ledger_kind') != SCIENCE_THEORY_FRONTIER_OUTCOME_LEDGER_KIND:
        raise ValueError('prior theory frontier outcome ledger has wrong ledger_kind')
    return validate_science_theory_frontier_outcome_ledger(ledger)


def _extract_frontier_outcome_rows(existing, frontier_ledger, theory_memory_ledger, runtime_memory_data, messages):
    rows: dict[str, dict[str, Any]] = {}
    for row in list(existing or []):
        _upsert_row(rows, row)
    for record in list(frontier_ledger.get('frontier_records') or []):
        _upsert_row(rows, _row_from_frontier_record(record))
    for row in list(frontier_ledger.get('frontier_rows') or []):
        _upsert_row(rows, _row_from_frontier_row(row))
    for memory_row in _memory_rows(theory_memory_ledger, runtime_memory_data):
        _upsert_row(rows, _row_from_theory_memory(memory_row))
    for message in messages:
        _upsert_row(rows, _row_from_message(message))
    return [_finalize_row(row) for row in rows.values()]


def _row_from_frontier_record(record):
    return {
        'frontier_id': record.get('frontier_id'),
        'source_outcome_id': _first(record.get('outcome_ids') or []),
        'action_id': _first(record.get('action_ids') or []),
        'benefit_id': _first(record.get('benefit_ids') or []),
        'scenario_id': _first(record.get('scenario_ids') or []),
        'hypothesis_id': _first(record.get('hypothesis_ids') or []),
        'planned_theory_move': record.get('selected_theory_move'),
        'observed_theory_evidence': [],
        'observed_sibling_evidence': list(record.get('observed_sibling_evidence') or []),
        'before_hypothesis_state': record.get('before_hypothesis_state') or 'planned',
        'after_hypothesis_state': record.get('after_hypothesis_state') or 'waiting',
        'theory_memory_state': 'promoted' if record.get('selected_theory_move') == 'promote_supported_hypothesis_to_theory_memory' else 'not_recorded',
        'promotion_state': 'promoted' if record.get('selected_theory_move') == 'promote_supported_hypothesis_to_theory_memory' else 'none',
        'refinement_state': record.get('refinement_state') or 'none',
        'retirement_block_state': record.get('retirement_block_state') or 'none',
        'campaign_frontier_state': 'scheduled' if record.get('selected_theory_move') == 'schedule_next_campaign_frontier_check' else 'none',
        'waiting_blocker_state': record.get('waiting_blocker_state') or 'waiting',
        'boundary_checkpoint_state': record.get('boundary_checkpoint_state') or 'clean',
        'boundary_notes': [],
        'label_leaks': [],
        'lineage': ['frontier_record'],
    }


def _row_from_frontier_row(row):
    return {
        'frontier_id': row.get('frontier_id'),
        'source_outcome_id': row.get('outcome_id'),
        'action_id': row.get('action_id'),
        'benefit_id': row.get('benefit_id'),
        'scenario_id': row.get('scenario_id'),
        'hypothesis_id': row.get('hypothesis_id'),
        'planned_theory_move': row.get('selected_theory_move') or row.get('planned_theory_move'),
        'observed_theory_evidence': list(row.get('observed_theory_evidence') or []),
        'observed_sibling_evidence': list(row.get('observed_sibling_evidence') or []),
        'before_hypothesis_state': row.get('before_hypothesis_state') or 'planned',
        'after_hypothesis_state': row.get('after_hypothesis_state') or 'waiting',
        'theory_memory_state': row.get('theory_memory_state') or 'not_recorded',
        'promotion_state': row.get('promotion_state') or 'none',
        'refinement_state': row.get('refinement_state') or 'none',
        'retirement_block_state': row.get('retirement_block_state') or 'none',
        'campaign_frontier_state': row.get('campaign_frontier_state') or 'none',
        'waiting_blocker_state': row.get('waiting_blocker_state') or 'waiting',
        'boundary_checkpoint_state': row.get('boundary_checkpoint_state') or 'clean',
        'boundary_notes': list(row.get('boundary_notes') or []),
        'label_leaks': list(row.get('label_leaks') or []),
        'lineage': ['frontier_row'],
    }


def _row_from_theory_memory(row):
    status = str(row.get('status') or row.get('theory_memory_state') or 'recorded')
    return {
        'frontier_id': row.get('frontier_id'),
        'source_outcome_id': row.get('outcome_id') or row.get('source_outcome_id'),
        'action_id': row.get('action_id'),
        'benefit_id': row.get('benefit_id'),
        'scenario_id': row.get('scenario_id') or row.get('hypothesis_id'),
        'hypothesis_id': row.get('hypothesis_id'),
        'planned_theory_move': row.get('planned_theory_move') or 'promote_supported_hypothesis_to_theory_memory',
        'observed_theory_evidence': [{'source': row.get('source') or 'theory_memory', 'status': status}],
        'observed_sibling_evidence': [],
        'before_hypothesis_state': row.get('before_hypothesis_state') or 'waiting',
        'after_hypothesis_state': row.get('after_hypothesis_state') or 'evidence_received',
        'theory_memory_state': 'promoted' if status in {'promoted', 'recorded', 'accepted'} else status,
        'promotion_state': 'promoted' if status in {'promoted', 'recorded', 'accepted'} else 'none',
        'refinement_state': row.get('refinement_state') or 'none',
        'retirement_block_state': row.get('retirement_block_state') or 'none',
        'campaign_frontier_state': row.get('campaign_frontier_state') or 'none',
        'waiting_blocker_state': row.get('waiting_blocker_state') or 'resolved',
        'boundary_checkpoint_state': row.get('boundary_checkpoint_state') or 'clean',
        'boundary_notes': list(row.get('boundary_notes') or []),
        'label_leaks': list(row.get('label_leaks') or []),
        'lineage': ['theory_memory'],
    }


def _row_from_message(message):
    body = dict(message.get('body') or {})
    evidence = dict(message.get('evidence') or {})
    sender = message.get('sender')
    response_kind = body.get('response_kind')
    scenario_id = (
        body.get('scenario_id')
        or evidence.get('scenario_id')
        or body.get('campaign_id')
        or evidence.get('campaign_id')
        or body.get('hypothesis_id')
        or evidence.get('hypothesis_id')
    )
    frontier_id = body.get('frontier_id') or evidence.get('frontier_id')
    planned_move = (
        body.get('planned_theory_move')
        or body.get('selected_theory_move')
        or evidence.get('selected_theory_move')
    )
    if response_kind == 'science_theory_frontier':
        planned_move = body.get('selected_theory_move') or planned_move
    selected_outcome = body.get('selected_outcome') or evidence.get('selected_outcome')
    sibling = list(body.get('observed_sibling_evidence') or [])
    theory_evidence = list(body.get('observed_theory_evidence') or [])
    gate = body.get('evidence_gate') or evidence.get('evidence_gate')
    status = str(body.get('status') or evidence.get('status') or '').lower()
    if sender in {'funfun', 'code_module', 'language_model_2'} and gate:
        sibling.append({'sender': sender, 'evidence_gate': str(gate), 'status': status or 'advisory'})
    if response_kind in {'theory_memory_recorded', 'science_theory_memory_recorded'} or body.get('theory_memory_state') in {'recorded', 'promoted'}:
        theory_evidence.append({'source': sender or 'ai_different', 'status': body.get('theory_memory_state') or 'recorded'})
    leaks = label_leak_terms({'body': body, 'evidence': evidence})
    third_party = bool(body.get('third_party_checkpoint_used') or evidence.get('third_party_checkpoint_used'))
    return {
        'frontier_id': frontier_id,
        'source_outcome_id': body.get('source_outcome_id') or body.get('outcome_id') or evidence.get('outcome_id'),
        'action_id': body.get('action_id') or evidence.get('action_id'),
        'benefit_id': body.get('benefit_id') or evidence.get('benefit_id'),
        'scenario_id': str(scenario_id) if scenario_id else None,
        'hypothesis_id': body.get('hypothesis_id') or evidence.get('hypothesis_id'),
        'planned_theory_move': planned_move or selected_outcome,
        'observed_theory_evidence': theory_evidence,
        'observed_sibling_evidence': sibling,
        'before_hypothesis_state': body.get('before_hypothesis_state') or 'planned',
        'after_hypothesis_state': body.get('after_hypothesis_state') or _after_state_from_status(status),
        'theory_memory_state': body.get('theory_memory_state') or ('recorded' if theory_evidence else 'not_recorded'),
        'promotion_state': body.get('promotion_state') or ('promoted' if planned_move == 'promote_supported_hypothesis_to_theory_memory' else 'none'),
        'refinement_state': body.get('refinement_state') or 'none',
        'retirement_block_state': body.get('retirement_block_state') or body.get('retirement_state') or ('blocked' if status in {'failed', 'blocked', 'rejected'} else 'none'),
        'campaign_frontier_state': body.get('campaign_frontier_state') or ('scheduled' if planned_move == 'schedule_next_campaign_frontier_check' else 'none'),
        'waiting_blocker_state': body.get('waiting_blocker_state') or ('blocked' if status in {'failed', 'blocked', 'rejected'} else 'waiting'),
        'boundary_checkpoint_state': 'repair' if leaks or third_party or body.get('boundary_checkpoint_state') == 'repair' else 'clean',
        'boundary_notes': list(body.get('boundary_notes') or ([] if not leaks else ['label leak detected'])),
        'label_leaks': leaks,
        'lineage': [f'message:{sender}'],
    }


def _upsert_row(rows, incoming):
    key = incoming.get('scenario_id') or incoming.get('hypothesis_id') or incoming.get('frontier_id') or incoming.get('action_id')
    if not key:
        return
    current = rows.setdefault(str(key), {
        'frontier_id': incoming.get('frontier_id'),
        'source_outcome_id': incoming.get('source_outcome_id'),
        'action_id': incoming.get('action_id'),
        'benefit_id': incoming.get('benefit_id'),
        'scenario_id': incoming.get('scenario_id'),
        'hypothesis_id': incoming.get('hypothesis_id'),
        'planned_theory_move': incoming.get('planned_theory_move'),
        'observed_theory_evidence': [],
        'observed_sibling_evidence': [],
        'before_hypothesis_state': 'planned',
        'after_hypothesis_state': 'waiting',
        'theory_memory_state': 'not_recorded',
        'promotion_state': 'none',
        'refinement_state': 'none',
        'retirement_block_state': 'none',
        'campaign_frontier_state': 'none',
        'waiting_blocker_state': 'waiting',
        'boundary_checkpoint_state': 'clean',
        'boundary_notes': [],
        'label_leaks': [],
        'lineage': [],
    })
    for field in ('frontier_id', 'source_outcome_id', 'action_id', 'benefit_id', 'scenario_id', 'hypothesis_id'):
        current[field] = current.get(field) or incoming.get(field)
    current['planned_theory_move'] = _dominant_move(current.get('planned_theory_move'), incoming.get('planned_theory_move'))
    current['observed_theory_evidence'] = _unique_dicts(list(current.get('observed_theory_evidence') or []) + list(incoming.get('observed_theory_evidence') or []))
    current['observed_sibling_evidence'] = _unique_dicts(list(current.get('observed_sibling_evidence') or []) + list(incoming.get('observed_sibling_evidence') or []))
    current['before_hypothesis_state'] = current.get('before_hypothesis_state') or incoming.get('before_hypothesis_state') or 'planned'
    current['after_hypothesis_state'] = _dominant_after_state(current.get('after_hypothesis_state'), incoming.get('after_hypothesis_state'))
    current['theory_memory_state'] = _dominant_theory_memory(current.get('theory_memory_state'), incoming.get('theory_memory_state'))
    current['promotion_state'] = _dominant_state(current.get('promotion_state'), incoming.get('promotion_state'))
    current['refinement_state'] = _dominant_state(current.get('refinement_state'), incoming.get('refinement_state'))
    current['retirement_block_state'] = _dominant_retirement(current.get('retirement_block_state'), incoming.get('retirement_block_state'))
    current['campaign_frontier_state'] = _dominant_state(current.get('campaign_frontier_state'), incoming.get('campaign_frontier_state'))
    current['waiting_blocker_state'] = _dominant_waiting(current.get('waiting_blocker_state'), incoming.get('waiting_blocker_state'))
    current['boundary_checkpoint_state'] = 'repair' if current.get('boundary_checkpoint_state') == 'repair' or incoming.get('boundary_checkpoint_state') == 'repair' else 'clean'
    current['boundary_notes'] = _unique_strings(list(current.get('boundary_notes') or []) + list(incoming.get('boundary_notes') or []))
    current['label_leaks'] = _unique_strings(list(current.get('label_leaks') or []) + list(incoming.get('label_leaks') or []))
    current['lineage'] = _unique_strings(list(current.get('lineage') or []) + list(incoming.get('lineage') or []))


def _finalize_row(row):
    move = row.get('planned_theory_move')
    if row.get('label_leaks'):
        row['boundary_checkpoint_state'] = 'repair'
    if move == 'promote_supported_hypothesis_to_theory_memory' and row.get('promotion_state') == 'none':
        row['promotion_state'] = 'promoted'
    if move == 'retire_or_block_refuted_hypothesis':
        row['retirement_block_state'] = 'blocked'
    if move == 'refine_hypothesis_from_outcome' and row.get('refinement_state') == 'none':
        row['refinement_state'] = 'accepted'
    if move == 'schedule_next_campaign_frontier_check' and row.get('campaign_frontier_state') == 'none':
        row['campaign_frontier_state'] = 'scheduled'
    row['frontier_outcome_hash'] = stable_digest({
        'frontier_id': row.get('frontier_id'),
        'scenario_id': row.get('scenario_id'),
        'move': move,
        'theory_memory_state': row.get('theory_memory_state'),
        'sibling': row.get('observed_sibling_evidence'),
        'boundary': row.get('boundary_checkpoint_state'),
    })
    return row


def _select_frontier_outcome(*, rows, assessed_keys, project_owned_boundary):
    for row in rows:
        if project_owned_boundary.get('third_party_checkpoint_used') or row.get('boundary_checkpoint_state') == 'repair':
            return _outcome(row, 'preserve_boundary_or_checkpoint_repair', 'preserve_boundary_or_checkpoint_repair', 'code_module')
    for row in rows:
        if _theory_memory_recorded(row) and not _is_blocked(row):
            return _outcome(row, 'theory_memory_recorded_or_hypothesis_promoted', 'record_theory_memory_promotion_outcome', 'broadcast')
    for row in rows:
        if _is_blocked(row) or row.get('planned_theory_move') == 'retire_or_block_refuted_hypothesis':
            return _outcome(row, 'refuted_hypothesis_retired_or_blocked', 'record_refuted_hypothesis_closed', 'broadcast')
    for sender, move, outcome, action, recipient in (
        ('funfun', 'request_funfun_certificate', 'funfun_certificate_supports_or_blocks_theory_move', 'record_funfun_certificate_outcome', 'broadcast'),
        ('code_module', 'request_code_experiment_or_counterexample', 'code_experiment_or_counterexample_changes_theory_move', 'record_code_experiment_outcome', 'broadcast'),
        ('language_model_2', 'request_language_protocol_clarification', 'language_protocol_clarification_resolves_theory_move', 'record_language_protocol_outcome', 'broadcast'),
    ):
        for row in rows:
            if row.get('planned_theory_move') == move and _has_sibling_evidence(row, sender):
                return _outcome(row, outcome, action, recipient)
    for row in rows:
        if row.get('planned_theory_move') == 'refine_hypothesis_from_outcome' or row.get('refinement_state') in {'accepted', 'updated'}:
            return _outcome(row, 'hypothesis_refinement_accepted', 'record_refinement_outcome', 'broadcast')
    for row in rows:
        if row.get('planned_theory_move') == 'schedule_next_campaign_frontier_check' or row.get('campaign_frontier_state') == 'scheduled':
            return _outcome(row, 'next_campaign_frontier_scheduled', 'record_frontier_scheduled', 'orchestrator')
    for row in rows:
        if _waiting_for_evidence(row) and _frontier_outcome_key_for(row, 'planned_theory_move_waiting_for_evidence') not in set(assessed_keys):
            return _outcome(row, 'planned_theory_move_waiting_for_evidence', 'wait_for_theory_frontier_evidence', 'orchestrator')
    for row in rows:
        if row.get('planned_theory_move') == 'record_no_measurable_theory_gain':
            return _outcome(row, 'no_measurable_theory_frontier_gain', 'record_no_measurable_theory_gain', 'orchestrator')
    return _noop_outcome('no theory-frontier outcome selected')


def _outcome(row, selected_outcome, action, recipient):
    return {
        'selected_outcome': selected_outcome,
        'selected_action': action,
        'selected_recipient': validate_participant(recipient),
        'frontier_id': row.get('frontier_id'),
        'source_outcome_id': row.get('source_outcome_id'),
        'action_id': row.get('action_id'),
        'benefit_id': row.get('benefit_id'),
        'scenario_id': row.get('scenario_id'),
        'hypothesis_id': row.get('hypothesis_id'),
        'planned_theory_move': row.get('planned_theory_move'),
        'observed_theory_evidence': list(row.get('observed_theory_evidence') or []),
        'observed_sibling_evidence': list(row.get('observed_sibling_evidence') or []),
        'before_hypothesis_state': row.get('before_hypothesis_state') or 'planned',
        'after_hypothesis_state': row.get('after_hypothesis_state') or 'waiting',
        'theory_memory_state': row.get('theory_memory_state') or 'not_recorded',
        'promotion_state': row.get('promotion_state') or 'none',
        'refinement_state': row.get('refinement_state') or 'none',
        'retirement_block_state': row.get('retirement_block_state') or 'none',
        'campaign_frontier_state': row.get('campaign_frontier_state') or 'none',
        'waiting_blocker_state': row.get('waiting_blocker_state') or 'waiting',
        'boundary_checkpoint_state': row.get('boundary_checkpoint_state') or 'clean',
        'boundary_notes': list(row.get('boundary_notes') or []),
        'before_after_delta': f"{row.get('before_hypothesis_state') or 'unknown'} -> {row.get('after_hypothesis_state') or 'unknown'}",
        'no_overclaiming_proof': 'local-owned checkpoint not claimed unless status capsule verifies it',
        'recommended_action': action,
        'label_leaks': list(row.get('label_leaks') or []),
    }


def _noop_outcome(reason):
    return {
        'selected_outcome': 'summarize_noop',
        'selected_action': 'summarize_noop',
        'selected_recipient': None,
        'frontier_id': None,
        'source_outcome_id': None,
        'action_id': None,
        'benefit_id': None,
        'scenario_id': None,
        'hypothesis_id': None,
        'planned_theory_move': None,
        'observed_theory_evidence': [],
        'observed_sibling_evidence': [],
        'before_hypothesis_state': None,
        'after_hypothesis_state': None,
        'theory_memory_state': 'not_recorded',
        'promotion_state': 'none',
        'refinement_state': 'none',
        'retirement_block_state': 'none',
        'campaign_frontier_state': 'none',
        'waiting_blocker_state': 'waiting',
        'boundary_checkpoint_state': 'clean',
        'boundary_notes': [],
        'before_after_delta': reason,
        'no_overclaiming_proof': 'no project-owned checkpoint claim made',
        'recommended_action': 'noop',
        'label_leaks': [],
    }


def _theory_memory_recorded(row):
    return (
        row.get('theory_memory_state') in {'recorded', 'promoted'}
        or row.get('promotion_state') == 'promoted'
        or row.get('planned_theory_move') == 'promote_supported_hypothesis_to_theory_memory'
    )


def _is_blocked(row):
    return (
        row.get('after_hypothesis_state') == 'blocked'
        or row.get('waiting_blocker_state') == 'blocked'
        or row.get('retirement_block_state') in {'blocked', 'retired'}
        or any(str(item.get('status')) in {'failed', 'blocked', 'rejected'} for item in row.get('observed_sibling_evidence') or [])
    )


def _has_sibling_evidence(row, sender):
    return any(item.get('sender') == sender and str(item.get('status') or '') for item in row.get('observed_sibling_evidence') or [])


def _waiting_for_evidence(row):
    if row.get('planned_theory_move') == 'record_no_measurable_theory_gain':
        return False
    return (
        str(row.get('planned_theory_move') or '').startswith('request_')
        or row.get('waiting_blocker_state') == 'waiting'
    )


def _state_counts(selected, rows):
    del rows
    counts = {
        'memory': 0,
        'promote': 0,
        'retire': 0,
        'block': 0,
        'funfun': 0,
        'code': 0,
        'language': 0,
        'refine': 0,
        'frontier': 0,
        'waiting': 0,
        'no_gain': 0,
        'repair': 0,
    }
    mapping = {
        'preserve_boundary_or_checkpoint_repair': 'repair',
        'theory_memory_recorded_or_hypothesis_promoted': 'memory',
        'refuted_hypothesis_retired_or_blocked': 'retire',
        'funfun_certificate_supports_or_blocks_theory_move': 'funfun',
        'code_experiment_or_counterexample_changes_theory_move': 'code',
        'language_protocol_clarification_resolves_theory_move': 'language',
        'hypothesis_refinement_accepted': 'refine',
        'next_campaign_frontier_scheduled': 'frontier',
        'planned_theory_move_waiting_for_evidence': 'waiting',
        'no_measurable_theory_frontier_gain': 'no_gain',
    }
    key = mapping.get(selected.get('selected_outcome'))
    if key:
        counts[key] += 1
    if selected.get('selected_outcome') == 'theory_memory_recorded_or_hypothesis_promoted':
        counts['promote'] += 1
    if selected.get('selected_outcome') == 'refuted_hypothesis_retired_or_blocked':
        counts['block'] += 1
    if selected.get('selected_outcome') == 'summarize_noop':
        counts['no_gain'] += 1
    return counts


def _source_hash(frontier, action_outcome, action_source, benefit, evaluator, prior_outcome, contracts, adjudicator, agenda, lifecycle, scorecard, campaign, campaign_outcome, theory_memory, prior_frontier_outcome, sibling, runtime_memory):
    return stable_digest({
        'frontier': frontier.get('ledger_hash'),
        'action_outcome': action_outcome.get('ledger_hash'),
        'action': action_source.get('ledger_hash'),
        'benefit': benefit.get('ledger_hash'),
        'evaluator': evaluator.get('ledger_hash') or evaluator.get('ledger_id'),
        'prior_outcome': prior_outcome.get('ledger_hash') or prior_outcome.get('ledger_id'),
        'contract': contracts.get('ledger_hash'),
        'adjudicator': adjudicator.get('ledger_hash'),
        'agenda': agenda.get('ledger_hash'),
        'lifecycle': lifecycle.get('ledger_hash'),
        'scorecard': scorecard.get('ledger_hash'),
        'campaign': campaign.get('ledger_hash'),
        'campaign_outcome': campaign_outcome.get('ledger_hash'),
        'theory_memory': theory_memory.get('ledger_hash') or stable_digest(theory_memory) if theory_memory else None,
        'prior_frontier_outcome': prior_frontier_outcome.get('ledger_hash'),
        'sibling': sibling,
        'runtime_keys': sorted(runtime_memory.keys()),
    })


def _memory_rows(theory_memory_ledger, runtime_memory_data):
    rows = []
    for source in (theory_memory_ledger or {}, runtime_memory_data or {}):
        for key in (
            'theory_memory_rows',
            'theory_records',
            'supported_hypotheses',
            'promoted_hypotheses',
            'theories',
        ):
            values = source.get(key)
            if isinstance(values, list):
                rows.extend([dict(item) for item in values if isinstance(item, dict)])
    return rows


def _find_row(rows, frontier_id, scenario_id):
    for row in rows:
        if frontier_id and row.get('frontier_id') == frontier_id:
            return row
        if scenario_id and row.get('scenario_id') == scenario_id:
            return row
    return None


def _frontier_outcome_key(selected):
    return _frontier_outcome_key_for(selected, selected.get('selected_outcome'))


def _frontier_outcome_key_for(row, outcome):
    return stable_digest({
        'frontier_id': row.get('frontier_id'),
        'scenario_id': row.get('scenario_id'),
        'hypothesis_id': row.get('hypothesis_id'),
        'outcome': outcome,
    })


def _after_state_from_status(status):
    if status in {'satisfied', 'resolved', 'passed', 'confirmed', 'accepted'}:
        return 'evidence_received'
    if status in {'failed', 'blocked', 'rejected', 'contradicted'}:
        return 'blocked'
    return 'waiting'


def _dominant_move(current, incoming):
    order = {
        'preserve_boundary_or_checkpoint_repair': 10,
        'promote_supported_hypothesis_to_theory_memory': 9,
        'retire_or_block_refuted_hypothesis': 8,
        'request_funfun_certificate': 7,
        'request_code_experiment_or_counterexample': 6,
        'request_language_protocol_clarification': 5,
        'refine_hypothesis_from_outcome': 4,
        'schedule_next_campaign_frontier_check': 3,
        'record_no_measurable_theory_gain': 2,
        None: 0,
    }
    return current if order.get(current, 0) >= order.get(incoming, 0) else incoming


def _dominant_after_state(current, incoming):
    order = {'blocked': 4, 'evidence_received': 3, 'updated': 3, 'closed': 3, 'waiting': 1, None: 0}
    return current if order.get(current, 0) >= order.get(incoming, 0) else incoming


def _dominant_theory_memory(current, incoming):
    order = {'promoted': 4, 'recorded': 3, 'not_recorded': 1, None: 0}
    return current if order.get(current, 0) >= order.get(incoming, 0) else incoming


def _dominant_state(current, incoming):
    order = {'promoted': 5, 'accepted': 5, 'updated': 4, 'scheduled': 4, 'closed': 4, 'blocked': 3, 'none': 1, None: 0}
    return current if order.get(current, 0) >= order.get(incoming, 0) else incoming


def _dominant_retirement(current, incoming):
    order = {'blocked': 4, 'retired': 4, 'closed': 3, 'none': 1, None: 0}
    return current if order.get(current, 0) >= order.get(incoming, 0) else incoming


def _dominant_waiting(current, incoming):
    order = {'blocked': 3, 'waiting': 2, 'resolved': 1, None: 0}
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
