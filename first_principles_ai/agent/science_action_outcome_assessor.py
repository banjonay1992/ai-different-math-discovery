"""Science-campaign action outcome assessor.

This layer checks whether a planned science campaign action received useful
plain-data evidence. It records symbolic campaign-management outcomes only:
no real-world science proof, no sibling imports, and no checkpoint overclaim.
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
from .science_benefit_evaluator import SCIENCE_BENEFIT_LEDGER_KIND
from .science_campaign_action_planner import SCIENCE_ACTION_LEDGER_KIND


SCIENCE_ACTION_OUTCOME_LEDGER_KIND = 'ai_different.science_campaign_action_outcome_ledger'
GATE_ORDER = ('math_proof', 'code_proof', 'language_epoch_plan')


def empty_science_action_outcome_ledger() -> dict[str, Any]:
    return {
        'schema_version': 1,
        'ledger_kind': SCIENCE_ACTION_OUTCOME_LEDGER_KIND,
        'processed_message_ids': [],
        'processed_source_hashes': [],
        'assessed_action_keys': [],
        'outcome_records': [],
        'action_rows': [],
        'outgoing_response_ids': [],
        'latest': {},
    }


def load_science_action_outcome_ledger(path: str | Path | None) -> dict[str, Any]:
    if not path or not Path(path).exists():
        return empty_science_action_outcome_ledger()
    with Path(path).open('r', encoding='utf-8') as handle:
        ledger = json.load(handle)
    return validate_science_action_outcome_ledger(ledger)


def write_science_action_outcome_ledger(path: str | Path, ledger: dict[str, Any]) -> Path:
    validated = validate_science_action_outcome_ledger(ledger)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(validated, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return output


def validate_science_action_outcome_ledger(ledger: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(ledger, dict):
        raise ValueError('science action outcome ledger must be a JSON object')
    if ledger.get('ledger_kind') != SCIENCE_ACTION_OUTCOME_LEDGER_KIND:
        raise ValueError('science action outcome ledger has wrong ledger_kind')
    for field in (
        'processed_message_ids',
        'processed_source_hashes',
        'assessed_action_keys',
        'outgoing_response_ids',
    ):
        values = ledger.get(field)
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            raise ValueError(f'{field} must be strings')
    for field in ('outcome_records', 'action_rows'):
        values = ledger.get(field)
        if not isinstance(values, list) or not all(isinstance(item, dict) for item in values):
            raise ValueError(f'{field} must be objects')
    latest = ledger.get('latest', {})
    if not isinstance(latest, dict):
        raise ValueError('science action outcome latest must be an object')
    validated = {
        'schema_version': int(ledger.get('schema_version', 1) or 1),
        'ledger_kind': SCIENCE_ACTION_OUTCOME_LEDGER_KIND,
        'processed_message_ids': _unique_strings(ledger['processed_message_ids']),
        'processed_source_hashes': _unique_strings(ledger['processed_source_hashes']),
        'assessed_action_keys': _unique_strings(ledger['assessed_action_keys']),
        'outcome_records': list(ledger['outcome_records']),
        'action_rows': list(ledger['action_rows']),
        'outgoing_response_ids': _unique_strings(ledger['outgoing_response_ids']),
        'latest': dict(latest),
    }
    if ledger.get('ledger_hash'):
        validated['ledger_hash'] = str(ledger['ledger_hash'])
    if ledger.get('artifact_path'):
        validated['artifact_path'] = str(ledger['artifact_path'])
    return validated


def read_science_action_outcome_transcript(path: str | Path | None) -> dict[str, Any]:
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
        raise ValueError('plain science action outcome input must be a JSON object')
    return value


def build_science_action_outcome_assessment(
    *,
    transcript_messages: list[dict[str, Any]],
    action_outcome_ledger: dict[str, Any],
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
    prior_action_outcome_ledger: dict[str, Any] | None = None,
    runtime_memory_data: dict[str, Any] | None = None,
    runtime_memory_hash_state: dict[str, Any],
    project_owned_boundary: dict[str, Any],
    artifact_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    del runtime_memory_data
    ledger = validate_science_action_outcome_ledger(action_outcome_ledger)
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
    prior_action_outcome = _valid_prior_action_outcome_or_empty(prior_action_outcome_ledger or {})
    source_hash = _source_hash(
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
        prior_action_outcome,
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
    rows = _extract_outcome_rows(ledger['action_rows'], action_source, campaign_outcome, transcript_messages)
    if not new_messages and (not source_is_new or rows):
        selected = _noop_action('no new science action outcome evidence or source ledger state')
    else:
        selected = _select_outcome(
            rows=rows,
            assessed_keys=ledger['assessed_action_keys'],
            project_owned_boundary=project_owned_boundary,
        )
    outcome_id = 'science_action_outcome_' + stable_digest({
        'source_hash': source_hash,
        'new_message_ids': [module_chat_message_id(message) for message in new_messages],
        'outcome': selected['selected_outcome'],
        'action_id': selected.get('action_id'),
        'scenario_id': selected.get('scenario_id'),
    })[:16]
    selected['outcome_id'] = outcome_id
    message = export_science_action_outcome_message(
        selected,
        rows=rows,
        runtime_memory_hash_state=runtime_memory_hash_state,
        project_owned_boundary=project_owned_boundary,
        source_hashes={
            'science_action_outcome_source_hash': source_hash,
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
            'prior_action_outcome_ledger_hash': prior_action_outcome.get('ledger_hash'),
        },
    )
    new_ids = [module_chat_message_id(message) for message in new_messages]
    outgoing_id = module_chat_message_id(message) if message else None
    outcome_key = _outcome_key(selected)
    if selected['selected_outcome'] != 'summarize_noop':
        ledger['assessed_action_keys'] = _unique_strings(list(ledger['assessed_action_keys']) + [outcome_key])
    if new_messages:
        ledger['processed_message_ids'] = _unique_strings(list(ledger['processed_message_ids']) + new_ids)
    if source_is_new and selected['selected_outcome'] in {'no_measurable_campaign_gain', 'summarize_noop'}:
        ledger['processed_source_hashes'] = _unique_strings(list(ledger['processed_source_hashes']) + [source_hash])
    if outgoing_id:
        ledger['outgoing_response_ids'] = _unique_strings(list(ledger['outgoing_response_ids']) + [outgoing_id])
    state_counts = _state_counts(selected, rows)
    latest = {
        'outcome_id': outcome_id,
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
        'outcome_id': outcome_id,
        'outcome_hash': stable_digest({'outcome_id': outcome_id, 'selected': selected}),
        'processed_message_ids': new_ids,
        'source_hash': source_hash,
        'action_ids': _unique_strings([row.get('action_id') for row in rows if row.get('action_id')]),
        'benefit_ids': _unique_strings([row.get('benefit_id') for row in rows if row.get('benefit_id')]),
        'scenario_ids': _unique_strings([row.get('scenario_id') for row in rows if row.get('scenario_id')]),
        'hypothesis_ids': _unique_strings([row.get('hypothesis_id') for row in rows if row.get('hypothesis_id')]),
        'planned_action': selected.get('planned_action'),
        'observed_sibling_evidence': selected.get('observed_sibling_evidence'),
        'before_hypothesis_state': selected.get('before_hypothesis_state'),
        'after_hypothesis_state': selected.get('after_hypothesis_state'),
        'theory_update_delta': selected.get('theory_update_delta'),
        'refinement_state': selected.get('refinement_state'),
        'retirement_state': selected.get('retirement_state'),
        'waiting_blocker_state': selected.get('waiting_blocker_state'),
        'boundary_checkpoint_state': selected.get('boundary_checkpoint_state'),
        'selected_next_action': selected.get('selected_action'),
        'selected_recipient': latest['chosen_recipient'],
        'outgoing_response_id': outgoing_id,
        'runtime_memory_hash_state': dict(runtime_memory_hash_state),
        'source_ledger_hashes': {
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
            'prior_action_outcome_ledger_hash': prior_action_outcome.get('ledger_hash'),
        },
    }
    ledger['action_rows'] = rows
    if new_messages or source_is_new or message is not None:
        ledger['outcome_records'] = list(ledger['outcome_records']) + [record]
    ledger['latest'] = latest
    if artifact_path is not None:
        ledger['artifact_path'] = str(Path(artifact_path))
    ledger['ledger_hash'] = stable_digest({key: value for key, value in ledger.items() if key != 'ledger_hash'})
    return ledger, message


def export_science_action_outcome_message(
    selected: dict[str, Any],
    *,
    rows: list[dict[str, Any]],
    runtime_memory_hash_state: dict[str, Any],
    project_owned_boundary: dict[str, Any],
    source_hashes: dict[str, Any],
    topic: str = 'ai_different.science_campaign_action_outcome',
) -> dict[str, Any] | None:
    if selected['selected_outcome'] == 'summarize_noop':
        return None
    recipient = validate_participant(selected.get('selected_recipient') or 'orchestrator')
    row = _find_row(rows, selected.get('action_id'), selected.get('scenario_id')) or {}
    leak_terms = label_leak_terms({'selected': selected, 'row': row})
    body = {
        'module': 'AI Different',
        'response_kind': 'science_campaign_action_outcome',
        'outcome_id': selected.get('outcome_id'),
        'action_id': selected.get('action_id'),
        'benefit_id': selected.get('benefit_id'),
        'scenario_id': selected.get('scenario_id'),
        'hypothesis_id': selected.get('hypothesis_id'),
        'selected_outcome': selected['selected_outcome'],
        'selected_action': selected['selected_action'],
        'selected_recipient': recipient,
        'planned_action': selected.get('planned_action'),
        'sibling_evidence_used': selected.get('observed_sibling_evidence') or [],
        'before_hypothesis_state': selected.get('before_hypothesis_state'),
        'after_hypothesis_state': selected.get('after_hypothesis_state'),
        'theory_update_delta': selected.get('theory_update_delta'),
        'refinement_state': selected.get('refinement_state'),
        'retirement_state': selected.get('retirement_state'),
        'waiting_blocker_state': selected.get('waiting_blocker_state'),
        'boundary_checkpoint_state': selected.get('boundary_checkpoint_state'),
        'boundary_notes': selected.get('boundary_notes') or [],
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
            'outcome_id': body['outcome_id'],
            'action_id': body['action_id'],
            'scenario_id': body['scenario_id'],
            'hypothesis_id': body['hypothesis_id'],
            'selected_outcome': body['selected_outcome'],
            'selected_action': body['selected_action'],
            'runtime_memory_hash_state': body['runtime_memory_hash_state'],
            'runtime_memory_mutated': body['runtime_memory_mutated'],
            'label_clean': body['label_clean'],
            'no_sibling_imports': True,
        },
        tags=['ai_different', 'science_campaign_action_outcome', body['selected_outcome'], 'label_clean' if body['label_clean'] else 'label_review_needed'],
    )


def write_science_action_outcome_outbox_jsonl(path: str | Path, message: dict[str, Any] | None) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text((json.dumps(message, sort_keys=True) + '\n') if message is not None else '', encoding='utf-8')
    return output


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


def _valid_prior_action_outcome_or_empty(ledger):
    if not ledger:
        return {}
    if ledger.get('ledger_kind') != SCIENCE_ACTION_OUTCOME_LEDGER_KIND:
        raise ValueError('prior action outcome ledger has wrong ledger_kind')
    return validate_science_action_outcome_ledger(ledger)


def _extract_outcome_rows(existing, action_ledger, campaign_outcome_ledger, messages):
    rows: dict[str, dict[str, Any]] = {}
    for row in list(existing or []):
        _upsert_row(rows, row)
    for record in list(action_ledger.get('action_records') or []):
        _upsert_row(rows, _row_from_action_record(record))
    for row in list(action_ledger.get('scenario_rows') or []):
        _upsert_row(rows, _row_from_action_row(row))
    for outcome in list(campaign_outcome_ledger.get('outcomes') or []):
        _upsert_row(rows, _row_from_campaign_outcome(outcome))
    for message in messages:
        _upsert_row(rows, _row_from_message(message))
    return [_finalize_row(row) for row in rows.values()]


def _row_from_action_record(record):
    scenario_id = _first(record.get('scenario_ids') or [])
    return {
        'action_id': record.get('action_id'),
        'benefit_id': _first(record.get('benefit_ids') or []),
        'scenario_id': scenario_id,
        'hypothesis_id': _first(record.get('hypothesis_ids') or []),
        'planned_action': record.get('selected_action'),
        'observed_sibling_evidence': list(record.get('sibling_evidence_used') or []),
        'before_hypothesis_state': 'planned',
        'after_hypothesis_state': 'waiting',
        'theory_update_delta': record.get('theory_update_intent'),
        'refinement_state': record.get('refinement_state') or 'none',
        'retirement_state': record.get('retirement_state') or 'none',
        'waiting_blocker_state': 'waiting',
        'boundary_checkpoint_state': record.get('boundary_checkpoint_state') or 'clean',
        'boundary_notes': list(record.get('boundary_notes') or []),
        'label_leaks': [],
        'lineage': ['action_record'],
    }


def _row_from_action_row(row):
    return {
        'action_id': row.get('action_id'),
        'benefit_id': row.get('benefit_id'),
        'scenario_id': row.get('scenario_id'),
        'hypothesis_id': row.get('hypothesis_id'),
        'planned_action': row.get('selected_action') or row.get('planned_action'),
        'observed_sibling_evidence': list(row.get('sibling_evidence_used') or []),
        'before_hypothesis_state': row.get('before_hypothesis_state') or 'planned',
        'after_hypothesis_state': row.get('after_hypothesis_state') or 'waiting',
        'theory_update_delta': row.get('theory_update_delta') or row.get('theory_update_intent'),
        'refinement_state': row.get('refinement_state') or 'none',
        'retirement_state': row.get('retirement_state') or 'none',
        'waiting_blocker_state': row.get('waiting_blocker_state') or 'waiting',
        'boundary_checkpoint_state': row.get('boundary_checkpoint_state') or 'clean',
        'boundary_notes': list(row.get('boundary_notes') or []),
        'label_leaks': list(row.get('label_leaks') or []),
        'lineage': ['action_row'],
    }


def _row_from_campaign_outcome(outcome):
    return {
        'action_id': outcome.get('action_id'),
        'benefit_id': outcome.get('benefit_id'),
        'scenario_id': outcome.get('scenario_id') or outcome.get('campaign_id') or outcome.get('hypothesis_id'),
        'hypothesis_id': outcome.get('hypothesis_id'),
        'planned_action': outcome.get('selected_action'),
        'observed_sibling_evidence': [],
        'before_hypothesis_state': 'campaign_outcome',
        'after_hypothesis_state': outcome.get('readiness_state') or 'waiting',
        'theory_update_delta': outcome.get('readiness_state'),
        'refinement_state': 'updated' if outcome.get('readiness_state') == 'refine' else 'none',
        'retirement_state': 'closed' if outcome.get('readiness_state') == 'retired' else 'none',
        'waiting_blocker_state': 'waiting',
        'boundary_checkpoint_state': 'repair' if 'safety_label_project_boundary' in set(outcome.get('rejected_evidence') or []) else 'clean',
        'boundary_notes': [],
        'label_leaks': [],
        'lineage': ['campaign_outcome'],
    }


def _row_from_message(message):
    body = dict(message.get('body') or {})
    evidence = dict(message.get('evidence') or {})
    sender = message.get('sender')
    scenario_id = body.get('scenario_id') or evidence.get('scenario_id') or body.get('campaign_id') or evidence.get('campaign_id') or body.get('hypothesis_id') or evidence.get('hypothesis_id')
    action_id = body.get('action_id') or evidence.get('action_id')
    planned = body.get('planned_action') or body.get('selected_action') or evidence.get('selected_action')
    if body.get('response_kind') == 'science_campaign_action_outcome':
        planned = body.get('planned_action') or planned
    gate = body.get('evidence_gate') or evidence.get('evidence_gate')
    status = str(body.get('status') or evidence.get('status') or '').lower()
    leaks = label_leak_terms({'body': body, 'evidence': evidence})
    third_party = bool(body.get('third_party_checkpoint_used') or evidence.get('third_party_checkpoint_used'))
    sibling = list(body.get('sibling_evidence_used') or [])
    if sender in {'funfun', 'code_module', 'language_model_2'} and gate:
        sibling.append({'sender': sender, 'evidence_gate': str(gate), 'status': status or 'advisory'})
    return {
        'action_id': action_id,
        'benefit_id': body.get('benefit_id') or evidence.get('benefit_id'),
        'scenario_id': str(scenario_id) if scenario_id else None,
        'hypothesis_id': body.get('hypothesis_id') or evidence.get('hypothesis_id'),
        'planned_action': planned,
        'observed_sibling_evidence': sibling,
        'before_hypothesis_state': body.get('before_hypothesis_state') or 'planned',
        'after_hypothesis_state': body.get('after_hypothesis_state') or _after_state_from_status(status),
        'theory_update_delta': body.get('theory_update_delta') or body.get('theory_update_intent'),
        'refinement_state': body.get('refinement_state') or ('updated' if planned == 'refine_hypothesis_with_connected_evidence' and status in {'satisfied', 'resolved', 'passed'} else 'none'),
        'retirement_state': (
            'closed'
            if planned == 'retire_failed_hypothesis_line'
            else body.get('retirement_state') or 'none'
        ),
        'waiting_blocker_state': body.get('waiting_blocker_state') or ('blocked' if status in {'failed', 'blocked', 'rejected'} else 'waiting'),
        'boundary_checkpoint_state': 'repair' if leaks or third_party or body.get('boundary_checkpoint_state') == 'repair' else 'clean',
        'boundary_notes': list(body.get('boundary_notes') or ([] if not leaks else ['label leak detected'])),
        'label_leaks': leaks,
        'lineage': [f'message:{sender}'],
    }


def _upsert_row(rows, incoming):
    scenario_id = incoming.get('scenario_id') or incoming.get('action_id')
    if not scenario_id:
        return
    current = rows.setdefault(str(scenario_id), {
        'scenario_id': incoming.get('scenario_id'),
        'action_id': incoming.get('action_id'),
        'benefit_id': incoming.get('benefit_id'),
        'hypothesis_id': incoming.get('hypothesis_id'),
        'planned_action': incoming.get('planned_action'),
        'observed_sibling_evidence': [],
        'before_hypothesis_state': 'planned',
        'after_hypothesis_state': 'waiting',
        'theory_update_delta': None,
        'refinement_state': 'none',
        'retirement_state': 'none',
        'waiting_blocker_state': 'waiting',
        'boundary_checkpoint_state': 'clean',
        'boundary_notes': [],
        'label_leaks': [],
        'lineage': [],
    })
    current['scenario_id'] = current.get('scenario_id') or incoming.get('scenario_id')
    current['action_id'] = current.get('action_id') or incoming.get('action_id')
    current['benefit_id'] = current.get('benefit_id') or incoming.get('benefit_id')
    current['hypothesis_id'] = current.get('hypothesis_id') or incoming.get('hypothesis_id')
    current['planned_action'] = _dominant_planned_action(current.get('planned_action'), incoming.get('planned_action'))
    current['observed_sibling_evidence'] = _unique_dicts(list(current.get('observed_sibling_evidence') or []) + list(incoming.get('observed_sibling_evidence') or []))
    current['before_hypothesis_state'] = current.get('before_hypothesis_state') or incoming.get('before_hypothesis_state') or 'planned'
    current['after_hypothesis_state'] = _dominant_after_state(current.get('after_hypothesis_state'), incoming.get('after_hypothesis_state'))
    current['theory_update_delta'] = incoming.get('theory_update_delta') or current.get('theory_update_delta')
    current['refinement_state'] = _dominant_state(current.get('refinement_state'), incoming.get('refinement_state'))
    current['retirement_state'] = _dominant_state(current.get('retirement_state'), incoming.get('retirement_state'))
    current['waiting_blocker_state'] = _dominant_waiting(current.get('waiting_blocker_state'), incoming.get('waiting_blocker_state'))
    current['boundary_checkpoint_state'] = 'repair' if current.get('boundary_checkpoint_state') == 'repair' or incoming.get('boundary_checkpoint_state') == 'repair' else 'clean'
    current['boundary_notes'] = _unique_strings(list(current.get('boundary_notes') or []) + list(incoming.get('boundary_notes') or []))
    current['label_leaks'] = _unique_strings(list(current.get('label_leaks') or []) + list(incoming.get('label_leaks') or []))
    current['lineage'] = _unique_strings(list(current.get('lineage') or []) + list(incoming.get('lineage') or []))


def _finalize_row(row):
    if row.get('label_leaks'):
        row['boundary_checkpoint_state'] = 'repair'
    row['outcome_hash'] = stable_digest({
        'action_id': row.get('action_id'),
        'scenario_id': row.get('scenario_id'),
        'planned': row.get('planned_action'),
        'evidence': row.get('observed_sibling_evidence'),
        'after': row.get('after_hypothesis_state'),
        'boundary': row.get('boundary_checkpoint_state'),
    })
    return row


def _select_outcome(*, rows, assessed_keys, project_owned_boundary):
    for row in rows:
        if project_owned_boundary.get('third_party_checkpoint_used') or row.get('boundary_checkpoint_state') == 'repair':
            return _outcome(row, 'preserve_boundary_or_checkpoint_repair', 'preserve_boundary_or_checkpoint_repair', 'code_module')
    outcome_map = (
        ('request_funfun_math_certificate', 'math_certificate_supports_or_blocks_claim', 'broadcast'),
        ('request_code_experiment_or_counterexample', 'code_experiment_or_counterexample_changes_decision', 'broadcast'),
        ('request_language_protocol_clarification', 'language_clarification_repairs_protocol', 'broadcast'),
        ('refine_hypothesis_with_connected_evidence', 'refinement_updates_hypothesis_safely', 'broadcast'),
        ('retire_failed_hypothesis_line', 'retirement_closes_failed_line', 'broadcast'),
    )
    for planned_action, outcome_name, recipient in outcome_map:
        for row in rows:
            if row.get('planned_action') == planned_action and _has_observed_signal(row):
                return _outcome(row, outcome_name, _next_action_for_outcome(outcome_name, row), recipient)
    for row in rows:
        if row.get('planned_action') == 'schedule_next_campaign_check':
            return _outcome(row, 'scheduled_action_waiting_for_evidence', 'schedule_next_campaign_check', 'orchestrator')
    for row in rows:
        if row.get('planned_action') == 'record_connected_no_safe_benefit':
            return _outcome(row, 'no_measurable_campaign_gain', 'record_connected_no_safe_benefit', 'orchestrator')
    for row in rows:
        if _outcome_key_for(row, 'scheduled_action_waiting_for_evidence') not in set(assessed_keys):
            return _outcome(row, 'scheduled_action_waiting_for_evidence', 'wait_for_sibling_evidence', 'orchestrator')
    return _noop_action('no science action outcome selected')


def _outcome(row, selected_outcome, selected_action, recipient):
    return {
        'selected_outcome': selected_outcome,
        'selected_action': selected_action,
        'selected_recipient': validate_participant(recipient),
        'action_id': row.get('action_id'),
        'benefit_id': row.get('benefit_id'),
        'scenario_id': row.get('scenario_id'),
        'hypothesis_id': row.get('hypothesis_id'),
        'planned_action': row.get('planned_action'),
        'observed_sibling_evidence': list(row.get('observed_sibling_evidence') or []),
        'before_hypothesis_state': row.get('before_hypothesis_state') or 'planned',
        'after_hypothesis_state': row.get('after_hypothesis_state') or 'waiting',
        'theory_update_delta': _delta_for(row, selected_outcome),
        'refinement_state': row.get('refinement_state') or 'none',
        'retirement_state': row.get('retirement_state') or 'none',
        'waiting_blocker_state': row.get('waiting_blocker_state') or 'waiting',
        'boundary_checkpoint_state': row.get('boundary_checkpoint_state') or 'clean',
        'boundary_notes': list(row.get('boundary_notes') or []),
        'recommended_action': selected_action,
        'label_leaks': list(row.get('label_leaks') or []),
    }


def _noop_action(reason):
    return {
        'selected_outcome': 'summarize_noop',
        'selected_action': 'summarize_noop',
        'selected_recipient': None,
        'action_id': None,
        'benefit_id': None,
        'scenario_id': None,
        'hypothesis_id': None,
        'planned_action': None,
        'observed_sibling_evidence': [],
        'before_hypothesis_state': None,
        'after_hypothesis_state': None,
        'theory_update_delta': reason,
        'refinement_state': 'none',
        'retirement_state': 'none',
        'waiting_blocker_state': 'waiting',
        'boundary_checkpoint_state': 'clean',
        'boundary_notes': [],
        'recommended_action': 'noop',
        'label_leaks': [],
    }


def _has_observed_signal(row):
    return bool(row.get('observed_sibling_evidence') or row.get('after_hypothesis_state') not in {None, 'waiting'} or row.get('retirement_state') == 'closed' or row.get('refinement_state') == 'updated')


def _next_action_for_outcome(outcome_name, row):
    if outcome_name == 'math_certificate_supports_or_blocks_claim':
        return 'record_math_certificate_outcome'
    if outcome_name == 'code_experiment_or_counterexample_changes_decision':
        return 'record_code_experiment_outcome'
    if outcome_name == 'language_clarification_repairs_protocol':
        return 'record_language_clarification_outcome'
    if outcome_name == 'refinement_updates_hypothesis_safely':
        return 'record_refinement_update'
    if outcome_name == 'retirement_closes_failed_line':
        return 'record_retirement'
    return row.get('planned_action') or 'record_outcome'


def _delta_for(row, selected_outcome):
    if row.get('theory_update_delta'):
        return row.get('theory_update_delta')
    return f"{row.get('planned_action') or 'planned_action'} -> {selected_outcome}"


def _state_counts(selected, rows):
    counts = {
        'math': 0,
        'code': 0,
        'language': 0,
        'refine': 0,
        'retire': 0,
        'waiting': 0,
        'no_gain': 0,
        'repair': 0,
    }
    mapping = {
        'preserve_boundary_or_checkpoint_repair': 'repair',
        'math_certificate_supports_or_blocks_claim': 'math',
        'code_experiment_or_counterexample_changes_decision': 'code',
        'language_clarification_repairs_protocol': 'language',
        'refinement_updates_hypothesis_safely': 'refine',
        'retirement_closes_failed_line': 'retire',
        'scheduled_action_waiting_for_evidence': 'waiting',
        'no_measurable_campaign_gain': 'no_gain',
    }
    key = mapping.get(selected.get('selected_outcome'))
    if key:
        counts[key] += 1
    if selected.get('selected_outcome') == 'summarize_noop' and not rows:
        counts['no_gain'] += 1
    return counts


def _source_hash(action_source, benefit, evaluator, prior_outcome, contracts, adjudicator, agenda, lifecycle, scorecard, campaign, campaign_outcome, prior_action_outcome):
    return stable_digest({
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
        'prior_action_outcome': prior_action_outcome.get('ledger_hash'),
    })


def _find_row(rows, action_id, scenario_id):
    for row in rows:
        if action_id and row.get('action_id') == action_id:
            return row
        if scenario_id and row.get('scenario_id') == scenario_id:
            return row
    return None


def _outcome_key(selected):
    return _outcome_key_for(selected, selected.get('selected_outcome'))


def _outcome_key_for(row, outcome):
    return stable_digest({
        'action_id': row.get('action_id'),
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


def _dominant_planned_action(current, incoming):
    return current or incoming


def _dominant_after_state(current, incoming):
    order = {'blocked': 4, 'evidence_received': 3, 'updated': 3, 'closed': 3, 'waiting': 1, None: 0}
    return current if order.get(current, 0) >= order.get(incoming, 0) else incoming


def _dominant_state(current, incoming):
    order = {'closed': 4, 'updated': 3, 'blocked': 2, 'none': 1, None: 0}
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
