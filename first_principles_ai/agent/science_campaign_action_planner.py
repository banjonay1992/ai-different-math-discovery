"""Science-benefit-to-campaign action planner.

This layer turns symbolic connected-vs-isolated benefit records into one safe
next science campaign action. It is campaign management only: no experiment
execution, no sibling imports, and no project-owned checkpoint claim unless the
status capsule verifies one.
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


SCIENCE_ACTION_LEDGER_KIND = 'ai_different.science_campaign_action_ledger'
GATE_ORDER = ('math_proof', 'code_proof', 'language_epoch_plan')


def empty_science_action_ledger() -> dict[str, Any]:
    return {
        'schema_version': 1,
        'ledger_kind': SCIENCE_ACTION_LEDGER_KIND,
        'processed_message_ids': [],
        'processed_source_hashes': [],
        'planned_action_keys': [],
        'action_records': [],
        'scenario_rows': [],
        'outgoing_response_ids': [],
        'latest': {},
    }


def load_science_action_ledger(path: str | Path | None) -> dict[str, Any]:
    if not path or not Path(path).exists():
        return empty_science_action_ledger()
    with Path(path).open('r', encoding='utf-8') as handle:
        ledger = json.load(handle)
    return validate_science_action_ledger(ledger)


def write_science_action_ledger(path: str | Path, ledger: dict[str, Any]) -> Path:
    validated = validate_science_action_ledger(ledger)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(validated, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return output


def validate_science_action_ledger(ledger: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(ledger, dict):
        raise ValueError('science campaign action ledger must be a JSON object')
    if ledger.get('ledger_kind') != SCIENCE_ACTION_LEDGER_KIND:
        raise ValueError('science campaign action ledger has wrong ledger_kind')
    for field in (
        'processed_message_ids',
        'processed_source_hashes',
        'planned_action_keys',
        'outgoing_response_ids',
    ):
        values = ledger.get(field)
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            raise ValueError(f'{field} must be strings')
    for field in ('action_records', 'scenario_rows'):
        values = ledger.get(field)
        if not isinstance(values, list) or not all(isinstance(item, dict) for item in values):
            raise ValueError(f'{field} must be objects')
    latest = ledger.get('latest', {})
    if not isinstance(latest, dict):
        raise ValueError('science campaign action latest must be an object')
    validated = {
        'schema_version': int(ledger.get('schema_version', 1) or 1),
        'ledger_kind': SCIENCE_ACTION_LEDGER_KIND,
        'processed_message_ids': _unique_strings(ledger['processed_message_ids']),
        'processed_source_hashes': _unique_strings(ledger['processed_source_hashes']),
        'planned_action_keys': _unique_strings(ledger['planned_action_keys']),
        'action_records': list(ledger['action_records']),
        'scenario_rows': list(ledger['scenario_rows']),
        'outgoing_response_ids': _unique_strings(ledger['outgoing_response_ids']),
        'latest': dict(latest),
    }
    if ledger.get('ledger_hash'):
        validated['ledger_hash'] = str(ledger['ledger_hash'])
    if ledger.get('artifact_path'):
        validated['artifact_path'] = str(ledger['artifact_path'])
    return validated


def read_science_action_transcript(path: str | Path | None) -> dict[str, Any]:
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
        raise ValueError('plain science action input must be a JSON object')
    return value


def build_science_campaign_action_plan(
    *,
    transcript_messages: list[dict[str, Any]],
    action_ledger: dict[str, Any],
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
    prior_action_ledger: dict[str, Any] | None = None,
    runtime_memory_data: dict[str, Any] | None = None,
    runtime_memory_hash_state: dict[str, Any],
    project_owned_boundary: dict[str, Any],
    artifact_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    del runtime_memory_data
    ledger = validate_science_action_ledger(action_ledger)
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
    prior_action = _valid_prior_action_or_empty(prior_action_ledger or {})
    source_hash = _source_hash(
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
        prior_action,
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
    rows = _extract_action_rows(ledger['scenario_rows'], benefit, campaign_outcome, transcript_messages)
    if not new_messages and (not source_is_new or rows):
        selected = _noop_action('no new science campaign action evidence or source ledger state')
    else:
        selected = _select_action(
            rows=rows,
            planned_keys=ledger['planned_action_keys'],
            project_owned_boundary=project_owned_boundary,
        )
    action_id = 'science_action_' + stable_digest({
        'source_hash': source_hash,
        'new_message_ids': [module_chat_message_id(message) for message in new_messages],
        'selected_action': selected['selected_action'],
        'scenario_id': selected.get('scenario_id'),
    })[:16]
    selected['action_id'] = action_id
    message = export_science_action_message(
        selected,
        rows=rows,
        runtime_memory_hash_state=runtime_memory_hash_state,
        project_owned_boundary=project_owned_boundary,
        source_hashes={
            'science_action_source_hash': source_hash,
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
            'prior_action_ledger_hash': prior_action.get('ledger_hash'),
        },
    )
    new_ids = [module_chat_message_id(message) for message in new_messages]
    outgoing_id = module_chat_message_id(message) if message else None
    action_key = _action_key(selected)
    if selected['selected_action'] != 'summarize_noop':
        ledger['planned_action_keys'] = _unique_strings(list(ledger['planned_action_keys']) + [action_key])
    if new_messages:
        ledger['processed_message_ids'] = _unique_strings(list(ledger['processed_message_ids']) + new_ids)
    if source_is_new and selected['selected_action'] in {'record_connected_no_safe_benefit', 'summarize_noop'}:
        ledger['processed_source_hashes'] = _unique_strings(list(ledger['processed_source_hashes']) + [source_hash])
    if outgoing_id:
        ledger['outgoing_response_ids'] = _unique_strings(list(ledger['outgoing_response_ids']) + [outgoing_id])
    state_counts = _state_counts(selected, rows)
    latest = {
        'action_id': action_id,
        'new_message_count': len(new_messages),
        'skipped_message_count': len(skipped_messages),
        'source_is_new': source_is_new,
        'state_counts': state_counts,
        'selected_action': selected['selected_action'],
        'chosen_recipient': message.get('recipient') if message else None,
        'outbox_count': 1 if message else 0,
        'label_leaks': selected.get('label_leaks', []),
        'runtime_memory_hash_state': dict(runtime_memory_hash_state),
        'runtime_memory_mutated': not bool(runtime_memory_hash_state.get('unchanged', True)),
    }
    record = {
        'action_id': action_id,
        'action_hash': stable_digest({'action_id': action_id, 'selected': selected}),
        'processed_message_ids': new_ids,
        'source_hash': source_hash,
        'benefit_ids': _unique_strings([row.get('benefit_id') for row in rows if row.get('benefit_id')]),
        'scenario_ids': _unique_strings([row.get('scenario_id') for row in rows if row.get('scenario_id')]),
        'hypothesis_ids': _unique_strings([row.get('hypothesis_id') for row in rows if row.get('hypothesis_id')]),
        'selected_action': selected['selected_action'],
        'selected_recipient': latest['chosen_recipient'],
        'required_evidence': selected.get('required_evidence'),
        'sibling_evidence_used': selected.get('sibling_evidence_used'),
        'theory_update_intent': selected.get('theory_update_intent'),
        'refinement_state': selected.get('refinement_state'),
        'retirement_state': selected.get('retirement_state'),
        'boundary_checkpoint_state': selected.get('boundary_checkpoint_state'),
        'outgoing_response_id': outgoing_id,
        'runtime_memory_hash_state': dict(runtime_memory_hash_state),
        'source_ledger_hashes': {
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
            'prior_action_ledger_hash': prior_action.get('ledger_hash'),
        },
    }
    ledger['scenario_rows'] = rows
    if new_messages or source_is_new or message is not None:
        ledger['action_records'] = list(ledger['action_records']) + [record]
    ledger['latest'] = latest
    if artifact_path is not None:
        ledger['artifact_path'] = str(Path(artifact_path))
    ledger['ledger_hash'] = stable_digest({key: value for key, value in ledger.items() if key != 'ledger_hash'})
    return ledger, message


def export_science_action_message(
    selected: dict[str, Any],
    *,
    rows: list[dict[str, Any]],
    runtime_memory_hash_state: dict[str, Any],
    project_owned_boundary: dict[str, Any],
    source_hashes: dict[str, Any],
    topic: str = 'ai_different.science_campaign_action',
) -> dict[str, Any] | None:
    if selected['selected_action'] == 'summarize_noop':
        return None
    recipient = validate_participant(selected.get('selected_recipient') or 'orchestrator')
    row = _find_row(rows, selected.get('scenario_id')) or {}
    leak_terms = label_leak_terms({'selected': selected, 'row': row})
    body = {
        'module': 'AI Different',
        'response_kind': 'science_campaign_action',
        'action_id': selected.get('action_id'),
        'benefit_id': selected.get('benefit_id'),
        'scenario_id': selected.get('scenario_id'),
        'hypothesis_id': selected.get('hypothesis_id'),
        'selected_action': selected['selected_action'],
        'selected_recipient': recipient,
        'required_evidence': selected.get('required_evidence') or [],
        'sibling_evidence_used': selected.get('sibling_evidence_used') or [],
        'theory_update_intent': selected.get('theory_update_intent'),
        'refinement_state': selected.get('refinement_state'),
        'retirement_state': selected.get('retirement_state'),
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
            'action_id': body['action_id'],
            'benefit_id': body['benefit_id'],
            'scenario_id': body['scenario_id'],
            'hypothesis_id': body['hypothesis_id'],
            'selected_action': body['selected_action'],
            'runtime_memory_hash_state': body['runtime_memory_hash_state'],
            'runtime_memory_mutated': body['runtime_memory_mutated'],
            'label_clean': body['label_clean'],
            'no_sibling_imports': True,
        },
        tags=['ai_different', 'science_campaign_action', body['selected_action'], 'label_clean' if body['label_clean'] else 'label_review_needed'],
    )


def write_science_action_outbox_jsonl(path: str | Path, message: dict[str, Any] | None) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text((json.dumps(message, sort_keys=True) + '\n') if message is not None else '', encoding='utf-8')
    return output


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


def _valid_prior_action_or_empty(ledger):
    if not ledger:
        return {}
    if ledger.get('ledger_kind') != SCIENCE_ACTION_LEDGER_KIND:
        raise ValueError('prior action ledger has wrong ledger_kind')
    return validate_science_action_ledger(ledger)


def _extract_action_rows(existing, benefit_ledger, campaign_outcome_ledger, messages):
    rows: dict[str, dict[str, Any]] = {}
    for row in list(existing or []):
        _upsert_row(rows, row)
    for record in list(benefit_ledger.get('benefit_records') or []):
        _upsert_row(rows, _row_from_benefit_record(record))
    for scenario in list(benefit_ledger.get('scenario_records') or []):
        _upsert_row(rows, _row_from_scenario_record(scenario))
    for outcome in list(campaign_outcome_ledger.get('outcomes') or []):
        _upsert_row(rows, _row_from_outcome_record(outcome))
    for message in messages:
        _upsert_row(rows, _row_from_message(message))
    return [_finalize_row(row) for row in rows.values()]


def _row_from_benefit_record(record):
    scenario_id = record.get('selected_scenario_id') or _first(record.get('scenario_ids') or [])
    return {
        'benefit_id': record.get('benefit_id'),
        'scenario_id': scenario_id,
        'hypothesis_id': _first(record.get('hypothesis_ids') or []),
        'benefit_classification': _classification_from_record(record),
        'required_evidence': _missing_from_record(record),
        'sibling_evidence_used': list(record.get('sibling_evidence_used') or []),
        'theory_update_delta': record.get('theory_update_delta'),
        'refinement_state': record.get('refinement_delta') or 'none',
        'retirement_state': record.get('retirement_delta') or 'none',
        'boundary_checkpoint_state': record.get('boundary_state') or 'clean',
        'boundary_notes': list(record.get('boundary_notes') or []),
        'lineage': ['benefit_record'],
    }


def _row_from_scenario_record(scenario):
    connected = dict(scenario.get('connected') or {})
    classification = 'connected_adds_no_safe_benefit'
    if connected.get('rejected'):
        classification = 'connected_retires_failed_line'
    elif all(gate in set(connected.get('accepted') or []) for gate in GATE_ORDER):
        classification = 'connected_accepts_with_verified_math_code_language'
    elif connected.get('missing'):
        classification = 'connected_requests_missing_targeted_evidence'
    return {
        'benefit_id': scenario.get('benefit_id'),
        'scenario_id': scenario.get('scenario_id'),
        'hypothesis_id': scenario.get('hypothesis_id'),
        'benefit_classification': classification,
        'required_evidence': list(connected.get('missing') or []),
        'sibling_evidence_used': _sibling_evidence_from_connected(connected),
        'theory_update_delta': scenario.get('theory_update_delta'),
        'refinement_state': scenario.get('refinement_delta') or 'none',
        'retirement_state': scenario.get('retirement_delta') or 'none',
        'boundary_checkpoint_state': scenario.get('boundary_state') or 'clean',
        'boundary_notes': list(scenario.get('boundary_notes') or []),
        'lineage': ['benefit_scenario'],
    }


def _row_from_outcome_record(outcome):
    return {
        'benefit_id': None,
        'scenario_id': outcome.get('scenario_id') or outcome.get('campaign_id') or outcome.get('hypothesis_id'),
        'hypothesis_id': outcome.get('hypothesis_id'),
        'benefit_classification': 'connected_adds_no_safe_benefit',
        'required_evidence': list(outcome.get('missing_evidence') or []),
        'sibling_evidence_used': [],
        'theory_update_delta': outcome.get('readiness_state'),
        'refinement_state': 'connected_added' if outcome.get('readiness_state') == 'refine' else 'none',
        'retirement_state': 'connected_added' if outcome.get('readiness_state') == 'retired' else 'none',
        'boundary_checkpoint_state': 'repair' if 'safety_label_project_boundary' in set(outcome.get('rejected_evidence') or []) else 'clean',
        'boundary_notes': [],
        'lineage': ['campaign_outcome'],
    }


def _row_from_message(message):
    body = dict(message.get('body') or {})
    evidence = dict(message.get('evidence') or {})
    scenario_id = body.get('scenario_id') or evidence.get('scenario_id') or body.get('campaign_id') or evidence.get('campaign_id') or body.get('hypothesis_id') or evidence.get('hypothesis_id')
    classification = body.get('benefit_classification') or evidence.get('benefit_classification')
    if body.get('response_kind') == 'science_campaign_action':
        classification = body.get('selected_action') or classification
    gate = body.get('evidence_gate') or evidence.get('evidence_gate')
    status = str(body.get('status') or evidence.get('status') or '').lower()
    required = []
    sibling = list(body.get('sibling_evidence_used') or [])
    if gate:
        if status in {'missing', 'needs_clarification'}:
            required.append(str(gate))
        elif message.get('sender') in {'funfun', 'code_module', 'language_model_2'}:
            sibling.append({'sender': message.get('sender'), 'evidence_gate': str(gate), 'status': status or 'advisory'})
    leaks = label_leak_terms({'body': body, 'evidence': evidence})
    third_party = bool(body.get('third_party_checkpoint_used') or evidence.get('third_party_checkpoint_used'))
    if leaks or third_party:
        classification = 'connected_prevents_boundary_or_checkpoint_overclaim'
    return {
        'benefit_id': body.get('benefit_id') or evidence.get('benefit_id'),
        'scenario_id': str(scenario_id) if scenario_id else None,
        'hypothesis_id': body.get('hypothesis_id') or evidence.get('hypothesis_id'),
        'benefit_classification': classification,
        'required_evidence': list(body.get('missing_evidence') or body.get('required_evidence') or required),
        'sibling_evidence_used': sibling,
        'theory_update_delta': body.get('theory_update_delta') or body.get('theory_update_intent'),
        'refinement_state': body.get('refinement_delta') or body.get('refinement_state') or 'none',
        'retirement_state': body.get('retirement_delta') or body.get('retirement_state') or 'none',
        'boundary_checkpoint_state': 'repair' if leaks or third_party else body.get('boundary_checkpoint_state') or 'clean',
        'boundary_notes': list(body.get('boundary_notes') or ([] if not leaks else ['label leak detected'])),
        'label_leaks': leaks,
        'lineage': [f'message:{message.get("sender")}'],
    }


def _upsert_row(rows, incoming):
    scenario_id = incoming.get('scenario_id')
    if not scenario_id:
        return
    current = rows.setdefault(str(scenario_id), {
        'scenario_id': str(scenario_id),
        'benefit_id': incoming.get('benefit_id'),
        'hypothesis_id': incoming.get('hypothesis_id'),
        'benefit_classification': incoming.get('benefit_classification'),
        'required_evidence': [],
        'sibling_evidence_used': [],
        'theory_update_delta': None,
        'refinement_state': 'none',
        'retirement_state': 'none',
        'boundary_checkpoint_state': 'clean',
        'boundary_notes': [],
        'label_leaks': [],
        'lineage': [],
    })
    current['benefit_id'] = current.get('benefit_id') or incoming.get('benefit_id')
    current['hypothesis_id'] = current.get('hypothesis_id') or incoming.get('hypothesis_id')
    current['benefit_classification'] = _dominant_classification(current.get('benefit_classification'), incoming.get('benefit_classification'))
    current['required_evidence'] = _unique_strings(list(current.get('required_evidence') or []) + list(incoming.get('required_evidence') or []))
    current['sibling_evidence_used'] = _unique_dicts(list(current.get('sibling_evidence_used') or []) + list(incoming.get('sibling_evidence_used') or []))
    current['theory_update_delta'] = incoming.get('theory_update_delta') or current.get('theory_update_delta')
    current['refinement_state'] = _dominant_state(current.get('refinement_state'), incoming.get('refinement_state'))
    current['retirement_state'] = _dominant_state(current.get('retirement_state'), incoming.get('retirement_state'))
    current['boundary_checkpoint_state'] = 'repair' if incoming.get('boundary_checkpoint_state') == 'repair' or current.get('boundary_checkpoint_state') == 'repair' else 'clean'
    current['boundary_notes'] = _unique_strings(list(current.get('boundary_notes') or []) + list(incoming.get('boundary_notes') or []))
    current['label_leaks'] = _unique_strings(list(current.get('label_leaks') or []) + list(incoming.get('label_leaks') or []))
    current['lineage'] = _unique_strings(list(current.get('lineage') or []) + list(incoming.get('lineage') or []))


def _finalize_row(row):
    classification = row.get('benefit_classification') or 'insufficient_evidence'
    if row.get('boundary_checkpoint_state') == 'repair' or row.get('label_leaks'):
        classification = 'connected_prevents_boundary_or_checkpoint_overclaim'
    row['benefit_classification'] = classification
    row['required_evidence'] = _unique_strings(row.get('required_evidence') or [])
    row['action_hash'] = stable_digest({
        'scenario_id': row.get('scenario_id'),
        'classification': classification,
        'required': row.get('required_evidence'),
        'sibling': row.get('sibling_evidence_used'),
    })
    return row


def _select_action(*, rows, planned_keys, project_owned_boundary):
    for row in rows:
        if project_owned_boundary.get('third_party_checkpoint_used') or row.get('benefit_classification') == 'connected_prevents_boundary_or_checkpoint_overclaim':
            return _action(row, 'preserve_boundary_or_checkpoint_repair', 'code_module', ['project_owned_checkpoint_boundary_explicit'])
    for row in rows:
        if row.get('benefit_classification') == 'connected_retires_failed_line' or row.get('retirement_state') == 'connected_added':
            return _action(row, 'retire_failed_hypothesis_line', 'broadcast', [])
    for gate, action_name, recipient in (
        ('math_proof', 'request_funfun_math_certificate', 'funfun'),
        ('code_proof', 'request_code_experiment_or_counterexample', 'code_module'),
        ('language_epoch_plan', 'request_language_protocol_clarification', 'language_model_2'),
    ):
        for row in rows:
            if gate in set(row.get('required_evidence') or []) and _action_key_for(row, action_name) not in set(planned_keys):
                return _action(row, action_name, recipient, [gate])
    for row in rows:
        if row.get('benefit_classification') == 'connected_refines_with_clearer_evidence' or row.get('refinement_state') == 'connected_added':
            return _action(row, 'refine_hypothesis_with_connected_evidence', 'broadcast', [])
    for row in rows:
        if row.get('benefit_classification') == 'connected_accepts_with_verified_math_code_language':
            return _action(row, 'schedule_next_campaign_check', 'orchestrator', list(GATE_ORDER))
    for row in rows:
        if row.get('benefit_classification') in {'connected_adds_no_safe_benefit', 'isolated_already_sufficient'}:
            return _action(row, 'record_connected_no_safe_benefit', 'orchestrator', [])
    return _noop_action('no science campaign action selected')


def _action(row, selected_action, recipient, required):
    return {
        'selected_action': selected_action,
        'selected_recipient': validate_participant(recipient),
        'benefit_id': row.get('benefit_id'),
        'scenario_id': row.get('scenario_id'),
        'hypothesis_id': row.get('hypothesis_id'),
        'required_evidence': list(required or row.get('required_evidence') or []),
        'sibling_evidence_used': list(row.get('sibling_evidence_used') or []),
        'theory_update_intent': _intent_for(selected_action, row),
        'refinement_state': row.get('refinement_state') or 'none',
        'retirement_state': row.get('retirement_state') or 'none',
        'boundary_checkpoint_state': row.get('boundary_checkpoint_state') or 'clean',
        'boundary_notes': list(row.get('boundary_notes') or []),
        'recommended_action': selected_action,
        'label_leaks': list(row.get('label_leaks') or []),
    }


def _noop_action(reason):
    return {
        'selected_action': 'summarize_noop',
        'selected_recipient': None,
        'benefit_id': None,
        'scenario_id': None,
        'hypothesis_id': None,
        'required_evidence': [],
        'sibling_evidence_used': [],
        'theory_update_intent': reason,
        'refinement_state': 'none',
        'retirement_state': 'none',
        'boundary_checkpoint_state': 'clean',
        'boundary_notes': [],
        'recommended_action': 'noop',
        'label_leaks': [],
    }


def _intent_for(action, row):
    return {
        'preserve_boundary_or_checkpoint_repair': 'preserve safety/project-owned checkpoint repair before further campaign work',
        'retire_failed_hypothesis_line': 'retire the failed hypothesis line using connected evidence',
        'request_funfun_math_certificate': 'request math certificate for the next missing evidence gate',
        'request_code_experiment_or_counterexample': 'request code experiment or counterexample for the next missing evidence gate',
        'request_language_protocol_clarification': 'request protocol clarification for the next missing evidence gate',
        'refine_hypothesis_with_connected_evidence': 'refine the hypothesis using connected evidence delta',
        'schedule_next_campaign_check': 'schedule the next campaign check after connected evidence satisfied all gates',
        'record_connected_no_safe_benefit': 'record that connected evidence adds no safe actionable benefit',
    }.get(action, row.get('theory_update_delta') or 'summarize action state')


def _state_counts(selected, rows):
    counts = {
        'repair': 0,
        'retire': 0,
        'math': 0,
        'code': 0,
        'language': 0,
        'refine': 0,
        'next': 0,
        'no_benefit': 0,
    }
    mapping = {
        'preserve_boundary_or_checkpoint_repair': 'repair',
        'retire_failed_hypothesis_line': 'retire',
        'request_funfun_math_certificate': 'math',
        'request_code_experiment_or_counterexample': 'code',
        'request_language_protocol_clarification': 'language',
        'refine_hypothesis_with_connected_evidence': 'refine',
        'schedule_next_campaign_check': 'next',
        'record_connected_no_safe_benefit': 'no_benefit',
    }
    key = mapping.get(selected.get('selected_action'))
    if key:
        counts[key] += 1
    if selected.get('selected_action') == 'summarize_noop' and not rows:
        counts['no_benefit'] += 1
    return counts


def _source_hash(benefit, evaluator, prior_outcome, contracts, adjudicator, agenda, lifecycle, scorecard, campaign, campaign_outcome, prior_action):
    return stable_digest({
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
        'prior_action': prior_action.get('ledger_hash'),
    })


def _classification_from_record(record):
    if record.get('benefit_classification'):
        return record.get('benefit_classification')
    selected = str(record.get('selected_action') or '')
    return {
        'repair_boundary': 'connected_prevents_boundary_or_checkpoint_overclaim',
        'retire_theory_line': 'connected_retires_failed_line',
        'accept_campaign': 'connected_accepts_with_verified_math_code_language',
        'refine_hypothesis': 'connected_refines_with_clearer_evidence',
    }.get(selected, 'connected_adds_no_safe_benefit')


def _missing_from_record(record):
    values = record.get('missing_evidence') or []
    if isinstance(values, dict):
        output = []
        for item in values.values():
            output.extend(list(item or []))
        return output
    return list(values or [])


def _sibling_evidence_from_connected(connected):
    used = []
    for module, values in dict(connected.get('evidence_by_module') or {}).items():
        for field in ('accepted', 'missing', 'rejected'):
            for gate in values.get(field) or []:
                used.append({'sender': module, 'evidence_gate': gate, 'status': field})
    return used


def _dominant_classification(current, incoming):
    order = {
        'connected_prevents_boundary_or_checkpoint_overclaim': 9,
        'connected_retires_failed_line': 8,
        'connected_accepts_with_verified_math_code_language': 7,
        'connected_refines_with_clearer_evidence': 6,
        'connected_requests_missing_targeted_evidence': 5,
        'connected_adds_no_safe_benefit': 4,
        'isolated_already_sufficient': 3,
        'insufficient_evidence': 2,
        'summarize_noop': 1,
    }
    current_value = str(current or 'insufficient_evidence')
    incoming_value = str(incoming or 'insufficient_evidence')
    return current_value if order.get(current_value, 0) >= order.get(incoming_value, 0) else incoming_value


def _dominant_state(current, incoming):
    order = {'connected_added': 3, 'unchanged': 2, 'none': 1}
    current_value = str(current or 'none')
    incoming_value = str(incoming or 'none')
    return current_value if order.get(current_value, 0) >= order.get(incoming_value, 0) else incoming_value


def _find_row(rows, scenario_id):
    for row in rows:
        if row.get('scenario_id') == scenario_id:
            return row
    return None


def _action_key(selected):
    return _action_key_for(selected, selected.get('selected_action'))


def _action_key_for(row, action):
    return stable_digest({
        'scenario_id': row.get('scenario_id'),
        'benefit_id': row.get('benefit_id'),
        'hypothesis_id': row.get('hypothesis_id'),
        'action': action,
    })


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
