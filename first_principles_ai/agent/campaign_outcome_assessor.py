"""Campaign outcome assessor and theory-update planner.

This layer reads returned module evidence for symbolic experiment campaigns and
decides whether the campaign is accepted, waiting on evidence, refined, retired,
or blocked by a boundary repair. It is plain-data orchestration only: no heavy
experiment run, no learned-model claim, and no sibling project imports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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


CAMPAIGN_OUTCOME_LEDGER_KIND = 'ai_different.experiment_campaign_outcome_ledger'
GATE_ORDER = ('math_proof', 'code_proof', 'language_epoch_plan')
REQUEST_ACTIONS = {
    'math_proof': ('request_more_math', 'funfun'),
    'code_proof': ('request_more_code', 'code_module'),
    'language_epoch_plan': ('request_more_language', 'language_model_2'),
}


def empty_campaign_outcome_ledger() -> dict[str, Any]:
    return {
        'schema_version': 1,
        'ledger_kind': CAMPAIGN_OUTCOME_LEDGER_KIND,
        'processed_message_ids': [],
        'processed_source_hashes': [],
        'accepted_campaign_ids': [],
        'refined_hypothesis_ids': [],
        'retired_hypothesis_ids': [],
        'repaired_campaign_ids': [],
        'outcomes': [],
        'outcome_records': [],
        'outgoing_response_ids': [],
        'latest': {},
    }


def load_campaign_outcome_ledger(path: str | Path | None) -> dict[str, Any]:
    if not path or not Path(path).exists():
        return empty_campaign_outcome_ledger()
    with Path(path).open('r', encoding='utf-8') as handle:
        ledger = json.load(handle)
    return validate_campaign_outcome_ledger(ledger)


def write_campaign_outcome_ledger(path: str | Path, ledger: dict[str, Any]) -> Path:
    validated = validate_campaign_outcome_ledger(ledger)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(validated, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return output


def validate_campaign_outcome_ledger(ledger: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(ledger, dict):
        raise ValueError('campaign outcome ledger must be a JSON object')
    if ledger.get('ledger_kind') != CAMPAIGN_OUTCOME_LEDGER_KIND:
        raise ValueError('campaign outcome ledger has wrong ledger_kind')
    for field in (
        'processed_message_ids',
        'processed_source_hashes',
        'accepted_campaign_ids',
        'refined_hypothesis_ids',
        'retired_hypothesis_ids',
        'repaired_campaign_ids',
        'outgoing_response_ids',
    ):
        values = ledger.get(field)
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            raise ValueError(f'{field} must be strings')
    for field in ('outcomes', 'outcome_records'):
        values = ledger.get(field)
        if not isinstance(values, list) or not all(isinstance(item, dict) for item in values):
            raise ValueError(f'{field} must be objects')
    latest = ledger.get('latest', {})
    if not isinstance(latest, dict):
        raise ValueError('campaign outcome latest must be an object')
    validated = {
        'schema_version': int(ledger.get('schema_version', 1) or 1),
        'ledger_kind': CAMPAIGN_OUTCOME_LEDGER_KIND,
        'processed_message_ids': _unique_strings(ledger['processed_message_ids']),
        'processed_source_hashes': _unique_strings(ledger['processed_source_hashes']),
        'accepted_campaign_ids': _unique_strings(ledger['accepted_campaign_ids']),
        'refined_hypothesis_ids': _unique_strings(ledger['refined_hypothesis_ids']),
        'retired_hypothesis_ids': _unique_strings(ledger['retired_hypothesis_ids']),
        'repaired_campaign_ids': _unique_strings(ledger['repaired_campaign_ids']),
        'outcomes': list(ledger['outcomes']),
        'outcome_records': list(ledger['outcome_records']),
        'outgoing_response_ids': _unique_strings(ledger['outgoing_response_ids']),
        'latest': dict(latest),
    }
    if ledger.get('ledger_hash'):
        validated['ledger_hash'] = str(ledger['ledger_hash'])
    if ledger.get('artifact_path'):
        validated['artifact_path'] = str(ledger['artifact_path'])
    return validated


def read_campaign_outcome_transcript(path: str | Path | None) -> dict[str, Any]:
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
        raise ValueError('plain campaign outcome input must be a JSON object')
    return value


def build_campaign_outcome_assessment(
    *,
    transcript_messages: list[dict[str, Any]],
    campaign_outcome_ledger: dict[str, Any],
    evaluator_ledger: dict[str, Any] | None = None,
    outcome_ledger: dict[str, Any] | None = None,
    contract_ledger: dict[str, Any] | None = None,
    adjudicator_ledger: dict[str, Any] | None = None,
    agenda_ledger: dict[str, Any] | None = None,
    lifecycle_ledger: dict[str, Any] | None = None,
    scorecard_ledger: dict[str, Any] | None = None,
    campaign_ledger: dict[str, Any] | None = None,
    runtime_memory_data: dict[str, Any] | None = None,
    runtime_memory_hash_state: dict[str, Any],
    project_owned_boundary: dict[str, Any],
    artifact_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    del runtime_memory_data
    ledger = validate_campaign_outcome_ledger(campaign_outcome_ledger)
    evaluator = _valid_evaluator_or_empty(evaluator_ledger or {})
    prior_outcome = _valid_prior_outcome_or_empty(outcome_ledger or {})
    contracts = _valid_contract_or_empty(contract_ledger or {})
    adjudicator = _valid_adjudicator_or_empty(adjudicator_ledger or {})
    agenda = _valid_agenda_or_empty(agenda_ledger or {})
    lifecycle = _valid_lifecycle_or_empty(lifecycle_ledger or {})
    scorecard = _valid_scorecard_or_empty(scorecard_ledger or {})
    campaign = _valid_campaign_or_empty(campaign_ledger or {})
    source_hash = _source_hash(evaluator, prior_outcome, contracts, adjudicator, agenda, lifecycle, scorecard, campaign)
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
    outcomes = _merge_outcomes(ledger['outcomes'], campaign, scorecard, lifecycle, agenda, contracts, transcript_messages)
    evidence_items = _extract_outcome_evidence(new_messages, outcomes)
    _apply_evidence_to_outcomes(outcomes, evidence_items)
    if not new_messages and (not source_is_new or outcomes):
        selected = _noop_action('no new campaign outcome evidence or source ledger state')
    else:
        selected = _select_outcome_action(
            outcomes=outcomes,
            evidence_items=evidence_items,
            accepted_ids=ledger['accepted_campaign_ids'],
            refined_ids=ledger['refined_hypothesis_ids'],
            retired_ids=ledger['retired_hypothesis_ids'],
            project_owned_boundary=project_owned_boundary,
        )
    outcome_id = 'campaign_outcome_' + stable_digest({
        'source_hash': source_hash,
        'new_message_ids': [module_chat_message_id(message) for message in new_messages],
        'selected_action': selected['selected_action'],
        'campaign_id': selected.get('campaign_id'),
        'hypothesis_id': selected.get('hypothesis_id'),
    })[:16]
    selected['outcome_id'] = outcome_id
    message = export_campaign_outcome_message(
        selected,
        outcomes=outcomes,
        evidence_items=evidence_items,
        runtime_memory_hash_state=runtime_memory_hash_state,
        project_owned_boundary=project_owned_boundary,
        source_hashes={
            'campaign_outcome_source_hash': source_hash,
            'evaluator_ledger_hash': evaluator.get('ledger_hash'),
            'outcome_ledger_hash': prior_outcome.get('ledger_hash'),
            'contract_ledger_hash': contracts.get('ledger_hash'),
            'adjudicator_ledger_hash': adjudicator.get('ledger_hash'),
            'agenda_ledger_hash': agenda.get('ledger_hash'),
            'lifecycle_ledger_hash': lifecycle.get('ledger_hash'),
            'scorecard_ledger_hash': scorecard.get('ledger_hash'),
            'campaign_ledger_hash': campaign.get('ledger_hash'),
        },
    )
    new_ids = [module_chat_message_id(message) for message in new_messages]
    outgoing_id = module_chat_message_id(message) if message else None
    campaign_id = selected.get('campaign_id')
    hypothesis_id = selected.get('hypothesis_id')
    action = selected['selected_action']
    if action == 'accept_campaign' and campaign_id:
        ledger['accepted_campaign_ids'] = _unique_strings(list(ledger['accepted_campaign_ids']) + [campaign_id])
    if action == 'refine_hypothesis' and hypothesis_id:
        ledger['refined_hypothesis_ids'] = _unique_strings(list(ledger['refined_hypothesis_ids']) + [hypothesis_id])
    if action == 'retire_theory_line' and hypothesis_id:
        ledger['retired_hypothesis_ids'] = _unique_strings(list(ledger['retired_hypothesis_ids']) + [hypothesis_id])
    if action == 'repair_boundary' and campaign_id:
        ledger['repaired_campaign_ids'] = _unique_strings(list(ledger['repaired_campaign_ids']) + [campaign_id])
    if new_messages:
        ledger['processed_message_ids'] = _unique_strings(list(ledger['processed_message_ids']) + new_ids)
    if source_is_new and action in {'accept_campaign', 'refine_hypothesis', 'noop'}:
        ledger['processed_source_hashes'] = _unique_strings(list(ledger['processed_source_hashes']) + [source_hash])
    if outgoing_id:
        ledger['outgoing_response_ids'] = _unique_strings(list(ledger['outgoing_response_ids']) + [outgoing_id])
    status_counts = _outcome_status_counts(outcomes, selected)
    latest = {
        'outcome_id': outcome_id,
        'new_message_count': len(new_messages),
        'skipped_message_count': len(skipped_messages),
        'source_is_new': source_is_new,
        'status_counts': status_counts,
        'selected_action': action,
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
        'campaign_ids': _unique_strings(_collect_field(outcomes, 'campaign_id')),
        'hypothesis_ids': _unique_strings(_collect_field(outcomes, 'hypothesis_id')),
        'source_scorecard_ids': _unique_strings(_collect_list_field(outcomes, 'source_scorecard_ids')),
        'source_lifecycle_ids': _unique_strings(_collect_list_field(outcomes, 'source_lifecycle_ids')),
        'source_agenda_ids': _unique_strings(_collect_list_field(outcomes, 'source_agenda_ids')),
        'source_contract_ids': _unique_strings(_collect_list_field(outcomes, 'source_contract_ids')),
        'acceptance_criteria': {item.get('campaign_id'): item.get('acceptance_criteria') for item in outcomes},
        'accepted_evidence': _field_map(outcomes, 'accepted_evidence'),
        'missing_evidence': _field_map(outcomes, 'missing_evidence'),
        'rejected_evidence': _field_map(outcomes, 'rejected_evidence'),
        'gate_results': {item.get('campaign_id'): item.get('gate_results') for item in outcomes},
        'theory_update_action': action,
        'updated_hypothesis_notes': selected.get('updated_hypothesis_notes'),
        'chosen_recipient': latest['chosen_recipient'],
        'outgoing_response_id': outgoing_id,
        'runtime_memory_hash_state': dict(runtime_memory_hash_state),
        'source_ledger_hashes': {
            'evaluator_ledger_hash': evaluator.get('ledger_hash'),
            'outcome_ledger_hash': prior_outcome.get('ledger_hash'),
            'contract_ledger_hash': contracts.get('ledger_hash'),
            'adjudicator_ledger_hash': adjudicator.get('ledger_hash'),
            'agenda_ledger_hash': agenda.get('ledger_hash'),
            'lifecycle_ledger_hash': lifecycle.get('ledger_hash'),
            'scorecard_ledger_hash': scorecard.get('ledger_hash'),
            'campaign_ledger_hash': campaign.get('ledger_hash'),
        },
    }
    ledger['outcomes'] = outcomes
    if new_messages or source_is_new or message is not None:
        ledger['outcome_records'] = list(ledger['outcome_records']) + [record]
    ledger['latest'] = latest
    if artifact_path is not None:
        ledger['artifact_path'] = str(Path(artifact_path))
    ledger['ledger_hash'] = stable_digest({key: value for key, value in ledger.items() if key != 'ledger_hash'})
    return ledger, message


def export_campaign_outcome_message(
    selected: dict[str, Any],
    *,
    outcomes: list[dict[str, Any]],
    evidence_items: list[dict[str, Any]],
    runtime_memory_hash_state: dict[str, Any],
    project_owned_boundary: dict[str, Any],
    source_hashes: dict[str, Any],
    topic: str = 'ai_different.experiment_campaign_outcome',
) -> dict[str, Any] | None:
    if selected['selected_action'] == 'noop':
        return None
    recipient = validate_participant(selected.get('chosen_recipient') or 'orchestrator')
    outcome = _find_outcome(outcomes, selected.get('campaign_id'), selected.get('hypothesis_id')) or {}
    leak_terms = label_leak_terms({'selected': selected, 'outcome': outcome, 'evidence': evidence_items})
    body = {
        'module': 'AI Different',
        'response_kind': 'experiment_campaign_outcome',
        'outcome_id': selected.get('outcome_id'),
        'campaign_id': selected.get('campaign_id'),
        'hypothesis_id': selected.get('hypothesis_id'),
        'selected_action': selected['selected_action'],
        'theory_update_action': selected['selected_action'],
        'priority_reason': selected.get('priority_reason'),
        'gate_verdicts': selected.get('gate_verdicts') or outcome.get('gate_results'),
        'accepted_evidence': selected.get('accepted_evidence') or outcome.get('accepted_evidence'),
        'missing_evidence': selected.get('missing_evidence') or outcome.get('missing_evidence'),
        'rejected_evidence': selected.get('rejected_evidence') or outcome.get('rejected_evidence'),
        'acceptance_criteria': selected.get('acceptance_criteria') or outcome.get('acceptance_criteria'),
        'theory_update_notes': selected.get('updated_hypothesis_notes'),
        'recommended_recipient': recipient,
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
            'campaign_id': body['campaign_id'],
            'hypothesis_id': body['hypothesis_id'],
            'selected_action': body['selected_action'],
            'gate_verdicts': body['gate_verdicts'],
            'runtime_memory_hash_state': body['runtime_memory_hash_state'],
            'runtime_memory_mutated': body['runtime_memory_mutated'],
            'label_clean': body['label_clean'],
            'no_sibling_imports': True,
        },
        tags=['ai_different', 'experiment_campaign_outcome', body['selected_action'], 'label_clean' if body['label_clean'] else 'label_review_needed'],
    )


def write_campaign_outcome_outbox_jsonl(path: str | Path, message: dict[str, Any] | None) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text((json.dumps(message, sort_keys=True) + '\n') if message is not None else '', encoding='utf-8')
    return output


def _valid_evaluator_or_empty(ledger: dict[str, Any]) -> dict[str, Any]:
    if not ledger:
        return {}
    if ledger.get('ledger_kind') != EVALUATOR_LEDGER_KIND:
        raise ValueError('evaluator ledger has wrong ledger_kind')
    return validate_evaluator_ledger(ledger)


def _valid_prior_outcome_or_empty(ledger: dict[str, Any]) -> dict[str, Any]:
    if not ledger:
        return {}
    if ledger.get('ledger_kind') not in {EVALUATOR_LEDGER_KIND, 'ai_different.outcome_ledger'}:
        raise ValueError('outcome ledger has wrong ledger_kind')
    return dict(ledger)


def _valid_contract_or_empty(ledger: dict[str, Any]) -> dict[str, Any]:
    if not ledger:
        return {'ledger_kind': CONTRACT_LEDGER_KIND, 'contracts': [], 'ledger_hash': None}
    if ledger.get('ledger_kind') != CONTRACT_LEDGER_KIND:
        raise ValueError('contract ledger has wrong ledger_kind')
    return dict(ledger)


def _valid_adjudicator_or_empty(ledger: dict[str, Any]) -> dict[str, Any]:
    if not ledger:
        return {'ledger_kind': ADJUDICATOR_LEDGER_KIND, 'contract_states': [], 'ledger_hash': None}
    if ledger.get('ledger_kind') != ADJUDICATOR_LEDGER_KIND:
        raise ValueError('adjudicator ledger has wrong ledger_kind')
    return dict(ledger)


def _valid_agenda_or_empty(ledger: dict[str, Any]) -> dict[str, Any]:
    if not ledger:
        return {'ledger_kind': AGENDA_LEDGER_KIND, 'hypotheses': [], 'agenda_records': [], 'ledger_hash': None}
    if ledger.get('ledger_kind') != AGENDA_LEDGER_KIND:
        raise ValueError('agenda ledger has wrong ledger_kind')
    return dict(ledger)


def _valid_lifecycle_or_empty(ledger: dict[str, Any]) -> dict[str, Any]:
    if not ledger:
        return {'ledger_kind': LIFECYCLE_LEDGER_KIND, 'hypotheses': [], 'lifecycle_records': [], 'ledger_hash': None}
    if ledger.get('ledger_kind') != LIFECYCLE_LEDGER_KIND:
        raise ValueError('lifecycle ledger has wrong ledger_kind')
    return dict(ledger)


def _valid_scorecard_or_empty(ledger: dict[str, Any]) -> dict[str, Any]:
    if not ledger:
        return {'ledger_kind': SCORECARD_LEDGER_KIND, 'scorecards': [], 'scorecard_records': [], 'ledger_hash': None}
    if ledger.get('ledger_kind') != SCORECARD_LEDGER_KIND:
        raise ValueError('scorecard ledger has wrong ledger_kind')
    return dict(ledger)


def _valid_campaign_or_empty(ledger: dict[str, Any]) -> dict[str, Any]:
    if not ledger:
        return {'ledger_kind': CAMPAIGN_LEDGER_KIND, 'campaigns': [], 'campaign_records': [], 'ledger_hash': None}
    if ledger.get('ledger_kind') != CAMPAIGN_LEDGER_KIND:
        raise ValueError('campaign ledger has wrong ledger_kind')
    return dict(ledger)


def _merge_outcomes(existing, campaign_ledger, scorecard_ledger, lifecycle_ledger, agenda_ledger, contract_ledger, messages):
    outcomes: dict[str, dict[str, Any]] = {
        str(item.get('campaign_id') or item.get('hypothesis_id') or item.get('contract_id')): dict(item)
        for item in existing
        if item.get('campaign_id') or item.get('hypothesis_id') or item.get('contract_id')
    }
    for campaign in list(campaign_ledger.get('campaigns') or []):
        _upsert_outcome(outcomes, _outcome_from_campaign(campaign, 'campaign_ledger'))
    for record in list(campaign_ledger.get('campaign_records') or []):
        _upsert_outcome(outcomes, _outcome_from_campaign_record(record))
    for card in list(scorecard_ledger.get('scorecards') or []):
        _upsert_outcome(outcomes, _outcome_from_gate_source(card, 'scorecard_ledger'))
    for hypothesis in list(lifecycle_ledger.get('hypotheses') or []):
        _upsert_outcome(outcomes, _outcome_from_gate_source(hypothesis, 'lifecycle_ledger'))
    for hypothesis in list(agenda_ledger.get('hypotheses') or []):
        _upsert_outcome(outcomes, _outcome_from_gate_source(hypothesis, 'agenda_ledger'))
    for contract in list(contract_ledger.get('contracts') or []):
        _upsert_outcome(outcomes, _outcome_from_gate_source(contract, 'contract_ledger'))
    for message in messages:
        body = dict(message.get('body') or {})
        if message.get('sender') != 'ai_different':
            continue
        if body.get('response_kind') == 'experiment_campaign':
            _upsert_outcome(outcomes, _outcome_from_campaign(body, 'campaign_message'))
        elif body.get('response_kind') == 'experiment_campaign_outcome':
            _upsert_outcome(outcomes, _outcome_from_prior_outcome_message(body))
    return [_finalize_outcome(item) for item in outcomes.values()]


def _outcome_from_campaign(item: dict[str, Any], source: str) -> dict[str, Any]:
    campaign_id = item.get('campaign_id') or item.get('campaign_hash')
    hypothesis_id = item.get('hypothesis_id')
    contract_id = item.get('contract_id')
    if not hypothesis_id and contract_id:
        hypothesis_id = f'hypothesis:{contract_id}'
    return {
        'campaign_id': str(campaign_id) if campaign_id else None,
        'hypothesis_id': str(hypothesis_id) if hypothesis_id else None,
        'contract_id': str(contract_id) if contract_id else None,
        'source_campaign_ids': [str(campaign_id)] if campaign_id else [],
        'source_contract_ids': list(item.get('source_contract_ids') or ([contract_id] if contract_id else [])),
        'source_agenda_ids': list(item.get('source_agenda_ids') or []),
        'source_lifecycle_ids': list(item.get('source_lifecycle_ids') or []),
        'source_scorecard_ids': list(item.get('source_scorecard_ids') or []),
        'acceptance_criteria': _criteria_from_item(item),
        'required_evidence': _required_gates(item),
        'accepted_evidence': list(item.get('accepted_evidence') or []),
        'missing_evidence': list(item.get('missing_evidence') or []),
        'rejected_evidence': list(item.get('rejected_evidence') or []),
        'campaign_type': item.get('campaign_type'),
        'lineage': [source],
    }


def _outcome_from_campaign_record(record: dict[str, Any]) -> dict[str, Any]:
    campaign_id = record.get('campaign_id')
    hypothesis_ids = list(record.get('hypothesis_ids') or [])
    selected = record.get('selected_action')
    accepted_map = dict(record.get('accepted_evidence') or {})
    missing_map = dict(record.get('missing_evidence') or {})
    rejected_map = dict(record.get('rejected_evidence') or {})
    hypothesis_id = hypothesis_ids[0] if hypothesis_ids else None
    return {
        'campaign_id': str(campaign_id) if campaign_id else None,
        'hypothesis_id': str(hypothesis_id) if hypothesis_id else None,
        'contract_id': None,
        'source_campaign_ids': [str(campaign_id)] if campaign_id else [],
        'source_contract_ids': list(record.get('source_contract_ids') or []),
        'source_agenda_ids': list(record.get('source_agenda_ids') or []),
        'source_lifecycle_ids': list(record.get('source_lifecycle_ids') or []),
        'source_scorecard_ids': list(record.get('source_scorecard_ids') or []),
        'acceptance_criteria': dict(record.get('acceptance_criteria') or {}).get(hypothesis_id) or [],
        'required_evidence': list(GATE_ORDER),
        'accepted_evidence': accepted_map.get(hypothesis_id, []),
        'missing_evidence': missing_map.get(hypothesis_id, []),
        'rejected_evidence': rejected_map.get(hypothesis_id, []),
        'campaign_type': selected,
        'lineage': ['campaign_record'],
    }


def _outcome_from_gate_source(item: dict[str, Any], source: str) -> dict[str, Any]:
    return _outcome_from_campaign({
        'campaign_id': item.get('campaign_id'),
        'hypothesis_id': item.get('hypothesis_id'),
        'contract_id': item.get('contract_id'),
        'source_contract_ids': item.get('source_contract_ids') or [],
        'source_agenda_ids': item.get('source_agenda_ids') or [],
        'source_lifecycle_ids': item.get('source_lifecycle_ids') or [],
        'source_scorecard_ids': item.get('source_scorecard_ids') or [],
        'required_evidence_gates': item.get('required_evidence_gates') or item.get('required_evidence') or [],
        'accepted_evidence': item.get('accepted_evidence_gates') or item.get('accepted_evidence') or [],
        'missing_evidence': item.get('missing_evidence_gates') or item.get('missing_evidence') or [],
        'rejected_evidence': item.get('rejected_evidence_gates') or item.get('rejected_evidence') or [],
    }, source)


def _outcome_from_prior_outcome_message(body: dict[str, Any]) -> dict[str, Any]:
    item = _outcome_from_campaign(body, 'campaign_outcome_message')
    item['accepted_evidence'] = list(body.get('accepted_evidence') or [])
    item['missing_evidence'] = list(body.get('missing_evidence') or [])
    item['rejected_evidence'] = list(body.get('rejected_evidence') or [])
    return item


def _upsert_outcome(outcomes: dict[str, dict[str, Any]], incoming: dict[str, Any]):
    key = str(incoming.get('campaign_id') or incoming.get('hypothesis_id') or incoming.get('contract_id') or '')
    if not key:
        return
    current = outcomes.setdefault(key, {})
    current.update({
        'campaign_id': incoming.get('campaign_id') or current.get('campaign_id'),
        'hypothesis_id': incoming.get('hypothesis_id') or current.get('hypothesis_id'),
        'contract_id': incoming.get('contract_id') or current.get('contract_id'),
        'source_campaign_ids': _unique_strings(list(current.get('source_campaign_ids') or []) + list(incoming.get('source_campaign_ids') or [])),
        'source_contract_ids': _unique_strings(list(current.get('source_contract_ids') or []) + list(incoming.get('source_contract_ids') or [])),
        'source_agenda_ids': _unique_strings(list(current.get('source_agenda_ids') or []) + list(incoming.get('source_agenda_ids') or [])),
        'source_lifecycle_ids': _unique_strings(list(current.get('source_lifecycle_ids') or []) + list(incoming.get('source_lifecycle_ids') or [])),
        'source_scorecard_ids': _unique_strings(list(current.get('source_scorecard_ids') or []) + list(incoming.get('source_scorecard_ids') or [])),
        'acceptance_criteria': list(incoming.get('acceptance_criteria') or current.get('acceptance_criteria') or []),
        'required_evidence': _unique_strings(list(current.get('required_evidence') or []) + list(incoming.get('required_evidence') or [])),
        'accepted_evidence': _unique_strings(list(current.get('accepted_evidence') or []) + list(incoming.get('accepted_evidence') or [])),
        'missing_evidence': _unique_strings(list(current.get('missing_evidence') or []) + list(incoming.get('missing_evidence') or [])),
        'rejected_evidence': _unique_strings(list(current.get('rejected_evidence') or []) + list(incoming.get('rejected_evidence') or [])),
        'campaign_type': incoming.get('campaign_type') or current.get('campaign_type'),
        'lineage': _unique_strings(list(current.get('lineage') or []) + list(incoming.get('lineage') or [])),
    })


def _extract_outcome_evidence(messages: list[dict[str, Any]], outcomes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    known_campaign_ids = {item.get('campaign_id') for item in outcomes}
    known_contract_ids = {item.get('contract_id') for item in outcomes}
    items = []
    for message in messages:
        body = dict(message.get('body') or {})
        evidence = dict(message.get('evidence') or {})
        sender = message.get('sender')
        campaign_id = body.get('campaign_id') or evidence.get('campaign_id')
        contract_id = body.get('contract_id') or evidence.get('contract_id')
        hypothesis_id = body.get('hypothesis_id') or evidence.get('hypothesis_id')
        if not hypothesis_id and contract_id:
            hypothesis_id = f'hypothesis:{contract_id}'
        item = {
            'evidence_id': str(body.get('evidence_id') or evidence.get('evidence_id') or module_chat_message_id(message)),
            'message_id': module_chat_message_id(message),
            'sender': sender,
            'topic': message.get('topic'),
            'campaign_id': str(campaign_id) if campaign_id else None,
            'hypothesis_id': str(hypothesis_id) if hypothesis_id else None,
            'contract_id': str(contract_id) if contract_id else None,
            'evidence_gate': _evidence_gate(message),
            'status': _evidence_status(message),
            'summary': body.get('summary') or body.get('note') or body.get('reason'),
            'label_leaks': label_leak_terms({'body': body, 'evidence': evidence}),
            'third_party_checkpoint_used': bool(body.get('third_party_checkpoint_used') or evidence.get('third_party_checkpoint_used')),
        }
        if (
            item['campaign_id'] in known_campaign_ids
            or item['contract_id'] in known_contract_ids
            or item['hypothesis_id']
            or sender in {'language_model_2', 'funfun', 'code_module'}
        ):
            items.append(item)
    return items


def _apply_evidence_to_outcomes(outcomes: list[dict[str, Any]], evidence_items: list[dict[str, Any]]):
    by_campaign = {item.get('campaign_id'): item for item in outcomes}
    by_hypothesis = {item.get('hypothesis_id'): item for item in outcomes}
    by_contract = {item.get('contract_id'): item for item in outcomes}
    for item in evidence_items:
        outcome = (
            by_campaign.get(item.get('campaign_id'))
            or by_hypothesis.get(item.get('hypothesis_id'))
            or by_contract.get(item.get('contract_id'))
        )
        if not outcome:
            continue
        gate = item.get('evidence_gate')
        if item.get('label_leaks') or item.get('third_party_checkpoint_used'):
            outcome.setdefault('rejected_evidence', []).append('safety_label_project_boundary')
        elif item.get('status') == 'failed':
            outcome.setdefault('rejected_evidence', []).append(gate)
        elif item.get('status') == 'satisfied':
            outcome.setdefault('accepted_evidence', []).append(gate)
        elif item.get('status') == 'missing':
            outcome.setdefault('missing_evidence', []).append(gate)
    for outcome in outcomes:
        _finalize_outcome(outcome)


def _finalize_outcome(outcome: dict[str, Any]) -> dict[str, Any]:
    required = _unique_strings(list(outcome.get('required_evidence') or []) or list(GATE_ORDER))
    accepted = _unique_strings(list(outcome.get('accepted_evidence') or []))
    rejected = _unique_strings(list(outcome.get('rejected_evidence') or []))
    missing = [
        gate for gate in _unique_strings(list(outcome.get('missing_evidence') or []))
        if gate in set(required) and gate not in set(accepted) and gate not in set(rejected)
    ]
    computed_missing = [gate for gate in required if gate not in accepted and gate not in rejected]
    missing = _unique_strings(missing + computed_missing)
    gate_results = {}
    for gate in required:
        if gate in set(rejected):
            gate_results[gate] = 'rejected'
        elif gate in set(accepted):
            gate_results[gate] = 'accepted'
        elif gate in set(missing):
            gate_results[gate] = 'missing'
        else:
            gate_results[gate] = 'unknown'
    outcome['required_evidence'] = required
    outcome['accepted_evidence'] = accepted
    outcome['rejected_evidence'] = rejected
    outcome['missing_evidence'] = missing
    outcome['gate_results'] = gate_results
    if 'safety_label_project_boundary' in set(rejected):
        outcome['readiness_state'] = 'repair'
    elif rejected:
        outcome['readiness_state'] = 'failed'
    elif missing:
        outcome['readiness_state'] = 'waiting'
    else:
        outcome['readiness_state'] = 'accepted'
    return outcome


def _select_outcome_action(*, outcomes, evidence_items, accepted_ids, refined_ids, retired_ids, project_owned_boundary):
    safety = _first_safety_failure(outcomes, evidence_items, project_owned_boundary)
    if safety:
        return _repair_action(safety, 'safety_label_or_project_owned_repair')
    protocol = _protocol_failure(evidence_items)
    if protocol:
        return _repair_action(protocol, 'protocol_or_memory_repair')
    for outcome in outcomes:
        if outcome.get('rejected_evidence') and outcome.get('hypothesis_id') not in set(retired_ids):
            if _should_refine(outcome) and outcome.get('hypothesis_id') not in set(refined_ids):
                return _outcome_action(outcome, 'refine_hypothesis', 'broadcast', 'failed gate can be refined once')
            return _outcome_action(outcome, 'retire_theory_line', 'broadcast', 'failed gate retires this theory line')
    for gate, (action, recipient) in REQUEST_ACTIONS.items():
        for outcome in outcomes:
            if gate in set(outcome.get('missing_evidence') or []):
                return _gate_action(outcome, action, recipient, gate)
    for outcome in outcomes:
        if outcome.get('readiness_state') == 'accepted' and outcome.get('campaign_id') not in set(accepted_ids):
            return _outcome_action(outcome, 'accept_campaign', 'broadcast', 'all acceptance criteria are met')
    for outcome in outcomes:
        if _should_refine(outcome) and outcome.get('hypothesis_id') not in set(refined_ids):
            return _outcome_action(outcome, 'refine_hypothesis', 'broadcast', 'accepted campaign can spawn one refinement')
    return _noop_action('no campaign outcome action selected')


def _outcome_action(outcome, action, recipient, reason):
    return {
        'selected_action': action,
        'chosen_recipient': validate_participant(recipient),
        'campaign_id': outcome.get('campaign_id'),
        'hypothesis_id': outcome.get('hypothesis_id'),
        'acceptance_criteria': list(outcome.get('acceptance_criteria') or []),
        'accepted_evidence': list(outcome.get('accepted_evidence') or []),
        'missing_evidence': list(outcome.get('missing_evidence') or []),
        'rejected_evidence': list(outcome.get('rejected_evidence') or []),
        'gate_verdicts': dict(outcome.get('gate_results') or {}),
        'recommended_action': action,
        'priority_reason': reason,
        'updated_hypothesis_notes': _theory_notes(outcome, action, reason),
        'label_leaks': label_leak_terms(outcome),
    }


def _gate_action(outcome, action, recipient, gate):
    selected = _outcome_action(outcome, action, recipient, f'{gate} remains missing')
    selected['missing_evidence'] = [gate]
    selected['recommended_action'] = f'provide_{gate}'
    return selected


def _repair_action(blocker, reason):
    return {
        'selected_action': 'repair_boundary',
        'chosen_recipient': validate_participant('code_module' if reason == 'safety_label_or_project_owned_repair' else 'language_model_2'),
        'campaign_id': blocker.get('campaign_id'),
        'hypothesis_id': blocker.get('hypothesis_id'),
        'acceptance_criteria': ['repair boundary/protocol evidence before campaign outcome assessment continues'],
        'accepted_evidence': [],
        'missing_evidence': [blocker.get('evidence_gate')] if blocker.get('evidence_gate') else [],
        'rejected_evidence': [blocker.get('evidence_id') or blocker.get('evidence_gate') or reason],
        'gate_verdicts': {blocker.get('evidence_gate') or 'boundary': 'rejected'},
        'recommended_action': reason,
        'priority_reason': reason,
        'updated_hypothesis_notes': ['campaign outcome assessment paused for repair'],
        'label_leaks': label_leak_terms(blocker),
    }


def _noop_action(reason):
    return {
        'selected_action': 'noop',
        'chosen_recipient': None,
        'campaign_id': None,
        'hypothesis_id': None,
        'acceptance_criteria': [],
        'accepted_evidence': [],
        'missing_evidence': [],
        'rejected_evidence': [],
        'gate_verdicts': {},
        'recommended_action': 'noop',
        'priority_reason': reason,
        'updated_hypothesis_notes': [],
        'label_leaks': [],
    }


def _should_refine(outcome):
    return (
        outcome.get('campaign_type') in {'continue_refinement', 'refine_hypothesis'}
        or outcome.get('readiness_state') == 'failed' and len(outcome.get('accepted_evidence') or []) >= 2
    )


def _first_safety_failure(outcomes, evidence_items, project_owned_boundary):
    if project_owned_boundary.get('third_party_checkpoint_used'):
        return {'reason': 'third-party checkpoint boundary failed', 'evidence_gate': 'project_owned_checkpoint_boundary_explicit', 'status': 'failed'}
    for item in evidence_items:
        if item.get('label_leaks') or item.get('third_party_checkpoint_used'):
            return item
    for outcome in outcomes:
        if 'safety_label_project_boundary' in set(outcome.get('rejected_evidence') or []):
            return {'campaign_id': outcome.get('campaign_id'), 'hypothesis_id': outcome.get('hypothesis_id'), 'evidence_gate': 'safety_label_project_boundary', 'status': 'failed'}
    return None


def _protocol_failure(evidence_items):
    for item in evidence_items:
        if item.get('sender') == 'language_model_2' and item.get('status') in {'missing', 'failed'}:
            return item
    return None


def _evidence_gate(message):
    body = dict(message.get('body') or {})
    evidence = dict(message.get('evidence') or {})
    explicit = body.get('evidence_gate') or evidence.get('evidence_gate') or body.get('gate') or evidence.get('gate')
    if explicit:
        return str(explicit)
    sender = message.get('sender')
    if sender == 'funfun':
        return 'math_proof'
    if sender == 'code_module':
        return 'code_proof'
    if sender == 'language_model_2':
        return 'language_epoch_plan'
    return 'advisory'


def _evidence_status(message):
    body = dict(message.get('body') or {})
    evidence = dict(message.get('evidence') or {})
    explicit = str(body.get('status') or evidence.get('status') or '').lower()
    text = json.dumps({'body': body, 'evidence': evidence, 'tags': message.get('tags')}, sort_keys=True).lower()
    if explicit in {'satisfied', 'resolved', 'passed', 'confirmed', 'accepted'}:
        return 'satisfied'
    if explicit in {'failed', 'blocked', 'contradicted', 'contradiction', 'rejected'}:
        return 'failed'
    if explicit in {'missing', 'needs_clarification', 'clarification_needed'} or 'missing' in text:
        return 'missing'
    if any(token in text for token in ('failed', 'blocked', 'contradict', 'rejected')):
        return 'failed'
    if any(token in text for token in ('satisfied', 'resolved', 'passed', 'confirmed', 'accepted')):
        return 'satisfied'
    return 'advisory'


def _required_gates(item):
    gates = set(item.get('required_evidence_gates') or item.get('required_evidence') or [])
    gates.update(GATE_ORDER)
    return [gate for gate in GATE_ORDER if gate in gates]


def _criteria_from_item(item):
    criteria = item.get('acceptance_criteria') or []
    if isinstance(criteria, list):
        return list(criteria)
    return []


def _theory_notes(outcome, action, reason):
    return [
        f'{action} for {outcome.get("hypothesis_id") or "unlinked_hypothesis"}',
        reason,
    ]


def _source_hash(evaluator, prior_outcome, contracts, adjudicator, agenda, lifecycle, scorecard, campaign):
    return stable_digest({
        'evaluator': evaluator.get('ledger_hash') or evaluator.get('ledger_id'),
        'prior_outcome': prior_outcome.get('ledger_hash') or prior_outcome.get('ledger_id'),
        'contract': contracts.get('ledger_hash'),
        'adjudicator': adjudicator.get('ledger_hash'),
        'agenda': agenda.get('ledger_hash'),
        'lifecycle': lifecycle.get('ledger_hash'),
        'scorecard': scorecard.get('ledger_hash'),
        'campaign': campaign.get('ledger_hash'),
    })


def _find_outcome(outcomes, campaign_id, hypothesis_id):
    for outcome in outcomes:
        if campaign_id and outcome.get('campaign_id') == campaign_id:
            return outcome
        if hypothesis_id and outcome.get('hypothesis_id') == hypothesis_id:
            return outcome
    return None


def _outcome_status_counts(outcomes, selected):
    counts = {
        'accepted': 0,
        'request': 0,
        'failed': 0,
        'refined': 0,
        'retired': 0,
        'repair': 0,
        'noop': 0,
    }
    for outcome in outcomes:
        state = outcome.get('readiness_state')
        if state == 'accepted':
            counts['accepted'] += 1
        elif state == 'waiting':
            counts['request'] += 1
        elif state == 'failed':
            counts['failed'] += 1
        elif state == 'repair':
            counts['repair'] += 1
    action = selected.get('selected_action')
    if action == 'refine_hypothesis':
        counts['refined'] += 1
    elif action == 'retire_theory_line':
        counts['retired'] += 1
    elif action == 'noop':
        counts['noop'] += 1
    return counts


def _field_map(outcomes, field):
    return {str(item.get('campaign_id') or item.get('hypothesis_id')): list(item.get(field) or []) for item in outcomes if item.get(field)}


def _collect_field(outcomes, field):
    values = []
    for outcome in outcomes:
        if outcome.get(field):
            values.append(outcome.get(field))
    return values


def _collect_list_field(outcomes, field):
    values = []
    for outcome in outcomes:
        values.extend(list(outcome.get(field) or []))
    return values


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
