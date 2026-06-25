"""Experiment campaign planner and acceptance-criteria bundler.

This layer turns scored hypothesis evidence into the next small symbolic
science campaign. It never runs heavy experiments or claims checkpoint
ownership; it only emits plain module-chat planning data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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


CAMPAIGN_LEDGER_KIND = 'ai_different.experiment_campaign_ledger'
GATE_ORDER = ('math_proof', 'code_proof', 'language_epoch_plan')


def empty_campaign_ledger() -> dict[str, Any]:
    return {
        'schema_version': 1,
        'ledger_kind': CAMPAIGN_LEDGER_KIND,
        'processed_message_ids': [],
        'processed_source_hashes': [],
        'retired_hypothesis_ids': [],
        'continued_refinement_ids': [],
        'acceptance_bundle_ids': [],
        'campaigns': [],
        'campaign_records': [],
        'outgoing_response_ids': [],
        'latest': {},
    }


def load_campaign_ledger(path: str | Path | None) -> dict[str, Any]:
    if not path or not Path(path).exists():
        return empty_campaign_ledger()
    with Path(path).open('r', encoding='utf-8') as handle:
        ledger = json.load(handle)
    return validate_campaign_ledger(ledger)


def write_campaign_ledger(path: str | Path, ledger: dict[str, Any]) -> Path:
    validated = validate_campaign_ledger(ledger)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(validated, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    return output


def validate_campaign_ledger(ledger: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(ledger, dict):
        raise ValueError('campaign ledger must be a JSON object')
    if ledger.get('ledger_kind') != CAMPAIGN_LEDGER_KIND:
        raise ValueError('campaign ledger has wrong ledger_kind')
    for field in (
        'processed_message_ids',
        'processed_source_hashes',
        'retired_hypothesis_ids',
        'continued_refinement_ids',
        'acceptance_bundle_ids',
        'outgoing_response_ids',
    ):
        values = ledger.get(field)
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            raise ValueError(f'{field} must be strings')
    for field in ('campaigns', 'campaign_records'):
        values = ledger.get(field)
        if not isinstance(values, list) or not all(isinstance(item, dict) for item in values):
            raise ValueError(f'{field} must be objects')
    latest = ledger.get('latest', {})
    if not isinstance(latest, dict):
        raise ValueError('campaign latest must be an object')
    validated = {
        'schema_version': int(ledger.get('schema_version', 1) or 1),
        'ledger_kind': CAMPAIGN_LEDGER_KIND,
        'processed_message_ids': _unique_strings(ledger['processed_message_ids']),
        'processed_source_hashes': _unique_strings(ledger['processed_source_hashes']),
        'retired_hypothesis_ids': _unique_strings(ledger['retired_hypothesis_ids']),
        'continued_refinement_ids': _unique_strings(ledger['continued_refinement_ids']),
        'acceptance_bundle_ids': _unique_strings(ledger['acceptance_bundle_ids']),
        'campaigns': list(ledger['campaigns']),
        'campaign_records': list(ledger['campaign_records']),
        'outgoing_response_ids': _unique_strings(ledger['outgoing_response_ids']),
        'latest': dict(latest),
    }
    if ledger.get('ledger_hash'):
        validated['ledger_hash'] = str(ledger['ledger_hash'])
    if ledger.get('artifact_path'):
        validated['artifact_path'] = str(ledger['artifact_path'])
    return validated


def read_campaign_transcript(path: str | Path | None) -> dict[str, Any]:
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
        raise ValueError('plain campaign input must be a JSON object')
    return value


def build_experiment_campaign(
    *,
    transcript_messages: list[dict[str, Any]],
    campaign_ledger: dict[str, Any],
    evaluator_ledger: dict[str, Any] | None = None,
    outcome_ledger: dict[str, Any] | None = None,
    contract_ledger: dict[str, Any] | None = None,
    adjudicator_ledger: dict[str, Any] | None = None,
    agenda_ledger: dict[str, Any] | None = None,
    lifecycle_ledger: dict[str, Any] | None = None,
    scorecard_ledger: dict[str, Any] | None = None,
    runtime_memory_data: dict[str, Any] | None = None,
    runtime_memory_hash_state: dict[str, Any],
    project_owned_boundary: dict[str, Any],
    artifact_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    ledger = validate_campaign_ledger(campaign_ledger)
    evaluator = _valid_evaluator_or_empty(evaluator_ledger or {})
    outcome = _valid_outcome_or_empty(outcome_ledger or {})
    contracts = _valid_contract_or_empty(contract_ledger or {})
    adjudicator = _valid_adjudicator_or_empty(adjudicator_ledger or {})
    agenda = _valid_agenda_or_empty(agenda_ledger or {})
    lifecycle = _valid_lifecycle_or_empty(lifecycle_ledger or {})
    scorecard = _valid_scorecard_or_empty(scorecard_ledger or {})
    source_hash = _source_hash(evaluator, outcome, contracts, adjudicator, agenda, lifecycle, scorecard)
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
    campaigns = _merge_campaigns(
        ledger['campaigns'],
        contracts,
        agenda,
        lifecycle,
        scorecard,
        transcript_messages,
    )
    evidence_items = _extract_campaign_evidence(new_messages, campaigns)
    _apply_evidence_to_campaigns(campaigns, evidence_items)
    refinement = _candidate_refinement(evaluator or outcome, scorecard, campaigns, runtime_memory_data or {}, source_hash)
    if not new_messages and (not source_is_new or campaigns):
        selected = _noop_action('no new campaign evidence or source ledger state')
    else:
        selected = _select_campaign_action(
            campaigns=campaigns,
            evidence_items=evidence_items,
            refinement=refinement,
            retired_ids=ledger['retired_hypothesis_ids'],
            continued_ids=ledger['continued_refinement_ids'],
            acceptance_ids=ledger['acceptance_bundle_ids'],
            project_owned_boundary=project_owned_boundary,
        )
    campaign_id = 'campaign_' + stable_digest({
        'source_hash': source_hash,
        'new_message_ids': [module_chat_message_id(message) for message in new_messages],
        'selected_action': selected['selected_action'],
        'hypothesis_id': selected.get('hypothesis_id'),
    })[:16]
    selected['campaign_id'] = campaign_id
    message = export_campaign_message(
        selected,
        campaigns=campaigns,
        evidence_items=evidence_items,
        refinement=refinement,
        runtime_memory_hash_state=runtime_memory_hash_state,
        project_owned_boundary=project_owned_boundary,
        source_hashes={
            'campaign_source_hash': source_hash,
            'evaluator_ledger_hash': evaluator.get('ledger_hash'),
            'outcome_ledger_hash': outcome.get('ledger_hash'),
            'contract_ledger_hash': contracts.get('ledger_hash'),
            'adjudicator_ledger_hash': adjudicator.get('ledger_hash'),
            'agenda_ledger_hash': agenda.get('ledger_hash'),
            'lifecycle_ledger_hash': lifecycle.get('ledger_hash'),
            'scorecard_ledger_hash': scorecard.get('ledger_hash'),
        },
    )
    new_ids = [module_chat_message_id(message) for message in new_messages]
    outgoing_id = module_chat_message_id(message) if message else None
    hypothesis_id = selected.get('hypothesis_id')
    if selected['selected_action'] == 'retire_stale_or_blocked_line' and hypothesis_id:
        ledger['retired_hypothesis_ids'] = _unique_strings(list(ledger['retired_hypothesis_ids']) + [hypothesis_id])
    if selected['selected_action'] == 'continue_refinement' and hypothesis_id:
        ledger['continued_refinement_ids'] = _unique_strings(list(ledger['continued_refinement_ids']) + [hypothesis_id])
    if selected['selected_action'] == 'emit_acceptance_bundle' and hypothesis_id:
        ledger['acceptance_bundle_ids'] = _unique_strings(list(ledger['acceptance_bundle_ids']) + [hypothesis_id])
    if new_messages:
        ledger['processed_message_ids'] = _unique_strings(list(ledger['processed_message_ids']) + new_ids)
    if source_is_new and selected['selected_action'] in {'summarize_noop', 'continue_refinement', 'emit_acceptance_bundle'}:
        ledger['processed_source_hashes'] = _unique_strings(list(ledger['processed_source_hashes']) + [source_hash])
    if outgoing_id:
        ledger['outgoing_response_ids'] = _unique_strings(list(ledger['outgoing_response_ids']) + [outgoing_id])
    latest = {
        'campaign_id': campaign_id,
        'new_message_count': len(new_messages),
        'skipped_message_count': len(skipped_messages),
        'source_is_new': source_is_new,
        'campaign_type_counts': _campaign_type_counts(campaigns),
        'selected_action': selected['selected_action'],
        'chosen_recipient': message.get('recipient') if message else None,
        'outbox_count': 1 if message else 0,
        'label_leaks': selected.get('label_leaks', []),
        'runtime_memory_hash_state': dict(runtime_memory_hash_state),
        'runtime_memory_mutated': not bool(runtime_memory_hash_state.get('unchanged', True)),
    }
    record = {
        'campaign_id': campaign_id,
        'campaign_hash': stable_digest({'campaign_id': campaign_id, 'selected': selected, 'refinement': refinement}),
        'processed_message_ids': new_ids,
        'source_hash': source_hash,
        'hypothesis_ids': [item.get('hypothesis_id') for item in campaigns],
        'source_scorecard_ids': _unique_strings(_collect_field(campaigns, 'source_scorecard_ids')),
        'source_lifecycle_ids': _unique_strings(_collect_field(campaigns, 'source_lifecycle_ids')),
        'source_agenda_ids': _unique_strings(_collect_field(campaigns, 'source_agenda_ids')),
        'source_contract_ids': _unique_strings(_collect_field(campaigns, 'source_contract_ids')),
        'accepted_evidence': _field_map(campaigns, 'accepted_evidence'),
        'missing_evidence': _field_map(campaigns, 'missing_evidence'),
        'rejected_evidence': _field_map(campaigns, 'rejected_evidence'),
        'campaign_types': {item.get('hypothesis_id'): item.get('campaign_type') for item in campaigns},
        'acceptance_criteria': {item.get('hypothesis_id'): item.get('acceptance_criteria') for item in campaigns},
        'selected_action': selected['selected_action'],
        'chosen_recipient': latest['chosen_recipient'],
        'outgoing_response_id': outgoing_id,
        'runtime_memory_hash_state': dict(runtime_memory_hash_state),
        'source_ledger_hashes': {
            'evaluator_ledger_hash': evaluator.get('ledger_hash'),
            'outcome_ledger_hash': outcome.get('ledger_hash'),
            'contract_ledger_hash': contracts.get('ledger_hash'),
            'adjudicator_ledger_hash': adjudicator.get('ledger_hash'),
            'agenda_ledger_hash': agenda.get('ledger_hash'),
            'lifecycle_ledger_hash': lifecycle.get('ledger_hash'),
            'scorecard_ledger_hash': scorecard.get('ledger_hash'),
        },
    }
    ledger['campaigns'] = campaigns
    if new_messages or source_is_new or message is not None:
        ledger['campaign_records'] = list(ledger['campaign_records']) + [record]
    ledger['latest'] = latest
    if artifact_path is not None:
        ledger['artifact_path'] = str(Path(artifact_path))
    ledger['ledger_hash'] = stable_digest({key: value for key, value in ledger.items() if key != 'ledger_hash'})
    return ledger, message


def export_campaign_message(
    selected: dict[str, Any],
    *,
    campaigns: list[dict[str, Any]],
    evidence_items: list[dict[str, Any]],
    refinement: dict[str, Any] | None,
    runtime_memory_hash_state: dict[str, Any],
    project_owned_boundary: dict[str, Any],
    source_hashes: dict[str, Any],
    topic: str = 'ai_different.experiment_campaign',
) -> dict[str, Any] | None:
    if selected['selected_action'] == 'summarize_noop':
        return None
    recipient = validate_participant(selected.get('chosen_recipient') or 'orchestrator')
    campaign = _find_campaign(campaigns, selected.get('hypothesis_id')) or {}
    leak_terms = label_leak_terms({'selected': selected, 'campaign': campaign, 'refinement': refinement, 'evidence': evidence_items})
    body = {
        'module': 'AI Different',
        'response_kind': 'experiment_campaign',
        'campaign_id': selected.get('campaign_id'),
        'hypothesis_id': selected.get('hypothesis_id'),
        'selected_action': selected['selected_action'],
        'campaign_type': selected.get('campaign_type') or campaign.get('campaign_type'),
        'acceptance_criteria': selected.get('acceptance_criteria') or campaign.get('acceptance_criteria'),
        'required_evidence': selected.get('required_evidence') or campaign.get('required_evidence'),
        'accepted_evidence': selected.get('accepted_evidence') or campaign.get('accepted_evidence'),
        'missing_evidence': selected.get('missing_evidence') or campaign.get('missing_evidence'),
        'rejected_evidence': selected.get('rejected_evidence') or campaign.get('rejected_evidence'),
        'recommended_recipient': recipient,
        'recommended_action': selected.get('recommended_action') or selected['selected_action'],
        'risk_unknowns': selected.get('risk_unknowns') or campaign.get('risk_unknowns'),
        'reason': selected.get('reason'),
        'refinement_lineage': refinement if selected['selected_action'] == 'continue_refinement' else None,
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
            'campaign_id': body['campaign_id'],
            'hypothesis_id': body['hypothesis_id'],
            'selected_action': body['selected_action'],
            'campaign_type': body['campaign_type'],
            'acceptance_criteria': body['acceptance_criteria'],
            'runtime_memory_hash_state': body['runtime_memory_hash_state'],
            'runtime_memory_mutated': body['runtime_memory_mutated'],
            'label_clean': body['label_clean'],
            'no_sibling_imports': True,
        },
        tags=['ai_different', 'experiment_campaign', body['selected_action'], 'label_clean' if body['label_clean'] else 'label_review_needed'],
    )


def write_campaign_outbox_jsonl(path: str | Path, message: dict[str, Any] | None) -> Path:
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


def _valid_outcome_or_empty(ledger: dict[str, Any]) -> dict[str, Any]:
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


def _merge_campaigns(
    existing: list[dict[str, Any]],
    contract_ledger: dict[str, Any],
    agenda_ledger: dict[str, Any],
    lifecycle_ledger: dict[str, Any],
    scorecard_ledger: dict[str, Any],
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    campaigns: dict[str, dict[str, Any]] = {
        str(item.get('hypothesis_id') or item.get('contract_id')): dict(item)
        for item in existing
        if item.get('hypothesis_id') or item.get('contract_id')
    }
    for contract in list(contract_ledger.get('contracts') or []):
        _upsert_campaign(campaigns, _campaign_from_contract(contract, 'contract_ledger'))
    for hypothesis in list(agenda_ledger.get('hypotheses') or []):
        _upsert_campaign(campaigns, _campaign_from_contract(hypothesis, 'agenda_ledger'))
    for hypothesis in list(lifecycle_ledger.get('hypotheses') or []):
        _upsert_campaign(campaigns, _campaign_from_contract(hypothesis, 'lifecycle_ledger'))
    for card in list(scorecard_ledger.get('scorecards') or []):
        _upsert_campaign(campaigns, _campaign_from_scorecard(card, scorecard_ledger))
    for message in messages:
        body = dict(message.get('body') or {})
        if message.get('sender') != 'ai_different':
            continue
        kind = body.get('response_kind')
        if kind == 'experiment_contract':
            _upsert_campaign(campaigns, _campaign_from_contract({
                'contract_id': body.get('contract_id'),
                'required_evidence_gates': body.get('required_evidence_gates') or [],
            }, 'transcript_contract'))
        elif kind == 'experiment_evidence_scorecard':
            _upsert_campaign(campaigns, _campaign_from_scorecard_message(body))
        elif kind == 'experiment_campaign':
            _upsert_campaign(campaigns, _campaign_from_prior_campaign(body))
    return [_finalize_campaign(item) for item in campaigns.values()]


def _campaign_from_contract(item: dict[str, Any], source: str) -> dict[str, Any]:
    contract_id = item.get('contract_id')
    return {
        'hypothesis_id': item.get('hypothesis_id') or (f'hypothesis:{contract_id}' if contract_id else None),
        'contract_id': contract_id,
        'source_contract_ids': [contract_id] if contract_id else [],
        'source_agenda_ids': list(item.get('source_agenda_ids') or []),
        'source_lifecycle_ids': list(item.get('source_lifecycle_ids') or []),
        'source_scorecard_ids': list(item.get('source_scorecard_ids') or []),
        'required_evidence': _required_gates(item),
        'accepted_evidence': list(item.get('accepted_evidence_gates') or item.get('satisfied_evidence_gates') or []),
        'missing_evidence': list(item.get('missing_evidence_gates') or []),
        'rejected_evidence': list(item.get('rejected_evidence_gates') or item.get('failed_evidence_gates') or []),
        'readiness_state': item.get('readiness_state') or item.get('lifecycle_state') or 'waiting',
        'lineage': [source],
    }


def _campaign_from_scorecard(card: dict[str, Any], ledger: dict[str, Any]) -> dict[str, Any]:
    item = _campaign_from_contract(card, 'scorecard_ledger')
    latest_id = dict(ledger.get('latest') or {}).get('scorecard_id')
    item['source_scorecard_ids'] = [latest_id] if latest_id else []
    item['accepted_evidence'] = list(card.get('accepted_evidence_gates') or [])
    item['missing_evidence'] = list(card.get('missing_evidence_gates') or [])
    item['rejected_evidence'] = list(card.get('rejected_evidence_gates') or [])
    item['readiness_state'] = card.get('readiness_state') or 'waiting'
    return item


def _campaign_from_scorecard_message(body: dict[str, Any]) -> dict[str, Any]:
    hypothesis_id = body.get('hypothesis_id')
    return {
        'hypothesis_id': hypothesis_id,
        'contract_id': str(hypothesis_id or '').replace('hypothesis:', ''),
        'source_contract_ids': [],
        'source_agenda_ids': [],
        'source_lifecycle_ids': [],
        'source_scorecard_ids': [body.get('scorecard_id')] if body.get('scorecard_id') else [],
        'required_evidence': _required_gates(body),
        'accepted_evidence': list(body.get('accepted_evidence') or []),
        'missing_evidence': list(body.get('missing_evidence') or []),
        'rejected_evidence': list(body.get('rejected_evidence') or []),
        'readiness_state': body.get('readiness_state') or 'waiting',
        'lineage': ['scorecard_message'],
    }


def _campaign_from_prior_campaign(body: dict[str, Any]) -> dict[str, Any]:
    item = _campaign_from_scorecard_message(body)
    item['source_campaign_ids'] = [body.get('campaign_id')] if body.get('campaign_id') else []
    item['campaign_type'] = body.get('campaign_type')
    return item


def _upsert_campaign(campaigns: dict[str, dict[str, Any]], incoming: dict[str, Any]):
    hypothesis_id = incoming.get('hypothesis_id')
    if not hypothesis_id:
        return
    current = campaigns.setdefault(str(hypothesis_id), {})
    current.update({
        'hypothesis_id': str(hypothesis_id),
        'contract_id': incoming.get('contract_id') or current.get('contract_id'),
        'source_contract_ids': _unique_strings(list(current.get('source_contract_ids') or []) + list(incoming.get('source_contract_ids') or [])),
        'source_agenda_ids': _unique_strings(list(current.get('source_agenda_ids') or []) + list(incoming.get('source_agenda_ids') or [])),
        'source_lifecycle_ids': _unique_strings(list(current.get('source_lifecycle_ids') or []) + list(incoming.get('source_lifecycle_ids') or [])),
        'source_scorecard_ids': _unique_strings(list(current.get('source_scorecard_ids') or []) + list(incoming.get('source_scorecard_ids') or [])),
        'required_evidence': _unique_strings(list(current.get('required_evidence') or []) + list(incoming.get('required_evidence') or [])),
        'accepted_evidence': _unique_strings(list(current.get('accepted_evidence') or []) + list(incoming.get('accepted_evidence') or [])),
        'missing_evidence': _unique_strings(list(current.get('missing_evidence') or []) + list(incoming.get('missing_evidence') or [])),
        'rejected_evidence': _unique_strings(list(current.get('rejected_evidence') or []) + list(incoming.get('rejected_evidence') or [])),
        'readiness_state': _dominant_state(current.get('readiness_state'), incoming.get('readiness_state')),
        'lineage': _unique_strings(list(current.get('lineage') or []) + list(incoming.get('lineage') or [])),
    })


def _extract_campaign_evidence(messages: list[dict[str, Any]], campaigns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    known_contract_ids = {item.get('contract_id') for item in campaigns}
    items = []
    for message in messages:
        body = dict(message.get('body') or {})
        evidence = dict(message.get('evidence') or {})
        sender = message.get('sender')
        contract_id = body.get('contract_id') or evidence.get('contract_id')
        hypothesis_id = body.get('hypothesis_id') or evidence.get('hypothesis_id')
        if not hypothesis_id and contract_id:
            hypothesis_id = f'hypothesis:{contract_id}'
        item = {
            'evidence_id': str(body.get('evidence_id') or evidence.get('evidence_id') or module_chat_message_id(message)),
            'message_id': module_chat_message_id(message),
            'sender': sender,
            'topic': message.get('topic'),
            'hypothesis_id': str(hypothesis_id) if hypothesis_id else None,
            'contract_id': str(contract_id) if contract_id else None,
            'evidence_gate': _evidence_gate(message),
            'status': _evidence_status(message),
            'summary': body.get('summary') or body.get('note') or body.get('reason'),
            'label_leaks': label_leak_terms({'body': body, 'evidence': evidence}),
            'third_party_checkpoint_used': bool(body.get('third_party_checkpoint_used') or evidence.get('third_party_checkpoint_used')),
        }
        if item['contract_id'] in known_contract_ids or item['hypothesis_id'] or sender in {'language_model_2', 'funfun', 'code_module'}:
            items.append(item)
    return items


def _apply_evidence_to_campaigns(campaigns: list[dict[str, Any]], evidence_items: list[dict[str, Any]]):
    by_hypothesis = {item.get('hypothesis_id'): item for item in campaigns}
    by_contract = {item.get('contract_id'): item for item in campaigns}
    for item in evidence_items:
        campaign = by_hypothesis.get(item.get('hypothesis_id')) or by_contract.get(item.get('contract_id'))
        if not campaign:
            continue
        gate = item.get('evidence_gate')
        if item.get('label_leaks') or item.get('third_party_checkpoint_used') or item.get('status') == 'failed':
            campaign.setdefault('rejected_evidence', []).append('safety_label_project_boundary' if item.get('label_leaks') or item.get('third_party_checkpoint_used') else gate)
        elif item.get('status') == 'satisfied':
            campaign.setdefault('accepted_evidence', []).append(gate)
        elif item.get('status') == 'missing':
            campaign.setdefault('missing_evidence', []).append(gate)
        elif item.get('status') in {'retired', 'blocked'}:
            campaign['readiness_state'] = 'retired'
    for campaign in campaigns:
        _finalize_campaign(campaign)


def _finalize_campaign(campaign: dict[str, Any]) -> dict[str, Any]:
    required = _unique_strings(list(campaign.get('required_evidence') or []) or list(GATE_ORDER))
    accepted = _unique_strings(list(campaign.get('accepted_evidence') or []))
    rejected = _unique_strings(list(campaign.get('rejected_evidence') or []))
    missing = [
        gate for gate in _unique_strings(list(campaign.get('missing_evidence') or []))
        if gate not in set(accepted) and gate not in set(rejected)
    ]
    computed_missing = [gate for gate in required if gate not in accepted and gate not in rejected]
    missing = _unique_strings(missing + computed_missing)
    campaign['required_evidence'] = required
    campaign['accepted_evidence'] = accepted
    campaign['rejected_evidence'] = rejected
    campaign['missing_evidence'] = missing
    if rejected:
        campaign['campaign_type'] = 'retire_line'
    elif 'math_proof' in missing:
        campaign['campaign_type'] = 'request_math_gate'
    elif 'code_proof' in missing:
        campaign['campaign_type'] = 'request_code_gate'
    elif 'language_epoch_plan' in missing:
        campaign['campaign_type'] = 'request_language_gate'
    elif campaign.get('readiness_state') in {'refine', 'refine_next'}:
        campaign['campaign_type'] = 'continue_refinement'
    else:
        campaign['campaign_type'] = 'emit_acceptance_bundle'
    campaign['acceptance_criteria'] = _acceptance_criteria(campaign)
    campaign['risk_unknowns'] = _risk_unknowns(campaign)
    campaign['campaign_hash'] = stable_digest({
        'hypothesis_id': campaign.get('hypothesis_id'),
        'type': campaign.get('campaign_type'),
        'accepted': accepted,
        'missing': missing,
        'rejected': rejected,
    })
    return campaign


def _select_campaign_action(
    *,
    campaigns: list[dict[str, Any]],
    evidence_items: list[dict[str, Any]],
    refinement: dict[str, Any] | None,
    retired_ids: list[str],
    continued_ids: list[str],
    acceptance_ids: list[str],
    project_owned_boundary: dict[str, Any],
) -> dict[str, Any]:
    safety = _first_safety_failure(campaigns, evidence_items, project_owned_boundary)
    if safety:
        return _repair_action('safety_label_or_project_owned_repair', 'code_module', safety, 'repair_boundary')
    protocol = _protocol_failure(evidence_items)
    if protocol:
        return _repair_action('protocol_or_readiness_repair', 'language_model_2', protocol, 'repair_boundary')
    for campaign in campaigns:
        if campaign.get('campaign_type') == 'retire_line' and campaign.get('hypothesis_id') not in set(retired_ids):
            return _campaign_action(campaign, 'retire_stale_or_blocked_line', 'broadcast')
    for campaign in campaigns:
        if campaign.get('campaign_type') == 'request_math_gate':
            return _gate_action(campaign, 'request_math_gate', 'funfun', 'math_proof')
    for campaign in campaigns:
        if campaign.get('campaign_type') == 'request_code_gate':
            return _gate_action(campaign, 'request_code_gate', 'code_module', 'code_proof')
    for campaign in campaigns:
        if campaign.get('campaign_type') == 'request_language_gate':
            return _gate_action(campaign, 'request_language_gate', 'language_model_2', 'language_epoch_plan')
    for campaign in campaigns:
        if (
            campaign.get('hypothesis_id') in set(acceptance_ids)
            and campaign.get('hypothesis_id') not in set(continued_ids)
            and refinement
        ):
            return _campaign_action(campaign, 'continue_refinement', 'broadcast', refinement=refinement)
    for campaign in campaigns:
        if campaign.get('campaign_type') == 'emit_acceptance_bundle' and campaign.get('hypothesis_id') not in set(acceptance_ids):
            return _campaign_action(campaign, 'emit_acceptance_bundle', 'broadcast')
    return _noop_action('no campaign action selected')


def _campaign_action(campaign: dict[str, Any], action: str, recipient: str, *, refinement: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        'selected_action': action,
        'chosen_recipient': recipient,
        'hypothesis_id': campaign.get('hypothesis_id'),
        'campaign_type': campaign.get('campaign_type') if action != 'continue_refinement' else 'continue_refinement',
        'acceptance_criteria': campaign.get('acceptance_criteria'),
        'required_evidence': list(campaign.get('required_evidence') or []),
        'accepted_evidence': list(campaign.get('accepted_evidence') or []),
        'missing_evidence': list(campaign.get('missing_evidence') or []),
        'rejected_evidence': list(campaign.get('rejected_evidence') or []),
        'recommended_action': action,
        'risk_unknowns': campaign.get('risk_unknowns'),
        'reason': _reason_for_action(action),
        'refinement': refinement,
        'label_leaks': label_leak_terms({'campaign': campaign, 'refinement': refinement}),
    }


def _gate_action(campaign: dict[str, Any], action: str, recipient: str, gate: str) -> dict[str, Any]:
    selected = _campaign_action(campaign, action, recipient)
    selected['missing_evidence'] = [gate]
    selected['recommended_action'] = f'provide_{gate}'
    return selected


def _repair_action(action: str, recipient: str, blocker: dict[str, Any], campaign_type: str) -> dict[str, Any]:
    return {
        'selected_action': action,
        'chosen_recipient': validate_participant(recipient),
        'hypothesis_id': blocker.get('hypothesis_id'),
        'campaign_type': campaign_type,
        'acceptance_criteria': ['repair boundary/protocol evidence before campaign planning continues'],
        'required_evidence': [blocker.get('evidence_gate')] if blocker.get('evidence_gate') else [],
        'accepted_evidence': [],
        'missing_evidence': [blocker.get('evidence_gate')] if blocker.get('evidence_gate') else [],
        'rejected_evidence': [blocker.get('evidence_id')] if blocker.get('evidence_id') else [],
        'recommended_action': action,
        'risk_unknowns': ['boundary/protocol evidence is not safe to plan through'],
        'reason': 'boundary or protocol repair outranks campaign selection',
        'label_leaks': label_leak_terms(blocker),
    }


def _noop_action(reason: str) -> dict[str, Any]:
    return {
        'selected_action': 'summarize_noop',
        'chosen_recipient': None,
        'hypothesis_id': None,
        'campaign_type': 'noop',
        'acceptance_criteria': [],
        'required_evidence': [],
        'accepted_evidence': [],
        'missing_evidence': [],
        'rejected_evidence': [],
        'recommended_action': 'noop',
        'risk_unknowns': [],
        'reason': reason,
        'label_leaks': [],
    }


def _candidate_refinement(source_ledger, scorecard_ledger, campaigns, runtime_memory_data, source_hash):
    selected = dict(source_ledger.get('selected_experiment') or {})
    if not selected:
        selected = {'world': 'hidden_procedural', 'probe': 'abstraction_transfer_probe'}
    ready = [item for item in campaigns if item.get('campaign_type') in {'continue_refinement', 'emit_acceptance_bundle'}]
    readiness = dict(runtime_memory_data.get('discovery_readiness') or {})
    if not ready or selected.get('runs_final', False):
        return None
    return {
        'refinement_id': 'campaign_refinement_' + stable_digest({
            'source_hash': source_hash,
            'ready': [item.get('hypothesis_id') for item in ready],
            'readiness_score': readiness.get('readiness_score'),
        })[:16],
        'source_hypothesis_ids': [item.get('hypothesis_id') for item in ready],
        'selected_world': selected.get('world'),
        'selected_probe': selected.get('probe'),
        'expected_transfer_signal': source_ledger.get('expected_transfer_signal') or selected.get('expected_transfer_signal'),
    }


def _first_safety_failure(campaigns, evidence_items, project_owned_boundary):
    if project_owned_boundary.get('third_party_checkpoint_used'):
        return {'sender': 'code_module', 'reason': 'third-party checkpoint boundary failed', 'evidence_gate': 'project_owned_checkpoint_boundary_explicit', 'status': 'failed'}
    for item in evidence_items:
        if item.get('label_leaks') or item.get('third_party_checkpoint_used'):
            return item
    for campaign in campaigns:
        if 'safety_label_project_boundary' in set(campaign.get('rejected_evidence') or []):
            return {'hypothesis_id': campaign.get('hypothesis_id'), 'sender': 'code_module', 'evidence_gate': 'safety_label_project_boundary', 'status': 'failed'}
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
    if explicit in {'satisfied', 'resolved', 'passed', 'confirmed'}:
        return 'satisfied'
    if explicit in {'failed', 'blocked', 'contradicted', 'contradiction'}:
        return 'failed'
    if explicit in {'missing', 'needs_clarification', 'clarification_needed'} or 'missing' in text:
        return 'missing'
    if explicit in {'retired', 'retire'}:
        return 'retired'
    if any(token in text for token in ('failed', 'blocked', 'contradict')):
        return 'failed'
    if any(token in text for token in ('satisfied', 'resolved', 'passed', 'confirmed')):
        return 'satisfied'
    return 'advisory'


def _required_gates(item):
    gates = set(item.get('required_evidence_gates') or [])
    gates.update(GATE_ORDER)
    return sorted(gate for gate in gates if gate in set(GATE_ORDER))


def _acceptance_criteria(campaign):
    criteria = []
    for gate in GATE_ORDER:
        status = 'accepted' if gate in set(campaign.get('accepted_evidence') or []) else 'missing'
        if gate in set(campaign.get('rejected_evidence') or []):
            status = 'rejected'
        criteria.append({'evidence_gate': gate, 'required_status': 'accepted', 'current_status': status})
    criteria.append({'evidence_gate': 'runtime_memory_not_mutated', 'required_status': 'preserved', 'current_status': 'checked_by_runner'})
    criteria.append({'evidence_gate': 'project_owned_checkpoint_boundary_explicit', 'required_status': 'no_overclaim', 'current_status': 'checked_by_runner'})
    return criteria


def _risk_unknowns(campaign):
    risks = []
    if campaign.get('missing_evidence'):
        risks.append('missing sibling evidence gates')
    if campaign.get('rejected_evidence'):
        risks.append('rejected or unsafe evidence present')
    if campaign.get('campaign_type') == 'continue_refinement':
        risks.append('refinement should remain non-final and cheap')
    return risks


def _reason_for_action(action):
    return {
        'retire_stale_or_blocked_line': 'stale or rejected evidence should retire this line',
        'request_math_gate': 'math gate is the next missing campaign criterion',
        'request_code_gate': 'code gate is the next missing campaign criterion',
        'request_language_gate': 'language handoff gate is the next missing campaign criterion',
        'continue_refinement': 'resolved evidence can continue one refinement campaign',
        'emit_acceptance_bundle': 'all evidence gates are accepted for the next module cycle',
    }.get(action, 'campaign action selected')


def _source_hash(evaluator, outcome, contracts, adjudicator, agenda, lifecycle, scorecard):
    return stable_digest({
        'evaluator': evaluator.get('ledger_hash') or evaluator.get('ledger_id'),
        'outcome': outcome.get('ledger_hash') or outcome.get('ledger_id'),
        'contract': contracts.get('ledger_hash'),
        'adjudicator': adjudicator.get('ledger_hash'),
        'agenda': agenda.get('ledger_hash'),
        'lifecycle': lifecycle.get('ledger_hash'),
        'scorecard': scorecard.get('ledger_hash'),
    })


def _find_campaign(campaigns, hypothesis_id):
    for campaign in campaigns:
        if campaign.get('hypothesis_id') == hypothesis_id:
            return campaign
    return None


def _campaign_type_counts(campaigns):
    counts = {
        'continue_refinement': 0,
        'request_math_gate': 0,
        'request_code_gate': 0,
        'request_language_gate': 0,
        'retire_line': 0,
        'emit_acceptance_bundle': 0,
        'repair_boundary': 0,
        'noop': 0,
    }
    for campaign in campaigns:
        value = str(campaign.get('campaign_type') or 'noop')
        counts[value] = counts.get(value, 0) + 1
    return counts


def _field_map(campaigns, field):
    return {str(item.get('hypothesis_id')): list(item.get(field) or []) for item in campaigns if item.get(field)}


def _collect_field(campaigns, field):
    values = []
    for campaign in campaigns:
        values.extend(list(campaign.get(field) or []))
    return values


def _dominant_state(current, incoming):
    order = {'retired': 5, 'refine': 4, 'repair': 3, 'resolved': 2, 'waiting': 1}
    current_state = str(current or 'waiting')
    incoming_state = str(incoming or 'waiting')
    return current_state if order.get(current_state, 0) >= order.get(incoming_state, 0) else incoming_state


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
