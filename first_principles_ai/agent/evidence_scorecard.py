"""Experiment evidence scorecard and refinement gate runner for AI Different.

The scorecard is a compact science-side orchestration artifact. It scores
hypotheses against accepted, rejected, and missing evidence gates, then emits at
most one neutral module-chat response.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .cross_module_adjudicator import ADJUDICATOR_LEDGER_KIND
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


SCORECARD_LEDGER_KIND = 'ai_different.experiment_evidence_scorecard_ledger'
REQUIRED_GATES = ('math_proof', 'code_proof', 'language_epoch_plan')


def empty_scorecard_ledger() -> dict[str, Any]:
    return {
        'schema_version': 1,
        'ledger_kind': SCORECARD_LEDGER_KIND,
        'processed_message_ids': [],
        'processed_source_hashes': [],
        'resolved_hypothesis_ids': [],
        'retired_hypothesis_ids': [],
        'refined_hypothesis_ids': [],
        'scorecards': [],
        'scorecard_records': [],
        'outgoing_response_ids': [],
        'latest': {},
    }


def load_scorecard_ledger(path: str | Path | None) -> dict[str, Any]:
    if not path or not Path(path).exists():
        return empty_scorecard_ledger()
    with Path(path).open('r', encoding='utf-8') as handle:
        ledger = json.load(handle)
    return validate_scorecard_ledger(ledger)


def write_scorecard_ledger(path: str | Path, ledger: dict[str, Any]) -> Path:
    validated = validate_scorecard_ledger(ledger)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(validated, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    return output


def validate_scorecard_ledger(ledger: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(ledger, dict):
        raise ValueError('scorecard ledger must be a JSON object')
    if ledger.get('ledger_kind') != SCORECARD_LEDGER_KIND:
        raise ValueError('scorecard ledger has wrong ledger_kind')
    for field in (
        'processed_message_ids',
        'processed_source_hashes',
        'resolved_hypothesis_ids',
        'retired_hypothesis_ids',
        'refined_hypothesis_ids',
        'outgoing_response_ids',
    ):
        values = ledger.get(field)
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            raise ValueError(f'{field} must be strings')
    for field in ('scorecards', 'scorecard_records'):
        values = ledger.get(field)
        if not isinstance(values, list) or not all(isinstance(item, dict) for item in values):
            raise ValueError(f'{field} must be objects')
    latest = ledger.get('latest', {})
    if not isinstance(latest, dict):
        raise ValueError('scorecard latest must be an object')
    validated = {
        'schema_version': int(ledger.get('schema_version', 1) or 1),
        'ledger_kind': SCORECARD_LEDGER_KIND,
        'processed_message_ids': _unique_strings(ledger['processed_message_ids']),
        'processed_source_hashes': _unique_strings(ledger['processed_source_hashes']),
        'resolved_hypothesis_ids': _unique_strings(ledger['resolved_hypothesis_ids']),
        'retired_hypothesis_ids': _unique_strings(ledger['retired_hypothesis_ids']),
        'refined_hypothesis_ids': _unique_strings(ledger['refined_hypothesis_ids']),
        'scorecards': list(ledger['scorecards']),
        'scorecard_records': list(ledger['scorecard_records']),
        'outgoing_response_ids': _unique_strings(ledger['outgoing_response_ids']),
        'latest': dict(latest),
    }
    if ledger.get('ledger_hash'):
        validated['ledger_hash'] = str(ledger['ledger_hash'])
    if ledger.get('artifact_path'):
        validated['artifact_path'] = str(ledger['artifact_path'])
    return validated


def read_scorecard_transcript(path: str | Path | None) -> dict[str, Any]:
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
        raise ValueError('plain scorecard input must be a JSON object')
    return value


def build_evidence_scorecard(
    *,
    transcript_messages: list[dict[str, Any]],
    scorecard_ledger: dict[str, Any],
    evaluator_ledger: dict[str, Any] | None = None,
    outcome_ledger: dict[str, Any] | None = None,
    contract_ledger: dict[str, Any] | None = None,
    adjudicator_ledger: dict[str, Any] | None = None,
    agenda_ledger: dict[str, Any] | None = None,
    lifecycle_ledger: dict[str, Any] | None = None,
    runtime_memory_data: dict[str, Any] | None = None,
    runtime_memory_hash_state: dict[str, Any],
    project_owned_boundary: dict[str, Any],
    artifact_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    ledger = validate_scorecard_ledger(scorecard_ledger)
    evaluator = _valid_evaluator_or_empty(evaluator_ledger or {})
    outcome = _valid_outcome_or_empty(outcome_ledger or {})
    contracts = _valid_contract_or_empty(contract_ledger or {})
    adjudicator = _valid_adjudicator_or_empty(adjudicator_ledger or {})
    agenda = _valid_agenda_or_empty(agenda_ledger or {})
    lifecycle = _valid_lifecycle_or_empty(lifecycle_ledger or {})
    source_hash = _source_hash(evaluator, outcome, contracts, adjudicator, agenda, lifecycle)
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
    scorecards = _merge_scorecards(
        ledger['scorecards'],
        contracts,
        adjudicator,
        agenda,
        lifecycle,
        transcript_messages,
    )
    evidence_items = _extract_scorecard_evidence(new_messages, scorecards)
    _apply_evidence_to_scorecards(scorecards, evidence_items)
    refinement = _candidate_refinement(
        evaluator or outcome,
        agenda,
        lifecycle,
        scorecards,
        runtime_memory_data or {},
        source_hash,
    )
    if not new_messages and (not source_is_new or scorecards):
        selected = _noop_action('no new scorecard evidence or source ledger state')
    else:
        selected = _select_scorecard_action(
            scorecards=scorecards,
            evidence_items=evidence_items,
            refinement=refinement,
            resolved_ids=ledger['resolved_hypothesis_ids'],
            retired_ids=ledger['retired_hypothesis_ids'],
            refined_ids=ledger['refined_hypothesis_ids'],
            project_owned_boundary=project_owned_boundary,
        )
    scorecard_id = 'scorecard_' + stable_digest({
        'source_hash': source_hash,
        'new_message_ids': [module_chat_message_id(message) for message in new_messages],
        'selected_action': selected['selected_action'],
        'hypothesis_id': selected.get('hypothesis_id'),
    })[:16]
    selected['scorecard_id'] = scorecard_id
    message = export_scorecard_message(
        selected,
        scorecards=scorecards,
        evidence_items=evidence_items,
        refinement=refinement,
        runtime_memory_hash_state=runtime_memory_hash_state,
        project_owned_boundary=project_owned_boundary,
        source_hashes={
            'scorecard_source_hash': source_hash,
            'evaluator_ledger_hash': evaluator.get('ledger_hash'),
            'outcome_ledger_hash': outcome.get('ledger_hash'),
            'contract_ledger_hash': contracts.get('ledger_hash'),
            'adjudicator_ledger_hash': adjudicator.get('ledger_hash'),
            'agenda_ledger_hash': agenda.get('ledger_hash'),
            'lifecycle_ledger_hash': lifecycle.get('ledger_hash'),
        },
    )
    new_ids = [module_chat_message_id(message) for message in new_messages]
    outgoing_id = module_chat_message_id(message) if message else None
    hypothesis_id = selected.get('hypothesis_id')
    if selected['selected_action'] == 'mark_scorecard_resolved' and hypothesis_id:
        ledger['resolved_hypothesis_ids'] = _unique_strings(
            list(ledger['resolved_hypothesis_ids']) + [str(hypothesis_id)]
        )
    if selected['selected_action'] == 'retire_unsatisfied_hypothesis' and hypothesis_id:
        ledger['retired_hypothesis_ids'] = _unique_strings(
            list(ledger['retired_hypothesis_ids']) + [str(hypothesis_id)]
        )
    if selected['selected_action'] == 'refine_next_hypothesis' and hypothesis_id:
        ledger['refined_hypothesis_ids'] = _unique_strings(
            list(ledger['refined_hypothesis_ids']) + [str(hypothesis_id)]
        )
    if new_messages:
        ledger['processed_message_ids'] = _unique_strings(
            list(ledger['processed_message_ids']) + new_ids
        )
    if source_is_new and selected['selected_action'] in {'summarize_noop', 'refine_next_hypothesis'}:
        ledger['processed_source_hashes'] = _unique_strings(
            list(ledger['processed_source_hashes']) + [source_hash]
        )
    if outgoing_id:
        ledger['outgoing_response_ids'] = _unique_strings(
            list(ledger['outgoing_response_ids']) + [outgoing_id]
        )
    latest = {
        'scorecard_id': scorecard_id,
        'new_message_count': len(new_messages),
        'skipped_message_count': len(skipped_messages),
        'source_is_new': source_is_new,
        'readiness_counts': _readiness_counts(scorecards),
        'selected_action': selected['selected_action'],
        'chosen_recipient': message.get('recipient') if message else None,
        'outbox_count': 1 if message else 0,
        'label_leaks': selected.get('label_leaks', []),
        'runtime_memory_hash_state': dict(runtime_memory_hash_state),
        'runtime_memory_mutated': not bool(runtime_memory_hash_state.get('unchanged', True)),
    }
    record = {
        'scorecard_id': scorecard_id,
        'scorecard_hash': stable_digest({
            'scorecard_id': scorecard_id,
            'selected': selected,
            'refinement': refinement,
        }),
        'processed_message_ids': new_ids,
        'source_hash': source_hash,
        'hypothesis_ids': [item.get('hypothesis_id') for item in scorecards],
        'source_agenda_ids': _unique_strings(_collect_field(scorecards, 'source_agenda_ids')),
        'source_contract_ids': _unique_strings(_collect_field(scorecards, 'source_contract_ids')),
        'source_lifecycle_ids': _unique_strings(_collect_field(scorecards, 'source_lifecycle_ids')),
        'source_adjudication_ids': _unique_strings(_collect_field(scorecards, 'source_adjudication_ids')),
        'required_evidence_gates': _gate_map(scorecards, 'required_evidence_gates'),
        'accepted_evidence': _gate_map(scorecards, 'accepted_evidence_gates'),
        'rejected_evidence': _gate_map(scorecards, 'rejected_evidence_gates'),
        'missing_evidence': _gate_map(scorecards, 'missing_evidence_gates'),
        'sibling_evidence_lineage': _sibling_lineage(scorecards),
        'lifecycle_states': {item.get('hypothesis_id'): item.get('lifecycle_state') for item in scorecards},
        'readiness_states': {item.get('hypothesis_id'): item.get('readiness_state') for item in scorecards},
        'retirement_or_refinement_reason': selected.get('retirement_or_refinement_reason'),
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
        },
    }
    ledger['scorecards'] = scorecards
    if new_messages or source_is_new or message is not None:
        ledger['scorecard_records'] = list(ledger['scorecard_records']) + [record]
    ledger['latest'] = latest
    if artifact_path is not None:
        ledger['artifact_path'] = str(Path(artifact_path))
    ledger['ledger_hash'] = stable_digest({
        key: value for key, value in ledger.items()
        if key != 'ledger_hash'
    })
    return ledger, message


def export_scorecard_message(
    selected: dict[str, Any],
    *,
    scorecards: list[dict[str, Any]],
    evidence_items: list[dict[str, Any]],
    refinement: dict[str, Any] | None,
    runtime_memory_hash_state: dict[str, Any],
    project_owned_boundary: dict[str, Any],
    source_hashes: dict[str, Any],
    topic: str = 'ai_different.experiment_evidence_scorecard',
) -> dict[str, Any] | None:
    if selected['selected_action'] == 'summarize_noop':
        return None
    recipient = validate_participant(selected.get('chosen_recipient') or 'orchestrator')
    card = _find_scorecard(scorecards, selected.get('hypothesis_id')) or {}
    leak_terms = label_leak_terms({
        'selected': selected,
        'scorecard': card,
        'refinement': refinement,
        'evidence': evidence_items,
    })
    body = {
        'module': 'AI Different',
        'response_kind': 'experiment_evidence_scorecard',
        'scorecard_id': selected.get('scorecard_id'),
        'hypothesis_id': selected.get('hypothesis_id'),
        'readiness_state': selected.get('readiness_state') or card.get('readiness_state'),
        'selected_action': selected['selected_action'],
        'recommended_recipient': recipient,
        'recommended_action': selected['selected_action'],
        'reason': selected.get('reason'),
        'accepted_evidence': selected.get('accepted_evidence') or list(card.get('accepted_evidence_ids') or []),
        'missing_evidence': selected.get('missing_evidence') or list(card.get('missing_evidence_gates') or []),
        'rejected_evidence': selected.get('rejected_evidence') or list(card.get('rejected_evidence_ids') or []),
        'repair_request': selected.get('repair_request'),
        'retirement_or_refinement_reason': selected.get('retirement_or_refinement_reason'),
        'refinement_lineage': refinement if selected['selected_action'] == 'refine_next_hypothesis' else None,
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
            'scorecard_id': body['scorecard_id'],
            'hypothesis_id': body['hypothesis_id'],
            'readiness_state': body['readiness_state'],
            'selected_action': body['selected_action'],
            'accepted_evidence': body['accepted_evidence'],
            'missing_evidence': body['missing_evidence'],
            'rejected_evidence': body['rejected_evidence'],
            'runtime_memory_hash_state': body['runtime_memory_hash_state'],
            'runtime_memory_mutated': body['runtime_memory_mutated'],
            'label_clean': body['label_clean'],
            'no_sibling_imports': True,
        },
        tags=[
            'ai_different',
            'experiment_evidence_scorecard',
            body['selected_action'],
            'label_clean' if body['label_clean'] else 'label_review_needed',
        ],
    )


def write_scorecard_outbox_jsonl(path: str | Path, message: dict[str, Any] | None) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        (json.dumps(message, sort_keys=True) + '\n') if message is not None else '',
        encoding='utf-8',
    )
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


def _merge_scorecards(
    existing: list[dict[str, Any]],
    contract_ledger: dict[str, Any],
    adjudicator_ledger: dict[str, Any],
    agenda_ledger: dict[str, Any],
    lifecycle_ledger: dict[str, Any],
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cards: dict[str, dict[str, Any]] = {
        str(item.get('hypothesis_id') or item.get('contract_id')): dict(item)
        for item in existing
        if item.get('hypothesis_id') or item.get('contract_id')
    }
    for contract in list(contract_ledger.get('contracts') or []):
        _upsert_card(cards, _card_from_contract(contract, 'contract_ledger'))
    for contract in list(adjudicator_ledger.get('contract_states') or []):
        _upsert_card(cards, _card_from_contract(contract, 'adjudicator_ledger'))
    for hypothesis in list(agenda_ledger.get('hypotheses') or []):
        _upsert_card(cards, _card_from_contract(hypothesis, 'agenda_ledger'))
    for record in list(agenda_ledger.get('agenda_records') or []):
        candidate = dict(record.get('candidate_next_experiment') or {})
        if candidate:
            _upsert_card(cards, _card_from_candidate(candidate, record))
    for hypothesis in list(lifecycle_ledger.get('hypotheses') or []):
        _upsert_card(cards, _card_from_lifecycle_hypothesis(hypothesis, lifecycle_ledger))
    for message in messages:
        body = dict(message.get('body') or {})
        if message.get('sender') != 'ai_different':
            continue
        kind = body.get('response_kind')
        if kind == 'experiment_contract':
            _upsert_card(cards, _card_from_contract({
                'contract_id': body.get('contract_id'),
                'signature': body.get('contract_signature'),
                'selected_world': body.get('selected_world'),
                'selected_probe': body.get('selected_probe'),
                'required_evidence_gates': body.get('required_evidence_gates') or [],
            }, 'transcript_contract'))
        elif kind == 'hypothesis_lifecycle':
            _upsert_card(cards, _card_from_lifecycle_message(body))
        elif kind == 'experiment_evidence_scorecard':
            _upsert_card(cards, _card_from_prior_scorecard(body))
    return [_finalize_card(card) for card in cards.values()]


def _card_from_contract(contract: dict[str, Any], source: str) -> dict[str, Any]:
    contract_id = contract.get('contract_id')
    return {
        'hypothesis_id': contract.get('hypothesis_id') or (f'hypothesis:{contract_id}' if contract_id else None),
        'contract_id': contract_id,
        'source_contract_ids': [contract_id] if contract_id else [],
        'source_agenda_ids': list(contract.get('source_agenda_ids') or []),
        'source_lifecycle_ids': list(contract.get('source_lifecycle_ids') or []),
        'source_adjudication_ids': list(contract.get('source_adjudication_ids') or []),
        'required_evidence_gates': _required_gates(contract),
        'accepted_evidence_gates': list(contract.get('accepted_evidence_gates') or contract.get('satisfied_evidence_gates') or []),
        'rejected_evidence_gates': list(contract.get('rejected_evidence_gates') or contract.get('failed_evidence_gates') or []),
        'accepted_evidence_ids': list(contract.get('accepted_evidence_ids') or contract.get('evidence_ids') or []),
        'rejected_evidence_ids': list(contract.get('rejected_evidence_ids') or []),
        'sibling_evidence_lineage': list(contract.get('sibling_evidence_lineage') or []),
        'lifecycle_state': _state_from_status(contract.get('lifecycle_state') or contract.get('status')),
        'lineage': [source],
    }


def _card_from_candidate(candidate: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    card = _card_from_contract(candidate, 'agenda_candidate')
    card['contract_id'] = candidate.get('contract_id')
    card['hypothesis_id'] = f"hypothesis:{candidate.get('contract_id')}" if candidate.get('contract_id') else None
    card['source_agenda_ids'] = [record.get('agenda_id')] if record.get('agenda_id') else []
    return card


def _card_from_lifecycle_hypothesis(hypothesis: dict[str, Any], lifecycle_ledger: dict[str, Any]) -> dict[str, Any]:
    card = _card_from_contract(hypothesis, 'lifecycle_ledger')
    card['source_lifecycle_ids'] = [lifecycle_ledger.get('latest', {}).get('lifecycle_id')] if lifecycle_ledger.get('latest', {}).get('lifecycle_id') else []
    return card


def _card_from_lifecycle_message(body: dict[str, Any]) -> dict[str, Any]:
    hypothesis_id = body.get('hypothesis_id')
    contract_id = body.get('contract_id') or str(hypothesis_id or '').replace('hypothesis:', '')
    return {
        'hypothesis_id': hypothesis_id,
        'contract_id': contract_id,
        'source_contract_ids': [contract_id] if contract_id else [],
        'source_agenda_ids': [],
        'source_lifecycle_ids': [body.get('lifecycle_id')] if body.get('lifecycle_id') else [],
        'source_adjudication_ids': [],
        'required_evidence_gates': _required_gates(body),
        'accepted_evidence_gates': list(body.get('accepted_evidence') or []),
        'rejected_evidence_gates': [],
        'accepted_evidence_ids': list(body.get('accepted_evidence') or []),
        'rejected_evidence_ids': [],
        'sibling_evidence_lineage': list(body.get('sibling_evidence_lineage') or []),
        'lifecycle_state': body.get('lifecycle_state') or 'waiting_evidence',
        'lineage': ['lifecycle_message'],
    }


def _card_from_prior_scorecard(body: dict[str, Any]) -> dict[str, Any]:
    hypothesis_id = body.get('hypothesis_id')
    return {
        'hypothesis_id': hypothesis_id,
        'contract_id': str(hypothesis_id or '').replace('hypothesis:', ''),
        'source_contract_ids': [],
        'source_agenda_ids': [],
        'source_lifecycle_ids': [],
        'source_adjudication_ids': [],
        'required_evidence_gates': _required_gates(body),
        'accepted_evidence_gates': list(body.get('accepted_evidence') or []),
        'rejected_evidence_gates': list(body.get('rejected_evidence') or []),
        'accepted_evidence_ids': list(body.get('accepted_evidence') or []),
        'rejected_evidence_ids': list(body.get('rejected_evidence') or []),
        'sibling_evidence_lineage': [],
        'lifecycle_state': body.get('readiness_state') or 'waiting_evidence',
        'lineage': ['prior_scorecard_message'],
    }


def _upsert_card(cards: dict[str, dict[str, Any]], incoming: dict[str, Any]):
    hypothesis_id = incoming.get('hypothesis_id')
    if not hypothesis_id:
        return
    current = cards.setdefault(str(hypothesis_id), {})
    current.update({
        'hypothesis_id': str(hypothesis_id),
        'contract_id': incoming.get('contract_id') or current.get('contract_id'),
        'required_evidence_gates': _unique_strings(
            list(current.get('required_evidence_gates') or [])
            + list(incoming.get('required_evidence_gates') or [])
        ),
        'accepted_evidence_gates': _unique_strings(
            list(current.get('accepted_evidence_gates') or [])
            + list(incoming.get('accepted_evidence_gates') or [])
        ),
        'rejected_evidence_gates': _unique_strings(
            list(current.get('rejected_evidence_gates') or [])
            + list(incoming.get('rejected_evidence_gates') or [])
        ),
        'accepted_evidence_ids': _unique_strings(
            list(current.get('accepted_evidence_ids') or [])
            + list(incoming.get('accepted_evidence_ids') or [])
        ),
        'rejected_evidence_ids': _unique_strings(
            list(current.get('rejected_evidence_ids') or [])
            + list(incoming.get('rejected_evidence_ids') or [])
        ),
        'source_agenda_ids': _unique_strings(list(current.get('source_agenda_ids') or []) + list(incoming.get('source_agenda_ids') or [])),
        'source_contract_ids': _unique_strings(list(current.get('source_contract_ids') or []) + list(incoming.get('source_contract_ids') or [])),
        'source_lifecycle_ids': _unique_strings(list(current.get('source_lifecycle_ids') or []) + list(incoming.get('source_lifecycle_ids') or [])),
        'source_adjudication_ids': _unique_strings(list(current.get('source_adjudication_ids') or []) + list(incoming.get('source_adjudication_ids') or [])),
        'sibling_evidence_lineage': list(current.get('sibling_evidence_lineage') or []) + list(incoming.get('sibling_evidence_lineage') or []),
        'lifecycle_state': _dominant_state(current.get('lifecycle_state'), incoming.get('lifecycle_state')),
        'lineage': _unique_strings(list(current.get('lineage') or []) + list(incoming.get('lineage') or [])),
    })


def _extract_scorecard_evidence(messages: list[dict[str, Any]], scorecards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    known_contract_ids = {card.get('contract_id') for card in scorecards}
    items = []
    for message in messages:
        body = dict(message.get('body') or {})
        evidence = dict(message.get('evidence') or {})
        sender = message.get('sender')
        contract_id = (
            body.get('contract_id')
            or evidence.get('contract_id')
            or body.get('experiment_contract_id')
            or evidence.get('experiment_contract_id')
        )
        hypothesis_id = body.get('hypothesis_id') or evidence.get('hypothesis_id')
        if not hypothesis_id and contract_id:
            hypothesis_id = f'hypothesis:{contract_id}'
        gate = _evidence_gate(message)
        status = _evidence_status(message)
        evidence_id = str(body.get('evidence_id') or evidence.get('evidence_id') or module_chat_message_id(message))
        item = {
            'evidence_id': evidence_id,
            'message_id': module_chat_message_id(message),
            'sender': sender,
            'topic': message.get('topic'),
            'hypothesis_id': str(hypothesis_id) if hypothesis_id else None,
            'contract_id': str(contract_id) if contract_id else None,
            'evidence_gate': gate,
            'status': status,
            'summary': body.get('summary') or body.get('note') or body.get('reason'),
            'label_leaks': label_leak_terms({'body': body, 'evidence': evidence}),
            'third_party_checkpoint_used': bool(body.get('third_party_checkpoint_used') or evidence.get('third_party_checkpoint_used')),
        }
        if item['contract_id'] in known_contract_ids or item['hypothesis_id'] or sender in {'language_model_2', 'funfun', 'code_module'}:
            items.append(item)
    return items


def _apply_evidence_to_scorecards(scorecards: list[dict[str, Any]], evidence_items: list[dict[str, Any]]):
    by_hypothesis = {card.get('hypothesis_id'): card for card in scorecards}
    by_contract = {card.get('contract_id'): card for card in scorecards}
    for item in evidence_items:
        card = by_hypothesis.get(item.get('hypothesis_id')) or by_contract.get(item.get('contract_id'))
        if not card:
            continue
        gate = item.get('evidence_gate')
        card.setdefault('sibling_evidence_lineage', []).append({
            'evidence_id': item['evidence_id'],
            'sender': item.get('sender'),
            'topic': item.get('topic'),
            'evidence_gate': gate,
            'status': item.get('status'),
        })
        if item.get('label_leaks') or item.get('third_party_checkpoint_used') or item.get('status') == 'failed':
            card.setdefault('rejected_evidence_gates', []).append(
                'safety_label_project_boundary'
                if item.get('label_leaks') or item.get('third_party_checkpoint_used')
                else gate
            )
            card.setdefault('rejected_evidence_ids', []).append(item['evidence_id'])
        elif item.get('status') == 'satisfied':
            card.setdefault('accepted_evidence_gates', []).append(gate)
            card.setdefault('accepted_evidence_ids', []).append(item['evidence_id'])
        elif item.get('status') == 'retired':
            card['lifecycle_state'] = 'retired'
        elif item.get('status') == 'refine_next':
            card['lifecycle_state'] = 'refine_next'
    for card in scorecards:
        _finalize_card(card)


def _finalize_card(card: dict[str, Any]) -> dict[str, Any]:
    card['required_evidence_gates'] = _unique_strings(list(card.get('required_evidence_gates') or []) or list(REQUIRED_GATES))
    card['accepted_evidence_gates'] = sorted(set(card.get('accepted_evidence_gates') or []))
    card['rejected_evidence_gates'] = sorted(set(card.get('rejected_evidence_gates') or []))
    card['accepted_evidence_ids'] = _unique_strings(list(card.get('accepted_evidence_ids') or []))
    card['rejected_evidence_ids'] = _unique_strings(list(card.get('rejected_evidence_ids') or []))
    card['missing_evidence_gates'] = sorted(
        set(card['required_evidence_gates'])
        - set(card['accepted_evidence_gates'])
        - set(card['rejected_evidence_gates'])
    )
    if card.get('lifecycle_state') in {'retired', 'refine_next'}:
        card['readiness_state'] = card['lifecycle_state']
    elif card['rejected_evidence_gates']:
        card['readiness_state'] = 'repair'
    elif not card['missing_evidence_gates']:
        card['readiness_state'] = 'resolved'
        card['lifecycle_state'] = 'resolved'
    else:
        card['readiness_state'] = 'waiting'
        card['lifecycle_state'] = card.get('lifecycle_state') or 'waiting_evidence'
    card['scorecard_hash'] = stable_digest({
        'hypothesis_id': card.get('hypothesis_id'),
        'accepted': card.get('accepted_evidence_gates'),
        'rejected': card.get('rejected_evidence_gates'),
        'missing': card.get('missing_evidence_gates'),
        'readiness': card.get('readiness_state'),
    })
    return card


def _select_scorecard_action(
    *,
    scorecards: list[dict[str, Any]],
    evidence_items: list[dict[str, Any]],
    refinement: dict[str, Any] | None,
    resolved_ids: list[str],
    retired_ids: list[str],
    refined_ids: list[str],
    project_owned_boundary: dict[str, Any],
) -> dict[str, Any]:
    safety_fail = _first_safety_failure(scorecards, evidence_items, project_owned_boundary)
    if safety_fail:
        return _repair_action('safety_label_or_project_owned_repair', 'code_module', safety_fail, 'failed safety, label, or project-owned boundary')
    protocol_fail = _protocol_failure(evidence_items)
    if protocol_fail:
        return _repair_action('protocol_or_handoff_repair', 'language_model_2', protocol_fail, 'language protocol or handoff evidence needs repair')
    for card in scorecards:
        if card.get('readiness_state') == 'waiting' and 'math_proof' in set(card.get('missing_evidence_gates') or []):
            return _missing_action(card, 'request_missing_math_evidence', 'funfun', 'math_proof')
    for card in scorecards:
        if card.get('readiness_state') == 'waiting' and 'code_proof' in set(card.get('missing_evidence_gates') or []):
            return _missing_action(card, 'request_missing_code_evidence', 'code_module', 'code_proof')
    for card in scorecards:
        if card.get('readiness_state') == 'resolved' and card.get('hypothesis_id') not in set(resolved_ids):
            return _card_action('mark_scorecard_resolved', 'broadcast', card, 'all required evidence gates are accepted')
    for card in scorecards:
        if card.get('readiness_state') == 'repair' and card.get('hypothesis_id') not in set(retired_ids):
            return _card_action(
                'retire_unsatisfied_hypothesis',
                'broadcast',
                card,
                'rejected evidence makes the scorecard unsatisfied',
                retirement_or_refinement_reason='rejected evidence or boundary failure',
            )
    for card in scorecards:
        if (
            card.get('readiness_state') == 'resolved'
            and card.get('hypothesis_id') in set(resolved_ids)
            and card.get('hypothesis_id') not in set(refined_ids)
            and refinement is not None
        ):
            return _card_action(
                'refine_next_hypothesis',
                'broadcast',
                card,
                'accepted evidence permits one refinement',
                retirement_or_refinement_reason='accepted scorecard gates support refinement',
            )
    return _noop_action('no new scorecard action')


def _candidate_refinement(
    source_ledger: dict[str, Any],
    agenda_ledger: dict[str, Any],
    lifecycle_ledger: dict[str, Any],
    scorecards: list[dict[str, Any]],
    runtime_memory_data: dict[str, Any],
    source_hash: str,
) -> dict[str, Any] | None:
    selected = dict(source_ledger.get('selected_experiment') or {})
    if not selected:
        records = list(agenda_ledger.get('agenda_records') or [])
        candidate = dict((records[-1] if records else {}).get('candidate_next_experiment') or {})
        selected = dict(candidate.get('selected_experiment') or {})
    if not selected.get('world') or not selected.get('probe') or selected.get('runs_final', False):
        return None
    resolved = [card for card in scorecards if card.get('readiness_state') == 'resolved']
    readiness = dict(runtime_memory_data.get('discovery_readiness') or {})
    return {
        'refinement_id': 'scorecard_refinement_' + stable_digest({
            'source_hash': source_hash,
            'selected': selected,
            'resolved': [card.get('hypothesis_id') for card in resolved],
            'readiness_score': readiness.get('readiness_score'),
            'lifecycle_hash': lifecycle_ledger.get('ledger_hash'),
        })[:16],
        'selected_world': selected.get('world'),
        'selected_probe': selected.get('probe'),
        'source_hypothesis_ids': [card.get('hypothesis_id') for card in resolved],
        'expected_transfer_signal': source_ledger.get('expected_transfer_signal') or selected.get('expected_transfer_signal'),
        'label_clean': not label_leak_terms(selected),
        'third_party_checkpoint_used': False,
    }


def _repair_action(action: str, recipient: str, blocker: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        'selected_action': action,
        'chosen_recipient': validate_participant(recipient),
        'hypothesis_id': blocker.get('hypothesis_id'),
        'readiness_state': 'repair',
        'reason': reason,
        'accepted_evidence': [],
        'missing_evidence': [blocker.get('evidence_gate')] if blocker.get('evidence_gate') else [],
        'rejected_evidence': [blocker.get('evidence_id')] if blocker.get('evidence_id') else [],
        'repair_request': dict(blocker),
        'retirement_or_refinement_reason': None,
        'label_leaks': label_leak_terms(blocker),
    }


def _missing_action(card: dict[str, Any], action: str, recipient: str, gate: str) -> dict[str, Any]:
    return {
        'selected_action': action,
        'chosen_recipient': recipient,
        'hypothesis_id': card.get('hypothesis_id'),
        'readiness_state': card.get('readiness_state'),
        'reason': f'missing {gate} evidence',
        'accepted_evidence': list(card.get('accepted_evidence_ids') or []),
        'missing_evidence': [gate],
        'rejected_evidence': list(card.get('rejected_evidence_ids') or []),
        'repair_request': {'hypothesis_id': card.get('hypothesis_id'), 'evidence_gate': gate, 'status': 'missing'},
        'retirement_or_refinement_reason': None,
        'label_leaks': label_leak_terms(card),
    }


def _card_action(
    action: str,
    recipient: str,
    card: dict[str, Any],
    reason: str,
    *,
    retirement_or_refinement_reason: str | None = None,
) -> dict[str, Any]:
    readiness = {
        'mark_scorecard_resolved': 'resolved',
        'retire_unsatisfied_hypothesis': 'retired',
        'refine_next_hypothesis': 'refine',
    }.get(action, card.get('readiness_state'))
    return {
        'selected_action': action,
        'chosen_recipient': recipient,
        'hypothesis_id': card.get('hypothesis_id'),
        'readiness_state': readiness,
        'reason': reason,
        'accepted_evidence': list(card.get('accepted_evidence_ids') or []),
        'missing_evidence': list(card.get('missing_evidence_gates') or []),
        'rejected_evidence': list(card.get('rejected_evidence_ids') or []),
        'repair_request': None,
        'retirement_or_refinement_reason': retirement_or_refinement_reason,
        'label_leaks': label_leak_terms(card),
    }


def _noop_action(reason: str) -> dict[str, Any]:
    return {
        'selected_action': 'summarize_noop',
        'chosen_recipient': None,
        'hypothesis_id': None,
        'readiness_state': None,
        'reason': reason,
        'accepted_evidence': [],
        'missing_evidence': [],
        'rejected_evidence': [],
        'repair_request': None,
        'retirement_or_refinement_reason': None,
        'label_leaks': [],
    }


def _first_safety_failure(scorecards: list[dict[str, Any]], evidence_items: list[dict[str, Any]], project_owned_boundary: dict[str, Any]) -> dict[str, Any] | None:
    if project_owned_boundary.get('third_party_checkpoint_used'):
        return {'sender': 'code_module', 'reason': 'third-party checkpoint boundary failed', 'evidence_gate': 'project_owned_checkpoint_boundary_explicit', 'status': 'failed'}
    for item in evidence_items:
        if item.get('label_leaks') or item.get('third_party_checkpoint_used'):
            return item
    for card in scorecards:
        if 'safety_label_project_boundary' in set(card.get('rejected_evidence_gates') or []):
            return {'hypothesis_id': card.get('hypothesis_id'), 'sender': 'code_module', 'evidence_gate': 'safety_label_project_boundary', 'status': 'failed'}
    return None


def _protocol_failure(evidence_items: list[dict[str, Any]]) -> dict[str, Any] | None:
    for item in evidence_items:
        if item.get('sender') == 'language_model_2' and item.get('status') in {'missing', 'failed'}:
            return item
    return None


def _evidence_gate(message: dict[str, Any]) -> str:
    body = dict(message.get('body') or {})
    evidence = dict(message.get('evidence') or {})
    explicit = body.get('evidence_gate') or evidence.get('evidence_gate') or body.get('gate') or evidence.get('gate') or body.get('missing_evidence_gate')
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


def _evidence_status(message: dict[str, Any]) -> str:
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
    if explicit in {'refine_next', 'refined'}:
        return 'refine_next'
    if any(token in text for token in ('failed', 'blocked', 'contradict')):
        return 'failed'
    if any(token in text for token in ('satisfied', 'resolved', 'passed', 'confirmed')):
        return 'satisfied'
    return 'advisory'


def _required_gates(item: dict[str, Any]) -> list[str]:
    explicit = set(item.get('required_evidence_gates') or [])
    gates = set(REQUIRED_GATES)
    gates.update(gate for gate in explicit if gate in set(REQUIRED_GATES))
    return sorted(gates)


def _state_from_status(status: Any) -> str:
    text = str(status or '').lower()
    if text in {'resolved', 'complete', 'satisfied'}:
        return 'resolved'
    if text in {'blocked', 'failed', 'repair'}:
        return 'repair'
    if text == 'retired':
        return 'retired'
    if text in {'refine_next', 'refine'}:
        return 'refine'
    return 'waiting'


def _dominant_state(current: Any, incoming: Any) -> str:
    order = {'retired': 5, 'refine': 4, 'repair': 3, 'resolved': 2, 'waiting': 1}
    current_state = _state_from_status(current)
    incoming_state = _state_from_status(incoming)
    return current_state if order.get(current_state, 0) >= order.get(incoming_state, 0) else incoming_state


def _source_hash(evaluator, outcome, contracts, adjudicator, agenda, lifecycle) -> str:
    return stable_digest({
        'evaluator': evaluator.get('ledger_hash') or evaluator.get('ledger_id'),
        'outcome': outcome.get('ledger_hash') or outcome.get('ledger_id'),
        'contract': contracts.get('ledger_hash'),
        'adjudicator': adjudicator.get('ledger_hash'),
        'agenda': agenda.get('ledger_hash'),
        'lifecycle': lifecycle.get('ledger_hash'),
    })


def _find_scorecard(scorecards: list[dict[str, Any]], hypothesis_id: str | None) -> dict[str, Any] | None:
    for card in scorecards:
        if card.get('hypothesis_id') == hypothesis_id:
            return card
    return None


def _readiness_counts(scorecards: list[dict[str, Any]]) -> dict[str, int]:
    counts = {'waiting': 0, 'resolved': 0, 'retired': 0, 'refine': 0, 'repair': 0}
    for card in scorecards:
        state = str(card.get('readiness_state') or 'waiting')
        counts[state] = counts.get(state, 0) + 1
    return counts


def _gate_map(scorecards: list[dict[str, Any]], field: str) -> dict[str, list[str]]:
    return {str(card.get('hypothesis_id')): list(card.get(field) or []) for card in scorecards if card.get(field)}


def _sibling_lineage(scorecards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for card in scorecards:
        for item in list(card.get('sibling_evidence_lineage') or []):
            if isinstance(item, dict):
                row = dict(item)
                row['hypothesis_id'] = card.get('hypothesis_id')
                rows.append(row)
    return rows


def _collect_field(scorecards: list[dict[str, Any]], field: str) -> list[Any]:
    values = []
    for card in scorecards:
        values.extend(list(card.get(field) or []))
    return values


def _unique_strings(values: list[Any]) -> list[str]:
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
