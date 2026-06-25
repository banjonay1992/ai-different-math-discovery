"""Hypothesis lifecycle and evidence memory curator for AI Different.

This module keeps the science-side memory of proposed hypotheses compact and
repeat-safe. It consumes only plain JSON ledgers/messages and emits at most one
neutral module-chat lifecycle update.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .cross_module_adjudicator import ADJUDICATOR_LEDGER_KIND
from .experiment_agenda import AGENDA_LEDGER_KIND
from .experiment_contracts import CONTRACT_LEDGER_KIND, validate_evaluator_ledger
from .family_outcome_evaluator import EVALUATOR_LEDGER_KIND
from .module_chat_adapter import (
    build_module_chat_message,
    label_leak_terms,
    module_chat_message_id,
    stable_digest,
    validate_module_chat_message,
    validate_participant,
)


LIFECYCLE_LEDGER_KIND = 'ai_different.hypothesis_lifecycle_ledger'
LIFECYCLE_STATES = {
    'proposed',
    'waiting_evidence',
    'blocked',
    'resolved',
    'retired',
    'refine_next',
}


def empty_hypothesis_lifecycle_ledger() -> dict[str, Any]:
    return {
        'schema_version': 1,
        'ledger_kind': LIFECYCLE_LEDGER_KIND,
        'processed_message_ids': [],
        'processed_source_hashes': [],
        'resolved_hypothesis_ids': [],
        'retired_hypothesis_ids': [],
        'refined_hypothesis_ids': [],
        'hypotheses': [],
        'lifecycle_records': [],
        'outgoing_response_ids': [],
        'latest': {},
    }


def load_hypothesis_lifecycle_ledger(path: str | Path | None) -> dict[str, Any]:
    if not path or not Path(path).exists():
        return empty_hypothesis_lifecycle_ledger()
    with Path(path).open('r', encoding='utf-8') as handle:
        ledger = json.load(handle)
    return validate_hypothesis_lifecycle_ledger(ledger)


def write_hypothesis_lifecycle_ledger(path: str | Path, ledger: dict[str, Any]) -> Path:
    validated = validate_hypothesis_lifecycle_ledger(ledger)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(validated, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    return output


def validate_hypothesis_lifecycle_ledger(ledger: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(ledger, dict):
        raise ValueError('hypothesis lifecycle ledger must be a JSON object')
    if ledger.get('ledger_kind') != LIFECYCLE_LEDGER_KIND:
        raise ValueError('hypothesis lifecycle ledger has wrong ledger_kind')
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
    for field in ('hypotheses', 'lifecycle_records'):
        values = ledger.get(field)
        if not isinstance(values, list) or not all(isinstance(item, dict) for item in values):
            raise ValueError(f'{field} must be objects')
    latest = ledger.get('latest', {})
    if not isinstance(latest, dict):
        raise ValueError('hypothesis lifecycle latest must be an object')
    validated = {
        'schema_version': int(ledger.get('schema_version', 1) or 1),
        'ledger_kind': LIFECYCLE_LEDGER_KIND,
        'processed_message_ids': _unique_strings(ledger['processed_message_ids']),
        'processed_source_hashes': _unique_strings(ledger['processed_source_hashes']),
        'resolved_hypothesis_ids': _unique_strings(ledger['resolved_hypothesis_ids']),
        'retired_hypothesis_ids': _unique_strings(ledger['retired_hypothesis_ids']),
        'refined_hypothesis_ids': _unique_strings(ledger['refined_hypothesis_ids']),
        'hypotheses': list(ledger['hypotheses']),
        'lifecycle_records': list(ledger['lifecycle_records']),
        'outgoing_response_ids': _unique_strings(ledger['outgoing_response_ids']),
        'latest': dict(latest),
    }
    if ledger.get('ledger_hash'):
        validated['ledger_hash'] = str(ledger['ledger_hash'])
    if ledger.get('artifact_path'):
        validated['artifact_path'] = str(ledger['artifact_path'])
    return validated


def read_lifecycle_transcript(path: str | Path | None) -> dict[str, Any]:
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
        raise ValueError('plain lifecycle input must be a JSON object')
    return value


def build_hypothesis_lifecycle(
    *,
    transcript_messages: list[dict[str, Any]],
    lifecycle_ledger: dict[str, Any],
    evaluator_ledger: dict[str, Any] | None = None,
    outcome_ledger: dict[str, Any] | None = None,
    contract_ledger: dict[str, Any] | None = None,
    adjudicator_ledger: dict[str, Any] | None = None,
    agenda_ledger: dict[str, Any] | None = None,
    runtime_memory_data: dict[str, Any] | None = None,
    runtime_memory_hash_state: dict[str, Any],
    project_owned_boundary: dict[str, Any],
    artifact_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    ledger = validate_hypothesis_lifecycle_ledger(lifecycle_ledger)
    evaluator = _valid_evaluator_or_empty(evaluator_ledger or {})
    outcome = _valid_outcome_or_empty(outcome_ledger or {})
    contracts = _valid_contract_or_empty(contract_ledger or {})
    adjudicator = _valid_adjudicator_or_empty(adjudicator_ledger or {})
    agenda = _valid_agenda_or_empty(agenda_ledger or {})
    source_hash = _source_hash(evaluator, outcome, contracts, adjudicator, agenda)
    source_is_new = bool(source_hash) and source_hash not in set(ledger['processed_source_hashes'])
    processed = set(ledger['processed_message_ids'])
    new_messages = [
        message for message in transcript_messages
        if module_chat_message_id(message) not in processed
    ]
    skipped_messages = [
        message for message in transcript_messages
        if module_chat_message_id(message) in processed
    ]
    hypotheses = _merge_hypotheses(
        ledger['hypotheses'],
        contracts,
        adjudicator,
        agenda,
        transcript_messages,
    )
    evidence = _extract_lifecycle_evidence(new_messages, hypotheses)
    _apply_evidence_to_hypotheses(hypotheses, evidence)
    refinement = _candidate_refinement(
        evaluator or outcome,
        agenda,
        hypotheses,
        runtime_memory_data or {},
        source_hash,
    )
    if not new_messages and (not source_is_new or hypotheses):
        selected = _noop_action('no new lifecycle evidence or source ledger state')
    else:
        selected = _select_lifecycle_action(
            hypotheses=hypotheses,
            evidence_items=evidence,
            refinement=refinement,
            resolved_ids=ledger['resolved_hypothesis_ids'],
            retired_ids=ledger['retired_hypothesis_ids'],
            refined_ids=ledger['refined_hypothesis_ids'],
            project_owned_boundary=project_owned_boundary,
        )
    lifecycle_id = 'lifecycle_' + stable_digest({
        'source_hash': source_hash,
        'new_message_ids': [module_chat_message_id(message) for message in new_messages],
        'selected_action': selected['selected_action'],
        'hypothesis_id': selected.get('hypothesis_id'),
    })[:16]
    selected['lifecycle_id'] = lifecycle_id
    message = export_lifecycle_message(
        selected,
        hypotheses=hypotheses,
        evidence_items=evidence,
        refinement=refinement,
        runtime_memory_hash_state=runtime_memory_hash_state,
        project_owned_boundary=project_owned_boundary,
        source_hashes={
            'lifecycle_source_hash': source_hash,
            'evaluator_ledger_hash': evaluator.get('ledger_hash'),
            'outcome_ledger_hash': outcome.get('ledger_hash'),
            'contract_ledger_hash': contracts.get('ledger_hash'),
            'adjudicator_ledger_hash': adjudicator.get('ledger_hash'),
            'agenda_ledger_hash': agenda.get('ledger_hash'),
        },
    )
    new_ids = [module_chat_message_id(message) for message in new_messages]
    outgoing_id = module_chat_message_id(message) if message else None
    hypothesis_id = selected.get('hypothesis_id')
    if selected['selected_action'] == 'mark_resolved' and hypothesis_id:
        ledger['resolved_hypothesis_ids'] = _unique_strings(
            list(ledger['resolved_hypothesis_ids']) + [str(hypothesis_id)]
        )
    if selected['selected_action'] == 'retire_blocked_hypothesis' and hypothesis_id:
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
    if source_is_new and selected['selected_action'] in {
        'summarize_noop',
        'refine_next_hypothesis',
    }:
        ledger['processed_source_hashes'] = _unique_strings(
            list(ledger['processed_source_hashes']) + [source_hash]
        )
    if outgoing_id:
        ledger['outgoing_response_ids'] = _unique_strings(
            list(ledger['outgoing_response_ids']) + [outgoing_id]
        )
    latest = {
        'lifecycle_id': lifecycle_id,
        'new_message_count': len(new_messages),
        'skipped_message_count': len(skipped_messages),
        'source_is_new': source_is_new,
        'state_counts': _state_counts(hypotheses),
        'selected_action': selected['selected_action'],
        'chosen_recipient': message.get('recipient') if message else None,
        'outbox_count': 1 if message else 0,
        'label_leaks': selected.get('label_leaks', []),
        'runtime_memory_hash_state': dict(runtime_memory_hash_state),
        'runtime_memory_mutated': not bool(runtime_memory_hash_state.get('unchanged', True)),
    }
    record = {
        'lifecycle_id': lifecycle_id,
        'lifecycle_hash': stable_digest({
            'lifecycle_id': lifecycle_id,
            'selected': selected,
            'refinement': refinement,
        }),
        'processed_message_ids': new_ids,
        'source_hash': source_hash,
        'hypothesis_ids': [item.get('hypothesis_id') for item in hypotheses],
        'hypothesis_hashes': [item.get('hypothesis_hash') for item in hypotheses],
        'source_agenda_ids': _unique_strings(_collect_field(hypotheses, 'source_agenda_ids')),
        'source_contract_ids': _unique_strings(_collect_field(hypotheses, 'source_contract_ids')),
        'source_adjudication_ids': _unique_strings(_collect_field(hypotheses, 'source_adjudication_ids')),
        'evidence_gates': _evidence_gate_map(hypotheses),
        'sibling_evidence_lineage': _sibling_evidence_lineage(hypotheses),
        'lifecycle_states': {item.get('hypothesis_id'): item.get('lifecycle_state') for item in hypotheses},
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
        },
    }
    ledger['hypotheses'] = hypotheses
    if new_messages or source_is_new or message is not None:
        ledger['lifecycle_records'] = list(ledger['lifecycle_records']) + [record]
    ledger['latest'] = latest
    if artifact_path is not None:
        ledger['artifact_path'] = str(Path(artifact_path))
    ledger['ledger_hash'] = stable_digest({
        key: value for key, value in ledger.items()
        if key != 'ledger_hash'
    })
    return ledger, message


def export_lifecycle_message(
    selected: dict[str, Any],
    *,
    hypotheses: list[dict[str, Any]],
    evidence_items: list[dict[str, Any]],
    refinement: dict[str, Any] | None,
    runtime_memory_hash_state: dict[str, Any],
    project_owned_boundary: dict[str, Any],
    source_hashes: dict[str, Any],
    topic: str = 'ai_different.hypothesis_lifecycle',
) -> dict[str, Any] | None:
    if selected['selected_action'] == 'summarize_noop':
        return None
    recipient = validate_participant(selected.get('chosen_recipient') or 'orchestrator')
    leak_terms = label_leak_terms({
        'selected': selected,
        'hypotheses': hypotheses,
        'refinement': refinement,
        'evidence': evidence_items,
    })
    hypothesis = _find_hypothesis(hypotheses, selected.get('hypothesis_id')) or {}
    body = {
        'module': 'AI Different',
        'response_kind': 'hypothesis_lifecycle',
        'lifecycle_id': selected.get('lifecycle_id'),
        'hypothesis_id': selected.get('hypothesis_id'),
        'hypothesis_hash': hypothesis.get('hypothesis_hash'),
        'lifecycle_state': selected.get('lifecycle_state') or hypothesis.get('lifecycle_state'),
        'selected_action': selected['selected_action'],
        'reason': selected.get('reason'),
        'missing_evidence_gate': selected.get('missing_gate'),
        'accepted_evidence': selected.get('accepted_evidence'),
        'repair_request': selected.get('repair_request'),
        'retirement_or_refinement_reason': selected.get('retirement_or_refinement_reason'),
        'refinement_lineage': refinement if selected['selected_action'] == 'refine_next_hypothesis' else None,
        'evidence_gates': _evidence_gate_map(hypotheses),
        'sibling_evidence_lineage': _sibling_evidence_lineage(hypotheses),
        'source_ledger_hashes': dict(source_hashes),
        'runtime_memory_hash_state': dict(runtime_memory_hash_state),
        'runtime_memory_mutated': not bool(runtime_memory_hash_state.get('unchanged', True)),
        'project_owned_boundary': dict(project_owned_boundary),
        'third_party_checkpoint_used': bool(
            project_owned_boundary.get('third_party_checkpoint_used')
        ),
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
            'lifecycle_id': body['lifecycle_id'],
            'hypothesis_id': body['hypothesis_id'],
            'selected_action': body['selected_action'],
            'lifecycle_state': body['lifecycle_state'],
            'missing_evidence_gate': body['missing_evidence_gate'],
            'accepted_evidence': body['accepted_evidence'],
            'runtime_memory_hash_state': body['runtime_memory_hash_state'],
            'runtime_memory_mutated': body['runtime_memory_mutated'],
            'label_clean': body['label_clean'],
            'no_sibling_imports': True,
        },
        tags=[
            'ai_different',
            'hypothesis_lifecycle',
            body['selected_action'],
            'label_clean' if body['label_clean'] else 'label_review_needed',
        ],
    )


def write_lifecycle_outbox_jsonl(
    path: str | Path,
    message: dict[str, Any] | None,
) -> Path:
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


def _merge_hypotheses(
    existing: list[dict[str, Any]],
    contract_ledger: dict[str, Any],
    adjudicator_ledger: dict[str, Any],
    agenda_ledger: dict[str, Any],
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {
        str(item.get('hypothesis_id') or item.get('contract_id')): dict(item)
        for item in existing
        if item.get('hypothesis_id') or item.get('contract_id')
    }
    for contract in list(contract_ledger.get('contracts') or []):
        _upsert_hypothesis(states, _hypothesis_from_contract(contract, 'contract_ledger'))
    for contract in list(adjudicator_ledger.get('contract_states') or []):
        _upsert_hypothesis(states, _hypothesis_from_contract(contract, 'adjudicator_ledger'))
    for hypothesis in list(agenda_ledger.get('hypotheses') or []):
        _upsert_hypothesis(states, _hypothesis_from_contract(hypothesis, 'agenda_ledger'))
    for record in list(agenda_ledger.get('agenda_records') or []):
        candidate = dict(record.get('candidate_next_experiment') or {})
        if candidate:
            _upsert_hypothesis(states, _hypothesis_from_candidate(candidate, record))
    for message in messages:
        body = dict(message.get('body') or {})
        if message.get('sender') != 'ai_different':
            continue
        kind = body.get('response_kind')
        if kind == 'experiment_contract':
            _upsert_hypothesis(states, _hypothesis_from_contract({
                'contract_id': body.get('contract_id'),
                'signature': body.get('contract_signature'),
                'selected_world': body.get('selected_world'),
                'selected_probe': body.get('selected_probe'),
                'required_evidence_gates': body.get('required_evidence_gates') or [],
                'status': 'open',
            }, 'transcript_contract'))
        elif kind == 'experiment_agenda':
            candidate = dict(body.get('compact_next_experiment_contract') or {})
            if candidate:
                _upsert_hypothesis(states, _hypothesis_from_candidate(candidate, body))
        elif kind == 'cross_module_adjudication':
            _upsert_hypothesis(states, _hypothesis_from_adjudication(body))
        elif kind == 'hypothesis_lifecycle':
            _upsert_hypothesis(states, _hypothesis_from_lifecycle(body))
    return [_finalize_hypothesis(item) for item in states.values()]


def _hypothesis_from_contract(contract: dict[str, Any], source: str) -> dict[str, Any]:
    contract_id = contract.get('contract_id')
    return {
        'hypothesis_id': contract.get('hypothesis_id') or (f'hypothesis:{contract_id}' if contract_id else None),
        'contract_id': contract_id,
        'source_contract_ids': [contract_id] if contract_id else [],
        'source_agenda_ids': list(contract.get('source_agenda_ids') or []),
        'source_adjudication_ids': list(contract.get('source_adjudication_ids') or []),
        'signature': contract.get('signature') or contract.get('contract_signature'),
        'selected_world': contract.get('selected_world'),
        'selected_probe': contract.get('selected_probe'),
        'required_evidence_gates': _required_gates(contract),
        'satisfied_evidence_gates': list(contract.get('satisfied_evidence_gates') or []),
        'failed_evidence_gates': list(contract.get('failed_evidence_gates') or []),
        'evidence_ids': list(contract.get('evidence_ids') or []),
        'sibling_evidence_lineage': list(contract.get('sibling_evidence_lineage') or []),
        'blockers': list(contract.get('blockers') or []),
        'lifecycle_state': _state_from_status(contract.get('status')),
        'lineage': [source],
    }


def _hypothesis_from_candidate(candidate: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    contract_id = candidate.get('contract_id')
    return {
        'hypothesis_id': f'hypothesis:{contract_id}' if contract_id else None,
        'contract_id': contract_id,
        'source_contract_ids': [contract_id] if contract_id else [],
        'source_agenda_ids': [record.get('agenda_id')] if record.get('agenda_id') else [],
        'source_adjudication_ids': [],
        'signature': candidate.get('contract_signature'),
        'selected_world': candidate.get('selected_world'),
        'selected_probe': candidate.get('selected_probe'),
        'required_evidence_gates': _required_gates(candidate),
        'satisfied_evidence_gates': [],
        'failed_evidence_gates': [],
        'evidence_ids': [],
        'sibling_evidence_lineage': [],
        'blockers': [],
        'lifecycle_state': 'proposed',
        'lineage': ['agenda_candidate'],
    }


def _hypothesis_from_adjudication(body: dict[str, Any]) -> dict[str, Any]:
    contract_id = body.get('contract_id')
    compact = dict(body.get('compact_evidence_contract') or {})
    return {
        'hypothesis_id': f'hypothesis:{contract_id}' if contract_id else None,
        'contract_id': contract_id,
        'source_contract_ids': [contract_id] if contract_id else [],
        'source_agenda_ids': [],
        'source_adjudication_ids': [body.get('agenda_id') or body.get('lifecycle_id') or body.get('contract_id')],
        'signature': body.get('contract_signature'),
        'required_evidence_gates': _required_gates(body),
        'satisfied_evidence_gates': _gates_from_contract_map(compact, contract_id, 'satisfied_evidence_gates'),
        'failed_evidence_gates': _gates_from_contract_map(compact, contract_id, 'failed_evidence_gates'),
        'evidence_ids': [],
        'sibling_evidence_lineage': [],
        'blockers': [],
        'lifecycle_state': 'resolved' if body.get('selected_action') == 'resolve_contract' else 'waiting_evidence',
        'lineage': ['adjudication_message'],
    }


def _hypothesis_from_lifecycle(body: dict[str, Any]) -> dict[str, Any]:
    hypothesis_id = body.get('hypothesis_id')
    contract_id = body.get('contract_id') or str(hypothesis_id or '').replace('hypothesis:', '')
    return {
        'hypothesis_id': hypothesis_id,
        'contract_id': contract_id,
        'source_contract_ids': [contract_id] if contract_id else [],
        'source_agenda_ids': [],
        'source_adjudication_ids': [],
        'signature': None,
        'required_evidence_gates': list(body.get('required_evidence_gates') or []),
        'satisfied_evidence_gates': [],
        'failed_evidence_gates': [],
        'evidence_ids': [],
        'sibling_evidence_lineage': list(body.get('sibling_evidence_lineage') or []),
        'blockers': [],
        'lifecycle_state': body.get('lifecycle_state') if body.get('lifecycle_state') in LIFECYCLE_STATES else 'waiting_evidence',
        'lineage': ['prior_lifecycle_message'],
    }


def _upsert_hypothesis(states: dict[str, dict[str, Any]], incoming: dict[str, Any]):
    hypothesis_id = incoming.get('hypothesis_id')
    if not hypothesis_id:
        return
    current = states.setdefault(str(hypothesis_id), {})
    current.update({
        'hypothesis_id': str(hypothesis_id),
        'contract_id': incoming.get('contract_id') or current.get('contract_id'),
        'signature': incoming.get('signature') or current.get('signature'),
        'selected_world': incoming.get('selected_world') or current.get('selected_world'),
        'selected_probe': incoming.get('selected_probe') or current.get('selected_probe'),
        'required_evidence_gates': _unique_strings(
            list(current.get('required_evidence_gates') or [])
            + list(incoming.get('required_evidence_gates') or [])
        ),
        'satisfied_evidence_gates': _unique_strings(
            list(current.get('satisfied_evidence_gates') or [])
            + list(incoming.get('satisfied_evidence_gates') or [])
        ),
        'failed_evidence_gates': _unique_strings(
            list(current.get('failed_evidence_gates') or [])
            + list(incoming.get('failed_evidence_gates') or [])
        ),
        'evidence_ids': _unique_strings(
            list(current.get('evidence_ids') or [])
            + list(incoming.get('evidence_ids') or [])
        ),
        'source_agenda_ids': _unique_strings(
            list(current.get('source_agenda_ids') or [])
            + list(incoming.get('source_agenda_ids') or [])
        ),
        'source_contract_ids': _unique_strings(
            list(current.get('source_contract_ids') or [])
            + list(incoming.get('source_contract_ids') or [])
        ),
        'source_adjudication_ids': _unique_strings(
            list(current.get('source_adjudication_ids') or [])
            + list(incoming.get('source_adjudication_ids') or [])
        ),
        'sibling_evidence_lineage': list(current.get('sibling_evidence_lineage') or [])
        + list(incoming.get('sibling_evidence_lineage') or []),
        'blockers': list(current.get('blockers') or []) + list(incoming.get('blockers') or []),
        'lineage': _unique_strings(
            list(current.get('lineage') or [])
            + list(incoming.get('lineage') or [])
        ),
        'lifecycle_state': _dominant_state(
            current.get('lifecycle_state'),
            incoming.get('lifecycle_state'),
        ),
    })


def _extract_lifecycle_evidence(
    messages: list[dict[str, Any]],
    hypotheses: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    known_contract_ids = {item.get('contract_id') for item in hypotheses}
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
        status = _evidence_status(message)
        gate = _evidence_gate(message)
        if sender == 'ai_different' and body.get('response_kind') == 'hypothesis_lifecycle':
            action = body.get('selected_action')
            if action == 'mark_resolved':
                status = 'satisfied'
            elif action == 'retire_blocked_hypothesis':
                status = 'retired'
            elif action == 'refine_next_hypothesis':
                status = 'refine_next'
        evidence_id = str(
            body.get('evidence_id')
            or evidence.get('evidence_id')
            or module_chat_message_id(message)
        )
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
            'third_party_checkpoint_used': bool(
                body.get('third_party_checkpoint_used')
                or evidence.get('third_party_checkpoint_used')
            ),
        }
        if item['contract_id'] in known_contract_ids or item['hypothesis_id'] or sender in {'language_model_2', 'funfun', 'code_module'}:
            items.append(item)
    return items


def _apply_evidence_to_hypotheses(
    hypotheses: list[dict[str, Any]],
    evidence_items: list[dict[str, Any]],
):
    by_hypothesis = {item.get('hypothesis_id'): item for item in hypotheses}
    by_contract = {item.get('contract_id'): item for item in hypotheses}
    for item in evidence_items:
        hypothesis = by_hypothesis.get(item.get('hypothesis_id')) or by_contract.get(item.get('contract_id'))
        if not hypothesis:
            continue
        gate = item.get('evidence_gate')
        hypothesis.setdefault('evidence_ids', []).append(item['evidence_id'])
        hypothesis.setdefault('sibling_evidence_lineage', []).append({
            'evidence_id': item['evidence_id'],
            'sender': item.get('sender'),
            'topic': item.get('topic'),
            'evidence_gate': gate,
            'status': item.get('status'),
        })
        if item.get('label_leaks') or item.get('third_party_checkpoint_used'):
            hypothesis.setdefault('failed_evidence_gates', []).append('safety_label_project_boundary')
            hypothesis.setdefault('blockers', []).append(item)
            hypothesis['lifecycle_state'] = 'blocked'
            continue
        if item.get('status') == 'failed':
            hypothesis.setdefault('failed_evidence_gates', []).append(gate)
            hypothesis.setdefault('blockers', []).append(item)
            hypothesis['lifecycle_state'] = 'blocked'
        elif item.get('status') == 'retired':
            hypothesis['lifecycle_state'] = 'retired'
        elif item.get('status') == 'refine_next':
            hypothesis['lifecycle_state'] = 'refine_next'
        elif item.get('status') == 'satisfied':
            hypothesis.setdefault('satisfied_evidence_gates', []).append(gate)
    for hypothesis in hypotheses:
        _finalize_hypothesis(hypothesis)


def _finalize_hypothesis(hypothesis: dict[str, Any]) -> dict[str, Any]:
    hypothesis['required_evidence_gates'] = _unique_strings(
        list(hypothesis.get('required_evidence_gates') or [])
        or ['math_proof', 'code_proof', 'language_epoch_plan']
    )
    hypothesis['satisfied_evidence_gates'] = sorted(set(hypothesis.get('satisfied_evidence_gates') or []))
    hypothesis['failed_evidence_gates'] = sorted(set(hypothesis.get('failed_evidence_gates') or []))
    if hypothesis.get('lifecycle_state') in {'retired', 'refine_next'}:
        pass
    elif hypothesis.get('failed_evidence_gates') or hypothesis.get('blockers'):
        hypothesis['lifecycle_state'] = 'blocked'
    elif {'math_proof', 'code_proof', 'language_epoch_plan'} <= set(hypothesis.get('satisfied_evidence_gates') or []):
        hypothesis['lifecycle_state'] = 'resolved'
    elif hypothesis.get('contract_id'):
        hypothesis['lifecycle_state'] = 'waiting_evidence'
    else:
        hypothesis['lifecycle_state'] = hypothesis.get('lifecycle_state') or 'proposed'
    hypothesis['hypothesis_hash'] = stable_digest({
        'hypothesis_id': hypothesis.get('hypothesis_id'),
        'contract_id': hypothesis.get('contract_id'),
        'gates': hypothesis.get('satisfied_evidence_gates'),
        'failed': hypothesis.get('failed_evidence_gates'),
        'state': hypothesis.get('lifecycle_state'),
    })
    return hypothesis


def _select_lifecycle_action(
    *,
    hypotheses: list[dict[str, Any]],
    evidence_items: list[dict[str, Any]],
    refinement: dict[str, Any] | None,
    resolved_ids: list[str],
    retired_ids: list[str],
    refined_ids: list[str],
    project_owned_boundary: dict[str, Any],
) -> dict[str, Any]:
    safety_fail = _first_safety_failure(hypotheses, evidence_items, project_owned_boundary)
    if safety_fail:
        recipient = 'code_module' if safety_fail.get('sender') == 'code_module' else 'funfun'
        if safety_fail.get('reason') == 'third-party checkpoint boundary failed':
            recipient = 'code_module'
        return _repair_action(
            'request_missing_code_evidence' if recipient == 'code_module' else 'request_missing_math_evidence',
            recipient,
            safety_fail,
            'failed safety, label, or project-owned boundary',
        )
    for hypothesis in hypotheses:
        if hypothesis.get('lifecycle_state') == 'blocked' and hypothesis.get('hypothesis_id') not in set(retired_ids):
            return _hypothesis_action(
                'retire_blocked_hypothesis',
                'broadcast',
                hypothesis,
                'blocked evidence retires this hypothesis before further refinement',
                retirement_or_refinement_reason='blocked evidence or boundary failure',
            )
    for hypothesis in hypotheses:
        if hypothesis.get('lifecycle_state') == 'waiting_evidence' and 'math_proof' not in set(hypothesis.get('satisfied_evidence_gates') or []):
            return _missing_gate_action(hypothesis, 'request_missing_math_evidence', 'funfun', 'math_proof')
    for hypothesis in hypotheses:
        if hypothesis.get('lifecycle_state') == 'waiting_evidence' and 'code_proof' not in set(hypothesis.get('satisfied_evidence_gates') or []):
            return _missing_gate_action(hypothesis, 'request_missing_code_evidence', 'code_module', 'code_proof')
    for hypothesis in hypotheses:
        if hypothesis.get('lifecycle_state') == 'waiting_evidence' and 'language_epoch_plan' not in set(hypothesis.get('satisfied_evidence_gates') or []):
            return _missing_gate_action(hypothesis, 'request_language_protocol_clarification', 'language_model_2', 'language_epoch_plan')
    for hypothesis in hypotheses:
        if hypothesis.get('lifecycle_state') == 'resolved' and hypothesis.get('hypothesis_id') not in set(resolved_ids):
            return _hypothesis_action(
                'mark_resolved',
                'broadcast',
                hypothesis,
                'all required evidence gates are satisfied',
                accepted_evidence=list(hypothesis.get('evidence_ids') or []),
            )
    for hypothesis in hypotheses:
        if (
            hypothesis.get('lifecycle_state') == 'resolved'
            and hypothesis.get('hypothesis_id') in set(resolved_ids)
            and hypothesis.get('hypothesis_id') not in set(refined_ids)
            and refinement is not None
        ):
            return _hypothesis_action(
                'refine_next_hypothesis',
                'broadcast',
                hypothesis,
                'resolved hypothesis can seed one next refinement',
                retirement_or_refinement_reason='resolved evidence supports one refinement',
            )
    return _noop_action('no new hypothesis lifecycle action')


def _candidate_refinement(
    source_ledger: dict[str, Any],
    agenda_ledger: dict[str, Any],
    hypotheses: list[dict[str, Any]],
    runtime_memory_data: dict[str, Any],
    source_hash: str,
) -> dict[str, Any] | None:
    selected = dict(source_ledger.get('selected_experiment') or {})
    if not selected:
        latest = dict(agenda_ledger.get('latest') or {})
        selected = {
            'world': latest.get('selected_world'),
            'probe': latest.get('selected_probe'),
        }
    if not selected.get('world') or not selected.get('probe') or selected.get('runs_final', False):
        return None
    resolved = [item for item in hypotheses if item.get('lifecycle_state') == 'resolved']
    readiness = dict(runtime_memory_data.get('discovery_readiness') or {})
    refinement_id = 'refinement_' + stable_digest({
        'source_hash': source_hash,
        'selected': selected,
        'resolved': [item.get('hypothesis_id') for item in resolved],
        'readiness_score': readiness.get('readiness_score'),
    })[:16]
    return {
        'refinement_id': refinement_id,
        'selected_world': selected.get('world'),
        'selected_probe': selected.get('probe'),
        'expected_transfer_signal': (
            source_ledger.get('expected_transfer_signal')
            or selected.get('expected_transfer_signal')
        ),
        'source_hypothesis_ids': [item.get('hypothesis_id') for item in resolved],
        'label_clean': not label_leak_terms(selected),
        'third_party_checkpoint_used': False,
    }


def _repair_action(
    action: str,
    recipient: str,
    blocker: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    return {
        'selected_action': action,
        'chosen_recipient': validate_participant(recipient),
        'hypothesis_id': blocker.get('hypothesis_id'),
        'lifecycle_state': 'blocked' if 'repair' not in action else 'waiting_evidence',
        'reason': reason,
        'missing_gate': blocker.get('evidence_gate'),
        'accepted_evidence': None,
        'repair_request': {
            'blocker_id': blocker.get('evidence_id') or blocker.get('message_id'),
            'sender': blocker.get('sender'),
            'evidence_gate': blocker.get('evidence_gate'),
            'status': blocker.get('status'),
            'summary': blocker.get('summary') or blocker.get('reason'),
        },
        'retirement_or_refinement_reason': None,
        'label_leaks': label_leak_terms(blocker),
    }


def _missing_gate_action(
    hypothesis: dict[str, Any],
    action: str,
    recipient: str,
    gate: str,
) -> dict[str, Any]:
    return {
        'selected_action': action,
        'chosen_recipient': recipient,
        'hypothesis_id': hypothesis.get('hypothesis_id'),
        'lifecycle_state': hypothesis.get('lifecycle_state'),
        'reason': f'missing {gate} evidence',
        'missing_gate': gate,
        'accepted_evidence': None,
        'repair_request': {
            'hypothesis_id': hypothesis.get('hypothesis_id'),
            'contract_id': hypothesis.get('contract_id'),
            'evidence_gate': gate,
            'status': 'missing',
        },
        'retirement_or_refinement_reason': None,
        'label_leaks': label_leak_terms(hypothesis),
    }


def _hypothesis_action(
    action: str,
    recipient: str,
    hypothesis: dict[str, Any],
    reason: str,
    *,
    accepted_evidence: list[str] | None = None,
    retirement_or_refinement_reason: str | None = None,
) -> dict[str, Any]:
    state = {
        'retire_blocked_hypothesis': 'retired',
        'mark_resolved': 'resolved',
        'refine_next_hypothesis': 'refine_next',
    }.get(action, hypothesis.get('lifecycle_state'))
    return {
        'selected_action': action,
        'chosen_recipient': recipient,
        'hypothesis_id': hypothesis.get('hypothesis_id'),
        'lifecycle_state': state,
        'reason': reason,
        'missing_gate': None,
        'accepted_evidence': list(accepted_evidence or []),
        'repair_request': None,
        'retirement_or_refinement_reason': retirement_or_refinement_reason,
        'label_leaks': label_leak_terms(hypothesis),
    }


def _noop_action(reason: str) -> dict[str, Any]:
    return {
        'selected_action': 'summarize_noop',
        'chosen_recipient': None,
        'hypothesis_id': None,
        'lifecycle_state': None,
        'reason': reason,
        'missing_gate': None,
        'accepted_evidence': None,
        'repair_request': None,
        'retirement_or_refinement_reason': None,
        'label_leaks': [],
    }


def _first_safety_failure(
    hypotheses: list[dict[str, Any]],
    evidence_items: list[dict[str, Any]],
    project_owned_boundary: dict[str, Any],
) -> dict[str, Any] | None:
    if project_owned_boundary.get('third_party_checkpoint_used'):
        return {
            'sender': 'code_module',
            'reason': 'third-party checkpoint boundary failed',
            'evidence_gate': 'project_owned_checkpoint_boundary_explicit',
            'status': 'failed',
        }
    for item in evidence_items:
        if item.get('label_leaks') or item.get('third_party_checkpoint_used'):
            return item
    for hypothesis in hypotheses:
        if 'safety_label_project_boundary' in set(hypothesis.get('failed_evidence_gates') or []):
            return (hypothesis.get('blockers') or [hypothesis])[-1]
    return None


def _evidence_gate(message: dict[str, Any]) -> str:
    body = dict(message.get('body') or {})
    evidence = dict(message.get('evidence') or {})
    explicit = (
        body.get('evidence_gate')
        or evidence.get('evidence_gate')
        or body.get('gate')
        or evidence.get('gate')
        or body.get('missing_evidence_gate')
        or body.get('missing_gate')
    )
    if explicit:
        return str(explicit)
    sender = message.get('sender')
    topic = str(message.get('topic') or '').lower()
    if sender == 'funfun':
        return 'math_proof'
    if sender == 'code_module':
        return 'code_proof'
    if sender == 'language_model_2':
        return 'language_epoch_plan' if 'epoch' in topic or 'protocol' in topic or 'lexicon' in topic else 'language_protocol'
    return 'advisory'


def _evidence_status(message: dict[str, Any]) -> str:
    body = dict(message.get('body') or {})
    evidence = dict(message.get('evidence') or {})
    explicit = str(body.get('status') or evidence.get('status') or '').lower()
    text = json.dumps({
        'body': body,
        'evidence': evidence,
        'tags': message.get('tags'),
        'topic': message.get('topic'),
    }, sort_keys=True).lower()
    if explicit in {'satisfied', 'resolved', 'passed', 'confirmed'}:
        return 'satisfied'
    if explicit in {'failed', 'blocked', 'contradicted', 'contradiction'}:
        return 'failed'
    if explicit in {'retired', 'retire'}:
        return 'retired'
    if explicit in {'refine_next', 'refined'}:
        return 'refine_next'
    if explicit in {'missing', 'needs_clarification', 'clarification_needed'}:
        return 'missing'
    if 'missing' in text or 'clarification_needed' in text:
        return 'missing'
    if any(token in text for token in ('failed', 'blocked', 'contradict')):
        return 'failed'
    if any(token in text for token in ('satisfied', 'resolved', 'passed', 'confirmed')):
        return 'satisfied'
    return 'advisory'


def _required_gates(contract: dict[str, Any]) -> list[str]:
    gates = set(contract.get('required_evidence_gates') or [])
    gates.update({'math_proof', 'code_proof', 'language_epoch_plan'})
    return sorted(gates)


def _state_from_status(status: Any) -> str:
    text = str(status or '').lower()
    if text in {'resolved', 'complete', 'satisfied'}:
        return 'resolved'
    if text in {'blocked', 'failed', 'contradicted'}:
        return 'blocked'
    if text == 'retired':
        return 'retired'
    if text == 'refine_next':
        return 'refine_next'
    if text in {'open', 'waiting_evidence'}:
        return 'waiting_evidence'
    return 'proposed'


def _dominant_state(current: Any, incoming: Any) -> str:
    order = {
        'retired': 6,
        'refine_next': 5,
        'blocked': 4,
        'resolved': 3,
        'waiting_evidence': 2,
        'proposed': 1,
    }
    current_state = str(current or 'proposed')
    incoming_state = str(incoming or 'proposed')
    return current_state if order.get(current_state, 0) >= order.get(incoming_state, 0) else incoming_state


def _source_hash(
    evaluator: dict[str, Any],
    outcome: dict[str, Any],
    contracts: dict[str, Any],
    adjudicator: dict[str, Any],
    agenda: dict[str, Any],
) -> str:
    return stable_digest({
        'evaluator': evaluator.get('ledger_hash') or evaluator.get('ledger_id'),
        'outcome': outcome.get('ledger_hash') or outcome.get('ledger_id'),
        'contract': contracts.get('ledger_hash'),
        'adjudicator': adjudicator.get('ledger_hash'),
        'agenda': agenda.get('ledger_hash'),
    })


def _gates_from_contract_map(
    compact: dict[str, Any] | None,
    contract_id: str | None,
    field: str,
) -> list[str]:
    value = dict(compact or {}).get(field) or {}
    if isinstance(value, dict):
        return list(value.get(str(contract_id)) or [])
    return []


def _find_hypothesis(
    hypotheses: list[dict[str, Any]],
    hypothesis_id: str | None,
) -> dict[str, Any] | None:
    for hypothesis in hypotheses:
        if hypothesis.get('hypothesis_id') == hypothesis_id:
            return hypothesis
    return None


def _state_counts(hypotheses: list[dict[str, Any]]) -> dict[str, int]:
    counts = {state: 0 for state in sorted(LIFECYCLE_STATES)}
    for hypothesis in hypotheses:
        state = hypothesis.get('lifecycle_state') or 'proposed'
        counts[state] = counts.get(state, 0) + 1
    return counts


def _evidence_gate_map(hypotheses: list[dict[str, Any]]) -> dict[str, dict[str, list[str]]]:
    output = {}
    for hypothesis in hypotheses:
        output[str(hypothesis.get('hypothesis_id'))] = {
            'required': list(hypothesis.get('required_evidence_gates') or []),
            'satisfied': list(hypothesis.get('satisfied_evidence_gates') or []),
            'failed': list(hypothesis.get('failed_evidence_gates') or []),
        }
    return output


def _sibling_evidence_lineage(hypotheses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lineage = []
    for hypothesis in hypotheses:
        for item in list(hypothesis.get('sibling_evidence_lineage') or []):
            if isinstance(item, dict):
                row = dict(item)
                row['hypothesis_id'] = hypothesis.get('hypothesis_id')
                lineage.append(row)
    return lineage


def _collect_field(hypotheses: list[dict[str, Any]], field: str) -> list[Any]:
    values = []
    for hypothesis in hypotheses:
        values.extend(list(hypothesis.get(field) or []))
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
