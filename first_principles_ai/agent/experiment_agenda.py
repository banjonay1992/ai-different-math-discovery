"""Experiment agenda scheduler for AI Different.

This is the planning layer above cross-module adjudication. It consumes only
plain JSON module-chat messages and local ledgers, then schedules the next safe
experiment contract, repair request, or no-op without importing sibling modules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .cross_module_adjudicator import ADJUDICATOR_LEDGER_KIND
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


AGENDA_LEDGER_KIND = 'ai_different.experiment_agenda_ledger'


def empty_experiment_agenda_ledger() -> dict[str, Any]:
    return {
        'schema_version': 1,
        'ledger_kind': AGENDA_LEDGER_KIND,
        'processed_message_ids': [],
        'processed_source_hashes': [],
        'scheduled_candidate_ids': [],
        'hypotheses': [],
        'agenda_records': [],
        'outgoing_response_ids': [],
        'latest': {},
    }


def load_experiment_agenda_ledger(path: str | Path | None) -> dict[str, Any]:
    if not path or not Path(path).exists():
        return empty_experiment_agenda_ledger()
    with Path(path).open('r', encoding='utf-8') as handle:
        ledger = json.load(handle)
    return validate_experiment_agenda_ledger(ledger)


def write_experiment_agenda_ledger(path: str | Path, ledger: dict[str, Any]) -> Path:
    validated = validate_experiment_agenda_ledger(ledger)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(validated, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    return output


def validate_experiment_agenda_ledger(ledger: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(ledger, dict):
        raise ValueError('experiment agenda ledger must be a JSON object')
    if ledger.get('ledger_kind') != AGENDA_LEDGER_KIND:
        raise ValueError('experiment agenda ledger has wrong ledger_kind')
    for field in (
        'processed_message_ids',
        'processed_source_hashes',
        'scheduled_candidate_ids',
        'outgoing_response_ids',
    ):
        values = ledger.get(field)
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            raise ValueError(f'{field} must be strings')
    for field in ('hypotheses', 'agenda_records'):
        values = ledger.get(field)
        if not isinstance(values, list) or not all(isinstance(item, dict) for item in values):
            raise ValueError(f'{field} must be objects')
    latest = ledger.get('latest', {})
    if not isinstance(latest, dict):
        raise ValueError('experiment agenda ledger latest must be an object')
    validated = {
        'schema_version': int(ledger.get('schema_version', 1) or 1),
        'ledger_kind': AGENDA_LEDGER_KIND,
        'processed_message_ids': _unique_strings(ledger['processed_message_ids']),
        'processed_source_hashes': _unique_strings(ledger['processed_source_hashes']),
        'scheduled_candidate_ids': _unique_strings(ledger['scheduled_candidate_ids']),
        'hypotheses': list(ledger['hypotheses']),
        'agenda_records': list(ledger['agenda_records']),
        'outgoing_response_ids': _unique_strings(ledger['outgoing_response_ids']),
        'latest': dict(latest),
    }
    if ledger.get('ledger_hash'):
        validated['ledger_hash'] = str(ledger['ledger_hash'])
    if ledger.get('artifact_path'):
        validated['artifact_path'] = str(ledger['artifact_path'])
    return validated


def read_agenda_transcript(path: str | Path | None) -> dict[str, Any]:
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
        raise ValueError('plain agenda input must be a JSON object')
    return value


def build_experiment_agenda(
    *,
    transcript_messages: list[dict[str, Any]],
    agenda_ledger: dict[str, Any],
    evaluator_ledger: dict[str, Any] | None = None,
    outcome_ledger: dict[str, Any] | None = None,
    contract_ledger: dict[str, Any] | None = None,
    adjudicator_ledger: dict[str, Any] | None = None,
    runtime_memory_data: dict[str, Any] | None = None,
    runtime_memory_hash_state: dict[str, Any],
    project_owned_boundary: dict[str, Any],
    artifact_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    ledger = validate_experiment_agenda_ledger(agenda_ledger)
    evaluator = _valid_evaluator_or_empty(evaluator_ledger or {})
    outcome = _valid_outcome_or_empty(outcome_ledger or {})
    contracts = _valid_contract_or_empty(contract_ledger or {})
    adjudicator = _valid_adjudicator_or_empty(adjudicator_ledger or {})
    source_hash = _source_hash(evaluator, outcome, contracts, adjudicator)
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
        transcript_messages,
    )
    evidence = _extract_agenda_evidence(new_messages, hypotheses)
    _apply_evidence_to_hypotheses(hypotheses, evidence)
    candidate = _candidate_next_contract(
        evaluator or outcome,
        hypotheses,
        runtime_memory_data or {},
        source_hash,
    )
    if not new_messages and (not source_is_new or hypotheses):
        selected = _noop_action('no new agenda evidence or source ledger state')
    else:
        selected = _select_agenda_action(
            hypotheses=hypotheses,
            evidence_items=evidence,
            candidate=candidate,
            scheduled_candidate_ids=ledger['scheduled_candidate_ids'],
            project_owned_boundary=project_owned_boundary,
        )
    agenda_id = 'agenda_' + stable_digest({
        'source_hash': source_hash,
        'new_message_ids': [module_chat_message_id(message) for message in new_messages],
        'selected_action': selected['selected_action'],
        'candidate_id': (candidate or {}).get('contract_id'),
    })[:16]
    selected['agenda_id'] = agenda_id
    message = export_agenda_message(
        selected,
        hypotheses=hypotheses,
        evidence_items=evidence,
        candidate=candidate,
        runtime_memory_hash_state=runtime_memory_hash_state,
        project_owned_boundary=project_owned_boundary,
        source_hashes={
            'agenda_source_hash': source_hash,
            'evaluator_ledger_hash': evaluator.get('ledger_hash'),
            'outcome_ledger_hash': outcome.get('ledger_hash'),
            'contract_ledger_hash': contracts.get('ledger_hash'),
            'adjudicator_ledger_hash': adjudicator.get('ledger_hash'),
        },
    )
    new_ids = [module_chat_message_id(message) for message in new_messages]
    outgoing_id = module_chat_message_id(message) if message else None
    if selected['selected_action'] == 'emit_next_experiment_contract' and candidate:
        ledger['scheduled_candidate_ids'] = _unique_strings(
            list(ledger['scheduled_candidate_ids']) + [str(candidate.get('contract_id'))]
        )
    if new_messages:
        ledger['processed_message_ids'] = _unique_strings(
            list(ledger['processed_message_ids']) + new_ids
        )
    if source_is_new and selected['selected_action'] in {
        'emit_next_experiment_contract',
        'summarize_noop',
    }:
        ledger['processed_source_hashes'] = _unique_strings(
            list(ledger['processed_source_hashes']) + [source_hash]
        )
    if outgoing_id:
        ledger['outgoing_response_ids'] = _unique_strings(
            list(ledger['outgoing_response_ids']) + [outgoing_id]
        )
    latest = {
        'agenda_id': agenda_id,
        'new_message_count': len(new_messages),
        'skipped_message_count': len(skipped_messages),
        'source_is_new': source_is_new,
        'open_gate_count': _gate_count(hypotheses, 'open'),
        'resolved_gate_count': _gate_count(hypotheses, 'satisfied'),
        'blocked_gate_count': _gate_count(hypotheses, 'failed'),
        'selected_action': selected['selected_action'],
        'chosen_recipient': message.get('recipient') if message else None,
        'outbox_count': 1 if message else 0,
        'label_leaks': selected.get('label_leaks', []),
        'runtime_memory_hash_state': dict(runtime_memory_hash_state),
        'runtime_memory_mutated': not bool(runtime_memory_hash_state.get('unchanged', True)),
    }
    record = {
        'agenda_id': agenda_id,
        'agenda_hash': stable_digest({
            'agenda_id': agenda_id,
            'selected': selected,
            'candidate': candidate,
        }),
        'processed_message_ids': new_ids,
        'source_hash': source_hash,
        'source_contract_ids': [item.get('contract_id') for item in hypotheses],
        'resolved_evidence_gates': _gates_by_status(hypotheses, 'satisfied'),
        'blocked_evidence_gates': _gates_by_status(hypotheses, 'failed'),
        'hypothesis_lineage': _hypothesis_lineage(hypotheses),
        'candidate_next_experiment': candidate,
        'required_evidence_gates': _required_agenda_gates(candidate),
        'safety_project_owned_boundary': dict(project_owned_boundary),
        'selected_action': selected['selected_action'],
        'chosen_recipient': latest['chosen_recipient'],
        'outgoing_response_id': outgoing_id,
        'source_ledger_hashes': {
            'evaluator_ledger_hash': evaluator.get('ledger_hash'),
            'outcome_ledger_hash': outcome.get('ledger_hash'),
            'contract_ledger_hash': contracts.get('ledger_hash'),
            'adjudicator_ledger_hash': adjudicator.get('ledger_hash'),
        },
    }
    ledger['hypotheses'] = hypotheses
    if new_messages or source_is_new or message is not None:
        ledger['agenda_records'] = list(ledger['agenda_records']) + [record]
    ledger['latest'] = latest
    if artifact_path is not None:
        ledger['artifact_path'] = str(Path(artifact_path))
    ledger['ledger_hash'] = stable_digest({
        key: value for key, value in ledger.items()
        if key != 'ledger_hash'
    })
    return ledger, message


def export_agenda_message(
    selected: dict[str, Any],
    *,
    hypotheses: list[dict[str, Any]],
    evidence_items: list[dict[str, Any]],
    candidate: dict[str, Any] | None,
    runtime_memory_hash_state: dict[str, Any],
    project_owned_boundary: dict[str, Any],
    source_hashes: dict[str, Any],
    topic: str = 'ai_different.experiment_agenda',
) -> dict[str, Any] | None:
    if selected['selected_action'] == 'summarize_noop':
        return None
    recipient = validate_participant(selected.get('chosen_recipient') or 'orchestrator')
    leak_terms = label_leak_terms({
        'selected': selected,
        'hypotheses': hypotheses,
        'candidate': candidate,
        'evidence': evidence_items,
    })
    body = {
        'module': 'AI Different',
        'response_kind': 'experiment_agenda',
        'agenda_id': selected.get('agenda_id'),
        'selected_action': selected['selected_action'],
        'reason': selected.get('reason'),
        'contract_id': selected.get('contract_id'),
        'missing_gate': selected.get('missing_gate'),
        'repair_request': selected.get('repair_request'),
        'compact_next_experiment_contract': candidate
        if selected['selected_action'] == 'emit_next_experiment_contract'
        else None,
        'hypothesis_lineage': _hypothesis_lineage(hypotheses),
        'required_evidence_gates': _required_agenda_gates(candidate),
        'resolved_evidence_gates': _gates_by_status(hypotheses, 'satisfied'),
        'blocked_evidence_gates': _gates_by_status(hypotheses, 'failed'),
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
            'agenda_id': body['agenda_id'],
            'selected_action': body['selected_action'],
            'contract_id': body['contract_id'],
            'candidate_contract_id': (candidate or {}).get('contract_id'),
            'missing_gate': body['missing_gate'],
            'required_evidence_gates': body['required_evidence_gates'],
            'runtime_memory_hash_state': body['runtime_memory_hash_state'],
            'runtime_memory_mutated': body['runtime_memory_mutated'],
            'label_clean': body['label_clean'],
            'no_sibling_imports': True,
        },
        tags=[
            'ai_different',
            'experiment_agenda',
            body['selected_action'],
            'label_clean' if body['label_clean'] else 'label_review_needed',
        ],
    )


def write_agenda_outbox_jsonl(
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
        return {
            'ledger_kind': ADJUDICATOR_LEDGER_KIND,
            'contract_states': [],
            'adjudication_records': [],
            'ledger_hash': None,
        }
    if ledger.get('ledger_kind') != ADJUDICATOR_LEDGER_KIND:
        raise ValueError('adjudicator ledger has wrong ledger_kind')
    return dict(ledger)


def _merge_hypotheses(
    existing: list[dict[str, Any]],
    contract_ledger: dict[str, Any],
    adjudicator_ledger: dict[str, Any],
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {
        str(item.get('contract_id')): dict(item)
        for item in existing
        if item.get('contract_id')
    }
    for contract in list(contract_ledger.get('contracts') or []):
        _upsert_hypothesis(states, contract, source='contract_ledger')
    for contract in list(adjudicator_ledger.get('contract_states') or []):
        _upsert_hypothesis(states, contract, source='adjudicator_ledger')
    for message in messages:
        body = dict(message.get('body') or {})
        if message.get('sender') != 'ai_different':
            continue
        if body.get('response_kind') == 'experiment_contract':
            _upsert_hypothesis(states, {
                'contract_id': body.get('contract_id'),
                'signature': body.get('contract_signature'),
                'selected_world': body.get('selected_world'),
                'selected_probe': body.get('selected_probe'),
                'required_evidence_gates': body.get('required_evidence_gates') or [],
                'status': 'open',
            }, source='transcript_contract')
        elif body.get('response_kind') == 'cross_module_adjudication':
            contract_id = body.get('contract_id')
            if not contract_id:
                continue
            _upsert_hypothesis(states, {
                'contract_id': contract_id,
                'signature': body.get('contract_signature'),
                'status': _status_from_adjudication(body),
                'satisfied_evidence_gates': _gates_from_contract_map(
                    body.get('compact_evidence_contract'),
                    contract_id,
                    'satisfied_evidence_gates',
                ),
                'failed_evidence_gates': _gates_from_contract_map(
                    body.get('compact_evidence_contract'),
                    contract_id,
                    'failed_evidence_gates',
                ),
            }, source='adjudication_message')
    return list(states.values())


def _upsert_hypothesis(
    states: dict[str, dict[str, Any]],
    contract: dict[str, Any],
    *,
    source: str,
):
    contract_id = contract.get('contract_id')
    if not contract_id:
        return
    current = states.setdefault(str(contract_id), {})
    current.update({
        'hypothesis_id': current.get('hypothesis_id') or f'hypothesis:{contract_id}',
        'contract_id': str(contract_id),
        'signature': contract.get('signature') or contract.get('contract_signature') or current.get('signature'),
        'selected_world': contract.get('selected_world') or current.get('selected_world'),
        'selected_probe': contract.get('selected_probe') or current.get('selected_probe'),
        'required_evidence_gates': _required_gates(contract),
        'status': contract.get('status') or current.get('status') or 'open',
        'satisfied_evidence_gates': _unique_strings(
            list(current.get('satisfied_evidence_gates') or [])
            + list(contract.get('satisfied_evidence_gates') or [])
        ),
        'failed_evidence_gates': _unique_strings(
            list(current.get('failed_evidence_gates') or [])
            + list(contract.get('failed_evidence_gates') or [])
        ),
        'evidence_ids': _unique_strings(
            list(current.get('evidence_ids') or [])
            + list(contract.get('evidence_ids') or [])
        ),
        'blockers': list(current.get('blockers') or []) + list(contract.get('blockers') or []),
        'lineage': _unique_strings(list(current.get('lineage') or []) + [source]),
    })


def _extract_agenda_evidence(
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
        gate = _evidence_gate(message)
        status = _evidence_status(message)
        evidence_id = str(
            body.get('evidence_id')
            or evidence.get('evidence_id')
            or module_chat_message_id(message)
        )
        if sender == 'ai_different' and body.get('response_kind') == 'experiment_contract':
            continue
        if sender == 'ai_different' and body.get('response_kind') == 'cross_module_adjudication':
            action = body.get('selected_action')
            if action == 'resolve_contract':
                status = 'satisfied'
            elif action in {'request_math_repair', 'request_code_repair'}:
                status = 'missing'
                gate = body.get('missing_gate') or gate
        item = {
            'evidence_id': evidence_id,
            'message_id': module_chat_message_id(message),
            'sender': sender,
            'topic': message.get('topic'),
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
        if item['contract_id'] in known_contract_ids or sender in {'language_model_2', 'funfun', 'code_module', 'ai_different'}:
            items.append(item)
    return items


def _apply_evidence_to_hypotheses(
    hypotheses: list[dict[str, Any]],
    evidence_items: list[dict[str, Any]],
):
    by_id = {item.get('contract_id'): item for item in hypotheses}
    for item in evidence_items:
        hypothesis = by_id.get(item.get('contract_id'))
        if not hypothesis:
            continue
        gate = item.get('evidence_gate')
        hypothesis.setdefault('evidence_ids', []).append(item['evidence_id'])
        if item.get('label_leaks') or item.get('third_party_checkpoint_used'):
            hypothesis.setdefault('failed_evidence_gates', []).append('safety_label_project_boundary')
            hypothesis.setdefault('blockers', []).append(item)
            hypothesis['status'] = 'blocked'
            continue
        if item.get('status') == 'failed':
            hypothesis.setdefault('failed_evidence_gates', []).append(gate)
            hypothesis.setdefault('blockers', []).append(item)
            hypothesis['status'] = 'blocked'
        elif item.get('status') == 'satisfied':
            hypothesis.setdefault('satisfied_evidence_gates', []).append(gate)
    for hypothesis in hypotheses:
        hypothesis['satisfied_evidence_gates'] = sorted(set(hypothesis.get('satisfied_evidence_gates') or []))
        hypothesis['failed_evidence_gates'] = sorted(set(hypothesis.get('failed_evidence_gates') or []))
        if hypothesis.get('status') == 'blocked':
            continue
        if {'math_proof', 'code_proof'} <= set(hypothesis.get('satisfied_evidence_gates') or []):
            hypothesis['status'] = 'resolved'
        else:
            hypothesis['status'] = 'open'


def _select_agenda_action(
    *,
    hypotheses: list[dict[str, Any]],
    evidence_items: list[dict[str, Any]],
    candidate: dict[str, Any] | None,
    scheduled_candidate_ids: list[str],
    project_owned_boundary: dict[str, Any],
) -> dict[str, Any]:
    safety_fail = _first_safety_failure(hypotheses, evidence_items, project_owned_boundary)
    if safety_fail:
        recipient = 'code_module' if safety_fail.get('sender') == 'code_module' else 'funfun'
        if safety_fail.get('reason') == 'third-party checkpoint boundary failed':
            recipient = 'code_module'
        return _repair_action(
            'request_code_repair' if recipient == 'code_module' else 'request_math_repair',
            recipient,
            safety_fail,
            'failed safety, label, or project-owned boundary',
        )
    for hypothesis in hypotheses:
        if hypothesis.get('status') == 'blocked':
            blocker = (hypothesis.get('blockers') or [{}])[-1]
            recipient = 'code_module' if blocker.get('sender') == 'code_module' else 'funfun'
            return _repair_action(
                'request_code_repair' if recipient == 'code_module' else 'request_math_repair',
                recipient,
                blocker,
                'contract evidence is blocked or contradictory',
            )
    for hypothesis in hypotheses:
        if hypothesis.get('status') == 'open' and 'math_proof' not in set(hypothesis.get('satisfied_evidence_gates') or []):
            return _gate_action(hypothesis, 'request_math_repair', 'funfun', 'math_proof')
    for hypothesis in hypotheses:
        if hypothesis.get('status') == 'open' and 'code_proof' not in set(hypothesis.get('satisfied_evidence_gates') or []):
            return _gate_action(hypothesis, 'request_code_repair', 'code_module', 'code_proof')
    language_missing = _language_clarification_item(evidence_items, candidate)
    if language_missing:
        return _repair_action(
            'language_clarification_needed',
            'language_model_2',
            language_missing,
            'language agenda/epoch evidence is missing or ambiguous',
        )
    if candidate and str(candidate.get('contract_id')) not in set(scheduled_candidate_ids):
        return {
            'selected_action': 'emit_next_experiment_contract',
            'reason': 'resolved evidence gates allow the next safe non-final experiment',
            'chosen_recipient': 'broadcast',
            'contract_id': candidate.get('contract_id'),
            'missing_gate': None,
            'repair_request': None,
            'label_leaks': label_leak_terms(candidate),
        }
    return _noop_action('no new safe agenda action')


def _candidate_next_contract(
    source_ledger: dict[str, Any],
    hypotheses: list[dict[str, Any]],
    runtime_memory_data: dict[str, Any],
    source_hash: str,
) -> dict[str, Any] | None:
    if not source_ledger or not source_ledger.get('label_clean', True):
        return None
    selected = dict(source_ledger.get('selected_experiment') or {})
    decision = dict(source_ledger.get('decision') or {})
    if decision and decision.get('decision_kind') != 'run_next_safe_experiment':
        return None
    if not selected or selected.get('runs_final', False):
        return None
    if not selected.get('world') or not selected.get('probe'):
        return None
    resolved = [
        item for item in hypotheses
        if item.get('status') == 'resolved'
    ]
    readiness = dict(runtime_memory_data.get('discovery_readiness') or {})
    candidate_id = 'agenda_contract_' + stable_digest({
        'source_hash': source_hash,
        'selected_experiment': selected,
        'resolved_contract_ids': [item.get('contract_id') for item in resolved],
        'readiness_score': readiness.get('readiness_score'),
    })[:16]
    return {
        'contract_id': candidate_id,
        'contract_signature': stable_digest({
            'contract_id': candidate_id,
            'source_hash': source_hash,
            'selected_experiment': selected,
        }),
        'selected_world': selected.get('world'),
        'selected_probe': selected.get('probe'),
        'selected_experiment': selected,
        'expected_transfer_signal': (
            source_ledger.get('expected_transfer_signal')
            or selected.get('expected_transfer_signal')
        ),
        'hypothesis_lineage': _hypothesis_lineage(resolved or hypotheses),
        'required_evidence_gates': [
            'math_proof',
            'code_proof',
            'language_epoch_plan',
            'label_clean_downstream_evidence',
            'runtime_memory_not_mutated',
            'project_owned_checkpoint_boundary_explicit',
        ],
        'target_recipient': 'broadcast',
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
        'reason': reason,
        'chosen_recipient': validate_participant(recipient),
        'contract_id': blocker.get('contract_id'),
        'missing_gate': blocker.get('evidence_gate'),
        'repair_request': {
            'blocker_id': blocker.get('evidence_id') or blocker.get('message_id'),
            'sender': blocker.get('sender'),
            'evidence_gate': blocker.get('evidence_gate'),
            'status': blocker.get('status'),
            'summary': blocker.get('summary') or blocker.get('reason'),
        },
        'label_leaks': label_leak_terms(blocker),
    }


def _gate_action(
    hypothesis: dict[str, Any],
    action: str,
    recipient: str,
    gate: str,
) -> dict[str, Any]:
    return {
        'selected_action': action,
        'reason': f'missing {gate} gate',
        'chosen_recipient': recipient,
        'contract_id': hypothesis.get('contract_id'),
        'missing_gate': gate,
        'repair_request': {
            'contract_id': hypothesis.get('contract_id'),
            'evidence_gate': gate,
            'status': 'missing',
        },
        'label_leaks': label_leak_terms(hypothesis),
    }


def _noop_action(reason: str) -> dict[str, Any]:
    return {
        'selected_action': 'summarize_noop',
        'reason': reason,
        'chosen_recipient': None,
        'contract_id': None,
        'missing_gate': None,
        'repair_request': None,
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


def _language_clarification_item(
    evidence_items: list[dict[str, Any]],
    candidate: dict[str, Any] | None,
) -> dict[str, Any] | None:
    for item in evidence_items:
        if item.get('sender') == 'language_model_2' and item.get('status') in {'missing', 'failed'}:
            item = dict(item)
            item['evidence_gate'] = item.get('evidence_gate') or 'language_epoch_plan'
            return item
    if candidate is None:
        return None
    selected = dict(candidate.get('selected_experiment') or {})
    if not selected.get('world') or not selected.get('probe'):
        return {
            'sender': 'language_model_2',
            'evidence_gate': 'language_epoch_plan',
            'status': 'missing',
            'summary': 'candidate experiment is missing world or probe',
        }
    return None


def _evidence_gate(message: dict[str, Any]) -> str:
    body = dict(message.get('body') or {})
    evidence = dict(message.get('evidence') or {})
    explicit = (
        body.get('evidence_gate')
        or evidence.get('evidence_gate')
        or body.get('gate')
        or evidence.get('gate')
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
        return 'language_epoch_plan' if 'epoch' in topic or 'agenda' in topic else 'language_clarification'
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
    if explicit in {'missing', 'needs_clarification', 'clarification_needed'}:
        return 'missing'
    if 'missing' in text or 'clarification_needed' in text:
        return 'missing'
    if any(token in text for token in ('failed', 'blocked', 'contradict')):
        return 'failed'
    if any(token in text for token in ('satisfied', 'resolved', 'passed', 'confirmed')):
        return 'satisfied'
    return 'advisory'


def _status_from_adjudication(body: dict[str, Any]) -> str:
    action = body.get('selected_action')
    if action == 'resolve_contract':
        return 'resolved'
    if action in {'request_math_repair', 'request_code_repair'}:
        return 'open'
    if action == 'defer':
        return 'blocked'
    return 'open'


def _source_hash(
    evaluator: dict[str, Any],
    outcome: dict[str, Any],
    contracts: dict[str, Any],
    adjudicator: dict[str, Any],
) -> str:
    return stable_digest({
        'evaluator': evaluator.get('ledger_hash') or evaluator.get('ledger_id'),
        'outcome': outcome.get('ledger_hash') or outcome.get('ledger_id'),
        'contract': contracts.get('ledger_hash'),
        'adjudicator': adjudicator.get('ledger_hash'),
    })


def _required_gates(contract: dict[str, Any]) -> list[str]:
    gates = set(contract.get('required_evidence_gates') or [])
    gates.update({'math_proof', 'code_proof'})
    return sorted(gates)


def _required_agenda_gates(candidate: dict[str, Any] | None) -> list[str]:
    if candidate:
        return list(candidate.get('required_evidence_gates') or [])
    return [
        'math_proof',
        'code_proof',
        'language_epoch_plan',
        'label_clean_downstream_evidence',
        'runtime_memory_not_mutated',
        'project_owned_checkpoint_boundary_explicit',
    ]


def _gates_from_contract_map(
    compact: dict[str, Any] | None,
    contract_id: str,
    field: str,
) -> list[str]:
    value = dict(compact or {}).get(field) or {}
    if isinstance(value, dict):
        return list(value.get(str(contract_id)) or [])
    return []


def _gates_by_status(hypotheses: list[dict[str, Any]], status: str) -> dict[str, list[str]]:
    field = 'satisfied_evidence_gates' if status == 'satisfied' else 'failed_evidence_gates'
    return {
        str(item.get('contract_id')): list(item.get(field) or [])
        for item in hypotheses
        if item.get(field)
    }


def _gate_count(hypotheses: list[dict[str, Any]], status: str) -> int:
    if status == 'open':
        return sum(
            len(set(item.get('required_evidence_gates') or []) - set(item.get('satisfied_evidence_gates') or []))
            for item in hypotheses
            if item.get('status') == 'open'
        )
    field = 'satisfied_evidence_gates' if status == 'satisfied' else 'failed_evidence_gates'
    return sum(len(item.get(field) or []) for item in hypotheses)


def _hypothesis_lineage(hypotheses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            'hypothesis_id': item.get('hypothesis_id'),
            'contract_id': item.get('contract_id'),
            'status': item.get('status'),
            'lineage': list(item.get('lineage') or []),
            'satisfied_evidence_gates': list(item.get('satisfied_evidence_gates') or []),
            'failed_evidence_gates': list(item.get('failed_evidence_gates') or []),
        }
        for item in hypotheses
    ]


def _unique_strings(values: list[Any]) -> list[str]:
    seen = set()
    output = []
    for value in values:
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output
