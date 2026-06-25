"""Cross-module evidence adjudicator for AI Different.

This module reconciles a family transcript and plain JSON ledgers without
importing sibling projects. It decides whether a contract is resolved, blocked,
needs proof repair, should spawn the next safe contract, or should no-op.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .experiment_contracts import (
    CONTRACT_LEDGER_KIND,
    build_experiment_contract_from_evaluator,
    validate_evaluator_ledger,
)
from .family_outcome_evaluator import EVALUATOR_LEDGER_KIND
from .module_chat_adapter import (
    build_module_chat_message,
    label_leak_terms,
    module_chat_message_id,
    stable_digest,
    validate_module_chat_message,
    validate_participant,
)


ADJUDICATOR_LEDGER_KIND = 'ai_different.cross_module_adjudicator_ledger'


def empty_adjudicator_ledger() -> dict[str, Any]:
    return {
        'schema_version': 1,
        'ledger_kind': ADJUDICATOR_LEDGER_KIND,
        'processed_message_ids': [],
        'processed_evaluator_ledger_ids': [],
        'contract_states': [],
        'adjudication_records': [],
        'outgoing_response_ids': [],
        'latest': {},
    }


def load_adjudicator_ledger(path: str | Path | None) -> dict[str, Any]:
    if not path or not Path(path).exists():
        return empty_adjudicator_ledger()
    with Path(path).open('r', encoding='utf-8') as handle:
        ledger = json.load(handle)
    return validate_adjudicator_ledger(ledger)


def write_adjudicator_ledger(path: str | Path, ledger: dict[str, Any]) -> Path:
    validated = validate_adjudicator_ledger(ledger)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(validated, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    return output


def validate_adjudicator_ledger(ledger: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(ledger, dict):
        raise ValueError('adjudicator ledger must be a JSON object')
    if ledger.get('ledger_kind') != ADJUDICATOR_LEDGER_KIND:
        raise ValueError('adjudicator ledger has wrong ledger_kind')
    for field in (
        'processed_message_ids',
        'processed_evaluator_ledger_ids',
        'outgoing_response_ids',
    ):
        values = ledger.get(field)
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            raise ValueError(f'{field} must be strings')
    for field in ('contract_states', 'adjudication_records'):
        values = ledger.get(field)
        if not isinstance(values, list) or not all(isinstance(item, dict) for item in values):
            raise ValueError(f'{field} must be objects')
    latest = ledger.get('latest', {})
    if not isinstance(latest, dict):
        raise ValueError('adjudicator ledger latest must be an object')
    validated = {
        'schema_version': int(ledger.get('schema_version', 1) or 1),
        'ledger_kind': ADJUDICATOR_LEDGER_KIND,
        'processed_message_ids': _unique_strings(ledger['processed_message_ids']),
        'processed_evaluator_ledger_ids': _unique_strings(
            ledger['processed_evaluator_ledger_ids']
        ),
        'contract_states': list(ledger['contract_states']),
        'adjudication_records': list(ledger['adjudication_records']),
        'outgoing_response_ids': _unique_strings(ledger['outgoing_response_ids']),
        'latest': dict(latest),
    }
    if ledger.get('ledger_hash'):
        validated['ledger_hash'] = str(ledger['ledger_hash'])
    if ledger.get('artifact_path'):
        validated['artifact_path'] = str(ledger['artifact_path'])
    return validated


def read_family_transcript(path: str | Path | None) -> dict[str, Any]:
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
        raise ValueError('plain ledger input must be a JSON object')
    return value


def build_adjudication(
    *,
    transcript_messages: list[dict[str, Any]],
    adjudicator_ledger: dict[str, Any],
    evaluator_ledger: dict[str, Any] | None = None,
    contract_ledger: dict[str, Any] | None = None,
    runtime_memory_hash_state: dict[str, Any],
    project_owned_boundary: dict[str, Any],
    artifact_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    ledger = validate_adjudicator_ledger(adjudicator_ledger)
    evaluator = _valid_evaluator_or_empty(evaluator_ledger or {})
    contract_ledger = _valid_contract_ledger_or_empty(contract_ledger or {})
    processed = set(ledger['processed_message_ids'])
    new_messages = [
        message for message in transcript_messages
        if module_chat_message_id(message) not in processed
    ]
    skipped_messages = [
        message for message in transcript_messages
        if module_chat_message_id(message) in processed
    ]
    contract_states = _merge_contract_states(
        ledger['contract_states'],
        contract_ledger,
        transcript_messages,
    )
    evidence = _extract_adjudication_evidence(new_messages, contract_states)
    _apply_evidence_to_contracts(contract_states, evidence)
    evaluator_is_new = bool(evaluator) and str(evaluator.get('ledger_id')) not in set(
        ledger['processed_evaluator_ledger_ids']
    )
    if not new_messages and (not evaluator_is_new or contract_states):
        selected = _noop_action('no new transcript or evaluator evidence since last adjudication')
    else:
        selected = _select_adjudication_action(
            contract_states,
            evidence,
            evaluator if evaluator_is_new else {},
            project_owned_boundary,
        )
    message = export_adjudication_message(
        selected,
        contract_states=contract_states,
        evidence_items=evidence,
        runtime_memory_hash_state=runtime_memory_hash_state,
        project_owned_boundary=project_owned_boundary,
        source_hashes={
            'evaluator_ledger_hash': evaluator.get('ledger_hash'),
            'contract_ledger_hash': contract_ledger.get('ledger_hash'),
        },
    )
    new_ids = [module_chat_message_id(message) for message in new_messages]
    outgoing_id = module_chat_message_id(message) if message is not None else None
    latest = {
        'new_message_count': len(new_messages),
        'skipped_message_count': len(skipped_messages),
        'open_contract_count': sum(1 for item in contract_states if item.get('status') == 'open'),
        'resolved_contract_count': sum(1 for item in contract_states if item.get('status') == 'resolved'),
        'blocked_contract_count': sum(1 for item in contract_states if item.get('status') == 'blocked'),
        'selected_action': selected['selected_action'],
        'chosen_recipient': message.get('recipient') if message else None,
        'outbox_count': 1 if message else 0,
        'label_leaks': selected.get('label_leaks', []),
        'runtime_memory_hash_state': dict(runtime_memory_hash_state),
        'runtime_memory_mutated': not bool(runtime_memory_hash_state.get('unchanged', True)),
    }
    record = {
        'record_id': 'adjudication_' + stable_digest({
            'new_message_ids': new_ids,
            'evaluator_ledger_id': evaluator.get('ledger_id'),
            'selected_action': selected['selected_action'],
        })[:16],
        'processed_message_ids': new_ids,
        'contract_ids': [item.get('contract_id') for item in contract_states],
        'contract_signatures': [item.get('signature') for item in contract_states],
        'evidence_ids_by_sender': _evidence_ids_by_sender(evidence),
        'satisfied_evidence_gates': _gates_by_status(contract_states, 'satisfied'),
        'failed_evidence_gates': _gates_by_status(contract_states, 'failed'),
        'selected_action': selected['selected_action'],
        'chosen_recipient': latest['chosen_recipient'],
        'outgoing_response_id': outgoing_id,
        'source_ledger_hashes': {
            'evaluator_ledger_hash': evaluator.get('ledger_hash'),
            'contract_ledger_hash': contract_ledger.get('ledger_hash'),
        },
    }
    ledger['processed_message_ids'] = _unique_strings(
        list(ledger['processed_message_ids']) + new_ids
    )
    if evaluator_is_new and selected['selected_action'] == 'emit_next_contract':
        ledger['processed_evaluator_ledger_ids'] = _unique_strings(
            list(ledger['processed_evaluator_ledger_ids'])
            + [str(evaluator.get('ledger_id'))]
        )
    ledger['contract_states'] = contract_states
    if new_messages or evaluator_is_new or message is not None:
        ledger['adjudication_records'] = list(ledger['adjudication_records']) + [record]
    if outgoing_id:
        ledger['outgoing_response_ids'] = _unique_strings(
            list(ledger['outgoing_response_ids']) + [outgoing_id]
        )
    ledger['latest'] = latest
    if artifact_path is not None:
        ledger['artifact_path'] = str(Path(artifact_path))
    ledger['ledger_hash'] = stable_digest({
        key: value for key, value in ledger.items()
        if key != 'ledger_hash'
    })
    return ledger, message


def export_adjudication_message(
    selected: dict[str, Any],
    *,
    contract_states: list[dict[str, Any]],
    evidence_items: list[dict[str, Any]],
    runtime_memory_hash_state: dict[str, Any],
    project_owned_boundary: dict[str, Any],
    source_hashes: dict[str, Any],
    topic: str = 'ai_different.cross_module_adjudication',
) -> dict[str, Any] | None:
    if selected['selected_action'] == 'summarize_noop':
        return None
    recipient = validate_participant(selected.get('chosen_recipient') or 'orchestrator')
    leak_terms = label_leak_terms({
        'selected': selected,
        'contract_states': contract_states,
        'evidence': evidence_items,
    })
    body = {
        'module': 'AI Different',
        'response_kind': 'cross_module_adjudication',
        'selected_action': selected['selected_action'],
        'reason': selected.get('reason'),
        'contract_id': selected.get('contract_id'),
        'contract_signature': selected.get('contract_signature'),
        'missing_gate': selected.get('missing_gate'),
        'resolution_proof': selected.get('resolution_proof'),
        'compact_evidence_contract': {
            'evidence_ids_by_sender': _evidence_ids_by_sender(evidence_items),
            'satisfied_evidence_gates': _gates_by_status(contract_states, 'satisfied'),
            'failed_evidence_gates': _gates_by_status(contract_states, 'failed'),
        },
        'source_ledger_hashes': dict(source_hashes),
        'runtime_memory_hash_state': dict(runtime_memory_hash_state),
        'runtime_memory_mutated': not bool(runtime_memory_hash_state.get('unchanged', True)),
        'project_owned_boundary': dict(project_owned_boundary),
        'third_party_checkpoint_used': bool(
            project_owned_boundary.get('third_party_checkpoint_used')
        ),
        'label_leaks': leak_terms,
        'label_clean': not leak_terms,
    }
    return build_module_chat_message(
        sender='ai_different',
        recipient=recipient,
        topic=topic,
        body=body,
        evidence={
            'selected_action': body['selected_action'],
            'contract_id': body['contract_id'],
            'missing_gate': body['missing_gate'],
            'resolution_proof': body['resolution_proof'],
            'source_ledger_hashes': body['source_ledger_hashes'],
            'runtime_memory_hash_state': body['runtime_memory_hash_state'],
            'runtime_memory_mutated': body['runtime_memory_mutated'],
            'label_clean': body['label_clean'],
        },
        tags=[
            'ai_different',
            'cross_module_adjudication',
            body['selected_action'],
            'label_clean' if body['label_clean'] else 'label_review_needed',
        ],
    )


def write_adjudication_outbox_jsonl(
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


def _valid_contract_ledger_or_empty(ledger: dict[str, Any]) -> dict[str, Any]:
    if not ledger:
        return {'ledger_kind': CONTRACT_LEDGER_KIND, 'contracts': [], 'ledger_hash': None}
    if ledger.get('ledger_kind') != CONTRACT_LEDGER_KIND:
        raise ValueError('contract ledger has wrong ledger_kind')
    return ledger


def _merge_contract_states(
    existing: list[dict[str, Any]],
    contract_ledger: dict[str, Any],
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {
        str(item.get('contract_id')): dict(item)
        for item in existing
        if item.get('contract_id')
    }
    for contract in list(contract_ledger.get('contracts') or []):
        _upsert_contract_state(states, contract)
    for message in messages:
        if message.get('sender') != 'ai_different':
            continue
        body = dict(message.get('body') or {})
        if body.get('response_kind') != 'experiment_contract':
            continue
        _upsert_contract_state(states, {
            'contract_id': body.get('contract_id'),
            'signature': body.get('contract_signature'),
            'required_evidence_gates': body.get('required_evidence_gates') or [],
            'selected_world': body.get('selected_world'),
            'selected_probe': body.get('selected_probe'),
            'source_evaluator_ledger_hash': body.get('source_evaluator_ledger_hash'),
            'status': 'open',
        })
    return list(states.values())


def _upsert_contract_state(states: dict[str, dict[str, Any]], contract: dict[str, Any]):
    contract_id = contract.get('contract_id')
    if not contract_id:
        return
    current = states.setdefault(str(contract_id), {})
    current.update({
        'contract_id': str(contract_id),
        'signature': contract.get('signature') or contract.get('contract_signature') or current.get('signature'),
        'required_evidence_gates': _required_gates(contract),
        'selected_world': contract.get('selected_world') or current.get('selected_world'),
        'selected_probe': contract.get('selected_probe') or current.get('selected_probe'),
        'source_evaluator_ledger_hash': contract.get('source_evaluator_ledger_hash') or current.get('source_evaluator_ledger_hash'),
        'status': contract.get('status') or current.get('status') or 'open',
        'satisfied_evidence_gates': list(current.get('satisfied_evidence_gates') or []),
        'failed_evidence_gates': list(current.get('failed_evidence_gates') or []),
        'evidence_ids': list(current.get('evidence_ids') or []),
        'blockers': list(current.get('blockers') or []),
    })


def _required_gates(contract: dict[str, Any]) -> list[str]:
    gates = set(contract.get('required_evidence_gates') or [])
    gates.update({'math_proof', 'code_proof', 'label_clean_downstream_evidence'})
    return sorted(gates)


def _extract_adjudication_evidence(
    messages: list[dict[str, Any]],
    contract_states: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    contract_ids = {item.get('contract_id') for item in contract_states}
    items = []
    for message in messages:
        sender = message.get('sender')
        body = dict(message.get('body') or {})
        evidence = dict(message.get('evidence') or {})
        contract_id = (
            body.get('contract_id')
            or evidence.get('contract_id')
            or body.get('experiment_contract_id')
            or evidence.get('experiment_contract_id')
        )
        if sender == 'ai_different' and body.get('response_kind') == 'experiment_contract':
            continue
        gate = _evidence_gate(message)
        status = _evidence_status(message)
        evidence_id = str(body.get('evidence_id') or evidence.get('evidence_id') or module_chat_message_id(message))
        item = {
            'evidence_id': evidence_id,
            'message_id': module_chat_message_id(message),
            'sender': sender,
            'topic': message.get('topic'),
            'contract_id': str(contract_id) if contract_id else None,
            'evidence_gate': gate,
            'status': status,
            'summary': body.get('summary') or body.get('note') or body.get('question'),
            'label_leaks': label_leak_terms({'body': body, 'evidence': evidence}),
            'third_party_checkpoint_used': bool(
                body.get('third_party_checkpoint_used')
                or evidence.get('third_party_checkpoint_used')
            ),
        }
        if item['contract_id'] in contract_ids or sender in {'language_model_2', 'funfun', 'code_module'}:
            items.append(item)
    return items


def _apply_evidence_to_contracts(
    contract_states: list[dict[str, Any]],
    evidence_items: list[dict[str, Any]],
):
    by_id = {item.get('contract_id'): item for item in contract_states}
    for item in evidence_items:
        contract = by_id.get(item.get('contract_id'))
        if not contract:
            continue
        gate = item.get('evidence_gate')
        contract.setdefault('evidence_ids', []).append(item['evidence_id'])
        if item.get('label_leaks') or item.get('third_party_checkpoint_used'):
            contract.setdefault('failed_evidence_gates', []).append('safety_label_project_boundary')
            contract.setdefault('blockers', []).append(item)
            contract['status'] = 'blocked'
            continue
        if item.get('status') == 'failed':
            contract.setdefault('failed_evidence_gates', []).append(gate)
            contract.setdefault('blockers', []).append(item)
            contract['status'] = 'blocked'
        elif item.get('status') == 'satisfied':
            contract.setdefault('satisfied_evidence_gates', []).append(gate)
    for contract in contract_states:
        contract['satisfied_evidence_gates'] = sorted(set(contract.get('satisfied_evidence_gates') or []))
        contract['failed_evidence_gates'] = sorted(set(contract.get('failed_evidence_gates') or []))
        if contract.get('status') == 'blocked':
            continue
        required_resolution = {'math_proof', 'code_proof'}
        if required_resolution <= set(contract.get('satisfied_evidence_gates') or []):
            contract['status'] = 'resolved'
        else:
            contract['status'] = 'open'


def _select_adjudication_action(
    contract_states: list[dict[str, Any]],
    evidence_items: list[dict[str, Any]],
    evaluator_ledger: dict[str, Any],
    project_owned_boundary: dict[str, Any],
) -> dict[str, Any]:
    safety_fail = _first_safety_failure(contract_states, evidence_items, project_owned_boundary)
    if safety_fail:
        return _action('request_code_repair' if safety_fail.get('sender') == 'code_module' else 'request_math_repair', safety_fail, 'failed safety, label, or project-owned boundary')
    for contract in contract_states:
        if contract.get('status') == 'blocked':
            blocker = (contract.get('blockers') or [{}])[-1]
            action = 'request_code_repair' if blocker.get('sender') == 'code_module' else 'request_math_repair'
            return _action(action, contract, 'contract has failed or contradictory evidence', blocker=blocker)
    for contract in contract_states:
        if contract.get('status') == 'open' and 'math_proof' not in set(contract.get('satisfied_evidence_gates') or []):
            return _action('request_math_repair', contract, 'missing math proof gate', missing_gate='math_proof')
    for contract in contract_states:
        if contract.get('status') == 'open' and 'code_proof' not in set(contract.get('satisfied_evidence_gates') or []):
            return _action('request_code_repair', contract, 'missing code proof gate', missing_gate='code_proof')
    for contract in contract_states:
        if contract.get('status') == 'resolved':
            return _action('resolve_contract', contract, 'math and code proof gates satisfied')
    if _evaluator_can_spawn(evaluator_ledger):
        candidate = build_experiment_contract_from_evaluator(evaluator_ledger)
        return _action('emit_next_contract', candidate, 'new safe evaluator decision available')
    return _noop_action('no new actionable cross-module evidence')


def _noop_action(reason: str) -> dict[str, Any]:
    return {
        'selected_action': 'summarize_noop',
        'reason': reason,
        'chosen_recipient': None,
        'contract_id': None,
        'contract_signature': None,
        'missing_gate': None,
        'resolution_proof': None,
        'label_leaks': [],
    }


def _action(
    action: str,
    contract: dict[str, Any],
    reason: str,
    *,
    missing_gate: str | None = None,
    blocker: dict[str, Any] | None = None,
) -> dict[str, Any]:
    recipient = {
        'request_math_repair': 'funfun',
        'request_code_repair': 'code_module',
        'resolve_contract': 'broadcast',
        'emit_next_contract': 'broadcast',
        'defer': 'orchestrator',
    }.get(action, 'orchestrator')
    return {
        'selected_action': action,
        'reason': reason,
        'chosen_recipient': recipient,
        'contract_id': contract.get('contract_id'),
        'contract_signature': contract.get('signature'),
        'missing_gate': missing_gate,
        'resolution_proof': {
            'satisfied_evidence_gates': list(contract.get('satisfied_evidence_gates') or []),
            'evidence_ids': list(contract.get('evidence_ids') or []),
        } if action == 'resolve_contract' else None,
        'blocker': blocker,
        'label_leaks': label_leak_terms({'contract': contract, 'blocker': blocker}),
    }


def _first_safety_failure(
    contract_states: list[dict[str, Any]],
    evidence_items: list[dict[str, Any]],
    project_owned_boundary: dict[str, Any],
) -> dict[str, Any] | None:
    if project_owned_boundary.get('third_party_checkpoint_used'):
        return {'sender': 'code_module', 'reason': 'third-party checkpoint boundary failed'}
    for item in evidence_items:
        if item.get('label_leaks') or item.get('third_party_checkpoint_used'):
            return item
    for contract in contract_states:
        if 'safety_label_project_boundary' in set(contract.get('failed_evidence_gates') or []):
            return (contract.get('blockers') or [contract])[-1]
    return None


def _evaluator_can_spawn(evaluator_ledger: dict[str, Any]) -> bool:
    if not evaluator_ledger or not evaluator_ledger.get('label_clean', True):
        return False
    decision = dict(evaluator_ledger.get('decision') or {})
    selected = dict(evaluator_ledger.get('selected_experiment') or {})
    return decision.get('decision_kind') == 'run_next_safe_experiment' and not selected.get('runs_final', False)


def _evidence_gate(message: dict[str, Any]) -> str:
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
        return 'turn_plan'
    return 'advisory'


def _evidence_status(message: dict[str, Any]) -> str:
    body = dict(message.get('body') or {})
    evidence = dict(message.get('evidence') or {})
    text = json.dumps({'body': body, 'evidence': evidence, 'tags': message.get('tags')}, sort_keys=True).lower()
    explicit = str(body.get('status') or evidence.get('status') or '').lower()
    if explicit in {'satisfied', 'resolved', 'passed', 'confirmed'}:
        return 'satisfied'
    if explicit in {'failed', 'blocked', 'contradicted', 'contradiction'}:
        return 'failed'
    if 'missing' in text:
        return 'missing'
    if any(token in text for token in ('failed', 'blocked', 'contradict')):
        return 'failed'
    if any(token in text for token in ('satisfied', 'resolved', 'passed', 'confirmed')):
        return 'satisfied'
    return 'advisory'


def _evidence_ids_by_sender(evidence_items: list[dict[str, Any]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for item in evidence_items:
        grouped.setdefault(str(item.get('sender') or 'unknown'), []).append(item['evidence_id'])
    return {key: _unique_strings(value) for key, value in grouped.items()}


def _gates_by_status(contract_states: list[dict[str, Any]], status: str) -> dict[str, list[str]]:
    field = 'satisfied_evidence_gates' if status == 'satisfied' else 'failed_evidence_gates'
    return {
        str(contract.get('contract_id')): list(contract.get(field) or [])
        for contract in contract_states
        if contract.get(field)
    }


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
