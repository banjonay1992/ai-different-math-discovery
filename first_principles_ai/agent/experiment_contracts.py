"""Plain-data experiment contracts for AI Different module-family decisions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .family_outcome_evaluator import EVALUATOR_LEDGER_KIND
from .module_chat_adapter import (
    build_module_chat_message,
    label_leak_terms,
    module_chat_message_id,
    stable_digest,
    validate_module_chat_message,
    validate_participant,
)


CONTRACT_LEDGER_KIND = 'ai_different.experiment_contract_ledger'
CONTRACT_MESSAGE_TOPIC = 'ai_different.experiment_contract'


def load_experiment_contract_ledger(path: str | Path | None) -> dict[str, Any]:
    if not path or not Path(path).exists():
        return empty_experiment_contract_ledger()
    with Path(path).open('r', encoding='utf-8') as handle:
        ledger = json.load(handle)
    return validate_experiment_contract_ledger(ledger)


def write_experiment_contract_ledger(path: str | Path, ledger: dict[str, Any]) -> Path:
    validated = validate_experiment_contract_ledger(ledger)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(validated, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    return output_path


def empty_experiment_contract_ledger() -> dict[str, Any]:
    return {
        'schema_version': 1,
        'ledger_kind': CONTRACT_LEDGER_KIND,
        'contracts': [],
        'processed_downstream_evidence_ids': [],
        'emitted_evaluator_ledger_ids': [],
        'outgoing_message_ids': [],
        'latest': {},
    }


def validate_experiment_contract_ledger(ledger: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(ledger, dict):
        raise ValueError('experiment contract ledger must be a JSON object')
    if ledger.get('ledger_kind') != CONTRACT_LEDGER_KIND:
        raise ValueError('experiment contract ledger has wrong ledger_kind')
    contracts = ledger.get('contracts')
    processed = ledger.get('processed_downstream_evidence_ids')
    emitted = ledger.get('emitted_evaluator_ledger_ids')
    outgoing = ledger.get('outgoing_message_ids')
    latest = ledger.get('latest', {})
    if not isinstance(contracts, list) or not all(isinstance(item, dict) for item in contracts):
        raise ValueError('experiment contract ledger contracts must be objects')
    if not isinstance(processed, list) or not all(isinstance(item, str) for item in processed):
        raise ValueError('processed_downstream_evidence_ids must be strings')
    if not isinstance(emitted, list) or not all(isinstance(item, str) for item in emitted):
        raise ValueError('emitted_evaluator_ledger_ids must be strings')
    if not isinstance(outgoing, list) or not all(isinstance(item, str) for item in outgoing):
        raise ValueError('outgoing_message_ids must be strings')
    if not isinstance(latest, dict):
        raise ValueError('experiment contract ledger latest must be an object')
    validated = {
        'schema_version': int(ledger.get('schema_version', 1) or 1),
        'ledger_kind': CONTRACT_LEDGER_KIND,
        'contracts': list(contracts),
        'processed_downstream_evidence_ids': _unique_strings(processed),
        'emitted_evaluator_ledger_ids': _unique_strings(emitted),
        'outgoing_message_ids': _unique_strings(outgoing),
        'latest': dict(latest),
    }
    if ledger.get('ledger_hash'):
        validated['ledger_hash'] = str(ledger.get('ledger_hash'))
    if ledger.get('artifact_path'):
        validated['artifact_path'] = str(ledger.get('artifact_path'))
    return validated


def validate_evaluator_ledger(ledger: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(ledger, dict):
        raise ValueError('evaluator ledger must be a JSON object')
    if ledger.get('ledger_kind') != EVALUATOR_LEDGER_KIND:
        raise ValueError('evaluator ledger has wrong ledger_kind')
    if not isinstance(ledger.get('decision'), dict):
        raise ValueError('evaluator ledger decision must be an object')
    if not isinstance(ledger.get('selected_experiment'), dict):
        raise ValueError('evaluator ledger selected_experiment must be an object')
    return dict(ledger)


def read_family_bus_messages(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {'path': None, 'messages': [], 'invalid_messages': []}
    bus_path = Path(path)
    if not bus_path.exists():
        return {'path': str(bus_path), 'messages': [], 'invalid_messages': []}
    messages = []
    invalid = []
    with bus_path.open('r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                messages.append(validate_module_chat_message(json.loads(text)))
            except (json.JSONDecodeError, ValueError) as error:
                invalid.append({'line': line_number, 'error': str(error), 'raw': text})
    return {'path': str(bus_path), 'messages': messages, 'invalid_messages': invalid}


def downstream_evidence_for_contracts(
    messages: list[dict[str, Any]],
    contracts: list[dict[str, Any]],
    processed_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    known_contract_ids = {contract.get('contract_id') for contract in contracts}
    processed = set(processed_ids or [])
    evidence_items = []
    for message in messages:
        if message.get('sender') == 'ai_different':
            continue
        body = dict(message.get('body') or {})
        evidence = dict(message.get('evidence') or {})
        contract_id = (
            body.get('contract_id')
            or evidence.get('contract_id')
            or body.get('experiment_contract_id')
            or evidence.get('experiment_contract_id')
        )
        if contract_id not in known_contract_ids:
            continue
        evidence_id = str(
            body.get('evidence_id')
            or evidence.get('evidence_id')
            or module_chat_message_id(message)
        )
        status = _downstream_status(message)
        evidence_items.append({
            'evidence_id': evidence_id,
            'contract_id': str(contract_id),
            'sender': message.get('sender'),
            'topic': message.get('topic'),
            'status': status,
            'summary': body.get('summary') or body.get('note') or body.get('question'),
            'message_id': module_chat_message_id(message),
            'processed': evidence_id in processed,
            'label_clean': not label_leak_terms({'body': body, 'evidence': evidence}),
            'leak_terms': label_leak_terms({'body': body, 'evidence': evidence}),
        })
    return evidence_items


def build_experiment_contract_from_evaluator(
    evaluator_ledger: dict[str, Any],
    *,
    target_recipient: str = 'broadcast',
) -> dict[str, Any]:
    ledger = validate_evaluator_ledger(evaluator_ledger)
    selected = dict(ledger.get('selected_experiment') or {})
    decision = dict(ledger.get('decision') or {})
    contract_id = 'contract_' + stable_digest({
        'evaluator_ledger_id': ledger.get('ledger_id'),
        'selected_experiment': selected,
        'decision_kind': decision.get('decision_kind'),
    })[:16]
    gates = [
        'label_clean_downstream_evidence',
        'runtime_memory_not_mutated',
        'project_owned_checkpoint_boundary_explicit',
        'falsifiable_transfer_signal_reported',
    ]
    return {
        'contract_id': contract_id,
        'signature': stable_digest({
            'contract_id': contract_id,
            'source_evaluator_ledger_hash': ledger.get('ledger_hash'),
            'selected_experiment': selected,
        }),
        'source_evaluator_ledger_id': ledger.get('ledger_id'),
        'source_evaluator_ledger_hash': ledger.get('ledger_hash'),
        'selected_evidence_ids': list(ledger.get('chosen_evidence_ids') or []),
        'selected_world': selected.get('world') or 'not_selected',
        'selected_probe': selected.get('probe') or selected.get('experiment_kind'),
        'selected_experiment': selected,
        'expected_transfer_signal': ledger.get('expected_transfer_signal'),
        'required_evidence_gates': gates,
        'target_recipient': validate_participant(target_recipient),
        'status': 'open',
        'downstream_evidence_ids': [],
        'blockers': [],
        'outgoing_message_ids': [],
        'label_clean': bool(ledger.get('label_clean')),
    }


def export_experiment_contract_message(
    contract: dict[str, Any],
    *,
    runtime_memory_hash_state: dict[str, Any],
    project_owned_boundary: dict[str, Any],
    topic: str = CONTRACT_MESSAGE_TOPIC,
) -> dict[str, Any]:
    recipient = validate_participant(str(contract.get('target_recipient') or 'broadcast'))
    body = {
        'module': 'AI Different',
        'response_kind': 'experiment_contract',
        'contract_id': contract.get('contract_id'),
        'contract_signature': contract.get('signature'),
        'source_evaluator_ledger_id': contract.get('source_evaluator_ledger_id'),
        'source_evaluator_ledger_hash': contract.get('source_evaluator_ledger_hash'),
        'selected_evidence_ids': list(contract.get('selected_evidence_ids') or []),
        'selected_world': contract.get('selected_world'),
        'selected_probe': contract.get('selected_probe'),
        'selected_experiment': dict(contract.get('selected_experiment') or {}),
        'expected_transfer_signal': contract.get('expected_transfer_signal'),
        'required_evidence_gates': list(contract.get('required_evidence_gates') or []),
        'runtime_memory_hash_state': dict(runtime_memory_hash_state),
        'runtime_memory_mutated': not bool(runtime_memory_hash_state.get('unchanged', True)),
        'project_owned_boundary': dict(project_owned_boundary),
        'third_party_checkpoint_used': bool(
            project_owned_boundary.get('third_party_checkpoint_used')
        ),
        'label_clean': bool(contract.get('label_clean')),
        'leak_terms': label_leak_terms(contract),
    }
    evidence = {
        'contract_id': body['contract_id'],
        'contract_signature': body['contract_signature'],
        'source_evaluator_ledger_hash': body['source_evaluator_ledger_hash'],
        'required_evidence_gates': body['required_evidence_gates'],
        'runtime_memory_hash_state': body['runtime_memory_hash_state'],
        'runtime_memory_mutated': body['runtime_memory_mutated'],
        'project_owned_boundary': body['project_owned_boundary'],
        'label_clean': body['label_clean'],
    }
    return build_module_chat_message(
        sender='ai_different',
        recipient=recipient,
        topic=topic,
        body=body,
        evidence=evidence,
        tags=[
            'ai_different',
            'experiment_contract',
            'abstraction_transfer',
            'label_clean' if body['label_clean'] and not body['leak_terms'] else 'label_review_needed',
        ],
    )


def export_contract_repair_message(
    contract: dict[str, Any],
    blocker: dict[str, Any],
    *,
    runtime_memory_hash_state: dict[str, Any],
    project_owned_boundary: dict[str, Any],
    recipient: str = 'orchestrator',
    topic: str = 'ai_different.experiment_contract_repair',
) -> dict[str, Any]:
    recipient = validate_participant(recipient)
    body = {
        'module': 'AI Different',
        'response_kind': 'experiment_contract_repair',
        'contract_id': contract.get('contract_id'),
        'contract_signature': contract.get('signature'),
        'decision': 'defer_or_repair',
        'reason': 'downstream evidence blocked or contradicted the experiment contract',
        'blocker': dict(blocker),
        'required_evidence_gates': list(contract.get('required_evidence_gates') or []),
        'runtime_memory_hash_state': dict(runtime_memory_hash_state),
        'runtime_memory_mutated': not bool(runtime_memory_hash_state.get('unchanged', True)),
        'project_owned_boundary': dict(project_owned_boundary),
        'third_party_checkpoint_used': bool(
            project_owned_boundary.get('third_party_checkpoint_used')
        ),
        'label_clean': not label_leak_terms({'contract': contract, 'blocker': blocker}),
        'leak_terms': label_leak_terms({'contract': contract, 'blocker': blocker}),
    }
    return build_module_chat_message(
        sender='ai_different',
        recipient=recipient,
        topic=topic,
        body=body,
        evidence={
            'contract_id': contract.get('contract_id'),
            'blocker_id': blocker.get('evidence_id'),
            'runtime_memory_hash_state': body['runtime_memory_hash_state'],
            'runtime_memory_mutated': body['runtime_memory_mutated'],
            'label_clean': body['label_clean'],
        },
        tags=[
            'ai_different',
            'experiment_contract_repair',
            'defer_or_repair',
            'label_clean' if body['label_clean'] else 'label_review_needed',
        ],
    )


def update_contract_ledger(
    contract_ledger: dict[str, Any],
    evaluator_ledger: dict[str, Any],
    bus_messages: list[dict[str, Any]],
    *,
    runtime_memory_hash_state: dict[str, Any],
    project_owned_boundary: dict[str, Any],
    target_recipient: str = 'broadcast',
    repair_recipient: str = 'orchestrator',
    artifact_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    ledger = validate_experiment_contract_ledger(contract_ledger)
    evaluator = validate_evaluator_ledger(evaluator_ledger)
    outgoing_message = None
    downstream = downstream_evidence_for_contracts(
        bus_messages,
        ledger['contracts'],
        ledger['processed_downstream_evidence_ids'],
    )
    fresh_downstream = [item for item in downstream if not item['processed']]
    resolved_count = 0
    blocked_count = 0
    for evidence in fresh_downstream:
        contract = _find_contract(ledger['contracts'], evidence['contract_id'])
        if contract is None:
            continue
        contract.setdefault('downstream_evidence_ids', []).append(evidence['evidence_id'])
        if evidence['status'] == 'satisfied':
            contract['status'] = 'resolved'
            contract['resolved_by'] = evidence
            resolved_count += 1
        elif evidence['status'] == 'blocked':
            contract['status'] = 'blocked'
            contract.setdefault('blockers', []).append(evidence)
            blocked_count += 1
            if outgoing_message is None:
                outgoing_message = export_contract_repair_message(
                    contract,
                    evidence,
                    runtime_memory_hash_state=runtime_memory_hash_state,
                    project_owned_boundary=project_owned_boundary,
                    recipient=repair_recipient,
                )
    emitted_ids = set(ledger['emitted_evaluator_ledger_ids'])
    evaluator_id = str(evaluator.get('ledger_id'))
    new_contract_count = 0
    skipped_count = 0
    if evaluator_id in emitted_ids:
        skipped_count += 1
    elif outgoing_message is None and _evaluator_wants_contract(evaluator):
        contract = build_experiment_contract_from_evaluator(
            evaluator,
            target_recipient=target_recipient,
        )
        message = export_experiment_contract_message(
            contract,
            runtime_memory_hash_state=runtime_memory_hash_state,
            project_owned_boundary=project_owned_boundary,
        )
        message_id = module_chat_message_id(message)
        contract['outgoing_message_ids'] = [message_id]
        ledger['contracts'].append(contract)
        ledger['emitted_evaluator_ledger_ids'].append(evaluator_id)
        outgoing_message = message
        new_contract_count += 1
    ledger['processed_downstream_evidence_ids'] = _unique_strings(
        list(ledger['processed_downstream_evidence_ids'])
        + [item['evidence_id'] for item in fresh_downstream]
    )
    if outgoing_message is not None:
        ledger['outgoing_message_ids'] = _unique_strings(
            list(ledger['outgoing_message_ids'])
            + [module_chat_message_id(outgoing_message)]
        )
    ledger['emitted_evaluator_ledger_ids'] = _unique_strings(
        ledger['emitted_evaluator_ledger_ids']
    )
    ledger['latest'] = {
        'new_contract_count': new_contract_count,
        'skipped_contract_count': skipped_count,
        'resolved_contract_count': resolved_count,
        'blocked_contract_count': blocked_count,
        'outbox_count': 1 if outgoing_message is not None else 0,
        'chosen_action': _message_action(outgoing_message),
        'chosen_recipient': outgoing_message.get('recipient') if outgoing_message else None,
        'runtime_memory_hash_state': dict(runtime_memory_hash_state),
        'runtime_memory_mutated': not bool(runtime_memory_hash_state.get('unchanged', True)),
        'label_leaks': label_leak_terms({
            'contracts': ledger['contracts'],
            'message': outgoing_message,
        }),
    }
    if artifact_path is not None:
        ledger['artifact_path'] = str(Path(artifact_path))
    ledger['ledger_hash'] = stable_digest({
        key: value for key, value in ledger.items()
        if key != 'ledger_hash'
    })
    return ledger, outgoing_message


def write_contract_outbox_jsonl(
    path: str | Path,
    message: dict[str, Any] | None,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        (json.dumps(message, sort_keys=True) + '\n') if message is not None else '',
        encoding='utf-8',
    )
    return output_path


def _downstream_status(message: dict[str, Any]) -> str:
    body = dict(message.get('body') or {})
    evidence = dict(message.get('evidence') or {})
    text = json.dumps({'body': body, 'evidence': evidence, 'tags': message.get('tags')}, sort_keys=True).lower()
    explicit = str(body.get('status') or evidence.get('status') or '').lower()
    if explicit in {'satisfied', 'resolved', 'passed', 'confirmed'}:
        return 'satisfied'
    if explicit in {'blocked', 'failed', 'contradicted', 'contradiction'}:
        return 'blocked'
    if any(token in text for token in ('blocked', 'failed', 'contradict')):
        return 'blocked'
    if any(token in text for token in ('satisfied', 'resolved', 'passed', 'confirmed')):
        return 'satisfied'
    return 'advisory'


def _find_contract(contracts: list[dict[str, Any]], contract_id: str) -> dict[str, Any] | None:
    for contract in contracts:
        if contract.get('contract_id') == contract_id:
            return contract
    return None


def _evaluator_wants_contract(evaluator: dict[str, Any]) -> bool:
    if not bool(evaluator.get('label_clean')):
        return False
    decision = dict(evaluator.get('decision') or {})
    if decision.get('decision_kind') != 'run_next_safe_experiment':
        return False
    selected = dict(evaluator.get('selected_experiment') or {})
    return bool(selected) and not bool(selected.get('runs_final', False))


def _message_action(message: dict[str, Any] | None) -> str:
    if not message:
        return 'none'
    body = dict(message.get('body') or {})
    return str(body.get('response_kind') or 'module_chat_response')


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
