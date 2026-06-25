"""Outcome evaluator for AI Different module-family response evidence.

This stays on the plain-data side of the module-chat bridge: it reads AI
Different rolling memory and response ledgers, classifies evidence, chooses one
safe next experiment/defer plan, and writes compact evaluator memory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .module_chat_adapter import (
    build_module_chat_message,
    label_leak_terms,
    stable_digest,
)


EVALUATOR_MEMORY_KIND = 'ai_different.family_outcome_evaluator_memory'
EVALUATOR_LEDGER_KIND = 'ai_different.family_outcome_evaluator_ledger'


def load_outcome_evaluator_memory(path: str | Path | None) -> dict[str, Any]:
    if not path or not Path(path).exists():
        return empty_outcome_evaluator_memory()
    with Path(path).open('r', encoding='utf-8') as handle:
        memory = json.load(handle)
    return validate_outcome_evaluator_memory(memory)


def write_outcome_evaluator_memory(path: str | Path, memory: dict[str, Any]) -> Path:
    validated = validate_outcome_evaluator_memory(memory)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(validated, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    return output_path


def empty_outcome_evaluator_memory() -> dict[str, Any]:
    return {
        'schema_version': 1,
        'memory_kind': EVALUATOR_MEMORY_KIND,
        'processed_evidence_ids': [],
        'processed_ledger_ids': [],
        'outgoing_response_ids': [],
        'decision_records': [],
        'latest': {},
    }


def validate_outcome_evaluator_memory(memory: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(memory, dict):
        raise ValueError('outcome evaluator memory must be a JSON object')
    if memory.get('memory_kind') != EVALUATOR_MEMORY_KIND:
        raise ValueError('outcome evaluator memory has wrong memory_kind')
    processed_evidence = memory.get('processed_evidence_ids')
    processed_ledgers = memory.get('processed_ledger_ids')
    outgoing = memory.get('outgoing_response_ids')
    records = memory.get('decision_records')
    latest = memory.get('latest', {})
    if not isinstance(processed_evidence, list) or not all(
        isinstance(item, str) for item in processed_evidence
    ):
        raise ValueError('processed_evidence_ids must be strings')
    if not isinstance(processed_ledgers, list) or not all(
        isinstance(item, str) for item in processed_ledgers
    ):
        raise ValueError('processed_ledger_ids must be strings')
    if not isinstance(outgoing, list) or not all(
        isinstance(item, str) for item in outgoing
    ):
        raise ValueError('outgoing_response_ids must be strings')
    if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
        raise ValueError('decision_records must be objects')
    if not isinstance(latest, dict):
        raise ValueError('latest must be an object')
    validated = {
        'schema_version': int(memory.get('schema_version', 1) or 1),
        'memory_kind': EVALUATOR_MEMORY_KIND,
        'processed_evidence_ids': _unique_strings(processed_evidence),
        'processed_ledger_ids': _unique_strings(processed_ledgers),
        'outgoing_response_ids': _unique_strings(outgoing),
        'decision_records': list(records),
        'latest': dict(latest),
    }
    if memory.get('memory_hash'):
        validated['memory_hash'] = str(memory.get('memory_hash'))
    return validated


def load_response_ledgers(paths: list[str | Path]) -> list[dict[str, Any]]:
    ledgers = []
    for path in paths:
        if not path:
            continue
        with Path(path).open('r', encoding='utf-8') as handle:
            ledger = json.load(handle)
        if not isinstance(ledger, dict):
            raise ValueError('response ledger must be a JSON object')
        ledgers.append(ledger)
    return ledgers


def collect_family_evidence_items(
    rolling_memory: dict[str, Any],
    response_ledgers: list[dict[str, Any]],
    evaluator_memory: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    processed = set((evaluator_memory or {}).get('processed_evidence_ids') or [])
    items = []
    for ledger in response_ledgers:
        ledger_id = str(ledger.get('ledger_id') or _ledger_fallback_id(ledger))
        selected = dict(ledger.get('selected_request') or {})
        if selected:
            items.append(_evidence_item(
                ledger_id,
                'selected_request',
                selected.get('requested_by') or selected.get('sender') or 'ai_different',
                selected.get('request_topic') or selected.get('topic'),
                selected,
                processed,
            ))
        for note in list(ledger.get('selected_evidence') or []):
            items.append(_evidence_item(
                ledger_id,
                'selected_evidence',
                note.get('sender') or 'unknown',
                note.get('topic'),
                note,
                processed,
            ))
        outcome = dict(ledger.get('outcome_or_plan') or {})
        if outcome:
            items.append(_evidence_item(
                ledger_id,
                'prior_ai_different_outcome',
                'ai_different',
                'ai_different.outcome_or_plan',
                outcome,
                processed,
            ))
        for record in list((rolling_memory or {}).get('response_records') or []):
            if record.get('response_ledger_id') != ledger_id:
                continue
            for rationale in list(record.get('evidence_rationale') or []):
                items.append(_evidence_item(
                    ledger_id,
                    'rolling_rationale',
                    rationale.get('sender') or 'unknown',
                    rationale.get('topic'),
                    rationale,
                    processed,
                ))
    unique: dict[str, dict[str, Any]] = {}
    for item in items:
        unique[item['evidence_id']] = item
    return list(unique.values())


def classify_family_evidence(item: dict[str, Any]) -> str:
    if bool(item.get('stale_duplicate')):
        return 'stale_duplicate_item'
    payload = item.get('payload') or {}
    text = json.dumps({
        'sender': item.get('sender'),
        'topic': item.get('topic'),
        'payload': payload,
    }, sort_keys=True).lower()
    if item.get('source_kind') == 'selected_request' and not item.get('leak_terms'):
        if payload.get('source') == 'inbox':
            return 'runnable_experiment_request'
    if 'missing' in text or 'blocker' in text or 'plan_only_missing' in text:
        return 'missing_evidence_blocker'
    if item.get('sender') == 'code_module' or 'proof' in text or 'explanation' in text:
        return 'proof_support'
    if item.get('sender') == 'funfun' or 'theorem' in text or 'advisory' in text:
        return 'advisory_note'
    if item.get('sender') == 'language_model_2' and 'warning' in text:
        return 'missing_evidence_blocker'
    return 'advisory_note'


def choose_family_outcome_decision(items: list[dict[str, Any]]) -> dict[str, Any]:
    fresh = [item for item in items if item.get('classification') != 'stale_duplicate_item']
    by_class = {
        name: [item for item in fresh if item.get('classification') == name]
        for name in (
            'runnable_experiment_request',
            'missing_evidence_blocker',
            'advisory_note',
            'proof_support',
        )
    }
    if by_class['runnable_experiment_request']:
        chosen = _highest_priority(by_class['runnable_experiment_request'])
        return _decision(
            'run_next_safe_experiment',
            chosen,
            'label-clean runnable experiment outranks blockers and advisory notes',
        )
    if by_class['missing_evidence_blocker']:
        chosen = _highest_priority(by_class['missing_evidence_blocker'])
        return _decision(
            'repair_missing_evidence',
            chosen,
            'missing-evidence repair outranks advisory notes',
        )
    if by_class['advisory_note'] or by_class['proof_support']:
        chosen = _highest_priority(by_class['advisory_note'] + by_class['proof_support'])
        return _decision(
            'defer_with_cross_module_advisory',
            chosen,
            'no runnable request; keep advisory evidence for the next probe',
        )
    return {
        'decision_kind': 'no_op',
        'priority_rank': 4,
        'reason': 'no new non-duplicate evaluator evidence',
        'chosen_evidence': {},
        'selected_experiment': {},
        'expected_transfer_signal': None,
        'unresolved_blockers': [],
        'should_export_response': False,
    }


def build_outcome_evaluator_ledger(
    *,
    rolling_memory: dict[str, Any],
    response_ledgers: list[dict[str, Any]],
    evaluator_memory: dict[str, Any],
    runtime_memory_hash_state: dict[str, Any],
    project_owned_boundary: dict[str, Any],
    ledger_path: str | Path | None = None,
) -> dict[str, Any]:
    items = collect_family_evidence_items(
        rolling_memory,
        response_ledgers,
        evaluator_memory,
    )
    for item in items:
        item['classification'] = classify_family_evidence(item)
    decision = choose_family_outcome_decision(items)
    fresh_items = [
        item for item in items
        if item.get('classification') != 'stale_duplicate_item'
    ]
    leak_terms = label_leak_terms({
        'items': fresh_items,
        'decision': decision,
    })
    ledger = {
        'schema_version': 1,
        'ledger_kind': EVALUATOR_LEDGER_KIND,
        'input_hashes': {
            'rolling_memory_hash': rolling_memory.get('memory_hash'),
            'response_ledger_hashes': [
                ledger.get('ledger_hash') for ledger in response_ledgers
            ],
        },
        'processed_ledger_ids': [
            str(ledger.get('ledger_id') or _ledger_fallback_id(ledger))
            for ledger in response_ledgers
        ],
        'processed_evidence_ids': [
            item['evidence_id'] for item in fresh_items
        ],
        'evidence_items': items,
        'classification_counts': _classification_counts(items),
        'chosen_evidence_ids': [
            decision.get('chosen_evidence', {}).get('evidence_id')
        ] if decision.get('chosen_evidence') else [],
        'chosen_evidence_senders': [
            decision.get('chosen_evidence', {}).get('sender')
        ] if decision.get('chosen_evidence') else [],
        'decision': decision,
        'selected_experiment': dict(decision.get('selected_experiment') or {}),
        'expected_transfer_signal': decision.get('expected_transfer_signal'),
        'unresolved_blockers': list(decision.get('unresolved_blockers') or []),
        'runtime_memory_hash_state': dict(runtime_memory_hash_state),
        'runtime_memory_mutated': not bool(runtime_memory_hash_state.get('unchanged', True)),
        'project_owned_boundary': dict(project_owned_boundary),
        'third_party_checkpoint_used': bool(
            project_owned_boundary.get('third_party_checkpoint_used')
        ),
        'outgoing_module_chat_response_ids': [],
        'label_clean': not leak_terms,
        'leak_terms': leak_terms,
    }
    if ledger_path is not None:
        ledger['artifact_path'] = str(Path(ledger_path))
    ledger['ledger_id'] = 'outcome_eval_' + stable_digest({
        key: value for key, value in ledger.items()
        if key != 'ledger_id'
    })[:16]
    ledger['ledger_hash'] = stable_digest(ledger)
    return ledger


def export_outcome_evaluator_message(
    ledger: dict[str, Any],
    *,
    recipient: str = 'orchestrator',
    topic: str = 'ai_different.family_outcome_evaluation',
) -> dict[str, Any] | None:
    decision = dict(ledger.get('decision') or {})
    if decision.get('decision_kind') == 'no_op':
        return None
    body = {
        'module': 'AI Different',
        'response_kind': 'family_outcome_evaluation',
        'ledger_id': ledger.get('ledger_id'),
        'ledger_path': ledger.get('artifact_path'),
        'ledger_hash': ledger.get('ledger_hash'),
        'decision': decision,
        'selected_experiment': dict(ledger.get('selected_experiment') or {}),
        'expected_transfer_signal': ledger.get('expected_transfer_signal'),
        'unresolved_blockers': list(ledger.get('unresolved_blockers') or []),
        'evidence_contract': {
            'classification_counts': dict(ledger.get('classification_counts') or {}),
            'chosen_evidence_ids': list(ledger.get('chosen_evidence_ids') or []),
            'chosen_evidence_senders': list(ledger.get('chosen_evidence_senders') or []),
        },
        'project_owned_boundary': dict(ledger.get('project_owned_boundary') or {}),
        'third_party_checkpoint_used': bool(ledger.get('third_party_checkpoint_used')),
        'runtime_memory_hash_state': dict(ledger.get('runtime_memory_hash_state') or {}),
        'runtime_memory_mutated': bool(ledger.get('runtime_memory_mutated')),
        'label_clean': bool(ledger.get('label_clean')),
        'leak_terms': list(ledger.get('leak_terms') or []),
    }
    evidence = {
        'ledger_id': body['ledger_id'],
        'ledger_hash': body['ledger_hash'],
        'decision_kind': decision.get('decision_kind'),
        'evidence_contract': body['evidence_contract'],
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
            'family_outcome_evaluator',
            'abstraction_transfer',
            'label_clean' if body['label_clean'] else 'label_review_needed',
        ],
    )


def append_outcome_evaluator_memory(
    memory: dict[str, Any],
    ledger: dict[str, Any],
    message: dict[str, Any] | None,
) -> dict[str, Any]:
    updated = validate_outcome_evaluator_memory(memory)
    message_id = _message_id(message) if message else None
    record = {
        'record_id': 'outcome_record_' + stable_digest({
            'ledger_id': ledger.get('ledger_id'),
            'message_id': message_id,
        })[:16],
        'ledger_id': ledger.get('ledger_id'),
        'ledger_hash': ledger.get('ledger_hash'),
        'processed_ledger_ids': list(ledger.get('processed_ledger_ids') or []),
        'processed_evidence_ids': list(ledger.get('processed_evidence_ids') or []),
        'decision': dict(ledger.get('decision') or {}),
        'outgoing_response_id': message_id,
        'label_clean': bool(ledger.get('label_clean')),
    }
    updated['processed_evidence_ids'] = _unique_strings(
        list(updated['processed_evidence_ids'])
        + list(ledger.get('processed_evidence_ids') or [])
    )
    updated['processed_ledger_ids'] = _unique_strings(
        list(updated['processed_ledger_ids'])
        + list(ledger.get('processed_ledger_ids') or [])
    )
    updated['outgoing_response_ids'] = _unique_strings(
        list(updated['outgoing_response_ids'])
        + ([message_id] if message_id else [])
    )
    if ledger.get('processed_evidence_ids') or message_id:
        updated['decision_records'] = list(updated['decision_records']) + [record]
        updated['latest'] = {
            'ledger_id': ledger.get('ledger_id'),
            'ledger_hash': ledger.get('ledger_hash'),
            'decision': dict(ledger.get('decision') or {}),
            'selected_experiment': dict(ledger.get('selected_experiment') or {}),
            'expected_transfer_signal': ledger.get('expected_transfer_signal'),
            'outgoing_response_id': message_id,
            'label_clean': bool(ledger.get('label_clean')),
        }
    updated['memory_hash'] = stable_digest({
        key: value for key, value in updated.items()
        if key != 'memory_hash'
    })
    return updated


def write_outcome_evaluator_ledger(path: str | Path, ledger: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    return output_path


def write_outcome_evaluator_message_jsonl(
    path: str | Path,
    message: dict[str, Any] | None,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = ''
    if message is not None:
        text = json.dumps(message, sort_keys=True) + '\n'
    output_path.write_text(text, encoding='utf-8')
    return output_path


def _evidence_item(
    ledger_id: str,
    source_kind: str,
    sender: str,
    topic: Any,
    payload: dict[str, Any],
    processed: set[str],
) -> dict[str, Any]:
    evidence_id = 'evidence_' + stable_digest({
        'ledger_id': ledger_id,
        'source_kind': source_kind,
        'sender': sender,
        'topic': topic,
        'payload': payload,
    })[:16]
    leak_terms = label_leak_terms(payload)
    return {
        'evidence_id': evidence_id,
        'ledger_id': ledger_id,
        'source_kind': source_kind,
        'sender': sender,
        'topic': topic,
        'payload': dict(payload),
        'priority': _payload_priority(payload),
        'stale_duplicate': evidence_id in processed,
        'leak_terms': leak_terms,
        'label_clean': not leak_terms,
    }


def _decision(kind: str, chosen: dict[str, Any], reason: str) -> dict[str, Any]:
    payload = dict(chosen.get('payload') or {})
    selected = _selected_experiment_from_payload(payload, kind)
    blockers = []
    if kind == 'repair_missing_evidence':
        blockers.append(payload.get('plan_reason') or payload.get('summary') or reason)
    return {
        'decision_kind': kind,
        'priority_rank': {
            'run_next_safe_experiment': 1,
            'repair_missing_evidence': 2,
            'defer_with_cross_module_advisory': 3,
        }[kind],
        'reason': reason,
        'chosen_evidence': {
            'evidence_id': chosen.get('evidence_id'),
            'sender': chosen.get('sender'),
            'classification': chosen.get('classification'),
            'topic': chosen.get('topic'),
        },
        'selected_experiment': selected,
        'expected_transfer_signal': selected.get('expected_transfer_signal'),
        'unresolved_blockers': blockers,
        'should_export_response': True,
    }


def _selected_experiment_from_payload(
    payload: dict[str, Any],
    decision_kind: str,
) -> dict[str, Any]:
    if decision_kind == 'run_next_safe_experiment':
        return {
            'experiment_kind': payload.get('experiment_kind')
            or 'abstraction_transfer_probe',
            'action_kind': payload.get('action_kind')
            or 'non_final_abstraction_transfer_campaign',
            'world': payload.get('world') or 'hidden_procedural',
            'probe': payload.get('probe') or 'abstraction_transfer_probe',
            'runs_final': False,
            'command': payload.get('command'),
            'expected_transfer_signal': (
                'compressed abstraction should improve held-out residuals without label leaks'
            ),
        }
    if decision_kind == 'repair_missing_evidence':
        return {
            'experiment_kind': 'missing_evidence_repair',
            'action_kind': 'collect_label_clean_budget_or_holdout_evidence',
            'world': 'not_selected',
            'probe': 'evidence_repair',
            'runs_final': False,
            'expected_transfer_signal': 'repair should unblock a later falsifiable transfer probe',
        }
    return {
        'experiment_kind': 'cross_module_advisory_review',
        'action_kind': 'defer_until_runnable_request_or_repair_evidence',
        'world': 'not_selected',
        'probe': 'advisory_review',
        'runs_final': False,
        'expected_transfer_signal': 'advisory should sharpen the next runnable request',
    }


def _highest_priority(items: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(
        items,
        key=lambda item: (
            float(item.get('priority', 0.0) or 0.0),
            item.get('sender') or '',
            item.get('evidence_id') or '',
        ),
        reverse=True,
    )[0]


def _payload_priority(payload: dict[str, Any]) -> float:
    for key in ('priority', 'selection_score'):
        if payload.get(key) is not None:
            try:
                return float(payload.get(key))
            except (TypeError, ValueError):
                return 0.5
    basis = dict(payload.get('selection_basis') or {})
    if basis.get('request_priority') is not None:
        try:
            return float(basis.get('request_priority'))
        except (TypeError, ValueError):
            return 0.5
    return 0.5


def _classification_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        key = str(item.get('classification') or 'unknown')
        counts[key] = counts.get(key, 0) + 1
    return counts


def _ledger_fallback_id(ledger: dict[str, Any]) -> str:
    return 'ledger_' + stable_digest(ledger)[:16]


def _message_id(message: dict[str, Any]) -> str:
    body = dict(message.get('body') or {})
    evidence = dict(message.get('evidence') or {})
    explicit = (
        body.get('message_id')
        or body.get('ledger_id')
        or evidence.get('message_id')
    )
    if explicit:
        return str(explicit)
    return 'msg_' + stable_digest(message)[:16]


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
