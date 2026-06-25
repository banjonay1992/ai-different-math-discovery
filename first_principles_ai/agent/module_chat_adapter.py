"""Plain-data module chat adapter for AI Different.

The orchestrator chat bus is treated as an external JSON contract. This module
does not import the orchestrator project; it only validates and emits messages
with sender, recipient, topic, body, evidence, and tags.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ALLOWED_PARTICIPANTS = {
    'ai_different',
    'funfun',
    'language_model_2',
    'code_module',
    'orchestrator',
    'broadcast',
}

HUMAN_LABEL_TERMS = (
    'gravity',
    'vortex',
    'repulsion',
    'sideways_wind',
    'inverse_square',
)

REQUEST_TERMS = (
    'abstraction',
    'transfer',
    'followup',
    'follow-up',
    'experiment',
    'non_final',
)


def validate_participant(participant: str) -> str:
    value = str(participant or '')
    if value not in ALLOWED_PARTICIPANTS:
        raise ValueError(f'unknown module-chat participant: {value}')
    return value


def build_module_chat_message(
    *,
    sender: str,
    recipient: str,
    topic: str,
    body: dict[str, Any],
    evidence: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise ValueError('module-chat body must be an object')
    if evidence is not None and not isinstance(evidence, dict):
        raise ValueError('module-chat evidence must be an object')
    if tags is not None and (
        not isinstance(tags, list)
        or not all(isinstance(tag, str) for tag in tags)
    ):
        raise ValueError('module-chat tags must be a list of strings')
    message = {
        'sender': validate_participant(sender),
        'recipient': validate_participant(recipient),
        'topic': _require_text(topic, 'topic'),
        'body': dict(body),
        'evidence': dict(evidence or {}),
        'tags': [str(tag) for tag in list(tags or [])],
    }
    return validate_module_chat_message(message)


def validate_module_chat_message(message: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(message, dict):
        raise ValueError('module-chat message must be a JSON object')
    for field in ('sender', 'recipient', 'topic', 'body', 'evidence', 'tags'):
        if field not in message:
            raise ValueError(f'module-chat message missing {field}')
    sender = validate_participant(str(message.get('sender') or ''))
    recipient = validate_participant(str(message.get('recipient') or ''))
    topic = _require_text(message.get('topic'), 'topic')
    body = message.get('body')
    evidence = message.get('evidence')
    tags = message.get('tags')
    if not isinstance(body, dict):
        raise ValueError('module-chat body must be an object')
    if not isinstance(evidence, dict):
        raise ValueError('module-chat evidence must be an object')
    if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
        raise ValueError('module-chat tags must be a list of strings')
    return {
        'sender': sender,
        'recipient': recipient,
        'topic': topic,
        'body': dict(body),
        'evidence': dict(evidence),
        'tags': list(tags),
    }


def export_capsule_chat_message(
    capsule: dict[str, Any],
    *,
    recipient: str = 'orchestrator',
    topic: str = 'ai_different.status_capsule',
    inbox_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    latest = dict(capsule.get('latest_verified_abstraction_transfer_result') or {})
    project_boundary = dict(capsule.get('project_owned_boundary') or {})
    runtime_memory = dict(capsule.get('runtime_memory') or {})
    chosen_request = choose_next_non_final_request(
        capsule,
        inbox_summary or {'experiment_requests': [], 'handoff_questions': []},
    )
    body = {
        'module': capsule.get('module', 'AI Different'),
        'capsule_kind': capsule.get('capsule_kind'),
        'current_capabilities': list(capsule.get('current_capabilities') or []),
        'latest_verified_abstraction_transfer_result': latest,
        'known_weak_spots': list(capsule.get('known_weak_spots') or []),
        'safe_handoff_questions': list(capsule.get('safe_handoff_questions') or []),
        'next_non_final_experiment': dict(capsule.get('next_non_final_experiment') or {}),
        'selected_chat_request': chosen_request,
        'project_owned_boundary': project_boundary,
        'runtime_memory': runtime_memory,
        'chat_bridge_available': True,
    }
    body['label_clean'] = not label_leak_terms({
        'latest_transfer': latest.get('agent_facing_evidence', {}),
        'selected_chat_request': chosen_request,
    })
    evidence = {
        'capsule_schema_version': capsule.get('schema_version'),
        'capsule_kind': capsule.get('capsule_kind'),
        'evidence_gates': list(capsule.get('evidence_gates') or []),
        'latest_verified_abstraction_transfer_result': latest,
        'latest_abstraction_transfer_label_clean': bool(
            latest.get('label_clean', True)
        ),
        'chat_bridge_available': True,
        'runtime_memory': runtime_memory,
        'project_owned_boundary': project_boundary,
        'inbox_summary': inbox_summary or {},
    }
    tags = [
        'ai_different',
        'status_capsule',
        'module_chat',
        'abstraction_transfer',
        'label_clean' if body['label_clean'] else 'label_review_needed',
    ]
    return build_module_chat_message(
        sender='ai_different',
        recipient=recipient,
        topic=topic,
        body=body,
        evidence=evidence,
        tags=tags,
    )


def read_module_chat_inbox(
    path: str | Path | None,
    *,
    participant: str = 'ai_different',
) -> dict[str, Any]:
    validate_participant(participant)
    if not path:
        return {
            'path': None,
            'messages': [],
            'invalid_messages': [],
            'handoff_questions': [],
            'experiment_requests': [],
            'evidence_notes': [],
        }
    inbox_path = Path(path)
    messages = []
    invalid_messages = []
    if not inbox_path.exists():
        return {
            'path': str(inbox_path),
            'messages': [],
            'invalid_messages': [],
            'handoff_questions': [],
            'experiment_requests': [],
            'evidence_notes': [],
        }
    with inbox_path.open('r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                message = validate_module_chat_message(json.loads(text))
            except (json.JSONDecodeError, ValueError) as error:
                invalid_messages.append({
                    'line': line_number,
                    'error': str(error),
                    'raw': text,
                })
                continue
            if message['recipient'] not in {participant, 'broadcast'}:
                continue
            messages.append(message)
    return {
        'path': str(inbox_path),
        'messages': messages,
        'invalid_messages': invalid_messages,
        'handoff_questions': extract_handoff_questions(messages),
        'experiment_requests': extract_experiment_requests(messages),
        'evidence_notes': extract_evidence_notes(messages),
    }


def read_module_chat_log(
    path: str | Path | None,
    *,
    participant: str = 'ai_different',
) -> dict[str, Any]:
    """Read a JSONL bus log, keeping inbound and prior outgoing AI Different rows."""
    validate_participant(participant)
    if not path:
        return _module_chat_log_summary(None, [], [], participant)
    log_path = Path(path)
    messages = []
    invalid_messages = []
    if not log_path.exists():
        return _module_chat_log_summary(str(log_path), [], [], participant)
    with log_path.open('r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                message = validate_module_chat_message(json.loads(text))
            except (json.JSONDecodeError, ValueError) as error:
                invalid_messages.append({
                    'line': line_number,
                    'error': str(error),
                    'raw': text,
                })
                continue
            if (
                message['recipient'] in {participant, 'broadcast'}
                or message['sender'] == participant
            ):
                messages.append(message)
    return _module_chat_log_summary(
        str(log_path),
        messages,
        invalid_messages,
        participant,
    )


def module_chat_summary_from_messages(
    messages: list[dict[str, Any]],
    *,
    path: str | None = None,
    invalid_messages: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        'path': path,
        'messages': list(messages),
        'invalid_messages': list(invalid_messages or []),
        'handoff_questions': extract_handoff_questions(messages),
        'experiment_requests': extract_experiment_requests(messages),
        'evidence_notes': extract_evidence_notes(messages),
    }


def load_rolling_family_memory(path: str | Path | None) -> dict[str, Any]:
    if not path or not Path(path).exists():
        return empty_rolling_family_memory()
    with Path(path).open('r', encoding='utf-8') as handle:
        memory = json.load(handle)
    return validate_rolling_family_memory(memory)


def write_rolling_family_memory(path: str | Path, memory: dict[str, Any]) -> Path:
    validated = validate_rolling_family_memory(memory)
    memory_path = Path(path)
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    memory_path.write_text(
        json.dumps(validated, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    return memory_path


def empty_rolling_family_memory() -> dict[str, Any]:
    return {
        'schema_version': 1,
        'memory_kind': 'ai_different.rolling_family_response_memory',
        'processed_message_ids': [],
        'outgoing_response_ids': [],
        'response_records': [],
        'latest': {},
    }


def validate_rolling_family_memory(memory: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(memory, dict):
        raise ValueError('rolling family memory must be a JSON object')
    if memory.get('memory_kind') != 'ai_different.rolling_family_response_memory':
        raise ValueError('rolling family memory has wrong memory_kind')
    processed = memory.get('processed_message_ids')
    outgoing = memory.get('outgoing_response_ids')
    records = memory.get('response_records')
    latest = memory.get('latest', {})
    if not isinstance(processed, list) or not all(
        isinstance(item, str) for item in processed
    ):
        raise ValueError('rolling family memory processed_message_ids must be strings')
    if not isinstance(outgoing, list) or not all(
        isinstance(item, str) for item in outgoing
    ):
        raise ValueError('rolling family memory outgoing_response_ids must be strings')
    if not isinstance(records, list) or not all(
        isinstance(item, dict) for item in records
    ):
        raise ValueError('rolling family memory response_records must be objects')
    if not isinstance(latest, dict):
        raise ValueError('rolling family memory latest must be an object')
    validated = {
        'schema_version': int(memory.get('schema_version', 1) or 1),
        'memory_kind': 'ai_different.rolling_family_response_memory',
        'processed_message_ids': _unique_strings(processed),
        'outgoing_response_ids': _unique_strings(outgoing),
        'response_records': list(records),
        'latest': dict(latest),
    }
    if memory.get('memory_hash'):
        validated['memory_hash'] = str(memory.get('memory_hash'))
    return validated


def rolling_unprocessed_inbound_messages(
    log_summary: dict[str, Any],
    rolling_memory: dict[str, Any],
    *,
    participant: str = 'ai_different',
) -> dict[str, Any]:
    processed = set(rolling_memory.get('processed_message_ids') or [])
    inbound = list(log_summary.get('inbound_messages') or [])
    new_messages = [
        message for message in inbound
        if module_chat_message_id(message) not in processed
    ]
    skipped_messages = [
        message for message in inbound
        if module_chat_message_id(message) in processed
    ]
    outgoing = [
        message for message in log_summary.get('outgoing_messages') or []
        if message.get('sender') == participant
    ]
    return {
        'new_messages': new_messages,
        'skipped_messages': skipped_messages,
        'new_message_ids': [module_chat_message_id(message) for message in new_messages],
        'skipped_message_ids': [
            module_chat_message_id(message) for message in skipped_messages
        ],
        'prior_outgoing_response_ids': [
            module_chat_message_id(message) for message in outgoing
        ],
    }


def append_rolling_family_record(
    rolling_memory: dict[str, Any],
    *,
    processed_messages: list[dict[str, Any]],
    ledger: dict[str, Any],
    response_message: dict[str, Any] | None,
    skipped_count: int,
    runtime_memory_hash_state: dict[str, Any],
    observed_outgoing_response_ids: list[str] | None = None,
) -> dict[str, Any]:
    updated = validate_rolling_family_memory(rolling_memory)
    processed_ids = _unique_strings(
        list(updated['processed_message_ids'])
        + [module_chat_message_id(message) for message in processed_messages]
    )
    response_id = (
        module_chat_message_id(response_message)
        if response_message is not None
        else None
    )
    outgoing_ids = _unique_strings(
        list(updated['outgoing_response_ids'])
        + list(observed_outgoing_response_ids or [])
        + ([response_id] if response_id else [])
    )
    record = {
        'record_id': 'record_' + stable_digest({
            'processed_message_ids': [
                module_chat_message_id(message) for message in processed_messages
            ],
            'ledger_id': ledger.get('ledger_id'),
            'response_id': response_id,
        })[:16],
        'processed_message_ids': [
            module_chat_message_id(message) for message in processed_messages
        ],
        'processed_message_count': len(processed_messages),
        'skipped_message_count': skipped_count,
        'response_ledger_id': ledger.get('ledger_id'),
        'response_ledger_hash': ledger.get('ledger_hash'),
        'response_ledger_path': ledger.get('artifact_path'),
        'selected_recipient': ledger.get('selected_recipient'),
        'evidence_counts_by_sender': dict(
            ledger.get('evidence_counts_by_sender') or {}
        ),
        'evidence_rationale': list(ledger.get('evidence_rationale') or []),
        'outcome_or_plan': dict(ledger.get('outcome_or_plan') or {}),
        'runtime_memory_hash_state': dict(runtime_memory_hash_state),
        'runtime_memory_mutated': bool(ledger.get('runtime_memory_mutated')),
        'outgoing_response_id': response_id,
        'label_clean': bool(ledger.get('label_clean')),
    }
    updated['processed_message_ids'] = processed_ids
    updated['outgoing_response_ids'] = outgoing_ids
    updated['response_records'] = list(updated['response_records']) + [record]
    updated['latest'] = {
        'selected_recipient': record['selected_recipient'],
        'evidence_counts_by_sender': record['evidence_counts_by_sender'],
        'outcome_or_plan': record['outcome_or_plan'],
        'response_ledger_id': record['response_ledger_id'],
        'response_ledger_hash': record['response_ledger_hash'],
        'response_ledger_path': record['response_ledger_path'],
        'outgoing_response_id': response_id,
        'label_clean': record['label_clean'],
        'runtime_memory_hash_state': record['runtime_memory_hash_state'],
        'runtime_memory_mutated': record['runtime_memory_mutated'],
    }
    updated['memory_hash'] = stable_digest({
        key: value for key, value in updated.items()
        if key != 'memory_hash'
    })
    return updated


def extract_handoff_questions(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    questions = []
    for message in messages:
        body = dict(message.get('body') or {})
        candidates = []
        if body.get('question'):
            candidates.append(str(body['question']))
        candidates.extend(str(item) for item in list(body.get('questions') or []))
        for question in candidates:
            questions.append({
                'message_id': module_chat_message_id(message),
                'sender': message.get('sender'),
                'topic': message.get('topic'),
                'question': question,
                'tags': list(message.get('tags') or []),
                'label_clean': not label_leak_terms({'question': question}),
            })
    return questions


def extract_experiment_requests(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    requests = []
    for message in messages:
        body = dict(message.get('body') or {})
        text = _message_search_text(message)
        request_kind = str(body.get('request_kind') or '')
        explicit_experiment = str(body.get('experiment_kind') or '')
        topic = str(message.get('topic') or '')
        tags = list(message.get('tags') or [])
        is_evidence_note = (
            body.get('note_kind') == 'evidence'
            or topic.startswith('evidence.')
            or 'evidence' in tags
        )
        has_explicit_request = bool(request_kind or explicit_experiment)
        if is_evidence_note and not has_explicit_request:
            continue
        asks_for_abstraction = (
            request_kind == 'abstraction_transfer_followup'
            or explicit_experiment == 'abstraction_transfer_probe'
            or all(term in text for term in ('abstraction', 'transfer'))
        )
        if not asks_for_abstraction:
            continue
        request = {
            'message_id': module_chat_message_id(message),
            'sender': message.get('sender'),
            'topic': message.get('topic'),
            'request_kind': request_kind or 'abstraction_transfer_followup',
            'experiment_kind': explicit_experiment or 'abstraction_transfer_probe',
            'question': body.get('question'),
            'outcome_mode': body.get('outcome_mode'),
            'priority': float(body.get('priority', 0.5) or 0.5),
            'tags': list(message.get('tags') or []),
            'label_clean': not label_leak_terms(body),
        }
        requests.append(request)
    requests.sort(
        key=lambda item: (
            item['experiment_kind'] == 'abstraction_transfer_probe',
            item['priority'],
            item['topic'] or '',
        ),
        reverse=True,
    )
    return requests


def extract_evidence_notes(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    notes = []
    for message in messages:
        evidence = dict(message.get('evidence') or {})
        body = dict(message.get('body') or {})
        tags = list(message.get('tags') or [])
        topic = str(message.get('topic') or '')
        has_evidence_signal = (
            'evidence' in topic
            or 'evidence' in tags
            or body.get('note_kind') == 'evidence'
        )
        if not has_evidence_signal:
            continue
        note = {
            'message_id': module_chat_message_id(message),
            'sender': message.get('sender'),
            'topic': topic,
            'summary': body.get('summary') or body.get('note') or body.get('question'),
            'evidence': evidence,
            'tags': tags,
            'priority': _evidence_priority(message),
            'why_it_matters': _evidence_rationale(message),
            'label_clean': not label_leak_terms({
                'summary': body.get('summary') or body.get('note'),
                'evidence': evidence,
            }),
        }
        notes.append(note)
    return notes


def choose_next_non_final_request(
    capsule: dict[str, Any],
    inbox_summary: dict[str, Any],
) -> dict[str, Any]:
    requests = list(inbox_summary.get('experiment_requests') or [])
    next_experiment = dict(capsule.get('next_non_final_experiment') or {})
    if requests:
        request = dict(requests[0])
        command = next_experiment.get('command') or (
            'python3 first_principles_ai/main.py '
            '--abstraction-transfer-campaign '
            '--theory-memory-file tmp/theory-memory.json'
        )
        outcome_mode = request.get('outcome_mode')
        if outcome_mode in {'confirmed', 'weak', 'absent'} and (
            '--abstraction-transfer-outcome' not in command
        ):
            command = f'{command} --abstraction-transfer-outcome {outcome_mode}'
        return {
            'source': 'inbox',
            'requested_by': request.get('sender'),
            'request_topic': request.get('topic'),
            'action_kind': 'non_final_abstraction_transfer_campaign',
            'experiment_kind': request.get('experiment_kind'),
            'command': command,
            'reason': request.get('question') or 'inbox requested abstraction transfer follow-up',
            'runs_final': False,
            'outcome_mode': outcome_mode if outcome_mode in {'confirmed', 'weak', 'absent'} else None,
            'label_clean': bool(request.get('label_clean', True)),
        }
    if next_experiment:
        return {
            'source': 'capsule',
            **next_experiment,
            'label_clean': True,
        }
    return {
        'source': 'default',
        'action_kind': 'non_final_abstraction_transfer_campaign',
        'command': (
            'python3 first_principles_ai/main.py '
            '--abstraction-transfer-campaign '
            '--theory-memory-file tmp/theory-memory.json'
        ),
        'reason': 'default safe abstraction-transfer follow-up',
        'runs_final': False,
        'label_clean': True,
    }


def choose_module_family_followup(
    capsule: dict[str, Any],
    inbox_summary: dict[str, Any],
) -> dict[str, Any]:
    """Choose a safe abstraction-transfer follow-up from a richer module inbox."""
    requests = list(inbox_summary.get('experiment_requests') or [])
    evidence_notes = sorted(
        list(inbox_summary.get('evidence_notes') or []),
        key=lambda note: (float(note.get('priority', 0.0) or 0.0), note.get('sender') or ''),
        reverse=True,
    )
    evidence_priority_total = round(
        sum(float(note.get('priority', 0.0) or 0.0) for note in evidence_notes),
        4,
    )
    cheap_no_save = any(_note_supports_no_save_run(note) for note in evidence_notes)
    typed_discovery = any(_note_supports_typed_discovery(note) for note in evidence_notes)
    if requests:
        scored = []
        for request in requests:
            request_score = float(request.get('priority', 0.5) or 0.5)
            if cheap_no_save:
                request_score += 0.1
            if typed_discovery:
                request_score += 0.05
            scored.append((round(request_score, 4), request))
        scored.sort(
            key=lambda item: (
                item[0],
                item[1].get('sender') == 'language_model_2',
                item[1].get('topic') or '',
            ),
            reverse=True,
        )
        selected = choose_next_non_final_request(
            capsule,
            {'experiment_requests': [scored[0][1]], 'handoff_questions': []},
        )
        selected['selection_score'] = scored[0][0]
        selected['selection_basis'] = {
            'request_priority': float(scored[0][1].get('priority', 0.5) or 0.5),
            'evidence_priority_total': evidence_priority_total,
            'cheap_no_save_supported': cheap_no_save,
            'typed_discovery_supported': typed_discovery,
            'fallback_used': False,
        }
        return selected
    selected = choose_next_non_final_request(
        capsule,
        {'experiment_requests': [], 'handoff_questions': []},
    )
    selected['source'] = 'deterministic_fallback'
    selected['selection_score'] = 0.0
    selected['selection_basis'] = {
        'request_priority': 0.0,
        'evidence_priority_total': evidence_priority_total,
        'cheap_no_save_supported': cheap_no_save,
        'typed_discovery_supported': typed_discovery,
        'fallback_used': True,
        'fallback_reason': 'no runnable abstraction-transfer request in inbox',
    }
    selected['runs_final'] = False
    selected['label_clean'] = bool(selected.get('label_clean', True))
    return selected


def choose_module_family_recipient(
    inbox_summary: dict[str, Any],
    selected_request: dict[str, Any],
    requested_recipient: str = 'orchestrator',
) -> str:
    if requested_recipient != 'auto':
        return validate_participant(requested_recipient)
    requested_by = selected_request.get('requested_by')
    if requested_by in {'language_model_2', 'orchestrator'}:
        return validate_participant(str(requested_by))
    senders = {message.get('sender') for message in inbox_summary.get('messages') or []}
    if 'language_model_2' in senders:
        return 'language_model_2'
    if 'orchestrator' in senders:
        return 'orchestrator'
    return 'broadcast'


def build_module_family_response_ledger(
    capsule: dict[str, Any],
    inbox_summary: dict[str, Any],
    *,
    selected_recipient: str = 'orchestrator',
    response_mode: str = 'plan',
    campaign_summary: dict[str, Any] | None = None,
    ran_campaign: bool = False,
    runtime_memory_hash_state: dict[str, Any] | None = None,
    ledger_path: str | Path | None = None,
) -> dict[str, Any]:
    selected = choose_module_family_followup(capsule, inbox_summary)
    evidence_notes = sorted(
        list(inbox_summary.get('evidence_notes') or []),
        key=lambda note: (float(note.get('priority', 0.0) or 0.0), note.get('sender') or ''),
        reverse=True,
    )
    message_records = [
        {
            'message_id': module_chat_message_id(message),
            'sender': message.get('sender'),
            'recipient': message.get('recipient'),
            'topic': message.get('topic'),
            'tags': list(message.get('tags') or []),
            'label_clean': not label_leak_terms({
                'topic': message.get('topic'),
                'body': message.get('body'),
                'evidence': message.get('evidence'),
            }),
        }
        for message in inbox_summary.get('messages') or []
    ]
    if ran_campaign and campaign_summary:
        transfer = dict(campaign_summary.get('transfer_result') or {})
        outcome_or_plan = {
            'mode': 'campaign_result',
            'ran_campaign': True,
            'runs_final': False,
            'campaign_run_kind': campaign_summary.get('run_kind'),
            'selected_plan': dict(campaign_summary.get('selected_plan') or {}),
            'outcome': dict(transfer.get('outcome') or {}),
            'abstraction_discovery_evidence': dict(
                campaign_summary.get('abstraction_discovery_evidence') or {}
            ),
            'memory_saved': False,
        }
    else:
        outcome_or_plan = {
            'mode': 'plan',
            'ran_campaign': False,
            'runs_final': False,
            'selected_plan': selected,
            'memory_saved': False,
            'plan_reason': _plan_reason(response_mode, selected, evidence_notes),
        }
    evidence_counts = _count_by_sender(evidence_notes)
    selected_evidence = evidence_notes[:3]
    runtime_hash = dict(runtime_memory_hash_state or {})
    project_boundary = dict(capsule.get('project_owned_boundary') or {})
    runtime_memory = dict(capsule.get('runtime_memory') or {})
    label_payload = {
        'selected_request': selected,
        'selected_evidence': selected_evidence,
        'outcome_or_plan': outcome_or_plan,
    }
    leak_terms = label_leak_terms(label_payload)
    ledger = {
        'schema_version': 1,
        'ledger_kind': 'ai_different.module_family_response_ledger',
        'module': capsule.get('module', 'AI Different'),
        'three_module_response_available': _three_module_response_available(
            inbox_summary
        ),
        'source_inbox_path': inbox_summary.get('path'),
        'message_records': message_records,
        'invalid_message_count': len(inbox_summary.get('invalid_messages') or []),
        'sender_counts': _count_by_sender(inbox_summary.get('messages') or []),
        'evidence_counts_by_sender': evidence_counts,
        'selected_request': selected,
        'selected_evidence': selected_evidence,
        'evidence_rationale': [
            {
                'message_id': note.get('message_id'),
                'sender': note.get('sender'),
                'topic': note.get('topic'),
                'priority': note.get('priority'),
                'why_it_matters': note.get('why_it_matters'),
            }
            for note in selected_evidence
        ],
        'selected_recipient': selected_recipient,
        'outcome_or_plan': outcome_or_plan,
        'evidence_gates': list(capsule.get('evidence_gates') or []),
        'run_decision': {
            'requested_mode': response_mode,
            'should_run_no_save_campaign': should_run_no_save_campaign(
                response_mode,
                selected,
                evidence_notes,
            ),
            'cheap_no_save_supported': any(
                _note_supports_no_save_run(note) for note in evidence_notes
            ),
            'runnable_request_present': selected.get('source') == 'inbox',
        },
        'project_owned_boundary': project_boundary,
        'runtime_memory': runtime_memory,
        'runtime_memory_hash_state': runtime_hash,
        'runtime_memory_mutated': not bool(runtime_hash.get('unchanged', True)),
        'label_clean': not leak_terms,
        'leak_terms': leak_terms,
    }
    if ledger_path is not None:
        ledger['artifact_path'] = str(Path(ledger_path))
    ledger['ledger_id'] = 'ledger_' + stable_digest({
        key: value for key, value in ledger.items()
        if key != 'ledger_id'
    })[:16]
    ledger['ledger_hash'] = stable_digest(ledger)
    return ledger


def export_module_family_response_message(
    ledger: dict[str, Any],
    *,
    recipient: str | None = None,
    topic: str = 'ai_different.module_family_response',
) -> dict[str, Any]:
    selected_recipient = validate_participant(
        recipient or ledger.get('selected_recipient') or 'orchestrator'
    )
    body = {
        'module': ledger.get('module', 'AI Different'),
        'response_kind': 'module_family_coordination_response',
        'three_module_response_available': bool(
            ledger.get('three_module_response_available')
        ),
        'ledger_id': ledger.get('ledger_id'),
        'ledger_path': ledger.get('artifact_path'),
        'ledger_hash': ledger.get('ledger_hash'),
        'selected_recipient': selected_recipient,
        'selected_chat_request': dict(ledger.get('selected_request') or {}),
        'selected_evidence': list(ledger.get('selected_evidence') or []),
        'evidence_counts_by_sender': dict(
            ledger.get('evidence_counts_by_sender') or {}
        ),
        'outcome_or_plan': dict(ledger.get('outcome_or_plan') or {}),
        'project_owned_boundary': dict(ledger.get('project_owned_boundary') or {}),
        'runtime_memory': dict(ledger.get('runtime_memory') or {}),
        'runtime_memory_hash_state': dict(
            ledger.get('runtime_memory_hash_state') or {}
        ),
        'runtime_memory_mutated': bool(ledger.get('runtime_memory_mutated')),
        'label_clean': bool(ledger.get('label_clean')),
        'leak_terms': list(ledger.get('leak_terms') or []),
    }
    evidence = {
        'ledger_id': body['ledger_id'],
        'ledger_path': body['ledger_path'],
        'ledger_hash': body['ledger_hash'],
        'evidence_gates': list(ledger.get('evidence_gates') or []),
        'evidence_counts_by_sender': body['evidence_counts_by_sender'],
        'selected_evidence': body['selected_evidence'],
        'outcome_or_plan': body['outcome_or_plan'],
        'project_owned_boundary': body['project_owned_boundary'],
        'runtime_memory_hash_state': body['runtime_memory_hash_state'],
        'runtime_memory_mutated': body['runtime_memory_mutated'],
        'label_clean': body['label_clean'],
    }
    tags = [
        'ai_different',
        'module_chat',
        'module_family_response',
        'abstraction_transfer',
        'campaign_run'
        if body['outcome_or_plan'].get('ran_campaign')
        else 'plan_only',
        'label_clean' if body['label_clean'] else 'label_review_needed',
    ]
    return build_module_chat_message(
        sender='ai_different',
        recipient=selected_recipient,
        topic=topic,
        body=body,
        evidence=evidence,
        tags=tags,
    )


def write_response_ledger(path: str | Path, ledger: dict[str, Any]) -> Path:
    ledger_path = Path(path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    return ledger_path


def load_response_ledger(path: str | Path) -> dict[str, Any]:
    with Path(path).open('r', encoding='utf-8') as handle:
        ledger = json.load(handle)
    if not isinstance(ledger, dict):
        raise ValueError('response ledger must be a JSON object')
    required = {'ledger_kind', 'ledger_id', 'ledger_hash', 'message_records'}
    missing = sorted(required - set(ledger))
    if missing:
        raise ValueError(f'response ledger missing {", ".join(missing)}')
    return ledger


def should_run_no_save_campaign(
    response_mode: str,
    selected_request: dict[str, Any],
    evidence_notes: list[dict[str, Any]],
) -> bool:
    return (
        response_mode == 'run'
        and selected_request.get('source') == 'inbox'
        and any(_note_supports_no_save_run(note) for note in evidence_notes)
    )


def build_chat_driven_response_payload(
    capsule: dict[str, Any],
    inbox_summary: dict[str, Any],
    *,
    campaign_summary: dict[str, Any] | None = None,
    ran_campaign: bool = False,
) -> dict[str, Any]:
    selected = choose_next_non_final_request(capsule, inbox_summary)
    evidence_notes = list(inbox_summary.get('evidence_notes') or [])
    latest = dict(capsule.get('latest_verified_abstraction_transfer_result') or {})
    runtime_memory = dict(capsule.get('runtime_memory') or {})
    project_boundary = dict(capsule.get('project_owned_boundary') or {})
    if ran_campaign and campaign_summary:
        transfer = dict(campaign_summary.get('transfer_result') or {})
        outcome_or_plan = {
            'mode': 'campaign_result',
            'ran_campaign': True,
            'runs_final': False,
            'campaign_run_kind': campaign_summary.get('run_kind'),
            'selected_plan': dict(campaign_summary.get('selected_plan') or {}),
            'outcome': dict(transfer.get('outcome') or {}),
            'abstraction_discovery_evidence': dict(
                campaign_summary.get('abstraction_discovery_evidence') or {}
            ),
            'memory_saved': False,
        }
    else:
        outcome_or_plan = {
            'mode': 'plan',
            'ran_campaign': False,
            'runs_final': False,
            'selected_plan': selected,
            'memory_saved': False,
        }
    label_payload = {
        'selected': selected,
        'outcome_or_plan': outcome_or_plan,
        'latest_transfer': latest.get('agent_facing_evidence', {}),
        'evidence_notes': evidence_notes,
    }
    leak_terms = label_leak_terms(label_payload)
    return {
        'module': capsule.get('module', 'AI Different'),
        'response_kind': 'abstraction_transfer_response',
        'chat_driven_response_available': True,
        'selected_chat_request': selected,
        'code_evidence_notes': [
            note for note in evidence_notes
            if note.get('sender') == 'code_module'
        ],
        'evidence_notes': evidence_notes,
        'outcome_or_plan': outcome_or_plan,
        'latest_verified_abstraction_transfer_result': latest,
        'evidence_gates': list(capsule.get('evidence_gates') or []),
        'project_owned_boundary': project_boundary,
        'runtime_memory': runtime_memory,
        'runtime_memory_mutated': False,
        'label_clean': not leak_terms,
        'leak_terms': leak_terms,
    }


def export_chat_driven_response_message(
    capsule: dict[str, Any],
    inbox_summary: dict[str, Any],
    *,
    campaign_summary: dict[str, Any] | None = None,
    ran_campaign: bool = False,
    recipient: str = 'orchestrator',
    topic: str = 'ai_different.abstraction_transfer_response',
) -> dict[str, Any]:
    body = build_chat_driven_response_payload(
        capsule,
        inbox_summary,
        campaign_summary=campaign_summary,
        ran_campaign=ran_campaign,
    )
    evidence = {
        'chat_bridge_available': True,
        'response_kind': body['response_kind'],
        'evidence_gates': body['evidence_gates'],
        'latest_verified_abstraction_transfer_result': (
            body['latest_verified_abstraction_transfer_result']
        ),
        'outcome_or_plan': body['outcome_or_plan'],
        'inbox_summary': {
            'message_count': len(inbox_summary.get('messages') or []),
            'invalid_message_count': len(inbox_summary.get('invalid_messages') or []),
            'experiment_request_count': len(
                inbox_summary.get('experiment_requests') or []
            ),
            'evidence_note_count': len(inbox_summary.get('evidence_notes') or []),
        },
        'project_owned_boundary': body['project_owned_boundary'],
        'runtime_memory': body['runtime_memory'],
        'runtime_memory_mutated': False,
        'label_clean': body['label_clean'],
    }
    tags = [
        'ai_different',
        'module_chat',
        'abstraction_transfer',
        'response_loop',
        'campaign_run' if ran_campaign else 'plan_only',
        'label_clean' if body['label_clean'] else 'label_review_needed',
    ]
    return build_module_chat_message(
        sender='ai_different',
        recipient=recipient,
        topic=topic,
        body=body,
        evidence=evidence,
        tags=tags,
    )


def module_chat_message_id(message: dict[str, Any]) -> str:
    body = dict(message.get('body') or {})
    evidence = dict(message.get('evidence') or {})
    explicit = (
        body.get('message_id')
        or body.get('id')
        or evidence.get('message_id')
        or evidence.get('id')
    )
    if explicit:
        return str(explicit)
    return 'msg_' + stable_digest({
        'sender': message.get('sender'),
        'recipient': message.get('recipient'),
        'topic': message.get('topic'),
        'body': body,
        'evidence': evidence,
        'tags': list(message.get('tags') or []),
    })[:16]


def stable_digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
    ).hexdigest()


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _evidence_priority(message: dict[str, Any]) -> float:
    body = dict(message.get('body') or {})
    evidence = dict(message.get('evidence') or {})
    explicit = body.get('priority', evidence.get('priority'))
    if explicit is not None:
        return round(_safe_float(explicit, 0.5), 4)
    sender = message.get('sender')
    topic = str(message.get('topic') or '').lower()
    tags = {str(tag).lower() for tag in list(message.get('tags') or [])}
    text = _message_search_text(message)
    if sender == 'code_module' and (
        'budget' in topic
        or 'handoff' in topic
        or 'local_safe' in tags
        or 'cheap' in text
    ):
        return 0.9
    if sender == 'funfun' and (
        'typed' in topic
        or 'typed_discovery' in tags
        or 'capability' in topic
        or 'typed' in text
    ):
        return 0.82
    if sender == 'language_model_2':
        return 0.65
    return 0.5


def _evidence_rationale(message: dict[str, Any]) -> str:
    sender = message.get('sender')
    topic = str(message.get('topic') or '').lower()
    text = _message_search_text(message)
    if sender == 'code_module':
        return 'bounds whether the follow-up can run cheaply without saving memory'
    if sender == 'funfun':
        return 'adds typed-discovery evidence for abstraction transfer across modules'
    if sender == 'language_model_2':
        return 'carries coordination intent and the safest user-facing question'
    if 'budget' in topic or 'cheap' in text:
        return 'helps decide whether to run or plan only'
    return 'supports the response ledger evidence trail'


def _note_supports_no_save_run(note: dict[str, Any]) -> bool:
    evidence = dict(note.get('evidence') or {})
    tags = {str(tag).lower() for tag in list(note.get('tags') or [])}
    text = json.dumps(note, sort_keys=True).lower()
    mutates = evidence.get('mutates_runtime_memory')
    return (
        note.get('sender') == 'code_module'
        and (
            'local_safe' in tags
            or 'cheap' in text
            or 'subsecond' in text
        )
        and mutates is not True
    )


def _note_supports_typed_discovery(note: dict[str, Any]) -> bool:
    text = json.dumps(note, sort_keys=True).lower()
    return note.get('sender') == 'funfun' and (
        'typed' in text or 'capability' in text or 'multiplicative' in text
    )


def _plan_reason(
    response_mode: str,
    selected_request: dict[str, Any],
    evidence_notes: list[dict[str, Any]],
) -> str:
    if selected_request.get('source') != 'inbox':
        return 'plan_only_no_runnable_inbox_request'
    if response_mode != 'run':
        return 'plan_only_requested'
    if not any(_note_supports_no_save_run(note) for note in evidence_notes):
        return 'plan_only_missing_cheap_no_save_evidence'
    return 'plan_only_before_campaign_execution'


def _count_by_sender(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        sender = str(item.get('sender') or 'unknown')
        counts[sender] = counts.get(sender, 0) + 1
    return counts


def _three_module_response_available(inbox_summary: dict[str, Any]) -> bool:
    senders = {message.get('sender') for message in inbox_summary.get('messages') or []}
    return {'language_model_2', 'code_module', 'funfun'} <= senders


def _module_chat_log_summary(
    path: str | None,
    messages: list[dict[str, Any]],
    invalid_messages: list[dict[str, Any]],
    participant: str,
) -> dict[str, Any]:
    inbound = [
        message for message in messages
        if message.get('sender') != participant
        and message.get('recipient') in {participant, 'broadcast'}
    ]
    outgoing = [
        message for message in messages
        if message.get('sender') == participant
    ]
    return {
        'path': path,
        'messages': list(messages),
        'inbound_messages': inbound,
        'outgoing_messages': outgoing,
        'invalid_messages': list(invalid_messages),
        'handoff_questions': extract_handoff_questions(inbound),
        'experiment_requests': extract_experiment_requests(inbound),
        'evidence_notes': extract_evidence_notes(inbound),
    }


def _unique_strings(values: list[Any]) -> list[str]:
    seen = set()
    unique = []
    for value in values:
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique


def label_leak_terms(payload: Any) -> list[str]:
    text = json.dumps(payload, sort_keys=True).lower()
    return sorted({term for term in HUMAN_LABEL_TERMS if term in text})


def _require_text(value: Any, field: str) -> str:
    text = str(value or '').strip()
    if not text:
        raise ValueError(f'module-chat {field} must be non-empty')
    return text


def _message_search_text(message: dict[str, Any]) -> str:
    return json.dumps({
        'topic': message.get('topic'),
        'body': message.get('body'),
        'tags': message.get('tags'),
    }, sort_keys=True).lower()
