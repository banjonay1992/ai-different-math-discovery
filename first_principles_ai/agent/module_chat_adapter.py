"""Plain-data module chat adapter for AI Different.

The orchestrator chat bus is treated as an external JSON contract. This module
does not import the orchestrator project; it only validates and emits messages
with sender, recipient, topic, body, evidence, and tags.
"""

from __future__ import annotations

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
            'sender': message.get('sender'),
            'topic': topic,
            'summary': body.get('summary') or body.get('note') or body.get('question'),
            'evidence': evidence,
            'tags': tags,
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
