import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'first_principles_ai'))
sys.path.insert(0, PROJECT_DIR)

from agent.module_chat_adapter import (  # noqa: E402
    append_rolling_family_record,
    build_module_family_response_ledger,
    build_module_chat_message,
    build_chat_driven_response_payload,
    choose_module_family_followup,
    choose_module_family_recipient,
    choose_next_non_final_request,
    export_module_family_response_message,
    export_chat_driven_response_message,
    export_capsule_chat_message,
    load_rolling_family_memory,
    load_response_ledger,
    read_module_chat_inbox,
    read_module_chat_log,
    rolling_unprocessed_inbound_messages,
    validate_module_chat_message,
    validate_participant,
    write_rolling_family_memory,
    write_response_ledger,
)
from agent.status_capsule import build_ai_different_status_capsule  # noqa: E402
from main import (  # noqa: E402
    run_module_chat_export,
    run_module_chat_family_response,
    run_module_chat_response_loop,
    run_module_chat_rolling_family_response,
)


def memory_fixture():
    latest = {
        'theory_kind': 'abstraction:scaled_domain_effect',
        'experiment_kind': 'abstraction_transfer_probe',
        'outcome': 'abstraction_transfer_confirmed',
        'context': 'hidden_procedural',
        'seed': 142,
        'abstraction_key': 'abstraction:scaled_domain_effect:memory',
        'abstraction_kind': 'scaled_domain_effect',
        'compressed_expression': 'effect := domain_weight * relative_basis / scale^p',
        'expected_result': 'compressed expression should reduce residual error',
        'falsifies_if': 'removing a compressed part gives the same held-out error',
    }
    return {
        'discovery_readiness': {
            'readiness_score': 0.81,
            'status': 'nearly_ready',
            'missing_gates': [],
            'recommended_actions': [{
                'action_kind': 'non_final_abstraction_transfer_campaign',
                'reason': 'collect abstraction transfer evidence',
                'command': (
                    'python3 first_principles_ai/main.py '
                    '--abstraction-transfer-campaign '
                    '--theory-memory-file tmp/theory-memory.json'
                ),
                'runs_final': False,
            }],
            'gates': {
                'abstraction_discovery_loop': {
                    'passed': True,
                    'evidence': {
                        'bridge_count': 4,
                        'transfer_outcome_count': 1,
                        'transfer_confirmed_count': 1,
                    },
                    'next_step': 'keep monitoring this gate',
                },
            },
        },
        'rediscovery_goal_progress': {
            'gates': {
                'abstraction_discovery_transfer': {
                    'score': 1.0,
                    'passed': True,
                    'evidence': {'transfer_outcome_count': 1},
                    'next_step': 'keep monitoring this gate',
                },
            },
        },
        'abstraction_discovery_evidence': {
            'record_count': 3,
            'bridge_count': 4,
            'reusable_count': 4,
            'transfer_experiment_count': 4,
            'transfer_outcome_count': 1,
            'transfer_confirmed_count': 1,
            'transfer_weak_count': 0,
            'transfer_absent_count': 0,
            'latest_transfer_outcome': latest,
        },
        'planned_outcomes': [latest],
    }


def capsule_fixture():
    return build_ai_different_status_capsule(
        memory_fixture(),
        git_status_text=' M tmp/theory-memory.json\n',
        runtime_memory_path='tmp/theory-memory.json',
    )


def three_module_messages(include_request=True):
    messages = []
    if include_request:
        messages.append(build_module_chat_message(
            sender='language_model_2',
            recipient='ai_different',
            topic='digest.abstraction_transfer.coordination',
            body={
                'message_id': 'lang-digest-1',
                'request_kind': 'abstraction_transfer_followup',
                'question': 'Use typed evidence to pick a safe transfer follow-up.',
                'outcome_mode': 'weak',
                'priority': 0.7,
            },
            evidence={'digest_kind': 'coordination', 'priority': 0.7},
            tags=['abstraction_transfer', 'coordination'],
        ))
    else:
        messages.append(build_module_chat_message(
            sender='language_model_2',
            recipient='ai_different',
            topic='digest.coordination',
            body={
                'message_id': 'lang-digest-1',
                'question': 'Which safe follow-up should we plan next?',
            },
            evidence={'digest_kind': 'coordination'},
            tags=['coordination'],
        ))
    messages.extend([
        build_module_chat_message(
            sender='code_module',
            recipient='ai_different',
            topic='evidence.abstraction_transfer.budget',
            body={
                'message_id': 'code-budget-1',
                'note_kind': 'evidence',
                'summary': 'No-save abstraction response is cheap enough locally.',
                'priority': 0.95,
            },
            evidence={
                'estimated_runtime': 'subsecond',
                'mutates_runtime_memory': False,
                'priority': 0.95,
            },
            tags=['evidence', 'local_safe', 'budget'],
        ),
        build_module_chat_message(
            sender='funfun',
            recipient='broadcast',
            topic='evidence.typed_discovery.capability',
            body={
                'message_id': 'funfun-typed-1',
                'note_kind': 'evidence',
                'summary': 'Typed discovery can expose reusable multiplicative structure.',
                'priority': 0.82,
            },
            evidence={
                'capability': 'typed_discovery',
                'transfer_relevance': 'cross_surface_abstraction',
                'priority': 0.82,
            },
            tags=['evidence', 'typed_discovery'],
        ),
    ])
    return messages


def two_round_family_log_messages():
    first = three_module_messages()
    second = [
        build_module_chat_message(
            sender='language_model_2',
            recipient='ai_different',
            topic='digest.abstraction_transfer.coordination',
            body={
                'message_id': 'lang-digest-2',
                'request_kind': 'abstraction_transfer_followup',
                'question': 'Plan the next transfer check after the first response.',
                'outcome_mode': 'absent',
                'priority': 0.6,
            },
            evidence={'digest_kind': 'coordination', 'priority': 0.6},
            tags=['abstraction_transfer', 'coordination'],
        ),
        build_module_chat_message(
            sender='code_module',
            recipient='ai_different',
            topic='evidence.abstraction_transfer.budget',
            body={
                'message_id': 'code-budget-2',
                'note_kind': 'evidence',
                'summary': 'Second no-save response remains cheap.',
                'priority': 0.88,
            },
            evidence={
                'estimated_runtime': 'subsecond',
                'mutates_runtime_memory': False,
                'priority': 0.88,
            },
            tags=['evidence', 'local_safe', 'budget'],
        ),
        build_module_chat_message(
            sender='funfun',
            recipient='broadcast',
            topic='evidence.typed_discovery.capability',
            body={
                'message_id': 'funfun-typed-2',
                'note_kind': 'evidence',
                'summary': 'Concept utility memory ranks reusable typed concepts.',
                'priority': 0.76,
            },
            evidence={
                'capability': 'concept_utility_memory',
                'transfer_relevance': 'cross_surface_abstraction',
                'priority': 0.76,
            },
            tags=['evidence', 'typed_discovery'],
        ),
    ]
    prior_outgoing = build_module_chat_message(
        sender='ai_different',
        recipient='orchestrator',
        topic='ai_different.module_family_response',
        body={
            'message_id': 'ai-different-prior-response',
            'response_kind': 'module_family_coordination_response',
            'ledger_id': 'ledger_prior',
            'label_clean': True,
        },
        evidence={'ledger_hash': 'prior-hash', 'label_clean': True},
        tags=['module_family_response', 'label_clean'],
    )
    return first + [prior_outgoing] + second


def append_language_request(path, message_id='lang-digest-3'):
    message = build_module_chat_message(
        sender='language_model_2',
        recipient='ai_different',
        topic='digest.abstraction_transfer.coordination',
        body={
            'message_id': message_id,
            'request_kind': 'abstraction_transfer_followup',
            'question': 'Plan a follow-up without fresh budget evidence.',
            'outcome_mode': 'weak',
            'priority': 0.66,
        },
        evidence={'digest_kind': 'coordination', 'priority': 0.66},
        tags=['abstraction_transfer', 'coordination'],
    )
    with Path(path).open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(message) + '\n')
    return message


def _write_messages(tmpdir, messages):
    inbox = Path(tmpdir) / 'inbox.jsonl'
    inbox.write_text(
        '\n'.join(json.dumps(message) for message in messages) + '\n',
        encoding='utf-8',
    )
    return inbox


class ModuleChatAdapterTests(unittest.TestCase):
    def test_builds_valid_capsule_message_with_evidence(self):
        message = export_capsule_chat_message(
            capsule_fixture(),
            recipient='orchestrator',
        )

        self.assertEqual('ai_different', message['sender'])
        self.assertEqual('orchestrator', message['recipient'])
        self.assertEqual('ai_different.status_capsule', message['topic'])
        self.assertTrue(message['body']['chat_bridge_available'])
        self.assertTrue(message['body']['label_clean'])
        self.assertIn('label_clean', message['tags'])
        self.assertIn('evidence_gates', message['evidence'])
        self.assertEqual(
            'abstraction_transfer_confirmed',
            message['evidence'][
                'latest_verified_abstraction_transfer_result'
            ]['outcome'],
        )
        self.assertFalse(
            message['body']['project_owned_boundary']['third_party_checkpoint_used']
        )

    def test_invalid_participant_and_message_handling(self):
        with self.assertRaisesRegex(ValueError, 'unknown module-chat participant'):
            validate_participant('not_a_module')
        with self.assertRaisesRegex(ValueError, 'missing evidence'):
            validate_module_chat_message({
                'sender': 'ai_different',
                'recipient': 'orchestrator',
                'topic': 'ai_different.status_capsule',
                'body': {},
                'tags': [],
            })
        with self.assertRaisesRegex(ValueError, 'body must be an object'):
            build_module_chat_message(
                sender='ai_different',
                recipient='orchestrator',
                topic='ai_different.status_capsule',
                body=[],
                evidence={},
                tags=[],
            )

    def test_reads_inbox_and_selects_abstraction_transfer_request(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            inbox = Path(tmpdir) / 'inbox.jsonl'
            messages = [
                build_module_chat_message(
                    sender='orchestrator',
                    recipient='ai_different',
                    topic='request.abstraction_transfer.followup',
                    body={
                        'request_kind': 'abstraction_transfer_followup',
                        'question': 'Please run a label-clean abstraction transfer follow-up.',
                        'outcome_mode': 'weak',
                        'priority': 0.9,
                    },
                    evidence={'source': 'unit_test'},
                    tags=['abstraction_transfer', 'followup'],
                ),
                build_module_chat_message(
                    sender='funfun',
                    recipient='funfun',
                    topic='request.unrelated',
                    body={'question': 'not for AI Different'},
                    evidence={},
                    tags=[],
                ),
            ]
            inbox.write_text(
                '\n'.join(json.dumps(message) for message in messages)
                + '\nnot-json\n',
                encoding='utf-8',
            )

            summary = read_module_chat_inbox(inbox)
            selected = choose_next_non_final_request(capsule_fixture(), summary)
            message = export_capsule_chat_message(
                capsule_fixture(),
                recipient='broadcast',
                inbox_summary=summary,
            )

        self.assertEqual(1, len(summary['messages']))
        self.assertEqual(1, len(summary['invalid_messages']))
        self.assertEqual(1, len(summary['experiment_requests']))
        self.assertEqual('inbox', selected['source'])
        self.assertEqual('orchestrator', selected['requested_by'])
        self.assertIn('--abstraction-transfer-campaign', selected['command'])
        self.assertIn('--abstraction-transfer-outcome weak', selected['command'])
        self.assertFalse(selected['runs_final'])
        self.assertEqual('broadcast', message['recipient'])
        self.assertEqual('inbox', message['body']['selected_chat_request']['source'])
        self.assertTrue(message['body']['selected_chat_request']['label_clean'])

    def test_multi_message_inbox_prioritizes_language_request_and_code_evidence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            inbox = Path(tmpdir) / 'inbox.jsonl'
            messages = [
                build_module_chat_message(
                    sender='language_model_2',
                    recipient='ai_different',
                    topic='request.abstraction_transfer.followup',
                    body={
                        'request_kind': 'abstraction_transfer_followup',
                        'question': 'Please test the compressed abstraction in another context.',
                        'outcome_mode': 'absent',
                        'priority': 0.95,
                    },
                    evidence={'prompt_source': 'language'},
                    tags=['abstraction_transfer'],
                ),
                build_module_chat_message(
                    sender='code_module',
                    recipient='ai_different',
                    topic='evidence.abstraction_transfer.cost',
                    body={
                        'note_kind': 'evidence',
                        'summary': 'The no-save campaign is lightweight enough for local response.',
                    },
                    evidence={'estimated_runtime': 'subsecond', 'mutates_runtime_memory': False},
                    tags=['evidence', 'local_safe'],
                ),
                build_module_chat_message(
                    sender='orchestrator',
                    recipient='ai_different',
                    topic='request.abstraction_transfer.followup',
                    body={
                        'request_kind': 'abstraction_transfer_followup',
                        'question': 'Lower priority request.',
                        'priority': 0.1,
                    },
                    evidence={'prompt_source': 'orchestrator'},
                    tags=['abstraction_transfer'],
                ),
            ]
            inbox.write_text(
                '\n'.join(json.dumps(message) for message in messages),
                encoding='utf-8',
            )

            summary = read_module_chat_inbox(inbox)
            selected = choose_next_non_final_request(capsule_fixture(), summary)
            payload = build_chat_driven_response_payload(capsule_fixture(), summary)

        self.assertEqual(3, len(summary['messages']))
        self.assertEqual(2, len(summary['experiment_requests']))
        self.assertEqual(1, len(summary['evidence_notes']))
        self.assertEqual('language_model_2', selected['requested_by'])
        self.assertIn('--abstraction-transfer-outcome absent', selected['command'])
        code_notes = [
            note for note in summary['evidence_notes']
            if note['sender'] == 'code_module'
        ]
        self.assertEqual(1, len(code_notes))
        self.assertTrue(code_notes[0]['label_clean'])
        self.assertEqual(1, len(payload['code_evidence_notes']))
        self.assertTrue(payload['label_clean'])

    def test_response_message_plan_shape_includes_project_and_runtime_evidence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            inbox = Path(tmpdir) / 'inbox.jsonl'
            inbox.write_text(
                json.dumps(build_module_chat_message(
                    sender='language_model_2',
                    recipient='ai_different',
                    topic='request.abstraction_transfer.followup',
                    body={
                        'request_kind': 'abstraction_transfer_followup',
                        'question': 'Plan the next abstraction transfer response.',
                        'priority': 0.8,
                    },
                    evidence={'source': 'unit_test'},
                    tags=['abstraction_transfer'],
                ))
                + '\n',
                encoding='utf-8',
            )
            summary = read_module_chat_inbox(inbox)

        message = export_chat_driven_response_message(
            capsule_fixture(),
            summary,
            recipient='broadcast',
        )

        self.assertEqual('ai_different', message['sender'])
        self.assertEqual('broadcast', message['recipient'])
        self.assertEqual('ai_different.abstraction_transfer_response', message['topic'])
        self.assertEqual('plan', message['body']['outcome_or_plan']['mode'])
        self.assertFalse(message['body']['outcome_or_plan']['ran_campaign'])
        self.assertFalse(message['body']['runtime_memory_mutated'])
        self.assertFalse(message['evidence']['runtime_memory_mutated'])
        self.assertFalse(
            message['body']['project_owned_boundary']['third_party_checkpoint_used']
        )
        self.assertTrue(message['body']['label_clean'])
        self.assertIn('plan_only', message['tags'])

    def test_run_module_chat_response_loop_can_run_lightweight_no_save_campaign(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            inbox = Path(tmpdir) / 'inbox.jsonl'
            runtime_memory = Path(tmpdir) / 'theory-memory.json'
            output_file = Path(tmpdir) / 'response.json'
            inbox.write_text(
                json.dumps(build_module_chat_message(
                    sender='language_model_2',
                    recipient='ai_different',
                    topic='request.abstraction_transfer.followup',
                    body={
                        'request_kind': 'abstraction_transfer_followup',
                        'question': 'Run the cheap abstraction transfer response.',
                        'outcome_mode': 'weak',
                        'priority': 0.9,
                    },
                    evidence={'source': 'unit_test'},
                    tags=['abstraction_transfer'],
                ))
                + '\n',
                encoding='utf-8',
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                message = run_module_chat_response_loop(
                    memory_data=memory_fixture(),
                    runtime_memory_path=str(runtime_memory),
                    recipient='orchestrator',
                    inbox_file=inbox,
                    output_file=output_file,
                    response_mode='run',
                    git_status_text='',
                    git_ignored_text='',
                )
            saved = json.loads(output_file.read_text(encoding='utf-8'))

        self.assertIn('AI_DIFFERENT_MODULE_CHAT_RESPONSE ', output.getvalue())
        self.assertFalse(runtime_memory.exists())
        self.assertEqual('campaign_result', message['body']['outcome_or_plan']['mode'])
        self.assertTrue(message['body']['outcome_or_plan']['ran_campaign'])
        self.assertFalse(message['body']['outcome_or_plan']['memory_saved'])
        self.assertFalse(message['body']['runtime_memory_mutated'])
        self.assertEqual(
            'abstraction_transfer_weak',
            message['body']['outcome_or_plan']['outcome']['outcome'],
        )
        self.assertTrue(message['body']['label_clean'])
        self.assertEqual(message['topic'], saved['topic'])

    def test_three_module_inbox_builds_family_ledger_with_funfun_evidence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            inbox = Path(tmpdir) / 'inbox.jsonl'
            inbox.write_text(
                '\n'.join(json.dumps(message) for message in three_module_messages())
                + '\n',
                encoding='utf-8',
            )
            summary = read_module_chat_inbox(inbox)

        selected = choose_module_family_followup(capsule_fixture(), summary)
        selected_recipient = choose_module_family_recipient(
            summary,
            selected,
            requested_recipient='auto',
        )
        ledger = build_module_family_response_ledger(
            capsule_fixture(),
            summary,
            selected_recipient=selected_recipient,
            response_mode='run',
            runtime_memory_hash_state={
                'path': 'tmp/theory-memory.json',
                'exists': True,
                'before_hash': 'same',
                'after_hash': 'same',
                'unchanged': True,
            },
            ledger_path='tmp/module-chat-response-ledger.json',
        )

        self.assertEqual(3, len(summary['messages']))
        self.assertEqual(1, len(summary['experiment_requests']))
        self.assertEqual(2, len(summary['evidence_notes']))
        self.assertEqual('language_model_2', selected['requested_by'])
        self.assertEqual('language_model_2', selected_recipient)
        self.assertTrue(selected['selection_basis']['cheap_no_save_supported'])
        self.assertTrue(selected['selection_basis']['typed_discovery_supported'])
        self.assertTrue(ledger['three_module_response_available'])
        self.assertEqual(1, ledger['evidence_counts_by_sender']['code_module'])
        self.assertEqual(1, ledger['evidence_counts_by_sender']['funfun'])
        self.assertEqual('funfun-typed-1', ledger['selected_evidence'][1]['message_id'])
        self.assertTrue(ledger['run_decision']['should_run_no_save_campaign'])
        self.assertTrue(ledger['label_clean'])
        self.assertFalse(ledger['runtime_memory_mutated'])

    def test_family_ledger_uses_deterministic_fallback_without_runnable_request(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            inbox = Path(tmpdir) / 'inbox.jsonl'
            inbox.write_text(
                '\n'.join(
                    json.dumps(message)
                    for message in three_module_messages(include_request=False)
                )
                + '\n',
                encoding='utf-8',
            )
            summary = read_module_chat_inbox(inbox)

        selected = choose_module_family_followup(capsule_fixture(), summary)
        ledger = build_module_family_response_ledger(
            capsule_fixture(),
            summary,
            selected_recipient='language_model_2',
            response_mode='run',
            runtime_memory_hash_state={'unchanged': True},
        )

        self.assertEqual(0, len(summary['experiment_requests']))
        self.assertEqual('deterministic_fallback', selected['source'])
        self.assertTrue(selected['selection_basis']['fallback_used'])
        self.assertFalse(ledger['run_decision']['should_run_no_save_campaign'])
        self.assertEqual('plan', ledger['outcome_or_plan']['mode'])
        self.assertEqual(
            'plan_only_no_runnable_inbox_request',
            ledger['outcome_or_plan']['plan_reason'],
        )

    def test_family_response_ledger_persists_and_exports_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = Path(tmpdir) / 'ledger.json'
            ledger = build_module_family_response_ledger(
                capsule_fixture(),
                {
                    'path': 'synthetic.jsonl',
                    'messages': three_module_messages(),
                    'invalid_messages': [],
                    'handoff_questions': [],
                    'experiment_requests': [{
                        'message_id': 'lang-digest-1',
                        'sender': 'language_model_2',
                        'topic': 'digest.abstraction_transfer.coordination',
                        'request_kind': 'abstraction_transfer_followup',
                        'experiment_kind': 'abstraction_transfer_probe',
                        'question': 'Use typed evidence to pick a safe transfer follow-up.',
                        'outcome_mode': 'weak',
                        'priority': 0.7,
                        'tags': ['abstraction_transfer', 'coordination'],
                        'label_clean': True,
                    }],
                    'evidence_notes': [
                        note for note in read_module_chat_inbox(
                            _write_messages(tmpdir, three_module_messages())
                        )['evidence_notes']
                    ],
                },
                selected_recipient='language_model_2',
                response_mode='plan',
                runtime_memory_hash_state={'path': 'tmp/theory-memory.json', 'unchanged': True},
                ledger_path=ledger_path,
            )
            write_response_ledger(ledger_path, ledger)
            loaded = load_response_ledger(ledger_path)
            message = export_module_family_response_message(loaded)

        self.assertEqual(ledger['ledger_id'], loaded['ledger_id'])
        self.assertEqual(ledger['ledger_hash'], loaded['ledger_hash'])
        self.assertEqual('ai_different', message['sender'])
        self.assertEqual('language_model_2', message['recipient'])
        self.assertEqual('ai_different.module_family_response', message['topic'])
        self.assertEqual(ledger['ledger_id'], message['body']['ledger_id'])
        self.assertTrue(message['body']['label_clean'])
        self.assertFalse(message['body']['runtime_memory_mutated'])
        self.assertFalse(
            message['body']['project_owned_boundary']['third_party_checkpoint_used']
        )

    def test_run_module_chat_family_response_preserves_runtime_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            inbox = Path(tmpdir) / 'inbox.jsonl'
            runtime_memory = Path(tmpdir) / 'theory-memory.json'
            output_file = Path(tmpdir) / 'response.json'
            ledger_file = Path(tmpdir) / 'ledger.json'
            inbox.write_text(
                '\n'.join(json.dumps(message) for message in three_module_messages())
                + '\n',
                encoding='utf-8',
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                message = run_module_chat_family_response(
                    memory_data=memory_fixture(),
                    runtime_memory_path=str(runtime_memory),
                    recipient='auto',
                    inbox_file=inbox,
                    output_file=output_file,
                    ledger_file=ledger_file,
                    response_mode='run',
                    git_status_text='',
                    git_ignored_text='',
                )
            saved = json.loads(output_file.read_text(encoding='utf-8'))
            ledger = load_response_ledger(ledger_file)

        self.assertIn('AI_DIFFERENT_MODULE_FAMILY_RESPONSE ', output.getvalue())
        self.assertFalse(runtime_memory.exists())
        self.assertEqual('language_model_2', message['recipient'])
        self.assertEqual('campaign_result', message['body']['outcome_or_plan']['mode'])
        self.assertTrue(message['body']['outcome_or_plan']['ran_campaign'])
        self.assertFalse(message['body']['outcome_or_plan']['memory_saved'])
        self.assertTrue(message['body']['runtime_memory_hash_state']['unchanged'])
        self.assertFalse(message['body']['runtime_memory_mutated'])
        self.assertTrue(message['body']['three_module_response_available'])
        self.assertEqual(1, message['body']['evidence_counts_by_sender']['code_module'])
        self.assertEqual(1, message['body']['evidence_counts_by_sender']['funfun'])
        self.assertEqual(message['body']['ledger_id'], ledger['ledger_id'])
        self.assertEqual(message['topic'], saved['topic'])

    def test_two_round_family_log_parsing_includes_prior_outgoing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = _write_messages(tmpdir, two_round_family_log_messages())
            summary = read_module_chat_log(log_path)

        self.assertEqual(7, len(summary['messages']))
        self.assertEqual(6, len(summary['inbound_messages']))
        self.assertEqual(1, len(summary['outgoing_messages']))
        self.assertEqual(2, len(summary['experiment_requests']))
        self.assertEqual(4, len(summary['evidence_notes']))
        self.assertEqual(
            'ai-different-prior-response',
            summary['outgoing_messages'][0]['body']['message_id'],
        )

    def test_rolling_family_response_is_idempotent_on_second_pass(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = _write_messages(tmpdir, two_round_family_log_messages())
            runtime_memory = Path(tmpdir) / 'theory-memory.json'
            output_file = Path(tmpdir) / 'rolling-output.json'
            ledger_file = Path(tmpdir) / 'ledger.json'
            rolling_memory_file = Path(tmpdir) / 'rolling-memory.json'
            with contextlib.redirect_stdout(io.StringIO()):
                first = run_module_chat_rolling_family_response(
                    memory_data=memory_fixture(),
                    runtime_memory_path=str(runtime_memory),
                    recipient='auto',
                    chat_log_file=log_path,
                    output_file=output_file,
                    ledger_file=ledger_file,
                    rolling_memory_file=rolling_memory_file,
                    response_mode='run',
                    git_status_text='',
                    git_ignored_text='',
                )
            first_memory = load_rolling_family_memory(rolling_memory_file)
            with contextlib.redirect_stdout(io.StringIO()):
                second = run_module_chat_rolling_family_response(
                    memory_data=memory_fixture(),
                    runtime_memory_path=str(runtime_memory),
                    recipient='auto',
                    chat_log_file=log_path,
                    output_file=output_file,
                    ledger_file=ledger_file,
                    rolling_memory_file=rolling_memory_file,
                    response_mode='run',
                    git_status_text='',
                    git_ignored_text='',
                )
            second_memory = load_rolling_family_memory(rolling_memory_file)

        self.assertEqual(6, first['new_message_count'])
        self.assertEqual(0, first['skipped_message_count'])
        self.assertEqual(6, first['processed_message_count'])
        self.assertIsNotNone(first['response_message'])
        self.assertIn('ai-different-prior-response', first['outgoing_response_ids'])
        self.assertEqual(0, second['new_message_count'])
        self.assertEqual(6, second['skipped_message_count'])
        self.assertIsNone(second['response_message'])
        self.assertEqual(first['selected_recipient'], second['selected_recipient'])
        self.assertEqual(first_memory['processed_message_ids'], second_memory['processed_message_ids'])
        self.assertFalse(second['runtime_memory_mutated'])
        self.assertTrue(second['label_clean'])

    def test_rolling_family_response_append_only_update_plans_without_budget(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = _write_messages(tmpdir, two_round_family_log_messages())
            runtime_memory = Path(tmpdir) / 'theory-memory.json'
            ledger_file = Path(tmpdir) / 'ledger.json'
            rolling_memory_file = Path(tmpdir) / 'rolling-memory.json'
            with contextlib.redirect_stdout(io.StringIO()):
                run_module_chat_rolling_family_response(
                    memory_data=memory_fixture(),
                    runtime_memory_path=str(runtime_memory),
                    recipient='auto',
                    chat_log_file=log_path,
                    ledger_file=ledger_file,
                    rolling_memory_file=rolling_memory_file,
                    response_mode='run',
                    git_status_text='',
                    git_ignored_text='',
                )
            append_language_request(log_path)
            with contextlib.redirect_stdout(io.StringIO()):
                update = run_module_chat_rolling_family_response(
                    memory_data=memory_fixture(),
                    runtime_memory_path=str(runtime_memory),
                    recipient='auto',
                    chat_log_file=log_path,
                    ledger_file=ledger_file,
                    rolling_memory_file=rolling_memory_file,
                    response_mode='run',
                    git_status_text='',
                    git_ignored_text='',
                )
            memory = load_rolling_family_memory(rolling_memory_file)

        self.assertEqual(1, update['new_message_count'])
        self.assertEqual(6, update['skipped_message_count'])
        self.assertEqual(7, update['processed_message_count'])
        self.assertIsNotNone(update['response_message'])
        self.assertEqual('plan', update['outcome_or_plan']['mode'])
        self.assertEqual(
            'plan_only_missing_cheap_no_save_evidence',
            update['outcome_or_plan']['plan_reason'],
        )
        self.assertEqual(2, len(memory['response_records']))
        self.assertEqual(['lang-digest-3'], update['new_message_ids'])
        self.assertFalse(update['runtime_memory_mutated'])

    def test_rolling_family_response_fallback_when_new_round_has_no_request(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = _write_messages(
                tmpdir,
                three_module_messages(include_request=False),
            )
            runtime_memory = Path(tmpdir) / 'theory-memory.json'
            rolling_memory_file = Path(tmpdir) / 'rolling-memory.json'
            with contextlib.redirect_stdout(io.StringIO()):
                result = run_module_chat_rolling_family_response(
                    memory_data=memory_fixture(),
                    runtime_memory_path=str(runtime_memory),
                    recipient='auto',
                    chat_log_file=log_path,
                    rolling_memory_file=rolling_memory_file,
                    response_mode='run',
                    git_status_text='',
                    git_ignored_text='',
                )

        self.assertEqual(3, result['new_message_count'])
        self.assertEqual('plan', result['outcome_or_plan']['mode'])
        self.assertEqual(
            'plan_only_no_runnable_inbox_request',
            result['outcome_or_plan']['plan_reason'],
        )
        self.assertEqual('language_model_2', result['selected_recipient'])
        self.assertTrue(result['label_clean'])

    def test_malformed_rolling_memory_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_path = Path(tmpdir) / 'rolling-memory.json'
            memory_path.write_text(
                json.dumps({'memory_kind': 'wrong', 'processed_message_ids': []}),
                encoding='utf-8',
            )
            with self.assertRaisesRegex(ValueError, 'wrong memory_kind'):
                load_rolling_family_memory(memory_path)

    def test_run_module_chat_export_prints_label_clean_bridge_without_mutating_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            inbox = Path(tmpdir) / 'inbox.jsonl'
            runtime_memory = Path(tmpdir) / 'theory-memory.json'
            output_file = Path(tmpdir) / 'message.json'
            inbox.write_text(
                json.dumps(build_module_chat_message(
                    sender='orchestrator',
                    recipient='ai_different',
                    topic='request.abstraction_transfer.followup',
                    body={
                        'request_kind': 'abstraction_transfer_followup',
                        'question': 'Please run the next abstraction transfer follow-up.',
                    },
                    evidence={'source': 'unit_test'},
                    tags=['abstraction_transfer'],
                ))
                + '\n',
                encoding='utf-8',
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                message = run_module_chat_export(
                    memory_data=memory_fixture(),
                    runtime_memory_path=str(runtime_memory),
                    recipient='orchestrator',
                    inbox_file=inbox,
                    output_file=output_file,
                    git_status_text='',
                    git_ignored_text='',
                )
            saved = json.loads(output_file.read_text(encoding='utf-8'))

        self.assertIn('AI_DIFFERENT_MODULE_CHAT_MESSAGE ', output.getvalue())
        self.assertFalse(runtime_memory.exists())
        self.assertTrue(message['body']['chat_bridge_available'])
        self.assertTrue(message['body']['label_clean'])
        self.assertEqual(message['topic'], saved['topic'])
        self.assertEqual('orchestrator', saved['recipient'])

    def test_adapter_has_no_external_orchestrator_project_imports(self):
        source = Path(
            PROJECT_DIR,
            'agent',
            'module_chat_adapter.py',
        ).read_text(encoding='utf-8')
        main_source = Path(PROJECT_DIR, 'main.py').read_text(encoding='utf-8')

        self.assertNotIn('Language model 2.0', source)
        self.assertNotIn('orchastratorrrr', source)
        self.assertNotIn('Code Module', source)
        self.assertNotIn('sys.path', source)
        self.assertNotIn('Language model 2.0', main_source)
        self.assertNotIn('orchastratorrrr', main_source)
        self.assertNotIn('Code Module', main_source)


if __name__ == '__main__':
    unittest.main()
