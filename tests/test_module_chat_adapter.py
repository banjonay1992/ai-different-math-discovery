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
    build_module_chat_message,
    choose_next_non_final_request,
    export_capsule_chat_message,
    read_module_chat_inbox,
    validate_module_chat_message,
    validate_participant,
)
from agent.status_capsule import build_ai_different_status_capsule  # noqa: E402
from main import run_module_chat_export  # noqa: E402


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

        self.assertNotIn('Language model 2.0', source)
        self.assertNotIn('orchastratorrrr', source)
        self.assertNotIn('Code Module', source)
        self.assertNotIn('sys.path', source)


if __name__ == '__main__':
    unittest.main()
