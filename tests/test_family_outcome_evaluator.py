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

from agent.family_outcome_evaluator import (  # noqa: E402
    build_outcome_evaluator_ledger,
    choose_family_outcome_decision,
    classify_family_evidence,
    collect_family_evidence_items,
    empty_outcome_evaluator_memory,
    export_outcome_evaluator_message,
    load_outcome_evaluator_memory,
    validate_outcome_evaluator_memory,
    write_outcome_evaluator_memory,
    write_outcome_evaluator_message_jsonl,
)
from agent.module_chat_adapter import write_response_ledger  # noqa: E402
from main import run_family_outcome_evaluator  # noqa: E402


def memory_fixture():
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
                    'evidence': {'transfer_outcome_count': 1},
                },
            },
        },
        'abstraction_discovery_evidence': {
            'transfer_outcome_count': 1,
            'transfer_confirmed_count': 1,
            'latest_transfer_outcome': {
                'outcome': 'abstraction_transfer_confirmed',
                'experiment_kind': 'abstraction_transfer_probe',
            },
        },
    }


def rolling_memory_fixture():
    return {
        'schema_version': 1,
        'memory_kind': 'ai_different.rolling_family_response_memory',
        'processed_message_ids': ['lang-1', 'code-1', 'funfun-1'],
        'outgoing_response_ids': ['ai-response-1'],
        'response_records': [{
            'record_id': 'record-1',
            'response_ledger_id': 'ledger-one',
            'response_ledger_hash': 'hash-one',
            'processed_message_ids': ['lang-1', 'code-1', 'funfun-1'],
            'selected_recipient': 'language_model_2',
            'evidence_counts_by_sender': {'code_module': 1, 'funfun': 1},
            'evidence_rationale': [{
                'message_id': 'code-proof-1',
                'sender': 'code_module',
                'topic': 'proof.abstraction_transfer',
                'priority': 0.91,
                'why_it_matters': 'proof explains why the no-save probe is cheap',
            }],
            'outcome_or_plan': {'mode': 'campaign_result', 'ran_campaign': True},
        }],
        'latest': {
            'response_ledger_id': 'ledger-one',
            'response_ledger_hash': 'hash-one',
            'selected_recipient': 'language_model_2',
        },
        'memory_hash': 'rolling-hash',
    }


def response_ledger_fixture(
    ledger_id='ledger-one',
    *,
    request_source='inbox',
    leak=False,
    missing=False,
):
    selected_request = {
        'source': request_source,
        'requested_by': 'language_model_2',
        'request_topic': 'digest.abstraction_transfer.coordination',
        'action_kind': 'non_final_abstraction_transfer_campaign',
        'experiment_kind': 'abstraction_transfer_probe',
        'command': 'python3 first_principles_ai/main.py --abstraction-transfer-campaign',
        'reason': 'Run label-clean transfer probe.',
        'runs_final': False,
        'outcome_mode': 'weak',
        'label_clean': not leak,
        'selection_score': 0.85,
    }
    if leak:
        selected_request['reason'] = 'Run gravity-labelled transfer probe.'
    outcome = {
        'mode': 'plan' if missing else 'campaign_result',
        'ran_campaign': False if missing else True,
        'runs_final': False,
        'memory_saved': False,
    }
    if missing:
        outcome['plan_reason'] = 'plan_only_missing_cheap_no_save_evidence'
    return {
        'schema_version': 1,
        'ledger_kind': 'ai_different.module_family_response_ledger',
        'ledger_id': ledger_id,
        'ledger_hash': f'{ledger_id}-hash',
        'selected_request': selected_request,
        'selected_evidence': [
            {
                'message_id': 'code-proof-1',
                'sender': 'code_module',
                'topic': 'proof.abstraction_transfer.explanation',
                'summary': 'Proof/explanation for cheap no-save response.',
                'priority': 0.91,
                'evidence': {'proof_status': 'passed'},
                'tags': ['proof', 'evidence'],
            },
            {
                'message_id': 'funfun-theorem-1',
                'sender': 'funfun',
                'topic': 'evidence.theorem_accounting',
                'summary': 'Real theorem evidence separated from internal theorem evidence.',
                'priority': 0.73,
                'evidence': {'real_theorem_count': 1, 'internal_theorem_count': 2},
                'tags': ['evidence', 'theorem_accounting'],
            },
        ],
        'outcome_or_plan': outcome,
        'project_owned_boundary': {'third_party_checkpoint_used': False},
        'runtime_memory_hash_state': {'unchanged': True},
        'label_clean': not leak,
        'leak_terms': ['gravity'] if leak else [],
    }


class FamilyOutcomeEvaluatorTests(unittest.TestCase):
    def test_classifies_cross_module_evidence_items(self):
        items = collect_family_evidence_items(
            rolling_memory_fixture(),
            [response_ledger_fixture(missing=True)],
            empty_outcome_evaluator_memory(),
        )
        for item in items:
            item['classification'] = classify_family_evidence(item)
        classes = {item['classification'] for item in items}

        self.assertIn('runnable_experiment_request', classes)
        self.assertIn('proof_support', classes)
        self.assertIn('advisory_note', classes)
        self.assertIn('missing_evidence_blocker', classes)

    def test_deterministic_selection_prioritizes_runnable_then_repair_then_advisory(self):
        runnable = {
            'evidence_id': 'run',
            'classification': 'runnable_experiment_request',
            'sender': 'language_model_2',
            'topic': 'request',
            'priority': 0.5,
            'payload': response_ledger_fixture()['selected_request'],
        }
        blocker = {
            'evidence_id': 'block',
            'classification': 'missing_evidence_blocker',
            'sender': 'language_model_2',
            'priority': 0.99,
            'payload': {'plan_reason': 'missing budget proof'},
        }
        advisory = {
            'evidence_id': 'adv',
            'classification': 'advisory_note',
            'sender': 'funfun',
            'priority': 1.0,
            'payload': {'summary': 'theorem accounting advisory'},
        }

        decision = choose_family_outcome_decision([advisory, blocker, runnable])
        repair = choose_family_outcome_decision([advisory, blocker])
        defer = choose_family_outcome_decision([advisory])

        self.assertEqual('run_next_safe_experiment', decision['decision_kind'])
        self.assertEqual('repair_missing_evidence', repair['decision_kind'])
        self.assertEqual('defer_with_cross_module_advisory', defer['decision_kind'])

    def test_noop_when_all_evidence_is_already_processed(self):
        memory = empty_outcome_evaluator_memory()
        initial = build_outcome_evaluator_ledger(
            rolling_memory=rolling_memory_fixture(),
            response_ledgers=[response_ledger_fixture()],
            evaluator_memory=memory,
            runtime_memory_hash_state={'unchanged': True},
            project_owned_boundary={'third_party_checkpoint_used': False},
        )
        memory['processed_evidence_ids'] = list(initial['processed_evidence_ids'])
        repeated = build_outcome_evaluator_ledger(
            rolling_memory=rolling_memory_fixture(),
            response_ledgers=[response_ledger_fixture()],
            evaluator_memory=memory,
            runtime_memory_hash_state={'unchanged': True},
            project_owned_boundary={'third_party_checkpoint_used': False},
        )

        self.assertEqual('no_op', repeated['decision']['decision_kind'])
        self.assertFalse(repeated['processed_evidence_ids'])
        self.assertIsNone(export_outcome_evaluator_message(repeated))

    def test_appended_ledger_update_processes_only_new_evidence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rolling_path = Path(tmpdir) / 'rolling.json'
            memory_path = Path(tmpdir) / 'evaluator-memory.json'
            ledger_one = Path(tmpdir) / 'ledger-one.json'
            ledger_two = Path(tmpdir) / 'ledger-two.json'
            rolling_path.write_text(
                json.dumps(rolling_memory_fixture()),
                encoding='utf-8',
            )
            write_response_ledger(ledger_one, response_ledger_fixture('ledger-one'))
            write_response_ledger(ledger_two, response_ledger_fixture('ledger-two', missing=True))
            with contextlib.redirect_stdout(io.StringIO()):
                first = run_family_outcome_evaluator(
                    memory_data=memory_fixture(),
                    runtime_memory_path=str(Path(tmpdir) / 'theory-memory.json'),
                    rolling_memory_file=rolling_path,
                    response_ledger_files=[ledger_one],
                    evaluator_memory_file=memory_path,
                    evaluator_ledger_file=Path(tmpdir) / 'outcome-ledger.json',
                    git_status_text='',
                    git_ignored_text='',
                )
            with contextlib.redirect_stdout(io.StringIO()):
                second = run_family_outcome_evaluator(
                    memory_data=memory_fixture(),
                    runtime_memory_path=str(Path(tmpdir) / 'theory-memory.json'),
                    rolling_memory_file=rolling_path,
                    response_ledger_files=[ledger_one, ledger_two],
                    evaluator_memory_file=memory_path,
                    evaluator_ledger_file=Path(tmpdir) / 'outcome-ledger.json',
                    git_status_text='',
                    git_ignored_text='',
                )

        self.assertGreater(len(first['processed_evidence_ids']), 0)
        self.assertGreater(len(second['processed_evidence_ids']), 0)
        self.assertIn('ledger-two', second['processed_ledger_ids'])
        self.assertLess(
            len(second['processed_evidence_ids']),
            len(first['processed_evidence_ids']) + len(second['processed_ledger_ids']) * 4,
        )

    def test_malformed_evaluator_memory_rejected(self):
        with self.assertRaisesRegex(ValueError, 'wrong memory_kind'):
            validate_outcome_evaluator_memory({
                'memory_kind': 'wrong',
                'processed_evidence_ids': [],
            })

    def test_outgoing_message_schema_and_jsonl_export(self):
        ledger = build_outcome_evaluator_ledger(
            rolling_memory=rolling_memory_fixture(),
            response_ledgers=[response_ledger_fixture()],
            evaluator_memory=empty_outcome_evaluator_memory(),
            runtime_memory_hash_state={'unchanged': True},
            project_owned_boundary={'third_party_checkpoint_used': False},
            ledger_path='tmp/outcome-ledger.json',
        )
        message = export_outcome_evaluator_message(ledger, recipient='broadcast')
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / 'message.jsonl'
            write_outcome_evaluator_message_jsonl(output, message)
            rows = [
                json.loads(line)
                for line in output.read_text(encoding='utf-8').splitlines()
            ]

        self.assertEqual('ai_different', message['sender'])
        self.assertEqual('broadcast', message['recipient'])
        self.assertEqual('family_outcome_evaluation', message['body']['response_kind'])
        self.assertTrue(message['body']['label_clean'])
        self.assertFalse(message['body']['third_party_checkpoint_used'])
        self.assertEqual(1, len(rows))
        self.assertEqual(message['topic'], rows[0]['topic'])

    def test_label_leak_guard_marks_unsafe_payload(self):
        ledger = build_outcome_evaluator_ledger(
            rolling_memory=rolling_memory_fixture(),
            response_ledgers=[response_ledger_fixture(leak=True)],
            evaluator_memory=empty_outcome_evaluator_memory(),
            runtime_memory_hash_state={'unchanged': True},
            project_owned_boundary={'third_party_checkpoint_used': False},
        )

        self.assertFalse(ledger['label_clean'])
        self.assertIn('gravity', ledger['leak_terms'])
        self.assertNotEqual('run_next_safe_experiment', ledger['decision']['decision_kind'])

    def test_runtime_memory_and_third_party_boundary_preserved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_memory = Path(tmpdir) / 'theory-memory.json'
            rolling_path = Path(tmpdir) / 'rolling.json'
            ledger_path = Path(tmpdir) / 'ledger.json'
            output_path = Path(tmpdir) / 'result.json'
            rolling_path.write_text(json.dumps(rolling_memory_fixture()), encoding='utf-8')
            write_response_ledger(ledger_path, response_ledger_fixture())
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = run_family_outcome_evaluator(
                    memory_data=memory_fixture(),
                    runtime_memory_path=str(runtime_memory),
                    rolling_memory_file=rolling_path,
                    response_ledger_files=[ledger_path],
                    output_file=output_path,
                    evaluator_ledger_file=Path(tmpdir) / 'outcome-ledger.json',
                    evaluator_memory_file=Path(tmpdir) / 'outcome-memory.json',
                    git_status_text='',
                    git_ignored_text='',
                )

        self.assertIn('AI_DIFFERENT_FAMILY_OUTCOME_EVALUATOR ', output.getvalue())
        self.assertFalse(runtime_memory.exists())
        self.assertTrue(result['runtime_memory_hash_state']['unchanged'])
        self.assertFalse(result['runtime_memory_mutated'])
        self.assertFalse(result['third_party_checkpoint_used'])
        self.assertTrue(result['label_clean'])

    def test_evaluator_has_no_sibling_project_imports(self):
        source = Path(
            PROJECT_DIR,
            'agent',
            'family_outcome_evaluator.py',
        ).read_text(encoding='utf-8')
        main_source = Path(PROJECT_DIR, 'main.py').read_text(encoding='utf-8')

        self.assertNotIn('Language model 2.0', source)
        self.assertNotIn('orchastratorrrr', source)
        self.assertNotIn('Code Module', source)
        self.assertNotIn('Language model 2.0', main_source)
        self.assertNotIn('orchastratorrrr', main_source)
        self.assertNotIn('Code Module', main_source)


if __name__ == '__main__':
    unittest.main()
