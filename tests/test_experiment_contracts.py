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

from agent.experiment_contracts import (  # noqa: E402
    build_experiment_contract_from_evaluator,
    downstream_evidence_for_contracts,
    empty_experiment_contract_ledger,
    export_experiment_contract_message,
    load_experiment_contract_ledger,
    update_contract_ledger,
    validate_evaluator_ledger,
    validate_experiment_contract_ledger,
    write_contract_outbox_jsonl,
    write_experiment_contract_ledger,
)
from main import run_experiment_contract_loop  # noqa: E402


def memory_fixture():
    return {
        'discovery_readiness': {
            'readiness_score': 0.82,
            'status': 'nearly_ready',
            'missing_gates': [],
            'recommended_actions': [],
            'gates': {'abstraction_discovery_loop': {'passed': True}},
        },
        'abstraction_discovery_evidence': {
            'transfer_outcome_count': 1,
            'latest_transfer_outcome': {'outcome': 'abstraction_transfer_confirmed'},
        },
    }


def evaluator_ledger_fixture(*, leak=False, decision='run_next_safe_experiment'):
    selected = {
        'experiment_kind': 'abstraction_transfer_probe',
        'action_kind': 'non_final_abstraction_transfer_campaign',
        'world': 'hidden_procedural',
        'probe': 'abstraction_transfer_probe',
        'runs_final': False,
        'command': 'python3 first_principles_ai/main.py --abstraction-transfer-campaign',
        'expected_transfer_signal': 'compressed abstraction should improve held-out residuals',
    }
    if leak:
        selected['world'] = 'gravity_world'
    return {
        'schema_version': 1,
        'ledger_kind': 'ai_different.family_outcome_evaluator_ledger',
        'ledger_id': 'outcome-eval-one',
        'ledger_hash': 'outcome-eval-hash-one',
        'chosen_evidence_ids': ['evidence-run-1'],
        'chosen_evidence_senders': ['language_model_2'],
        'decision': {
            'decision_kind': decision,
            'reason': 'label-clean runnable experiment outranks blockers',
            'selected_experiment': selected,
        },
        'selected_experiment': selected,
        'expected_transfer_signal': selected['expected_transfer_signal'],
        'unresolved_blockers': [],
        'runtime_memory_hash_state': {'unchanged': True},
        'project_owned_boundary': {'third_party_checkpoint_used': False},
        'third_party_checkpoint_used': False,
        'label_clean': not leak,
        'leak_terms': ['gravity'] if leak else [],
    }


def downstream_message(contract_id, *, status='satisfied', sender='code_module'):
    return {
        'sender': sender,
        'recipient': 'ai_different',
        'topic': f'evidence.contract.{status}',
        'body': {
            'message_id': f'{contract_id}-{status}-{sender}',
            'contract_id': contract_id,
            'evidence_id': f'{contract_id}-{status}-evidence',
            'status': status,
            'summary': f'{status} contract evidence',
        },
        'evidence': {'contract_id': contract_id, 'status': status},
        'tags': ['evidence', status],
    }


class ExperimentContractsTests(unittest.TestCase):
    def test_contract_message_schema_and_jsonl_export(self):
        contract = build_experiment_contract_from_evaluator(
            evaluator_ledger_fixture(),
            target_recipient='broadcast',
        )
        message = export_experiment_contract_message(
            contract,
            runtime_memory_hash_state={'unchanged': True},
            project_owned_boundary={'third_party_checkpoint_used': False},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            outbox = Path(tmpdir) / 'outbox.jsonl'
            write_contract_outbox_jsonl(outbox, message)
            rows = [
                json.loads(line)
                for line in outbox.read_text(encoding='utf-8').splitlines()
            ]

        self.assertEqual('ai_different', message['sender'])
        self.assertEqual('broadcast', message['recipient'])
        self.assertEqual('experiment_contract', message['body']['response_kind'])
        self.assertEqual('hidden_procedural', message['body']['selected_world'])
        self.assertIn('falsifiable_transfer_signal_reported', message['body']['required_evidence_gates'])
        self.assertTrue(message['body']['label_clean'])
        self.assertFalse(message['body']['third_party_checkpoint_used'])
        self.assertEqual(1, len(rows))

    def test_evaluator_ledger_input_validation(self):
        with self.assertRaisesRegex(ValueError, 'wrong ledger_kind'):
            validate_evaluator_ledger({'ledger_kind': 'wrong', 'decision': {}})
        with self.assertRaisesRegex(ValueError, 'selected_experiment'):
            validate_evaluator_ledger({
                'ledger_kind': 'ai_different.family_outcome_evaluator_ledger',
                'decision': {},
            })

    def test_idempotent_repeat_skips_already_emitted_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            eval_path = Path(tmpdir) / 'eval.json'
            bus_path = Path(tmpdir) / 'bus.jsonl'
            ledger_path = Path(tmpdir) / 'contracts.json'
            outbox = Path(tmpdir) / 'outbox.jsonl'
            eval_path.write_text(json.dumps(evaluator_ledger_fixture()), encoding='utf-8')
            bus_path.write_text('', encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()):
                first = run_experiment_contract_loop(
                    memory_data=memory_fixture(),
                    runtime_memory_path=str(Path(tmpdir) / 'theory-memory.json'),
                    evaluator_ledger_file=eval_path,
                    family_bus_file=bus_path,
                    contract_ledger_file=ledger_path,
                    outbox_file=outbox,
                    git_status_text='',
                    git_ignored_text='',
                )
            with contextlib.redirect_stdout(io.StringIO()):
                second = run_experiment_contract_loop(
                    memory_data=memory_fixture(),
                    runtime_memory_path=str(Path(tmpdir) / 'theory-memory.json'),
                    evaluator_ledger_file=eval_path,
                    family_bus_file=bus_path,
                    contract_ledger_file=ledger_path,
                    outbox_file=outbox,
                    git_status_text='',
                    git_ignored_text='',
                )

        self.assertEqual(1, first['new_contract_count'])
        self.assertEqual(1, first['outbox_count'])
        self.assertEqual(0, second['new_contract_count'])
        self.assertEqual(1, second['skipped_contract_count'])
        self.assertEqual(0, second['outbox_count'])

    def test_appended_downstream_evidence_resolves_contract_without_new_emit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            eval_path = Path(tmpdir) / 'eval.json'
            bus_path = Path(tmpdir) / 'bus.jsonl'
            ledger_path = Path(tmpdir) / 'contracts.json'
            eval_path.write_text(json.dumps(evaluator_ledger_fixture()), encoding='utf-8')
            bus_path.write_text('', encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()):
                first = run_experiment_contract_loop(
                    memory_data=memory_fixture(),
                    runtime_memory_path=str(Path(tmpdir) / 'theory-memory.json'),
                    evaluator_ledger_file=eval_path,
                    family_bus_file=bus_path,
                    contract_ledger_file=ledger_path,
                    git_status_text='',
                    git_ignored_text='',
                )
            ledger = load_experiment_contract_ledger(ledger_path)
            contract_id = ledger['contracts'][0]['contract_id']
            bus_path.write_text(
                json.dumps(downstream_message(contract_id, status='satisfied')) + '\n',
                encoding='utf-8',
            )
            with contextlib.redirect_stdout(io.StringIO()):
                resolved = run_experiment_contract_loop(
                    memory_data=memory_fixture(),
                    runtime_memory_path=str(Path(tmpdir) / 'theory-memory.json'),
                    evaluator_ledger_file=eval_path,
                    family_bus_file=bus_path,
                    contract_ledger_file=ledger_path,
                    git_status_text='',
                    git_ignored_text='',
                )

        self.assertEqual(1, first['new_contract_count'])
        self.assertEqual(1, resolved['resolved_contract_count'])
        self.assertEqual(0, resolved['outbox_count'])
        self.assertEqual(1, resolved['resolved_total'])
        self.assertEqual(0, resolved['open_contract_count'])

    def test_blocked_contract_emits_repair_not_new_experiment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            eval_path = Path(tmpdir) / 'eval.json'
            bus_path = Path(tmpdir) / 'bus.jsonl'
            ledger_path = Path(tmpdir) / 'contracts.json'
            outbox = Path(tmpdir) / 'outbox.jsonl'
            eval_path.write_text(json.dumps(evaluator_ledger_fixture()), encoding='utf-8')
            bus_path.write_text('', encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()):
                run_experiment_contract_loop(
                    memory_data=memory_fixture(),
                    runtime_memory_path=str(Path(tmpdir) / 'theory-memory.json'),
                    evaluator_ledger_file=eval_path,
                    family_bus_file=bus_path,
                    contract_ledger_file=ledger_path,
                    outbox_file=outbox,
                    git_status_text='',
                    git_ignored_text='',
                )
            ledger = load_experiment_contract_ledger(ledger_path)
            contract_id = ledger['contracts'][0]['contract_id']
            bus_path.write_text(
                json.dumps(downstream_message(contract_id, status='blocked')) + '\n',
                encoding='utf-8',
            )
            with contextlib.redirect_stdout(io.StringIO()):
                blocked = run_experiment_contract_loop(
                    memory_data=memory_fixture(),
                    runtime_memory_path=str(Path(tmpdir) / 'theory-memory.json'),
                    evaluator_ledger_file=eval_path,
                    family_bus_file=bus_path,
                    contract_ledger_file=ledger_path,
                    outbox_file=outbox,
                    git_status_text='',
                    git_ignored_text='',
                )
            rows = [
                json.loads(line)
                for line in outbox.read_text(encoding='utf-8').splitlines()
            ]

        self.assertEqual(1, blocked['blocked_contract_count'])
        self.assertEqual(0, blocked['new_contract_count'])
        self.assertEqual(1, blocked['outbox_count'])
        self.assertEqual('experiment_contract_repair', rows[0]['body']['response_kind'])

    def test_malformed_contract_ledger_rejected(self):
        with self.assertRaisesRegex(ValueError, 'wrong ledger_kind'):
            validate_experiment_contract_ledger({
                'ledger_kind': 'wrong',
                'contracts': [],
                'processed_downstream_evidence_ids': [],
                'emitted_evaluator_ledger_ids': [],
                'outgoing_message_ids': [],
            })

    def test_label_leak_guard_blocks_contract_emit(self):
        ledger = empty_experiment_contract_ledger()
        updated, message = update_contract_ledger(
            ledger,
            evaluator_ledger_fixture(leak=True),
            [],
            runtime_memory_hash_state={'unchanged': True},
            project_owned_boundary={'third_party_checkpoint_used': False},
        )

        self.assertIsNone(message)
        self.assertEqual(0, len(updated['contracts']))

    def test_downstream_evidence_classification(self):
        contract = build_experiment_contract_from_evaluator(evaluator_ledger_fixture())
        messages = [
            downstream_message(contract['contract_id'], status='satisfied'),
            downstream_message(contract['contract_id'], status='blocked', sender='funfun'),
        ]
        items = downstream_evidence_for_contracts(messages, [contract])
        statuses = {item['status'] for item in items}

        self.assertEqual({'satisfied', 'blocked'}, statuses)
        self.assertTrue(all(item['label_clean'] for item in items))

    def test_cli_runtime_boundary_and_no_third_party_claim(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            eval_path = Path(tmpdir) / 'eval.json'
            bus_path = Path(tmpdir) / 'bus.jsonl'
            ledger_path = Path(tmpdir) / 'contracts.json'
            outbox = Path(tmpdir) / 'outbox.jsonl'
            runtime_memory = Path(tmpdir) / 'theory-memory.json'
            eval_path.write_text(json.dumps(evaluator_ledger_fixture()), encoding='utf-8')
            bus_path.write_text('', encoding='utf-8')
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                result = run_experiment_contract_loop(
                    memory_data=memory_fixture(),
                    runtime_memory_path=str(runtime_memory),
                    evaluator_ledger_file=eval_path,
                    family_bus_file=bus_path,
                    contract_ledger_file=ledger_path,
                    outbox_file=outbox,
                    git_status_text='',
                    git_ignored_text='',
                )

        self.assertIn('AI_DIFFERENT_EXPERIMENT_CONTRACT ', stdout.getvalue())
        self.assertFalse(runtime_memory.exists())
        self.assertTrue(result['runtime_memory_hash_state']['unchanged'])
        self.assertFalse(result['runtime_memory_mutated'])
        self.assertFalse(result['third_party_checkpoint_used'])
        self.assertEqual([], result['label_leaks'])

    def test_contract_module_has_no_sibling_project_imports(self):
        source = Path(PROJECT_DIR, 'agent', 'experiment_contracts.py').read_text(encoding='utf-8')
        main_source = Path(PROJECT_DIR, 'main.py').read_text(encoding='utf-8')

        self.assertNotIn('Language model 2.0', source)
        self.assertNotIn('orchastratorrrr', source)
        self.assertNotIn('Code Module', source)
        self.assertNotIn('Language model 2.0', main_source)
        self.assertNotIn('orchastratorrrr', main_source)
        self.assertNotIn('Code Module', main_source)


if __name__ == '__main__':
    unittest.main()
