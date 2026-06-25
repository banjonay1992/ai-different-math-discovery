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

from agent.cross_module_adjudicator import (  # noqa: E402
    ADJUDICATOR_LEDGER_KIND,
    build_adjudication,
    empty_adjudicator_ledger,
    load_adjudicator_ledger,
    read_family_transcript,
    validate_adjudicator_ledger,
    write_adjudication_outbox_jsonl,
    write_adjudicator_ledger,
)
from agent.experiment_contracts import (  # noqa: E402
    CONTRACT_LEDGER_KIND,
    build_experiment_contract_from_evaluator,
    export_experiment_contract_message,
)
from agent.module_chat_adapter import build_module_chat_message  # noqa: E402
from main import run_cross_module_adjudicator  # noqa: E402


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


def evaluator_ledger_fixture():
    selected = {
        'experiment_kind': 'abstraction_transfer_probe',
        'action_kind': 'non_final_abstraction_transfer_campaign',
        'world': 'hidden_procedural',
        'probe': 'abstraction_transfer_probe',
        'runs_final': False,
        'command': 'python3 first_principles_ai/main.py --abstraction-transfer-campaign',
        'expected_transfer_signal': 'compressed abstraction should improve held-out residuals',
    }
    return {
        'schema_version': 1,
        'ledger_kind': 'ai_different.family_outcome_evaluator_ledger',
        'ledger_id': 'outcome-eval-one',
        'ledger_hash': 'outcome-eval-hash-one',
        'chosen_evidence_ids': ['evidence-run-1'],
        'chosen_evidence_senders': ['language_model_2'],
        'decision': {
            'decision_kind': 'run_next_safe_experiment',
            'reason': 'label-clean runnable experiment outranks blockers',
            'selected_experiment': selected,
        },
        'selected_experiment': selected,
        'expected_transfer_signal': selected['expected_transfer_signal'],
        'unresolved_blockers': [],
        'runtime_memory_hash_state': {'unchanged': True},
        'project_owned_boundary': {'third_party_checkpoint_used': False},
        'third_party_checkpoint_used': False,
        'label_clean': True,
        'leak_terms': [],
    }


def contract_fixture():
    return build_experiment_contract_from_evaluator(evaluator_ledger_fixture())


def contract_message(contract):
    return export_experiment_contract_message(
        contract,
        runtime_memory_hash_state={'unchanged': True},
        project_owned_boundary={'third_party_checkpoint_used': False},
    )


def evidence_message(contract_id, *, sender, status='satisfied', gate=None, leak=False):
    if gate is None:
        gate = 'math_proof' if sender == 'funfun' else 'code_proof'
    summary = f'{gate} {status} downstream proof'
    if leak:
        summary = 'gravity label leaked into downstream proof'
    return build_module_chat_message(
        sender=sender,
        recipient='ai_different',
        topic=f'{sender}.{gate}.{status}',
        body={
            'evidence_id': f'{contract_id}-{sender}-{gate}-{status}',
            'contract_id': contract_id,
            'evidence_gate': gate,
            'status': status,
            'summary': summary,
        },
        evidence={
            'contract_id': contract_id,
            'evidence_gate': gate,
            'status': status,
        },
        tags=['evidence', gate, status],
    )


def language_turn_message():
    return build_module_chat_message(
        sender='language_model_2',
        recipient='ai_different',
        topic='language.turn_plan',
        body={
            'message_id': 'language-plan-one',
            'summary': 'next speaker should provide proof evidence',
            'status': 'advisory',
        },
        evidence={'status': 'advisory'},
        tags=['turn_plan'],
    )


def build_once(messages, *, ledger=None, evaluator=None, project_boundary=None):
    updated, message = build_adjudication(
        transcript_messages=messages,
        adjudicator_ledger=ledger or empty_adjudicator_ledger(),
        evaluator_ledger=evaluator or {},
        contract_ledger={},
        runtime_memory_hash_state={'unchanged': True, 'before': 'same', 'after': 'same'},
        project_owned_boundary=project_boundary or {'third_party_checkpoint_used': False},
    )
    return updated, message


class CrossModuleAdjudicatorTests(unittest.TestCase):
    def test_transcript_parsing_accepts_three_module_family_and_rejects_malformed(self):
        contract = contract_fixture()
        rows = [
            contract_message(contract),
            language_turn_message(),
            evidence_message(contract['contract_id'], sender='funfun'),
            evidence_message(contract['contract_id'], sender='code_module'),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            transcript = Path(tmpdir) / 'family.jsonl'
            transcript.write_text(
                ''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows)
                + '{"sender": "not_allowed"}\n',
                encoding='utf-8',
            )
            parsed = read_family_transcript(transcript)

        self.assertEqual(4, len(parsed['messages']))
        self.assertEqual(1, len(parsed['invalid_messages']))
        self.assertEqual({'ai_different', 'language_model_2', 'funfun', 'code_module'}, {
            message['sender'] for message in parsed['messages']
        })

    def test_priority_requests_math_before_code_then_resolves(self):
        contract = contract_fixture()
        first_ledger, first_message = build_once([contract_message(contract)])
        self.assertEqual('request_math_repair', first_ledger['latest']['selected_action'])
        self.assertEqual('funfun', first_message['recipient'])

        second_ledger, second_message = build_once(
            [evidence_message(contract['contract_id'], sender='funfun')],
            ledger=first_ledger,
        )
        self.assertEqual('request_code_repair', second_ledger['latest']['selected_action'])
        self.assertEqual('code_module', second_message['recipient'])

        third_ledger, third_message = build_once(
            [evidence_message(contract['contract_id'], sender='code_module')],
            ledger=second_ledger,
        )
        self.assertEqual('resolve_contract', third_ledger['latest']['selected_action'])
        self.assertEqual('broadcast', third_message['recipient'])
        self.assertEqual(1, third_ledger['latest']['resolved_contract_count'])

    def test_idempotent_repeat_skips_duplicate_outbox(self):
        contract = contract_fixture()
        transcript = [contract_message(contract)]
        first_ledger, first_message = build_once(transcript)
        second_ledger, second_message = build_once(transcript, ledger=first_ledger)

        self.assertIsNotNone(first_message)
        self.assertIsNone(second_message)
        self.assertEqual('summarize_noop', second_ledger['latest']['selected_action'])
        self.assertEqual(0, second_ledger['latest']['outbox_count'])
        self.assertEqual(1, second_ledger['latest']['skipped_message_count'])

    def test_blocked_repair_routing_and_label_leak_guard(self):
        contract = contract_fixture()
        ledger, _ = build_once([contract_message(contract)])
        code_blocked, code_message = build_once(
            [evidence_message(contract['contract_id'], sender='code_module', status='blocked')],
            ledger=ledger,
        )
        self.assertEqual('request_code_repair', code_blocked['latest']['selected_action'])
        self.assertEqual('code_module', code_message['recipient'])

        fresh, _ = build_once([contract_message(contract)])
        leak_ledger, leak_message = build_once(
            [evidence_message(contract['contract_id'], sender='funfun', leak=True)],
            ledger=fresh,
        )
        self.assertEqual('request_math_repair', leak_ledger['latest']['selected_action'])
        self.assertEqual('funfun', leak_message['recipient'])
        self.assertIn('gravity', leak_message['body']['label_leaks'])
        self.assertFalse(leak_message['body']['label_clean'])

    def test_evaluator_can_spawn_when_no_open_contract_exists(self):
        ledger, message = build_once([], evaluator=evaluator_ledger_fixture())

        self.assertEqual('emit_next_contract', ledger['latest']['selected_action'])
        self.assertEqual('broadcast', message['recipient'])
        self.assertEqual(1, ledger['latest']['outbox_count'])
        self.assertIn('outcome-eval-one', ledger['processed_evaluator_ledger_ids'])
        self.assertFalse(message['body']['third_party_checkpoint_used'])

    def test_project_owned_boundary_failure_has_priority(self):
        contract = contract_fixture()
        ledger, message = build_once(
            [contract_message(contract), evidence_message(contract['contract_id'], sender='funfun')],
            project_boundary={'third_party_checkpoint_used': True},
        )

        self.assertEqual('request_code_repair', ledger['latest']['selected_action'])
        self.assertEqual('code_module', message['recipient'])
        self.assertTrue(message['body']['third_party_checkpoint_used'])

    def test_malformed_adjudicator_ledger_rejected_and_jsonl_export_shape(self):
        with self.assertRaisesRegex(ValueError, 'wrong ledger_kind'):
            validate_adjudicator_ledger({'ledger_kind': 'wrong'})

        contract = contract_fixture()
        ledger, message = build_once([contract_message(contract)])
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = Path(tmpdir) / 'adjudicator.json'
            outbox = Path(tmpdir) / 'outbox.jsonl'
            write_adjudicator_ledger(ledger_path, ledger)
            loaded = load_adjudicator_ledger(ledger_path)
            write_adjudication_outbox_jsonl(outbox, message)
            rows = [json.loads(line) for line in outbox.read_text(encoding='utf-8').splitlines()]

        self.assertEqual(ADJUDICATOR_LEDGER_KIND, loaded['ledger_kind'])
        self.assertEqual('cross_module_adjudication', rows[0]['body']['response_kind'])
        self.assertEqual('request_math_repair', rows[0]['body']['selected_action'])

    def test_cli_status_proof_preserves_runtime_memory(self):
        contract = contract_fixture()
        contract_ledger = {
            'schema_version': 1,
            'ledger_kind': CONTRACT_LEDGER_KIND,
            'contracts': [contract],
            'processed_downstream_evidence_ids': [],
            'emitted_evaluator_ledger_ids': ['outcome-eval-one'],
            'outgoing_message_ids': [],
            'latest': {},
            'ledger_hash': 'contract-ledger-hash-one',
        }
        rows = [
            contract_message(contract),
            language_turn_message(),
            evidence_message(contract['contract_id'], sender='funfun'),
            evidence_message(contract['contract_id'], sender='code_module'),
            evidence_message(contract['contract_id'], sender='code_module'),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            memory_path = tmp / 'runtime-memory.json'
            transcript = tmp / 'family.jsonl'
            evaluator_path = tmp / 'evaluator.json'
            contract_path = tmp / 'contract.json'
            ledger_path = tmp / 'adjudicator.json'
            outbox = tmp / 'outbox.jsonl'
            memory_path.write_text(json.dumps(memory_fixture()), encoding='utf-8')
            transcript.write_text(
                ''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows),
                encoding='utf-8',
            )
            evaluator_path.write_text(json.dumps(evaluator_ledger_fixture()), encoding='utf-8')
            contract_path.write_text(json.dumps(contract_ledger), encoding='utf-8')
            before = memory_path.read_text(encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()):
                first = run_cross_module_adjudicator(
                    memory_data=memory_fixture(),
                    runtime_memory_path=memory_path,
                    transcript_file=transcript,
                    evaluator_ledger_file=evaluator_path,
                    contract_ledger_file=contract_path,
                    adjudicator_ledger_file=ledger_path,
                    outbox_file=outbox,
                    git_status_text='',
                    git_ignored_text='',
                )
            with contextlib.redirect_stdout(io.StringIO()):
                second = run_cross_module_adjudicator(
                    memory_data=memory_fixture(),
                    runtime_memory_path=memory_path,
                    transcript_file=transcript,
                    evaluator_ledger_file=evaluator_path,
                    contract_ledger_file=contract_path,
                    adjudicator_ledger_file=ledger_path,
                    outbox_file=outbox,
                    git_status_text='',
                    git_ignored_text='',
                )
            after = memory_path.read_text(encoding='utf-8')

        self.assertEqual('resolve_contract', first['selected_action'])
        self.assertEqual(1, first['outbox_count'])
        self.assertEqual('summarize_noop', second['selected_action'])
        self.assertEqual(0, second['outbox_count'])
        self.assertTrue(first['runtime_memory_hash_state']['unchanged'])
        self.assertEqual(before, after)
        self.assertEqual([], first['label_leaks'])
        self.assertFalse(first['third_party_checkpoint_used'])
        self.assertTrue(first['no_sibling_imports'])

    def test_no_sibling_project_imports_are_introduced(self):
        root = Path(PROJECT_DIR)
        checked = [
            root / 'agent' / 'cross_module_adjudicator.py',
            root / 'main.py',
        ]
        forbidden = [
            'Language model 2.0',
            'Code Module',
            'orchastratorrrr',
            'from funfun',
            'import funfun',
        ]
        for path in checked:
            text = path.read_text(encoding='utf-8')
            for token in forbidden:
                self.assertNotIn(token, text)


if __name__ == '__main__':
    unittest.main()
