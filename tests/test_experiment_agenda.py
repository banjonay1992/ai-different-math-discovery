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

from agent.experiment_agenda import (  # noqa: E402
    AGENDA_LEDGER_KIND,
    build_experiment_agenda,
    empty_experiment_agenda_ledger,
    load_experiment_agenda_ledger,
    read_agenda_transcript,
    validate_experiment_agenda_ledger,
    write_agenda_outbox_jsonl,
    write_experiment_agenda_ledger,
)
from agent.experiment_contracts import (  # noqa: E402
    CONTRACT_LEDGER_KIND,
    build_experiment_contract_from_evaluator,
    export_experiment_contract_message,
)
from agent.module_chat_adapter import build_module_chat_message  # noqa: E402
from main import run_experiment_agenda_scheduler  # noqa: E402


def memory_fixture():
    return {
        'discovery_readiness': {
            'readiness_score': 0.84,
            'status': 'nearly_ready',
        },
        'abstraction_discovery_evidence': {
            'transfer_outcome_count': 2,
            'latest_transfer_outcome': {'outcome': 'abstraction_transfer_confirmed'},
        },
    }


def evaluator_ledger_fixture(*, missing_language=False):
    selected = {
        'experiment_kind': 'abstraction_transfer_probe',
        'action_kind': 'non_final_abstraction_transfer_campaign',
        'world': '' if missing_language else 'hidden_procedural',
        'probe': 'abstraction_transfer_probe',
        'runs_final': False,
        'command': 'python3 first_principles_ai/main.py --abstraction-transfer-campaign',
        'expected_transfer_signal': 'compressed abstraction should improve held-out residuals',
    }
    return {
        'schema_version': 1,
        'ledger_kind': 'ai_different.family_outcome_evaluator_ledger',
        'ledger_id': 'agenda-eval-one',
        'ledger_hash': 'agenda-eval-hash-one',
        'chosen_evidence_ids': ['agenda-evidence-one'],
        'chosen_evidence_senders': ['language_model_2'],
        'decision': {
            'decision_kind': 'run_next_safe_experiment',
            'reason': 'safe runnable next experiment',
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
    summary = f'{gate} {status} evidence'
    if leak:
        summary = 'gravity label leaked into agenda evidence'
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
        tags=['agenda_evidence', gate, status],
    )


def language_message(*, status='advisory'):
    return build_module_chat_message(
        sender='language_model_2',
        recipient='ai_different',
        topic='language.epoch_agenda',
        body={
            'evidence_id': f'language-{status}',
            'evidence_gate': 'language_epoch_plan',
            'status': status,
            'summary': 'language epoch agenda needs clarification'
            if status == 'missing'
            else 'language epoch agenda available',
        },
        evidence={'evidence_gate': 'language_epoch_plan', 'status': status},
        tags=['language_epoch', status],
    )


def build_once(messages, *, ledger=None, evaluator=None, project_boundary=None):
    updated, message = build_experiment_agenda(
        transcript_messages=messages,
        agenda_ledger=ledger or empty_experiment_agenda_ledger(),
        evaluator_ledger=evaluator if evaluator is not None else evaluator_ledger_fixture(),
        outcome_ledger={},
        contract_ledger={},
        adjudicator_ledger={},
        runtime_memory_data=memory_fixture(),
        runtime_memory_hash_state={'unchanged': True, 'before': 'same', 'after': 'same'},
        project_owned_boundary=project_boundary or {'third_party_checkpoint_used': False},
    )
    return updated, message


class ExperimentAgendaTests(unittest.TestCase):
    def test_agenda_ledger_persistence_load_and_malformed_rejection(self):
        ledger, message = build_once([])
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = Path(tmpdir) / 'agenda.json'
            outbox = Path(tmpdir) / 'outbox.jsonl'
            write_experiment_agenda_ledger(ledger_path, ledger)
            loaded = load_experiment_agenda_ledger(ledger_path)
            write_agenda_outbox_jsonl(outbox, message)
            rows = [json.loads(line) for line in outbox.read_text(encoding='utf-8').splitlines()]

        self.assertEqual(AGENDA_LEDGER_KIND, loaded['ledger_kind'])
        self.assertEqual('experiment_agenda', rows[0]['body']['response_kind'])
        with self.assertRaisesRegex(ValueError, 'wrong ledger_kind'):
            validate_experiment_agenda_ledger({'ledger_kind': 'wrong'})

    def test_transcript_parsing_handles_duplicates_and_invalid_messages(self):
        contract = contract_fixture()
        rows = [
            contract_message(contract),
            language_message(),
            evidence_message(contract['contract_id'], sender='funfun'),
            evidence_message(contract['contract_id'], sender='code_module'),
            evidence_message(contract['contract_id'], sender='code_module'),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            transcript = Path(tmpdir) / 'agenda.jsonl'
            transcript.write_text(
                ''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows)
                + '{"sender":"nope"}\n',
                encoding='utf-8',
            )
            parsed = read_agenda_transcript(transcript)

        self.assertEqual(5, len(parsed['messages']))
        self.assertEqual(1, len(parsed['invalid_messages']))

    def test_priority_missing_math_then_code_then_emit_next_contract(self):
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
        self.assertEqual('emit_next_experiment_contract', third_ledger['latest']['selected_action'])
        self.assertEqual('broadcast', third_message['recipient'])
        self.assertIsNotNone(third_message['body']['compact_next_experiment_contract'])

    def test_safety_project_boundary_and_label_leak_repair_routing(self):
        contract = contract_fixture()
        boundary_ledger, boundary_message = build_once(
            [contract_message(contract)],
            project_boundary={'third_party_checkpoint_used': True},
        )
        self.assertEqual('request_code_repair', boundary_ledger['latest']['selected_action'])
        self.assertEqual('code_module', boundary_message['recipient'])
        self.assertTrue(boundary_message['body']['third_party_checkpoint_used'])

        fresh, _ = build_once([contract_message(contract)])
        leak_ledger, leak_message = build_once(
            [evidence_message(contract['contract_id'], sender='funfun', leak=True)],
            ledger=fresh,
        )
        self.assertEqual('request_math_repair', leak_ledger['latest']['selected_action'])
        self.assertIn('gravity', leak_message['body']['label_leaks'])
        self.assertFalse(leak_message['body']['label_clean'])

    def test_language_clarification_priority_before_next_contract(self):
        ledger, message = build_once([language_message(status='missing')])

        self.assertEqual('language_clarification_needed', ledger['latest']['selected_action'])
        self.assertEqual('language_model_2', message['recipient'])
        self.assertEqual('language_epoch_plan', message['body']['repair_request']['evidence_gate'])

    def test_duplicate_idempotence_and_appended_unlock(self):
        contract = contract_fixture()
        first_ledger, first_message = build_once([contract_message(contract)])
        repeat_ledger, repeat_message = build_once([contract_message(contract)], ledger=first_ledger)
        self.assertIsNotNone(first_message)
        self.assertIsNone(repeat_message)
        self.assertEqual('summarize_noop', repeat_ledger['latest']['selected_action'])

        math_ledger, math_message = build_once(
            [evidence_message(contract['contract_id'], sender='funfun')],
            ledger=first_ledger,
        )
        code_ledger, code_message = build_once(
            [evidence_message(contract['contract_id'], sender='code_module')],
            ledger=math_ledger,
        )
        self.assertEqual('request_code_repair', math_ledger['latest']['selected_action'])
        self.assertEqual('code_module', math_message['recipient'])
        self.assertEqual('emit_next_experiment_contract', code_ledger['latest']['selected_action'])
        self.assertEqual('broadcast', code_message['recipient'])

    def test_cli_status_proof_preserves_runtime_memory(self):
        contract = contract_fixture()
        rows = [
            contract_message(contract),
            language_message(),
            evidence_message(contract['contract_id'], sender='funfun'),
            evidence_message(contract['contract_id'], sender='code_module'),
            evidence_message(contract['contract_id'], sender='code_module'),
        ]
        contract_ledger = {
            'schema_version': 1,
            'ledger_kind': CONTRACT_LEDGER_KIND,
            'contracts': [contract],
            'processed_downstream_evidence_ids': [],
            'emitted_evaluator_ledger_ids': ['agenda-eval-one'],
            'outgoing_message_ids': [],
            'latest': {},
            'ledger_hash': 'agenda-contract-ledger-hash-one',
        }
        adjudicator_ledger = {
            'schema_version': 1,
            'ledger_kind': 'ai_different.cross_module_adjudicator_ledger',
            'processed_message_ids': [],
            'processed_evaluator_ledger_ids': [],
            'contract_states': [],
            'adjudication_records': [],
            'outgoing_response_ids': [],
            'latest': {},
            'ledger_hash': 'agenda-adjudicator-ledger-hash-one',
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            memory_path = tmp / 'runtime-memory.json'
            transcript = tmp / 'family.jsonl'
            evaluator_path = tmp / 'evaluator.json'
            contract_path = tmp / 'contract.json'
            adjudicator_path = tmp / 'adjudicator.json'
            agenda_path = tmp / 'agenda.json'
            outbox = tmp / 'outbox.jsonl'
            memory_path.write_text(json.dumps(memory_fixture()), encoding='utf-8')
            transcript.write_text(
                ''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows),
                encoding='utf-8',
            )
            evaluator_path.write_text(json.dumps(evaluator_ledger_fixture()), encoding='utf-8')
            contract_path.write_text(json.dumps(contract_ledger), encoding='utf-8')
            adjudicator_path.write_text(json.dumps(adjudicator_ledger), encoding='utf-8')
            before = memory_path.read_text(encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()):
                first = run_experiment_agenda_scheduler(
                    memory_data=memory_fixture(),
                    runtime_memory_path=memory_path,
                    transcript_file=transcript,
                    evaluator_ledger_file=evaluator_path,
                    contract_ledger_file=contract_path,
                    adjudicator_ledger_file=adjudicator_path,
                    agenda_ledger_file=agenda_path,
                    outbox_file=outbox,
                    git_status_text='',
                    git_ignored_text='',
                )
            with contextlib.redirect_stdout(io.StringIO()):
                second = run_experiment_agenda_scheduler(
                    memory_data=memory_fixture(),
                    runtime_memory_path=memory_path,
                    transcript_file=transcript,
                    evaluator_ledger_file=evaluator_path,
                    contract_ledger_file=contract_path,
                    adjudicator_ledger_file=adjudicator_path,
                    agenda_ledger_file=agenda_path,
                    outbox_file=outbox,
                    git_status_text='',
                    git_ignored_text='',
                )
            after = memory_path.read_text(encoding='utf-8')

        self.assertEqual('emit_next_experiment_contract', first['selected_action'])
        self.assertEqual(1, first['outbox_count'])
        self.assertEqual('summarize_noop', second['selected_action'])
        self.assertEqual(0, second['outbox_count'])
        self.assertTrue(first['runtime_memory_hash_state']['unchanged'])
        self.assertEqual(before, after)
        self.assertEqual([], first['label_leaks'])
        self.assertFalse(first['third_party_checkpoint_used'])
        self.assertTrue(first['no_sibling_imports'])
        self.assertFalse(first['project_owned_checkpoint_claimed'])

    def test_no_sibling_project_imports_are_introduced(self):
        root = Path(PROJECT_DIR)
        checked = [
            root / 'agent' / 'experiment_agenda.py',
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
