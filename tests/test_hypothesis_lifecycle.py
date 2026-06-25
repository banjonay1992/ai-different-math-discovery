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
    CONTRACT_LEDGER_KIND,
    build_experiment_contract_from_evaluator,
    export_experiment_contract_message,
)
from agent.hypothesis_lifecycle import (  # noqa: E402
    LIFECYCLE_LEDGER_KIND,
    build_hypothesis_lifecycle,
    empty_hypothesis_lifecycle_ledger,
    load_hypothesis_lifecycle_ledger,
    read_lifecycle_transcript,
    validate_hypothesis_lifecycle_ledger,
    write_hypothesis_lifecycle_ledger,
    write_lifecycle_outbox_jsonl,
)
from agent.module_chat_adapter import build_module_chat_message  # noqa: E402
from main import run_hypothesis_lifecycle_curator  # noqa: E402


def memory_fixture():
    return {
        'discovery_readiness': {'readiness_score': 0.85, 'status': 'nearly_ready'},
        'abstraction_discovery_evidence': {
            'transfer_outcome_count': 3,
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
        'expected_transfer_signal': 'compressed abstraction should improve held-out residuals',
    }
    return {
        'schema_version': 1,
        'ledger_kind': 'ai_different.family_outcome_evaluator_ledger',
        'ledger_id': 'lifecycle-eval-one',
        'ledger_hash': 'lifecycle-eval-hash-one',
        'chosen_evidence_ids': ['lifecycle-evidence-one'],
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
    summary = f'{gate} {status} lifecycle evidence'
    if leak:
        summary = 'gravity label leaked into lifecycle evidence'
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
        tags=['hypothesis_lifecycle', gate, status],
    )


def language_message(contract_id, *, status='satisfied'):
    return build_module_chat_message(
        sender='language_model_2',
        recipient='ai_different',
        topic='language.lexicon_protocol',
        body={
            'evidence_id': f'{contract_id}-language-{status}',
            'contract_id': contract_id,
            'evidence_gate': 'language_epoch_plan',
            'status': status,
            'summary': 'language protocol evidence',
        },
        evidence={
            'contract_id': contract_id,
            'evidence_gate': 'language_epoch_plan',
            'status': status,
        },
        tags=['language_protocol', status],
    )


def lifecycle_message(hypothesis_id, *, status='resolved'):
    return build_module_chat_message(
        sender='ai_different',
        recipient='broadcast',
        topic='ai_different.hypothesis_lifecycle',
        body={
            'response_kind': 'hypothesis_lifecycle',
            'hypothesis_id': hypothesis_id,
            'selected_action': 'mark_resolved' if status == 'resolved' else 'retire_blocked_hypothesis',
            'lifecycle_state': status,
            'status': status,
        },
        evidence={'hypothesis_id': hypothesis_id, 'status': status},
        tags=['hypothesis_lifecycle', status],
    )


def build_once(messages, *, ledger=None, evaluator=None, project_boundary=None):
    updated, message = build_hypothesis_lifecycle(
        transcript_messages=messages,
        lifecycle_ledger=ledger or empty_hypothesis_lifecycle_ledger(),
        evaluator_ledger=evaluator if evaluator is not None else evaluator_ledger_fixture(),
        outcome_ledger={},
        contract_ledger={},
        adjudicator_ledger={},
        agenda_ledger={},
        runtime_memory_data=memory_fixture(),
        runtime_memory_hash_state={'unchanged': True, 'before': 'same', 'after': 'same'},
        project_owned_boundary=project_boundary or {'third_party_checkpoint_used': False},
    )
    return updated, message


class HypothesisLifecycleTests(unittest.TestCase):
    def test_lifecycle_ledger_persistence_load_and_malformed_rejection(self):
        contract = contract_fixture()
        ledger, message = build_once([contract_message(contract)])
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = Path(tmpdir) / 'lifecycle.json'
            outbox = Path(tmpdir) / 'outbox.jsonl'
            write_hypothesis_lifecycle_ledger(ledger_path, ledger)
            loaded = load_hypothesis_lifecycle_ledger(ledger_path)
            write_lifecycle_outbox_jsonl(outbox, message)
            rows = [json.loads(line) for line in outbox.read_text(encoding='utf-8').splitlines()]

        self.assertEqual(LIFECYCLE_LEDGER_KIND, loaded['ledger_kind'])
        self.assertEqual('hypothesis_lifecycle', rows[0]['body']['response_kind'])
        with self.assertRaisesRegex(ValueError, 'wrong ledger_kind'):
            validate_hypothesis_lifecycle_ledger({'ledger_kind': 'wrong'})

    def test_hypothesis_extraction_lineage_and_transcript_invalids(self):
        contract = contract_fixture()
        with tempfile.TemporaryDirectory() as tmpdir:
            transcript = Path(tmpdir) / 'lifecycle.jsonl'
            transcript.write_text(
                json.dumps(contract_message(contract), sort_keys=True) + '\n'
                + json.dumps(evidence_message(contract['contract_id'], sender='funfun'), sort_keys=True) + '\n'
                + '{"sender":"bad"}\n',
                encoding='utf-8',
            )
            parsed = read_lifecycle_transcript(transcript)
        ledger, _ = build_once(parsed['messages'])

        self.assertEqual(2, len(parsed['messages']))
        self.assertEqual(1, len(parsed['invalid_messages']))
        self.assertEqual('hypothesis:' + contract['contract_id'], ledger['hypotheses'][0]['hypothesis_id'])
        self.assertIn('transcript_contract', ledger['hypotheses'][0]['lineage'])
        self.assertIn('math_proof', ledger['hypotheses'][0]['satisfied_evidence_gates'])

    def test_priority_missing_math_code_language_then_resolve(self):
        contract = contract_fixture()
        first_ledger, first_message = build_once([contract_message(contract)])
        self.assertEqual('request_missing_math_evidence', first_ledger['latest']['selected_action'])
        self.assertEqual('funfun', first_message['recipient'])

        second_ledger, second_message = build_once(
            [evidence_message(contract['contract_id'], sender='funfun')],
            ledger=first_ledger,
        )
        self.assertEqual('request_missing_code_evidence', second_ledger['latest']['selected_action'])
        self.assertEqual('code_module', second_message['recipient'])

        third_ledger, third_message = build_once(
            [evidence_message(contract['contract_id'], sender='code_module')],
            ledger=second_ledger,
        )
        self.assertEqual('request_language_protocol_clarification', third_ledger['latest']['selected_action'])
        self.assertEqual('language_model_2', third_message['recipient'])

        fourth_ledger, fourth_message = build_once(
            [language_message(contract['contract_id'])],
            ledger=third_ledger,
        )
        self.assertEqual('mark_resolved', fourth_ledger['latest']['selected_action'])
        self.assertEqual('broadcast', fourth_message['recipient'])
        self.assertEqual(1, fourth_ledger['latest']['state_counts']['resolved'])

    def test_blocked_retirement_and_label_boundary_repair(self):
        contract = contract_fixture()
        ledger, _ = build_once([contract_message(contract)])
        blocked_ledger, blocked_message = build_once(
            [evidence_message(contract['contract_id'], sender='code_module', status='failed')],
            ledger=ledger,
        )
        self.assertEqual('retire_blocked_hypothesis', blocked_ledger['latest']['selected_action'])
        self.assertEqual('broadcast', blocked_message['recipient'])

        fresh, _ = build_once([contract_message(contract)])
        leak_ledger, leak_message = build_once(
            [evidence_message(contract['contract_id'], sender='funfun', leak=True)],
            ledger=fresh,
        )
        self.assertEqual('request_missing_math_evidence', leak_ledger['latest']['selected_action'])
        self.assertIn('gravity', leak_message['body']['label_leaks'])
        self.assertFalse(leak_message['body']['label_clean'])

        boundary_ledger, boundary_message = build_once(
            [contract_message(contract)],
            project_boundary={'third_party_checkpoint_used': True},
        )
        self.assertEqual('request_missing_code_evidence', boundary_ledger['latest']['selected_action'])
        self.assertEqual('code_module', boundary_message['recipient'])

    def test_duplicate_idempotence_and_refinement_scheduling_once(self):
        contract = contract_fixture()
        ledger, first_message = build_once([contract_message(contract)])
        repeat_ledger, repeat_message = build_once([contract_message(contract)], ledger=ledger)
        self.assertIsNotNone(first_message)
        self.assertIsNone(repeat_message)
        self.assertEqual('summarize_noop', repeat_ledger['latest']['selected_action'])

        math_ledger, _ = build_once(
            [evidence_message(contract['contract_id'], sender='funfun')],
            ledger=ledger,
        )
        code_ledger, _ = build_once(
            [evidence_message(contract['contract_id'], sender='code_module')],
            ledger=math_ledger,
        )
        resolved_ledger, resolved_message = build_once(
            [language_message(contract['contract_id'])],
            ledger=code_ledger,
        )
        hypothesis_id = resolved_message['body']['hypothesis_id']
        refined_ledger, refined_message = build_once(
            [lifecycle_message(hypothesis_id)],
            ledger=resolved_ledger,
        )
        repeated_refine_ledger, repeated_refine_message = build_once(
            [lifecycle_message(hypothesis_id)],
            ledger=refined_ledger,
        )

        self.assertEqual('mark_resolved', resolved_ledger['latest']['selected_action'])
        self.assertEqual('refine_next_hypothesis', refined_ledger['latest']['selected_action'])
        self.assertEqual('broadcast', refined_message['recipient'])
        self.assertEqual('summarize_noop', repeated_refine_ledger['latest']['selected_action'])
        self.assertIsNone(repeated_refine_message)

    def test_cli_status_proof_preserves_runtime_memory(self):
        contract = contract_fixture()
        rows = [
            contract_message(contract),
            evidence_message(contract['contract_id'], sender='funfun'),
            evidence_message(contract['contract_id'], sender='code_module'),
            language_message(contract['contract_id']),
            evidence_message(contract['contract_id'], sender='code_module'),
        ]
        contract_ledger = {
            'schema_version': 1,
            'ledger_kind': CONTRACT_LEDGER_KIND,
            'contracts': [contract],
            'processed_downstream_evidence_ids': [],
            'emitted_evaluator_ledger_ids': ['lifecycle-eval-one'],
            'outgoing_message_ids': [],
            'latest': {},
            'ledger_hash': 'lifecycle-contract-ledger-hash-one',
        }
        empty_adjudicator = {
            'schema_version': 1,
            'ledger_kind': 'ai_different.cross_module_adjudicator_ledger',
            'processed_message_ids': [],
            'processed_evaluator_ledger_ids': [],
            'contract_states': [],
            'adjudication_records': [],
            'outgoing_response_ids': [],
            'latest': {},
            'ledger_hash': 'lifecycle-adjudicator-hash-one',
        }
        empty_agenda = {
            'schema_version': 1,
            'ledger_kind': 'ai_different.experiment_agenda_ledger',
            'processed_message_ids': [],
            'processed_source_hashes': [],
            'scheduled_candidate_ids': [],
            'hypotheses': [],
            'agenda_records': [],
            'outgoing_response_ids': [],
            'latest': {},
            'ledger_hash': 'lifecycle-agenda-hash-one',
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            memory_path = tmp / 'runtime-memory.json'
            transcript = tmp / 'family.jsonl'
            evaluator_path = tmp / 'evaluator.json'
            contract_path = tmp / 'contract.json'
            adjudicator_path = tmp / 'adjudicator.json'
            agenda_path = tmp / 'agenda.json'
            lifecycle_path = tmp / 'lifecycle.json'
            outbox = tmp / 'outbox.jsonl'
            memory_path.write_text(json.dumps(memory_fixture()), encoding='utf-8')
            transcript.write_text(
                ''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows),
                encoding='utf-8',
            )
            evaluator_path.write_text(json.dumps(evaluator_ledger_fixture()), encoding='utf-8')
            contract_path.write_text(json.dumps(contract_ledger), encoding='utf-8')
            adjudicator_path.write_text(json.dumps(empty_adjudicator), encoding='utf-8')
            agenda_path.write_text(json.dumps(empty_agenda), encoding='utf-8')
            before = memory_path.read_text(encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()):
                first = run_hypothesis_lifecycle_curator(
                    memory_data=memory_fixture(),
                    runtime_memory_path=memory_path,
                    transcript_file=transcript,
                    evaluator_ledger_file=evaluator_path,
                    contract_ledger_file=contract_path,
                    adjudicator_ledger_file=adjudicator_path,
                    agenda_ledger_file=agenda_path,
                    lifecycle_ledger_file=lifecycle_path,
                    outbox_file=outbox,
                    git_status_text='',
                    git_ignored_text='',
                )
            with contextlib.redirect_stdout(io.StringIO()):
                second = run_hypothesis_lifecycle_curator(
                    memory_data=memory_fixture(),
                    runtime_memory_path=memory_path,
                    transcript_file=transcript,
                    evaluator_ledger_file=evaluator_path,
                    contract_ledger_file=contract_path,
                    adjudicator_ledger_file=adjudicator_path,
                    agenda_ledger_file=agenda_path,
                    lifecycle_ledger_file=lifecycle_path,
                    outbox_file=outbox,
                    git_status_text='',
                    git_ignored_text='',
                )
            after = memory_path.read_text(encoding='utf-8')

        self.assertEqual('mark_resolved', first['selected_action'])
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
            root / 'agent' / 'hypothesis_lifecycle.py',
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
