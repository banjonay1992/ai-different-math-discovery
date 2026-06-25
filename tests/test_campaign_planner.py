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

from agent.campaign_planner import (  # noqa: E402
    CAMPAIGN_LEDGER_KIND,
    build_experiment_campaign,
    empty_campaign_ledger,
    load_campaign_ledger,
    read_campaign_transcript,
    validate_campaign_ledger,
    write_campaign_ledger,
    write_campaign_outbox_jsonl,
)
from agent.experiment_contracts import (  # noqa: E402
    CONTRACT_LEDGER_KIND,
    build_experiment_contract_from_evaluator,
    export_experiment_contract_message,
)
from agent.module_chat_adapter import build_module_chat_message  # noqa: E402
from main import run_experiment_campaign_planner  # noqa: E402


def memory_fixture():
    return {
        'discovery_readiness': {'readiness_score': 0.87, 'status': 'nearly_ready'},
        'abstraction_discovery_evidence': {'transfer_outcome_count': 5},
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
        'ledger_id': 'campaign-eval-one',
        'ledger_hash': 'campaign-eval-hash-one',
        'chosen_evidence_ids': ['campaign-evidence-one'],
        'chosen_evidence_senders': ['language_model_2'],
        'decision': {'decision_kind': 'run_next_safe_experiment', 'selected_experiment': selected},
        'selected_experiment': selected,
        'expected_transfer_signal': selected['expected_transfer_signal'],
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
    summary = f'{gate} {status} campaign evidence'
    if leak:
        summary = 'gravity label leaked into campaign evidence'
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
        evidence={'contract_id': contract_id, 'evidence_gate': gate, 'status': status},
        tags=['campaign', gate, status],
    )


def language_message(contract_id, *, status='satisfied'):
    return evidence_message(
        contract_id,
        sender='language_model_2',
        status=status,
        gate='language_epoch_plan',
    )


def prior_campaign_message(hypothesis_id):
    return build_module_chat_message(
        sender='ai_different',
        recipient='broadcast',
        topic='ai_different.experiment_campaign',
        body={
            'response_kind': 'experiment_campaign',
            'campaign_id': 'prior-campaign-one',
            'hypothesis_id': hypothesis_id,
            'campaign_type': 'emit_acceptance_bundle',
            'selected_action': 'emit_acceptance_bundle',
            'accepted_evidence': ['math_proof', 'code_proof', 'language_epoch_plan'],
            'missing_evidence': [],
            'rejected_evidence': [],
        },
        evidence={'hypothesis_id': hypothesis_id},
        tags=['campaign'],
    )


def build_once(messages, *, ledger=None, project_boundary=None):
    updated, message = build_experiment_campaign(
        transcript_messages=messages,
        campaign_ledger=ledger or empty_campaign_ledger(),
        evaluator_ledger=evaluator_ledger_fixture(),
        outcome_ledger={},
        contract_ledger={},
        adjudicator_ledger={},
        agenda_ledger={},
        lifecycle_ledger={},
        scorecard_ledger={},
        runtime_memory_data=memory_fixture(),
        runtime_memory_hash_state={'unchanged': True, 'before': 'same', 'after': 'same'},
        project_owned_boundary=project_boundary or {'third_party_checkpoint_used': False},
    )
    return updated, message


class CampaignPlannerTests(unittest.TestCase):
    def test_campaign_ledger_persistence_load_and_malformed_rejection(self):
        contract = contract_fixture()
        ledger, message = build_once([contract_message(contract)])
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = Path(tmpdir) / 'campaign.json'
            outbox = Path(tmpdir) / 'outbox.jsonl'
            write_campaign_ledger(ledger_path, ledger)
            loaded = load_campaign_ledger(ledger_path)
            write_campaign_outbox_jsonl(outbox, message)
            rows = [json.loads(line) for line in outbox.read_text(encoding='utf-8').splitlines()]

        self.assertEqual(CAMPAIGN_LEDGER_KIND, loaded['ledger_kind'])
        self.assertEqual('experiment_campaign', rows[0]['body']['response_kind'])
        with self.assertRaisesRegex(ValueError, 'wrong ledger_kind'):
            validate_campaign_ledger({'ledger_kind': 'wrong'})

    def test_scorecard_to_campaign_extraction_and_invalids(self):
        contract = contract_fixture()
        with tempfile.TemporaryDirectory() as tmpdir:
            transcript = Path(tmpdir) / 'campaign.jsonl'
            transcript.write_text(
                json.dumps(contract_message(contract), sort_keys=True) + '\n'
                + json.dumps(evidence_message(contract['contract_id'], sender='funfun'), sort_keys=True) + '\n'
                + '{"sender":"bad"}\n',
                encoding='utf-8',
            )
            parsed = read_campaign_transcript(transcript)
        ledger, _ = build_once(parsed['messages'])

        self.assertEqual(2, len(parsed['messages']))
        self.assertEqual(1, len(parsed['invalid_messages']))
        campaign = ledger['campaigns'][0]
        self.assertEqual('hypothesis:' + contract['contract_id'], campaign['hypothesis_id'])
        self.assertIn('math_proof', campaign['accepted_evidence'])
        self.assertEqual('request_code_gate', campaign['campaign_type'])

    def test_priority_protocol_math_code_language_then_acceptance_bundle(self):
        contract = contract_fixture()
        protocol_ledger, protocol_message = build_once([
            contract_message(contract),
            language_message(contract['contract_id'], status='missing'),
        ])
        self.assertEqual('protocol_or_readiness_repair', protocol_ledger['latest']['selected_action'])
        self.assertEqual('language_model_2', protocol_message['recipient'])

        first_ledger, first_message = build_once([contract_message(contract)])
        self.assertEqual('request_math_gate', first_ledger['latest']['selected_action'])
        self.assertEqual('funfun', first_message['recipient'])

        second_ledger, second_message = build_once(
            [evidence_message(contract['contract_id'], sender='funfun')],
            ledger=first_ledger,
        )
        self.assertEqual('request_code_gate', second_ledger['latest']['selected_action'])
        self.assertEqual('code_module', second_message['recipient'])

        third_ledger, third_message = build_once(
            [evidence_message(contract['contract_id'], sender='code_module')],
            ledger=second_ledger,
        )
        self.assertEqual('request_language_gate', third_ledger['latest']['selected_action'])
        self.assertEqual('language_model_2', third_message['recipient'])

        fourth_ledger, fourth_message = build_once(
            [language_message(contract['contract_id'])],
            ledger=third_ledger,
        )
        self.assertEqual('emit_acceptance_bundle', fourth_ledger['latest']['selected_action'])
        self.assertEqual('broadcast', fourth_message['recipient'])
        self.assertTrue(fourth_message['body']['acceptance_criteria'])

    def test_stale_retirement_boundary_and_refinement_once(self):
        contract = contract_fixture()
        base, _ = build_once([
            contract_message(contract),
            evidence_message(contract['contract_id'], sender='funfun'),
            language_message(contract['contract_id']),
        ])
        retired, retired_message = build_once(
            [evidence_message(contract['contract_id'], sender='code_module', status='failed')],
            ledger=base,
        )
        self.assertEqual('retire_stale_or_blocked_line', retired['latest']['selected_action'])
        self.assertEqual('broadcast', retired_message['recipient'])

        leak_ledger, leak_message = build_once([
            contract_message(contract),
            evidence_message(contract['contract_id'], sender='funfun', leak=True),
        ])
        self.assertEqual('safety_label_or_project_owned_repair', leak_ledger['latest']['selected_action'])
        self.assertEqual('code_module', leak_message['recipient'])
        self.assertIn('gravity', leak_message['body']['label_leaks'])

        boundary_ledger, boundary_message = build_once(
            [contract_message(contract)],
            project_boundary={'third_party_checkpoint_used': True},
        )
        self.assertEqual('safety_label_or_project_owned_repair', boundary_ledger['latest']['selected_action'])
        self.assertEqual('code_module', boundary_message['recipient'])

        ready, ready_message = build_once([
            contract_message(contract),
            evidence_message(contract['contract_id'], sender='funfun'),
            evidence_message(contract['contract_id'], sender='code_module'),
            language_message(contract['contract_id']),
        ])
        hypothesis_id = ready_message['body']['hypothesis_id']
        refined, refined_message = build_once([prior_campaign_message(hypothesis_id)], ledger=ready)
        repeated, repeated_message = build_once([prior_campaign_message(hypothesis_id)], ledger=refined)
        self.assertEqual('continue_refinement', refined['latest']['selected_action'])
        self.assertEqual('broadcast', refined_message['recipient'])
        self.assertEqual('summarize_noop', repeated['latest']['selected_action'])
        self.assertIsNone(repeated_message)

    def test_duplicate_idempotence(self):
        contract = contract_fixture()
        ledger, message = build_once([contract_message(contract)])
        repeat, repeat_message = build_once([contract_message(contract)], ledger=ledger)

        self.assertIsNotNone(message)
        self.assertIsNone(repeat_message)
        self.assertEqual('summarize_noop', repeat['latest']['selected_action'])

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
            'emitted_evaluator_ledger_ids': ['campaign-eval-one'],
            'outgoing_message_ids': [],
            'latest': {},
            'ledger_hash': 'campaign-contract-ledger-hash-one',
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
            'ledger_hash': 'campaign-adjudicator-hash-one',
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
            'ledger_hash': 'campaign-agenda-hash-one',
        }
        empty_lifecycle = {
            'schema_version': 1,
            'ledger_kind': 'ai_different.hypothesis_lifecycle_ledger',
            'processed_message_ids': [],
            'processed_source_hashes': [],
            'resolved_hypothesis_ids': [],
            'retired_hypothesis_ids': [],
            'refined_hypothesis_ids': [],
            'hypotheses': [],
            'lifecycle_records': [],
            'outgoing_response_ids': [],
            'latest': {},
            'ledger_hash': 'campaign-lifecycle-hash-one',
        }
        empty_scorecard = {
            'schema_version': 1,
            'ledger_kind': 'ai_different.experiment_evidence_scorecard_ledger',
            'processed_message_ids': [],
            'processed_source_hashes': [],
            'resolved_hypothesis_ids': [],
            'retired_hypothesis_ids': [],
            'refined_hypothesis_ids': [],
            'scorecards': [],
            'scorecard_records': [],
            'outgoing_response_ids': [],
            'latest': {},
            'ledger_hash': 'campaign-scorecard-hash-one',
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
            scorecard_path = tmp / 'scorecard.json'
            campaign_path = tmp / 'campaign.json'
            outbox = tmp / 'outbox.jsonl'
            memory_path.write_text(json.dumps(memory_fixture()), encoding='utf-8')
            transcript.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows), encoding='utf-8')
            evaluator_path.write_text(json.dumps(evaluator_ledger_fixture()), encoding='utf-8')
            contract_path.write_text(json.dumps(contract_ledger), encoding='utf-8')
            adjudicator_path.write_text(json.dumps(empty_adjudicator), encoding='utf-8')
            agenda_path.write_text(json.dumps(empty_agenda), encoding='utf-8')
            lifecycle_path.write_text(json.dumps(empty_lifecycle), encoding='utf-8')
            scorecard_path.write_text(json.dumps(empty_scorecard), encoding='utf-8')
            before = memory_path.read_text(encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()):
                first = run_experiment_campaign_planner(
                    memory_data=memory_fixture(),
                    runtime_memory_path=memory_path,
                    transcript_file=transcript,
                    evaluator_ledger_file=evaluator_path,
                    contract_ledger_file=contract_path,
                    adjudicator_ledger_file=adjudicator_path,
                    agenda_ledger_file=agenda_path,
                    lifecycle_ledger_file=lifecycle_path,
                    scorecard_ledger_file=scorecard_path,
                    campaign_ledger_file=campaign_path,
                    outbox_file=outbox,
                    git_status_text='',
                    git_ignored_text='',
                )
            with contextlib.redirect_stdout(io.StringIO()):
                second = run_experiment_campaign_planner(
                    memory_data=memory_fixture(),
                    runtime_memory_path=memory_path,
                    transcript_file=transcript,
                    evaluator_ledger_file=evaluator_path,
                    contract_ledger_file=contract_path,
                    adjudicator_ledger_file=adjudicator_path,
                    agenda_ledger_file=agenda_path,
                    lifecycle_ledger_file=lifecycle_path,
                    scorecard_ledger_file=scorecard_path,
                    campaign_ledger_file=campaign_path,
                    outbox_file=outbox,
                    git_status_text='',
                    git_ignored_text='',
                )
            after = memory_path.read_text(encoding='utf-8')

        self.assertEqual('emit_acceptance_bundle', first['selected_action'])
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
            root / 'agent' / 'campaign_planner.py',
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
