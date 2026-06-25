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

from agent.campaign_outcome_assessor import (  # noqa: E402
    CAMPAIGN_OUTCOME_LEDGER_KIND,
    build_campaign_outcome_assessment,
    empty_campaign_outcome_ledger,
    load_campaign_outcome_ledger,
    read_campaign_outcome_transcript,
    validate_campaign_outcome_ledger,
    write_campaign_outcome_ledger,
    write_campaign_outcome_outbox_jsonl,
)
from agent.campaign_planner import CAMPAIGN_LEDGER_KIND  # noqa: E402
from agent.module_chat_adapter import build_module_chat_message  # noqa: E402
from main import run_campaign_outcome_assessor  # noqa: E402


def memory_fixture():
    return {
        'discovery_readiness': {'readiness_score': 0.88, 'status': 'nearly_ready'},
        'abstraction_discovery_evidence': {'transfer_outcome_count': 6},
    }


def campaign_message(
    *,
    campaign_id='campaign-one',
    hypothesis_id='hypothesis:contract-one',
    accepted=None,
    missing=None,
    rejected=None,
    campaign_type='emit_acceptance_bundle',
):
    accepted = list(accepted or [])
    missing = list(missing or ['math_proof', 'code_proof', 'language_epoch_plan'])
    rejected = list(rejected or [])
    return build_module_chat_message(
        sender='ai_different',
        recipient='broadcast',
        topic='ai_different.experiment_campaign',
        body={
            'response_kind': 'experiment_campaign',
            'campaign_id': campaign_id,
            'hypothesis_id': hypothesis_id,
            'campaign_type': campaign_type,
            'selected_action': campaign_type,
            'acceptance_criteria': [
                {'evidence_gate': 'math_proof', 'required_status': 'accepted'},
                {'evidence_gate': 'code_proof', 'required_status': 'accepted'},
                {'evidence_gate': 'language_epoch_plan', 'required_status': 'accepted'},
            ],
            'required_evidence': ['math_proof', 'code_proof', 'language_epoch_plan'],
            'accepted_evidence': accepted,
            'missing_evidence': missing,
            'rejected_evidence': rejected,
        },
        evidence={'campaign_id': campaign_id, 'hypothesis_id': hypothesis_id},
        tags=['campaign'],
    )


def evidence_message(campaign_id='campaign-one', *, sender, gate=None, status='satisfied', leak=False, third_party=False):
    if gate is None:
        gate = 'math_proof' if sender == 'funfun' else 'code_proof'
    summary = f'{gate} {status} outcome evidence'
    if leak:
        summary = 'gravity label leaked into campaign outcome evidence'
    return build_module_chat_message(
        sender=sender,
        recipient='ai_different',
        topic=f'{sender}.{gate}.{status}',
        body={
            'evidence_id': f'{campaign_id}-{sender}-{gate}-{status}',
            'campaign_id': campaign_id,
            'hypothesis_id': 'hypothesis:contract-one',
            'evidence_gate': gate,
            'status': status,
            'summary': summary,
            'third_party_checkpoint_used': third_party,
        },
        evidence={'campaign_id': campaign_id, 'evidence_gate': gate, 'status': status},
        tags=['campaign_outcome', gate, status],
    )


def language_message(campaign_id='campaign-one', *, status='satisfied'):
    return evidence_message(
        campaign_id,
        sender='language_model_2',
        gate='language_epoch_plan',
        status=status,
    )


def campaign_ledger_fixture(*, accepted=None, missing=None, rejected=None):
    accepted = list(accepted or [])
    missing = list(missing or ['math_proof', 'code_proof', 'language_epoch_plan'])
    rejected = list(rejected or [])
    return {
        'schema_version': 1,
        'ledger_kind': CAMPAIGN_LEDGER_KIND,
        'processed_message_ids': [],
        'processed_source_hashes': [],
        'retired_hypothesis_ids': [],
        'continued_refinement_ids': [],
        'acceptance_bundle_ids': [],
        'campaigns': [{
            'campaign_id': 'campaign-one',
            'hypothesis_id': 'hypothesis:contract-one',
            'contract_id': 'contract-one',
            'required_evidence': ['math_proof', 'code_proof', 'language_epoch_plan'],
            'accepted_evidence': accepted,
            'missing_evidence': missing,
            'rejected_evidence': rejected,
            'acceptance_criteria': [
                {'evidence_gate': 'math_proof', 'required_status': 'accepted'},
                {'evidence_gate': 'code_proof', 'required_status': 'accepted'},
                {'evidence_gate': 'language_epoch_plan', 'required_status': 'accepted'},
            ],
            'campaign_type': 'emit_acceptance_bundle',
        }],
        'campaign_records': [],
        'outgoing_response_ids': [],
        'latest': {},
        'ledger_hash': 'campaign-ledger-hash-one',
    }


def empty_ledger(kind, **extra):
    ledger = {'schema_version': 1, 'ledger_kind': kind, 'latest': {}, 'ledger_hash': f'{kind}-hash'}
    ledger.update(extra)
    return ledger


def build_once(messages, *, ledger=None, campaign_ledger=None, project_boundary=None):
    updated, message = build_campaign_outcome_assessment(
        transcript_messages=messages,
        campaign_outcome_ledger=ledger or empty_campaign_outcome_ledger(),
        evaluator_ledger={},
        outcome_ledger={},
        contract_ledger={},
        adjudicator_ledger={},
        agenda_ledger={},
        lifecycle_ledger={},
        scorecard_ledger={},
        campaign_ledger=campaign_ledger or campaign_ledger_fixture(),
        runtime_memory_data=memory_fixture(),
        runtime_memory_hash_state={'unchanged': True, 'before': 'same', 'after': 'same'},
        project_owned_boundary=project_boundary or {'third_party_checkpoint_used': False},
    )
    return updated, message


class CampaignOutcomeAssessorTests(unittest.TestCase):
    def test_outcome_ledger_persistence_load_and_malformed_rejection(self):
        ledger, message = build_once([campaign_message()])
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = Path(tmpdir) / 'outcome.json'
            outbox = Path(tmpdir) / 'outbox.jsonl'
            write_campaign_outcome_ledger(ledger_path, ledger)
            loaded = load_campaign_outcome_ledger(ledger_path)
            write_campaign_outcome_outbox_jsonl(outbox, message)
            rows = [json.loads(line) for line in outbox.read_text(encoding='utf-8').splitlines()]

        self.assertEqual(CAMPAIGN_OUTCOME_LEDGER_KIND, loaded['ledger_kind'])
        self.assertEqual('experiment_campaign_outcome', rows[0]['body']['response_kind'])
        with self.assertRaisesRegex(ValueError, 'wrong ledger_kind'):
            validate_campaign_outcome_ledger({'ledger_kind': 'wrong'})

    def test_campaign_and_evidence_extraction_with_invalid_record(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            transcript = Path(tmpdir) / 'outcome.jsonl'
            transcript.write_text(
                json.dumps(campaign_message(), sort_keys=True) + '\n'
                + json.dumps(evidence_message(sender='funfun'), sort_keys=True) + '\n'
                + '{"sender":"bad"}\n',
                encoding='utf-8',
            )
            parsed = read_campaign_outcome_transcript(transcript)
        ledger, _ = build_once(parsed['messages'])

        self.assertEqual(2, len(parsed['messages']))
        self.assertEqual(1, len(parsed['invalid_messages']))
        outcome = ledger['outcomes'][0]
        self.assertEqual('campaign-one', outcome['campaign_id'])
        self.assertIn('math_proof', outcome['accepted_evidence'])
        self.assertEqual('request_more_code', ledger['latest']['selected_action'])

    def test_priority_routes_math_code_language_then_acceptance(self):
        first, first_message = build_once([campaign_message()])
        self.assertEqual('request_more_math', first['latest']['selected_action'])
        self.assertEqual('funfun', first_message['recipient'])

        second, second_message = build_once([evidence_message(sender='funfun')], ledger=first)
        self.assertEqual('request_more_code', second['latest']['selected_action'])
        self.assertEqual('code_module', second_message['recipient'])

        third, third_message = build_once([evidence_message(sender='code_module')], ledger=second)
        self.assertEqual('request_more_language', third['latest']['selected_action'])
        self.assertEqual('language_model_2', third_message['recipient'])

        fourth, fourth_message = build_once([language_message()], ledger=third)
        self.assertEqual('accept_campaign', fourth['latest']['selected_action'])
        self.assertEqual('broadcast', fourth_message['recipient'])
        self.assertEqual('accept_campaign', fourth_message['body']['theory_update_action'])

    def test_boundary_repair_failed_gate_refine_and_retire(self):
        leak_ledger, leak_message = build_once([
            campaign_message(),
            evidence_message(sender='funfun', leak=True),
        ])
        self.assertEqual('repair_boundary', leak_ledger['latest']['selected_action'])
        self.assertEqual('code_module', leak_message['recipient'])
        self.assertIn('gravity', leak_message['body']['label_leaks'])

        boundary_ledger, boundary_message = build_once(
            [campaign_message()],
            project_boundary={'third_party_checkpoint_used': True},
        )
        self.assertEqual('repair_boundary', boundary_ledger['latest']['selected_action'])
        self.assertEqual('code_module', boundary_message['recipient'])

        refined, refined_message = build_once(
            [evidence_message(sender='code_module', status='failed')],
            campaign_ledger=campaign_ledger_fixture(
                accepted=['math_proof', 'language_epoch_plan'],
                missing=['code_proof'],
            ),
        )
        self.assertEqual('refine_hypothesis', refined['latest']['selected_action'])
        self.assertEqual('broadcast', refined_message['recipient'])

        retired, retired_message = build_once(
            [evidence_message(sender='funfun', status='failed')],
            campaign_ledger=campaign_ledger_fixture(
                accepted=[],
                missing=['math_proof', 'code_proof', 'language_epoch_plan'],
            ),
        )
        self.assertEqual('retire_theory_line', retired['latest']['selected_action'])
        self.assertEqual('broadcast', retired_message['recipient'])

    def test_duplicate_idempotence(self):
        ledger, message = build_once([campaign_message()])
        repeat, repeat_message = build_once([campaign_message()], ledger=ledger)

        self.assertIsNotNone(message)
        self.assertIsNone(repeat_message)
        self.assertEqual('noop', repeat['latest']['selected_action'])

    def test_cli_status_proof_preserves_runtime_memory(self):
        rows = [
            campaign_message(),
            evidence_message(sender='funfun'),
            evidence_message(sender='code_module'),
            language_message(),
        ]
        empty_adjudicator = empty_ledger(
            'ai_different.cross_module_adjudicator_ledger',
            processed_message_ids=[],
            processed_evaluator_ledger_ids=[],
            contract_states=[],
            adjudication_records=[],
            outgoing_response_ids=[],
        )
        empty_agenda = empty_ledger(
            'ai_different.experiment_agenda_ledger',
            processed_message_ids=[],
            processed_source_hashes=[],
            scheduled_candidate_ids=[],
            hypotheses=[],
            agenda_records=[],
            outgoing_response_ids=[],
        )
        empty_lifecycle = empty_ledger(
            'ai_different.hypothesis_lifecycle_ledger',
            processed_message_ids=[],
            processed_source_hashes=[],
            resolved_hypothesis_ids=[],
            retired_hypothesis_ids=[],
            refined_hypothesis_ids=[],
            hypotheses=[],
            lifecycle_records=[],
            outgoing_response_ids=[],
        )
        empty_scorecard = empty_ledger(
            'ai_different.experiment_evidence_scorecard_ledger',
            processed_message_ids=[],
            processed_source_hashes=[],
            resolved_hypothesis_ids=[],
            retired_hypothesis_ids=[],
            refined_hypothesis_ids=[],
            scorecards=[],
            scorecard_records=[],
            outgoing_response_ids=[],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            memory_path = tmp / 'runtime-memory.json'
            transcript = tmp / 'family.jsonl'
            adjudicator_path = tmp / 'adjudicator.json'
            agenda_path = tmp / 'agenda.json'
            lifecycle_path = tmp / 'lifecycle.json'
            scorecard_path = tmp / 'scorecard.json'
            campaign_path = tmp / 'campaign.json'
            outcome_path = tmp / 'campaign-outcome.json'
            outbox = tmp / 'outbox.jsonl'
            memory_path.write_text(json.dumps(memory_fixture()), encoding='utf-8')
            transcript.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows), encoding='utf-8')
            adjudicator_path.write_text(json.dumps(empty_adjudicator), encoding='utf-8')
            agenda_path.write_text(json.dumps(empty_agenda), encoding='utf-8')
            lifecycle_path.write_text(json.dumps(empty_lifecycle), encoding='utf-8')
            scorecard_path.write_text(json.dumps(empty_scorecard), encoding='utf-8')
            campaign_path.write_text(json.dumps(campaign_ledger_fixture()), encoding='utf-8')
            before = memory_path.read_text(encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()):
                first = run_campaign_outcome_assessor(
                    memory_data=memory_fixture(),
                    runtime_memory_path=memory_path,
                    transcript_file=transcript,
                    evaluator_ledger_file=None,
                    contract_ledger_file=None,
                    adjudicator_ledger_file=adjudicator_path,
                    agenda_ledger_file=agenda_path,
                    lifecycle_ledger_file=lifecycle_path,
                    scorecard_ledger_file=scorecard_path,
                    campaign_ledger_file=campaign_path,
                    campaign_outcome_ledger_file=outcome_path,
                    outbox_file=outbox,
                    git_status_text='',
                    git_ignored_text='',
                )
            with contextlib.redirect_stdout(io.StringIO()):
                second = run_campaign_outcome_assessor(
                    memory_data=memory_fixture(),
                    runtime_memory_path=memory_path,
                    transcript_file=transcript,
                    evaluator_ledger_file=None,
                    contract_ledger_file=None,
                    adjudicator_ledger_file=adjudicator_path,
                    agenda_ledger_file=agenda_path,
                    lifecycle_ledger_file=lifecycle_path,
                    scorecard_ledger_file=scorecard_path,
                    campaign_ledger_file=campaign_path,
                    campaign_outcome_ledger_file=outcome_path,
                    outbox_file=outbox,
                    git_status_text='',
                    git_ignored_text='',
                )
            after = memory_path.read_text(encoding='utf-8')

        self.assertEqual('accept_campaign', first['selected_action'])
        self.assertEqual(1, first['outbox_count'])
        self.assertEqual('noop', second['selected_action'])
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
            root / 'agent' / 'campaign_outcome_assessor.py',
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
