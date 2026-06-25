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

from agent.module_chat_adapter import build_module_chat_message  # noqa: E402
from agent.science_benefit_evaluator import SCIENCE_BENEFIT_LEDGER_KIND  # noqa: E402
from agent.science_campaign_action_planner import (  # noqa: E402
    SCIENCE_ACTION_LEDGER_KIND,
    build_science_campaign_action_plan,
    empty_science_action_ledger,
    load_science_action_ledger,
    read_science_action_transcript,
    validate_science_action_ledger,
    write_science_action_ledger,
    write_science_action_outbox_jsonl,
)
from main import run_science_campaign_action_planner  # noqa: E402


def memory_fixture():
    return {
        'discovery_readiness': {'readiness_score': 0.9, 'status': 'nearly_ready'},
        'abstraction_discovery_evidence': {'transfer_outcome_count': 8},
    }


def benefit_message(
    scenario_id='scenario-one',
    *,
    classification='connected_requests_missing_targeted_evidence',
    missing=None,
    sibling=None,
    leak=False,
    third_party=False,
):
    missing = ['math_proof'] if missing is None else list(missing)
    sibling = [] if sibling is None else list(sibling)
    body = {
        'response_kind': 'science_campaign_benefit',
        'benefit_id': f'benefit-{scenario_id}',
        'scenario_id': scenario_id,
        'hypothesis_id': f'hypothesis:{scenario_id}',
        'benefit_classification': classification,
        'selected_action': classification,
        'missing_evidence': missing,
        'sibling_evidence_used': sibling,
        'theory_update_delta': 'isolated missing -> connected targeted',
        'refinement_delta': 'connected_added' if classification == 'connected_refines_with_clearer_evidence' else 'none',
        'retirement_delta': 'connected_added' if classification == 'connected_retires_failed_line' else 'none',
        'boundary_checkpoint_state': 'repair' if classification == 'connected_prevents_boundary_or_checkpoint_overclaim' else 'clean',
        'boundary_notes': ['repair boundary'] if classification == 'connected_prevents_boundary_or_checkpoint_overclaim' else [],
        'third_party_checkpoint_used': third_party,
    }
    if leak:
        body['boundary_notes'] = ['gravity label leaked']
    return build_module_chat_message(
        sender='ai_different',
        recipient='orchestrator',
        topic='ai_different.science_campaign_benefit',
        body=body,
        evidence={'scenario_id': scenario_id, 'benefit_classification': classification},
        tags=['science_benefit'],
    )


def sibling_evidence(scenario_id='scenario-one', *, sender='funfun', gate='math_proof', status='satisfied'):
    return build_module_chat_message(
        sender=sender,
        recipient='ai_different',
        topic=f'{sender}.{gate}.{status}',
        body={
            'scenario_id': scenario_id,
            'hypothesis_id': f'hypothesis:{scenario_id}',
            'evidence_gate': gate,
            'status': status,
            'summary': f'{gate} {status}',
        },
        evidence={'scenario_id': scenario_id, 'evidence_gate': gate, 'status': status},
        tags=['science_action', gate, status],
    )


def benefit_ledger_fixture(classification='connected_requests_missing_targeted_evidence', *, scenario_id='scenario-one', missing=None):
    return {
        'schema_version': 1,
        'ledger_kind': SCIENCE_BENEFIT_LEDGER_KIND,
        'processed_message_ids': [],
        'processed_source_hashes': [],
        'evaluated_scenario_ids': [],
        'scenario_records': [],
        'benefit_records': [{
            'benefit_id': f'benefit-{scenario_id}',
            'selected_scenario_id': scenario_id,
            'scenario_ids': [scenario_id],
            'hypothesis_ids': [f'hypothesis:{scenario_id}'],
            'benefit_classification': classification,
            'missing_evidence': list(missing or []),
            'sibling_evidence_used': [],
            'theory_update_delta': 'benefit delta',
            'refinement_delta': 'connected_added' if classification == 'connected_refines_with_clearer_evidence' else 'none',
            'retirement_delta': 'connected_added' if classification == 'connected_retires_failed_line' else 'none',
            'boundary_state': 'repair' if classification == 'connected_prevents_boundary_or_checkpoint_overclaim' else 'clean',
            'boundary_notes': [],
        }],
        'outgoing_response_ids': [],
        'latest': {},
        'ledger_hash': f'benefit-{scenario_id}-hash',
    }


def empty_ledger(kind, **extra):
    ledger = {'schema_version': 1, 'ledger_kind': kind, 'latest': {}, 'ledger_hash': f'{kind}-hash'}
    ledger.update(extra)
    return ledger


def build_once(messages, *, ledger=None, benefit_ledger=None, project_boundary=None):
    updated, message = build_science_campaign_action_plan(
        transcript_messages=messages,
        action_ledger=ledger or empty_science_action_ledger(),
        benefit_ledger=benefit_ledger or {},
        evaluator_ledger={},
        outcome_ledger={},
        contract_ledger={},
        adjudicator_ledger={},
        agenda_ledger={},
        lifecycle_ledger={},
        scorecard_ledger={},
        campaign_ledger={},
        campaign_outcome_ledger={},
        prior_action_ledger={},
        runtime_memory_data=memory_fixture(),
        runtime_memory_hash_state={'unchanged': True, 'before': 'same', 'after': 'same'},
        project_owned_boundary=project_boundary or {'third_party_checkpoint_used': False},
    )
    return updated, message


class ScienceCampaignActionPlannerTests(unittest.TestCase):
    def test_action_ledger_persistence_load_and_malformed_rejection(self):
        ledger, message = build_once([benefit_message()])
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = Path(tmpdir) / 'action.json'
            outbox = Path(tmpdir) / 'outbox.jsonl'
            write_science_action_ledger(ledger_path, ledger)
            loaded = load_science_action_ledger(ledger_path)
            write_science_action_outbox_jsonl(outbox, message)
            rows = [json.loads(line) for line in outbox.read_text(encoding='utf-8').splitlines()]

        self.assertEqual(SCIENCE_ACTION_LEDGER_KIND, loaded['ledger_kind'])
        self.assertEqual('science_campaign_action', rows[0]['body']['response_kind'])
        with self.assertRaisesRegex(ValueError, 'wrong ledger_kind'):
            validate_science_action_ledger({'ledger_kind': 'wrong'})

    def test_benefit_to_action_extraction_and_invalid_record(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            transcript = Path(tmpdir) / 'action.jsonl'
            transcript.write_text(
                json.dumps(benefit_message(missing=['code_proof']), sort_keys=True) + '\n'
                + json.dumps(sibling_evidence(sender='funfun'), sort_keys=True) + '\n'
                + '{"sender":"bad"}\n',
                encoding='utf-8',
            )
            parsed = read_science_action_transcript(transcript)
        ledger, _ = build_once(parsed['messages'])

        self.assertEqual(2, len(parsed['messages']))
        self.assertEqual(1, len(parsed['invalid_messages']))
        row = ledger['scenario_rows'][0]
        self.assertEqual('scenario-one', row['scenario_id'])
        self.assertEqual('request_code_experiment_or_counterexample', ledger['latest']['selected_action'])

    def test_deterministic_priority_and_action_routing(self):
        cases = [
            (
                benefit_message('repair', classification='connected_prevents_boundary_or_checkpoint_overclaim'),
                'preserve_boundary_or_checkpoint_repair',
                'code_module',
            ),
            (
                benefit_message('retire', classification='connected_retires_failed_line', missing=[]),
                'retire_failed_hypothesis_line',
                'broadcast',
            ),
            (
                benefit_message('math', missing=['math_proof']),
                'request_funfun_math_certificate',
                'funfun',
            ),
            (
                benefit_message('code', missing=['code_proof']),
                'request_code_experiment_or_counterexample',
                'code_module',
            ),
            (
                benefit_message('language', missing=['language_epoch_plan']),
                'request_language_protocol_clarification',
                'language_model_2',
            ),
            (
                benefit_message('refine', classification='connected_refines_with_clearer_evidence', missing=[]),
                'refine_hypothesis_with_connected_evidence',
                'broadcast',
            ),
            (
                benefit_message('next', classification='connected_accepts_with_verified_math_code_language', missing=[]),
                'schedule_next_campaign_check',
                'orchestrator',
            ),
            (
                benefit_message('nobenefit', classification='connected_adds_no_safe_benefit', missing=[]),
                'record_connected_no_safe_benefit',
                'orchestrator',
            ),
        ]
        for message, action, recipient in cases:
            with self.subTest(action=action):
                ledger, out = build_once([message])
                self.assertEqual(action, ledger['latest']['selected_action'])
                self.assertEqual(recipient, out['recipient'])

    def test_boundary_checkpoint_repair_and_label_guard(self):
        leak_ledger, leak_message = build_once([
            benefit_message(classification='connected_prevents_boundary_or_checkpoint_overclaim', leak=True),
        ])
        self.assertEqual('preserve_boundary_or_checkpoint_repair', leak_ledger['latest']['selected_action'])
        self.assertEqual('code_module', leak_message['recipient'])
        self.assertIn('gravity', leak_message['body']['label_leaks'])

        boundary_ledger, boundary_message = build_once(
            [benefit_message()],
            project_boundary={'third_party_checkpoint_used': True},
        )
        self.assertEqual('preserve_boundary_or_checkpoint_repair', boundary_ledger['latest']['selected_action'])
        self.assertEqual('code_module', boundary_message['recipient'])

    def test_duplicate_idempotence_and_appended_evidence_update(self):
        first, first_message = build_once([benefit_message(missing=['math_proof'])])
        repeat, repeat_message = build_once([benefit_message(missing=['math_proof'])], ledger=first)
        appended, appended_message = build_once([
            benefit_message('scenario-two', missing=['code_proof']),
        ], ledger=repeat)

        self.assertIsNotNone(first_message)
        self.assertIsNone(repeat_message)
        self.assertEqual('summarize_noop', repeat['latest']['selected_action'])
        self.assertEqual('request_code_experiment_or_counterexample', appended['latest']['selected_action'])
        self.assertEqual('code_module', appended_message['recipient'])

    def test_cli_status_proof_preserves_runtime_memory(self):
        rows = [benefit_message(classification='connected_accepts_with_verified_math_code_language', missing=[])]
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
        empty_campaign = empty_ledger(
            'ai_different.experiment_campaign_ledger',
            processed_message_ids=[],
            processed_source_hashes=[],
            retired_hypothesis_ids=[],
            continued_refinement_ids=[],
            acceptance_bundle_ids=[],
            campaigns=[],
            campaign_records=[],
            outgoing_response_ids=[],
        )
        empty_campaign_outcome = empty_ledger(
            'ai_different.experiment_campaign_outcome_ledger',
            processed_message_ids=[],
            processed_source_hashes=[],
            accepted_campaign_ids=[],
            refined_hypothesis_ids=[],
            retired_hypothesis_ids=[],
            repaired_campaign_ids=[],
            outcomes=[],
            outcome_records=[],
            outgoing_response_ids=[],
        )
        empty_benefit = empty_ledger(
            'ai_different.science_campaign_benefit_ledger',
            processed_message_ids=[],
            processed_source_hashes=[],
            evaluated_scenario_ids=[],
            benefit_records=[],
            scenario_records=[],
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
            campaign_outcome_path = tmp / 'campaign-outcome.json'
            benefit_path = tmp / 'benefit.json'
            action_path = tmp / 'action.json'
            outbox = tmp / 'outbox.jsonl'
            memory_path.write_text(json.dumps(memory_fixture()), encoding='utf-8')
            transcript.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows), encoding='utf-8')
            adjudicator_path.write_text(json.dumps(empty_adjudicator), encoding='utf-8')
            agenda_path.write_text(json.dumps(empty_agenda), encoding='utf-8')
            lifecycle_path.write_text(json.dumps(empty_lifecycle), encoding='utf-8')
            scorecard_path.write_text(json.dumps(empty_scorecard), encoding='utf-8')
            campaign_path.write_text(json.dumps(empty_campaign), encoding='utf-8')
            campaign_outcome_path.write_text(json.dumps(empty_campaign_outcome), encoding='utf-8')
            benefit_path.write_text(json.dumps(empty_benefit), encoding='utf-8')
            before = memory_path.read_text(encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()):
                first = run_science_campaign_action_planner(
                    memory_data=memory_fixture(),
                    runtime_memory_path=memory_path,
                    transcript_file=transcript,
                    benefit_ledger_file=benefit_path,
                    evaluator_ledger_file=None,
                    contract_ledger_file=None,
                    adjudicator_ledger_file=adjudicator_path,
                    agenda_ledger_file=agenda_path,
                    lifecycle_ledger_file=lifecycle_path,
                    scorecard_ledger_file=scorecard_path,
                    campaign_ledger_file=campaign_path,
                    campaign_outcome_ledger_file=campaign_outcome_path,
                    action_ledger_file=action_path,
                    outbox_file=outbox,
                    git_status_text='',
                    git_ignored_text='',
                )
            with contextlib.redirect_stdout(io.StringIO()):
                second = run_science_campaign_action_planner(
                    memory_data=memory_fixture(),
                    runtime_memory_path=memory_path,
                    transcript_file=transcript,
                    benefit_ledger_file=benefit_path,
                    evaluator_ledger_file=None,
                    contract_ledger_file=None,
                    adjudicator_ledger_file=adjudicator_path,
                    agenda_ledger_file=agenda_path,
                    lifecycle_ledger_file=lifecycle_path,
                    scorecard_ledger_file=scorecard_path,
                    campaign_ledger_file=campaign_path,
                    campaign_outcome_ledger_file=campaign_outcome_path,
                    action_ledger_file=action_path,
                    outbox_file=outbox,
                    git_status_text='',
                    git_ignored_text='',
                )
            after = memory_path.read_text(encoding='utf-8')

        self.assertEqual('schedule_next_campaign_check', first['selected_action'])
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
            root / 'agent' / 'science_campaign_action_planner.py',
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
