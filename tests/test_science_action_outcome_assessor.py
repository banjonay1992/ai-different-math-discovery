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
from agent.science_action_outcome_assessor import (  # noqa: E402
    SCIENCE_ACTION_OUTCOME_LEDGER_KIND,
    build_science_action_outcome_assessment,
    empty_science_action_outcome_ledger,
    load_science_action_outcome_ledger,
    read_science_action_outcome_transcript,
    validate_science_action_outcome_ledger,
    write_science_action_outcome_ledger,
    write_science_action_outcome_outbox_jsonl,
)
from agent.science_campaign_action_planner import SCIENCE_ACTION_LEDGER_KIND  # noqa: E402
from main import run_science_action_outcome_assessor  # noqa: E402


def memory_fixture():
    return {
        'discovery_readiness': {'readiness_score': 0.91, 'status': 'nearly_ready'},
        'abstraction_discovery_evidence': {'transfer_outcome_count': 9},
    }


def action_message(
    scenario_id='scenario-one',
    *,
    action_id=None,
    selected_action='request_funfun_math_certificate',
    recipient='funfun',
    required=None,
    leak=False,
    third_party=False,
):
    action_id = action_id or f'action-{scenario_id}'
    required = ['math_proof'] if required is None else list(required)
    body = {
        'response_kind': 'science_campaign_action',
        'action_id': action_id,
        'benefit_id': f'benefit-{scenario_id}',
        'scenario_id': scenario_id,
        'hypothesis_id': f'hypothesis:{scenario_id}',
        'selected_action': selected_action,
        'selected_recipient': recipient,
        'required_evidence': required,
        'sibling_evidence_used': [],
        'theory_update_intent': f'{selected_action} intent',
        'refinement_state': 'none',
        'retirement_state': 'none',
        'boundary_checkpoint_state': 'repair' if selected_action == 'preserve_boundary_or_checkpoint_repair' else 'clean',
        'boundary_notes': ['repair boundary'] if selected_action == 'preserve_boundary_or_checkpoint_repair' else [],
        'third_party_checkpoint_used': third_party,
    }
    if leak:
        body['boundary_notes'] = ['gravity label leaked']
    return build_module_chat_message(
        sender='ai_different',
        recipient=recipient,
        topic='ai_different.science_campaign_action',
        body=body,
        evidence={'action_id': action_id, 'scenario_id': scenario_id, 'selected_action': selected_action},
        tags=['science_action'],
    )


def sibling_evidence(
    scenario_id='scenario-one',
    *,
    sender='funfun',
    gate='math_proof',
    status='satisfied',
):
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
        tags=['science_action_outcome', gate, status],
    )


def action_ledger_fixture(selected_action='request_funfun_math_certificate', *, scenario_id='scenario-one'):
    return {
        'schema_version': 1,
        'ledger_kind': SCIENCE_ACTION_LEDGER_KIND,
        'processed_message_ids': [],
        'processed_source_hashes': [],
        'planned_action_keys': [],
        'scenario_rows': [],
        'action_records': [{
            'action_id': f'action-{scenario_id}',
            'benefit_ids': [f'benefit-{scenario_id}'],
            'scenario_ids': [scenario_id],
            'hypothesis_ids': [f'hypothesis:{scenario_id}'],
            'selected_action': selected_action,
            'sibling_evidence_used': [],
            'theory_update_intent': f'{selected_action} intent',
            'refinement_state': 'none',
            'retirement_state': 'none',
            'boundary_checkpoint_state': 'clean',
        }],
        'outgoing_response_ids': [],
        'latest': {},
        'ledger_hash': f'action-{scenario_id}-hash',
    }


def empty_ledger(kind, **extra):
    ledger = {'schema_version': 1, 'ledger_kind': kind, 'latest': {}, 'ledger_hash': f'{kind}-hash'}
    ledger.update(extra)
    return ledger


def build_once(messages, *, ledger=None, action_ledger=None, project_boundary=None):
    updated, message = build_science_action_outcome_assessment(
        transcript_messages=messages,
        action_outcome_ledger=ledger or empty_science_action_outcome_ledger(),
        action_ledger=action_ledger or {},
        benefit_ledger={},
        evaluator_ledger={},
        outcome_ledger={},
        contract_ledger={},
        adjudicator_ledger={},
        agenda_ledger={},
        lifecycle_ledger={},
        scorecard_ledger={},
        campaign_ledger={},
        campaign_outcome_ledger={},
        prior_action_outcome_ledger={},
        runtime_memory_data=memory_fixture(),
        runtime_memory_hash_state={'unchanged': True, 'before': 'same', 'after': 'same'},
        project_owned_boundary=project_boundary or {'third_party_checkpoint_used': False},
    )
    return updated, message


class ScienceActionOutcomeAssessorTests(unittest.TestCase):
    def test_outcome_ledger_persistence_load_and_malformed_rejection(self):
        ledger, message = build_once([
            action_message(),
            sibling_evidence(),
        ])
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = Path(tmpdir) / 'outcome.json'
            outbox = Path(tmpdir) / 'outbox.jsonl'
            write_science_action_outcome_ledger(ledger_path, ledger)
            loaded = load_science_action_outcome_ledger(ledger_path)
            write_science_action_outcome_outbox_jsonl(outbox, message)
            rows = [json.loads(line) for line in outbox.read_text(encoding='utf-8').splitlines()]

        self.assertEqual(SCIENCE_ACTION_OUTCOME_LEDGER_KIND, loaded['ledger_kind'])
        self.assertEqual('science_campaign_action_outcome', rows[0]['body']['response_kind'])
        with self.assertRaisesRegex(ValueError, 'wrong ledger_kind'):
            validate_science_action_outcome_ledger({'ledger_kind': 'wrong'})

    def test_action_to_outcome_extraction_and_invalid_record(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            transcript = Path(tmpdir) / 'outcome.jsonl'
            transcript.write_text(
                json.dumps(action_message(), sort_keys=True) + '\n'
                + json.dumps(sibling_evidence(), sort_keys=True) + '\n'
                + '{"sender":"bad"}\n',
                encoding='utf-8',
            )
            parsed = read_science_action_outcome_transcript(transcript)
        ledger, _ = build_once(parsed['messages'])

        self.assertEqual(2, len(parsed['messages']))
        self.assertEqual(1, len(parsed['invalid_messages']))
        row = ledger['action_rows'][0]
        self.assertEqual('scenario-one', row['scenario_id'])
        self.assertEqual('math_certificate_supports_or_blocks_claim', ledger['latest']['selected_outcome'])

    def test_deterministic_priority_selection(self):
        cases = [
            (
                [action_message('repair', selected_action='preserve_boundary_or_checkpoint_repair', recipient='code_module')],
                'preserve_boundary_or_checkpoint_repair',
                'code_module',
            ),
            (
                [
                    action_message('math', selected_action='request_funfun_math_certificate', recipient='funfun'),
                    sibling_evidence('math', sender='funfun', gate='math_proof'),
                ],
                'math_certificate_supports_or_blocks_claim',
                'broadcast',
            ),
            (
                [
                    action_message('code', selected_action='request_code_experiment_or_counterexample', recipient='code_module', required=['code_proof']),
                    sibling_evidence('code', sender='code_module', gate='code_proof', status='failed'),
                ],
                'code_experiment_or_counterexample_changes_decision',
                'broadcast',
            ),
            (
                [
                    action_message('language', selected_action='request_language_protocol_clarification', recipient='language_model_2', required=['language_epoch_plan']),
                    sibling_evidence('language', sender='language_model_2', gate='language_epoch_plan'),
                ],
                'language_clarification_repairs_protocol',
                'broadcast',
            ),
            (
                [
                    action_message('refine', selected_action='refine_hypothesis_with_connected_evidence', recipient='broadcast', required=[]),
                    sibling_evidence('refine', sender='funfun', gate='math_proof'),
                ],
                'refinement_updates_hypothesis_safely',
                'broadcast',
            ),
            (
                [action_message('retire', selected_action='retire_failed_hypothesis_line', recipient='broadcast', required=[])],
                'retirement_closes_failed_line',
                'broadcast',
            ),
            (
                [action_message('waiting', selected_action='schedule_next_campaign_check', recipient='orchestrator', required=[])],
                'scheduled_action_waiting_for_evidence',
                'orchestrator',
            ),
            (
                [action_message('nogain', selected_action='record_connected_no_safe_benefit', recipient='orchestrator', required=[])],
                'no_measurable_campaign_gain',
                'orchestrator',
            ),
        ]
        for rows, outcome, recipient in cases:
            with self.subTest(outcome=outcome):
                ledger, message = build_once(rows)
                self.assertEqual(outcome, ledger['latest']['selected_outcome'])
                self.assertEqual(recipient, message['recipient'])

    def test_boundary_checkpoint_repair_and_label_guard(self):
        leak_ledger, leak_message = build_once([
            action_message('repair-leak', leak=True),
        ])
        self.assertEqual('preserve_boundary_or_checkpoint_repair', leak_ledger['latest']['selected_outcome'])
        self.assertEqual('code_module', leak_message['recipient'])
        self.assertIn('gravity', leak_message['body']['label_leaks'])

        boundary_ledger, boundary_message = build_once(
            [action_message()],
            project_boundary={'third_party_checkpoint_used': True},
        )
        self.assertEqual('preserve_boundary_or_checkpoint_repair', boundary_ledger['latest']['selected_outcome'])
        self.assertEqual('code_module', boundary_message['recipient'])

    def test_duplicate_idempotence_and_appended_evidence_update(self):
        first, first_message = build_once([
            action_message('waiting', selected_action='request_funfun_math_certificate', recipient='funfun'),
        ])
        repeat, repeat_message = build_once([
            action_message('waiting', selected_action='request_funfun_math_certificate', recipient='funfun'),
        ], ledger=first)
        appended, appended_message = build_once([
            sibling_evidence('waiting', sender='funfun', gate='math_proof'),
        ], ledger=repeat)

        self.assertEqual('scheduled_action_waiting_for_evidence', first['latest']['selected_outcome'])
        self.assertIsNotNone(first_message)
        self.assertEqual('summarize_noop', repeat['latest']['selected_outcome'])
        self.assertIsNone(repeat_message)
        self.assertEqual('math_certificate_supports_or_blocks_claim', appended['latest']['selected_outcome'])
        self.assertEqual('broadcast', appended_message['recipient'])

    def test_cli_status_proof_preserves_runtime_memory(self):
        rows = [
            action_message(),
            sibling_evidence(),
        ]
        empty_action = empty_ledger(
            'ai_different.science_campaign_action_ledger',
            processed_message_ids=[],
            processed_source_hashes=[],
            planned_action_keys=[],
            action_records=[],
            scenario_rows=[],
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
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            memory_path = tmp / 'runtime-memory.json'
            transcript = tmp / 'family.jsonl'
            action_path = tmp / 'action.json'
            benefit_path = tmp / 'benefit.json'
            adjudicator_path = tmp / 'adjudicator.json'
            agenda_path = tmp / 'agenda.json'
            lifecycle_path = tmp / 'lifecycle.json'
            scorecard_path = tmp / 'scorecard.json'
            campaign_path = tmp / 'campaign.json'
            campaign_outcome_path = tmp / 'campaign-outcome.json'
            outcome_path = tmp / 'action-outcome.json'
            outbox = tmp / 'outbox.jsonl'
            memory_path.write_text(json.dumps(memory_fixture()), encoding='utf-8')
            transcript.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows), encoding='utf-8')
            action_path.write_text(json.dumps(empty_action), encoding='utf-8')
            benefit_path.write_text(json.dumps(empty_benefit), encoding='utf-8')
            adjudicator_path.write_text(json.dumps(empty_adjudicator), encoding='utf-8')
            agenda_path.write_text(json.dumps(empty_agenda), encoding='utf-8')
            lifecycle_path.write_text(json.dumps(empty_lifecycle), encoding='utf-8')
            scorecard_path.write_text(json.dumps(empty_scorecard), encoding='utf-8')
            campaign_path.write_text(json.dumps(empty_campaign), encoding='utf-8')
            campaign_outcome_path.write_text(json.dumps(empty_campaign_outcome), encoding='utf-8')
            before = memory_path.read_text(encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()):
                first = run_science_action_outcome_assessor(
                    memory_data=memory_fixture(),
                    runtime_memory_path=memory_path,
                    transcript_file=transcript,
                    action_ledger_file=action_path,
                    benefit_ledger_file=benefit_path,
                    evaluator_ledger_file=None,
                    contract_ledger_file=None,
                    adjudicator_ledger_file=adjudicator_path,
                    agenda_ledger_file=agenda_path,
                    lifecycle_ledger_file=lifecycle_path,
                    scorecard_ledger_file=scorecard_path,
                    campaign_ledger_file=campaign_path,
                    campaign_outcome_ledger_file=campaign_outcome_path,
                    action_outcome_ledger_file=outcome_path,
                    outbox_file=outbox,
                    git_status_text='',
                    git_ignored_text='',
                )
            with contextlib.redirect_stdout(io.StringIO()):
                second = run_science_action_outcome_assessor(
                    memory_data=memory_fixture(),
                    runtime_memory_path=memory_path,
                    transcript_file=transcript,
                    action_ledger_file=action_path,
                    benefit_ledger_file=benefit_path,
                    evaluator_ledger_file=None,
                    contract_ledger_file=None,
                    adjudicator_ledger_file=adjudicator_path,
                    agenda_ledger_file=agenda_path,
                    lifecycle_ledger_file=lifecycle_path,
                    scorecard_ledger_file=scorecard_path,
                    campaign_ledger_file=campaign_path,
                    campaign_outcome_ledger_file=campaign_outcome_path,
                    action_outcome_ledger_file=outcome_path,
                    outbox_file=outbox,
                    git_status_text='',
                    git_ignored_text='',
                )
            after = memory_path.read_text(encoding='utf-8')

        self.assertEqual('math_certificate_supports_or_blocks_claim', first['selected_outcome'])
        self.assertEqual(1, first['outbox_count'])
        self.assertEqual('summarize_noop', second['selected_outcome'])
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
            root / 'agent' / 'science_action_outcome_assessor.py',
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
