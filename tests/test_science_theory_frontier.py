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

from agent.campaign_outcome_assessor import CAMPAIGN_OUTCOME_LEDGER_KIND  # noqa: E402
from agent.module_chat_adapter import build_module_chat_message  # noqa: E402
from agent.science_action_outcome_assessor import SCIENCE_ACTION_OUTCOME_LEDGER_KIND  # noqa: E402
from agent.science_campaign_action_planner import SCIENCE_ACTION_LEDGER_KIND  # noqa: E402
from agent.science_theory_frontier import (  # noqa: E402
    SCIENCE_THEORY_FRONTIER_LEDGER_KIND,
    build_science_theory_frontier_plan,
    empty_science_theory_frontier_ledger,
    load_science_theory_frontier_ledger,
    read_science_theory_frontier_transcript,
    validate_science_theory_frontier_ledger,
    write_science_theory_frontier_ledger,
    write_science_theory_frontier_outbox_jsonl,
)
from main import run_science_theory_frontier_planner  # noqa: E402


def memory_fixture():
    return {
        'discovery_readiness': {'readiness_score': 0.92, 'status': 'nearly_ready'},
        'abstraction_discovery_evidence': {'transfer_outcome_count': 10},
    }


def outcome_message(
    scenario_id='scenario-one',
    *,
    outcome_id=None,
    selected_outcome='math_certificate_supports_or_blocks_claim',
    planned_action='request_funfun_math_certificate',
    status='accepted',
    leak=False,
    third_party=False,
):
    outcome_id = outcome_id or f'outcome-{scenario_id}'
    body = {
        'response_kind': 'science_campaign_action_outcome',
        'outcome_id': outcome_id,
        'action_id': f'action-{scenario_id}',
        'benefit_id': f'benefit-{scenario_id}',
        'scenario_id': scenario_id,
        'hypothesis_id': f'hypothesis:{scenario_id}',
        'selected_outcome': selected_outcome,
        'selected_action': selected_outcome,
        'planned_action': planned_action,
        'observed_sibling_evidence': [],
        'before_hypothesis_state': 'waiting',
        'after_hypothesis_state': 'blocked' if status in {'blocked', 'failed'} else 'evidence_received',
        'theory_update_delta': f'{selected_outcome} delta',
        'refinement_state': 'updated' if selected_outcome == 'refinement_updates_hypothesis_safely' else 'none',
        'retirement_block_state': 'retired' if selected_outcome == 'retirement_closes_failed_line' else 'none',
        'waiting_blocker_state': 'blocked' if status in {'blocked', 'failed'} else 'resolved',
        'boundary_checkpoint_state': 'repair' if selected_outcome == 'preserve_boundary_or_checkpoint_repair' else 'clean',
        'boundary_notes': ['repair boundary'] if selected_outcome == 'preserve_boundary_or_checkpoint_repair' else [],
        'third_party_checkpoint_used': third_party,
    }
    if leak:
        body['boundary_notes'] = ['gravity label leaked']
    return build_module_chat_message(
        sender='ai_different',
        recipient='orchestrator',
        topic='ai_different.science_campaign_action_outcome',
        body=body,
        evidence={'outcome_id': outcome_id, 'scenario_id': scenario_id, 'selected_outcome': selected_outcome},
        tags=['science_action_outcome'],
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
        tags=['science_theory_frontier', gate, status],
    )


def action_outcome_ledger_fixture(selected_outcome='math_certificate_supports_or_blocks_claim', *, scenario_id='scenario-one'):
    return {
        'schema_version': 1,
        'ledger_kind': SCIENCE_ACTION_OUTCOME_LEDGER_KIND,
        'processed_message_ids': [],
        'processed_source_hashes': [],
        'assessed_action_keys': [],
        'action_rows': [],
        'outcome_records': [{
            'outcome_id': f'outcome-{scenario_id}',
            'action_ids': [f'action-{scenario_id}'],
            'benefit_ids': [f'benefit-{scenario_id}'],
            'scenario_ids': [scenario_id],
            'hypothesis_ids': [f'hypothesis:{scenario_id}'],
            'selected_outcome': selected_outcome,
            'planned_action': 'request_funfun_math_certificate',
            'observed_sibling_evidence': [{'sender': 'funfun', 'evidence_gate': 'math_proof', 'status': 'satisfied'}],
            'before_hypothesis_state': 'waiting',
            'after_hypothesis_state': 'evidence_received',
            'theory_update_delta': 'ledger delta',
            'refinement_state': 'none',
            'retirement_state': 'none',
            'waiting_blocker_state': 'resolved',
            'boundary_checkpoint_state': 'clean',
            'boundary_notes': [],
        }],
        'outgoing_response_ids': [],
        'latest': {},
        'ledger_hash': f'action-outcome-{scenario_id}-hash',
    }


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


def campaign_outcome_ledger_fixture(*, scenario_id='scenario-one', selected_outcome='accept_campaign'):
    return {
        'schema_version': 1,
        'ledger_kind': CAMPAIGN_OUTCOME_LEDGER_KIND,
        'processed_message_ids': [],
        'outcomes': [{
            'outcome_id': f'campaign-outcome-{scenario_id}',
            'action_id': f'action-{scenario_id}',
            'benefit_id': f'benefit-{scenario_id}',
            'scenario_id': scenario_id,
            'hypothesis_id': f'hypothesis:{scenario_id}',
            'selected_outcome': selected_outcome,
            'readiness_state': 'resolved',
            'rejected_evidence': [],
        }],
        'outgoing_response_ids': [],
        'latest': {},
        'ledger_hash': f'campaign-outcome-{scenario_id}-hash',
    }


def build_once(messages, *, ledger=None, action_outcome_ledger=None, action_ledger=None, campaign_outcome_ledger=None, project_boundary=None):
    updated, message = build_science_theory_frontier_plan(
        transcript_messages=messages,
        frontier_ledger=ledger or empty_science_theory_frontier_ledger(),
        action_outcome_ledger=action_outcome_ledger or {},
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
        campaign_outcome_ledger=campaign_outcome_ledger or {},
        prior_frontier_ledger={},
        sibling_outcome_ledgers={},
        runtime_memory_data=memory_fixture(),
        runtime_memory_hash_state={'unchanged': True, 'before': 'same', 'after': 'same'},
        project_owned_boundary=project_boundary or {'third_party_checkpoint_used': False},
    )
    return updated, message


class ScienceTheoryFrontierTests(unittest.TestCase):
    def test_frontier_ledger_persistence_load_and_malformed_rejection(self):
        ledger, message = build_once([outcome_message()])
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = Path(tmpdir) / 'frontier.json'
            outbox = Path(tmpdir) / 'outbox.jsonl'
            write_science_theory_frontier_ledger(ledger_path, ledger)
            loaded = load_science_theory_frontier_ledger(ledger_path)
            write_science_theory_frontier_outbox_jsonl(outbox, message)
            rows = [json.loads(line) for line in outbox.read_text(encoding='utf-8').splitlines()]

        self.assertEqual(SCIENCE_THEORY_FRONTIER_LEDGER_KIND, loaded['ledger_kind'])
        self.assertEqual('science_theory_frontier', rows[0]['body']['response_kind'])
        with self.assertRaisesRegex(ValueError, 'wrong ledger_kind'):
            validate_science_theory_frontier_ledger({'ledger_kind': 'wrong'})

    def test_outcome_to_frontier_extraction_and_invalid_record(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            transcript = Path(tmpdir) / 'frontier.jsonl'
            transcript.write_text(
                json.dumps(outcome_message(), sort_keys=True) + '\n'
                + json.dumps(sibling_evidence(), sort_keys=True) + '\n'
                + '{"sender":"bad"}\n',
                encoding='utf-8',
            )
            parsed = read_science_theory_frontier_transcript(transcript)
        ledger, _ = build_once(parsed['messages'])

        self.assertEqual(2, len(parsed['messages']))
        self.assertEqual(1, len(parsed['invalid_messages']))
        row = ledger['frontier_rows'][0]
        self.assertEqual('scenario-one', row['scenario_id'])
        self.assertEqual('promote_supported_hypothesis_to_theory_memory', ledger['latest']['selected_theory_move'])

    def test_deterministic_priority_selection_and_routing(self):
        cases = [
            (
                [outcome_message('repair', selected_outcome='preserve_boundary_or_checkpoint_repair')],
                'preserve_boundary_or_checkpoint_repair',
                'code_module',
            ),
            (
                [outcome_message('promote')],
                'promote_supported_hypothesis_to_theory_memory',
                'broadcast',
            ),
            (
                [outcome_message('block', selected_outcome='code_experiment_or_counterexample_changes_decision', status='blocked')],
                'retire_or_block_refuted_hypothesis',
                'broadcast',
            ),
            (
                [outcome_message('math', selected_outcome='scheduled_action_waiting_for_evidence', planned_action='request_funfun_math_certificate')],
                'request_funfun_certificate',
                'funfun',
            ),
            (
                [outcome_message('code', selected_outcome='scheduled_action_waiting_for_evidence', planned_action='request_code_experiment_or_counterexample')],
                'request_code_experiment_or_counterexample',
                'code_module',
            ),
            (
                [outcome_message('language', selected_outcome='scheduled_action_waiting_for_evidence', planned_action='request_language_protocol_clarification')],
                'request_language_protocol_clarification',
                'language_model_2',
            ),
            (
                [outcome_message('refine', selected_outcome='refinement_updates_hypothesis_safely')],
                'refine_hypothesis_from_outcome',
                'broadcast',
            ),
            (
                [outcome_message('frontier', selected_outcome='scheduled_action_waiting_for_evidence', planned_action='schedule_next_campaign_check')],
                'schedule_next_campaign_frontier_check',
                'orchestrator',
            ),
            (
                [outcome_message('nogain', selected_outcome='no_measurable_campaign_gain')],
                'record_no_measurable_theory_gain',
                'orchestrator',
            ),
        ]
        for messages, move, recipient in cases:
            with self.subTest(move=move):
                ledger, message = build_once(messages)
                self.assertEqual(move, ledger['latest']['selected_theory_move'])
                self.assertEqual(recipient, message['recipient'])

    def test_ledger_sources_and_boundary_guards(self):
        ledger, message = build_once([], action_outcome_ledger=action_outcome_ledger_fixture())
        self.assertEqual('promote_supported_hypothesis_to_theory_memory', ledger['latest']['selected_theory_move'])
        self.assertEqual('broadcast', message['recipient'])

        action_ledger, action_message_obj = build_once([], action_ledger=action_ledger_fixture())
        self.assertEqual('request_funfun_certificate', action_ledger['latest']['selected_theory_move'])
        self.assertEqual('funfun', action_message_obj['recipient'])

        campaign_ledger, campaign_message = build_once([], campaign_outcome_ledger=campaign_outcome_ledger_fixture(selected_outcome='no_measurable_campaign_gain'))
        self.assertEqual('record_no_measurable_theory_gain', campaign_ledger['latest']['selected_theory_move'])
        self.assertEqual('orchestrator', campaign_message['recipient'])

        repair_ledger, repair_message = build_once(
            [outcome_message('leak', leak=True)],
            project_boundary={'third_party_checkpoint_used': False},
        )
        self.assertEqual('preserve_boundary_or_checkpoint_repair', repair_ledger['latest']['selected_theory_move'])
        self.assertEqual('code_module', repair_message['recipient'])
        self.assertIn('gravity', repair_message['body']['label_leaks'])

        third_party_ledger, third_party_message = build_once(
            [outcome_message('third-party')],
            project_boundary={'third_party_checkpoint_used': True},
        )
        self.assertEqual('preserve_boundary_or_checkpoint_repair', third_party_ledger['latest']['selected_theory_move'])
        self.assertFalse(third_party_message['body'].get('project_owned_checkpoint_claimed', False))

    def test_duplicate_idempotence_and_appended_evidence_update(self):
        first_ledger, first_message = build_once([
            outcome_message('waiting', selected_outcome='scheduled_action_waiting_for_evidence', planned_action='request_funfun_math_certificate'),
        ])
        self.assertEqual('request_funfun_certificate', first_ledger['latest']['selected_theory_move'])
        self.assertIsNotNone(first_message)

        repeat_ledger, repeat_message = build_once([
            outcome_message('waiting', selected_outcome='scheduled_action_waiting_for_evidence', planned_action='request_funfun_math_certificate'),
        ], ledger=first_ledger)
        self.assertEqual('summarize_noop', repeat_ledger['latest']['selected_theory_move'])
        self.assertIsNone(repeat_message)

        appended_ledger, appended_message = build_once([
            outcome_message('waiting', selected_outcome='scheduled_action_waiting_for_evidence', planned_action='request_funfun_math_certificate'),
            outcome_message('waiting', outcome_id='outcome-waiting-confirmed', selected_outcome='math_certificate_supports_or_blocks_claim'),
        ], ledger=repeat_ledger)
        self.assertEqual('promote_supported_hypothesis_to_theory_memory', appended_ledger['latest']['selected_theory_move'])
        self.assertIsNotNone(appended_message)

    def test_cli_status_proof_preserves_runtime_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            runtime = tmp / 'theory-memory.json'
            runtime.write_text(json.dumps(memory_fixture(), sort_keys=True), encoding='utf-8')
            transcript = tmp / 'frontier.jsonl'
            transcript.write_text(json.dumps(outcome_message(), sort_keys=True) + '\n', encoding='utf-8')
            ledger_path = tmp / 'frontier-ledger.json'
            outbox = tmp / 'outbox.jsonl'

            before = runtime.read_text(encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()):
                result = run_science_theory_frontier_planner(
                    runtime_memory_path=runtime,
                    transcript_file=transcript,
                    action_outcome_ledger_file=tmp / 'missing-action-outcome.json',
                    action_ledger_file=tmp / 'missing-action.json',
                    benefit_ledger_file=tmp / 'missing-benefit.json',
                    evaluator_ledger_file=tmp / 'missing-evaluator.json',
                    contract_ledger_file=tmp / 'missing-contract.json',
                    adjudicator_ledger_file=tmp / 'missing-adjudicator.json',
                    agenda_ledger_file=tmp / 'missing-agenda.json',
                    lifecycle_ledger_file=tmp / 'missing-lifecycle.json',
                    scorecard_ledger_file=tmp / 'missing-scorecard.json',
                    campaign_ledger_file=tmp / 'missing-campaign.json',
                    campaign_outcome_ledger_file=tmp / 'missing-campaign-outcome.json',
                    frontier_ledger_file=ledger_path,
                    outbox_file=outbox,
                    memory_data=memory_fixture(),
                    git_status_text=' M tmp/theory-memory.json\n',
                    git_ignored_text='',
                )
            with contextlib.redirect_stdout(io.StringIO()):
                repeat = run_science_theory_frontier_planner(
                    runtime_memory_path=runtime,
                    transcript_file=transcript,
                    action_outcome_ledger_file=tmp / 'missing-action-outcome.json',
                    action_ledger_file=tmp / 'missing-action.json',
                    benefit_ledger_file=tmp / 'missing-benefit.json',
                    evaluator_ledger_file=tmp / 'missing-evaluator.json',
                    contract_ledger_file=tmp / 'missing-contract.json',
                    adjudicator_ledger_file=tmp / 'missing-adjudicator.json',
                    agenda_ledger_file=tmp / 'missing-agenda.json',
                    lifecycle_ledger_file=tmp / 'missing-lifecycle.json',
                    scorecard_ledger_file=tmp / 'missing-scorecard.json',
                    campaign_ledger_file=tmp / 'missing-campaign.json',
                    campaign_outcome_ledger_file=tmp / 'missing-campaign-outcome.json',
                    frontier_ledger_file=ledger_path,
                    outbox_file=outbox,
                    memory_data=memory_fixture(),
                    git_status_text=' M tmp/theory-memory.json\n',
                    git_ignored_text='',
                )
            after = runtime.read_text(encoding='utf-8')

        self.assertEqual(before, after)
        self.assertEqual('promote_supported_hypothesis_to_theory_memory', result['selected_theory_move'])
        self.assertEqual(1, result['outbox_count'])
        self.assertEqual('summarize_noop', repeat['selected_theory_move'])
        self.assertEqual(0, repeat['outbox_count'])
        self.assertFalse(result['runtime_memory_mutated'])
        self.assertEqual([], result['label_leaks'])
        self.assertTrue(result['no_sibling_imports'])
        self.assertFalse(result['project_owned_checkpoint_claimed'])

    def test_no_sibling_imports(self):
        module = Path(PROJECT_DIR) / 'agent' / 'science_theory_frontier.py'
        main = Path(PROJECT_DIR) / 'main.py'
        for source in (module.read_text(encoding='utf-8'), main.read_text(encoding='utf-8')):
            import_lines = [line for line in source.splitlines() if line.startswith(('import ', 'from '))]
            joined = '\n'.join(import_lines)
            self.assertNotIn('funfun', joined)
            self.assertNotIn('Language model 2.0', joined)
            self.assertNotIn('Code Module', joined)
            self.assertNotIn('orchastratorrrr', joined)


if __name__ == '__main__':
    unittest.main()
