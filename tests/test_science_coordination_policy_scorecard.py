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
from agent.science_coordination_policy_outcome import SCIENCE_COORDINATION_POLICY_OUTCOME_LEDGER_KIND  # noqa: E402
from agent.science_coordination_policy_scorecard import (  # noqa: E402
    SCIENCE_COORDINATION_POLICY_SCORECARD_LEDGER_KIND,
    build_science_coordination_policy_scorecard,
    empty_science_coordination_policy_scorecard_ledger,
    load_science_coordination_policy_scorecard_ledger,
    read_science_coordination_policy_scorecard_transcript,
    validate_science_coordination_policy_scorecard_ledger,
    write_science_coordination_policy_scorecard_ledger,
    write_science_coordination_policy_scorecard_outbox_jsonl,
)
from main import run_science_coordination_policy_scorecard  # noqa: E402


def memory_fixture():
    return {
        'discovery_readiness': {'readiness_score': 0.99, 'status': 'scorecard_ready'},
        'abstraction_discovery_evidence': {'transfer_outcome_count': 41},
    }


def outcome_message(
    scenario_id='scenario-one',
    *,
    selected_policy='try_code_simulation_before_science_campaign',
    selected_outcome='code_simulation_policy_improved_science_campaign',
    retention='retained',
    leak=False,
    third_party=False,
):
    body = {
        'response_kind': 'science_coordination_policy_outcome',
        'science_coordination_policy_outcome_id': f'policy-outcome-{scenario_id}-{selected_outcome}',
        'science_coordination_policy_id': f'policy-{scenario_id}',
        'source_policy_ids': [f'policy-{scenario_id}'],
        'scenario_id': scenario_id,
        'hypothesis_id': f'hypothesis:{scenario_id}',
        'campaign_id': f'campaign-{scenario_id}',
        'family_id': f'family-{scenario_id}',
        'selected_policy': selected_policy,
        'selected_outcome': selected_outcome,
        'selected_action': selected_outcome,
        'policy_retention_state': retention,
        'observed_science_side_evidence': [{'source': 'policy_outcome', 'status': selected_outcome}],
        'source_commits': ['4600e520da64b9b513ef04348473738436d69a2e'],
        'source_tests': ['tests.test_science_coordination_policy_outcome:7'],
        'checkpoint_boundary_state': 'repair' if selected_outcome == 'preserve_checkpoint_boundary' else 'clean',
        'checkpoint_boundary_notes': ['boundary preserved'] if selected_outcome == 'preserve_checkpoint_boundary' else [],
        'candidate_not_causal': True,
        'third_party_checkpoint_used': third_party,
    }
    if leak:
        body['checkpoint_boundary_notes'] = ['gravity label leaked']
    return build_module_chat_message(
        sender='ai_different',
        recipient='orchestrator',
        topic='ai_different.science_coordination_policy_outcome',
        body=body,
        evidence={
            'scenario_id': scenario_id,
            'science_coordination_policy_outcome_id': body['science_coordination_policy_outcome_id'],
            'science_coordination_policy_id': f'policy-{scenario_id}',
            'selected_policy': selected_policy,
            'selected_outcome': selected_outcome,
        },
        tags=['science_coordination_policy_outcome', selected_outcome],
    )


def outcome_ledger_fixture(
    *,
    scenario_id='scenario-one',
    selected_policy='try_code_simulation_before_science_campaign',
    outcomes=None,
):
    outcomes = outcomes or ['code_simulation_policy_improved_science_campaign']
    return {
        'schema_version': 1,
        'ledger_kind': SCIENCE_COORDINATION_POLICY_OUTCOME_LEDGER_KIND,
        'processed_message_ids': [],
        'processed_source_hashes': [],
        'assessed_policy_keys': [],
        'policy_rows': [],
        'outcome_records': [
            {
                'science_coordination_policy_outcome_id': f'policy-outcome-{scenario_id}-{index}',
                'science_coordination_policy_id': f'policy-{scenario_id}',
                'source_policy_ids': [f'policy-{scenario_id}'],
                'selected_policy': selected_policy,
                'selected_outcome': outcome,
                'scenario_id': scenario_id,
                'hypothesis_id': f'hypothesis:{scenario_id}',
                'campaign_id': f'campaign-{scenario_id}',
                'source_commits': ['4600e520da64b9b513ef04348473738436d69a2e'],
                'source_tests': ['tests.test_science_coordination_policy_outcome:7'],
                'checkpoint_boundary_state': 'clean',
                'checkpoint_boundary_notes': [],
            }
            for index, outcome in enumerate(outcomes, start=1)
        ],
        'outgoing_response_ids': [],
        'latest': {},
        'ledger_hash': f'policy-outcome-{scenario_id}-hash',
    }


def build_once(messages, *, ledger=None, outcome_ledger=None, project_boundary=None):
    updated, message = build_science_coordination_policy_scorecard(
        transcript_messages=messages,
        scorecard_ledger=ledger or empty_science_coordination_policy_scorecard_ledger(),
        policy_outcome_ledger=outcome_ledger or {},
        policy_ledger={},
        history_ledger={},
        cycle_strategy_outcome_ledger={},
        theory_memory_ledger={},
        campaign_cycle_memory_ledger={},
        hypothesis_ledger={},
        experiment_ledger={},
        module_chat_ledger={},
        prior_scorecard_ledger={},
        runtime_memory_data=memory_fixture(),
        runtime_memory_hash_state={'unchanged': True, 'before': 'same', 'after': 'same'},
        project_owned_boundary=project_boundary or {'third_party_checkpoint_used': False},
    )
    return updated, message


class ScienceCoordinationPolicyScorecardTests(unittest.TestCase):
    def test_scorecard_ledger_persistence_load_and_outbox_schema(self):
        ledger, message = build_once([
            outcome_message('a'),
            outcome_message('b'),
        ])
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = Path(tmpdir) / 'scorecard.json'
            outbox = Path(tmpdir) / 'outbox.jsonl'
            write_science_coordination_policy_scorecard_ledger(ledger_path, ledger)
            loaded = load_science_coordination_policy_scorecard_ledger(ledger_path)
            write_science_coordination_policy_scorecard_outbox_jsonl(outbox, message)
            rows = [json.loads(line) for line in outbox.read_text(encoding='utf-8').splitlines()]

        self.assertEqual(SCIENCE_COORDINATION_POLICY_SCORECARD_LEDGER_KIND, loaded['ledger_kind'])
        self.assertEqual('science_coordination_policy_scorecard', rows[0]['body']['response_kind'])
        self.assertEqual('retain_policy_after_repeated_science_campaign_improvement', rows[0]['body']['selected_retention_decision'])
        self.assertEqual('retained', rows[0]['body']['policy_should'])
        self.assertTrue(rows[0]['body']['candidate_not_causal'])
        self.assertIn('no causal proof', rows[0]['body']['no_overclaiming_proof'])
        with self.assertRaisesRegex(ValueError, 'wrong ledger_kind'):
            validate_science_coordination_policy_scorecard_ledger({'ledger_kind': 'wrong'})

    def test_outcome_to_scorecard_extraction_and_invalid_record(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            transcript = Path(tmpdir) / 'scorecard.jsonl'
            transcript.write_text(
                json.dumps(outcome_message('a'), sort_keys=True) + '\n'
                + json.dumps(outcome_message('b'), sort_keys=True) + '\n'
                + '{"sender":"bad"}\n',
                encoding='utf-8',
            )
            parsed = read_science_coordination_policy_scorecard_transcript(transcript)
        ledger, message = build_once(parsed['messages'])

        self.assertEqual(2, len(parsed['messages']))
        self.assertEqual(1, len(parsed['invalid_messages']))
        self.assertEqual('retain_policy_after_repeated_science_campaign_improvement', ledger['latest']['selected_retention_decision'])
        self.assertEqual('strong', message['body']['recommendation_strength'])

    def test_deterministic_priority_selection(self):
        cases = [
            (
                [outcome_message('boundary', selected_outcome='preserve_checkpoint_boundary', third_party=True)],
                'preserve_checkpoint_boundary',
                'retained',
            ),
            (
                [outcome_message('code-a'), outcome_message('code-b')],
                'retain_policy_after_repeated_science_campaign_improvement',
                'retained',
            ),
            (
                [
                    outcome_message('mixed-a'),
                    outcome_message('mixed-b', selected_outcome='no_measurable_policy_gain', retention='weakened'),
                ],
                'weaken_policy_after_mixed_evidence',
                'weakened',
            ),
            (
                [
                    outcome_message('noop-a', selected_outcome='no_measurable_policy_gain', retention='weakened'),
                    outcome_message('noop-b', selected_outcome='repeated_noop_policy_retired', retention='retired'),
                ],
                'retire_policy_after_repeated_noop_or_no_gain',
                'retired',
            ),
            (
                [outcome_message('waiting', selected_outcome='policy_waiting_for_sibling_evidence', retention='waiting')],
                'keep_policy_waiting_for_more_evidence',
                'waiting',
            ),
            (
                [outcome_message('probe')],
                'schedule_science_policy_ab_probe',
                'probed',
            ),
            (
                [outcome_message('nogain', selected_outcome='no_measurable_policy_gain', retention='weakened')],
                'record_no_measurable_scorecard_gain',
                'weakened',
            ),
        ]
        for messages, decision, policy_should in cases:
            with self.subTest(decision=decision):
                ledger, message = build_once(messages)
                self.assertEqual(decision, ledger['latest']['selected_retention_decision'])
                self.assertEqual(policy_should, message['body']['policy_should'])

    def test_boundary_preservation_and_no_overclaiming_from_project_state(self):
        ledger, message = build_once(
            [outcome_message('code-a'), outcome_message('code-b')],
            project_boundary={'third_party_checkpoint_used': True, 'project_owned_checkpoint_verified': False},
        )

        self.assertEqual('preserve_checkpoint_boundary', ledger['latest']['selected_retention_decision'])
        self.assertEqual('repair', message['body']['checkpoint_boundary_state'])
        self.assertFalse(message['body']['project_owned_checkpoint_claimed'])
        self.assertTrue(message['body']['third_party_checkpoint_used'])
        self.assertIn('Candidate-not-causal', message['body']['candidate_not_causal_wording'])

    def test_duplicate_idempotence_and_appended_evidence_update(self):
        waiting = outcome_message('same-policy', selected_outcome='policy_waiting_for_sibling_evidence', retention='waiting')
        first_ledger, first_message = build_once([waiting])
        repeat_ledger, repeat_message = build_once([waiting], ledger=first_ledger)
        appended_ledger, appended_message = build_once(
            [
                waiting,
                outcome_message('same-policy', selected_outcome='code_simulation_policy_improved_science_campaign'),
            ],
            ledger=repeat_ledger,
        )

        self.assertEqual('keep_policy_waiting_for_more_evidence', first_ledger['latest']['selected_retention_decision'])
        self.assertIsNotNone(first_message)
        self.assertEqual('summarize_noop', repeat_ledger['latest']['selected_retention_decision'])
        self.assertIsNone(repeat_message)
        self.assertEqual('weaken_policy_after_mixed_evidence', appended_ledger['latest']['selected_retention_decision'])
        self.assertEqual('weakened', appended_message['body']['policy_should'])

    def test_cli_status_proof_preserves_runtime_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            runtime = tmp / 'theory-memory.json'
            runtime.write_text(json.dumps(memory_fixture(), sort_keys=True), encoding='utf-8')
            transcript = tmp / 'inbox.jsonl'
            transcript.write_text(
                json.dumps(outcome_message('a'), sort_keys=True) + '\n'
                + json.dumps(outcome_message('b'), sort_keys=True) + '\n',
                encoding='utf-8',
            )
            scorecard_ledger = tmp / 'scorecard.json'
            outbox = tmp / 'outbox.jsonl'

            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                first = run_science_coordination_policy_scorecard(
                    theory_memory_file=runtime,
                    runtime_memory_path=str(runtime),
                    transcript_file=transcript,
                    policy_outcome_ledger_file=tmp / 'missing-policy-outcome.json',
                    policy_ledger_file=tmp / 'missing-policy.json',
                    history_ledger_file=tmp / 'missing-history.json',
                    cycle_strategy_outcome_ledger_file=tmp / 'missing-cycle-outcome.json',
                    scorecard_ledger_file=scorecard_ledger,
                    outbox_file=outbox,
                    memory_data=memory_fixture(),
                    git_status_text=' M tmp/theory-memory.json',
                    git_ignored_text='',
                )
                repeat = run_science_coordination_policy_scorecard(
                    theory_memory_file=runtime,
                    runtime_memory_path=str(runtime),
                    transcript_file=transcript,
                    policy_outcome_ledger_file=tmp / 'missing-policy-outcome.json',
                    policy_ledger_file=tmp / 'missing-policy.json',
                    history_ledger_file=tmp / 'missing-history.json',
                    cycle_strategy_outcome_ledger_file=tmp / 'missing-cycle-outcome.json',
                    scorecard_ledger_file=scorecard_ledger,
                    outbox_file=outbox,
                    memory_data=memory_fixture(),
                    git_status_text=' M tmp/theory-memory.json',
                    git_ignored_text='',
                )

        self.assertIn('AI_DIFFERENT_SCIENCE_COORDINATION_POLICY_SCORECARD', stream.getvalue())
        self.assertTrue(first['science_coordination_policy_scorecard_capability'])
        self.assertEqual('retain_policy_after_repeated_science_campaign_improvement', first['selected_decision'])
        self.assertEqual('strong', first['recommendation_strength'])
        self.assertEqual(1, first['outbox_count'])
        self.assertFalse(first['runtime_memory_mutated'])
        self.assertEqual([], first['label_leaks'])
        self.assertTrue(first['no_sibling_imports'])
        self.assertFalse(first['project_owned_checkpoint_claimed'])
        self.assertFalse(first['hf_validation_used'])
        self.assertEqual('summarize_noop', repeat['selected_decision'])
        self.assertEqual(0, repeat['outbox_count'])
        self.assertFalse(repeat['runtime_memory_mutated'])

    def test_no_sibling_imports(self):
        source = Path(PROJECT_DIR) / 'agent' / 'science_coordination_policy_scorecard.py'
        forbidden = ('funfun', 'Language model 2.0', 'Code Module', 'orchastratorrrr')
        imports = [
            line
            for line in source.read_text(encoding='utf-8').splitlines()
            if line.startswith(('import ', 'from '))
        ]
        self.assertFalse([line for line in imports if any(term in line for term in forbidden)])


if __name__ == '__main__':
    unittest.main()
