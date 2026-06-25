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
from agent.science_coordination_history import SCIENCE_COORDINATION_HISTORY_LEDGER_KIND  # noqa: E402
from agent.science_coordination_policy import (  # noqa: E402
    SCIENCE_COORDINATION_POLICY_LEDGER_KIND,
    build_science_coordination_policy_recommendation,
    empty_science_coordination_policy_ledger,
    load_science_coordination_policy_ledger,
    read_science_coordination_policy_transcript,
    validate_science_coordination_policy_ledger,
    write_science_coordination_policy_ledger,
    write_science_coordination_policy_outbox_jsonl,
)
from main import run_science_coordination_policy_recommender  # noqa: E402


def memory_fixture():
    return {
        'discovery_readiness': {'readiness_score': 0.98, 'status': 'policy_ready'},
        'abstraction_discovery_evidence': {'transfer_outcome_count': 31},
    }


def history_message(
    scenario_id='scenario-one',
    *,
    payoff='code_simulation_before_science_campaign_paid_off',
    strength='moderate',
    leak=False,
    third_party=False,
):
    body = {
        'response_kind': 'science_coordination_history',
        'science_coordination_history_id': f'history-{scenario_id}',
        'scenario_id': scenario_id,
        'hypothesis_id': f'hypothesis:{scenario_id}',
        'campaign_id': f'campaign-{scenario_id}',
        'family_id': f'family-{scenario_id}',
        'interaction_sequence': ['science_history', payoff],
        'observed_sequences': ['science_history', payoff],
        'involved_modules': ['ai_different'],
        'source_commits': ['3f3bd96e5c2475ae447ecccc9ad61ad9fc81a0b1'],
        'source_tests': ['tests.test_science_coordination_history:7'],
        'selected_science_outcome': payoff,
        'payoff_class': payoff,
        'payoff_classes': [payoff],
        'recommendation_strength': strength,
        'checkpoint_boundary_state': 'repair' if payoff == 'preserve_checkpoint_boundary' else 'clean',
        'checkpoint_boundary_notes': ['boundary preserved'] if payoff == 'preserve_checkpoint_boundary' else [],
        'third_party_checkpoint_used': third_party,
    }
    if leak:
        body['checkpoint_boundary_notes'] = ['gravity label leaked']
    return build_module_chat_message(
        sender='ai_different',
        recipient='history',
        topic='ai_different.science_coordination_history',
        body=body,
        evidence={'scenario_id': scenario_id, 'payoff_class': payoff},
        tags=['science_coordination_history', payoff],
    )


def history_ledger_fixture(
    *,
    scenario_id='scenario-one',
    payoff='code_simulation_before_science_campaign_paid_off',
    strength='moderate',
):
    return {
        'schema_version': 1,
        'ledger_kind': SCIENCE_COORDINATION_HISTORY_LEDGER_KIND,
        'processed_message_ids': [],
        'processed_source_hashes': [],
        'exported_history_keys': [],
        'history_rows': [],
        'event_records': [{
            'science_coordination_history_id': f'history-{scenario_id}',
            'scenario_id': scenario_id,
            'hypothesis_id': f'hypothesis:{scenario_id}',
            'campaign_id': f'campaign-{scenario_id}',
            'family_id': f'family-{scenario_id}',
            'observed_sequences': ['science_history', payoff],
            'payoff_class': payoff,
            'payoff_classes': [payoff],
            'recommendation_strength': strength,
            'source_commits': ['3f3bd96e5c2475ae447ecccc9ad61ad9fc81a0b1'],
            'source_tests': ['tests.test_science_coordination_history:7'],
            'checkpoint_boundary_state': 'clean',
            'checkpoint_boundary_notes': [],
        }],
        'outgoing_response_ids': [],
        'latest': {},
        'ledger_hash': f'history-{scenario_id}-hash',
    }


def build_once(messages, *, ledger=None, history_ledger=None, project_boundary=None):
    updated, message = build_science_coordination_policy_recommendation(
        transcript_messages=messages,
        policy_ledger=ledger or empty_science_coordination_policy_ledger(),
        history_ledger=history_ledger or {},
        cycle_strategy_outcome_ledger={},
        cycle_strategy_ledger={},
        strategy_outcome_ledger={},
        strategy_ledger={},
        frontier_outcome_ledger={},
        theory_memory_ledger={},
        campaign_cycle_memory_ledger={},
        hypothesis_ledger={},
        campaign_ledger={},
        module_chat_ledger={},
        prior_policy_ledger={},
        runtime_memory_data=memory_fixture(),
        runtime_memory_hash_state={'unchanged': True, 'before': 'same', 'after': 'same'},
        project_owned_boundary=project_boundary or {'third_party_checkpoint_used': False},
    )
    return updated, message


class ScienceCoordinationPolicyTests(unittest.TestCase):
    def test_policy_ledger_persistence_load_and_outbox_schema(self):
        ledger, message = build_once([history_message()])
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = Path(tmpdir) / 'policy.json'
            outbox = Path(tmpdir) / 'outbox.jsonl'
            write_science_coordination_policy_ledger(ledger_path, ledger)
            loaded = load_science_coordination_policy_ledger(ledger_path)
            write_science_coordination_policy_outbox_jsonl(outbox, message)
            rows = [json.loads(line) for line in outbox.read_text(encoding='utf-8').splitlines()]

        self.assertEqual(SCIENCE_COORDINATION_POLICY_LEDGER_KIND, loaded['ledger_kind'])
        self.assertEqual('science_coordination_policy', rows[0]['body']['response_kind'])
        self.assertEqual('code_module', rows[0]['recipient'])
        self.assertTrue(rows[0]['body']['candidate_not_causal'])
        self.assertIn('not causal proof', rows[0]['body']['no_overclaiming_proof'])
        with self.assertRaisesRegex(ValueError, 'wrong ledger_kind'):
            validate_science_coordination_policy_ledger({'ledger_kind': 'wrong'})

    def test_history_to_policy_extraction_and_invalid_record(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            transcript = Path(tmpdir) / 'policy.jsonl'
            transcript.write_text(
                json.dumps(history_message(), sort_keys=True) + '\n'
                + '{"sender":"bad"}\n',
                encoding='utf-8',
            )
            parsed = read_science_coordination_policy_transcript(transcript)
        ledger, message = build_once(parsed['messages'])

        self.assertEqual(1, len(parsed['messages']))
        self.assertEqual(1, len(parsed['invalid_messages']))
        self.assertEqual('try_code_simulation_before_science_campaign', ledger['latest']['selected_policy'])
        self.assertEqual('request_code_simulation_before_campaign', message['body']['selected_action'])

    def test_deterministic_priority_selection(self):
        cases = [
            ('preserve_checkpoint_boundary', 'preserve_checkpoint_boundary', 'code_module'),
            ('code_simulation_before_science_campaign_paid_off', 'try_code_simulation_before_science_campaign', 'code_module'),
            ('funfun_formalization_before_science_campaign_paid_off', 'try_funfun_formalization_before_science_campaign', 'funfun'),
            ('language_terminology_before_science_campaign_paid_off', 'try_language_terminology_before_science_campaign', 'language_model_2'),
            ('theory_repair_loop_helped', 'try_theory_repair_before_next_campaign', 'orchestrator'),
            ('repeated_no_gain_or_noop_loop', 'avoid_repeated_noop_sequence', 'orchestrator'),
            ('sibling_request_waiting_for_evidence', 'gather_more_sibling_evidence_before_policy_change', 'broadcast'),
            ('next_frontier_hypothesis_campaign_helped', 'schedule_hypothesis_campaign_cycle', 'orchestrator'),
            ('no_measurable_coordination_payoff', 'record_no_measurable_policy_gain', 'orchestrator'),
        ]
        for payoff, selected_policy, recipient in cases:
            with self.subTest(payoff=payoff):
                ledger, message = build_once([history_message(payoff, payoff=payoff)])
                self.assertEqual(selected_policy, ledger['latest']['selected_policy'])
                self.assertEqual(recipient, message['recipient'])

    def test_ledger_sources_boundary_and_no_overclaiming(self):
        ledger, message = build_once(
            [],
            history_ledger=history_ledger_fixture(payoff='funfun_formalization_before_science_campaign_paid_off'),
            project_boundary={'third_party_checkpoint_used': True, 'project_owned_checkpoint_verified': False},
        )

        self.assertEqual('preserve_checkpoint_boundary', ledger['latest']['selected_policy'])
        self.assertEqual('repair', message['body']['checkpoint_boundary_state'])
        self.assertFalse(message['body']['project_owned_checkpoint_claimed'])
        self.assertTrue(message['body']['third_party_checkpoint_used'])
        self.assertIn('Candidate-not-causal', message['body']['recommendation_text'])

    def test_duplicate_idempotence_and_appended_evidence_update(self):
        waiting = history_message('same-scenario', payoff='sibling_request_waiting_for_evidence')
        first_ledger, first_message = build_once([waiting])
        repeat_ledger, repeat_message = build_once([waiting], ledger=first_ledger)
        appended_ledger, appended_message = build_once(
            [
                waiting,
                history_message('same-scenario', payoff='code_simulation_before_science_campaign_paid_off'),
            ],
            ledger=repeat_ledger,
        )

        self.assertEqual('gather_more_sibling_evidence_before_policy_change', first_ledger['latest']['selected_policy'])
        self.assertIsNotNone(first_message)
        self.assertEqual('summarize_noop', repeat_ledger['latest']['selected_policy'])
        self.assertIsNone(repeat_message)
        self.assertEqual('try_code_simulation_before_science_campaign', appended_ledger['latest']['selected_policy'])
        self.assertEqual('code_module', appended_message['recipient'])

    def test_cli_status_proof_preserves_runtime_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            runtime = tmp / 'theory-memory.json'
            runtime.write_text(json.dumps(memory_fixture(), sort_keys=True), encoding='utf-8')
            transcript = tmp / 'inbox.jsonl'
            transcript.write_text(json.dumps(history_message(), sort_keys=True) + '\n', encoding='utf-8')
            policy_ledger = tmp / 'policy.json'
            outbox = tmp / 'outbox.jsonl'

            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                first = run_science_coordination_policy_recommender(
                    theory_memory_file=runtime,
                    runtime_memory_path=str(runtime),
                    transcript_file=transcript,
                    history_ledger_file=tmp / 'missing-history.json',
                    cycle_strategy_outcome_ledger_file=tmp / 'missing-cycle-outcome.json',
                    cycle_strategy_ledger_file=tmp / 'missing-cycle-strategy.json',
                    strategy_outcome_ledger_file=tmp / 'missing-strategy-outcome.json',
                    strategy_ledger_file=tmp / 'missing-strategy.json',
                    frontier_outcome_ledger_file=tmp / 'missing-frontier-outcome.json',
                    policy_ledger_file=policy_ledger,
                    outbox_file=outbox,
                    memory_data=memory_fixture(),
                    git_status_text=' M tmp/theory-memory.json',
                    git_ignored_text='',
                )
                repeat = run_science_coordination_policy_recommender(
                    theory_memory_file=runtime,
                    runtime_memory_path=str(runtime),
                    transcript_file=transcript,
                    history_ledger_file=tmp / 'missing-history.json',
                    cycle_strategy_outcome_ledger_file=tmp / 'missing-cycle-outcome.json',
                    cycle_strategy_ledger_file=tmp / 'missing-cycle-strategy.json',
                    strategy_outcome_ledger_file=tmp / 'missing-strategy-outcome.json',
                    strategy_ledger_file=tmp / 'missing-strategy.json',
                    frontier_outcome_ledger_file=tmp / 'missing-frontier-outcome.json',
                    policy_ledger_file=policy_ledger,
                    outbox_file=outbox,
                    memory_data=memory_fixture(),
                    git_status_text=' M tmp/theory-memory.json',
                    git_ignored_text='',
                )

        self.assertIn('AI_DIFFERENT_SCIENCE_COORDINATION_POLICY', stream.getvalue())
        self.assertTrue(first['science_coordination_policy_capability'])
        self.assertEqual('try_code_simulation_before_science_campaign', first['selected_policy'])
        self.assertEqual('code_module', first['chosen_recipient'])
        self.assertEqual(1, first['outbox_count'])
        self.assertFalse(first['runtime_memory_mutated'])
        self.assertEqual([], first['label_leaks'])
        self.assertTrue(first['no_sibling_imports'])
        self.assertFalse(first['project_owned_checkpoint_claimed'])
        self.assertEqual('summarize_noop', repeat['selected_policy'])
        self.assertEqual(0, repeat['outbox_count'])
        self.assertFalse(repeat['runtime_memory_mutated'])

    def test_no_sibling_imports(self):
        source = Path(PROJECT_DIR) / 'agent' / 'science_coordination_policy.py'
        forbidden = ('funfun', 'Language model 2.0', 'Code Module', 'orchastratorrrr')
        imports = [
            line
            for line in source.read_text(encoding='utf-8').splitlines()
            if line.startswith(('import ', 'from '))
        ]
        self.assertFalse([line for line in imports if any(term in line for term in forbidden)])


if __name__ == '__main__':
    unittest.main()
