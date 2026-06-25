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
from agent.science_coordination_ab_probe import (  # noqa: E402
    SCIENCE_COORDINATION_AB_PROBE_LEDGER_KIND,
    build_science_coordination_ab_probe_plan,
    empty_science_coordination_ab_probe_ledger,
    load_science_coordination_ab_probe_ledger,
    read_science_coordination_ab_probe_transcript,
    validate_science_coordination_ab_probe_ledger,
    write_science_coordination_ab_probe_ledger,
    write_science_coordination_ab_probe_outbox_jsonl,
)
from agent.science_coordination_policy_scorecard import SCIENCE_COORDINATION_POLICY_SCORECARD_LEDGER_KIND  # noqa: E402
from main import run_science_coordination_ab_probe_planner  # noqa: E402


def memory_fixture():
    return {
        'discovery_readiness': {'readiness_score': 0.99, 'status': 'ab_probe_ready'},
        'abstraction_discovery_evidence': {'transfer_outcome_count': 47},
    }


def scorecard_message(
    scenario_id='scenario-one',
    *,
    policy_class='try_code_simulation_before_science_campaign',
    decision='retain_policy_after_repeated_science_campaign_improvement',
    third_party=False,
):
    body = {
        'response_kind': 'science_coordination_policy_scorecard',
        'science_coordination_policy_scorecard_id': f'scorecard-{scenario_id}',
        'source_policy_ids': [f'policy-{scenario_id}'],
        'source_outcome_ids': [f'outcome-{scenario_id}'],
        'policy_class': policy_class,
        'selected_retention_decision': decision,
        'selected_action': decision,
        'recommendation_strength': 'strong' if 'retain' in decision else 'weak',
        'tested_commits': ['efb9c376d114a54bd3e5c4f7a06a2c5952f30277'],
        'tested_tests': ['tests.test_science_coordination_policy_scorecard:7'],
        'campaign_evidence_ids': [f'campaign-{scenario_id}'],
        'hypothesis_evidence_ids': [f'hypothesis:{scenario_id}'],
        'checkpoint_boundary_state': 'repair' if decision == 'preserve_checkpoint_boundary' else 'clean',
        'checkpoint_boundary_notes': ['boundary preserved'] if decision == 'preserve_checkpoint_boundary' else [],
        'candidate_not_causal': True,
        'third_party_checkpoint_used': third_party,
    }
    return build_module_chat_message(
        sender='ai_different',
        recipient='orchestrator',
        topic='ai_different.science_coordination_policy_scorecard',
        body=body,
        evidence={
            'science_coordination_policy_scorecard_id': f'scorecard-{scenario_id}',
            'policy_class': policy_class,
            'selected_retention_decision': decision,
        },
        tags=['science_coordination_policy_scorecard', decision],
    )


def scorecard_ledger_fixture(
    *,
    scenario_id='scenario-one',
    policy_class='try_code_simulation_before_science_campaign',
    decision='retain_policy_after_repeated_science_campaign_improvement',
):
    return {
        'schema_version': 1,
        'ledger_kind': SCIENCE_COORDINATION_POLICY_SCORECARD_LEDGER_KIND,
        'processed_message_ids': [],
        'processed_source_hashes': [],
        'scored_policy_keys': [],
        'scorecard_rows': [],
        'scorecard_records': [{
            'science_coordination_policy_scorecard_id': f'scorecard-{scenario_id}',
            'source_policy_ids': [f'policy-{scenario_id}'],
            'source_outcome_ids': [f'outcome-{scenario_id}'],
            'policy_class': policy_class,
            'selected_retention_decision': decision,
            'tested_commits': ['efb9c376d114a54bd3e5c4f7a06a2c5952f30277'],
            'tested_tests': ['tests.test_science_coordination_policy_scorecard:7'],
            'campaign_evidence_ids': [f'campaign-{scenario_id}'],
            'hypothesis_evidence_ids': [f'hypothesis:{scenario_id}'],
            'checkpoint_boundary_state': 'clean',
            'checkpoint_boundary_notes': [],
        }],
        'outgoing_response_ids': [],
        'latest': {},
        'ledger_hash': f'scorecard-{scenario_id}-hash',
    }


def build_once(messages, *, ledger=None, scorecard_ledger=None, project_boundary=None):
    updated, message = build_science_coordination_ab_probe_plan(
        transcript_messages=messages,
        ab_probe_ledger=ledger or empty_science_coordination_ab_probe_ledger(),
        scorecard_ledger=scorecard_ledger or {},
        policy_outcome_ledger={},
        policy_ledger={},
        history_ledger={},
        cycle_strategy_outcome_ledger={},
        theory_memory_ledger={},
        campaign_cycle_memory_ledger={},
        hypothesis_ledger={},
        experiment_ledger={},
        module_chat_ledger={},
        prior_ab_probe_ledger={},
        runtime_memory_data=memory_fixture(),
        runtime_memory_hash_state={'unchanged': True, 'before': 'same', 'after': 'same'},
        project_owned_boundary=project_boundary or {'third_party_checkpoint_used': False},
        hf_use_status={'hf_validation_used': False},
    )
    return updated, message


class ScienceCoordinationABProbeTests(unittest.TestCase):
    def test_ab_probe_ledger_persistence_load_and_outbox_schema(self):
        ledger, message = build_once([scorecard_message()])
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = Path(tmpdir) / 'probe.json'
            outbox = Path(tmpdir) / 'outbox.jsonl'
            write_science_coordination_ab_probe_ledger(ledger_path, ledger)
            loaded = load_science_coordination_ab_probe_ledger(ledger_path)
            write_science_coordination_ab_probe_outbox_jsonl(outbox, message)
            rows = [json.loads(line) for line in outbox.read_text(encoding='utf-8').splitlines()]

        self.assertEqual(SCIENCE_COORDINATION_AB_PROBE_LEDGER_KIND, loaded['ledger_kind'])
        self.assertEqual('science_coordination_ab_probe', rows[0]['body']['response_kind'])
        self.assertEqual('schedule_code_simulation_vs_science_only_probe', rows[0]['body']['selected_probe_decision'])
        self.assertEqual(['code_simulation', 'science_campaign'], rows[0]['body']['candidate_sequence'])
        self.assertTrue(rows[0]['body']['candidate_not_causal'])
        self.assertIn('no causal proof', rows[0]['body']['no_overclaiming_proof'])
        self.assertFalse(rows[0]['body']['hf_validation_used'])
        with self.assertRaisesRegex(ValueError, 'wrong ledger_kind'):
            validate_science_coordination_ab_probe_ledger({'ledger_kind': 'wrong'})

    def test_scorecard_to_probe_extraction_and_invalid_record(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            transcript = Path(tmpdir) / 'probe.jsonl'
            transcript.write_text(
                json.dumps(scorecard_message(), sort_keys=True) + '\n'
                + '{"sender":"bad"}\n',
                encoding='utf-8',
            )
            parsed = read_science_coordination_ab_probe_transcript(transcript)
        ledger, message = build_once(parsed['messages'])

        self.assertEqual(1, len(parsed['messages']))
        self.assertEqual(1, len(parsed['invalid_messages']))
        self.assertEqual('schedule_code_simulation_vs_science_only_probe', ledger['latest']['selected_probe_decision'])
        self.assertEqual('orchestrator', message['recipient'])

    def test_deterministic_priority_selection(self):
        cases = [
            (
                [scorecard_message('boundary', decision='preserve_checkpoint_boundary', third_party=True)],
                'preserve_checkpoint_boundary',
                ['preserve_checkpoint_boundary', 'defer_probe'],
            ),
            (
                [scorecard_message('code', policy_class='try_code_simulation_before_science_campaign')],
                'schedule_code_simulation_vs_science_only_probe',
                ['code_simulation', 'science_campaign'],
            ),
            (
                [scorecard_message('proof', policy_class='try_funfun_formalization_before_science_campaign')],
                'schedule_proof_before_hypothesis_probe',
                ['formal_proof', 'hypothesis_campaign'],
            ),
            (
                [scorecard_message('language', policy_class='try_language_terminology_before_science_campaign')],
                'schedule_language_summary_before_campaign_probe',
                ['language_summary', 'science_campaign'],
            ),
            (
                [scorecard_message('history', policy_class='history_policy_memory')],
                'schedule_history_policy_memory_probe',
                ['history_policy_memory', 'science_campaign'],
            ),
            (
                [scorecard_message('waiting', decision='keep_policy_waiting_for_more_evidence')],
                'keep_probe_waiting_for_more_scorecard_evidence',
                ['wait_for_more_scorecard_evidence'],
            ),
            (
                [scorecard_message('retired', decision='retire_policy_after_repeated_noop_or_no_gain')],
                'retire_probe_for_repeated_no_gain_policy',
                ['retire_policy_probe'],
            ),
            (
                [scorecard_message('nogain', decision='record_no_measurable_scorecard_gain')],
                'record_no_measurable_probe_gain',
                ['record_no_measurable_probe_gain'],
            ),
        ]
        for messages, decision, candidate in cases:
            with self.subTest(decision=decision):
                ledger, message = build_once(messages)
                self.assertEqual(decision, ledger['latest']['selected_probe_decision'])
                self.assertEqual(candidate, message['body']['candidate_sequence'])

    def test_boundary_preservation_and_no_overclaiming_from_project_state(self):
        ledger, message = build_once(
            [scorecard_message()],
            project_boundary={'third_party_checkpoint_used': True, 'project_owned_checkpoint_verified': False},
        )

        self.assertEqual('preserve_checkpoint_boundary', ledger['latest']['selected_probe_decision'])
        self.assertEqual('repair', message['body']['checkpoint_boundary_state'])
        self.assertFalse(message['body']['project_owned_checkpoint_claimed'])
        self.assertTrue(message['body']['third_party_checkpoint_used'])
        self.assertIn('Candidate-not-causal', message['body']['candidate_not_causal_wording'])

    def test_duplicate_idempotence_and_appended_evidence_update(self):
        waiting = scorecard_message('same-policy', decision='keep_policy_waiting_for_more_evidence')
        first_ledger, first_message = build_once([waiting])
        repeat_ledger, repeat_message = build_once([waiting], ledger=first_ledger)
        appended_ledger, appended_message = build_once(
            [
                waiting,
                scorecard_message(
                    'same-policy',
                    policy_class='try_code_simulation_before_science_campaign',
                    decision='schedule_science_policy_ab_probe',
                ),
            ],
            ledger=repeat_ledger,
        )

        self.assertEqual('keep_probe_waiting_for_more_scorecard_evidence', first_ledger['latest']['selected_probe_decision'])
        self.assertIsNotNone(first_message)
        self.assertEqual('summarize_noop', repeat_ledger['latest']['selected_probe_decision'])
        self.assertIsNone(repeat_message)
        self.assertEqual('schedule_code_simulation_vs_science_only_probe', appended_ledger['latest']['selected_probe_decision'])
        self.assertEqual(['code_simulation', 'science_campaign'], appended_message['body']['candidate_sequence'])

    def test_cli_status_proof_preserves_runtime_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            runtime = tmp / 'theory-memory.json'
            runtime.write_text(json.dumps(memory_fixture(), sort_keys=True), encoding='utf-8')
            transcript = tmp / 'inbox.jsonl'
            transcript.write_text(json.dumps(scorecard_message(), sort_keys=True) + '\n', encoding='utf-8')
            probe_ledger = tmp / 'probe.json'
            outbox = tmp / 'outbox.jsonl'

            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                first = run_science_coordination_ab_probe_planner(
                    theory_memory_file=runtime,
                    runtime_memory_path=str(runtime),
                    transcript_file=transcript,
                    scorecard_ledger_file=tmp / 'missing-scorecard.json',
                    policy_outcome_ledger_file=tmp / 'missing-policy-outcome.json',
                    policy_ledger_file=tmp / 'missing-policy.json',
                    history_ledger_file=tmp / 'missing-history.json',
                    cycle_strategy_outcome_ledger_file=tmp / 'missing-cycle-outcome.json',
                    ab_probe_ledger_file=probe_ledger,
                    outbox_file=outbox,
                    memory_data=memory_fixture(),
                    git_status_text=' M tmp/theory-memory.json',
                    git_ignored_text='',
                )
                repeat = run_science_coordination_ab_probe_planner(
                    theory_memory_file=runtime,
                    runtime_memory_path=str(runtime),
                    transcript_file=transcript,
                    scorecard_ledger_file=tmp / 'missing-scorecard.json',
                    policy_outcome_ledger_file=tmp / 'missing-policy-outcome.json',
                    policy_ledger_file=tmp / 'missing-policy.json',
                    history_ledger_file=tmp / 'missing-history.json',
                    cycle_strategy_outcome_ledger_file=tmp / 'missing-cycle-outcome.json',
                    ab_probe_ledger_file=probe_ledger,
                    outbox_file=outbox,
                    memory_data=memory_fixture(),
                    git_status_text=' M tmp/theory-memory.json',
                    git_ignored_text='',
                )

        self.assertIn('AI_DIFFERENT_SCIENCE_COORDINATION_AB_PROBE', stream.getvalue())
        self.assertTrue(first['science_coordination_ab_probe_capability'])
        self.assertEqual('schedule_code_simulation_vs_science_only_probe', first['selected_decision'])
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
        source = Path(PROJECT_DIR) / 'agent' / 'science_coordination_ab_probe.py'
        forbidden = ('funfun', 'Language model 2.0', 'Code Module', 'orchastratorrrr')
        imports = [
            line
            for line in source.read_text(encoding='utf-8').splitlines()
            if line.startswith(('import ', 'from '))
        ]
        self.assertFalse([line for line in imports if any(term in line for term in forbidden)])


if __name__ == '__main__':
    unittest.main()
