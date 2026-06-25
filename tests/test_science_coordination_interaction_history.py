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
from agent.science_coordination_ab_probe_outcome import SCIENCE_COORDINATION_AB_PROBE_OUTCOME_LEDGER_KIND  # noqa: E402
from agent.science_coordination_interaction_history import (  # noqa: E402
    SCIENCE_COORDINATION_INTERACTION_HISTORY_LEDGER_KIND,
    build_science_coordination_interaction_history_scorecard,
    empty_science_coordination_interaction_history_ledger,
    load_science_coordination_interaction_history_ledger,
    read_science_coordination_interaction_history_transcript,
    validate_science_coordination_interaction_history_ledger,
    write_science_coordination_interaction_history_ledger,
    write_science_coordination_interaction_history_outbox_jsonl,
)
from main import run_science_coordination_interaction_history_scorecard  # noqa: E402


def memory_fixture():
    return {
        'discovery_readiness': {'readiness_score': 0.99, 'status': 'interaction_history_ready'},
        'abstraction_discovery_evidence': {'transfer_outcome_count': 59},
    }


def outcome_message(
    scenario_id='scenario-one',
    *,
    outcome='candidate_code_simulation_probe_improved_science_campaign',
    decision='schedule_code_simulation_vs_science_only_probe',
    candidate=None,
    control=None,
    third_party=False,
):
    candidate = candidate or ['code_simulation', 'science_campaign']
    control = control or ['science_campaign_only']
    body = {
        'response_kind': 'science_coordination_ab_probe_outcome',
        'science_coordination_ab_probe_outcome_id': f'ab-outcome-{scenario_id}-{outcome}',
        'science_coordination_ab_probe_id': f'probe-{scenario_id}',
        'source_probe_ids': [f'probe-{scenario_id}'],
        'source_scorecard_ids': [f'scorecard-{scenario_id}'],
        'selected_probe_decision': decision,
        'selected_outcome_class': outcome,
        'selected_action': outcome,
        'retention_decision': 'retained' if outcome.startswith('candidate') else 'waiting',
        'candidate_sequence': candidate,
        'control_sequence': control,
        'evidence_counts': {'candidate_win': 1} if outcome.startswith('candidate') else {},
        'tested_commits': ['b2b18a7c83de05bbf7c5ebb5ef5052cc73690864'],
        'tested_tests': ['tests.test_science_coordination_ab_probe_outcome:7'],
        'candidate_not_causal': True,
        'checkpoint_boundary_state': 'repair' if outcome == 'preserve_checkpoint_boundary' else 'clean',
        'checkpoint_boundary_notes': ['boundary preserved'] if outcome == 'preserve_checkpoint_boundary' else [],
        'third_party_checkpoint_used': third_party,
        'hf_validation_used': False,
    }
    return build_module_chat_message(
        sender='ai_different',
        recipient='history',
        topic='ai_different.science_coordination_ab_probe_outcome',
        body=body,
        evidence={
            'science_coordination_ab_probe_outcome_id': body['science_coordination_ab_probe_outcome_id'],
            'science_coordination_ab_probe_id': body['science_coordination_ab_probe_id'],
            'selected_outcome_class': outcome,
            'selected_probe_decision': decision,
        },
        tags=['science_coordination_ab_probe_outcome', outcome],
    )


def outcome_ledger_fixture(
    *,
    outcomes=None,
    decision='schedule_code_simulation_vs_science_only_probe',
):
    outcomes = outcomes or [
        'candidate_code_simulation_probe_improved_science_campaign',
        'candidate_code_simulation_probe_improved_science_campaign',
    ]
    return {
        'schema_version': 1,
        'ledger_kind': SCIENCE_COORDINATION_AB_PROBE_OUTCOME_LEDGER_KIND,
        'processed_message_ids': [],
        'processed_source_hashes': [],
        'assessed_probe_keys': [],
        'probe_rows': [],
        'outcome_records': [
            {
                'science_coordination_ab_probe_outcome_id': f'ab-outcome-ledger-{index}',
                'science_coordination_ab_probe_id': f'probe-ledger-{index}',
                'source_probe_ids': [f'probe-ledger-{index}'],
                'source_scorecard_ids': [f'scorecard-ledger-{index}'],
                'selected_probe_decision': decision,
                'selected_outcome_class': outcome,
                'selected_action': outcome,
                'candidate_sequence': ['code_simulation', 'science_campaign'],
                'control_sequence': ['science_campaign_only'],
                'tested_commits': ['b2b18a7c83de05bbf7c5ebb5ef5052cc73690864'],
                'tested_tests': ['tests.test_science_coordination_ab_probe_outcome:7'],
                'checkpoint_boundary_state': 'clean',
            }
            for index, outcome in enumerate(outcomes, start=1)
        ],
        'outgoing_response_ids': [],
        'latest': {},
        'ledger_hash': 'ab-probe-outcome-ledger-hash',
    }


def build_once(messages, *, ledger=None, outcome_ledger=None, project_boundary=None):
    updated, message = build_science_coordination_interaction_history_scorecard(
        transcript_messages=messages,
        interaction_history_ledger=ledger or empty_science_coordination_interaction_history_ledger(),
        ab_probe_outcome_ledger=outcome_ledger or {},
        ab_probe_ledger={},
        policy_scorecard_ledger={},
        policy_outcome_ledger={},
        history_ledger={},
        cycle_strategy_outcome_ledger={},
        theory_memory_ledger={},
        campaign_ledger={},
        hypothesis_ledger={},
        experiment_ledger={},
        module_chat_ledger={},
        prior_interaction_history_ledger={},
        runtime_memory_data=memory_fixture(),
        runtime_memory_hash_state={'unchanged': True, 'before': 'same', 'after': 'same'},
        project_owned_boundary=project_boundary or {'third_party_checkpoint_used': False},
        hf_use_status={'hf_validation_used': False},
    )
    return updated, message


class ScienceCoordinationInteractionHistoryTests(unittest.TestCase):
    def test_history_ledger_persistence_load_and_outbox_schema(self):
        ledger, message = build_once([
            outcome_message('code-a'),
            outcome_message('code-b'),
        ])
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = Path(tmpdir) / 'history.json'
            outbox = Path(tmpdir) / 'outbox.jsonl'
            write_science_coordination_interaction_history_ledger(ledger_path, ledger)
            loaded = load_science_coordination_interaction_history_ledger(ledger_path)
            write_science_coordination_interaction_history_outbox_jsonl(outbox, message)
            rows = [json.loads(line) for line in outbox.read_text(encoding='utf-8').splitlines()]

        self.assertEqual(SCIENCE_COORDINATION_INTERACTION_HISTORY_LEDGER_KIND, loaded['ledger_kind'])
        self.assertEqual('science_coordination_interaction_history', rows[0]['body']['response_kind'])
        self.assertEqual('retain_code_simulation_before_science_when_repeatedly_helpful', rows[0]['body']['selected_decision'])
        self.assertEqual('retained', rows[0]['body']['retention_decision'])
        self.assertTrue(rows[0]['body']['candidate_not_causal'])
        self.assertIn('no causal proof', rows[0]['body']['no_overclaiming_proof'])
        self.assertFalse(rows[0]['body']['hf_validation_used'])
        with self.assertRaisesRegex(ValueError, 'wrong ledger_kind'):
            validate_science_coordination_interaction_history_ledger({'ledger_kind': 'wrong'})

    def test_outcome_to_history_extraction_and_invalid_record(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            transcript = Path(tmpdir) / 'interaction-history.jsonl'
            transcript.write_text(
                json.dumps(outcome_message('code-a'), sort_keys=True) + '\n'
                + json.dumps(outcome_message('code-b'), sort_keys=True) + '\n'
                + '{"sender":"bad"}\n',
                encoding='utf-8',
            )
            parsed = read_science_coordination_interaction_history_transcript(transcript)
        ledger, message = build_once(parsed['messages'])

        self.assertEqual(2, len(parsed['messages']))
        self.assertEqual(1, len(parsed['invalid_messages']))
        self.assertEqual('retain_code_simulation_before_science_when_repeatedly_helpful', ledger['latest']['selected_decision'])
        self.assertEqual('history', message['recipient'])

    def test_deterministic_priority_selection(self):
        cases = [
            (
                [outcome_message('boundary', outcome='preserve_checkpoint_boundary', third_party=True)],
                'preserve_checkpoint_boundary',
                'retained',
            ),
            (
                [outcome_message('code-a'), outcome_message('code-b')],
                'retain_code_simulation_before_science_when_repeatedly_helpful',
                'retained',
            ),
            (
                [
                    outcome_message('proof-a', outcome='candidate_proof_before_hypothesis_probe_improved_campaign', decision='schedule_proof_before_hypothesis_probe', candidate=['formal_proof', 'hypothesis_campaign']),
                    outcome_message('proof-b', outcome='candidate_proof_before_hypothesis_probe_improved_campaign', decision='schedule_proof_before_hypothesis_probe', candidate=['formal_proof', 'hypothesis_campaign']),
                ],
                'retain_proof_before_hypothesis_when_repeatedly_helpful',
                'retained',
            ),
            (
                [
                    outcome_message('language-a', outcome='candidate_language_summary_probe_improved_campaign', decision='schedule_language_summary_before_campaign_probe', candidate=['language_summary', 'science_campaign']),
                    outcome_message('language-b', outcome='candidate_language_summary_probe_improved_campaign', decision='schedule_language_summary_before_campaign_probe', candidate=['language_summary', 'science_campaign']),
                ],
                'retain_language_summary_before_science_when_repeatedly_helpful',
                'retained',
            ),
            (
                [
                    outcome_message('history-a', outcome='history_policy_memory_probe_improved_coordination', decision='schedule_history_policy_memory_probe', candidate=['history_policy_memory', 'science_campaign']),
                    outcome_message('history-b', outcome='history_policy_memory_probe_improved_coordination', decision='schedule_history_policy_memory_probe', candidate=['history_policy_memory', 'science_campaign']),
                ],
                'retain_history_policy_memory_when_repeatedly_helpful',
                'retained',
            ),
            (
                [
                    outcome_message('control-a', outcome='control_outperformed_candidate'),
                    outcome_message('control-b', outcome='control_outperformed_candidate'),
                ],
                'weaken_or_retire_ordering_after_repeated_control_wins',
                'retired',
            ),
            (
                [outcome_message('waiting', outcome='probe_underpowered_waiting_for_more_evidence')],
                'keep_ordering_underpowered_waiting_for_more_evidence',
                'waiting',
            ),
            (
                [outcome_message('nogain', outcome='record_no_measurable_ab_probe_gain')],
                'record_no_measurable_interaction_gain',
                'weakened',
            ),
        ]
        for messages, decision, retention in cases:
            with self.subTest(decision=decision):
                ledger, message = build_once(messages)
                self.assertEqual(decision, ledger['latest']['selected_decision'])
                self.assertEqual(retention, message['body']['retention_decision'])

    def test_boundary_preservation_and_no_overclaiming_from_project_state(self):
        ledger, message = build_once(
            [outcome_message('code-a'), outcome_message('code-b')],
            project_boundary={'third_party_checkpoint_used': True, 'project_owned_checkpoint_verified': False},
        )

        self.assertEqual('preserve_checkpoint_boundary', ledger['latest']['selected_decision'])
        self.assertEqual('repair', message['body']['checkpoint_boundary_state'])
        self.assertFalse(message['body']['project_owned_checkpoint_claimed'])
        self.assertTrue(message['body']['third_party_checkpoint_used'])
        self.assertIn('Candidate-not-causal', message['body']['candidate_not_causal_wording'])

    def test_duplicate_idempotence_and_appended_evidence_update(self):
        waiting = outcome_message('same-order', outcome='probe_underpowered_waiting_for_more_evidence')
        first_ledger, first_message = build_once([waiting])
        repeat_ledger, repeat_message = build_once([waiting], ledger=first_ledger)
        appended_ledger, appended_message = build_once(
            [
                waiting,
                outcome_message('same-order-a'),
                outcome_message('same-order-b'),
            ],
            ledger=repeat_ledger,
        )

        self.assertEqual('keep_ordering_underpowered_waiting_for_more_evidence', first_ledger['latest']['selected_decision'])
        self.assertIsNotNone(first_message)
        self.assertEqual('summarize_noop', repeat_ledger['latest']['selected_decision'])
        self.assertIsNone(repeat_message)
        self.assertEqual('retain_code_simulation_before_science_when_repeatedly_helpful', appended_ledger['latest']['selected_decision'])
        self.assertEqual('retained', appended_message['body']['retention_decision'])

    def test_cli_status_proof_preserves_runtime_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            runtime = tmp / 'theory-memory.json'
            runtime.write_text(json.dumps(memory_fixture(), sort_keys=True), encoding='utf-8')
            transcript = tmp / 'inbox.jsonl'
            transcript.write_text(
                json.dumps(outcome_message('code-a'), sort_keys=True) + '\n'
                + json.dumps(outcome_message('code-b'), sort_keys=True) + '\n',
                encoding='utf-8',
            )
            history_ledger = tmp / 'interaction-history.json'
            outbox = tmp / 'outbox.jsonl'

            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                first = run_science_coordination_interaction_history_scorecard(
                    theory_memory_file=runtime,
                    runtime_memory_path=str(runtime),
                    transcript_file=transcript,
                    interaction_history_ledger_file=history_ledger,
                    outbox_file=outbox,
                    memory_data=memory_fixture(),
                    git_status_text=' M tmp/theory-memory.json',
                    git_ignored_text='',
                )
                repeat = run_science_coordination_interaction_history_scorecard(
                    theory_memory_file=runtime,
                    runtime_memory_path=str(runtime),
                    transcript_file=transcript,
                    interaction_history_ledger_file=history_ledger,
                    outbox_file=outbox,
                    memory_data=memory_fixture(),
                    git_status_text=' M tmp/theory-memory.json',
                    git_ignored_text='',
                )

        self.assertIn('AI_DIFFERENT_SCIENCE_COORDINATION_INTERACTION_HISTORY', stream.getvalue())
        self.assertTrue(first['science_coordination_interaction_history_capability'])
        self.assertEqual('retain_code_simulation_before_science_when_repeatedly_helpful', first['selected_decision'])
        self.assertEqual('retained', first['retention_decision'])
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
        source = Path(PROJECT_DIR) / 'agent' / 'science_coordination_interaction_history.py'
        forbidden = ('funfun', 'Language model 2.0', 'Code Module', 'orchastratorrrr')
        imports = [
            line
            for line in source.read_text(encoding='utf-8').splitlines()
            if line.startswith(('import ', 'from '))
        ]
        self.assertFalse([line for line in imports if any(term in line for term in forbidden)])


if __name__ == '__main__':
    unittest.main()
