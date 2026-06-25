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
from agent.science_history_guided_policy_outcome import SCIENCE_HISTORY_GUIDED_POLICY_OUTCOME_LEDGER_KIND  # noqa: E402
from agent.science_history_guided_policy_retention import (  # noqa: E402
    SCIENCE_HISTORY_GUIDED_POLICY_RETENTION_LEDGER_KIND,
    build_science_history_guided_policy_retention_scorecard,
    empty_science_history_guided_policy_retention_ledger,
    load_science_history_guided_policy_retention_ledger,
    read_science_history_guided_policy_retention_transcript,
    validate_science_history_guided_policy_retention_ledger,
    write_science_history_guided_policy_retention_ledger,
    write_science_history_guided_policy_retention_outbox_jsonl,
)
from main import run_science_history_guided_policy_retention_scorecard  # noqa: E402


def memory_fixture():
    return {
        'discovery_readiness': {'readiness_score': 0.99, 'status': 'history_guided_policy_retention_ready'},
        'abstraction_discovery_evidence': {'transfer_outcome_count': 71},
    }


def outcome_message(
    scenario_id='scenario-one',
    *,
    ordering='code_simulation_before_science',
    outcome='retained_code_simulation_ordering_improved_science_campaign',
    retention='retained',
    third_party=False,
):
    body = {
        'response_kind': 'science_history_guided_policy_outcome',
        'science_history_guided_policy_outcome_id': f'policy-outcome-{scenario_id}',
        'science_history_guided_policy_id': f'policy-{scenario_id}',
        'source_interaction_history_ids': [f'interaction-history-{scenario_id}'],
        'selected_outcome_class': outcome,
        'retention_decision': retention,
        'selected_ordering_key': ordering,
        'target_campaign_ids': [f'campaign-{scenario_id}'],
        'target_hypothesis_ids': [f'hypothesis:{scenario_id}'],
        'target_simulation_ids': [f'simulation-{scenario_id}'],
        'tested_commits': ['656a493fdf9ee4b60c6fd609fc649834146845f9'],
        'tested_tests': ['tests.test_science_history_guided_policy_outcome:7'],
        'candidate_not_causal': True,
        'checkpoint_boundary_state': 'repair' if third_party or outcome == 'preserve_checkpoint_boundary' else 'clean',
        'checkpoint_boundary_notes': ['checkpoint boundary preserved'] if third_party or outcome == 'preserve_checkpoint_boundary' else [],
        'third_party_checkpoint_used': third_party,
        'hf_validation_used': False,
    }
    return build_module_chat_message(
        sender='ai_different',
        recipient='history',
        topic='ai_different.science_history_guided_policy_outcome',
        body=body,
        evidence={
            'science_history_guided_policy_outcome_id': body['science_history_guided_policy_outcome_id'],
            'selected_outcome_class': outcome,
            'retention_decision': retention,
            'selected_ordering_key': ordering,
            'label_clean': True,
        },
        tags=['science_history_guided_policy_outcome', outcome],
    )


def outcome_ledger_fixture(*messages):
    records = [dict(message['body']) for message in messages]
    return {
        'schema_version': 1,
        'ledger_kind': SCIENCE_HISTORY_GUIDED_POLICY_OUTCOME_LEDGER_KIND,
        'processed_message_ids': [],
        'processed_source_hashes': [],
        'assessed_policy_keys': [],
        'outcome_records': records,
        'policy_rows': [],
        'outgoing_response_ids': [],
        'latest': {},
        'ledger_hash': 'history-guided-policy-outcome-fixture-hash',
    }


def build_once(messages, *, ledger=None, outcome_ledger=None, project_boundary=None):
    updated, message = build_science_history_guided_policy_retention_scorecard(
        transcript_messages=messages,
        retention_ledger=ledger or empty_science_history_guided_policy_retention_ledger(),
        policy_outcome_ledger=outcome_ledger or {},
        history_guided_policy_ledger={},
        interaction_history_ledger={},
        ab_probe_outcome_ledger={},
        ab_probe_ledger={},
        theory_memory_ledger={},
        campaign_ledger={},
        hypothesis_ledger={},
        experiment_ledger={},
        module_chat_ledger={},
        prior_retention_ledger={},
        runtime_memory_data=memory_fixture(),
        runtime_memory_hash_state={'unchanged': True, 'before': 'same', 'after': 'same'},
        project_owned_boundary=project_boundary or {'third_party_checkpoint_used': False},
        hf_use_status={'hf_validation_used': False},
    )
    return updated, message


class ScienceHistoryGuidedPolicyRetentionTests(unittest.TestCase):
    def test_retention_ledger_persistence_load_and_outbox_schema(self):
        ledger, message = build_once([
            outcome_message('code-a'),
            outcome_message('code-b'),
        ])
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = Path(tmpdir) / 'retention.json'
            outbox = Path(tmpdir) / 'outbox.jsonl'
            write_science_history_guided_policy_retention_ledger(ledger_path, ledger)
            loaded = load_science_history_guided_policy_retention_ledger(ledger_path)
            write_science_history_guided_policy_retention_outbox_jsonl(outbox, message)
            rows = [json.loads(line) for line in outbox.read_text(encoding='utf-8').splitlines()]

        self.assertEqual(SCIENCE_HISTORY_GUIDED_POLICY_RETENTION_LEDGER_KIND, loaded['ledger_kind'])
        self.assertEqual('science_history_guided_policy_retention', rows[0]['body']['response_kind'])
        self.assertEqual('retain_code_simulation_ordering_when_repeatedly_helpful', rows[0]['body']['selected_decision'])
        self.assertEqual('retained', rows[0]['body']['retention_decision'])
        self.assertEqual('code_simulation_before_science', rows[0]['body']['retained_policy_key'])
        self.assertIn('Candidate-not-causal', rows[0]['body']['candidate_not_causal_wording'])
        self.assertIn('no causal proof', rows[0]['body']['no_overclaiming_proof'])
        self.assertFalse(rows[0]['body']['hf_validation_used'])
        with self.assertRaisesRegex(ValueError, 'wrong ledger_kind'):
            validate_science_history_guided_policy_retention_ledger({'ledger_kind': 'wrong'})

    def test_outcome_to_retention_extraction_and_invalid_record(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            transcript = Path(tmpdir) / 'retention.jsonl'
            transcript.write_text(
                json.dumps(outcome_message('code-a'), sort_keys=True) + '\n'
                + json.dumps(outcome_message('code-b'), sort_keys=True) + '\n'
                + '{"sender":"bad"}\n',
                encoding='utf-8',
            )
            parsed = read_science_history_guided_policy_retention_transcript(transcript)
        ledger, message = build_once(parsed['messages'])

        self.assertEqual(2, len(parsed['messages']))
        self.assertEqual(1, len(parsed['invalid_messages']))
        self.assertEqual('retain_code_simulation_ordering_when_repeatedly_helpful', ledger['latest']['selected_decision'])
        self.assertEqual('history', message['recipient'])

    def test_deterministic_priority_selection(self):
        cases = [
            (
                [outcome_message('boundary', outcome='preserve_checkpoint_boundary', third_party=True)],
                'preserve_checkpoint_boundary_memory',
                'retained',
            ),
            (
                [outcome_message('code-a'), outcome_message('code-b')],
                'retain_code_simulation_ordering_when_repeatedly_helpful',
                'retained',
            ),
            (
                [
                    outcome_message('proof-a', ordering='proof_before_hypothesis', outcome='retained_proof_before_hypothesis_ordering_improved_campaign'),
                    outcome_message('proof-b', ordering='proof_before_hypothesis', outcome='retained_proof_before_hypothesis_ordering_improved_campaign'),
                ],
                'retain_proof_before_hypothesis_ordering_when_repeatedly_helpful',
                'retained',
            ),
            (
                [
                    outcome_message('language-a', ordering='language_summary_before_science', outcome='retained_language_summary_ordering_improved_campaign'),
                    outcome_message('language-b', ordering='language_summary_before_science', outcome='retained_language_summary_ordering_improved_campaign'),
                ],
                'retain_language_summary_ordering_when_repeatedly_helpful',
                'retained',
            ),
            (
                [
                    outcome_message('history-a', ordering='history_policy_memory_before_science', outcome='retained_history_policy_memory_ordering_improved_coordination'),
                    outcome_message('history-b', ordering='history_policy_memory_before_science', outcome='retained_history_policy_memory_ordering_improved_coordination'),
                ],
                'retain_history_policy_memory_ordering_when_repeatedly_helpful',
                'retained',
            ),
            (
                [
                    outcome_message('control-a', outcome='control_or_regression_outperformed_history_policy', retention='weakened'),
                    outcome_message('control-b', outcome='control_or_regression_outperformed_history_policy', retention='weakened'),
                ],
                'weaken_or_retire_policy_after_repeated_control_or_regression_wins',
                'retired',
            ),
            (
                [outcome_message('waiting', outcome='policy_underpowered_waiting_for_more_evidence', retention='waiting')],
                'keep_policy_underpowered_waiting_for_more_evidence',
                'waiting',
            ),
            (
                [outcome_message('nogain', outcome='record_no_measurable_history_policy_gain', retention='weakened')],
                'record_no_measurable_policy_gain',
                'weakened',
            ),
        ]
        for messages, decision, retention in cases:
            with self.subTest(decision=decision):
                ledger, message = build_once(messages)
                self.assertEqual(decision, ledger['latest']['selected_decision'])
                self.assertEqual(retention, message['body']['retention_decision'])

    def test_outcome_ledger_source_can_drive_retention_without_transcript(self):
        source_messages = [outcome_message('code-a'), outcome_message('code-b')]
        ledger, message = build_once([], outcome_ledger=outcome_ledger_fixture(*source_messages))

        self.assertEqual('retain_code_simulation_ordering_when_repeatedly_helpful', ledger['latest']['selected_decision'])
        self.assertEqual(['policy-outcome-code-a', 'policy-outcome-code-b'], message['body']['source_outcome_ids'])

    def test_boundary_preservation_and_no_overclaiming_from_project_state(self):
        ledger, message = build_once(
            [outcome_message('code-a'), outcome_message('code-b')],
            project_boundary={'third_party_checkpoint_used': True, 'project_owned_checkpoint_verified': False},
        )

        self.assertEqual('preserve_checkpoint_boundary_memory', ledger['latest']['selected_decision'])
        self.assertEqual('repair', message['body']['checkpoint_boundary_state'])
        self.assertFalse(message['body']['project_owned_checkpoint_claimed'])
        self.assertTrue(message['body']['third_party_checkpoint_used'])
        self.assertIn('Candidate-not-causal', message['body']['candidate_not_causal_wording'])

    def test_duplicate_idempotence_and_appended_evidence_update(self):
        waiting = [outcome_message('same-policy', outcome='policy_underpowered_waiting_for_more_evidence', retention='waiting')]
        first_ledger, first_message = build_once(waiting)
        repeat_ledger, repeat_message = build_once(waiting, ledger=first_ledger)
        appended_ledger, appended_message = build_once(
            waiting + [outcome_message('code-a'), outcome_message('code-b')],
            ledger=repeat_ledger,
        )

        self.assertEqual('keep_policy_underpowered_waiting_for_more_evidence', first_ledger['latest']['selected_decision'])
        self.assertIsNotNone(first_message)
        self.assertEqual('summarize_noop', repeat_ledger['latest']['selected_decision'])
        self.assertIsNone(repeat_message)
        self.assertEqual('retain_code_simulation_ordering_when_repeatedly_helpful', appended_ledger['latest']['selected_decision'])
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
            retention_ledger = tmp / 'retention.json'
            outbox = tmp / 'outbox.jsonl'

            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                first = run_science_history_guided_policy_retention_scorecard(
                    theory_memory_file=runtime,
                    runtime_memory_path=str(runtime),
                    transcript_file=transcript,
                    retention_ledger_file=retention_ledger,
                    outbox_file=outbox,
                    memory_data=memory_fixture(),
                    git_status_text=' M tmp/theory-memory.json',
                    git_ignored_text='',
                )
                repeat = run_science_history_guided_policy_retention_scorecard(
                    theory_memory_file=runtime,
                    runtime_memory_path=str(runtime),
                    transcript_file=transcript,
                    retention_ledger_file=retention_ledger,
                    outbox_file=outbox,
                    memory_data=memory_fixture(),
                    git_status_text=' M tmp/theory-memory.json',
                    git_ignored_text='',
                )

        self.assertIn('AI_DIFFERENT_SCIENCE_HISTORY_GUIDED_POLICY_RETENTION', stream.getvalue())
        self.assertTrue(first['science_history_guided_policy_retention_capability'])
        self.assertEqual('retain_code_simulation_ordering_when_repeatedly_helpful', first['selected_decision'])
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
        source = Path(PROJECT_DIR) / 'agent' / 'science_history_guided_policy_retention.py'
        forbidden = ('funfun', 'Language model 2.0', 'Code Module', 'orchastratorrrr')
        imports = [
            line
            for line in source.read_text(encoding='utf-8').splitlines()
            if line.startswith(('import ', 'from '))
        ]
        self.assertFalse([line for line in imports if any(term in line for term in forbidden)])


if __name__ == '__main__':
    unittest.main()
