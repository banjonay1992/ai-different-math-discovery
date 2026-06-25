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
from agent.science_history_guided_policy_retention import SCIENCE_HISTORY_GUIDED_POLICY_RETENTION_LEDGER_KIND  # noqa: E402
from agent.science_retained_policy_replay import (  # noqa: E402
    SCIENCE_RETAINED_POLICY_REPLAY_LEDGER_KIND,
    build_science_retained_policy_replay_evaluation,
    empty_science_retained_policy_replay_ledger,
    load_science_retained_policy_replay_ledger,
    read_science_retained_policy_replay_transcript,
    validate_science_retained_policy_replay_ledger,
    write_science_retained_policy_replay_ledger,
    write_science_retained_policy_replay_outbox_jsonl,
)
from main import run_science_retained_policy_replay_evaluator  # noqa: E402


def memory_fixture():
    return {
        'discovery_readiness': {'readiness_score': 0.99, 'status': 'retained_policy_replay_ready'},
        'abstraction_discovery_evidence': {'transfer_outcome_count': 73},
    }


def retention_message(
    scenario_id='scenario-one',
    *,
    ordering='code_simulation_before_science',
    decision='retain_code_simulation_ordering_when_repeatedly_helpful',
    retention='retained',
    third_party=False,
):
    body = {
        'response_kind': 'science_history_guided_policy_retention',
        'science_history_guided_policy_retention_id': f'retention-{scenario_id}',
        'source_outcome_ids': [f'policy-outcome-{scenario_id}'],
        'source_policy_ids': [f'policy-{scenario_id}'],
        'source_interaction_history_ids': [f'interaction-history-{scenario_id}'],
        'selected_decision': decision,
        'selected_action': decision,
        'retention_decision': retention,
        'selected_ordering_key': ordering,
        'retained_policy_key': ordering if retention == 'retained' else None,
        'weakened_policy_key': ordering if retention == 'weakened' else None,
        'retired_policy_key': ordering if retention == 'retired' else None,
        'replay_target_ids': [f'campaign-{scenario_id}', f'hypothesis:{scenario_id}', f'simulation-{scenario_id}'],
        'tested_commits': ['7c7f04f0a543b95b42168b51f12144ac126cd0f1'],
        'tested_tests': ['tests.test_science_history_guided_policy_retention:8'],
        'candidate_not_causal': True,
        'checkpoint_boundary_state': 'repair' if third_party or decision == 'preserve_checkpoint_boundary_memory' else 'clean',
        'checkpoint_boundary_notes': ['checkpoint boundary preserved'] if third_party or decision == 'preserve_checkpoint_boundary_memory' else [],
        'third_party_checkpoint_used': third_party,
        'hf_validation_used': False,
    }
    return build_module_chat_message(
        sender='ai_different',
        recipient='history',
        topic='ai_different.science_history_guided_policy_retention',
        body=body,
        evidence={
            'science_history_guided_policy_retention_id': body['science_history_guided_policy_retention_id'],
            'selected_decision': decision,
            'retention_decision': retention,
            'selected_ordering_key': ordering,
            'label_clean': True,
        },
        tags=['science_history_guided_policy_retention', decision],
    )


def replay_signal_message(
    scenario_id='scenario-one',
    *,
    ordering='code_simulation_before_science',
    replay_class='replay_control_or_regression_beats_retained_policy',
    candidate_score=0,
    control_score=2,
):
    body = {
        'response_kind': 'science_retained_policy_replay',
        'science_retained_policy_replay_id': f'prior-replay-{scenario_id}',
        'selected_replay_class': replay_class,
        'retained_policy_key': ordering,
        'control_policy_key': 'science_only_control',
        'candidate_evidence_score': candidate_score,
        'control_evidence_score': control_score,
        'replay_target_ids': [f'campaign-{scenario_id}'],
        'tested_commits': ['prior-replay-commit'],
        'tested_tests': ['prior-replay:test'],
        'candidate_not_causal': True,
        'checkpoint_boundary_state': 'clean',
        'hf_validation_used': False,
    }
    return build_module_chat_message(
        sender='history',
        recipient='ai_different',
        topic='history.science_retained_policy_replay',
        body=body,
        evidence={
            'selected_replay_class': replay_class,
            'retained_policy_key': ordering,
            'candidate_evidence_score': candidate_score,
            'control_evidence_score': control_score,
            'label_clean': True,
        },
        tags=['science_retained_policy_replay', replay_class],
    )


def retention_ledger_fixture(*messages):
    records = [dict(message['body']) for message in messages]
    return {
        'schema_version': 1,
        'ledger_kind': SCIENCE_HISTORY_GUIDED_POLICY_RETENTION_LEDGER_KIND,
        'processed_message_ids': [],
        'processed_source_hashes': [],
        'scored_policy_keys': [],
        'retention_records': records,
        'retention_rows': [],
        'outgoing_response_ids': [],
        'latest': {},
        'ledger_hash': 'retention-fixture-hash',
    }


def build_once(messages, *, ledger=None, retention_ledger=None, project_boundary=None):
    updated, message = build_science_retained_policy_replay_evaluation(
        transcript_messages=messages,
        replay_ledger=ledger or empty_science_retained_policy_replay_ledger(),
        retention_ledger=retention_ledger or {},
        policy_outcome_ledger={},
        history_guided_policy_ledger={},
        interaction_history_ledger={},
        theory_memory_ledger={},
        campaign_ledger={},
        hypothesis_ledger={},
        experiment_ledger={},
        module_chat_ledger={},
        prior_replay_ledger={},
        runtime_memory_data=memory_fixture(),
        runtime_memory_hash_state={'unchanged': True, 'before': 'same', 'after': 'same'},
        project_owned_boundary=project_boundary or {'third_party_checkpoint_used': False},
        hf_use_status={'hf_validation_used': False},
    )
    return updated, message


class ScienceRetainedPolicyReplayTests(unittest.TestCase):
    def test_replay_ledger_persistence_load_and_outbox_schema(self):
        ledger, message = build_once([retention_message()])
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = Path(tmpdir) / 'replay.json'
            outbox = Path(tmpdir) / 'outbox.jsonl'
            write_science_retained_policy_replay_ledger(ledger_path, ledger)
            loaded = load_science_retained_policy_replay_ledger(ledger_path)
            write_science_retained_policy_replay_outbox_jsonl(outbox, message)
            rows = [json.loads(line) for line in outbox.read_text(encoding='utf-8').splitlines()]

        self.assertEqual(SCIENCE_RETAINED_POLICY_REPLAY_LEDGER_KIND, loaded['ledger_kind'])
        self.assertEqual('science_retained_policy_replay', rows[0]['body']['response_kind'])
        self.assertEqual('replay_retained_code_simulation_ordering_beats_control', rows[0]['body']['selected_replay_class'])
        self.assertEqual('code_simulation_before_science', rows[0]['body']['retained_policy_key'])
        self.assertTrue(rows[0]['evidence']['replay_candidate_benefit'])
        self.assertIn('Replay_candidate_benefit', rows[0]['body']['candidate_not_causal_wording'])
        self.assertIn('not causal proof', rows[0]['body']['no_overclaiming_proof'])
        self.assertFalse(rows[0]['body']['hf_validation_used'])
        with self.assertRaisesRegex(ValueError, 'wrong ledger_kind'):
            validate_science_retained_policy_replay_ledger({'ledger_kind': 'wrong'})

    def test_retention_to_replay_extraction_and_invalid_record(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            transcript = Path(tmpdir) / 'replay.jsonl'
            transcript.write_text(
                json.dumps(retention_message(), sort_keys=True) + '\n'
                + '{"sender":"bad"}\n',
                encoding='utf-8',
            )
            parsed = read_science_retained_policy_replay_transcript(transcript)
        ledger, message = build_once(parsed['messages'])

        self.assertEqual(1, len(parsed['messages']))
        self.assertEqual(1, len(parsed['invalid_messages']))
        self.assertEqual('replay_retained_code_simulation_ordering_beats_control', ledger['latest']['selected_replay_class'])
        self.assertEqual('history', message['recipient'])

    def test_deterministic_priority_selection(self):
        cases = [
            (
                [retention_message('boundary', decision='preserve_checkpoint_boundary_memory', third_party=True)],
                'preserve_checkpoint_boundary_replay',
            ),
            (
                [retention_message('code')],
                'replay_retained_code_simulation_ordering_beats_control',
            ),
            (
                [retention_message('proof', ordering='proof_before_hypothesis', decision='retain_proof_before_hypothesis_ordering_when_repeatedly_helpful')],
                'replay_retained_proof_before_hypothesis_ordering_beats_control',
            ),
            (
                [retention_message('language', ordering='language_summary_before_science', decision='retain_language_summary_ordering_when_repeatedly_helpful')],
                'replay_retained_language_summary_ordering_beats_control',
            ),
            (
                [retention_message('history', ordering='history_policy_memory_before_science', decision='retain_history_policy_memory_ordering_when_repeatedly_helpful')],
                'replay_retained_history_memory_ordering_beats_control',
            ),
            (
                [replay_signal_message('control')],
                'replay_control_or_regression_beats_retained_policy',
            ),
            (
                [replay_signal_message('waiting', replay_class='replay_underpowered_waiting_for_more_evidence', candidate_score=0, control_score=0)],
                'replay_underpowered_waiting_for_more_evidence',
            ),
            (
                [replay_signal_message('nogain', replay_class='replay_no_measurable_retained_policy_gain', candidate_score=1, control_score=1)],
                'replay_no_measurable_retained_policy_gain',
            ),
        ]
        for messages, replay_class in cases:
            with self.subTest(replay_class=replay_class):
                ledger, message = build_once(messages)
                self.assertEqual(replay_class, ledger['latest']['selected_replay_class'])
                self.assertEqual(replay_class, message['body']['selected_action'])

    def test_retention_ledger_source_can_drive_replay_without_transcript(self):
        ledger, message = build_once([], retention_ledger=retention_ledger_fixture(retention_message()))

        self.assertEqual('replay_retained_code_simulation_ordering_beats_control', ledger['latest']['selected_replay_class'])
        self.assertEqual(['retention-scenario-one'], message['body']['source_retention_ids'])

    def test_boundary_preservation_and_no_overclaiming_from_project_state(self):
        ledger, message = build_once(
            [retention_message()],
            project_boundary={'third_party_checkpoint_used': True, 'project_owned_checkpoint_verified': False},
        )

        self.assertEqual('preserve_checkpoint_boundary_replay', ledger['latest']['selected_replay_class'])
        self.assertEqual('repair', message['body']['checkpoint_boundary_state'])
        self.assertFalse(message['body']['project_owned_checkpoint_claimed'])
        self.assertTrue(message['body']['third_party_checkpoint_used'])
        self.assertIn('not causal proof', message['body']['no_overclaiming_proof'])

    def test_duplicate_idempotence_and_appended_evidence_update(self):
        waiting = [replay_signal_message('same-policy', replay_class='replay_underpowered_waiting_for_more_evidence', candidate_score=0, control_score=0)]
        first_ledger, first_message = build_once(waiting)
        repeat_ledger, repeat_message = build_once(waiting, ledger=first_ledger)
        appended_ledger, appended_message = build_once(waiting + [retention_message('code')], ledger=repeat_ledger)

        self.assertEqual('replay_underpowered_waiting_for_more_evidence', first_ledger['latest']['selected_replay_class'])
        self.assertIsNotNone(first_message)
        self.assertEqual('summarize_noop', repeat_ledger['latest']['selected_replay_class'])
        self.assertIsNone(repeat_message)
        self.assertEqual('replay_retained_code_simulation_ordering_beats_control', appended_ledger['latest']['selected_replay_class'])
        self.assertTrue(appended_message['evidence']['replay_candidate_benefit'])

    def test_cli_status_proof_preserves_runtime_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            runtime = tmp / 'theory-memory.json'
            runtime.write_text(json.dumps(memory_fixture(), sort_keys=True), encoding='utf-8')
            transcript = tmp / 'inbox.jsonl'
            transcript.write_text(json.dumps(retention_message(), sort_keys=True) + '\n', encoding='utf-8')
            replay_ledger = tmp / 'replay.json'
            outbox = tmp / 'outbox.jsonl'

            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                first = run_science_retained_policy_replay_evaluator(
                    theory_memory_file=runtime,
                    runtime_memory_path=str(runtime),
                    transcript_file=transcript,
                    replay_ledger_file=replay_ledger,
                    outbox_file=outbox,
                    memory_data=memory_fixture(),
                    git_status_text=' M tmp/theory-memory.json',
                    git_ignored_text='',
                )
                repeat = run_science_retained_policy_replay_evaluator(
                    theory_memory_file=runtime,
                    runtime_memory_path=str(runtime),
                    transcript_file=transcript,
                    replay_ledger_file=replay_ledger,
                    outbox_file=outbox,
                    memory_data=memory_fixture(),
                    git_status_text=' M tmp/theory-memory.json',
                    git_ignored_text='',
                )

        self.assertIn('AI_DIFFERENT_SCIENCE_RETAINED_POLICY_REPLAY', stream.getvalue())
        self.assertTrue(first['science_retained_policy_replay_capability'])
        self.assertEqual('replay_retained_code_simulation_ordering_beats_control', first['selected_replay_class'])
        self.assertEqual(1, first['outbox_count'])
        self.assertFalse(first['runtime_memory_mutated'])
        self.assertEqual([], first['label_leaks'])
        self.assertTrue(first['no_sibling_imports'])
        self.assertFalse(first['project_owned_checkpoint_claimed'])
        self.assertFalse(first['hf_validation_used'])
        self.assertEqual('summarize_noop', repeat['selected_replay_class'])
        self.assertEqual(0, repeat['outbox_count'])
        self.assertFalse(repeat['runtime_memory_mutated'])

    def test_no_sibling_imports(self):
        source = Path(PROJECT_DIR) / 'agent' / 'science_retained_policy_replay.py'
        forbidden = ('funfun', 'Language model 2.0', 'Code Module', 'orchastratorrrr')
        imports = [
            line
            for line in source.read_text(encoding='utf-8').splitlines()
            if line.startswith(('import ', 'from '))
        ]
        self.assertFalse([line for line in imports if any(term in line for term in forbidden)])


if __name__ == '__main__':
    unittest.main()
