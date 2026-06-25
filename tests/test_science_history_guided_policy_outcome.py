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
from agent.science_history_guided_policy import SCIENCE_HISTORY_GUIDED_POLICY_LEDGER_KIND  # noqa: E402
from agent.science_history_guided_policy_outcome import (  # noqa: E402
    SCIENCE_HISTORY_GUIDED_POLICY_OUTCOME_LEDGER_KIND,
    build_science_history_guided_policy_outcome_assessment,
    empty_science_history_guided_policy_outcome_ledger,
    load_science_history_guided_policy_outcome_ledger,
    read_science_history_guided_policy_outcome_transcript,
    validate_science_history_guided_policy_outcome_ledger,
    write_science_history_guided_policy_outcome_ledger,
    write_science_history_guided_policy_outcome_outbox_jsonl,
)
from main import run_science_history_guided_policy_outcome_assessor  # noqa: E402


def memory_fixture():
    return {
        'discovery_readiness': {'readiness_score': 0.99, 'status': 'history_guided_policy_outcome_ready'},
        'abstraction_discovery_evidence': {'transfer_outcome_count': 67},
    }


def policy_message(
    scenario_id='scenario-one',
    *,
    ordering='code_simulation_before_science',
    decision='apply_retained_code_simulation_before_science_ordering',
    third_party=False,
):
    body = {
        'response_kind': 'science_history_guided_policy',
        'science_history_guided_policy_id': f'policy-{scenario_id}',
        'source_interaction_history_ids': [f'interaction-history-{scenario_id}'],
        'selected_policy_decision': decision,
        'selected_action': decision,
        'selected_ordering_key': ordering,
        'selected_ordering': ordering,
        'target_campaign_ids': [f'campaign-{scenario_id}'],
        'target_hypothesis_ids': [f'hypothesis:{scenario_id}'],
        'target_simulation_ids': [f'simulation-{scenario_id}'],
        'required_evidence_gate': 'candidate_win',
        'stop_condition': 'stop after one bounded campaign cycle',
        'tested_commits': ['9e22b3ed2446400c55f250c4fc91d007ffcfa9dd'],
        'tested_tests': ['tests.test_science_history_guided_policy:7'],
        'candidate_not_causal': True,
        'checkpoint_boundary_state': 'repair' if decision == 'preserve_checkpoint_boundary' else 'clean',
        'checkpoint_boundary_notes': ['boundary preserved'] if decision == 'preserve_checkpoint_boundary' else [],
        'third_party_checkpoint_used': third_party,
        'hf_validation_used': False,
    }
    return build_module_chat_message(
        sender='ai_different',
        recipient='orchestrator',
        topic='ai_different.science_history_guided_policy',
        body=body,
        evidence={
            'science_history_guided_policy_id': body['science_history_guided_policy_id'],
            'selected_policy_decision': decision,
            'selected_ordering_key': ordering,
        },
        tags=['science_history_guided_policy', decision],
    )


def evidence_message(
    scenario_id='scenario-one',
    *,
    outcome='candidate_win',
    gate='candidate_win',
    sender='code_module',
):
    return build_module_chat_message(
        sender=sender,
        recipient='ai_different',
        topic=f'{sender}.science_history_guided_policy.{gate}',
        body={
            'science_history_guided_policy_id': f'policy-{scenario_id}',
            'selected_outcome_class': outcome,
            'evidence_gate': gate,
            'campaign_id': f'campaign-{scenario_id}',
            'hypothesis_id': f'hypothesis:{scenario_id}',
            'simulation_id': f'simulation-{scenario_id}',
            'source_commits': [f'{sender}-history-policy-outcome'],
            'source_tests': [f'{sender}:{gate}:passed'],
        },
        evidence={'science_history_guided_policy_id': f'policy-{scenario_id}', 'selected_outcome_class': outcome, 'evidence_gate': gate},
        tags=['science_history_guided_policy_outcome', outcome],
    )


def policy_ledger_fixture(*, scenario_id='scenario-one', ordering='code_simulation_before_science'):
    return {
        'schema_version': 1,
        'ledger_kind': SCIENCE_HISTORY_GUIDED_POLICY_LEDGER_KIND,
        'processed_message_ids': [],
        'processed_source_hashes': [],
        'planned_policy_keys': [],
        'policy_rows': [],
        'policy_records': [{
            'science_history_guided_policy_id': f'policy-{scenario_id}',
            'source_interaction_history_ids': [f'interaction-history-{scenario_id}'],
            'selected_ordering_key': ordering,
            'selected_policy_decision': 'apply_retained_code_simulation_before_science_ordering',
            'target_campaign_ids': [f'campaign-{scenario_id}'],
            'target_hypothesis_ids': [f'hypothesis:{scenario_id}'],
            'target_simulation_ids': [f'simulation-{scenario_id}'],
            'tested_commits': ['9e22b3ed2446400c55f250c4fc91d007ffcfa9dd'],
            'tested_tests': ['tests.test_science_history_guided_policy:7'],
            'checkpoint_boundary_state': 'clean',
        }],
        'outgoing_response_ids': [],
        'latest': {},
        'ledger_hash': f'policy-{scenario_id}-hash',
    }


def build_once(messages, *, ledger=None, policy_ledger=None, project_boundary=None):
    updated, message = build_science_history_guided_policy_outcome_assessment(
        transcript_messages=messages,
        policy_outcome_ledger=ledger or empty_science_history_guided_policy_outcome_ledger(),
        history_guided_policy_ledger=policy_ledger or {},
        interaction_history_ledger={},
        ab_probe_outcome_ledger={},
        ab_probe_ledger={},
        theory_memory_ledger={},
        campaign_ledger={},
        hypothesis_ledger={},
        experiment_ledger={},
        module_chat_ledger={},
        prior_policy_outcome_ledger={},
        runtime_memory_data=memory_fixture(),
        runtime_memory_hash_state={'unchanged': True, 'before': 'same', 'after': 'same'},
        project_owned_boundary=project_boundary or {'third_party_checkpoint_used': False},
        hf_use_status={'hf_validation_used': False},
    )
    return updated, message


class ScienceHistoryGuidedPolicyOutcomeTests(unittest.TestCase):
    def test_outcome_ledger_persistence_load_and_outbox_schema(self):
        ledger, message = build_once([policy_message(), evidence_message()])
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = Path(tmpdir) / 'outcome.json'
            outbox = Path(tmpdir) / 'outbox.jsonl'
            write_science_history_guided_policy_outcome_ledger(ledger_path, ledger)
            loaded = load_science_history_guided_policy_outcome_ledger(ledger_path)
            write_science_history_guided_policy_outcome_outbox_jsonl(outbox, message)
            rows = [json.loads(line) for line in outbox.read_text(encoding='utf-8').splitlines()]

        self.assertEqual(SCIENCE_HISTORY_GUIDED_POLICY_OUTCOME_LEDGER_KIND, loaded['ledger_kind'])
        self.assertEqual('science_history_guided_policy_outcome', rows[0]['body']['response_kind'])
        self.assertEqual('retained_code_simulation_ordering_improved_science_campaign', rows[0]['body']['selected_outcome_class'])
        self.assertEqual('retained', rows[0]['body']['retention_decision'])
        self.assertTrue(rows[0]['body']['candidate_not_causal'])
        self.assertIn('no causal proof', rows[0]['body']['no_overclaiming_proof'])
        self.assertFalse(rows[0]['body']['hf_validation_used'])
        with self.assertRaisesRegex(ValueError, 'wrong ledger_kind'):
            validate_science_history_guided_policy_outcome_ledger({'ledger_kind': 'wrong'})

    def test_policy_to_outcome_extraction_and_invalid_record(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            transcript = Path(tmpdir) / 'policy-outcome.jsonl'
            transcript.write_text(
                json.dumps(policy_message(), sort_keys=True) + '\n'
                + json.dumps(evidence_message(), sort_keys=True) + '\n'
                + '{"sender":"bad"}\n',
                encoding='utf-8',
            )
            parsed = read_science_history_guided_policy_outcome_transcript(transcript)
        ledger, message = build_once(parsed['messages'])

        self.assertEqual(2, len(parsed['messages']))
        self.assertEqual(1, len(parsed['invalid_messages']))
        self.assertEqual('retained_code_simulation_ordering_improved_science_campaign', ledger['latest']['selected_outcome_class'])
        self.assertEqual('orchestrator', message['recipient'])

    def test_deterministic_priority_selection(self):
        cases = [
            (
                [policy_message('boundary', decision='preserve_checkpoint_boundary', third_party=True), evidence_message('boundary')],
                'preserve_checkpoint_boundary',
                'retained',
            ),
            (
                [policy_message('code'), evidence_message('code')],
                'retained_code_simulation_ordering_improved_science_campaign',
                'retained',
            ),
            (
                [policy_message('proof', ordering='proof_before_hypothesis', decision='apply_retained_proof_before_hypothesis_ordering'), evidence_message('proof', sender='funfun')],
                'retained_proof_before_hypothesis_ordering_improved_campaign',
                'retained',
            ),
            (
                [policy_message('language', ordering='language_summary_before_science', decision='apply_retained_language_summary_before_science_ordering'), evidence_message('language', sender='language_model_2')],
                'retained_language_summary_ordering_improved_campaign',
                'retained',
            ),
            (
                [policy_message('history', ordering='history_policy_memory_before_science', decision='apply_retained_history_policy_memory_ordering'), evidence_message('history', sender='history')],
                'retained_history_policy_memory_ordering_improved_coordination',
                'retained',
            ),
            (
                [policy_message('control'), evidence_message('control', outcome='control_win', gate='control_win')],
                'control_or_regression_outperformed_history_policy',
                'weakened',
            ),
            (
                [policy_message('waiting'), evidence_message('waiting', outcome='underpowered', gate='underpowered')],
                'policy_underpowered_waiting_for_more_evidence',
                'waiting',
            ),
            (
                [policy_message('retired'), evidence_message('retired', outcome='no_gain', gate='no_gain'), evidence_message('retired', outcome='no_gain', gate='no_gain_2')],
                'retire_or_weaken_history_policy_after_repeated_no_gain',
                'retired',
            ),
            (
                [policy_message('nogain'), evidence_message('nogain', outcome='no_gain', gate='no_gain')],
                'record_no_measurable_history_policy_gain',
                'weakened',
            ),
        ]
        for messages, outcome, retention in cases:
            with self.subTest(outcome=outcome):
                ledger, message = build_once(messages)
                self.assertEqual(outcome, ledger['latest']['selected_outcome_class'])
                self.assertEqual(retention, message['body']['retention_decision'])

    def test_boundary_preservation_and_no_overclaiming_from_project_state(self):
        ledger, message = build_once(
            [policy_message(), evidence_message()],
            project_boundary={'third_party_checkpoint_used': True, 'project_owned_checkpoint_verified': False},
        )

        self.assertEqual('preserve_checkpoint_boundary', ledger['latest']['selected_outcome_class'])
        self.assertEqual('repair', message['body']['checkpoint_boundary_state'])
        self.assertFalse(message['body']['project_owned_checkpoint_claimed'])
        self.assertTrue(message['body']['third_party_checkpoint_used'])
        self.assertIn('Candidate-not-causal', message['body']['candidate_not_causal_wording'])

    def test_duplicate_idempotence_and_appended_evidence_update(self):
        waiting = [policy_message('same-policy'), evidence_message('same-policy', outcome='underpowered', gate='underpowered')]
        first_ledger, first_message = build_once(waiting)
        repeat_ledger, repeat_message = build_once(waiting, ledger=first_ledger)
        appended_ledger, appended_message = build_once(
            waiting + [evidence_message('same-policy')],
            ledger=repeat_ledger,
        )

        self.assertEqual('policy_underpowered_waiting_for_more_evidence', first_ledger['latest']['selected_outcome_class'])
        self.assertIsNotNone(first_message)
        self.assertEqual('summarize_noop', repeat_ledger['latest']['selected_outcome_class'])
        self.assertIsNone(repeat_message)
        self.assertEqual('retained_code_simulation_ordering_improved_science_campaign', appended_ledger['latest']['selected_outcome_class'])
        self.assertEqual('retained', appended_message['body']['retention_decision'])

    def test_cli_status_proof_preserves_runtime_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            runtime = tmp / 'theory-memory.json'
            runtime.write_text(json.dumps(memory_fixture(), sort_keys=True), encoding='utf-8')
            transcript = tmp / 'inbox.jsonl'
            transcript.write_text(
                json.dumps(policy_message(), sort_keys=True) + '\n'
                + json.dumps(evidence_message(), sort_keys=True) + '\n',
                encoding='utf-8',
            )
            outcome_ledger = tmp / 'policy-outcome.json'
            outbox = tmp / 'outbox.jsonl'

            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                first = run_science_history_guided_policy_outcome_assessor(
                    theory_memory_file=runtime,
                    runtime_memory_path=str(runtime),
                    transcript_file=transcript,
                    history_guided_policy_outcome_ledger_file=outcome_ledger,
                    outbox_file=outbox,
                    memory_data=memory_fixture(),
                    git_status_text=' M tmp/theory-memory.json',
                    git_ignored_text='',
                )
                repeat = run_science_history_guided_policy_outcome_assessor(
                    theory_memory_file=runtime,
                    runtime_memory_path=str(runtime),
                    transcript_file=transcript,
                    history_guided_policy_outcome_ledger_file=outcome_ledger,
                    outbox_file=outbox,
                    memory_data=memory_fixture(),
                    git_status_text=' M tmp/theory-memory.json',
                    git_ignored_text='',
                )

        self.assertIn('AI_DIFFERENT_SCIENCE_HISTORY_GUIDED_POLICY_OUTCOME', stream.getvalue())
        self.assertTrue(first['science_history_guided_policy_outcome_capability'])
        self.assertEqual('retained_code_simulation_ordering_improved_science_campaign', first['selected_outcome'])
        self.assertEqual('retained', first['retention_decision'])
        self.assertEqual(1, first['outbox_count'])
        self.assertFalse(first['runtime_memory_mutated'])
        self.assertEqual([], first['label_leaks'])
        self.assertTrue(first['no_sibling_imports'])
        self.assertFalse(first['project_owned_checkpoint_claimed'])
        self.assertFalse(first['hf_validation_used'])
        self.assertEqual('summarize_noop', repeat['selected_outcome'])
        self.assertEqual(0, repeat['outbox_count'])
        self.assertFalse(repeat['runtime_memory_mutated'])

    def test_no_sibling_imports(self):
        source = Path(PROJECT_DIR) / 'agent' / 'science_history_guided_policy_outcome.py'
        forbidden = ('funfun', 'Language model 2.0', 'Code Module', 'orchastratorrrr')
        imports = [
            line
            for line in source.read_text(encoding='utf-8').splitlines()
            if line.startswith(('import ', 'from '))
        ]
        self.assertFalse([line for line in imports if any(term in line for term in forbidden)])


if __name__ == '__main__':
    unittest.main()
