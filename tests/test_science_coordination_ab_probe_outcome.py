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
from agent.science_coordination_ab_probe import SCIENCE_COORDINATION_AB_PROBE_LEDGER_KIND  # noqa: E402
from agent.science_coordination_ab_probe_outcome import (  # noqa: E402
    SCIENCE_COORDINATION_AB_PROBE_OUTCOME_LEDGER_KIND,
    build_science_coordination_ab_probe_outcome_assessment,
    empty_science_coordination_ab_probe_outcome_ledger,
    load_science_coordination_ab_probe_outcome_ledger,
    read_science_coordination_ab_probe_outcome_transcript,
    validate_science_coordination_ab_probe_outcome_ledger,
    write_science_coordination_ab_probe_outcome_ledger,
    write_science_coordination_ab_probe_outcome_outbox_jsonl,
)
from main import run_science_coordination_ab_probe_outcome_assessor  # noqa: E402


def memory_fixture():
    return {
        'discovery_readiness': {'readiness_score': 0.99, 'status': 'ab_probe_outcome_ready'},
        'abstraction_discovery_evidence': {'transfer_outcome_count': 53},
    }


def probe_message(
    scenario_id='scenario-one',
    *,
    decision='schedule_code_simulation_vs_science_only_probe',
    candidate=None,
    control=None,
    third_party=False,
):
    candidate = candidate or ['code_simulation', 'science_campaign']
    control = control or ['science_campaign_only']
    body = {
        'response_kind': 'science_coordination_ab_probe',
        'science_coordination_ab_probe_id': f'probe-{scenario_id}',
        'source_scorecard_ids': [f'scorecard-{scenario_id}'],
        'source_outcome_ids': [f'outcome-{scenario_id}'],
        'policy_class': 'try_code_simulation_before_science_campaign',
        'selected_probe_decision': decision,
        'selected_action': decision,
        'candidate_sequence': candidate,
        'control_sequence': control,
        'target_campaign_ids': [f'campaign-{scenario_id}'],
        'target_hypothesis_ids': [f'hypothesis:{scenario_id}'],
        'target_simulation_ids': [f'simulation-{scenario_id}'],
        'success_metric': 'candidate yields stronger label-clean campaign outcome than control',
        'stop_condition': 'stop after one matched candidate/control cycle',
        'tested_commits': ['84cbeeb1d85e855ba3ce191d19204bbc828e78e1'],
        'tested_tests': ['tests.test_science_coordination_ab_probe:7'],
        'checkpoint_boundary_state': 'clean',
        'checkpoint_boundary_notes': [],
        'candidate_not_causal': True,
        'third_party_checkpoint_used': third_party,
    }
    return build_module_chat_message(
        sender='ai_different',
        recipient='orchestrator',
        topic='ai_different.science_coordination_ab_probe',
        body=body,
        evidence={'science_coordination_ab_probe_id': f'probe-{scenario_id}', 'selected_probe_decision': decision},
        tags=['science_coordination_ab_probe', decision],
    )


def outcome_evidence(
    scenario_id='scenario-one',
    *,
    selected_outcome='candidate_win',
    gate='candidate_win',
    sender='code_module',
):
    return build_module_chat_message(
        sender=sender,
        recipient='ai_different',
        topic=f'{sender}.science_ab_probe.{gate}',
        body={
            'science_coordination_ab_probe_id': f'probe-{scenario_id}',
            'policy_class': 'try_code_simulation_before_science_campaign',
            'selected_outcome_class': selected_outcome,
            'evidence_gate': gate,
            'campaign_id': f'campaign-{scenario_id}',
            'hypothesis_id': f'hypothesis:{scenario_id}',
            'simulation_id': f'simulation-{scenario_id}',
            'source_commits': [f'{sender}-ab-outcome-commit'],
            'source_tests': [f'{sender}:{gate}:passed'],
        },
        evidence={'science_coordination_ab_probe_id': f'probe-{scenario_id}', 'selected_outcome_class': selected_outcome, 'evidence_gate': gate},
        tags=['science_coordination_ab_probe_outcome', selected_outcome],
    )


def probe_ledger_fixture(
    *,
    scenario_id='scenario-one',
    decision='schedule_code_simulation_vs_science_only_probe',
):
    return {
        'schema_version': 1,
        'ledger_kind': SCIENCE_COORDINATION_AB_PROBE_LEDGER_KIND,
        'processed_message_ids': [],
        'processed_source_hashes': [],
        'planned_probe_keys': [],
        'probe_rows': [],
        'probe_records': [{
            'science_coordination_ab_probe_id': f'probe-{scenario_id}',
            'source_scorecard_ids': [f'scorecard-{scenario_id}'],
            'policy_class': 'try_code_simulation_before_science_campaign',
            'selected_probe_decision': decision,
            'candidate_sequence': ['code_simulation', 'science_campaign'],
            'control_sequence': ['science_campaign_only'],
            'target_campaign_ids': [f'campaign-{scenario_id}'],
            'target_hypothesis_ids': [f'hypothesis:{scenario_id}'],
            'target_simulation_ids': [f'simulation-{scenario_id}'],
            'success_metric': 'candidate yields stronger label-clean campaign outcome than control',
            'stop_condition': 'stop after one matched candidate/control cycle',
            'tested_commits': ['84cbeeb1d85e855ba3ce191d19204bbc828e78e1'],
            'tested_tests': ['tests.test_science_coordination_ab_probe:7'],
            'checkpoint_boundary_state': 'clean',
            'checkpoint_boundary_notes': [],
        }],
        'outgoing_response_ids': [],
        'latest': {},
        'ledger_hash': f'probe-{scenario_id}-hash',
    }


def build_once(messages, *, ledger=None, probe_ledger=None, project_boundary=None):
    updated, message = build_science_coordination_ab_probe_outcome_assessment(
        transcript_messages=messages,
        ab_probe_outcome_ledger=ledger or empty_science_coordination_ab_probe_outcome_ledger(),
        ab_probe_ledger=probe_ledger or {},
        scorecard_ledger={},
        policy_outcome_ledger={},
        history_ledger={},
        cycle_strategy_outcome_ledger={},
        theory_memory_ledger={},
        campaign_ledger={},
        hypothesis_ledger={},
        experiment_ledger={},
        module_chat_ledger={},
        prior_ab_probe_outcome_ledger={},
        runtime_memory_data=memory_fixture(),
        runtime_memory_hash_state={'unchanged': True, 'before': 'same', 'after': 'same'},
        project_owned_boundary=project_boundary or {'third_party_checkpoint_used': False},
        hf_use_status={'hf_validation_used': False},
    )
    return updated, message


class ScienceCoordinationABProbeOutcomeTests(unittest.TestCase):
    def test_outcome_ledger_persistence_load_and_outbox_schema(self):
        ledger, message = build_once([probe_message(), outcome_evidence()])
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = Path(tmpdir) / 'outcome.json'
            outbox = Path(tmpdir) / 'outbox.jsonl'
            write_science_coordination_ab_probe_outcome_ledger(ledger_path, ledger)
            loaded = load_science_coordination_ab_probe_outcome_ledger(ledger_path)
            write_science_coordination_ab_probe_outcome_outbox_jsonl(outbox, message)
            rows = [json.loads(line) for line in outbox.read_text(encoding='utf-8').splitlines()]

        self.assertEqual(SCIENCE_COORDINATION_AB_PROBE_OUTCOME_LEDGER_KIND, loaded['ledger_kind'])
        self.assertEqual('science_coordination_ab_probe_outcome', rows[0]['body']['response_kind'])
        self.assertEqual('candidate_code_simulation_probe_improved_science_campaign', rows[0]['body']['selected_outcome_class'])
        self.assertEqual('retained', rows[0]['body']['retention_decision'])
        self.assertTrue(rows[0]['body']['candidate_not_causal'])
        self.assertIn('no causal proof', rows[0]['body']['no_overclaiming_proof'])
        self.assertFalse(rows[0]['body']['hf_validation_used'])
        with self.assertRaisesRegex(ValueError, 'wrong ledger_kind'):
            validate_science_coordination_ab_probe_outcome_ledger({'ledger_kind': 'wrong'})

    def test_probe_to_outcome_extraction_and_invalid_record(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            transcript = Path(tmpdir) / 'probe-outcome.jsonl'
            transcript.write_text(
                json.dumps(probe_message(), sort_keys=True) + '\n'
                + json.dumps(outcome_evidence(), sort_keys=True) + '\n'
                + '{"sender":"bad"}\n',
                encoding='utf-8',
            )
            parsed = read_science_coordination_ab_probe_outcome_transcript(transcript)
        ledger, message = build_once(parsed['messages'])

        self.assertEqual(2, len(parsed['messages']))
        self.assertEqual(1, len(parsed['invalid_messages']))
        self.assertEqual('candidate_code_simulation_probe_improved_science_campaign', ledger['latest']['selected_outcome_class'])
        self.assertEqual('orchestrator', message['recipient'])

    def test_deterministic_priority_selection(self):
        cases = [
            (
                [probe_message('boundary', third_party=True), outcome_evidence('boundary')],
                'preserve_checkpoint_boundary',
                'retained',
            ),
            (
                [probe_message('code'), outcome_evidence('code', selected_outcome='candidate_win')],
                'candidate_code_simulation_probe_improved_science_campaign',
                'retained',
            ),
            (
                [
                    probe_message('proof', decision='schedule_proof_before_hypothesis_probe', candidate=['formal_proof', 'hypothesis_campaign'], control=['hypothesis_campaign_only']),
                    outcome_evidence('proof', selected_outcome='candidate_win', gate='proof_candidate_win', sender='funfun'),
                ],
                'candidate_proof_before_hypothesis_probe_improved_campaign',
                'retained',
            ),
            (
                [
                    probe_message('language', decision='schedule_language_summary_before_campaign_probe', candidate=['language_summary', 'science_campaign'], control=['science_campaign_no_summary']),
                    outcome_evidence('language', selected_outcome='candidate_win', gate='language_candidate_win', sender='language_model_2'),
                ],
                'candidate_language_summary_probe_improved_campaign',
                'retained',
            ),
            (
                [
                    probe_message('history', decision='schedule_history_policy_memory_probe', candidate=['history_policy_memory', 'science_campaign'], control=['science_campaign_without_history_policy_memory']),
                    outcome_evidence('history', selected_outcome='candidate_win', gate='history_candidate_win', sender='history'),
                ],
                'history_policy_memory_probe_improved_coordination',
                'retained',
            ),
            ([probe_message('control'), outcome_evidence('control', selected_outcome='control_win', gate='control_win')], 'control_outperformed_candidate', 'weakened'),
            ([probe_message('waiting'), outcome_evidence('waiting', selected_outcome='underpowered', gate='underpowered')], 'probe_underpowered_waiting_for_more_evidence', 'waiting'),
            ([probe_message('retired'), outcome_evidence('retired', selected_outcome='regression', gate='regression')], 'retire_probe_for_repeated_no_gain_or_regression', 'retired'),
            ([probe_message('nogain'), outcome_evidence('nogain', selected_outcome='no_gain', gate='no_gain')], 'record_no_measurable_ab_probe_gain', 'weakened'),
        ]
        for messages, outcome, retention in cases:
            with self.subTest(outcome=outcome):
                ledger, message = build_once(messages)
                self.assertEqual(outcome, ledger['latest']['selected_outcome_class'])
                self.assertEqual(retention, message['body']['retention_decision'])

    def test_boundary_preservation_and_no_overclaiming_from_project_state(self):
        ledger, message = build_once(
            [probe_message(), outcome_evidence()],
            project_boundary={'third_party_checkpoint_used': True, 'project_owned_checkpoint_verified': False},
        )

        self.assertEqual('preserve_checkpoint_boundary', ledger['latest']['selected_outcome_class'])
        self.assertEqual('repair', message['body']['checkpoint_boundary_state'])
        self.assertFalse(message['body']['project_owned_checkpoint_claimed'])
        self.assertTrue(message['body']['third_party_checkpoint_used'])
        self.assertIn('Candidate-not-causal', message['body']['candidate_not_causal_wording'])

    def test_duplicate_idempotence_and_appended_evidence_update(self):
        underpowered = [probe_message('same-probe'), outcome_evidence('same-probe', selected_outcome='underpowered', gate='underpowered')]
        first_ledger, first_message = build_once(underpowered)
        repeat_ledger, repeat_message = build_once(underpowered, ledger=first_ledger)
        appended_ledger, appended_message = build_once(
            underpowered + [outcome_evidence('same-probe', selected_outcome='candidate_win', gate='candidate_win')],
            ledger=repeat_ledger,
        )

        self.assertEqual('probe_underpowered_waiting_for_more_evidence', first_ledger['latest']['selected_outcome_class'])
        self.assertIsNotNone(first_message)
        self.assertEqual('summarize_noop', repeat_ledger['latest']['selected_outcome_class'])
        self.assertIsNone(repeat_message)
        self.assertEqual('candidate_code_simulation_probe_improved_science_campaign', appended_ledger['latest']['selected_outcome_class'])
        self.assertEqual('retained', appended_message['body']['retention_decision'])

    def test_cli_status_proof_preserves_runtime_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            runtime = tmp / 'theory-memory.json'
            runtime.write_text(json.dumps(memory_fixture(), sort_keys=True), encoding='utf-8')
            transcript = tmp / 'inbox.jsonl'
            transcript.write_text(
                json.dumps(probe_message(), sort_keys=True) + '\n'
                + json.dumps(outcome_evidence(), sort_keys=True) + '\n',
                encoding='utf-8',
            )
            outcome_ledger = tmp / 'probe-outcome.json'
            outbox = tmp / 'outbox.jsonl'

            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                first = run_science_coordination_ab_probe_outcome_assessor(
                    theory_memory_file=runtime,
                    runtime_memory_path=str(runtime),
                    transcript_file=transcript,
                    ab_probe_ledger_file=tmp / 'missing-probe.json',
                    scorecard_ledger_file=tmp / 'missing-scorecard.json',
                    policy_outcome_ledger_file=tmp / 'missing-policy-outcome.json',
                    history_ledger_file=tmp / 'missing-history.json',
                    cycle_strategy_outcome_ledger_file=tmp / 'missing-cycle-outcome.json',
                    ab_probe_outcome_ledger_file=outcome_ledger,
                    outbox_file=outbox,
                    memory_data=memory_fixture(),
                    git_status_text=' M tmp/theory-memory.json',
                    git_ignored_text='',
                )
                repeat = run_science_coordination_ab_probe_outcome_assessor(
                    theory_memory_file=runtime,
                    runtime_memory_path=str(runtime),
                    transcript_file=transcript,
                    ab_probe_ledger_file=tmp / 'missing-probe.json',
                    scorecard_ledger_file=tmp / 'missing-scorecard.json',
                    policy_outcome_ledger_file=tmp / 'missing-policy-outcome.json',
                    history_ledger_file=tmp / 'missing-history.json',
                    cycle_strategy_outcome_ledger_file=tmp / 'missing-cycle-outcome.json',
                    ab_probe_outcome_ledger_file=outcome_ledger,
                    outbox_file=outbox,
                    memory_data=memory_fixture(),
                    git_status_text=' M tmp/theory-memory.json',
                    git_ignored_text='',
                )

        self.assertIn('AI_DIFFERENT_SCIENCE_COORDINATION_AB_PROBE_OUTCOME', stream.getvalue())
        self.assertTrue(first['science_coordination_ab_probe_outcome_capability'])
        self.assertEqual('candidate_code_simulation_probe_improved_science_campaign', first['selected_outcome'])
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
        source = Path(PROJECT_DIR) / 'agent' / 'science_coordination_ab_probe_outcome.py'
        forbidden = ('funfun', 'Language model 2.0', 'Code Module', 'orchastratorrrr')
        imports = [
            line
            for line in source.read_text(encoding='utf-8').splitlines()
            if line.startswith(('import ', 'from '))
        ]
        self.assertFalse([line for line in imports if any(term in line for term in forbidden)])


if __name__ == '__main__':
    unittest.main()
