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
from agent.science_coordination_policy import SCIENCE_COORDINATION_POLICY_LEDGER_KIND  # noqa: E402
from agent.science_coordination_policy_outcome import (  # noqa: E402
    SCIENCE_COORDINATION_POLICY_OUTCOME_LEDGER_KIND,
    build_science_coordination_policy_outcome_assessment,
    empty_science_coordination_policy_outcome_ledger,
    load_science_coordination_policy_outcome_ledger,
    read_science_coordination_policy_outcome_transcript,
    validate_science_coordination_policy_outcome_ledger,
    write_science_coordination_policy_outcome_ledger,
    write_science_coordination_policy_outcome_outbox_jsonl,
)
from main import run_science_coordination_policy_outcome_assessor  # noqa: E402


def memory_fixture():
    return {
        'discovery_readiness': {'readiness_score': 0.99, 'status': 'policy_outcome_ready'},
        'abstraction_discovery_evidence': {'transfer_outcome_count': 37},
    }


def policy_message(
    scenario_id='scenario-one',
    *,
    selected_policy='try_code_simulation_before_science_campaign',
    recipient='code_module',
    leak=False,
    third_party=False,
):
    body = {
        'response_kind': 'science_coordination_policy',
        'science_coordination_policy_id': f'policy-{scenario_id}',
        'scenario_id': scenario_id,
        'hypothesis_id': f'hypothesis:{scenario_id}',
        'campaign_id': f'campaign-{scenario_id}',
        'family_id': f'family-{scenario_id}',
        'selected_policy': selected_policy,
        'selected_action': selected_policy,
        'selected_recipient': recipient,
        'source_history_ids': [f'history-{scenario_id}'],
        'source_commits': ['3fb611988f70835c9180f0463331a66a33951164'],
        'source_tests': ['tests.test_science_coordination_policy:7'],
        'checkpoint_boundary_state': 'clean',
        'checkpoint_boundary_notes': [],
        'candidate_not_causal': True,
        'third_party_checkpoint_used': third_party,
    }
    if leak:
        body['checkpoint_boundary_notes'] = ['gravity label leaked']
    return build_module_chat_message(
        sender='ai_different',
        recipient=recipient,
        topic='ai_different.science_coordination_policy',
        body=body,
        evidence={'scenario_id': scenario_id, 'science_coordination_policy_id': f'policy-{scenario_id}', 'selected_policy': selected_policy},
        tags=['science_coordination_policy', selected_policy],
    )


def evidence_message(
    scenario_id='scenario-one',
    *,
    sender='code_module',
    selected_outcome='code_simulation_or_primitive_capability_help_received',
    gate='primitive_simulation_help',
    status='received',
):
    return build_module_chat_message(
        sender=sender,
        recipient='ai_different',
        topic=f'{sender}.{gate}.{status}',
        body={
            'scenario_id': scenario_id,
            'hypothesis_id': f'hypothesis:{scenario_id}',
            'campaign_id': f'campaign-{scenario_id}',
            'family_id': f'family-{scenario_id}',
            'selected_outcome': selected_outcome,
            'evidence_gate': gate,
            'status': status,
            'source_commits': [f'{sender}-evidence-commit'],
            'source_tests': [f'{sender}:{gate}:passed'],
        },
        evidence={'scenario_id': scenario_id, 'selected_outcome': selected_outcome, 'evidence_gate': gate, 'status': status},
        tags=['science_coordination_policy_outcome', gate, status],
    )


def policy_ledger_fixture(
    *,
    scenario_id='scenario-one',
    selected_policy='try_code_simulation_before_science_campaign',
):
    return {
        'schema_version': 1,
        'ledger_kind': SCIENCE_COORDINATION_POLICY_LEDGER_KIND,
        'processed_message_ids': [],
        'processed_source_hashes': [],
        'planned_policy_keys': [],
        'policy_rows': [],
        'policy_records': [{
            'science_coordination_policy_id': f'policy-{scenario_id}',
            'scenario_id': scenario_id,
            'hypothesis_id': f'hypothesis:{scenario_id}',
            'campaign_id': f'campaign-{scenario_id}',
            'family_id': f'family-{scenario_id}',
            'source_policy_ids': [f'policy-{scenario_id}'],
            'source_history_ids': [f'history-{scenario_id}'],
            'selected_policy': selected_policy,
            'selected_action': selected_policy,
            'selected_recipient': 'code_module',
            'source_commits': ['3fb611988f70835c9180f0463331a66a33951164'],
            'source_tests': ['tests.test_science_coordination_policy:7'],
            'checkpoint_boundary_state': 'clean',
            'checkpoint_boundary_notes': [],
        }],
        'outgoing_response_ids': [],
        'latest': {},
        'ledger_hash': f'policy-{scenario_id}-hash',
    }


def build_once(messages, *, ledger=None, policy_ledger=None, project_boundary=None):
    updated, message = build_science_coordination_policy_outcome_assessment(
        transcript_messages=messages,
        policy_outcome_ledger=ledger or empty_science_coordination_policy_outcome_ledger(),
        policy_ledger=policy_ledger or {},
        history_ledger={},
        cycle_strategy_outcome_ledger={},
        cycle_strategy_ledger={},
        strategy_outcome_ledger={},
        frontier_outcome_ledger={},
        theory_memory_ledger={},
        campaign_cycle_memory_ledger={},
        hypothesis_ledger={},
        campaign_ledger={},
        experiment_ledger={},
        module_chat_ledger={},
        prior_policy_outcome_ledger={},
        runtime_memory_data=memory_fixture(),
        runtime_memory_hash_state={'unchanged': True, 'before': 'same', 'after': 'same'},
        project_owned_boundary=project_boundary or {'third_party_checkpoint_used': False},
    )
    return updated, message


class ScienceCoordinationPolicyOutcomeTests(unittest.TestCase):
    def test_outcome_ledger_persistence_load_and_outbox_schema(self):
        ledger, message = build_once([policy_message(), evidence_message()])
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = Path(tmpdir) / 'outcome.json'
            outbox = Path(tmpdir) / 'outbox.jsonl'
            write_science_coordination_policy_outcome_ledger(ledger_path, ledger)
            loaded = load_science_coordination_policy_outcome_ledger(ledger_path)
            write_science_coordination_policy_outcome_outbox_jsonl(outbox, message)
            rows = [json.loads(line) for line in outbox.read_text(encoding='utf-8').splitlines()]

        self.assertEqual(SCIENCE_COORDINATION_POLICY_OUTCOME_LEDGER_KIND, loaded['ledger_kind'])
        self.assertEqual('science_coordination_policy_outcome', rows[0]['body']['response_kind'])
        self.assertEqual('code_simulation_policy_improved_science_campaign', rows[0]['body']['selected_outcome'])
        self.assertEqual('retained', rows[0]['body']['policy_retention_state'])
        self.assertTrue(rows[0]['body']['candidate_not_causal'])
        self.assertIn('no causal proof', rows[0]['body']['no_overclaiming_proof'])
        with self.assertRaisesRegex(ValueError, 'wrong ledger_kind'):
            validate_science_coordination_policy_outcome_ledger({'ledger_kind': 'wrong'})

    def test_policy_to_outcome_extraction_and_invalid_record(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            transcript = Path(tmpdir) / 'policy-outcome.jsonl'
            transcript.write_text(
                json.dumps(policy_message(), sort_keys=True) + '\n'
                + json.dumps(evidence_message(), sort_keys=True) + '\n'
                + '{"sender":"bad"}\n',
                encoding='utf-8',
            )
            parsed = read_science_coordination_policy_outcome_transcript(transcript)
        ledger, message = build_once(parsed['messages'])

        self.assertEqual(2, len(parsed['messages']))
        self.assertEqual(1, len(parsed['invalid_messages']))
        self.assertEqual('code_simulation_policy_improved_science_campaign', ledger['latest']['selected_outcome'])
        self.assertEqual('code_module', message['recipient'])

    def test_deterministic_priority_selection(self):
        cases = [
            (
                [policy_message('boundary', third_party=True), evidence_message('boundary')],
                'preserve_checkpoint_boundary',
                'retained',
            ),
            (
                [policy_message('code'), evidence_message('code', sender='code_module', selected_outcome='code_simulation_or_primitive_capability_help_received')],
                'code_simulation_policy_improved_science_campaign',
                'retained',
            ),
            (
                [
                    policy_message('funfun', selected_policy='try_funfun_formalization_before_science_campaign', recipient='funfun'),
                    evidence_message('funfun', sender='funfun', selected_outcome='funfun_formal_or_proof_help_received', gate='formal_proof_certificate'),
                ],
                'funfun_formalization_policy_improved_science_campaign',
                'retained',
            ),
            (
                [
                    policy_message('language', selected_policy='try_language_terminology_before_science_campaign', recipient='language_model_2'),
                    evidence_message('language', sender='language_model_2', selected_outcome='language_terminology_or_protocol_help_received', gate='protocol_clarification'),
                ],
                'language_terminology_policy_improved_science_campaign',
                'retained',
            ),
            (
                [
                    policy_message('repair', selected_policy='try_theory_repair_before_next_campaign', recipient='orchestrator'),
                    evidence_message('repair', sender='ai_different', selected_outcome='theory_repair_cycle_reopened', gate='repair_reopened'),
                ],
                'theory_repair_policy_improved_next_campaign',
                'retained',
            ),
            (
                [
                    policy_message('schedule', selected_policy='schedule_hypothesis_campaign_cycle', recipient='orchestrator'),
                    evidence_message('schedule', sender='ai_different', selected_outcome='next_frontier_hypothesis_campaign_scheduled', gate='scheduled'),
                ],
                'hypothesis_campaign_cycle_scheduled_or_closed',
                'retained',
            ),
            ([policy_message('noop', selected_policy='avoid_repeated_noop_sequence', recipient='orchestrator')], 'repeated_noop_policy_retired', 'retired'),
            ([policy_message('waiting', selected_policy='gather_more_sibling_evidence_before_policy_change', recipient='broadcast')], 'policy_waiting_for_sibling_evidence', 'waiting'),
            ([policy_message('nogain', selected_policy='record_no_measurable_policy_gain', recipient='orchestrator')], 'no_measurable_policy_gain', 'weakened'),
        ]
        for messages, selected_outcome, retention in cases:
            with self.subTest(outcome=selected_outcome):
                ledger, message = build_once(messages)
                self.assertEqual(selected_outcome, ledger['latest']['selected_outcome'])
                self.assertEqual(retention, ledger['latest']['policy_retention_state'])
                self.assertIsNotNone(message)

    def test_boundary_preservation_and_no_overclaiming_from_project_state(self):
        ledger, message = build_once(
            [policy_message(), evidence_message()],
            project_boundary={'third_party_checkpoint_used': True, 'project_owned_checkpoint_verified': False},
        )

        self.assertEqual('preserve_checkpoint_boundary', ledger['latest']['selected_outcome'])
        self.assertEqual('repair', message['body']['checkpoint_boundary_state'])
        self.assertFalse(message['body']['project_owned_checkpoint_claimed'])
        self.assertTrue(message['body']['third_party_checkpoint_used'])
        self.assertIn('Candidate-not-causal', message['body']['candidate_not_causal_wording'])

    def test_duplicate_idempotence_and_appended_evidence_update(self):
        waiting_policy = policy_message(
            'same-scenario',
            selected_policy='gather_more_sibling_evidence_before_policy_change',
            recipient='broadcast',
        )
        first_ledger, first_message = build_once([waiting_policy])
        repeat_ledger, repeat_message = build_once([waiting_policy], ledger=first_ledger)
        appended_ledger, appended_message = build_once(
            [waiting_policy, evidence_message('same-scenario')],
            ledger=repeat_ledger,
        )

        self.assertEqual('policy_waiting_for_sibling_evidence', first_ledger['latest']['selected_outcome'])
        self.assertIsNotNone(first_message)
        self.assertEqual('summarize_noop', repeat_ledger['latest']['selected_outcome'])
        self.assertIsNone(repeat_message)
        self.assertEqual('code_simulation_policy_improved_science_campaign', appended_ledger['latest']['selected_outcome'])
        self.assertEqual('retained', appended_ledger['latest']['policy_retention_state'])
        self.assertEqual('code_module', appended_message['recipient'])

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
                first = run_science_coordination_policy_outcome_assessor(
                    theory_memory_file=runtime,
                    runtime_memory_path=str(runtime),
                    transcript_file=transcript,
                    policy_ledger_file=tmp / 'missing-policy.json',
                    history_ledger_file=tmp / 'missing-history.json',
                    cycle_strategy_outcome_ledger_file=tmp / 'missing-cycle-outcome.json',
                    cycle_strategy_ledger_file=tmp / 'missing-cycle-strategy.json',
                    strategy_outcome_ledger_file=tmp / 'missing-strategy-outcome.json',
                    frontier_outcome_ledger_file=tmp / 'missing-frontier-outcome.json',
                    policy_outcome_ledger_file=outcome_ledger,
                    outbox_file=outbox,
                    memory_data=memory_fixture(),
                    git_status_text=' M tmp/theory-memory.json',
                    git_ignored_text='',
                )
                repeat = run_science_coordination_policy_outcome_assessor(
                    theory_memory_file=runtime,
                    runtime_memory_path=str(runtime),
                    transcript_file=transcript,
                    policy_ledger_file=tmp / 'missing-policy.json',
                    history_ledger_file=tmp / 'missing-history.json',
                    cycle_strategy_outcome_ledger_file=tmp / 'missing-cycle-outcome.json',
                    cycle_strategy_ledger_file=tmp / 'missing-cycle-strategy.json',
                    strategy_outcome_ledger_file=tmp / 'missing-strategy-outcome.json',
                    frontier_outcome_ledger_file=tmp / 'missing-frontier-outcome.json',
                    policy_outcome_ledger_file=outcome_ledger,
                    outbox_file=outbox,
                    memory_data=memory_fixture(),
                    git_status_text=' M tmp/theory-memory.json',
                    git_ignored_text='',
                )

        self.assertIn('AI_DIFFERENT_SCIENCE_COORDINATION_POLICY_OUTCOME', stream.getvalue())
        self.assertTrue(first['science_coordination_policy_outcome_capability'])
        self.assertEqual('code_simulation_policy_improved_science_campaign', first['selected_outcome'])
        self.assertEqual('retained', first['policy_retention_state'])
        self.assertEqual('code_module', first['chosen_recipient'])
        self.assertEqual(1, first['outbox_count'])
        self.assertFalse(first['runtime_memory_mutated'])
        self.assertEqual([], first['label_leaks'])
        self.assertTrue(first['no_sibling_imports'])
        self.assertFalse(first['project_owned_checkpoint_claimed'])
        self.assertEqual('summarize_noop', repeat['selected_outcome'])
        self.assertEqual(0, repeat['outbox_count'])
        self.assertFalse(repeat['runtime_memory_mutated'])

    def test_no_sibling_imports(self):
        source = Path(PROJECT_DIR) / 'agent' / 'science_coordination_policy_outcome.py'
        forbidden = ('funfun', 'Language model 2.0', 'Code Module', 'orchastratorrrr')
        imports = [
            line
            for line in source.read_text(encoding='utf-8').splitlines()
            if line.startswith(('import ', 'from '))
        ]
        self.assertFalse([line for line in imports if any(term in line for term in forbidden)])


if __name__ == '__main__':
    unittest.main()
