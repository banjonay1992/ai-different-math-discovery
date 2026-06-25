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
from agent.science_coordination_interaction_history import SCIENCE_COORDINATION_INTERACTION_HISTORY_LEDGER_KIND  # noqa: E402
from agent.science_history_guided_policy import (  # noqa: E402
    SCIENCE_HISTORY_GUIDED_POLICY_LEDGER_KIND,
    build_science_history_guided_policy_plan,
    empty_science_history_guided_policy_ledger,
    load_science_history_guided_policy_ledger,
    read_science_history_guided_policy_transcript,
    validate_science_history_guided_policy_ledger,
    write_science_history_guided_policy_ledger,
    write_science_history_guided_policy_outbox_jsonl,
)
from main import run_science_history_guided_policy_planner  # noqa: E402


def memory_fixture():
    return {
        'discovery_readiness': {'readiness_score': 0.99, 'status': 'history_guided_policy_ready'},
        'abstraction_discovery_evidence': {'transfer_outcome_count': 61},
    }


def history_message(
    scenario_id='scenario-one',
    *,
    ordering='code_simulation_before_science',
    decision='retain_code_simulation_before_science_when_repeatedly_helpful',
    retention='retained',
    third_party=False,
):
    body = {
        'response_kind': 'science_coordination_interaction_history',
        'science_coordination_interaction_history_id': f'interaction-history-{scenario_id}',
        'selected_decision': decision,
        'selected_action': decision,
        'retention_decision': retention,
        'interaction_ordering_key': ordering,
        'selected_ordering_key': ordering,
        'selected_ordering': ordering if retention == 'retained' else None,
        'retired_ordering': ordering if retention == 'retired' else None,
        'source_outcome_ids': [f'ab-outcome-{scenario_id}'],
        'source_probe_ids': [f'probe-{scenario_id}'],
        'target_campaign_ids': [f'campaign-{scenario_id}'],
        'target_hypothesis_ids': [f'hypothesis:{scenario_id}'],
        'target_simulation_ids': [f'simulation-{scenario_id}'],
        'tested_commits': ['d3f91f5335aa34f9b9c5a4a5c15fa19b17e14afc'],
        'tested_tests': ['tests.test_science_coordination_interaction_history:7'],
        'candidate_not_causal': True,
        'checkpoint_boundary_state': 'repair' if decision == 'preserve_checkpoint_boundary' else 'clean',
        'checkpoint_boundary_notes': ['boundary preserved'] if decision == 'preserve_checkpoint_boundary' else [],
        'third_party_checkpoint_used': third_party,
        'hf_validation_used': False,
    }
    return build_module_chat_message(
        sender='ai_different',
        recipient='orchestrator',
        topic='ai_different.science_coordination_interaction_history',
        body=body,
        evidence={
            'science_coordination_interaction_history_id': body['science_coordination_interaction_history_id'],
            'selected_decision': decision,
            'interaction_ordering_key': ordering,
            'retention_decision': retention,
        },
        tags=['science_coordination_interaction_history', decision],
    )


def interaction_history_ledger_fixture(
    *,
    scenario_id='scenario-one',
    ordering='code_simulation_before_science',
    decision='retain_code_simulation_before_science_when_repeatedly_helpful',
    retention='retained',
):
    return {
        'schema_version': 1,
        'ledger_kind': SCIENCE_COORDINATION_INTERACTION_HISTORY_LEDGER_KIND,
        'processed_message_ids': [],
        'processed_source_hashes': [],
        'scored_ordering_keys': [],
        'ordering_rows': [],
        'scorecard_records': [{
            'science_coordination_interaction_history_id': f'interaction-history-{scenario_id}',
            'selected_decision': decision,
            'selected_action': decision,
            'retention_decision': retention,
            'interaction_ordering_key': ordering,
            'source_outcome_ids': [f'ab-outcome-{scenario_id}'],
            'target_campaign_ids': [f'campaign-{scenario_id}'],
            'target_hypothesis_ids': [f'hypothesis:{scenario_id}'],
            'target_simulation_ids': [f'simulation-{scenario_id}'],
            'tested_commits': ['d3f91f5335aa34f9b9c5a4a5c15fa19b17e14afc'],
            'tested_tests': ['tests.test_science_coordination_interaction_history:7'],
            'checkpoint_boundary_state': 'clean',
        }],
        'outgoing_response_ids': [],
        'latest': {},
        'ledger_hash': f'interaction-history-{scenario_id}-hash',
    }


def build_once(messages, *, ledger=None, interaction_history_ledger=None, project_boundary=None):
    updated, message = build_science_history_guided_policy_plan(
        transcript_messages=messages,
        history_guided_policy_ledger=ledger or empty_science_history_guided_policy_ledger(),
        interaction_history_ledger=interaction_history_ledger or {},
        ab_probe_outcome_ledger={},
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
        prior_history_guided_policy_ledger={},
        runtime_memory_data=memory_fixture(),
        runtime_memory_hash_state={'unchanged': True, 'before': 'same', 'after': 'same'},
        project_owned_boundary=project_boundary or {'third_party_checkpoint_used': False},
        hf_use_status={'hf_validation_used': False},
    )
    return updated, message


class ScienceHistoryGuidedPolicyTests(unittest.TestCase):
    def test_policy_ledger_persistence_load_and_outbox_schema(self):
        ledger, message = build_once([history_message()])
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = Path(tmpdir) / 'policy.json'
            outbox = Path(tmpdir) / 'outbox.jsonl'
            write_science_history_guided_policy_ledger(ledger_path, ledger)
            loaded = load_science_history_guided_policy_ledger(ledger_path)
            write_science_history_guided_policy_outbox_jsonl(outbox, message)
            rows = [json.loads(line) for line in outbox.read_text(encoding='utf-8').splitlines()]

        self.assertEqual(SCIENCE_HISTORY_GUIDED_POLICY_LEDGER_KIND, loaded['ledger_kind'])
        self.assertEqual('science_history_guided_policy', rows[0]['body']['response_kind'])
        self.assertEqual('apply_retained_code_simulation_before_science_ordering', rows[0]['body']['selected_policy_decision'])
        self.assertEqual('code_simulation_result_before_science_campaign', rows[0]['body']['required_evidence_gate'])
        self.assertTrue(rows[0]['body']['candidate_not_causal'])
        self.assertIn('no autonomous self-modification', rows[0]['body']['no_overclaiming_proof'])
        self.assertFalse(rows[0]['body']['hf_validation_used'])
        with self.assertRaisesRegex(ValueError, 'wrong ledger_kind'):
            validate_science_history_guided_policy_ledger({'ledger_kind': 'wrong'})

    def test_history_to_policy_extraction_and_invalid_record(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            transcript = Path(tmpdir) / 'history-guided-policy.jsonl'
            transcript.write_text(
                json.dumps(history_message(), sort_keys=True) + '\n'
                + '{"sender":"bad"}\n',
                encoding='utf-8',
            )
            parsed = read_science_history_guided_policy_transcript(transcript)
        ledger, message = build_once(parsed['messages'])

        self.assertEqual(1, len(parsed['messages']))
        self.assertEqual(1, len(parsed['invalid_messages']))
        self.assertEqual('apply_retained_code_simulation_before_science_ordering', ledger['latest']['selected_policy_decision'])
        self.assertEqual('orchestrator', message['recipient'])

    def test_deterministic_priority_selection(self):
        cases = [
            (
                [history_message('boundary', decision='preserve_checkpoint_boundary', third_party=True)],
                'preserve_checkpoint_boundary',
                'verify_no_project_owned_checkpoint_overclaim',
            ),
            (
                [history_message('code')],
                'apply_retained_code_simulation_before_science_ordering',
                'code_simulation_result_before_science_campaign',
            ),
            (
                [history_message('proof', ordering='proof_before_hypothesis', decision='retain_proof_before_hypothesis_when_repeatedly_helpful')],
                'apply_retained_proof_before_hypothesis_ordering',
                'funfun_formal_or_proof_certificate_before_hypothesis',
            ),
            (
                [history_message('language', ordering='language_summary_before_science', decision='retain_language_summary_before_science_when_repeatedly_helpful')],
                'apply_retained_language_summary_before_science_ordering',
                'language_protocol_summary_before_science_campaign',
            ),
            (
                [history_message('history', ordering='history_policy_memory_before_science', decision='retain_history_policy_memory_when_repeatedly_helpful')],
                'apply_retained_history_policy_memory_ordering',
                'history_policy_memory_receipt_before_campaign',
            ),
            (
                [history_message('control', decision='weaken_or_retire_ordering_after_repeated_control_wins', retention='retired')],
                'weaken_or_retire_ordering_after_control_wins',
                'confirm_retired_ordering_not_reused_without_new_evidence',
            ),
            (
                [history_message('waiting', decision='keep_ordering_underpowered_waiting_for_more_evidence', retention='waiting')],
                'keep_history_guided_policy_waiting_for_more_evidence',
                'collect_more_interaction_history_evidence',
            ),
            (
                [history_message('nogain', decision='record_no_measurable_interaction_gain', retention='none')],
                'record_no_actionable_history_policy_gain',
                'record_no_actionable_gain',
            ),
        ]
        for messages, decision, gate in cases:
            with self.subTest(decision=decision):
                ledger, message = build_once(messages)
                self.assertEqual(decision, ledger['latest']['selected_policy_decision'])
                self.assertEqual(gate, message['body']['required_evidence_gate'])

    def test_boundary_preservation_and_no_overclaiming_from_project_state(self):
        ledger, message = build_once(
            [history_message('code')],
            project_boundary={'third_party_checkpoint_used': True, 'project_owned_checkpoint_verified': False},
        )

        self.assertEqual('preserve_checkpoint_boundary', ledger['latest']['selected_policy_decision'])
        self.assertEqual('repair', message['body']['checkpoint_boundary_state'])
        self.assertFalse(message['body']['project_owned_checkpoint_claimed'])
        self.assertTrue(message['body']['third_party_checkpoint_used'])
        self.assertIn('Candidate-not-causal', message['body']['candidate_not_causal_wording'])

    def test_duplicate_idempotence_and_appended_evidence_update(self):
        waiting = history_message('same-order', decision='keep_ordering_underpowered_waiting_for_more_evidence', retention='waiting')
        first_ledger, first_message = build_once([waiting])
        repeat_ledger, repeat_message = build_once([waiting], ledger=first_ledger)
        appended_ledger, appended_message = build_once(
            [
                waiting,
                history_message('same-order'),
            ],
            ledger=repeat_ledger,
        )

        self.assertEqual('keep_history_guided_policy_waiting_for_more_evidence', first_ledger['latest']['selected_policy_decision'])
        self.assertIsNotNone(first_message)
        self.assertEqual('summarize_noop', repeat_ledger['latest']['selected_policy_decision'])
        self.assertIsNone(repeat_message)
        self.assertEqual('apply_retained_code_simulation_before_science_ordering', appended_ledger['latest']['selected_policy_decision'])
        self.assertEqual('code_simulation_before_science', appended_message['body']['selected_ordering'])

    def test_cli_status_proof_preserves_runtime_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            runtime = tmp / 'theory-memory.json'
            runtime.write_text(json.dumps(memory_fixture(), sort_keys=True), encoding='utf-8')
            transcript = tmp / 'inbox.jsonl'
            transcript.write_text(json.dumps(history_message(), sort_keys=True) + '\n', encoding='utf-8')
            policy_ledger = tmp / 'history-guided-policy.json'
            outbox = tmp / 'outbox.jsonl'

            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                first = run_science_history_guided_policy_planner(
                    theory_memory_file=runtime,
                    runtime_memory_path=str(runtime),
                    transcript_file=transcript,
                    history_guided_policy_ledger_file=policy_ledger,
                    outbox_file=outbox,
                    memory_data=memory_fixture(),
                    git_status_text=' M tmp/theory-memory.json',
                    git_ignored_text='',
                )
                repeat = run_science_history_guided_policy_planner(
                    theory_memory_file=runtime,
                    runtime_memory_path=str(runtime),
                    transcript_file=transcript,
                    history_guided_policy_ledger_file=policy_ledger,
                    outbox_file=outbox,
                    memory_data=memory_fixture(),
                    git_status_text=' M tmp/theory-memory.json',
                    git_ignored_text='',
                )

        self.assertIn('AI_DIFFERENT_SCIENCE_HISTORY_GUIDED_POLICY', stream.getvalue())
        self.assertTrue(first['science_history_guided_policy_capability'])
        self.assertEqual('apply_retained_code_simulation_before_science_ordering', first['selected_decision'])
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
        source = Path(PROJECT_DIR) / 'agent' / 'science_history_guided_policy.py'
        forbidden = ('funfun', 'Language model 2.0', 'Code Module', 'orchastratorrrr')
        imports = [
            line
            for line in source.read_text(encoding='utf-8').splitlines()
            if line.startswith(('import ', 'from '))
        ]
        self.assertFalse([line for line in imports if any(term in line for term in forbidden)])


if __name__ == '__main__':
    unittest.main()
