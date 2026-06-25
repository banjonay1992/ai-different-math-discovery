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
from agent.science_campaign_strategy import (  # noqa: E402
    SCIENCE_CAMPAIGN_STRATEGY_LEDGER_KIND,
    build_science_campaign_strategy_plan,
    empty_science_campaign_strategy_ledger,
    load_science_campaign_strategy_ledger,
    read_science_campaign_strategy_transcript,
    validate_science_campaign_strategy_ledger,
    write_science_campaign_strategy_ledger,
    write_science_campaign_strategy_outbox_jsonl,
)
from agent.science_theory_frontier_outcome import SCIENCE_THEORY_FRONTIER_OUTCOME_LEDGER_KIND  # noqa: E402
from main import run_science_campaign_strategy_planner  # noqa: E402


def memory_fixture():
    return {
        'discovery_readiness': {'readiness_score': 0.94, 'status': 'nearly_ready'},
        'abstraction_discovery_evidence': {'transfer_outcome_count': 12},
    }


def strategy_source_message(
    scenario_id='scenario-one',
    *,
    outcome_id=None,
    selected_outcome='theory_memory_recorded_or_hypothesis_promoted',
    planned_move='promote_supported_hypothesis_to_theory_memory',
    leak=False,
    third_party=False,
):
    outcome_id = outcome_id or f'frontier-outcome-{scenario_id}'
    body = {
        'response_kind': 'science_theory_frontier_outcome',
        'frontier_outcome_id': outcome_id,
        'frontier_id': f'frontier-{scenario_id}',
        'source_outcome_id': f'outcome-{scenario_id}',
        'scenario_id': scenario_id,
        'hypothesis_id': f'hypothesis:{scenario_id}',
        'campaign_id': f'campaign-{scenario_id}',
        'planned_theory_move': planned_move,
        'selected_outcome': selected_outcome,
        'selected_action': selected_outcome,
        'selected_recipient': 'broadcast',
        'observed_theory_evidence': [{'source': 'theory_memory', 'status': 'recorded'}],
        'observed_sibling_evidence': [],
        'before_theory_memory_state': 'not_recorded',
        'after_theory_memory_state': 'recorded',
        'promotion_state': 'promoted' if selected_outcome == 'theory_memory_recorded_or_hypothesis_promoted' else 'none',
        'retirement_block_state': 'retired' if selected_outcome == 'refuted_hypothesis_retired_or_blocked' else 'none',
        'repair_state': 'none',
        'closure_state': 'stable' if selected_outcome == 'close_stable_science_campaign' else 'none',
        'boundary_checkpoint_state': 'repair' if selected_outcome == 'preserve_boundary_or_checkpoint_repair' else 'clean',
        'boundary_notes': ['repair boundary'] if selected_outcome == 'preserve_boundary_or_checkpoint_repair' else [],
        'third_party_checkpoint_used': third_party,
    }
    if leak:
        body['boundary_notes'] = ['gravity label leaked']
    return build_module_chat_message(
        sender='ai_different',
        recipient='orchestrator',
        topic='ai_different.science_theory_frontier_outcome',
        body=body,
        evidence={'frontier_outcome_id': outcome_id, 'scenario_id': scenario_id, 'selected_outcome': selected_outcome},
        tags=['science_theory_frontier_outcome'],
    )


def sibling_evidence(
    scenario_id='scenario-one',
    *,
    sender='funfun',
    gate='formal_proof',
    status='needed',
):
    return build_module_chat_message(
        sender=sender,
        recipient='ai_different',
        topic=f'{sender}.{gate}.{status}',
        body={
            'scenario_id': scenario_id,
            'hypothesis_id': f'hypothesis:{scenario_id}',
            'campaign_id': f'campaign-{scenario_id}',
            'evidence_gate': gate,
            'status': status,
        },
        evidence={'scenario_id': scenario_id, 'evidence_gate': gate, 'status': status},
        tags=['science_campaign_strategy', gate, status],
    )


def frontier_outcome_ledger_fixture(selected_outcome='theory_memory_recorded_or_hypothesis_promoted', *, scenario_id='scenario-one'):
    return {
        'schema_version': 1,
        'ledger_kind': SCIENCE_THEORY_FRONTIER_OUTCOME_LEDGER_KIND,
        'processed_message_ids': [],
        'processed_source_hashes': [],
        'assessed_frontier_keys': [],
        'frontier_rows': [],
        'outcome_records': [{
            'frontier_outcome_id': f'frontier-outcome-{scenario_id}',
            'frontier_id': f'frontier-{scenario_id}',
            'source_outcome_id': f'outcome-{scenario_id}',
            'hypothesis_id': f'hypothesis:{scenario_id}',
            'campaign_id': f'campaign-{scenario_id}',
            'scenario_id': scenario_id,
            'selected_outcome': selected_outcome,
            'planned_theory_move': 'promote_supported_hypothesis_to_theory_memory',
            'observed_theory_evidence': [{'source': 'theory_memory', 'status': 'recorded'}],
            'observed_sibling_evidence': [],
            'promotion_state': 'promoted' if selected_outcome == 'theory_memory_recorded_or_hypothesis_promoted' else 'none',
            'retirement_block_state': 'none',
            'boundary_checkpoint_state': 'clean',
            'boundary_notes': [],
            'theory_memory_state': 'recorded',
        }],
        'outgoing_response_ids': [],
        'latest': {},
        'ledger_hash': f'frontier-outcome-{scenario_id}-hash',
    }


def theory_memory_ledger_fixture(*, scenario_id='scenario-one'):
    return {
        'schema_version': 1,
        'ledger_kind': 'ai_different.symbolic_theory_memory',
        'theory_memory_rows': [{
            'frontier_outcome_id': f'frontier-outcome-{scenario_id}',
            'frontier_id': f'frontier-{scenario_id}',
            'scenario_id': scenario_id,
            'hypothesis_id': f'hypothesis:{scenario_id}',
            'campaign_id': f'campaign-{scenario_id}',
            'status': 'recorded',
            'source': 'theory_memory',
        }],
        'ledger_hash': f'theory-memory-{scenario_id}-hash',
    }


def build_once(messages, *, ledger=None, frontier_outcome_ledger=None, theory_memory_ledger=None, project_boundary=None):
    updated, message = build_science_campaign_strategy_plan(
        transcript_messages=messages,
        strategy_ledger=ledger or empty_science_campaign_strategy_ledger(),
        frontier_outcome_ledger=frontier_outcome_ledger or {},
        frontier_ledger={},
        campaign_ledger={},
        theory_memory_ledger=theory_memory_ledger or {},
        hypothesis_ledger={},
        experiment_ledger={},
        module_chat_ledger={},
        prior_strategy_ledger={},
        runtime_memory_data=memory_fixture(),
        runtime_memory_hash_state={'unchanged': True, 'before': 'same', 'after': 'same'},
        project_owned_boundary=project_boundary or {'third_party_checkpoint_used': False},
    )
    return updated, message


class ScienceCampaignStrategyTests(unittest.TestCase):
    def test_strategy_ledger_persistence_load_and_malformed_rejection(self):
        ledger, message = build_once([strategy_source_message()])
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = Path(tmpdir) / 'strategy.json'
            outbox = Path(tmpdir) / 'outbox.jsonl'
            write_science_campaign_strategy_ledger(ledger_path, ledger)
            loaded = load_science_campaign_strategy_ledger(ledger_path)
            write_science_campaign_strategy_outbox_jsonl(outbox, message)
            rows = [json.loads(line) for line in outbox.read_text(encoding='utf-8').splitlines()]

        self.assertEqual(SCIENCE_CAMPAIGN_STRATEGY_LEDGER_KIND, loaded['ledger_kind'])
        self.assertEqual('science_campaign_strategy', rows[0]['body']['response_kind'])
        with self.assertRaisesRegex(ValueError, 'wrong ledger_kind'):
            validate_science_campaign_strategy_ledger({'ledger_kind': 'wrong'})

    def test_outcome_to_strategy_extraction_and_invalid_record(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            transcript = Path(tmpdir) / 'strategy.jsonl'
            transcript.write_text(
                json.dumps(strategy_source_message(), sort_keys=True) + '\n'
                + json.dumps(sibling_evidence(), sort_keys=True) + '\n'
                + '{"sender":"bad"}\n',
                encoding='utf-8',
            )
            parsed = read_science_campaign_strategy_transcript(transcript)
        ledger, _ = build_once(parsed['messages'])

        self.assertEqual(2, len(parsed['messages']))
        self.assertEqual(1, len(parsed['invalid_messages']))
        row = ledger['strategy_rows'][0]
        self.assertEqual('scenario-one', row['scenario_id'])
        self.assertEqual('promote_validated_hypothesis_to_theory_memory', ledger['latest']['selected_strategy'])

    def test_deterministic_priority_selection_and_routing(self):
        cases = [
            (
                [strategy_source_message('boundary', selected_outcome='preserve_boundary_or_checkpoint_repair')],
                'preserve_checkpoint_boundary',
                'code_module',
            ),
            (
                [strategy_source_message('promote')],
                'promote_validated_hypothesis_to_theory_memory',
                'broadcast',
            ),
            (
                [strategy_source_message('retire', selected_outcome='refuted_hypothesis_retired_or_blocked', planned_move='retire_or_block_refuted_hypothesis')],
                'retire_weak_hypothesis_route',
                'broadcast',
            ),
            (
                [strategy_source_message('funfun', selected_outcome='funfun_certificate_supports_or_blocks_theory_move', planned_move='request_funfun_certificate')],
                'request_funfun_formal_or_proof_clarification',
                'funfun',
            ),
            (
                [strategy_source_message('code', selected_outcome='code_experiment_or_counterexample_changes_theory_move', planned_move='request_code_experiment_or_counterexample')],
                'request_code_simulation_or_primitive_capability',
                'code_module',
            ),
            (
                [strategy_source_message('language', selected_outcome='language_protocol_clarification_resolves_theory_move', planned_move='request_language_protocol_clarification')],
                'request_language_terminology_or_protocol_clarification',
                'language_model_2',
            ),
            (
                [strategy_source_message('repair', selected_outcome='hypothesis_refinement_accepted', planned_move='refine_hypothesis_from_outcome')],
                'reopen_theory_repair_cycle',
                'orchestrator',
            ),
            (
                [strategy_source_message('close', selected_outcome='close_stable_science_campaign')],
                'close_stable_science_campaign',
                'orchestrator',
            ),
            (
                [strategy_source_message('schedule', selected_outcome='next_campaign_frontier_scheduled', planned_move='schedule_next_campaign_frontier_check')],
                'schedule_next_frontier_hypothesis',
                'orchestrator',
            ),
            (
                [strategy_source_message('nogain', selected_outcome='no_measurable_theory_frontier_gain', planned_move='record_no_measurable_theory_gain')],
                'record_no_measurable_science_gain',
                'orchestrator',
            ),
        ]
        for messages, strategy, recipient in cases:
            with self.subTest(strategy=strategy):
                ledger, message = build_once(messages)
                self.assertEqual(strategy, ledger['latest']['selected_strategy'])
                self.assertEqual(recipient, message['recipient'])

    def test_ledger_sources_theory_memory_and_boundary_guards(self):
        ledger, message = build_once([], frontier_outcome_ledger=frontier_outcome_ledger_fixture())
        self.assertEqual('promote_validated_hypothesis_to_theory_memory', ledger['latest']['selected_strategy'])
        self.assertEqual('broadcast', message['recipient'])

        memory_ledger, memory_message = build_once([], theory_memory_ledger=theory_memory_ledger_fixture())
        self.assertEqual('promote_validated_hypothesis_to_theory_memory', memory_ledger['latest']['selected_strategy'])
        self.assertEqual('broadcast', memory_message['recipient'])

        repair_ledger, repair_message = build_once([strategy_source_message('leak', leak=True)])
        self.assertEqual('preserve_checkpoint_boundary', repair_ledger['latest']['selected_strategy'])
        self.assertEqual('code_module', repair_message['recipient'])
        self.assertIn('gravity', repair_message['body']['label_leaks'])

        third_party_ledger, third_party_message = build_once(
            [strategy_source_message('third-party')],
            project_boundary={'third_party_checkpoint_used': True},
        )
        self.assertEqual('preserve_checkpoint_boundary', third_party_ledger['latest']['selected_strategy'])
        self.assertIn('no local-owned checkpoint claim', third_party_message['body']['no_overclaiming_proof'])
        self.assertFalse(third_party_message['body'].get('project_owned_checkpoint_claimed', False))

    def test_duplicate_idempotence_and_appended_evidence_update(self):
        first_ledger, first_message = build_once([
            strategy_source_message('waiting', selected_outcome='planned_theory_move_waiting_for_evidence', planned_move='request_funfun_certificate'),
        ])
        self.assertEqual('request_funfun_formal_or_proof_clarification', first_ledger['latest']['selected_strategy'])
        self.assertIsNotNone(first_message)

        repeat_ledger, repeat_message = build_once([
            strategy_source_message('waiting', selected_outcome='planned_theory_move_waiting_for_evidence', planned_move='request_funfun_certificate'),
        ], ledger=first_ledger)
        self.assertEqual('summarize_noop', repeat_ledger['latest']['selected_strategy'])
        self.assertIsNone(repeat_message)

        appended_ledger, appended_message = build_once([
            strategy_source_message('waiting', selected_outcome='planned_theory_move_waiting_for_evidence', planned_move='request_funfun_certificate'),
            strategy_source_message('waiting', outcome_id='frontier-outcome-waiting-promoted', selected_outcome='theory_memory_recorded_or_hypothesis_promoted'),
        ], ledger=repeat_ledger)
        self.assertEqual('promote_validated_hypothesis_to_theory_memory', appended_ledger['latest']['selected_strategy'])
        self.assertIsNotNone(appended_message)

    def test_cli_status_proof_preserves_runtime_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            runtime = tmp / 'theory-memory.json'
            runtime.write_text(json.dumps(memory_fixture(), sort_keys=True), encoding='utf-8')
            transcript = tmp / 'strategy.jsonl'
            transcript.write_text(json.dumps(strategy_source_message(), sort_keys=True) + '\n', encoding='utf-8')
            ledger_path = tmp / 'strategy-ledger.json'
            outbox = tmp / 'outbox.jsonl'

            before = runtime.read_text(encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()):
                result = run_science_campaign_strategy_planner(
                    runtime_memory_path=runtime,
                    transcript_file=transcript,
                    frontier_outcome_ledger_file=tmp / 'missing-frontier-outcome.json',
                    frontier_ledger_file=tmp / 'missing-frontier.json',
                    campaign_ledger_file=tmp / 'missing-campaign.json',
                    strategy_ledger_file=ledger_path,
                    outbox_file=outbox,
                    memory_data=memory_fixture(),
                    git_status_text=' M tmp/theory-memory.json\n',
                    git_ignored_text='',
                )
            with contextlib.redirect_stdout(io.StringIO()):
                repeat = run_science_campaign_strategy_planner(
                    runtime_memory_path=runtime,
                    transcript_file=transcript,
                    frontier_outcome_ledger_file=tmp / 'missing-frontier-outcome.json',
                    frontier_ledger_file=tmp / 'missing-frontier.json',
                    campaign_ledger_file=tmp / 'missing-campaign.json',
                    strategy_ledger_file=ledger_path,
                    outbox_file=outbox,
                    memory_data=memory_fixture(),
                    git_status_text=' M tmp/theory-memory.json\n',
                    git_ignored_text='',
                )
            after = runtime.read_text(encoding='utf-8')

        self.assertEqual(before, after)
        self.assertEqual('promote_validated_hypothesis_to_theory_memory', result['selected_strategy'])
        self.assertEqual(1, result['outbox_count'])
        self.assertEqual('summarize_noop', repeat['selected_strategy'])
        self.assertEqual(0, repeat['outbox_count'])
        self.assertFalse(result['runtime_memory_mutated'])
        self.assertEqual([], result['label_leaks'])
        self.assertTrue(result['no_sibling_imports'])
        self.assertFalse(result['project_owned_checkpoint_claimed'])

    def test_no_sibling_imports(self):
        module = Path(PROJECT_DIR) / 'agent' / 'science_campaign_strategy.py'
        main = Path(PROJECT_DIR) / 'main.py'
        for source in (module.read_text(encoding='utf-8'), main.read_text(encoding='utf-8')):
            import_lines = [line for line in source.splitlines() if line.startswith(('import ', 'from '))]
            joined = '\n'.join(import_lines)
            self.assertNotIn('funfun', joined)
            self.assertNotIn('Language model 2.0', joined)
            self.assertNotIn('Code Module', joined)
            self.assertNotIn('orchastratorrrr', joined)


if __name__ == '__main__':
    unittest.main()
