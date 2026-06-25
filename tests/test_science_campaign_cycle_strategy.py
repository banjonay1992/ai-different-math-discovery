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
from agent.science_campaign_strategy_outcome import SCIENCE_CAMPAIGN_STRATEGY_OUTCOME_LEDGER_KIND  # noqa: E402
from agent.science_campaign_cycle_strategy import (  # noqa: E402
    SCIENCE_CAMPAIGN_CYCLE_STRATEGY_LEDGER_KIND,
    build_science_campaign_cycle_strategy_plan,
    empty_science_campaign_cycle_strategy_ledger,
    load_science_campaign_cycle_strategy_ledger,
    read_science_campaign_cycle_strategy_transcript,
    validate_science_campaign_cycle_strategy_ledger,
    write_science_campaign_cycle_strategy_ledger,
    write_science_campaign_cycle_strategy_outbox_jsonl,
)
from main import run_science_campaign_cycle_strategy_planner  # noqa: E402


def memory_fixture():
    return {
        'discovery_readiness': {'readiness_score': 0.95, 'status': 'nearly_ready'},
        'abstraction_discovery_evidence': {'transfer_outcome_count': 17},
    }


def strategy_outcome_message(
    scenario_id='scenario-one',
    *,
    outcome_id=None,
    strategy_id=None,
    selected_outcome='validated_hypothesis_promoted_to_theory_memory',
    planned_strategy='promote_validated_hypothesis_to_theory_memory',
    leak=False,
    third_party=False,
):
    outcome_id = outcome_id or f'strategy-outcome-{scenario_id}'
    strategy_id = strategy_id or f'strategy-{scenario_id}'
    body = {
        'response_kind': 'science_campaign_strategy_outcome',
        'campaign_strategy_outcome_id': outcome_id,
        'strategy_id': strategy_id,
        'frontier_outcome_id': f'frontier-outcome-{scenario_id}',
        'frontier_id': f'frontier-{scenario_id}',
        'scenario_id': scenario_id,
        'hypothesis_id': f'hypothesis:{scenario_id}',
        'campaign_id': f'campaign-{scenario_id}',
        'family_id': f'family-{scenario_id}',
        'planned_strategy': planned_strategy,
        'selected_outcome': selected_outcome,
        'selected_action': selected_outcome,
        'selected_recipient': 'broadcast',
        'source_evidence_used': [{'source': 'strategy_outcome', 'status': 'assessed'}],
        'observed_sibling_evidence': [],
        'promotion_state': 'promoted' if selected_outcome == 'validated_hypothesis_promoted_to_theory_memory' else 'none',
        'retirement_state': 'retired' if selected_outcome == 'weak_hypothesis_route_retired' else 'none',
        'repair_state': 'open' if selected_outcome == 'theory_repair_cycle_reopened' else 'none',
        'closure_state': 'closed' if selected_outcome == 'stable_science_campaign_closed' else 'none',
        'waiting_blocker_state': 'waiting' if 'waiting' in selected_outcome else 'resolved',
        'before_theory_memory_state': 'not_recorded',
        'after_theory_memory_state': 'recorded' if selected_outcome == 'validated_hypothesis_promoted_to_theory_memory' else 'unknown',
        'checkpoint_boundary_state': 'repair' if selected_outcome == 'preserve_checkpoint_boundary' else 'clean',
        'checkpoint_boundary_notes': ['checkpoint boundary repair'] if selected_outcome == 'preserve_checkpoint_boundary' else [],
        'third_party_checkpoint_used': third_party,
    }
    if leak:
        body['checkpoint_boundary_notes'] = ['gravity label leaked']
    return build_module_chat_message(
        sender='ai_different',
        recipient='orchestrator',
        topic='ai_different.science_campaign_strategy_outcome',
        body=body,
        evidence={'campaign_strategy_outcome_id': outcome_id, 'scenario_id': scenario_id, 'selected_outcome': selected_outcome},
        tags=['science_campaign_strategy_outcome'],
    )


def sibling_evidence(
    scenario_id='scenario-one',
    *,
    sender='funfun',
    gate='formal_proof',
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
            'evidence_gate': gate,
            'status': status,
        },
        evidence={'scenario_id': scenario_id, 'evidence_gate': gate, 'status': status},
        tags=['science_campaign_cycle_strategy', gate, status],
    )


def strategy_outcome_ledger_fixture(
    *,
    scenario_id='scenario-one',
    selected_outcome='validated_hypothesis_promoted_to_theory_memory',
):
    return {
        'schema_version': 1,
        'ledger_kind': SCIENCE_CAMPAIGN_STRATEGY_OUTCOME_LEDGER_KIND,
        'processed_message_ids': [],
        'processed_source_hashes': [],
        'assessed_strategy_keys': [],
        'strategy_rows': [],
        'outcome_records': [{
            'campaign_strategy_outcome_id': f'strategy-outcome-{scenario_id}',
            'strategy_id': f'strategy-{scenario_id}',
            'frontier_outcome_id': f'frontier-outcome-{scenario_id}',
            'frontier_id': f'frontier-{scenario_id}',
            'hypothesis_id': f'hypothesis:{scenario_id}',
            'campaign_id': f'campaign-{scenario_id}',
            'family_id': f'family-{scenario_id}',
            'scenario_id': scenario_id,
            'selected_outcome': selected_outcome,
            'planned_strategy': 'promote_validated_hypothesis_to_theory_memory',
            'source_evidence_used': [{'source': 'outcome_ledger', 'status': 'recorded'}],
            'observed_sibling_evidence': [],
            'promotion_state': 'promoted' if selected_outcome == 'validated_hypothesis_promoted_to_theory_memory' else 'none',
            'retirement_state': 'none',
            'repair_state': 'none',
            'closure_state': 'none',
            'waiting_blocker_state': 'resolved',
            'before_theory_memory_state': 'not_recorded',
            'after_theory_memory_state': 'recorded',
            'checkpoint_boundary_notes': [],
        }],
        'outgoing_response_ids': [],
        'latest': {},
        'ledger_hash': f'strategy-outcome-{scenario_id}-hash',
    }


def cycle_memory_ledger_fixture(*, scenario_id='scenario-one'):
    return {
        'schema_version': 1,
        'ledger_kind': 'ai_different.symbolic_campaign_cycle_memory',
        'campaign_cycle_memory_rows': [{
            'campaign_strategy_outcome_id': f'strategy-outcome-{scenario_id}',
            'scenario_id': scenario_id,
            'hypothesis_id': f'hypothesis:{scenario_id}',
            'campaign_id': f'campaign-{scenario_id}',
            'family_id': f'family-{scenario_id}',
            'status': 'recorded',
            'source': 'campaign_cycle_memory',
        }],
        'ledger_hash': f'cycle-memory-{scenario_id}-hash',
    }


def build_once(messages, *, ledger=None, strategy_outcome_ledger=None, cycle_memory_ledger=None, project_boundary=None):
    updated, message = build_science_campaign_cycle_strategy_plan(
        transcript_messages=messages,
        cycle_strategy_ledger=ledger or empty_science_campaign_cycle_strategy_ledger(),
        strategy_outcome_ledger=strategy_outcome_ledger or {},
        strategy_ledger={},
        frontier_outcome_ledger={},
        frontier_ledger={},
        campaign_ledger={},
        theory_memory_ledger=cycle_memory_ledger or {},
        hypothesis_ledger={},
        experiment_ledger={},
        module_chat_ledger={},
        prior_cycle_strategy_ledger={},
        runtime_memory_data=memory_fixture(),
        runtime_memory_hash_state={'unchanged': True, 'before': 'same', 'after': 'same'},
        project_owned_boundary=project_boundary or {'third_party_checkpoint_used': False},
    )
    return updated, message


class ScienceCampaignCycleStrategyTests(unittest.TestCase):
    def test_cycle_strategy_ledger_persistence_load_and_malformed_rejection(self):
        ledger, message = build_once([strategy_outcome_message()])
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = Path(tmpdir) / 'cycle-strategy.json'
            outbox = Path(tmpdir) / 'outbox.jsonl'
            write_science_campaign_cycle_strategy_ledger(ledger_path, ledger)
            loaded = load_science_campaign_cycle_strategy_ledger(ledger_path)
            write_science_campaign_cycle_strategy_outbox_jsonl(outbox, message)
            rows = [json.loads(line) for line in outbox.read_text(encoding='utf-8').splitlines()]

        self.assertEqual(SCIENCE_CAMPAIGN_CYCLE_STRATEGY_LEDGER_KIND, loaded['ledger_kind'])
        self.assertEqual('science_campaign_cycle_strategy', rows[0]['body']['response_kind'])
        with self.assertRaisesRegex(ValueError, 'wrong ledger_kind'):
            validate_science_campaign_cycle_strategy_ledger({'ledger_kind': 'wrong'})

    def test_outcome_to_cycle_strategy_extraction_and_invalid_record(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            transcript = Path(tmpdir) / 'cycle-strategy.jsonl'
            transcript.write_text(
                json.dumps(strategy_outcome_message(), sort_keys=True) + '\n'
                + json.dumps(sibling_evidence(), sort_keys=True) + '\n'
                + '{"sender":"bad"}\n',
                encoding='utf-8',
            )
            parsed = read_science_campaign_cycle_strategy_transcript(transcript)
        ledger, _ = build_once(parsed['messages'])

        self.assertEqual(2, len(parsed['messages']))
        self.assertEqual(1, len(parsed['invalid_messages']))
        row = ledger['cycle_strategy_rows'][0]
        self.assertEqual('scenario-one', row['scenario_id'])
        self.assertEqual('promote_validated_hypothesis_strategy_to_campaign_cycle_memory', ledger['latest']['selected_strategy'])

    def test_deterministic_priority_selection_and_routing(self):
        cases = [
            (
                [strategy_outcome_message('boundary', selected_outcome='preserve_checkpoint_boundary')],
                'preserve_checkpoint_boundary',
                'code_module',
            ),
            ([strategy_outcome_message('promote')], 'promote_validated_hypothesis_strategy_to_campaign_cycle_memory', 'broadcast'),
            (
                [strategy_outcome_message('retire', selected_outcome='weak_hypothesis_route_retired')],
                'retire_weak_hypothesis_family',
                'broadcast',
            ),
            (
                [strategy_outcome_message('funfun', selected_outcome='funfun_formal_or_proof_clarification_received')],
                'request_funfun_formal_or_proof_help',
                'funfun',
            ),
            (
                [strategy_outcome_message('code', selected_outcome='code_simulation_or_primitive_capability_received')],
                'request_code_simulation_or_primitive_capability_help',
                'code_module',
            ),
            (
                [strategy_outcome_message('language', selected_outcome='language_terminology_or_protocol_clarification_received')],
                'request_language_terminology_or_protocol_help',
                'language_model_2',
            ),
            (
                [strategy_outcome_message('repair', selected_outcome='theory_repair_cycle_reopened')],
                'reopen_theory_repair_cycle',
                'orchestrator',
            ),
            (
                [strategy_outcome_message('close', selected_outcome='stable_science_campaign_closed')],
                'close_stable_science_campaign_cycle',
                'orchestrator',
            ),
            (
                [strategy_outcome_message('schedule', selected_outcome='next_frontier_hypothesis_scheduled')],
                'schedule_next_frontier_hypothesis_campaign',
                'orchestrator',
            ),
            (
                [strategy_outcome_message('nogain', selected_outcome='no_measurable_science_strategy_gain')],
                'record_no_measurable_science_cycle_gain',
                'orchestrator',
            ),
        ]
        for messages, strategy, recipient in cases:
            with self.subTest(strategy=strategy):
                ledger, message = build_once(messages)
                self.assertEqual(strategy, ledger['latest']['selected_strategy'])
                self.assertEqual(recipient, message['recipient'])

    def test_ledger_sources_cycle_memory_and_boundary_guards(self):
        ledger, message = build_once([], strategy_outcome_ledger=strategy_outcome_ledger_fixture())
        self.assertEqual('promote_validated_hypothesis_strategy_to_campaign_cycle_memory', ledger['latest']['selected_strategy'])
        self.assertEqual('broadcast', message['recipient'])

        memory_ledger, memory_message = build_once([], cycle_memory_ledger=cycle_memory_ledger_fixture())
        self.assertEqual('promote_validated_hypothesis_strategy_to_campaign_cycle_memory', memory_ledger['latest']['selected_strategy'])
        self.assertEqual('broadcast', memory_message['recipient'])

        repair_ledger, repair_message = build_once([strategy_outcome_message('leak', leak=True)])
        self.assertEqual('preserve_checkpoint_boundary', repair_ledger['latest']['selected_strategy'])
        self.assertEqual('code_module', repair_message['recipient'])
        self.assertIn('gravity', repair_message['body']['label_leaks'])

        third_party_ledger, third_party_message = build_once(
            [strategy_outcome_message('third-party')],
            project_boundary={'third_party_checkpoint_used': True},
        )
        self.assertEqual('preserve_checkpoint_boundary', third_party_ledger['latest']['selected_strategy'])
        self.assertIn('no local-owned checkpoint claim', third_party_message['body']['no_overclaiming_proof'])
        self.assertFalse(third_party_message['body']['project_owned_checkpoint_claimed'])

    def test_duplicate_idempotence_and_appended_evidence_update(self):
        first_ledger, first_message = build_once([
            strategy_outcome_message('waiting', selected_outcome='no_measurable_science_strategy_gain'),
        ])
        self.assertEqual('record_no_measurable_science_cycle_gain', first_ledger['latest']['selected_strategy'])
        self.assertIsNotNone(first_message)

        repeat_ledger, repeat_message = build_once([
            strategy_outcome_message('waiting', selected_outcome='no_measurable_science_strategy_gain'),
        ], ledger=first_ledger)
        self.assertEqual('summarize_noop', repeat_ledger['latest']['selected_strategy'])
        self.assertIsNone(repeat_message)

        appended_ledger, appended_message = build_once([
            strategy_outcome_message('waiting', selected_outcome='no_measurable_science_strategy_gain'),
            strategy_outcome_message('promoted-new', selected_outcome='validated_hypothesis_promoted_to_theory_memory'),
        ], ledger=repeat_ledger)
        self.assertEqual('promote_validated_hypothesis_strategy_to_campaign_cycle_memory', appended_ledger['latest']['selected_strategy'])
        self.assertIsNotNone(appended_message)

    def test_cli_status_proof_preserves_runtime_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            runtime = tmp / 'theory-memory.json'
            runtime.write_text(json.dumps(memory_fixture(), sort_keys=True), encoding='utf-8')
            transcript = tmp / 'cycle-strategy.jsonl'
            transcript.write_text(json.dumps(strategy_outcome_message(), sort_keys=True) + '\n', encoding='utf-8')
            ledger_path = tmp / 'cycle-strategy-ledger.json'
            outbox = tmp / 'outbox.jsonl'

            before = runtime.read_text(encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()):
                result = run_science_campaign_cycle_strategy_planner(
                    runtime_memory_path=runtime,
                    transcript_file=transcript,
                    strategy_outcome_ledger_file=tmp / 'missing-strategy-outcome.json',
                    strategy_ledger_file=tmp / 'missing-strategy.json',
                    frontier_outcome_ledger_file=tmp / 'missing-frontier-outcome.json',
                    frontier_ledger_file=tmp / 'missing-frontier.json',
                    campaign_ledger_file=tmp / 'missing-campaign.json',
                    cycle_strategy_ledger_file=ledger_path,
                    outbox_file=outbox,
                    memory_data=memory_fixture(),
                    git_status_text=' M tmp/theory-memory.json\n',
                    git_ignored_text='',
                )
            with contextlib.redirect_stdout(io.StringIO()):
                repeat = run_science_campaign_cycle_strategy_planner(
                    runtime_memory_path=runtime,
                    transcript_file=transcript,
                    strategy_outcome_ledger_file=tmp / 'missing-strategy-outcome.json',
                    strategy_ledger_file=tmp / 'missing-strategy.json',
                    frontier_outcome_ledger_file=tmp / 'missing-frontier-outcome.json',
                    frontier_ledger_file=tmp / 'missing-frontier.json',
                    campaign_ledger_file=tmp / 'missing-campaign.json',
                    cycle_strategy_ledger_file=ledger_path,
                    outbox_file=outbox,
                    memory_data=memory_fixture(),
                    git_status_text=' M tmp/theory-memory.json\n',
                    git_ignored_text='',
                )
            after = runtime.read_text(encoding='utf-8')

        self.assertEqual(before, after)
        self.assertEqual('promote_validated_hypothesis_strategy_to_campaign_cycle_memory', result['selected_strategy'])
        self.assertEqual(1, result['outbox_count'])
        self.assertEqual('summarize_noop', repeat['selected_strategy'])
        self.assertEqual(0, repeat['outbox_count'])
        self.assertFalse(result['runtime_memory_mutated'])
        self.assertEqual([], result['label_leaks'])
        self.assertTrue(result['no_sibling_imports'])
        self.assertFalse(result['project_owned_checkpoint_claimed'])

    def test_no_sibling_imports(self):
        module = Path(PROJECT_DIR) / 'agent' / 'science_campaign_cycle_strategy.py'
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
