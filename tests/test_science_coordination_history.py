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

from agent.module_chat_adapter import build_module_chat_message, validate_participant  # noqa: E402
from agent.science_campaign_cycle_strategy_outcome import SCIENCE_CAMPAIGN_CYCLE_STRATEGY_OUTCOME_LEDGER_KIND  # noqa: E402
from agent.science_coordination_history import (  # noqa: E402
    SCIENCE_COORDINATION_HISTORY_LEDGER_KIND,
    build_science_coordination_history_export,
    empty_science_coordination_history_ledger,
    load_science_coordination_history_ledger,
    read_science_coordination_history_transcript,
    validate_science_coordination_history_ledger,
    write_science_coordination_history_ledger,
    write_science_coordination_history_outbox_jsonl,
)
from main import run_science_coordination_history_exporter  # noqa: E402


def memory_fixture():
    return {
        'discovery_readiness': {'readiness_score': 0.97, 'status': 'history_ready'},
        'abstraction_discovery_evidence': {'transfer_outcome_count': 23},
    }


def cycle_outcome_message(
    scenario_id='scenario-one',
    *,
    selected_outcome='campaign_cycle_memory_promoted',
    leak=False,
    third_party=False,
    rework_noop_count=0,
):
    body = {
        'response_kind': 'science_campaign_cycle_strategy_outcome',
        'campaign_cycle_strategy_outcome_id': f'cycle-outcome-{scenario_id}',
        'campaign_cycle_strategy_id': f'cycle-strategy-{scenario_id}',
        'scenario_id': scenario_id,
        'hypothesis_id': f'hypothesis:{scenario_id}',
        'campaign_id': f'campaign-{scenario_id}',
        'family_id': f'family-{scenario_id}',
        'selected_outcome': selected_outcome,
        'selected_action': selected_outcome,
        'interaction_sequence': ['cycle_strategy', 'cycle_strategy_outcome'],
        'involved_modules': ['ai_different'],
        'source_commits': ['9638ea274058de48bac8b8bc5bd778f1b3fc444d'],
        'source_tests': ['tests.test_science_campaign_cycle_strategy_outcome:7'],
        'source_evidence_used': [{'source': 'cycle_strategy_outcome', 'status': 'assessed'}],
        'observed_sibling_evidence': [],
        'checkpoint_boundary_state': 'repair' if selected_outcome == 'preserve_checkpoint_boundary' else 'clean',
        'checkpoint_boundary_notes': ['boundary preserved'] if selected_outcome == 'preserve_checkpoint_boundary' else [],
        'waiting_blocker_state': 'waiting' if 'waiting' in selected_outcome else 'resolved',
        'rework_noop_count': rework_noop_count,
        'third_party_checkpoint_used': third_party,
    }
    if leak:
        body['checkpoint_boundary_notes'] = ['gravity label leaked']
    return build_module_chat_message(
        sender='ai_different',
        recipient='orchestrator',
        topic='ai_different.science_campaign_cycle_strategy_outcome',
        body=body,
        evidence={'scenario_id': scenario_id, 'selected_outcome': selected_outcome},
        tags=['science_campaign_cycle_strategy_outcome'],
    )


def sibling_message(
    scenario_id='scenario-one',
    *,
    sender='code_module',
    gate='simulation_help',
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
            'source_commits': ['sibling-proof-commit'],
            'source_tests': [f'{sender}:{gate}:passed'],
        },
        evidence={'scenario_id': scenario_id, 'evidence_gate': gate, 'status': status},
        tags=['science_coordination_history', gate, status],
    )


def cycle_outcome_ledger_fixture(*, scenario_id='scenario-one', selected_outcome='campaign_cycle_memory_promoted'):
    return {
        'schema_version': 1,
        'ledger_kind': SCIENCE_CAMPAIGN_CYCLE_STRATEGY_OUTCOME_LEDGER_KIND,
        'processed_message_ids': [],
        'processed_source_hashes': [],
        'assessed_cycle_strategy_keys': [],
        'cycle_strategy_rows': [],
        'outcome_records': [{
            'campaign_cycle_strategy_outcome_id': f'cycle-outcome-{scenario_id}',
            'campaign_cycle_strategy_id': f'cycle-strategy-{scenario_id}',
            'hypothesis_id': f'hypothesis:{scenario_id}',
            'campaign_id': f'campaign-{scenario_id}',
            'family_id': f'family-{scenario_id}',
            'scenario_id': scenario_id,
            'selected_outcome': selected_outcome,
            'source_evidence_used': [{'source': 'outcome_ledger', 'status': 'recorded'}],
            'observed_sibling_evidence': [],
            'checkpoint_boundary_notes': [],
        }],
        'outgoing_response_ids': [],
        'latest': {},
        'ledger_hash': f'cycle-outcome-{scenario_id}-hash',
    }


def build_once(messages, *, ledger=None, cycle_outcome_ledger=None, project_boundary=None):
    updated, message = build_science_coordination_history_export(
        transcript_messages=messages,
        history_ledger=ledger or empty_science_coordination_history_ledger(),
        cycle_strategy_outcome_ledger=cycle_outcome_ledger or {},
        cycle_strategy_ledger={},
        strategy_outcome_ledger={},
        strategy_ledger={},
        frontier_outcome_ledger={},
        frontier_ledger={},
        theory_memory_ledger={},
        campaign_cycle_memory_ledger={},
        hypothesis_ledger={},
        campaign_ledger={},
        experiment_ledger={},
        module_chat_ledger={},
        prior_history_ledger={},
        runtime_memory_data=memory_fixture(),
        runtime_memory_hash_state={'unchanged': True, 'before': 'same', 'after': 'same'},
        project_owned_boundary=project_boundary or {'third_party_checkpoint_used': False},
    )
    return updated, message


class ScienceCoordinationHistoryTests(unittest.TestCase):
    def test_history_participant_and_ledger_persistence_load_rejection(self):
        self.assertEqual('history', validate_participant('history'))
        ledger, message = build_once([cycle_outcome_message()])
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = Path(tmpdir) / 'history.json'
            outbox = Path(tmpdir) / 'outbox.jsonl'
            write_science_coordination_history_ledger(ledger_path, ledger)
            loaded = load_science_coordination_history_ledger(ledger_path)
            write_science_coordination_history_outbox_jsonl(outbox, message)
            rows = [json.loads(line) for line in outbox.read_text(encoding='utf-8').splitlines()]

        self.assertEqual(SCIENCE_COORDINATION_HISTORY_LEDGER_KIND, loaded['ledger_kind'])
        self.assertEqual('history', rows[0]['recipient'])
        self.assertEqual('science_coordination_history', rows[0]['body']['response_kind'])
        with self.assertRaisesRegex(ValueError, 'wrong ledger_kind'):
            validate_science_coordination_history_ledger({'ledger_kind': 'wrong'})

    def test_science_outcome_to_history_extraction_and_invalid_record(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            transcript = Path(tmpdir) / 'history.jsonl'
            transcript.write_text(
                json.dumps(cycle_outcome_message(), sort_keys=True) + '\n'
                + json.dumps(sibling_message(sender='funfun', gate='formal_proof'), sort_keys=True) + '\n'
                + '{"sender":"bad"}\n',
                encoding='utf-8',
            )
            parsed = read_science_coordination_history_transcript(transcript)
        ledger, _ = build_once(parsed['messages'])

        self.assertEqual(2, len(parsed['messages']))
        self.assertEqual(1, len(parsed['invalid_messages']))
        row = ledger['history_rows'][0]
        self.assertEqual('scenario-one', row['scenario_id'])
        self.assertEqual('validated_hypothesis_cycle_memory_payoff', ledger['latest']['payoff_class'])

    def test_deterministic_priority_selection(self):
        cases = [
            ([cycle_outcome_message('boundary', selected_outcome='preserve_checkpoint_boundary')], 'preserve_checkpoint_boundary'),
            ([cycle_outcome_message('memory')], 'validated_hypothesis_cycle_memory_payoff'),
            (
                [
                    cycle_outcome_message('code', selected_outcome='planned_science_campaign_cycle_strategy_waiting_for_evidence'),
                    sibling_message('code', sender='code_module', gate='primitive_simulation_help'),
                ],
                'code_simulation_before_science_campaign_paid_off',
            ),
            (
                [
                    cycle_outcome_message('funfun', selected_outcome='planned_science_campaign_cycle_strategy_waiting_for_evidence'),
                    sibling_message('funfun', sender='funfun', gate='formal_proof_certificate'),
                ],
                'funfun_formalization_before_science_campaign_paid_off',
            ),
            (
                [
                    cycle_outcome_message('language', selected_outcome='planned_science_campaign_cycle_strategy_waiting_for_evidence'),
                    sibling_message('language', sender='language_model_2', gate='protocol_clarification'),
                ],
                'language_terminology_before_science_campaign_paid_off',
            ),
            ([cycle_outcome_message('waiting', selected_outcome='planned_science_campaign_cycle_strategy_waiting_for_evidence')], 'sibling_request_waiting_for_evidence'),
            ([cycle_outcome_message('noop', selected_outcome='summarize_noop', rework_noop_count=2)], 'repeated_no_gain_or_noop_loop'),
            ([cycle_outcome_message('repair', selected_outcome='theory_repair_cycle_reopened')], 'theory_repair_loop_helped'),
            ([cycle_outcome_message('closed', selected_outcome='stable_science_campaign_cycle_closed')], 'stable_science_campaign_cycle_closed'),
            ([cycle_outcome_message('frontier', selected_outcome='next_frontier_hypothesis_campaign_scheduled')], 'next_frontier_hypothesis_campaign_helped'),
            ([cycle_outcome_message('nogain', selected_outcome='no_measurable_science_campaign_cycle_strategy_gain')], 'no_measurable_coordination_payoff'),
        ]
        for messages, payoff in cases:
            with self.subTest(payoff=payoff):
                ledger, message = build_once(messages)
                self.assertEqual(payoff, ledger['latest']['payoff_class'])
                self.assertEqual('history', message['recipient'])

    def test_ledger_sources_boundary_and_no_overclaiming(self):
        ledger, message = build_once([], cycle_outcome_ledger=cycle_outcome_ledger_fixture())
        self.assertEqual('validated_hypothesis_cycle_memory_payoff', ledger['latest']['payoff_class'])
        self.assertEqual('history', message['recipient'])

        repair_ledger, repair_message = build_once([cycle_outcome_message('leak', leak=True)])
        self.assertEqual('preserve_checkpoint_boundary', repair_ledger['latest']['payoff_class'])
        self.assertIn('gravity', repair_message['body']['label_leaks'])

        third_party_ledger, third_party_message = build_once(
            [cycle_outcome_message('third-party')],
            project_boundary={'third_party_checkpoint_used': True},
        )
        self.assertEqual('preserve_checkpoint_boundary', third_party_ledger['latest']['payoff_class'])
        self.assertIn('no local-owned checkpoint claim', third_party_message['body']['no_overclaiming_proof'])
        self.assertFalse(third_party_message['body']['project_owned_checkpoint_claimed'])

    def test_duplicate_idempotence_and_appended_evidence_update(self):
        first_ledger, first_message = build_once([
            cycle_outcome_message('waiting', selected_outcome='planned_science_campaign_cycle_strategy_waiting_for_evidence'),
        ])
        self.assertEqual('sibling_request_waiting_for_evidence', first_ledger['latest']['payoff_class'])
        self.assertIsNotNone(first_message)

        repeat_ledger, repeat_message = build_once([
            cycle_outcome_message('waiting', selected_outcome='planned_science_campaign_cycle_strategy_waiting_for_evidence'),
        ], ledger=first_ledger)
        self.assertEqual('summarize_noop', repeat_ledger['latest']['payoff_class'])
        self.assertIsNone(repeat_message)

        appended_ledger, appended_message = build_once([
            cycle_outcome_message('waiting', selected_outcome='planned_science_campaign_cycle_strategy_waiting_for_evidence'),
            sibling_message('waiting', sender='code_module', gate='simulation_counterexample'),
        ], ledger=repeat_ledger)
        self.assertEqual('code_simulation_before_science_campaign_paid_off', appended_ledger['latest']['payoff_class'])
        self.assertIsNotNone(appended_message)

    def test_cli_status_proof_preserves_runtime_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            runtime = tmp / 'theory-memory.json'
            runtime.write_text(json.dumps(memory_fixture(), sort_keys=True), encoding='utf-8')
            transcript = tmp / 'history.jsonl'
            transcript.write_text(json.dumps(cycle_outcome_message(), sort_keys=True) + '\n', encoding='utf-8')
            ledger_path = tmp / 'history-ledger.json'
            outbox = tmp / 'outbox.jsonl'

            before = runtime.read_text(encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()):
                result = run_science_coordination_history_exporter(
                    runtime_memory_path=runtime,
                    transcript_file=transcript,
                    cycle_strategy_outcome_ledger_file=tmp / 'missing-cycle-outcome.json',
                    cycle_strategy_ledger_file=tmp / 'missing-cycle-strategy.json',
                    strategy_outcome_ledger_file=tmp / 'missing-strategy-outcome.json',
                    strategy_ledger_file=tmp / 'missing-strategy.json',
                    frontier_outcome_ledger_file=tmp / 'missing-frontier-outcome.json',
                    frontier_ledger_file=tmp / 'missing-frontier.json',
                    history_ledger_file=ledger_path,
                    outbox_file=outbox,
                    memory_data=memory_fixture(),
                    git_status_text=' M tmp/theory-memory.json\n',
                    git_ignored_text='',
                )
            with contextlib.redirect_stdout(io.StringIO()):
                repeat = run_science_coordination_history_exporter(
                    runtime_memory_path=runtime,
                    transcript_file=transcript,
                    cycle_strategy_outcome_ledger_file=tmp / 'missing-cycle-outcome.json',
                    cycle_strategy_ledger_file=tmp / 'missing-cycle-strategy.json',
                    strategy_outcome_ledger_file=tmp / 'missing-strategy-outcome.json',
                    strategy_ledger_file=tmp / 'missing-strategy.json',
                    frontier_outcome_ledger_file=tmp / 'missing-frontier-outcome.json',
                    frontier_ledger_file=tmp / 'missing-frontier.json',
                    history_ledger_file=ledger_path,
                    outbox_file=outbox,
                    memory_data=memory_fixture(),
                    git_status_text=' M tmp/theory-memory.json\n',
                    git_ignored_text='',
                )
            after = runtime.read_text(encoding='utf-8')

        self.assertEqual(before, after)
        self.assertEqual('validated_hypothesis_cycle_memory_payoff', result['payoff_class'])
        self.assertEqual(1, result['outbox_count'])
        self.assertEqual('summarize_noop', repeat['payoff_class'])
        self.assertEqual(0, repeat['outbox_count'])
        self.assertFalse(result['runtime_memory_mutated'])
        self.assertEqual([], result['label_leaks'])
        self.assertTrue(result['no_sibling_imports'])
        self.assertFalse(result['project_owned_checkpoint_claimed'])

    def test_no_sibling_imports(self):
        module = Path(PROJECT_DIR) / 'agent' / 'science_coordination_history.py'
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
