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
from agent.science_campaign_strategy import SCIENCE_CAMPAIGN_STRATEGY_LEDGER_KIND  # noqa: E402
from agent.science_campaign_strategy_outcome import (  # noqa: E402
    SCIENCE_CAMPAIGN_STRATEGY_OUTCOME_LEDGER_KIND,
    build_science_campaign_strategy_outcome_assessment,
    empty_science_campaign_strategy_outcome_ledger,
    load_science_campaign_strategy_outcome_ledger,
    read_science_campaign_strategy_outcome_transcript,
    validate_science_campaign_strategy_outcome_ledger,
    write_science_campaign_strategy_outcome_ledger,
    write_science_campaign_strategy_outcome_outbox_jsonl,
)
from main import run_science_campaign_strategy_outcome_assessor  # noqa: E402


def memory_fixture():
    return {
        'discovery_readiness': {'readiness_score': 0.95, 'status': 'nearly_ready'},
        'abstraction_discovery_evidence': {'transfer_outcome_count': 13},
    }


def strategy_message(
    scenario_id='scenario-one',
    *,
    strategy_id=None,
    strategy='promote_validated_hypothesis_to_theory_memory',
    leak=False,
    third_party=False,
):
    strategy_id = strategy_id or f'strategy-{scenario_id}'
    body = {
        'response_kind': 'science_campaign_strategy',
        'strategy_id': strategy_id,
        'frontier_outcome_id': f'frontier-outcome-{scenario_id}',
        'frontier_id': f'frontier-{scenario_id}',
        'source_outcome_id': f'outcome-{scenario_id}',
        'scenario_id': scenario_id,
        'hypothesis_id': f'hypothesis:{scenario_id}',
        'campaign_id': f'campaign-{scenario_id}',
        'selected_strategy': strategy,
        'selected_action': strategy,
        'selected_recipient': 'broadcast',
        'source_evidence_used': [{'source': 'strategy', 'status': 'planned'}],
        'sibling_evidence_used': [],
        'promotion_state': 'promoted' if strategy == 'promote_validated_hypothesis_to_theory_memory' else 'none',
        'retirement_state': 'retired' if strategy == 'retire_weak_hypothesis_route' else 'none',
        'repair_state': 'open' if strategy == 'reopen_theory_repair_cycle' else 'none',
        'closure_state': 'closed' if strategy == 'close_stable_science_campaign' else 'none',
        'before_theory_memory_state': 'not_recorded',
        'after_theory_memory_state': 'recorded' if strategy == 'promote_validated_hypothesis_to_theory_memory' else 'unknown',
        'checkpoint_boundary_state': 'repair' if strategy == 'preserve_checkpoint_boundary' else 'clean',
        'checkpoint_boundary_notes': ['repair boundary'] if strategy == 'preserve_checkpoint_boundary' else [],
        'third_party_checkpoint_used': third_party,
    }
    if leak:
        body['checkpoint_boundary_notes'] = ['gravity label leaked']
    return build_module_chat_message(
        sender='ai_different',
        recipient='orchestrator',
        topic='ai_different.science_campaign_strategy',
        body=body,
        evidence={'strategy_id': strategy_id, 'scenario_id': scenario_id, 'selected_strategy': strategy},
        tags=['science_campaign_strategy'],
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
            'evidence_gate': gate,
            'status': status,
        },
        evidence={'scenario_id': scenario_id, 'evidence_gate': gate, 'status': status},
        tags=['science_campaign_strategy_outcome', gate, status],
    )


def strategy_ledger_fixture(strategy='promote_validated_hypothesis_to_theory_memory', *, scenario_id='scenario-one'):
    return {
        'schema_version': 1,
        'ledger_kind': SCIENCE_CAMPAIGN_STRATEGY_LEDGER_KIND,
        'processed_message_ids': [],
        'processed_source_hashes': [],
        'planned_strategy_keys': [],
        'strategy_rows': [],
        'strategy_records': [{
            'strategy_id': f'strategy-{scenario_id}',
            'frontier_outcome_id': f'frontier-outcome-{scenario_id}',
            'frontier_id': f'frontier-{scenario_id}',
            'source_outcome_id': f'outcome-{scenario_id}',
            'hypothesis_id': f'hypothesis:{scenario_id}',
            'campaign_id': f'campaign-{scenario_id}',
            'scenario_id': scenario_id,
            'selected_strategy': strategy,
            'selected_action': strategy,
            'requested_sibling_evidence': [],
            'promotion_state': 'promoted' if strategy == 'promote_validated_hypothesis_to_theory_memory' else 'none',
            'retirement_state': 'none',
            'repair_state': 'none',
            'closure_state': 'none',
            'before_theory_memory_state': 'not_recorded',
            'after_theory_memory_state': 'recorded',
            'checkpoint_boundary_notes': [],
        }],
        'outgoing_response_ids': [],
        'latest': {},
        'ledger_hash': f'strategy-{scenario_id}-hash',
    }


def theory_memory_ledger_fixture(*, scenario_id='scenario-one'):
    return {
        'schema_version': 1,
        'ledger_kind': 'ai_different.symbolic_theory_memory',
        'theory_memory_rows': [{
            'strategy_id': f'strategy-{scenario_id}',
            'scenario_id': scenario_id,
            'hypothesis_id': f'hypothesis:{scenario_id}',
            'campaign_id': f'campaign-{scenario_id}',
            'status': 'recorded',
            'source': 'theory_memory',
        }],
        'ledger_hash': f'theory-memory-{scenario_id}-hash',
    }


def build_once(messages, *, ledger=None, strategy_ledger=None, theory_memory_ledger=None, project_boundary=None):
    updated, message = build_science_campaign_strategy_outcome_assessment(
        transcript_messages=messages,
        strategy_outcome_ledger=ledger or empty_science_campaign_strategy_outcome_ledger(),
        strategy_ledger=strategy_ledger or {},
        frontier_outcome_ledger={},
        frontier_ledger={},
        campaign_ledger={},
        theory_memory_ledger=theory_memory_ledger or {},
        hypothesis_ledger={},
        experiment_ledger={},
        module_chat_ledger={},
        prior_strategy_outcome_ledger={},
        runtime_memory_data=memory_fixture(),
        runtime_memory_hash_state={'unchanged': True, 'before': 'same', 'after': 'same'},
        project_owned_boundary=project_boundary or {'third_party_checkpoint_used': False},
    )
    return updated, message


class ScienceCampaignStrategyOutcomeTests(unittest.TestCase):
    def test_outcome_ledger_persistence_load_and_malformed_rejection(self):
        ledger, message = build_once([strategy_message()])
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = Path(tmpdir) / 'strategy-outcome.json'
            outbox = Path(tmpdir) / 'outbox.jsonl'
            write_science_campaign_strategy_outcome_ledger(ledger_path, ledger)
            loaded = load_science_campaign_strategy_outcome_ledger(ledger_path)
            write_science_campaign_strategy_outcome_outbox_jsonl(outbox, message)
            rows = [json.loads(line) for line in outbox.read_text(encoding='utf-8').splitlines()]

        self.assertEqual(SCIENCE_CAMPAIGN_STRATEGY_OUTCOME_LEDGER_KIND, loaded['ledger_kind'])
        self.assertEqual('science_campaign_strategy_outcome', rows[0]['body']['response_kind'])
        with self.assertRaisesRegex(ValueError, 'wrong ledger_kind'):
            validate_science_campaign_strategy_outcome_ledger({'ledger_kind': 'wrong'})

    def test_strategy_to_outcome_extraction_and_invalid_record(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            transcript = Path(tmpdir) / 'strategy-outcome.jsonl'
            transcript.write_text(
                json.dumps(strategy_message(), sort_keys=True) + '\n'
                + json.dumps(sibling_evidence(), sort_keys=True) + '\n'
                + '{"sender":"bad"}\n',
                encoding='utf-8',
            )
            parsed = read_science_campaign_strategy_outcome_transcript(transcript)
        ledger, _ = build_once(parsed['messages'])

        self.assertEqual(2, len(parsed['messages']))
        self.assertEqual(1, len(parsed['invalid_messages']))
        row = ledger['strategy_rows'][0]
        self.assertEqual('scenario-one', row['scenario_id'])
        self.assertEqual('validated_hypothesis_promoted_to_theory_memory', ledger['latest']['selected_outcome'])

    def test_deterministic_priority_selection_and_routing(self):
        cases = [
            ([strategy_message('boundary', strategy='preserve_checkpoint_boundary')], 'preserve_checkpoint_boundary', 'code_module'),
            ([strategy_message('promote')], 'validated_hypothesis_promoted_to_theory_memory', 'broadcast'),
            ([strategy_message('retire', strategy='retire_weak_hypothesis_route')], 'weak_hypothesis_route_retired', 'broadcast'),
            (
                [
                    strategy_message('funfun', strategy='request_funfun_formal_or_proof_clarification'),
                    sibling_evidence('funfun', sender='funfun', gate='formal_proof'),
                ],
                'funfun_formal_or_proof_clarification_received',
                'broadcast',
            ),
            (
                [
                    strategy_message('code', strategy='request_code_simulation_or_primitive_capability'),
                    sibling_evidence('code', sender='code_module', gate='primitive_capability'),
                ],
                'code_simulation_or_primitive_capability_received',
                'broadcast',
            ),
            (
                [
                    strategy_message('language', strategy='request_language_terminology_or_protocol_clarification'),
                    sibling_evidence('language', sender='language_model_2', gate='protocol_clarification'),
                ],
                'language_terminology_or_protocol_clarification_received',
                'broadcast',
            ),
            ([strategy_message('repair', strategy='reopen_theory_repair_cycle')], 'theory_repair_cycle_reopened', 'orchestrator'),
            ([strategy_message('close', strategy='close_stable_science_campaign')], 'stable_science_campaign_closed', 'orchestrator'),
            ([strategy_message('schedule', strategy='schedule_next_frontier_hypothesis')], 'next_frontier_hypothesis_scheduled', 'orchestrator'),
            ([strategy_message('nogain', strategy='record_no_measurable_science_gain')], 'no_measurable_science_strategy_gain', 'orchestrator'),
        ]
        for messages, outcome, recipient in cases:
            with self.subTest(outcome=outcome):
                ledger, message = build_once(messages)
                self.assertEqual(outcome, ledger['latest']['selected_outcome'])
                self.assertEqual(recipient, message['recipient'])

    def test_ledger_sources_theory_memory_and_boundary_guards(self):
        ledger, message = build_once([], strategy_ledger=strategy_ledger_fixture())
        self.assertEqual('validated_hypothesis_promoted_to_theory_memory', ledger['latest']['selected_outcome'])
        self.assertEqual('broadcast', message['recipient'])

        memory_ledger, memory_message = build_once([], theory_memory_ledger=theory_memory_ledger_fixture())
        self.assertEqual('validated_hypothesis_promoted_to_theory_memory', memory_ledger['latest']['selected_outcome'])
        self.assertEqual('broadcast', memory_message['recipient'])

        repair_ledger, repair_message = build_once([strategy_message('leak', leak=True)])
        self.assertEqual('preserve_checkpoint_boundary', repair_ledger['latest']['selected_outcome'])
        self.assertEqual('code_module', repair_message['recipient'])
        self.assertIn('gravity', repair_message['body']['label_leaks'])

        third_party_ledger, third_party_message = build_once(
            [strategy_message('third-party')],
            project_boundary={'third_party_checkpoint_used': True},
        )
        self.assertEqual('preserve_checkpoint_boundary', third_party_ledger['latest']['selected_outcome'])
        self.assertIn('no local-owned checkpoint claim', third_party_message['body']['no_overclaiming_proof'])
        self.assertFalse(third_party_message['body'].get('project_owned_checkpoint_claimed', False))

    def test_duplicate_idempotence_and_appended_evidence_update(self):
        first_ledger, first_message = build_once([
            strategy_message('waiting', strategy='request_funfun_formal_or_proof_clarification'),
        ])
        self.assertEqual('planned_science_campaign_strategy_waiting_for_evidence', first_ledger['latest']['selected_outcome'])
        self.assertIsNotNone(first_message)

        repeat_ledger, repeat_message = build_once([
            strategy_message('waiting', strategy='request_funfun_formal_or_proof_clarification'),
        ], ledger=first_ledger)
        self.assertEqual('summarize_noop', repeat_ledger['latest']['selected_outcome'])
        self.assertIsNone(repeat_message)

        appended_ledger, appended_message = build_once([
            strategy_message('waiting', strategy='request_funfun_formal_or_proof_clarification'),
            sibling_evidence('waiting', sender='funfun', gate='formal_proof'),
        ], ledger=repeat_ledger)
        self.assertEqual('funfun_formal_or_proof_clarification_received', appended_ledger['latest']['selected_outcome'])
        self.assertIsNotNone(appended_message)

    def test_cli_status_proof_preserves_runtime_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            runtime = tmp / 'theory-memory.json'
            runtime.write_text(json.dumps(memory_fixture(), sort_keys=True), encoding='utf-8')
            transcript = tmp / 'strategy-outcome.jsonl'
            transcript.write_text(json.dumps(strategy_message(), sort_keys=True) + '\n', encoding='utf-8')
            ledger_path = tmp / 'strategy-outcome-ledger.json'
            outbox = tmp / 'outbox.jsonl'

            before = runtime.read_text(encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()):
                result = run_science_campaign_strategy_outcome_assessor(
                    runtime_memory_path=runtime,
                    transcript_file=transcript,
                    strategy_ledger_file=tmp / 'missing-strategy.json',
                    frontier_outcome_ledger_file=tmp / 'missing-frontier-outcome.json',
                    frontier_ledger_file=tmp / 'missing-frontier.json',
                    campaign_ledger_file=tmp / 'missing-campaign.json',
                    strategy_outcome_ledger_file=ledger_path,
                    outbox_file=outbox,
                    memory_data=memory_fixture(),
                    git_status_text=' M tmp/theory-memory.json\n',
                    git_ignored_text='',
                )
            with contextlib.redirect_stdout(io.StringIO()):
                repeat = run_science_campaign_strategy_outcome_assessor(
                    runtime_memory_path=runtime,
                    transcript_file=transcript,
                    strategy_ledger_file=tmp / 'missing-strategy.json',
                    frontier_outcome_ledger_file=tmp / 'missing-frontier-outcome.json',
                    frontier_ledger_file=tmp / 'missing-frontier.json',
                    campaign_ledger_file=tmp / 'missing-campaign.json',
                    strategy_outcome_ledger_file=ledger_path,
                    outbox_file=outbox,
                    memory_data=memory_fixture(),
                    git_status_text=' M tmp/theory-memory.json\n',
                    git_ignored_text='',
                )
            after = runtime.read_text(encoding='utf-8')

        self.assertEqual(before, after)
        self.assertEqual('validated_hypothesis_promoted_to_theory_memory', result['selected_outcome'])
        self.assertEqual(1, result['outbox_count'])
        self.assertEqual('summarize_noop', repeat['selected_outcome'])
        self.assertEqual(0, repeat['outbox_count'])
        self.assertFalse(result['runtime_memory_mutated'])
        self.assertEqual([], result['label_leaks'])
        self.assertTrue(result['no_sibling_imports'])
        self.assertFalse(result['project_owned_checkpoint_claimed'])

    def test_no_sibling_imports(self):
        module = Path(PROJECT_DIR) / 'agent' / 'science_campaign_strategy_outcome.py'
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
