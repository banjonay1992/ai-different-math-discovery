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
from agent.science_theory_frontier import SCIENCE_THEORY_FRONTIER_LEDGER_KIND  # noqa: E402
from agent.science_theory_frontier_outcome import (  # noqa: E402
    SCIENCE_THEORY_FRONTIER_OUTCOME_LEDGER_KIND,
    build_science_theory_frontier_outcome_assessment,
    empty_science_theory_frontier_outcome_ledger,
    load_science_theory_frontier_outcome_ledger,
    read_science_theory_frontier_outcome_transcript,
    validate_science_theory_frontier_outcome_ledger,
    write_science_theory_frontier_outcome_ledger,
    write_science_theory_frontier_outcome_outbox_jsonl,
)
from main import run_science_theory_frontier_outcome_assessor  # noqa: E402


def memory_fixture():
    return {
        'discovery_readiness': {'readiness_score': 0.93, 'status': 'nearly_ready'},
        'abstraction_discovery_evidence': {'transfer_outcome_count': 11},
    }


def frontier_message(
    scenario_id='scenario-one',
    *,
    frontier_id=None,
    move='promote_supported_hypothesis_to_theory_memory',
    leak=False,
    third_party=False,
):
    frontier_id = frontier_id or f'frontier-{scenario_id}'
    body = {
        'response_kind': 'science_theory_frontier',
        'frontier_id': frontier_id,
        'outcome_id': f'outcome-{scenario_id}',
        'action_id': f'action-{scenario_id}',
        'benefit_id': f'benefit-{scenario_id}',
        'scenario_id': scenario_id,
        'hypothesis_id': f'hypothesis:{scenario_id}',
        'selected_theory_move': move,
        'selected_action': move,
        'selected_recipient': 'broadcast',
        'observed_sibling_evidence': [],
        'before_hypothesis_state': 'waiting',
        'after_hypothesis_state': 'blocked' if move == 'retire_or_block_refuted_hypothesis' else 'evidence_received',
        'theory_memory_delta': f'{move} delta',
        'campaign_frontier_delta': f'frontier -> {move}',
        'refinement_state': 'accepted' if move == 'refine_hypothesis_from_outcome' else 'none',
        'retirement_block_state': 'blocked' if move == 'retire_or_block_refuted_hypothesis' else 'none',
        'waiting_blocker_state': 'waiting' if move.startswith('request_') else 'resolved',
        'boundary_checkpoint_state': 'repair' if move == 'preserve_boundary_or_checkpoint_repair' else 'clean',
        'boundary_notes': ['repair boundary'] if move == 'preserve_boundary_or_checkpoint_repair' else [],
        'third_party_checkpoint_used': third_party,
    }
    if leak:
        body['boundary_notes'] = ['gravity label leaked']
    return build_module_chat_message(
        sender='ai_different',
        recipient='orchestrator',
        topic='ai_different.science_theory_frontier',
        body=body,
        evidence={'frontier_id': frontier_id, 'scenario_id': scenario_id, 'selected_theory_move': move},
        tags=['science_theory_frontier'],
    )


def sibling_evidence(
    scenario_id='scenario-one',
    *,
    sender='funfun',
    gate='math_proof',
    status='satisfied',
):
    return build_module_chat_message(
        sender=sender,
        recipient='ai_different',
        topic=f'{sender}.{gate}.{status}',
        body={
            'scenario_id': scenario_id,
            'hypothesis_id': f'hypothesis:{scenario_id}',
            'evidence_gate': gate,
            'status': status,
            'summary': f'{gate} {status}',
        },
        evidence={'scenario_id': scenario_id, 'evidence_gate': gate, 'status': status},
        tags=['science_theory_frontier_outcome', gate, status],
    )


def frontier_ledger_fixture(move='promote_supported_hypothesis_to_theory_memory', *, scenario_id='scenario-one'):
    return {
        'schema_version': 1,
        'ledger_kind': SCIENCE_THEORY_FRONTIER_LEDGER_KIND,
        'processed_message_ids': [],
        'processed_source_hashes': [],
        'planned_frontier_keys': [],
        'frontier_rows': [],
        'frontier_records': [{
            'frontier_id': f'frontier-{scenario_id}',
            'outcome_ids': [f'outcome-{scenario_id}'],
            'action_ids': [f'action-{scenario_id}'],
            'benefit_ids': [f'benefit-{scenario_id}'],
            'scenario_ids': [scenario_id],
            'hypothesis_ids': [f'hypothesis:{scenario_id}'],
            'selected_theory_move': move,
            'observed_sibling_evidence': [],
            'before_hypothesis_state': 'waiting',
            'after_hypothesis_state': 'evidence_received',
            'refinement_state': 'none',
            'retirement_block_state': 'none',
            'waiting_blocker_state': 'resolved',
            'boundary_checkpoint_state': 'clean',
        }],
        'outgoing_response_ids': [],
        'latest': {},
        'ledger_hash': f'frontier-{scenario_id}-hash',
    }


def theory_memory_ledger_fixture(*, scenario_id='scenario-one', status='recorded'):
    return {
        'schema_version': 1,
        'ledger_kind': 'ai_different.symbolic_theory_memory',
        'theory_memory_rows': [{
            'frontier_id': f'frontier-{scenario_id}',
            'outcome_id': f'outcome-{scenario_id}',
            'scenario_id': scenario_id,
            'hypothesis_id': f'hypothesis:{scenario_id}',
            'status': status,
            'source': 'theory_memory',
        }],
        'ledger_hash': f'theory-memory-{scenario_id}-hash',
    }


def build_once(messages, *, ledger=None, frontier_ledger=None, theory_memory_ledger=None, project_boundary=None):
    updated, message = build_science_theory_frontier_outcome_assessment(
        transcript_messages=messages,
        frontier_outcome_ledger=ledger or empty_science_theory_frontier_outcome_ledger(),
        frontier_ledger=frontier_ledger or {},
        action_outcome_ledger={},
        action_ledger={},
        benefit_ledger={},
        evaluator_ledger={},
        outcome_ledger={},
        contract_ledger={},
        adjudicator_ledger={},
        agenda_ledger={},
        lifecycle_ledger={},
        scorecard_ledger={},
        campaign_ledger={},
        campaign_outcome_ledger={},
        theory_memory_ledger=theory_memory_ledger or {},
        prior_frontier_outcome_ledger={},
        sibling_frontier_outcome_ledgers={},
        runtime_memory_data=memory_fixture(),
        runtime_memory_hash_state={'unchanged': True, 'before': 'same', 'after': 'same'},
        project_owned_boundary=project_boundary or {'third_party_checkpoint_used': False},
    )
    return updated, message


class ScienceTheoryFrontierOutcomeTests(unittest.TestCase):
    def test_outcome_ledger_persistence_load_and_malformed_rejection(self):
        ledger, message = build_once([frontier_message()])
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = Path(tmpdir) / 'frontier-outcome.json'
            outbox = Path(tmpdir) / 'outbox.jsonl'
            write_science_theory_frontier_outcome_ledger(ledger_path, ledger)
            loaded = load_science_theory_frontier_outcome_ledger(ledger_path)
            write_science_theory_frontier_outcome_outbox_jsonl(outbox, message)
            rows = [json.loads(line) for line in outbox.read_text(encoding='utf-8').splitlines()]

        self.assertEqual(SCIENCE_THEORY_FRONTIER_OUTCOME_LEDGER_KIND, loaded['ledger_kind'])
        self.assertEqual('science_theory_frontier_outcome', rows[0]['body']['response_kind'])
        with self.assertRaisesRegex(ValueError, 'wrong ledger_kind'):
            validate_science_theory_frontier_outcome_ledger({'ledger_kind': 'wrong'})

    def test_frontier_to_outcome_extraction_and_invalid_record(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            transcript = Path(tmpdir) / 'frontier-outcome.jsonl'
            transcript.write_text(
                json.dumps(frontier_message(), sort_keys=True) + '\n'
                + json.dumps(sibling_evidence(), sort_keys=True) + '\n'
                + '{"sender":"bad"}\n',
                encoding='utf-8',
            )
            parsed = read_science_theory_frontier_outcome_transcript(transcript)
        ledger, _ = build_once(parsed['messages'])

        self.assertEqual(2, len(parsed['messages']))
        self.assertEqual(1, len(parsed['invalid_messages']))
        row = ledger['frontier_rows'][0]
        self.assertEqual('scenario-one', row['scenario_id'])
        self.assertEqual('theory_memory_recorded_or_hypothesis_promoted', ledger['latest']['selected_outcome'])

    def test_deterministic_priority_selection_and_routing(self):
        cases = [
            (
                [frontier_message('repair', move='preserve_boundary_or_checkpoint_repair')],
                'preserve_boundary_or_checkpoint_repair',
                'code_module',
            ),
            (
                [frontier_message('promote')],
                'theory_memory_recorded_or_hypothesis_promoted',
                'broadcast',
            ),
            (
                [frontier_message('block', move='retire_or_block_refuted_hypothesis')],
                'refuted_hypothesis_retired_or_blocked',
                'broadcast',
            ),
            (
                [
                    frontier_message('math', move='request_funfun_certificate'),
                    sibling_evidence('math', sender='funfun', gate='math_proof'),
                ],
                'funfun_certificate_supports_or_blocks_theory_move',
                'broadcast',
            ),
            (
                [
                    frontier_message('code', move='request_code_experiment_or_counterexample'),
                    sibling_evidence('code', sender='code_module', gate='code_proof', status='failed'),
                ],
                'refuted_hypothesis_retired_or_blocked',
                'broadcast',
            ),
            (
                [
                    frontier_message('language', move='request_language_protocol_clarification'),
                    sibling_evidence('language', sender='language_model_2', gate='language_epoch_plan'),
                ],
                'language_protocol_clarification_resolves_theory_move',
                'broadcast',
            ),
            (
                [frontier_message('refine', move='refine_hypothesis_from_outcome')],
                'hypothesis_refinement_accepted',
                'broadcast',
            ),
            (
                [frontier_message('frontier', move='schedule_next_campaign_frontier_check')],
                'next_campaign_frontier_scheduled',
                'orchestrator',
            ),
            (
                [frontier_message('waiting', move='request_funfun_certificate')],
                'planned_theory_move_waiting_for_evidence',
                'orchestrator',
            ),
            (
                [frontier_message('nogain', move='record_no_measurable_theory_gain')],
                'no_measurable_theory_frontier_gain',
                'orchestrator',
            ),
        ]
        for messages, outcome, recipient in cases:
            with self.subTest(outcome=outcome):
                ledger, message = build_once(messages)
                self.assertEqual(outcome, ledger['latest']['selected_outcome'])
                self.assertEqual(recipient, message['recipient'])

    def test_ledger_sources_theory_memory_and_boundary_guards(self):
        ledger, message = build_once([], frontier_ledger=frontier_ledger_fixture())
        self.assertEqual('theory_memory_recorded_or_hypothesis_promoted', ledger['latest']['selected_outcome'])
        self.assertEqual('broadcast', message['recipient'])

        memory_ledger, memory_message = build_once([], theory_memory_ledger=theory_memory_ledger_fixture())
        self.assertEqual('theory_memory_recorded_or_hypothesis_promoted', memory_ledger['latest']['selected_outcome'])
        self.assertEqual('broadcast', memory_message['recipient'])

        repair_ledger, repair_message = build_once([frontier_message('leak', leak=True)])
        self.assertEqual('preserve_boundary_or_checkpoint_repair', repair_ledger['latest']['selected_outcome'])
        self.assertEqual('code_module', repair_message['recipient'])
        self.assertIn('gravity', repair_message['body']['label_leaks'])

        third_party_ledger, third_party_message = build_once(
            [frontier_message('third-party')],
            project_boundary={'third_party_checkpoint_used': True},
        )
        self.assertEqual('preserve_boundary_or_checkpoint_repair', third_party_ledger['latest']['selected_outcome'])
        self.assertFalse(third_party_message['body'].get('project_owned_checkpoint_claimed', False))

    def test_duplicate_idempotence_and_appended_evidence_update(self):
        first_ledger, first_message = build_once([
            frontier_message('waiting', move='request_funfun_certificate'),
        ])
        self.assertEqual('planned_theory_move_waiting_for_evidence', first_ledger['latest']['selected_outcome'])
        self.assertIsNotNone(first_message)

        repeat_ledger, repeat_message = build_once([
            frontier_message('waiting', move='request_funfun_certificate'),
        ], ledger=first_ledger)
        self.assertEqual('summarize_noop', repeat_ledger['latest']['selected_outcome'])
        self.assertIsNone(repeat_message)

        appended_ledger, appended_message = build_once([
            frontier_message('waiting', move='request_funfun_certificate'),
            sibling_evidence('waiting', sender='funfun', gate='math_proof'),
        ], ledger=repeat_ledger)
        self.assertEqual('funfun_certificate_supports_or_blocks_theory_move', appended_ledger['latest']['selected_outcome'])
        self.assertIsNotNone(appended_message)

    def test_cli_status_proof_preserves_runtime_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            runtime = tmp / 'theory-memory.json'
            runtime.write_text(json.dumps(memory_fixture(), sort_keys=True), encoding='utf-8')
            transcript = tmp / 'frontier-outcome.jsonl'
            transcript.write_text(json.dumps(frontier_message(), sort_keys=True) + '\n', encoding='utf-8')
            ledger_path = tmp / 'frontier-outcome-ledger.json'
            outbox = tmp / 'outbox.jsonl'

            before = runtime.read_text(encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()):
                result = run_science_theory_frontier_outcome_assessor(
                    runtime_memory_path=runtime,
                    transcript_file=transcript,
                    frontier_ledger_file=tmp / 'missing-frontier.json',
                    action_outcome_ledger_file=tmp / 'missing-action-outcome.json',
                    action_ledger_file=tmp / 'missing-action.json',
                    benefit_ledger_file=tmp / 'missing-benefit.json',
                    evaluator_ledger_file=tmp / 'missing-evaluator.json',
                    contract_ledger_file=tmp / 'missing-contract.json',
                    adjudicator_ledger_file=tmp / 'missing-adjudicator.json',
                    agenda_ledger_file=tmp / 'missing-agenda.json',
                    lifecycle_ledger_file=tmp / 'missing-lifecycle.json',
                    scorecard_ledger_file=tmp / 'missing-scorecard.json',
                    campaign_ledger_file=tmp / 'missing-campaign.json',
                    campaign_outcome_ledger_file=tmp / 'missing-campaign-outcome.json',
                    frontier_outcome_ledger_file=ledger_path,
                    outbox_file=outbox,
                    memory_data=memory_fixture(),
                    git_status_text=' M tmp/theory-memory.json\n',
                    git_ignored_text='',
                )
            with contextlib.redirect_stdout(io.StringIO()):
                repeat = run_science_theory_frontier_outcome_assessor(
                    runtime_memory_path=runtime,
                    transcript_file=transcript,
                    frontier_ledger_file=tmp / 'missing-frontier.json',
                    action_outcome_ledger_file=tmp / 'missing-action-outcome.json',
                    action_ledger_file=tmp / 'missing-action.json',
                    benefit_ledger_file=tmp / 'missing-benefit.json',
                    evaluator_ledger_file=tmp / 'missing-evaluator.json',
                    contract_ledger_file=tmp / 'missing-contract.json',
                    adjudicator_ledger_file=tmp / 'missing-adjudicator.json',
                    agenda_ledger_file=tmp / 'missing-agenda.json',
                    lifecycle_ledger_file=tmp / 'missing-lifecycle.json',
                    scorecard_ledger_file=tmp / 'missing-scorecard.json',
                    campaign_ledger_file=tmp / 'missing-campaign.json',
                    campaign_outcome_ledger_file=tmp / 'missing-campaign-outcome.json',
                    frontier_outcome_ledger_file=ledger_path,
                    outbox_file=outbox,
                    memory_data=memory_fixture(),
                    git_status_text=' M tmp/theory-memory.json\n',
                    git_ignored_text='',
                )
            after = runtime.read_text(encoding='utf-8')

        self.assertEqual(before, after)
        self.assertEqual('theory_memory_recorded_or_hypothesis_promoted', result['selected_outcome'])
        self.assertEqual(1, result['outbox_count'])
        self.assertEqual('summarize_noop', repeat['selected_outcome'])
        self.assertEqual(0, repeat['outbox_count'])
        self.assertFalse(result['runtime_memory_mutated'])
        self.assertEqual([], result['label_leaks'])
        self.assertTrue(result['no_sibling_imports'])
        self.assertFalse(result['project_owned_checkpoint_claimed'])

    def test_no_sibling_imports(self):
        module = Path(PROJECT_DIR) / 'agent' / 'science_theory_frontier_outcome.py'
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
