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
from agent.science_benefit_evaluator import (  # noqa: E402
    SCIENCE_BENEFIT_LEDGER_KIND,
    build_science_benefit_evaluation,
    empty_science_benefit_ledger,
    load_science_benefit_ledger,
    read_science_benefit_transcript,
    validate_science_benefit_ledger,
    write_science_benefit_ledger,
    write_science_benefit_outbox_jsonl,
)
from main import run_science_benefit_evaluator  # noqa: E402


def memory_fixture():
    return {
        'discovery_readiness': {'readiness_score': 0.89, 'status': 'nearly_ready'},
        'abstraction_discovery_evidence': {'transfer_outcome_count': 7},
    }


def isolated_message(
    scenario_id='scenario-one',
    *,
    accepted=None,
    missing=None,
    rejected=None,
    action='request_more_math',
):
    accepted = list(accepted or [])
    missing = list(missing or ['math_proof', 'code_proof', 'language_epoch_plan'])
    rejected = list(rejected or [])
    return build_module_chat_message(
        sender='ai_different',
        recipient='orchestrator',
        topic='ai_different.isolated_campaign_baseline',
        body={
            'response_kind': 'science_benefit_baseline',
            'comparison_mode': 'isolated',
            'scenario_id': scenario_id,
            'hypothesis_id': f'hypothesis:{scenario_id}',
            'selected_action': action,
            'accepted_evidence': accepted,
            'missing_evidence': missing,
            'rejected_evidence': rejected,
        },
        evidence={'scenario_id': scenario_id, 'comparison_mode': 'isolated'},
        tags=['science_benefit', 'isolated'],
    )


def connected_evidence(
    scenario_id='scenario-one',
    *,
    sender,
    gate=None,
    status='satisfied',
    action=None,
    leak=False,
    third_party=False,
):
    if gate is None:
        gate = {
            'funfun': 'math_proof',
            'code_module': 'code_proof',
            'language_model_2': 'language_epoch_plan',
        }.get(sender, 'advisory')
    summary = f'{gate} {status} connected evidence'
    if leak:
        summary = 'gravity label leaked into benefit evidence'
    return build_module_chat_message(
        sender=sender,
        recipient='ai_different',
        topic=f'{sender}.{gate}.{status}',
        body={
            'evidence_id': f'{scenario_id}-{sender}-{gate}-{status}',
            'scenario_id': scenario_id,
            'hypothesis_id': f'hypothesis:{scenario_id}',
            'comparison_mode': 'connected',
            'evidence_gate': gate,
            'status': status,
            'summary': summary,
            'selected_action': action,
            'third_party_checkpoint_used': third_party,
        },
        evidence={'scenario_id': scenario_id, 'evidence_gate': gate, 'status': status},
        tags=['science_benefit', 'connected', gate, status],
    )


def connected_ai_message(
    scenario_id='scenario-one',
    *,
    accepted=None,
    missing=None,
    rejected=None,
    action='refine_hypothesis',
):
    return build_module_chat_message(
        sender='ai_different',
        recipient='broadcast',
        topic='ai_different.connected_campaign_outcome',
        body={
            'response_kind': 'experiment_campaign_outcome',
            'comparison_mode': 'connected',
            'scenario_id': scenario_id,
            'hypothesis_id': f'hypothesis:{scenario_id}',
            'selected_action': action,
            'theory_update_action': action,
            'accepted_evidence': list(accepted or []),
            'missing_evidence': list(missing or []),
            'rejected_evidence': list(rejected or []),
        },
        evidence={'scenario_id': scenario_id, 'comparison_mode': 'connected'},
        tags=['science_benefit', 'connected'],
    )


def empty_ledger(kind, **extra):
    ledger = {'schema_version': 1, 'ledger_kind': kind, 'latest': {}, 'ledger_hash': f'{kind}-hash'}
    ledger.update(extra)
    return ledger


def build_once(messages, *, ledger=None, project_boundary=None):
    updated, message = build_science_benefit_evaluation(
        transcript_messages=messages,
        benefit_ledger=ledger or empty_science_benefit_ledger(),
        evaluator_ledger={},
        outcome_ledger={},
        contract_ledger={},
        adjudicator_ledger={},
        agenda_ledger={},
        lifecycle_ledger={},
        scorecard_ledger={},
        campaign_ledger={},
        campaign_outcome_ledger={},
        prior_benefit_ledger={},
        runtime_memory_data=memory_fixture(),
        runtime_memory_hash_state={'unchanged': True, 'before': 'same', 'after': 'same'},
        project_owned_boundary=project_boundary or {'third_party_checkpoint_used': False},
    )
    return updated, message


class ScienceBenefitEvaluatorTests(unittest.TestCase):
    def test_benefit_ledger_persistence_load_and_malformed_rejection(self):
        ledger, message = build_once([
            isolated_message(),
            connected_evidence(sender='funfun'),
            connected_evidence(sender='code_module'),
            connected_evidence(sender='language_model_2'),
        ])
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = Path(tmpdir) / 'benefit.json'
            outbox = Path(tmpdir) / 'outbox.jsonl'
            write_science_benefit_ledger(ledger_path, ledger)
            loaded = load_science_benefit_ledger(ledger_path)
            write_science_benefit_outbox_jsonl(outbox, message)
            rows = [json.loads(line) for line in outbox.read_text(encoding='utf-8').splitlines()]

        self.assertEqual(SCIENCE_BENEFIT_LEDGER_KIND, loaded['ledger_kind'])
        self.assertEqual('science_campaign_benefit', rows[0]['body']['response_kind'])
        with self.assertRaisesRegex(ValueError, 'wrong ledger_kind'):
            validate_science_benefit_ledger({'ledger_kind': 'wrong'})

    def test_scenario_extraction_and_invalid_record(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            transcript = Path(tmpdir) / 'benefit.jsonl'
            transcript.write_text(
                json.dumps(isolated_message(), sort_keys=True) + '\n'
                + json.dumps(connected_evidence(sender='funfun'), sort_keys=True) + '\n'
                + '{"sender":"bad"}\n',
                encoding='utf-8',
            )
            parsed = read_science_benefit_transcript(transcript)
        ledger, _ = build_once(parsed['messages'])

        self.assertEqual(2, len(parsed['messages']))
        self.assertEqual(1, len(parsed['invalid_messages']))
        scenario = ledger['scenario_records'][0]
        self.assertEqual('scenario-one', scenario['scenario_id'])
        self.assertIn('math_proof', scenario['connected']['accepted'])
        self.assertEqual('connected_requests_missing_targeted_evidence', ledger['latest']['benefit_classification'])

    def test_connected_math_code_language_acceptance_and_appended_update(self):
        first, first_message = build_once([
            isolated_message(),
            connected_evidence(sender='funfun'),
        ])
        self.assertEqual('connected_requests_missing_targeted_evidence', first['latest']['benefit_classification'])
        self.assertEqual('code_module', first_message['recipient'])

        second, second_message = build_once([
            connected_evidence(sender='code_module'),
            connected_evidence(sender='language_model_2'),
        ], ledger=first)
        self.assertEqual('connected_accepts_with_verified_math_code_language', second['latest']['benefit_classification'])
        self.assertEqual('broadcast', second_message['recipient'])
        self.assertEqual('accept_campaign', second_message['body']['selected_action'])

    def test_connected_refinement_retirement_and_targeted_request(self):
        refined, refined_message = build_once([
            isolated_message('refine-scenario'),
            connected_evidence('refine-scenario', sender='funfun'),
            connected_evidence('refine-scenario', sender='code_module'),
            connected_ai_message('refine-scenario', accepted=['math_proof', 'code_proof'], missing=['language_epoch_plan']),
        ])
        self.assertEqual('connected_refines_with_clearer_evidence', refined['latest']['benefit_classification'])
        self.assertEqual('refine_hypothesis', refined_message['body']['selected_action'])

        retired, retired_message = build_once([
            isolated_message('retire-scenario'),
            connected_evidence('retire-scenario', sender='funfun', status='failed'),
        ])
        self.assertEqual('connected_retires_failed_line', retired['latest']['benefit_classification'])
        self.assertEqual('retire_theory_line', retired_message['body']['selected_action'])

        requested, requested_message = build_once([
            isolated_message('request-scenario'),
            connected_evidence('request-scenario', sender='funfun'),
        ])
        self.assertEqual('connected_requests_missing_targeted_evidence', requested['latest']['benefit_classification'])
        self.assertEqual('code_module', requested_message['recipient'])

    def test_boundary_repair_and_isolated_already_sufficient(self):
        leak_ledger, leak_message = build_once([
            isolated_message(),
            connected_evidence(sender='funfun', leak=True),
        ])
        self.assertEqual('connected_prevents_boundary_or_checkpoint_overclaim', leak_ledger['latest']['benefit_classification'])
        self.assertEqual('code_module', leak_message['recipient'])
        self.assertIn('gravity', leak_message['body']['label_leaks'])

        boundary_ledger, boundary_message = build_once(
            [isolated_message()],
            project_boundary={'third_party_checkpoint_used': True},
        )
        self.assertEqual('connected_prevents_boundary_or_checkpoint_overclaim', boundary_ledger['latest']['benefit_classification'])
        self.assertEqual('code_module', boundary_message['recipient'])

        sufficient, sufficient_message = build_once([
            isolated_message(
                'sufficient-scenario',
                accepted=['math_proof', 'code_proof', 'language_epoch_plan'],
                missing=[],
                action='accept_campaign',
            )
        ])
        self.assertEqual('isolated_already_sufficient', sufficient['latest']['benefit_classification'])
        self.assertEqual('broadcast', sufficient_message['recipient'])

    def test_duplicate_idempotence(self):
        rows = [
            isolated_message(),
            connected_evidence(sender='funfun'),
            connected_evidence(sender='code_module'),
            connected_evidence(sender='language_model_2'),
        ]
        ledger, message = build_once(rows)
        repeat, repeat_message = build_once(rows, ledger=ledger)

        self.assertIsNotNone(message)
        self.assertIsNone(repeat_message)
        self.assertEqual('summarize_noop', repeat['latest']['benefit_classification'])

    def test_cli_status_proof_preserves_runtime_memory(self):
        rows = [
            isolated_message(),
            connected_evidence(sender='funfun'),
            connected_evidence(sender='code_module'),
            connected_evidence(sender='language_model_2'),
        ]
        empty_adjudicator = empty_ledger(
            'ai_different.cross_module_adjudicator_ledger',
            processed_message_ids=[],
            processed_evaluator_ledger_ids=[],
            contract_states=[],
            adjudication_records=[],
            outgoing_response_ids=[],
        )
        empty_agenda = empty_ledger(
            'ai_different.experiment_agenda_ledger',
            processed_message_ids=[],
            processed_source_hashes=[],
            scheduled_candidate_ids=[],
            hypotheses=[],
            agenda_records=[],
            outgoing_response_ids=[],
        )
        empty_lifecycle = empty_ledger(
            'ai_different.hypothesis_lifecycle_ledger',
            processed_message_ids=[],
            processed_source_hashes=[],
            resolved_hypothesis_ids=[],
            retired_hypothesis_ids=[],
            refined_hypothesis_ids=[],
            hypotheses=[],
            lifecycle_records=[],
            outgoing_response_ids=[],
        )
        empty_scorecard = empty_ledger(
            'ai_different.experiment_evidence_scorecard_ledger',
            processed_message_ids=[],
            processed_source_hashes=[],
            resolved_hypothesis_ids=[],
            retired_hypothesis_ids=[],
            refined_hypothesis_ids=[],
            scorecards=[],
            scorecard_records=[],
            outgoing_response_ids=[],
        )
        empty_campaign = empty_ledger(
            'ai_different.experiment_campaign_ledger',
            processed_message_ids=[],
            processed_source_hashes=[],
            retired_hypothesis_ids=[],
            continued_refinement_ids=[],
            acceptance_bundle_ids=[],
            campaigns=[],
            campaign_records=[],
            outgoing_response_ids=[],
        )
        empty_campaign_outcome = empty_ledger(
            'ai_different.experiment_campaign_outcome_ledger',
            processed_message_ids=[],
            processed_source_hashes=[],
            accepted_campaign_ids=[],
            refined_hypothesis_ids=[],
            retired_hypothesis_ids=[],
            repaired_campaign_ids=[],
            outcomes=[],
            outcome_records=[],
            outgoing_response_ids=[],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            memory_path = tmp / 'runtime-memory.json'
            transcript = tmp / 'family.jsonl'
            adjudicator_path = tmp / 'adjudicator.json'
            agenda_path = tmp / 'agenda.json'
            lifecycle_path = tmp / 'lifecycle.json'
            scorecard_path = tmp / 'scorecard.json'
            campaign_path = tmp / 'campaign.json'
            campaign_outcome_path = tmp / 'campaign-outcome.json'
            benefit_path = tmp / 'benefit.json'
            outbox = tmp / 'outbox.jsonl'
            memory_path.write_text(json.dumps(memory_fixture()), encoding='utf-8')
            transcript.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows), encoding='utf-8')
            adjudicator_path.write_text(json.dumps(empty_adjudicator), encoding='utf-8')
            agenda_path.write_text(json.dumps(empty_agenda), encoding='utf-8')
            lifecycle_path.write_text(json.dumps(empty_lifecycle), encoding='utf-8')
            scorecard_path.write_text(json.dumps(empty_scorecard), encoding='utf-8')
            campaign_path.write_text(json.dumps(empty_campaign), encoding='utf-8')
            campaign_outcome_path.write_text(json.dumps(empty_campaign_outcome), encoding='utf-8')
            before = memory_path.read_text(encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()):
                first = run_science_benefit_evaluator(
                    memory_data=memory_fixture(),
                    runtime_memory_path=memory_path,
                    transcript_file=transcript,
                    evaluator_ledger_file=None,
                    contract_ledger_file=None,
                    adjudicator_ledger_file=adjudicator_path,
                    agenda_ledger_file=agenda_path,
                    lifecycle_ledger_file=lifecycle_path,
                    scorecard_ledger_file=scorecard_path,
                    campaign_ledger_file=campaign_path,
                    campaign_outcome_ledger_file=campaign_outcome_path,
                    benefit_ledger_file=benefit_path,
                    outbox_file=outbox,
                    git_status_text='',
                    git_ignored_text='',
                )
            with contextlib.redirect_stdout(io.StringIO()):
                second = run_science_benefit_evaluator(
                    memory_data=memory_fixture(),
                    runtime_memory_path=memory_path,
                    transcript_file=transcript,
                    evaluator_ledger_file=None,
                    contract_ledger_file=None,
                    adjudicator_ledger_file=adjudicator_path,
                    agenda_ledger_file=agenda_path,
                    lifecycle_ledger_file=lifecycle_path,
                    scorecard_ledger_file=scorecard_path,
                    campaign_ledger_file=campaign_path,
                    campaign_outcome_ledger_file=campaign_outcome_path,
                    benefit_ledger_file=benefit_path,
                    outbox_file=outbox,
                    git_status_text='',
                    git_ignored_text='',
                )
            after = memory_path.read_text(encoding='utf-8')

        self.assertEqual('connected_accepts_with_verified_math_code_language', first['benefit_classification'])
        self.assertEqual(1, first['outbox_count'])
        self.assertEqual('summarize_noop', second['benefit_classification'])
        self.assertEqual(0, second['outbox_count'])
        self.assertTrue(first['runtime_memory_hash_state']['unchanged'])
        self.assertEqual(before, after)
        self.assertEqual([], first['label_leaks'])
        self.assertFalse(first['third_party_checkpoint_used'])
        self.assertTrue(first['no_sibling_imports'])
        self.assertFalse(first['project_owned_checkpoint_claimed'])

    def test_no_sibling_project_imports_are_introduced(self):
        root = Path(PROJECT_DIR)
        checked = [
            root / 'agent' / 'science_benefit_evaluator.py',
            root / 'main.py',
        ]
        forbidden = [
            'Language model 2.0',
            'Code Module',
            'orchastratorrrr',
            'from funfun',
            'import funfun',
        ]
        for path in checked:
            text = path.read_text(encoding='utf-8')
            for token in forbidden:
                self.assertNotIn(token, text)


if __name__ == '__main__':
    unittest.main()
