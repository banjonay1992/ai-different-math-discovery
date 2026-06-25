import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock


PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'first_principles_ai'))
sys.path.insert(0, PROJECT_DIR)

from agent.status_capsule import (  # noqa: E402
    build_ai_different_status_capsule,
    git_status_for_path,
    runtime_memory_state,
)
from main import run_status_capsule  # noqa: E402


def capsule_memory_fixture():
    latest = {
        'theory_kind': 'abstraction:scaled_domain_effect',
        'experiment_kind': 'abstraction_transfer_probe',
        'outcome': 'abstraction_transfer_confirmed',
        'context': 'hidden_procedural',
        'seed': 142,
        'abstraction_key': 'abstraction:scaled_domain_effect:memory',
        'abstraction_kind': 'scaled_domain_effect',
        'compressed_expression': 'effect := domain_weight * relative_basis / scale^p',
        'expected_result': 'compressed expression should reduce residual error',
        'falsifies_if': 'removing a compressed part gives the same held-out error',
    }
    return {
        'discovery_readiness': {
            'readiness_score': 0.81,
            'status': 'nearly_ready',
            'missing_gates': ['operator_prior_feedback'],
            'recommended_actions': [{
                'action_kind': 'non_final_abstraction_transfer_campaign',
                'reason': 'collect abstraction transfer evidence',
                'command': (
                    'python3 first_principles_ai/main.py '
                    '--abstraction-transfer-campaign '
                    '--theory-memory-file tmp/theory-memory.json'
                ),
                'runs_final': False,
            }],
            'gates': {
                'residual_to_theory': {
                    'passed': True,
                    'evidence': {'records_with_theories': 3},
                    'next_step': 'keep monitoring this gate',
                },
                'abstraction_discovery_loop': {
                    'passed': True,
                    'evidence': {
                        'bridge_count': 4,
                        'transfer_outcome_count': 1,
                        'transfer_confirmed_count': 1,
                    },
                    'next_step': 'keep monitoring this gate',
                },
            },
        },
        'rediscovery_goal_progress': {
            'progress_percent': 82.0,
            'blockers': ['operator_prior_feedback'],
            'gates': {
                'abstraction_discovery_transfer': {
                    'score': 1.0,
                    'passed': True,
                    'evidence': {'transfer_outcome_count': 1},
                    'next_step': 'keep monitoring this gate',
                },
            },
        },
        'abstraction_discovery_evidence': {
            'record_count': 3,
            'bridge_count': 4,
            'reusable_count': 4,
            'transfer_experiment_count': 4,
            'transfer_outcome_count': 1,
            'transfer_confirmed_count': 1,
            'transfer_weak_count': 0,
            'transfer_absent_count': 0,
            'latest_transfer_outcome': latest,
        },
        'planned_outcomes': [latest],
    }


class StatusCapsuleTests(unittest.TestCase):
    def test_capsule_shape_and_project_boundary(self):
        capsule = build_ai_different_status_capsule(
            capsule_memory_fixture(),
            git_status_text=' M tmp/theory-memory.json\n',
            runtime_memory_path='tmp/theory-memory.json',
        )

        self.assertEqual('AI Different', capsule['module'])
        self.assertEqual('orchestrator_status', capsule['capsule_kind'])
        self.assertFalse(
            capsule['project_owned_boundary']['third_party_checkpoint_used']
        )
        self.assertTrue(capsule['current_capabilities'])
        self.assertTrue(capsule['evidence_gates'])
        gate_keys = {gate['key'] for gate in capsule['evidence_gates']}
        self.assertIn('abstraction_discovery_loop', gate_keys)
        self.assertIn('abstraction_discovery_transfer', gate_keys)
        self.assertEqual(
            'non_final_abstraction_transfer_campaign',
            capsule['next_non_final_experiment']['action_kind'],
        )
        self.assertFalse(capsule['next_non_final_experiment']['runs_final'])
        self.assertTrue(capsule['runtime_memory']['dirty'])
        self.assertFalse(capsule['runtime_memory']['ignored'])

    def test_capsule_keeps_abstraction_transfer_evidence_label_clean(self):
        capsule = build_ai_different_status_capsule(
            capsule_memory_fixture(),
            git_status_text='',
        )
        latest = capsule['latest_verified_abstraction_transfer_result']

        self.assertEqual('verified', latest['status'])
        self.assertEqual('success', latest['result_strength'])
        self.assertEqual('abstraction_transfer_confirmed', latest['outcome'])
        self.assertTrue(latest['label_clean'])
        self.assertEqual([], latest['leak_terms'])
        agent_text = json.dumps(latest['agent_facing_evidence']).lower()
        for forbidden in ('gravity', 'vortex', 'repulsion', 'sideways_wind'):
            self.assertNotIn(forbidden, agent_text)

    def test_capsule_flags_label_leak_terms_when_transfer_uses_human_world_label(self):
        fixture = capsule_memory_fixture()
        fixture['abstraction_discovery_evidence']['latest_transfer_outcome'] = {
            **fixture['abstraction_discovery_evidence']['latest_transfer_outcome'],
            'context': 'localized_gravity',
        }

        capsule = build_ai_different_status_capsule(fixture)
        latest = capsule['latest_verified_abstraction_transfer_result']

        self.assertFalse(latest['label_clean'])
        self.assertIn('gravity', latest['leak_terms'])

    def test_runtime_memory_state_reports_dirty_and_ignored(self):
        dirty = runtime_memory_state(
            git_status_text=' M tmp/theory-memory.json\n',
            runtime_memory_path='tmp/theory-memory.json',
        )
        ignored = runtime_memory_state(
            git_status_text='!! tmp/theory-memory.json\n',
            runtime_memory_path='tmp/theory-memory.json',
        )

        self.assertTrue(dirty['dirty'])
        self.assertTrue(dirty['unstaged'])
        self.assertFalse(dirty['staged'])
        self.assertEqual('leave_unstaged_runtime_memory', dirty['recommendation'])
        self.assertFalse(ignored['dirty'])
        self.assertTrue(ignored['ignored'])

    def test_git_status_for_path_preserves_porcelain_leading_space(self):
        completed = mock.Mock()
        completed.stdout = ' M tmp/theory-memory.json\n'
        with mock.patch('agent.status_capsule.subprocess.run', return_value=completed):
            status = git_status_for_path('tmp/theory-memory.json')

        self.assertEqual(' M tmp/theory-memory.json', status)
        state = runtime_memory_state(
            git_status_text=status,
            runtime_memory_path='tmp/theory-memory.json',
        )
        self.assertFalse(state['staged'])
        self.assertTrue(state['unstaged'])

    def test_run_status_capsule_prints_and_writes_json(self):
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, 'capsule.json')
            with contextlib.redirect_stdout(output):
                capsule = run_status_capsule(
                    memory_data=capsule_memory_fixture(),
                    git_status_text=' M tmp/theory-memory.json\n',
                    git_ignored_text='',
                    output_file=output_file,
                )
            with open(output_file, 'r', encoding='utf-8') as handle:
                saved = json.load(handle)

        text = output.getvalue()
        self.assertIn('AI_DIFFERENT_STATUS_CAPSULE ', text)
        self.assertEqual('AI Different', capsule['module'])
        self.assertEqual(capsule['module'], saved['module'])
        self.assertEqual(output_file, capsule['artifact_path'])


if __name__ == '__main__':
    unittest.main()
