import contextlib
import io
import json
import os
import sys
import tempfile
import unittest


PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'first_principles_ai'))
sys.path.insert(0, PROJECT_DIR)

from agent.discovery_loop import CumulativeTheoryMemory
from agent.super_system import (
    build_super_system_report,
    planned_probe_actions,
    super_system_summary_lines,
)
from main import run_super_system_audit


class SuperSystemTests(unittest.TestCase):
    def test_super_system_report_connects_core_subsystems_and_artifacts(self):
        class DesignMemory(CumulativeTheoryMemory):
            def planned_experiments(self, world_types, object_counts, steps, limit):
                return [{
                    'theory_kind': 'distance_scaled_direction_residual',
                    'experiment_kind': 'model_disagreement_probe',
                    'priority': 0.91,
                    'world_type': world_types[0],
                    'seed': 7,
                    'object_count': object_counts[0],
                    'steps': steps,
                    'reason': 'current theories disagree after replay',
                    'expected_result': 'moving the object separates the residual',
                    'falsifies_if': 'the rival predicts the moved object better',
                    'probe_action': {
                        'type': 'move',
                        'object_id': 1,
                        'x': 15.0,
                        'y': 10.0,
                    },
                }]

        report = build_super_system_report(
            DesignMemory(),
            world_types=['vortex'],
            object_counts=[5],
            steps=240,
            limit=1,
            latest_artifact={
                'run_kind': 'math_final_discovery',
                'runs_final': True,
                'hf_upload': {
                    'status': 'failed',
                    'error': 'permission denied',
                },
            },
        )

        self.assertFalse(report['runs_final'])
        self.assertIn('experiment_design', report['subsystems'])
        self.assertIn('operator_system', report['subsystems'])
        self.assertIn('domain_rediscovery', report['subsystems'])
        self.assertEqual(
            'move',
            report['subsystems']['experiment_design']['cockpit'][0]['intervention']['type'],
        )
        self.assertIn(
            'artifact_upload_failed',
            {gap['gap_kind'] for gap in report['connection_gaps']},
        )
        self.assertIn(
            'freeze',
            {action['type'] for action in report['action_surface']},
        )
        self.assertTrue(any(
            status['connection'] == 'theories_to_experiment_design'
            for status in report['connection_status']
        ))

    def test_planned_probe_actions_preserves_direct_interventions(self):
        for action_type in ('move', 'freeze', 'duplicate'):
            plan = {
                'experiment_kind': 'model_disagreement_probe',
                'probe_action': {
                    'type': action_type,
                    'object_id': 3,
                    'x': 4.0,
                    'y': 5.0,
                },
            }

            actions = planned_probe_actions(plan)

            self.assertEqual(action_type, actions[0]['type'])
            self.assertEqual('planned_model_disagreement_probe', actions[0]['source'])

    def test_super_system_summary_and_cli_artifact_are_non_final(self):
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = os.path.join(tmp_dir, 'super-system.json')
            with contextlib.redirect_stdout(output):
                report = run_super_system_audit(
                    CumulativeTheoryMemory(),
                    output_file=output_file,
                    world_types=['standard'],
                    object_counts=[3],
                    steps=80,
                    limit=1,
                )
            with open(output_file, 'r', encoding='utf-8') as handle:
                saved = json.load(handle)

        text = output.getvalue()
        lines = '\n'.join(super_system_summary_lines(report))
        self.assertIn('SUPER SYSTEM AUDIT', text)
        self.assertIn('Watched final run: not run by this audit', text)
        self.assertIn('Experiment design cockpit:', lines)
        self.assertFalse(report['runs_final'])
        self.assertFalse(saved['runs_final'])
        self.assertEqual('super_system_audit', saved['run_kind'])


if __name__ == '__main__':
    unittest.main()
