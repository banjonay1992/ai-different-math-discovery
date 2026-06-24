import contextlib
import io
import os
import sys
import unittest


PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'first_principles_ai'))
sys.path.insert(0, PROJECT_DIR)

from agent.equation_workbench import EquationWorkbench
from agent.discovery_loop import CumulativeTheoryMemory
from agent.math_discovery import EmergentMathDiscovery
from agent.math_foundation import MathFoundationWorkbench
from agent.representation import KnowledgeBase
from main import (
    _foundation_metrics_from_knowledge,
    _print_section_study_summary,
    _select_interesting_equation,
    run_discovery_readiness_audit,
    run_experiment,
    run_math_final_discovery,
    run_math_foundation_prep,
)


def object_state(count, step, vx=0.2, vy=0.1):
    objects = []
    for index in range(count):
        objects.append({
            'id': index + 1,
            'position': (2.0 + index * 2.0 + vx * step * 0.016, 3.0 + index + vy * step * 0.016),
            'velocity': (vx, vy),
            'mass': 1.0 + index * 0.1,
            'radius': 0.5,
        })
    return {
        'objects': objects,
        'time': step * 0.016,
        'step': step,
        'world_size': (20.0, 20.0),
    }


class MathFoundationTests(unittest.TestCase):
    def build_ready_report(self):
        kb = KnowledgeBase()
        discovery = EmergentMathDiscovery(
            kb,
            concept_evidence_threshold=1,
            operation_evidence_threshold=1,
        )
        equations = EquationWorkbench(
            kb,
            min_samples=4,
            install_score_threshold=0.6,
        )

        transitions = [
            (object_state(2, 0), object_state(3, 1), {'type': 'spawn'}),
            (object_state(3, 1), object_state(2, 2), {'type': 'remove'}),
            (object_state(2, 2), object_state(3, 3), {'type': 'spawn'}),
            (object_state(3, 3), object_state(3, 4), {'type': 'wait'}),
            (object_state(3, 4), object_state(3, 5), {'type': 'wait'}),
            (object_state(3, 5), object_state(3, 6), {'type': 'wait'}),
            (object_state(3, 6), object_state(3, 7), {'type': 'wait'}),
            (object_state(3, 7), object_state(3, 8), {'type': 'wait'}),
        ]
        for step, (before, after, action) in enumerate(transitions, start=1):
            discovery.observe_transition(before, after, action, step)
            equations.observe_transition(before, after, action, step)

        equations.discover(step=len(transitions))
        foundation = MathFoundationWorkbench(kb, discovery, equations)
        return foundation.evaluate(install=True), foundation, kb

    def test_foundation_report_marks_core_gates_ready(self):
        report, foundation, _ = self.build_ready_report()

        self.assertTrue(report.gates['number_system_stability'])
        self.assertTrue(report.gates['equation_templates'])
        self.assertTrue(report.gates['composition_inverse_planning'])
        self.assertTrue(report.gates['check_traces'])
        self.assertTrue(report.gates['geometry_basis'])
        self.assertGreaterEqual(report.readiness_score, 0.84)
        self.assertEqual([], foundation.label_leaks(report))

    def test_foundation_extracts_equation_templates_and_path_plans(self):
        report, _, _ = self.build_ready_report()
        artifact_keys = {artifact.key for artifact in report.artifacts}

        self.assertIn('raw_foundation:channel_step_template', artifact_keys)
        self.assertIn('raw_foundation:extent_action_template', artifact_keys)
        self.assertIn('raw_foundation:return_path_planner', artifact_keys)

    def test_foundation_installs_check_traces_into_knowledge_base(self):
        report, _, kb = self.build_ready_report()
        traces = [
            rule for rule in kb.rules.values()
            if rule.properties.get('source') == 'math_foundation'
            and rule.properties.get('hypothesis_type') == 'foundation_check_trace'
        ]

        self.assertTrue(report.proof_traces)
        self.assertTrue(traces)
        self.assertTrue(all(rule.confidence > 0 for rule in traces))

    def test_foundation_metrics_handle_missing_report(self):
        metrics = _foundation_metrics_from_knowledge(KnowledgeBase())

        self.assertFalse(metrics['ready_for_final'])
        self.assertIn('number_system_stability', metrics['missing_gates'])

    def test_run_experiment_passes_operator_feedback_budget_to_workbench(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            _, kb, _ = run_experiment(
                num_steps=1,
                num_initial_objects=1,
                seed=0,
                verbose=False,
                report_interval=1,
                equation_max_operator_feedback_rows=64,
                equation_max_operator_feedback_operators=2,
            )

        self.assertEqual(64, kb.equation_workbench.max_operator_feedback_rows)
        self.assertEqual(2, kb.equation_workbench.max_operator_feedback_operators)

    def test_math_foundation_prep_runs_without_final_campaign(self):
        theory_memory = CumulativeTheoryMemory()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            results = run_math_foundation_prep(
                seeds=1,
                steps=80,
                object_counts=[3],
                world_types=['standard'],
                hidden_worlds=0,
                num_agents=2,
                theory_memory=theory_memory,
            )

        text = output.getvalue()
        self.assertEqual(1, len(results))
        self.assertIn('readiness_score', results[0])
        self.assertIn('probes', results[0])
        self.assertIn('discovery_loop', results[0])
        self.assertIn('Theory discovery readiness:', text)
        self.assertIn('Final command is ready, but not run:', text)
        self.assertIn('--theory-memory-file tmp/theory-memory.json', text)
        self.assertEqual(1, len(theory_memory.records))

    def test_discovery_readiness_audit_prints_without_running_final_campaign(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            report = run_discovery_readiness_audit(CumulativeTheoryMemory())

        text = output.getvalue()
        self.assertEqual('early', report['status'])
        self.assertIn('evidence_dossier', report)
        self.assertIn('DISCOVERY READINESS AUDIT', text)
        self.assertIn('Final discovery run: held for user-watched session', text)
        self.assertIn('Evidence dossier:', text)
        self.assertIn('domain world arithmetic_quantity', text)
        self.assertIn('executable_domain_worlds', report['gates'])
        self.assertTrue(report['gates']['executable_domain_worlds']['passed'])
        self.assertIn('Recommended non-final next actions:', text)
        self.assertIn('[SAFE] non_final_equation_campaign', text)
        self.assertNotIn('FINAL WATCHED MATH DISCOVERY CAMPAIGN', text)
        self.assertFalse(report['recommended_actions'][0]['runs_final'])

    def test_math_final_discovery_runs_performance_campaign(self):
        theory_memory = CumulativeTheoryMemory()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            results = run_math_final_discovery(
                seeds=1,
                steps=80,
                object_counts=[3],
                world_types=['standard'],
                hidden_worlds=1,
                num_agents=2,
                theory_memory=theory_memory,
            )

        text = output.getvalue()
        self.assertEqual(2, len(results))
        self.assertIn('FINAL WATCHED MATH DISCOVERY CAMPAIGN', text)
        self.assertIn('Running final case: standard seed=0', text)
        self.assertIn('Running final case: hidden_00_0000 seed=0', text)
        self.assertNotIn('held for user-watched session', text)
        self.assertTrue(all('equation_count' in result for result in results))
        self.assertTrue(all('ready_for_final' in result for result in results))
        self.assertTrue(all('equation_passed' in result for result in results))
        self.assertEqual(2, len(theory_memory.records))

    def test_math_final_discovery_can_repeat_and_consolidate_each_section(self):
        theory_memory = CumulativeTheoryMemory()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            results = run_math_final_discovery(
                seeds=1,
                steps=60,
                object_counts=[3],
                world_types=['standard'],
                hidden_worlds=1,
                num_agents=2,
                section_study_cycles=2,
                theory_memory=theory_memory,
            )

        text = output.getvalue()
        standard_results = [
            result for result in results
            if result['context'] == 'standard'
        ]
        hidden_results = [
            result for result in results
            if result['context'].startswith('hidden_')
        ]

        self.assertEqual(4, len(results))
        self.assertEqual([0, 1], [result['seed'] for result in standard_results])
        self.assertEqual([0, 1], [result['seed'] for result in hidden_results])
        self.assertIn('Section study cycles: 2', text)
        self.assertIn('Section study cycle 1/2: standard', text)
        self.assertIn('Section study cycle 2/2: standard', text)
        self.assertIn('Section study cycle 1/2: hidden_00_0000', text)
        self.assertIn('Section study cycle 2/2: hidden_00_0000', text)
        self.assertIn('Section study summary: standard cycle=1/2', text)
        self.assertIn('Section study summary: standard cycle=2/2', text)
        self.assertIn('Section study summary: hidden_00_0000 cycle=1/2', text)
        self.assertIn('Section study summary: hidden_00_0000 cycle=2/2', text)
        self.assertIn('Families:', text)
        self.assertIn('Best so far:', text)
        self.assertEqual(4, len(theory_memory.records))

    def test_math_final_discovery_executes_section_followup_replays(self):
        class ReplayMemory(CumulativeTheoryMemory):
            def planned_experiments(self, world_types, object_counts, steps, limit):
                if not self.equation_case_records:
                    return []
                return [{
                    'theory_kind': 'residual_after_transition',
                    'experiment_kind': 'post_run_replay_revision',
                    'priority': 0.99,
                    'world_type': 'sideways_wind',
                    'seed': 0,
                    'object_count': object_counts[0],
                    'steps': steps,
                    'hidden_holdout': False,
                    'reason': 'force residual-first replay of seed=0',
                    'expected_result': 'surface a residual law',
                    'falsifies_if': 'baseline transition stays dominant',
                    'source_status': 'robust_law',
                    'source_context': 'sideways_wind',
                    'target_context': 'post_run_replay_context',
                    'replay_key': 'post_run_replay:test:0:baseline',
                    'replay_issue': 'baseline_headline_needs_residual_replay',
                    'original_record': {
                        'context': 'sideways_wind',
                        'seed': 0,
                        'target': 'next_x',
                        'expression': 'x + vx * dt',
                        'role': 'position_update_equation',
                    },
                    'learned_invariant': {
                        'context': 'sideways_wind',
                        'law_family': 'linear_transition',
                    },
                    'equation_invariant': {
                        'context': 'sideways_wind',
                        'law_family': 'linear_transition',
                    },
                    'residual_first': True,
                    'replay_from_start': True,
                }]

        theory_memory = ReplayMemory()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            results = run_math_final_discovery(
                seeds=1,
                steps=40,
                object_counts=[3],
                world_types=['sideways_wind'],
                hidden_worlds=0,
                num_agents=2,
                section_study_cycles=2,
                theory_memory=theory_memory,
            )

        text = output.getvalue()
        self.assertEqual([0, 0], [result['seed'] for result in results])
        self.assertIn('plan=post_run_replay_revision', text)
        self.assertEqual(
            'post_run_replay_revision',
            results[1]['planned_experiment']['experiment_kind'],
        )
        self.assertTrue(results[1]['planned_experiment']['residual_first'])
        self.assertEqual(
            'math_final_discovery_followup',
            results[1]['phase'],
        )
        self.assertEqual(2, len(theory_memory.equation_case_records))

    def test_hidden_math_final_discovery_executes_section_followup_replays(self):
        class HiddenReplayMemory(CumulativeTheoryMemory):
            def planned_experiments(self, world_types, object_counts, steps, limit):
                if not self.equation_case_records:
                    return []
                return [{
                    'theory_kind': 'hidden_residual_replay',
                    'experiment_kind': 'post_run_replay_revision',
                    'priority': 0.99,
                    'world_type': 'hidden_00_0000',
                    'seed': 0,
                    'object_count': object_counts[0],
                    'steps': steps,
                    'hidden_holdout': False,
                    'reason': 'replay hidden seed=0 residual-first',
                    'expected_result': 'surface a hidden residual law',
                    'falsifies_if': 'baseline transition stays dominant',
                    'source_status': 'robust_law',
                    'source_context': 'hidden_00_0000',
                    'target_context': 'post_run_replay_context',
                    'replay_key': 'post_run_replay:hidden:0:baseline',
                    'replay_issue': 'baseline_headline_needs_residual_replay',
                    'original_record': {
                        'context': 'hidden_00_0000',
                        'seed': 0,
                        'target': 'next_x',
                        'expression': 'x + vx * dt',
                        'role': 'position_update_equation',
                    },
                    'learned_invariant': {
                        'context': 'hidden_00_0000',
                        'law_family': 'linear_transition',
                    },
                    'equation_invariant': {
                        'context': 'hidden_00_0000',
                        'law_family': 'linear_transition',
                    },
                    'residual_first': True,
                    'replay_from_start': True,
                }]

        theory_memory = HiddenReplayMemory()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            results = run_math_final_discovery(
                seeds=1,
                steps=40,
                object_counts=[3],
                world_types=['standard'],
                hidden_worlds=1,
                num_agents=2,
                section_study_cycles=2,
                theory_memory=theory_memory,
            )

        text = output.getvalue()
        hidden_results = [
            result for result in results
            if result['context'] == 'hidden_00_0000'
        ]
        self.assertEqual([0, 0], [result['seed'] for result in hidden_results])
        self.assertIn('Section study cycle 2/2: hidden_00_0000', text)
        self.assertIn(
            'Running final case: hidden_00_0000 seed=0 objects=3 steps=40 '
            'plan=post_run_replay_revision',
            text,
        )
        self.assertEqual(
            'post_run_replay_revision',
            hidden_results[1]['planned_experiment']['experiment_kind'],
        )
        self.assertTrue(hidden_results[1]['planned_experiment']['residual_first'])
        self.assertEqual(
            'math_final_discovery_followup',
            hidden_results[1]['phase'],
        )

    def test_residual_first_selection_prefers_residual_over_baseline_motion(self):
        baseline = {
            'target': 'next_x',
            'expression': 'x + vx * dt',
            'role': 'position_update_equation',
            'score': 0.97,
            'complexity': 2,
        }
        residual = {
            'target': 'baseline_adjusted_delta_velocity',
            'expression': 'k * unit_inferred_vector',
            'role': 'residual_direction_equation',
            'score': 0.31,
            'complexity': 3,
        }
        pack = {
            'top_equations': [baseline],
            'interesting_equations': [baseline],
            'categories': {
                'motion_updates': [baseline],
                'residual_dynamics': [residual],
            },
        }

        selected = _select_interesting_equation(pack, residual_first=True)
        normal = _select_interesting_equation(pack, residual_first=False)

        self.assertEqual('residual_direction_equation', selected['role'])
        self.assertEqual('position_update_equation', normal['role'])

    def test_section_study_summary_filters_cross_section_followups(self):
        class FakeMemory:
            def planned_experiments(self, world_types, object_counts, steps, limit):
                return [
                    {
                        'experiment_kind': 'post_run_replay_revision',
                        'world_type': 'central_force',
                        'source_context': 'central_force',
                        'seed': 0,
                        'priority': 0.99,
                        'reason': 'replay central_force seed=0',
                    },
                    {
                        'experiment_kind': 'post_run_replay_revision',
                        'world_type': 'repulsion',
                        'source_context': 'repulsion',
                        'seed': 2,
                        'priority': 0.95,
                        'reason': 'replay repulsion seed=2',
                    },
                ]

        section_results = [{
            'passed': True,
            'label_leaks': [],
            'interesting_score': 0.87,
            'interesting_equation': {
                'target': 'baseline_adjusted_delta_velocity',
                'expression': (
                    'k * taper(separation, 7_273) * '
                    'unit_generated_center_vector / separation^0_5'
                ),
                'role': 'generated_operator_tapered_distance_direction_equation',
                'parameters': {
                    'operator_kind': 'localized_tapered_power',
                    'distance_exponent': 0.5,
                },
            },
        }]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            _print_section_study_summary(
                'repulsion',
                section_results,
                FakeMemory(),
                object_counts=[5],
                steps=600,
                cycle=1,
                total_cycles=2,
            )

        text = output.getvalue()
        self.assertIn('Section follow-up probes:', text)
        self.assertIn('replay repulsion seed=2', text)
        self.assertNotIn('replay central_force seed=0', text)


if __name__ == '__main__':
    unittest.main()
