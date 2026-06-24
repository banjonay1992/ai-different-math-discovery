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

import main as main_module

from agent.equation_workbench import EquationWorkbench
from agent.discovery_loop import CumulativeTheoryMemory
from agent.math_discovery import EmergentMathDiscovery
from agent.math_foundation import MathFoundationWorkbench
from agent.perception import Perception
from agent.representation import KnowledgeBase
from main import (
    _artifact_log_chunks,
    _checkpoint_theory_memory,
    _experiment_design_cockpit,
    _foundation_metrics_from_knowledge,
    _math_final_artifact_summary,
    _persist_math_final_artifact,
    _print_section_study_summary,
    _runtime_profile_summary,
    _section_best_result,
    _section_composite_decomposition,
    _section_leak_diagnosis,
    _section_parameter_consolidation,
    _select_interesting_equation,
    _should_force_residual_first,
    _weak_case_diagnostics,
    merge_final_artifacts,
    parse_live_progress_line,
    run_backend_profile_comparison,
    run_gpu_feasibility_benchmark,
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

    def test_run_experiment_reuses_perception_between_steps(self):
        original_perceive = main_module.Perception.perceive
        perceive_calls = 0

        def counted_perceive(raw_state):
            nonlocal perceive_calls
            perceive_calls += 1
            return original_perceive(raw_state)

        with mock.patch.object(
            main_module.Perception,
            'perceive',
            side_effect=counted_perceive,
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                run_experiment(
                    num_steps=4,
                    num_initial_objects=1,
                    seed=0,
                    verbose=False,
                    report_interval=4,
                )

        self.assertEqual(5, perceive_calls)

    def test_perception_feature_vector_matches_derived_properties(self):
        observation = Perception.perceive(object_state(3, 2, vx=0.4, vy=-0.2))
        features = observation.get_feature_vector()

        self.assertEqual(observation.count, features['count'])
        self.assertEqual(
            round(observation.total_momentum_x, 6),
            features['total_momentum_x'],
        )
        self.assertEqual(
            round(observation.total_momentum_y, 6),
            features['total_momentum_y'],
        )
        self.assertEqual(round(observation.total_momentum, 6), features['total_momentum'])
        self.assertEqual(
            round(observation.total_kinetic_energy, 6),
            features['total_kinetic_energy'],
        )
        self.assertEqual(round(observation.total_mass, 6), features['total_mass'])
        self.assertEqual(
            round(observation.center_of_mass_x, 6),
            features['center_of_mass_x'],
        )
        self.assertEqual(
            round(observation.center_of_mass_y, 6),
            features['center_of_mass_y'],
        )
        self.assertEqual(
            round(sum(observation.pairwise_distances) / len(observation.pairwise_distances), 6),
            features['mean_distance'],
        )

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

    def test_math_final_discovery_parallel_cases_preserve_result_order(self):
        class ImmediateFuture:
            def __init__(self, result):
                self._result = result

            def result(self):
                return self._result

        class FakeExecutor:
            max_workers_seen = None
            payloads = []

            def __init__(self, max_workers):
                FakeExecutor.max_workers_seen = max_workers

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def submit(self, fn, payload):
                FakeExecutor.payloads.append(dict(payload))
                return ImmediateFuture(fn(payload))

        def fake_case(payload):
            seed = payload['seed']
            return {
                'context': payload['context'],
                'seed': seed,
                'objects': payload['object_count'],
                'steps': payload['steps'],
                'readiness_score': 1.0,
                'missing_gates': [],
                'gates': {},
                'artifact_count': 0,
                'proof_trace_count': 0,
                'probe_count': 0,
                'ready_for_final': True,
                'probes': [],
                'equation_count': 1,
                'installed_count': 1,
                'label_leaks': [],
                'probe_suggestions': [],
                'interesting_score': 0.9,
                'interesting_equation': {
                    'target': 'next_x',
                    'expression': f'seed_{seed}',
                    'role': 'position_update_equation',
                    'score': 0.9,
                },
                'discovery_loop': {},
                'generated_operator_prior_count': 0,
                'equation_passed': True,
                'passed': True,
            }

        FakeExecutor.payloads = []
        theory_memory = CumulativeTheoryMemory()
        output = io.StringIO()
        with (
            mock.patch('main.concurrent.futures.ProcessPoolExecutor', FakeExecutor),
            mock.patch(
                'main.concurrent.futures.as_completed',
                side_effect=lambda futures: list(reversed(list(futures))),
            ),
            mock.patch('main._execute_math_final_case_payload', side_effect=fake_case),
            contextlib.redirect_stdout(output),
        ):
            results = run_math_final_discovery(
                seeds=3,
                steps=40,
                object_counts=[3],
                world_types=['standard'],
                hidden_worlds=0,
                num_agents=2,
                theory_memory=theory_memory,
                parallel_cases=3,
            )

        self.assertEqual(3, FakeExecutor.max_workers_seen)
        self.assertEqual([0, 1, 2], [payload['seed'] for payload in FakeExecutor.payloads])
        self.assertEqual([0, 1, 2], [result['seed'] for result in results])
        self.assertEqual(
            [0, 1, 2],
            [record['seed'] for record in theory_memory.equation_case_records],
        )
        self.assertIn('Parallel case workers: 3', output.getvalue())

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

    def test_math_final_discovery_runs_self_authored_hidden_worlds(self):
        class SelfAuthoredMemory(CumulativeTheoryMemory):
            def autonomous_experiment_design_agenda(self, limit=8):
                return [{
                    'design_key': 'autonomous_design:invariant_resolution:test',
                    'source': 'invariant_resolution',
                    'experiment_kind': 'equation_invariant_exponent_resolution',
                    'priority': 0.91,
                    'question': 'Which distance exponent wins a near/mid/far race?',
                    'expected_result': 'select a robust exponent',
                    'falsifies_if': 'all exponents tie on heldout rows',
                    'evidence': {'support_count': 2},
                }][:limit]

            def planned_experiments(self, world_types, object_counts, steps, limit):
                return []

        theory_memory = SelfAuthoredMemory()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            results = run_math_final_discovery(
                seeds=1,
                steps=40,
                object_counts=[3],
                world_types=['standard'],
                hidden_worlds=0,
                self_authored_worlds=1,
                num_agents=2,
                section_study_cycles=2,
                theory_memory=theory_memory,
            )

        text = output.getvalue()
        authored_results = [
            result for result in results
            if result['context'] == 'authored_00_0000'
        ]
        manifest_components = [
            component['type']
            for component in authored_results[0]['manifest']['components']
        ]

        self.assertEqual(4, len(results))
        self.assertEqual(2, len(authored_results))
        self.assertEqual([0, 1], [result['seed'] for result in authored_results])
        self.assertIn('Self-authored hidden worlds: 1', text)
        self.assertIn('Self-authored hidden world: authored_00_0000', text)
        self.assertIn('Section study cycle 1/2: authored_00_0000', text)
        self.assertIn('Section study cycle 2/2: authored_00_0000', text)
        self.assertIn('Running final case: authored_00_0000 seed=0', text)
        self.assertEqual(
            {'self_authored'},
            {result['manifest_source'] for result in authored_results},
        )
        self.assertIn('self_authored_world_design', authored_results[0])
        self.assertEqual(
            'invariant_resolution',
            authored_results[0]['self_authored_world_design']['source'],
        )
        self.assertIn('piecewise_radial', manifest_components)
        self.assertIn('cutoff_radial_push', manifest_components)

    def test_math_final_discovery_can_resume_hidden_world_offset(self):
        def hidden_manifest(index, variant):
            manifest = mock.Mock()
            manifest.hidden_id = f"hidden_{index:02d}_{variant:04d}"
            return manifest

        output = io.StringIO()
        with (
            mock.patch('main.generate_hidden_world_manifest', side_effect=hidden_manifest) as generate,
            mock.patch('main._run_hidden_manifest_final_section') as run_section,
            contextlib.redirect_stdout(output),
        ):
            results = run_math_final_discovery(
                seeds=1,
                steps=40,
                object_counts=[3],
                world_types=[],
                hidden_worlds=2,
                hidden_world_start=7,
                num_agents=2,
                section_study_cycles=1,
                theory_memory=CumulativeTheoryMemory(),
            )

        text = output.getvalue()
        self.assertEqual([], results)
        self.assertIn('Worlds: (none)', text)
        self.assertIn('Hidden generated world start: 7', text)
        self.assertEqual(
            [mock.call(7, variant=7), mock.call(8, variant=8)],
            generate.call_args_list,
        )
        self.assertEqual(
            ['hidden_07_0007', 'hidden_08_0008'],
            [call.args[0].hidden_id for call in run_section.call_args_list],
        )

    def test_theory_memory_checkpoint_writes_resumable_snapshot(self):
        theory_memory = CumulativeTheoryMemory()
        theory_memory.records.append({
            'context': 'standard',
            'seed': 0,
            'phase': 'math_final_discovery',
        })

        output = io.StringIO()
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'theory-memory.json')
            with contextlib.redirect_stdout(output):
                _checkpoint_theory_memory(
                    theory_memory,
                    checkpoint_path,
                    label='standard cycle 1/1',
                )
            with open(checkpoint_path, 'r', encoding='utf-8') as handle:
                data = json.load(handle)

        self.assertEqual('standard', data['records'][0]['context'])
        self.assertIn('Theory memory checkpoint saved:', output.getvalue())

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

    def test_section_memory_forces_residual_first_after_non_motion_law(self):
        theory_memory = CumulativeTheoryMemory()

        self.assertFalse(_should_force_residual_first(theory_memory, 'sideways_wind'))

        theory_memory.equation_case_records.append({
            'context': 'sideways_wind',
            'seed': 5,
            'role': 'residual_direction_equation',
            'parameters': {},
            'passed': True,
            'label_leak_count': 0,
        })
        theory_memory.equation_case_records.append({
            'context': 'standard',
            'seed': 7,
            'role': 'residual_direction_equation',
            'parameters': {},
            'passed': True,
            'label_leak_count': 0,
        })

        self.assertTrue(_should_force_residual_first(theory_memory, 'sideways_wind'))
        self.assertFalse(_should_force_residual_first(theory_memory, 'standard'))

    def test_section_best_result_prefers_robust_non_motion_signature(self):
        baseline = {
            'passed': True,
            'equation_passed': True,
            'label_leaks': [],
            'interesting_score': 0.97,
            'interesting_equation': {
                'target': 'next_x',
                'expression': 'x + vx * dt',
                'role': 'position_update_equation',
                'score': 0.97,
                'parameters': {},
            },
        }
        periodic_a = {
            'passed': True,
            'equation_passed': True,
            'label_leaks': [],
            'interesting_score': 0.91,
            'interesting_equation': {
                'target': 'baseline_adjusted_delta_vy',
                'expression': 'a * sin(step/76) + b * cos(step/76)',
                'role': 'residual_periodic_equation',
                'score': 0.91,
                'parameters': {'operator_kind': 'phase_basis'},
            },
        }
        periodic_b = {
            **periodic_a,
            'interesting_score': 0.92,
            'interesting_equation': {
                **periodic_a['interesting_equation'],
                'score': 0.92,
            },
        }

        selected = _section_best_result(
            'hidden_02_0002',
            [baseline, periodic_a, periodic_b],
        )

        self.assertEqual(
            'residual_periodic_equation',
            selected['interesting_equation']['role'],
        )

    def test_section_consolidation_selects_majority_exponent_and_radius(self):
        def row(seed, exponent, radius):
            return {
                'context': 'hidden_07_0007',
                'seed': seed,
                'passed': True,
                'equation_passed': True,
                'label_leaks': [],
                'interesting_score': 0.88,
                'interesting_equation': {
                    'role': 'generated_operator_tapered_distance_perpendicular_equation',
                    'target': 'baseline_adjusted_delta_velocity',
                    'expression': (
                        'k * taper(separation, 7_2) * '
                        'perpendicular(unit_generated_center_vector) '
                        f'/ separation^{exponent}'
                    ),
                    'parameters': {
                        'operator_kind': 'localized_tapered_power',
                        'distance_exponent': exponent,
                        'cutoff_radius': radius,
                        'relation': 'perpendicular',
                    },
                },
            }

        consolidation = _section_parameter_consolidation(
            'hidden_07_0007',
            [row(0, 0.5, 7.2), row(1, 0.5, 7.35), row(2, 0.75, 8.4)],
        )

        self.assertEqual('localized_tapered_power', consolidation['dominant_family'])
        self.assertEqual(0.5, consolidation['selected_distance_exponent'])
        self.assertAlmostEqual(0.667, consolidation['distance_exponent_confidence'])
        self.assertAlmostEqual(7.275, consolidation['selected_cutoff_radius'])

    def test_section_leak_diagnosis_summarizes_labels_and_rows(self):
        diagnosis = _section_leak_diagnosis(
            'localized_gravity',
            [{
                'context': 'localized_gravity',
                'seed': 2,
                'phase': 'math_final_discovery',
                'passed': False,
                'label_leaks': [{
                    'labels': ['gravity', 'localized_gravity'],
                    'description': 'mentions gravity',
                }],
                'interesting_equation': {
                    'role': 'localized_tapered_power',
                    'expression': 'gravity',
                },
            }],
        )

        self.assertEqual(1, diagnosis['leak_count'])
        self.assertEqual(1, diagnosis['affected_row_count'])
        self.assertEqual(1, diagnosis['label_counts']['gravity'])
        self.assertIn('block leaked rows', diagnosis['recommendation'])

    def test_composite_decomposition_splits_phase_and_tangential_components(self):
        phase = {
            'context': 'hidden_04_0004',
            'seed': 1,
            'passed': True,
            'equation_passed': True,
            'label_leaks': [],
            'interesting_equation': {
                'role': 'residual_periodic_equation',
                'expression': 'a * sin(step/75) + b * cos(step/75)',
                'parameters': {'operator_kind': 'phase_basis'},
            },
            'manifest': {
                'components': [{'type': 'time_wave'}, {'type': 'tangential_flow'}],
            },
        }
        tangential = {
            'context': 'hidden_04_0004',
            'seed': 2,
            'passed': True,
            'equation_passed': True,
            'label_leaks': [],
            'interesting_equation': {
                'role': 'generated_operator_tapered_distance_perpendicular_equation',
                'expression': (
                    'k * taper(separation, 7_5) * '
                    'perpendicular(unit_generated_center_vector)'
                ),
                'parameters': {'operator_kind': 'localized_tapered_power'},
            },
            'manifest': phase['manifest'],
        }

        decomposition = _section_composite_decomposition(
            'hidden_04_0004',
            [phase, tangential],
        )

        self.assertEqual('composite_hypothesis', decomposition['status'])
        components = {
            item['component'] for item in decomposition['inferred_components']
        }
        self.assertIn('time_varying_component', components)
        self.assertIn('tangential_component', components)
        self.assertEqual(1, decomposition['benchmark_manifest_components']['time_wave'])

    def test_experiment_design_cockpit_exposes_beliefs_and_intervention(self):
        class DesignMemory(CumulativeTheoryMemory):
            def planned_experiments(self, world_types, object_counts, steps, limit):
                return [{
                    'theory_kind': 'distance_scaled_direction_residual',
                    'experiment_kind': 'model_disagreement_probe',
                    'priority': 0.91,
                    'world_type': 'vortex',
                    'seed': 12,
                    'object_count': object_counts[0],
                    'steps': steps,
                    'reason': 'current distance and direction theories disagree',
                    'expected_result': 'near and far probes separate the exponent',
                    'falsifies_if': 'the rival has lower residual error on both probes',
                    'disagreement_signature': {
                        'question': 'Which exponent survives near/far intervention?',
                        'rival_predictions': [
                            {
                                'theory_key': 'inverse_square',
                                'score': 0.63,
                                'prediction': 'near probe has much larger residual',
                                'falsified_if': 'near/far ratio is weak',
                            },
                            {
                                'theory_key': 'inverse_linear',
                                'score': 0.27,
                                'prediction': 'near/far ratio is milder',
                                'falsified_if': 'near/far ratio is steep',
                            },
                        ],
                    },
                    'probe_action': {
                        'type': 'move',
                        'object_id': 1,
                        'x': 15.0,
                        'y': 10.0,
                    },
                }]

        designs = _experiment_design_cockpit(
            DesignMemory(),
            world_types=['vortex'],
            object_counts=[5],
            steps=240,
            limit=1,
        )

        self.assertEqual(1, len(designs))
        design = designs[0]
        self.assertIn('Which exponent survives', design['question'])
        self.assertIn('move object 1', design['intervention_text'])
        self.assertEqual('high', design['counterexample_reward'])
        self.assertAlmostEqual(
            1.0,
            sum(item['belief'] for item in design['beliefs']),
            places=3,
        )

    def test_math_final_artifact_writes_summary_with_diagnostics(self):
        theory_memory = CumulativeTheoryMemory()
        result = {
            'context': 'hidden_07_0007',
            'seed': 0,
            'objects': 5,
            'steps': 120,
            'passed': True,
            'ready_for_final': True,
            'equation_passed': True,
            'label_leaks': [],
            'interesting_score': 0.91,
            'interesting_equation': {
                'role': 'generated_operator_tapered_distance_perpendicular_equation',
                'target': 'baseline_adjusted_delta_velocity',
                'expression': (
                    'k * taper(separation, 7_2) * '
                    'perpendicular(unit_generated_center_vector) / separation^0.5'
                ),
                'parameters': {
                    'operator_kind': 'localized_tapered_power',
                    'distance_exponent': 0.5,
                    'cutoff_radius': 7.2,
                },
            },
            'manifest': {'components': [{'type': 'tangential_flow'}]},
        }
        theory_memory.record_equation_case_result(
            'hidden_07_0007',
            0,
            result,
            phase='math_final_discovery',
        )
        starting = CumulativeTheoryMemory().memory_checkpoint_summary()
        artifact = _math_final_artifact_summary(
            [result],
            theory_memory,
            run_id='demo-run',
            run_config={'seeds': 1},
            starting_memory_summary=starting,
        )

        self.assertTrue(artifact['runs_final'])
        self.assertEqual(1, artifact['passed_count'])
        self.assertEqual('demo-run', artifact['run_id'])
        self.assertEqual(1, artifact['memory_delta']['new_equation_cases'])
        self.assertEqual(
            'localized_tapered_power',
            artifact['section_consolidations'][0]['dominant_family'],
        )
        self.assertIn('weak_case_diagnostics', artifact)
        self.assertIn('super_system_snapshot', artifact)
        self.assertIn('resource_efficiency', artifact)
        self.assertIn('runtime_profile', artifact)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, 'final.json')
            with mock.patch.dict(
                os.environ,
                {'HF_OUTPUT_REPO': '', 'HF_RUN_REPO': '', 'HF_TOKEN': ''},
            ), contextlib.redirect_stdout(io.StringIO()):
                persisted = _persist_math_final_artifact(
                    [result],
                    theory_memory,
                    artifact_output_file=output_file,
                    hf_output_repo=None,
                    run_id='demo-run',
                    run_config={'seeds': 1},
                    starting_memory_summary=starting,
                )
            with open(output_file, 'r', encoding='utf-8') as handle:
                saved = json.load(handle)

        self.assertEqual('skipped', persisted['hf_upload']['status'])
        self.assertEqual('demo-run', saved['run_id'])
        self.assertIn('artifact_path', saved)

    def test_weak_case_diagnostics_classifies_failures_and_conflicts(self):
        diagnostics = _weak_case_diagnostics([
            {
                'context': 'localized_gravity',
                'seed': 1,
                'passed': False,
                'ready_for_final': True,
                'equation_passed': False,
                'label_leaks': [{'labels': ['gravity']}],
                'interesting_score': 0.11,
                'interesting_equation': {'role': 'local_residual_direction_equation'},
                'planned_experiment_outcome': {'outcome': 'blind_holdout_conflicted'},
            },
            {
                'context': 'standard',
                'seed': 0,
                'passed': True,
                'ready_for_final': True,
                'equation_passed': True,
                'label_leaks': [],
                'interesting_score': 0.95,
                'interesting_equation': {'role': 'position_update_equation'},
            },
        ])

        self.assertEqual('needs_diagnosis', diagnostics['status'])
        self.assertEqual(1, diagnostics['weak_case_count'])
        self.assertEqual(1, diagnostics['reason_counts']['label_leak'])
        self.assertEqual(1, diagnostics['reason_counts']['equation_not_clean'])
        self.assertEqual(
            1,
            diagnostics['reason_counts']['planned_holdout_or_repair_conflict'],
        )
        self.assertTrue(diagnostics['next_actions'])

    def test_runtime_profile_summary_ranks_slow_contexts(self):
        profile = _runtime_profile_summary(
            [
                {
                    'context': 'standard',
                    'seed': 0,
                    'phase': 'math_final_discovery',
                    'passed': True,
                    'equation_count': 2,
                    'interesting_score': 0.9,
                    'case_elapsed_seconds': 1.0,
                },
                {
                    'context': 'vortex',
                    'seed': 1,
                    'phase': 'math_final_discovery',
                    'passed': True,
                    'equation_count': 4,
                    'interesting_score': 0.7,
                    'case_elapsed_seconds': 3.5,
                },
            ],
            [{'event': 'section_cycle_profile', 'context': 'vortex'}],
        )

        self.assertTrue(profile['enabled'])
        self.assertEqual(2, profile['case_count'])
        self.assertEqual('vortex', profile['by_context'][0]['context'])
        self.assertEqual(1, len(profile['section_events']))

    def test_merge_final_artifacts_combines_shards_and_profiles(self):
        shard_a = {
            'run_kind': 'math_final_discovery',
            'run_id': 'shard-a',
            'result_count': 1,
            'passed_count': 1,
            'rows': [{
                'context': 'standard',
                'seed': 0,
                'objects': 5,
                'steps': 40,
                'phase': 'math_final_discovery',
                'passed': True,
                'ready_for_final': True,
                'equation_passed': True,
                'leak_count': 0,
                'interesting_score': 0.97,
                'interesting_role': 'position_update_equation',
                'interesting_target': 'next_x',
                'interesting_expression': 'x + vx * dt',
                'interesting_parameters': {},
                'case_elapsed_seconds': 1.2,
            }],
            'runtime_profile': {
                'section_events': [{
                    'event': 'section_cycle_profile',
                    'context': 'standard',
                    'elapsed_seconds': 1.3,
                }],
            },
        }
        shard_b = {
            'run_kind': 'math_final_discovery',
            'run_id': 'shard-b',
            'result_count': 1,
            'passed_count': 0,
            'rows': [{
                'context': 'localized_gravity',
                'seed': 1,
                'objects': 5,
                'steps': 40,
                'phase': 'math_final_discovery',
                'passed': False,
                'ready_for_final': True,
                'equation_passed': False,
                'leak_count': 1,
                'interesting_score': 0.1,
                'interesting_role': 'local_residual_direction_equation',
                'interesting_target': 'baseline_adjusted_delta_velocity',
                'interesting_expression': 'leaky expression',
                'interesting_parameters': {},
                'case_elapsed_seconds': 2.4,
            }],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path_a = os.path.join(tmpdir, 'a.json')
            path_b = os.path.join(tmpdir, 'b.json')
            merged_path = os.path.join(tmpdir, 'merged.json')
            with open(path_a, 'w', encoding='utf-8') as handle:
                json.dump(shard_a, handle)
            with open(path_b, 'w', encoding='utf-8') as handle:
                json.dump(shard_b, handle)

            merged = merge_final_artifacts(
                [path_a, path_b],
                output_file=merged_path,
                run_id='merged-demo',
            )

            with open(merged_path, 'r', encoding='utf-8') as handle:
                saved = json.load(handle)

        self.assertEqual('math_final_shard_merge', merged['run_kind'])
        self.assertEqual(2, merged['result_count'])
        self.assertEqual(1, merged['passed_count'])
        self.assertEqual(1, merged['weak_case_diagnostics']['weak_case_count'])
        self.assertTrue(merged['runtime_profile']['enabled'])
        self.assertEqual('merged-demo', saved['run_id'])

    def test_gpu_feasibility_benchmark_is_non_final_and_writes_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, 'gpu.json')
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                report = run_gpu_feasibility_benchmark(
                    sample_count=256,
                    repeats=1,
                    prefer_cuda=False,
                    output_file=output_file,
                )
            with open(output_file, 'r', encoding='utf-8') as handle:
                saved = json.load(handle)

        self.assertFalse(report['runs_final'])
        self.assertEqual('gpu_feasibility_benchmark', report['run_kind'])
        self.assertIn('recommendation', report)
        self.assertIn('physics_force_backend', report)
        self.assertIn('python_force_kernel_seconds', report)
        self.assertIn('physics_force_backend_seconds', report)
        self.assertTrue(report['physics_force_parity_passed'])
        self.assertEqual(report['recommendation'], saved['recommendation'])
        self.assertEqual(
            report['physics_force_backend'],
            saved['physics_force_backend'],
        )
        self.assertIn('GPU FEASIBILITY BENCHMARK', output.getvalue())
        self.assertIn('physics_force_kernel=', output.getvalue())

    def test_run_experiment_records_timing_profile_when_enabled(self):
        with contextlib.redirect_stdout(io.StringIO()):
            _, kb, _ = run_experiment(
                num_steps=4,
                num_initial_objects=2,
                seed=0,
                verbose=False,
                report_interval=4,
                force_backend='python',
                profile_timings=True,
            )

        profile = kb.runtime_profile
        stage_names = {item['stage'] for item in profile['stages']}

        self.assertTrue(profile['enabled'])
        self.assertEqual(4, profile['steps'])
        self.assertEqual('python', profile['force_backend'])
        self.assertIn('environment_step', stage_names)
        self.assertIn('perception', stage_names)
        self.assertIn('equation_final_discover', stage_names)
        self.assertIsNotNone(profile['hot_stage'])

    def test_backend_profile_comparison_is_non_final_and_writes_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, 'backend-profile.json')
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                report = run_backend_profile_comparison(
                    backends=['python', 'numpy'],
                    seeds=1,
                    steps=6,
                    object_counts=[2],
                    world_types=['standard'],
                    output_file=output_file,
                )
            with open(output_file, 'r', encoding='utf-8') as handle:
                saved = json.load(handle)

        self.assertFalse(report['runs_final'])
        self.assertEqual('backend_profile_comparison', report['run_kind'])
        self.assertEqual(2, len(report['rows']))
        self.assertTrue(report['all_metric_matches'])
        self.assertEqual(report['all_metric_matches'], saved['all_metric_matches'])
        self.assertIn('backend_summaries', report)
        self.assertTrue(report['rows'][0]['profile']['enabled'])
        self.assertIn('BACKEND PROFILE COMPARISON', output.getvalue())
        self.assertIn('All backend metrics match reference', output.getvalue())

    def test_math_final_artifact_prints_compact_upload_failure_summary(self):
        theory_memory = CumulativeTheoryMemory()
        result = {
            'context': 'standard',
            'seed': 0,
            'objects': 5,
            'steps': 40,
            'passed': True,
            'ready_for_final': True,
            'equation_passed': True,
            'label_leaks': [],
            'probe_suggestions': [],
            'interesting_score': 0.97,
            'interesting_equation': {
                'role': 'position_update_equation',
                'target': 'next_x',
                'expression': 'x + vx * dt',
                'score': 0.97,
                'parameters': {},
            },
        }
        starting = CumulativeTheoryMemory().memory_checkpoint_summary()
        upload_failure = {
            'status': 'failed',
            'repo_id': 'demo/artifacts',
            'path_in_repo': 'runs/demo-run/summary.json',
            'reason': 'upload_failed',
            'error': '403 Forbidden',
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, 'final.json')
            output = io.StringIO()
            with (
                mock.patch('main._upload_math_final_artifact', return_value=upload_failure),
                contextlib.redirect_stdout(output),
            ):
                persisted = _persist_math_final_artifact(
                    [result],
                    theory_memory,
                    artifact_output_file=output_file,
                    hf_output_repo='demo/artifacts',
                    run_id='demo-run',
                    run_config={'seeds': 1, 'world_types': ['standard']},
                    starting_memory_summary=starting,
                )

        summary_line = next(
            line for line in output.getvalue().splitlines()
            if line.startswith('HF_ARTIFACT_SUMMARY ')
        )
        parsed = parse_live_progress_line(summary_line)
        self.assertEqual('failed', persisted['hf_upload']['status'])
        self.assertEqual('failed', parsed['hf_upload']['status'])
        self.assertEqual('upload_failed', parsed['hf_upload']['reason'])
        self.assertEqual(1, parsed['result_count'])
        self.assertIn('experiment_design_cockpit', parsed)
        self.assertIn('weak_case_diagnostics', parsed)

    def test_failed_hf_upload_emits_reconstructable_log_chunks(self):
        theory_memory = CumulativeTheoryMemory()
        result = {
            'context': 'standard',
            'seed': 0,
            'objects': 5,
            'steps': 40,
            'passed': True,
            'ready_for_final': True,
            'equation_passed': True,
            'label_leaks': [],
            'probe_suggestions': [],
            'interesting_score': 0.97,
            'interesting_equation': {
                'role': 'position_update_equation',
                'target': 'next_x',
                'expression': 'x + vx * dt',
                'score': 0.97,
                'parameters': {},
            },
        }
        starting = CumulativeTheoryMemory().memory_checkpoint_summary()
        upload_failure = {
            'status': 'failed',
            'repo_id': 'demo/artifacts',
            'path_in_repo': 'runs/demo-run/summary.json',
            'reason': 'upload_failed',
            'error': '403 Forbidden',
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, 'final.json')
            output = io.StringIO()
            with (
                mock.patch('main._upload_math_final_artifact', return_value=upload_failure),
                contextlib.redirect_stdout(output),
            ):
                persisted = _persist_math_final_artifact(
                    [result],
                    theory_memory,
                    artifact_output_file=output_file,
                    hf_output_repo='demo/artifacts',
                    run_id='demo-run',
                    run_config={'seeds': 1, 'world_types': ['standard']},
                    starting_memory_summary=starting,
                )

        chunk_lines = [
            line for line in output.getvalue().splitlines()
            if line.startswith('HF_ARTIFACT_CHUNK ')
        ]
        self.assertTrue(chunk_lines)
        chunk = parse_live_progress_line(chunk_lines[0])
        self.assertEqual('hf_artifact_chunk', chunk['stream'])
        self.assertEqual('zlib+base64+json', chunk['encoding'])
        self.assertEqual(persisted['log_artifact_chunks'][0]['data'], chunk['data'])

    def test_artifact_log_chunks_split_large_artifact(self):
        chunks = _artifact_log_chunks(
            {'payload': 'x' * 5000},
            run_id='demo-run',
            max_chars=1024,
        )

        self.assertGreaterEqual(len(chunks), 1)
        self.assertEqual('demo-run', chunks[0]['run_id'])
        self.assertEqual(len(chunks), chunks[0]['total'])
        self.assertEqual('zlib+base64+json', chunks[0]['encoding'])

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
