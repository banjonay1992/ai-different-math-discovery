import contextlib
import io
import os
import sys
import tempfile
import unittest


PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'first_principles_ai'))
sys.path.insert(0, PROJECT_DIR)

from agent.discovery_loop import AutonomousDiscoveryLoop, CumulativeTheoryMemory
from agent.equation_workbench import EquationWorkbench, PrimitiveEquation
from agent.representation import KnowledgeBase
from main import (
    _equation_campaign_artifact_summary,
    _equation_metrics_from_knowledge,
    _planned_probe_actions,
    _parse_abstraction_transfer_worlds,
    _print_cumulative_theory_review,
    _run_equation_followup_cases,
    parse_live_progress_line,
    run_abstraction_transfer_campaign,
    run_live_progress_viewer,
    run_memory_efficiency_review,
    run_rediscovery_goal_progress_audit,
    upload_hf_artifact_file,
)


def equation(
    key,
    role,
    score=0.82,
    target='baseline_adjusted_delta_velocity',
    expression='k * unit_local_inferred_vector',
    parameters=None,
):
    return PrimitiveEquation(
        key=key,
        target=target,
        expression=expression,
        description='candidate',
        score=score,
        mse=0.02,
        baseline_mse=0.30,
        complexity=5,
        sample_count=120,
        parameters=parameters or {},
        role=role,
    )


class DiscoveryLoopTests(unittest.TestCase):
    def test_periodic_residual_generates_phase_concept_and_wait_probe(self):
        loop = AutonomousDiscoveryLoop()
        residual = equation(
            key='raw_eq:residual_periodic_x_80',
            role='residual_periodic_equation',
            target='baseline_adjusted_delta_vx',
            expression='a * sin(step/80) + b * cos(step/80)',
            parameters={'period_steps': 80, 'amplitude': 0.18},
        )

        report = loop.build_report([residual], step=180, current_count=5)
        packed = report.to_dict()

        self.assertEqual('probe_ready', packed['phase'])
        self.assertEqual('periodic_residual', packed['theories'][0]['theory_kind'])
        self.assertTrue(any(
            item['concept_kind'] == 'phase'
            for item in packed['concept_proposals']
        ))
        self.assertEqual('wait', packed['probe_plan']['action']['type'])
        self.assertIn('phase', packed['probe_plan']['expected_contrast'])

    def test_local_residual_generates_center_concept_and_spawn_probe(self):
        loop = AutonomousDiscoveryLoop()
        residual = equation(
            key='raw_eq:residual_local_direction_vector',
            role='local_residual_direction_equation',
            parameters={'center_x': 7.0, 'center_y': 13.0, 'k': 0.2},
        )

        report = loop.build_report(
            [residual],
            step=120,
            current_count=2,
            world_width=20.0,
            world_height=20.0,
        )
        packed = report.to_dict()

        self.assertEqual('local_direction_residual', packed['theories'][0]['theory_kind'])
        self.assertTrue(any(
            item['concept_kind'] == 'local_center'
            and item['parameters']['center_x'] == 7.0
            for item in packed['concept_proposals']
        ))
        self.assertEqual('spawn', packed['probe_plan']['action']['type'])
        self.assertAlmostEqual(10.6, packed['probe_plan']['action']['x'], delta=0.01)
        self.assertAlmostEqual(13.0, packed['probe_plan']['action']['y'], delta=0.01)

    def test_local_residual_generates_cutoff_operator_proposal(self):
        loop = AutonomousDiscoveryLoop()
        residual = equation(
            key='raw_eq:residual_local_direction_vector',
            role='local_residual_direction_equation',
            parameters={'center_x': 7.0, 'center_y': 13.0, 'k': 0.2},
        )

        report = loop.build_report(
            [residual],
            step=120,
            current_count=2,
            world_width=20.0,
            world_height=20.0,
        )
        packed = report.to_dict()

        self.assertTrue(any(
            item['operator_kind'] == 'localized_cutoff_window'
            and item['parameters']['relation'] == 'direction'
            and item['parameters']['center_x'] == 7.0
            for item in packed['operator_proposals']
        ))
        self.assertTrue(any(
            item['operator_kind'] == 'localized_tapered_power'
            and item['parameters']['relation'] == 'direction'
            and item['parameters']['center_x'] == 7.0
            for item in packed['operator_proposals']
        ))

    def test_discovery_loop_builds_abstraction_bridge_from_equivalent_concepts(self):
        loop = AutonomousDiscoveryLoop()
        equations = [
            equation(
                key='raw_eq:local_direction',
                role='local_residual_direction_equation',
                score=0.78,
                parameters={'center_x': 7.0, 'center_y': 13.0, 'k': 0.2},
            ),
            equation(
                key='raw_eq:cutoff_direction',
                role='generated_operator_cutoff_direction_equation',
                score=0.82,
                parameters={
                    'center_x': 7.0,
                    'center_y': 13.0,
                    'cutoff_radius': 5.5,
                    'cutoff_mse_improvement': 0.12,
                    'cutoff_vs_smooth_improvement': 0.09,
                },
            ),
            equation(
                key='raw_eq:tapered_direction',
                role='generated_operator_tapered_distance_direction_equation',
                score=0.88,
                parameters={
                    'center_x': 7.0,
                    'center_y': 13.0,
                    'cutoff_radius': 5.5,
                    'distance_exponent': 2.0,
                    'distance_mse_improvement': 0.15,
                    'tapered_vs_cutoff_improvement': 0.07,
                    'tapered_vs_smooth_improvement': 0.11,
                },
            ),
        ]

        report = loop.build_report(equations, step=240, current_count=4)
        bridges = report.to_dict()['abstraction_bridges']

        self.assertTrue(bridges)
        bridge_kinds = {bridge['abstraction_kind'] for bridge in bridges}
        self.assertTrue({
            'localized_context',
            'domain_boundary',
            'scaled_domain_effect',
        } & bridge_kinds)
        top_bridge = bridges[0]
        self.assertGreaterEqual(len(top_bridge['source_concept_keys']), 2)
        self.assertTrue(top_bridge['compressed_expression'])
        self.assertTrue(top_bridge['unrelated_world'])
        self.assertTrue(top_bridge['solve_hint'])

    def test_cumulative_memory_promotes_abstraction_to_unrelated_transfer_plan(self):
        loop = AutonomousDiscoveryLoop()
        memory = CumulativeTheoryMemory()
        equations = [
            equation(
                key='raw_eq:local_direction',
                role='local_residual_direction_equation',
                score=0.80,
                parameters={'center_x': 8.0, 'center_y': 12.0, 'k': 0.2},
            ),
            equation(
                key='raw_eq:cutoff_direction',
                role='generated_operator_cutoff_direction_equation',
                score=0.84,
                parameters={
                    'center_x': 8.0,
                    'center_y': 12.0,
                    'cutoff_radius': 6.0,
                    'cutoff_mse_improvement': 0.12,
                    'cutoff_vs_smooth_improvement': 0.08,
                },
            ),
            equation(
                key='raw_eq:tapered_direction',
                role='generated_operator_tapered_distance_direction_equation',
                score=0.90,
                parameters={
                    'center_x': 8.0,
                    'center_y': 12.0,
                    'cutoff_radius': 6.0,
                    'distance_exponent': 2.0,
                    'distance_mse_improvement': 0.16,
                    'tapered_vs_cutoff_improvement': 0.06,
                    'tapered_vs_smooth_improvement': 0.10,
                },
            ),
        ]

        for seed, context in enumerate(('localized_gravity', 'time_varying')):
            report = loop.build_report(equations, step=200 + seed, current_count=4)
            memory.record_result(context, seed, report)

        bridges = memory.abstraction_bridges(limit=5)
        self.assertTrue(bridges)
        self.assertIn(
            bridges[0]['status'],
            {'reusable_abstraction', 'transfer_ready'},
        )
        self.assertGreaterEqual(bridges[0]['support_count'], 2)

        probes = memory.abstraction_transfer_experiments(limit=5)
        self.assertTrue(probes)
        self.assertEqual('abstraction_transfer_probe', probes[0]['experiment_kind'])
        self.assertEqual('abstraction_unrelated_world', probes[0]['target_context'])
        self.assertTrue(probes[0]['compressed_expression'])
        self.assertNotIn(
            probes[0]['unrelated_world'],
            set(probes[0]['source_contexts']),
        )

        plans = memory.planned_experiments(
            world_types=[
                'standard',
                'localized_gravity',
                'time_varying',
                'hidden_procedural',
            ],
            object_counts=[5],
            steps=240,
            limit=8,
        )
        abstraction_plans = [
            plan for plan in plans
            if plan['experiment_kind'] == 'abstraction_transfer_probe'
        ]
        self.assertTrue(abstraction_plans)
        self.assertTrue(abstraction_plans[0]['compressed_expression'])
        self.assertEqual(
            abstraction_plans[0]['world_type'],
            abstraction_plans[0]['unrelated_world'],
        )

    def test_abstraction_records_roundtrip_readiness_and_summary(self):
        loop = AutonomousDiscoveryLoop()
        memory = CumulativeTheoryMemory()
        report = loop.build_report([
            equation(
                key='raw_eq:local_direction',
                role='local_residual_direction_equation',
                parameters={'center_x': 9.0, 'center_y': 11.0, 'k': 0.2},
            ),
            equation(
                key='raw_eq:tapered_direction',
                role='generated_operator_tapered_distance_direction_equation',
                score=0.87,
                parameters={
                    'center_x': 9.0,
                    'center_y': 11.0,
                    'cutoff_radius': 5.0,
                    'distance_exponent': 2.0,
                    'distance_mse_improvement': 0.12,
                    'tapered_vs_cutoff_improvement': 0.06,
                    'tapered_vs_smooth_improvement': 0.09,
                },
            ),
        ], step=160, current_count=3)
        memory.record_result('localized_gravity', 0, report)

        packed = memory.to_dict()
        restored = CumulativeTheoryMemory.from_dict(packed)
        readiness = restored.discovery_readiness_report()

        self.assertTrue(restored.abstraction_records)
        self.assertTrue(restored.abstraction_bridges())
        self.assertIn('abstraction_discovery_loop', readiness['gates'])
        self.assertIn('abstraction_bridges', packed)
        self.assertIn('Abstraction discovery', restored.summary())

    def test_abstraction_transfer_campaign_records_empirical_outcomes(self):
        self.assertEqual(
            ['standard', 'hidden_procedural'],
            _parse_abstraction_transfer_worlds('standard,hidden_procedural'),
        )
        with self.assertRaisesRegex(ValueError, 'Unknown abstraction target'):
            _parse_abstraction_transfer_worlds('standard,not_a_world')

        expected_outcomes = {
            'confirmed': 'abstraction_transfer_confirmed',
            'weak': 'abstraction_transfer_weak',
            'absent': 'abstraction_transfer_absent',
        }
        for mode, expected_outcome in expected_outcomes.items():
            with self.subTest(mode=mode):
                memory = CumulativeTheoryMemory()
                with contextlib.redirect_stdout(io.StringIO()):
                    summary = run_abstraction_transfer_campaign(
                        theory_memory=memory,
                        seed_start=10,
                        steps=90,
                        target_world_types=[
                            'standard',
                            'time_varying',
                            'hidden_procedural',
                        ],
                        outcome_mode=mode,
                    )

                self.assertEqual('abstraction_transfer_campaign', summary['run_kind'])
                self.assertFalse(summary['runs_final'])
                self.assertGreaterEqual(len(summary['source_results']), 2)
                self.assertGreaterEqual(summary['bridge_count'], 2)
                source_contexts = {
                    item['context'] for item in summary['source_results']
                }
                self.assertGreaterEqual(len(source_contexts), 2)
                self.assertEqual(
                    'abstraction_transfer_probe',
                    summary['selected_plan']['experiment_kind'],
                )
                transfer = summary['transfer_result']
                self.assertEqual(
                    expected_outcome,
                    transfer['outcome']['outcome'],
                )
                evidence = summary['abstraction_discovery_evidence']
                self.assertEqual(1, evidence['transfer_outcome_count'])
                self.assertEqual(
                    int(mode == 'confirmed'),
                    evidence['transfer_confirmed_count'],
                )
                self.assertEqual(
                    int(mode == 'weak'),
                    evidence['transfer_weak_count'],
                )
                self.assertEqual(
                    int(mode == 'absent'),
                    evidence['transfer_absent_count'],
                )
                readiness_gate = summary['readiness']['gates'][
                    'abstraction_discovery_loop'
                ]
                self.assertTrue(readiness_gate['passed'])
                self.assertEqual(
                    1,
                    readiness_gate['evidence']['transfer_outcome_count'],
                )
                progress_gate = summary['rediscovery_goal_progress']['gates'][
                    'abstraction_discovery_transfer'
                ]
                self.assertGreaterEqual(progress_gate['score'], 0.9)
                self.assertIn('Abstraction discovery', memory.summary())

    def test_distance_scaled_residual_generates_strength_law_concept(self):
        loop = AutonomousDiscoveryLoop()
        residual = equation(
            key='raw_eq:residual_inferred_direction_distance_2',
            role='residual_distance_scaled_direction_equation',
            expression='k * unit_inferred_vector / separation^2',
            parameters={
                'center_x': 8.0,
                'center_y': 12.0,
                'distance_exponent': 2.0,
                'distance_mse_improvement': 0.44,
            },
        )

        report = loop.build_report(
            [residual],
            step=160,
            current_count=4,
            world_width=20.0,
            world_height=20.0,
        )
        packed = report.to_dict()

        self.assertEqual(
            'distance_scaled_direction_residual',
            packed['theories'][0]['theory_kind'],
        )
        self.assertTrue(any(
            item['concept_kind'] == 'distance_strength_law'
            and item['parameters']['distance_exponent'] == 2.0
            for item in packed['concept_proposals']
        ))
        self.assertEqual('spawn', packed['probe_plan']['action']['type'])
        self.assertIn('separation^2.0', packed['probe_plan']['expected_contrast'])
        self.assertTrue(any(
            item['operator_kind'] == 'inverse_separation_power'
            and item['parameters']['distance_exponent'] == 2.0
            for item in packed['operator_proposals']
        ))
        self.assertTrue(any(
            item['check_kind'] == 'simpler_model_contrast'
            and item['status'] == 'passed'
            for item in packed['proof_checks']
        ))

    def test_cutoff_residual_generates_boundary_concept_and_probe(self):
        loop = AutonomousDiscoveryLoop()
        residual = equation(
            key='raw_eq:generated_cutoff_direction_4',
            role='generated_operator_cutoff_direction_equation',
            expression='k * inside(separation <= 4) * unit_generated_center_vector',
            parameters={
                'center_x': 8.0,
                'center_y': 12.0,
                'cutoff_radius': 4.0,
                'cutoff_mse_improvement': 0.52,
                'cutoff_vs_smooth_improvement': 0.31,
            },
        )

        report = loop.build_report(
            [residual],
            step=160,
            current_count=4,
            world_width=20.0,
            world_height=20.0,
        )
        packed = report.to_dict()

        self.assertEqual(
            'generated_cutoff_direction_residual',
            packed['theories'][0]['theory_kind'],
        )
        self.assertTrue(any(
            item['concept_kind'] == 'localized_cutoff_region'
            and item['parameters']['cutoff_radius'] == 4.0
            for item in packed['concept_proposals']
        ))
        self.assertTrue(any(
            item['operator_kind'] == 'localized_tapered_power'
            and item['parameters']['cutoff_radius'] == 4.0
            for item in packed['operator_proposals']
        ))
        self.assertTrue(any(
            item['check_kind'] == 'near_far_contrast'
            and item['status'] == 'passed'
            for item in packed['proof_checks']
        ))
        self.assertEqual('spawn', packed['probe_plan']['action']['type'])
        self.assertIn('boundary', packed['probe_plan']['reason'])
        self.assertIn('beyond radius 4.0', packed['probe_plan']['expected_contrast'])

    def test_tapered_residual_generates_shape_concept_and_probe(self):
        loop = AutonomousDiscoveryLoop()
        residual = equation(
            key='raw_eq:generated_tapered_direction_4_2',
            role='generated_operator_tapered_distance_direction_equation',
            expression='k * taper(separation, 4) * unit_generated_center_vector / separation^2',
            parameters={
                'center_x': 8.0,
                'center_y': 12.0,
                'cutoff_radius': 4.0,
                'distance_exponent': 2.0,
                'tapered_mse_improvement': 0.64,
                'tapered_vs_smooth_improvement': 0.32,
                'tapered_vs_cutoff_improvement': 0.28,
            },
        )

        report = loop.build_report(
            [residual],
            step=160,
            current_count=4,
            world_width=20.0,
            world_height=20.0,
        )
        packed = report.to_dict()

        self.assertEqual(
            'generated_tapered_distance_direction_residual',
            packed['theories'][0]['theory_kind'],
        )
        self.assertTrue(any(
            item['concept_kind'] == 'boundary_taper'
            and item['parameters']['cutoff_radius'] == 4.0
            for item in packed['concept_proposals']
        ))
        self.assertFalse(any(
            item['operator_kind'] in {
                'localized_cutoff_window',
                'localized_tapered_power',
            }
            for item in packed['operator_proposals']
        ))
        self.assertTrue(any(
            item['check_kind'] == 'shape_contrast'
            and item['status'] == 'passed'
            for item in packed['proof_checks']
        ))
        self.assertEqual('spawn', packed['probe_plan']['action']['type'])
        self.assertIn('taper shape', packed['probe_plan']['reason'])
        self.assertIn('tapering toward radius 4.0', packed['probe_plan']['expected_contrast'])

    def test_competing_residual_theories_choose_disagreement_probe(self):
        loop = AutonomousDiscoveryLoop()
        direction = equation(
            key='raw_eq:residual_local_direction_vector',
            role='local_residual_direction_equation',
            score=0.82,
            parameters={'center_x': 10.0, 'center_y': 10.0},
        )
        perpendicular = equation(
            key='raw_eq:residual_local_perpendicular_vector',
            role='local_residual_perpendicular_equation',
            score=0.79,
            expression='k * perpendicular(unit_local_inferred_vector)',
            parameters={'center_x': 10.0, 'center_y': 10.0},
        )

        report = loop.build_report(
            [direction, perpendicular],
            step=140,
            current_count=3,
            world_width=20.0,
            world_height=20.0,
        )
        packed = report.to_dict()

        self.assertEqual(2, len(packed['probe_plan']['theory_keys']))
        self.assertIn('competing residual theories', packed['probe_plan']['reason'])
        self.assertIn('weaken', packed['probe_plan']['expected_contrast'])
        signature = packed['probe_plan']['disagreement_signature']
        self.assertEqual('vector_direction_disagreement', signature['mode'])
        self.assertEqual(
            ['direction_test_east', 'direction_test_north'],
            [point['label'] for point in signature['probe_points']],
        )
        self.assertTrue(any(
            'quarter turn' in item['prediction']
            for item in signature['rival_predictions']
        ))

    def test_competing_cutoff_and_distance_scaled_choose_boundary_probe(self):
        loop = AutonomousDiscoveryLoop()
        cutoff = equation(
            key='raw_eq:generated_cutoff_direction_4',
            role='generated_operator_cutoff_direction_equation',
            score=0.82,
            expression='k * inside(separation <= 4) * unit_generated_center_vector',
            parameters={
                'center_x': 8.0,
                'center_y': 12.0,
                'cutoff_radius': 4.0,
                'cutoff_mse_improvement': 0.52,
                'cutoff_vs_smooth_improvement': 0.31,
            },
        )
        distance = equation(
            key='raw_eq:generated_operator_distance_2',
            role='generated_operator_distance_scaled_direction_equation',
            score=0.78,
            expression='k * unit_generated_center_vector / separation^2',
            parameters={
                'center_x': 8.0,
                'center_y': 12.0,
                'distance_exponent': 2.0,
                'distance_mse_improvement': 0.41,
            },
        )

        report = loop.build_report(
            [cutoff, distance],
            step=170,
            current_count=3,
            world_width=20.0,
            world_height=20.0,
        )
        packed = report.to_dict()

        self.assertEqual(2, len(packed['probe_plan']['theory_keys']))
        self.assertIn('cutoff locality', packed['probe_plan']['reason'])
        self.assertIn('outside samples should collapse', packed['probe_plan']['expected_contrast'])
        signature = packed['probe_plan']['disagreement_signature']
        self.assertEqual('cutoff_boundary_vs_smooth_falloff', signature['mode'])
        self.assertEqual(
            ['inside_boundary', 'just_outside_boundary'],
            [point['label'] for point in signature['probe_points']],
        )
        self.assertTrue(any(
            'near-zero outside' in item['prediction']
            for item in signature['rival_predictions']
        ))
        self.assertTrue(any(
            'smooth nonzero falloff' in item['prediction']
            for item in signature['rival_predictions']
        ))

    def test_competing_tapered_and_cutoff_choose_shape_probe(self):
        loop = AutonomousDiscoveryLoop()
        tapered = equation(
            key='raw_eq:generated_tapered_direction_4_2',
            role='generated_operator_tapered_distance_direction_equation',
            score=0.83,
            expression='k * taper(separation, 4) * unit_generated_center_vector / separation^2',
            parameters={
                'center_x': 8.0,
                'center_y': 12.0,
                'cutoff_radius': 4.0,
                'distance_exponent': 2.0,
                'tapered_mse_improvement': 0.64,
                'tapered_vs_smooth_improvement': 0.32,
                'tapered_vs_cutoff_improvement': 0.28,
            },
        )
        cutoff = equation(
            key='raw_eq:generated_cutoff_direction_4',
            role='generated_operator_cutoff_direction_equation',
            score=0.79,
            expression='k * inside(separation <= 4) * unit_generated_center_vector',
            parameters={
                'center_x': 8.0,
                'center_y': 12.0,
                'cutoff_radius': 4.0,
                'cutoff_mse_improvement': 0.52,
                'cutoff_vs_smooth_improvement': 0.31,
            },
        )

        report = loop.build_report(
            [tapered, cutoff],
            step=170,
            current_count=3,
            world_width=20.0,
            world_height=20.0,
        )
        packed = report.to_dict()

        self.assertEqual(2, len(packed['probe_plan']['theory_keys']))
        self.assertIn('tapered local shape', packed['probe_plan']['reason'])
        self.assertIn('graded residual shape', packed['probe_plan']['expected_contrast'])
        signature = packed['probe_plan']['disagreement_signature']
        self.assertEqual('taper_shape_vs_hard_boundary', signature['mode'])
        self.assertEqual(
            ['near_center', 'mid_region', 'just_inside_boundary', 'just_outside_boundary'],
            [point['label'] for point in signature['probe_points']],
        )
        self.assertTrue(any(
            'graded residual' in item['prediction']
            for item in signature['rival_predictions']
        ))

    def test_competing_generated_exponents_choose_near_far_probe(self):
        loop = AutonomousDiscoveryLoop()
        shallow = equation(
            key='raw_eq:generated_operator_distance_1_5',
            role='generated_operator_distance_scaled_direction_equation',
            score=0.78,
            expression='k * unit_generated_center_vector / separation^1_5',
            parameters={
                'center_x': 10.0,
                'center_y': 10.0,
                'distance_exponent': 1.5,
                'distance_mse_improvement': 0.18,
            },
        )
        steep = equation(
            key='raw_eq:generated_operator_distance_2',
            role='generated_operator_distance_scaled_direction_equation',
            score=0.76,
            expression='k * unit_generated_center_vector / separation^2',
            parameters={
                'center_x': 10.0,
                'center_y': 10.0,
                'distance_exponent': 2.0,
                'distance_mse_improvement': 0.22,
            },
        )

        report = loop.build_report(
            [shallow, steep],
            step=180,
            current_count=3,
            world_width=20.0,
            world_height=20.0,
        )
        packed = report.to_dict()

        self.assertEqual(2, len(packed['probe_plan']['theory_keys']))
        self.assertIn('competing distance exponents', packed['probe_plan']['reason'])
        self.assertIn('exponents', packed['probe_plan']['expected_contrast'])
        signature = packed['probe_plan']['disagreement_signature']
        self.assertEqual('distance_exponent_race', signature['mode'])
        self.assertEqual(
            ['near_center', 'far_from_center'],
            [point['label'] for point in signature['probe_points']],
        )
        self.assertTrue(all(
            'separation^-' in item['prediction']
            for item in signature['rival_predictions']
        ))
        memory = CumulativeTheoryMemory()
        memory.record_result('inverse_square_repulsion', 0, report)
        experiment = memory.to_dict()['next_experiments'][0]
        self.assertEqual('model_disagreement_probe', experiment['experiment_kind'])
        self.assertIn('separation^-', experiment['primary_theory_label'])
        self.assertTrue(experiment['rival_theory_labels'])
        self.assertIn('separation^-', experiment['rival_theory_labels'][0])
        before_priority = experiment['priority']
        original_plan = memory.planned_experiments(
            world_types=['inverse_square_repulsion'],
            object_counts=[5],
            steps=240,
            seed_start=0,
            limit=1,
        )[0]
        still_open_outcome = memory.record_planned_result(
            original_plan,
            context='inverse_square_repulsion',
            seed=1,
            report=report,
        )
        self.assertEqual('disagreement_still_open', still_open_outcome['outcome'])
        refined = memory.disagreement_experiments(limit=1)[0]
        self.assertEqual('needs_refinement', refined['stagnation_status'])
        self.assertEqual(1, refined['proof_evidence']['still_open_count'])
        self.assertLess(refined['priority'], before_priority)
        self.assertEqual(1, refined['disagreement_signature']['refinement_level'])
        self.assertEqual(
            ['very_near_center', 'mid_log_check', 'very_far_from_center'],
            [
                point['label']
                for point in refined['disagreement_signature']['probe_points']
            ],
        )
        refined_plan = memory.planned_experiments(
            world_types=['inverse_square_repulsion'],
            object_counts=[5],
            steps=240,
            seed_start=0,
            limit=1,
        )[0]
        if refined_plan['experiment_kind'] == 'model_disagreement_probe':
            self.assertEqual(
                ['very_near_center', 'mid_log_check', 'very_far_from_center'],
                [action['probe_label'] for action in _planned_probe_actions(refined_plan)],
            )
        rival_only_report = loop.build_report(
            [steep],
            step=190,
            current_count=3,
            world_width=20.0,
            world_height=20.0,
        )
        outcome = memory.evaluate_planned_result(
            original_plan,
            context='inverse_square_repulsion',
            seed=1,
            report=rival_only_report,
        )
        self.assertEqual('rival_confirmed', outcome['outcome'])
        self.assertFalse(outcome['found_family'])
        self.assertTrue(outcome['rival_found'])

    def test_cumulative_memory_promotes_recorded_model_disagreement(self):
        loop = AutonomousDiscoveryLoop()
        cutoff = equation(
            key='raw_eq:generated_cutoff_direction_4',
            role='generated_operator_cutoff_direction_equation',
            score=0.82,
            expression='k * inside(separation <= 4) * unit_generated_center_vector',
            parameters={
                'center_x': 8.0,
                'center_y': 12.0,
                'cutoff_radius': 4.0,
                'cutoff_mse_improvement': 0.52,
                'cutoff_vs_smooth_improvement': 0.31,
            },
        )
        distance = equation(
            key='raw_eq:generated_operator_distance_2',
            role='generated_operator_distance_scaled_direction_equation',
            score=0.78,
            expression='k * unit_generated_center_vector / separation^2',
            parameters={
                'center_x': 8.0,
                'center_y': 12.0,
                'distance_exponent': 2.0,
                'distance_mse_improvement': 0.41,
            },
        )
        report = loop.build_report(
            [cutoff, distance],
            step=170,
            current_count=3,
            world_width=20.0,
            world_height=20.0,
        )
        memory = CumulativeTheoryMemory()

        memory.record_result('localized_gravity', 0, report)
        packed = memory.to_dict()

        self.assertEqual(1, len(packed['disagreement_records']))
        self.assertEqual(
            'cutoff_boundary_vs_smooth_falloff',
            packed['disagreement_records'][0]['mode'],
        )
        self.assertEqual(
            'model_disagreement_probe',
            packed['next_experiments'][0]['experiment_kind'],
        )
        self.assertEqual(
            'cutoff_boundary_vs_smooth_falloff',
            packed['next_experiments'][0]['disagreement_signature']['mode'],
        )
        self.assertIn(
            'distance_scaled_direction_residual',
            packed['next_experiments'][0]['rival_theory_kinds'],
        )
        plan = memory.planned_experiments(
            world_types=['localized_gravity', 'inverse_square_repulsion'],
            object_counts=[5],
            steps=240,
            seed_start=0,
            limit=1,
        )[0]
        self.assertEqual('localized_gravity', plan['world_type'])
        self.assertEqual('model_disagreement_probe', plan['experiment_kind'])
        self.assertEqual(
            'cutoff_boundary_vs_smooth_falloff',
            plan['disagreement_signature']['mode'],
        )
        self.assertEqual('spawn', plan['probe_action']['type'])
        actions = _planned_probe_actions(plan)
        self.assertEqual(
            ['inside_boundary', 'just_outside_boundary'],
            [action['probe_label'] for action in actions],
        )
        self.assertTrue(all(
            action['source'] == 'planned_model_disagreement_probe'
            for action in actions
        ))
        self.assertIn('Disagreement probes:', memory.summary())
        rival_report = loop.build_report(
            [distance],
            step=190,
            current_count=3,
            world_width=20.0,
            world_height=20.0,
        )
        outcome = memory.record_planned_result(
            plan,
            context='localized_gravity',
            seed=1,
            report=rival_report,
        )
        self.assertEqual('rival_confirmed', outcome['outcome'])
        self.assertTrue(outcome['rival_found'])
        self.assertIn('localized_gravity/', outcome['target_scope'])
        revised_family = memory.to_dict()['families'][plan['theory_kind']]
        self.assertEqual('domain_limited', revised_family['generalization_status'])
        self.assertIn(
            outcome['target_scope'],
            revised_family['domain_hypothesis']['excluded_contexts'],
        )
        self.assertNotIn(
            'localized_gravity',
            revised_family['domain_hypothesis']['excluded_contexts'],
        )
        self.assertNotEqual(
            'model_disagreement_probe',
            memory.next_experiments(limit=1)[0]['experiment_kind'],
        )
        self.assertEqual(
            'rival_recently_confirmed',
            memory.disagreement_experiments(limit=1)[0]['stagnation_status'],
        )

    def test_model_disagreement_both_winners_becomes_domain_split(self):
        loop = AutonomousDiscoveryLoop()
        shallow = equation(
            key='raw_eq:generated_operator_distance_1',
            role='generated_operator_distance_scaled_direction_equation',
            score=0.78,
            expression='k * unit_generated_center_vector / separation^1',
            parameters={
                'center_x': 10.0,
                'center_y': 10.0,
                'distance_exponent': 1.0,
            },
        )
        steep = equation(
            key='raw_eq:generated_operator_distance_2',
            role='generated_operator_distance_scaled_direction_equation',
            score=0.76,
            expression='k * unit_generated_center_vector / separation^2',
            parameters={
                'center_x': 10.0,
                'center_y': 10.0,
                'distance_exponent': 2.0,
            },
        )
        memory = CumulativeTheoryMemory()
        memory.record_result(
            'inverse_square_repulsion',
            0,
            loop.build_report([shallow, steep], step=180),
        )
        plan = memory.planned_experiments(
            world_types=['inverse_square_repulsion'],
            object_counts=[5],
            steps=240,
            seed_start=0,
            limit=1,
        )[0]
        target_outcome = memory.evaluate_planned_result(
            plan,
            context='inverse_square_repulsion',
            seed=1,
            report={
                'theories': [{
                    'theory_kind': 'generated_distance_scaled_direction_residual',
                    'parameters': {'distance_exponent': 1.0},
                    'score': 0.78,
                }],
                'proof_checks': [],
            },
        )
        rival_outcome = memory.evaluate_planned_result(
            plan,
            context='inverse_square_repulsion',
            seed=2,
            report={
                'theories': [{
                    'theory_kind': 'generated_distance_scaled_direction_residual',
                    'parameters': {'distance_exponent': 2.0},
                    'score': 0.76,
                }],
                'proof_checks': [],
            },
        )
        memory.planned_outcomes.extend([target_outcome, rival_outcome])

        split = memory.model_disagreement_domain_split_experiments(limit=1)[0]
        split_plan = memory.planned_experiments(
            world_types=['inverse_square_repulsion'],
            object_counts=[5],
            steps=240,
            seed_start=0,
            limit=1,
        )[0]
        supported = memory.evaluate_planned_result(
            split_plan,
            context='inverse_square_repulsion',
            seed=split_plan['seed'],
            report={
                'theories': [
                    {
                        'theory_kind': 'generated_distance_scaled_direction_residual',
                        'parameters': {'distance_exponent': 1.0},
                        'score': 0.78,
                    },
                    {
                        'theory_kind': 'generated_distance_scaled_direction_residual',
                        'parameters': {'distance_exponent': 2.0},
                        'score': 0.76,
                    },
                ],
                'proof_checks': [],
            },
        )

        self.assertEqual([], memory.disagreement_experiments(limit=1))
        self.assertEqual(
            'model_disagreement_domain_split',
            split['experiment_kind'],
        )
        self.assertEqual('split_suspected', split['domain_split_hypothesis']['status'])
        self.assertTrue(split['quick_probe'])
        self.assertEqual(
            'model_disagreement_domain_split',
            split_plan['experiment_kind'],
        )
        self.assertLess(split_plan['steps'], split_plan['full_steps'])
        self.assertEqual(
            'planned_model_disagreement_domain_split',
            _planned_probe_actions(split_plan)[0]['source'],
        )
        self.assertEqual(
            'model_disagreement_domain_split_supported',
            supported['outcome'],
        )
        self.assertIn(
            'model_disagreement_domain_split_experiments',
            memory.to_dict(),
        )
        self.assertIn('Model-disagreement domain splits:', memory.summary())

    def test_cumulative_memory_builds_representation_agenda_from_disagreement(self):
        loop = AutonomousDiscoveryLoop()
        shallow = equation(
            key='raw_eq:generated_operator_distance_1_5',
            role='generated_operator_distance_scaled_direction_equation',
            score=0.78,
            expression='k * unit_generated_center_vector / separation^1_5',
            parameters={
                'center_x': 10.0,
                'center_y': 10.0,
                'distance_exponent': 1.5,
                'distance_mse_improvement': 0.18,
            },
        )
        steep = equation(
            key='raw_eq:generated_operator_distance_2',
            role='generated_operator_distance_scaled_direction_equation',
            score=0.76,
            expression='k * unit_generated_center_vector / separation^2',
            parameters={
                'center_x': 10.0,
                'center_y': 10.0,
                'distance_exponent': 2.0,
                'distance_mse_improvement': 0.22,
            },
        )
        report = loop.build_report(
            [shallow, steep],
            step=180,
            current_count=3,
            world_width=20.0,
            world_height=20.0,
        )
        memory = CumulativeTheoryMemory()

        memory.record_result('inverse_square_repulsion', 0, report)
        agenda = memory.representation_agenda(limit=3)
        priors = memory.generated_operator_priors(limit=4)
        packed = memory.to_dict()

        self.assertTrue(agenda)
        exponent_item = next(
            item for item in agenda
            if item['name'] == 'separation_exponent_from_log_ratio'
        )
        self.assertEqual('derived_variable', exponent_item['proposal_kind'])
        self.assertEqual('model_disagreement', exponent_item['source'])
        self.assertIn('log(abs(residual_near)', exponent_item['expression'])
        self.assertEqual(
            'distance_exponent_race',
            exponent_item['evidence']['mode'],
        )
        self.assertIn('representation_agenda', packed)
        self.assertTrue(any(
            item['name'] == 'separation_exponent_from_log_ratio'
            for item in packed['representation_agenda']
        ))
        self.assertIn('generated_operator_priors', packed)
        self.assertIn('first_principles_basis', packed)
        self.assertIn('adaptive_dimension_agenda', packed)
        self.assertIn('algebraic_foundation_baseline', packed)
        self.assertIn('algebraic_expression_agenda', packed)
        self.assertIn('operator_prior_budget_report', packed)
        self.assertTrue(priors)
        prior = next(
            prior for prior in priors
            if prior['operator_kind'] == 'inverse_separation_power'
            and prior['parameters']['distance_exponent'] == 1.5
            and prior['parameters']['center_x'] == 10.0
        )
        self.assertIn('order_metric', prior['first_principles'])
        self.assertIn('residual_strength_exponent', prior['adaptive_dimensions'])
        self.assertIn('power_law', prior['algebraic_families'])
        self.assertIn('rational_ratio', prior['algebraic_families'])
        self.assertIn('field', prior['algebraic_structures'])
        self.assertIn('metric_space', prior['algebraic_structures'])
        self.assertIn('domain_nonzero_positive', prior['algebraic_proof_obligations'])
        self.assertTrue(prior['algebraic_search_controls']['require_heldout_score'])
        foundation = memory.algebraic_foundation_baseline()
        self.assertGreaterEqual(foundation['expression_family_count'], 16)
        self.assertGreaterEqual(foundation['structure_count'], 10)
        self.assertGreaterEqual(foundation['proof_obligation_count'], 10)
        family_keys = {
            item['key']
            for item in foundation['expression_families']
        }
        self.assertIn('polynomial_basis', family_keys)
        self.assertIn('matrix_linear_transform', family_keys)
        self.assertIn('probability_statistics', family_keys)
        algebraic_agenda = memory.algebraic_expression_agenda(limit=4)
        self.assertTrue(any(
            'power_law' in item['expression_families']
            and 'heldout_counterexample' in item['proof_obligations']
            for item in algebraic_agenda
        ))
        dimensions = memory.adaptive_dimension_agenda(limit=4)
        self.assertTrue(any(
            item['name'] == 'residual_strength_exponent'
            for item in dimensions
        ))
        self.assertIn('Representation agenda:', memory.summary())
        self.assertIn('Adaptive dimensions:', memory.summary())
        self.assertIn('Algebraic foundation:', memory.summary())
        self.assertIn('Algebraic expression agenda:', memory.summary())

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            _print_cumulative_theory_review(memory)
        printed = output.getvalue()

        self.assertIn('Theory representation agenda:', printed)
        self.assertIn('Theory adaptive dimensions:', printed)
        self.assertIn('Theory algebraic foundation:', printed)
        self.assertIn('Theory algebraic expression agenda:', printed)
        self.assertIn('Theory generated operator priors:', printed)
        self.assertIn('separation_exponent_from_log_ratio', printed)

        recorded = memory.record_operator_prior_results(
            'inverse_square_repulsion',
            1,
            {
                'operator_prior_results': [{
                    'operator_key': prior['key'],
                    'operator_kind': prior['operator_kind'],
                    'outcome': 'confirmed',
                    'best_score': 0.84,
                    'matching_equation_count': 1,
                    'parameters': prior['parameters'],
                    'best_equation': {
                        'role': 'generated_operator_distance_scaled_direction_equation',
                        'expression': 'k * unit_generated_center_vector / separation^2.25',
                        'score': 0.84,
                        'parameters': {
                            **prior['parameters'],
                            'distance_exponent': 2.25,
                        },
                    },
                }]
            },
        )
        boosted = next(
            item for item in memory.generated_operator_priors(limit=4)
            if item['key'] == prior['key']
        )
        packed_after_feedback = memory.to_dict()

        self.assertEqual(1, len(recorded))
        self.assertGreater(boosted['usefulness'], prior['usefulness'])
        self.assertEqual(1, boosted['feedback']['confirmed_count'])
        self.assertAlmostEqual(
            2.25,
            boosted['parameters']['distance_exponent'],
            delta=0.01,
        )
        self.assertEqual(prior['key'], boosted['refined_from_operator_key'])
        self.assertEqual(
            'confirmed',
            packed_after_feedback['operator_prior_outcomes'][0]['outcome'],
        )
        self.assertAlmostEqual(
            2.25,
            packed_after_feedback['operator_prior_outcomes'][0]['refined_parameters']['distance_exponent'],
            delta=0.01,
        )
        self.assertIn('operator_prior_feedback', packed_after_feedback)
        self.assertIn('Operator prior feedback:', memory.summary())
        domain = memory.operator_prior_domains(limit=3)[0]
        self.assertEqual(prior['key'], domain['operator_key'])
        self.assertIn(
            'inverse_square_repulsion',
            domain['domain_hypothesis']['included_contexts'],
        )
        self.assertIn('operator_prior_domains', packed_after_feedback)
        self.assertIn('Operator prior domains:', memory.summary())

        memory.record_operator_prior_results(
            'standard',
            2,
            {
                'operator_prior_results': [{
                    'operator_key': prior['key'],
                    'operator_kind': prior['operator_kind'],
                    'outcome': 'unmatched',
                    'best_score': 0.0,
                    'matching_equation_count': 0,
                    'parameters': prior['parameters'],
                }]
            },
        )
        domain_after_failure = next(
            item for item in memory.operator_prior_domains(limit=3)
            if item['operator_key'] == prior['key']
        )

        self.assertIn(
            'standard',
            domain_after_failure['domain_hypothesis']['excluded_contexts'],
        )
        self.assertTrue(memory.generated_operator_priors(context='inverse_square_repulsion'))
        self.assertFalse(any(
            item['key'] == prior['key']
            for item in memory.generated_operator_priors(context='standard')
        ))
        anomalies = memory.operator_prior_anomalies(limit=3)
        self.assertTrue(anomalies)
        anomaly = next(
            item for item in anomalies
            if item['operator_key'] == prior['key']
        )
        self.assertEqual('domain_break', anomaly['anomaly_kind'])
        self.assertEqual('standard', anomaly['failure_context'])
        self.assertIn('inverse_square_repulsion', anomaly['support_contexts'])
        self.assertIn('operator_prior_anomalies', memory.to_dict())
        self.assertIn('Operator prior anomalies:', memory.summary())
        anomaly_agenda = memory.representation_agenda(limit=8)
        anomaly_domain_item = next(
            item for item in anomaly_agenda
            if item['source'] == 'operator_prior_anomaly'
        )
        self.assertEqual(
            'operator_prior_domain_predicate',
            anomaly_domain_item['name'],
        )
        self.assertEqual('domain_predicate', anomaly_domain_item['proposal_kind'])
        self.assertIn(
            'residual_strength_ratio_matches_refined_exponent',
            anomaly_domain_item['expression'],
        )
        self.assertEqual(
            'standard',
            anomaly_domain_item['evidence']['failure_context'],
        )
        self.assertEqual(
            prior['key'],
            anomaly_domain_item['evidence']['operator_key'],
        )
        anomaly_dimensions = memory.adaptive_dimension_agenda(limit=8)
        self.assertTrue(any(
            item['dimension_kind'] == 'failure_separator_axis'
            and item['evidence']['operator_key'] == prior['key']
            for item in anomaly_dimensions
        ))
        claim_experiment = next(
            item for item in memory.operator_prior_claim_experiments(limit=5)
            if item['operator_prior_key'] == prior['key']
        )
        self.assertEqual(
            'operator_prior_domain_predicate_validation',
            claim_experiment['experiment_kind'],
        )
        self.assertEqual('operator_prior_failure_context', claim_experiment['target_context'])
        self.assertEqual('standard', claim_experiment['failure_context'])
        self.assertEqual(
            'operator_prior_claim_domain_limited',
            claim_experiment['family_status'],
        )
        domain_plan = next(
            plan for plan in memory.planned_experiments(
                world_types=['standard', 'inverse_square_repulsion', 'localized_gravity'],
                object_counts=[5],
                steps=240,
                seed_start=0,
                limit=8,
            )
            if plan['experiment_kind'] == 'operator_prior_domain_predicate_validation'
        )
        self.assertEqual('standard', domain_plan['world_type'])
        self.assertEqual(prior['key'], domain_plan['operator_prior_key'])
        self.assertEqual(
            'operator_prior_domain_predicate_deferred_to_prior_feedback',
            memory.evaluate_planned_result(
                domain_plan,
                context='standard',
                seed=domain_plan['seed'],
                report={},
            )['outcome'],
        )
        failed_domain_outcome = memory.evaluate_planned_result(
            domain_plan,
            context='standard',
            seed=domain_plan['seed'],
            report={},
            operator_prior_records=[{
                'context': 'standard',
                'operator_key': prior['key'],
                'operator_kind': prior['operator_kind'],
                'outcome': 'confirmed',
                'best_score': 0.82,
                'matching_equation_count': 1,
                'parameters': prior['parameters'],
            }],
        )
        failed_memory = CumulativeTheoryMemory.from_dict(memory.to_dict())
        failed_memory.planned_outcomes.append(failed_domain_outcome)
        repeated_predicates = [
            item for item in failed_memory.operator_prior_claim_experiments(limit=8)
            if item['operator_prior_key'] == prior['key']
            and item['experiment_kind'] == 'operator_prior_domain_predicate_validation'
        ]
        legacy_failed_memory = CumulativeTheoryMemory.from_dict(memory.to_dict())
        legacy_failed_memory.planned_outcomes.append({
            'theory_kind': f"operator_prior:{prior['operator_kind']}",
            'experiment_kind': 'operator_prior_domain_predicate_validation',
            'outcome': 'operator_prior_domain_predicate_failed',
            'context': 'standard',
            'seed': domain_plan['seed'],
        })
        legacy_repeated_predicates = [
            item for item in legacy_failed_memory.operator_prior_claim_experiments(limit=8)
            if item['operator_prior_key'] == prior['key']
            and item['experiment_kind'] == 'operator_prior_domain_predicate_validation'
        ]
        domain_outcome = memory.evaluate_planned_result(
            domain_plan,
            context='standard',
            seed=domain_plan['seed'],
            report={},
            operator_prior_records=[{
                'context': 'standard',
                'operator_key': prior['key'],
                'operator_kind': prior['operator_kind'],
                'outcome': 'unmatched',
                'best_score': 0.0,
                'matching_equation_count': 0,
                'parameters': prior['parameters'],
            }],
        )
        self.assertEqual(
            'operator_prior_domain_predicate_failed',
            failed_domain_outcome['outcome'],
        )
        self.assertEqual([], repeated_predicates)
        self.assertEqual([], legacy_repeated_predicates)
        self.assertEqual(
            'operator_prior_domain_predicate_confirmed',
            domain_outcome['outcome'],
        )
        self.assertEqual('unmatched', domain_outcome['operator_prior_feedback_outcome'])
        repairs = memory.operator_prior_repair_experiments(limit=3)
        self.assertTrue(repairs)
        repair = next(
            item for item in repairs
            if item['operator_prior_key'] == prior['key']
        )
        self.assertEqual('operator_prior_domain_repair', repair['experiment_kind'])
        self.assertEqual('operator_prior_failure_context', repair['target_context'])
        self.assertEqual('standard', repair['failure_context'])
        self.assertEqual(
            'planned_operator_prior_repair',
            repair['probe_action']['source'],
        )
        self.assertIn('operator_prior_repair_experiments', memory.to_dict())
        self.assertIn('Operator prior repair:', memory.summary())
        validations = memory.operator_prior_validation_experiments(limit=3)
        self.assertTrue(validations)
        validation = next(
            item for item in validations
            if item['operator_prior_key'] == prior['key']
        )
        self.assertEqual(
            'operator_prior_refinement_validation',
            validation['experiment_kind'],
        )
        self.assertEqual(
            2.25,
            validation['operator_prior_parameters']['distance_exponent'],
        )
        self.assertIn(
            'standard',
            validation['avoid_contexts'],
        )
        packed_with_validation = memory.to_dict()
        self.assertIn('operator_prior_validation_experiments', packed_with_validation)
        self.assertIn('Operator prior validation:', memory.summary())
        plans = memory.planned_experiments(
            world_types=['standard', 'inverse_square_repulsion', 'localized_gravity'],
            object_counts=[5],
            steps=240,
            seed_start=0,
            limit=8,
        )
        repair_plan = next(
            plan for plan in plans
            if plan['experiment_kind'] == 'operator_prior_domain_repair'
        )
        self.assertEqual('standard', repair_plan['world_type'])
        self.assertEqual(prior['key'], repair_plan['operator_prior_key'])
        self.assertEqual(
            'planned_operator_prior_repair',
            repair_plan['probe_action']['source'],
        )
        self.assertEqual(
            'operator_prior_repair_deferred_to_prior_feedback',
            memory.evaluate_planned_result(
                repair_plan,
                context='standard',
                seed=repair_plan['seed'],
                report={},
            )['outcome'],
        )
        repair_outcome = memory.evaluate_planned_result(
            repair_plan,
            context='standard',
            seed=repair_plan['seed'],
            report={},
            operator_prior_records=[{
                'operator_key': prior['key'],
                'operator_kind': prior['operator_kind'],
                'outcome': 'confirmed',
                'best_score': 0.76,
                'matching_equation_count': 1,
                'parameters': prior['parameters'],
                'best_equation': {
                    'role': 'generated_operator_distance_scaled_direction_equation',
                    'score': 0.76,
                    'parameters': {
                        **prior['parameters'],
                        'distance_exponent': 2.1,
                    },
                },
                'refined_parameters': {
                    **prior['parameters'],
                    'distance_exponent': 2.1,
                },
            }],
        )
        self.assertEqual('operator_prior_repair_confirmed', repair_outcome['outcome'])
        self.assertTrue(repair_outcome['operator_prior_evaluated'])
        self.assertTrue(repair_outcome['operator_prior_refinement_detected'])
        self.assertEqual('confirmed', repair_outcome['operator_prior_feedback_outcome'])
        validation_plan = next(
            plan for plan in plans
            if plan['experiment_kind'] == 'operator_prior_refinement_validation'
        )
        self.assertEqual('localized_gravity', validation_plan['world_type'])
        self.assertEqual(prior['key'], validation_plan['operator_prior_key'])
        self.assertEqual(
            2.25,
            validation_plan['operator_prior_parameters']['distance_exponent'],
        )
        validation_outcome = memory.evaluate_planned_result(
            validation_plan,
            context='localized_gravity',
            seed=validation_plan['seed'],
            report={},
            operator_prior_records=[],
        )
        self.assertEqual(
            'operator_prior_validation_failed',
            validation_outcome['outcome'],
        )
        self.assertTrue(validation_outcome['operator_prior_evaluated'])

    def test_generated_operator_priors_apply_signature_budget(self):
        memory = CumulativeTheoryMemory()
        contexts = [
            'repulsion',
            'central_force',
            'localized_gravity',
            'inverse_square_repulsion',
        ]
        for seed, context in enumerate(contexts):
            memory.disagreement_records.append({
                'context': context,
                'seed': seed,
                'mode': 'distance_exponent_race',
                'family_kinds': [
                    'distance_scaled_direction_residual',
                    'generated_distance_scaled_direction_residual',
                ],
                'rival_labels': [
                    'distance exponent candidate',
                    'generated distance exponent candidate',
                ],
                'mean_rival_score': 0.82,
                'probe_points': [
                    {'x': 12.0, 'y': 10.0, 'distance_from_center': 2.0},
                    {'x': 14.0, 'y': 10.0, 'distance_from_center': 4.0},
                ],
                'rival_predictions': [
                    {'prediction': 'candidate / separation^-0.5'},
                    {'prediction': 'candidate / separation^-1.0'},
                    {'prediction': 'candidate / separation^-1.5'},
                    {'prediction': 'candidate / separation^-2.0'},
                ],
            })

        priors = memory.generated_operator_priors(limit=8)
        budget = memory.operator_prior_budget_report(limit=8)

        self.assertEqual(8, len(priors))
        self.assertGreaterEqual(budget['candidate_count'], 16)
        self.assertEqual(8, budget['selected_count'])
        self.assertTrue(budget['signature_counts'])
        self.assertLessEqual(max(budget['signature_counts'].values()), 2)
        self.assertTrue(all(
            prior['operator_kind'] == 'inverse_separation_power'
            for prior in priors
        ))

    def test_equation_case_records_consolidate_robust_operator_invariants(self):
        memory = CumulativeTheoryMemory()
        cases = [
            (0, 'k * unit_anchor_vector / separation^3', 0.76),
            (1, 'k * unit_anchor_vector / separation^2', 0.63),
            (3, 'k * unit_anchor_vector / separation^3', 0.77),
            (
                2,
                'k * taper(separation, 7_053) * '
                'unit_generated_center_vector / separation^0_5',
                0.89,
            ),
        ]
        for seed, expression, score in cases:
            memory.record_equation_case_result(
                'repulsion',
                seed,
                {
                    'interesting_equation': {
                        'target': 'baseline_adjusted_delta_velocity',
                        'expression': expression,
                        'role': 'generated_operator_distance_scaled_direction_equation',
                        'score': score,
                    },
                    'passed': True,
                    'label_leaks': [],
                },
                phase='math_final_discovery',
            )

        invariants = memory.operator_prior_invariant_consolidations(limit=3)
        top = invariants[0]
        exponents = {
            item['value']: item['support']
            for item in top['parameter_candidates']
        }
        packed = memory.to_dict()
        restored = CumulativeTheoryMemory.from_dict(packed)

        self.assertEqual('repulsion', top['context'])
        self.assertEqual('inverse_separation_power', top['law_family'])
        self.assertEqual('unit_anchor_vector', top['vector_basis'])
        self.assertEqual('robust_family_parameter_unresolved', top['status'])
        self.assertEqual(3, top['support_count'])
        self.assertEqual([0, 1, 3], top['support_seeds'])
        self.assertEqual(2, exponents[3.0])
        self.assertEqual(1, exponents[2.0])
        self.assertIn('distance exponent', top['next_experiment'])
        self.assertIn('equation_case_records', packed)
        self.assertIn('operator_prior_invariant_consolidations', packed)
        self.assertEqual(
            top,
            restored.operator_prior_invariant_consolidations(limit=3)[0],
        )
        self.assertIn('Robust equation invariants:', memory.summary())

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            _print_cumulative_theory_review(memory)
        printed = output.getvalue()

        self.assertIn('Theory robust equation invariants:', printed)
        self.assertIn('robust_family_parameter_unresolved', printed)

    def test_invariant_exponent_resolution_designs_near_mid_far_probe(self):
        memory = CumulativeTheoryMemory()
        cases = [
            (0, 'k * unit_anchor_vector / separation^3', 0.76, 3.0),
            (1, 'k * unit_anchor_vector / separation^2', 0.63, 2.0),
            (3, 'k * unit_anchor_vector / separation^3', 0.77, 3.0),
        ]
        for seed, expression, score, exponent in cases:
            memory.record_equation_case_result(
                'repulsion',
                seed,
                {
                    'interesting_equation': {
                        'target': 'baseline_adjusted_delta_velocity',
                        'expression': expression,
                        'role': 'generated_operator_distance_scaled_direction_equation',
                        'score': score,
                        'parameters': {
                            'center_x': 9.0,
                            'center_y': 11.0,
                            'distance_exponent': exponent,
                        },
                    },
                    'passed': True,
                    'label_leaks': [],
                },
                phase='math_final_discovery',
            )

        experiment = memory.equation_invariant_resolution_experiments(limit=1)[0]
        signature = experiment['disagreement_signature']
        plan = memory.planned_experiments(
            world_types=['standard', 'repulsion'],
            object_counts=[5],
            steps=240,
            limit=1,
        )[0]
        actions = _planned_probe_actions(plan)
        selected = memory.evaluate_planned_result(
            plan,
            context='repulsion',
            seed=plan['seed'],
            report={
                'theories': [{
                    'theory_kind': 'distance_scaled_direction_residual',
                    'parameters': {'distance_exponent': 3.0},
                    'score': 0.81,
                }],
                'proof_checks': [],
            },
        )

        self.assertEqual(
            'equation_invariant_exponent_resolution',
            experiment['experiment_kind'],
        )
        self.assertEqual('repulsion', plan['world_type'])
        self.assertEqual('distance_exponent_race', signature['mode'])
        self.assertEqual([3.0, 2.0], signature['candidate_exponents'])
        self.assertEqual(
            [
                'very_near_exponent_ratio',
                'near_exponent_ratio',
                'mid_log_slope',
                'far_log_slope',
                'very_far_exponent_ratio',
            ],
            [point['label'] for point in signature['probe_points']],
        )
        self.assertEqual(
            [
                'very_near_exponent_ratio',
                'near_exponent_ratio',
                'mid_log_slope',
                'far_log_slope',
            ],
            [action['probe_label'] for action in actions],
        )
        self.assertEqual('five_point_log_slope_ladder', signature['probe_strategy'])
        self.assertGreaterEqual(len(signature['log_slope_probe_pairs']), 4)
        self.assertTrue(all(
            action['source'] == 'planned_equation_invariant_resolution'
            for action in actions
        ))
        self.assertEqual(plan['invariant_key'], actions[0]['invariant_key'])
        self.assertIn('equation_invariant_resolution_experiments', memory.to_dict())
        self.assertIn('Equation invariant resolution:', memory.summary())
        self.assertEqual('invariant_exponent_selected', selected['outcome'])

        memory.planned_outcomes.append({
            'experiment_kind': 'equation_invariant_exponent_resolution',
            'invariant_key': plan['invariant_key'],
            'outcome': 'invariant_resolution_still_unresolved',
        })
        refined = memory.equation_invariant_resolution_experiments(limit=1)[0]
        self.assertEqual(
            ['very_near_center', 'mid_log_check', 'very_far_from_center'],
            [
                point['label']
                for point in refined['disagreement_signature']['probe_points']
            ],
        )

        memory.planned_outcomes.append({
            'experiment_kind': 'equation_invariant_exponent_resolution',
            'invariant_key': plan['invariant_key'],
            'outcome': 'invariant_exponent_selected',
        })
        self.assertEqual([], memory.equation_invariant_resolution_experiments(limit=1))

    def test_tapered_invariant_exponent_resolution_matches_exponent_label(self):
        memory = CumulativeTheoryMemory()
        plan = {
            'theory_kind': 'tapered_distance_direction_residual',
            'experiment_kind': 'equation_invariant_exponent_resolution',
            'primary_theory_label': (
                'tapered_distance_direction_residual/separation^-0.5'
            ),
            'rival_theory_labels': [
                'tapered_distance_direction_residual/separation^-1.5',
            ],
            'expected_result': 'one tapered distance exponent should win',
            'falsifies_if': 'near/far ratio rejects the exponent',
        }

        outcome = memory.evaluate_planned_result(
            plan,
            context='repulsion',
            seed=5,
            report={
                'theories': [{
                    'theory_kind': 'tapered_distance_direction_residual',
                    'parameters': {'distance_exponent': 0.5},
                    'score': 0.89,
                }],
                'proof_checks': [],
            },
        )

        self.assertEqual('invariant_exponent_selected', outcome['outcome'])
        self.assertEqual(1, outcome['matching_theory_count'])
        self.assertEqual(0, outcome['rival_theory_count'])

    def test_selected_invariant_exponent_consolidates_and_replays_law(self):
        memory = CumulativeTheoryMemory()
        cases = [
            (0, 'k * unit_anchor_vector / separation^3', 0.76, 3.0),
            (1, 'k * unit_anchor_vector / separation^2', 0.63, 2.0),
            (3, 'k * unit_anchor_vector / separation^3', 0.77, 3.0),
        ]
        for seed, expression, score, exponent in cases:
            memory.record_equation_case_result(
                'repulsion',
                seed,
                {
                    'interesting_equation': {
                        'target': 'baseline_adjusted_delta_velocity',
                        'expression': expression,
                        'role': 'generated_operator_distance_scaled_direction_equation',
                        'score': score,
                        'parameters': {'distance_exponent': exponent},
                    },
                    'passed': True,
                    'label_leaks': [],
                },
                phase='math_final_discovery',
            )

        resolution = memory.equation_invariant_resolution_experiments(limit=1)[0]
        selected = memory.evaluate_planned_result(
            resolution,
            context='repulsion',
            seed=5,
            report={
                'theories': [{
                    'theory_kind': 'distance_scaled_direction_residual',
                    'parameters': {'distance_exponent': 3.0},
                    'score': 0.82,
                }],
                'proof_checks': [],
            },
        )
        memory.planned_outcomes.append(selected)
        invariant = memory.operator_prior_invariant_consolidations(limit=1)[0]
        replay = memory.selected_law_replay_agenda(limit=1)[0]
        replay_outcome = memory.evaluate_planned_result(
            replay,
            context='repulsion',
            seed=6,
            report={
                'theories': [{
                    'theory_kind': 'distance_scaled_direction_residual',
                    'parameters': {'distance_exponent': 3.0},
                    'score': 0.84,
                }],
                'proof_checks': [],
            },
        )

        self.assertEqual('robust_law_parameter_selected', invariant['status'])
        self.assertEqual(3.0, invariant['selected_distance_exponent'])
        self.assertTrue(any(
            candidate.get('selected') and candidate['value'] == 3.0
            for candidate in invariant['parameter_candidates']
        ))
        self.assertIn('replay the selected exponent', invariant['next_experiment'])
        self.assertEqual([], memory.equation_invariant_resolution_experiments(limit=1))
        self.assertEqual('selected_law_replay', replay['experiment_kind'])
        self.assertEqual(
            'distance_scaled_direction_residual/separation^-3.0',
            replay['primary_theory_label'],
        )
        self.assertIn(
            'distance_scaled_direction_residual/separation^-2.0',
            replay['rival_theory_labels'],
        )
        self.assertEqual(
            'planned_selected_law_replay',
            _planned_probe_actions(replay)[0]['source'],
        )
        self.assertEqual(
            'selected_law_replay_confirmed',
            replay_outcome['outcome'],
        )
        self.assertEqual(
            replay['selected_law_replay_key'],
            replay_outcome['selected_law_replay_key'],
        )
        self.assertIn('selected_law_replay_agenda', memory.to_dict())
        self.assertIn('Selected-law replay agenda:', memory.summary())
        memory.planned_outcomes.append(replay_outcome)
        self.assertEqual(
            'heldout_confirmed',
            memory.theorem_memory(limit=1)[0]['status'],
        )

    def _build_selected_tapered_law_memory(self) -> CumulativeTheoryMemory:
        memory = CumulativeTheoryMemory()
        for seed, exponent in [(0, 0.5), (1, 1.5), (2, 0.5)]:
            memory.record_equation_case_result(
                'localized_gravity',
                seed,
                {
                    'interesting_equation': {
                        'target': 'baseline_adjusted_delta_velocity',
                        'expression': (
                            'k * taper(separation, 9) * '
                            'unit_generated_center_vector / '
                            f"separation^{str(exponent).replace('.', '_')}"
                        ),
                        'role': 'generated_operator_distance_scaled_direction_equation',
                        'score': 0.74,
                        'parameters': {
                            'cutoff_radius': 9.0,
                            'distance_exponent': exponent,
                        },
                    },
                    'passed': True,
                    'label_leaks': [],
                },
                phase='math_final_discovery',
            )
        resolution = memory.equation_invariant_resolution_experiments(limit=1)[0]
        selected = memory.evaluate_planned_result(
            resolution,
            context='localized_gravity',
            seed=5,
            report={
                'theories': [{
                    'theory_kind': 'tapered_distance_direction_residual',
                    'parameters': {
                        'distance_exponent': 0.5,
                        'cutoff_radius': 9.0,
                    },
                    'score': 0.79,
                }],
                'proof_checks': [],
            },
        )
        memory.planned_outcomes.append(selected)
        return memory

    def test_domain_design_theorem_and_blind_holdout_memory_are_connected(self):
        memory = CumulativeTheoryMemory()
        for seed, exponent in [(0, 0.5), (1, 1.5), (2, 0.5)]:
            memory.record_equation_case_result(
                'localized_gravity',
                seed,
                {
                    'interesting_equation': {
                        'target': 'baseline_adjusted_delta_velocity',
                        'expression': (
                            'k * taper(separation, 9) * '
                            'unit_generated_center_vector / '
                            f"separation^{str(exponent).replace('.', '_')}"
                        ),
                        'role': 'generated_operator_distance_scaled_direction_equation',
                        'score': 0.74,
                        'parameters': {
                            'cutoff_radius': 9.0,
                            'distance_exponent': exponent,
                        },
                    },
                    'passed': True,
                    'label_leaks': [],
                },
                phase='math_final_discovery',
            )
        resolution = memory.equation_invariant_resolution_experiments(limit=1)[0]
        selected = memory.evaluate_planned_result(
            resolution,
            context='localized_gravity',
            seed=5,
            report={
                'theories': [{
                    'theory_kind': 'tapered_distance_direction_residual',
                    'parameters': {'distance_exponent': 0.5},
                    'score': 0.79,
                }],
                'proof_checks': [],
            },
        )
        memory.planned_outcomes.append(selected)

        domain_plans = memory.domain_rediscovery_experiments(limit=20)
        designs = memory.autonomous_experiment_design_agenda(limit=20)
        theorems = memory.theorem_memory(limit=5)
        blind = memory.blind_holdout_benchmark_report(limit=8)
        packed = memory.to_dict()
        printed_output = io.StringIO()
        with contextlib.redirect_stdout(printed_output):
            _print_cumulative_theory_review(memory)
        printed = printed_output.getvalue()

        self.assertTrue({
            'algebra_equations',
            'logic_proof',
            'higher_dimensions',
        } <= {item['domain_key'] for item in domain_plans})
        self.assertTrue(all(
            item['experiment_kind'] == 'domain_world_rediscovery_probe'
            and item['leak_constraints']['withhold_manifest']
            and not item['leak_constraints']['leaks_benchmark_truth']
            for item in domain_plans
        ))
        self.assertIn(
            'selected_law_replay',
            {item['source'] for item in designs},
        )
        self.assertIn(
            'domain_rediscovery',
            {item['source'] for item in designs},
        )
        selected_theorem = next(
            item for item in theorems
            if item['theorem_kind'] == 'selected_equation_invariant'
        )
        self.assertEqual(
            {'distance_exponent': 0.5},
            selected_theorem['selected_parameters'],
        )
        self.assertIn(
            'blind_holdout_counterexample',
            selected_theorem['proof_obligations'],
        )
        self.assertTrue(blind['ready_for_blind_run'])
        self.assertEqual(0, blind['leak_blocker_count'])
        self.assertIn(
            'selected_law_blind_holdout',
            blind['benchmark_kinds'],
        )
        self.assertIn('domain_world_withheld_manifest', blind['benchmark_kinds'])
        self.assertIn('domain_rediscovery_experiments', packed)
        self.assertIn('autonomous_experiment_design_agenda', packed)
        self.assertIn('theorem_memory', packed)
        self.assertIn('blind_holdout_benchmark', packed)
        self.assertIn('Theorem memory:', memory.summary())
        self.assertIn('Autonomous experiment designs:', memory.summary())
        self.assertIn('Blind holdout benchmark:', memory.summary())
        self.assertIn('Theory theorem memory:', printed)
        self.assertIn('Theory blind holdout benchmark:', printed)

        replay = memory.selected_law_replay_agenda(limit=1)[0]
        conflicted = memory.evaluate_planned_result(
            replay,
            context='localized_gravity',
            seed=6,
            report={
                'theories': [{
                    'theory_kind': 'tapered_distance_direction_residual',
                    'parameters': {'distance_exponent': 1.5},
                    'score': 0.82,
                }],
                'proof_checks': [],
            },
        )
        memory.planned_outcomes.append(conflicted)
        self.assertEqual('selected_law_replay_conflicted', conflicted['outcome'])
        self.assertEqual(
            'holdout_conflicted',
            memory.theorem_memory(limit=1)[0]['status'],
        )

    def test_selected_law_conflict_designs_multi_parameter_race_and_domain_split(self):
        memory = self._build_selected_tapered_law_memory()
        replay = memory.selected_law_replay_agenda(limit=1)[0]
        conflicted = memory.evaluate_planned_result(
            replay,
            context='localized_gravity',
            seed=6,
            report={
                'theories': [{
                    'theory_kind': 'tapered_distance_direction_residual',
                    'parameters': {
                        'distance_exponent': 1.5,
                        'cutoff_radius': 9.0,
                    },
                    'score': 0.82,
                }],
                'proof_checks': [],
            },
        )
        memory.record_equation_case_result(
            'localized_gravity',
            6,
            {
                'interesting_equation': {
                    'target': 'baseline_adjusted_delta_velocity',
                    'expression': (
                        'k * taper(separation, 9) * '
                        'unit_generated_center_vector / separation^1_5'
                    ),
                    'role': 'generated_operator_distance_scaled_direction_equation',
                    'score': 0.82,
                    'parameters': {
                        'cutoff_radius': 9.0,
                        'distance_exponent': 1.5,
                    },
                },
                'passed': True,
                'label_leaks': [],
            },
            phase='equation_followup',
        )
        memory.planned_outcomes.append(conflicted)

        conflict = memory.selected_law_conflict_experiments(limit=1)[0]
        domain_split = memory.law_domain_split_hypotheses(limit=1)[0]
        plan = memory.planned_experiments(
            world_types=['localized_gravity'],
            object_counts=[5],
            steps=240,
            limit=1,
        )[0]
        actions = _planned_probe_actions(plan)
        outcome = memory.evaluate_planned_result(
            plan,
            context='localized_gravity',
            seed=plan['seed'],
            report={
                'theories': [{
                    'theory_kind': 'tapered_distance_direction_residual',
                    'parameters': {
                        'distance_exponent': 1.5,
                        'cutoff_radius': 9.0,
                    },
                    'score': 0.84,
                }],
                'proof_checks': [],
            },
        )
        packed = memory.to_dict()
        printed_output = io.StringIO()
        with contextlib.redirect_stdout(printed_output):
            _print_cumulative_theory_review(memory)
        printed = printed_output.getvalue()

        self.assertEqual(
            'selected_law_replay_conflicted',
            conflicted['outcome'],
        )
        self.assertEqual(
            'selected_law_conflict_resolution',
            conflict['experiment_kind'],
        )
        self.assertEqual(
            'multi_parameter_law_race',
            conflict['disagreement_signature']['mode'],
        )
        self.assertEqual('split_suspected', domain_split['status'])
        self.assertEqual(
            'selected_law_conflict_resolution',
            plan['experiment_kind'],
        )
        self.assertTrue(plan['selected_multi_parameter_variant'])
        self.assertTrue(plan['rival_multi_parameter_variants'])
        self.assertEqual(
            'planned_selected_law_conflict_resolution',
            actions[0]['source'],
        )
        self.assertEqual('conflict_rival_supported', outcome['outcome'])
        self.assertEqual(1, outcome['rival_theory_count'])
        self.assertIn('selected_law_conflict_experiments', packed)
        self.assertIn('law_domain_split_hypotheses', packed)
        self.assertIn('Selected-law conflict resolution:', memory.summary())
        self.assertIn('Law domain split hypotheses:', memory.summary())
        self.assertIn('Theory selected-law conflict resolution:', printed)
        self.assertIn('Theory law domain split hypotheses:', printed)

    def test_settled_conflict_surfaces_domain_predicate_learning(self):
        memory = self._build_selected_tapered_law_memory()
        replay = memory.selected_law_replay_agenda(limit=1)[0]
        conflicted = memory.evaluate_planned_result(
            replay,
            context='localized_gravity',
            seed=6,
            report={
                'theories': [{
                    'theory_kind': 'tapered_distance_direction_residual',
                    'parameters': {
                        'distance_exponent': 1.5,
                        'cutoff_radius': 9.0,
                    },
                    'score': 0.82,
                }],
                'proof_checks': [],
            },
        )
        memory.record_equation_case_result(
            'localized_gravity',
            6,
            {
                'interesting_equation': {
                    'target': 'baseline_adjusted_delta_velocity',
                    'expression': (
                        'k * taper(separation, 9) * '
                        'unit_generated_center_vector / separation^1_5'
                    ),
                    'role': 'generated_operator_distance_scaled_direction_equation',
                    'score': 0.82,
                    'parameters': {
                        'cutoff_radius': 9.0,
                        'distance_exponent': 1.5,
                    },
                },
                'passed': True,
                'label_leaks': [],
            },
            phase='equation_followup',
        )
        memory.planned_outcomes.append(conflicted)
        conflict_plan = memory.selected_law_conflict_experiments(limit=1)[0]
        conflict_plan = {
            **conflict_plan,
            'world_type': 'localized_gravity',
            'seed': 7,
            'object_count': 5,
            'steps': conflict_plan.get('quick_steps', 170),
            'hidden_holdout': False,
        }
        conflict_outcome = memory.evaluate_planned_result(
            conflict_plan,
            context='localized_gravity',
            seed=conflict_plan['seed'],
            report={
                'theories': [{
                    'theory_kind': 'tapered_distance_direction_residual',
                    'parameters': {
                        'distance_exponent': 1.5,
                        'cutoff_radius': 9.0,
                    },
                    'score': 0.84,
                }],
                'proof_checks': [],
            },
        )
        memory.planned_outcomes.append(conflict_outcome)

        self.assertEqual([], memory.selected_law_conflict_experiments(limit=1))
        split = memory.law_domain_split_hypotheses(limit=1)[0]
        agenda = memory.domain_predicate_learning_agenda(limit=1)[0]
        plan = dict(agenda)
        plan.update({
            'world_type': agenda['source_context'],
            'seed': 8,
            'object_count': 5,
            'steps': agenda['quick_steps'],
            'hidden_holdout': False,
        })
        outcome = memory.evaluate_planned_result(
            plan,
            context='localized_gravity',
            seed=8,
            report={
                'theories': [
                    {
                        'theory_kind': 'tapered_distance_direction_residual',
                        'parameters': {
                            'distance_exponent': 0.5,
                            'cutoff_radius': 9.0,
                        },
                        'score': 0.79,
                    },
                    {
                        'theory_kind': 'tapered_distance_direction_residual',
                        'parameters': {
                            'distance_exponent': 1.5,
                            'cutoff_radius': 9.0,
                        },
                        'score': 0.84,
                    },
                ],
                'proof_checks': [],
            },
        )

        self.assertEqual('split_suspected', split['status'])
        self.assertEqual('domain_predicate_discovery', agenda['experiment_kind'])
        self.assertEqual(
            'planned_domain_predicate_discovery',
            _planned_probe_actions(agenda)[0]['source'],
        )
        self.assertTrue(agenda['candidate_predicate'])
        self.assertEqual(
            'domain_predicate_split_supported',
            outcome['outcome'],
        )
        self.assertIn('domain_predicate_learning_agenda', memory.to_dict())
        self.assertIn('Domain predicate learning agenda:', memory.summary())

    def test_theorem_consolidation_marks_conflicted_selected_law_as_piecewise(self):
        memory = self._build_selected_tapered_law_memory()
        replay = memory.selected_law_replay_agenda(limit=1)[0]
        conflicted = memory.evaluate_planned_result(
            replay,
            context='localized_gravity',
            seed=6,
            report={
                'theories': [{
                    'theory_kind': 'tapered_distance_direction_residual',
                    'parameters': {
                        'distance_exponent': 1.5,
                        'cutoff_radius': 9.0,
                    },
                    'score': 0.82,
                }],
                'proof_checks': [],
            },
        )
        memory.planned_outcomes.append(conflicted)

        consolidation = memory.theorem_consolidations(limit=1)[0]

        self.assertEqual(
            'domain_limited_or_piecewise',
            consolidation['status'],
        )
        self.assertEqual('holdout_conflicted', consolidation['theorem_status'])
        self.assertIn('approximate_variants', consolidation)
        self.assertIn('theorem_consolidations', memory.to_dict())
        self.assertIn('Theorem consolidations:', memory.summary())

    def test_blind_holdout_validation_executes_selected_law_and_rivals(self):
        memory = self._build_selected_tapered_law_memory()
        validation = memory.blind_holdout_validation_experiments(limit=1)[0]
        plans = memory.planned_experiments(
            world_types=['localized_gravity'],
            object_counts=[5],
            steps=240,
            limit=4,
        )
        plan = next(
            item for item in plans
            if item['experiment_kind'] == 'blind_holdout_validation'
        )
        actions = _planned_probe_actions(plan)
        confirmed = memory.evaluate_planned_result(
            plan,
            context='hidden:blind',
            seed=plan['seed'],
            report={
                'theories': [{
                    'theory_kind': 'tapered_distance_direction_residual',
                    'parameters': {
                        'distance_exponent': 0.5,
                        'cutoff_radius': 9.0,
                    },
                    'score': 0.81,
                }],
                'proof_checks': [],
            },
        )
        conflicted = memory.evaluate_planned_result(
            plan,
            context='hidden:blind',
            seed=plan['seed'],
            report={
                'theories': [{
                    'theory_kind': 'tapered_distance_direction_residual',
                    'parameters': {
                        'distance_exponent': 1.5,
                        'cutoff_radius': 9.0,
                    },
                    'score': 0.82,
                }],
                'proof_checks': [],
            },
        )

        self.assertEqual(
            'blind_holdout_validation',
            validation['experiment_kind'],
        )
        self.assertEqual(
            'blind_selected_law_holdout',
            validation['disagreement_signature']['mode'],
        )
        self.assertTrue(validation['rival_multi_parameter_variants'])
        self.assertEqual('hidden_procedural', plan['world_type'])
        self.assertTrue(plan['hidden_holdout'])
        self.assertEqual(
            'planned_blind_holdout_validation',
            actions[0]['source'],
        )
        self.assertEqual('blind_holdout_confirmed', confirmed['outcome'])
        self.assertEqual('blind_holdout_conflicted', conflicted['outcome'])
        self.assertIn(
            'blind_holdout_validation_experiments',
            memory.to_dict(),
        )
        self.assertIn(
            'Blind holdout validation agenda:',
            memory.summary(),
        )

    def test_blind_holdout_validation_retries_absent_holdout_in_source_context(self):
        memory = self._build_selected_tapered_law_memory()
        validation = memory.blind_holdout_validation_experiments(limit=1)[0]
        absent = memory.evaluate_planned_result(
            validation,
            context='hidden_00_0000',
            seed=0,
            report={
                'theories': [],
                'proof_checks': [],
            },
        )
        memory.planned_outcomes.append(absent)
        memory.record_result('hidden_00_0000', 0, {'theories': []})
        loop = AutonomousDiscoveryLoop()
        for seed in range(2):
            memory.record_result(
                'inverse_square_repulsion',
                20 + seed,
                loop.build_report([
                    equation(
                        key=f'raw_eq:residual_inferred_direction_distance_2_{seed}',
                        role='residual_distance_scaled_direction_equation',
                        expression='k * unit_inferred_vector / separation^2',
                        parameters={
                            'center_x': 8.0 + seed,
                            'center_y': 12.0,
                            'distance_exponent': 2.0,
                            'distance_mse_improvement': 0.42,
                        },
                    )
                ], step=170 + seed),
            )

        retry = memory.blind_holdout_validation_experiments(limit=1)[0]
        prioritized = memory.next_experiments(limit=3)[0]
        plans = memory.planned_experiments(
            world_types=['hidden_procedural', 'localized_gravity'],
            object_counts=[5],
            steps=240,
            limit=6,
        )
        retry_plan = next(
            plan for plan in plans
            if plan['experiment_kind'] == 'blind_holdout_validation'
        )
        confirmed = memory.evaluate_planned_result(
            retry,
            context='localized_gravity',
            seed=retry_plan['seed'],
            report={
                'theories': [{
                    'theory_kind': 'tapered_distance_direction_residual',
                    'parameters': {
                        'distance_exponent': 0.5,
                        'cutoff_radius': 9.0,
                    },
                    'score': 0.83,
                }],
                'proof_checks': [],
            },
        )
        memory.planned_outcomes.append(confirmed)

        self.assertEqual('blind_holdout_absent', absent['outcome'])
        self.assertEqual('blind_holdout_validation', retry['experiment_kind'])
        self.assertEqual(1, retry['proof_evidence']['attempt_count'])
        self.assertEqual(1, retry['proof_evidence']['absent_count'])
        self.assertTrue(retry['proof_evidence']['retry_in_source_context'])
        self.assertIn('fresh selected-source seed', retry['reason'])
        self.assertEqual('selected_law_holdout_context', retry['target_context'])
        self.assertEqual('blind_holdout_validation', prioritized['experiment_kind'])
        self.assertEqual('localized_gravity', retry_plan['world_type'])
        self.assertFalse(retry_plan['hidden_holdout'])
        self.assertEqual('blind_holdout_confirmed', confirmed['outcome'])
        self.assertEqual([], memory.blind_holdout_validation_experiments(limit=1))

    def test_blind_holdout_conflict_routes_to_selected_law_resolution(self):
        memory = self._build_selected_tapered_law_memory()
        plan = next(
            item for item in memory.planned_experiments(
                world_types=['hidden_procedural', 'localized_gravity'],
                object_counts=[5],
                steps=240,
                limit=4,
            )
            if item['experiment_kind'] == 'blind_holdout_validation'
        )
        memory.planned_outcomes.append({
            'theory_kind': plan['theory_kind'],
            'experiment_kind': 'selected_law_conflict_resolution',
            'outcome': 'conflict_rival_supported',
            'context': 'localized_gravity',
            'seed': 99,
            'selected_law_replay_key': plan['selected_law_replay_key'],
        })
        conflicted = memory.evaluate_planned_result(
            plan,
            context='localized_gravity',
            seed=plan['seed'],
            report={
                'theories': [{
                    'theory_kind': 'tapered_distance_direction_residual',
                    'parameters': {
                        'distance_exponent': 1.5,
                        'cutoff_radius': 9.0,
                    },
                    'score': 0.82,
                }],
                'proof_checks': [],
            },
        )
        memory.planned_outcomes.append(conflicted)
        memory.record_equation_case_result(
            'localized_gravity',
            plan['seed'],
            {
                'interesting_equation': {
                    'target': 'baseline_adjusted_delta_velocity',
                    'expression': (
                        'k * taper(separation, 9) * '
                        'unit_generated_center_vector / separation^1_5'
                    ),
                    'role': 'generated_operator_distance_scaled_direction_equation',
                    'score': 0.82,
                    'parameters': {
                        'cutoff_radius': 9.0,
                        'distance_exponent': 1.5,
                    },
                },
                'passed': True,
                'label_leaks': [],
            },
            phase='equation_followup',
        )

        conflict = memory.selected_law_conflict_experiments(limit=1)[0]
        next_plan = memory.planned_experiments(
            world_types=['hidden_procedural', 'localized_gravity'],
            object_counts=[5],
            steps=240,
            limit=1,
        )[0]
        theorem = next(
            item for item in memory.theorem_memory(limit=4)
            if item['theorem_kind'] == 'selected_equation_invariant'
        )

        self.assertEqual('blind_holdout_conflicted', conflicted['outcome'])
        self.assertEqual([], memory.blind_holdout_validation_experiments(limit=1))
        self.assertEqual(
            'selected_law_conflict_resolution',
            conflict['experiment_kind'],
        )
        self.assertEqual(1, conflict['proof_evidence']['blind_conflicted_count'])
        self.assertEqual(1, conflict['proof_evidence']['attempt_count'])
        self.assertEqual(
            'selected_law_conflict_resolution',
            next_plan['experiment_kind'],
        )
        self.assertEqual('holdout_conflicted', theorem['status'])

    def test_rival_supported_conflict_reselects_only_same_family_parameter(self):
        memory = self._build_selected_tapered_law_memory()
        invariant = memory.operator_prior_invariant_consolidations(limit=4)[0]
        replay_key = memory._selected_law_replay_key(invariant)
        memory.planned_outcomes.append({
            'theory_kind': 'tapered_distance_direction_residual',
            'experiment_kind': 'selected_law_conflict_resolution',
            'outcome': 'conflict_rival_supported',
            'context': 'localized_gravity',
            'seed': 8,
            'invariant_key': invariant['key'],
            'selected_law_replay_key': replay_key,
            'target_scope': (
                'localized_gravity/'
                'tapered_distance_direction_residual/separation^-0.5'
            ),
            'rival_scope': (
                'localized_gravity/'
                'tapered_distance_direction_residual/separation^-1.5/'
                'tapered/cutoff~9/unit_generated_center_vector'
            ),
            'primary_theory_label': (
                'tapered_distance_direction_residual/separation^-0.5'
            ),
            'rival_theory_labels': [
                'tapered_distance_direction_residual/separation^-1.5/'
                'tapered/cutoff~9/unit_generated_center_vector',
            ],
            'best_score': 0.12,
            'rival_best_score': 0.70,
        })

        reselected = next(
            item for item in memory.operator_prior_invariant_consolidations(limit=4)
            if item['key'] == invariant['key']
        )
        memory.planned_outcomes.append({
            'theory_kind': 'tapered_distance_direction_residual',
            'experiment_kind': 'selected_law_conflict_resolution',
            'outcome': 'conflict_rival_supported',
            'context': 'localized_gravity',
            'seed': 9,
            'invariant_key': invariant['key'],
            'selected_law_replay_key': replay_key,
            'target_scope': (
                'localized_gravity/'
                'tapered_distance_direction_residual/separation^-1.5'
            ),
            'rival_scope': (
                'localized_gravity/'
                'distance_scaled_direction_residual/separation^-2.0'
            ),
            'primary_theory_label': (
                'tapered_distance_direction_residual/separation^-1.5'
            ),
            'rival_theory_labels': [
                'distance_scaled_direction_residual/separation^-2.0',
            ],
            'best_score': 0.10,
            'rival_best_score': 0.99,
        })
        after_cross_family = next(
            item for item in memory.operator_prior_invariant_consolidations(limit=4)
            if item['key'] == invariant['key']
        )

        self.assertEqual(1.5, reselected['selected_distance_exponent'])
        self.assertEqual(
            'conflict_rival_supported',
            reselected['selected_resolution']['outcome'],
        )
        self.assertEqual(1.5, after_cross_family['selected_distance_exponent'])

    def test_rediscovery_goal_progress_penalizes_unresolved_selected_laws(self):
        memory = self._build_selected_tapered_law_memory()
        replay = memory.selected_law_replay_agenda(limit=1)[0]
        conflicted = memory.evaluate_planned_result(
            replay,
            context='localized_gravity',
            seed=6,
            report={
                'theories': [{
                    'theory_kind': 'tapered_distance_direction_residual',
                    'parameters': {
                        'distance_exponent': 1.5,
                        'cutoff_radius': 9.0,
                    },
                    'score': 0.82,
                }],
                'proof_checks': [],
            },
        )
        memory.record_equation_case_result(
            'localized_gravity',
            6,
            {
                'interesting_equation': {
                    'target': 'baseline_adjusted_delta_velocity',
                    'expression': (
                        'k * taper(separation, 9) * '
                        'unit_generated_center_vector / separation^1_5'
                    ),
                    'role': 'generated_operator_distance_scaled_direction_equation',
                    'score': 0.82,
                    'parameters': {
                        'cutoff_radius': 9.0,
                        'distance_exponent': 1.5,
                    },
                },
                'passed': True,
                'label_leaks': [],
            },
            phase='equation_followup',
        )
        memory.planned_outcomes.append(conflicted)

        progress = memory.rediscovery_goal_progress_report()

        self.assertLess(progress['progress_score'], 0.85)
        self.assertFalse(progress['target_reached'])
        self.assertIn('heldout_law_stability', progress['blockers'])
        self.assertIn('blind_holdout_validation', progress['blockers'])
        self.assertEqual(
            1,
            progress['evidence_summary']['selected_law_replay_conflicted_count'],
        )
        self.assertIn('rediscovery_goal_progress', memory.to_dict())

    def test_rediscovery_goal_progress_improves_after_replay_and_blind_holdout(self):
        memory = self._build_selected_tapered_law_memory()
        before = memory.rediscovery_goal_progress_report()
        replay = memory.selected_law_replay_agenda(limit=1)[0]
        replay_confirmed = memory.evaluate_planned_result(
            replay,
            context='localized_gravity',
            seed=6,
            report={
                'theories': [{
                    'theory_kind': 'tapered_distance_direction_residual',
                    'parameters': {
                        'distance_exponent': 0.5,
                        'cutoff_radius': 9.0,
                    },
                    'score': 0.84,
                }],
                'proof_checks': [],
            },
        )
        memory.planned_outcomes.append(replay_confirmed)
        validation = memory.blind_holdout_validation_experiments(limit=1)[0]
        blind_confirmed = memory.evaluate_planned_result(
            validation,
            context='hidden:blind',
            seed=validation.get('seed', 7),
            report={
                'theories': [{
                    'theory_kind': 'tapered_distance_direction_residual',
                    'parameters': {
                        'distance_exponent': 0.5,
                        'cutoff_radius': 9.0,
                    },
                    'score': 0.82,
                }],
                'proof_checks': [],
            },
        )
        memory.planned_outcomes.append(blind_confirmed)

        after = memory.rediscovery_goal_progress_report()

        self.assertGreater(after['progress_score'], before['progress_score'])
        self.assertGreater(
            after['gates']['heldout_law_stability']['score'],
            before['gates']['heldout_law_stability']['score'],
        )
        self.assertGreater(
            after['gates']['blind_holdout_validation']['score'],
            before['gates']['blind_holdout_validation']['score'],
        )
        self.assertEqual(
            1,
            after['evidence_summary']['blind_holdout_confirmed_count'],
        )

    def test_rediscovery_goal_progress_counts_domain_scoped_blind_conflicts(self):
        memory = self._build_selected_tapered_law_memory()
        replay = memory.selected_law_replay_agenda(limit=1)[0]
        replay_conflicted = memory.evaluate_planned_result(
            replay,
            context='localized_gravity',
            seed=6,
            report={
                'theories': [{
                    'theory_kind': 'tapered_distance_direction_residual',
                    'parameters': {
                        'distance_exponent': 1.5,
                        'cutoff_radius': 9.0,
                    },
                    'score': 0.82,
                }],
                'proof_checks': [],
            },
        )
        memory.record_equation_case_result(
            'localized_gravity',
            6,
            {
                'interesting_equation': {
                    'target': 'baseline_adjusted_delta_velocity',
                    'expression': (
                        'k * taper(separation, 9) * '
                        'unit_generated_center_vector / separation^1_5'
                    ),
                    'role': 'generated_operator_distance_scaled_direction_equation',
                    'score': 0.82,
                    'parameters': {
                        'cutoff_radius': 9.0,
                        'distance_exponent': 1.5,
                    },
                },
                'passed': True,
                'label_leaks': [],
            },
            phase='equation_followup',
        )
        memory.planned_outcomes.append(replay_conflicted)
        validation = memory.blind_holdout_validation_experiments(limit=1)[0]
        blind_conflicted = memory.evaluate_planned_result(
            validation,
            context='hidden:blind',
            seed=7,
            report={
                'theories': [{
                    'theory_kind': 'tapered_distance_direction_residual',
                    'parameters': {
                        'distance_exponent': 1.5,
                        'cutoff_radius': 9.0,
                    },
                    'score': 0.83,
                }],
                'proof_checks': [],
            },
        )
        memory.planned_outcomes.append(blind_conflicted)
        conflict_plan = memory.selected_law_conflict_experiments(limit=1)[0]
        conflict_outcome = memory.evaluate_planned_result(
            conflict_plan,
            context='localized_gravity',
            seed=8,
            report={
                'theories': [
                    {
                        'theory_kind': 'tapered_distance_direction_residual',
                        'parameters': {
                            'distance_exponent': 0.5,
                            'cutoff_radius': 9.0,
                        },
                        'score': 0.81,
                    },
                    {
                        'theory_kind': 'tapered_distance_direction_residual',
                        'parameters': {
                            'distance_exponent': 1.5,
                            'cutoff_radius': 9.0,
                        },
                        'score': 0.84,
                    },
                ],
                'proof_checks': [],
            },
        )
        memory.planned_outcomes.append(conflict_outcome)
        predicate_plan = memory.domain_predicate_learning_agenda(limit=1)[0]
        predicate_outcome = memory.evaluate_planned_result(
            predicate_plan,
            context='localized_gravity',
            seed=9,
            report={
                'theories': [{
                    'theory_kind': 'tapered_distance_direction_residual',
                    'parameters': {
                        'distance_exponent': 1.5,
                        'cutoff_radius': 9.0,
                    },
                    'score': 0.84,
                }],
                'proof_checks': [],
            },
        )
        memory.planned_outcomes.append(predicate_outcome)

        progress = memory.rediscovery_goal_progress_report()

        self.assertEqual('blind_holdout_conflicted', blind_conflicted['outcome'])
        self.assertEqual('conflict_domain_split_supported', conflict_outcome['outcome'])
        self.assertEqual(
            'domain_predicate_rival_region',
            predicate_outcome['outcome'],
        )
        self.assertGreaterEqual(
            progress['gates']['heldout_law_stability']['score'],
            0.85,
        )
        self.assertGreaterEqual(
            progress['gates']['blind_holdout_validation']['score'],
            0.40,
        )
        self.assertLess(
            progress['gates']['blind_holdout_validation']['score'],
            0.85,
        )
        self.assertEqual(
            1,
            progress['gates']['heldout_law_stability']['evidence'][
                'domain_scoped_law_count'
            ],
        )
        self.assertEqual(
            1,
            progress['gates']['blind_holdout_validation']['evidence'][
                'blind_holdout_scoped_count'
            ],
        )

    def test_rediscovery_goal_progress_audit_prints_stricter_score(self):
        memory = self._build_selected_tapered_law_memory()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            report = run_rediscovery_goal_progress_audit(theory_memory=memory)

        self.assertIn('REDISCOVERY GOAL PROGRESS', output.getvalue())
        self.assertIn('Watched final run: not run', output.getvalue())
        self.assertIn('progress_score', report)
        self.assertIn('selected_law_parameterization', report['gates'])

    def test_localized_gravity_simple_headline_gets_structure_probe(self):
        memory = CumulativeTheoryMemory()
        memory.record_equation_case_result(
            'localized_gravity',
            0,
            {
                'interesting_equation': {
                    'target': 'baseline_adjusted_delta_velocity',
                    'expression': 'k * unit_local_inferred_vector',
                    'role': 'local_residual_direction_equation',
                    'score': 0.71,
                    'parameters': {
                        'center_x': 8.0,
                        'center_y': 12.0,
                        'cutoff_radius': 7.0,
                    },
                },
                'passed': True,
                'label_leaks': [],
            },
            phase='equation_campaign',
        )

        probe = memory.localized_gravity_structure_experiments(limit=1)[0]
        plan = memory.planned_experiments(
            world_types=['standard', 'localized_gravity'],
            object_counts=[5],
            steps=260,
            limit=1,
        )[0]
        actions = _planned_probe_actions(plan)
        outcome = memory.evaluate_planned_result(
            plan,
            context='localized_gravity',
            seed=plan['seed'],
            report={
                'theories': [{
                    'theory_kind': 'tapered_distance_direction_residual',
                    'parameters': {'distance_exponent': 1.0},
                    'score': 0.82,
                }],
                'proof_checks': [],
            },
        )

        self.assertEqual(
            'localized_gravity_structure_probe',
            probe['experiment_kind'],
        )
        self.assertEqual('localized_gravity', plan['world_type'])
        self.assertTrue(plan['quick_probe'])
        self.assertLess(plan['steps'], plan['full_steps'])
        self.assertEqual(
            [
                'inside_local_region',
                'boundary_margin',
                'outside_local_tail',
            ],
            [point['label'] for point in probe['disagreement_signature']['probe_points']],
        )
        self.assertEqual(
            'planned_localized_gravity_structure_probe',
            actions[0]['source'],
        )
        self.assertEqual(
            'localized_gravity_structure_found',
            outcome['outcome'],
        )
        self.assertIn(
            'localized_gravity_structure_experiments',
            memory.to_dict(),
        )
        self.assertIn(
            'Localized-gravity structure probes:',
            memory.summary(),
        )

    def test_post_run_replay_agenda_flags_baseline_headline_replay(self):
        memory = CumulativeTheoryMemory()
        for seed in range(4):
            memory.record_equation_case_result(
                'sideways_wind',
                seed,
                {
                    'interesting_equation': {
                        'target': 'next_y',
                        'expression': 'y + vy * dt',
                        'role': 'simple_transition',
                        'score': 0.97,
                    },
                    'passed': True,
                    'label_leaks': [],
                },
                phase='math_final_discovery',
            )

        agenda = memory.post_run_replay_agenda(limit=3)
        replay = agenda[0]
        replay_issue_groups = {
            (item['source_context'], item['replay_issue'])
            for item in agenda
        }
        plan = memory.planned_experiments(
            world_types=['sideways_wind'],
            object_counts=[5],
            steps=240,
            limit=1,
        )[0]
        outcome = memory.evaluate_planned_result(
            plan,
            context='sideways_wind',
            seed=plan['seed'],
            report={
                'interesting_equation': {
                    'target': 'baseline_adjusted_delta_velocity',
                    'expression': 'k * unit_local_inferred_vector',
                    'role': 'residual_direction_equation',
                    'score': 0.71,
                },
                'passed': True,
                'label_leaks': [],
            },
        )

        self.assertEqual('post_run_replay_revision', replay['experiment_kind'])
        self.assertEqual(
            'baseline_headline_needs_residual_replay',
            replay['replay_issue'],
        )
        self.assertTrue(replay['residual_first'])
        self.assertEqual('sideways_wind', replay['source_context'])
        self.assertEqual(0, replay['replay_seed'])
        self.assertEqual(1, len(agenda))
        self.assertEqual(1, len(replay_issue_groups))
        self.assertTrue(plan['replay_from_start'])
        self.assertEqual(0, plan['seed'])
        self.assertEqual('replay_found_residual_headline', outcome['outcome'])
        self.assertIn('post_run_replay_agenda', memory.to_dict())
        self.assertIn('Post-run replay agenda:', memory.summary())
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            _print_cumulative_theory_review(memory)
        printed = output.getvalue()
        self.assertIn('Theory post-run replay agenda:', printed)
        self.assertIn('baseline_headline_needs_residual_replay', printed)

        memory.planned_outcomes.append({
            'experiment_kind': 'post_run_replay_revision',
            'replay_key': replay['replay_key'],
            'outcome': 'replay_still_baseline_headline',
        })
        self.assertEqual([], memory.post_run_replay_agenda(limit=3))

    def test_post_run_replay_confirms_later_invariant_on_old_case(self):
        memory = CumulativeTheoryMemory()
        for seed in (0, 1, 3):
            memory.record_equation_case_result(
                'inverse_square_repulsion',
                seed,
                {
                    'interesting_equation': {
                        'target': 'baseline_adjusted_delta_velocity',
                        'expression': (
                            'k * unit_local_inferred_vector / separation^2'
                        ),
                        'role': 'residual_distance_scaled_direction_equation',
                        'score': 0.88,
                        'parameters': {'distance_exponent': 2.0},
                    },
                    'passed': True,
                    'label_leaks': [],
                },
                phase='math_final_discovery',
            )
        memory.record_equation_case_result(
            'inverse_square_repulsion',
            2,
            {
                'interesting_equation': {
                    'target': 'baseline_adjusted_delta_velocity',
                    'expression': (
                        'k * taper(separation, 8_88) * '
                        'unit_generated_center_vector / separation^0_5'
                    ),
                    'role': 'generated_operator_distance_scaled_direction_equation',
                    'score': 0.85,
                    'parameters': {'distance_exponent': 0.5},
                },
                'passed': True,
                'label_leaks': [],
            },
            phase='math_final_discovery',
        )

        agenda = memory.post_run_replay_agenda(limit=3)
        replay = next(
            item for item in agenda
            if item['replay_issue'] == 'old_headline_conflicts_with_robust_law'
        )
        plan = memory.planned_experiments(
            world_types=['inverse_square_repulsion'],
            object_counts=[5],
            steps=240,
            limit=1,
        )[0]
        outcome = memory.evaluate_planned_result(
            plan,
            context='inverse_square_repulsion',
            seed=plan['seed'],
            report={
                'interesting_equation': {
                    'target': 'baseline_adjusted_delta_velocity',
                    'expression': 'k * unit_local_inferred_vector / separation^2',
                    'role': 'residual_distance_scaled_direction_equation',
                    'score': 0.91,
                    'parameters': {'distance_exponent': 2.0},
                },
                'passed': True,
                'label_leaks': [],
            },
        )

        self.assertEqual(2, replay['replay_seed'])
        self.assertEqual(
            'inverse_separation_power',
            replay['learned_invariant']['law_family'],
        )
        self.assertEqual(2, plan['seed'])
        self.assertEqual(
            'replay_confirmed_learned_invariant',
            outcome['outcome'],
        )
        self.assertEqual(
            'k * unit_local_inferred_vector / separation^2',
            outcome['replay_expression'],
        )

    def test_post_run_replay_parameter_resolution_does_not_repeat_same_case(self):
        memory = CumulativeTheoryMemory()
        cases = [
            (0, 'k * taper(separation, 6) * unit_generated_center_vector / separation^0_5', 0.89, 0.5),
            (1, 'k * taper(separation, 6) * unit_generated_center_vector / separation^1_5', 0.91, 1.5),
            (2, 'k * taper(separation, 6) * unit_generated_center_vector / separation^0_5', 0.90, 0.5),
        ]
        for seed, expression, score, exponent in cases:
            memory.record_equation_case_result(
                'repulsion',
                seed,
                {
                    'interesting_equation': {
                        'target': 'baseline_adjusted_delta_velocity',
                        'expression': expression,
                        'role': 'generated_operator_distance_scaled_direction_equation',
                        'score': score,
                        'parameters': {
                            'cutoff_radius': 6.0,
                            'distance_exponent': exponent,
                        },
                    },
                    'passed': True,
                    'label_leaks': [],
                },
                phase='math_final_discovery',
            )

        replay = memory.post_run_replay_agenda(limit=3)[0]
        alternate_issue = (
            'old_headline_conflicts_with_robust_law'
            if replay['replay_issue'] != 'old_headline_conflicts_with_robust_law'
            else 'parameter_variant_needs_replay'
        )
        alternate_key = replay['replay_key'].rsplit(':', 1)[0] + f':{alternate_issue}'
        memory.planned_outcomes.append({
            'experiment_kind': 'post_run_replay_revision',
            'replay_key': alternate_key,
            'outcome': 'replay_needs_parameter_resolution',
        })
        replay_keys = {
            item['replay_key']
            for item in memory.post_run_replay_agenda(limit=5)
        }

        self.assertNotIn(replay['replay_key'], replay_keys)
        self.assertTrue(memory.equation_invariant_resolution_experiments(limit=1))

    def test_domain_counterexample_adds_representation_domain_predicate(self):
        loop = AutonomousDiscoveryLoop()
        support = loop.build_report([
            equation(
                key='raw_eq:residual_inferred_direction_distance_2_a',
                role='residual_distance_scaled_direction_equation',
                expression='k * unit_inferred_vector / separation^2',
                parameters={
                    'center_x': 8.0,
                    'center_y': 12.0,
                    'distance_exponent': 2.0,
                    'distance_mse_improvement': 0.44,
                },
            )
        ], step=160)
        repeat = loop.build_report([
            equation(
                key='raw_eq:residual_inferred_direction_distance_2_b',
                role='residual_distance_scaled_direction_equation',
                expression='k * unit_inferred_vector / separation^2',
                parameters={
                    'center_x': 8.5,
                    'center_y': 11.5,
                    'distance_exponent': 2.0,
                    'distance_mse_improvement': 0.38,
                },
            )
        ], step=170)
        counterexample = loop.build_report([
            equation(
                key='raw_eq:residual_local_direction_vector',
                role='local_residual_direction_equation',
                parameters={'center_x': 7.0, 'center_y': 13.0},
            )
        ], step=180)
        memory = CumulativeTheoryMemory()
        memory.record_result('inverse_square_repulsion', 0, support)
        memory.record_result('inverse_square_repulsion', 1, repeat)
        plan = memory.planned_experiments(
            world_types=['inverse_square_repulsion'],
            object_counts=[5],
            steps=240,
            seed_start=0,
            limit=1,
        )[0]

        memory.record_planned_result(
            plan,
            context='hidden_01_0001',
            seed=2,
            report=counterexample,
        )
        agenda = memory.representation_agenda(limit=5)

        domain_item = next(
            item for item in agenda
            if item['proposal_kind'] == 'domain_predicate'
        )
        self.assertEqual('learned_domain_predicate', domain_item['name'])
        self.assertEqual('domain_revision', domain_item['source'])
        self.assertIn('included and context not in excluded', domain_item['expression'])
        self.assertIn(
            'hidden_01_0001',
            domain_item['evidence']['excluded_contexts'],
        )

    def test_cumulative_theory_memory_persists_cross_run_notebook(self):
        loop = AutonomousDiscoveryLoop()
        shallow = equation(
            key='raw_eq:generated_operator_distance_1_5',
            role='generated_operator_distance_scaled_direction_equation',
            score=0.78,
            expression='k * unit_generated_center_vector / separation^1_5',
            parameters={
                'center_x': 10.0,
                'center_y': 10.0,
                'distance_exponent': 1.5,
                'distance_mse_improvement': 0.18,
            },
        )
        steep = equation(
            key='raw_eq:generated_operator_distance_2',
            role='generated_operator_distance_scaled_direction_equation',
            score=0.76,
            expression='k * unit_generated_center_vector / separation^2',
            parameters={
                'center_x': 10.0,
                'center_y': 10.0,
                'distance_exponent': 2.0,
                'distance_mse_improvement': 0.22,
            },
        )
        report = loop.build_report(
            [shallow, steep],
            step=180,
            current_count=3,
            world_width=20.0,
            world_height=20.0,
        )
        memory = CumulativeTheoryMemory()
        memory.record_result('inverse_square_repulsion', 0, report)
        plan = memory.planned_experiments(
            world_types=['inverse_square_repulsion'],
            object_counts=[5],
            steps=240,
            seed_start=0,
            limit=1,
        )[0]
        memory.record_planned_result(
            plan,
            context='inverse_square_repulsion',
            seed=1,
            report=report,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'theory-memory.json')
            memory.save(path)
            loaded = CumulativeTheoryMemory.load(path)

        packed = loaded.to_dict()
        self.assertEqual(2, len(packed['records']))
        self.assertEqual(1, len(packed['planned_outcomes']))
        self.assertEqual(
            'disagreement_still_open',
            packed['planned_outcomes'][0]['outcome'],
        )
        self.assertIn('distance_scaled_direction_residual', packed['families'])
        refined = loaded.disagreement_experiments(limit=1)[0]
        self.assertEqual('needs_refinement', refined['stagnation_status'])
        self.assertEqual(1, refined['disagreement_signature']['refinement_level'])
        self.assertEqual(
            ['very_near_center', 'mid_log_check', 'very_far_from_center'],
            [
                point['label']
                for point in refined['disagreement_signature']['probe_points']
            ],
        )

    def test_cumulative_theory_memory_consolidates_cross_world_families(self):
        loop = AutonomousDiscoveryLoop()
        first = loop.build_report([
            equation(
                key='raw_eq:residual_inferred_direction_distance_2_a',
                role='residual_distance_scaled_direction_equation',
                expression='k * unit_inferred_vector / separation^2',
                parameters={
                    'center_x': 8.0,
                    'center_y': 12.0,
                    'distance_exponent': 2.0,
                    'distance_mse_improvement': 0.44,
                },
            )
        ], step=160)
        second = loop.build_report([
            equation(
                key='raw_eq:residual_local_direction_distance_2_b',
                role='local_residual_distance_scaled_direction_equation',
                expression='k * unit_local_inferred_vector / separation^2',
                parameters={
                    'center_x': 9.0,
                    'center_y': 11.0,
                    'distance_exponent': 2.0,
                    'distance_mse_improvement': 0.31,
                },
            )
        ], step=170)
        memory = CumulativeTheoryMemory()

        memory.record_result('inverse_square_repulsion', 0, first)
        memory.record_result('hidden_00_0000', 1, second)
        packed = memory.to_dict()

        self.assertEqual(2, len(packed['records']))
        families = packed['reusable_families']
        self.assertTrue(families)
        self.assertIn('inverse_separation_power', families[0]['operator_kinds'])
        self.assertGreaterEqual(families[0]['generalization_score'], 0.6)
        self.assertEqual('reusable', families[0]['generalization_status'])
        self.assertIn('another seed', families[0]['next_proof_obligation'])
        certificate = families[0]['proof_certificate']
        self.assertEqual('reusable', certificate['status'])
        self.assertTrue(
            any('proof-like checks pass' in reason for reason in certificate['accepted_because'])
        )
        self.assertIn('hidden holdouts', certificate['not_universal_because'][-1])
        self.assertIn('proof_certificates', packed)
        self.assertEqual('reusable', packed['proof_certificates'][0]['status'])
        self.assertIn('Proof certificates:', memory.summary())
        self.assertEqual(
            'reusable',
            packed['family_evaluations'][0]['status'],
        )
        self.assertEqual(
            'replication_or_holdout',
            packed['next_experiments'][0]['experiment_kind'],
        )
        self.assertEqual(
            'new_seed_or_hidden_holdout',
            packed['next_experiments'][0]['target_context'],
        )
        plan = memory.planned_experiments(
            world_types=['inverse_square_repulsion', 'localized_gravity'],
            object_counts=[5],
            steps=240,
            seed_start=0,
            limit=1,
        )[0]
        self.assertEqual('hidden_procedural', plan['world_type'])
        self.assertTrue(plan['hidden_holdout'])
        self.assertEqual(240, plan['steps'])

    def test_cumulative_memory_writes_self_authored_equation_templates(self):
        loop = AutonomousDiscoveryLoop()
        radial_a = loop.build_report([
            equation(
                key='raw_eq:radial_a',
                role='residual_distance_scaled_direction_equation',
                expression='k * unit_inferred_vector / separation^2',
                parameters={
                    'center_x': 8.0,
                    'center_y': 12.0,
                    'distance_exponent': 2.0,
                    'distance_mse_improvement': 0.44,
                },
            )
        ], step=160)
        radial_b = loop.build_report([
            equation(
                key='raw_eq:radial_b',
                role='local_residual_distance_scaled_direction_equation',
                expression='k * unit_local_inferred_vector / separation^2',
                parameters={
                    'center_x': 9.0,
                    'center_y': 11.0,
                    'distance_exponent': 2.0,
                    'distance_mse_improvement': 0.31,
                },
            )
        ], step=170)
        radial_rough = loop.build_report([
            equation(
                key='raw_eq:radial_rough',
                role='generated_operator_distance_scaled_direction_equation',
                expression='k * unit_generated_center_vector / separation^1_5',
                parameters={
                    'center_x': 9.5,
                    'center_y': 11.5,
                    'distance_exponent': 1.5,
                    'distance_mse_improvement': 0.18,
                },
            )
        ], step=180)
        vortex_a = loop.build_report([
            equation(
                key='raw_eq:vortex_a',
                role='generated_operator_tapered_distance_perpendicular_equation',
                expression='k * taper(separation, 8) * perpendicular(unit_generated_center_vector) / separation^0_5',
                parameters={
                    'center_x': 10.0,
                    'center_y': 10.0,
                    'cutoff_radius': 8.0,
                    'distance_exponent': 0.5,
                    'tapered_vs_smooth_improvement': 0.25,
                    'tapered_vs_cutoff_improvement': 0.18,
                },
            )
        ], step=190)
        vortex_b = loop.build_report([
            equation(
                key='raw_eq:vortex_b',
                role='generated_operator_tapered_distance_perpendicular_equation',
                expression='k * taper(separation, 7.5) * perpendicular(unit_generated_center_vector) / separation^0_5',
                parameters={
                    'center_x': 10.5,
                    'center_y': 9.5,
                    'cutoff_radius': 7.5,
                    'distance_exponent': 0.5,
                    'tapered_vs_smooth_improvement': 0.22,
                    'tapered_vs_cutoff_improvement': 0.16,
                },
            )
        ], step=200)
        memory = CumulativeTheoryMemory()

        memory.record_result('inverse_square_repulsion', 0, radial_a)
        memory.record_result('hidden_00_0000', 1, radial_b)
        memory.record_result('inverse_square_repulsion', 2, radial_rough)
        memory.record_result('vortex', 0, vortex_a)
        memory.record_result('vortex', 1, vortex_b)
        authored = memory.self_authored_equations(limit=5)
        packed = memory.to_dict()

        radial = next(
            item for item in authored
            if item['equation_kind'] == 'distance_scaled_direction_residual'
        )
        self.assertEqual(3, radial['support_count'])
        self.assertEqual(2.0, radial['dominant_parameters']['distance_exponent'])
        self.assertEqual(
            'baseline_adjusted_delta_velocity ~= k * unit(center - position) / separation^2',
            radial['expression'],
        )
        self.assertIn('domain_nonzero_positive', radial['proof_obligations'])
        self.assertTrue(any(
            '1.5' in note and '2' in note
            for note in radial['approximation_notes']
        ))
        self.assertTrue(any(
            'near/far holdouts' in test
            for test in radial['falsification_tests']
        ))

        vortex = next(
            item for item in authored
            if item['equation_kind'] == 'tapered_distance_perpendicular_residual'
        )
        self.assertEqual(2, vortex['support_count'])
        self.assertIn(
            'perpendicular(unit(center - position))',
            vortex['expression'],
        )
        self.assertIn('symmetry_invariance', vortex['proof_obligations'])
        self.assertTrue(any(
            'quarter turn' in test
            for test in vortex['falsification_tests']
        ))

        self.assertIn('self_authored_equations', packed)
        self.assertTrue(packed['self_authored_equations'])
        self.assertTrue(
            packed['discovery_readiness']['gates'][
                'self_authored_equation_synthesis'
            ]['passed']
        )
        self.assertTrue(
            packed['discovery_evidence_dossier']['self_authored_equations']
        )
        self.assertIn('Self-authored equations:', memory.summary())

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            _print_cumulative_theory_review(memory)
        printed = output.getvalue()

        self.assertIn('Theory self-authored equations:', printed)
        self.assertIn('baseline_adjusted_delta_velocity', printed)
        self.assertIn('perpendicular(unit(center - position))', printed)

    def test_domain_curriculum_covers_core_math_and_transfer_bridges(self):
        memory = CumulativeTheoryMemory()

        curriculum = memory.math_domain_curriculum()
        agenda = memory.domain_curriculum_agenda(limit=20)
        transfers = memory.domain_transfer_experiments(limit=20)
        readiness = memory.discovery_readiness_report()
        packed = memory.to_dict()

        required = {
            'arithmetic_quantity',
            'algebra_equations',
            'geometry_space',
            'calculus_change',
            'probability_uncertainty',
            'logic_proof',
            'discrete_structures',
            'symmetry_invariance',
            'optimization_extrema',
            'dynamics_systems',
            'information_computation',
            'higher_dimensions',
        }
        self.assertEqual(required, set(curriculum['required_domains']))
        self.assertGreaterEqual(curriculum['domain_count'], 12)
        self.assertGreaterEqual(curriculum['transfer_bridge_count'], 12)
        self.assertIn('curriculum_policy', curriculum)
        self.assertEqual(
            required,
            {item['domain_key'] for item in agenda},
        )
        self.assertTrue(all(item['target_primitives'] for item in agenda))
        self.assertTrue(all(item['proof_pressure'] for item in agenda))
        self.assertTrue(transfers)
        self.assertTrue(all(
            item['experiment_kind'] == 'domain_transfer_probe'
            and item['transfer_question']
            and item['falsifies_if']
            and item['suggested_world_seed']['combined_pressure']
            for item in transfers
        ))
        self.assertTrue(
            readiness['gates']['broad_domain_curriculum']['passed']
        )
        self.assertTrue(
            readiness['gates']['domain_transfer_loop']['passed']
        )
        self.assertIn('math_domain_curriculum', packed)
        self.assertIn('domain_curriculum_agenda', packed)
        self.assertIn('domain_transfer_experiments', packed)
        self.assertIn('domain_transfer_probes', packed['discovery_evidence_dossier'])
        self.assertEqual([], packed['discovery_evidence_dossier']['domain_transfer_probes'])
        self.assertIn('Math domain curriculum:', memory.summary())
        self.assertIn('Domain transfer probes:', memory.summary())

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            _print_cumulative_theory_review(memory)
        printed = output.getvalue()

        self.assertIn('Theory math domain curriculum:', printed)
        self.assertIn('Theory domain transfer probes:', printed)
        self.assertIn('arithmetic_quantity', printed)

    def test_cumulative_theory_memory_marks_local_family_as_transfer_gap(self):
        loop = AutonomousDiscoveryLoop()
        first = loop.build_report([
            equation(
                key='raw_eq:residual_inferred_direction_distance_2_a',
                role='residual_distance_scaled_direction_equation',
                expression='k * unit_inferred_vector / separation^2',
                parameters={
                    'center_x': 8.0,
                    'center_y': 12.0,
                    'distance_exponent': 2.0,
                    'distance_mse_improvement': 0.44,
                },
            )
        ], step=160)
        second = loop.build_report([
            equation(
                key='raw_eq:residual_inferred_direction_distance_2_b',
                role='residual_distance_scaled_direction_equation',
                expression='k * unit_inferred_vector / separation^2',
                parameters={
                    'center_x': 8.5,
                    'center_y': 11.5,
                    'distance_exponent': 2.0,
                    'distance_mse_improvement': 0.38,
                },
            )
        ], step=170)
        memory = CumulativeTheoryMemory()

        memory.record_result('inverse_square_repulsion', 0, first)
        memory.record_result('inverse_square_repulsion', 1, second)
        packed = memory.to_dict()

        family = packed['reusable_families'][0]
        self.assertEqual('local', family['generalization_status'])
        self.assertIn('different world context', family['next_proof_obligation'])
        self.assertEqual('local', packed['generalization_gaps'][0]['status'])
        self.assertEqual(
            'transfer_test',
            packed['next_experiments'][0]['experiment_kind'],
        )
        self.assertEqual(
            'unseen_world_context',
            packed['next_experiments'][0]['target_context'],
        )
        self.assertIn(
            'inverse_square_repulsion',
            packed['next_experiments'][0]['avoid_contexts'],
        )
        plan = memory.planned_experiments(
            world_types=['inverse_square_repulsion', 'localized_gravity'],
            object_counts=[5, 7],
            steps=260,
            seed_start=0,
            limit=1,
        )[0]
        self.assertEqual('localized_gravity', plan['world_type'])
        self.assertEqual(5, plan['object_count'])
        self.assertEqual(260, plan['steps'])
        self.assertEqual('distance_scaled_direction_residual', plan['theory_kind'])
        transfer_report = loop.build_report([
            equation(
                key='raw_eq:residual_inferred_direction_distance_2_transfer',
                role='residual_distance_scaled_direction_equation',
                expression='k * unit_inferred_vector / separation^2',
                parameters={
                    'center_x': 7.0,
                    'center_y': 13.0,
                    'distance_exponent': 2.0,
                    'distance_mse_improvement': 0.36,
                },
            )
        ], step=190)
        outcome = memory.record_planned_result(
            plan,
            context='localized_gravity',
            seed=0,
            report=transfer_report,
        )

        self.assertEqual('transfer_confirmed', outcome['outcome'])
        self.assertTrue(outcome['new_context'])
        self.assertTrue(outcome['proof_passed'])
        self.assertEqual(
            'transfer_confirmed',
            memory.to_dict()['planned_outcomes'][0]['outcome'],
        )
        revised_family = memory.to_dict()['families']['distance_scaled_direction_residual']
        self.assertGreaterEqual(
            revised_family['proof_evidence']['transfer_success_count'],
            1,
        )
        self.assertNotEqual('local', revised_family['generalization_status'])
        self.assertIn(
            'localized_gravity',
            revised_family['domain_hypothesis']['included_contexts'],
        )
        self.assertFalse(revised_family['domain_hypothesis']['revision_needed'])

    def test_equation_followup_cases_run_plans_and_record_outcomes(self):
        loop = AutonomousDiscoveryLoop()
        first = loop.build_report([
            equation(
                key='raw_eq:residual_inferred_direction_distance_2_a',
                role='residual_distance_scaled_direction_equation',
                expression='k * unit_inferred_vector / separation^2',
                parameters={
                    'center_x': 8.0,
                    'center_y': 12.0,
                    'distance_exponent': 2.0,
                    'distance_mse_improvement': 0.44,
                },
            )
        ], step=160)
        second = loop.build_report([
            equation(
                key='raw_eq:residual_inferred_direction_distance_2_b',
                role='residual_distance_scaled_direction_equation',
                expression='k * unit_inferred_vector / separation^2',
                parameters={
                    'center_x': 8.5,
                    'center_y': 11.5,
                    'distance_exponent': 2.0,
                    'distance_mse_improvement': 0.38,
                },
            )
        ], step=170)
        transfer_report = loop.build_report([
            equation(
                key='raw_eq:residual_inferred_direction_distance_2_transfer',
                role='residual_distance_scaled_direction_equation',
                expression='k * unit_inferred_vector / separation^2',
                parameters={
                    'center_x': 7.0,
                    'center_y': 13.0,
                    'distance_exponent': 2.0,
                    'distance_mse_improvement': 0.36,
                },
            )
        ], step=190)
        memory = CumulativeTheoryMemory()
        memory.record_result('inverse_square_repulsion', 0, first)
        memory.record_result('inverse_square_repulsion', 1, second)
        progress_events = []

        def fake_run_case(**kwargs):
            self.assertEqual('localized_gravity', kwargs['context'])
            self.assertEqual('localized_gravity', kwargs['world_type'])
            self.assertIn('planned_actions', kwargs)
            return {
                'context': kwargs['context'],
                'seed': kwargs['seed'],
                'objects': kwargs['object_count'],
                'steps': kwargs['steps'],
                'equation_count': 1,
                'installed_count': 1,
                'interesting_equation': {
                    'target': 'baseline_adjusted_delta_velocity',
                    'expression': 'k * unit_local_inferred_vector / separation^2',
                    'role': 'residual_distance_scaled_direction_equation',
                    'score': 0.36,
                    'parameters': {
                        'center_x': 7.0,
                        'center_y': 13.0,
                        'distance_exponent': 2.0,
                    },
                },
                'interesting_score': 0.36,
                'label_leaks': [],
                'probe_suggestions': [],
                'discovery_loop': transfer_report.to_dict(),
                'passed': True,
            }

        followups = _run_equation_followup_cases(
            theory_memory=memory,
            world_types=['inverse_square_repulsion', 'localized_gravity'],
            object_counts=[5],
            steps=240,
            num_agents=2,
            limit=1,
            run_case_fn=fake_run_case,
            progress_fn=lambda event, payload: progress_events.append(
                (event, dict(payload))
            ),
        )

        self.assertEqual(1, len(followups))
        self.assertEqual(['start', 'finish'], [event for event, _ in progress_events])
        self.assertEqual(1, progress_events[0][1]['iteration'])
        self.assertEqual('localized_gravity', progress_events[0][1]['context'])
        self.assertEqual(
            'transfer_test',
            progress_events[0][1]['plan']['experiment_kind'],
        )
        self.assertEqual(
            'transfer_confirmed',
            progress_events[1][1]['outcome']['outcome'],
        )
        self.assertEqual('autonomous_followup', followups[0]['phase'])
        self.assertEqual(1, len(memory.equation_case_records))
        self.assertEqual(
            'equation_followup',
            memory.equation_case_records[0]['phase'],
        )
        self.assertEqual(
            'k * unit_local_inferred_vector / separation^2',
            memory.equation_case_records[0]['expression'],
        )
        self.assertEqual(
            'transfer_test',
            followups[0]['planned_experiment']['experiment_kind'],
        )
        self.assertEqual(
            'transfer_confirmed',
            followups[0]['planned_experiment_outcome']['outcome'],
        )
        revised_family = memory.to_dict()['families']['distance_scaled_direction_residual']
        self.assertNotEqual('local', revised_family['generalization_status'])
        self.assertIn(
            'localized_gravity',
            revised_family['domain_hypothesis']['included_contexts'],
        )

    def test_equation_followup_cases_resolve_operator_prior_feedback_outcomes(self):
        operator_key = 'operator:memory_prior:inverse_separation_power:2.25:direction:inverse_square_repulsion:0'
        memory = CumulativeTheoryMemory()
        memory.operator_prior_outcomes = [
            {
                'context': 'inverse_square_repulsion',
                'seed': 0,
                'operator_key': operator_key,
                'operator_kind': 'inverse_separation_power',
                'outcome': 'confirmed',
                'best_score': 0.84,
                'matching_equation_count': 1,
                'parameters': {
                    'center_x': 10.0,
                    'center_y': 10.0,
                    'distance_exponent': 2.25,
                    'relation': 'direction',
                },
                'refined_parameters': {
                    'center_x': 10.0,
                    'center_y': 10.0,
                    'distance_exponent': 2.25,
                    'relation': 'direction',
                },
            },
            {
                'context': 'standard',
                'seed': 1,
                'operator_key': operator_key,
                'operator_kind': 'inverse_separation_power',
                'outcome': 'unmatched',
                'best_score': 0.0,
                'matching_equation_count': 0,
                'parameters': {
                    'center_x': 10.0,
                    'center_y': 10.0,
                    'distance_exponent': 2.25,
                    'relation': 'direction',
                },
            },
        ]

        def fake_run_case(**kwargs):
            self.assertEqual('standard', kwargs['context'])
            self.assertEqual('standard', kwargs['world_type'])
            self.assertEqual(
                'planned_operator_prior_repair',
                kwargs['planned_actions'][0]['source'],
            )
            return {
                'context': kwargs['context'],
                'seed': kwargs['seed'],
                'objects': kwargs['object_count'],
                'steps': kwargs['steps'],
                'equation_count': 1,
                'installed_count': 1,
                'interesting_equation': {},
                'interesting_score': 0.76,
                'label_leaks': [],
                'probe_suggestions': [],
                'discovery_loop': {'theories': [], 'proof_checks': []},
                'operator_prior_results': [{
                    'operator_key': operator_key,
                    'operator_kind': 'inverse_separation_power',
                    'outcome': 'confirmed',
                    'best_score': 0.76,
                    'matching_equation_count': 1,
                    'parameters': {
                        'center_x': 10.0,
                        'center_y': 10.0,
                        'distance_exponent': 2.25,
                        'relation': 'direction',
                    },
                    'best_equation': {
                        'role': 'generated_operator_distance_scaled_direction_equation',
                        'score': 0.76,
                        'parameters': {
                            'center_x': 10.0,
                            'center_y': 10.0,
                            'distance_exponent': 2.1,
                            'relation': 'direction',
                        },
                    },
                }],
                'passed': True,
            }

        followups = _run_equation_followup_cases(
            theory_memory=memory,
            world_types=['standard', 'localized_gravity'],
            object_counts=[5],
            steps=240,
            num_agents=2,
            limit=1,
            run_case_fn=fake_run_case,
        )

        self.assertEqual(1, len(followups))
        outcome = followups[0]['planned_experiment_outcome']
        self.assertEqual(
            'operator_prior_domain_repair',
            followups[0]['planned_experiment']['experiment_kind'],
        )
        self.assertEqual('operator_prior_repair_confirmed', outcome['outcome'])
        self.assertTrue(outcome['operator_prior_evaluated'])
        self.assertEqual('confirmed', outcome['operator_prior_feedback_outcome'])
        self.assertTrue(outcome['operator_prior_refinement_detected'])
        packed = memory.to_dict()
        self.assertEqual(
            'operator_prior_repair_confirmed',
            packed['planned_outcomes'][0]['outcome'],
        )
        self.assertEqual(1, packed['records'][0]['operator_prior_result_count'])
        claim = next(
            item for item in memory.operator_prior_discovery_claims(limit=3)
            if item['operator_key'] == operator_key
        )
        self.assertEqual('repaired', claim['status'])
        self.assertIn('inverse_separation_power', claim['claim'])
        self.assertEqual(2, claim['proof_evidence']['confirmed_count'])
        self.assertEqual(1, claim['proof_evidence']['repair_confirmed_count'])
        self.assertIn('operator_prior_discovery_claims', packed)
        self.assertIn('Operator prior discovery claims:', memory.summary())
        chain = next(
            item for item in memory.operator_prior_discovery_chains(limit=3)
            if item['operator_key'] == operator_key
        )
        self.assertEqual('repaired', chain['status'])
        self.assertIn('standard', chain['failure_contexts'])
        step_kinds = [step['step_kind'] for step in chain['steps']]
        self.assertIn('operator_prior_invented', step_kinds)
        self.assertIn('operator_prior_tested', step_kinds)
        self.assertIn('anomaly_detected', step_kinds)
        self.assertIn('planned_outcome_recorded', step_kinds)
        self.assertIn('discovery_claim_synthesized', step_kinds)
        self.assertIn('next_experiment_selected', step_kinds)
        self.assertIn('operator_prior_discovery_chains', packed)
        self.assertIn('Operator prior discovery chains:', memory.summary())
        self.assertFalse(any(
            item['operator_prior_key'] == operator_key
            for item in memory.operator_prior_repair_experiments(limit=3)
        ))
        claim_experiment = next(
            item for item in memory.operator_prior_claim_experiments(limit=3)
            if item['operator_prior_key'] == operator_key
        )
        self.assertEqual(
            'operator_prior_refinement_validation',
            claim_experiment['experiment_kind'],
        )
        self.assertEqual(
            'operator_prior_unseen_context',
            claim_experiment['target_context'],
        )
        self.assertIn('operator_prior_claim_experiments', packed)
        self.assertIn('Operator prior claim experiments:', memory.summary())
        claim_prior = next(
            item for item in memory.generated_operator_priors(
                context='localized_gravity',
                limit=5,
            )
            if item['key'] == operator_key
        )
        self.assertEqual('operator_prior_discovery_claim', claim_prior['generated_from'])
        self.assertAlmostEqual(
            2.1,
            claim_prior['parameters']['distance_exponent'],
            delta=0.01,
        )
        validation_plan = next(
            plan for plan in memory.planned_experiments(
                world_types=['standard', 'inverse_square_repulsion', 'localized_gravity'],
                object_counts=[5],
                steps=240,
                limit=3,
            )
            if plan['experiment_kind'] == 'operator_prior_refinement_validation'
        )
        self.assertEqual('localized_gravity', validation_plan['world_type'])
        self.assertEqual(operator_key, validation_plan['operator_prior_key'])
        self.assertEqual('repaired', validation_plan['operator_prior_claim']['status'])
        memory.planned_outcomes.append({
            'experiment_kind': 'operator_prior_refinement_validation',
            'operator_prior_key': operator_key,
            'context': 'localized_gravity',
            'outcome': 'operator_prior_validation_confirmed',
        })
        hidden_holdout = next(
            item for item in memory.operator_prior_claim_experiments(limit=3)
            if item['operator_prior_key'] == operator_key
        )
        self.assertEqual(
            'operator_prior_hidden_holdout_counterexample',
            hidden_holdout['experiment_kind'],
        )
        self.assertEqual('hidden_holdout', hidden_holdout['target_context'])

    def test_discovery_readiness_audits_human_style_loop_gates(self):
        loop = AutonomousDiscoveryLoop()
        shallow = equation(
            key='raw_eq:generated_operator_distance_1_5',
            role='generated_operator_distance_scaled_direction_equation',
            score=0.78,
            expression='k * unit_generated_center_vector / separation^1_5',
            parameters={
                'center_x': 10.0,
                'center_y': 10.0,
                'distance_exponent': 1.5,
                'distance_mse_improvement': 0.18,
            },
        )
        steep = equation(
            key='raw_eq:generated_operator_distance_2',
            role='generated_operator_distance_scaled_direction_equation',
            score=0.76,
            expression='k * unit_generated_center_vector / separation^2',
            parameters={
                'center_x': 10.0,
                'center_y': 10.0,
                'distance_exponent': 2.0,
                'distance_mse_improvement': 0.22,
            },
        )
        memory = CumulativeTheoryMemory()
        memory.record_result(
            'inverse_square_repulsion',
            0,
            loop.build_report([shallow, steep], step=180),
        )
        memory.record_result(
            'inverse_square_repulsion',
            1,
            loop.build_report([steep], step=181),
        )
        prior = next(
            item for item in memory.generated_operator_priors(limit=5)
            if item['operator_kind'] == 'inverse_separation_power'
        )
        memory.record_operator_prior_results(
            'inverse_square_repulsion',
            2,
            {
                'operator_prior_results': [{
                    'operator_key': prior['key'],
                    'operator_kind': prior['operator_kind'],
                    'outcome': 'confirmed',
                    'best_score': 0.84,
                    'matching_equation_count': 1,
                    'parameters': prior['parameters'],
                    'best_equation': {
                        'role': 'generated_operator_distance_scaled_direction_equation',
                        'score': 0.84,
                        'parameters': {
                            **prior['parameters'],
                            'distance_exponent': 2.25,
                        },
                    },
                }]
            },
        )
        memory.record_operator_prior_results(
            'standard',
            3,
            {
                'operator_prior_results': [{
                    'operator_key': prior['key'],
                    'operator_kind': prior['operator_kind'],
                    'outcome': 'unmatched',
                    'best_score': 0.0,
                    'matching_equation_count': 0,
                    'parameters': prior['parameters'],
                }]
            },
        )
        memory.planned_outcomes.append({
            'experiment_kind': 'operator_prior_domain_repair',
            'operator_prior_key': prior['key'],
            'context': 'standard',
            'outcome': 'operator_prior_repair_confirmed',
            'operator_prior_refined_parameters': {
                **prior['parameters'],
                'distance_exponent': 2.1,
            },
        })
        memory.record_autonomous_scientist_loop(seed_count=2, variants=[0, 1])

        readiness = memory.discovery_readiness_report()
        packed = memory.to_dict()

        self.assertGreaterEqual(readiness['readiness_score'], 0.8)
        self.assertIn(readiness['status'], {'nearly_ready', 'ready_for_watched_final'})
        self.assertTrue(readiness['gates']['residual_to_theory']['passed'])
        self.assertTrue(readiness['gates']['representation_agenda']['passed'])
        self.assertTrue(
            readiness['gates']['first_principles_adaptive_dimensions']['passed']
        )
        self.assertTrue(
            readiness['gates']['algebraic_foundation_baseline']['passed']
        )
        self.assertTrue(
            readiness['gates']['baseline_experiment_templates']['passed']
        )
        self.assertGreaterEqual(
            readiness['gates']['algebraic_foundation_baseline']['evidence'][
                'expression_family_count'
            ],
            16,
        )
        self.assertTrue(readiness['gates']['operator_discovery_claims']['passed'])
        self.assertTrue(
            readiness['gates']['self_authored_equation_synthesis']['passed']
        )
        self.assertTrue(
            readiness['gates']['scientist_invariant_consolidation']['passed']
        )
        self.assertTrue(
            readiness['gates']['scientist_residual_experiment_loop']['passed']
        )
        self.assertTrue(
            readiness['gates']['scientist_harder_hidden_worlds']['passed']
        )
        self.assertTrue(
            readiness['gates']['scientist_richer_equation_writing']['passed']
        )
        self.assertTrue(readiness['gates']['scientist_live_trace']['passed'])
        self.assertGreaterEqual(
            readiness['gates']['operator_discovery_claims']['evidence']['chain_count'],
            1,
        )
        self.assertTrue(readiness['gates']['claim_driven_planning']['passed'])
        dossier = readiness['evidence_dossier']
        self.assertTrue(dossier['chains'])
        self.assertEqual(prior['key'], dossier['chains'][0]['operator_key'])
        self.assertGreaterEqual(dossier['chains'][0]['step_count'], 4)
        self.assertTrue(dossier['claims'])
        self.assertEqual(prior['key'], dossier['claims'][0]['operator_key'])
        self.assertTrue(dossier['planned_tests'])
        planned_kinds = {
            item['experiment_kind']
            for item in dossier['planned_tests']
        }
        self.assertTrue(
            planned_kinds
            & {
                'operator_prior_refinement_validation',
                'operator_prior_domain_repair',
                'operator_prior_hidden_holdout_counterexample',
            }
        )
        self.assertTrue(readiness['recommended_actions'])
        if readiness['ready_for_watched_final']:
            self.assertEqual(
                'hold_for_user',
                readiness['recommended_actions'][0]['action_kind'],
            )
            self.assertTrue(readiness['recommended_actions'][0]['runs_final'])
            self.assertIn(
                '--theory-memory-file tmp/theory-memory.json',
                readiness['recommended_actions'][0]['command'],
            )
        else:
            self.assertTrue(all(
                not action['runs_final']
                for action in readiness['recommended_actions']
            ))
        self.assertIn('discovery_readiness', packed)
        self.assertIn('discovery_evidence_dossier', packed)
        self.assertIn('self_authored_equations', packed)
        self.assertIn('first_principles_basis', packed)
        self.assertIn('baseline_experiment_templates', packed)
        self.assertIn('adaptive_dimension_agenda', packed)
        self.assertIn('algebraic_foundation_baseline', packed)
        self.assertIn('algebraic_expression_agenda', packed)
        self.assertIn('autonomous_scientist_records', packed)
        self.assertIn('autonomous_scientist_evidence', packed)
        self.assertIn('Discovery readiness:', memory.summary())
        self.assertIn('Discovery evidence dossier:', memory.summary())
        self.assertIn('Self-authored equations:', memory.summary())
        self.assertIn('Adaptive dimensions:', memory.summary())
        self.assertIn('Algebraic foundation:', memory.summary())
        self.assertIn('Baseline experiment templates:', memory.summary())
        self.assertIn('Autonomous scientist loop:', memory.summary())
        primitive_keys = {
            item['key']
            for item in packed['first_principles_basis']
        }
        self.assertTrue({
            'cardinality_grouping',
            'rate_change',
            'uncertainty_sampling',
            'falsification_probe',
        } <= primitive_keys)
        template_kinds = {
            item['template_kind']
            for item in packed['baseline_experiment_templates']
        }
        self.assertTrue({
            'residual_perturbation',
            'equation_race',
            'dimension_lift',
            'compression_replay',
        } <= template_kinds)

        empty_readiness = CumulativeTheoryMemory().discovery_readiness_report()
        self.assertEqual('early', empty_readiness['status'])
        self.assertFalse(
            empty_readiness['gates']['first_principles_adaptive_dimensions']['passed']
        )
        self.assertTrue(
            empty_readiness['gates']['algebraic_foundation_baseline']['passed']
        )
        self.assertTrue(
            empty_readiness['gates']['baseline_experiment_templates']['passed']
        )
        self.assertEqual([], empty_readiness['evidence_dossier']['chains'])
        self.assertEqual([], empty_readiness['evidence_dossier']['planned_tests'])
        self.assertEqual([], empty_readiness['evidence_dossier']['self_authored_equations'])
        self.assertTrue(empty_readiness['recommended_actions'])
        self.assertIn(
            'non_final_autonomous_scientist_loop',
            {
                action['action_kind']
                for action in empty_readiness['recommended_actions']
            },
        )
        self.assertTrue(all(
            not action['runs_final']
            for action in empty_readiness['recommended_actions']
        ))

    def test_equation_followup_cases_replan_after_each_outcome(self):
        loop = AutonomousDiscoveryLoop()
        first = loop.build_report([
            equation(
                key='raw_eq:residual_inferred_direction_distance_2_a',
                role='residual_distance_scaled_direction_equation',
                expression='k * unit_inferred_vector / separation^2',
                parameters={
                    'center_x': 8.0,
                    'center_y': 12.0,
                    'distance_exponent': 2.0,
                    'distance_mse_improvement': 0.44,
                },
            )
        ], step=160)
        second = loop.build_report([
            equation(
                key='raw_eq:residual_inferred_direction_distance_2_b',
                role='residual_distance_scaled_direction_equation',
                expression='k * unit_inferred_vector / separation^2',
                parameters={
                    'center_x': 8.5,
                    'center_y': 11.5,
                    'distance_exponent': 2.0,
                    'distance_mse_improvement': 0.38,
                },
            )
        ], step=170)
        transfer_report = loop.build_report([
            equation(
                key='raw_eq:residual_inferred_direction_distance_2_transfer',
                role='residual_distance_scaled_direction_equation',
                expression='k * unit_inferred_vector / separation^2',
                parameters={
                    'center_x': 7.0,
                    'center_y': 13.0,
                    'distance_exponent': 2.0,
                    'distance_mse_improvement': 0.36,
                },
            )
        ], step=190)
        memory = CumulativeTheoryMemory()
        memory.record_result('inverse_square_repulsion', 0, first)
        memory.record_result('inverse_square_repulsion', 1, second)
        calls = []

        def fake_run_case(**kwargs):
            calls.append(kwargs)
            return {
                'context': kwargs['context'],
                'seed': kwargs['seed'],
                'objects': kwargs['object_count'],
                'steps': kwargs['steps'],
                'equation_count': 1,
                'installed_count': 1,
                'interesting_equation': {},
                'interesting_score': 0.36,
                'label_leaks': [],
                'probe_suggestions': [],
                'discovery_loop': transfer_report.to_dict(),
                'passed': True,
            }

        followups = _run_equation_followup_cases(
            theory_memory=memory,
            world_types=['inverse_square_repulsion', 'localized_gravity'],
            object_counts=[5],
            steps=240,
            num_agents=2,
            limit=2,
            run_case_fn=fake_run_case,
        )

        self.assertEqual(2, len(followups))
        self.assertEqual('transfer_test', followups[0]['planned_experiment']['experiment_kind'])
        self.assertEqual(
            'hidden_holdout_counterexample',
            followups[1]['planned_experiment']['experiment_kind'],
        )
        self.assertEqual(1, followups[0]['followup_iteration'])
        self.assertEqual(2, followups[1]['followup_iteration'])
        self.assertEqual('localized_gravity', calls[0]['world_type'])
        self.assertEqual('hidden_procedural', calls[1]['world_type'])

    def test_cumulative_theory_memory_marks_weak_proof_family_for_counterexample(self):
        loop = AutonomousDiscoveryLoop()
        first = loop.build_report([
            equation(
                key='raw_eq:residual_inferred_direction_distance_2_a',
                role='residual_distance_scaled_direction_equation',
                expression='k * unit_inferred_vector / separation^2',
                parameters={
                    'center_x': 8.0,
                    'center_y': 12.0,
                    'distance_exponent': 2.0,
                    'distance_mse_improvement': 0.01,
                },
            )
        ], step=160)
        second = loop.build_report([
            equation(
                key='raw_eq:residual_local_direction_distance_2_b',
                role='local_residual_distance_scaled_direction_equation',
                expression='k * unit_local_inferred_vector / separation^2',
                parameters={
                    'center_x': 9.0,
                    'center_y': 11.0,
                    'distance_exponent': 2.0,
                    'distance_mse_improvement': 0.01,
                },
            )
        ], step=170)
        memory = CumulativeTheoryMemory()

        memory.record_result('inverse_square_repulsion', 0, first)
        memory.record_result('hidden_00_0000', 1, second)
        packed = memory.to_dict()

        family = packed['reusable_families'][0]
        self.assertEqual('needs_counterexample', family['generalization_status'])
        self.assertEqual(0.0, family['proof_rate'])
        self.assertEqual('needs_counterexample', family['proof_certificate']['status'])
        self.assertTrue(
            any(
                'proof-like checks pass at only' in reason
                for reason in family['proof_certificate']['not_universal_because']
            )
        )
        self.assertEqual('needs_counterexample', packed['proof_gaps'][0]['status'])
        self.assertIn('disagreement probe', packed['proof_gaps'][0]['next_check'])
        self.assertEqual(
            'disagreement_counterexample',
            packed['next_experiments'][0]['experiment_kind'],
        )
        self.assertEqual(
            'rival_or_hidden_context',
            packed['next_experiments'][0]['target_context'],
        )
        self.assertIn(
            'rival theories',
            packed['next_experiments'][0]['falsifies_if'],
        )
        plan = memory.planned_experiments(
            world_types=['inverse_square_repulsion'],
            object_counts=[5],
            steps=240,
            seed_start=0,
            limit=1,
        )[0]
        self.assertEqual('hidden_procedural', plan['world_type'])
        self.assertTrue(plan['hidden_holdout'])
        self.assertEqual('disagreement_counterexample', plan['experiment_kind'])
        counterexample_report = loop.build_report([
            equation(
                key='raw_eq:residual_local_direction_vector',
                role='local_residual_direction_equation',
                parameters={'center_x': 7.0, 'center_y': 13.0},
            )
        ], step=180)
        outcome = memory.record_planned_result(
            plan,
            context='hidden_01_0001',
            seed=2,
            report=counterexample_report,
        )

        self.assertEqual('counterexample_found', outcome['outcome'])
        self.assertFalse(outcome['found_family'])
        self.assertEqual(
            'counterexample_found',
            memory.to_dict()['planned_outcomes'][0]['outcome'],
        )
        packed_after_revision = memory.to_dict()
        revised_family = packed_after_revision['families']['distance_scaled_direction_residual']
        self.assertEqual('domain_limited', revised_family['generalization_status'])
        self.assertEqual(
            1,
            revised_family['proof_evidence']['counterexample_count'],
        )
        self.assertIn(
            'inverse_square_repulsion',
            revised_family['domain_hypothesis']['included_contexts'],
        )
        self.assertIn(
            'hidden_01_0001',
            revised_family['domain_hypothesis']['excluded_contexts'],
        )
        self.assertTrue(revised_family['domain_hypothesis']['revision_needed'])
        self.assertIn(
            'exclude hidden_01_0001',
            revised_family['domain_hypothesis']['claim'],
        )
        revised_certificate = revised_family['proof_certificate']
        self.assertEqual('domain_limited', revised_certificate['status'])
        self.assertIn('excluded contexts', revised_certificate['would_break_if'])
        self.assertTrue(
            any(
                'hidden_01_0001' in reason
                for reason in revised_certificate['not_universal_because']
            )
        )
        self.assertIn('revise the family domain', revised_certificate['next_obligation'])
        self.assertEqual(
            'distance_scaled_direction_residual',
            packed_after_revision['domain_revisions'][0]['theory_kind'],
        )
        self.assertIn(
            'hidden_01_0001',
            packed_after_revision['domain_revisions'][0]['domain_hypothesis']['excluded_contexts'],
        )
        self.assertIn('Domain revisions:', memory.summary())
        self.assertEqual(
            'domain_refinement',
            revised_family['experiment_recommendation']['experiment_kind'],
        )

    def test_equation_review_pack_contains_discovery_loop_report(self):
        workbench = EquationWorkbench()
        workbench.equations = {
            'raw_eq:residual_periodic_x_80': equation(
                key='raw_eq:residual_periodic_x_80',
                role='residual_periodic_equation',
                target='baseline_adjusted_delta_vx',
                expression='a * sin(step/80) + b * cos(step/80)',
                parameters={'period_steps': 80},
            )
        }

        pack = workbench.review_pack()

        self.assertIn('discovery_loop', pack)
        self.assertEqual('probe_ready', pack['discovery_loop']['phase'])
        self.assertEqual(
            'periodic_residual',
            pack['discovery_loop']['theories'][0]['theory_kind'],
        )

    def test_equation_metrics_expose_discovery_loop(self):
        kb = KnowledgeBase()
        workbench = EquationWorkbench()
        workbench.equations = {
            'raw_eq:residual_local_direction_vector': equation(
                key='raw_eq:residual_local_direction_vector',
                role='local_residual_direction_equation',
                parameters={'center_x': 7.0, 'center_y': 13.0},
            )
        }
        kb.equation_workbench = workbench

        metrics = _equation_metrics_from_knowledge(kb)

        self.assertEqual('probe_ready', metrics['discovery_loop']['phase'])
        self.assertTrue(metrics['discovery_loop']['concept_proposals'])

    def test_memory_compaction_builds_quantized_shards_and_keeps_anchors(self):
        memory = CumulativeTheoryMemory()
        for index in range(30):
            context = 'localized_gravity' if index % 2 else 'inverse_square_repulsion'
            memory.records.append({
                'context': context,
                'seed': index,
                'phase': 'probe_ready',
                'theory_count': 2 + index % 3,
                'operator_count': index % 4,
                'proof_check_count': index % 5,
                'disagreement_mode': 'distance_exponent_race',
            })
            operator_kind = (
                'localized_tapered_power'
                if index % 2
                else 'inverse_separation_power'
            )
            memory.operator_prior_outcomes.append({
                'context': context,
                'seed': index,
                'operator_key': f'operator:{operator_kind}:{index % 3}',
                'operator_kind': operator_kind,
                'outcome': 'confirmed' if index % 5 == 0 else 'unmatched',
                'best_score': 0.91 if index % 5 == 0 else 0.0,
                'matching_equation_count': 1 if index % 5 == 0 else 0,
                'parameters': {
                    'distance_exponent': 1.5 + 0.1 * (index % 4),
                    'cutoff_radius': 7.1 + 0.05 * (index % 5),
                    'relation': 'direction',
                    'source_context': context,
                },
            })

        before = memory.resource_efficiency_report(
            recommended_record_window=5,
            recommended_operator_window=8,
        )
        self.assertFalse(before['long_run_ready'])

        after = memory.compact_experience(
            keep_recent_records=5,
            keep_recent_operator_outcomes=6,
            max_operator_anchors=1,
        )

        self.assertEqual(5, len(memory.records))
        self.assertGreater(len(memory.operator_prior_outcomes), 6)
        self.assertEqual(1, len(memory.compressed_experience_shards))
        shard = memory.compressed_experience_shards[0]
        self.assertEqual(25, shard['source_counts']['records'])
        self.assertEqual(24, shard['source_counts']['operator_prior_outcomes'])
        self.assertTrue(shard['record_signatures'])
        self.assertTrue(shard['operator_prior_signatures'])
        self.assertGreater(after['detail_reduction_ratio'], 1.0)
        self.assertTrue(after['long_run_ready'])

        packed = memory.to_dict()
        self.assertIn('resource_efficiency', packed)
        restored = CumulativeTheoryMemory.from_dict(packed)
        self.assertEqual(
            shard['shard_id'],
            restored.compressed_experience_shards[0]['shard_id'],
        )
        self.assertIn('Resource efficiency:', restored.summary())

    def test_memory_efficiency_review_prints_and_can_compact(self):
        memory = CumulativeTheoryMemory()
        for index in range(10):
            memory.records.append({
                'context': 'time_varying',
                'seed': index,
                'phase': 'probe_ready',
                'theory_count': 2,
                'operator_count': 1,
                'proof_check_count': 1,
                'disagreement_mode': 'phase_frequency_race',
            })
            memory.operator_prior_outcomes.append({
                'context': 'time_varying',
                'seed': index,
                'operator_key': 'operator:phase:sinusoid',
                'operator_kind': 'sinusoidal_phase',
                'outcome': 'confirmed',
                'best_score': 0.88,
                'matching_equation_count': 1,
                'parameters': {
                    'relation': 'phase',
                    'source_context': 'time_varying',
                },
            })

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            report = run_memory_efficiency_review(
                theory_memory=memory,
                compact=True,
                keep_recent_records=3,
                keep_recent_operator_outcomes=3,
            )

        self.assertIn('MEMORY EFFICIENCY REVIEW', output.getvalue())
        self.assertEqual(1, report['compressed_shard_count'])
        self.assertEqual(3, len(memory.records))

    def test_equation_campaign_artifact_summary_survives_upload_blockers(self):
        memory = CumulativeTheoryMemory()
        starting = memory.memory_checkpoint_summary()
        result = {
            'context': 'repulsion',
            'seed': 0,
            'steps': 180,
            'equation_count': 3,
            'label_leaks': [],
            'interesting_score': 0.91,
            'interesting_equation': {
                'expression': 'k * unit_center_vector / separation^2',
            },
            'planned_experiment': {
                'experiment_kind': 'model_disagreement_domain_split',
            },
            'planned_experiment_outcome': {
                'outcome': 'model_disagreement_domain_split_supported',
            },
            'passed': True,
        }

        memory.record_equation_case_result(
            'repulsion',
            0,
            result,
            phase='equation_campaign',
        )
        summary = _equation_campaign_artifact_summary(
            [result],
            memory,
            starting_memory_summary=starting,
        )

        self.assertFalse(summary['runs_final'])
        self.assertEqual(1, summary['passed_count'])
        self.assertEqual(
            'model_disagreement_domain_split',
            summary['rows'][0]['planned_experiment_kind'],
        )
        self.assertEqual(
            'model_disagreement_domain_split_supported',
            summary['rows'][0]['planned_outcome'],
        )
        self.assertIn('readiness', summary)
        self.assertIn('memory_delta', summary)
        self.assertEqual(1, summary['memory_delta']['new_equation_cases'])
        self.assertIn('theorem_consolidations', summary)
        self.assertIn('domain_predicate_learning_agenda', summary)
        self.assertIn('planned_outcomes_tail', summary)

    def test_hf_artifact_upload_retries_with_create_pr_when_required(self):
        class FakeApi:
            def __init__(self):
                self.calls = []

            def upload_file(self, **kwargs):
                self.calls.append(dict(kwargs))
                if len(self.calls) == 1:
                    raise RuntimeError(
                        '403 Forbidden: pass create_pr=1 as a query parameter '
                        'to create a Pull Request.'
                    )
                return 'https://huggingface.test/pr/1'

        api = FakeApi()
        result = upload_hf_artifact_file(
            api,
            path_or_fileobj='tmp/summary.json',
            path_in_repo='runs/demo/summary.json',
            repo_id='demo/artifacts',
        )

        self.assertEqual('uploaded_via_pr', result['status'])
        self.assertEqual('create_pr_required', result['fallback_reason'])
        self.assertFalse(api.calls[0]['create_pr'])
        self.assertTrue(api.calls[1]['create_pr'])
        self.assertEqual('runs/demo/summary.json', api.calls[1]['path_in_repo'])

    def test_hf_artifact_upload_retries_generic_forbidden_with_create_pr(self):
        class FakeApi:
            def __init__(self):
                self.calls = []

            def upload_file(self, **kwargs):
                self.calls.append(dict(kwargs))
                if len(self.calls) == 1:
                    raise RuntimeError('403 Forbidden: Authorization error')
                return 'https://huggingface.test/pr/2'

        api = FakeApi()
        result = upload_hf_artifact_file(
            api,
            path_or_fileobj='tmp/summary.json',
            path_in_repo='runs/demo/summary.json',
            repo_id='demo/artifacts',
        )

        self.assertEqual('uploaded_via_pr', result['status'])
        self.assertEqual(
            'forbidden_retry_create_pr',
            result['fallback_reason'],
        )
        self.assertFalse(api.calls[0]['create_pr'])
        self.assertTrue(api.calls[1]['create_pr'])

    def test_hf_artifact_upload_does_not_swallow_unrelated_errors(self):
        class FakeApi:
            def upload_file(self, **kwargs):
                raise RuntimeError('network dropped before upload started')

        with self.assertRaisesRegex(RuntimeError, 'network dropped'):
            upload_hf_artifact_file(
                FakeApi(),
                path_or_fileobj='tmp/summary.json',
                path_in_repo='runs/demo/summary.json',
                repo_id='demo/artifacts',
            )

    def test_summary_only_compaction_makes_efficient_runs_long_ready(self):
        memory = CumulativeTheoryMemory()
        for index in range(4):
            memory.records.append({
                'context': 'localized_gravity',
                'seed': index,
                'phase': 'probe_ready',
                'theory_count': 2,
                'operator_count': 1,
                'proof_check_count': 1,
                'disagreement_mode': 'taper_shape_vs_hard_boundary',
            })
            memory.operator_prior_outcomes.append({
                'context': 'localized_gravity',
                'seed': index,
                'operator_key': 'operator:localized:taper',
                'operator_kind': 'localized_tapered_power',
                'outcome': 'weak',
                'best_score': 0.31,
                'matching_equation_count': 1,
                'parameters': {
                    'center_x': 8.0,
                    'center_y': 12.0,
                    'cutoff_radius': 6.0,
                    'distance_exponent': 1.0,
                    'relation': 'direction',
                },
            })

        before = memory.resource_efficiency_report(
            recommended_record_window=8,
            recommended_operator_window=8,
        )
        self.assertFalse(before['long_run_ready'])

        after = memory.compact_experience(
            keep_recent_records=8,
            keep_recent_operator_outcomes=8,
            force_summary=True,
        )

        self.assertEqual(4, len(memory.records))
        self.assertEqual(4, len(memory.operator_prior_outcomes))
        self.assertEqual(1, len(memory.compressed_experience_shards))
        self.assertTrue(memory.compressed_experience_shards[0]['summary_only'])
        self.assertTrue(after['long_run_ready'])

    def test_unmatched_operator_prior_creates_repair_claim_and_plan(self):
        memory = CumulativeTheoryMemory()
        operator_key = 'operator:memory_prior:localized_tapered_power:6:1:direction:demo'
        for context in ('standard', 'zero_gravity'):
            memory.operator_prior_outcomes.append({
                'context': context,
                'seed': 0,
                'operator_key': operator_key,
                'operator_kind': 'localized_tapered_power',
                'outcome': 'unmatched',
                'best_score': 0.0,
                'matching_equation_count': 0,
                'parameters': {
                    'center_x': 9.0,
                    'center_y': 11.0,
                    'cutoff_radius': 6.0,
                    'distance_exponent': 1.0,
                    'relation': 'direction',
                },
            })

        claim = memory.operator_prior_discovery_claims(limit=1)[0]
        self.assertEqual(operator_key, claim['operator_key'])
        self.assertEqual('needs_repair', claim['status'])
        self.assertIn('repair target', claim['accepted_because'][0])

        repair = memory.operator_prior_claim_experiments(limit=1)[0]
        self.assertEqual('operator_prior_domain_repair', repair['experiment_kind'])
        self.assertEqual('operator_prior_claim_needs_repair', repair['family_status'])
        self.assertEqual('operator_prior_failure_context', repair['target_context'])
        self.assertEqual(operator_key, repair['operator_prior_key'])

        readiness = memory.discovery_readiness_report()
        self.assertTrue(readiness['gates']['operator_discovery_claims']['passed'])
        self.assertTrue(readiness['gates']['claim_driven_planning']['passed'])
        self.assertTrue(readiness['gates']['anomaly_repair_loop']['passed'])

    def test_next_experiments_diversify_repeated_disagreement_modes(self):
        memory = CumulativeTheoryMemory()
        recommendations = []
        for index in range(4):
            recommendations.append({
                'experiment_kind': 'model_disagreement_probe',
                'theory_kind': 'distance_scaled_direction_residual',
                'priority': 0.96 - index * 0.01,
                'target_context': 'recorded_disagreement_context',
                'family_status': 'disagreement',
                'reason': 'resolve exponent race',
                'expected_result': 'one exponent should win',
                'falsifies_if': 'near/far ratio rejects the exponent',
                'proof_evidence': {'support_count': 3 - index},
                'disagreement_signature': {'mode': 'distance_exponent_race'},
            })
        recommendations.append({
            'experiment_kind': 'model_disagreement_probe',
            'theory_kind': 'tapered_distance_direction_residual',
            'priority': 0.91,
            'target_context': 'recorded_disagreement_context',
            'family_status': 'disagreement',
            'reason': 'resolve taper shape',
            'expected_result': 'boundary samples decide',
            'falsifies_if': 'boundary samples reject taper',
            'proof_evidence': {'support_count': 2},
            'disagreement_signature': {'mode': 'taper_shape_vs_hard_boundary'},
        })

        selected = memory._diversified_experiments(recommendations, limit=5)
        first_three_modes = [
            item['disagreement_signature']['mode']
            for item in selected[:3]
        ]

        self.assertEqual(
            2,
            first_three_modes.count('distance_exponent_race'),
        )
        self.assertIn('taper_shape_vs_hard_boundary', first_three_modes)

    def test_live_progress_viewer_parses_hf_and_scientist_events(self):
        line = 'HF_PROGRESS {"event":"finish","readiness_score":0.879}'
        parsed = parse_live_progress_line(line)
        self.assertEqual('hf_progress', parsed['stream'])
        self.assertEqual('finish', parsed['event'])

        summary_line = (
            'HF_ARTIFACT_SUMMARY {"run_kind":"equation_campaign",'
            '"runs_final":false,"result_count":2,"passed_count":2,'
            '"readiness":{"readiness_score":0.85,"status":"nearly_ready"},'
            '"memory_delta":{"new_equation_cases":2,"new_planned_outcomes":1}}'
        )
        parsed_summary = parse_live_progress_line(summary_line)
        self.assertEqual('hf_artifact_summary', parsed_summary['stream'])
        self.assertEqual('equation_campaign', parsed_summary['run_kind'])
        self.assertFalse(parsed_summary['runs_final'])

        with tempfile.NamedTemporaryFile('w+', encoding='utf-8') as handle:
            handle.write(
                'SCIENTIST_EVENT {"event":"invariant_consolidated",'
                '"relation_kind":"distance","status":"robust_law"}\n'
            )
            handle.write(
                'HF_PROGRESS {"event":"adaptive_comparison_finish",'
                '"recommendation":"keep_targeted_adaptive"}\n'
            )
            handle.write('HF_ARTIFACT {"run_id":"demo","report":"https://example.test/r"}\n')
            handle.write(summary_line + '\n')
            handle.flush()

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                summary = run_live_progress_viewer(progress_file=handle.name)

        self.assertEqual(1, summary['counts']['scientist_event'])
        self.assertEqual(1, summary['counts']['hf_progress'])
        self.assertEqual(1, summary['counts']['hf_artifact'])
        self.assertEqual(1, summary['counts']['hf_artifact_summary'])
        self.assertIn('LIVE DISCOVERY PROGRESS VIEW', output.getvalue())
        self.assertIn('HF_ARTIFACT_SUMMARY', output.getvalue())
        self.assertEqual('demo', summary['artifacts']['run_id'])
        self.assertEqual(2, summary['artifact_summary']['result_count'])


if __name__ == '__main__':
    unittest.main()
