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
    _equation_metrics_from_knowledge,
    _planned_probe_actions,
    _print_cumulative_theory_review,
    _run_equation_followup_cases,
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
        self.assertIn('Autonomous scientist loop:', memory.summary())

        empty_readiness = CumulativeTheoryMemory().discovery_readiness_report()
        self.assertEqual('early', empty_readiness['status'])
        self.assertFalse(
            empty_readiness['gates']['first_principles_adaptive_dimensions']['passed']
        )
        self.assertTrue(
            empty_readiness['gates']['algebraic_foundation_baseline']['passed']
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


if __name__ == '__main__':
    unittest.main()
