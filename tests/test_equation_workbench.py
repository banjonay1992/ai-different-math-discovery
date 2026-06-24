import contextlib
import io
import math
import os
import sys
import unittest


PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'first_principles_ai'))
sys.path.insert(0, PROJECT_DIR)

from agent.equation_workbench import EquationWorkbench, PrimitiveEquation
from agent.equation_tensor_backend import available_equation_scoring_backends
from agent.representation import KnowledgeBase
from main import _equation_metrics_from_knowledge, _interesting_result_rank, run_equation_campaign


def state(step, time, objects):
    return {
        'step': step,
        'time': time,
        'world_size': (20.0, 20.0),
        'objects': objects,
    }


def obj(object_id, x, y, vx, vy, mass=1.0, radius=0.5):
    return {
        'id': object_id,
        'position': (x, y),
        'velocity': (vx, vy),
        'mass': mass,
        'radius': radius,
    }


class EquationWorkbenchTests(unittest.TestCase):
    def test_records_direct_intervention_action_channels(self):
        workbench = EquationWorkbench(min_samples=2)
        for index, action_type in enumerate(('move', 'freeze', 'duplicate'), start=1):
            workbench.observe_transition(
                state(index, index * 0.016, [obj(1, 1.0, 1.0, 0.0, 0.0)]),
                state(index + 1, (index + 1) * 0.016, [obj(1, 1.0, 1.0, 0.0, 0.0)]),
                {'type': action_type},
                index,
            )

        self.assertEqual(1.0, workbench.aggregate_rows[0]['action_move'])
        self.assertEqual(1.0, workbench.aggregate_rows[1]['action_freeze'])
        self.assertEqual(1.0, workbench.aggregate_rows[2]['action_duplicate'])

    def test_discovers_count_and_position_equations_from_starter_kit(self):
        kb = KnowledgeBase()
        workbench = EquationWorkbench(
            kb,
            min_samples=4,
            install_score_threshold=0.8,
        )

        for step in range(8):
            before_objects = [obj(1, 1.0 + step, 2.0, 2.0, 0.0)]
            after_objects = [obj(1, 1.0 + step + 0.032, 2.0, 2.0, 0.0)]
            action = {'type': 'wait'}
            if step % 2 == 0:
                action = {'type': 'spawn'}
                after_objects.append(obj(100 + step, 5.0, 5.0, 0.0, 0.0))
            workbench.observe_transition(
                state(step, step * 0.016, before_objects),
                state(step + 1, (step + 1) * 0.016, after_objects),
                action,
                step + 1,
            )

        equations = workbench.discover(step=8)
        keys = {equation.key for equation in equations}
        rules = [
            rule for rule in kb.get_confirmed_rules()
            if rule.properties.get('source') == 'equation_workbench'
        ]

        self.assertIn('raw_eq:delta_count_from_action', keys)
        self.assertIn('raw_eq:next_x_from_velocity', keys)
        self.assertTrue(rules)
        self.assertFalse(workbench.label_leaks())

    def test_discovers_scaled_direction_vector_equation(self):
        workbench = EquationWorkbench(min_samples=8)
        for step in range(20):
            angle = (2 * math.pi * step) / 20
            x = 10.0 + math.cos(angle) * 4.0
            y = 10.0 + math.sin(angle) * 4.0
            dx = 10.0 - x
            dy = 10.0 - y
            dist = math.sqrt(dx * dx + dy * dy)
            dvx = 0.1 * dx / dist
            dvy = 0.1 * dy / dist
            workbench.observe_transition(
                state(step, step * 0.016, [obj(1, x, y, 0.0, 0.0)]),
                state(step + 1, (step + 1) * 0.016, [obj(1, x, y, dvx, dvy)]),
                {'type': 'wait'},
                step + 1,
            )

        equations = workbench.discover(step=20)
        radial = next(
            equation for equation in equations
            if equation.key == 'raw_eq:anchor_radial_delta_vector'
        )

        self.assertGreater(radial.score, 0.9)
        self.assertAlmostEqual(0.1, radial.parameters['k'], delta=0.01)

    def test_numpy_scoring_backend_matches_python_equations(self):
        if not available_equation_scoring_backends()['numpy']:
            self.skipTest('numpy equation scoring backend unavailable')

        python_workbench = EquationWorkbench(min_samples=8, scoring_backend='python')
        numpy_workbench = EquationWorkbench(min_samples=8, scoring_backend='numpy')
        for step in range(30):
            angle = (2 * math.pi * step) / 30
            x = 10.0 + math.cos(angle) * 4.0
            y = 10.0 + math.sin(angle) * 4.0
            dx = 10.0 - x
            dy = 10.0 - y
            dist = math.sqrt(dx * dx + dy * dy)
            dvx = 0.12 * dx / dist
            dvy = 0.12 * dy / dist
            before = state(step, step * 0.016, [obj(1, x, y, 0.0, 0.0)])
            after = state(step + 1, (step + 1) * 0.016, [obj(1, x, y, dvx, dvy)])
            python_workbench.observe_transition(before, after, {'type': 'wait'}, step + 1)
            numpy_workbench.observe_transition(before, after, {'type': 'wait'}, step + 1)

        python_equations = {
            equation.key: equation
            for equation in python_workbench.discover(step=30)
        }
        numpy_equations = {
            equation.key: equation
            for equation in numpy_workbench.discover(step=30)
        }

        self.assertEqual('numpy', numpy_workbench.scoring_backend)
        for key in (
            'raw_eq:anchor_radial_delta_vx',
            'raw_eq:anchor_radial_delta_vy',
            'raw_eq:anchor_radial_delta_vector',
        ):
            self.assertIn(key, python_equations)
            self.assertIn(key, numpy_equations)
            self.assertAlmostEqual(
                python_equations[key].score,
                numpy_equations[key].score,
                places=9,
            )
            self.assertAlmostEqual(
                python_equations[key].parameters['k'],
                numpy_equations[key].parameters['k'],
                places=9,
            )

    def test_review_pack_groups_foundations_and_interesting_dynamics(self):
        workbench = EquationWorkbench(min_samples=8)
        for step in range(20):
            angle = (2 * math.pi * step) / 20
            x = 10.0 + math.cos(angle) * 4.0
            y = 10.0 + math.sin(angle) * 4.0
            dx = 10.0 - x
            dy = 10.0 - y
            dist = math.sqrt(dx * dx + dy * dy)
            dvx = 0.1 * dx / dist
            dvy = 0.1 * dy / dist
            workbench.observe_transition(
                state(step, step * 0.016, [obj(1, x, y, 0.0, 0.0)]),
                state(step + 1, (step + 1) * 0.016, [obj(1, x, y, dvx, dvy)]),
                {'type': 'wait'},
                step + 1,
            )

        workbench.discover(step=20)
        pack = workbench.review_pack()

        self.assertIn('foundation_invariants', pack['categories'])
        self.assertIn('direction_vectors', pack['categories'])
        self.assertTrue(pack['interesting_equations'])
        self.assertNotEqual('invariant_equation', pack['interesting_equations'][0]['role'])

    def test_residual_dynamics_compete_with_easy_position_updates(self):
        kb = KnowledgeBase()
        workbench = EquationWorkbench(
            kb,
            min_samples=12,
            install_score_threshold=0.5,
        )
        dt = 0.016
        for step in range(96):
            angle = (2 * math.pi * step) / 96
            radius = 4.0 + (step % 4) * 0.25
            x = 10.0 + math.cos(angle) * radius
            y = 10.0 + math.sin(angle) * radius
            vx = ((step % 9) - 4) * 0.05
            vy = ((step % 7) - 3) * 0.04
            dx = x - 10.0
            dy = y - 10.0
            dist = math.sqrt(dx * dx + dy * dy)
            baseline_x = 0.04 - 0.02 * vx
            baseline_y = -0.03 - 0.015 * vy
            residual_x = -0.16 * dy / dist
            residual_y = 0.16 * dx / dist
            dvx = baseline_x + residual_x
            dvy = baseline_y + residual_y

            workbench.observe_transition(
                state(step, step * dt, [obj(1, x, y, vx, vy)]),
                state(
                    step + 1,
                    (step + 1) * dt,
                    [obj(1, x + vx * dt, y + vy * dt, vx + dvx, vy + dvy)],
                ),
                {'type': 'wait'},
                step + 1,
            )

        equations = workbench.discover(step=96)
        residuals = [
            equation for equation in equations
            if equation.role in {
                'residual_perpendicular_equation',
                'local_residual_perpendicular_equation',
            }
        ]
        pack = workbench.review_pack()
        installed_roles = {
            rule.properties.get('role')
            for rule in kb.get_confirmed_rules()
            if rule.properties.get('source') == 'equation_workbench'
        }

        self.assertTrue(residuals)
        self.assertGreater(max(equation.score for equation in residuals), 0.5)
        self.assertTrue(any(
            equation.parameters.get('directional_alignment', 0.0) > 0.5
            for equation in residuals
        ))
        self.assertIn(
            pack['interesting_equations'][0]['role'],
            {
                'residual_perpendicular_equation',
                'local_residual_perpendicular_equation',
                'generated_operator_periodic_equation',
            },
        )
        self.assertIn('residual_dynamics', pack['categories'])
        self.assertTrue(installed_roles & {
            'residual_perpendicular_equation',
            'local_residual_perpendicular_equation',
        })
        self.assertFalse(workbench.label_leaks())

    def test_temporal_residual_equation_surfaces_repeating_template(self):
        workbench = EquationWorkbench(min_samples=16)
        dt = 0.016
        period = 80
        for step in range(180):
            x = 4.0 + (step % 9) * 0.2
            y = 6.0 + (step % 7) * 0.15
            vx = ((step % 11) - 5) * 0.03
            vy = ((step % 5) - 2) * 0.02
            dvx = 0.02 - 0.01 * vx + 0.18 * math.sin((2 * math.pi * step) / period)
            dvy = -0.01 - 0.01 * vy

            workbench.observe_transition(
                state(step, step * dt, [obj(1, x, y, vx, vy)]),
                state(
                    step + 1,
                    (step + 1) * dt,
                    [obj(1, x + vx * dt, y + vy * dt, vx + dvx, vy + dvy)],
                ),
                {'type': 'wait'},
                step + 1,
            )

        equations = workbench.discover(step=180)
        periodic = [
            equation for equation in equations
            if equation.role == 'residual_periodic_equation'
        ]
        pack = workbench.review_pack()
        action = workbench.suggest_probe_action(
            current_count=12,
            world_width=20.0,
            world_height=20.0,
            step=220,
        )

        self.assertTrue(periodic)
        best = max(periodic, key=lambda equation: equation.score)
        self.assertGreater(best.score, 0.7)
        self.assertAlmostEqual(period, best.parameters['period_steps'], delta=8)
        self.assertIn(pack['interesting_equations'][0]['role'], {
            'residual_periodic_equation',
            'generated_operator_periodic_equation',
        })
        self.assertEqual('wait', action['type'])
        self.assertEqual('equation_workbench_probe', action['source'])

    def test_local_residual_center_search_surfaces_off_center_structure(self):
        workbench = EquationWorkbench(min_samples=16)
        dt = 0.016
        center_x = 7.0
        center_y = 13.0
        for step in range(120):
            angle = (2 * math.pi * step) / 120
            near_x = center_x + math.cos(angle) * 3.0
            near_y = center_y + math.sin(angle) * 3.0
            near_vx = ((step % 7) - 3) * 0.04
            near_vy = ((step % 5) - 2) * 0.04
            dx = center_x - near_x
            dy = center_y - near_y
            dist = math.sqrt(dx * dx + dy * dy)
            near_dvx = 0.02 - 0.01 * near_vx + 0.2 * dx / dist
            near_dvy = -0.03 - 0.01 * near_vy + 0.2 * dy / dist

            far_x = 16.0 + (step % 4) * 0.1
            far_y = 4.0 + (step % 6) * 0.1
            far_vx = ((step % 3) - 1) * 0.03
            far_vy = ((step % 4) - 2) * 0.03
            far_dvx = 0.02 - 0.01 * far_vx
            far_dvy = -0.03 - 0.01 * far_vy

            workbench.observe_transition(
                state(step, step * dt, [
                    obj(1, near_x, near_y, near_vx, near_vy),
                    obj(2, far_x, far_y, far_vx, far_vy),
                ]),
                state(
                    step + 1,
                    (step + 1) * dt,
                    [
                        obj(1, near_x + near_vx * dt, near_y + near_vy * dt, near_vx + near_dvx, near_vy + near_dvy),
                        obj(2, far_x + far_vx * dt, far_y + far_vy * dt, far_vx + far_dvx, far_vy + far_dvy),
                    ],
                ),
                {'type': 'wait'},
                step + 1,
            )

        equations = workbench.discover(step=120)
        local = [
            equation for equation in equations
            if equation.role == 'local_residual_direction_equation'
        ]
        pack = workbench.review_pack()
        action = workbench.suggest_probe_action(
            current_count=2,
            world_width=20.0,
            world_height=20.0,
            step=150,
        )

        self.assertTrue(local)
        best = max(local, key=lambda equation: equation.score)
        self.assertGreater(best.score, 0.6)
        self.assertAlmostEqual(center_x, best.parameters['center_x'], delta=0.5)
        self.assertAlmostEqual(center_y, best.parameters['center_y'], delta=0.5)
        self.assertIn(pack['interesting_equations'][0]['role'], {
            'local_residual_direction_equation',
            'local_residual_distance_scaled_direction_equation',
            'generated_operator_tapered_distance_direction_equation',
        })
        self.assertEqual('spawn', action['type'])
        self.assertAlmostEqual(center_y, action['y'], delta=0.5)
        self.assertFalse(workbench.label_leaks())

    def test_distance_scaled_residual_surfaces_hidden_strength_law(self):
        workbench = EquationWorkbench(min_samples=16)
        dt = 0.016
        center_x = 9.0
        center_y = 11.0
        for step in range(180):
            angle = (2 * math.pi * step) / 45
            radius = 2.0 + (step % 9) * 0.65
            x = center_x + math.cos(angle) * radius
            y = center_y + math.sin(angle) * radius
            vx = ((step % 7) - 3) * 0.03
            vy = ((step % 5) - 2) * 0.03
            dx = center_x - x
            dy = center_y - y
            dist = math.sqrt(dx * dx + dy * dy)
            strength = 2.4 / (dist * dist)
            dvx = 0.01 - 0.01 * vx + strength * dx / dist
            dvy = -0.02 - 0.01 * vy + strength * dy / dist

            workbench.observe_transition(
                state(step, step * dt, [obj(1, x, y, vx, vy)]),
                state(
                    step + 1,
                    (step + 1) * dt,
                    [obj(1, x + vx * dt, y + vy * dt, vx + dvx, vy + dvy)],
                ),
                {'type': 'wait'},
                step + 1,
            )

        equations = workbench.discover(step=180)
        distance_scaled = [
            equation for equation in equations
            if equation.role in {
                'residual_distance_scaled_direction_equation',
                'local_residual_distance_scaled_direction_equation',
            }
        ]
        pack = workbench.review_pack()

        self.assertTrue(distance_scaled)
        best = max(distance_scaled, key=lambda equation: equation.score)
        self.assertGreater(best.score, 0.7)
        self.assertAlmostEqual(2.0, best.parameters['distance_exponent'], delta=0.01)
        self.assertGreater(best.parameters['distance_mse_improvement'], 0.1)
        self.assertIn('distance_scaled', pack['interesting_equations'][0]['role'])
        self.assertIn('residual_strength', pack['categories'])
        self.assertFalse(workbench.label_leaks())

    def test_distance_scaled_residual_requires_strength_improvement(self):
        workbench = EquationWorkbench(min_samples=16)
        dt = 0.016
        center_x = 10.0
        center_y = 10.0
        for step in range(120):
            angle = (2 * math.pi * step) / 40
            radius = 2.0 + (step % 8) * 0.7
            x = center_x + math.cos(angle) * radius
            y = center_y + math.sin(angle) * radius
            vx = ((step % 5) - 2) * 0.02
            vy = ((step % 7) - 3) * 0.02
            dx = center_x - x
            dy = center_y - y
            dist = math.sqrt(dx * dx + dy * dy)
            dvx = 0.02 - 0.01 * vx + 0.18 * dx / dist
            dvy = -0.01 - 0.01 * vy + 0.18 * dy / dist

            workbench.observe_transition(
                state(step, step * dt, [obj(1, x, y, vx, vy)]),
                state(
                    step + 1,
                    (step + 1) * dt,
                    [obj(1, x + vx * dt, y + vy * dt, vx + dvx, vy + dvy)],
                ),
                {'type': 'wait'},
                step + 1,
            )

        equations = workbench.discover(step=120)
        distance_scaled = [
            equation for equation in equations
            if 'distance_scaled' in equation.role
        ]

        self.assertFalse(distance_scaled)

    def test_discover_refreshes_generated_operator_bank_from_theories(self):
        workbench = EquationWorkbench(min_samples=16)
        dt = 0.016
        center_x = 9.0
        center_y = 11.0
        for step in range(160):
            angle = (2 * math.pi * step) / 40
            radius = 2.0 + (step % 8) * 0.7
            x = center_x + math.cos(angle) * radius
            y = center_y + math.sin(angle) * radius
            vx = ((step % 7) - 3) * 0.03
            vy = ((step % 5) - 2) * 0.03
            dx = center_x - x
            dy = center_y - y
            dist = math.sqrt(dx * dx + dy * dy)
            strength = 2.2 / (dist * dist)
            dvx = 0.01 - 0.01 * vx + strength * dx / dist
            dvy = -0.02 - 0.01 * vy + strength * dy / dist

            workbench.observe_transition(
                state(step, step * dt, [obj(1, x, y, vx, vy)]),
                state(
                    step + 1,
                    (step + 1) * dt,
                    [obj(1, x + vx * dt, y + vy * dt, vx + dvx, vy + dvy)],
                ),
                {'type': 'wait'},
                step + 1,
            )

        workbench.discover(step=160)

        self.assertTrue(any(
            operator['operator_kind'] == 'inverse_separation_power'
            for operator in workbench.generated_operator_bank.values()
        ))
        self.assertTrue(any(
            equation.role == 'generated_operator_distance_scaled_direction_equation'
            for equation in workbench.discovered_equations()
        ))

    def test_generated_operator_feedback_scores_refined_distance_exponent(self):
        workbench = EquationWorkbench(min_samples=16)
        center_x = 8.0
        center_y = 12.0
        workbench.generated_operator_bank = {
            'operator:inverse_separation_power:2.25:direction': {
                'key': 'operator:inverse_separation_power:2.25:direction',
                'operator_kind': 'inverse_separation_power',
                'inputs': ['center', 'position', 'distance_exponent'],
                'expression': 'unit(center - position) / separation^2.25',
                'generated_from': 'concept:distance_strength_exponent:2.25',
                'usefulness': 0.9,
                'test_hint': 'compare near and far residual magnitudes',
                'parameters': {
                    'center_x': center_x,
                    'center_y': center_y,
                    'distance_exponent': 2.25,
                    'relation': 'direction',
                },
            }
        }
        dt = 0.016
        for step in range(180):
            angle = (2 * math.pi * step) / 45
            radius = 2.0 + (step % 9) * 0.65
            x = center_x + math.cos(angle) * radius
            y = center_y + math.sin(angle) * radius
            vx = ((step % 7) - 3) * 0.03
            vy = ((step % 5) - 2) * 0.03
            dx = center_x - x
            dy = center_y - y
            dist = math.sqrt(dx * dx + dy * dy)
            strength = 2.6 / (dist ** 2.25)
            dvx = 0.01 - 0.01 * vx + strength * dx / dist
            dvy = -0.02 - 0.01 * vy + strength * dy / dist

            workbench.observe_transition(
                state(step, step * dt, [obj(1, x, y, vx, vy)]),
                state(
                    step + 1,
                    (step + 1) * dt,
                    [obj(1, x + vx * dt, y + vy * dt, vx + dvx, vy + dvy)],
                ),
                {'type': 'wait'},
                step + 1,
            )

        equations = workbench.discover(step=180)
        generated = [
            equation for equation in equations
            if equation.role == 'generated_operator_distance_scaled_direction_equation'
        ]

        self.assertTrue(generated)
        best = max(generated, key=lambda equation: equation.score)
        self.assertGreater(best.score, 0.7)
        self.assertAlmostEqual(2.25, best.parameters['distance_exponent'], delta=0.01)
        self.assertEqual(
            'operator:inverse_separation_power:2.25:direction',
            best.parameters['operator_key'],
        )

    def test_constructor_operator_priors_drive_generated_feedback(self):
        center_x = 8.0
        center_y = 12.0
        workbench = EquationWorkbench(
            min_samples=16,
            generated_operator_priors=[{
                'key': 'operator:memory_prior:inverse_separation_power:2.0:direction:test',
                'operator_kind': 'inverse_separation_power',
                'inputs': ['center', 'position', 'distance_exponent'],
                'expression': 'unit(center - position) / separation^2.0',
                'generated_from': 'representation:separation_exponent_from_log_ratio',
                'usefulness': 0.86,
                'test_hint': 'score remembered exponent on held-out residuals',
                'parameters': {
                    'center_x': center_x,
                    'center_y': center_y,
                    'distance_exponent': 2.0,
                    'relation': 'direction',
                },
            }],
        )
        dt = 0.016
        for step in range(180):
            angle = (2 * math.pi * step) / 45
            radius = 2.0 + (step % 9) * 0.65
            x = center_x + math.cos(angle) * radius
            y = center_y + math.sin(angle) * radius
            vx = ((step % 7) - 3) * 0.03
            vy = ((step % 5) - 2) * 0.03
            dx = center_x - x
            dy = center_y - y
            dist = math.sqrt(dx * dx + dy * dy)
            strength = 2.6 / (dist ** 2.25)
            dvx = 0.01 - 0.01 * vx + strength * dx / dist
            dvy = -0.02 - 0.01 * vy + strength * dy / dist

            workbench.observe_transition(
                state(step, step * dt, [obj(1, x, y, vx, vy)]),
                state(
                    step + 1,
                    (step + 1) * dt,
                    [obj(1, x + vx * dt, y + vy * dt, vx + dvx, vy + dvy)],
                ),
                {'type': 'wait'},
                step + 1,
            )

        equations = workbench.discover(step=180)
        generated = [
            equation for equation in equations
            if equation.role == 'generated_operator_distance_scaled_direction_equation'
        ]

        self.assertEqual(1, workbench.generated_operator_prior_count)
        self.assertTrue(generated)
        best = max(generated, key=lambda equation: equation.score)
        self.assertGreater(best.score, 0.7)
        self.assertIn('operator:memory_prior:', best.parameters['operator_key'])
        pack = workbench.review_pack()
        self.assertEqual(1, len(pack['operator_prior_results']))
        self.assertEqual('confirmed', pack['operator_prior_results'][0]['outcome'])
        self.assertGreater(pack['operator_prior_results'][0]['best_score'], 0.7)
        self.assertAlmostEqual(
            2.25,
            pack['operator_prior_results'][0]['best_equation']['parameters']['distance_exponent'],
            delta=0.01,
        )

    def test_generated_cutoff_operator_scores_localized_residual_region(self):
        workbench = EquationWorkbench(min_samples=16)
        center_x = 8.0
        center_y = 12.0
        cutoff_radius = 4.0
        workbench.generated_operator_bank = {
            'operator:localized_cutoff_window:local:direction': {
                'key': 'operator:localized_cutoff_window:local:direction',
                'operator_kind': 'localized_cutoff_window',
                'inputs': ['center', 'position', 'cutoff_radius'],
                'expression': 'inside(separation <= 4.0) * unit(center - position)',
                'generated_from': 'concept:local_high_change_region',
                'usefulness': 0.9,
                'test_hint': 'compare inside and outside the inferred radius',
                'parameters': {
                    'center_x': center_x,
                    'center_y': center_y,
                    'cutoff_radius': cutoff_radius,
                    'relation': 'direction',
                },
            }
        }
        dt = 0.016
        for step in range(200):
            angle = (2 * math.pi * step) / 40
            if step % 2 == 0:
                radius = 2.0 + ((step // 2) % 4) * 0.55
            else:
                radius = 4.4 + ((step // 2) % 4) * 0.7
            x = center_x + math.cos(angle) * radius
            y = center_y + math.sin(angle) * radius
            vx = ((step % 7) - 3) * 0.025
            vy = ((step % 5) - 2) * 0.025
            dx = center_x - x
            dy = center_y - y
            dist = math.sqrt(dx * dx + dy * dy)
            residual = 0.26 if dist <= cutoff_radius else 0.0
            dvx = 0.01 - 0.012 * vx + residual * dx / dist
            dvy = -0.015 - 0.012 * vy + residual * dy / dist

            workbench.observe_transition(
                state(step, step * dt, [obj(1, x, y, vx, vy)]),
                state(
                    step + 1,
                    (step + 1) * dt,
                    [obj(1, x + vx * dt, y + vy * dt, vx + dvx, vy + dvy)],
                ),
                {'type': 'wait'},
                step + 1,
            )

        equations = workbench.discover(step=200)
        generated = [
            equation for equation in equations
            if equation.role == 'generated_operator_cutoff_direction_equation'
        ]

        self.assertTrue(generated)
        best = max(generated, key=lambda equation: equation.score)
        self.assertGreater(best.score, 0.7)
        self.assertGreater(best.parameters['cutoff_mse_improvement'], 0.25)
        self.assertGreater(best.parameters['inside_valid_count'], 3)
        self.assertGreater(best.parameters['outside_valid_count'], 3)
        self.assertLess(best.parameters['outside_residual_fraction'], 0.16)
        self.assertLess(best.parameters['inside_projection_cv'], 0.35)
        self.assertGreaterEqual(best.parameters['cutoff_radius'], 2.0)
        self.assertLess(best.parameters['cutoff_radius'], 4.5)

    def test_generated_cutoff_operator_requires_near_far_improvement(self):
        workbench = EquationWorkbench(min_samples=16)
        center_x = 8.0
        center_y = 12.0
        workbench.generated_operator_bank = {
            'operator:localized_cutoff_window:local:direction': {
                'key': 'operator:localized_cutoff_window:local:direction',
                'operator_kind': 'localized_cutoff_window',
                'inputs': ['center', 'position', 'cutoff_radius'],
                'expression': 'inside(separation <= 4.0) * unit(center - position)',
                'generated_from': 'concept:local_high_change_region',
                'usefulness': 0.9,
                'test_hint': 'compare inside and outside the inferred radius',
                'parameters': {
                    'center_x': center_x,
                    'center_y': center_y,
                    'cutoff_radius': 4.0,
                    'relation': 'direction',
                },
            }
        }
        dt = 0.016
        for step in range(200):
            angle = (2 * math.pi * step) / 40
            radius = 2.2 + (step % 10) * 0.55
            x = center_x + math.cos(angle) * radius
            y = center_y + math.sin(angle) * radius
            vx = ((step % 7) - 3) * 0.025
            vy = ((step % 5) - 2) * 0.025
            dx = center_x - x
            dy = center_y - y
            dist = math.sqrt(dx * dx + dy * dy)
            dvx = 0.01 - 0.012 * vx + 0.22 * dx / dist
            dvy = -0.015 - 0.012 * vy + 0.22 * dy / dist

            workbench.observe_transition(
                state(step, step * dt, [obj(1, x, y, vx, vy)]),
                state(
                    step + 1,
                    (step + 1) * dt,
                    [obj(1, x + vx * dt, y + vy * dt, vx + dvx, vy + dvy)],
                ),
                {'type': 'wait'},
                step + 1,
            )

        equations = workbench.discover(step=200)
        generated = [
            equation for equation in equations
            if equation.role == 'generated_operator_cutoff_direction_equation'
        ]

        self.assertFalse(generated)

    def test_generated_cutoff_operator_uses_balanced_local_contrast(self):
        workbench = EquationWorkbench(min_samples=16)
        center_x = 8.0
        center_y = 12.0
        cutoff_radius = 4.0
        workbench.generated_operator_bank = {
            'operator:localized_cutoff_window:noisy:direction': {
                'key': 'operator:localized_cutoff_window:noisy:direction',
                'operator_kind': 'localized_cutoff_window',
                'inputs': ['center', 'position', 'cutoff_radius'],
                'expression': 'inside(separation <= 4.0) * unit(center - position)',
                'generated_from': 'concept:local_high_change_region',
                'usefulness': 0.9,
                'test_hint': 'compare local contrast against far-field clutter',
                'parameters': {
                    'center_x': center_x,
                    'center_y': center_y,
                    'cutoff_radius': cutoff_radius,
                    'relation': 'direction',
                },
            }
        }
        dt = 0.016
        for step in range(260):
            angle = (2 * math.pi * step) / 52
            if step % 4 == 0:
                radius = 2.3 + ((step // 4) % 3) * 0.55
            elif step % 4 == 1:
                radius = 4.45 + ((step // 4) % 3) * 0.45
            else:
                radius = 8.5 + (step % 6) * 0.8
            x = center_x + math.cos(angle) * radius
            y = center_y + math.sin(angle) * radius
            vx = ((step % 7) - 3) * 0.025
            vy = ((step % 5) - 2) * 0.025
            dx = center_x - x
            dy = center_y - y
            dist = math.sqrt(dx * dx + dy * dy)
            residual = 0.32 if dist <= cutoff_radius else 0.0
            clutter = 0.02 * math.sin(step * 0.7)
            dvx = 0.01 - 0.012 * vx + residual * dx / dist + clutter
            dvy = -0.015 - 0.012 * vy + residual * dy / dist - clutter * 0.5

            workbench.observe_transition(
                state(step, step * dt, [obj(1, x, y, vx, vy)]),
                state(
                    step + 1,
                    (step + 1) * dt,
                    [obj(1, x + vx * dt, y + vy * dt, vx + dvx, vy + dvy)],
                ),
                {'type': 'wait'},
                step + 1,
            )

        equations = workbench.discover(step=260)
        generated = [
            equation for equation in equations
            if equation.role == 'generated_operator_cutoff_direction_equation'
        ]

        self.assertTrue(generated)
        best = max(generated, key=lambda equation: equation.score)
        self.assertGreater(best.parameters['cutoff_mse_improvement'], 0.04)
        self.assertGreater(best.parameters['mse_improvement'], 0.08)
        self.assertLess(best.parameters['outside_residual_fraction'], 0.16)
        self.assertLess(best.parameters['inside_projection_cv'], 0.35)
        self.assertLess(best.sample_count, len(workbench.object_rows))

    def test_generated_cutoff_operator_refines_offset_center(self):
        workbench = EquationWorkbench(min_samples=16)
        true_center_x = 8.0
        true_center_y = 12.0
        cutoff_radius = 4.0
        workbench.generated_operator_bank = {
            'operator:localized_cutoff_window:offset:direction': {
                'key': 'operator:localized_cutoff_window:offset:direction',
                'operator_kind': 'localized_cutoff_window',
                'inputs': ['center', 'position', 'cutoff_radius'],
                'expression': 'inside(separation <= 4.0) * unit(center - position)',
                'generated_from': 'concept:local_high_change_region',
                'usefulness': 0.9,
                'test_hint': 'refine a noisy inferred center',
                'parameters': {
                    'center_x': true_center_x,
                    'center_y': true_center_y - 2.25,
                    'cutoff_radius': cutoff_radius,
                    'relation': 'direction',
                },
            }
        }
        dt = 0.016
        for step in range(220):
            angle = (2 * math.pi * step) / 44
            radius = [2.1, 2.8, 3.5, 4.5, 5.2, 6.0][step % 6]
            x = true_center_x + math.cos(angle) * radius
            y = true_center_y + math.sin(angle) * radius
            vx = ((step % 7) - 3) * 0.025
            vy = ((step % 5) - 2) * 0.025
            dx = true_center_x - x
            dy = true_center_y - y
            dist = math.sqrt(dx * dx + dy * dy)
            residual = 0.3 if dist <= cutoff_radius else 0.0
            dvx = 0.01 - 0.012 * vx + residual * dx / dist
            dvy = -0.015 - 0.012 * vy + residual * dy / dist

            workbench.observe_transition(
                state(step, step * dt, [obj(1, x, y, vx, vy)]),
                state(
                    step + 1,
                    (step + 1) * dt,
                    [obj(1, x + vx * dt, y + vy * dt, vx + dvx, vy + dvy)],
                ),
                {'type': 'wait'},
                step + 1,
            )

        equations = workbench.discover(step=220)
        generated = [
            equation for equation in equations
            if equation.role == 'generated_operator_cutoff_direction_equation'
        ]

        self.assertTrue(generated)
        best = max(generated, key=lambda equation: equation.score)
        self.assertAlmostEqual(true_center_x, best.parameters['center_x'], delta=0.2)
        self.assertAlmostEqual(true_center_y, best.parameters['center_y'], delta=0.2)

    def test_generated_cutoff_operator_rejects_smooth_distance_law(self):
        workbench = EquationWorkbench(min_samples=16)
        center_x = 8.0
        center_y = 12.0
        workbench.generated_operator_bank = {
            'operator:localized_cutoff_window:local:direction': {
                'key': 'operator:localized_cutoff_window:local:direction',
                'operator_kind': 'localized_cutoff_window',
                'inputs': ['center', 'position', 'cutoff_radius'],
                'expression': 'inside(separation <= 4.0) * unit(center - position)',
                'generated_from': 'concept:local_high_change_region',
                'usefulness': 0.9,
                'test_hint': 'compare inside and outside the inferred radius',
                'parameters': {
                    'center_x': center_x,
                    'center_y': center_y,
                    'cutoff_radius': 4.0,
                    'relation': 'direction',
                },
            }
        }
        dt = 0.016
        for step in range(200):
            angle = (2 * math.pi * step) / 40
            radius = 2.2 + (step % 10) * 0.55
            x = center_x + math.cos(angle) * radius
            y = center_y + math.sin(angle) * radius
            vx = ((step % 7) - 3) * 0.025
            vy = ((step % 5) - 2) * 0.025
            dx = center_x - x
            dy = center_y - y
            dist = math.sqrt(dx * dx + dy * dy)
            strength = 2.0 / (dist * dist)
            dvx = 0.01 - 0.012 * vx + strength * dx / dist
            dvy = -0.015 - 0.012 * vy + strength * dy / dist

            workbench.observe_transition(
                state(step, step * dt, [obj(1, x, y, vx, vy)]),
                state(
                    step + 1,
                    (step + 1) * dt,
                    [obj(1, x + vx * dt, y + vy * dt, vx + dvx, vy + dvy)],
                ),
                {'type': 'wait'},
                step + 1,
            )

        equations = workbench.discover(step=200)
        generated = [
            equation for equation in equations
            if equation.role == 'generated_operator_cutoff_direction_equation'
        ]

        self.assertFalse(generated)

    def test_generated_tapered_operator_scores_local_shape(self):
        workbench = EquationWorkbench(min_samples=16)
        center_x = 8.0
        center_y = 12.0
        cutoff_radius = 4.0
        workbench.generated_operator_bank = {
            'operator:localized_tapered_power:local:direction': {
                'key': 'operator:localized_tapered_power:local:direction',
                'operator_kind': 'localized_tapered_power',
                'inputs': ['center', 'position', 'cutoff_radius', 'distance_exponent'],
                'expression': 'inside(separation <= 4.0) * taper * unit(center - position) / separation^2',
                'generated_from': 'concept:local_high_change_region',
                'usefulness': 0.9,
                'test_hint': 'compare center, middle, boundary, and outside samples',
                'parameters': {
                    'center_x': center_x,
                    'center_y': center_y,
                    'cutoff_radius': cutoff_radius,
                    'distance_exponent': 2.0,
                    'relation': 'direction',
                },
            }
        }
        dt = 0.016
        for step in range(240):
            angle = (2 * math.pi * step) / 48
            radius_options = [0.9, 1.4, 2.1, 2.8, 3.45, 4.35, 5.1, 5.8]
            radius = radius_options[step % len(radius_options)]
            x = center_x + math.cos(angle) * radius
            y = center_y + math.sin(angle) * radius
            vx = ((step % 7) - 3) * 0.02
            vy = ((step % 5) - 2) * 0.02
            dx = center_x - x
            dy = center_y - y
            dist = math.sqrt(dx * dx + dy * dy)
            taper = max(0.0, 1.0 - dist / cutoff_radius)
            residual = 2.4 * taper / max(dist * dist, 0.25)
            dvx = 0.01 - 0.012 * vx + residual * dx / dist
            dvy = -0.015 - 0.012 * vy + residual * dy / dist

            workbench.observe_transition(
                state(step, step * dt, [obj(1, x, y, vx, vy)]),
                state(
                    step + 1,
                    (step + 1) * dt,
                    [obj(1, x + vx * dt, y + vy * dt, vx + dvx, vy + dvy)],
                ),
                {'type': 'wait'},
                step + 1,
            )

        equations = workbench.discover(step=240)
        generated = [
            equation for equation in equations
            if equation.role == 'generated_operator_tapered_distance_direction_equation'
        ]

        self.assertTrue(generated)
        best = max(generated, key=lambda equation: equation.score)
        self.assertGreater(best.score, 0.7)
        self.assertGreaterEqual(best.parameters['distance_exponent'], 0.25)
        self.assertLessEqual(best.parameters['distance_exponent'], 2.25)
        self.assertGreaterEqual(best.parameters['cutoff_radius'], 2.0)
        self.assertLess(best.parameters['cutoff_radius'], 4.5)
        self.assertGreater(best.parameters['tapered_vs_smooth_improvement'], 0.05)
        self.assertGreater(best.parameters['tapered_vs_cutoff_improvement'], 0.1)

    def test_operator_feedback_bounds_large_row_sets_before_taper_scoring(self):
        workbench = EquationWorkbench(
            min_samples=16,
            max_operator_feedback_rows=64,
        )
        center_x = 8.0
        center_y = 12.0
        cutoff_radius = 4.0
        workbench.generated_operator_bank = {
            'operator:localized_tapered_power:local:direction': {
                'key': 'operator:localized_tapered_power:local:direction',
                'operator_kind': 'localized_tapered_power',
                'inputs': ['center', 'position', 'cutoff_radius', 'distance_exponent'],
                'expression': 'inside(separation <= 4.0) * taper * unit(center - position) / separation^2',
                'generated_from': 'concept:local_high_change_region',
                'usefulness': 0.9,
                'test_hint': 'compare center, middle, boundary, and outside samples',
                'parameters': {
                    'center_x': center_x,
                    'center_y': center_y,
                    'cutoff_radius': cutoff_radius,
                    'distance_exponent': 2.0,
                    'relation': 'direction',
                },
            }
        }
        dt = 0.016
        for step in range(240):
            angle = (2 * math.pi * step) / 48
            radius_options = [0.9, 1.4, 2.1, 2.8, 3.45, 4.35, 5.1, 5.8]
            radius = radius_options[step % len(radius_options)]
            x = center_x + math.cos(angle) * radius
            y = center_y + math.sin(angle) * radius
            vx = ((step % 7) - 3) * 0.02
            vy = ((step % 5) - 2) * 0.02
            dx = center_x - x
            dy = center_y - y
            dist = math.sqrt(dx * dx + dy * dy)
            taper = max(0.0, 1.0 - dist / cutoff_radius)
            residual = 2.4 * taper / max(dist * dist, 0.25)
            dvx = 0.01 - 0.012 * vx + residual * dx / dist
            dvy = -0.015 - 0.012 * vy + residual * dy / dist

            workbench.observe_transition(
                state(step, step * dt, [obj(1, x, y, vx, vy)]),
                state(
                    step + 1,
                    (step + 1) * dt,
                    [obj(1, x + vx * dt, y + vy * dt, vx + dvx, vy + dvy)],
                ),
                {'type': 'wait'},
                step + 1,
            )

        seen_row_counts = []

        def fake_taper_score(rows, **_kwargs):
            seen_row_counts.append(len(rows))
            return None

        workbench._score_generated_operator_tapered_distance_vector_scale = fake_taper_score

        equations = workbench._operator_feedback_equations(workbench.object_rows)

        self.assertEqual([], equations)
        self.assertTrue(seen_row_counts)
        self.assertLessEqual(max(seen_row_counts), 64)

    def test_operator_feedback_scores_highest_usefulness_operators_first(self):
        workbench = EquationWorkbench(max_operator_feedback_operators=2)
        workbench.generated_operator_bank = {
            'operator:low': {
                'key': 'operator:low',
                'operator_kind': 'inverse_separation_power',
                'usefulness': 0.1,
            },
            'operator:top': {
                'key': 'operator:top',
                'operator_kind': 'localized_tapered_power',
                'usefulness': 0.9,
            },
            'operator:middle': {
                'key': 'operator:middle',
                'operator_kind': 'localized_cutoff_window',
                'usefulness': 0.5,
            },
            'operator:ignored': {
                'key': 'operator:ignored',
                'operator_kind': 'center_from_residual_lines',
                'usefulness': 1.0,
            },
        }

        ranked = workbench._ranked_operator_feedback_items()

        self.assertEqual(['operator:top', 'operator:middle'], [
            item['key'] for item in ranked
        ])

    def test_generated_tapered_operator_rejects_hard_cutoff_shape(self):
        workbench = EquationWorkbench(min_samples=16)
        center_x = 8.0
        center_y = 12.0
        cutoff_radius = 4.0
        workbench.generated_operator_bank = {
            'operator:localized_tapered_power:local:direction': {
                'key': 'operator:localized_tapered_power:local:direction',
                'operator_kind': 'localized_tapered_power',
                'inputs': ['center', 'position', 'cutoff_radius', 'distance_exponent'],
                'expression': 'inside(separation <= 4.0) * taper * unit(center - position) / separation^2',
                'generated_from': 'concept:local_high_change_region',
                'usefulness': 0.9,
                'test_hint': 'compare center, middle, boundary, and outside samples',
                'parameters': {
                    'center_x': center_x,
                    'center_y': center_y,
                    'cutoff_radius': cutoff_radius,
                    'distance_exponent': 2.0,
                    'relation': 'direction',
                },
            }
        }
        dt = 0.016
        for step in range(240):
            angle = (2 * math.pi * step) / 48
            radius_options = [0.9, 1.4, 2.1, 2.8, 3.45, 4.35, 5.1, 5.8]
            radius = radius_options[step % len(radius_options)]
            x = center_x + math.cos(angle) * radius
            y = center_y + math.sin(angle) * radius
            vx = ((step % 7) - 3) * 0.02
            vy = ((step % 5) - 2) * 0.02
            dx = center_x - x
            dy = center_y - y
            dist = math.sqrt(dx * dx + dy * dy)
            residual = 0.28 if dist <= cutoff_radius else 0.0
            dvx = 0.01 - 0.012 * vx + residual * dx / dist
            dvy = -0.015 - 0.012 * vy + residual * dy / dist

            workbench.observe_transition(
                state(step, step * dt, [obj(1, x, y, vx, vy)]),
                state(
                    step + 1,
                    (step + 1) * dt,
                    [obj(1, x + vx * dt, y + vy * dt, vx + dvx, vy + dvy)],
                ),
                {'type': 'wait'},
                step + 1,
            )

        equations = workbench.discover(step=240)
        generated = [
            equation for equation in equations
            if equation.role == 'generated_operator_tapered_distance_direction_equation'
        ]

        self.assertFalse(generated)

    def test_generated_tapered_operator_rejects_global_smooth_distance_law(self):
        workbench = EquationWorkbench(min_samples=16)
        center_x = 8.0
        center_y = 12.0
        workbench.generated_operator_bank = {
            'operator:localized_tapered_power:local:direction': {
                'key': 'operator:localized_tapered_power:local:direction',
                'operator_kind': 'localized_tapered_power',
                'inputs': ['center', 'position', 'cutoff_radius', 'distance_exponent'],
                'expression': 'inside(separation <= 4.0) * taper * unit(center - position) / separation^2',
                'generated_from': 'concept:local_high_change_region',
                'usefulness': 0.9,
                'test_hint': 'compare center, middle, boundary, and outside samples',
                'parameters': {
                    'center_x': center_x,
                    'center_y': center_y,
                    'cutoff_radius': 4.0,
                    'distance_exponent': 2.0,
                    'relation': 'direction',
                },
            }
        }
        dt = 0.016
        for step in range(240):
            angle = (2 * math.pi * step) / 48
            radius = 0.9 + (step % 8) * 0.7
            x = center_x + math.cos(angle) * radius
            y = center_y + math.sin(angle) * radius
            vx = ((step % 7) - 3) * 0.02
            vy = ((step % 5) - 2) * 0.02
            dx = center_x - x
            dy = center_y - y
            dist = math.sqrt(dx * dx + dy * dy)
            residual = 0.7 / math.sqrt(max(dist, 0.5))
            dvx = 0.01 - 0.012 * vx + residual * dx / dist
            dvy = -0.015 - 0.012 * vy + residual * dy / dist

            workbench.observe_transition(
                state(step, step * dt, [obj(1, x, y, vx, vy)]),
                state(
                    step + 1,
                    (step + 1) * dt,
                    [obj(1, x + vx * dt, y + vy * dt, vx + dvx, vy + dvy)],
                ),
                {'type': 'wait'},
                step + 1,
            )

        equations = workbench.discover(step=240)
        generated = [
            equation for equation in equations
            if equation.role == 'generated_operator_tapered_distance_direction_equation'
        ]

        self.assertFalse(generated)

    def test_local_residual_needs_meaningful_residual_energy(self):
        workbench = EquationWorkbench(min_samples=16)
        dt = 0.016
        for step in range(120):
            angle = (2 * math.pi * step) / 120
            x = 10.0 + math.cos(angle) * 4.0
            y = 10.0 + math.sin(angle) * 4.0
            vx = ((step % 7) - 3) * 0.04
            vy = ((step % 5) - 2) * 0.04
            tiny_x = 0.0001 * math.cos(angle)
            tiny_y = 0.0001 * math.sin(angle)
            dvx = 0.02 - 0.01 * vx + tiny_x
            dvy = -0.03 - 0.01 * vy + tiny_y

            workbench.observe_transition(
                state(step, step * dt, [obj(1, x, y, vx, vy)]),
                state(
                    step + 1,
                    (step + 1) * dt,
                    [obj(1, x + vx * dt, y + vy * dt, vx + dvx, vy + dvy)],
                ),
                {'type': 'wait'},
                step + 1,
            )

        equations = workbench.discover(step=120)
        pack = workbench.review_pack()
        local_scores = [
            equation.score for equation in equations
            if equation.role.startswith('local_residual')
        ]

        self.assertTrue(local_scores)
        self.assertTrue(all(score == 0.0 for score in local_scores))
        self.assertEqual('position_update_equation', pack['interesting_equations'][0]['role'])

    def test_discover_preserves_best_seen_equation_score(self):
        workbench = EquationWorkbench(min_samples=12)
        dt = 0.016
        for step in range(96):
            angle = (2 * math.pi * step) / 96
            x = 10.0 + math.cos(angle) * 4.0
            y = 10.0 + math.sin(angle) * 4.0
            vx = ((step % 7) - 3) * 0.04
            vy = ((step % 5) - 2) * 0.04
            residual_x = -0.16 * math.sin(angle)
            residual_y = 0.16 * math.cos(angle)
            workbench.observe_transition(
                state(step, step * dt, [obj(1, x, y, vx, vy)]),
                state(
                    step + 1,
                    (step + 1) * dt,
                    [obj(1, x + vx * dt, y + vy * dt, vx + residual_x, vy + residual_y)],
                ),
                {'type': 'wait'},
                step + 1,
            )

        first = workbench.discover(step=96)
        first_best = max(
            (
                equation for equation in first
                if equation.role == 'local_residual_perpendicular_equation'
            ),
            key=lambda equation: equation.score,
        )

        for step in range(96, 170):
            x = 3.0 + (step % 6) * 0.2
            y = 4.0 + (step % 5) * 0.2
            vx = ((step % 4) - 2) * 0.02
            vy = ((step % 3) - 1) * 0.02
            workbench.observe_transition(
                state(step, step * dt, [obj(1, x, y, vx, vy)]),
                state(
                    step + 1,
                    (step + 1) * dt,
                    [obj(1, x + vx * dt, y + vy * dt, vx, vy)],
                ),
                {'type': 'wait'},
                step + 1,
            )

        second = workbench.discover(step=170)
        second_by_key = {equation.key: equation for equation in second}

        self.assertGreater(first_best.score, 0.5)
        self.assertEqual(first_best.score, second_by_key[first_best.key].score)

    def test_best_seen_retention_is_limited_to_local_and_periodic_residuals(self):
        workbench = EquationWorkbench()
        old_global = PrimitiveEquation(
            key='old_global',
            target='baseline_adjusted_delta_velocity',
            expression='k * unit_inferred_vector',
            description='global residual',
            score=0.9,
            mse=0.1,
            baseline_mse=0.2,
            complexity=5,
            sample_count=100,
            role='residual_direction_equation',
        )
        old_local = PrimitiveEquation(
            key='old_local',
            target='baseline_adjusted_delta_velocity',
            expression='k * unit_local_inferred_vector',
            description='local residual',
            score=0.5,
            mse=0.1,
            baseline_mse=0.2,
            complexity=5,
            sample_count=100,
            role='local_residual_direction_equation',
        )
        weaker_local = PrimitiveEquation(
            key='old_local',
            target='baseline_adjusted_delta_velocity',
            expression='k * unit_local_inferred_vector',
            description='local residual',
            score=0.2,
            mse=0.1,
            baseline_mse=0.2,
            complexity=5,
            sample_count=100,
            role='local_residual_direction_equation',
        )

        self.assertFalse(workbench._should_retain_best_equation(old_global, weaker_local))
        self.assertTrue(workbench._should_retain_best_equation(old_local, weaker_local))

    def test_local_residual_quality_gate_requires_error_reduction(self):
        workbench = EquationWorkbench()
        alignment_only = PrimitiveEquation(
            key='alignment_only',
            target='baseline_adjusted_delta_velocity',
            expression='k * unit_local_inferred_vector',
            description='alignment without fit improvement',
            score=0.7,
            mse=0.08021,
            baseline_mse=0.08025,
            complexity=5,
            sample_count=100,
            role='local_residual_direction_equation',
        )
        useful = PrimitiveEquation(
            key='useful',
            target='baseline_adjusted_delta_velocity',
            expression='k * unit_local_inferred_vector',
            description='alignment with fit improvement',
            score=0.7,
            mse=0.47,
            baseline_mse=0.48,
            complexity=5,
            sample_count=100,
            role='local_residual_direction_equation',
        )

        self.assertFalse(workbench._local_residual_passes_quality_gate(alignment_only))
        self.assertTrue(workbench._local_residual_passes_quality_gate(useful))

    def test_equation_probe_suggests_spawn_from_direction_equation(self):
        workbench = EquationWorkbench(min_samples=8)
        for step in range(20):
            angle = (2 * math.pi * step) / 20
            x = 10.0 + math.cos(angle) * 4.0
            y = 10.0 + math.sin(angle) * 4.0
            dx = 10.0 - x
            dy = 10.0 - y
            dist = math.sqrt(dx * dx + dy * dy)
            dvx = 0.1 * dx / dist
            dvy = 0.1 * dy / dist
            workbench.observe_transition(
                state(step, step * 0.016, [obj(1, x, y, 0.0, 0.0)]),
                state(step + 1, (step + 1) * 0.016, [obj(1, x, y, dvx, dvy)]),
                {'type': 'wait'},
                step + 1,
            )
        workbench.discover(step=20)

        action = workbench.suggest_probe_action(
            current_count=1,
            world_width=20.0,
            world_height=20.0,
            step=50,
        )

        self.assertEqual('spawn', action['type'])
        self.assertEqual('equation_workbench_probe', action['source'])
        self.assertEqual(1, len(workbench.review_pack()['probe_suggestions']))

    def test_label_leaks_are_reported_for_review(self):
        workbench = EquationWorkbench()
        workbench.equations['bad'] = PrimitiveEquation(
            key='raw_eq:bad_gravity_label',
            target='dvx',
            expression='gravity',
            description='mentions gravity',
            score=1.0,
            mse=0.0,
            baseline_mse=1.0,
            complexity=1,
            sample_count=10,
        )

        leaks = workbench.label_leaks()

        self.assertEqual(1, len(leaks))
        self.assertIn('gravity', leaks[0]['labels'])

    def test_equation_campaign_returns_review_pack(self):
        with contextlib.redirect_stdout(io.StringIO()):
            results = run_equation_campaign(
                seeds=1,
                steps=80,
                object_counts=[3],
                world_types=['standard'],
                hidden_worlds=1,
                num_agents=2,
            )

        self.assertEqual(2, len(results))
        self.assertTrue(all(result['equation_count'] > 0 for result in results))
        self.assertTrue(all(not result['label_leaks'] for result in results))
        self.assertIn('top_equation', results[0])
        self.assertIn('interesting_equation', results[0])
        self.assertIn('categories', results[0])
        self.assertIn('probe_suggestions', results[0])

    def test_equation_metrics_handle_missing_workbench(self):
        metrics = _equation_metrics_from_knowledge(KnowledgeBase())

        self.assertFalse(metrics['passed'])
        self.assertEqual(0, metrics['equation_count'])

    def test_final_summary_rank_prefers_residual_roles_over_easy_fit(self):
        easy = {
            'interesting_score': 0.97,
            'interesting_equation': {'role': 'position_update_equation'},
        }
        residual = {
            'interesting_score': 0.55,
            'interesting_equation': {'role': 'residual_periodic_equation'},
        }

        self.assertGreater(_interesting_result_rank(residual), _interesting_result_rank(easy))


if __name__ == '__main__':
    unittest.main()
