import math
import os
import sys
import unittest


PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'first_principles_ai'))
sys.path.insert(0, PROJECT_DIR)

from agent.law_learning import DynamicsLawLearner, MotionSample


def circular_samples(force_fn, steps: int = 120) -> list[MotionSample]:
    return circular_samples_around(10.0, 10.0, force_fn, steps)


def circular_samples_around(
    center_x: float,
    center_y: float,
    force_fn,
    steps: int = 120,
) -> list[MotionSample]:
    samples = []
    for step in range(steps):
        angle = (2 * math.pi * step) / steps
        radius = 5.0 + (step % 5) * 0.4
        x = center_x + math.cos(angle) * radius
        y = center_y + math.sin(angle) * radius
        vx = ((step % 7) - 3) * 0.03
        vy = ((step % 5) - 2) * 0.03
        dvx, dvy = force_fn(step, x, y, vx, vy)
        samples.append(MotionSample(step, x, y, vx, vy, dvx, dvy))
    return samples


class DynamicsLawLearnerTests(unittest.TestCase):
    def test_learns_uniform_horizontal_acceleration(self):
        learner = DynamicsLawLearner()
        samples = circular_samples(
            lambda step, x, y, vx, vy: (0.13 - 0.001 * vx, -0.001 * vy)
        )

        laws = learner.discover(samples)
        uniform = next(law for law in laws if law.law_type == 'uniform_acceleration')

        self.assertEqual('x', uniform.properties['axis'])
        self.assertEqual('rightward', uniform.properties['direction'])
        self.assertGreater(uniform.confidence, 0.9)

    def test_learns_radial_attraction(self):
        learner = DynamicsLawLearner()

        def force(step, x, y, vx, vy):
            dx = 10.0 - x
            dy = 10.0 - y
            dist = math.sqrt(dx * dx + dy * dy)
            return (0.22 * dx / dist, 0.22 * dy / dist)

        laws = learner.discover(circular_samples(force))
        radial = next(law for law in laws if law.law_type == 'radial_field')

        self.assertEqual('attractive', radial.properties['direction'])
        self.assertAlmostEqual(10.0, radial.properties['center_x'], delta=2.0)
        self.assertAlmostEqual(10.0, radial.properties['center_y'], delta=2.0)
        self.assertGreater(radial.confidence, 0.5)

    def test_learns_tangential_vortex_field(self):
        learner = DynamicsLawLearner()

        def force(step, x, y, vx, vy):
            dx = x - 10.0
            dy = y - 10.0
            dist = math.sqrt(dx * dx + dy * dy)
            return (-0.20 * dy / dist, 0.20 * dx / dist)

        laws = learner.discover(circular_samples(force))
        tangent = next(law for law in laws if law.law_type == 'tangential_field')

        self.assertEqual('counterclockwise', tangent.properties['spin'])
        self.assertAlmostEqual(10.0, tangent.properties['center_x'], delta=2.0)
        self.assertAlmostEqual(10.0, tangent.properties['center_y'], delta=2.0)
        self.assertGreater(tangent.confidence, 0.5)

    def test_learns_time_varying_force(self):
        learner = DynamicsLawLearner()

        def force(step, x, y, vx, vy):
            return (0.14 * math.sin((2 * math.pi * step) / 80), 0.0)

        laws = learner.discover(circular_samples(force, steps=160))
        timed = next(law for law in laws if law.law_type == 'time_varying_field')

        self.assertEqual('x', timed.properties['axis'])
        self.assertEqual(80, timed.properties['period_steps'])
        self.assertGreater(timed.confidence, 0.7)

    def test_invents_off_grid_radial_center_from_residuals(self):
        learner = DynamicsLawLearner()
        center_x = 11.3
        center_y = 7.7

        def force(step, x, y, vx, vy):
            dx = center_x - x
            dy = center_y - y
            dist = math.sqrt(dx * dx + dy * dy)
            return (0.20 * dx / dist, 0.20 * dy / dist)

        samples = circular_samples_around(center_x, center_y, force, steps=160)
        laws = learner.discover(samples)
        radial = next(law for law in laws if law.law_type == 'radial_field')

        self.assertEqual('invented', radial.properties['term_origin'])
        self.assertAlmostEqual(center_x, radial.properties['center_x'], delta=0.25)
        self.assertAlmostEqual(center_y, radial.properties['center_y'], delta=0.25)

    def test_invents_non_seeded_temporal_period(self):
        learner = DynamicsLawLearner()
        period = 72

        def force(step, x, y, vx, vy):
            return (0.14 * math.sin((2 * math.pi * step) / period), 0.0)

        samples = circular_samples_around(10.0, 10.0, force, steps=240)
        laws = learner.discover(samples)
        timed = next(law for law in laws if law.law_type == 'time_varying_field')

        self.assertEqual('invented', timed.properties['term_origin'])
        self.assertEqual(period, timed.properties['period_steps'])
        self.assertGreater(timed.confidence, 0.9)

    def test_composes_radial_and_periodic_terms(self):
        learner = DynamicsLawLearner()
        center_x = 11.3
        center_y = 7.7
        period = 80

        def force(step, x, y, vx, vy):
            dx = center_x - x
            dy = center_y - y
            dist = math.sqrt(dx * dx + dy * dy)
            radial_x = 0.16 * dx / dist
            radial_y = 0.16 * dy / dist
            periodic_x = 0.11 * math.sin((2 * math.pi * step) / period)
            return (radial_x + periodic_x, radial_y)

        samples = circular_samples_around(center_x, center_y, force, steps=240)
        laws = learner.discover(samples, max_laws=10)
        composed = next(law for law in laws if law.law_type == 'composed_dynamics')
        radial = next(law for law in laws if law.law_type == 'radial_field')

        self.assertGreater(composed.confidence, 0.75)
        self.assertLess(composed.mse, radial.mse)
        self.assertEqual(2, composed.properties['component_count'])
        self.assertIn('radial_field', composed.properties['components'])
        self.assertIn('time_varying_field', composed.properties['components'])
        self.assertIn('invented', composed.properties['term_origins'])
        self.assertGreater(composed.properties['complexity_penalty'], 0)
        self.assertGreater(composed.properties['elegance_score'], radial.confidence)

    def test_single_radial_law_is_not_reported_as_composed(self):
        learner = DynamicsLawLearner()
        center_x = 11.3
        center_y = 7.7

        def force(step, x, y, vx, vy):
            dx = center_x - x
            dy = center_y - y
            dist = math.sqrt(dx * dx + dy * dy)
            return (0.20 * dx / dist, 0.20 * dy / dist)

        laws = learner.discover(
            circular_samples_around(center_x, center_y, force, steps=160),
            max_laws=10,
        )

        self.assertFalse(any(law.law_type == 'composed_dynamics' for law in laws))

    def test_single_tangential_law_is_not_overfit_as_composed(self):
        learner = DynamicsLawLearner()
        center_x = 10.0
        center_y = 10.0

        def force(step, x, y, vx, vy):
            dx = x - center_x
            dy = y - center_y
            dist = math.sqrt(dx * dx + dy * dy)
            return (-0.20 * dy / dist, 0.20 * dx / dist)

        laws = learner.discover(
            circular_samples_around(center_x, center_y, force, steps=240),
            max_laws=10,
        )
        tangent = next(law for law in laws if law.law_type == 'tangential_field')

        self.assertGreater(tangent.confidence, 0.9)
        self.assertFalse(any(law.law_type == 'composed_dynamics' for law in laws))

    def test_memory_prior_center_can_guide_law_search(self):
        center_x = 11.0
        center_y = 9.0
        learner = DynamicsLawLearner(law_priors=[{
            'law_type': 'radial_field',
            'transfer_score': 0.9,
            'parameter_ranges': {
                'center_x': (center_x, center_x),
                'center_y': (center_y, center_y),
            },
        }])

        def force(step, x, y, vx, vy):
            dx = center_x - x
            dy = center_y - y
            dist = math.sqrt(dx * dx + dy * dy)
            return (0.20 * dx / dist, 0.20 * dy / dist)

        laws = learner.discover(
            circular_samples_around(center_x, center_y, force, steps=160),
            max_laws=10,
        )
        radial = next(law for law in laws if law.law_type == 'radial_field')

        self.assertEqual('memory', radial.properties['term_origin'])
        self.assertAlmostEqual(center_x, radial.properties['center_x'], delta=0.1)
        self.assertAlmostEqual(center_y, radial.properties['center_y'], delta=0.1)


if __name__ == '__main__':
    unittest.main()
