import contextlib
import io
import os
import sys
import unittest


PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'first_principles_ai'))
sys.path.insert(0, PROJECT_DIR)

from agent.curiosity import Curiosity
from agent.perception import Perception
from agent.predictor import Predictor
from agent.representation import KnowledgeBase
from main import EXPECTED_NOVEL_BY_WORLD, run_benchmark, run_transfer_benchmark
from world.environment import Environment
from world.physics import PhysicsObject


def run_learning_world(world_type: str, steps: int = 600) -> KnowledgeBase:
    PhysicsObject._next_id = 0
    env = Environment(num_initial_objects=5, seed=42, world_type=world_type)
    kb = KnowledgeBase()
    predictor = Predictor(kb)
    curiosity = Curiosity(env)

    for step in range(steps):
        raw_state = env.observe()
        observation = Perception.perceive(raw_state)
        features = observation.get_feature_vector()
        action = predictor.suggest_experiment_action(
            current_count=features.get('count', 0),
            world_width=observation.world_width or 20.0,
            world_height=observation.world_height or 20.0,
        )
        if action is None:
            action = curiosity.select_action(predictor, features)

        raw_state = env.step(action)
        observation = Perception.perceive(raw_state)
        predictor.observe(
            observation.get_feature_vector(),
            len(observation.collisions) > 0,
            step + 1,
            raw_objects=raw_state['objects'],
        )
        curiosity.decay_exploration()

    return kb


def confirmed_novel_types(kb: KnowledgeBase) -> set[str]:
    return {
        rule.properties.get('novel_type')
        for rule in kb.get_confirmed_rules()
        if rule.properties.get('hypothesis_type') == 'novel_physics'
    }


class NovelPhysicsDetectionTests(unittest.TestCase):
    def test_standard_world_has_no_novel_physics_false_positive(self):
        kb = run_learning_world('standard')

        self.assertEqual(set(), confirmed_novel_types(kb))

    def test_central_force_world_confirms_attractive_field(self):
        kb = run_learning_world('central_force')

        self.assertIn('central_force', confirmed_novel_types(kb))

    def test_repulsion_world_confirms_repulsive_field(self):
        kb = run_learning_world('repulsion')

        self.assertIn('repulsion', confirmed_novel_types(kb))

    def test_zero_gravity_world_confirms_missing_gravity(self):
        kb = run_learning_world('zero_gravity')

        self.assertIn('zero_gravity', confirmed_novel_types(kb))

    def test_sideways_wind_world_confirms_uniform_horizontal_force(self):
        kb = run_learning_world('sideways_wind')

        self.assertIn('uniform_horizontal_force', confirmed_novel_types(kb))

    def test_vortex_world_confirms_tangential_force_field(self):
        kb = run_learning_world('vortex')

        self.assertIn('vortex', confirmed_novel_types(kb))

    def test_inverse_square_repulsion_world_confirms_repulsion_source(self):
        kb = run_learning_world('inverse_square_repulsion')

        self.assertEqual({'repulsion'}, confirmed_novel_types(kb))

    def test_localized_gravity_world_confirms_central_force(self):
        kb = run_learning_world('localized_gravity')

        self.assertIn('central_force', confirmed_novel_types(kb))

    def test_time_varying_world_confirms_temporal_force(self):
        kb = run_learning_world('time_varying')

        self.assertIn('time_varying_force', confirmed_novel_types(kb))

    def test_benchmark_reports_expected_results_for_small_sweep(self):
        with contextlib.redirect_stdout(io.StringIO()):
            results = run_benchmark(
                seeds=1,
                steps=600,
                object_counts=[5],
                world_types=['standard', 'sideways_wind'],
                num_agents=2,
            )

        self.assertEqual(2, len(results))
        self.assertTrue(all(result['passed'] for result in results))
        self.assertEqual(EXPECTED_NOVEL_BY_WORLD['standard'], results[0]['expected'])
        self.assertIn('memory_transfer', results[0])
        self.assertIn('observed', results[0]['memory_transfer'])

    def test_benchmark_vortex_seed_zero_recovers_from_rejected_candidate(self):
        with contextlib.redirect_stdout(io.StringIO()):
            results = run_benchmark(
                seeds=1,
                steps=500,
                object_counts=[5],
                world_types=['vortex'],
                num_agents=2,
            )

        self.assertEqual({'vortex'}, results[0]['detected'])
        self.assertTrue(results[0]['passed'])

    def test_transfer_benchmark_reports_cold_and_warm_metrics(self):
        with contextlib.redirect_stdout(io.StringIO()):
            results = run_transfer_benchmark(
                seeds=1,
                steps=120,
                object_counts=[5],
                world_types=['standard'],
                num_agents=2,
            )

        self.assertEqual(1, len(results))
        self.assertIn('cold', results[0])
        self.assertIn('warm', results[0])
        self.assertIn('learned_rule_count', results[0]['warm'])
        self.assertIn('transfer_report', results[0])


if __name__ == '__main__':
    unittest.main()
