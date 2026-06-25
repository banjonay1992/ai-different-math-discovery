import os
import sys
import unittest


PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'first_principles_ai'))
sys.path.insert(0, PROJECT_DIR)

from agent.predictor import Predictor
from agent.representation import KnowledgeBase



def feature_set(
    count: int = 5,
    total_momentum_x: float = 1.2,
    total_momentum_y: float = 0.8,
    total_mass: float = 10.0,
    total_kinetic_energy: float = 3.5,
    num_collisions: int = 0,
    world_x_bounds=(5.0, 11.0),
    world_y_bounds=(6.0, 12.0),
    mean_distance: float = 4.2,
    mean_speed: float = 1.1,
):
    max_x, min_x = world_x_bounds[1], world_x_bounds[0]
    max_y, min_y = world_y_bounds[1], world_y_bounds[0]
    total_momentum = (total_momentum_x ** 2 + total_momentum_y ** 2) ** 0.5
    return {
        'count': count,
        'total_momentum_x': total_momentum_x,
        'total_momentum_y': total_momentum_y,
        'total_momentum': total_momentum,
        'total_kinetic_energy': total_kinetic_energy,
        'total_mass': total_mass,
        'center_of_mass_x': (max_x + min_x) / 2,
        'center_of_mass_y': (max_y + min_y) / 2,
        'num_collisions': num_collisions,
        'mean_distance': mean_distance,
        'max_x': max_x,
        'min_x': min_x,
        'max_y': max_y,
        'min_y': min_y,
        'mean_speed': mean_speed,
    }


class PredictorTests(unittest.TestCase):
    def test_tracks_features_and_discovers_core_concepts(self):
        kb = KnowledgeBase()
        predictor = Predictor(kb)

        for step in range(31):
            predictor.observe(feature_set(), had_collision=False, step=step, raw_objects=[])

        discovered_features = {concept.feature_key for concept in kb.get_all_concepts()}
        self.assertIn('count', discovered_features)
        self.assertIn('total_mass', discovered_features)
        self.assertIn('total_momentum', discovered_features)

    def test_confirms_count_increment_rule_with_repeated_spawn_events(self):
        kb = KnowledgeBase()
        predictor = Predictor(kb)

        for step in range(35):
            predictor.observe(feature_set(count=5), had_collision=False, step=step, raw_objects=[])

        # Simulate repeated object appearances. Count increases by exactly 1 each step.
        count = 5
        for step in range(35, 65):
            count += 1
            predictor.observe(feature_set(count=count), had_collision=False, step=step, raw_objects=[])

        confirmed = [
            rule for rule in kb.get_confirmed_rules()
            if rule.properties.get('hypothesis_type') == 'arithmetic'
            and rule.feature_key == 'count'
            and rule.properties.get('delta') == 1
        ]

        self.assertEqual(1, len(confirmed))
        self.assertEqual(1, confirmed[0].properties.get('delta'))

    def test_confirms_mass_conservation_under_stable_observations(self):
        kb = KnowledgeBase()
        predictor = Predictor(kb)

        for step in range(80):
            predictor.observe(feature_set(total_mass=10.0), had_collision=False, step=step, raw_objects=[])

        confirmed = [
            rule for rule in kb.get_confirmed_rules()
            if rule.properties.get('hypothesis_type') == 'global_conservation'
            and rule.feature_key == 'total_mass'
        ]
        self.assertEqual(1, len(confirmed))

    def test_flags_collision_conservation_when_collision_has_no_extra_delta(self):
        kb = KnowledgeBase()
        predictor = Predictor(kb)

        # Build non-collision history first.
        for step in range(40):
            predictor.observe(
                feature_set(total_momentum_x=2.0),
                had_collision=False,
                step=step,
                raw_objects=[],
            )

        # Then only collision transitions with identical feature deltas.
        for step in range(40, 100):
            predictor.observe(
                feature_set(total_momentum_x=2.0),
                had_collision=True,
                step=step,
                raw_objects=[],
            )

        confirmed = [
            rule for rule in kb.get_confirmed_rules()
            if rule.properties.get('hypothesis_type') == 'collision_conservation'
            and rule.feature_key == 'total_momentum_x'
        ]
        self.assertEqual(1, len(confirmed))

    def test_suggest_experiment_action_targets_active_novel_physics_hypothesis(self):
        kb = KnowledgeBase()
        predictor = Predictor(kb, allow_memory_probes=False)

        kb.add_hypothesis(
            conditions='an object is near a field center',
            prediction='motion drifts toward center',
            feature_key='mean_distance',
            step=0,
            properties={
                'hypothesis_type': 'novel_physics',
                'novel_type': 'central_force',
                'attractor_x': 10.0,
                'attractor_y': 10.0,
            },
        )
        predictor.observe(feature_set(), had_collision=False, step=0, raw_objects=[])

        action = predictor.suggest_experiment_action(current_count=5, world_width=20.0, world_height=20.0)
        self.assertIsNotNone(action)
        self.assertEqual('spawn', action['type'])
        self.assertEqual(0.0, action['vx'])
        self.assertEqual(0.0, action['vy'])
        self.assertGreaterEqual(action['x'], 2.0)
        self.assertLessEqual(action['x'], 20.0 - 2.0)
        self.assertGreaterEqual(action['y'], 2.0)
        self.assertLessEqual(action['y'], 20.0 - 2.0)

        # Cooldown: immediate re-suggestion should not happen.
        self.assertIsNone(
            predictor.suggest_experiment_action(current_count=5, world_width=20.0, world_height=20.0)
        )

    def test_memory_probe_questions_generate_spawn_action_after_sufficient_steps(self):
        kb = KnowledgeBase()
        predictor = Predictor(
            kb,
            allow_memory_probes=True,
            law_priors=[
                {
                    'law_type': 'radial_field',
                    'transfer_score': 0.91,
                    'parameter_ranges': {
                        'center_x': (9.0, 9.0),
                        'center_y': (13.0, 13.0),
                    },
                }
            ],
        )

        for step in range(60):
            predictor.observe(feature_set(), had_collision=False, step=step, raw_objects=[])

        action = predictor.suggest_experiment_action(current_count=3, world_width=20.0, world_height=20.0)
        self.assertIsNotNone(action)
        self.assertEqual('spawn', action['type'])
        self.assertEqual(0.0, action['vx'])
        self.assertEqual(0.0, action['vy'])


if __name__ == '__main__':
    unittest.main()
