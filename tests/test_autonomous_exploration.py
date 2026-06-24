import contextlib
import io
import os
import random
import sys
import unittest


PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'first_principles_ai'))
sys.path.insert(0, PROJECT_DIR)

from agent.explorer import ExplorationPlanner, score_exploration_outcome
from agent.law_memory import LawMemory
from agent.representation import RuleStatus
from main import (
    _blind_hidden_discoveries,
    run_autonomous_exploration,
    run_hidden_autonomous_exploration,
    run_hidden_holdout_benchmark,
)
from world.environment import Environment
from world.hidden_worlds import (
    generate_hidden_world_manifest,
    generate_self_authored_hidden_world_manifest,
    hidden_manifest_from_observation,
    registered_hidden_component_types,
)
from world.physics import registered_force_component_types


def learned_law(law_type: str, **properties):
    return {
        'name': f'learned_{law_type}',
        'law_type': law_type,
        'description': f'{law_type} law',
        'confidence': 0.9,
        'mse': 0.01,
        'baseline_mse': 0.10,
        'improvement': 0.8,
        'sample_count': 120,
        'coefficients': {},
        'properties': properties,
    }


class AutonomousExplorationTests(unittest.TestCase):
    def test_planner_prefers_untested_transfer_contexts(self):
        memory = LawMemory()
        memory.record_experiment(
            world_type='central_force',
            seed=0,
            object_count=5,
            step_count=100,
            learned_laws=[learned_law('radial_field')],
        )
        planner = ExplorationPlanner(
            world_types=['central_force', 'vortex'],
            object_counts=[5],
            steps=100,
            seed_start=0,
            seed_span=2,
        )

        candidates = planner.propose(memory, limit=3)

        self.assertEqual('vortex', candidates[0].world_type)
        self.assertGreater(candidates[0].reasons['transfer_gap'], 0.0)

    def test_outcome_score_rewards_learning_and_falsification(self):
        value = score_exploration_outcome({
            'learned_rule_count': 2,
            'detected': {'vortex'},
            'first_learned_step': 50,
            'steps': 100,
            'memory_transfer': {
                'matched_priors': [{'law_type': 'radial_field'}],
                'missing_priors': [{'law_type': 'time_varying_field'}],
            },
        })

        self.assertGreater(value, 0.4)

    def test_autonomous_exploration_runs_budgeted_real_experiments(self):
        memory = LawMemory()

        with contextlib.redirect_stdout(io.StringIO()):
            results = run_autonomous_exploration(
                budget=2,
                steps=90,
                object_counts=[5],
                world_types=['standard', 'sideways_wind'],
                num_agents=2,
                law_memory=memory,
                seed_start=0,
                seed_span=2,
            )

        self.assertEqual(2, len(results))
        self.assertEqual(2, len(memory.episodes))
        keys = {
            (
                result['candidate']['world_type'],
                result['candidate']['seed'],
                result['candidate']['object_count'],
                result['candidate']['steps'],
            )
            for result in results
        }
        self.assertEqual(2, len(keys))
        self.assertIn('outcome_score', results[0])
        self.assertIn('memory_transfer', results[0]['metrics'])

    def test_hidden_world_observation_does_not_expose_manifest(self):
        manifest = generate_hidden_world_manifest(seed=3, variant=2)
        env = Environment(
            num_initial_objects=2,
            seed=3,
            world_type='hidden_procedural',
            hidden_manifest=manifest,
        )

        observation = env.observe()

        self.assertNotIn('world_type', observation)
        self.assertEqual('hidden_procedural', env.benchmark_metadata()['world_type'])
        self.assertEqual(manifest.hidden_id, env.benchmark_metadata()['hidden_id'])
        self.assertFalse(hidden_manifest_from_observation(observation))
        self.assertNotIn('components', observation)
        self.assertNotIn('expected_discoveries', observation)

    def test_environment_seed_is_reproducible_without_global_random_bleed(self):
        random.seed(123456)
        expected_global = random.Random(123456)

        first = Environment(
            num_initial_objects=3,
            seed=19,
            world_type='standard',
        )
        second = Environment(
            num_initial_objects=3,
            seed=19,
            world_type='standard',
        )

        self.assertEqual(first.observe(), second.observe())
        self.assertEqual([0, 1, 2], first.get_object_ids())
        self.assertEqual([0, 1, 2], second.get_object_ids())
        self.assertEqual(expected_global.random(), random.random())

    def test_environment_supports_direct_causal_interventions(self):
        env = Environment(
            num_initial_objects=1,
            seed=5,
            world_type='zero_gravity',
        )
        object_id = env.get_object_ids()[0]

        moved = env.step({
            'type': 'move',
            'object_id': object_id,
            'x': 6.0,
            'y': 7.0,
            'vx': 0.0,
            'vy': 0.0,
        })
        moved_object = next(
            item for item in moved['objects']
            if item['id'] == object_id
        )
        self.assertEqual((6.0, 7.0), moved_object['position'])

        env.step({'type': 'push', 'object_id': object_id, 'fx': 3.0, 'fy': 0.0})
        frozen = env.step({'type': 'freeze', 'object_id': object_id})
        frozen_object = next(
            item for item in frozen['objects']
            if item['id'] == object_id
        )
        self.assertEqual((0.0, 0.0), frozen_object['velocity'])

        before_count = len(env.get_object_ids())
        duplicated = env.step({'type': 'duplicate', 'object_id': object_id})
        self.assertEqual(before_count + 1, len(duplicated['objects']))

    def test_hidden_world_generator_covers_more_component_recipes(self):
        component_signatures = {
            tuple(
                component.component_type
                for component in generate_hidden_world_manifest(
                    seed=11,
                    variant=variant,
                ).components
            )
            for variant in range(25)
        }

        self.assertGreaterEqual(len(component_signatures), 20)
        self.assertIn(('zero_gravity', 'uniform_push'), component_signatures)
        self.assertIn(('soft_drag', 'uniform_push'), component_signatures)
        self.assertIn(('localized_push', 'uniform_push'), component_signatures)

    def test_localized_push_hidden_component_uses_repulsion_zone_without_leaking(self):
        manifest = generate_hidden_world_manifest(seed=11, variant=15)
        env = Environment(
            num_initial_objects=2,
            seed=11,
            world_type='hidden_procedural',
            hidden_manifest=manifest,
        )
        observation = env.observe()

        self.assertEqual(
            ['localized_push', 'uniform_push'],
            [component.component_type for component in manifest.components],
        )
        self.assertTrue(env.world.repulsion_zones)
        self.assertFalse(hidden_manifest_from_observation(observation))

    def test_adversarial_hidden_components_are_registry_backed_and_blind(self):
        cutoff = generate_hidden_world_manifest(seed=13, variant=25)
        piecewise = generate_hidden_world_manifest(seed=13, variant=27)

        registered_hidden = set(registered_hidden_component_types())
        registered_forces = set(registered_force_component_types())
        self.assertIn('cutoff_radial_pull', registered_hidden)
        self.assertIn('piecewise_radial', registered_hidden)
        self.assertIn('cutoff_radial', registered_forces)
        self.assertIn('piecewise_radial', registered_forces)

        cutoff_env = Environment(
            num_initial_objects=2,
            seed=13,
            world_type='hidden_procedural',
            hidden_manifest=cutoff,
        )
        piecewise_env = Environment(
            num_initial_objects=2,
            seed=13,
            world_type='hidden_procedural',
            hidden_manifest=piecewise,
        )

        self.assertIn(
            'cutoff_radial_pull',
            [component.component_type for component in cutoff.components],
        )
        self.assertIn(
            'piecewise_radial',
            [component.component_type for component in piecewise.components],
        )
        self.assertIn('piecewise_component', cutoff.expected_discoveries)
        self.assertIn('piecewise_component', piecewise.expected_discoveries)
        self.assertIn(
            'cutoff_radial',
            [component['type'] for component in cutoff_env.world.force_components],
        )
        self.assertIn(
            'piecewise_radial',
            [component['type'] for component in piecewise_env.world.force_components],
        )
        self.assertFalse(hidden_manifest_from_observation(cutoff_env.observe()))
        self.assertFalse(hidden_manifest_from_observation(piecewise_env.observe()))

    def test_self_authored_hidden_world_manifest_stays_blind(self):
        design = {
            'design_key': 'autonomous_design:invariant_resolution:test',
            'source': 'invariant_resolution',
            'question': 'Which distance exponent wins a near/mid/far race?',
        }

        manifest = generate_self_authored_hidden_world_manifest(
            design,
            seed=4,
            variant=2,
        )
        component_types = [
            component.component_type
            for component in manifest.components
        ]
        env = Environment(
            num_initial_objects=2,
            seed=4,
            world_type='hidden_procedural',
            hidden_manifest=manifest,
        )
        observation = env.observe()

        self.assertEqual('authored_02_0004', manifest.hidden_id)
        self.assertIn('piecewise_radial', component_types)
        self.assertIn('cutoff_radial_push', component_types)
        self.assertFalse(hidden_manifest_from_observation(observation))
        self.assertNotIn('question', observation)

    def test_blind_hidden_discovery_maps_laws_without_manifest_labels(self):
        kb = LawMemoryKnowledgeBuilder.with_laws([
            learned_law('uniform_acceleration'),
            learned_law('radial_field', direction='attractive'),
            learned_law('tangential_field'),
            learned_law('composed_dynamics'),
        ])

        discoveries = _blind_hidden_discoveries(kb)

        self.assertIn('uniform_component', discoveries)
        self.assertIn('radial_component', discoveries)
        self.assertIn('tangential_component', discoveries)
        self.assertIn('composed_law', discoveries)

    def test_hidden_holdout_benchmark_runs_blind_train_and_holdout(self):
        memory = LawMemory()

        with contextlib.redirect_stdout(io.StringIO()):
            results = run_hidden_holdout_benchmark(
                train_worlds=1,
                holdout_worlds=1,
                steps=120,
                object_count=4,
                num_agents=2,
                law_memory=memory,
                seed_start=0,
            )

        self.assertEqual(2, len(results))
        self.assertEqual('train', results[0]['phase'])
        self.assertEqual('holdout', results[1]['phase'])
        self.assertIn('expected_discoveries', results[1]['warm'])
        self.assertFalse(results[1]['warm']['observation_leak'])

    def test_hidden_autonomous_exploration_reports_proposals(self):
        memory = LawMemory()

        with contextlib.redirect_stdout(io.StringIO()):
            results = run_hidden_autonomous_exploration(
                budget=1,
                steps=120,
                object_count=4,
                num_agents=2,
                law_memory=memory,
                seed_start=1,
            )

        self.assertEqual(1, len(results))
        self.assertIn('experiment_proposals', results[0]['metrics'])
        self.assertIn('score', results[0]['metrics'])


class LawMemoryKnowledgeBuilder:
    @staticmethod
    def with_laws(laws):
        from agent.representation import KnowledgeBase

        kb = KnowledgeBase()
        for index, law in enumerate(laws, start=1):
            rule = kb.add_hypothesis(
                conditions='object state observed',
                prediction=law['description'],
                feature_key='mean_speed',
                step=index,
                properties={
                    'hypothesis_type': 'learned_dynamics',
                    'law_type': law['law_type'],
                    **law.get('properties', {}),
                },
            )
            rule.evidence_for = 10
            rule.confidence = 1.0
            rule.status = RuleStatus.CONFIRMED
        return kb


if __name__ == '__main__':
    unittest.main()
