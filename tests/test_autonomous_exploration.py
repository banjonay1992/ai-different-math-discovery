import contextlib
import io
import os
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
from world.hidden_worlds import generate_hidden_world_manifest, hidden_manifest_from_observation


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

        self.assertEqual('hidden_procedural', observation['world_type'])
        self.assertFalse(hidden_manifest_from_observation(observation))
        self.assertNotIn('components', observation)
        self.assertNotIn('expected_discoveries', observation)

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
