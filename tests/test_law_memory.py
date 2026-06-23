import os
import sys
import tempfile
import unittest


PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'first_principles_ai'))
sys.path.insert(0, PROJECT_DIR)

from agent.law_memory import LawMemory
from agent.representation import KnowledgeBase


def learned_law(law_type: str, confidence: float = 0.9, improvement: float = 0.8, **properties):
    return {
        'name': f'learned_{law_type}',
        'law_type': law_type,
        'description': f'{law_type} law',
        'confidence': confidence,
        'mse': 0.01,
        'baseline_mse': 0.10,
        'improvement': improvement,
        'sample_count': 120,
        'coefficients': {},
        'properties': properties,
    }


class LawMemoryTests(unittest.TestCase):
    def test_records_episodes_and_consolidates_law_schemas(self):
        memory = LawMemory()

        memory.record_experiment(
            world_type='central_force',
            seed=0,
            object_count=5,
            step_count=500,
            learned_laws=[learned_law(
                'radial_field',
                center_x=10.0,
                center_y=10.0,
                direction='attractive',
                term_origin='invented',
            )],
        )
        memory.record_experiment(
            world_type='localized_gravity',
            seed=1,
            object_count=5,
            step_count=500,
            learned_laws=[learned_law(
                'radial_field',
                center_x=12.0,
                center_y=8.0,
                direction='attractive',
                term_origin='invented',
            )],
        )

        schema = memory.schemas['radial_field']
        self.assertEqual(2, len(memory.episodes))
        self.assertEqual(2, schema.support_count)
        self.assertEqual({'central_force', 'localized_gravity'}, schema.source_worlds)
        self.assertEqual((10.0, 12.0), schema.parameter_ranges['center_x'])
        self.assertEqual((8.0, 10.0), schema.parameter_ranges['center_y'])
        self.assertIn('radial_field', memory.suggest_priors()[0]['law_type'])

    def test_transfer_report_compares_new_run_to_prior_memory_before_update(self):
        memory = LawMemory()
        memory.record_experiment(
            world_type='central_force',
            seed=0,
            object_count=5,
            step_count=500,
            learned_laws=[learned_law('radial_field')],
        )

        record = memory.record_experiment(
            world_type='vortex',
            seed=1,
            object_count=5,
            step_count=500,
            learned_laws=[learned_law('tangential_field', spin='counterclockwise')],
        )

        missing = [item['law_type'] for item in record.transfer_report['missing_priors']]
        self.assertIn('radial_field', missing)
        self.assertIn('tangential_field', memory.schemas)
        self.assertIn('vortex', memory.schemas['radial_field'].absent_worlds)

    def test_principles_include_time_and_composed_law_families(self):
        memory = LawMemory()
        memory.record_experiment(
            world_type='mixed',
            seed=0,
            object_count=5,
            step_count=500,
            learned_laws=[
                learned_law('time_varying_field', period_steps=80, axis='x'),
                learned_law(
                    'composed_dynamics',
                    components=['radial_field', 'time_varying_field'],
                    component_count=2,
                ),
            ],
        )

        principles = "\n".join(memory.principles())
        self.assertIn('vary over time', principles)
        self.assertIn('combine multiple simpler law families', principles)

    def test_memory_saves_loads_and_keeps_transfer_scores(self):
        memory = LawMemory()
        memory.record_experiment(
            world_type='time_varying',
            seed=0,
            object_count=5,
            step_count=500,
            learned_laws=[learned_law('time_varying_field', period_steps=80, axis='x')],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'law-memory.json')
            memory.save(path)
            loaded = LawMemory.load(path)

        self.assertEqual(1, len(loaded.episodes))
        self.assertIn('time_varying_field', loaded.schemas)
        self.assertGreater(loaded.schemas['time_varying_field'].transfer_score, 0.0)

    def test_loading_empty_memory_file_starts_fresh(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'empty.json')
            open(path, 'w', encoding='utf-8').close()

            loaded = LawMemory.load(path)

        self.assertEqual(0, len(loaded.episodes))
        self.assertEqual({}, loaded.schemas)

    def test_probe_questions_and_principle_installation_are_explicit(self):
        memory = LawMemory()
        memory.record_experiment(
            world_type='central_force',
            seed=0,
            object_count=5,
            step_count=500,
            learned_laws=[learned_law('radial_field', center_x=10.0, center_y=12.0)],
        )
        kb = KnowledgeBase()

        questions = memory.suggest_probe_questions()
        memory.install_principles(kb, step=500)

        self.assertEqual('radial_field', questions[0]['law_type'])
        self.assertEqual('point', questions[0]['target']['kind'])
        self.assertTrue(any(
            concept.properties.get('source') == 'law_memory'
            for concept in kb.get_meta_concepts()
        ))

    def test_uniform_acceleration_prior_does_not_spawn_probe_question(self):
        memory = LawMemory()
        memory.record_experiment(
            world_type='standard',
            seed=0,
            object_count=5,
            step_count=500,
            learned_laws=[learned_law('uniform_acceleration')],
        )

        self.assertEqual([], memory.suggest_probe_questions())


if __name__ == '__main__':
    unittest.main()
