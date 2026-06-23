import contextlib
import io
import os
import sys
import unittest


PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'first_principles_ai'))
sys.path.insert(0, PROJECT_DIR)

from agent.discovery_loop import CumulativeTheoryMemory, MATH_DOMAIN_CURRICULUM
from main import run_domain_curriculum_preview
from world.math_domain_worlds import (
    DOMAIN_WORLD_GENERATORS,
    generate_all_math_domain_world_manifests,
    generate_math_domain_world_manifest,
    math_domain_manifest_from_observation,
)


class MathDomainWorldTests(unittest.TestCase):
    def test_domain_generators_cover_curriculum_without_observation_leaks(self):
        required = {domain['key'] for domain in MATH_DOMAIN_CURRICULUM}

        self.assertEqual(required, set(DOMAIN_WORLD_GENERATORS))

        for manifest in generate_all_math_domain_world_manifests(seed=3, variant=1):
            with self.subTest(domain=manifest.domain_key):
                self.assertGreater(len(manifest.samples), 0)
                self.assertTrue(manifest.expected_discoveries)
                self.assertTrue(manifest.falsifiers)
                self.assertTrue(manifest.transfer_targets)
                self.assertEqual(len(manifest.samples), len(manifest.observations()))

                for observation in manifest.observations():
                    self.assertFalse(math_domain_manifest_from_observation(observation))
                    self.assertIn('sample_id', observation)
                    self.assertIn('observation_kind', observation)
                    self.assertNotIn('domain_key', observation)
                    self.assertNotIn('expected_discoveries', observation)
                    self.assertNotIn('falsifiers', observation)

                manifest_dict = manifest.to_dict()
                self.assertIn('domain_key', manifest_dict)
                self.assertIn('expected_discoveries', manifest_dict)
                self.assertIn('falsifiers', manifest_dict)
                self.assertIn('observation_schema', manifest_dict)

    def test_domain_generation_is_deterministic_but_variant_sensitive(self):
        first = generate_math_domain_world_manifest(
            'calculus_change',
            seed=11,
            variant=0,
        ).to_dict()
        second = generate_math_domain_world_manifest(
            'calculus_change',
            seed=11,
            variant=0,
        ).to_dict()
        changed = generate_math_domain_world_manifest(
            'calculus_change',
            seed=11,
            variant=1,
        ).to_dict()

        self.assertEqual(first, second)
        self.assertNotEqual(first['samples'], changed['samples'])

    def test_cumulative_memory_reports_executable_domain_worlds(self):
        memory = CumulativeTheoryMemory()

        blueprints = memory.domain_world_blueprints(limit=20)
        readiness = memory.discovery_readiness_report()
        packed = memory.to_dict()
        summary = memory.summary()

        required = {domain['key'] for domain in MATH_DOMAIN_CURRICULUM}
        self.assertEqual(required, {item['domain_key'] for item in blueprints})
        self.assertTrue(all(item['sample_count'] > 0 for item in blueprints))
        self.assertTrue(all(item['falsifier_count'] > 0 for item in blueprints))
        self.assertTrue(all(not item['leaks_benchmark_truth'] for item in blueprints))
        self.assertTrue(
            readiness['gates']['executable_domain_worlds']['passed']
        )
        self.assertIn('domain_world_blueprints', readiness)
        self.assertIn('domain_world_blueprints', packed)
        self.assertIn(
            'domain_world_blueprints',
            packed['discovery_evidence_dossier'],
        )
        self.assertIn('Domain world blueprints:', summary)

    def test_domain_curriculum_preview_is_non_final_and_lists_worlds(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            blueprints = run_domain_curriculum_preview(CumulativeTheoryMemory())

        printed = output.getvalue()

        self.assertEqual(len(MATH_DOMAIN_CURRICULUM), len(blueprints))
        self.assertIn('DOMAIN CURRICULUM WORLD PREVIEW', printed)
        self.assertIn('Final discovery run: not run', printed)
        self.assertIn('arithmetic_quantity', printed)
        self.assertIn('higher_dimensions', printed)


if __name__ == '__main__':
    unittest.main()
