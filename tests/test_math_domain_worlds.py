import contextlib
import io
import os
import sys
import unittest


PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'first_principles_ai'))
sys.path.insert(0, PROJECT_DIR)

from agent.discovery_loop import (
    CumulativeTheoryMemory,
    MATH_DOMAIN_CURRICULUM,
    MATH_DOMAIN_TRANSFER_BRIDGES,
)
from agent.domain_world_discovery import (
    build_domain_transfer_evidence,
    discover_all_domain_worlds,
    discover_domain_world_manifest,
)
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
        discoveries = memory.domain_world_discovery_reports(limit=20)
        transfer_evidence = memory.domain_world_transfer_evidence(limit=20)
        readiness = memory.discovery_readiness_report()
        packed = memory.to_dict()
        summary = memory.summary()

        required = {domain['key'] for domain in MATH_DOMAIN_CURRICULUM}
        self.assertEqual(required, {item['domain_key'] for item in blueprints})
        self.assertTrue(all(item['sample_count'] > 0 for item in blueprints))
        self.assertTrue(all(item['falsifier_count'] > 0 for item in blueprints))
        self.assertTrue(all(not item['leaks_benchmark_truth'] for item in blueprints))
        self.assertEqual(required, {item['domain_key'] for item in discoveries})
        self.assertTrue(all(item['candidate_count'] > 0 for item in discoveries))
        self.assertTrue(all(item['self_authored_equations'] for item in discoveries))
        self.assertTrue(all(
            item['benchmark_coverage'] >= 1.0
            and item['falsification_test_count'] > 0
            and not item['leaked_manifest']
            for item in discoveries
        ))
        self.assertEqual(
            {bridge['key'] for bridge in MATH_DOMAIN_TRANSFER_BRIDGES},
            {item['bridge_key'] for item in transfer_evidence},
        )
        self.assertTrue(all(
            item['status'] == 'transfer_link_ready'
            and item['source_matches']
            and item['target_matches']
            for item in transfer_evidence
        ))
        self.assertTrue(
            readiness['gates']['executable_domain_worlds']['passed']
        )
        self.assertTrue(
            readiness['gates']['domain_world_discovery_loop']['passed']
        )
        self.assertTrue(
            readiness['gates']['domain_world_transfer_evidence']['passed']
        )
        self.assertIn('domain_world_blueprints', readiness)
        self.assertIn('domain_world_discoveries', readiness)
        self.assertIn('domain_world_transfer_evidence', readiness)
        self.assertIn('domain_world_blueprints', packed)
        self.assertIn('domain_world_discoveries', packed)
        self.assertIn('domain_world_transfer_evidence', packed)
        self.assertIn(
            'domain_world_blueprints',
            packed['discovery_evidence_dossier'],
        )
        self.assertIn(
            'domain_world_discoveries',
            packed['discovery_evidence_dossier'],
        )
        self.assertIn(
            'domain_world_transfer_evidence',
            packed['discovery_evidence_dossier'],
        )
        self.assertIn('Domain world blueprints:', summary)
        self.assertIn('Domain world discoveries:', summary)
        self.assertIn('Domain world transfer evidence:', summary)

    def test_domain_discovery_infers_candidates_before_manifest_scoring(self):
        reports = discover_all_domain_worlds(seed=5, variant=0)

        self.assertEqual(
            {domain['key'] for domain in MATH_DOMAIN_CURRICULUM},
            {report.domain_key for report in reports},
        )
        for report in reports:
            with self.subTest(domain=report.domain_key):
                packed = report.to_dict()
                self.assertGreater(packed['candidate_count'], 0)
                self.assertTrue(packed['self_authored_equations'])
                self.assertGreater(packed['falsification_test_count'], 0)
                self.assertEqual(1.0, packed['benchmark_coverage'])
                self.assertEqual([], packed['missing_comparison_tags'])
                self.assertFalse(packed['leaked_manifest'])

        transfer_evidence = build_domain_transfer_evidence(
            reports,
            MATH_DOMAIN_TRANSFER_BRIDGES,
        )
        self.assertEqual(len(MATH_DOMAIN_TRANSFER_BRIDGES), len(transfer_evidence))
        self.assertTrue(all(
            item['status'] == 'transfer_link_ready'
            and item['falsifies_if']
            for item in transfer_evidence
        ))

    def test_single_manifest_discovery_uses_public_observations(self):
        manifest = generate_math_domain_world_manifest(
            'algebra_equations',
            seed=7,
            variant=0,
        )

        report = discover_domain_world_manifest(manifest)
        packed = report.to_dict()

        self.assertEqual('algebra_equations', report.domain_key)
        self.assertEqual(1.0, packed['benchmark_coverage'])
        self.assertTrue(any(
            'missing input reverses the steps' in equation['expression']
            for equation in packed['self_authored_equations']
        ))
        self.assertFalse(packed['leaked_manifest'])

    def test_domain_curriculum_preview_is_non_final_and_lists_worlds(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            blueprints = run_domain_curriculum_preview(CumulativeTheoryMemory())

        printed = output.getvalue()

        self.assertEqual(len(MATH_DOMAIN_CURRICULUM), len(blueprints))
        self.assertIn('DOMAIN CURRICULUM WORLD PREVIEW', printed)
        self.assertIn('Final discovery run: not run', printed)
        self.assertIn('Cand', printed)
        self.assertIn('100%', printed)
        self.assertIn('arithmetic_quantity', printed)
        self.assertIn('higher_dimensions', printed)


if __name__ == '__main__':
    unittest.main()
