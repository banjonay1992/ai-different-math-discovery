import contextlib
import io
import json
import os
import sys
import tempfile
import unittest


PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'first_principles_ai'))
sys.path.insert(0, PROJECT_DIR)

from agent.compute_budget import plan_adaptive_compute_budget
from agent.discovery_loop import (
    CumulativeTheoryMemory,
    MATH_DOMAIN_CURRICULUM,
    MATH_DOMAIN_TRANSFER_BRIDGES,
)
from agent.arithmetic_rediscovery import run_arithmetic_rediscovery_probe
from agent.domain_world_discovery import (
    build_domain_transfer_evidence,
    discover_all_domain_worlds,
    discover_domain_world_manifest,
)
from main import (
    run_autonomous_scientist_loop,
    run_domain_curriculum_preview,
    run_domain_world_discovery_ingest,
    run_hf_adaptive_comparison,
    run_hf_non_final_campaign,
)
from world.math_domain_worlds import (
    DOMAIN_WORLD_GENERATORS,
    FORBIDDEN_OBSERVATION_KEYS,
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
                self.assertGreaterEqual(len(manifest.samples), 3)
                self.assertTrue(manifest.expected_discoveries)
                self.assertTrue(manifest.falsifiers)
                self.assertTrue(manifest.transfer_targets)
                self.assertEqual(len(manifest.samples), len(manifest.observations()))
                self.assertIn(
                    'primitive_contrast',
                    {
                        observation['observation_kind']
                        for observation in manifest.observations()
                    },
                )

                for observation in manifest.observations():
                    self.assertFalse(math_domain_manifest_from_observation(observation))
                    self.assertIn('sample_id', observation)
                    self.assertIn('observation_kind', observation)
                    self.assertNotIn('domain_key', observation)
                    self.assertNotIn('expected_discoveries', observation)
                    self.assertNotIn('falsifiers', observation)
                    self.assertTrue(
                        FORBIDDEN_OBSERVATION_KEYS.isdisjoint(observation.keys())
                    )

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
        rediscovery_experiments = memory.domain_rediscovery_experiments(limit=20)
        readiness = memory.discovery_readiness_report()
        packed = memory.to_dict()
        summary = memory.summary()

        required = {domain['key'] for domain in MATH_DOMAIN_CURRICULUM}
        self.assertEqual(required, {item['domain_key'] for item in blueprints})
        self.assertTrue(all(item['sample_count'] > 0 for item in blueprints))
        self.assertTrue(all(item['sample_count'] >= 3 for item in blueprints))
        self.assertTrue(all(item['falsifier_count'] > 0 for item in blueprints))
        self.assertTrue(all(not item['leaks_benchmark_truth'] for item in blueprints))
        self.assertTrue(all(
            item['blind_observation_policy']['withhold_benchmark_truth']
            and item['blind_observation_policy']['score_after_candidate_generation']
            and item['public_sample_preview']
            for item in blueprints
        ))
        for item in blueprints:
            with self.subTest(policy=item['domain_key']):
                for observation in item['public_sample_preview']:
                    self.assertFalse(math_domain_manifest_from_observation(observation))
        self.assertEqual(required, {item['domain_key'] for item in discoveries})
        self.assertTrue(all(item['candidate_count'] > 0 for item in discoveries))
        self.assertTrue(all(item['self_authored_equations'] for item in discoveries))
        self.assertTrue(all(
            any(
                candidate['observation_kind'] == 'primitive_contrast'
                for candidate in item['candidates']
            )
            for item in discoveries
        ))
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
        self.assertEqual(required, {item['domain_key'] for item in rediscovery_experiments})
        for item in rediscovery_experiments:
            with self.subTest(rediscovery=item['domain_key']):
                self.assertNotIn('expected_discoveries', item)
                self.assertNotIn('comparison_hits', item)
                self.assertNotIn('missing_comparison_tags', item)
                self.assertTrue(item['experiment_templates'])
                self.assertTrue(
                    item['blind_observation_policy']['withhold_benchmark_truth']
                )
                for observation in item['public_sample_preview']:
                    self.assertFalse(math_domain_manifest_from_observation(observation))
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
        self.assertTrue(
            readiness['gates']['baseline_experiment_templates']['passed']
        )
        self.assertIn('domain_world_blueprints', readiness)
        self.assertIn('domain_world_discoveries', readiness)
        self.assertIn('domain_world_transfer_evidence', readiness)
        self.assertIn('baseline_experiment_templates', readiness)
        self.assertIn('domain_world_blueprints', packed)
        self.assertIn('domain_world_discoveries', packed)
        self.assertIn('domain_world_transfer_evidence', packed)
        self.assertIn('baseline_experiment_templates', packed)
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
        self.assertIn('Baseline experiment templates:', summary)

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
                self.assertTrue(any(
                    candidate['observation_kind'] == 'primitive_contrast'
                    and candidate['falsification_tests']
                    for candidate in packed['candidates']
                ))
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

    def test_domain_discovery_ingest_persists_curriculum_evidence(self):
        memory = CumulativeTheoryMemory()

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            records = run_domain_world_discovery_ingest(memory, seed=2, variant=1)

        text = output.getvalue()
        agenda = memory.domain_curriculum_agenda(limit=20)
        packed = memory.to_dict()
        restored = CumulativeTheoryMemory.from_dict(packed)

        self.assertEqual(len(MATH_DOMAIN_CURRICULUM), len(records))
        self.assertIn('DOMAIN WORLD DISCOVERY INGEST', text)
        self.assertEqual(
            len(MATH_DOMAIN_CURRICULUM),
            len(memory.domain_world_records),
        )
        self.assertEqual(
            {'transfer_ready'},
            {item['status'] for item in agenda},
        )
        self.assertTrue(all(
            item['evidence']['domain_world_record_count'] >= 1
            and item['evidence']['domain_world_equations']
            for item in agenda
        ))
        self.assertIn('domain_world_records', packed)
        self.assertEqual(
            len(memory.domain_world_records),
            len(restored.domain_world_records),
        )

    def test_hf_non_final_campaign_writes_json_artifact_without_final_run(self):
        memory = CumulativeTheoryMemory()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, 'hf-report.json')
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = run_hf_non_final_campaign(
                    theory_memory=memory,
                    output_file=output_file,
                    include_prep=False,
                )

            text = output.getvalue()
            with open(output_file, 'r', encoding='utf-8') as handle:
                artifact = json.load(handle)

        self.assertFalse(result['runs_final'])
        self.assertEqual('hf_non_final_campaign', result['run_kind'])
        self.assertEqual(len(MATH_DOMAIN_CURRICULUM), result['domain_world_record_count'])
        self.assertEqual(0, result['prep_result_count'])
        self.assertIn('autonomous_scientist_report', result)
        self.assertGreater(
            result['autonomous_scientist_report']['coverage']['robust_invariant_count'],
            0,
        )
        self.assertIn('HF_PROGRESS', text)
        self.assertIn('domain_world_discoveries_recorded', text)
        self.assertIn('SCIENTIST_EVENT', text)
        self.assertEqual('hf_non_final_campaign', artifact['run_kind'])
        self.assertIn('theory_memory', artifact)
        self.assertIn('autonomous_scientist_report', artifact)
        self.assertEqual(
            len(MATH_DOMAIN_CURRICULUM),
            len(artifact['theory_memory']['domain_world_records']),
        )
        self.assertTrue(artifact['theory_memory']['autonomous_scientist_records'])
        self.assertIn('compute_budget_plan', artifact)
        self.assertIn('resource_efficiency', artifact)
        self.assertIn('compaction_events', artifact)
        self.assertIn('arithmetic_rediscovery_report', artifact)
        self.assertEqual(
            'arithmetic_ready',
            artifact['arithmetic_rediscovery_report']['status'],
        )
        self.assertTrue(
            artifact['theory_memory']['arithmetic_rediscovery_records']
        )
        self.assertIn(
            'canonical_law_compression',
            artifact['resource_efficiency'],
        )

    def test_arithmetic_probe_rediscovers_counting_without_manifest_leak(self):
        report = run_arithmetic_rediscovery_probe(
            seed_start=0,
            seed_count=2,
            variants=(0, 1),
        )

        self.assertFalse(report['runs_final'])
        self.assertEqual('arithmetic_ready', report['status'])
        self.assertEqual(1.0, report['coverage'])
        self.assertFalse(report['leaked_manifest'])
        self.assertEqual(
            {
                'cardinality_invariance',
                'addition_as_composition',
                'permutation_invariance',
                'successor_step',
                'predecessor_step',
            },
            set(report['discovered_targets']),
        )
        self.assertTrue(all(
            item['expression']
            and item['falsification_tests']
            and item['proof_obligations']
            for item in report['self_authored_equations']
        ))

    def test_memory_compacts_canonical_laws_from_arithmetic_and_scientist(self):
        memory = CumulativeTheoryMemory()
        memory.record_domain_world_discoveries(seed=0, variant=0)
        memory.record_arithmetic_rediscovery(seed_start=0, seed_count=2, variants=(0, 1))
        memory.record_autonomous_scientist_loop(
            seed_start=0,
            seed_count=2,
            variants=(0, 1),
        )

        report = memory.compact_canonical_laws(source='test')
        packed = memory.to_dict()
        restored = CumulativeTheoryMemory.from_dict(packed)
        readiness = memory.discovery_readiness_report()

        self.assertGreater(report['canonical_law_shard_count'], 0)
        self.assertGreater(report['canonical_law_count'], 0)
        self.assertTrue(report['long_run_law_ready'])
        self.assertTrue(
            readiness['gates']['arithmetic_rediscovery_probe']['passed']
        )
        self.assertTrue(
            readiness['gates']['canonical_law_compression']['passed']
        )
        self.assertEqual(
            len(memory.canonical_law_shards),
            len(restored.canonical_law_shards),
        )

    def test_compute_budget_does_not_expand_without_residual_pressure(self):
        plan = plan_adaptive_compute_budget(
            readiness={'missing_gates': [], 'next_steps': []},
            scientist_report={
                'coverage': {},
                'next_actions': [],
                'harder_stress_worlds': [],
            },
            resource_report={
                'long_run_ready': True,
                'raw_operator_prior_outcome_count': 0,
                'bounded_windows': {'recommended_operator_window': 192},
            },
            requested_steps=80,
            requested_seeds=1,
            requested_hidden_worlds=0,
            max_steps=160,
            max_seeds=2,
            max_hidden_worlds=1,
        )

        self.assertFalse(plan['expanded'])
        self.assertEqual(plan['requested'], plan['effective'])
        self.assertEqual(0.0, plan['residual_pressure']['score'])

    def test_compute_budget_targets_high_value_worlds_before_expanding(self):
        plan = plan_adaptive_compute_budget(
            readiness={
                'missing_gates': [
                    'anomaly_repair_loop',
                    'operator_discovery_claims',
                    'claim_driven_planning',
                ],
                'next_steps': [],
            },
            scientist_report={
                'coverage': {
                    'residual_experiment_count': 24,
                    'stress_world_count': 5,
                },
                'next_actions': [{
                    'action_kind': 'run_harder_hidden_world',
                    'suggested_world_type': 'localized_gravity',
                }],
                'harder_stress_worlds': [
                    {
                        'key': 'localized_off_center_gravity',
                        'world_type': 'localized_gravity',
                        'priority': 0.92,
                    },
                    {
                        'key': 'time_varying_force',
                        'world_type': 'time_varying',
                        'priority': 0.78,
                    },
                ],
            },
            resource_report={
                'long_run_ready': True,
                'bounded_windows': {'operator_outcomes_within_window': True},
            },
            requested_steps=80,
            requested_seeds=1,
            requested_hidden_worlds=1,
            requested_world_types=[
                'standard',
                'inverse_square_repulsion',
                'localized_gravity',
                'time_varying',
            ],
            max_steps=160,
            max_seeds=2,
            max_hidden_worlds=2,
        )

        targeting = plan['targeting_plan']
        self.assertTrue(plan['targeted'])
        self.assertTrue(targeting['focused'])
        self.assertEqual(
            ['standard', 'localized_gravity', 'time_varying'],
            targeting['effective_world_types'],
        )
        self.assertEqual(plan['requested']['steps'], plan['effective']['steps'])
        self.assertLess(
            targeting['effective_case_count'],
            targeting['requested_case_count'],
        )

    def test_hf_campaign_expands_compute_only_from_residual_pressure(self):
        memory = CumulativeTheoryMemory()
        captured = {}

        def fake_prep_runner(**kwargs):
            captured.update(kwargs)
            return []

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = run_hf_non_final_campaign(
                theory_memory=memory,
                seeds=1,
                steps=40,
                object_counts=[3],
                world_types=['standard'],
                hidden_worlds=0,
                include_prep=True,
                live_scientist=False,
                max_adaptive_steps=80,
                max_adaptive_seeds=2,
                max_adaptive_hidden_worlds=1,
                prep_runner=fake_prep_runner,
            )

        plan = result['compute_budget_plan']
        self.assertTrue(plan['expanded'])
        self.assertGreater(plan['effective']['steps'], plan['requested']['steps'])
        self.assertEqual(plan['effective']['steps'], captured['steps'])
        self.assertEqual(plan['effective']['seeds'], captured['seeds'])
        self.assertIn('compute_budget_plan', output.getvalue())

    def test_hf_campaign_targets_prep_worlds_for_efficiency(self):
        memory = CumulativeTheoryMemory()
        captured = {}

        def fake_prep_runner(**kwargs):
            captured.update(kwargs)
            return [
                {
                    'context': world_type,
                    'seed': 0,
                    'objects': 3,
                    'steps': kwargs['steps'],
                    'ready_for_final': True,
                    'equation_passed': True,
                    'passed': True,
                }
                for world_type in kwargs['world_types']
            ]

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = run_hf_non_final_campaign(
                theory_memory=memory,
                seeds=1,
                steps=40,
                object_counts=[3],
                world_types=[
                    'standard',
                    'inverse_square_repulsion',
                    'localized_gravity',
                    'time_varying',
                ],
                hidden_worlds=1,
                include_prep=True,
                live_scientist=False,
                scientist_seed_count=2,
                scientist_variants=[0, 1],
                max_adaptive_steps=80,
                max_adaptive_seeds=2,
                max_adaptive_hidden_worlds=2,
                prep_runner=fake_prep_runner,
            )

        plan = result['compute_budget_plan']
        prep_plan = result['prep_execution_plan']
        self.assertTrue(plan['targeted'])
        self.assertTrue(prep_plan['targeted'])
        self.assertLess(len(captured['world_types']), 4)
        self.assertEqual(
            ['standard', 'localized_gravity', 'time_varying'],
            captured['world_types'],
        )
        self.assertLess(
            prep_plan['estimated_effective_compute_units'],
            prep_plan['estimated_requested_compute_units'],
        )
        self.assertIn('compute_targeted', output.getvalue())

    def test_hf_adaptive_comparison_runs_fixed_and_adaptive_from_same_snapshot(self):
        memory = CumulativeTheoryMemory()
        captured_steps = []

        def fake_prep_runner(**kwargs):
            captured_steps.append(kwargs['steps'])
            return [{
                'context': 'standard',
                'seed': index,
                'ready_for_final': True,
                'equation_passed': True,
                'passed': True,
            } for index in range(kwargs['seeds'])]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, 'comparison.json')
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                report = run_hf_adaptive_comparison(
                    theory_memory=memory,
                    seeds=1,
                    steps=40,
                    object_counts=[3],
                    world_types=['standard'],
                    hidden_worlds=0,
                    output_file=output_file,
                    scientist_seed_count=2,
                    scientist_variants=[0, 1],
                    live_scientist=False,
                    max_adaptive_steps=80,
                    max_adaptive_seeds=2,
                    max_adaptive_hidden_worlds=1,
                    prep_runner=fake_prep_runner,
                )

            with open(output_file, 'r', encoding='utf-8') as handle:
                artifact = json.load(handle)

        fixed, adaptive = report['variants']
        self.assertFalse(report['runs_final'])
        self.assertEqual('hf_adaptive_comparison', report['run_kind'])
        self.assertEqual(2, report['variant_count'])
        self.assertEqual('fixed_budget', fixed['variant'])
        self.assertEqual('adaptive_budget', adaptive['variant'])
        self.assertFalse(fixed['result']['compute_budget_plan']['expanded'])
        self.assertTrue(adaptive['result']['compute_budget_plan']['expanded'])
        self.assertLess(captured_steps[0], captured_steps[1])
        self.assertIn('adaptive_comparison_finish', output.getvalue())
        self.assertEqual('hf_adaptive_comparison', artifact['run_kind'])
        self.assertIn('comparison', artifact)
        self.assertIn('readiness_delta', artifact['comparison'])

    def test_hf_adaptive_comparison_can_improve_readiness_per_compute(self):
        memory = CumulativeTheoryMemory()

        def fake_prep_runner(**kwargs):
            results = []
            for world_type in kwargs['world_types']:
                for seed in range(kwargs['seeds']):
                    results.append({
                        'context': world_type,
                        'seed': seed,
                        'objects': 3,
                        'steps': kwargs['steps'],
                        'ready_for_final': True,
                        'equation_passed': True,
                        'passed': True,
                    })
            for index in range(kwargs['hidden_worlds']):
                results.append({
                    'context': f'hidden_{index:02d}',
                    'seed': index,
                    'objects': 3,
                    'steps': kwargs['steps'],
                    'ready_for_final': True,
                    'equation_passed': True,
                    'passed': True,
                })
            return results

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            report = run_hf_adaptive_comparison(
                theory_memory=memory,
                seeds=1,
                steps=40,
                object_counts=[3],
                world_types=[
                    'standard',
                    'inverse_square_repulsion',
                    'localized_gravity',
                    'time_varying',
                ],
                hidden_worlds=1,
                scientist_seed_count=2,
                scientist_variants=[0, 1],
                live_scientist=False,
                max_adaptive_steps=80,
                max_adaptive_seeds=2,
                max_adaptive_hidden_worlds=2,
                prep_runner=fake_prep_runner,
            )

        fixed, adaptive = report['variants']
        comparison = report['comparison']
        self.assertFalse(fixed['telemetry']['compute_targeted'])
        self.assertTrue(adaptive['telemetry']['compute_targeted'])
        self.assertLess(
            adaptive['telemetry']['compute_units'],
            fixed['telemetry']['compute_units'],
        )
        self.assertGreater(
            adaptive['telemetry']['readiness_per_compute_unit'],
            fixed['telemetry']['readiness_per_compute_unit'],
        )
        self.assertTrue(comparison['adaptive_improved_readiness_per_compute'])
        self.assertIn('recommendation', comparison)

    def test_hf_campaign_compacts_memory_between_batches(self):
        memory = CumulativeTheoryMemory()
        for index in range(12):
            memory.records.append({
                'context': 'standard',
                'seed': index,
                'phase': 'probe_ready',
                'theory_count': 2,
                'operator_count': 1,
                'proof_check_count': 1,
                'disagreement_mode': 'distance_exponent_race',
            })
            memory.operator_prior_outcomes.append({
                'context': 'standard',
                'seed': index,
                'operator_key': f'operator:inverse:{index % 2}',
                'operator_kind': 'inverse_separation_power',
                'outcome': 'confirmed' if index % 3 == 0 else 'unmatched',
                'best_score': 0.88 if index % 3 == 0 else 0.0,
                'matching_equation_count': 1 if index % 3 == 0 else 0,
                'parameters': {
                    'distance_exponent': 2.0,
                    'relation': 'direction',
                    'source_context': 'standard',
                },
            })

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = run_hf_non_final_campaign(
                theory_memory=memory,
                include_prep=False,
                live_scientist=False,
                compact_keep_records=4,
                compact_keep_operator_outcomes=4,
            )

        self.assertGreaterEqual(
            result['resource_efficiency']['compressed_shard_count'],
            1,
        )
        self.assertTrue(result['compaction_events'])
        self.assertIn('memory_compaction_checkpoint', output.getvalue())

    def test_autonomous_scientist_loop_adds_all_five_discovery_upgrades(self):
        memory = CumulativeTheoryMemory()

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            report = run_autonomous_scientist_loop(
                memory,
                seed_start=0,
                seed_count=2,
                variants=[0, 1],
                live=True,
                event_limit=80,
            )

        text = output.getvalue()
        coverage = report['coverage']
        readiness = memory.discovery_readiness_report()
        packed = memory.to_dict()
        restored = CumulativeTheoryMemory.from_dict(packed)

        self.assertFalse(report['runs_final'])
        self.assertEqual('autonomous_scientist_loop', report['run_kind'])
        self.assertGreater(coverage['robust_invariant_count'], 0)
        self.assertGreater(coverage['residual_experiment_count'], 0)
        self.assertGreaterEqual(coverage['stress_world_count'], 4)
        self.assertGreater(coverage['authored_equation_extension_count'], 0)
        self.assertGreater(coverage['live_event_count'], 0)
        self.assertIn('SCIENTIST_EVENT', text)
        self.assertTrue(any(
            event['event'] == 'invariant_consolidated'
            for event in report['live_events']
        ))
        self.assertTrue(any(
            item['status'] == 'robust_law'
            for item in report['invariant_consolidations']
        ))
        self.assertTrue(all(
            item['designed_next_experiment']
            and item['falsifies_if']
            for item in report['residual_experiments'][:5]
        ))
        self.assertTrue(any(
            item['key'] == 'higher_dimensional_projection'
            for item in report['harder_stress_worlds']
        ))
        self.assertTrue(all(
            item['expression']
            and item['proof_obligations']
            and item['falsification_tests']
            for item in report['authored_equation_extensions'][:5]
        ))
        for gate in (
            'scientist_invariant_consolidation',
            'scientist_residual_experiment_loop',
            'scientist_harder_hidden_worlds',
            'scientist_richer_equation_writing',
            'scientist_live_trace',
        ):
            self.assertTrue(readiness['gates'][gate]['passed'], gate)
        self.assertTrue(packed['autonomous_scientist_records'])
        self.assertTrue(packed['latest_autonomous_scientist_report'])
        self.assertEqual(
            len(memory.autonomous_scientist_records),
            len(restored.autonomous_scientist_records),
        )
        self.assertIn('Autonomous scientist loop:', memory.summary())

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
