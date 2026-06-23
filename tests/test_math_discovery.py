import os
import sys
import unittest
import contextlib
import io


PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'first_principles_ai'))
sys.path.insert(0, PROJECT_DIR)

from agent.math_discovery import EmergentMathDiscovery
from agent.representation import ConceptType, KnowledgeBase
from main import _emergent_math_label_leaks, _math_metrics_from_knowledge, run_math_benchmark


def raw_state(*object_ids):
    objects = []
    for index, object_id in enumerate(object_ids):
        objects.append({
            'id': object_id,
            'position': (1.0 + index * 2.0, 2.0 + index),
            'velocity': (0.1 * (index + 1), -0.2),
            'mass': 1.0,
            'radius': 0.5,
        })
    return {'objects': objects}


class EmergentMathDiscoveryTests(unittest.TestCase):
    def test_discovers_internal_patterns_separate_from_human_math_labels(self):
        kb = KnowledgeBase()
        discovery = EmergentMathDiscovery(
            kb,
            concept_evidence_threshold=1,
            operation_evidence_threshold=1,
        )

        discovery.observe_transition(
            raw_state(1, 2),
            raw_state(1, 2, 3),
            {'type': 'spawn'},
            step=1,
        )

        emergent_concepts = [
            concept for concept in kb.get_all_concepts()
            if concept.properties.get('source') == 'emergent_math'
        ]
        descriptions = ' '.join(concept.description.lower() for concept in emergent_concepts)
        comparisons = {
            item.human_concept
            for item in discovery.compare_to_human_math()
        }

        self.assertTrue(emergent_concepts)
        self.assertNotIn('addition', descriptions)
        self.assertNotIn('subtraction', descriptions)
        self.assertNotIn('successor', descriptions)
        self.assertNotIn('predecessor', descriptions)
        self.assertIn('successor-like operation', comparisons)

    def test_discovers_inverse_and_repeatable_extent_operations(self):
        kb = KnowledgeBase()
        discovery = EmergentMathDiscovery(
            kb,
            concept_evidence_threshold=1,
            operation_evidence_threshold=1,
        )

        discovery.observe_transition(raw_state(1, 2), raw_state(1, 2, 3), {'type': 'spawn'}, step=1)
        discovery.observe_transition(raw_state(1, 2, 3), raw_state(1, 2, 3, 4), {'type': 'spawn'}, step=2)
        discovery.observe_transition(raw_state(1, 2, 3, 4), raw_state(1, 2, 3), {'type': 'remove'}, step=3)

        comparisons = {
            item.human_concept
            for item in discovery.compare_to_human_math()
        }
        operations = [
            concept for concept in kb.get_concepts_by_type(ConceptType.OPERATION)
            if concept.properties.get('source') == 'emergent_math'
        ]
        confirmed_operation_rules = [
            rule for rule in kb.get_confirmed_rules()
            if rule.properties.get('hypothesis_type') == 'emergent_math_operation'
        ]

        self.assertIn('successor-like operation', comparisons)
        self.assertIn('predecessor-like operation', comparisons)
        self.assertIn('iteration / composition', comparisons)
        self.assertIn('inverse operation pair', comparisons)
        self.assertTrue(operations)
        self.assertTrue(confirmed_operation_rules)
        self.assertTrue(all(op.notation not in ('+', '-', '=') for op in operations))

    def test_discovers_adjacent_transform_sequences_without_taught_composition_label(self):
        kb = KnowledgeBase()
        discovery = EmergentMathDiscovery(
            kb,
            concept_evidence_threshold=1,
            operation_evidence_threshold=1,
        )

        discovery.observe_transition(raw_state(1), raw_state(1, 2), {'type': 'spawn'}, step=1)
        discovery.observe_transition(raw_state(1, 2), raw_state(1, 2, 3), {'type': 'spawn'}, step=2)

        comparisons = {
            item.human_concept
            for item in discovery.compare_to_human_math()
        }
        emergent_descriptions = ' '.join(
            concept.description.lower()
            for concept in kb.get_all_concepts()
            if concept.properties.get('source') == 'emergent_math'
        )
        sequence_patterns = [
            pattern for pattern in discovery.discovered_patterns()
            if pattern.properties.get('structural_role') == 'paired_step_transform'
        ]

        self.assertIn('operation composition', comparisons)
        self.assertTrue(sequence_patterns)
        self.assertEqual(2, sequence_patterns[0].properties['net_delta'])
        self.assertNotIn('composition', emergent_descriptions)
        self.assertNotIn('addition', emergent_descriptions)

    def test_discovers_returning_sequences_as_identity_like_only_in_comparison(self):
        kb = KnowledgeBase()
        discovery = EmergentMathDiscovery(
            kb,
            concept_evidence_threshold=1,
            operation_evidence_threshold=1,
        )

        discovery.observe_transition(raw_state(1), raw_state(1, 2), {'type': 'spawn'}, step=1)
        discovery.observe_transition(raw_state(1, 2), raw_state(1), {'type': 'remove'}, step=2)

        comparisons = {
            item.human_concept
            for item in discovery.compare_to_human_math()
        }
        emergent_descriptions = ' '.join(
            concept.description.lower()
            for concept in kb.get_all_concepts()
            if concept.properties.get('source') == 'emergent_math'
        )

        self.assertIn('identity-like cancellation', comparisons)
        self.assertIn('operation composition', comparisons)
        self.assertNotIn('identity', emergent_descriptions)
        self.assertNotIn('cancellation', emergent_descriptions)

    def test_discovers_same_net_sequence_classes_without_taught_equivalence_label(self):
        kb = KnowledgeBase()
        discovery = EmergentMathDiscovery(
            kb,
            concept_evidence_threshold=1,
            operation_evidence_threshold=1,
        )

        discovery.observe_transition(raw_state(1), raw_state(1, 2), {'type': 'spawn'}, step=1)
        discovery.observe_transition(raw_state(1, 2), raw_state(1), {'type': 'remove'}, step=2)
        discovery.observe_transition(raw_state(1), raw_state(1, 2), {'type': 'spawn'}, step=3)

        comparisons = {
            item.human_concept
            for item in discovery.compare_to_human_math()
        }
        emergent_descriptions = ' '.join(
            concept.description.lower()
            for concept in kb.get_all_concepts()
            if concept.properties.get('source') == 'emergent_math'
        )
        same_net_patterns = [
            pattern for pattern in discovery.discovered_patterns()
            if pattern.properties.get('structural_role') == 'same_net_sequence_set'
        ]
        swapped_patterns = [
            pattern for pattern in discovery.discovered_patterns()
            if pattern.properties.get('structural_role') == 'swapped_sequence_same_net'
        ]

        self.assertIn('operation equivalence class', comparisons)
        self.assertIn('commutativity-like operation behavior', comparisons)
        self.assertEqual(0, same_net_patterns[0].properties['net_delta'])
        self.assertEqual(2, same_net_patterns[0].properties['sequence_count'])
        self.assertTrue(swapped_patterns)
        self.assertNotIn('equivalence', emergent_descriptions)
        self.assertNotIn('commutative', emergent_descriptions)
        self.assertNotIn('commutativity', emergent_descriptions)

    def test_summary_reports_internal_patterns_and_comparison_candidates(self):
        kb = KnowledgeBase()
        discovery = EmergentMathDiscovery(
            kb,
            concept_evidence_threshold=1,
            operation_evidence_threshold=1,
        )

        discovery.observe_transition(raw_state(1, 2), raw_state(1, 2), {'type': 'wait'}, step=1)
        summary = discovery.summary()

        self.assertIn('Agent-internal structures', summary)
        self.assertIn('Human-math comparison candidates', summary)
        self.assertIn('identity / object permanence', summary)

    def test_discovers_metric_scale_symmetry_and_mapping_without_taught_labels(self):
        kb = KnowledgeBase()
        discovery = EmergentMathDiscovery(
            kb,
            concept_evidence_threshold=1,
            operation_evidence_threshold=1,
        )

        discovery.observe_transition(raw_state(1, 2, 3), raw_state(1, 2, 3), {'type': 'wait'}, step=1)
        discovery.observe_transition(raw_state(1), raw_state(1, 2), {'type': 'spawn'}, step=2)
        discovery.observe_transition(raw_state(1), raw_state(1, 2), {'type': 'spawn'}, step=3)

        comparisons = {
            item.human_concept
            for item in discovery.compare_to_human_math()
        }
        emergent_descriptions = ' '.join(
            concept.description.lower()
            for concept in kb.get_all_concepts()
            if concept.properties.get('source') == 'emergent_math'
        )

        self.assertIn('metric / distance structure', comparisons)
        self.assertIn('ratio-like relation', comparisons)
        self.assertIn('symmetry-like balance', comparisons)
        self.assertIn('function-like mapping', comparisons)
        self.assertIn('conditional rule', comparisons)
        self.assertNotIn('function-like', emergent_descriptions)
        self.assertNotIn('ratio-like', emergent_descriptions)
        self.assertNotIn('symmetry-like', emergent_descriptions)

    def test_discovers_periodicity_like_recurrence_without_taught_label(self):
        kb = KnowledgeBase()
        discovery = EmergentMathDiscovery(
            kb,
            concept_evidence_threshold=1,
            operation_evidence_threshold=1,
        )

        states = [
            {
                'objects': [{
                    'id': 1,
                    'position': (1.0, 1.0),
                    'velocity': (1.0, 0.0),
                    'mass': 1.0,
                    'radius': 0.5,
                }]
            },
            {
                'objects': [{
                    'id': 1,
                    'position': (1.1, 1.0),
                    'velocity': (-1.0, 0.0),
                    'mass': 1.0,
                    'radius': 0.5,
                }]
            },
            {
                'objects': [{
                    'id': 1,
                    'position': (1.0, 1.0),
                    'velocity': (1.0, 0.0),
                    'mass': 1.0,
                    'radius': 0.5,
                }]
            },
            {
                'objects': [{
                    'id': 1,
                    'position': (1.1, 1.0),
                    'velocity': (-1.0, 0.0),
                    'mass': 1.0,
                    'radius': 0.5,
                }]
            },
        ]
        discovery.observe_transition(states[0], states[1], {'type': 'wait'}, step=1)
        discovery.observe_transition(states[1], states[2], {'type': 'wait'}, step=2)
        discovery.observe_transition(states[2], states[3], {'type': 'wait'}, step=3)

        comparisons = {
            item.human_concept
            for item in discovery.compare_to_human_math()
        }
        emergent_descriptions = ' '.join(
            concept.description.lower()
            for concept in kb.get_all_concepts()
            if concept.properties.get('source') == 'emergent_math'
        )

        self.assertIn('periodicity-like recurrence', comparisons)
        self.assertNotIn('period', emergent_descriptions)

    def test_math_metrics_measure_coverage_and_agent_label_leaks(self):
        kb = KnowledgeBase()
        discovery = EmergentMathDiscovery(
            kb,
            concept_evidence_threshold=1,
            operation_evidence_threshold=1,
        )
        kb.emergent_math_discovery = discovery

        discovery.observe_transition(raw_state(1), raw_state(1, 2), {'type': 'spawn'}, step=1)
        kb.add_concept(
            ConceptType.PATTERN,
            'leaky addition label in agent-facing concept',
            'raw_pattern:leaky',
            step=2,
            properties={'source': 'emergent_math'},
        )

        metrics = _math_metrics_from_knowledge(
            kb,
            required_concepts={'successor-like operation'},
        )
        leaks = _emergent_math_label_leaks(kb)

        self.assertEqual(1.0, metrics['coverage'])
        self.assertFalse(metrics['passed'])
        self.assertEqual(1, len(leaks))
        self.assertIn('addition', leaks[0]['labels'])

    def test_math_benchmark_runs_real_experiment_and_reports_coverage(self):
        required = {
            'discrete quantity / cardinality',
            'identity / object permanence',
            'order relation',
        }

        with contextlib.redirect_stdout(io.StringIO()):
            results = run_math_benchmark(
                seeds=1,
                steps=80,
                object_counts=[3],
                world_types=['standard'],
                num_agents=2,
                required_concepts=required,
            )

        self.assertEqual(1, len(results))
        self.assertTrue(required <= results[0]['human_concepts'])
        self.assertEqual([], results[0]['label_leaks'])
        self.assertTrue(results[0]['passed'])


if __name__ == '__main__':
    unittest.main()
