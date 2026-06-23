import os
import sys
import unittest


PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'first_principles_ai'))
sys.path.insert(0, PROJECT_DIR)

from agent.language import CommunicationSystem, MultiAgentExperiment, Signal
from agent.math_discovery import RawMathPattern


class LanguageGroundingTests(unittest.TestCase):
    def test_agents_prefer_emergent_math_meanings_over_feature_fallbacks(self):
        experiment = MultiAgentExperiment(num_agents=2, seed=1)
        pattern = RawMathPattern(
            key='raw_operator:intervention:spawn:extent_delta:1',
            kind='operation',
            description='internal transform',
            first_step=1,
            evidence=10,
            concept_name='C_009',
        )

        signals = experiment.step_agents(
            features={
                'count': 5,
                'mean_speed': 2.0,
                'total_kinetic_energy': 20.0,
            },
            had_collision=True,
            step=1,
            math_patterns=[pattern],
        )

        self.assertTrue(signals)
        self.assertTrue(all(signal.meaning == 'math:C_009' for signal in signals))
        self.assertTrue(all(signal.context['_meaning_source'] == 'emergent_math' for signal in signals))
        self.assertTrue(all('_math_grounding' not in signal.context for signal in signals))
        self.assertTrue(all(signal.context['_meaning_candidates'] == ['math:C_009'] for signal in signals))
        self.assertEqual(2, experiment.math_grounded_signal_count)
        self.assertEqual(0, experiment.feature_fallback_signal_count)
        self.assertEqual(
            'raw_operator:intervention:spawn:extent_delta:1',
            experiment.math_meaning_sources['math:C_009']['pattern_key'],
        )

    def test_receiver_infers_from_context_without_trusting_sender_meaning(self):
        receiver = CommunicationSystem(agent_id=1)
        signal = Signal(
            sender_id=0,
            receiver_id=1,
            symbol='s0',
            meaning='math:WRONG_PRIVATE_INTENT',
            step=1,
            context={'_meaning_candidates': ['math:C_001']},
        )

        inferred = receiver.receive_signal(signal)

        self.assertEqual('math:C_001', inferred)
        self.assertEqual('s0', receiver.meaning_to_symbol['math:C_001'])
        self.assertNotIn('math:WRONG_PRIVATE_INTENT', receiver.meaning_to_symbol)
        self.assertNotIn(
            'math:WRONG_PRIVATE_INTENT',
            receiver.symbol_to_meanings['s0'],
        )

    def test_receiver_reinforces_inference_when_math_meaning_matches_transition(self):
        receiver = CommunicationSystem(agent_id=1)
        signal = Signal(
            sender_id=0,
            receiver_id=1,
            symbol='s0',
            meaning='math:PRIVATE',
            step=7,
            context={'_meaning_candidates': ['math:C_008']},
        )

        receiver.receive_signal(signal)
        success = receiver.reinforce_inference(
            signal,
            active_meanings={'math:C_008'},
            action_relevant_meanings={'math:C_008'},
        )

        self.assertTrue(success)
        self.assertEqual(1, receiver.successful_inference_count)
        self.assertEqual(0, receiver.failed_inference_count)
        self.assertEqual(1, receiver.action_useful_inference_count)
        self.assertEqual('success', signal.context['_receiver_feedback'])
        self.assertTrue(signal.context['_receiver_action_useful'])
        self.assertGreater(receiver.symbol_to_meanings['s0']['math:C_008'], 1.0)

    def test_receiver_does_not_reinforce_inactive_math_guess(self):
        receiver = CommunicationSystem(agent_id=1)
        signal = Signal(
            sender_id=0,
            receiver_id=1,
            symbol='s0',
            meaning='math:PRIVATE',
            step=7,
            context={'_meaning_candidates': ['math:C_OLD']},
        )

        receiver.receive_signal(signal)
        success = receiver.reinforce_inference(signal, active_meanings={'math:C_NEW'})

        self.assertFalse(success)
        self.assertEqual(0, receiver.successful_inference_count)
        self.assertEqual(1, receiver.failed_inference_count)
        self.assertEqual('failed', signal.context['_receiver_feedback'])

    def test_experiment_counts_action_useful_math_grounding(self):
        experiment = MultiAgentExperiment(num_agents=2, seed=1)
        pattern = RawMathPattern(
            key='raw_operator:intervention:spawn:extent_delta:1',
            kind='operation',
            description='internal transform',
            first_step=1,
            evidence=10,
            last_step=7,
            properties={'structural_role': 'intervention_transform', 'action_type': 'spawn', 'delta': 1},
            concept_name='C_008',
        )

        experiment.step_agents({}, False, step=7, math_patterns=[pattern])

        self.assertEqual(2, experiment.successful_grounding_count)
        self.assertEqual(0, experiment.failed_grounding_count)
        self.assertEqual(2, experiment.action_useful_signal_count)

    def test_agents_use_feature_fallback_only_before_math_concepts_exist(self):
        experiment = MultiAgentExperiment(num_agents=2, seed=1)

        signals = experiment.step_agents(
            features={'count': 5, 'mean_speed': 2.0, 'total_kinetic_energy': 20.0},
            had_collision=False,
            step=1,
            math_patterns=[],
        )

        self.assertTrue(signals)
        self.assertTrue(all(not signal.meaning.startswith('math:') for signal in signals))
        self.assertTrue(all(signal.context['_meaning_source'] == 'feature_fallback' for signal in signals))
        self.assertTrue(all(signal.context['_meaning_candidates'] for signal in signals))
        self.assertEqual(0, experiment.math_grounded_signal_count)
        self.assertEqual(2, experiment.feature_fallback_signal_count)

    def test_summary_reports_math_grounding_sources(self):
        experiment = MultiAgentExperiment(num_agents=2, seed=1)
        pattern = RawMathPattern(
            key='raw_pattern:token_continuity',
            kind='relation',
            description='internal persistence',
            first_step=1,
            evidence=12,
            concept_name='C_001',
        )

        experiment.step_agents({}, False, step=1, math_patterns=[pattern])
        summary = experiment.summary()

        self.assertIn('Math-grounded meanings', summary)
        self.assertIn('math:C_001 grounded in raw_pattern:token_continuity', summary)
        self.assertIn('Feature-fallback signals: 0', summary)


if __name__ == '__main__':
    unittest.main()
