import contextlib
import copy
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'first_principles_ai'))
sys.path.insert(0, PROJECT_DIR)

from agent.abstraction_replay_schema import (  # noqa: E402
    AbstractionReplayArtifactValidationError,
    artifact_content_hash,
    validate_abstraction_replay_artifact,
)
from main import (  # noqa: E402
    run_bounded_abstraction_transfer_negative_control_matrix,
    run_bounded_abstraction_transfer_negative_control_sweep,
    run_bounded_abstraction_transfer_replay_pack,
    run_bounded_abstraction_transfer_replay_matrix,
)


class AbstractionReplaySchemaTests(unittest.TestCase):
    def _artifact_from_runner(self, runner, **kwargs):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / 'artifact.json'
            with contextlib.redirect_stdout(io.StringIO()):
                result = runner(
                    seed_start=7,
                    steps=90,
                    object_count=5,
                    target_world_types=[
                        'standard',
                        'time_varying',
                        'hidden_procedural',
                    ],
                    output_file=artifact_path,
                    **kwargs,
                )
            artifact = json.loads(artifact_path.read_text(encoding='utf-8'))
        return result, artifact

    def test_valid_pack_matrix_and_negative_control_artifacts(self):
        cases = [
            (
                run_bounded_abstraction_transfer_replay_pack,
                'bounded_abstraction_transfer_replay_pack',
                'replay_candidate_benefit',
            ),
            (
                run_bounded_abstraction_transfer_replay_matrix,
                'bounded_abstraction_transfer_replay_matrix',
                'candidate_replay_matrix_evidence',
            ),
            (
                run_bounded_abstraction_transfer_negative_control_matrix,
                'bounded_abstraction_transfer_negative_control_matrix',
                'candidate_replay_negative_control_evidence',
            ),
            (
                run_bounded_abstraction_transfer_negative_control_sweep,
                'bounded_abstraction_transfer_negative_control_sweep',
                'candidate_replay_negative_control_sweep_evidence',
            ),
        ]
        for runner, run_kind, evidence_type in cases:
            with self.subTest(run_kind=run_kind):
                result, artifact = self._artifact_from_runner(runner)
                validation = validate_abstraction_replay_artifact(artifact)

                self.assertEqual(run_kind, validation['run_kind'])
                self.assertEqual(evidence_type, validation['evidence_type'])
                self.assertEqual(artifact['artifact_content_hash'], validation['content_hash'])
                self.assertEqual(validation, result['artifact_schema_validation'])
                self.assertFalse(artifact['mutates_runtime_theory_memory'])
                self.assertFalse(artifact['project_owned_checkpoint_claimed'])
                self.assertIn('project-owned', artifact['candidate_not_causal_wording'])

    def test_negative_control_artifact_validates_hold_without_promotion(self):
        result, artifact = self._artifact_from_runner(
            run_bounded_abstraction_transfer_negative_control_matrix
        )
        validation = validate_abstraction_replay_artifact(artifact)

        self.assertTrue(validation['valid'])
        self.assertEqual('hold_for_more_evidence', artifact['decision']['decision'])
        self.assertFalse(artifact['decision']['promote_bridge'])
        self.assertEqual('hold_for_more_evidence', result['decision']['decision'])
        self.assertFalse(result['decision']['promote_bridge'])
        self.assertEqual(0.4, artifact['candidate_win_rate'])
        self.assertEqual(0.4, artifact['candidate_survives_negative_controls_rate'])

    def test_negative_control_sweep_validates_schema_and_holds(self):
        result, artifact = self._artifact_from_runner(
            run_bounded_abstraction_transfer_negative_control_sweep,
            seed_count=2,
        )
        validation = validate_abstraction_replay_artifact(artifact)

        self.assertTrue(validation['valid'])
        self.assertEqual('bounded_abstraction_transfer_negative_control_sweep', result['run_kind'])
        self.assertEqual('candidate_replay_negative_control_sweep_evidence', artifact['evidence_type'])
        self.assertEqual(2, result['matrix_count'])
        self.assertEqual(10, result['comparison_count'])
        self.assertEqual(10, artifact['aggregate_counts']['comparison_count'])
        self.assertEqual(4, artifact['aggregate_counts']['candidate_win_count'])
        self.assertEqual(4, artifact['aggregate_counts']['candidate_survives_negative_controls_count'])
        self.assertEqual(0, artifact['aggregate_counts']['shuffled_or_mismatched_control_win_count'])
        self.assertEqual(0.4, artifact['candidate_survives_negative_controls_rate'])
        self.assertEqual('hold_for_more_evidence', artifact['decision']['decision'])
        self.assertFalse(artifact['decision']['promote_bridge'])
        self.assertFalse(artifact['hf_validation_used'])
        self.assertFalse(artifact['hf_recommended_for_broader_sweep'])
        self.assertEqual(2, len(artifact['matrix_results']))
        self.assertIn('cheap HF', artifact['hf_repro_plan']['use_hf_when'])
        wording = artifact['candidate_not_causal_wording']
        self.assertIn('Candidate_replay_negative_control_sweep_evidence', wording)
        self.assertIn('do not establish causal proof', wording)
        self.assertIn('benchmark proof', wording)
        self.assertIn('project-owned model capability', wording)
        for row in artifact['comparisons']:
            self.assertEqual('no_bridge_plus_mismatched_bridge', row['negative_control_type'])
            self.assertEqual('mismatched_bridge_same_budget', row['shuffled_control_type'])
            self.assertTrue(row['source_matrix_id'])
            self.assertTrue(row['sweep_case_id'])

    def test_negative_control_sweep_is_deterministic_and_non_mutating(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / 'negative-control-sweep.json'
            runtime_memory = Path(tmpdir) / 'theory-memory.json'
            runtime_memory.write_text('{"runtime":"unchanged"}', encoding='utf-8')
            before_runtime = runtime_memory.read_text(encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()):
                first = run_bounded_abstraction_transfer_negative_control_sweep(
                    seed_start=4,
                    seed_count=2,
                    steps=90,
                    object_count=5,
                    target_world_types=[
                        'standard',
                        'time_varying',
                        'hidden_procedural',
                    ],
                    output_file=artifact_path,
                )
            first_artifact = json.loads(artifact_path.read_text(encoding='utf-8'))
            after_first_runtime = runtime_memory.read_text(encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()):
                second = run_bounded_abstraction_transfer_negative_control_sweep(
                    seed_start=4,
                    seed_count=2,
                    steps=90,
                    object_count=5,
                    target_world_types=[
                        'standard',
                        'time_varying',
                        'hidden_procedural',
                    ],
                    output_file=artifact_path,
                )
            second_artifact = json.loads(artifact_path.read_text(encoding='utf-8'))
            after_second_runtime = runtime_memory.read_text(encoding='utf-8')

        self.assertEqual(first['artifact_content_hash'], second['artifact_content_hash'])
        self.assertEqual(first['artifact_sha256'], second['artifact_sha256'])
        self.assertEqual(first_artifact, second_artifact)
        self.assertEqual(before_runtime, after_first_runtime)
        self.assertEqual(before_runtime, after_second_runtime)
        self.assertEqual('hold_for_more_evidence', first['decision']['decision'])

    def test_missing_no_overclaim_wording_fails_clearly(self):
        _, artifact = self._artifact_from_runner(run_bounded_abstraction_transfer_replay_matrix)
        malformed = copy.deepcopy(artifact)
        malformed['candidate_not_causal_wording'] = 'candidate looks good'
        malformed['artifact_content_hash'] = artifact_content_hash(malformed)

        with self.assertRaisesRegex(
            AbstractionReplayArtifactValidationError,
            'candidate_not_causal_wording: missing no-overclaim wording',
        ):
            validate_abstraction_replay_artifact(malformed)

    def test_missing_checkpoint_boundary_fails_clearly(self):
        _, artifact = self._artifact_from_runner(run_bounded_abstraction_transfer_replay_pack)
        malformed = copy.deepcopy(artifact)
        malformed.pop('project_owned_checkpoint_claimed')
        malformed['artifact_content_hash'] = artifact_content_hash(malformed)

        with self.assertRaisesRegex(
            AbstractionReplayArtifactValidationError,
            'project_owned_checkpoint_claimed: missing required field',
        ):
            validate_abstraction_replay_artifact(malformed)

    def test_missing_runtime_nonmutation_fails_clearly(self):
        _, artifact = self._artifact_from_runner(run_bounded_abstraction_transfer_replay_pack)
        malformed = copy.deepcopy(artifact)
        malformed.pop('mutates_runtime_theory_memory')
        malformed['artifact_content_hash'] = artifact_content_hash(malformed)

        with self.assertRaisesRegex(
            AbstractionReplayArtifactValidationError,
            'mutates_runtime_theory_memory: missing required field',
        ):
            validate_abstraction_replay_artifact(malformed)

    def test_malformed_aggregate_counts_fail_clearly(self):
        _, artifact = self._artifact_from_runner(run_bounded_abstraction_transfer_replay_matrix)
        malformed = copy.deepcopy(artifact)
        malformed['aggregate_counts']['candidate_win_count'] = '2'
        malformed['artifact_content_hash'] = artifact_content_hash(malformed)

        with self.assertRaisesRegex(
            AbstractionReplayArtifactValidationError,
            'aggregate_counts.candidate_win_count: aggregate count must be an integer',
        ):
            validate_abstraction_replay_artifact(malformed)

    def test_wrong_evidence_type_fails_clearly(self):
        _, artifact = self._artifact_from_runner(
            run_bounded_abstraction_transfer_negative_control_matrix
        )
        malformed = copy.deepcopy(artifact)
        malformed['evidence_type'] = 'candidate_replay_matrix_evidence'
        malformed['artifact_content_hash'] = artifact_content_hash(malformed)

        with self.assertRaisesRegex(
            AbstractionReplayArtifactValidationError,
            'evidence_type: wrong evidence type',
        ):
            validate_abstraction_replay_artifact(malformed)


if __name__ == '__main__':
    unittest.main()
