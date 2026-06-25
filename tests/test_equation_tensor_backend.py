import os
import sys
import unittest
from unittest.mock import patch


PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'first_principles_ai'))
sys.path.insert(0, PROJECT_DIR)

from agent.equation_tensor_backend import (
    fitted_scale_reduce,
    resolve_equation_scoring_backend,
    vector_scale_reduce,
)


class EquationTensorBackendTests(unittest.TestCase):
    def test_resolve_equation_scoring_backend_auto_prefers_cuda_when_available(self):
        with patch(
            'agent.equation_tensor_backend.available_equation_scoring_backends',
            return_value={
                'numpy': True,
                'torch': True,
                'cuda': True,
                'cuda_device': 'Mock CUDA',
            },
        ):
            status = resolve_equation_scoring_backend('auto')

        self.assertEqual('auto', status['requested_backend'])
        self.assertEqual('cuda', status['backend'])
        self.assertIsNone(status['fallback_reason'])
        self.assertTrue(status['available_backends']['cuda'])

    def test_resolve_equation_scoring_backend_cuda_falls_back_to_torch(self):
        with patch(
            'agent.equation_tensor_backend.available_equation_scoring_backends',
            return_value={
                'numpy': True,
                'torch': True,
                'cuda': False,
                'cuda_device': None,
            },
        ):
            status = resolve_equation_scoring_backend('cuda')

        self.assertEqual('cuda', status['requested_backend'])
        self.assertEqual('torch', status['backend'])
        self.assertEqual('cuda_unavailable_using_torch_cpu', status['fallback_reason'])

    def test_fitted_scale_reduce_falls_back_to_torch_on_cuda_error(self):
        calls = []

        def fake_torch_fit(features, targets, backend):
            calls.append(backend)
            if len(calls) == 1:
                raise RuntimeError('simulated cuda failure')
            return 1.23

        with patch('agent.equation_tensor_backend._torch_fitted_scale', side_effect=fake_torch_fit):
            value = fitted_scale_reduce([1.0, 2.0, 3.0], [2.0, 4.0, 6.0], backend='cuda')

        self.assertEqual(['cuda', 'torch'], calls)
        self.assertAlmostEqual(1.23, value)

    def test_vector_scale_reduce_falls_back_to_torch_on_cuda_error(self):
        calls = []

        def fake_torch_vector(x_features, y_features, x_targets, y_targets, backend):
            calls.append(backend)
            if len(calls) == 1:
                raise RuntimeError('simulated cuda failure')
            return -0.5

        with patch('agent.equation_tensor_backend._torch_vector_scale', side_effect=fake_torch_vector):
            value = vector_scale_reduce(
                [1.0, 2.0],
                [0.5, 1.0],
                [2.0, 4.0],
                [1.0, 2.0],
                backend='cuda',
            )

        self.assertEqual(['cuda', 'torch'], calls)
        self.assertAlmostEqual(-0.5, value)
