import os
import sys
import unittest
from unittest.mock import patch


PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'first_principles_ai'))
sys.path.insert(0, PROJECT_DIR)

from world.environment import Environment
from world.physics import PhysicsObject, PhysicsWorld, Vec2
from world.tensor_backend import (
    available_force_backends,
    compute_external_force_deltas,
    resolve_force_backend,
)


def build_mixed_force_world(force_backend='python'):
    world = PhysicsWorld(force_backend=force_backend)
    world.gravity = 9.8
    world.friction = 0.999
    world.restitution = 0.85
    world.uniform_force = {'x': 8.0, 'y': -0.25}
    world.central_force = {'x': 10.0, 'y': 10.0, 'strength': 200.0}
    world.gravity_wells = [{
        'x': 7.0,
        'y': 13.0,
        'strength': 220.0,
        'radius': 8.5,
    }]
    world.repulsion_zones = [{
        'x': 13.0,
        'y': 8.0,
        'strength': 80.0,
        'radius': 6.0,
    }]
    world.inverse_square_repulsions = [{
        'x': 5.0,
        'y': 5.0,
        'strength': 130.0,
    }]
    world.vortex_fields = [{
        'x': 10.0,
        'y': 10.0,
        'strength': 140.0,
        'radius': 9.0,
        'direction': 1.0,
    }]
    world.time_varying_force = {
        'axis': 'x',
        'amplitude': 12.0,
        'period': 1.28,
        'phase': 0.0,
    }
    world.force_components = [
        {
            'type': 'cutoff_radial',
            'params': {
                'x': 11.0,
                'y': 6.0,
                'strength': 155.0,
                'radius': 6.0,
                'exponent': 2.0,
                'direction': -1.0,
            },
        },
        {
            'type': 'piecewise_radial',
            'params': {
                'x': 9.0,
                'y': 12.0,
                'split_radius': 6.0,
                'inner': {
                    'strength': 120.0,
                    'exponent': 1.0,
                    'direction': 1.0,
                },
                'outer': {
                    'strength': 70.0,
                    'exponent': 1.5,
                    'direction': -1.0,
                },
            },
        },
        {
            'type': 'periodic_regime_uniform',
            'params': {
                'period': 1.1,
                'phase': 0.0,
                'duty_cycle': 0.6,
                'on_force': {'x': 5.0, 'y': 0.0},
                'off_force': {'x': -2.0, 'y': 0.0},
            },
        },
    ]
    for index, (x, y, vx, vy) in enumerate([
        (3.0, 3.5, 0.2, -0.1),
        (6.0, 7.0, -0.15, 0.05),
        (9.0, 4.0, 0.1, 0.2),
        (14.0, 12.0, -0.2, -0.05),
        (17.0, 16.0, 0.05, -0.15),
    ]):
        world.add_object(
            PhysicsObject(
                position=Vec2(x, y),
                velocity=Vec2(vx, vy),
                mass=1.0,
                radius=0.05,
                object_id=index,
            )
        )
    return world


class TensorPhysicsBackendTests(unittest.TestCase):
    def test_resolve_force_backend_auto_prefers_cuda_when_available(self):
        with patch(
            'world.tensor_backend.available_force_backends',
            return_value={
                'numpy': True,
                'torch': True,
                'cuda': True,
                'cuda_device': 'Mock CUDA',
            },
        ):
            status = resolve_force_backend('auto')

        self.assertEqual('auto', status['requested_backend'])
        self.assertEqual('cuda', status['backend'])
        self.assertIsNone(status['fallback_reason'])
        self.assertTrue(status['available_backends']['cuda'])

    def test_resolve_force_backend_cuda_falls_back_to_torch(self):
        with patch(
            'world.tensor_backend.available_force_backends',
            return_value={
                'numpy': True,
                'torch': True,
                'cuda': False,
                'cuda_device': None,
            },
        ):
            status = resolve_force_backend('cuda')

        self.assertEqual('cuda', status['requested_backend'])
        self.assertEqual('torch', status['backend'])
        self.assertEqual('cuda_unavailable_using_torch_cpu', status['fallback_reason'])

    def test_resolve_force_backend_cuda_falls_back_to_numpy_when_torch_unavailable(self):
        with patch(
            'world.tensor_backend.available_force_backends',
            return_value={
                'numpy': True,
                'torch': False,
                'cuda': False,
                'cuda_device': None,
            },
        ):
            status = resolve_force_backend('cuda')

        self.assertEqual('cuda', status['requested_backend'])
        self.assertEqual('numpy', status['backend'])
        self.assertEqual('cuda_unavailable_using_numpy', status['fallback_reason'])

    def test_numpy_force_backend_matches_python_step(self):
        if not available_force_backends()['numpy']:
            self.skipTest('numpy backend unavailable')

        python_world = build_mixed_force_world(force_backend='python')
        numpy_world = build_mixed_force_world(force_backend='numpy')

        for _ in range(4):
            python_world.step()
            numpy_world.step()

        self.assertEqual('numpy', numpy_world.force_backend)
        for expected, actual in zip(python_world.objects, numpy_world.objects):
            self.assertAlmostEqual(expected.position.x, actual.position.x, places=9)
            self.assertAlmostEqual(expected.position.y, actual.position.y, places=9)
            self.assertAlmostEqual(expected.velocity.x, actual.velocity.x, places=9)
            self.assertAlmostEqual(expected.velocity.y, actual.velocity.y, places=9)

    def test_force_delta_kernel_matches_python_reference(self):
        if not available_force_backends()['numpy']:
            self.skipTest('numpy backend unavailable')

        python_world = build_mixed_force_world(force_backend='python')
        python_world.time = 0.016
        python_world._apply_external_forces_python(0.016)
        expected = [
            (obj.velocity.x - base_vx, obj.velocity.y - base_vy)
            for obj, base_vx, base_vy in zip(
                python_world.objects,
                [0.2, -0.15, 0.1, -0.2, 0.05],
                [-0.1, 0.05, 0.2, -0.05, -0.15],
            )
        ]

        numpy_world = build_mixed_force_world(force_backend='numpy')
        actual = compute_external_force_deltas(
            positions=[
                (obj.position.x, obj.position.y)
                for obj in numpy_world.objects
            ],
            dt=0.016,
            time_value=0.016,
            gravity=numpy_world.gravity,
            uniform_force=numpy_world.uniform_force,
            central_force=numpy_world.central_force,
            gravity_wells=numpy_world.gravity_wells,
            repulsion_zones=numpy_world.repulsion_zones,
            inverse_square_repulsions=numpy_world.inverse_square_repulsions,
            vortex_fields=numpy_world.vortex_fields,
            time_varying_force=numpy_world.time_varying_force,
            force_components=numpy_world.force_components,
            backend='numpy',
        )

        for (expected_x, expected_y), (actual_x, actual_y) in zip(expected, actual):
            self.assertAlmostEqual(expected_x, actual_x, places=9)
            self.assertAlmostEqual(expected_y, actual_y, places=9)

    def test_environment_accepts_force_backend_without_leaking_agent_labels(self):
        env = Environment(num_initial_objects=2, seed=1, force_backend='numpy')

        self.assertEqual('numpy', env.world.force_backend)
        observation = env.observe()
        self.assertNotIn('force_backend', observation)
        self.assertNotIn('world_type', observation)

    def test_cuda_request_falls_back_to_available_tensor_backend(self):
        status = resolve_force_backend('cuda')

        self.assertEqual('cuda', status['requested_backend'])
        self.assertIn(status['backend'], {'cuda', 'torch', 'numpy', 'python'})
        if status['backend'] != 'cuda':
            self.assertTrue(status['fallback_reason'])

    def test_cuda_force_deltas_fall_back_to_torch_path_on_cuda_error(self):
        calls = []

        def fake_torch_delta_call(**kwargs):
            calls.append(kwargs['backend'])
            if len(calls) == 1:
                raise RuntimeError('simulated cuda failure')
            return [(1.0, 2.0)] * len(kwargs['positions'])

        with patch('world.tensor_backend._torch_external_force_deltas', side_effect=fake_torch_delta_call):
            result = compute_external_force_deltas(
                positions=[(1.0, 2.0), (3.0, 4.0)],
                dt=0.01,
                time_value=0.0,
                gravity=1.0,
                uniform_force=None,
                central_force=None,
                gravity_wells=[],
                repulsion_zones=[],
                inverse_square_repulsions=[],
                vortex_fields=[],
                time_varying_force=None,
                force_components=[],
                backend='cuda',
            )

        self.assertEqual(['cuda', 'torch'], calls)
        self.assertEqual([(1.0, 2.0), (1.0, 2.0)], result)


if __name__ == '__main__':
    unittest.main()
