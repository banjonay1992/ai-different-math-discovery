from __future__ import annotations

"""
Environment — the agent's interface to the physics world.

The agent can:
  - OBSERVE: See the world state (positions, velocities, masses of all objects)
  - PUSH: Apply an impulse to an object
  - SPAWN: Create a new object
  - REMOVE: Remove an object
  - MOVE/FREEZE/DUPLICATE: Controlled causal interventions for theory tests
  - WAIT: Let the world evolve without intervention

The agent is NOT told what actions exist. It discovers them through exploration.
For the prototype, we provide the action space but the agent must learn what they do.
"""

import random
from typing import Callable, Optional
from .physics import PhysicsWorld
from .hidden_worlds import HiddenWorldManifest, apply_hidden_world


WorldConfigurator = Callable[[PhysicsWorld, float, float], None]
WORLD_CONFIGURATORS: dict[str, WorldConfigurator] = {}


def register_world_type(world_type: str, configurator: WorldConfigurator):
    """Register a benchmark world without putting labels in observations."""
    if not world_type:
        raise ValueError("world_type must be non-empty")
    WORLD_CONFIGURATORS[world_type] = configurator


def registered_world_types() -> tuple[str, ...]:
    return tuple(sorted(WORLD_CONFIGURATORS))


class Environment:
    """Wraps the physics world and provides a clean action/observation interface."""

    def __init__(self, width: float = 20.0, height: float = 20.0,
                 num_initial_objects: int = 5, seed: int = None,
                 world_type: str = 'standard',
                 hidden_manifest: HiddenWorldManifest | None = None):
        self.rng = random.Random(seed)
        self._benchmark_world_type = (
            'hidden_procedural' if hidden_manifest is not None else world_type
        )
        self._hidden_id = hidden_manifest.hidden_id if hidden_manifest is not None else None
        self.world = PhysicsWorld(width, height, rng=self.rng)
        self.world.world_type = self._benchmark_world_type
        self.step_count = 0

        if hidden_manifest is not None:
            apply_hidden_world(self.world, hidden_manifest)
        else:
            configurator = WORLD_CONFIGURATORS.get(world_type)
            if configurator is None:
                raise ValueError(f"Unknown world type: {world_type}")
            configurator(self.world, width, height)

        self._populate(num_initial_objects)

    def _populate(self, n: int):
        for _ in range(n):
            self.world.spawn_object(
                x=self.rng.uniform(2, self.world.width - 2),
                y=self.rng.uniform(2, self.world.height - 2),
                vx=self.rng.uniform(-3, 3),
                vy=self.rng.uniform(-3, 3),
            )

    def observe(self) -> dict:
        """Return the full observable state of the world."""
        return {
            'objects': self.world.get_state(),
            'collisions': self.world.get_collisions(),
            'time': round(self.world.time, 6),
            'step': self.step_count,
            'world_size': (self.world.width, self.world.height),
        }

    def benchmark_metadata(self) -> dict:
        """Return benchmark/report metadata that must stay outside agent input."""
        metadata = {'world_type': self._benchmark_world_type}
        if self._hidden_id is not None:
            metadata['hidden_id'] = self._hidden_id
        return metadata

    def step(self, action: dict = None) -> dict:
        """
        Execute one timestep. If an action is provided, apply it before stepping.

        Actions:
            {'type': 'wait'} — just observe
            {'type': 'push', 'object_id': int, 'fx': float, 'fy': float}
            {'type': 'spawn', 'x': float, 'y': float, 'vx': float, 'vy': float}
            {'type': 'remove', 'object_id': int}
            {'type': 'move', 'object_id': int, 'x': float, 'y': float, 'vx': float, 'vy': float}
            {'type': 'freeze', 'object_id': int}
            {'type': 'duplicate', 'object_id': int}
        """
        if action is None or action.get('type') == 'wait':
            pass
        elif action['type'] == 'push':
            self.world.apply_impulse(action['object_id'], action['fx'], action['fy'])
        elif action['type'] == 'spawn':
            self.world.spawn_object(
                action['x'], action['y'],
                action.get('vx', 0), action.get('vy', 0),
            )
        elif action['type'] == 'remove':
            self.world.remove_object(action['object_id'])
        elif action['type'] == 'move':
            self.world.move_object(
                action['object_id'],
                action['x'],
                action['y'],
                action.get('vx'),
                action.get('vy'),
            )
        elif action['type'] == 'freeze':
            self.world.freeze_object(action['object_id'])
        elif action['type'] == 'duplicate':
            self.world.duplicate_object(action['object_id'])

        self.world.step()
        self.step_count += 1
        return self.observe()

    def get_object_ids(self) -> list[int]:
        return [obj.id for obj in self.world.objects]

    def get_random_object_id(self) -> Optional[int]:
        ids = self.get_object_ids()
        return self.rng.choice(ids) if ids else None


def _standard_world(world: PhysicsWorld, width: float, height: float):
    return None


def _central_force_world(world: PhysicsWorld, width: float, height: float):
    world.central_force = {
        'x': width / 2,
        'y': height / 2,
        'strength': 200.0,
    }


def _repulsion_world(world: PhysicsWorld, width: float, height: float):
    world.repulsion_zones = [{
        'x': width / 2,
        'y': height / 2,
        'strength': 80.0,
        'radius': 6.0,
    }]


def _zero_gravity_world(world: PhysicsWorld, width: float, height: float):
    world.gravity = 0.0


def _sideways_wind_world(world: PhysicsWorld, width: float, height: float):
    world.uniform_force = {'x': 8.0, 'y': 0.0}


def _vortex_world(world: PhysicsWorld, width: float, height: float):
    world.vortex_fields = [{
        'x': width / 2,
        'y': height / 2,
        'strength': 140.0,
        'radius': 9.0,
        'direction': 1.0,
    }]


def _inverse_square_repulsion_world(world: PhysicsWorld, width: float, height: float):
    world.inverse_square_repulsions = [{
        'x': width / 2,
        'y': height / 2,
        'strength': 130.0,
    }]


def _localized_gravity_world(world: PhysicsWorld, width: float, height: float):
    world.gravity_wells = [{
        'x': width * 0.35,
        'y': height * 0.65,
        'strength': 260.0,
        'radius': 9.0,
    }]


def _time_varying_world(world: PhysicsWorld, width: float, height: float):
    world.time_varying_force = {
        'axis': 'x',
        'amplitude': 12.0,
        'period': 1.28,
        'phase': 0.0,
    }


register_world_type('standard', _standard_world)
register_world_type('central_force', _central_force_world)
register_world_type('repulsion', _repulsion_world)
register_world_type('zero_gravity', _zero_gravity_world)
register_world_type('sideways_wind', _sideways_wind_world)
register_world_type('vortex', _vortex_world)
register_world_type('inverse_square_repulsion', _inverse_square_repulsion_world)
register_world_type('localized_gravity', _localized_gravity_world)
register_world_type('time_varying', _time_varying_world)
