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
from typing import Optional
from .physics import PhysicsWorld, PhysicsObject, Vec2
from .hidden_worlds import HiddenWorldManifest, apply_hidden_world


class Environment:
    """Wraps the physics world and provides a clean action/observation interface."""

    def __init__(self, width: float = 20.0, height: float = 20.0,
                 num_initial_objects: int = 5, seed: int = None,
                 world_type: str = 'standard',
                 hidden_manifest: HiddenWorldManifest | None = None):
        if seed is not None:
            random.seed(seed)
        self.world = PhysicsWorld(width, height)
        self.world.world_type = world_type
        self.step_count = 0

        # Configure novel physics based on world type
        if hidden_manifest is not None:
            apply_hidden_world(self.world, hidden_manifest)
        elif world_type == 'central_force':
            # Add a gravitational well at the center of the world
            self.world.central_force = {
                'x': width / 2,
                'y': height / 2,
                'strength': 200.0,  # Strong enough to be noticeable at typical distances
            }
        elif world_type == 'repulsion':
            # Add a repulsion zone at the center
            self.world.repulsion_zones = [{
                'x': width / 2,
                'y': height / 2,
                'strength': 80.0,
                'radius': 6.0,
            }]
        elif world_type == 'zero_gravity':
            # No gravity — test if agent notices the absence
            self.world.gravity = 0.0
        elif world_type == 'sideways_wind':
            # A steady horizontal acceleration layered on top of gravity
            self.world.uniform_force = {'x': 8.0, 'y': 0.0}
        elif world_type == 'vortex':
            # A rotating tangential field centered in the world
            self.world.vortex_fields = [{
                'x': width / 2,
                'y': height / 2,
                'strength': 140.0,
                'radius': 9.0,
                'direction': 1.0,
            }]
        elif world_type == 'inverse_square_repulsion':
            # A point source that pushes objects away with inverse-square falloff
            self.world.inverse_square_repulsions = [{
                'x': width / 2,
                'y': height / 2,
                'strength': 130.0,
            }]
        elif world_type == 'localized_gravity':
            # An off-center attractive well with finite range
            self.world.gravity_wells = [{
                'x': width * 0.35,
                'y': height * 0.65,
                'strength': 260.0,
                'radius': 9.0,
            }]
        elif world_type == 'time_varying':
            # A global horizontal force that periodically reverses direction
            self.world.time_varying_force = {
                'axis': 'x',
                'amplitude': 12.0,
                'period': 1.28,
                'phase': 0.0,
            }

        self._populate(num_initial_objects)

    def _populate(self, n: int):
        for _ in range(n):
            self.world.spawn_object(
                x=random.uniform(2, self.world.width - 2),
                y=random.uniform(2, self.world.height - 2),
                vx=random.uniform(-3, 3),
                vy=random.uniform(-3, 3),
            )

    def observe(self) -> dict:
        """Return the full observable state of the world."""
        return {
            'objects': self.world.get_state(),
            'collisions': self.world.get_collisions(),
            'time': round(self.world.time, 6),
            'step': self.step_count,
            'world_size': (self.world.width, self.world.height),
            'world_type': self.world.world_type,
        }

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
        return random.choice(ids) if ids else None
