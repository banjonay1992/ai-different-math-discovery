from __future__ import annotations

"""
Perception — the agent's sensory system.

Takes raw world state and extracts structured features.
The agent is NOT told what features to extract. It starts with:
  - "There are things in the world" (object detection)
  - "Things have properties" (position, velocity, mass)

From these, it must DERIVE higher-level features:
  - Count (how many things)
  - Momentum (mass × velocity)
  - Energy (½mv²)
  - Distance (between things)
  - Ordering (left/right, up/down)

The agent discovers these derived features through its hypothesis system,
not because we hardcoded them. However, for the prototype to work in
reasonable time, we provide the raw perceptual primitives and let the
agent's reasoning system discover the relationships.
"""

import math
from dataclasses import dataclass, field


@dataclass
class ObjectObservation:
    """The agent's perception of a single object at one moment in time."""
    id: int
    x: float
    y: float
    vx: float
    vy: float
    mass: float
    radius: float
    speed: float = 0.0  # derived: magnitude of velocity

    def __post_init__(self):
        self.speed = math.sqrt(self.vx ** 2 + self.vy ** 2)


@dataclass
class WorldObservation:
    """The agent's complete perception of the world at one moment."""
    objects: list[ObjectObservation] = field(default_factory=list)
    collisions: list[dict] = field(default_factory=list)
    time: float = 0.0
    step: int = 0
    world_width: float = 0.0
    world_height: float = 0.0

    @property
    def count(self) -> int:
        return len(self.objects)

    @property
    def total_momentum_x(self) -> float:
        return sum(o.mass * o.vx for o in self.objects)

    @property
    def total_momentum_y(self) -> float:
        return sum(o.mass * o.vy for o in self.objects)

    @property
    def total_momentum(self) -> float:
        return math.sqrt(self.total_momentum_x ** 2 + self.total_momentum_y ** 2)

    @property
    def total_kinetic_energy(self) -> float:
        return sum(0.5 * o.mass * o.speed ** 2 for o in self.objects)

    @property
    def total_mass(self) -> float:
        return sum(o.mass for o in self.objects)

    @property
    def center_of_mass_x(self) -> float:
        tm = self.total_mass
        return sum(o.mass * o.x for o in self.objects) / tm if tm > 0 else 0.0

    @property
    def center_of_mass_y(self) -> float:
        tm = self.total_mass
        return sum(o.mass * o.y for o in self.objects) / tm if tm > 0 else 0.0

    @property
    def pairwise_distances(self) -> list[float]:
        dists = []
        objs = self.objects
        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                dx = objs[i].x - objs[j].x
                dy = objs[i].y - objs[j].y
                dists.append(math.sqrt(dx ** 2 + dy ** 2))
        return dists

    @property
    def x_ordering(self) -> list[int]:
        """Object IDs sorted by x position (leftmost first)."""
        return [o.id for o in sorted(self.objects, key=lambda o: o.x)]

    @property
    def y_ordering(self) -> list[int]:
        """Object IDs sorted by y position (bottom first)."""
        return [o.id for o in sorted(self.objects, key=lambda o: o.y)]

    def get_feature_vector(self) -> dict:
        """
        Return all observable features as a flat dictionary.
        This is what the agent works with — it does NOT know the names mean.
        It discovers what's important through observation.
        """
        return {
            'count': self.count,
            'total_momentum_x': round(self.total_momentum_x, 6),
            'total_momentum_y': round(self.total_momentum_y, 6),
            'total_momentum': round(self.total_momentum, 6),
            'total_kinetic_energy': round(self.total_kinetic_energy, 6),
            'total_mass': round(self.total_mass, 6),
            'center_of_mass_x': round(self.center_of_mass_x, 6),
            'center_of_mass_y': round(self.center_of_mass_y, 6),
            'num_collisions': len(self.collisions),
            'mean_distance': round(
                sum(self.pairwise_distances) / max(len(self.pairwise_distances), 1), 6
            ) if self.pairwise_distances else 0.0,
            'max_x': max((o.x for o in self.objects), default=0.0),
            'min_x': min((o.x for o in self.objects), default=0.0),
            'max_y': max((o.y for o in self.objects), default=0.0),
            'min_y': min((o.y for o in self.objects), default=0.0),
            'mean_speed': round(
                sum(o.speed for o in self.objects) / max(self.count, 1), 6
            ),
        }


class Perception:
    """Converts raw environment observations into structured perceptions."""

    @staticmethod
    def perceive(raw_state: dict) -> WorldObservation:
        objects = []
        for obj_dict in raw_state['objects']:
            objects.append(ObjectObservation(
                id=obj_dict['id'],
                x=obj_dict['position'][0],
                y=obj_dict['position'][1],
                vx=obj_dict['velocity'][0],
                vy=obj_dict['velocity'][1],
                mass=obj_dict['mass'],
                radius=obj_dict['radius'],
            ))

        w, h = raw_state.get('world_size', (0, 0))
        return WorldObservation(
            objects=objects,
            collisions=raw_state.get('collisions', []),
            time=raw_state.get('time', 0.0),
            step=raw_state.get('step', 0),
            world_width=w,
            world_height=h,
        )
