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
        count = len(self.objects)
        total_momentum_x = 0.0
        total_momentum_y = 0.0
        total_kinetic_energy = 0.0
        total_mass = 0.0
        weighted_x = 0.0
        weighted_y = 0.0
        speed_sum = 0.0
        max_x = max_y = 0.0
        min_x = min_y = 0.0

        if self.objects:
            max_x = min_x = self.objects[0].x
            max_y = min_y = self.objects[0].y

        for obj in self.objects:
            momentum_x = obj.mass * obj.vx
            momentum_y = obj.mass * obj.vy
            total_momentum_x += momentum_x
            total_momentum_y += momentum_y
            total_kinetic_energy += 0.5 * obj.mass * obj.speed ** 2
            total_mass += obj.mass
            weighted_x += obj.mass * obj.x
            weighted_y += obj.mass * obj.y
            speed_sum += obj.speed
            max_x = max(max_x, obj.x)
            min_x = min(min_x, obj.x)
            max_y = max(max_y, obj.y)
            min_y = min(min_y, obj.y)

        pairwise_total = 0.0
        pairwise_count = 0
        for i in range(count):
            first = self.objects[i]
            for j in range(i + 1, count):
                second = self.objects[j]
                dx = first.x - second.x
                dy = first.y - second.y
                pairwise_total += math.sqrt(dx ** 2 + dy ** 2)
                pairwise_count += 1

        center_of_mass_x = weighted_x / total_mass if total_mass > 0 else 0.0
        center_of_mass_y = weighted_y / total_mass if total_mass > 0 else 0.0
        mean_distance = pairwise_total / pairwise_count if pairwise_count else 0.0
        mean_speed = speed_sum / max(count, 1)
        return {
            'count': count,
            'total_momentum_x': round(total_momentum_x, 6),
            'total_momentum_y': round(total_momentum_y, 6),
            'total_momentum': round(
                math.sqrt(total_momentum_x ** 2 + total_momentum_y ** 2),
                6,
            ),
            'total_kinetic_energy': round(total_kinetic_energy, 6),
            'total_mass': round(total_mass, 6),
            'center_of_mass_x': round(center_of_mass_x, 6),
            'center_of_mass_y': round(center_of_mass_y, 6),
            'num_collisions': len(self.collisions),
            'mean_distance': round(mean_distance, 6),
            'max_x': max_x,
            'min_x': min_x,
            'max_y': max_y,
            'min_y': min_y,
            'mean_speed': round(mean_speed, 6),
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
