from __future__ import annotations

"""
2D Physics engine built from first principles.
No external physics libraries — just Newton's laws implemented directly.

The agent observes THIS world and must discover its laws independently.
We do not tell the agent about gravity, momentum, or energy.
It must discover them by watching what happens.
"""

import math
import random


from typing import Callable, Optional
from .tensor_backend import (
    can_vectorize_force_components,
    compute_external_force_deltas,
    resolve_force_backend,
)


ForceComponentApplier = Callable[[object, dict, float], None]
FORCE_COMPONENT_APPLIERS: dict[str, ForceComponentApplier] = {}


def register_force_component(component_type: str, applier: ForceComponentApplier):
    """Register a simulator-only force component for hidden/adversarial worlds."""
    if not component_type:
        raise ValueError("component_type must be non-empty")
    FORCE_COMPONENT_APPLIERS[component_type] = applier


def registered_force_component_types() -> tuple[str, ...]:
    return tuple(sorted(FORCE_COMPONENT_APPLIERS))


class Vec2:
    """2D vector — the most fundamental mathematical object the agent could discover."""
    __slots__ = ('x', 'y')

    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.x = float(x)
        self.y = float(y)

    def __add__(self, other):
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        return Vec2(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar):
        return Vec2(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar):
        return Vec2(self.x / scalar, self.y / scalar)

    def __repr__(self):
        return f"Vec2({self.x:.3f}, {self.y:.3f})"

    def magnitude(self) -> float:
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def magnitude_squared(self) -> float:
        return self.x ** 2 + self.y ** 2

    def normalize(self):
        m = self.magnitude()
        return Vec2(self.x / m, self.y / m) if m > 1e-12 else Vec2(0, 0)

    def dot(self, other) -> float:
        return self.x * other.x + self.y * other.y

    def to_tuple(self) -> tuple:
        return (round(self.x, 6), round(self.y, 6))


class PhysicsObject:
    """A discrete object in the physics world. The agent must discover that objects exist."""
    _next_id = 0

    def __init__(self, position: Vec2, velocity: Vec2 = None,
                 mass: float = 1.0, radius: float = 0.5,
                 object_id: int | None = None):
        if object_id is None:
            self.id = PhysicsObject._next_id
            PhysicsObject._next_id += 1
        else:
            self.id = int(object_id)
        self.position = position
        self.velocity = velocity or Vec2(0, 0)
        self.mass = mass
        self.radius = radius
        self.alive = True

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'position': self.position.to_tuple(),
            'velocity': self.velocity.to_tuple(),
            'mass': round(self.mass, 6),
            'radius': round(self.radius, 6),
        }


class PhysicsWorld:
    """
    A 2D world with gravity, wall collisions, object collisions, and friction.

    Physics is real — momentum IS conserved in collisions (minus friction).
    The agent must discover this by observation, not by being told.
    """

    def __init__(self, width: float = 20.0, height: float = 20.0,
                 gravity: float = 9.8, friction: float = 0.999,
                 restitution: float = 0.85,
                 rng: random.Random | None = None,
                 force_backend: str = 'python'):
        self.width = width
        self.height = height
        self.gravity = gravity
        self.friction = friction
        self.restitution = restitution
        self.rng = rng or random.Random()
        self.objects: list[PhysicsObject] = []
        self.time: float = 0.0
        self.collision_events: list[dict] = []
        self._event_idx = 0
        self._next_object_id = 0
        # Novel physics: no extra forces by default
        self.central_force: Optional[dict] = None  # {'x': cx, 'y': cy, 'strength': G}
        self.gravity_wells: list[dict] = []  # Localized attractive wells
        self.repulsion_zones: list = []  # Regions where objects are repelled
        self.inverse_square_repulsions: list[dict] = []  # Point repulsors
        self.uniform_force: Optional[dict] = None  # Constant acceleration
        self.vortex_fields: list[dict] = []  # Tangential force fields
        self.time_varying_force: Optional[dict] = None  # Sinusoidal acceleration
        self.force_components: list[dict] = []  # Registry-driven hidden/adversarial forces
        self.world_type = 'standard'
        self.force_backend_request = 'python'
        self.force_backend = 'python'
        self.force_backend_status = {}
        self.set_force_backend(force_backend)

    def add_object(self, obj: PhysicsObject) -> PhysicsObject:
        self.objects.append(obj)
        return obj

    def spawn_object(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0,
                     mass: float = None, radius: float = None) -> PhysicsObject:
        obj = PhysicsObject(
            position=Vec2(x, y),
            velocity=Vec2(vx, vy),
            mass=self.rng.uniform(0.5, 3.0) if mass is None else mass,
            radius=self.rng.uniform(0.3, 0.8) if radius is None else radius,
            object_id=self._allocate_object_id(),
        )
        return self.add_object(obj)

    def _allocate_object_id(self) -> int:
        object_id = self._next_object_id
        self._next_object_id += 1
        return object_id

    def add_force_component(self, component_type: str, params: dict):
        if component_type not in FORCE_COMPONENT_APPLIERS:
            raise ValueError(f"Unknown force component type: {component_type}")
        self.force_components.append({
            'type': component_type,
            'params': dict(params),
        })

    def remove_object(self, obj_id: int):
        self.objects = [o for o in self.objects if o.id != obj_id]

    def get_object(self, obj_id: int) -> Optional[PhysicsObject]:
        for obj in self.objects:
            if obj.id == obj_id:
                return obj
        return None

    def move_object(
        self,
        obj_id: int,
        x: float,
        y: float,
        vx: float | None = None,
        vy: float | None = None,
    ):
        obj = self.get_object(obj_id)
        if obj is None:
            return
        obj.position = Vec2(x, y)
        if vx is not None:
            obj.velocity.x = float(vx)
        if vy is not None:
            obj.velocity.y = float(vy)

    def freeze_object(self, obj_id: int):
        obj = self.get_object(obj_id)
        if obj is None:
            return
        obj.velocity = Vec2(0.0, 0.0)

    def duplicate_object(self, obj_id: int) -> Optional[PhysicsObject]:
        obj = self.get_object(obj_id)
        if obj is None:
            return None
        return self.spawn_object(
            obj.position.x,
            obj.position.y,
            obj.velocity.x,
            obj.velocity.y,
            mass=obj.mass,
            radius=obj.radius,
        )

    def apply_impulse(self, obj_id: int, fx: float, fy: float):
        """Apply an instantaneous impulse to an object. The agent uses this to interact."""
        for obj in self.objects:
            if obj.id == obj_id:
                obj.velocity.x += fx / obj.mass
                obj.velocity.y += fy / obj.mass
                break

    def set_force_backend(self, force_backend: str | None):
        """Choose the simulator-side force backend without changing observations."""
        status = resolve_force_backend(force_backend)
        self.force_backend_request = status['requested_backend']
        self.force_backend = status['backend']
        self.force_backend_status = status

    def _can_use_tensor_force_backend(self) -> bool:
        return (
            self.force_backend != 'python'
            and bool(self.objects)
            and can_vectorize_force_components(self.force_components)
        )

    def _apply_external_forces(self, dt: float):
        if self._can_use_tensor_force_backend():
            self._apply_external_forces_tensor(dt)
        else:
            self._apply_external_forces_python(dt)

    def _apply_external_forces_tensor(self, dt: float):
        positions = [
            (obj.position.x, obj.position.y)
            for obj in self.objects
        ]
        deltas = compute_external_force_deltas(
            positions=positions,
            dt=dt,
            time_value=self.time,
            gravity=self.gravity,
            uniform_force=self.uniform_force,
            central_force=self.central_force,
            gravity_wells=self.gravity_wells,
            repulsion_zones=self.repulsion_zones,
            inverse_square_repulsions=self.inverse_square_repulsions,
            vortex_fields=self.vortex_fields,
            time_varying_force=self.time_varying_force,
            force_components=self.force_components,
            backend=self.force_backend,
        )
        for obj, (dvx, dvy) in zip(self.objects, deltas):
            obj.velocity.x += dvx
            obj.velocity.y += dvy

    def _apply_external_forces_python(self, dt: float):
        # Gravity — constant downward acceleration
        for obj in self.objects:
            obj.velocity.y -= self.gravity * dt

        # Novel physics: uniform field (like steady sideways wind)
        if self.uniform_force is not None:
            fx = self.uniform_force.get('x', 0.0)
            fy = self.uniform_force.get('y', 0.0)
            for obj in self.objects:
                obj.velocity.x += fx * dt
                obj.velocity.y += fy * dt

        # Novel physics: central force (gravitational well toward a point)
        if self.central_force is not None:
            cx = self.central_force['x']
            cy = self.central_force['y']
            strength = self.central_force['strength']
            for obj in self.objects:
                dx = cx - obj.position.x
                dy = cy - obj.position.y
                dist_sq = dx * dx + dy * dy
                if dist_sq < 0.01:
                    dist_sq = 0.01
                dist = math.sqrt(dist_sq)
                # Inverse-square-like attraction
                force = strength / dist_sq
                obj.velocity.x += (dx / dist) * force * dt
                obj.velocity.y += (dy / dist) * force * dt

        # Novel physics: localized gravity wells with finite radius
        for well in self.gravity_wells:
            wx = well['x']
            wy = well['y']
            strength = well['strength']
            radius = well['radius']
            for obj in self.objects:
                dx = wx - obj.position.x
                dy = wy - obj.position.y
                dist_sq = dx * dx + dy * dy
                dist = math.sqrt(dist_sq)
                if dist < 0.01 or dist > radius:
                    continue
                falloff = 1.0 - (dist / radius)
                force = strength * falloff / max(dist_sq, 0.25)
                obj.velocity.x += (dx / dist) * force * dt
                obj.velocity.y += (dy / dist) * force * dt

        # Novel physics: repulsion zones (objects pushed away from certain points)
        for zone in self.repulsion_zones:
            zx = zone['x']
            zy = zone['y']
            z_strength = zone['strength']
            z_radius = zone['radius']
            for obj in self.objects:
                dx = obj.position.x - zx
                dy = obj.position.y - zy
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < z_radius and dist > 0.01:
                    force = z_strength * (1.0 - dist / z_radius)
                    obj.velocity.x += (dx / dist) * force * dt
                    obj.velocity.y += (dy / dist) * force * dt

        # Novel physics: inverse-square repulsion from point sources
        for source in self.inverse_square_repulsions:
            sx = source['x']
            sy = source['y']
            strength = source['strength']
            for obj in self.objects:
                dx = obj.position.x - sx
                dy = obj.position.y - sy
                dist_sq = dx * dx + dy * dy
                if dist_sq < 0.25:
                    dist_sq = 0.25
                dist = math.sqrt(dist_sq)
                force = strength / dist_sq
                obj.velocity.x += (dx / dist) * force * dt
                obj.velocity.y += (dy / dist) * force * dt

        # Novel physics: vortex field (tangential acceleration around a point)
        for vortex in self.vortex_fields:
            vx = vortex['x']
            vy = vortex['y']
            strength = vortex['strength']
            radius = vortex['radius']
            direction = vortex.get('direction', 1.0)
            for obj in self.objects:
                dx = obj.position.x - vx
                dy = obj.position.y - vy
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < 0.01 or dist > radius:
                    continue
                falloff = 1.0 - (dist / radius)
                tangent_x = -dy / dist * direction
                tangent_y = dx / dist * direction
                obj.velocity.x += tangent_x * strength * falloff * dt
                obj.velocity.y += tangent_y * strength * falloff * dt

        # Novel physics: time-varying global field
        if self.time_varying_force is not None:
            axis = self.time_varying_force.get('axis', 'x')
            amplitude = self.time_varying_force.get('amplitude', 0.0)
            period = max(self.time_varying_force.get('period', 1.0), 1e-6)
            phase = self.time_varying_force.get('phase', 0.0)
            force = amplitude * math.sin((2 * math.pi * self.time / period) + phase)
            for obj in self.objects:
                if axis == 'x':
                    obj.velocity.x += force * dt
                else:
                    obj.velocity.y += force * dt

        for component in self.force_components:
            applier = FORCE_COMPONENT_APPLIERS.get(component.get('type'))
            if applier is None:
                continue
            applier(self, component.get('params', {}), dt)

    def step(self, dt: float = 0.016):
        """Advance the simulation by one timestep."""
        self.time += dt
        self.collision_events.clear()
        self._apply_external_forces(dt)

        # Friction — multiplicative damping (breaks strict conservation, as in reality)
        for obj in self.objects:
            obj.velocity = obj.velocity * self.friction

        # Update positions
        for obj in self.objects:
            obj.position = obj.position + obj.velocity * dt

        # Wall collisions
        for obj in self.objects:
            if obj.position.x - obj.radius < 0:
                obj.position.x = obj.radius
                obj.velocity.x = -obj.velocity.x * self.restitution
            elif obj.position.x + obj.radius > self.width:
                obj.position.x = self.width - obj.radius
                obj.velocity.x = -obj.velocity.x * self.restitution
            if obj.position.y - obj.radius < 0:
                obj.position.y = obj.radius
                obj.velocity.y = -obj.velocity.y * self.restitution
            elif obj.position.y + obj.radius > self.height:
                obj.position.y = self.height - obj.radius
                obj.velocity.y = -obj.velocity.y * self.restitution

        # Object-object collisions (impulse-based, conserves momentum)
        n = len(self.objects)
        for i in range(n):
            for j in range(i + 1, n):
                self._resolve_collision(self.objects[i], self.objects[j])

    def _resolve_collision(self, a: PhysicsObject, b: PhysicsObject):
        delta = b.position - a.position
        dist = delta.magnitude()
        min_dist = a.radius + b.radius

        if dist >= min_dist or dist < 1e-12:
            return

        # Positional correction
        normal = delta.normalize()
        overlap = min_dist - dist
        a.position = a.position - normal * (overlap * 0.5)
        b.position = b.position + normal * (overlap * 0.5)

        # Impulse-based collision response (momentum is conserved here)
        rel_vel = b.velocity - a.velocity
        vel_along_normal = rel_vel.dot(normal)

        if vel_along_normal > 0:
            return  # Already separating

        e = self.restitution
        j_impulse = -(1 + e) * vel_along_normal
        j_impulse /= (1.0 / a.mass + 1.0 / b.mass)

        impulse = normal * j_impulse
        a.velocity = a.velocity - impulse * (1.0 / a.mass)
        b.velocity = b.velocity + impulse * (1.0 / b.mass)

        # Record the collision event for the agent to observe
        self.collision_events.append({
            'object_a': a.id,
            'object_b': b.id,
            'time': round(self.time, 6),
        })

    def get_state(self) -> list[dict]:
        return [obj.to_dict() for obj in self.objects]

    def get_collisions(self) -> list[dict]:
        return list(self.collision_events)


def _radial_force(
    world: PhysicsWorld,
    *,
    x: float,
    y: float,
    strength: float,
    exponent: float,
    direction: float,
    dt: float,
    max_radius: float | None = None,
):
    exponent = max(float(exponent), 0.0)
    for obj in world.objects:
        dx = obj.position.x - x
        dy = obj.position.y - y
        dist_sq = dx * dx + dy * dy
        dist = math.sqrt(max(dist_sq, 0.25))
        if max_radius is not None and dist > max_radius:
            continue
        force = strength / max(dist ** exponent, 0.25)
        obj.velocity.x += direction * (dx / dist) * force * dt
        obj.velocity.y += direction * (dy / dist) * force * dt


def _apply_cutoff_radial(world: PhysicsWorld, params: dict, dt: float):
    _radial_force(
        world,
        x=float(params['x']),
        y=float(params['y']),
        strength=float(params['strength']),
        exponent=float(params.get('exponent', 2.0)),
        direction=float(params.get('direction', 1.0)),
        max_radius=float(params['radius']),
        dt=dt,
    )


def _apply_piecewise_radial(world: PhysicsWorld, params: dict, dt: float):
    x = float(params['x'])
    y = float(params['y'])
    split_radius = float(params['split_radius'])
    inner = dict(params.get('inner', {}))
    outer = dict(params.get('outer', {}))
    for obj in world.objects:
        dx = obj.position.x - x
        dy = obj.position.y - y
        dist_sq = dx * dx + dy * dy
        dist = math.sqrt(max(dist_sq, 0.25))
        regime = inner if dist <= split_radius else outer
        strength = float(regime.get('strength', 0.0))
        if abs(strength) < 1e-12:
            continue
        exponent = max(float(regime.get('exponent', 2.0)), 0.0)
        direction = float(regime.get('direction', 1.0))
        force = strength / max(dist ** exponent, 0.25)
        obj.velocity.x += direction * (dx / dist) * force * dt
        obj.velocity.y += direction * (dy / dist) * force * dt


def _apply_periodic_regime_uniform(world: PhysicsWorld, params: dict, dt: float):
    period = max(float(params.get('period', 1.0)), 1e-6)
    duty_cycle = min(max(float(params.get('duty_cycle', 0.5)), 0.0), 1.0)
    phase = float(params.get('phase', 0.0))
    cycle_position = ((world.time + phase) % period) / period
    if cycle_position <= duty_cycle:
        force = params.get('on_force', {})
    else:
        force = params.get('off_force', {})
    fx = float(force.get('x', 0.0))
    fy = float(force.get('y', 0.0))
    for obj in world.objects:
        obj.velocity.x += fx * dt
        obj.velocity.y += fy * dt


register_force_component('cutoff_radial', _apply_cutoff_radial)
register_force_component('piecewise_radial', _apply_piecewise_radial)
register_force_component('periodic_regime_uniform', _apply_periodic_regime_uniform)
