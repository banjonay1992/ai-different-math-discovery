from __future__ import annotations

"""
Deterministic hidden physics worlds.

The manifest is a benchmark-only truth object. The environment applies the
components to the simulator, but public observations expose no world labels or
component labels, so the agent has to infer laws from state transitions.
"""

from dataclasses import dataclass, field
import hashlib
import random
from typing import Callable


HIDDEN_COMPONENT_TO_EXPECTED_DISCOVERY = {
    'uniform_push': 'uniform_component',
    'radial_attraction': 'radial_component',
    'radial_repulsion': 'repulsive_component',
    'tangential_flow': 'tangential_component',
    'time_wave': 'time_varying_component',
    'localized_pull': 'radial_component',
    'localized_push': 'repulsive_component',
    'zero_gravity': 'zero_gravity',
    'soft_drag': 'damping_component',
    'cutoff_radial_pull': ('radial_component', 'piecewise_component'),
    'cutoff_radial_push': ('repulsive_component', 'piecewise_component'),
    'piecewise_radial': (
        'radial_component',
        'repulsive_component',
        'piecewise_component',
    ),
    'periodic_regime_uniform': (
        'uniform_component',
        'time_varying_component',
        'piecewise_component',
    ),
}


HiddenComponentFactory = Callable[[random.Random], 'HiddenWorldComponent']
HiddenComponentApplier = Callable[[object, dict], None]
HIDDEN_COMPONENT_FACTORIES: dict[str, HiddenComponentFactory] = {}
HIDDEN_COMPONENT_APPLIERS: dict[str, HiddenComponentApplier] = {}


def register_hidden_component(
    component_type: str,
    factory: HiddenComponentFactory,
    applier: HiddenComponentApplier,
):
    """Register a benchmark-only component without changing agent code."""
    if not component_type:
        raise ValueError("component_type must be non-empty")
    HIDDEN_COMPONENT_FACTORIES[component_type] = factory
    HIDDEN_COMPONENT_APPLIERS[component_type] = applier


def registered_hidden_component_types() -> tuple[str, ...]:
    return tuple(sorted(HIDDEN_COMPONENT_FACTORIES))


@dataclass(frozen=True)
class HiddenWorldComponent:
    """One hidden simulator component used only to configure physics."""
    component_type: str
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'type': self.component_type,
            'params': dict(self.params),
        }


@dataclass(frozen=True)
class HiddenWorldManifest:
    """Benchmark-side truth for a generated hidden world."""
    hidden_id: str
    seed: int
    components: tuple[HiddenWorldComponent, ...]

    @property
    def expected_discoveries(self) -> set[str]:
        discoveries: set[str] = set()
        for component in self.components:
            expected = HIDDEN_COMPONENT_TO_EXPECTED_DISCOVERY.get(
                component.component_type
            )
            if isinstance(expected, str):
                discoveries.add(expected)
            elif expected:
                discoveries.update(str(item) for item in expected)
        return discoveries

    def to_dict(self) -> dict:
        return {
            'hidden_id': self.hidden_id,
            'seed': self.seed,
            'components': [
                component.to_dict()
                for component in self.components
            ],
            'expected_discoveries': sorted(self.expected_discoveries),
        }


def generate_hidden_world_manifest(seed: int, variant: int = 0) -> HiddenWorldManifest:
    """
    Build a deterministic mixed-force world.

    Variants cycle through deliberately different component families so train
    and holdout suites exercise transfer instead of one repeated recipe.
    """
    rng = random.Random((seed + 1) * 7919 + variant * 104729)
    recipes = [
        ('uniform_push', 'radial_attraction'),
        ('radial_repulsion', 'uniform_push'),
        ('time_wave', 'uniform_push'),
        ('localized_pull', 'uniform_push', 'tangential_flow'),
        ('time_wave', 'radial_repulsion', 'uniform_push'),
        ('radial_attraction', 'tangential_flow'),
        ('localized_pull', 'time_wave'),
        ('radial_repulsion', 'tangential_flow'),
        ('zero_gravity', 'uniform_push'),
        ('time_wave', 'localized_pull', 'radial_repulsion'),
        ('tangential_flow',),
        ('radial_attraction', 'radial_repulsion'),
        ('localized_pull', 'localized_pull'),
        ('soft_drag', 'uniform_push'),
        ('zero_gravity', 'time_wave', 'tangential_flow'),
        ('localized_push', 'uniform_push'),
        ('localized_push', 'time_wave'),
        ('localized_push', 'tangential_flow', 'soft_drag'),
        ('radial_attraction', 'localized_push', 'time_wave'),
        ('radial_repulsion', 'localized_pull', 'soft_drag'),
        ('uniform_push', 'time_wave', 'soft_drag'),
        ('localized_pull', 'tangential_flow', 'radial_repulsion'),
        ('zero_gravity', 'localized_push', 'uniform_push'),
        ('radial_attraction', 'localized_pull', 'localized_push'),
        ('time_wave', 'tangential_flow', 'soft_drag'),
        ('cutoff_radial_pull', 'uniform_push'),
        ('cutoff_radial_push', 'tangential_flow'),
        ('piecewise_radial', 'time_wave'),
        ('periodic_regime_uniform', 'localized_push'),
        ('piecewise_radial', 'soft_drag', 'uniform_push'),
    ]
    component_types = recipes[variant % len(recipes)]
    components = tuple(
        _component_for(component_type, rng)
        for component_type in component_types
    )
    return HiddenWorldManifest(
        hidden_id=f"hidden_{variant:02d}_{seed:04d}",
        seed=seed,
        components=components,
    )


def generate_self_authored_hidden_world_manifest(
    design: dict,
    seed: int = 0,
    variant: int = 0,
) -> HiddenWorldManifest:
    """
    Convert an autonomous theory question into a hidden physics world.

    The design stays outside the observation channel. Only the simulator receives
    the resulting component recipe, so the agent must rediscover the effect from
    public state transitions.
    """
    design_key = str(design.get('design_key') or design.get('question') or variant)
    digest = hashlib.sha256(design_key.encode('utf-8')).hexdigest()
    design_offset = int(digest[:8], 16)
    rng = random.Random((seed + 17) * 15485863 + variant * 32452843 + design_offset)
    recipe = _self_authored_recipe_for_design(design)
    components = tuple(
        _component_for(component_type, rng)
        for component_type in recipe
    )
    return HiddenWorldManifest(
        hidden_id=f"authored_{variant:02d}_{seed:04d}",
        seed=seed,
        components=components,
    )


def apply_hidden_world(world, manifest: HiddenWorldManifest):
    """Apply hidden components to a PhysicsWorld."""
    world.world_type = 'hidden_procedural'
    for component in manifest.components:
        applier = HIDDEN_COMPONENT_APPLIERS.get(component.component_type)
        if applier is None:
            raise ValueError(
                f"Unknown hidden component type: {component.component_type}"
            )
        applier(world, component.params)


def hidden_manifest_from_observation(observation: dict) -> bool:
    """Return True if a raw observation appears to leak hidden manifest data."""
    forbidden = {
        'components',
        'expected_discoveries',
        'hidden_id',
        'truth',
        'manifest',
        'world_type',
    }
    return any(key in observation for key in forbidden)


def _random_center(rng: random.Random) -> tuple[float, float]:
    return rng.uniform(7.0, 13.0), rng.uniform(7.0, 13.0)


def _uniform_push_component(rng: random.Random) -> HiddenWorldComponent:
    direction = rng.choice([-1.0, 1.0])
    return HiddenWorldComponent('uniform_push', {
        'x': direction * rng.uniform(5.5, 9.5),
        'y': rng.uniform(-1.0, 1.0),
    })


def _radial_attraction_component(rng: random.Random) -> HiddenWorldComponent:
    center_x, center_y = _random_center(rng)
    return HiddenWorldComponent('radial_attraction', {
        'x': center_x,
        'y': center_y,
        'strength': rng.uniform(140.0, 230.0),
    })


def _radial_repulsion_component(rng: random.Random) -> HiddenWorldComponent:
    center_x, center_y = _random_center(rng)
    return HiddenWorldComponent('radial_repulsion', {
        'x': center_x,
        'y': center_y,
        'strength': rng.uniform(95.0, 155.0),
    })


def _tangential_flow_component(rng: random.Random) -> HiddenWorldComponent:
    center_x, center_y = _random_center(rng)
    return HiddenWorldComponent('tangential_flow', {
        'x': center_x,
        'y': center_y,
        'strength': rng.uniform(95.0, 145.0),
        'radius': rng.uniform(7.0, 9.5),
        'direction': rng.choice([-1.0, 1.0]),
    })


def _time_wave_component(rng: random.Random) -> HiddenWorldComponent:
    return HiddenWorldComponent('time_wave', {
        'axis': rng.choice(['x', 'y']),
        'amplitude': rng.uniform(8.0, 13.0),
        'period': rng.uniform(0.96, 1.44),
        'phase': rng.uniform(0.0, 1.0),
    })


def _localized_pull_component(rng: random.Random) -> HiddenWorldComponent:
    center_x, center_y = _random_center(rng)
    return HiddenWorldComponent('localized_pull', {
        'x': center_x,
        'y': center_y,
        'strength': rng.uniform(180.0, 280.0),
        'radius': rng.uniform(7.0, 9.5),
    })


def _localized_push_component(rng: random.Random) -> HiddenWorldComponent:
    center_x, center_y = _random_center(rng)
    return HiddenWorldComponent('localized_push', {
        'x': center_x,
        'y': center_y,
        'strength': rng.uniform(70.0, 125.0),
        'radius': rng.uniform(5.5, 8.5),
    })


def _zero_gravity_component(rng: random.Random) -> HiddenWorldComponent:
    return HiddenWorldComponent('zero_gravity', {})


def _soft_drag_component(rng: random.Random) -> HiddenWorldComponent:
    return HiddenWorldComponent('soft_drag', {
        'friction': rng.uniform(0.985, 0.994),
    })


def _cutoff_radial_pull_component(rng: random.Random) -> HiddenWorldComponent:
    center_x, center_y = _random_center(rng)
    return HiddenWorldComponent('cutoff_radial_pull', {
        'x': center_x,
        'y': center_y,
        'strength': rng.uniform(155.0, 260.0),
        'radius': rng.uniform(4.5, 7.0),
        'exponent': rng.choice([1.0, 1.5, 2.0, 2.5]),
        'direction': -1.0,
    })


def _cutoff_radial_push_component(rng: random.Random) -> HiddenWorldComponent:
    center_x, center_y = _random_center(rng)
    return HiddenWorldComponent('cutoff_radial_push', {
        'x': center_x,
        'y': center_y,
        'strength': rng.uniform(90.0, 175.0),
        'radius': rng.uniform(4.0, 6.5),
        'exponent': rng.choice([1.0, 1.5, 2.0, 2.5]),
        'direction': 1.0,
    })


def _piecewise_radial_component(rng: random.Random) -> HiddenWorldComponent:
    center_x, center_y = _random_center(rng)
    inner_direction = rng.choice([-1.0, 1.0])
    return HiddenWorldComponent('piecewise_radial', {
        'x': center_x,
        'y': center_y,
        'split_radius': rng.uniform(5.0, 7.5),
        'inner': {
            'strength': rng.uniform(95.0, 180.0),
            'exponent': rng.choice([1.0, 2.0]),
            'direction': inner_direction,
        },
        'outer': {
            'strength': rng.uniform(40.0, 110.0),
            'exponent': rng.choice([0.5, 1.5, 2.5]),
            'direction': -inner_direction,
        },
    })


def _periodic_regime_uniform_component(rng: random.Random) -> HiddenWorldComponent:
    axis = rng.choice(['x', 'y'])
    direction = rng.choice([-1.0, 1.0])
    magnitude = rng.uniform(6.0, 11.0)
    on_force = {'x': 0.0, 'y': 0.0}
    off_force = {'x': 0.0, 'y': 0.0}
    on_force[axis] = direction * magnitude
    off_force[axis] = -direction * rng.uniform(0.0, magnitude * 0.45)
    return HiddenWorldComponent('periodic_regime_uniform', {
        'period': rng.uniform(0.72, 1.52),
        'phase': rng.uniform(0.0, 1.0),
        'duty_cycle': rng.uniform(0.35, 0.65),
        'on_force': on_force,
        'off_force': off_force,
    })


def _apply_uniform_push(world, params: dict):
    world.uniform_force = {'x': params['x'], 'y': params['y']}


def _apply_radial_attraction(world, params: dict):
    world.central_force = {
        'x': params['x'],
        'y': params['y'],
        'strength': params['strength'],
    }


def _apply_radial_repulsion(world, params: dict):
    world.inverse_square_repulsions.append({
        'x': params['x'],
        'y': params['y'],
        'strength': params['strength'],
    })


def _apply_tangential_flow(world, params: dict):
    world.vortex_fields.append({
        'x': params['x'],
        'y': params['y'],
        'strength': params['strength'],
        'radius': params['radius'],
        'direction': params['direction'],
    })


def _apply_time_wave(world, params: dict):
    world.time_varying_force = {
        'axis': params['axis'],
        'amplitude': params['amplitude'],
        'period': params['period'],
        'phase': params['phase'],
    }


def _apply_localized_pull(world, params: dict):
    world.gravity_wells.append({
        'x': params['x'],
        'y': params['y'],
        'strength': params['strength'],
        'radius': params['radius'],
    })


def _apply_localized_push(world, params: dict):
    world.repulsion_zones.append({
        'x': params['x'],
        'y': params['y'],
        'strength': params['strength'],
        'radius': params['radius'],
    })


def _apply_zero_gravity(world, params: dict):
    world.gravity = 0.0


def _apply_soft_drag(world, params: dict):
    world.friction = params['friction']


def _apply_cutoff_radial(world, params: dict):
    world.add_force_component('cutoff_radial', params)


def _apply_piecewise_radial(world, params: dict):
    world.add_force_component('piecewise_radial', params)


def _apply_periodic_regime_uniform(world, params: dict):
    world.add_force_component('periodic_regime_uniform', params)


def _register_builtin_components():
    register_hidden_component('uniform_push', _uniform_push_component, _apply_uniform_push)
    register_hidden_component(
        'radial_attraction',
        _radial_attraction_component,
        _apply_radial_attraction,
    )
    register_hidden_component(
        'radial_repulsion',
        _radial_repulsion_component,
        _apply_radial_repulsion,
    )
    register_hidden_component(
        'tangential_flow',
        _tangential_flow_component,
        _apply_tangential_flow,
    )
    register_hidden_component('time_wave', _time_wave_component, _apply_time_wave)
    register_hidden_component(
        'localized_pull',
        _localized_pull_component,
        _apply_localized_pull,
    )
    register_hidden_component(
        'localized_push',
        _localized_push_component,
        _apply_localized_push,
    )
    register_hidden_component('zero_gravity', _zero_gravity_component, _apply_zero_gravity)
    register_hidden_component('soft_drag', _soft_drag_component, _apply_soft_drag)
    register_hidden_component(
        'cutoff_radial_pull',
        _cutoff_radial_pull_component,
        _apply_cutoff_radial,
    )
    register_hidden_component(
        'cutoff_radial_push',
        _cutoff_radial_push_component,
        _apply_cutoff_radial,
    )
    register_hidden_component(
        'piecewise_radial',
        _piecewise_radial_component,
        _apply_piecewise_radial,
    )
    register_hidden_component(
        'periodic_regime_uniform',
        _periodic_regime_uniform_component,
        _apply_periodic_regime_uniform,
    )


_register_builtin_components()


def _component_for(component_type: str, rng: random.Random) -> HiddenWorldComponent:
    factory = HIDDEN_COMPONENT_FACTORIES.get(component_type)
    if factory is None:
        raise ValueError(f"Unknown hidden component type: {component_type}")
    return factory(rng)


def _self_authored_recipe_for_design(design: dict) -> tuple[str, ...]:
    text = ' '.join(
        str(design.get(key, ''))
        for key in (
            'design_key',
            'source',
            'experiment_kind',
            'question',
            'hypothesis',
            'experiment',
            'expected_result',
            'falsifies_if',
            'next_action',
        )
    ).lower()
    probe_or_world = str(design.get('probe_or_world') or '').lower()
    text = f"{text} {probe_or_world}"
    if 'regime' in text or 'on/off' in text or 'duty' in text:
        return ('periodic_regime_uniform', 'localized_push')
    if 'period' in text or 'phase' in text or 'time' in text:
        return ('time_wave', 'uniform_push')
    if (
        'vector_direction' in text
        or 'perpendicular' in text
        or 'vortex' in text
        or 'tangent' in text
    ):
        return ('tangential_flow', 'radial_attraction')
    if 'repulsion' in text or 'push' in text or 'domain split' in text:
        return ('localized_push', 'time_wave')
    if 'cutoff' in text or 'boundary' in text or 'localized' in text:
        return ('cutoff_radial_pull', 'uniform_push')
    if (
        'exponent' in text
        or 'near/mid/far' in text
        or 'distance' in text
        or 'inverse' in text
    ):
        return ('piecewise_radial', 'cutoff_radial_push')
    if 'baseline' in text or 'residual' in text:
        return ('uniform_push', 'time_wave')
    if 'holdout' in text or 'counterexample' in text or 'hidden' in text:
        return ('radial_attraction', 'radial_repulsion', 'uniform_push')
    recipes = [
        ('uniform_push', 'radial_attraction'),
        ('radial_repulsion', 'tangential_flow'),
        ('localized_pull', 'time_wave'),
        ('localized_push', 'soft_drag'),
        ('zero_gravity', 'uniform_push'),
    ]
    design_key = str(design.get('design_key') or design.get('question') or '')
    digest = hashlib.sha256(design_key.encode('utf-8')).hexdigest()
    return recipes[int(digest[:4] or '0', 16) % len(recipes)]
