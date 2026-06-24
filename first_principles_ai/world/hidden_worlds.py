from __future__ import annotations

"""
Deterministic hidden physics worlds.

The manifest is a benchmark-only truth object. The environment applies the
components to the simulator, but observations expose only a generic hidden
world marker so the agent cannot branch on component labels.
"""

from dataclasses import dataclass, field
import hashlib
import random


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
}


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
        return {
            HIDDEN_COMPONENT_TO_EXPECTED_DISCOVERY[component.component_type]
            for component in self.components
            if component.component_type in HIDDEN_COMPONENT_TO_EXPECTED_DISCOVERY
        }

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
        params = component.params
        if component.component_type == 'uniform_push':
            world.uniform_force = {
                'x': params['x'],
                'y': params['y'],
            }
        elif component.component_type == 'radial_attraction':
            world.central_force = {
                'x': params['x'],
                'y': params['y'],
                'strength': params['strength'],
            }
        elif component.component_type == 'radial_repulsion':
            world.inverse_square_repulsions.append({
                'x': params['x'],
                'y': params['y'],
                'strength': params['strength'],
            })
        elif component.component_type == 'tangential_flow':
            world.vortex_fields.append({
                'x': params['x'],
                'y': params['y'],
                'strength': params['strength'],
                'radius': params['radius'],
                'direction': params['direction'],
            })
        elif component.component_type == 'time_wave':
            world.time_varying_force = {
                'axis': params['axis'],
                'amplitude': params['amplitude'],
                'period': params['period'],
                'phase': params['phase'],
            }
        elif component.component_type == 'localized_pull':
            world.gravity_wells.append({
                'x': params['x'],
                'y': params['y'],
                'strength': params['strength'],
                'radius': params['radius'],
            })
        elif component.component_type == 'localized_push':
            world.repulsion_zones.append({
                'x': params['x'],
                'y': params['y'],
                'strength': params['strength'],
                'radius': params['radius'],
            })
        elif component.component_type == 'zero_gravity':
            world.gravity = 0.0
        elif component.component_type == 'soft_drag':
            world.friction = params['friction']


def hidden_manifest_from_observation(observation: dict) -> bool:
    """Return True if a raw observation appears to leak hidden manifest data."""
    forbidden = {
        'components',
        'expected_discoveries',
        'hidden_id',
        'truth',
        'manifest',
    }
    return any(key in observation for key in forbidden)


def _component_for(component_type: str, rng: random.Random) -> HiddenWorldComponent:
    center_x = rng.uniform(7.0, 13.0)
    center_y = rng.uniform(7.0, 13.0)
    if component_type == 'uniform_push':
        direction = rng.choice([-1.0, 1.0])
        return HiddenWorldComponent(component_type, {
            'x': direction * rng.uniform(5.5, 9.5),
            'y': rng.uniform(-1.0, 1.0),
        })
    if component_type == 'radial_attraction':
        return HiddenWorldComponent(component_type, {
            'x': center_x,
            'y': center_y,
            'strength': rng.uniform(140.0, 230.0),
        })
    if component_type == 'radial_repulsion':
        return HiddenWorldComponent(component_type, {
            'x': center_x,
            'y': center_y,
            'strength': rng.uniform(95.0, 155.0),
        })
    if component_type == 'tangential_flow':
        return HiddenWorldComponent(component_type, {
            'x': center_x,
            'y': center_y,
            'strength': rng.uniform(95.0, 145.0),
            'radius': rng.uniform(7.0, 9.5),
            'direction': rng.choice([-1.0, 1.0]),
        })
    if component_type == 'time_wave':
        return HiddenWorldComponent(component_type, {
            'axis': rng.choice(['x', 'y']),
            'amplitude': rng.uniform(8.0, 13.0),
            'period': rng.uniform(0.96, 1.44),
            'phase': rng.uniform(0.0, 1.0),
        })
    if component_type == 'localized_pull':
        return HiddenWorldComponent(component_type, {
            'x': center_x,
            'y': center_y,
            'strength': rng.uniform(180.0, 280.0),
            'radius': rng.uniform(7.0, 9.5),
        })
    if component_type == 'localized_push':
        return HiddenWorldComponent(component_type, {
            'x': center_x,
            'y': center_y,
            'strength': rng.uniform(70.0, 125.0),
            'radius': rng.uniform(5.5, 8.5),
        })
    if component_type == 'zero_gravity':
        return HiddenWorldComponent(component_type, {})
    if component_type == 'soft_drag':
        return HiddenWorldComponent(component_type, {
            'friction': rng.uniform(0.985, 0.994),
        })
    raise ValueError(f"Unknown hidden component type: {component_type}")


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
        return ('localized_pull', 'uniform_push')
    if (
        'exponent' in text
        or 'near/mid/far' in text
        or 'distance' in text
        or 'inverse' in text
    ):
        return ('radial_repulsion', 'localized_pull')
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
