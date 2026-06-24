from __future__ import annotations

"""Optional vector/tensor kernels for simulator force fields.

The discovery system still reasons over ordinary Python observations.  This
module only accelerates simulator-side force math, keeping benchmark labels and
world configuration outside the agent-facing data path.
"""

import math
from typing import Any


SUPPORTED_FORCE_COMPONENT_TYPES = {
    'cutoff_radial',
    'piecewise_radial',
    'periodic_regime_uniform',
}


def available_force_backends() -> dict[str, Any]:
    """Report optional vector/tensor backends without importing them at module load."""
    report = {
        'numpy': False,
        'torch': False,
        'cuda': False,
        'cuda_device': None,
    }
    try:
        import numpy  # noqa: F401

        report['numpy'] = True
    except Exception:
        pass
    try:
        import torch  # type: ignore

        report['torch'] = True
        report['cuda'] = bool(torch.cuda.is_available())
        if report['cuda']:
            report['cuda_device'] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return report


def resolve_force_backend(requested: str | None) -> dict[str, Any]:
    """Resolve a requested backend into an executable backend plus fallback reason."""
    requested = (requested or 'python').lower().replace('-', '_')
    if requested in {'none', 'python'}:
        return {
            'requested_backend': requested,
            'backend': 'python',
            'accelerated': False,
            'fallback_reason': None,
            'available_backends': available_force_backends(),
        }

    available = available_force_backends()
    if requested in {'auto', 'tensor'}:
        if available['cuda']:
            backend = 'cuda'
        elif available['torch']:
            backend = 'torch'
        elif available['numpy']:
            backend = 'numpy'
        else:
            backend = 'python'
        return {
            'requested_backend': requested,
            'backend': backend,
            'accelerated': backend != 'python',
            'fallback_reason': None if backend != 'python' else 'no_tensor_backend_available',
            'available_backends': available,
        }

    if requested == 'cuda':
        if available['cuda']:
            backend = 'cuda'
            reason = None
        elif available['torch']:
            backend = 'torch'
            reason = 'cuda_unavailable_using_torch_cpu'
        elif available['numpy']:
            backend = 'numpy'
            reason = 'cuda_unavailable_using_numpy'
        else:
            backend = 'python'
            reason = 'cuda_unavailable_no_tensor_backend'
        return {
            'requested_backend': requested,
            'backend': backend,
            'accelerated': backend != 'python',
            'fallback_reason': reason,
            'available_backends': available,
        }

    if requested == 'torch':
        if available['torch']:
            backend = 'torch'
            reason = None
        elif available['numpy']:
            backend = 'numpy'
            reason = 'torch_unavailable_using_numpy'
        else:
            backend = 'python'
            reason = 'torch_unavailable_no_tensor_backend'
        return {
            'requested_backend': requested,
            'backend': backend,
            'accelerated': backend != 'python',
            'fallback_reason': reason,
            'available_backends': available,
        }

    if requested == 'numpy':
        backend = 'numpy' if available['numpy'] else 'python'
        return {
            'requested_backend': requested,
            'backend': backend,
            'accelerated': backend != 'python',
            'fallback_reason': None if backend == 'numpy' else 'numpy_unavailable',
            'available_backends': available,
        }

    raise ValueError(f"Unknown force backend: {requested}")


def can_vectorize_force_components(force_components: list[dict[str, Any]]) -> bool:
    return all(
        component.get('type') in SUPPORTED_FORCE_COMPONENT_TYPES
        for component in force_components
    )


def compute_external_force_deltas(
    *,
    positions: list[tuple[float, float]],
    dt: float,
    time_value: float,
    gravity: float,
    uniform_force: dict[str, Any] | None,
    central_force: dict[str, Any] | None,
    gravity_wells: list[dict[str, Any]],
    repulsion_zones: list[dict[str, Any]],
    inverse_square_repulsions: list[dict[str, Any]],
    vortex_fields: list[dict[str, Any]],
    time_varying_force: dict[str, Any] | None,
    force_components: list[dict[str, Any]],
    backend: str,
) -> list[tuple[float, float]]:
    """Return velocity deltas from all external force fields for one timestep."""
    if not positions:
        return []
    backend = (backend or 'numpy').lower()
    if backend in {'torch', 'cuda'}:
        try:
            return _torch_external_force_deltas(
                positions=positions,
                dt=dt,
                time_value=time_value,
                gravity=gravity,
                uniform_force=uniform_force,
                central_force=central_force,
                gravity_wells=gravity_wells,
                repulsion_zones=repulsion_zones,
                inverse_square_repulsions=inverse_square_repulsions,
                vortex_fields=vortex_fields,
                time_varying_force=time_varying_force,
                force_components=force_components,
                backend=backend,
            )
        except Exception:
            if backend == 'cuda':
                return _torch_external_force_deltas(
                    positions=positions,
                    dt=dt,
                    time_value=time_value,
                    gravity=gravity,
                    uniform_force=uniform_force,
                    central_force=central_force,
                    gravity_wells=gravity_wells,
                    repulsion_zones=repulsion_zones,
                    inverse_square_repulsions=inverse_square_repulsions,
                    vortex_fields=vortex_fields,
                    time_varying_force=time_varying_force,
                    force_components=force_components,
                    backend='torch',
                )
            raise
    return _numpy_external_force_deltas(
        positions=positions,
        dt=dt,
        time_value=time_value,
        gravity=gravity,
        uniform_force=uniform_force,
        central_force=central_force,
        gravity_wells=gravity_wells,
        repulsion_zones=repulsion_zones,
        inverse_square_repulsions=inverse_square_repulsions,
        vortex_fields=vortex_fields,
        time_varying_force=time_varying_force,
        force_components=force_components,
    )


def _numpy_external_force_deltas(
    *,
    positions: list[tuple[float, float]],
    dt: float,
    time_value: float,
    gravity: float,
    uniform_force: dict[str, Any] | None,
    central_force: dict[str, Any] | None,
    gravity_wells: list[dict[str, Any]],
    repulsion_zones: list[dict[str, Any]],
    inverse_square_repulsions: list[dict[str, Any]],
    vortex_fields: list[dict[str, Any]],
    time_varying_force: dict[str, Any] | None,
    force_components: list[dict[str, Any]],
) -> list[tuple[float, float]]:
    import numpy as np

    pos = np.asarray(positions, dtype=np.float64)
    x = pos[:, 0]
    y = pos[:, 1]
    dvx = np.zeros_like(x)
    dvy = np.zeros_like(y)

    dvy -= float(gravity) * dt
    if uniform_force is not None:
        dvx += float(uniform_force.get('x', 0.0)) * dt
        dvy += float(uniform_force.get('y', 0.0)) * dt

    if central_force is not None:
        dx = float(central_force['x']) - x
        dy = float(central_force['y']) - y
        dist_sq = np.maximum(dx * dx + dy * dy, 0.01)
        dist = np.sqrt(dist_sq)
        force = float(central_force['strength']) / dist_sq
        dvx += (dx / dist) * force * dt
        dvy += (dy / dist) * force * dt

    for well in gravity_wells:
        dx = float(well['x']) - x
        dy = float(well['y']) - y
        dist_sq_raw = dx * dx + dy * dy
        dist = np.sqrt(dist_sq_raw)
        radius = float(well['radius'])
        mask = (dist >= 0.01) & (dist <= radius)
        safe_dist_sq = np.maximum(dist_sq_raw, 0.25)
        safe_dist = np.where(mask, dist, 1.0)
        falloff = 1.0 - (dist / radius)
        force = float(well['strength']) * falloff / safe_dist_sq
        dvx += np.where(mask, (dx / safe_dist) * force * dt, 0.0)
        dvy += np.where(mask, (dy / safe_dist) * force * dt, 0.0)

    for zone in repulsion_zones:
        dx = x - float(zone['x'])
        dy = y - float(zone['y'])
        dist = np.sqrt(dx * dx + dy * dy)
        radius = float(zone['radius'])
        mask = (dist < radius) & (dist > 0.01)
        safe_dist = np.where(mask, dist, 1.0)
        force = float(zone['strength']) * (1.0 - dist / radius)
        dvx += np.where(mask, (dx / safe_dist) * force * dt, 0.0)
        dvy += np.where(mask, (dy / safe_dist) * force * dt, 0.0)

    for source in inverse_square_repulsions:
        dx = x - float(source['x'])
        dy = y - float(source['y'])
        dist_sq = np.maximum(dx * dx + dy * dy, 0.25)
        dist = np.sqrt(dist_sq)
        force = float(source['strength']) / dist_sq
        dvx += (dx / dist) * force * dt
        dvy += (dy / dist) * force * dt

    for vortex in vortex_fields:
        dx = x - float(vortex['x'])
        dy = y - float(vortex['y'])
        dist = np.sqrt(dx * dx + dy * dy)
        radius = float(vortex['radius'])
        mask = (dist >= 0.01) & (dist <= radius)
        safe_dist = np.where(mask, dist, 1.0)
        falloff = 1.0 - (dist / radius)
        direction = float(vortex.get('direction', 1.0))
        tangent_x = -dy / safe_dist * direction
        tangent_y = dx / safe_dist * direction
        force = float(vortex['strength']) * falloff
        dvx += np.where(mask, tangent_x * force * dt, 0.0)
        dvy += np.where(mask, tangent_y * force * dt, 0.0)

    _apply_time_varying_numpy(dvx, dvy, time_value, dt, time_varying_force)
    for component in force_components:
        _apply_component_numpy(dvx, dvy, x, y, time_value, dt, component)

    return [
        (float(delta_x), float(delta_y))
        for delta_x, delta_y in zip(dvx.tolist(), dvy.tolist())
    ]


def _apply_time_varying_numpy(dvx, dvy, time_value, dt, force_config):
    if force_config is None:
        return
    axis = force_config.get('axis', 'x')
    amplitude = float(force_config.get('amplitude', 0.0))
    period = max(float(force_config.get('period', 1.0)), 1e-6)
    phase = float(force_config.get('phase', 0.0))
    force = amplitude * math.sin((2 * math.pi * time_value / period) + phase)
    if axis == 'x':
        dvx += force * dt
    else:
        dvy += force * dt


def _apply_component_numpy(dvx, dvy, x, y, time_value, dt, component):
    import numpy as np

    component_type = component.get('type')
    params = dict(component.get('params') or {})
    if component_type == 'cutoff_radial':
        _apply_radial_numpy(
            dvx,
            dvy,
            x,
            y,
            center_x=float(params['x']),
            center_y=float(params['y']),
            strength=float(params['strength']),
            exponent=float(params.get('exponent', 2.0)),
            direction=float(params.get('direction', 1.0)),
            dt=dt,
            max_radius=float(params['radius']),
        )
    elif component_type == 'piecewise_radial':
        center_x = float(params['x'])
        center_y = float(params['y'])
        split_radius = float(params['split_radius'])
        dx = x - center_x
        dy = y - center_y
        dist_sq = dx * dx + dy * dy
        dist = np.sqrt(np.maximum(dist_sq, 0.25))
        inner = dict(params.get('inner') or {})
        outer = dict(params.get('outer') or {})
        for mask, regime in (
            (dist <= split_radius, inner),
            (dist > split_radius, outer),
        ):
            strength = float(regime.get('strength', 0.0))
            if abs(strength) < 1e-12:
                continue
            exponent = max(float(regime.get('exponent', 2.0)), 0.0)
            direction = float(regime.get('direction', 1.0))
            force = strength / np.maximum(np.power(dist, exponent), 0.25)
            dvx += np.where(mask, direction * (dx / dist) * force * dt, 0.0)
            dvy += np.where(mask, direction * (dy / dist) * force * dt, 0.0)
    elif component_type == 'periodic_regime_uniform':
        period = max(float(params.get('period', 1.0)), 1e-6)
        duty_cycle = min(max(float(params.get('duty_cycle', 0.5)), 0.0), 1.0)
        phase = float(params.get('phase', 0.0))
        cycle_position = ((time_value + phase) % period) / period
        force = params.get('on_force', {}) if cycle_position <= duty_cycle else params.get('off_force', {})
        dvx += float(force.get('x', 0.0)) * dt
        dvy += float(force.get('y', 0.0)) * dt


def _apply_radial_numpy(
    dvx,
    dvy,
    x,
    y,
    *,
    center_x: float,
    center_y: float,
    strength: float,
    exponent: float,
    direction: float,
    dt: float,
    max_radius: float | None,
):
    import numpy as np

    exponent = max(float(exponent), 0.0)
    dx = x - center_x
    dy = y - center_y
    dist_sq = dx * dx + dy * dy
    dist = np.sqrt(np.maximum(dist_sq, 0.25))
    if max_radius is None:
        mask = np.ones_like(dist, dtype=bool)
    else:
        mask = dist <= float(max_radius)
    force = strength / np.maximum(np.power(dist, exponent), 0.25)
    dvx += np.where(mask, direction * (dx / dist) * force * dt, 0.0)
    dvy += np.where(mask, direction * (dy / dist) * force * dt, 0.0)


def _torch_external_force_deltas(
    *,
    positions: list[tuple[float, float]],
    dt: float,
    time_value: float,
    gravity: float,
    uniform_force: dict[str, Any] | None,
    central_force: dict[str, Any] | None,
    gravity_wells: list[dict[str, Any]],
    repulsion_zones: list[dict[str, Any]],
    inverse_square_repulsions: list[dict[str, Any]],
    vortex_fields: list[dict[str, Any]],
    time_varying_force: dict[str, Any] | None,
    force_components: list[dict[str, Any]],
    backend: str,
) -> list[tuple[float, float]]:
    import torch  # type: ignore

    device_name = 'cuda' if backend == 'cuda' and torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    pos = torch.tensor(positions, dtype=torch.float64, device=device)
    x = pos[:, 0]
    y = pos[:, 1]
    dvx = torch.zeros_like(x)
    dvy = torch.zeros_like(y)

    dvy -= float(gravity) * dt
    if uniform_force is not None:
        dvx += float(uniform_force.get('x', 0.0)) * dt
        dvy += float(uniform_force.get('y', 0.0)) * dt

    if central_force is not None:
        dx = float(central_force['x']) - x
        dy = float(central_force['y']) - y
        dist_sq = torch.clamp(dx * dx + dy * dy, min=0.01)
        dist = torch.sqrt(dist_sq)
        force = float(central_force['strength']) / dist_sq
        dvx += (dx / dist) * force * dt
        dvy += (dy / dist) * force * dt

    for well in gravity_wells:
        dx = float(well['x']) - x
        dy = float(well['y']) - y
        dist_sq_raw = dx * dx + dy * dy
        dist = torch.sqrt(dist_sq_raw)
        radius = float(well['radius'])
        mask = (dist >= 0.01) & (dist <= radius)
        safe_dist_sq = torch.clamp(dist_sq_raw, min=0.25)
        safe_dist = torch.where(mask, dist, torch.ones_like(dist))
        falloff = 1.0 - (dist / radius)
        force = float(well['strength']) * falloff / safe_dist_sq
        dvx += torch.where(mask, (dx / safe_dist) * force * dt, torch.zeros_like(dvx))
        dvy += torch.where(mask, (dy / safe_dist) * force * dt, torch.zeros_like(dvy))

    for zone in repulsion_zones:
        dx = x - float(zone['x'])
        dy = y - float(zone['y'])
        dist = torch.sqrt(dx * dx + dy * dy)
        radius = float(zone['radius'])
        mask = (dist < radius) & (dist > 0.01)
        safe_dist = torch.where(mask, dist, torch.ones_like(dist))
        force = float(zone['strength']) * (1.0 - dist / radius)
        dvx += torch.where(mask, (dx / safe_dist) * force * dt, torch.zeros_like(dvx))
        dvy += torch.where(mask, (dy / safe_dist) * force * dt, torch.zeros_like(dvy))

    for source in inverse_square_repulsions:
        dx = x - float(source['x'])
        dy = y - float(source['y'])
        dist_sq = torch.clamp(dx * dx + dy * dy, min=0.25)
        dist = torch.sqrt(dist_sq)
        force = float(source['strength']) / dist_sq
        dvx += (dx / dist) * force * dt
        dvy += (dy / dist) * force * dt

    for vortex in vortex_fields:
        dx = x - float(vortex['x'])
        dy = y - float(vortex['y'])
        dist = torch.sqrt(dx * dx + dy * dy)
        radius = float(vortex['radius'])
        mask = (dist >= 0.01) & (dist <= radius)
        safe_dist = torch.where(mask, dist, torch.ones_like(dist))
        falloff = 1.0 - (dist / radius)
        direction = float(vortex.get('direction', 1.0))
        tangent_x = -dy / safe_dist * direction
        tangent_y = dx / safe_dist * direction
        force = float(vortex['strength']) * falloff
        dvx += torch.where(mask, tangent_x * force * dt, torch.zeros_like(dvx))
        dvy += torch.where(mask, tangent_y * force * dt, torch.zeros_like(dvy))

    _apply_time_varying_torch(dvx, dvy, time_value, dt, time_varying_force)
    for component in force_components:
        _apply_component_torch(dvx, dvy, x, y, time_value, dt, component)

    deltas = torch.stack((dvx, dvy), dim=1).detach().cpu().tolist()
    return [(float(delta_x), float(delta_y)) for delta_x, delta_y in deltas]


def _apply_time_varying_torch(dvx, dvy, time_value, dt, force_config):
    if force_config is None:
        return
    axis = force_config.get('axis', 'x')
    amplitude = float(force_config.get('amplitude', 0.0))
    period = max(float(force_config.get('period', 1.0)), 1e-6)
    phase = float(force_config.get('phase', 0.0))
    force = amplitude * math.sin((2 * math.pi * time_value / period) + phase)
    if axis == 'x':
        dvx += force * dt
    else:
        dvy += force * dt


def _apply_component_torch(dvx, dvy, x, y, time_value, dt, component):
    import torch  # type: ignore

    component_type = component.get('type')
    params = dict(component.get('params') or {})
    if component_type == 'cutoff_radial':
        _apply_radial_torch(
            dvx,
            dvy,
            x,
            y,
            center_x=float(params['x']),
            center_y=float(params['y']),
            strength=float(params['strength']),
            exponent=float(params.get('exponent', 2.0)),
            direction=float(params.get('direction', 1.0)),
            dt=dt,
            max_radius=float(params['radius']),
        )
    elif component_type == 'piecewise_radial':
        center_x = float(params['x'])
        center_y = float(params['y'])
        split_radius = float(params['split_radius'])
        dx = x - center_x
        dy = y - center_y
        dist_sq = dx * dx + dy * dy
        dist = torch.sqrt(torch.clamp(dist_sq, min=0.25))
        inner = dict(params.get('inner') or {})
        outer = dict(params.get('outer') or {})
        for mask, regime in (
            (dist <= split_radius, inner),
            (dist > split_radius, outer),
        ):
            strength = float(regime.get('strength', 0.0))
            if abs(strength) < 1e-12:
                continue
            exponent = max(float(regime.get('exponent', 2.0)), 0.0)
            direction = float(regime.get('direction', 1.0))
            force = strength / torch.clamp(torch.pow(dist, exponent), min=0.25)
            dvx += torch.where(mask, direction * (dx / dist) * force * dt, torch.zeros_like(dvx))
            dvy += torch.where(mask, direction * (dy / dist) * force * dt, torch.zeros_like(dvy))
    elif component_type == 'periodic_regime_uniform':
        period = max(float(params.get('period', 1.0)), 1e-6)
        duty_cycle = min(max(float(params.get('duty_cycle', 0.5)), 0.0), 1.0)
        phase = float(params.get('phase', 0.0))
        cycle_position = ((time_value + phase) % period) / period
        force = params.get('on_force', {}) if cycle_position <= duty_cycle else params.get('off_force', {})
        dvx += float(force.get('x', 0.0)) * dt
        dvy += float(force.get('y', 0.0)) * dt


def _apply_radial_torch(
    dvx,
    dvy,
    x,
    y,
    *,
    center_x: float,
    center_y: float,
    strength: float,
    exponent: float,
    direction: float,
    dt: float,
    max_radius: float | None,
):
    import torch  # type: ignore

    exponent = max(float(exponent), 0.0)
    dx = x - center_x
    dy = y - center_y
    dist_sq = dx * dx + dy * dy
    dist = torch.sqrt(torch.clamp(dist_sq, min=0.25))
    if max_radius is None:
        mask = torch.ones_like(dist, dtype=torch.bool)
    else:
        mask = dist <= float(max_radius)
    force = strength / torch.clamp(torch.pow(dist, exponent), min=0.25)
    dvx += torch.where(mask, direction * (dx / dist) * force * dt, torch.zeros_like(dvx))
    dvy += torch.where(mask, direction * (dy / dist) * force * dt, torch.zeros_like(dvy))
