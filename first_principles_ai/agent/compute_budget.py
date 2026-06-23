from __future__ import annotations

"""Adaptive compute-budget policy for long discovery campaigns."""

from typing import Any


def plan_adaptive_compute_budget(
    *,
    readiness: dict[str, Any],
    scientist_report: dict[str, Any] | None,
    resource_report: dict[str, Any],
    requested_steps: int,
    requested_seeds: int,
    requested_hidden_worlds: int,
    requested_world_types: list[str] | tuple[str, ...] | None = None,
    max_steps: int,
    max_seeds: int,
    max_hidden_worlds: int,
    enabled: bool = True,
    expansion_threshold: float = 0.55,
) -> dict[str, Any]:
    """
    Expand compute only when residual evidence justifies it.

    The policy is deliberately conservative: memory/resource pressure can block
    expansion, and all increases are capped by explicit caller-provided limits.
    """
    requested = {
        'steps': max(1, int(requested_steps or 1)),
        'seeds': max(1, int(requested_seeds or 1)),
        'hidden_worlds': max(0, int(requested_hidden_worlds or 0)),
    }
    max_limits = {
        'steps': max(requested['steps'], int(max_steps or requested['steps'])),
        'seeds': max(requested['seeds'], int(max_seeds or requested['seeds'])),
        'hidden_worlds': max(
            requested['hidden_worlds'],
            int(max_hidden_worlds or requested['hidden_worlds']),
        ),
    }
    pressure = _residual_pressure(
        readiness=readiness,
        scientist_report=scientist_report or {},
    )
    targeting_plan = _targeted_prep_plan(
        requested_world_types=list(requested_world_types or []),
        requested=requested,
        pressure=pressure,
        scientist_report=scientist_report or {},
        readiness=readiness,
        enabled=enabled,
        expansion_threshold=expansion_threshold,
    )
    blockers = []
    if not enabled:
        blockers.append('adaptive_compute_disabled')
    bounded_windows = dict(resource_report.get('bounded_windows') or {})
    if not bounded_windows.get('operator_outcomes_within_window', True):
        blockers.append('memory_window_exceeded')

    should_expand = (
        enabled
        and not blockers
        and pressure['score'] >= expansion_threshold
    )
    effective = dict(requested)
    if should_expand and targeting_plan['focused']:
        effective['hidden_worlds'] = int(
            targeting_plan['effective_hidden_worlds']
        )
    elif should_expand:
        step_multiplier = 1.0 + min(0.75, pressure['score'] * 0.7)
        effective['steps'] = min(
            max_limits['steps'],
            max(requested['steps'], int(round(requested['steps'] * step_multiplier))),
        )
        if pressure['score'] >= 0.72:
            effective['seeds'] = min(max_limits['seeds'], requested['seeds'] + 1)
        if pressure['hidden_pressure']:
            effective['hidden_worlds'] = min(
                max_limits['hidden_worlds'],
                requested['hidden_worlds'] + 1,
            )

    return {
        'version': 1,
        'policy': 'adaptive_residual_compute',
        'enabled': enabled,
        'expansion_threshold': expansion_threshold,
        'requested': requested,
        'max_limits': max_limits,
        'effective': effective,
        'expanded': effective != requested,
        'targeted': bool(targeting_plan['focused']),
        'targeting_plan': targeting_plan,
        'blocked': bool(blockers),
        'blockers': blockers,
        'residual_pressure': pressure,
        'quality_gate': (
            'expansion changes compute only; proof/readiness/leak gates must still pass'
        ),
    }


def _targeted_prep_plan(
    *,
    requested_world_types: list[str],
    requested: dict[str, int],
    pressure: dict[str, Any],
    scientist_report: dict[str, Any],
    readiness: dict[str, Any],
    enabled: bool,
    expansion_threshold: float,
) -> dict[str, Any]:
    """Choose a smaller high-value prep slice when enough worlds are available."""
    requested_world_types = _dedupe_world_types(requested_world_types)
    requested_hidden_worlds = max(0, int(requested.get('hidden_worlds', 0) or 0))
    requested_case_count = len(requested_world_types) + requested_hidden_worlds
    if not requested_world_types:
        return _targeting_plan(
            requested_world_types=[],
            effective_world_types=[],
            requested_hidden_worlds=requested_hidden_worlds,
            effective_hidden_worlds=requested_hidden_worlds,
            focused=False,
            reasons=['no_world_types_supplied'],
        )
    if (
        not enabled
        or pressure.get('score', 0.0) < expansion_threshold
        or len(requested_world_types) <= 3
    ):
        return _targeting_plan(
            requested_world_types=requested_world_types,
            effective_world_types=requested_world_types,
            requested_hidden_worlds=requested_hidden_worlds,
            effective_hidden_worlds=requested_hidden_worlds,
            focused=False,
            reasons=['insufficient_pressure_or_world_count'],
        )

    priorities = _world_priorities_from_scientist(
        requested_world_types=requested_world_types,
        scientist_report=scientist_report,
        readiness=readiness,
    )
    selected = []
    reasons = []
    if 'standard' in requested_world_types:
        selected.append('standard')
        reasons.append('keep_standard_anchor')
    for world_type, reason in priorities:
        if world_type in selected:
            continue
        selected.append(world_type)
        reasons.append(reason)
        if len(selected) >= min(3, len(requested_world_types)):
            break
    if not priorities:
        for world_type in requested_world_types:
            if world_type in selected:
                continue
            selected.append(world_type)
            reasons.append('fallback_requested_world_order')
            if len(selected) >= min(3, len(requested_world_types)):
                break

    selected = _dedupe_world_types(selected)
    if len(selected) >= len(requested_world_types):
        return _targeting_plan(
            requested_world_types=requested_world_types,
            effective_world_types=requested_world_types,
            requested_hidden_worlds=requested_hidden_worlds,
            effective_hidden_worlds=requested_hidden_worlds,
            focused=False,
            reasons=['targeting_would_not_reduce_world_count'],
        )

    effective_hidden_worlds = requested_hidden_worlds
    hidden_requested = bool(pressure.get('hidden_pressure'))
    if hidden_requested and requested_hidden_worlds == 0:
        proposed_cases = len(selected) + 1
        if proposed_cases <= max(1, requested_case_count):
            effective_hidden_worlds = 1
            reasons.append('spend_saved_world_sweep_on_one_hidden_stress_world')

    return _targeting_plan(
        requested_world_types=requested_world_types,
        effective_world_types=selected,
        requested_hidden_worlds=requested_hidden_worlds,
        effective_hidden_worlds=effective_hidden_worlds,
        focused=True,
        reasons=reasons,
    )


def _targeting_plan(
    *,
    requested_world_types: list[str],
    effective_world_types: list[str],
    requested_hidden_worlds: int,
    effective_hidden_worlds: int,
    focused: bool,
    reasons: list[str],
) -> dict[str, Any]:
    requested_case_count = len(requested_world_types) + requested_hidden_worlds
    effective_case_count = len(effective_world_types) + effective_hidden_worlds
    saved_cases = max(0, requested_case_count - effective_case_count)
    return {
        'policy': 'targeted_readiness_per_compute',
        'focused': bool(focused),
        'requested_world_types': list(requested_world_types),
        'effective_world_types': list(effective_world_types),
        'requested_hidden_worlds': requested_hidden_worlds,
        'effective_hidden_worlds': effective_hidden_worlds,
        'requested_case_count': requested_case_count,
        'effective_case_count': effective_case_count,
        'estimated_saved_case_count': saved_cases,
        'estimated_case_reduction_ratio': _safe_ratio(
            requested_case_count,
            effective_case_count,
        ),
        'reasons': list(reasons),
    }


def _world_priorities_from_scientist(
    *,
    requested_world_types: list[str],
    scientist_report: dict[str, Any],
    readiness: dict[str, Any],
) -> list[tuple[str, str]]:
    requested = set(requested_world_types)
    ranked: list[tuple[float, str, str]] = []
    for item in scientist_report.get('harder_stress_worlds') or []:
        world_type = str(item.get('world_type') or '')
        mapped = _map_stress_world_to_prep_world(world_type, str(item.get('key') or ''))
        if mapped in requested:
            ranked.append((
                float(item.get('priority', 0.0) or 0.0),
                mapped,
                f"stress_world:{item.get('key') or world_type}",
            ))
    for action in scientist_report.get('next_actions') or []:
        world_type = _map_stress_world_to_prep_world(
            str(action.get('suggested_world_type') or ''),
            str(action.get('target') or ''),
        )
        if world_type in requested:
            ranked.append((
                float(action.get('priority', 0.75) or 0.75),
                world_type,
                f"next_action:{action.get('action_kind') or 'scientist'}",
            ))
    missing = set(readiness.get('missing_gates') or [])
    if (
        missing
        & {
            'anomaly_repair_loop',
            'operator_discovery_claims',
            'claim_driven_planning',
        }
        and 'inverse_square_repulsion' in requested
    ):
        ranked.append((
            0.7,
            'inverse_square_repulsion',
            'missing_operator_claim_gate',
        ))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [(world_type, reason) for _priority, world_type, reason in ranked]


def _map_stress_world_to_prep_world(world_type: str, key: str) -> str | None:
    if world_type in {
        'standard',
        'inverse_square_repulsion',
        'localized_gravity',
        'time_varying',
        'sideways_wind',
        'vortex',
        'repulsion',
        'central_force',
        'zero_gravity',
    }:
        return world_type
    if 'localized' in key:
        return 'localized_gravity'
    if 'time' in key:
        return 'time_varying'
    if 'inverse' in key or 'distance' in key:
        return 'inverse_square_repulsion'
    return None


def _dedupe_world_types(world_types: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for world_type in world_types:
        if not world_type or world_type in seen:
            continue
        seen.add(world_type)
        deduped.append(world_type)
    return deduped


def _residual_pressure(
    *,
    readiness: dict[str, Any],
    scientist_report: dict[str, Any],
) -> dict[str, Any]:
    missing_gates = list(readiness.get('missing_gates') or [])
    next_steps = list(readiness.get('next_steps') or [])
    next_actions = list((scientist_report or {}).get('next_actions') or [])
    coverage = dict((scientist_report or {}).get('coverage') or {})
    harder_worlds = list((scientist_report or {}).get('harder_stress_worlds') or [])

    reasons = []
    score = 0.0
    hidden_pressure = False

    if missing_gates:
        gate_pressure = min(0.35, 0.08 * len(missing_gates))
        score += gate_pressure
        reasons.append({
            'reason_kind': 'missing_readiness_gates',
            'score': round(gate_pressure, 3),
            'count': len(missing_gates),
            'examples': missing_gates[:4],
        })
    high_priority_next = _high_priority_count(next_actions)
    if high_priority_next:
        action_pressure = min(0.22, 0.06 * high_priority_next)
        score += action_pressure
        reasons.append({
            'reason_kind': 'scientist_next_actions',
            'score': round(action_pressure, 3),
            'count': high_priority_next,
        })
    residual_experiments = int(coverage.get('residual_experiment_count', 0) or 0)
    if residual_experiments:
        residual_pressure = min(0.25, 0.01 * residual_experiments)
        score += residual_pressure
        reasons.append({
            'reason_kind': 'residual_experiment_backlog',
            'score': round(residual_pressure, 3),
            'count': residual_experiments,
        })
    stress_worlds = int(coverage.get('stress_world_count', 0) or 0)
    if stress_worlds or harder_worlds:
        hidden_pressure = True
        stress_pressure = min(0.18, 0.04 * max(stress_worlds, len(harder_worlds)))
        score += stress_pressure
        reasons.append({
            'reason_kind': 'harder_stress_worlds',
            'score': round(stress_pressure, 3),
            'count': max(stress_worlds, len(harder_worlds)),
        })
    if any('hidden' in str(step).lower() for step in next_steps):
        hidden_pressure = True
        score += 0.08
        reasons.append({
            'reason_kind': 'hidden_holdout_requested',
            'score': 0.08,
        })
    score = round(min(1.0, score), 3)
    return {
        'score': score,
        'hidden_pressure': hidden_pressure,
        'reason_count': len(reasons),
        'reasons': reasons,
    }


def _high_priority_count(actions: list[dict[str, Any]]) -> int:
    return sum(
        1 for action in actions
        if float(action.get('priority', 0.0) or 0.0) >= 0.75
        or action.get('action_kind') in {
            'run_residual_probe',
            'run_harder_hidden_world',
            'falsify_authored_equation',
        }
    )


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    denominator = float(denominator or 0)
    if denominator <= 0:
        return 0.0
    return round(float(numerator or 0) / denominator, 3)
