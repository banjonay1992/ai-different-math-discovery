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
    if should_expand:
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
        'blocked': bool(blockers),
        'blockers': blockers,
        'residual_pressure': pressure,
        'quality_gate': (
            'expansion changes compute only; proof/readiness/leak gates must still pass'
        ),
    }


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
