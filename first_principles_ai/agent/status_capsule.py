"""Orchestrator-facing status capsule for AI Different.

This module is intentionally standalone: it consumes plain dictionaries and
does not import sibling discovery modules. That lets an external orchestrator
read a memory artifact and ask what this module can safely do next.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


HUMAN_LABEL_TERMS = (
    'gravity',
    'vortex',
    'repulsion',
    'sideways_wind',
    'inverse_square',
)

IMPORTANT_READINESS_GATES = (
    'residual_to_theory',
    'proof_like_evaluation',
    'model_disagreement_planning',
    'self_authored_equation_synthesis',
    'domain_world_transfer_evidence',
    'abstraction_discovery_loop',
    'autonomous_next_experiments',
)

IMPORTANT_PROGRESS_GATES = (
    'abstraction_discovery_transfer',
    'heldout_law_stability',
    'blind_holdout_validation',
    'bounded_memory_and_compression',
)


def load_capsule_memory_data(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    memory_path = Path(path)
    if not memory_path.exists() or memory_path.stat().st_size == 0:
        return {}
    with memory_path.open('r', encoding='utf-8') as handle:
        loaded = json.load(handle)
    return dict(loaded or {})


def git_status_for_path(path: str = 'tmp/theory-memory.json') -> str:
    return _run_git_command(['git', 'status', '--short', '--ignored', '--', path])


def git_check_ignore_for_path(path: str = 'tmp/theory-memory.json') -> str:
    return _run_git_command(['git', 'check-ignore', '-v', path], allow_failure=True)


def build_ai_different_status_capsule(
    memory_data: dict[str, Any] | None = None,
    *,
    git_status_text: str = '',
    git_ignored_text: str = '',
    runtime_memory_path: str = 'tmp/theory-memory.json',
) -> dict[str, Any]:
    data = dict(memory_data or {})
    readiness = dict(data.get('discovery_readiness') or {})
    progress = dict(data.get('rediscovery_goal_progress') or {})
    abstraction = dict(data.get('abstraction_discovery_evidence') or {})
    runtime_memory = runtime_memory_state(
        git_status_text=git_status_text,
        git_ignored_text=git_ignored_text,
        runtime_memory_path=runtime_memory_path,
    )
    latest_transfer = latest_abstraction_transfer_result(data, abstraction)
    gates = evidence_gate_capsule(readiness, progress)
    next_experiment = next_non_final_experiment(readiness)
    return {
        'schema_version': 1,
        'capsule_kind': 'orchestrator_status',
        'module': 'AI Different',
        'project_owned_boundary': {
            'project_owned_code_only': True,
            'third_party_checkpoint_used': False,
            'external_model_claim': 'none',
            'note': (
                'Capsule summarizes project memory and local code; it does not '
                'claim ownership of any third-party checkpoint.'
            ),
        },
        'current_capabilities': current_capabilities(latest_transfer, abstraction),
        'evidence_gates': gates,
        'latest_verified_abstraction_transfer_result': latest_transfer,
        'known_weak_spots': known_weak_spots(
            readiness=readiness,
            progress=progress,
            abstraction=abstraction,
            latest_transfer=latest_transfer,
            runtime_memory=runtime_memory,
        ),
        'safe_handoff_questions': safe_handoff_questions(runtime_memory),
        'next_non_final_experiment': next_experiment,
        'runtime_memory': runtime_memory,
    }


def latest_abstraction_transfer_result(
    memory_data: dict[str, Any],
    abstraction: dict[str, Any],
) -> dict[str, Any]:
    latest = dict(abstraction.get('latest_transfer_outcome') or {})
    if not latest:
        outcomes = [
            dict(outcome)
            for outcome in list(memory_data.get('planned_outcomes') or [])
            if outcome.get('experiment_kind') == 'abstraction_transfer_probe'
        ]
        latest = outcomes[-1] if outcomes else {}
    if not latest:
        return {
            'status': 'missing',
            'outcome': None,
            'result_strength': 'not_run',
            'label_clean': True,
            'leak_terms': [],
            'agent_facing_evidence': {},
            'counts': _abstraction_counts(abstraction),
        }

    outcome = str(latest.get('outcome') or 'unknown')
    result_strength = {
        'abstraction_transfer_confirmed': 'success',
        'abstraction_transfer_weak': 'weak',
        'abstraction_reused_same_context': 'weak',
        'abstraction_transfer_absent': 'absent',
    }.get(outcome, 'unknown')
    agent_facing = {
        'abstraction_kind': latest.get('abstraction_kind'),
        'compressed_expression': latest.get('compressed_expression'),
        'target_context': latest.get('context'),
        'outcome': outcome,
        'expected_result': latest.get('expected_result'),
        'falsifies_if': latest.get('falsifies_if'),
    }
    leak_terms = label_leak_terms(agent_facing)
    return {
        'status': 'verified' if result_strength != 'unknown' else 'unclassified',
        'outcome': outcome,
        'result_strength': result_strength,
        'abstraction_key': latest.get('abstraction_key'),
        'abstraction_kind': latest.get('abstraction_kind'),
        'target_context': latest.get('context'),
        'compressed_expression': latest.get('compressed_expression'),
        'label_clean': not leak_terms,
        'leak_terms': leak_terms,
        'agent_facing_evidence': agent_facing,
        'counts': _abstraction_counts(abstraction),
    }


def evidence_gate_capsule(
    readiness: dict[str, Any],
    progress: dict[str, Any],
) -> list[dict[str, Any]]:
    gates = []
    readiness_gates = dict(readiness.get('gates') or {})
    for key in IMPORTANT_READINESS_GATES:
        gate = dict(readiness_gates.get(key) or {})
        if not gate:
            continue
        gates.append({
            'source': 'discovery_readiness',
            'key': key,
            'passed': bool(gate.get('passed')),
            'score': 1.0 if gate.get('passed') else 0.0,
            'evidence': dict(gate.get('evidence') or {}),
            'next_step': gate.get('next_step'),
        })

    progress_gates = dict(progress.get('gates') or {})
    for key in IMPORTANT_PROGRESS_GATES:
        gate = dict(progress_gates.get(key) or {})
        if not gate:
            continue
        gates.append({
            'source': 'rediscovery_goal_progress',
            'key': key,
            'passed': bool(gate.get('passed')),
            'score': float(gate.get('score', 0.0) or 0.0),
            'evidence': dict(gate.get('evidence') or {}),
            'next_step': gate.get('next_step'),
        })
    return gates


def current_capabilities(
    latest_transfer: dict[str, Any],
    abstraction: dict[str, Any],
) -> list[dict[str, Any]]:
    transfer_outcomes = int(abstraction.get('transfer_outcome_count', 0) or 0)
    return [
        {
            'key': 'raw_observation_to_equation_candidates',
            'status': 'available',
            'evidence': 'equation campaign and discovery-loop tests',
        },
        {
            'key': 'residual_to_concept_to_operator_loop',
            'status': 'available',
            'evidence': 'concept proposals, operator priors, and proof checks',
        },
        {
            'key': 'concept_equivalence_compression',
            'status': 'available',
            'evidence': (
                f"{int(abstraction.get('bridge_count', 0) or 0)} abstraction bridge(s)"
            ),
        },
        {
            'key': 'empirical_abstraction_transfer_probe',
            'status': 'verified' if transfer_outcomes else 'planned',
            'evidence': latest_transfer.get('outcome') or 'no transfer outcome yet',
        },
        {
            'key': 'bounded_runtime_memory_reporting',
            'status': 'available',
            'evidence': 'capsule reports dirty/ignored runtime memory state',
        },
    ]


def known_weak_spots(
    *,
    readiness: dict[str, Any],
    progress: dict[str, Any],
    abstraction: dict[str, Any],
    latest_transfer: dict[str, Any],
    runtime_memory: dict[str, Any],
) -> list[dict[str, Any]]:
    weak_spots = []
    missing = list(readiness.get('missing_gates') or [])
    for gate in missing[:6]:
        weak_spots.append({
            'key': f'missing_gate:{gate}',
            'status': 'open',
            'evidence': gate,
        })
    blockers = list(progress.get('blockers') or [])
    for blocker in blockers[:4]:
        key = f'progress_blocker:{blocker}'
        if any(item['key'] == key for item in weak_spots):
            continue
        weak_spots.append({
            'key': key,
            'status': 'open',
            'evidence': blocker,
        })
    if int(abstraction.get('transfer_outcome_count', 0) or 0) <= 0:
        weak_spots.append({
            'key': 'abstraction_transfer_not_empirical',
            'status': 'open',
            'evidence': 'no persisted abstraction-transfer outcome',
        })
    elif latest_transfer.get('result_strength') in {'weak', 'absent'}:
        weak_spots.append({
            'key': 'abstraction_transfer_not_confirmed',
            'status': 'open',
            'evidence': latest_transfer.get('outcome'),
        })
    if runtime_memory.get('dirty') and not runtime_memory.get('ignored'):
        weak_spots.append({
            'key': 'runtime_memory_dirty_unignored',
            'status': 'watch',
            'evidence': runtime_memory.get('path'),
        })
    return weak_spots[:8]


def safe_handoff_questions(runtime_memory: dict[str, Any]) -> list[str]:
    questions = [
        'Should the next pass run the non-final abstraction transfer campaign again with confirmed, weak, or absent pressure?',
        'Which missing readiness gate should be repaired before any watched final run?',
        'Should heavier validation stay local or move to the cheapest suitable Hugging Face job?',
        'Should the capsule be attached to the orchestrator summary after every non-final run?',
    ]
    if runtime_memory.get('dirty'):
        questions.insert(
            1,
            'Should tmp/theory-memory.json remain runtime-only, or should this exact memory snapshot be checkpointed intentionally?',
        )
    return questions


def next_non_final_experiment(readiness: dict[str, Any]) -> dict[str, Any]:
    actions = [
        dict(action)
        for action in list(readiness.get('recommended_actions') or [])
        if not action.get('runs_final')
    ]
    if actions:
        action = actions[0]
        return {
            'action_kind': action.get('action_kind'),
            'command': action.get('command'),
            'reason': action.get('reason'),
            'runs_final': False,
            'safe_to_run_without_user_watched_final': True,
        }
    return {
        'action_kind': 'non_final_abstraction_transfer_campaign',
        'command': (
            'python3 first_principles_ai/main.py '
            '--abstraction-transfer-campaign '
            '--theory-memory-file tmp/theory-memory.json'
        ),
        'reason': 'default safe probe for empirical abstraction-transfer evidence',
        'runs_final': False,
        'safe_to_run_without_user_watched_final': True,
    }


def runtime_memory_state(
    *,
    git_status_text: str,
    git_ignored_text: str = '',
    runtime_memory_path: str = 'tmp/theory-memory.json',
) -> dict[str, Any]:
    lines = [
        line for line in git_status_text.splitlines()
        if runtime_memory_path in line
    ]
    ignored = bool(git_ignored_text.strip()) or any(
        line.startswith('!!') for line in lines
    )
    dirty_lines = [
        line for line in lines
        if not line.startswith('!!')
    ]
    staged = any(
        len(line) >= 2 and line[0] not in {' ', '?', '!'}
        for line in dirty_lines
    )
    unstaged = any(
        len(line) >= 2 and line[1] not in {' ', '?', '!'}
        for line in dirty_lines
    )
    untracked = any(line.startswith('??') for line in dirty_lines)
    dirty = bool(dirty_lines)
    recommendation = (
        'leave_unstaged_runtime_memory'
        if dirty and not staged
        else 'review_before_staging'
        if dirty
        else 'clean'
    )
    return {
        'path': runtime_memory_path,
        'dirty': dirty,
        'ignored': ignored,
        'staged': staged,
        'unstaged': unstaged,
        'untracked': untracked,
        'status_lines': dirty_lines,
        'recommendation': recommendation,
    }


def label_leak_terms(payload: dict[str, Any]) -> list[str]:
    text = json.dumps(payload, sort_keys=True).lower()
    return sorted({term for term in HUMAN_LABEL_TERMS if term in text})


def _abstraction_counts(abstraction: dict[str, Any]) -> dict[str, int]:
    return {
        'transfer_outcome_count': int(abstraction.get('transfer_outcome_count', 0) or 0),
        'transfer_confirmed_count': int(
            abstraction.get('transfer_confirmed_count', 0) or 0
        ),
        'transfer_weak_count': int(abstraction.get('transfer_weak_count', 0) or 0),
        'transfer_absent_count': int(abstraction.get('transfer_absent_count', 0) or 0),
        'bridge_count': int(abstraction.get('bridge_count', 0) or 0),
    }


def _run_git_command(command: list[str], allow_failure: bool = False) -> str:
    try:
        completed = subprocess.run(
            command,
            check=not allow_failure,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        if allow_failure:
            return ''
        raise RuntimeError(f"failed to run {' '.join(command)}") from error
    return completed.stdout.rstrip('\n')
