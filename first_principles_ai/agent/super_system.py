"""
Consolidated audit layer for the discovery system.

This module does not run experiments. It connects the current theory memory,
experiment design, operator repair, domain rediscovery, compression, theorem,
and artifact surfaces into one report that can be printed or saved.
"""

from __future__ import annotations

from typing import Any

from .discovery_loop import CumulativeTheoryMemory


DEFAULT_SUPER_SYSTEM_WORLDS = [
    'standard',
    'sideways_wind',
    'vortex',
    'inverse_square_repulsion',
    'localized_gravity',
    'time_varying',
]

SUPPORTED_INTERVENTIONS = [
    {
        'type': 'wait',
        'purpose': 'observe a natural transition when time or phase is the variable',
    },
    {
        'type': 'push',
        'purpose': 'apply a controlled velocity or force perturbation',
    },
    {
        'type': 'spawn',
        'purpose': 'place a new probe object at a discriminating location',
    },
    {
        'type': 'remove',
        'purpose': 'test whether a candidate source object is causal',
    },
    {
        'type': 'move',
        'purpose': 'relocate an existing object to a theory-disagreement point',
    },
    {
        'type': 'freeze',
        'purpose': 'hold an object fixed to separate source motion from field effects',
    },
    {
        'type': 'duplicate',
        'purpose': 'copy an object to test superposition, symmetry, or scaling',
    },
]

SUPPORTED_INTERVENTION_TYPES = {
    item['type']
    for item in SUPPORTED_INTERVENTIONS
}

FORCE_PLAN_SOURCE_KINDS = {
    'selected_law_conflict_resolution',
    'blind_holdout_validation',
    'model_disagreement_domain_split',
    'localized_gravity_structure_probe',
    'domain_predicate_discovery',
}

CONNECTION_GRAPH = [
    {
        'from': 'public_observations',
        'to': 'residual_equations',
        'connection': 'equation workbench extracts candidate laws',
    },
    {
        'from': 'residual_equations',
        'to': 'cumulative_memory',
        'connection': 'theory families store proof, support, and counterexamples',
    },
    {
        'from': 'cumulative_memory',
        'to': 'experiment_design',
        'connection': 'memory emits falsification probes and intervention plans',
    },
    {
        'from': 'experiment_design',
        'to': 'world_actions',
        'connection': 'plans compile into wait, push, spawn, remove, move, freeze, duplicate',
    },
    {
        'from': 'domain_worlds',
        'to': 'transfer',
        'connection': 'math-domain discoveries create cross-domain tests',
    },
    {
        'from': 'operator_priors',
        'to': 'repair_loop',
        'connection': 'invented operators produce anomalies, repairs, and validation probes',
    },
    {
        'from': 'theorem_memory',
        'to': 'blind_holdouts',
        'connection': 'consolidated laws become holdout and replay obligations',
    },
    {
        'from': 'compression',
        'to': 'long_runs',
        'connection': 'bounded summaries keep future runs small enough to repeat',
    },
]


def _experiment_source(plan: dict[str, Any]) -> str:
    experiment_kind = plan.get('experiment_kind')
    if experiment_kind == 'equation_invariant_exponent_resolution':
        return 'planned_equation_invariant_resolution'
    if experiment_kind == 'selected_law_replay':
        return 'planned_selected_law_replay'
    if experiment_kind == 'selected_law_conflict_resolution':
        return 'planned_selected_law_conflict_resolution'
    if experiment_kind == 'blind_holdout_validation':
        return 'planned_blind_holdout_validation'
    if experiment_kind == 'model_disagreement_domain_split':
        return 'planned_model_disagreement_domain_split'
    if experiment_kind == 'localized_gravity_structure_probe':
        return 'planned_localized_gravity_structure_probe'
    if experiment_kind == 'domain_predicate_discovery':
        return 'planned_domain_predicate_discovery'
    if str(experiment_kind).startswith('operator_prior_'):
        return 'planned_operator_prior_probe'
    return 'planned_model_disagreement_probe'


def planned_probe_actions(
    plan: dict[str, Any],
    max_actions: int = 4,
) -> list[dict[str, Any]]:
    """Compile a planned experiment into supported public world actions."""
    signature = dict(plan.get('disagreement_signature') or {})
    source = _experiment_source(plan)
    actions = []
    for point in list(signature.get('probe_points') or [])[:max_actions]:
        if not {'x', 'y'} <= set(point):
            continue
        action = {
            'type': 'spawn',
            'x': float(point['x']),
            'y': float(point['y']),
            'vx': 0.0,
            'vy': 0.0,
            'source': source,
            'probe_label': point.get('label'),
            'disagreement_mode': signature.get('mode'),
        }
        if plan.get('invariant_key'):
            action['invariant_key'] = plan.get('invariant_key')
        if plan.get('operator_prior_key'):
            action['operator_prior_key'] = plan.get('operator_prior_key')
        actions.append(action)
    if actions:
        return actions

    action = dict(plan.get('probe_action') or {})
    if action.get('type') in SUPPORTED_INTERVENTION_TYPES:
        if plan.get('experiment_kind') in FORCE_PLAN_SOURCE_KINDS:
            action['source'] = source
        else:
            action.setdefault('source', source)
        return [action]
    return []


def theory_beliefs_from_plan(plan: dict[str, Any]) -> list[dict[str, Any]]:
    signature = dict(plan.get('disagreement_signature') or {})
    predictions = [
        dict(prediction)
        for prediction in list(signature.get('rival_predictions') or [])
    ]
    if not predictions:
        label = (
            plan.get('primary_theory_label')
            or plan.get('theory_kind')
            or 'candidate_theory'
        )
        return [{
            'theory': str(label),
            'belief': 1.0,
            'score': round(float(plan.get('priority', 1.0) or 1.0), 3),
            'prediction': plan.get('expected_result'),
            'falsified_if': plan.get('falsifies_if'),
        }]

    raw_scores = [
        max(0.0, float(prediction.get('score', 0.0) or 0.0))
        for prediction in predictions
    ]
    score_total = sum(raw_scores)
    equal_belief = 1.0 / max(1, len(predictions))
    beliefs = []
    for prediction, raw_score in zip(predictions, raw_scores):
        belief = raw_score / score_total if score_total > 0 else equal_belief
        beliefs.append({
            'theory': str(
                prediction.get('theory_key')
                or prediction.get('theory_kind')
                or 'candidate_theory'
            ),
            'belief': round(belief, 3),
            'score': round(raw_score, 3),
            'prediction': prediction.get('prediction'),
            'falsified_if': prediction.get('falsified_if'),
        })
    return beliefs


def _float_text(value: Any, default: float = 0.0) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return f"{default:.2f}"


def format_intervention_action(action: dict[str, Any]) -> str:
    if not action:
        return 'observe/wait for the next natural transition'
    action_type = str(action.get('type') or 'wait')
    if action_type == 'wait':
        return 'observe/wait for the next natural transition'
    if action_type == 'spawn':
        return (
            f"spawn probe at x={_float_text(action.get('x'))}, "
            f"y={_float_text(action.get('y'))}"
        )
    if action_type == 'move':
        return (
            f"move object {action.get('object_id', '?')} to "
            f"x={_float_text(action.get('x'))}, "
            f"y={_float_text(action.get('y'))}"
        )
    if action_type == 'push':
        return (
            f"push object {action.get('object_id', '?')} with "
            f"fx={_float_text(action.get('fx'))}, "
            f"fy={_float_text(action.get('fy'))}"
        )
    if action_type in {'remove', 'freeze', 'duplicate'}:
        return f"{action_type} object {action.get('object_id', '?')}"
    return action_type


def experiment_design_from_plan(plan: dict[str, Any]) -> dict[str, Any]:
    planned_actions = planned_probe_actions(plan)
    action = (
        dict(plan.get('probe_action') or {})
        if plan.get('probe_action')
        else (planned_actions[0] if planned_actions else {'type': 'wait'})
    )
    if action.get('type') in SUPPORTED_INTERVENTION_TYPES:
        action.setdefault('source', _experiment_source(plan))
    signature = dict(plan.get('disagreement_signature') or {})
    question = (
        signature.get('question')
        or plan.get('reason')
        or 'Which theory loses predictive power under the next probe?'
    )
    return {
        'experiment_kind': plan.get('experiment_kind'),
        'priority': round(float(plan.get('priority', 0.0) or 0.0), 3),
        'world_type': plan.get('world_type'),
        'seed': plan.get('seed'),
        'object_count': plan.get('object_count'),
        'steps': plan.get('steps'),
        'question': question,
        'beliefs': theory_beliefs_from_plan(plan),
        'intervention': action,
        'intervention_text': format_intervention_action(action),
        'expected_result': plan.get('expected_result'),
        'falsifies_if': plan.get('falsifies_if'),
        'counterexample_reward': (
            'high'
            if 'counterexample' in str(plan.get('experiment_kind', ''))
            or plan.get('falsifies_if')
            else 'normal'
        ),
    }


def build_experiment_design_cockpit(
    theory_memory: CumulativeTheoryMemory,
    *,
    world_types: list[str] | None = None,
    object_counts: list[int] | None = None,
    steps: int = 240,
    limit: int = 5,
) -> list[dict[str, Any]]:
    plans = theory_memory.planned_experiments(
        world_types=world_types or DEFAULT_SUPER_SYSTEM_WORLDS,
        object_counts=object_counts or [5],
        steps=steps,
        limit=limit,
    )
    return [experiment_design_from_plan(plan) for plan in plans]


def _safe_call(
    memory: CumulativeTheoryMemory,
    errors: list[dict[str, str]],
    method: str,
    default: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    try:
        return getattr(memory, method)(*args, **kwargs)
    except Exception as error:  # pragma: no cover - exercised by defensive CLI use
        errors.append({'method': method, 'error': str(error)})
        return default


def _artifact_state(latest_artifact: dict[str, Any] | None) -> dict[str, Any]:
    if not latest_artifact:
        return {'provided': False, 'upload_status': 'not_provided'}
    upload = dict(latest_artifact.get('hf_upload') or {})
    upload_status = (
        upload.get('status')
        or latest_artifact.get('upload_status')
        or 'unknown'
    )
    return {
        'provided': True,
        'run_kind': latest_artifact.get('run_kind'),
        'runs_final': bool(latest_artifact.get('runs_final')),
        'artifact_path': latest_artifact.get('artifact_path'),
        'upload_status': upload_status,
        'repo_id': upload.get('repo_id') or latest_artifact.get('repo_id'),
        'path_in_repo': upload.get('path_in_repo') or latest_artifact.get('path_in_repo'),
        'reason': upload.get('reason') or latest_artifact.get('reason'),
        'error': upload.get('error') or latest_artifact.get('error'),
        'create_repo_error': (
            upload.get('create_repo_error')
            or latest_artifact.get('create_repo_error')
        ),
    }


def _connection_statuses(report: dict[str, Any]) -> list[dict[str, Any]]:
    subsystems = report['subsystems']
    readiness = report['readiness']
    checkpoint = subsystems['memory']['checkpoint']
    foundation = subsystems['foundation']
    equations = subsystems['equation_science']
    experiments = subsystems['experiment_design']
    domains = subsystems['domain_rediscovery']
    operators = subsystems['operator_system']
    compression = subsystems['memory_compression']
    artifacts = subsystems['artifact_persistence']

    def state(active: bool, waiting: bool = False) -> str:
        if active:
            return 'active'
        if waiting:
            return 'waiting'
        return 'attention'

    statuses = [
        {
            'connection': 'memory_to_readiness',
            'status': state(bool(readiness)),
            'evidence': {
                'records': checkpoint.get('record_count', 0),
                'families': checkpoint.get('family_count', 0),
                'status': readiness.get('status'),
            },
        },
        {
            'connection': 'foundation_to_equation_search',
            'status': state(
                bool(foundation['first_principles_basis'])
                and bool(foundation['algebraic_foundation'])
            ),
            'evidence': {
                'basis': len(foundation['first_principles_basis']),
                'expression_families': (
                    foundation['algebraic_foundation'].get(
                        'expression_family_count',
                        0,
                    )
                ),
            },
        },
        {
            'connection': 'equations_to_theorem_memory',
            'status': state(
                bool(equations['self_authored_equations'])
                or bool(subsystems['theorem_memory']['theorems'])
                or bool(equations['reusable_families']),
                waiting=bool(equations['reusable_families']),
            ),
            'evidence': {
                'families': len(equations['reusable_families']),
                'self_authored': len(equations['self_authored_equations']),
                'theorems': len(subsystems['theorem_memory']['theorems']),
            },
        },
        {
            'connection': 'theories_to_experiment_design',
            'status': state(
                bool(experiments['planned_experiments'])
                and bool(experiments['cockpit'])
            ),
            'evidence': {
                'next': len(experiments['next_experiments']),
                'planned': len(experiments['planned_experiments']),
                'cockpit': len(experiments['cockpit']),
            },
        },
        {
            'connection': 'domain_worlds_to_transfer',
            'status': state(
                bool(domains['domain_world_blueprints'])
                and (
                    bool(domains['domain_transfer_experiments'])
                    or bool(domains['domain_world_transfer_evidence'])
                )
            ),
            'evidence': {
                'blueprints': len(domains['domain_world_blueprints']),
                'transfer_experiments': len(domains['domain_transfer_experiments']),
                'transfer_evidence': len(domains['domain_world_transfer_evidence']),
            },
        },
        {
            'connection': 'operator_priors_to_repair',
            'status': state(
                bool(operators['generated_operator_priors'])
                and (
                    bool(operators['operator_prior_feedback'])
                    or bool(operators['operator_prior_repairs'])
                    or bool(operators['operator_prior_claim_experiments'])
                ),
                waiting=bool(operators['generated_operator_priors']),
            ),
            'evidence': {
                'priors': len(operators['generated_operator_priors']),
                'feedback': len(operators['operator_prior_feedback']),
                'repairs': len(operators['operator_prior_repairs']),
            },
        },
        {
            'connection': 'compression_to_long_runs',
            'status': state(
                bool(compression['resource_efficiency'].get('long_run_ready'))
                or bool(
                    compression['canonical_law_compression'].get(
                        'long_run_law_ready'
                    )
                )
            ),
            'evidence': {
                'resource_long_run_ready': compression['resource_efficiency'].get(
                    'long_run_ready'
                ),
                'canonical_long_run_ready': (
                    compression['canonical_law_compression'].get(
                        'long_run_law_ready'
                    )
                ),
            },
        },
        {
            'connection': 'artifacts_to_remote_runs',
            'status': state(
                artifacts['latest_artifact'].get('upload_status') in {
                    'uploaded',
                    'success',
                    'skipped',
                },
                waiting=(
                    not artifacts['latest_artifact'].get('provided')
                    or artifacts['latest_artifact'].get('upload_status')
                    == 'not_provided'
                ),
            ),
            'evidence': {
                'upload_status': artifacts['latest_artifact'].get('upload_status'),
                'path_in_repo': artifacts['latest_artifact'].get('path_in_repo'),
            },
        },
    ]
    return statuses


def _connection_gaps(report: dict[str, Any]) -> list[dict[str, Any]]:
    gaps = []
    readiness = report['readiness']
    for gate_key in list(readiness.get('missing_gates') or [])[:8]:
        gate = dict(readiness.get('gates', {}).get(gate_key) or {})
        gaps.append({
            'gap_kind': 'readiness_gate',
            'key': gate_key,
            'next_step': gate.get('next_step', 'collect more evidence'),
        })

    experiments = report['subsystems']['experiment_design']
    if not experiments['planned_experiments']:
        gaps.append({
            'gap_kind': 'no_executable_planned_experiments',
            'key': 'experiment_design',
            'next_step': 'record or replay enough theory evidence to create a falsification plan',
        })
    if not experiments['cockpit']:
        gaps.append({
            'gap_kind': 'no_experiment_cockpit_rows',
            'key': 'experiment_design_cockpit',
            'next_step': 'generate planned experiments and expose beliefs plus interventions',
        })

    artifacts = report['subsystems']['artifact_persistence']['latest_artifact']
    if artifacts.get('upload_status') == 'failed':
        gaps.append({
            'gap_kind': 'artifact_upload_failed',
            'key': 'hf_upload',
            'next_step': 'repair Hugging Face artifact persistence before the next long run',
            'error': artifacts.get('error') or artifacts.get('reason'),
        })

    resource = report['subsystems']['memory_compression']['resource_efficiency']
    if resource and not resource.get('long_run_ready'):
        gaps.append({
            'gap_kind': 'memory_not_long_run_ready',
            'key': 'memory_compression',
            'next_step': 'compact raw records and operator outcomes before very long runs',
        })
    return gaps


def _recommended_commands(report: dict[str, Any]) -> list[dict[str, Any]]:
    commands = [{
        'action_kind': 'super_system_snapshot',
        'reason': 'save one connected audit before changing or running the system again',
        'command': (
            'python3 first_principles_ai/main.py --super-system-audit '
            '--theory-memory-file tmp/theory-memory.json '
            '--super-system-output-file tmp/super-system-audit.json'
        ),
        'runs_final': False,
    }]

    gap_kinds = {
        gap['gap_kind']
        for gap in report.get('connection_gaps', [])
    }
    if 'memory_not_long_run_ready' in gap_kinds:
        commands.append({
            'action_kind': 'memory_efficiency_review',
            'reason': 'confirm raw evidence is bounded before remote long runs',
            'command': (
                'python3 first_principles_ai/main.py --memory-efficiency-review '
                '--theory-memory-file tmp/theory-memory.json'
            ),
            'runs_final': False,
        })
    if 'artifact_upload_failed' in gap_kinds:
        commands.append({
            'action_kind': 'artifact_persistence_recheck',
            'reason': 'make the next remote run durable instead of log-only',
            'command': (
                'python3 first_principles_ai/main.py --hf-non-final-campaign '
                '--hf-log-artifact-summary --theory-memory-file tmp/theory-memory.json '
                '--hf-output-file tmp/hf-non-final-campaign.json'
            ),
            'runs_final': False,
        })
    if report['readiness'].get('ready_for_watched_final'):
        commands.append({
            'action_kind': 'watched_final_ready_but_held',
            'reason': 'the full discovery run should wait for the user to watch it live',
            'command': (
                'python3 first_principles_ai/main.py --math-final-discovery '
                '--theory-memory-file tmp/theory-memory.json --parallel-cases 4'
            ),
            'runs_final': True,
            'held_for_user': True,
        })
    else:
        commands.append({
            'action_kind': 'rediscovery_progress_audit',
            'reason': 'track the stricter progress target without running final discovery',
            'command': (
                'python3 first_principles_ai/main.py --rediscovery-goal-progress '
                '--theory-memory-file tmp/theory-memory.json'
            ),
            'runs_final': False,
        })
    return commands


def build_super_system_report(
    theory_memory: CumulativeTheoryMemory | None = None,
    *,
    world_types: list[str] | None = None,
    object_counts: list[int] | None = None,
    steps: int = 240,
    limit: int = 5,
    latest_artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a connected, non-final report over every discovery subsystem."""
    memory = theory_memory or CumulativeTheoryMemory()
    worlds = list(world_types or DEFAULT_SUPER_SYSTEM_WORLDS)
    counts = list(object_counts or [5])
    errors: list[dict[str, str]] = []

    readiness = _safe_call(memory, errors, 'discovery_readiness_report', {})
    rediscovery = _safe_call(memory, errors, 'rediscovery_goal_progress_report', {})
    memory_checkpoint = _safe_call(memory, errors, 'memory_checkpoint_summary', {})
    resource_efficiency = _safe_call(memory, errors, 'resource_efficiency_report', {})
    canonical_law_compression = _safe_call(
        memory,
        errors,
        'canonical_law_compression_report',
        {},
    )
    planned = _safe_call(
        memory,
        errors,
        'planned_experiments',
        [],
        world_types=worlds,
        object_counts=counts,
        steps=steps,
        limit=limit,
    )
    try:
        cockpit = [experiment_design_from_plan(plan) for plan in planned]
    except Exception as error:  # pragma: no cover - keeps CLI audit defensive
        errors.append({'method': 'experiment_design_cockpit', 'error': str(error)})
        cockpit = []

    report = {
        'run_kind': 'super_system_audit',
        'runs_final': False,
        'world_types': worlds,
        'object_counts': counts,
        'steps': steps,
        'readiness': readiness,
        'rediscovery_goal_progress': rediscovery,
        'connection_graph': list(CONNECTION_GRAPH),
        'action_surface': list(SUPPORTED_INTERVENTIONS),
        'subsystems': {
            'memory': {
                'checkpoint': memory_checkpoint,
            },
            'foundation': {
                'first_principles_basis': _safe_call(
                    memory,
                    errors,
                    'first_principles_basis',
                    [],
                ),
                'baseline_experiment_templates': _safe_call(
                    memory,
                    errors,
                    'baseline_experiment_templates',
                    [],
                ),
                'adaptive_dimension_agenda': _safe_call(
                    memory,
                    errors,
                    'adaptive_dimension_agenda',
                    [],
                    limit=limit,
                ),
                'algebraic_foundation': _safe_call(
                    memory,
                    errors,
                    'algebraic_foundation_baseline',
                    {},
                ),
                'algebraic_expression_agenda': _safe_call(
                    memory,
                    errors,
                    'algebraic_expression_agenda',
                    [],
                    limit=limit,
                ),
            },
            'equation_science': {
                'reusable_families': _safe_call(
                    memory,
                    errors,
                    'reusable_families',
                    [],
                )[:limit],
                'proof_certificates': _safe_call(
                    memory,
                    errors,
                    'proof_certificates',
                    [],
                    limit=limit,
                ),
                'self_authored_equations': _safe_call(
                    memory,
                    errors,
                    'self_authored_equations',
                    [],
                    limit=limit,
                ),
                'equation_invariant_resolution_experiments': _safe_call(
                    memory,
                    errors,
                    'equation_invariant_resolution_experiments',
                    [],
                    limit=limit,
                ),
                'selected_law_replay_agenda': _safe_call(
                    memory,
                    errors,
                    'selected_law_replay_agenda',
                    [],
                    limit=limit,
                ),
                'selected_law_conflict_experiments': _safe_call(
                    memory,
                    errors,
                    'selected_law_conflict_experiments',
                    [],
                    limit=limit,
                ),
                'law_domain_split_hypotheses': _safe_call(
                    memory,
                    errors,
                    'law_domain_split_hypotheses',
                    [],
                    limit=limit,
                ),
            },
            'experiment_design': {
                'next_experiments': _safe_call(
                    memory,
                    errors,
                    'next_experiments',
                    [],
                    limit=limit,
                ),
                'planned_experiments': planned,
                'cockpit': cockpit,
                'disagreement_experiments': _safe_call(
                    memory,
                    errors,
                    'disagreement_experiments',
                    [],
                    limit=limit,
                ),
                'blind_holdout_validation_experiments': _safe_call(
                    memory,
                    errors,
                    'blind_holdout_validation_experiments',
                    [],
                    limit=limit,
                ),
                'post_run_replay_agenda': _safe_call(
                    memory,
                    errors,
                    'post_run_replay_agenda',
                    [],
                    limit=limit,
                ),
            },
            'domain_rediscovery': {
                'math_domain_curriculum': _safe_call(
                    memory,
                    errors,
                    'math_domain_curriculum',
                    {},
                ),
                'domain_curriculum_agenda': _safe_call(
                    memory,
                    errors,
                    'domain_curriculum_agenda',
                    [],
                    limit=limit,
                ),
                'domain_world_blueprints': _safe_call(
                    memory,
                    errors,
                    'domain_world_blueprints',
                    [],
                    limit=limit,
                ),
                'domain_world_discoveries': _safe_call(
                    memory,
                    errors,
                    'domain_world_discovery_reports',
                    [],
                    limit=limit,
                ),
                'domain_world_transfer_evidence': _safe_call(
                    memory,
                    errors,
                    'domain_world_transfer_evidence',
                    [],
                    limit=limit,
                ),
                'domain_transfer_experiments': _safe_call(
                    memory,
                    errors,
                    'domain_transfer_experiments',
                    [],
                    limit=limit,
                ),
                'domain_rediscovery_experiments': _safe_call(
                    memory,
                    errors,
                    'domain_rediscovery_experiments',
                    [],
                    limit=limit,
                ),
                'autonomous_experiment_design_agenda': _safe_call(
                    memory,
                    errors,
                    'autonomous_experiment_design_agenda',
                    [],
                    limit=limit,
                ),
                'autonomous_scientist_evidence': _safe_call(
                    memory,
                    errors,
                    'autonomous_scientist_evidence',
                    {},
                ),
                'arithmetic_rediscovery_evidence': _safe_call(
                    memory,
                    errors,
                    'arithmetic_rediscovery_evidence',
                    {},
                ),
            },
            'operator_system': {
                'representation_agenda': _safe_call(
                    memory,
                    errors,
                    'representation_agenda',
                    [],
                    limit=limit,
                ),
                'generated_operator_priors': _safe_call(
                    memory,
                    errors,
                    'generated_operator_priors',
                    [],
                    limit=limit,
                ),
                'operator_prior_feedback': _safe_call(
                    memory,
                    errors,
                    'operator_prior_feedback',
                    [],
                    limit=limit,
                ),
                'operator_prior_domains': _safe_call(
                    memory,
                    errors,
                    'operator_prior_domains',
                    [],
                    limit=limit,
                ),
                'operator_prior_anomalies': _safe_call(
                    memory,
                    errors,
                    'operator_prior_anomalies',
                    [],
                    limit=limit,
                ),
                'operator_prior_discovery_claims': _safe_call(
                    memory,
                    errors,
                    'operator_prior_discovery_claims',
                    [],
                    limit=limit,
                ),
                'operator_prior_discovery_chains': _safe_call(
                    memory,
                    errors,
                    'operator_prior_discovery_chains',
                    [],
                    limit=limit,
                ),
                'operator_prior_claim_experiments': _safe_call(
                    memory,
                    errors,
                    'operator_prior_claim_experiments',
                    [],
                    limit=limit,
                ),
                'operator_prior_repairs': _safe_call(
                    memory,
                    errors,
                    'operator_prior_repair_experiments',
                    [],
                    limit=limit,
                ),
                'operator_prior_validation_experiments': _safe_call(
                    memory,
                    errors,
                    'operator_prior_validation_experiments',
                    [],
                    limit=limit,
                ),
                'operator_prior_invariant_consolidations': _safe_call(
                    memory,
                    errors,
                    'operator_prior_invariant_consolidations',
                    [],
                    limit=limit,
                ),
            },
            'theorem_memory': {
                'theorems': _safe_call(
                    memory,
                    errors,
                    'theorem_memory',
                    [],
                    limit=limit,
                ),
                'theorem_consolidations': _safe_call(
                    memory,
                    errors,
                    'theorem_consolidations',
                    [],
                    limit=limit,
                ),
                'blind_holdout_benchmark': _safe_call(
                    memory,
                    errors,
                    'blind_holdout_benchmark_report',
                    {},
                    limit=limit,
                ),
            },
            'memory_compression': {
                'resource_efficiency': resource_efficiency,
                'canonical_law_compression': canonical_law_compression,
            },
            'artifact_persistence': {
                'latest_artifact': _artifact_state(latest_artifact),
            },
        },
        'integration_errors': errors,
    }
    report['connection_status'] = _connection_statuses(report)
    report['connection_gaps'] = _connection_gaps(report)
    report['recommended_commands'] = _recommended_commands(report)
    return report


def _short_evidence(evidence: dict[str, Any]) -> str:
    parts = []
    for key, value in evidence.items():
        if value is None:
            continue
        parts.append(f'{key}={value}')
    return ', '.join(parts[:3]) if parts else 'no evidence'


def super_system_summary_lines(
    report: dict[str, Any],
    *,
    limit: int = 4,
) -> list[str]:
    readiness = report.get('readiness') or {}
    checkpoint = (
        report.get('subsystems', {})
        .get('memory', {})
        .get('checkpoint', {})
    )
    lines = [
        '=' * 70,
        'SUPER SYSTEM AUDIT',
        '=' * 70,
        'Watched final run: not run by this audit',
        (
            f"runs_final={report.get('runs_final')} "
            f"score={float(readiness.get('readiness_score', 0.0) or 0.0):.0%} "
            f"status={readiness.get('status', 'unknown')} "
            f"gates={readiness.get('passed_gate_count', 0)}/"
            f"{readiness.get('gate_count', 0)}"
        ),
        (
            'Memory: '
            f"records={checkpoint.get('record_count', 0)} "
            f"families={checkpoint.get('family_count', 0)} "
            f"theorems={checkpoint.get('theorem_count', 0)} "
            f"canonical_shards={checkpoint.get('canonical_law_shard_count', 0)}"
        ),
        '',
        'Connected subsystems:',
    ]
    for status in list(report.get('connection_status') or [])[:limit + 4]:
        lines.append(
            f"  {status['connection']}: {status['status']} "
            f"({_short_evidence(status.get('evidence') or {})})"
        )
    action_types = ', '.join(
        item['type']
        for item in report.get('action_surface', [])
    )
    lines.extend(['', f'Intervention surface: {action_types}'])

    cockpit = (
        report.get('subsystems', {})
        .get('experiment_design', {})
        .get('cockpit', [])
    )
    lines.append('')
    lines.append('Experiment design cockpit:')
    if cockpit:
        for design in cockpit[:limit]:
            lines.append(
                f"  {design.get('experiment_kind')}: "
                f"{design.get('world_type')} seed={design.get('seed')} "
                f"beliefs={len(design.get('beliefs') or [])}"
            )
            lines.append(f"    ask: {design.get('question')}")
            lines.append(f"    do: {design.get('intervention_text')}")
            lines.append(f"    falsifier: {design.get('falsifies_if')}")
    else:
        lines.append('  no cockpit rows yet')

    gaps = list(report.get('connection_gaps') or [])
    lines.append('')
    lines.append('Connection gaps:')
    if gaps:
        for gap in gaps[:limit]:
            lines.append(
                f"  {gap.get('gap_kind')}: {gap.get('key')} - "
                f"{gap.get('next_step')}"
            )
    else:
        lines.append('  none')

    commands = list(report.get('recommended_commands') or [])
    lines.append('')
    lines.append('Recommended next commands:')
    for command in commands[:limit]:
        label = 'FINAL-HELD' if command.get('runs_final') else 'SAFE'
        lines.append(f"  [{label}] {command.get('action_kind')}: {command.get('reason')}")
        lines.append(f"    {command.get('command')}")
    if report.get('integration_errors'):
        lines.append('')
        lines.append('Integration errors:')
        for error in report['integration_errors'][:limit]:
            lines.append(f"  {error.get('method')}: {error.get('error')}")
    return lines
