from __future__ import annotations

"""
Autonomous scientist layer for domain-world discoveries.

The domain discovery layer writes local equations from public observations.
This layer works one level up: it compares repeated runs, consolidates stable
laws, turns residuals into next experiments, picks harder stress worlds, writes
richer equation forms, and emits a live event stream.
"""

from collections import Counter, defaultdict
from typing import Any


HARDER_STRESS_WORLD_TEMPLATES: tuple[dict[str, Any], ...] = (
    {
        'key': 'localized_off_center_gravity',
        'world_type': 'localized_gravity',
        'target_domains': ('dynamics_systems', 'geometry_space'),
        'pressure': 'move the inferred center away from the origin and sample near/mid/far shells',
        'why': 'separates global center assumptions from local inverse-distance structure',
        'falsifies_if': 'the same force equation only works when the center is at the default frame',
    },
    {
        'key': 'time_varying_force',
        'world_type': 'time_varying',
        'target_domains': ('dynamics_systems', 'calculus_change'),
        'pressure': 'hold position pattern similar while shifting time phase',
        'why': 'tests whether residuals are state-only or need an explicit time variable',
        'falsifies_if': 'the residual repeats by position but not by phase or step offset',
    },
    {
        'key': 'mixed_law_composition',
        'world_type': 'hidden_procedural',
        'target_domains': ('algebra_equations', 'dynamics_systems'),
        'pressure': 'compose uniform, radial, tangential, and distance-scaled components',
        'why': 'forces the loop to write equations as sums of simpler discovered operators',
        'falsifies_if': 'one headline law fits easy samples but leaves structured residual fields',
    },
    {
        'key': 'higher_dimensional_projection',
        'world_type': 'higher_dimensions',
        'target_domains': ('higher_dimensions', 'geometry_space', 'dynamics_systems'),
        'pressure': 'hide one or more axes and ask whether a latent coordinate removes residuals',
        'why': 'tests whether the system can invent useful dimensions instead of overfitting 2D',
        'falsifies_if': 'the lifted axis helps one projection but fails a matched projection holdout',
    },
    {
        'key': 'noisy_sparse_observations',
        'world_type': 'probabilistic_hidden',
        'target_domains': ('probability_uncertainty', 'information_computation'),
        'pressure': 'drop observations and add bounded noise after the deterministic rule is found',
        'why': 'separates deterministic structure from uncertainty and compression claims',
        'falsifies_if': 'a noise model explains less than a hidden deterministic state or code',
    },
)


RELATION_REWRITE_FORMS: dict[str, dict[str, Any]] = {
    'collection_extent_join': {
        'expression': 'N(A union B) = N(A) + N(B) - N(A intersect B)',
        'grammar': ('set_relation_cardinality', 'inclusion_exclusion', 'equality'),
        'proofs': ('partition_exhaustion', 'label_invariance'),
    },
    'collection_extent_step': {
        'expression': 'N(S after event) = N(S before event) + delta(event)',
        'grammar': ('successor', 'integer_delta', 'recurrence'),
        'proofs': ('closure', 'one_step_holdout'),
    },
    'reversible_machine_relation': {
        'expression': 'x = inverse(stretch)(y - shift); y = stretch*x + shift',
        'grammar': ('affine_linear', 'inverse_operation', 'symbolic_substitution'),
        'proofs': ('inverse_check', 'heldout_substitution'),
    },
    'rewrite_equivalence': {
        'expression': 'forall slot in D: left_form(slot) - right_form(slot) = 0',
        'grammar': ('equivalence_class', 'polynomial_or_affine_rewrite'),
        'proofs': ('identity_substitution', 'heldout_rewrite'),
    },
    'frame_stable_reading': {
        'expression': 'd(T(p), T(q)) = d(p, q) for allowed frame transform T',
        'grammar': ('metric_space', 'symmetry_invariant', 'coordinate_transform'),
        'proofs': ('basis_invariance', 'inverse_transform_check'),
    },
    'partition_boundary': {
        'expression': 'region(x) = low | boundary | high, with exactly one assignment',
        'grammar': ('piecewise_predicate', 'topological_boundary'),
        'proofs': ('partition_exhaustion', 'boundary_holdout'),
    },
    'local_change_accumulation': {
        'expression': 'sum_i delta_i = x_T - x_0',
        'grammar': ('finite_difference_calculus', 'accumulation_integral'),
        'proofs': ('telescoping_balance', 'step_size_stability'),
    },
    'local_prediction_rule': {
        'expression': 'x(t + h) ~= x(t) + h * local_rate(t), with residual checked by distance from t',
        'grammar': ('local_linearization', 'finite_difference', 'residual_model'),
        'proofs': ('near_far_holdout', 'locality_boundary'),
    },
    'repeated_frequency_split': {
        'expression': 'P(symbol) ~= count(symbol in window) / window_size, confidence grows with window',
        'grammar': ('probability_statistics', 'sample_window', 'calibration'),
        'proofs': ('heldout_calibration', 'window_growth_check'),
    },
    'evidence_condition_split': {
        'expression': 'P(next | evidence) narrows support relative to P(next)',
        'grammar': ('conditional_probability', 'information_gain'),
        'proofs': ('evidence_ablation', 'heldout_split'),
    },
    'claim_counterexample_search': {
        'expression': 'claim holds on D only if no minimal counterexample is found in D',
        'grammar': ('logic_proof', 'counterexample_search', 'domain_predicate'),
        'proofs': ('smallest_counterexample', 'domain_restriction'),
    },
    'predicate_partition': {
        'expression': 'forall x: exactly_one(predicate_i(x))',
        'grammar': ('boolean_algebra', 'partition', 'exhaustive_case_split'),
        'proofs': ('overlap_check', 'gap_check'),
    },
    'link_walk_composition': {
        'expression': 'path(a,c) exists when edge(a,b) and path(b,c) compose',
        'grammar': ('graph_relation_path', 'composition'),
        'proofs': ('path_composition', 'reverse_or_reorder_holdout'),
    },
    'state_step_composition': {
        'expression': 'state_after(composed_moves) = fold(step, moves, state_0)',
        'grammar': ('finite_transition_algebra', 'recurrence_iteration'),
        'proofs': ('associativity_or_order_sensitivity', 'longer_sequence_holdout'),
    },
    'transform_invariant_reading': {
        'expression': 'reading(T(x)) = reading(x) for allowed transform family',
        'grammar': ('symmetry_invariant', 'group_action'),
        'proofs': ('new_transform_holdout', 'inverse_transform_check'),
    },
    'transform_order_rule': {
        'expression': 'route_b(route_a(x)) may differ from route_a(route_b(x)); structure decides order',
        'grammar': ('non_commutative_composition', 'group_like_transform'),
        'proofs': ('swap_order_check', 'identity_route_check'),
    },
    'choice_extremum': {
        'expression': 'chosen = argmin_x cost(x) subject to observed constraints',
        'grammar': ('optimization_extremum', 'constraint_solving'),
        'proofs': ('objective_decreases', 'neighbor_holdout'),
    },
    'constraint_boundary_rule': {
        'expression': 'argmin_x cost(x) over allowed(x); boundary can replace unconstrained optimum',
        'grammar': ('constrained_optimization', 'piecewise_predicate'),
        'proofs': ('boundary_behavior', 'outside_rejection'),
    },
    'state_update_residual_rule': {
        'expression': 'state[t+1] = F(state[t]) + residual(theta, t, latent_z)',
        'grammar': ('recurrence_iteration', 'residual_corrector', 'latent_variable'),
        'proofs': ('long_trace_rollout', 'residual_independence_check'),
    },
    'cycle_recurrence_rule': {
        'expression': 'phase(t + T) = phase(t), so reading(t + T) ~= reading(t)',
        'grammar': ('sinusoidal_phase', 'periodic_recurrence'),
        'proofs': ('phase_holdout', 'period_shift_check'),
    },
    'compression_description_rule': {
        'expression': 'best_model = argmin(description_length + reconstruction_error)',
        'grammar': ('information_computation', 'compression', 'heldout_reconstruction'),
        'proofs': ('decode_holdout', 'short_model_penalty'),
    },
    'hidden_state_recurrence': {
        'expression': 'visible[t] = decode(hidden[t]); hidden[t+1] = G(hidden[t])',
        'grammar': ('hidden_state_machine', 'recurrence_iteration'),
        'proofs': ('long_cycle_holdout', 'minimal_state_check'),
    },
    'projection_lift_rule': {
        'expression': 'visible = P(latent); invent z when residual(visible) is structured',
        'grammar': ('latent_axis', 'projection', 'dimension_lift'),
        'proofs': ('projection_holdout', 'latent_axis_transfer'),
    },
    'dimension_generalization_rule': {
        'expression': 'law_d(x_1...x_d) keeps form as d changes, with dimension-aware operators',
        'grammar': ('dimension_independent_law', 'vector_space', 'basis_change'),
        'proofs': ('heldout_axis_count', 'basis_invariance'),
    },
}


def run_domain_scientist_cycle(
    seed_start: int = 0,
    seed_count: int = 3,
    variants: tuple[int, ...] | list[int] = (0,),
    event_limit: int = 80,
) -> dict[str, Any]:
    """Run the non-final scientist loop across generated domain worlds."""
    try:
        from agent.domain_world_discovery import (
            build_domain_transfer_evidence,
            discover_all_domain_worlds,
        )
        from agent.discovery_loop import MATH_DOMAIN_TRANSFER_BRIDGES
    except ImportError:  # pragma: no cover - package import fallback
        from first_principles_ai.agent.domain_world_discovery import (
            build_domain_transfer_evidence,
            discover_all_domain_worlds,
        )
        from first_principles_ai.agent.discovery_loop import MATH_DOMAIN_TRANSFER_BRIDGES

    safe_seed_count = max(1, int(seed_count or 1))
    safe_variants = tuple(int(variant) for variant in variants) or (0,)
    all_reports = []
    run_summaries = []
    latest_by_domain = {}
    for variant in safe_variants:
        for seed in range(int(seed_start), int(seed_start) + safe_seed_count):
            reports = discover_all_domain_worlds(seed=seed, variant=variant)
            all_reports.extend(reports)
            for report in reports:
                latest_by_domain[report.domain_key] = report
            run_summaries.append({
                'seed': seed,
                'variant': variant,
                'domain_count': len(reports),
                'candidate_count': sum(len(report.candidates) for report in reports),
                'mean_coverage': round(
                    sum(report.benchmark_coverage for report in reports) / len(reports),
                    3,
                ) if reports else 0.0,
            })
    transfer_evidence = build_domain_transfer_evidence(
        list(latest_by_domain.values()),
        MATH_DOMAIN_TRANSFER_BRIDGES,
    )
    return build_autonomous_scientist_report(
        reports=all_reports,
        run_summaries=run_summaries,
        transfer_evidence=transfer_evidence,
        event_limit=event_limit,
    )


def build_autonomous_scientist_report(
    reports: list[Any],
    run_summaries: list[dict[str, Any]] | None = None,
    transfer_evidence: list[dict[str, Any]] | None = None,
    event_limit: int = 80,
) -> dict[str, Any]:
    """Build a meta-scientist report from domain discovery reports."""
    records = _candidate_records(reports)
    consolidations = _consolidate_invariants(records)
    residual_experiments = _residual_experiment_loop(records, consolidations)
    stress_worlds = _harder_stress_worlds(residual_experiments, consolidations)
    authored_equations = _richer_equation_extensions(
        consolidations,
        transfer_evidence or [],
    )
    live_events = _live_events(
        records,
        consolidations,
        residual_experiments,
        stress_worlds,
        authored_equations,
        event_limit=event_limit,
    )
    robust_count = sum(
        1 for item in consolidations
        if item.get('status') == 'robust_law'
    )
    coverage = {
        'input_report_count': len(reports),
        'input_candidate_count': len(records),
        'run_count': len(run_summaries or []),
        'invariant_count': len(consolidations),
        'robust_invariant_count': robust_count,
        'residual_experiment_count': len(residual_experiments),
        'stress_world_count': len(stress_worlds),
        'authored_equation_extension_count': len(authored_equations),
        'live_event_count': len(live_events),
    }
    return {
        'run_kind': 'autonomous_scientist_loop',
        'runs_final': False,
        'status': 'scientist_loop_ready' if robust_count else 'collect_more_runs',
        'run_summaries': list(run_summaries or []),
        'coverage': coverage,
        'invariant_consolidations': consolidations,
        'residual_experiments': residual_experiments,
        'harder_stress_worlds': stress_worlds,
        'authored_equation_extensions': authored_equations,
        'domain_transfer_evidence': list(transfer_evidence or []),
        'live_events': live_events,
        'next_actions': _scientist_next_actions(
            residual_experiments,
            stress_worlds,
            authored_equations,
        ),
    }


def _candidate_records(reports: list[Any]) -> list[dict[str, Any]]:
    records = []
    for report in reports:
        for candidate in report.candidates:
            records.append({
                'domain_key': report.domain_key,
                'seed': report.seed,
                'variant': report.variant,
                'candidate_key': candidate.key,
                'observation_kind': candidate.observation_kind,
                'relation_kind': candidate.relation_kind,
                'expression': candidate.expression,
                'confidence': candidate.confidence,
                'evidence': _json_copy(candidate.evidence),
                'falsification_tests': list(candidate.falsification_tests),
                'transfer_basis': list(candidate.transfer_basis),
                'comparison_tags': list(candidate.comparison_tags),
                'support_sample_ids': list(candidate.support_sample_ids),
                'benchmark_coverage': report.benchmark_coverage,
                'missing_comparison_tags': list(report.missing_comparison_tags),
            })
    return records


def _consolidate_invariants(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get('relation_kind', 'unknown'))].append(record)

    consolidations = []
    for relation_kind, items in grouped.items():
        expressions = Counter(str(item.get('expression', '')) for item in items)
        dominant_expression, dominant_count = expressions.most_common(1)[0]
        seeds = sorted({int(item.get('seed', 0) or 0) for item in items})
        variants = sorted({int(item.get('variant', 0) or 0) for item in items})
        domains = sorted({str(item.get('domain_key', 'unknown')) for item in items})
        confidences = [float(item.get('confidence', 0.0) or 0.0) for item in items]
        mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        transfer_basis = sorted({
            str(basis)
            for item in items
            for basis in item.get('transfer_basis', [])
        })
        status = (
            'robust_law'
            if len(seeds) >= 2 and mean_confidence >= 0.75
            else 'variant_stable_law'
            if len(variants) >= 2 and mean_confidence >= 0.75
            else 'local_candidate'
        )
        approximation_notes = []
        if len(expressions) > 1:
            approximation_notes.append(
                'multiple expression variants survived; keep the dominant form and test rewrites'
            )
        if min(confidences or [1.0]) < 0.9:
            approximation_notes.append(
                'some supporting candidates are lower-confidence and need residual probes'
            )
        if not approximation_notes:
            approximation_notes.append(
                'no public residual failure yet; next step is a harder held-out falsifier'
            )
        falsification_tests = _unique(
            test
            for item in items
            for test in item.get('falsification_tests', [])
        )[:5]
        consolidations.append({
            'key': f'invariant:{relation_kind}',
            'relation_kind': relation_kind,
            'status': status,
            'law_expression': dominant_expression,
            'support_count': len(items),
            'dominant_expression_support': dominant_count,
            'support_domains': domains,
            'support_seeds': seeds,
            'support_variants': variants,
            'mean_confidence': round(mean_confidence, 3),
            'transfer_basis': transfer_basis,
            'comparison_tags': _unique(
                tag
                for item in items
                for tag in item.get('comparison_tags', [])
            ),
            'variant_expressions': [
                expression for expression, _count in expressions.most_common(5)
            ],
            'approximation_notes': approximation_notes,
            'falsification_tests': falsification_tests,
            'next_test': falsification_tests[0] if falsification_tests else (
                'repeat on a new seed and a harder hidden world'
            ),
        })

    status_rank = {'robust_law': 2, 'variant_stable_law': 1, 'local_candidate': 0}
    consolidations.sort(
        key=lambda item: (
            status_rank.get(str(item.get('status')), 0),
            int(item.get('support_count', 0) or 0),
            float(item.get('mean_confidence', 0.0) or 0.0),
            str(item.get('relation_kind')),
        ),
        reverse=True,
    )
    return consolidations


def _residual_experiment_loop(
    records: list[dict[str, Any]],
    consolidations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    consolidation_by_relation = {
        str(item.get('relation_kind')): item
        for item in consolidations
    }
    experiments = []
    seen = set()
    for record in records:
        relation_kind = str(record.get('relation_kind', 'unknown'))
        key = (str(record.get('domain_key')), relation_kind)
        if key in seen:
            continue
        seen.add(key)
        residual_notes = _residual_notes(record)
        confidence = float(record.get('confidence', 0.0) or 0.0)
        priority = min(1.0, 0.42 + (1.0 - confidence) + 0.08 * len(residual_notes))
        consolidation = consolidation_by_relation.get(relation_kind, {})
        falsification_tests = list(record.get('falsification_tests') or [])
        next_experiment = (
            falsification_tests[0]
            if falsification_tests
            else str(consolidation.get('next_test') or 'repeat with a held-out contrast')
        )
        experiments.append({
            'key': f"residual_probe:{record.get('domain_key')}:{relation_kind}",
            'domain_key': record.get('domain_key'),
            'relation_kind': relation_kind,
            'status': (
                'residual_pressure'
                if any(note.startswith('failure') for note in residual_notes)
                else 'falsification_pressure'
            ),
            'priority': round(priority, 3),
            'current_expression': record.get('expression'),
            'where_failed': residual_notes,
            'designed_next_experiment': next_experiment,
            'refinement_rule': _refinement_rule(record, residual_notes),
            'expected_signal': _expected_signal(record),
            'falsifies_if': (
                'the residual keeps structure after the proposed refinement, '
                'or a rival equation predicts the held-out case better'
            ),
            'source_candidate': record.get('candidate_key'),
        })
    experiments.sort(
        key=lambda item: (
            float(item.get('priority', 0.0) or 0.0),
            str(item.get('domain_key')),
            str(item.get('relation_kind')),
        ),
        reverse=True,
    )
    return experiments[:24]


def _harder_stress_worlds(
    residual_experiments: list[dict[str, Any]],
    consolidations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    basis = {
        str(item)
        for consolidation in consolidations
        for item in consolidation.get('transfer_basis', [])
    }
    residual_by_domain = defaultdict(list)
    for experiment in residual_experiments:
        residual_by_domain[str(experiment.get('domain_key'))].append(experiment['key'])

    worlds = []
    for template in HARDER_STRESS_WORLD_TEMPLATES:
        target_domains = tuple(template['target_domains'])
        connected_residuals = _unique(
            residual
            for domain in target_domains
            for residual in residual_by_domain.get(domain, [])
        )[:5]
        priority = 0.54 + 0.06 * len(connected_residuals)
        if 'dimension_lift' in basis and template['key'] == 'higher_dimensional_projection':
            priority += 0.18
        if 'residual' in basis and template['key'] in {
            'localized_off_center_gravity',
            'mixed_law_composition',
        }:
            priority += 0.14
        if 'uncertainty' in basis and template['key'] == 'noisy_sparse_observations':
            priority += 0.12
        worlds.append({
            **template,
            'priority': round(min(1.0, priority), 3),
            'status': 'ready_for_non_final_campaign',
            'connected_residuals': connected_residuals,
            'runs_final': False,
        })
    worlds.sort(key=lambda item: (item['priority'], item['key']), reverse=True)
    return worlds


def _richer_equation_extensions(
    consolidations: list[dict[str, Any]],
    transfer_evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    authored = []
    for consolidation in consolidations:
        relation_kind = str(consolidation.get('relation_kind', 'unknown'))
        rewrite = RELATION_REWRITE_FORMS.get(relation_kind)
        if rewrite is None:
            rewrite = {
                'expression': consolidation.get('law_expression', ''),
                'grammar': tuple(consolidation.get('transfer_basis', [])),
                'proofs': ('heldout_counterexample',),
            }
        authored.append({
            'key': f"authored_extension:{relation_kind}",
            'source': 'invariant_consolidation',
            'relation_kind': relation_kind,
            'status': (
                'candidate_law'
                if consolidation.get('status') == 'robust_law'
                else 'working_equation'
            ),
            'expression': rewrite['expression'],
            'base_expression': consolidation.get('law_expression'),
            'grammar_terms': list(rewrite['grammar']),
            'support_count': consolidation.get('support_count', 0),
            'support_domains': list(consolidation.get('support_domains') or []),
            'proof_obligations': list(rewrite['proofs']),
            'falsification_tests': list(consolidation.get('falsification_tests') or []),
            'why_it_goes_further': (
                'turns a local observed relation into a reusable equation grammar '
                'with explicit proof and falsification pressure'
            ),
        })

    for item in transfer_evidence[:8]:
        if item.get('status') != 'transfer_link_ready':
            continue
        bridge_key = str(item.get('bridge_key'))
        authored.append({
            'key': f"authored_transfer_equation:{bridge_key}",
            'source': 'domain_transfer_bridge',
            'relation_kind': bridge_key,
            'status': 'transfer_equation',
            'expression': (
                f"target_rule({item.get('target_domain')}) := "
                f"translate(source_rule({item.get('source_domain')}), bridge={bridge_key})"
            ),
            'base_expression': item.get('transfer_question'),
            'grammar_terms': _unique(
                list(item.get('source_matches') or [])
                + list(item.get('target_matches') or [])
                + list(item.get('shared_basis') or [])
            ),
            'support_count': (
                int(item.get('source_candidate_count', 0) or 0)
                + int(item.get('target_candidate_count', 0) or 0)
            ),
            'support_domains': [
                str(item.get('source_domain')),
                str(item.get('target_domain')),
            ],
            'proof_obligations': ['bridge_holdout', 'domain_translation_check'],
            'falsification_tests': [str(item.get('falsifies_if'))],
            'why_it_goes_further': (
                'lets an equation written in one domain become a testable '
                'candidate in another domain'
            ),
        })

    authored.sort(
        key=lambda item: (
            item['status'] == 'candidate_law',
            int(item.get('support_count', 0) or 0),
            item['key'],
        ),
        reverse=True,
    )
    return authored[:36]


def _live_events(
    records: list[dict[str, Any]],
    consolidations: list[dict[str, Any]],
    residual_experiments: list[dict[str, Any]],
    stress_worlds: list[dict[str, Any]],
    authored_equations: list[dict[str, Any]],
    event_limit: int,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = [{
        'event': 'scientist_loop_start',
        'candidate_count': len(records),
        'domain_count': len({record.get('domain_key') for record in records}),
    }]
    for record in records[:12]:
        events.append({
            'event': 'candidate_seen',
            'domain_key': record.get('domain_key'),
            'relation_kind': record.get('relation_kind'),
            'confidence': record.get('confidence'),
            'expression': record.get('expression'),
        })
    for invariant in consolidations[:12]:
        events.append({
            'event': 'invariant_consolidated',
            'relation_kind': invariant.get('relation_kind'),
            'status': invariant.get('status'),
            'support_count': invariant.get('support_count'),
            'law_expression': invariant.get('law_expression'),
        })
    for experiment in residual_experiments[:12]:
        events.append({
            'event': 'residual_probe_designed',
            'domain_key': experiment.get('domain_key'),
            'relation_kind': experiment.get('relation_kind'),
            'priority': experiment.get('priority'),
            'next_experiment': experiment.get('designed_next_experiment'),
        })
    for world in stress_worlds[:8]:
        events.append({
            'event': 'stress_world_selected',
            'key': world.get('key'),
            'world_type': world.get('world_type'),
            'priority': world.get('priority'),
            'pressure': world.get('pressure'),
        })
    for equation in authored_equations[:12]:
        events.append({
            'event': 'equation_written',
            'key': equation.get('key'),
            'status': equation.get('status'),
            'expression': equation.get('expression'),
        })
    events.append({
        'event': 'scientist_loop_finish',
        'invariant_count': len(consolidations),
        'residual_experiment_count': len(residual_experiments),
        'stress_world_count': len(stress_worlds),
        'authored_equation_count': len(authored_equations),
    })
    return events[:max(1, int(event_limit or 1))]


def _scientist_next_actions(
    residual_experiments: list[dict[str, Any]],
    stress_worlds: list[dict[str, Any]],
    authored_equations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    actions = []
    if residual_experiments:
        first = residual_experiments[0]
        actions.append({
            'action_kind': 'run_residual_probe',
            'runs_final': False,
            'reason': first['where_failed'][0],
            'target': first['key'],
            'suggested_experiment': first['designed_next_experiment'],
        })
    if stress_worlds:
        world = stress_worlds[0]
        actions.append({
            'action_kind': 'run_harder_hidden_world',
            'runs_final': False,
            'reason': world['why'],
            'target': world['key'],
            'suggested_world_type': world['world_type'],
        })
    if authored_equations:
        equation = authored_equations[0]
        actions.append({
            'action_kind': 'falsify_authored_equation',
            'runs_final': False,
            'reason': equation['why_it_goes_further'],
            'target': equation['key'],
            'suggested_test': (
                equation.get('falsification_tests') or ['new held-out seed']
            )[0],
        })
    return actions


def _residual_notes(record: dict[str, Any]) -> list[str]:
    evidence = dict(record.get('evidence') or {})
    notes = []
    confidence = float(record.get('confidence', 0.0) or 0.0)
    if confidence < 0.9:
        notes.append(f'failure: candidate confidence is only {confidence:.2f}')
    if int(evidence.get('mismatch_count', 0) or 0) > 0:
        notes.append('failure: observed checks contain mismatches')
    if evidence.get('stable') is False or evidence.get('stable_residual') is False:
        notes.append('failure: public readings are not stable under the claimed transformation')
    if (
        'neighbor_sum' in evidence
        and 'endpoint_change' in evidence
        and abs(float(evidence['neighbor_sum']) - float(evidence['endpoint_change'])) > 0.002
    ):
        notes.append('failure: local accumulated changes do not match the endpoint change')
    if (
        'left_extent' in evidence
        and 'right_extent' in evidence
        and evidence['left_extent'] != evidence['right_extent']
    ):
        notes.append('failure: extent expression does not balance')
    if not notes:
        notes.append('no public residual failure yet; run the named falsifier on a harder holdout')
    return notes


def _refinement_rule(record: dict[str, Any], residual_notes: list[str]) -> str:
    relation_kind = str(record.get('relation_kind', 'unknown'))
    if any('not stable' in note for note in residual_notes):
        return 'add a domain predicate or latent coordinate before promoting the law'
    if relation_kind in {'state_update_residual_rule', 'cycle_recurrence_rule'}:
        return 'split residual into state, phase, and hidden-variable components'
    if relation_kind in {'frame_stable_reading', 'transform_invariant_reading'}:
        return 'test a new transform family and keep only transform-invariant terms'
    if relation_kind in {'choice_extremum', 'constraint_boundary_rule'}:
        return 'compare local and constrained optima on a boundary holdout'
    if relation_kind in {'projection_lift_rule', 'dimension_generalization_rule'}:
        return 'invent a latent axis only if it transfers to another projection'
    return 'repeat the equation on a new seed and revise only where residuals persist'


def _expected_signal(record: dict[str, Any]) -> str:
    basis = set(record.get('transfer_basis') or [])
    if 'residual' in basis:
        return 'residuals become less structured after the refined equation is installed'
    if 'invariance' in basis:
        return 'the same reading survives the held-out transform'
    if 'uncertainty' in basis:
        return 'held-out uncertainty or compression improves against the easy baseline'
    if 'dimension_lift' in basis:
        return 'a lifted coordinate explains a projection failure without overfitting'
    return 'the held-out contrast preserves the proposed relation'


def _unique(values) -> list[str]:
    seen = set()
    result = []
    for value in values:
        text = str(value)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _json_copy(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_copy(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_copy(item) for item in value]
    if isinstance(value, list):
        return [_json_copy(item) for item in value]
    return value
