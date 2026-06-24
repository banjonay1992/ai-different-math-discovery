from __future__ import annotations

"""
Discovery over generated math-domain worlds.

This layer reads only learner-facing observations. Benchmark labels from the
manifest are used after candidate generation to score coverage, mirroring the
project's "discover first, compare later" rule.
"""

from dataclasses import dataclass, field
from typing import Any


RELATION_TO_COMPARISON_TAGS: dict[str, tuple[str, ...]] = {
    'collection_extent_join': (
        'count invariance',
        'conservation of total under regrouping',
    ),
    'collection_extent_step': ('successor arithmetic',),
    'reversible_machine_relation': ('symbolic substitution', 'equation balance'),
    'rewrite_equivalence': ('factor/rewrite equivalence', 'symbolic substitution'),
    'frame_stable_reading': ('metric distance', 'coordinate transform'),
    'partition_boundary': ('local/global shape distinction', 'coordinate transform'),
    'local_change_accumulation': (
        'derivative-like rate',
        'integral-like accumulation',
    ),
    'local_prediction_rule': ('local linear approximation', 'derivative-like rate'),
    'repeated_frequency_split': ('frequency convergence', 'conditional split'),
    'evidence_condition_split': ('conditional split', 'expected error minimization'),
    'claim_counterexample_search': ('falsification', 'domain restriction'),
    'predicate_partition': (
        'domain restriction',
        'proof by repeated invariant check',
    ),
    'link_walk_composition': ('path composition', 'connectivity'),
    'state_step_composition': ('finite transition algebra', 'path composition'),
    'transform_invariant_reading': ('invariant quantity', 'coordinate-free law'),
    'transform_order_rule': ('group-like composition', 'coordinate-free law'),
    'choice_extremum': ('least-error fit', 'tradeoff curve'),
    'constraint_boundary_rule': ('constraint boundary', 'least-error fit'),
    'state_update_residual_rule': ('state update law', 'residual field'),
    'cycle_recurrence_rule': ('phase or conservation law', 'state update law'),
    'compression_description_rule': (
        'compression preference',
        'algorithmic recurrence',
    ),
    'hidden_state_recurrence': (
        'hidden-state inference',
        'algorithmic recurrence',
    ),
    'projection_lift_rule': ('latent axis', 'projection invariance'),
    'dimension_generalization_rule': (
        'dimension-independent law',
        'projection invariance',
    ),
    'primitive_cardinality_balance': (
        'count invariance',
        'conservation of total under regrouping',
    ),
    'primitive_reversible_substitution': (
        'symbolic substitution',
        'equation balance',
    ),
    'primitive_coordinate_measurement': ('metric distance', 'coordinate transform'),
    'primitive_finite_difference_accumulation': (
        'derivative-like rate',
        'integral-like accumulation',
    ),
    'primitive_sample_noise_split': (
        'frequency convergence',
        'conditional split',
    ),
    'primitive_counterexample_search': ('falsification', 'domain restriction'),
    'primitive_path_composition': ('path composition', 'connectivity'),
    'primitive_transform_invariance': (
        'invariant quantity',
        'coordinate-free law',
    ),
    'primitive_objective_comparison': ('least-error fit', 'tradeoff curve'),
    'primitive_transition_rollout': ('state update law', 'residual field'),
    'primitive_compression_holdout': (
        'compression preference',
        'algorithmic recurrence',
    ),
    'primitive_projection_residual': ('latent axis', 'projection invariance'),
}


BRIDGE_BASIS_REQUIREMENTS: dict[str, dict[str, tuple[str, ...]]] = {
    'quantity_to_algebra': {
        'source_any': ('extent', 'composition'),
        'target_any': ('composition', 'relation', 'inverse'),
    },
    'algebra_to_geometry': {
        'source_any': ('equivalence', 'invariance'),
        'target_any': ('invariance', 'metric'),
    },
    'geometry_to_calculus': {
        'source_any': ('locality', 'partition'),
        'target_any': ('local_change', 'accumulation'),
    },
    'calculus_to_dynamics': {
        'source_any': ('state_update', 'local_change'),
        'target_any': ('state_update', 'recurrence'),
    },
    'probability_to_information': {
        'source_any': ('uncertainty', 'conditional'),
        'target_any': ('uncertainty', 'compression'),
    },
    'logic_to_all_domains': {
        'source_any': ('falsifier', 'partition'),
        'target_any': ('falsifier', 'relation'),
    },
    'discrete_to_algebra': {
        'source_any': ('composition', 'relation'),
        'target_any': ('composition', 'relation'),
    },
    'symmetry_to_geometry': {
        'source_any': ('invariance', 'transform'),
        'target_any': ('invariance', 'metric'),
    },
    'optimization_to_calculus': {
        'source_any': ('local_change', 'constraint'),
        'target_any': ('local_change', 'accumulation'),
    },
    'dynamics_to_probability': {
        'source_any': ('residual', 'state_update'),
        'target_any': ('uncertainty', 'residual'),
    },
    'information_to_logic': {
        'source_any': ('recurrence', 'falsifier'),
        'target_any': ('falsifier', 'partition'),
    },
    'higher_dimensions_to_all_domains': {
        'source_any': ('dimension_lift', 'projection'),
        'target_any': ('residual', 'state_update'),
    },
}


PRIMITIVE_CONTRAST_RELATIONS: dict[str, dict[str, Any]] = {
    'cardinality_balance': {
        'relation_kind': 'primitive_cardinality_balance',
        'expression': 'extent is preserved by regrouping, relabeling, and order changes',
        'basis': ('extent', 'composition', 'invariance', 'falsifier'),
    },
    'reversible_substitution': {
        'relation_kind': 'primitive_reversible_substitution',
        'expression': 'hidden slot can be recovered by reversing the observed transformation chain',
        'basis': ('composition', 'inverse', 'relation', 'falsifier'),
    },
    'coordinate_measurement': {
        'relation_kind': 'primitive_coordinate_measurement',
        'expression': 'metric readings should remain stable under equivalent coordinate views',
        'basis': ('invariance', 'metric', 'projection', 'falsifier'),
    },
    'finite_difference_accumulation': {
        'relation_kind': 'primitive_finite_difference_accumulation',
        'expression': 'local differences should sum to the observed endpoint change',
        'basis': ('accumulation', 'local_change', 'state_update', 'falsifier'),
    },
    'sample_noise_split': {
        'relation_kind': 'primitive_sample_noise_split',
        'expression': 'repeated samples and conditions separate stable signal from noise',
        'basis': ('frequency', 'uncertainty', 'conditional', 'falsifier'),
    },
    'counterexample_search': {
        'relation_kind': 'primitive_counterexample_search',
        'expression': 'candidate rule must name a domain and a smallest breaking observation',
        'basis': ('implication', 'falsifier', 'partition'),
    },
    'path_composition': {
        'relation_kind': 'primitive_path_composition',
        'expression': 'local relations can compose into longer paths only when endpoints match',
        'basis': ('relation', 'composition', 'path', 'falsifier'),
    },
    'transform_invariance': {
        'relation_kind': 'primitive_transform_invariance',
        'expression': 'allowed transforms preserve invariant readings while changing presentation',
        'basis': ('invariance', 'transform', 'equivalence', 'falsifier'),
    },
    'objective_comparison': {
        'relation_kind': 'primitive_objective_comparison',
        'expression': 'local objective comparisons identify better choices under constraints',
        'basis': ('local_change', 'extremum', 'optimization', 'falsifier'),
    },
    'transition_rollout': {
        'relation_kind': 'primitive_transition_rollout',
        'expression': 'state residuals become reusable transition rules only if rollout survives',
        'basis': ('state_update', 'residual', 'recurrence', 'falsifier'),
    },
    'compression_holdout': {
        'relation_kind': 'primitive_compression_holdout',
        'expression': 'shorter descriptions are promoted only when they replay held-out data',
        'basis': ('compression', 'recurrence', 'uncertainty', 'falsifier'),
    },
    'projection_residual': {
        'relation_kind': 'primitive_projection_residual',
        'expression': 'latent coordinates are useful only when they explain projection residuals',
        'basis': ('projection', 'dimension_lift', 'invariance', 'falsifier'),
    },
}


@dataclass(frozen=True)
class DomainDiscoveryCandidate:
    """A rule candidate inferred from public observations."""

    key: str
    observation_kind: str
    relation_kind: str
    expression: str
    confidence: float
    support_sample_ids: tuple[str, ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)
    falsification_tests: tuple[str, ...] = ()
    transfer_basis: tuple[str, ...] = ()

    @property
    def comparison_tags(self) -> tuple[str, ...]:
        return RELATION_TO_COMPARISON_TAGS.get(self.relation_kind, ())

    def to_dict(self) -> dict[str, Any]:
        return {
            'key': self.key,
            'observation_kind': self.observation_kind,
            'relation_kind': self.relation_kind,
            'expression': self.expression,
            'confidence': self.confidence,
            'support_sample_ids': list(self.support_sample_ids),
            'evidence': _json_copy(self.evidence),
            'falsification_tests': list(self.falsification_tests),
            'transfer_basis': list(self.transfer_basis),
            'comparison_tags': list(self.comparison_tags),
        }


@dataclass(frozen=True)
class DomainWorldDiscoveryReport:
    """Benchmark-side report for one generated domain world."""

    domain_key: str
    seed: int
    variant: int
    candidates: tuple[DomainDiscoveryCandidate, ...]
    expected_discoveries: tuple[str, ...]
    observed_sample_count: int
    leaked_manifest: bool = False

    @property
    def comparison_hits(self) -> tuple[str, ...]:
        candidate_tags = {
            tag
            for candidate in self.candidates
            for tag in candidate.comparison_tags
        }
        return tuple(sorted(set(self.expected_discoveries) & candidate_tags))

    @property
    def missing_comparison_tags(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.expected_discoveries) - set(self.comparison_hits)))

    @property
    def benchmark_coverage(self) -> float:
        if not self.expected_discoveries:
            return 0.0
        return round(len(self.comparison_hits) / len(self.expected_discoveries), 3)

    @property
    def transfer_basis(self) -> tuple[str, ...]:
        return tuple(sorted({
            basis
            for candidate in self.candidates
            for basis in candidate.transfer_basis
        }))

    @property
    def falsification_tests(self) -> tuple[str, ...]:
        return tuple(
            test
            for candidate in self.candidates
            for test in candidate.falsification_tests
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            'domain_key': self.domain_key,
            'seed': self.seed,
            'variant': self.variant,
            'observed_sample_count': self.observed_sample_count,
            'candidate_count': len(self.candidates),
            'candidates': [candidate.to_dict() for candidate in self.candidates],
            'self_authored_equations': [
                {
                    'candidate_key': candidate.key,
                    'expression': candidate.expression,
                    'falsification_tests': list(candidate.falsification_tests),
                    'confidence': candidate.confidence,
                }
                for candidate in self.candidates
            ],
            'expected_discoveries': list(self.expected_discoveries),
            'comparison_hits': list(self.comparison_hits),
            'missing_comparison_tags': list(self.missing_comparison_tags),
            'benchmark_coverage': self.benchmark_coverage,
            'transfer_basis': list(self.transfer_basis),
            'falsification_test_count': len(self.falsification_tests),
            'leaked_manifest': self.leaked_manifest,
        }


def discover_domain_world_manifest(manifest) -> DomainWorldDiscoveryReport:
    """Infer candidate rules from a manifest's public observation stream."""
    try:
        from world.math_domain_worlds import math_domain_manifest_from_observation
    except ImportError:  # pragma: no cover - package import fallback
        from first_principles_ai.world.math_domain_worlds import (
            math_domain_manifest_from_observation,
        )

    observations = manifest.observations()
    candidates = []
    leaked_manifest = False
    for observation in observations:
        leaked_manifest = leaked_manifest or math_domain_manifest_from_observation(observation)
        candidates.extend(_infer_candidates(observation))
    return DomainWorldDiscoveryReport(
        domain_key=manifest.domain_key,
        seed=manifest.seed,
        variant=manifest.variant,
        candidates=tuple(candidates),
        expected_discoveries=tuple(manifest.expected_discoveries),
        observed_sample_count=len(observations),
        leaked_manifest=leaked_manifest,
    )


def discover_all_domain_worlds(
    seed: int = 0,
    variant: int = 0,
) -> list[DomainWorldDiscoveryReport]:
    """Run the lightweight discovery evaluator over every generated domain."""
    try:
        from world.math_domain_worlds import generate_all_math_domain_world_manifests
    except ImportError:  # pragma: no cover - package import fallback
        from first_principles_ai.world.math_domain_worlds import (
            generate_all_math_domain_world_manifests,
        )

    return [
        discover_domain_world_manifest(manifest)
        for manifest in generate_all_math_domain_world_manifests(
            seed=seed,
            variant=variant,
        )
    ]


def build_domain_transfer_evidence(
    reports: list[DomainWorldDiscoveryReport],
    bridges: list[dict[str, Any]],
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Score whether discovered relation bases connect source and target domains."""
    by_domain = {report.domain_key: report for report in reports}
    evidence = []
    for bridge in bridges:
        source_key = str(bridge.get('source_domain'))
        target_key = str(bridge.get('target_domain'))
        source = by_domain.get(source_key)
        target = by_domain.get(target_key)
        requirements = BRIDGE_BASIS_REQUIREMENTS.get(str(bridge.get('key')), {})
        source_basis = set(source.transfer_basis if source else ())
        target_basis = set(target.transfer_basis if target else ())
        required_source = set(requirements.get('source_any') or ())
        required_target = set(requirements.get('target_any') or ())
        source_matches = sorted(source_basis & required_source)
        target_matches = sorted(target_basis & required_target)
        shared_basis = sorted(source_basis & target_basis)
        passed = bool(source_matches and target_matches)
        evidence.append({
            'key': f"domain_transfer_evidence:{bridge.get('key')}",
            'bridge_key': bridge.get('key'),
            'source_domain': source_key,
            'target_domain': target_key,
            'status': 'transfer_link_ready' if passed else 'needs_observation',
            'source_basis': sorted(source_basis),
            'target_basis': sorted(target_basis),
            'required_source_basis': sorted(required_source),
            'required_target_basis': sorted(required_target),
            'source_matches': source_matches,
            'target_matches': target_matches,
            'shared_basis': shared_basis,
            'source_candidate_count': len(source.candidates) if source else 0,
            'target_candidate_count': len(target.candidates) if target else 0,
            'falsifies_if': bridge.get('falsifier'),
            'transfer_question': bridge.get('transfer_question'),
        })
    evidence.sort(
        key=lambda item: (
            item['status'] == 'transfer_link_ready',
            len(item['source_matches']) + len(item['target_matches']),
            item['bridge_key'] or '',
        ),
        reverse=True,
    )
    return evidence[:limit] if limit is not None else evidence


def _infer_candidates(observation: dict[str, Any]) -> list[DomainDiscoveryCandidate]:
    kind = str(observation.get('observation_kind', 'unknown'))
    if kind == 'collection_event':
        return [_candidate_from_collection_event(observation)]
    if kind == 'balanced_machine':
        return [_candidate_from_balanced_machine(observation)]
    if kind == 'rewrite_contrast':
        return [_candidate_from_rewrite_contrast(observation)]
    if kind == 'frame_change':
        return [_candidate_from_frame_change(observation)]
    if kind == 'boundary_probe':
        return [_candidate_from_boundary_probe(observation)]
    if kind == 'refined_series':
        return [_candidate_from_refined_series(observation)]
    if kind == 'local_prediction':
        return [_candidate_from_local_prediction(observation)]
    if kind == 'repeated_draws':
        return [_candidate_from_repeated_draws(observation)]
    if kind == 'evidence_split':
        return [_candidate_from_evidence_split(observation)]
    if kind == 'claim_checks':
        return [_candidate_from_claim_checks(observation)]
    if kind == 'partition_checks':
        return [_candidate_from_partition_checks(observation)]
    if kind == 'link_walks':
        return [_candidate_from_link_walks(observation)]
    if kind == 'state_steps':
        return [_candidate_from_state_steps(observation)]
    if kind == 'transform_family':
        return [_candidate_from_transform_family(observation)]
    if kind == 'operation_order':
        return [_candidate_from_operation_order(observation)]
    if kind == 'choice_scores':
        return [_candidate_from_choice_scores(observation)]
    if kind == 'constraint_scores':
        return [_candidate_from_constraint_scores(observation)]
    if kind == 'state_trace':
        return [_candidate_from_state_trace(observation)]
    if kind == 'cycle_trace':
        return [_candidate_from_cycle_trace(observation)]
    if kind == 'message_codes':
        return [_candidate_from_message_codes(observation)]
    if kind == 'hidden_state_machine':
        return [_candidate_from_hidden_state_machine(observation)]
    if kind == 'projection_pairs':
        return [_candidate_from_projection_pairs(observation)]
    if kind == 'dimension_change':
        return [_candidate_from_dimension_change(observation)]
    if kind == 'primitive_contrast':
        return [_candidate_from_primitive_contrast(observation)]
    return []


def _candidate_from_collection_event(observation: dict[str, Any]) -> DomainDiscoveryCandidate:
    before = dict(observation.get('before') or {})
    after = dict(observation.get('after') or {})
    event = dict(observation.get('event') or {})
    if 'group_a' in before and 'group_b' in before:
        left = _extent(before.get('group_a')) + _extent(before.get('group_b'))
        right = _extent(after.get('group_c'))
        return _candidate(
            observation,
            'collection_extent_join',
            'extent(after.group_c) == extent(before.group_a) + extent(before.group_b)',
            {'left_extent': left, 'right_extent': right},
            ('permute labels and require extent expression to stay fixed',),
            ('extent', 'composition', 'invariance', 'falsifier'),
            confidence=1.0 if left == right else 0.45,
        )
    before_extent = _extent(before.get('group'))
    after_extent = _extent(after.get('group'))
    direction = -1 if event.get('move') == 'remove_one' else 1
    return _candidate(
        observation,
        'collection_extent_step',
        'extent(after.group) == extent(before.group) + local_event_delta',
        {
            'before_extent': before_extent,
            'after_extent': after_extent,
            'observed_delta': after_extent - before_extent,
            'event_delta': direction,
        },
        ('repeat one-token event on a held-out collection size',),
        ('extent', 'local_change', 'falsifier'),
        confidence=1.0 if after_extent - before_extent == direction else 0.45,
    )


def _candidate_from_balanced_machine(observation: dict[str, Any]) -> DomainDiscoveryCandidate:
    pairs = list(observation.get('seen_pairs') or [])
    slope = None
    intercept = None
    if len(pairs) >= 2:
        first, second = pairs[0], pairs[1]
        dx = second['input'] - first['input']
        dy = second['output'] - first['output']
        slope = dy / dx if dx else None
        intercept = second['output'] - slope * second['input'] if slope is not None else None
    return _candidate(
        observation,
        'reversible_machine_relation',
        'output == stretch * input + shift; missing input reverses the steps',
        {'slope': slope, 'intercept': intercept, 'pair_count': len(pairs)},
        ('ask the inverse relation to recover a held-out input slot',),
        ('composition', 'inverse', 'relation', 'falsifier'),
    )


def _candidate_from_rewrite_contrast(observation: dict[str, Any]) -> DomainDiscoveryCandidate:
    checks = list(observation.get('checks') or [])
    mismatches = [
        check for check in checks
        if check.get('left') != check.get('right')
    ]
    return _candidate(
        observation,
        'rewrite_equivalence',
        'left_form(slot) - right_form(slot) == 0 over tested slots',
        {'check_count': len(checks), 'mismatch_count': len(mismatches)},
        ('evaluate both descriptions on a held-out slot',),
        ('equivalence', 'invariance', 'relation', 'falsifier'),
        confidence=1.0 if checks and not mismatches else 0.4,
    )


def _candidate_from_frame_change(observation: dict[str, Any]) -> DomainDiscoveryCandidate:
    readings = list(observation.get('paired_readings') or [])
    stable = len({reading.get('separation_reading') for reading in readings}) <= 1
    return _candidate(
        observation,
        'frame_stable_reading',
        'paired_reading(view_a) == paired_reading(view_b) after shared frame change',
        {'reading_count': len(readings), 'stable': stable},
        ('apply another shared frame shift and remeasure the paired reading',),
        ('invariance', 'metric', 'projection', 'falsifier'),
        confidence=1.0 if stable else 0.45,
    )


def _candidate_from_boundary_probe(observation: dict[str, Any]) -> DomainDiscoveryCandidate:
    checks = list(observation.get('membership_checks') or [])
    sides = sorted({check.get('side') for check in checks})
    return _candidate(
        observation,
        'partition_boundary',
        'marks split into low/boundary/high regions under re-description',
        {'check_count': len(checks), 'observed_sides': sides},
        ('shift the frame and require the same boundary membership pattern',),
        ('partition', 'locality', 'invariance', 'falsifier'),
    )


def _candidate_from_refined_series(observation: dict[str, Any]) -> DomainDiscoveryCandidate:
    changes = [float(value) for value in observation.get('neighbor_changes') or []]
    endpoint = float(observation.get('endpoint_change', 0.0) or 0.0)
    total_change = round(sum(changes), 3)
    return _candidate(
        observation,
        'local_change_accumulation',
        'sum(neighbor_changes) == endpoint_change',
        {'neighbor_sum': total_change, 'endpoint_change': endpoint},
        ('refine the trace and check whether summed local changes still match endpoints',),
        ('accumulation', 'local_change', 'state_update', 'falsifier'),
        confidence=1.0 if abs(total_change - endpoint) <= 0.002 else 0.45,
    )


def _candidate_from_local_prediction(observation: dict[str, Any]) -> DomainDiscoveryCandidate:
    return _candidate(
        observation,
        'local_prediction_rule',
        'nearby readings define a local step estimate that should beat far reuse',
        {
            'nearby_count': len(observation.get('nearby') or []),
            'has_far_probe': bool(observation.get('far')),
        },
        ('compare local estimate against a farther held-out sample',),
        ('local_change', 'approximation', 'falsifier'),
    )


def _candidate_from_repeated_draws(observation: dict[str, Any]) -> DomainDiscoveryCandidate:
    windows = list(observation.get('windows') or [])
    return _candidate(
        observation,
        'repeated_frequency_split',
        'longer windows should stabilize symbol mix better than short windows',
        {'window_count': len(windows), 'trial_count': len(observation.get('trial_log') or [])},
        ('append a later window and reject the rule if the mix systematically reverses',),
        ('frequency', 'uncertainty', 'partition', 'residual', 'falsifier'),
    )


def _candidate_from_evidence_split(observation: dict[str, Any]) -> DomainDiscoveryCandidate:
    candidates = list(observation.get('candidate_symbols_after_flag') or [])
    return _candidate(
        observation,
        'evidence_condition_split',
        'observed flag narrows candidate symbols for the next prediction',
        {'candidate_count_after_evidence': len(candidates)},
        ('remove the evidence flag and check whether the best prediction changes',),
        ('conditional', 'uncertainty', 'partition', 'falsifier'),
    )


def _candidate_from_claim_checks(observation: dict[str, Any]) -> DomainDiscoveryCandidate:
    return _candidate(
        observation,
        'claim_counterexample_search',
        'a claim is promoted only with supporting cases and a smallest breaking-case search',
        {'claim_count': len(observation.get('candidate_claims') or [])},
        ('search for the smallest observed case that breaks the implication',),
        ('implication', 'falsifier', 'partition'),
    )


def _candidate_from_partition_checks(observation: dict[str, Any]) -> DomainDiscoveryCandidate:
    return _candidate(
        observation,
        'predicate_partition',
        'predicate bins should cover observed cases without overlap or gaps',
        {'bin_count': len(observation.get('bins') or [])},
        ('add a held-out case and require exactly one bin assignment',),
        ('partition', 'falsifier', 'relation'),
    )


def _candidate_from_link_walks(observation: dict[str, Any]) -> DomainDiscoveryCandidate:
    return _candidate(
        observation,
        'link_walk_composition',
        'valid local links can be chained into reachable walks',
        {
            'link_count': len(observation.get('links') or []),
            'walk_count': len(observation.get('walks') or []),
        },
        ('reverse or reorder a walk and check whether local links still justify it',),
        ('relation', 'composition', 'path', 'falsifier'),
    )


def _candidate_from_state_steps(observation: dict[str, Any]) -> DomainDiscoveryCandidate:
    return _candidate(
        observation,
        'state_step_composition',
        'piecewise moves compose into the same final state as the combined move',
        {'step_count': len(observation.get('steps') or [])},
        ('compare a composed move against the same moves applied one at a time',),
        ('state_update', 'composition', 'relation', 'falsifier'),
    )


def _candidate_from_transform_family(observation: dict[str, Any]) -> DomainDiscoveryCandidate:
    return _candidate(
        observation,
        'transform_invariant_reading',
        'allowed transforms preserve a reading pattern while changing coordinates',
        {
            'view_count': len(observation.get('views') or []),
            'reading_count': len(observation.get('readings') or []),
        },
        ('apply a new transform and require the reading pattern to survive',),
        ('invariance', 'transform', 'equivalence', 'falsifier'),
    )


def _candidate_from_operation_order(observation: dict[str, Any]) -> DomainDiscoveryCandidate:
    return _candidate(
        observation,
        'transform_order_rule',
        'transform routes can preserve structure while still being order-sensitive',
        {'route_count': len(observation.get('routes') or [])},
        ('swap route order and check whether the final marks differ',),
        ('composition', 'order', 'transform', 'falsifier'),
    )


def _candidate_from_choice_scores(observation: dict[str, Any]) -> DomainDiscoveryCandidate:
    candidates = list(observation.get('candidates') or [])
    best = min(candidates, key=lambda item: item.get('cost_reading', 0.0)) if candidates else {}
    return _candidate(
        observation,
        'choice_extremum',
        'best choice has lower cost than its nearby alternatives',
        {'candidate_count': len(candidates), 'best_choice': best.get('choice')},
        ('step in the predicted improving direction and reject if cost rises',),
        ('local_change', 'extremum', 'optimization', 'falsifier'),
    )


def _candidate_from_constraint_scores(observation: dict[str, Any]) -> DomainDiscoveryCandidate:
    return _candidate(
        observation,
        'constraint_boundary_rule',
        'allowed boundaries can change which low-cost choice is usable',
        {
            'boundary_check_count': len(observation.get('boundary_checks') or []),
            'allowed_interval': observation.get('allowed_interval'),
        },
        ('try the unconstrained best outside the allowed interval and require rejection',),
        ('constraint', 'boundary', 'optimization', 'partition', 'falsifier'),
    )


def _candidate_from_state_trace(observation: dict[str, Any]) -> DomainDiscoveryCandidate:
    residuals = [float(value) for value in observation.get('residual_readings') or []]
    stable_residual = len({round(value, 3) for value in residuals}) <= 1
    return _candidate(
        observation,
        'state_update_residual_rule',
        'next state is better explained by a repeated residual update than by repeating the last step',
        {'trace_count': len(observation.get('trace') or []), 'stable_residual': stable_residual},
        ('roll the one-step residual update across a longer held-out trace',),
        ('state_update', 'residual', 'recurrence', 'falsifier'),
    )


def _candidate_from_cycle_trace(observation: dict[str, Any]) -> DomainDiscoveryCandidate:
    return _candidate(
        observation,
        'cycle_recurrence_rule',
        'phase marks repeat after a stable gap and predict later readings',
        {'repeat_gap': observation.get('repeat_gap')},
        ('advance by another repeat gap and require the phase mark to recur',),
        ('recurrence', 'state_update', 'periodic', 'falsifier'),
    )


def _candidate_from_message_codes(observation: dict[str, Any]) -> DomainDiscoveryCandidate:
    descriptions = list(observation.get('descriptions') or [])
    best = min(descriptions, key=lambda item: item.get('units', 0)) if descriptions else {}
    return _candidate(
        observation,
        'compression_description_rule',
        'shorter descriptions are preferred only when they reconstruct held-out observations',
        {'description_count': len(descriptions), 'shortest_method': best.get('method')},
        ('decode the shorter description on a held-out prefix',),
        ('compression', 'recurrence', 'uncertainty', 'falsifier'),
    )


def _candidate_from_hidden_state_machine(observation: dict[str, Any]) -> DomainDiscoveryCandidate:
    return _candidate(
        observation,
        'hidden_state_recurrence',
        'small internal state predicts recurring visible outputs across longer cycles',
        {
            'observation_count': len(observation.get('observations') or []),
            'candidate_memory_sizes': observation.get('candidate_memory_sizes'),
        },
        ('extend the cycle and reject the smallest state that fails to predict output',),
        ('state_update', 'recurrence', 'compression', 'falsifier'),
    )


def _candidate_from_projection_pairs(observation: dict[str, Any]) -> DomainDiscoveryCandidate:
    return _candidate(
        observation,
        'projection_lift_rule',
        'extra readings can explain differences invisible in one projection',
        {
            'visible_count': len(observation.get('visible_marks') or []),
            'paired_count': len(observation.get('paired_readings') or []),
        },
        ('change projection view and require the extra coordinate to keep explaining residuals',),
        ('projection', 'dimension_lift', 'invariance', 'falsifier'),
    )


def _candidate_from_dimension_change(observation: dict[str, Any]) -> DomainDiscoveryCandidate:
    return _candidate(
        observation,
        'dimension_generalization_rule',
        'a rule promoted as general should keep its form when axis count changes',
        {'family_count': len(observation.get('families') or [])},
        ('test the same rule on a held-out axis count',),
        ('dimension_lift', 'invariance', 'generalization', 'falsifier'),
    )


def _candidate_from_primitive_contrast(observation: dict[str, Any]) -> DomainDiscoveryCandidate:
    contrast_kind = str(observation.get('contrast_kind') or 'unknown')
    config = PRIMITIVE_CONTRAST_RELATIONS.get(contrast_kind)
    if not config:
        return _candidate(
            observation,
            'primitive_unknown_contrast',
            'public primitive contrast needs a candidate relation',
            {'contrast_kind': contrast_kind},
            ('design a held-out contrast that separates the primitive relation',),
            ('falsifier',),
            confidence=0.35,
        )
    heldout = dict(observation.get('heldout_probe') or {})
    falsifier = heldout.get('ask') or 'run the public held-out contrast'
    evidence = {
        'contrast_kind': contrast_kind,
        'primitive_count': len(observation.get('public_primitives') or []),
        'example_count': len(observation.get('observed_examples') or []),
        'has_heldout_probe': bool(heldout),
        'control_question': observation.get('control_question'),
    }
    confidence = 0.92 if evidence['has_heldout_probe'] else 0.55
    return _candidate(
        observation,
        str(config['relation_kind']),
        str(config['expression']),
        evidence,
        (str(falsifier),),
        tuple(config['basis']),
        confidence=confidence,
    )


def _candidate(
    observation: dict[str, Any],
    relation_kind: str,
    expression: str,
    evidence: dict[str, Any],
    falsification_tests: tuple[str, ...],
    transfer_basis: tuple[str, ...],
    confidence: float = 0.9,
) -> DomainDiscoveryCandidate:
    sample_id = str(observation.get('sample_id', 'unknown'))
    observation_kind = str(observation.get('observation_kind', 'unknown'))
    return DomainDiscoveryCandidate(
        key=f"{observation_kind}:{sample_id}:{relation_kind}",
        observation_kind=observation_kind,
        relation_kind=relation_kind,
        expression=expression,
        confidence=round(float(confidence), 3),
        support_sample_ids=(sample_id,),
        evidence=evidence,
        falsification_tests=falsification_tests,
        transfer_basis=transfer_basis,
    )


def _extent(value: Any) -> int:
    if isinstance(value, (list, tuple, set)):
        return len(value)
    if isinstance(value, dict):
        return len(value)
    return 0


def _json_copy(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_copy(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_copy(item) for item in value]
    if isinstance(value, list):
        return [_json_copy(item) for item in value]
    return value
