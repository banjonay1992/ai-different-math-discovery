from __future__ import annotations

"""
Autonomous discovery loop for equation-driven science.

The equation workbench answers "what fits?"  This layer asks the next
scientific questions: what theory does that imply, what concept did the agent
need to invent, where does the theory still fail, and what probe would
separate it from competing explanations?
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .resource_efficiency import (
    build_canonical_law_shard,
    build_compressed_experience_shard,
    canonical_law_compression_report,
    operator_outcome_anchor_indexes,
    resource_efficiency_report,
)


FIRST_PRINCIPLES_BASIS: list[dict[str, Any]] = [
    {
        'key': 'identity_equality',
        'name': 'identity and equality',
        'primitive_kind': 'logic',
        'expression': 'a = a; if a = b then substitute b for a',
        'unlocks': ['proof_checks', 'domain_predicates', 'equation_comparison'],
    },
    {
        'key': 'composition',
        'name': 'composition',
        'primitive_kind': 'algebra',
        'expression': 'h(x) = f(g(x))',
        'unlocks': ['operator_chains', 'derived_variables', 'nested_transforms'],
    },
    {
        'key': 'inverse_operation',
        'name': 'inverse operation',
        'primitive_kind': 'algebra',
        'expression': 'solve by applying an operation that undoes another',
        'unlocks': ['ratio_exponents', 'residual_correction', 'hidden_parameter_search'],
    },
    {
        'key': 'order_metric',
        'name': 'order and metric comparison',
        'primitive_kind': 'measurement',
        'expression': 'compare magnitudes, distances, and before/after order',
        'unlocks': ['distance', 'boundary_margin', 'nearest_counterexample'],
    },
    {
        'key': 'symmetry_transform',
        'name': 'symmetry transform',
        'primitive_kind': 'geometry',
        'expression': 'translate, rotate, reflect, or change basis while preserving structure',
        'unlocks': ['orthogonal_projection', 'invariant_coordinates', 'relative_frame'],
    },
    {
        'key': 'recursion_iteration',
        'name': 'recursion and iteration',
        'primitive_kind': 'process',
        'expression': 'state[n+1] = F(state[n])',
        'unlocks': ['phase', 'induction', 'long_horizon_pattern'],
    },
    {
        'key': 'conservation_balance',
        'name': 'conservation and balance',
        'primitive_kind': 'invariant',
        'expression': 'quantity_before - quantity_after ~= 0 unless a domain event applies',
        'unlocks': ['invariants', 'counterexample_search', 'domain_breaks'],
    },
    {
        'key': 'domain_partition',
        'name': 'domain partition',
        'primitive_kind': 'logic',
        'expression': 'rule applies under predicate P and fails or changes under not-P',
        'unlocks': ['piecewise_laws', 'domain_limited_claims', 'adaptive_dimensions'],
    },
    {
        'key': 'dimension_lift',
        'name': 'dimension lift',
        'primitive_kind': 'representation',
        'expression': 'invent a coordinate z = phi(observations, residuals) when current axes fail',
        'unlocks': ['latent_dimensions', 'projection_axes', 'new_operator_inputs'],
    },
]


ALGEBRAIC_EXPRESSION_FAMILIES: list[dict[str, Any]] = [
    {
        'key': 'constant_identity',
        'name': 'constant and identity',
        'expression_schema': ['c', 'x'],
        'search_parameters': ['constant', 'identity_channel'],
        'structure_primitives': ['identity_equality'],
        'good_for': ['null_models', 'invariants', 'baseline_comparison'],
        'proof_obligations': ['identity_substitution', 'dimensional_consistency'],
        'complexity_cost': 1,
    },
    {
        'key': 'affine_linear',
        'name': 'affine and linear forms',
        'expression_schema': ['a*x + b', 'dot(w, x) + b'],
        'search_parameters': ['slope', 'offset', 'weight_vector'],
        'structure_primitives': ['composition', 'conservation_balance'],
        'good_for': ['rates', 'proportionality', 'local_tangent_models'],
        'proof_obligations': ['closure_under_addition', 'dimensional_consistency'],
        'complexity_cost': 2,
    },
    {
        'key': 'polynomial_basis',
        'name': 'polynomial basis',
        'expression_schema': ['sum_i a_i*x^i', 'sum_ij a_ij*x_i*x_j'],
        'search_parameters': ['degree', 'coefficients', 'cross_terms'],
        'structure_primitives': ['composition', 'recursion_iteration'],
        'good_for': ['smooth_curvature', 'local_series', 'interaction_terms'],
        'proof_obligations': ['closure_under_addition', 'closure_under_multiplication'],
        'complexity_cost': 4,
    },
    {
        'key': 'rational_ratio',
        'name': 'rational and ratio forms',
        'expression_schema': ['P(x) / Q(x)', 'a / (x + eps)'],
        'search_parameters': ['numerator_degree', 'denominator_degree', 'safe_domain'],
        'structure_primitives': ['inverse_operation', 'domain_partition'],
        'good_for': ['inverse_response', 'saturation', 'scale_free_comparison'],
        'proof_obligations': ['nonzero_domain', 'equivalence_under_rewrite'],
        'complexity_cost': 5,
    },
    {
        'key': 'power_law',
        'name': 'power-law forms',
        'expression_schema': ['a*x^p', 'a / x^p'],
        'search_parameters': ['exponent', 'scale', 'safe_domain'],
        'structure_primitives': ['order_metric', 'inverse_operation', 'composition'],
        'good_for': ['distance_decay', 'growth_scaling', 'scale_invariance'],
        'proof_obligations': ['monotonicity', 'nonzero_domain', 'dimensional_consistency'],
        'complexity_cost': 4,
    },
    {
        'key': 'logarithmic',
        'name': 'logarithmic forms',
        'expression_schema': ['a*log(x + c) + b', 'log(x/y)'],
        'search_parameters': ['scale', 'offset', 'ratio_channels'],
        'structure_primitives': ['inverse_operation', 'order_metric'],
        'good_for': ['exponent_estimation', 'multiplicative_residuals', 'compression'],
        'proof_obligations': ['positive_domain', 'equivalence_under_rewrite'],
        'complexity_cost': 4,
    },
    {
        'key': 'exponential',
        'name': 'exponential forms',
        'expression_schema': ['a*exp(k*x) + b', 'a*r^t'],
        'search_parameters': ['rate', 'base', 'scale'],
        'structure_primitives': ['recursion_iteration', 'inverse_operation'],
        'good_for': ['compound_growth', 'decay', 'repeated_transform'],
        'proof_obligations': ['recurrence_consistency', 'monotonicity'],
        'complexity_cost': 4,
    },
    {
        'key': 'sinusoidal_phase',
        'name': 'sinusoidal and phase forms',
        'expression_schema': ['a*sin(w*t + p)', 'a*cos(w*t + p)'],
        'search_parameters': ['frequency', 'phase', 'amplitude'],
        'structure_primitives': ['recursion_iteration', 'symmetry_transform'],
        'good_for': ['periodicity', 'rotation', 'oscillation'],
        'proof_obligations': ['periodicity_closure', 'phase_shift_invariance'],
        'complexity_cost': 5,
    },
    {
        'key': 'piecewise_predicate',
        'name': 'piecewise predicate forms',
        'expression_schema': ['if P(x) then f(x) else g(x)'],
        'search_parameters': ['predicate', 'boundary', 'inside_model', 'outside_model'],
        'structure_primitives': ['domain_partition', 'identity_equality'],
        'good_for': ['domain_limited_laws', 'event_rules', 'counterexample_repair'],
        'proof_obligations': ['predicate_partition_exhaustive', 'boundary_behavior'],
        'complexity_cost': 6,
    },
    {
        'key': 'recurrence_iteration',
        'name': 'recurrence and iteration forms',
        'expression_schema': ['x[n+1] = F(x[n])', 'x[n+k] = G(history)'],
        'search_parameters': ['lag', 'transition_operator', 'memory_length'],
        'structure_primitives': ['recursion_iteration', 'composition'],
        'good_for': ['motion_update', 'induction', 'state_machine_patterns'],
        'proof_obligations': ['induction_step', 'fixed_point_or_cycle_check'],
        'complexity_cost': 5,
    },
    {
        'key': 'finite_difference_calculus',
        'name': 'finite-difference calculus',
        'expression_schema': ['delta(x)/delta(t)', 'delta(delta(x))'],
        'search_parameters': ['time_step', 'difference_order', 'smoothing_window'],
        'structure_primitives': ['recursion_iteration', 'conservation_balance'],
        'good_for': ['velocity', 'acceleration', 'curvature'],
        'proof_obligations': ['step_size_stability', 'dimensional_consistency'],
        'complexity_cost': 4,
    },
    {
        'key': 'accumulation_integral',
        'name': 'accumulation and integral forms',
        'expression_schema': ['sum_t f(t)*dt', 'running_total += flow'],
        'search_parameters': ['window', 'flow_channel', 'initial_value'],
        'structure_primitives': ['conservation_balance', 'recursion_iteration'],
        'good_for': ['conserved_totals', 'area_under_change', 'budget_balance'],
        'proof_obligations': ['telescoping_balance', 'initial_condition_check'],
        'complexity_cost': 4,
    },
    {
        'key': 'vector_projection_norm',
        'name': 'vector projection and norm forms',
        'expression_schema': ['dot(v, u)', 'norm(v)', 'unit(v)'],
        'search_parameters': ['basis_vector', 'norm_power', 'projection_axis'],
        'structure_primitives': ['symmetry_transform', 'order_metric'],
        'good_for': ['direction_fields', 'orthogonal_components', 'coordinate_lifts'],
        'proof_obligations': ['basis_invariance', 'normalization_domain'],
        'complexity_cost': 4,
    },
    {
        'key': 'matrix_linear_transform',
        'name': 'matrix and linear-transform forms',
        'expression_schema': ['A*x', 'R(theta)*x', 'A^-1*x'],
        'search_parameters': ['matrix_entries', 'rank', 'basis'],
        'structure_primitives': ['symmetry_transform', 'composition', 'inverse_operation'],
        'good_for': ['basis_change', 'rotation', 'coupled_channels'],
        'proof_obligations': ['rank_or_invertibility', 'composition_consistency'],
        'complexity_cost': 6,
    },
    {
        'key': 'set_relation_cardinality',
        'name': 'set relation and cardinality forms',
        'expression_schema': ['x in S', 'count(S)', 'S subset T'],
        'search_parameters': ['membership_predicate', 'count_channel', 'subset_relation'],
        'structure_primitives': ['identity_equality', 'domain_partition'],
        'good_for': ['classification', 'existence', 'counting'],
        'proof_obligations': ['membership_consistency', 'partition_exhaustion'],
        'complexity_cost': 3,
    },
    {
        'key': 'graph_relation_path',
        'name': 'graph relation and path forms',
        'expression_schema': ['edge(a,b)', 'path_length(a,b)', 'neighborhood(a)'],
        'search_parameters': ['edge_predicate', 'path_metric', 'neighborhood_radius'],
        'structure_primitives': ['composition', 'order_metric'],
        'good_for': ['adjacency', 'causal_chains', 'interaction_networks'],
        'proof_obligations': ['path_composition', 'reachability_counterexample'],
        'complexity_cost': 5,
    },
    {
        'key': 'probability_statistics',
        'name': 'probability and statistics forms',
        'expression_schema': ['mean(x)', 'var(x)', 'P(event | context)'],
        'search_parameters': ['sample_window', 'event_predicate', 'confidence_threshold'],
        'structure_primitives': ['order_metric', 'domain_partition'],
        'good_for': ['noisy_rules', 'confidence', 'rare_counterexamples'],
        'proof_obligations': ['sample_support', 'heldout_calibration'],
        'complexity_cost': 4,
    },
    {
        'key': 'optimization_extremum',
        'name': 'optimization and extremum forms',
        'expression_schema': ['argmin_x loss(x)', 'max_x score(x)'],
        'search_parameters': ['objective', 'candidate_space', 'constraint'],
        'structure_primitives': ['order_metric', 'domain_partition'],
        'good_for': ['inferred_centers', 'best_explanation', 'constraint_solving'],
        'proof_obligations': ['objective_decreases', 'counterexample_search'],
        'complexity_cost': 5,
    },
    {
        'key': 'symmetry_invariant',
        'name': 'symmetry and invariant forms',
        'expression_schema': ['f(T(x)) = f(x)', 'T^-1(T(x)) = x'],
        'search_parameters': ['transform', 'invariant_channel', 'basis'],
        'structure_primitives': ['symmetry_transform', 'inverse_operation'],
        'good_for': ['coordinate_free_laws', 'conservation', 'basis_change'],
        'proof_obligations': ['invariance_under_transform', 'inverse_transform_check'],
        'complexity_cost': 5,
    },
]


ALGEBRAIC_STRUCTURES: list[dict[str, Any]] = [
    {
        'key': 'magma',
        'name': 'magma',
        'operations': ['binary_combine'],
        'axioms': ['closure'],
        'discovery_use': 'ask whether a learned combine operation stays inside the observed domain',
    },
    {
        'key': 'semigroup',
        'name': 'semigroup',
        'operations': ['binary_combine'],
        'axioms': ['closure', 'associativity'],
        'discovery_use': 'test whether repeated composition can be regrouped without changing outcomes',
    },
    {
        'key': 'monoid',
        'name': 'monoid',
        'operations': ['binary_combine', 'identity'],
        'axioms': ['closure', 'associativity', 'identity'],
        'discovery_use': 'look for a do-nothing element in transformations or counts',
    },
    {
        'key': 'group',
        'name': 'group',
        'operations': ['combine', 'identity', 'inverse'],
        'axioms': ['closure', 'associativity', 'identity', 'inverse'],
        'discovery_use': 'test reversible transforms, rotations, translations, and phase cycles',
    },
    {
        'key': 'ring',
        'name': 'ring',
        'operations': ['addition', 'multiplication'],
        'axioms': ['additive_group', 'multiplicative_associativity', 'distributivity'],
        'discovery_use': 'support polynomial and recurrence equations over additive and multiplicative terms',
    },
    {
        'key': 'field',
        'name': 'field',
        'operations': ['addition', 'multiplication', 'division'],
        'axioms': ['ring_axioms', 'multiplicative_inverse_for_nonzero'],
        'discovery_use': 'support ratios, inverse laws, normalized coordinates, and equation solving',
    },
    {
        'key': 'vector_space',
        'name': 'vector space',
        'operations': ['vector_addition', 'scalar_multiplication'],
        'axioms': ['closure', 'zero_vector', 'additive_inverse', 'distributivity'],
        'discovery_use': 'support basis changes, projections, directions, and linear residual decompositions',
    },
    {
        'key': 'ordered_set',
        'name': 'ordered set',
        'operations': ['less_than', 'compare'],
        'axioms': ['reflexivity_or_totality', 'antisymmetry', 'transitivity'],
        'discovery_use': 'support ranking, thresholds, boundaries, and monotonic relations',
    },
    {
        'key': 'metric_space',
        'name': 'metric space',
        'operations': ['distance'],
        'axioms': ['nonnegative_distance', 'identity_of_indiscernibles', 'symmetry', 'triangle_inequality'],
        'discovery_use': 'support separation, neighborhoods, nearest counterexamples, and local laws',
    },
    {
        'key': 'lattice_boolean_algebra',
        'name': 'lattice and boolean algebra',
        'operations': ['and', 'or', 'not', 'meet', 'join'],
        'axioms': ['idempotence', 'absorption', 'complement', 'distributivity'],
        'discovery_use': 'support domain predicates, rule partitions, and logical counterexample repair',
    },
    {
        'key': 'topological_neighborhood',
        'name': 'topological neighborhood',
        'operations': ['open_region', 'boundary', 'closure'],
        'axioms': ['neighborhood_contains_point', 'unions_preserve_openness'],
        'discovery_use': 'support locality, continuity, inside/outside laws, and boundary probes',
    },
    {
        'key': 'graph',
        'name': 'graph',
        'operations': ['edge', 'path', 'neighborhood'],
        'axioms': ['path_composition', 'adjacency_consistency'],
        'discovery_use': 'support interaction networks, causal paths, and relational structure',
    },
    {
        'key': 'probability_space',
        'name': 'probability space',
        'operations': ['event', 'measure', 'conditional_probability'],
        'axioms': ['nonnegative_measure', 'total_measure_one', 'additivity'],
        'discovery_use': 'support noisy hypotheses, calibration, and probabilistic counterexamples',
    },
]


ALGEBRAIC_PROOF_OBLIGATIONS: list[dict[str, Any]] = [
    {
        'key': 'closure',
        'statement': 'candidate operation maps observed inputs back into its claimed domain',
        'catches': ['type_leakage', 'invalid_generated_dimension'],
        'applies_to': ['magma', 'semigroup', 'vector_space', 'ring'],
    },
    {
        'key': 'identity_substitution',
        'statement': 'substituting equal expressions preserves prediction and score',
        'catches': ['spurious_feature_alias', 'notation_without_equivalence'],
        'applies_to': ['constant_identity', 'affine_linear'],
    },
    {
        'key': 'associativity',
        'statement': 'regrouping repeated composition does not alter the result',
        'catches': ['order_sensitive_composition_claimed_as_general'],
        'applies_to': ['semigroup', 'monoid', 'group'],
    },
    {
        'key': 'identity_and_inverse',
        'statement': 'the system can find no-op and undo operations when a structure claims them',
        'catches': ['irreversible_transform_marked_reversible'],
        'applies_to': ['monoid', 'group', 'field', 'matrix_linear_transform'],
    },
    {
        'key': 'commutativity_or_order_sensitivity',
        'statement': 'the system states whether swapping operands preserves or changes the result',
        'catches': ['wrongly_symmetric_operator', 'hidden_time_order'],
        'applies_to': ['ring', 'field', 'graph_relation_path'],
    },
    {
        'key': 'distributivity',
        'statement': 'multiplication or scaling distributes over claimed addition-like structure',
        'catches': ['invalid_polynomial_rewrite'],
        'applies_to': ['ring', 'field', 'vector_space', 'polynomial_basis'],
    },
    {
        'key': 'domain_nonzero_positive',
        'statement': 'division, inverse, power, and log forms declare excluded or positive domains',
        'catches': ['division_by_zero', 'log_of_nonpositive', 'invalid_power_domain'],
        'applies_to': ['rational_ratio', 'power_law', 'logarithmic', 'field'],
    },
    {
        'key': 'dimensional_consistency',
        'statement': 'both sides of an equation use compatible units or explicitly learned dimensions',
        'catches': ['unit_mismatch', 'dimensionless_exponent_misuse'],
        'applies_to': ['all_equation_families'],
    },
    {
        'key': 'monotonicity_or_extremum',
        'statement': 'order claims identify where a relation increases, decreases, or reaches an extremum',
        'catches': ['wrong_direction_of_effect', 'local_optimum_misread_as_global'],
        'applies_to': ['power_law', 'logarithmic', 'exponential', 'optimization_extremum'],
    },
    {
        'key': 'boundary_and_partition',
        'statement': 'piecewise domains cover the relevant cases and state boundary behavior',
        'catches': ['missing_outside_case', 'boundary_discontinuity_without_event'],
        'applies_to': ['piecewise_predicate', 'set_relation_cardinality', 'topological_neighborhood'],
    },
    {
        'key': 'induction_or_recurrence',
        'statement': 'a transition rule proves a base case and repeated-step consistency',
        'catches': ['one_step_fit_that_fails_long_horizon'],
        'applies_to': ['recurrence_iteration', 'exponential', 'finite_difference_calculus'],
    },
    {
        'key': 'symmetry_invariance',
        'statement': 'claimed coordinate-free laws survive transform, inverse transform, or basis change',
        'catches': ['coordinate_artifact', 'orientation_overfit'],
        'applies_to': ['symmetry_invariant', 'matrix_linear_transform', 'vector_projection_norm'],
    },
    {
        'key': 'heldout_counterexample',
        'statement': 'every promoted form names a held-out setting that would break it',
        'catches': ['overfit_expression', 'unfalsifiable_claim'],
        'applies_to': ['all_equation_families'],
    },
]


ALGEBRAIC_SEARCH_CONTROLS: dict[str, Any] = {
    'baseline_version': 1,
    'complexity_budget': {
        'start_max_cost': 6,
        'expand_when': 'heldout residual or disagreement remains after simpler families',
        'penalize': ['unused_terms', 'unproven_domain_split', 'unchecked_inverse'],
    },
    'anti_template_policy': (
        'families are allowed search moves, not answers; every promoted equation '
        'must win held-out scoring and state falsification conditions'
    ),
    'dimension_policy': (
        'new variables may be invented from residuals, but they must improve transfer, '
        'separate rivals, or explain a domain-limited failure'
    ),
    'proof_policy': (
        'attach the cheapest relevant obligation first, then escalate to algebraic '
        'structure checks when an operator is reused across contexts'
    ),
}


MATH_DOMAIN_CURRICULUM: list[dict[str, Any]] = [
    {
        'key': 'arithmetic_quantity',
        'name': 'arithmetic and quantity',
        'curriculum_order': 1,
        'world_seed': 'finite objects that can appear, disappear, combine, and split',
        'observation_tasks': [
            'count stable objects under permutation',
            'compare more, fewer, same, zero, and one-more transitions',
            'compose combine/split actions and check whether totals agree',
        ],
        'primitive_targets': ['number', 'successor', 'addition', 'subtraction', 'equality'],
        'equation_families': ['constant_identity', 'affine_linear', 'set_relation_cardinality'],
        'proof_pressure': ['closure', 'identity_substitution', 'associativity'],
        'expected_discoveries': ['count invariance', 'successor arithmetic', 'conservation of total under regrouping'],
    },
    {
        'key': 'algebra_equations',
        'name': 'algebra and symbolic equations',
        'curriculum_order': 2,
        'world_seed': 'unknown quantities hidden behind reversible operations',
        'observation_tasks': [
            'solve missing values from balanced transformations',
            'compose and invert operations',
            'compare equivalent expressions on held-out substitutions',
        ],
        'primitive_targets': ['variable', 'inverse operation', 'composition', 'equivalence class'],
        'equation_families': ['affine_linear', 'polynomial_basis', 'rational_ratio'],
        'proof_pressure': ['identity_substitution', 'distributivity', 'domain_nonzero_positive'],
        'expected_discoveries': ['symbolic substitution', 'equation balance', 'factor/rewrite equivalence'],
    },
    {
        'key': 'geometry_space',
        'name': 'geometry and measurement',
        'curriculum_order': 3,
        'world_seed': 'points, distances, angles, boundaries, and coordinate changes',
        'observation_tasks': [
            'measure invariant distance under translation and rotation',
            'infer centers, axes, and boundaries from observations',
            'compare coordinate descriptions of the same shape',
        ],
        'primitive_targets': ['point', 'distance', 'angle', 'coordinate frame', 'shape invariant'],
        'equation_families': ['vector_projection_norm', 'matrix_linear_transform', 'symmetry_invariant'],
        'proof_pressure': ['symmetry_invariance', 'dimensional_consistency', 'boundary_and_partition'],
        'expected_discoveries': ['metric distance', 'coordinate transform', 'local/global shape distinction'],
    },
    {
        'key': 'calculus_change',
        'name': 'calculus and change',
        'curriculum_order': 4,
        'world_seed': 'smooth and abrupt change sampled over time',
        'observation_tasks': [
            'estimate velocity, acceleration, and curvature from finite differences',
            'compare accumulated change against endpoint differences',
            'separate smooth trends from boundary events',
        ],
        'primitive_targets': ['rate', 'finite difference', 'accumulation', 'limit-like refinement'],
        'equation_families': ['finite_difference_calculus', 'accumulation_integral', 'optimization_extremum'],
        'proof_pressure': ['induction_or_recurrence', 'dimensional_consistency', 'monotonicity_or_extremum'],
        'expected_discoveries': ['derivative-like rate', 'integral-like accumulation', 'local linear approximation'],
    },
    {
        'key': 'probability_uncertainty',
        'name': 'probability and uncertainty',
        'curriculum_order': 5,
        'world_seed': 'stochastic events, noisy measurements, and repeated trials',
        'observation_tasks': [
            'estimate frequencies from repeated samples',
            'update beliefs after evidence',
            'distinguish noise from deterministic residual structure',
        ],
        'primitive_targets': ['sample space', 'frequency', 'conditional probability', 'expectation'],
        'equation_families': ['probability_statistics', 'optimization_extremum', 'piecewise_predicate'],
        'proof_pressure': ['boundary_and_partition', 'heldout_counterexample', 'dimensional_consistency'],
        'expected_discoveries': ['frequency convergence', 'conditional split', 'expected error minimization'],
    },
    {
        'key': 'logic_proof',
        'name': 'logic and proof',
        'curriculum_order': 6,
        'world_seed': 'statements, predicates, implications, contradictions, and counterexamples',
        'observation_tasks': [
            'test whether predicates partition examples',
            'build implication chains from observed rules',
            'search for the smallest counterexample to a claim',
        ],
        'primitive_targets': ['predicate', 'implication', 'contradiction', 'counterexample', 'proof obligation'],
        'equation_families': ['piecewise_predicate', 'set_relation_cardinality', 'symmetry_invariant'],
        'proof_pressure': ['boundary_and_partition', 'heldout_counterexample', 'identity_substitution'],
        'expected_discoveries': ['falsification', 'domain restriction', 'proof by repeated invariant check'],
    },
    {
        'key': 'discrete_structures',
        'name': 'discrete structures and graphs',
        'curriculum_order': 7,
        'world_seed': 'nodes, edges, paths, neighborhoods, orderings, and finite state transitions',
        'observation_tasks': [
            'discover adjacency and reachability',
            'compare path composition and shortest routes',
            'track state-machine transitions over repeated steps',
        ],
        'primitive_targets': ['set', 'relation', 'graph', 'path', 'state transition'],
        'equation_families': ['graph_relation_path', 'set_relation_cardinality', 'recurrence_iteration'],
        'proof_pressure': ['closure', 'associativity', 'induction_or_recurrence'],
        'expected_discoveries': ['path composition', 'connectivity', 'finite transition algebra'],
    },
    {
        'key': 'symmetry_invariance',
        'name': 'symmetry and invariance',
        'curriculum_order': 8,
        'world_seed': 'transformations that preserve or break structure',
        'observation_tasks': [
            'apply translations, rotations, reflections, and relabelings',
            'detect quantities unchanged by transformations',
            'separate true invariants from coordinate artifacts',
        ],
        'primitive_targets': ['transform', 'orbit', 'invariant', 'equivalence under symmetry'],
        'equation_families': ['symmetry_invariant', 'matrix_linear_transform', 'vector_projection_norm'],
        'proof_pressure': ['symmetry_invariance', 'identity_substitution', 'commutativity_or_order_sensitivity'],
        'expected_discoveries': ['group-like composition', 'invariant quantity', 'coordinate-free law'],
    },
    {
        'key': 'optimization_extrema',
        'name': 'optimization and extrema',
        'curriculum_order': 9,
        'world_seed': 'actions with costs, rewards, constraints, and tradeoffs',
        'observation_tasks': [
            'compare candidate rules by residual/error cost',
            'search local and global optima',
            'discover constraints and active boundaries',
        ],
        'primitive_targets': ['objective', 'constraint', 'minimum', 'maximum', 'gradient-like direction'],
        'equation_families': ['optimization_extremum', 'finite_difference_calculus', 'probability_statistics'],
        'proof_pressure': ['monotonicity_or_extremum', 'boundary_and_partition', 'heldout_counterexample'],
        'expected_discoveries': ['least-error fit', 'constraint boundary', 'tradeoff curve'],
    },
    {
        'key': 'dynamics_systems',
        'name': 'dynamics and systems',
        'curriculum_order': 10,
        'world_seed': 'interacting objects with hidden forces, phases, feedback, and conservation',
        'observation_tasks': [
            'infer residual fields after a simple transition baseline',
            'discover attractors, rotations, periodic forcing, and conservation',
            'compose multiple laws and test held-out trajectories',
        ],
        'primitive_targets': ['state', 'transition', 'field', 'feedback', 'conserved quantity'],
        'equation_families': ['recurrence_iteration', 'finite_difference_calculus', 'sinusoidal_phase'],
        'proof_pressure': ['induction_or_recurrence', 'dimensional_consistency', 'heldout_counterexample'],
        'expected_discoveries': ['state update law', 'residual field', 'phase or conservation law'],
    },
    {
        'key': 'information_computation',
        'name': 'information and computation',
        'curriculum_order': 11,
        'world_seed': 'messages, encodings, compression, decision trees, and algorithms',
        'observation_tasks': [
            'compare descriptions by compression and prediction cost',
            'infer hidden state from partial observations',
            'learn procedures that generalize across input sizes',
        ],
        'primitive_targets': ['encoding', 'entropy-like uncertainty', 'algorithm', 'state memory'],
        'equation_families': ['probability_statistics', 'recurrence_iteration', 'graph_relation_path'],
        'proof_pressure': ['induction_or_recurrence', 'heldout_counterexample', 'dimensional_consistency'],
        'expected_discoveries': ['compression preference', 'hidden-state inference', 'algorithmic recurrence'],
    },
    {
        'key': 'higher_dimensions',
        'name': 'higher-dimensional worlds',
        'curriculum_order': 12,
        'world_seed': 'latent axes, projections, manifolds, and arbitrary-dimensional coordinates',
        'observation_tasks': [
            'invent new coordinates when residuals cannot be expressed in visible axes',
            'project high-dimensional structure into observable views',
            'test whether learned laws survive dimension changes',
        ],
        'primitive_targets': ['dimension lift', 'projection', 'basis', 'latent coordinate', 'manifold-like neighborhood'],
        'equation_families': ['vector_projection_norm', 'matrix_linear_transform', 'symmetry_invariant'],
        'proof_pressure': ['symmetry_invariance', 'heldout_counterexample', 'dimensional_consistency'],
        'expected_discoveries': ['latent axis', 'projection invariance', 'dimension-independent law'],
    },
]


MATH_DOMAIN_TRANSFER_BRIDGES: list[dict[str, Any]] = [
    {
        'key': 'quantity_to_algebra',
        'source_domain': 'arithmetic_quantity',
        'target_domain': 'algebra_equations',
        'bridge_principle': 'counts become variables and operations become reusable transformations',
        'transfer_question': 'Does a counting rule still hold when the count is hidden behind an unknown?',
        'falsifier': 'a substitution that preserves counts breaks the symbolic equation',
    },
    {
        'key': 'algebra_to_geometry',
        'source_domain': 'algebra_equations',
        'target_domain': 'geometry_space',
        'bridge_principle': 'equations define coordinate constraints and geometric loci',
        'transfer_question': 'Can equivalent expressions describe the same shape under a coordinate change?',
        'falsifier': 'two algebraically equivalent forms produce different measured shapes',
    },
    {
        'key': 'geometry_to_calculus',
        'source_domain': 'geometry_space',
        'target_domain': 'calculus_change',
        'bridge_principle': 'local geometric differences become rates and curvature',
        'transfer_question': 'Does a small spatial displacement predict the observed change rate?',
        'falsifier': 'refined samples reject the local linear or curvature estimate',
    },
    {
        'key': 'calculus_to_dynamics',
        'source_domain': 'calculus_change',
        'target_domain': 'dynamics_systems',
        'bridge_principle': 'rates compose into state transition laws',
        'transfer_question': 'Can finite differences explain future state transitions across trajectories?',
        'falsifier': 'held-out trajectories keep residual structure after the rate law is applied',
    },
    {
        'key': 'probability_to_information',
        'source_domain': 'probability_uncertainty',
        'target_domain': 'information_computation',
        'bridge_principle': 'uncertainty becomes description length and inference cost',
        'transfer_question': 'Does the lower-uncertainty model also compress future observations better?',
        'falsifier': 'a probabilistically better model has worse held-out description cost',
    },
    {
        'key': 'logic_to_all_domains',
        'source_domain': 'logic_proof',
        'target_domain': 'algebra_equations',
        'bridge_principle': 'proof obligations and counterexamples govern every promoted rule',
        'transfer_question': 'Does the rule name a domain, a falsifier, and a counterexample search?',
        'falsifier': 'the claim cannot produce a setting that would break it',
    },
    {
        'key': 'discrete_to_algebra',
        'source_domain': 'discrete_structures',
        'target_domain': 'algebra_equations',
        'bridge_principle': 'path and state composition behave like algebraic operations',
        'transfer_question': 'Does composing paths match composing symbolic transformations?',
        'falsifier': 'path composition is order-sensitive but the algebraic rule treats it as commutative',
    },
    {
        'key': 'symmetry_to_geometry',
        'source_domain': 'symmetry_invariance',
        'target_domain': 'geometry_space',
        'bridge_principle': 'geometric facts are promoted only if they survive allowed transformations',
        'transfer_question': 'Does the geometric law survive rotation, reflection, translation, or relabeling?',
        'falsifier': 'the law succeeds only in the original coordinate frame',
    },
    {
        'key': 'optimization_to_calculus',
        'source_domain': 'optimization_extrema',
        'target_domain': 'calculus_change',
        'bridge_principle': 'extrema are found by comparing local change directions under constraints',
        'transfer_question': 'Does the local change signal identify a better or worse candidate?',
        'falsifier': 'following the predicted improvement direction raises error on a holdout',
    },
    {
        'key': 'dynamics_to_probability',
        'source_domain': 'dynamics_systems',
        'target_domain': 'probability_uncertainty',
        'bridge_principle': 'unexplained deterministic residuals become noise models only after structure fails',
        'transfer_question': 'Is the residual random after known dynamics, or still structured?',
        'falsifier': 'phase, position, or hidden-state features predict the residual better than noise',
    },
    {
        'key': 'information_to_logic',
        'source_domain': 'information_computation',
        'target_domain': 'logic_proof',
        'bridge_principle': 'short programs need proof obligations before they become laws',
        'transfer_question': 'Does the shortest discovered procedure generalize by induction?',
        'falsifier': 'a larger held-out input breaks the compressed procedure',
    },
    {
        'key': 'higher_dimensions_to_all_domains',
        'source_domain': 'higher_dimensions',
        'target_domain': 'dynamics_systems',
        'bridge_principle': 'new latent axes are allowed only when they improve transfer or falsification',
        'transfer_question': 'Does a lifted coordinate explain residuals across domains without overfitting?',
        'falsifier': 'the latent axis helps one run but fails same-structure holdouts',
    },
]


@dataclass
class ConceptProposal:
    """A candidate internal concept generated from a theory."""
    key: str
    concept_kind: str
    basis: str
    expression_seed: str
    usefulness: float
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'key': self.key,
            'concept_kind': self.concept_kind,
            'basis': self.basis,
            'expression_seed': self.expression_seed,
            'usefulness': round(self.usefulness, 3),
            'parameters': _rounded_dict(self.parameters),
        }


@dataclass
class OperatorProposal:
    """A generated operator candidate built from proposed concepts."""
    key: str
    operator_kind: str
    inputs: list[str]
    expression: str
    generated_from: str
    usefulness: float
    test_hint: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'key': self.key,
            'operator_kind': self.operator_kind,
            'inputs': list(self.inputs),
            'expression': self.expression,
            'generated_from': self.generated_from,
            'usefulness': round(self.usefulness, 3),
            'test_hint': self.test_hint,
            'parameters': _rounded_dict(self.parameters),
        }


@dataclass
class ProofCheck:
    """A lightweight proof-like check for a generated theory."""
    key: str
    check_kind: str
    status: str
    statement: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'key': self.key,
            'check_kind': self.check_kind,
            'status': self.status,
            'statement': self.statement,
            'evidence': _rounded_dict(self.evidence),
        }


@dataclass
class TheoryRecord:
    """One explanation in the agent's theory ledger."""
    key: str
    theory_kind: str
    source_equation_key: str
    claim: str
    explains: list[str]
    failures: list[str]
    score: float
    uncertainty: float
    concept_keys: list[str]
    next_experiment: str
    status: str
    target: str = ''
    expression: str = ''
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'key': self.key,
            'theory_kind': self.theory_kind,
            'source_equation_key': self.source_equation_key,
            'claim': self.claim,
            'explains': list(self.explains),
            'failures': list(self.failures),
            'score': round(self.score, 3),
            'uncertainty': round(self.uncertainty, 3),
            'concept_keys': list(self.concept_keys),
            'next_experiment': self.next_experiment,
            'status': self.status,
            'target': self.target,
            'expression': self.expression,
            'parameters': _rounded_dict(self.parameters),
        }


@dataclass
class SelfAuthoredEquation:
    """A generalized equation synthesized from repeated theory evidence."""
    key: str
    equation_kind: str
    target: str
    expression: str
    support_count: int
    support_contexts: list[str]
    support_seeds: list[int]
    confidence: float
    status: str
    dominant_parameters: dict[str, Any]
    variant_expressions: list[str]
    approximation_notes: list[str]
    proof_obligations: list[str]
    falsification_tests: list[str]
    generated_from: list[str]

    def to_dict(self) -> dict:
        return {
            'key': self.key,
            'equation_kind': self.equation_kind,
            'target': self.target,
            'expression': self.expression,
            'support_count': self.support_count,
            'support_contexts': list(self.support_contexts),
            'support_seeds': list(self.support_seeds),
            'confidence': round(self.confidence, 3),
            'status': self.status,
            'dominant_parameters': _rounded_dict(self.dominant_parameters),
            'variant_expressions': list(self.variant_expressions),
            'approximation_notes': list(self.approximation_notes),
            'proof_obligations': list(self.proof_obligations),
            'falsification_tests': list(self.falsification_tests),
            'generated_from': list(self.generated_from),
        }


@dataclass
class DiscoveryProbePlan:
    """A probe selected because it can sharpen or falsify theories."""
    action: dict
    theory_keys: list[str]
    reason: str
    expected_contrast: str
    source: str = 'discovery_loop'
    disagreement_signature: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'source': self.source,
            'action': _rounded_dict(self.action),
            'theory_keys': list(self.theory_keys),
            'reason': self.reason,
            'expected_contrast': self.expected_contrast,
            'disagreement_signature': _rounded_dict(self.disagreement_signature),
        }


@dataclass
class DiscoveryCycleReport:
    """A snapshot of one residual-to-theory-to-probe cycle."""
    step: int
    theories: list[TheoryRecord]
    concept_proposals: list[ConceptProposal]
    operator_proposals: list[OperatorProposal]
    proof_checks: list[ProofCheck]
    probe_plan: DiscoveryProbePlan | None
    open_questions: list[str]

    @property
    def phase(self) -> str:
        if self.probe_plan is not None:
            return 'probe_ready'
        if self.theories:
            return 'theory_forming'
        return 'collect_more_observations'

    def to_dict(self) -> dict:
        return {
            'step': self.step,
            'phase': self.phase,
            'theories': [theory.to_dict() for theory in self.theories],
            'concept_proposals': [
                proposal.to_dict() for proposal in self.concept_proposals
            ],
            'operator_proposals': [
                proposal.to_dict() for proposal in self.operator_proposals
            ],
            'proof_checks': [check.to_dict() for check in self.proof_checks],
            'probe_plan': self.probe_plan.to_dict() if self.probe_plan else None,
            'open_questions': list(self.open_questions),
        }


@dataclass
class TheoryFamily:
    """A cross-world family of similar theories."""
    theory_kind: str
    support_count: int = 0
    contexts: set[str] = field(default_factory=set)
    operator_kinds: set[str] = field(default_factory=set)
    concept_kinds: set[str] = field(default_factory=set)
    best_score: float = 0.0
    total_score: float = 0.0
    proof_passes: int = 0
    proof_checks: int = 0
    examples: list[dict[str, Any]] = field(default_factory=list)
    planned_outcomes: list[dict[str, Any]] = field(default_factory=list)

    @property
    def mean_score(self) -> float:
        if self.support_count == 0:
            return 0.0
        return self.total_score / self.support_count

    @property
    def proof_rate(self) -> float:
        if self.proof_checks == 0:
            return 0.0
        return self.proof_passes / self.proof_checks

    @property
    def counterexample_count(self) -> int:
        return sum(
            1 for outcome in self.planned_outcomes
            if outcome.get('outcome') in {
                'counterexample_found',
                'transfer_absent',
                'transfer_weak',
                'replication_failed',
                'rival_confirmed',
            }
        )

    @property
    def transfer_success_count(self) -> int:
        return sum(
            1 for outcome in self.planned_outcomes
            if outcome.get('outcome') == 'transfer_confirmed'
        )

    @property
    def transfer_failure_count(self) -> int:
        return sum(
            1 for outcome in self.planned_outcomes
            if outcome.get('outcome') in {'transfer_absent', 'transfer_weak'}
        )

    @property
    def success_contexts(self) -> set[str]:
        contexts = set(self.contexts)
        for outcome in self.planned_outcomes:
            if outcome.get('outcome') in {
                'transfer_confirmed',
                'counterexample_not_found',
                'holdout_survived',
                'replication_confirmed',
                'evidence_confirmed',
                'target_confirmed',
            }:
                context = outcome.get('target_scope') or outcome.get('context')
                if context:
                    contexts.add(str(context))
        return contexts

    @property
    def failure_contexts(self) -> set[str]:
        contexts = set()
        for outcome in self.planned_outcomes:
            if outcome.get('outcome') in {
                'counterexample_found',
                'transfer_absent',
                'transfer_weak',
                'replication_failed',
                'rival_confirmed',
            }:
                context = outcome.get('target_scope') or outcome.get('context')
                if context:
                    contexts.add(str(context))
        return contexts

    @property
    def domain_hypothesis(self) -> dict[str, Any]:
        success = sorted(self.success_contexts)
        failure = sorted(self.failure_contexts)
        if failure:
            claim = (
                f'{self.theory_kind} is domain-limited: keep it for '
                f'{", ".join(success) if success else "known successes"} '
                f'and exclude {", ".join(failure)} until a narrower condition is found'
            )
        elif success:
            claim = (
                f'{self.theory_kind} currently transfers across '
                f'{", ".join(success)}'
            )
        else:
            claim = f'{self.theory_kind} has no tested domain yet'
        return {
            'claim': claim,
            'included_contexts': success,
            'excluded_contexts': failure,
            'revision_needed': bool(failure),
            'next_test': self.next_proof_obligation,
        }

    @property
    def generalization_score(self) -> float:
        context_bonus = min(1.0, len(self.contexts) / 3.0)
        operator_bonus = min(1.0, len(self.operator_kinds) / 2.0)
        return max(0.0, min(
            1.0,
            0.45 * self.mean_score
            + 0.25 * self.proof_rate
            + 0.20 * context_bonus
            + 0.10 * operator_bonus,
        ))

    @property
    def generalization_status(self) -> str:
        if self.counterexample_count > 0:
            return 'domain_limited'
        if self.support_count < 2:
            return 'provisional'
        if self.proof_rate < 0.75:
            return 'needs_counterexample'
        if len(self.contexts) < 2:
            return 'local'
        if (
            self.support_count >= 3
            and self.proof_rate >= 0.9
            and self.generalization_score >= 0.78
        ):
            return 'established'
        return 'reusable'

    @property
    def next_proof_obligation(self) -> str:
        if self.counterexample_count > 0:
            return 'revise the family domain and test the narrower theory against known success and failure contexts'
        if self.support_count < 2:
            return 'find a second independent occurrence before treating this as a family'
        if self.proof_rate < 0.75:
            return 'run a disagreement probe designed to make this family fail against a rival'
        if len(self.contexts) < 2:
            return 'test transfer in a different world context before calling it reusable'
        if self.support_count < 3:
            return 'repeat on another seed or hidden holdout before treating it as established'
        return 'seek a hidden-holdout counterexample that should break the family if it is overgeneralized'

    @property
    def proof_evidence(self) -> dict[str, Any]:
        return {
            'support_count': self.support_count,
            'context_count': len(self.contexts),
            'proof_passes': self.proof_passes,
            'proof_checks': self.proof_checks,
            'proof_rate': round(self.proof_rate, 3),
            'mean_score': round(self.mean_score, 3),
            'operator_count': len(self.operator_kinds),
            'planned_outcome_count': len(self.planned_outcomes),
            'counterexample_count': self.counterexample_count,
            'transfer_success_count': self.transfer_success_count,
            'transfer_failure_count': self.transfer_failure_count,
        }

    @property
    def proof_certificate(self) -> dict[str, Any]:
        status = self.generalization_status
        contexts = sorted(self.contexts)
        domain = self.domain_hypothesis
        accepted_because = []
        not_universal_because = []

        if self.support_count >= 2:
            accepted_because.append(
                f'observed in {self.support_count} independent supporting runs'
            )
        if len(contexts) >= 2:
            accepted_because.append(
                f'transfers across recorded contexts: {", ".join(contexts)}'
            )
        if self.proof_checks > 0 and self.proof_rate >= 0.75:
            accepted_because.append(
                f'proof-like checks pass at {self.proof_rate:.2f}'
            )
        if self.operator_kinds:
            accepted_because.append(
                f'reuses generated operators: {", ".join(sorted(self.operator_kinds))}'
            )
        if self.transfer_success_count > 0:
            accepted_because.append(
                f'planned transfer confirmed {self.transfer_success_count} time(s)'
            )
        if status == 'established':
            accepted_because.append(
                'meets established support, proof-rate, and generalization thresholds'
            )
        if not accepted_because:
            accepted_because.append(
                'recorded as a candidate family, but accepted only provisionally'
            )

        if self.support_count < 2:
            not_universal_because.append(
                'only one supporting occurrence has been recorded'
            )
        if self.proof_checks == 0:
            not_universal_because.append(
                'no proof-like checks have been attached yet'
            )
        elif self.proof_rate < 0.75:
            not_universal_because.append(
                f'proof-like checks pass at only {self.proof_rate:.2f}'
            )
        if len(contexts) < 2:
            not_universal_because.append(
                'not yet recovered in a different recorded world context'
            )
        if self.support_count < 3 and status in {'reusable', 'established'}:
            not_universal_because.append(
                'needs another seed or hidden holdout before being treated as broad'
            )
        if self.failure_contexts:
            not_universal_because.append(
                'counterexample or failed-transfer contexts recorded: '
                + ', '.join(sorted(self.failure_contexts))
            )
        elif status in {'reusable', 'established'}:
            not_universal_because.append(
                'hidden holdouts can still narrow the claim'
            )

        if status == 'domain_limited':
            would_break_if = (
                'the narrowed domain predicts one of its excluded contexts '
                'or loses a known success context'
            )
        elif status == 'needs_counterexample':
            would_break_if = (
                'a rival explains targeted disagreement samples while this '
                'family keeps failing proof checks'
            )
        elif status == 'local':
            would_break_if = (
                'the family only reappears in the original context and fails '
                'a transfer context'
            )
        elif status == 'provisional':
            would_break_if = (
                'an independent repeat fails to recover the same family'
            )
        else:
            would_break_if = (
                'a hidden holdout or new seed collapses the score, proof rate, '
                'or required operator family'
            )

        return {
            'theory_kind': self.theory_kind,
            'status': status,
            'claim': domain['claim'],
            'support': self.proof_evidence,
            'contexts': contexts,
            'accepted_because': accepted_because,
            'not_universal_because': not_universal_because,
            'would_break_if': would_break_if,
            'next_obligation': self.next_proof_obligation,
            'domain_hypothesis': domain,
            'examples': list(self.examples),
            'recent_outcomes': list(self.planned_outcomes[-3:]),
        }

    def evaluation(self) -> dict[str, Any]:
        return {
            'theory_kind': self.theory_kind,
            'status': self.generalization_status,
            'generalization_score': round(self.generalization_score, 3),
            'proof_evidence': self.proof_evidence,
            'proof_certificate': self.proof_certificate,
            'next_obligation': self.next_proof_obligation,
            'contexts': sorted(self.contexts),
            'domain_hypothesis': self.domain_hypothesis,
        }

    def experiment_recommendation(self) -> dict[str, Any]:
        status = self.generalization_status
        if status == 'provisional':
            experiment_kind = 'replication_seed'
            target_context = 'same_or_similar_context'
            expected_result = 'a second independent run should recover the same theory family'
            falsifies_if = 'the family disappears or drops below proof thresholds on repeat'
            priority_base = 0.45
        elif status == 'needs_counterexample':
            experiment_kind = 'disagreement_counterexample'
            target_context = 'rival_or_hidden_context'
            expected_result = 'a targeted disagreement probe should either restore proof rate or expose a failure mode'
            falsifies_if = 'rival theories explain the new samples while this family keeps failing proof checks'
            priority_base = 1.0
        elif status == 'domain_limited':
            experiment_kind = 'domain_refinement'
            target_context = 'known_success_and_failure_contexts'
            expected_result = 'a narrowed family should keep successes while excluding recorded counterexamples'
            falsifies_if = 'the revised domain still predicts a context where the family was already contradicted'
            priority_base = 0.88
        elif status == 'local':
            experiment_kind = 'transfer_test'
            target_context = 'unseen_world_context'
            expected_result = 'the family should reappear with passing proof checks outside its original context'
            falsifies_if = 'the family only fits repeated runs from the same context'
            priority_base = 0.9
        elif status == 'reusable':
            experiment_kind = 'replication_or_holdout'
            target_context = 'new_seed_or_hidden_holdout'
            expected_result = 'the family should repeat without losing proof rate or requiring special-case operators'
            falsifies_if = 'support grows but proof rate or held-out score collapses'
            priority_base = 0.72
        else:
            experiment_kind = 'hidden_holdout_counterexample'
            target_context = 'hidden_holdout'
            expected_result = 'the family should survive a blind holdout, or fail in a way that sharpens its domain'
            falsifies_if = 'a hidden holdout shows the family was overgeneralized from its known contexts'
            priority_base = 0.62
        priority = min(
            1.0,
            priority_base
            + 0.12 * (1.0 - self.proof_rate)
            + 0.06 * min(1.0, self.support_count / 5.0),
        )
        return {
            'theory_kind': self.theory_kind,
            'experiment_kind': experiment_kind,
            'priority': round(priority, 3),
            'family_status': status,
            'target_context': target_context,
            'avoid_contexts': sorted(self.contexts),
            'reason': self.next_proof_obligation,
            'expected_result': expected_result,
            'falsifies_if': falsifies_if,
            'proof_evidence': self.proof_evidence,
            'proof_certificate': self.proof_certificate,
            'domain_hypothesis': self.domain_hypothesis,
            'suggested_campaign': {
                'command_family': 'equation_campaign',
                'world_selection': target_context,
                'enable_equation_probes': True,
            },
        }

    def update(
        self,
        context: str,
        seed: int,
        theory: dict,
        operator_proposals: list[dict],
        concept_proposals: list[dict],
        proof_checks: list[dict],
    ):
        score = float(theory.get('score', 0.0) or 0.0)
        self.support_count += 1
        self.contexts.add(context)
        self.best_score = max(self.best_score, score)
        self.total_score += score
        self.operator_kinds.update(
            str(item.get('operator_kind'))
            for item in operator_proposals
            if item.get('operator_kind')
        )
        self.concept_kinds.update(
            str(item.get('concept_kind'))
            for item in concept_proposals
            if item.get('concept_kind')
        )
        self.proof_checks += len(proof_checks)
        self.proof_passes += sum(1 for item in proof_checks if item.get('status') == 'passed')
        if len(self.examples) < 5:
            self.examples.append({
                'context': context,
                'seed': seed,
                'theory_kind': theory.get('theory_kind'),
                'source_equation_key': theory.get('source_equation_key'),
                'target': theory.get('target') or _target_from_claim(str(theory.get('claim', ''))),
                'expression': theory.get('expression'),
                'parameters': _rounded_dict(dict(theory.get('parameters') or {})),
                'claim': theory.get('claim'),
                'score': round(score, 3),
            })

    def record_planned_outcome(self, outcome: dict[str, Any]):
        self.planned_outcomes.append(dict(outcome))
        if len(self.planned_outcomes) > 8:
            self.planned_outcomes = self.planned_outcomes[-8:]

    def to_dict(self) -> dict:
        return {
            'theory_kind': self.theory_kind,
            'support_count': self.support_count,
            'contexts': sorted(self.contexts),
            'operator_kinds': sorted(self.operator_kinds),
            'concept_kinds': sorted(self.concept_kinds),
            'best_score': round(self.best_score, 3),
            'mean_score': round(self.mean_score, 3),
            'proof_rate': round(self.proof_rate, 3),
            'generalization_score': round(self.generalization_score, 3),
            'generalization_status': self.generalization_status,
            'next_proof_obligation': self.next_proof_obligation,
            'proof_evidence': self.proof_evidence,
            'proof_certificate': self.proof_certificate,
            'domain_hypothesis': self.domain_hypothesis,
            'evaluation': self.evaluation(),
            'experiment_recommendation': self.experiment_recommendation(),
            'planned_outcomes': list(self.planned_outcomes),
            'examples': list(self.examples),
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'TheoryFamily':
        support_count = int(data.get('support_count', 0) or 0)
        proof_evidence = dict(data.get('proof_evidence') or {})
        mean_score = float(
            data.get('mean_score', proof_evidence.get('mean_score', 0.0)) or 0.0
        )
        family = cls(
            theory_kind=str(data.get('theory_kind', 'unknown')),
            support_count=support_count,
            contexts=set(data.get('contexts', [])),
            operator_kinds=set(data.get('operator_kinds', [])),
            concept_kinds=set(data.get('concept_kinds', [])),
            best_score=float(data.get('best_score', 0.0) or 0.0),
            total_score=mean_score * support_count,
            proof_passes=int(proof_evidence.get('proof_passes', 0) or 0),
            proof_checks=int(proof_evidence.get('proof_checks', 0) or 0),
            examples=list(data.get('examples', [])),
            planned_outcomes=list(data.get('planned_outcomes', [])),
        )
        return family


class CumulativeTheoryMemory:
    """Accumulates discovery-loop reports into reusable theory families."""

    def __init__(self):
        self.records: list[dict] = []
        self.families: dict[str, TheoryFamily] = {}
        self.planned_outcomes: list[dict] = []
        self.disagreement_records: list[dict] = []
        self.operator_prior_outcomes: list[dict] = []
        self.equation_case_records: list[dict] = []
        self.domain_world_records: list[dict] = []
        self.autonomous_scientist_records: list[dict] = []
        self.arithmetic_rediscovery_records: list[dict] = []
        self.compressed_experience_shards: list[dict] = []
        self.canonical_law_shards: list[dict] = []

    def record_result(self, context: str, seed: int, report) -> dict:
        report_dict = self._report_dict(report)
        theories = list(report_dict.get('theories', []))
        operator_proposals = list(report_dict.get('operator_proposals', []))
        concept_proposals = list(report_dict.get('concept_proposals', []))
        proof_checks = list(report_dict.get('proof_checks', []))
        probe_plan = report_dict.get('probe_plan') or {}
        disagreement_record = self._disagreement_record(
            context=context,
            seed=seed,
            probe_plan=probe_plan,
        )
        record = {
            'context': context,
            'seed': seed,
            'phase': report_dict.get('phase', 'unknown'),
            'theory_count': len(theories),
            'operator_count': len(operator_proposals),
            'proof_check_count': len(proof_checks),
            'disagreement_mode': (
                disagreement_record.get('mode') if disagreement_record else None
            ),
        }
        self.records.append(record)
        if disagreement_record:
            self.disagreement_records.append(disagreement_record)
            if len(self.disagreement_records) > 20:
                self.disagreement_records = self.disagreement_records[-20:]
        for theory in theories:
            kind = str(theory.get('theory_kind', 'unknown'))
            if kind == 'simple_transition':
                continue
            family_kind = self._family_key(kind)
            family = self.families.setdefault(
                family_kind,
                TheoryFamily(theory_kind=family_kind),
            )
            relevant_concepts = self._relevant_concepts(theory, concept_proposals)
            relevant_operators = self._relevant_operators(theory, operator_proposals)
            relevant_checks = self._relevant_checks(theory, proof_checks)
            family.update(
                context=context,
                seed=seed,
                theory=theory,
                operator_proposals=relevant_operators,
                concept_proposals=relevant_concepts,
                proof_checks=relevant_checks,
            )
        return record

    def evaluate_planned_result(
        self,
        plan: dict,
        context: str,
        seed: int,
        report,
        operator_prior_records: list[dict[str, Any]] | None = None,
    ) -> dict:
        report_dict = self._report_dict(report)
        target_family = str(plan.get('theory_kind', 'unknown'))
        matching_theories = [
            theory for theory in report_dict.get('theories', [])
            if self._matches_plan_target(plan, theory)
        ]
        rival_theories = [
            theory for theory in report_dict.get('theories', [])
            if self._matches_plan_rival(plan, theory)
        ]
        proof_checks = list(report_dict.get('proof_checks', []))
        matching_check_count = 0
        matching_pass_count = 0
        for theory in matching_theories:
            checks = self._relevant_checks(theory, proof_checks)
            matching_check_count += len(checks)
            matching_pass_count += sum(1 for check in checks if check.get('status') == 'passed')
        rival_check_count = 0
        rival_pass_count = 0
        for theory in rival_theories:
            checks = self._relevant_checks(theory, proof_checks)
            rival_check_count += len(checks)
            rival_pass_count += sum(1 for check in checks if check.get('status') == 'passed')
        found_family = bool(matching_theories)
        proof_passed = matching_pass_count > 0 or (
            found_family and matching_check_count == 0
        )
        rival_found = bool(rival_theories)
        rival_proof_passed = rival_pass_count > 0 or (
            rival_found and rival_check_count == 0
        )
        previous_family = self.families.get(target_family)
        previous_contexts = set(previous_family.contexts) if previous_family else set()
        new_context = context not in previous_contexts
        experiment_kind = str(plan.get('experiment_kind', 'unknown'))
        operator_prior_evidence: dict[str, Any] = {}
        if experiment_kind in {
            'operator_prior_refinement_validation',
            'operator_prior_domain_repair',
            'operator_prior_hidden_holdout_counterexample',
            'operator_prior_domain_predicate_validation',
        }:
            operator_prior_evidence = self._operator_prior_plan_evidence(
                plan,
                operator_prior_records,
            )
            outcome = operator_prior_evidence['outcome']
        elif experiment_kind == 'model_disagreement_probe':
            outcome = self._model_disagreement_outcome_label(
                target_found=found_family,
                target_proof_passed=proof_passed,
                rival_found=rival_found,
                rival_proof_passed=rival_proof_passed,
            )
        elif experiment_kind == 'equation_invariant_exponent_resolution':
            outcome = self._equation_invariant_resolution_outcome_label(
                target_found=found_family,
                target_proof_passed=proof_passed,
                rival_found=rival_found,
                rival_proof_passed=rival_proof_passed,
            )
        else:
            outcome = self._planned_outcome_label(
                experiment_kind=experiment_kind,
                found_family=found_family,
                proof_passed=proof_passed,
                new_context=new_context,
            )
        best_score = max(
            (float(theory.get('score', 0.0) or 0.0) for theory in matching_theories),
            default=0.0,
        )
        rival_best_score = max(
            (float(theory.get('score', 0.0) or 0.0) for theory in rival_theories),
            default=0.0,
        )
        return {
            'theory_kind': target_family,
            'experiment_kind': experiment_kind,
            'outcome': outcome,
            'context': context,
            'seed': seed,
            'target_scope': self._plan_target_scope(plan, context),
            'rival_scope': self._plan_rival_scope(plan, context),
            'disagreement_mode': plan.get('disagreement_signature', {}).get('mode'),
            'invariant_key': plan.get('invariant_key'),
            'primary_theory_label': plan.get('primary_theory_label'),
            'rival_theory_labels': list(plan.get('rival_theory_labels') or []),
            'found_family': found_family,
            'proof_passed': proof_passed,
            'rival_found': rival_found,
            'rival_proof_passed': rival_proof_passed,
            'new_context': new_context,
            'matching_theory_count': len(matching_theories),
            'matching_proof_passes': matching_pass_count,
            'matching_proof_checks': matching_check_count,
            'rival_theory_count': len(rival_theories),
            'rival_proof_passes': rival_pass_count,
            'rival_proof_checks': rival_check_count,
            'best_score': round(best_score, 3),
            'rival_best_score': round(rival_best_score, 3),
            'expected_result': plan.get('expected_result'),
            'falsifies_if': plan.get('falsifies_if'),
            **operator_prior_evidence,
        }

    def record_planned_result(
        self,
        plan: dict,
        context: str,
        seed: int,
        report,
        operator_prior_result: dict[str, Any] | None = None,
    ) -> dict:
        operator_prior_records = (
            self._operator_prior_records_from_result(context, seed, operator_prior_result)
            if operator_prior_result is not None
            else None
        )
        outcome = self.evaluate_planned_result(
            plan,
            context,
            seed,
            report,
            operator_prior_records=operator_prior_records,
        )
        record = self.record_result(context, seed, report)
        if operator_prior_records:
            self._append_operator_prior_result_records(operator_prior_records)
        record['planned_experiment_outcome'] = outcome
        self.planned_outcomes.append(outcome)
        target_family = self.families.get(str(outcome.get('theory_kind', 'unknown')))
        if target_family is not None:
            target_family.record_planned_outcome(outcome)
        return outcome

    def record_operator_prior_results(
        self,
        context: str,
        seed: int,
        result: dict[str, Any],
    ) -> list[dict[str, Any]]:
        records = self._operator_prior_records_from_result(context, seed, result)
        if not records:
            return []
        self._append_operator_prior_result_records(records)
        return records

    def record_equation_case_result(
        self,
        context: str,
        seed: int,
        result: dict[str, Any],
        phase: str = 'equation_case',
    ) -> dict[str, Any]:
        """Keep the headline equation from a run for cross-seed consolidation."""
        equation = dict(result.get('interesting_equation') or {})
        if not equation:
            return {}
        expression = str(equation.get('expression') or '')
        if not expression:
            return {}
        record = {
            'context': context,
            'seed': int(seed),
            'phase': phase,
            'target': equation.get('target'),
            'expression': expression,
            'role': equation.get('role'),
            'score': round(float(equation.get('score', 0.0) or 0.0), 3),
            'parameters': _rounded_dict(dict(equation.get('parameters') or {})),
            'passed': bool(result.get('passed') or result.get('equation_passed')),
            'label_leak_count': len(result.get('label_leaks') or []),
            'probe_suggestion_count': len(result.get('probe_suggestions') or []),
        }
        self.equation_case_records.append(record)
        if len(self.equation_case_records) > 160:
            self.equation_case_records = self.equation_case_records[-160:]
        return record

    def _operator_prior_records_from_result(
        self,
        context: str,
        seed: int,
        result: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        records = []
        if not result:
            return records
        for item in list(result.get('operator_prior_results') or []):
            operator_key = item.get('operator_key')
            if not operator_key:
                continue
            record = {
                'context': context,
                'seed': seed,
                'operator_key': operator_key,
                'operator_kind': item.get('operator_kind'),
                'outcome': item.get('outcome', 'unknown'),
                'best_score': round(float(item.get('best_score', 0.0) or 0.0), 3),
                'matching_equation_count': int(item.get('matching_equation_count', 0) or 0),
                'parameters': _rounded_dict(dict(item.get('parameters') or {})),
                'best_equation': _rounded_dict(dict(item.get('best_equation') or {})),
            }
            best_parameters = dict(record['best_equation'].get('parameters') or {})
            if best_parameters:
                record['refined_parameters'] = _rounded_dict({
                    **dict(record['parameters']),
                    **best_parameters,
                    })
            records.append(record)
        return records

    def _append_operator_prior_result_records(
        self,
        records: list[dict[str, Any]],
    ):
        self.operator_prior_outcomes.extend(records)
        if len(self.operator_prior_outcomes) > 80:
            self.operator_prior_outcomes = self.operator_prior_outcomes[-80:]
        if self.records:
            self.records[-1]['operator_prior_result_count'] = len(records)
            self.records[-1]['operator_prior_confirmed_count'] = sum(
                1 for record in records
                if record.get('outcome') == 'confirmed'
            )

    def record_domain_world_discoveries(
        self,
        seed: int = 0,
        variant: int = 0,
    ) -> list[dict[str, Any]]:
        """Persist generated domain-world discoveries as curriculum evidence."""
        reports = self.domain_world_discovery_reports(
            limit=len(MATH_DOMAIN_CURRICULUM),
            seed=seed,
            variant=variant,
        )
        records = []
        for report in reports:
            record = {
                'phase': 'domain_world_discovery',
                'domain_key': report.get('domain_key'),
                'seed': int(report.get('seed', seed) or seed),
                'variant': int(report.get('variant', variant) or variant),
                'candidate_count': int(report.get('candidate_count', 0) or 0),
                'benchmark_coverage': float(report.get('benchmark_coverage', 0.0) or 0.0),
                'comparison_hits': list(report.get('comparison_hits') or []),
                'missing_comparison_tags': list(report.get('missing_comparison_tags') or []),
                'falsification_test_count': int(
                    report.get('falsification_test_count', 0) or 0
                ),
                'transfer_basis': list(report.get('transfer_basis') or []),
                'self_authored_equations': list(
                    report.get('self_authored_equations') or []
                )[:5],
                'leaked_manifest': bool(report.get('leaked_manifest')),
            }
            records.append(record)

        seen = {
            (
                item.get('domain_key'),
                int(item.get('seed', 0) or 0),
                int(item.get('variant', 0) or 0),
            )
            for item in self.domain_world_records
        }
        for record in records:
            key = (
                record.get('domain_key'),
                int(record.get('seed', 0) or 0),
                int(record.get('variant', 0) or 0),
            )
            if key in seen:
                continue
            self.domain_world_records.append(record)
            seen.add(key)
        if len(self.domain_world_records) > 96:
            self.domain_world_records = self.domain_world_records[-96:]
        return records

    def record_autonomous_scientist_loop(
        self,
        seed_start: int = 0,
        seed_count: int = 3,
        variants: list[int] | tuple[int, ...] = (0,),
        event_limit: int = 80,
    ) -> dict[str, Any]:
        """Persist a non-final scientist pass over repeated domain worlds."""
        try:
            from agent.autonomous_scientist import run_domain_scientist_cycle
        except ImportError:  # pragma: no cover - package import fallback
            from first_principles_ai.agent.autonomous_scientist import (
                run_domain_scientist_cycle,
            )

        report = run_domain_scientist_cycle(
            seed_start=seed_start,
            seed_count=seed_count,
            variants=tuple(variants or (0,)),
            event_limit=event_limit,
        )
        record = {
            'phase': 'autonomous_scientist_loop',
            'seed_start': int(seed_start),
            'seed_count': int(seed_count),
            'variants': [int(variant) for variant in (variants or (0,))],
            'coverage': dict(report.get('coverage') or {}),
            'status': report.get('status'),
            'invariant_consolidations': list(
                report.get('invariant_consolidations') or []
            )[:24],
            'residual_experiments': list(report.get('residual_experiments') or [])[:24],
            'harder_stress_worlds': list(report.get('harder_stress_worlds') or [])[:12],
            'authored_equation_extensions': list(
                report.get('authored_equation_extensions') or []
            )[:36],
            'live_events': list(report.get('live_events') or [])[:event_limit],
            'next_actions': list(report.get('next_actions') or []),
        }
        key = (
            record['seed_start'],
            record['seed_count'],
            tuple(record['variants']),
        )
        existing = {
            (
                int(item.get('seed_start', 0) or 0),
                int(item.get('seed_count', 0) or 0),
                tuple(int(variant) for variant in item.get('variants', [])),
            )
            for item in self.autonomous_scientist_records
        }
        if key not in existing:
            self.autonomous_scientist_records.append(record)
        if len(self.autonomous_scientist_records) > 24:
            self.autonomous_scientist_records = self.autonomous_scientist_records[-24:]
        return report

    def record_arithmetic_rediscovery(
        self,
        seed_start: int = 0,
        seed_count: int = 2,
        variants: list[int] | tuple[int, ...] = (0,),
    ) -> dict[str, Any]:
        """Persist an observation-only counting/arithmetic rediscovery probe."""
        try:
            from agent.arithmetic_rediscovery import run_arithmetic_rediscovery_probe
        except ImportError:  # pragma: no cover - package import fallback
            from first_principles_ai.agent.arithmetic_rediscovery import (
                run_arithmetic_rediscovery_probe,
            )

        report = run_arithmetic_rediscovery_probe(
            seed_start=seed_start,
            seed_count=seed_count,
            variants=tuple(variants or (0,)),
        )
        record = {
            'phase': 'arithmetic_rediscovery',
            'seed_start': int(seed_start),
            'seed_count': int(seed_count),
            'variants': [int(variant) for variant in (variants or (0,))],
            'status': report.get('status'),
            'coverage': float(report.get('coverage', 0.0) or 0.0),
            'target_count': int(report.get('target_count', 0) or 0),
            'discovered_target_count': int(
                report.get('discovered_target_count', 0) or 0
            ),
            'candidate_count': int(report.get('candidate_count', 0) or 0),
            'observation_count': int(report.get('observation_count', 0) or 0),
            'discovered_targets': list(report.get('discovered_targets') or []),
            'missing_targets': list(report.get('missing_targets') or []),
            'leaked_manifest': bool(report.get('leaked_manifest')),
            'self_authored_equations': list(
                report.get('self_authored_equations') or []
            )[:12],
            'live_events': list(report.get('live_events') or [])[:40],
        }
        key = (
            record['seed_start'],
            record['seed_count'],
            tuple(record['variants']),
        )
        existing = {
            (
                int(item.get('seed_start', 0) or 0),
                int(item.get('seed_count', 0) or 0),
                tuple(int(variant) for variant in item.get('variants', [])),
            )
            for item in self.arithmetic_rediscovery_records
        }
        if key not in existing:
            self.arithmetic_rediscovery_records.append(record)
        if len(self.arithmetic_rediscovery_records) > 24:
            self.arithmetic_rediscovery_records = self.arithmetic_rediscovery_records[-24:]
        return report

    def _operator_prior_plan_evidence(
        self,
        plan: dict,
        operator_prior_records: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        experiment_kind = str(plan.get('experiment_kind', 'unknown'))
        prefix = (
            'operator_prior_repair'
            if experiment_kind == 'operator_prior_domain_repair'
            else 'operator_prior_holdout'
            if experiment_kind == 'operator_prior_hidden_holdout_counterexample'
            else 'operator_prior_domain_predicate'
            if experiment_kind == 'operator_prior_domain_predicate_validation'
            else 'operator_prior_validation'
        )
        if operator_prior_records is None:
            return {
                'outcome': self._planned_outcome_label(
                    experiment_kind,
                    found_family=False,
                    proof_passed=False,
                    new_context=False,
                ),
                'operator_prior_evaluated': False,
                'operator_prior_key': plan.get('operator_prior_key'),
                'operator_prior_result_count': 0,
            }

        operator_key = plan.get('operator_prior_key')
        matches = [
            record for record in operator_prior_records
            if not operator_key or record.get('operator_key') == operator_key
        ]
        if not matches:
            return {
                'outcome': f'{prefix}_failed',
                'operator_prior_evaluated': True,
                'operator_prior_key': operator_key,
                'operator_prior_feedback_outcome': 'missing',
                'operator_prior_result_count': 0,
                'operator_prior_best_score': 0.0,
            }

        best = max(
            matches,
            key=lambda record: (
                float(record.get('best_score', 0.0) or 0.0),
                int(record.get('matching_equation_count', 0) or 0),
            ),
        )
        feedback_outcome = str(best.get('outcome', 'unknown'))
        if experiment_kind == 'operator_prior_domain_predicate_validation':
            domain = dict(plan.get('operator_prior_domain') or {})
            included = set(domain.get('included_contexts') or [])
            excluded = set(domain.get('excluded_contexts') or [])
            context = str(best.get('context') or '')
            if context in excluded or plan.get('failure_context') == context:
                if feedback_outcome in {'weak', 'unmatched'}:
                    outcome = f'{prefix}_confirmed'
                else:
                    outcome = f'{prefix}_failed'
            elif context in included:
                if feedback_outcome == 'confirmed':
                    outcome = f'{prefix}_confirmed'
                elif feedback_outcome == 'weak':
                    outcome = f'{prefix}_weak'
                else:
                    outcome = f'{prefix}_failed'
            elif feedback_outcome == 'confirmed':
                outcome = f'{prefix}_confirmed'
            elif feedback_outcome == 'weak':
                outcome = f'{prefix}_weak'
            else:
                outcome = f'{prefix}_failed'
        elif feedback_outcome == 'confirmed':
            outcome = f'{prefix}_confirmed'
        elif feedback_outcome == 'weak':
            outcome = f'{prefix}_weak'
        else:
            outcome = f'{prefix}_failed'
        refined_parameters = dict(best.get('refined_parameters') or {})
        plan_parameters = dict(plan.get('operator_prior_parameters') or {})
        refinement_detected = bool(
            refined_parameters
            and _rounded_dict(refined_parameters) != _rounded_dict(plan_parameters)
        )
        return {
            'outcome': outcome,
            'operator_prior_evaluated': True,
            'operator_prior_key': best.get('operator_key') or operator_key,
            'operator_prior_kind': best.get('operator_kind'),
            'operator_prior_feedback_outcome': feedback_outcome,
            'operator_prior_result_count': len(matches),
            'operator_prior_best_score': round(
                float(best.get('best_score', 0.0) or 0.0),
                3,
            ),
            'operator_prior_matching_equation_count': int(
                best.get('matching_equation_count', 0) or 0
            ),
            'operator_prior_best_equation': _rounded_dict(
                dict(best.get('best_equation') or {})
            ),
            'operator_prior_refined_parameters': _rounded_dict(refined_parameters),
            'operator_prior_refinement_detected': refinement_detected,
        }

    def reusable_families(self, min_support: int = 2) -> list[dict]:
        families = [
            family.to_dict()
            for family in self.families.values()
            if family.support_count >= min_support
        ]
        families.sort(
            key=lambda item: (
                item['generalization_score'],
                item['support_count'],
                item['best_score'],
            ),
            reverse=True,
        )
        return families

    def family_evaluations(self, min_support: int = 1) -> list[dict]:
        status_rank = {
            'established': 4,
            'reusable': 3,
            'local': 2,
            'domain_limited': 2,
            'needs_counterexample': 1,
            'provisional': 0,
        }
        evaluations = [
            family.evaluation()
            for family in self.families.values()
            if family.support_count >= min_support
        ]
        evaluations.sort(
            key=lambda item: (
                status_rank.get(item['status'], 0),
                item['generalization_score'],
                item['proof_evidence']['support_count'],
            ),
            reverse=True,
        )
        return evaluations

    def self_authored_equations(
        self,
        limit: int = 5,
        min_support: int = 2,
    ) -> list[dict[str, Any]]:
        """Write generalized equation templates from repeated theory evidence."""
        authored = []
        for family in self.families.values():
            equation = self._self_authored_equation_from_family(
                family,
                min_support=min_support,
            )
            if equation is not None:
                authored.append(equation.to_dict())
        authored.sort(
            key=lambda item: (
                item['confidence'],
                item['support_count'],
                item['equation_kind'],
            ),
            reverse=True,
        )
        return authored[:limit]

    def _self_authored_equation_from_family(
        self,
        family: TheoryFamily,
        min_support: int,
    ) -> SelfAuthoredEquation | None:
        if family.support_count < min_support:
            return None
        examples = [dict(example) for example in family.examples]
        if not examples:
            return None
        distinct_occurrences = {
            (str(example.get('context', 'unknown')), example.get('seed'))
            for example in examples
        }
        if len(distinct_occurrences) < min_support:
            return None

        kind = family.theory_kind
        relation = 'perpendicular' if 'perpendicular' in kind else 'direction'
        target = self._dominant_text(
            [
                str(example.get('target') or _target_from_claim(str(example.get('claim', ''))))
                for example in examples
            ],
            fallback='baseline_adjusted_delta_velocity',
        )
        dominant_parameters = self._dominant_equation_parameters(kind, examples)
        expression = self._self_authored_expression(
            kind=kind,
            target=target,
            relation=relation,
            parameters=dominant_parameters,
        )
        if not expression:
            return None

        support_seeds = sorted({
            int(example['seed'])
            for example in examples
            if isinstance(example.get('seed'), int)
        })
        variant_expressions = self._distinct_nonempty(
            str(example.get('expression') or '')
            for example in examples
        )[:5]
        confidence = max(0.0, min(
            1.0,
            0.62 * family.generalization_score
            + 0.28 * family.proof_rate
            + 0.10 * min(1.0, family.support_count / 5.0),
        ))
        if family.generalization_status in {'established', 'reusable'}:
            status = 'candidate_law'
        elif family.support_count >= 2:
            status = 'working_equation'
        else:
            status = 'hypothesis'

        return SelfAuthoredEquation(
            key=self._self_authored_equation_key(kind, dominant_parameters),
            equation_kind=kind,
            target=target,
            expression=expression,
            support_count=len(distinct_occurrences),
            support_contexts=sorted(family.contexts),
            support_seeds=support_seeds,
            confidence=confidence,
            status=status,
            dominant_parameters=dominant_parameters,
            variant_expressions=variant_expressions,
            approximation_notes=self._self_authored_approximation_notes(
                kind,
                dominant_parameters,
                examples,
                family,
            ),
            proof_obligations=self._self_authored_proof_obligations(kind),
            falsification_tests=self._self_authored_falsification_tests(
                kind,
                dominant_parameters,
            ),
            generated_from=self._distinct_nonempty(
                str(example.get('source_equation_key') or '')
                for example in examples
            )[:5],
        )

    def _dominant_equation_parameters(
        self,
        kind: str,
        examples: list[dict[str, Any]],
    ) -> dict[str, Any]:
        parameters: dict[str, Any] = {}
        if any(name in kind for name in ('distance_scaled', 'tapered_distance')):
            exponent = self._dominant_numeric_parameter(examples, 'distance_exponent')
            if exponent is not None:
                parameters['distance_exponent'] = exponent
        if 'cutoff' in kind or 'tapered_distance' in kind:
            cutoff_radius = self._dominant_numeric_parameter(examples, 'cutoff_radius')
            if cutoff_radius is not None:
                parameters['cutoff_radius'] = cutoff_radius
        if kind in {'periodic_residual', 'generated_periodic_residual'}:
            period = self._dominant_numeric_parameter(examples, 'period_steps')
            if period is not None:
                parameters['period_steps'] = period
        centers = [
            dict(example.get('parameters') or {})
            for example in examples
            if isinstance(example.get('parameters'), dict)
        ]
        center_x = self._dominant_numeric_parameter_from_dicts(centers, 'center_x')
        center_y = self._dominant_numeric_parameter_from_dicts(centers, 'center_y')
        if center_x is not None:
            parameters['example_center_x'] = center_x
        if center_y is not None:
            parameters['example_center_y'] = center_y
        return parameters

    def _self_authored_expression(
        self,
        kind: str,
        target: str,
        relation: str,
        parameters: dict[str, Any],
    ) -> str | None:
        vector = (
            'perpendicular(unit(center - position))'
            if relation == 'perpendicular'
            else 'unit(center - position)'
        )
        if kind in {'periodic_residual', 'generated_periodic_residual'}:
            period = _format_number(parameters.get('period_steps', 'T'))
            return (
                f'{target} ~= a*sin(2*pi*step/{period}) '
                f'+ b*cos(2*pi*step/{period})'
            )
        if 'tapered_distance' in kind:
            radius = _format_number(parameters.get('cutoff_radius', 'R'))
            exponent = _format_number(parameters.get('distance_exponent', 'p'))
            return (
                f'{target} ~= k * taper(separation, {radius}) * '
                f'{vector} / separation^{exponent}'
            )
        if 'cutoff' in kind:
            radius = _format_number(parameters.get('cutoff_radius', 'R'))
            return (
                f'{target} ~= k * inside(separation <= {radius}) * {vector}'
            )
        if 'distance_scaled' in kind:
            exponent = _format_number(parameters.get('distance_exponent', 'p'))
            return f'{target} ~= k * {vector} / separation^{exponent}'
        if 'perpendicular' in kind or 'direction' in kind:
            return f'{target} ~= k * {vector}'
        return None

    def _self_authored_approximation_notes(
        self,
        kind: str,
        dominant_parameters: dict[str, Any],
        examples: list[dict[str, Any]],
        family: TheoryFamily,
    ) -> list[str]:
        notes = []
        for name in ('distance_exponent', 'cutoff_radius', 'period_steps'):
            values = self._numeric_parameter_values(examples, name)
            if len(values) > 1:
                dominant = dominant_parameters.get(name)
                notes.append(
                    f'observed {name} variants {", ".join(_format_number(value) for value in values)}; '
                    f'using {_format_number(dominant)} as the current dominant template'
                )
        if family.proof_rate < 0.75:
            notes.append('proof checks are still weak, so treat this as an authored hypothesis')
        if family.support_count < 3:
            notes.append('needs another seed or hidden holdout before being treated as stable')
        if 'distance_scaled' in kind and 'distance_exponent' not in dominant_parameters:
            notes.append('distance exponent is symbolic until near/far residual ratios settle it')
        return notes

    def _self_authored_proof_obligations(self, kind: str) -> list[str]:
        obligations = ['heldout_counterexample', 'dimensional_consistency']
        if 'distance_scaled' in kind or 'tapered_distance' in kind:
            obligations.extend(['domain_nonzero_positive', 'monotonicity_or_extremum'])
        if 'cutoff' in kind or 'tapered_distance' in kind:
            obligations.extend(['boundary_and_partition'])
        if 'perpendicular' in kind:
            obligations.extend(['symmetry_invariance'])
        if kind in {'periodic_residual', 'generated_periodic_residual'}:
            obligations.extend(['induction_or_recurrence', 'symmetry_invariance'])
        return self._distinct_nonempty(obligations)

    def _self_authored_falsification_tests(
        self,
        kind: str,
        parameters: dict[str, Any],
    ) -> list[str]:
        tests = []
        if 'distance_scaled' in kind:
            exponent = _format_number(parameters.get('distance_exponent', 'p'))
            tests.append(
                f'near/far holdouts should preserve residual magnitude ratio for exponent {exponent}'
            )
        if 'cutoff' in kind:
            radius = _format_number(parameters.get('cutoff_radius', 'R'))
            tests.append(
                f'samples just outside radius {radius} should lose the residual family'
            )
        if 'tapered_distance' in kind:
            radius = _format_number(parameters.get('cutoff_radius', 'R'))
            tests.append(
                f'center/mid/boundary samples should show graded strength before radius {radius}'
            )
        if 'perpendicular' in kind:
            tests.append(
                'rotating around the inferred center should rotate residual vectors by a quarter turn'
            )
        if 'direction' in kind and 'perpendicular' not in kind:
            tests.append(
                'new positions around the inferred center should keep center-aligned residuals'
            )
        if kind in {'periodic_residual', 'generated_periodic_residual'}:
            period = _format_number(parameters.get('period_steps', 'T'))
            tests.append(
                f'matched states separated by period {period} should repeat residual phase'
            )
        tests.append('a hidden holdout should recover the same equation family or narrow its domain')
        return tests

    def _self_authored_equation_key(
        self,
        kind: str,
        parameters: dict[str, Any],
    ) -> str:
        parameter_bits = [
            f'{name}:{_format_number(value)}'
            for name, value in sorted(parameters.items())
            if name in {'distance_exponent', 'cutoff_radius', 'period_steps'}
        ]
        suffix = ':'.join(parameter_bits) if parameter_bits else 'template'
        return f'authored_equation:{kind}:{suffix}'

    def _dominant_numeric_parameter(
        self,
        examples: list[dict[str, Any]],
        name: str,
    ) -> float | None:
        return self._dominant_numeric_parameter_from_dicts(
            [
                dict(example.get('parameters') or {})
                for example in examples
                if isinstance(example.get('parameters'), dict)
            ],
            name,
        )

    def _dominant_numeric_parameter_from_dicts(
        self,
        items: list[dict[str, Any]],
        name: str,
    ) -> float | None:
        values = [
            round(float(item[name]), 6)
            for item in items
            if isinstance(item.get(name), (int, float))
        ]
        if not values:
            return None
        counts: dict[float, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        return max(counts.items(), key=lambda item: (item[1], -abs(item[0])))[0]

    def _numeric_parameter_values(
        self,
        examples: list[dict[str, Any]],
        name: str,
    ) -> list[float]:
        values = set()
        for example in examples:
            parameters = dict(example.get('parameters') or {})
            value = parameters.get(name)
            if isinstance(value, (int, float)):
                values.add(round(float(value), 6))
        return sorted(values)

    def _dominant_text(self, values: list[str], fallback: str) -> str:
        counts: dict[str, int] = {}
        for value in values:
            clean = value.strip()
            if not clean:
                continue
            counts[clean] = counts.get(clean, 0) + 1
        if not counts:
            return fallback
        return max(counts.items(), key=lambda item: (item[1], item[0]))[0]

    def _distinct_nonempty(self, values) -> list[str]:
        seen = set()
        result = []
        for value in values:
            text = str(value).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            result.append(text)
        return result

    def proof_gaps(self) -> list[dict]:
        gaps = []
        for family in self.families.values():
            if family.support_count >= 2 and family.proof_rate < 0.75:
                gaps.append({
                    'theory_kind': family.theory_kind,
                    'support_count': family.support_count,
                    'proof_rate': round(family.proof_rate, 3),
                    'status': family.generalization_status,
                    'next_check': family.next_proof_obligation,
                })
        gaps.sort(key=lambda item: (item['proof_rate'], -item['support_count']))
        return gaps

    def generalization_gaps(self) -> list[dict]:
        gaps = []
        for family in self.families.values():
            if family.support_count < 2:
                continue
            if family.generalization_status in {
                'provisional',
                'local',
                'needs_counterexample',
                'domain_limited',
            }:
                gaps.append(family.evaluation())
        gaps.sort(
            key=lambda item: (
                item['proof_evidence']['context_count'],
                item['proof_evidence']['proof_rate'],
                -item['proof_evidence']['support_count'],
            )
        )
        return gaps

    def domain_revisions(self) -> list[dict]:
        revisions = [
            family.evaluation()
            for family in self.families.values()
            if family.domain_hypothesis.get('revision_needed')
        ]
        revisions.sort(
            key=lambda item: (
                item['proof_evidence']['counterexample_count'],
                item['proof_evidence']['support_count'],
                item['theory_kind'],
            ),
            reverse=True,
        )
        return revisions

    def discovery_readiness_report(self) -> dict[str, Any]:
        """Non-final audit of the autonomous mathematical discovery loop."""
        records_with_theories = sum(
            1 for record in self.records
            if int(record.get('theory_count', 0) or 0) > 0
        )
        records_with_proofs = sum(
            1 for record in self.records
            if int(record.get('proof_check_count', 0) or 0) > 0
        )
        representation_agenda = self.representation_agenda(limit=5)
        operator_priors = self.generated_operator_priors(limit=5)
        prior_feedback = self.operator_prior_feedback(limit=5)
        prior_anomalies = self.operator_prior_anomalies(limit=5)
        prior_repairs = self.operator_prior_repair_experiments(limit=5)
        prior_claims = self.operator_prior_discovery_claims(limit=5)
        prior_chains = self.operator_prior_discovery_chains(limit=5)
        claim_experiments = self.operator_prior_claim_experiments(limit=5)
        disagreement_experiments = self.disagreement_experiments(limit=5)
        next_experiments = self.next_experiments(limit=5)
        planned_experiments = self.planned_experiments(
            world_types=[
                'standard',
                'sideways_wind',
                'vortex',
                'inverse_square_repulsion',
                'localized_gravity',
                'time_varying',
            ],
            object_counts=[5],
            steps=240,
            limit=5,
        )
        proof_certificates = self.proof_certificates(limit=5)
        self_authored_equations = self.self_authored_equations(limit=5)
        first_principles = self.first_principles_basis()
        adaptive_dimensions = self.adaptive_dimension_agenda(limit=5)
        algebraic_foundation = self.algebraic_foundation_baseline()
        algebraic_agenda = self.algebraic_expression_agenda(limit=5)
        domain_curriculum = self.math_domain_curriculum()
        domain_world_blueprints = self.domain_world_blueprints(
            limit=len(MATH_DOMAIN_CURRICULUM),
        )
        domain_world_discoveries = self.domain_world_discovery_reports(
            limit=len(MATH_DOMAIN_CURRICULUM),
        )
        domain_world_transfer_evidence = self.domain_world_transfer_evidence(
            limit=len(MATH_DOMAIN_TRANSFER_BRIDGES),
        )
        domain_transfer_experiments = self.domain_transfer_experiments(limit=5)
        autonomous_scientist = self.autonomous_scientist_evidence()
        arithmetic_rediscovery = self.arithmetic_rediscovery_evidence()
        canonical_law_compression = self.canonical_law_compression_report()
        repair_confirmed_count = sum(
            1 for outcome in self.planned_outcomes
            if outcome.get('outcome') == 'operator_prior_repair_confirmed'
        )
        validation_confirmed_count = sum(
            1 for outcome in self.planned_outcomes
            if outcome.get('outcome') == 'operator_prior_validation_confirmed'
        )
        claim_repair_count = sum(
            1 for experiment in claim_experiments
            if experiment.get('experiment_kind') == 'operator_prior_domain_repair'
        )

        gate_specs = [
            (
                'residual_to_theory',
                'residual observations produce candidate theory families',
                records_with_theories > 0 and bool(self.families),
                1.0,
                {'records_with_theories': records_with_theories, 'family_count': len(self.families)},
                'run an equation campaign until residual theories are recorded',
            ),
            (
                'proof_like_evaluation',
                'theories carry proof-like checks or certificates',
                records_with_proofs > 0 or bool(proof_certificates),
                1.0,
                {'records_with_proofs': records_with_proofs, 'certificate_count': len(proof_certificates)},
                'collect runs with proof checks or reusable family certificates',
            ),
            (
                'model_disagreement_planning',
                'rival theories can choose falsification probes',
                bool(disagreement_experiments) or any(
                    experiment.get('experiment_kind') == 'model_disagreement_probe'
                    for experiment in next_experiments
                ),
                1.0,
                {'disagreement_records': len(self.disagreement_records), 'recommendations': len(disagreement_experiments)},
                'create a run where rival equations disagree on a targeted probe',
            ),
            (
                'representation_agenda',
                'the loop proposes new variables, operators, or domain predicates',
                bool(representation_agenda),
                0.9,
                {'proposal_count': len(representation_agenda)},
                'derive a representation proposal from a disagreement or domain revision',
            ),
            (
                'executable_operator_priors',
                'representation proposals become executable generated operator priors',
                bool(operator_priors),
                1.0,
                {'prior_count': len(operator_priors)},
                'promote a representation proposal into an operator prior',
            ),
            (
                'operator_prior_feedback',
                'generated priors are judged confirmed, weak, or unmatched',
                bool(prior_feedback),
                1.0,
                {'feedback_count': len(prior_feedback)},
                'run a workbench case with memory-generated operator priors installed',
            ),
            (
                'anomaly_repair_loop',
                'failed priors create anomalies or repair experiments',
                (
                    bool(prior_anomalies)
                    or bool(prior_repairs)
                    or claim_repair_count > 0
                    or repair_confirmed_count > 0
                ),
                1.0,
                {
                    'anomaly_count': len(prior_anomalies),
                    'repair_experiment_count': len(prior_repairs),
                    'claim_repair_experiment_count': claim_repair_count,
                    'repair_confirmed_count': repair_confirmed_count,
                },
                'confirm a prior in one context and test it in a context where it breaks',
            ),
            (
                'operator_discovery_claims',
                'invented operators become proof-like discovery claims',
                bool(prior_claims),
                1.0,
                {
                    'claim_count': len(prior_claims),
                    'chain_count': len(prior_chains),
                },
                'collect enough operator-prior feedback to synthesize a discovery claim',
            ),
            (
                'claim_driven_planning',
                'operator claims choose validation or holdout tests',
                bool(claim_experiments) or validation_confirmed_count > 0,
                1.0,
                {
                    'claim_experiment_count': len(claim_experiments),
                    'validation_confirmed_count': validation_confirmed_count,
                },
                'let a repaired or supported operator claim schedule unseen-context validation',
            ),
            (
                'self_authored_equation_synthesis',
                'repeated discoveries are consolidated into self-authored equation templates',
                bool(self_authored_equations),
                1.0,
                {
                    'authored_count': len(self_authored_equations),
                    'best_confidence': (
                        self_authored_equations[0]['confidence']
                        if self_authored_equations else 0.0
                    ),
                },
                'cluster repeated discoveries and write a generalized equation template with falsification tests',
            ),
            (
                'first_principles_adaptive_dimensions',
                'primitive rules can lift residuals into new dimensions',
                bool(first_principles) and bool(adaptive_dimensions),
                1.0,
                {
                    'basis_count': len(first_principles),
                    'adaptive_dimension_count': len(adaptive_dimensions),
                },
                'seed first-principles primitives and let residual failures propose dimensions',
            ),
            (
                'algebraic_foundation_baseline',
                'broad equation and algebra grammar is available with proof obligations',
                (
                    algebraic_foundation['expression_family_count'] >= 16
                    and algebraic_foundation['structure_count'] >= 10
                    and algebraic_foundation['proof_obligation_count'] >= 10
                    and bool(algebraic_foundation['search_controls'])
                ),
                1.0,
                {
                    'expression_family_count': algebraic_foundation['expression_family_count'],
                    'structure_count': algebraic_foundation['structure_count'],
                    'proof_obligation_count': algebraic_foundation['proof_obligation_count'],
                    'agenda_count': len(algebraic_agenda),
                },
                'seed broad algebraic expression families, structures, proof obligations, and search controls',
            ),
            (
                'broad_domain_curriculum',
                'a multi-domain rediscovery curriculum covers core mathematical pressure sources',
                (
                    domain_curriculum['domain_count'] >= 12
                    and domain_curriculum['transfer_bridge_count'] >= 12
                    and set(domain_curriculum['required_domains']) >= {
                        'arithmetic_quantity',
                        'algebra_equations',
                        'geometry_space',
                        'calculus_change',
                        'probability_uncertainty',
                        'logic_proof',
                        'discrete_structures',
                        'symmetry_invariance',
                        'optimization_extrema',
                        'dynamics_systems',
                        'information_computation',
                        'higher_dimensions',
                    }
                ),
                1.0,
                {
                    'domain_count': domain_curriculum['domain_count'],
                    'transfer_bridge_count': domain_curriculum['transfer_bridge_count'],
                    'active_domain_count': domain_curriculum['coverage']['active_domain_count'],
                },
                'seed the missing math domains and bridges in the domain curriculum',
            ),
            (
                'executable_domain_worlds',
                'each math domain has generated observations plus benchmark-only falsifiers',
                (
                    len(domain_world_blueprints) >= domain_curriculum['domain_count']
                    and all(
                        int(item.get('sample_count', 0) or 0) > 0
                        and item.get('expected_discoveries')
                        and int(item.get('falsifier_count', 0) or 0) > 0
                        and not item.get('leaks_benchmark_truth')
                        for item in domain_world_blueprints
                    )
                ),
                1.0,
                {
                    'world_blueprint_count': len(domain_world_blueprints),
                    'domains_with_falsifiers': sum(
                        1 for item in domain_world_blueprints
                        if int(item.get('falsifier_count', 0) or 0) > 0
                    ),
                    'leaky_observation_count': sum(
                        1 for item in domain_world_blueprints
                        if item.get('leaks_benchmark_truth')
                    ),
                },
                'generate label-clean observation worlds with falsifiers for every math domain',
            ),
            (
                'domain_transfer_loop',
                'the curriculum emits cross-domain transfer probes with falsifiers',
                bool(domain_transfer_experiments) and all(
                    item.get('falsifies_if') and item.get('transfer_question')
                    for item in domain_transfer_experiments
                ),
                1.0,
                {
                    'transfer_experiment_count': len(domain_transfer_experiments),
                    'top_priority': (
                        domain_transfer_experiments[0]['priority']
                        if domain_transfer_experiments else 0.0
                    ),
                },
                'generate cross-domain transfer experiments with expected results and falsifiers',
            ),
            (
                'domain_world_discovery_loop',
                'generated domain worlds yield self-authored candidate equations and falsifiers',
                (
                    len(domain_world_discoveries) >= domain_curriculum['domain_count']
                    and all(
                        int(item.get('candidate_count', 0) or 0) > 0
                        and item.get('self_authored_equations')
                        and int(item.get('falsification_test_count', 0) or 0) > 0
                        and float(item.get('benchmark_coverage', 0.0) or 0.0) >= 1.0
                        and not item.get('leaked_manifest')
                        for item in domain_world_discoveries
                    )
                ),
                1.0,
                {
                    'discovery_report_count': len(domain_world_discoveries),
                    'covered_domain_count': sum(
                        1 for item in domain_world_discoveries
                        if float(item.get('benchmark_coverage', 0.0) or 0.0) >= 1.0
                    ),
                    'candidate_count': sum(
                        int(item.get('candidate_count', 0) or 0)
                        for item in domain_world_discoveries
                    ),
                },
                'run generated domain observations through the lightweight discovery evaluator',
            ),
            (
                'domain_world_transfer_evidence',
                'discovered relation bases connect source and target math domains',
                (
                    len(domain_world_transfer_evidence) >= domain_curriculum['transfer_bridge_count']
                    and all(
                        item.get('status') == 'transfer_link_ready'
                        and item.get('falsifies_if')
                        for item in domain_world_transfer_evidence
                    )
                ),
                1.0,
                {
                    'transfer_evidence_count': len(domain_world_transfer_evidence),
                    'ready_transfer_count': sum(
                        1 for item in domain_world_transfer_evidence
                        if item.get('status') == 'transfer_link_ready'
                    ),
                },
                'derive transfer evidence from discovered domain-world relation bases',
            ),
            (
                'arithmetic_rediscovery_probe',
                'counting and unit arithmetic are rediscovered from public observations',
                (
                    arithmetic_rediscovery['record_count'] > 0
                    and arithmetic_rediscovery['best_coverage'] >= 1.0
                    and arithmetic_rediscovery['leaked_manifest_count'] == 0
                    and set(arithmetic_rediscovery['discovered_targets']) >= {
                        'cardinality_invariance',
                        'addition_as_composition',
                        'permutation_invariance',
                        'successor_step',
                        'predecessor_step',
                    }
                ),
                1.0,
                {
                    'record_count': arithmetic_rediscovery['record_count'],
                    'best_coverage': arithmetic_rediscovery['best_coverage'],
                    'discovered_targets': arithmetic_rediscovery['discovered_targets'],
                    'leaked_manifest_count': arithmetic_rediscovery['leaked_manifest_count'],
                },
                'run the arithmetic rediscovery probe on generated public observations',
            ),
            (
                'scientist_invariant_consolidation',
                'repeated runs are consolidated into robust invariant law candidates',
                (
                    autonomous_scientist['record_count'] > 0
                    and autonomous_scientist['robust_invariant_count'] > 0
                ),
                1.0,
                {
                    'scientist_record_count': autonomous_scientist['record_count'],
                    'invariant_count': autonomous_scientist['invariant_count'],
                    'robust_invariant_count': autonomous_scientist[
                        'robust_invariant_count'
                    ],
                },
                'run the autonomous scientist loop across multiple domain-world seeds',
            ),
            (
                'scientist_residual_experiment_loop',
                'candidate failures or untested falsifiers design the next experiment',
                autonomous_scientist['residual_experiment_count'] > 0,
                1.0,
                {
                    'residual_experiment_count': autonomous_scientist[
                        'residual_experiment_count'
                    ],
                    'next_action_count': len(autonomous_scientist['latest_next_actions']),
                },
                'turn domain-world candidate residuals into explicit next experiments',
            ),
            (
                'scientist_harder_hidden_worlds',
                'the loop selects harder hidden worlds for stress-testing laws',
                autonomous_scientist['stress_world_count'] >= 4,
                1.0,
                {
                    'stress_world_count': autonomous_scientist['stress_world_count'],
                    'latest_status': autonomous_scientist['latest_status'],
                },
                'select localized, time-varying, mixed-law, noisy, and higher-dimensional stress worlds',
            ),
            (
                'scientist_richer_equation_writing',
                'the loop rewrites local equations into richer reusable equation grammar',
                autonomous_scientist['authored_equation_extension_count'] > 0,
                1.0,
                {
                    'authored_equation_extension_count': autonomous_scientist[
                        'authored_equation_extension_count'
                    ],
                    'latest_input_candidate_count': autonomous_scientist[
                        'latest_coverage'
                    ].get('input_candidate_count', 0),
                },
                'let the scientist loop write equation extensions from consolidated invariants',
            ),
            (
                'scientist_live_trace',
                'live events expose what the scientist is seeing and doing',
                autonomous_scientist['live_event_count'] > 0,
                1.0,
                {'live_event_count': autonomous_scientist['live_event_count']},
                'emit a live scientist event stream during non-final campaigns',
            ),
            (
                'canonical_law_compression',
                'repeated laws are compacted into reusable canonical summaries',
                (
                    canonical_law_compression['canonical_law_shard_count'] > 0
                    and canonical_law_compression['canonical_law_count'] > 0
                    and canonical_law_compression['long_run_law_ready']
                ),
                0.8,
                {
                    'canonical_law_shard_count': canonical_law_compression[
                        'canonical_law_shard_count'
                    ],
                    'canonical_law_count': canonical_law_compression[
                        'canonical_law_count'
                    ],
                    'robust_law_count': canonical_law_compression['robust_law_count'],
                    'estimated_law_compression_ratio': canonical_law_compression[
                        'estimated_law_compression_ratio'
                    ],
                },
                'compact repeated domain, arithmetic, and scientist laws into canonical summaries',
            ),
            (
                'autonomous_next_experiments',
                'the memory notebook can emit concrete next experiments',
                bool(next_experiments) and bool(planned_experiments),
                1.0,
                {'next_count': len(next_experiments), 'planned_count': len(planned_experiments)},
                'produce at least one concrete planned experiment from memory',
            ),
        ]
        gates = {}
        score_total = 0.0
        weight_total = 0.0
        for key, description, passed, weight, evidence, next_step in gate_specs:
            score_total += weight if passed else 0.0
            weight_total += weight
            gates[key] = {
                'description': description,
                'passed': bool(passed),
                'weight': weight,
                'evidence': evidence,
                'next_step': 'keep monitoring this gate' if passed else next_step,
            }
        readiness_score = score_total / weight_total if weight_total else 0.0
        missing = [
            key for key, gate in gates.items()
            if not gate['passed']
        ]
        if readiness_score >= 0.85 and not missing:
            status = 'ready_for_watched_final'
        elif readiness_score >= 0.65:
            status = 'nearly_ready'
        elif readiness_score >= 0.35:
            status = 'building'
        else:
            status = 'early'
        return {
            'readiness_score': round(readiness_score, 3),
            'status': status,
            'ready_for_watched_final': status == 'ready_for_watched_final',
            'missing_gates': missing,
            'passed_gate_count': len(gates) - len(missing),
            'gate_count': len(gates),
            'gates': gates,
            'next_steps': [
                gates[key]['next_step']
                for key in missing[:3]
            ],
            'recommended_actions': self._discovery_readiness_actions(
                status,
                missing,
            ),
            'evidence_dossier': self.discovery_evidence_dossier(
                prior_chains=prior_chains,
                prior_claims=prior_claims,
                next_experiments=next_experiments,
                planned_experiments=planned_experiments,
                proof_certificates=proof_certificates,
                disagreement_experiments=disagreement_experiments,
                self_authored_equations=self_authored_equations,
                domain_transfer_experiments=(
                    domain_transfer_experiments if self.records else []
                ),
                domain_world_blueprints=domain_world_blueprints,
                domain_world_discoveries=domain_world_discoveries,
                domain_world_transfer_evidence=domain_world_transfer_evidence,
                autonomous_scientist=self.latest_autonomous_scientist_report(),
            ),
            'first_principles_basis': first_principles,
            'adaptive_dimension_agenda': adaptive_dimensions,
            'algebraic_foundation_baseline': algebraic_foundation,
            'algebraic_expression_agenda': algebraic_agenda,
            'self_authored_equations': self_authored_equations,
            'math_domain_curriculum': domain_curriculum,
            'domain_curriculum_agenda': self.domain_curriculum_agenda(limit=12),
            'domain_world_blueprints': domain_world_blueprints,
            'domain_world_discoveries': domain_world_discoveries,
            'domain_world_transfer_evidence': domain_world_transfer_evidence,
            'domain_transfer_experiments': domain_transfer_experiments,
            'autonomous_scientist_evidence': autonomous_scientist,
            'arithmetic_rediscovery_evidence': arithmetic_rediscovery,
            'canonical_law_compression': canonical_law_compression,
        }

    def discovery_evidence_dossier(
        self,
        limit: int = 3,
        *,
        prior_chains: list[dict[str, Any]] | None = None,
        prior_claims: list[dict[str, Any]] | None = None,
        next_experiments: list[dict[str, Any]] | None = None,
        planned_experiments: list[dict[str, Any]] | None = None,
        proof_certificates: list[dict[str, Any]] | None = None,
        disagreement_experiments: list[dict[str, Any]] | None = None,
        self_authored_equations: list[dict[str, Any]] | None = None,
        domain_transfer_experiments: list[dict[str, Any]] | None = None,
        domain_world_blueprints: list[dict[str, Any]] | None = None,
        domain_world_discoveries: list[dict[str, Any]] | None = None,
        domain_world_transfer_evidence: list[dict[str, Any]] | None = None,
        autonomous_scientist: dict[str, Any] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Compact evidence trail for the non-final readiness score."""
        chains = (
            prior_chains
            if prior_chains is not None
            else self.operator_prior_discovery_chains(limit=limit)
        )
        claims = (
            prior_claims
            if prior_claims is not None
            else self.operator_prior_discovery_claims(limit=limit)
        )
        next_items = (
            next_experiments
            if next_experiments is not None
            else self.next_experiments(limit=limit)
        )
        planned = (
            planned_experiments
            if planned_experiments is not None
            else self.planned_experiments(
                world_types=[
                    'standard',
                    'sideways_wind',
                    'vortex',
                    'inverse_square_repulsion',
                    'localized_gravity',
                    'time_varying',
                ],
                object_counts=[5],
                steps=240,
                limit=limit,
            )
        )
        certificates = (
            proof_certificates
            if proof_certificates is not None
            else self.proof_certificates(limit=limit)
        )
        disagreements = (
            disagreement_experiments
            if disagreement_experiments is not None
            else self.disagreement_experiments(limit=limit)
        )
        authored = (
            self_authored_equations
            if self_authored_equations is not None
            else self.self_authored_equations(limit=limit)
        )
        domain_transfers = (
            domain_transfer_experiments
            if domain_transfer_experiments is not None
            else (
                self.domain_transfer_experiments(limit=limit)
                if self.records else []
            )
        )
        domain_worlds = (
            domain_world_blueprints
            if domain_world_blueprints is not None
            else self.domain_world_blueprints(limit=limit)
        )
        domain_discoveries = (
            domain_world_discoveries
            if domain_world_discoveries is not None
            else self.domain_world_discovery_reports(limit=limit)
        )
        domain_transfer_evidence = (
            domain_world_transfer_evidence
            if domain_world_transfer_evidence is not None
            else self.domain_world_transfer_evidence(limit=limit)
        )
        scientist = (
            autonomous_scientist
            if autonomous_scientist is not None
            else self.latest_autonomous_scientist_report()
        )

        chain_summaries = []
        for chain in chains[:limit]:
            evidence = dict(chain.get('proof_evidence') or {})
            steps = list(chain.get('steps') or [])
            latest_step = steps[-1] if steps else {}
            chain_summaries.append({
                'operator_key': chain.get('operator_key'),
                'operator_kind': chain.get('operator_kind'),
                'status': chain.get('status'),
                'step_count': len(steps),
                'latest_step': latest_step.get('step_kind'),
                'latest_summary': latest_step.get('summary'),
                'support_contexts': list(chain.get('support_contexts') or [])[:limit],
                'failure_contexts': list(chain.get('failure_contexts') or [])[:limit],
                'best_score': round(float(evidence.get('best_score', 0.0) or 0.0), 3),
                'confirmed_count': int(evidence.get('confirmed_count', 0) or 0),
                'weak_count': int(evidence.get('weak_count', 0) or 0),
                'unmatched_count': int(evidence.get('unmatched_count', 0) or 0),
                'next_obligation': chain.get('next_obligation'),
            })

        claim_summaries = []
        for claim in claims[:limit]:
            evidence = dict(claim.get('proof_evidence') or {})
            claim_summaries.append({
                'operator_key': claim.get('operator_key'),
                'operator_kind': claim.get('operator_kind'),
                'status': claim.get('status'),
                'expression': claim.get('expression'),
                'best_score': round(float(evidence.get('best_score', 0.0) or 0.0), 3),
                'confirmed_count': int(evidence.get('confirmed_count', 0) or 0),
                'weak_count': int(evidence.get('weak_count', 0) or 0),
                'validation_confirmed_count': int(
                    evidence.get('validation_confirmed_count', 0) or 0
                ),
                'accepted_because': list(claim.get('accepted_because') or [])[:2],
                'not_universal_because': list(claim.get('not_universal_because') or [])[:2],
                'next_obligation': claim.get('next_obligation'),
            })

        planned_summaries = []
        for plan in planned[:limit]:
            planned_summaries.append({
                'experiment_kind': plan.get('experiment_kind'),
                'theory_kind': plan.get('theory_kind'),
                'priority': round(float(plan.get('priority', 0.0) or 0.0), 3),
                'world_type': plan.get('world_type'),
                'seed': plan.get('seed'),
                'object_count': plan.get('object_count'),
                'steps': plan.get('steps'),
                'operator_prior_kind': plan.get('operator_prior_kind'),
                'reason': plan.get('reason'),
                'falsifies_if': plan.get('falsifies_if'),
            })

        next_summaries = []
        for experiment in next_items[:limit]:
            next_summaries.append({
                'experiment_kind': experiment.get('experiment_kind'),
                'theory_kind': experiment.get('theory_kind'),
                'priority': round(float(experiment.get('priority', 0.0) or 0.0), 3),
                'target_context': experiment.get('target_context'),
                'operator_prior_kind': experiment.get('operator_prior_kind'),
                'reason': experiment.get('reason'),
            })

        proof_summaries = []
        for certificate in certificates[:limit]:
            support = dict(certificate.get('support') or {})
            proof_summaries.append({
                'theory_kind': certificate.get('theory_kind'),
                'status': certificate.get('status'),
                'support_count': int(support.get('support_count', 0) or 0),
                'proof_rate': round(float(support.get('proof_rate', 0.0) or 0.0), 3),
                'next_obligation': certificate.get('next_obligation'),
            })

        disagreement_summaries = []
        for experiment in disagreements[:limit]:
            signature = dict(experiment.get('disagreement_signature') or {})
            disagreement_summaries.append({
                'theory_kind': experiment.get('theory_kind'),
                'experiment_kind': experiment.get('experiment_kind'),
                'priority': round(float(experiment.get('priority', 0.0) or 0.0), 3),
                'mode': signature.get('mode'),
                'question': signature.get('question'),
                'primary_theory_label': experiment.get('primary_theory_label'),
                'rival_theory_kinds': list(experiment.get('rival_theory_kinds') or []),
            })

        authored_summaries = []
        for equation in authored[:limit]:
            authored_summaries.append({
                'key': equation.get('key'),
                'equation_kind': equation.get('equation_kind'),
                'status': equation.get('status'),
                'confidence': round(float(equation.get('confidence', 0.0) or 0.0), 3),
                'support_count': int(equation.get('support_count', 0) or 0),
                'expression': equation.get('expression'),
                'proof_obligations': list(equation.get('proof_obligations') or [])[:3],
                'falsification_tests': list(equation.get('falsification_tests') or [])[:2],
            })

        domain_transfer_summaries = []
        for experiment in domain_transfers[:limit]:
            domain_transfer_summaries.append({
                'key': experiment.get('key'),
                'source_domain': experiment.get('source_domain'),
                'target_domain': experiment.get('target_domain'),
                'priority': round(float(experiment.get('priority', 0.0) or 0.0), 3),
                'source_status': experiment.get('source_status'),
                'target_status': experiment.get('target_status'),
                'transfer_question': experiment.get('transfer_question'),
                'falsifies_if': experiment.get('falsifies_if'),
            })

        domain_world_summaries = []
        for blueprint in domain_worlds[:limit]:
            domain_world_summaries.append({
                'domain_key': blueprint.get('domain_key'),
                'sample_count': int(blueprint.get('sample_count', 0) or 0),
                'falsifier_count': int(blueprint.get('falsifier_count', 0) or 0),
                'transfer_targets': list(blueprint.get('transfer_targets') or [])[:limit],
                'leaks_benchmark_truth': bool(blueprint.get('leaks_benchmark_truth')),
                'next_pressure': blueprint.get('next_pressure'),
            })

        domain_discovery_summaries = []
        for discovery in domain_discoveries[:limit]:
            equations = list(discovery.get('self_authored_equations') or [])
            domain_discovery_summaries.append({
                'domain_key': discovery.get('domain_key'),
                'candidate_count': int(discovery.get('candidate_count', 0) or 0),
                'benchmark_coverage': float(discovery.get('benchmark_coverage', 0.0) or 0.0),
                'comparison_hits': list(discovery.get('comparison_hits') or [])[:limit],
                'falsification_test_count': int(
                    discovery.get('falsification_test_count', 0) or 0
                ),
                'top_expression': (
                    equations[0].get('expression')
                    if equations else None
                ),
            })

        domain_transfer_evidence_summaries = []
        for item in domain_transfer_evidence[:limit]:
            domain_transfer_evidence_summaries.append({
                'bridge_key': item.get('bridge_key'),
                'source_domain': item.get('source_domain'),
                'target_domain': item.get('target_domain'),
                'status': item.get('status'),
                'source_matches': list(item.get('source_matches') or [])[:limit],
                'target_matches': list(item.get('target_matches') or [])[:limit],
                'falsifies_if': item.get('falsifies_if'),
            })

        scientist_summaries = []
        if scientist:
            coverage = dict(scientist.get('coverage') or {})
            next_actions = list(scientist.get('next_actions') or [])
            scientist_summaries.append({
                'status': scientist.get('status'),
                'invariant_count': int(coverage.get('invariant_count', 0) or 0),
                'robust_invariant_count': int(
                    coverage.get('robust_invariant_count', 0) or 0
                ),
                'residual_experiment_count': int(
                    coverage.get('residual_experiment_count', 0) or 0
                ),
                'stress_world_count': int(coverage.get('stress_world_count', 0) or 0),
                'authored_equation_extension_count': int(
                    coverage.get('authored_equation_extension_count', 0) or 0
                ),
                'live_event_count': int(coverage.get('live_event_count', 0) or 0),
                'top_next_action': next_actions[0] if next_actions else {},
            })

        return {
            'chains': chain_summaries,
            'claims': claim_summaries,
            'planned_tests': planned_summaries,
            'next_experiments': next_summaries,
            'proof_certificates': proof_summaries,
            'disagreement_probes': disagreement_summaries,
            'self_authored_equations': authored_summaries,
            'domain_transfer_probes': domain_transfer_summaries,
            'domain_world_blueprints': domain_world_summaries,
            'domain_world_discoveries': domain_discovery_summaries,
            'domain_world_transfer_evidence': domain_transfer_evidence_summaries,
            'autonomous_scientist': scientist_summaries,
        }

    def _discovery_readiness_actions(
        self,
        status: str,
        missing_gates: list[str],
    ) -> list[dict[str, Any]]:
        if status == 'ready_for_watched_final':
            return [{
                'action_kind': 'hold_for_user',
                'reason': 'all discovery-loop readiness gates are satisfied',
                'command': (
                    'python3 first_principles_ai/main.py --math-final-discovery '
                    '--benchmark-steps 600 --object-counts 5 --equation-hidden-worlds 3 '
                    '--theory-memory-file tmp/theory-memory.json'
                ),
                'runs_final': True,
            }]
        if not missing_gates:
            return []
        actions = []
        missing = set(missing_gates)
        if missing & {
            'broad_domain_curriculum',
            'executable_domain_worlds',
            'domain_transfer_loop',
            'domain_world_discovery_loop',
            'domain_world_transfer_evidence',
            'arithmetic_rediscovery_probe',
        }:
            actions.append({
                'action_kind': 'non_final_domain_curriculum_review',
                'reason': 'inspect domain coverage, arithmetic probes, and bridge probes before expanding simulator worlds',
                'command': (
                    'python3 first_principles_ai/main.py --discovery-readiness '
                    '--theory-memory-file tmp/theory-memory.json'
                ),
                'runs_final': False,
            })
        if missing & {
            'scientist_invariant_consolidation',
            'scientist_residual_experiment_loop',
            'scientist_harder_hidden_worlds',
            'scientist_richer_equation_writing',
            'scientist_live_trace',
        }:
            actions.append({
                'action_kind': 'non_final_autonomous_scientist_loop',
                'reason': 'consolidate domain-world equations into invariants and next experiments',
                'command': (
                    'python3 first_principles_ai/main.py --autonomous-scientist-loop '
                    '--scientist-seed-count 3 --scientist-variants 0,1 '
                    '--theory-memory-file tmp/theory-memory.json'
                ),
                'runs_final': False,
            })
        if missing & {
            'residual_to_theory',
            'proof_like_evaluation',
            'model_disagreement_planning',
            'representation_agenda',
            'executable_operator_priors',
            'self_authored_equation_synthesis',
            'first_principles_adaptive_dimensions',
            'algebraic_foundation_baseline',
        }:
            actions.append({
                'action_kind': 'non_final_equation_campaign',
                'reason': 'collect residual theories, disagreements, and representation proposals',
                'command': (
                    'python3 first_principles_ai/main.py --equation-campaign '
                    '--seeds 1 --benchmark-steps 240 '
                    '--world-types standard,inverse_square_repulsion,localized_gravity '
                    '--equation-hidden-worlds 1 --theory-memory-file tmp/theory-memory.json'
                ),
                'runs_final': False,
            })
        if missing & {
            'operator_prior_feedback',
            'anomaly_repair_loop',
            'operator_discovery_claims',
            'claim_driven_planning',
        }:
            actions.append({
                'action_kind': 'non_final_followup_campaign',
                'reason': 'test generated priors, repair anomalies, and let claims choose next probes',
                'command': (
                    'python3 first_principles_ai/main.py --equation-campaign '
                    '--seeds 1 --benchmark-steps 260 '
                    '--world-types standard,inverse_square_repulsion,localized_gravity '
                    '--equation-followup-budget 3 '
                    '--theory-memory-file tmp/theory-memory.json'
                ),
                'runs_final': False,
            })
        if 'canonical_law_compression' in missing:
            actions.append({
                'action_kind': 'non_final_canonical_law_compaction',
                'reason': 'compress repeated domain, arithmetic, and scientist laws before longer runs',
                'command': (
                    'python3 first_principles_ai/main.py --compact-theory-memory '
                    '--memory-keep-records 96 --memory-keep-operator-outcomes 192 '
                    '--theory-memory-file tmp/theory-memory.json'
                ),
                'runs_final': False,
            })
        if 'autonomous_next_experiments' in missing:
            actions.append({
                'action_kind': 'non_final_readiness_recheck',
                'reason': 'inspect whether the notebook can emit concrete next experiments',
                'command': (
                    'python3 first_principles_ai/main.py --discovery-readiness '
                    '--theory-memory-file tmp/theory-memory.json'
                ),
                'runs_final': False,
            })
        return actions[:3]

    def first_principles_basis(self) -> list[dict[str, Any]]:
        """Return the primitive moves the system may compose before naming laws."""
        return [
            {
                **item,
                'unlocks': list(item.get('unlocks') or []),
            }
            for item in FIRST_PRINCIPLES_BASIS
        ]

    def adaptive_dimension_agenda(self, limit: int = 8) -> list[dict[str, Any]]:
        """
        Propose new coordinates from residual pressure.

        These are not fixed world axes. They are candidate dimensions the system
        can add when current observations are too small to express a theory.
        """
        dimensions: dict[str, dict[str, Any]] = {}
        for proposal in self.representation_agenda(limit=max(12, limit * 3)):
            dimension = self._adaptive_dimension_from_representation(proposal)
            if dimension:
                dimensions[dimension['key']] = dimension
        for anomaly in self.operator_prior_anomalies(limit=max(8, limit * 3)):
            dimension = self._adaptive_dimension_from_operator_anomaly(anomaly)
            if dimension:
                dimensions[dimension['key']] = dimension
        for gap in self.generalization_gaps()[:max(4, limit)]:
            dimension = self._adaptive_dimension_from_generalization_gap(gap)
            if dimension:
                dimensions.setdefault(dimension['key'], dimension)
        ranked = sorted(
            dimensions.values(),
            key=lambda item: (
                item['priority'],
                item['evidence'].get('support_count', 0),
                item['key'],
            ),
            reverse=True,
        )
        return ranked[:limit]

    def algebraic_foundation_baseline(self) -> dict[str, Any]:
        """Return the broad equation/algebra grammar available to discovery."""
        expression_families = json.loads(json.dumps(ALGEBRAIC_EXPRESSION_FAMILIES))
        structures = json.loads(json.dumps(ALGEBRAIC_STRUCTURES))
        proof_obligations = json.loads(json.dumps(ALGEBRAIC_PROOF_OBLIGATIONS))
        search_controls = json.loads(json.dumps(ALGEBRAIC_SEARCH_CONTROLS))
        return {
            'expression_family_count': len(expression_families),
            'structure_count': len(structures),
            'proof_obligation_count': len(proof_obligations),
            'expression_families': expression_families,
            'algebraic_structures': structures,
            'proof_obligations': proof_obligations,
            'search_controls': search_controls,
        }

    def algebraic_expression_agenda(self, limit: int = 8) -> list[dict[str, Any]]:
        """
        Pick algebraic families to try from the current residual pressure.

        The foundation is intentionally broad. This agenda is the narrow,
        evidence-driven slice that the agent should actually spend search on.
        """
        agenda: dict[str, dict[str, Any]] = {}
        for proposal in self.representation_agenda(limit=max(8, limit * 3)):
            item = self._algebraic_agenda_item_from_signal(
                key=f"algebraic:representation:{proposal.get('key')}",
                source='representation_agenda',
                signal=proposal,
                priority=float(proposal.get('priority', 0.5) or 0.5),
            )
            if item:
                agenda[item['key']] = item
        for dimension in self.adaptive_dimension_agenda(limit=max(6, limit * 2)):
            item = self._algebraic_agenda_item_from_signal(
                key=f"algebraic:dimension:{dimension.get('key')}",
                source='adaptive_dimension_agenda',
                signal=dimension,
                priority=float(dimension.get('priority', 0.45) or 0.45) - 0.02,
            )
            if item:
                agenda.setdefault(item['key'], item)
        for anomaly in self.operator_prior_anomalies(limit=max(6, limit * 2)):
            item = self._algebraic_agenda_item_from_signal(
                key=f"algebraic:anomaly:{anomaly.get('operator_key')}:{anomaly.get('failure_context')}",
                source='operator_prior_anomaly',
                signal=anomaly,
                priority=float(anomaly.get('severity', 0.5) or 0.5) + 0.08,
            )
            if item:
                agenda.setdefault(item['key'], item)
        for gap in self.generalization_gaps()[:max(4, limit)]:
            item = self._algebraic_agenda_item_from_signal(
                key=f"algebraic:generalization:{gap.get('theory_kind')}",
                source='generalization_gap',
                signal=gap,
                priority=float(gap.get('priority', 0.5) or 0.5) + 0.02,
            )
            if item:
                agenda.setdefault(item['key'], item)
        ranked = sorted(
            agenda.values(),
            key=lambda item: (
                item['priority'],
                len(item['expression_families']),
                item['key'],
            ),
            reverse=True,
        )
        return ranked[:limit]

    def math_domain_curriculum(self) -> dict[str, Any]:
        """Return the broad math-domain curriculum used to accelerate rediscovery."""
        domains = json.loads(json.dumps(MATH_DOMAIN_CURRICULUM))
        bridges = json.loads(json.dumps(MATH_DOMAIN_TRANSFER_BRIDGES))
        agenda = self.domain_curriculum_agenda(limit=len(domains))
        covered = [
            item['domain_key']
            for item in agenda
            if item['status'] in {'active', 'transfer_ready'}
        ]
        return {
            'version': 1,
            'domain_count': len(domains),
            'transfer_bridge_count': len(bridges),
            'required_domains': [domain['key'] for domain in domains],
            'domains': domains,
            'transfer_bridges': bridges,
            'coverage': {
                'active_domain_count': len(covered),
                'active_domains': covered,
                'pending_domains': [
                    item['domain_key']
                    for item in agenda
                    if item['status'] == 'seeded_pending_world'
                ],
            },
            'curriculum_policy': (
                'each domain is a pressure source, not an answer bank; a '
                'candidate concept must be rediscovered from observations, '
                'transferred across at least one bridge, and survive a named falsifier'
            ),
        }

    def domain_curriculum_agenda(self, limit: int = 12) -> list[dict[str, Any]]:
        """Rank domains by current evidence and missing rediscovery pressure."""
        agenda = []
        for domain in MATH_DOMAIN_CURRICULUM:
            evidence = self._domain_curriculum_evidence(domain)
            support = int(evidence.get('support_count', 0) or 0)
            bridge_count = self._domain_bridge_count(str(domain['key']))
            if support >= 2:
                status = 'transfer_ready'
            elif support == 1:
                status = 'active'
            else:
                status = 'seeded_pending_world'
            priority = (
                1.0
                - 0.05 * min(8, support)
                + 0.01 * int(domain.get('curriculum_order', 0) or 0)
            )
            if status == 'seeded_pending_world':
                priority += 0.18
            agenda.append({
                'domain_key': domain['key'],
                'name': domain['name'],
                'curriculum_order': domain['curriculum_order'],
                'status': status,
                'priority': round(max(0.1, min(1.0, priority)), 3),
                'support_count': support,
                'bridge_count': bridge_count,
                'evidence': evidence,
                'next_pressure': self._domain_next_pressure(domain, status),
                'target_primitives': list(domain.get('primitive_targets') or []),
                'equation_families': list(domain.get('equation_families') or []),
                'proof_pressure': list(domain.get('proof_pressure') or []),
            })
        agenda.sort(
            key=lambda item: (
                item['priority'],
                -item['support_count'],
                -item['curriculum_order'],
            ),
            reverse=True,
        )
        return agenda[:limit]

    def domain_transfer_experiments(self, limit: int = 8) -> list[dict[str, Any]]:
        """Suggest cross-domain probes that force discovered math to transfer."""
        agenda = {
            item['domain_key']: item
            for item in self.domain_curriculum_agenda(limit=len(MATH_DOMAIN_CURRICULUM))
        }
        experiments = []
        for bridge in MATH_DOMAIN_TRANSFER_BRIDGES:
            source = agenda.get(str(bridge['source_domain']), {})
            target = agenda.get(str(bridge['target_domain']), {})
            source_support = int(source.get('support_count', 0) or 0)
            target_support = int(target.get('support_count', 0) or 0)
            priority = 0.52
            if source_support > 0:
                priority += 0.16
            if target_support == 0:
                priority += 0.18
            if source.get('status') == 'transfer_ready':
                priority += 0.08
            if target.get('status') == 'seeded_pending_world':
                priority += 0.06
            experiments.append({
                'key': f"domain_transfer:{bridge['key']}",
                'experiment_kind': 'domain_transfer_probe',
                'source_domain': bridge['source_domain'],
                'target_domain': bridge['target_domain'],
                'source_status': source.get('status', 'seeded_pending_world'),
                'target_status': target.get('status', 'seeded_pending_world'),
                'priority': round(max(0.1, min(1.0, priority)), 3),
                'bridge_principle': bridge['bridge_principle'],
                'transfer_question': bridge['transfer_question'],
                'expected_result': (
                    'a self-authored rule or operator from the source domain '
                    'should explain held-out observations in the target domain'
                ),
                'falsifies_if': bridge['falsifier'],
                'suggested_world_seed': self._domain_bridge_world_seed(bridge),
                'source_evidence': dict(source.get('evidence') or {}),
                'target_evidence': dict(target.get('evidence') or {}),
            })
        experiments.sort(
            key=lambda item: (
                item['priority'],
                item['source_status'] == 'transfer_ready',
                item['key'],
            ),
            reverse=True,
        )
        return experiments[:limit]

    def domain_world_blueprints(
        self,
        limit: int = 12,
        seed: int = 0,
        variant: int = 0,
    ) -> list[dict[str, Any]]:
        """Summarize executable observation worlds for the broad math curriculum."""
        try:
            from world.math_domain_worlds import (
                generate_math_domain_world_manifest,
                math_domain_manifest_from_observation,
            )
        except ImportError:  # pragma: no cover - package import fallback
            from first_principles_ai.world.math_domain_worlds import (
                generate_math_domain_world_manifest,
                math_domain_manifest_from_observation,
            )

        agenda = {
            item['domain_key']: item
            for item in self.domain_curriculum_agenda(limit=len(MATH_DOMAIN_CURRICULUM))
        }
        blueprints = []
        for domain in MATH_DOMAIN_CURRICULUM:
            key = str(domain['key'])
            item = agenda.get(key, {})
            manifest = generate_math_domain_world_manifest(
                key,
                seed=seed + int(domain.get('curriculum_order', 0) or 0),
                variant=variant,
            )
            observations = manifest.observations()
            leak_count = sum(
                1 for observation in observations
                if math_domain_manifest_from_observation(observation)
            )
            blueprints.append({
                'domain_key': key,
                'name': domain.get('name'),
                'curriculum_order': domain.get('curriculum_order'),
                'status': item.get('status', 'seeded_pending_world'),
                'priority': item.get('priority', 0.0),
                'seed': manifest.seed,
                'variant': manifest.variant,
                'sample_count': len(observations),
                'observation_kinds': sorted({
                    str(observation.get('observation_kind', 'unknown'))
                    for observation in observations
                }),
                'observation_schema': manifest.observation_schema(),
                'expected_discoveries': list(manifest.expected_discoveries),
                'falsifier_count': len(manifest.falsifiers),
                'falsifiers': list(manifest.falsifiers),
                'transfer_targets': list(manifest.transfer_targets),
                'leaks_benchmark_truth': leak_count > 0,
                'leaky_observation_count': leak_count,
                'next_pressure': item.get(
                    'next_pressure',
                    self._domain_next_pressure(domain, 'seeded_pending_world'),
                ),
            })
        blueprints.sort(
            key=lambda item: (
                float(item.get('priority', 0.0) or 0.0),
                -int(item.get('curriculum_order', 0) or 0),
                str(item.get('domain_key', '')),
            ),
            reverse=True,
        )
        return blueprints[:limit]

    def domain_world_discovery_reports(
        self,
        limit: int = 12,
        seed: int = 0,
        variant: int = 0,
    ) -> list[dict[str, Any]]:
        """Run lightweight discovery over generated domain-world observations."""
        reports = self._domain_world_discovery_report_objects(
            seed=seed,
            variant=variant,
        )
        packed = [report.to_dict() for report in reports]
        packed.sort(
            key=lambda item: (
                float(item.get('benchmark_coverage', 0.0) or 0.0),
                int(item.get('candidate_count', 0) or 0),
                str(item.get('domain_key', '')),
            ),
            reverse=True,
        )
        return packed[:limit]

    def domain_world_transfer_evidence(
        self,
        limit: int = 12,
        seed: int = 0,
        variant: int = 0,
    ) -> list[dict[str, Any]]:
        """Score transfer bridges using discovered relation bases."""
        try:
            from agent.domain_world_discovery import build_domain_transfer_evidence
        except ImportError:  # pragma: no cover - package import fallback
            from first_principles_ai.agent.domain_world_discovery import (
                build_domain_transfer_evidence,
            )

        reports = self._domain_world_discovery_report_objects(
            seed=seed,
            variant=variant,
        )
        return build_domain_transfer_evidence(
            reports,
            MATH_DOMAIN_TRANSFER_BRIDGES,
            limit=limit,
        )

    def autonomous_scientist_evidence(self) -> dict[str, Any]:
        """Aggregate persisted scientist-loop evidence for readiness gates."""
        latest = (
            self.autonomous_scientist_records[-1]
            if self.autonomous_scientist_records
            else {}
        )
        records = list(self.autonomous_scientist_records)
        invariant_count = sum(
            len(record.get('invariant_consolidations') or [])
            for record in records
        )
        robust_invariant_count = sum(
            1
            for record in records
            for item in record.get('invariant_consolidations') or []
            if item.get('status') == 'robust_law'
        )
        residual_count = sum(
            len(record.get('residual_experiments') or [])
            for record in records
        )
        stress_world_count = sum(
            len(record.get('harder_stress_worlds') or [])
            for record in records
        )
        equation_count = sum(
            len(record.get('authored_equation_extensions') or [])
            for record in records
        )
        live_event_count = sum(
            len(record.get('live_events') or [])
            for record in records
        )
        return {
            'record_count': len(records),
            'latest_status': latest.get('status'),
            'invariant_count': invariant_count,
            'robust_invariant_count': robust_invariant_count,
            'residual_experiment_count': residual_count,
            'stress_world_count': stress_world_count,
            'authored_equation_extension_count': equation_count,
            'live_event_count': live_event_count,
            'latest_coverage': dict(latest.get('coverage') or {}),
            'latest_next_actions': list(latest.get('next_actions') or []),
        }

    def arithmetic_rediscovery_evidence(self) -> dict[str, Any]:
        """Aggregate observation-only counting/arithmetic rediscovery evidence."""
        latest = (
            self.arithmetic_rediscovery_records[-1]
            if self.arithmetic_rediscovery_records
            else {}
        )
        records = list(self.arithmetic_rediscovery_records)
        discovered_targets = sorted({
            str(target)
            for record in records
            for target in record.get('discovered_targets') or []
        })
        equation_count = sum(
            len(record.get('self_authored_equations') or [])
            for record in records
        )
        leaked_count = sum(
            1 for record in records
            if record.get('leaked_manifest')
        )
        best_coverage = max(
            [float(record.get('coverage', 0.0) or 0.0) for record in records]
            or [0.0]
        )
        return {
            'record_count': len(records),
            'latest_status': latest.get('status'),
            'best_coverage': round(best_coverage, 3),
            'discovered_target_count': len(discovered_targets),
            'discovered_targets': discovered_targets,
            'equation_count': equation_count,
            'leaked_manifest_count': leaked_count,
            'latest_missing_targets': list(latest.get('missing_targets') or []),
            'latest_live_event_count': len(latest.get('live_events') or []),
        }

    def latest_autonomous_scientist_report(self) -> dict[str, Any]:
        if not self.autonomous_scientist_records:
            return {}
        return dict(self.autonomous_scientist_records[-1])

    def _domain_world_discovery_report_objects(
        self,
        seed: int = 0,
        variant: int = 0,
    ):
        try:
            from agent.domain_world_discovery import discover_domain_world_manifest
            from world.math_domain_worlds import generate_math_domain_world_manifest
        except ImportError:  # pragma: no cover - package import fallback
            from first_principles_ai.agent.domain_world_discovery import (
                discover_domain_world_manifest,
            )
            from first_principles_ai.world.math_domain_worlds import (
                generate_math_domain_world_manifest,
            )

        reports = []
        for domain in MATH_DOMAIN_CURRICULUM:
            manifest = generate_math_domain_world_manifest(
                str(domain['key']),
                seed=seed + int(domain.get('curriculum_order', 0) or 0),
                variant=variant,
            )
            reports.append(discover_domain_world_manifest(manifest))
        return reports

    def _domain_curriculum_evidence(self, domain: dict[str, Any]) -> dict[str, Any]:
        keywords = self._domain_keywords(domain)
        matched_families = []
        support = 0
        for family in self.families.values():
            text = self._family_search_text(family)
            if any(keyword in text for keyword in keywords):
                matched_families.append(family.theory_kind)
                support += family.support_count
        authored = []
        for equation in self.self_authored_equations(limit=12):
            text = ' '.join(
                str(equation.get(field, ''))
                for field in ('equation_kind', 'expression', 'target')
            ).lower()
            if any(keyword in text for keyword in keywords):
                authored.append(str(equation.get('key')))
        adaptive = []
        for dimension in self.adaptive_dimension_agenda(limit=12):
            text = ' '.join(
                str(dimension.get(field, ''))
                for field in ('name', 'dimension_kind', 'expression', 'theory_kind')
            ).lower()
            if any(keyword in text for keyword in keywords):
                adaptive.append(str(dimension.get('key')))
        domain_world_records = [
            record for record in self.domain_world_records
            if record.get('domain_key') == domain.get('key')
        ]
        discovered_equations = []
        transfer_basis = set()
        for record in domain_world_records:
            support += int(record.get('candidate_count', 0) or 0)
            for equation in list(record.get('self_authored_equations') or [])[:3]:
                if equation.get('expression'):
                    discovered_equations.append(str(equation['expression']))
            transfer_basis.update(str(item) for item in record.get('transfer_basis') or [])
        return {
            'support_count': support,
            'matched_families': sorted(set(matched_families))[:5],
            'self_authored_equations': authored[:5],
            'adaptive_dimensions': adaptive[:5],
            'domain_world_record_count': len(domain_world_records),
            'domain_world_equations': discovered_equations[:5],
            'domain_world_transfer_basis': sorted(transfer_basis)[:8],
            'evidence_keywords': sorted(keywords)[:8],
        }

    def _domain_keywords(self, domain: dict[str, Any]) -> set[str]:
        explicit = {
            'arithmetic_quantity': {'count', 'cardinality', 'successor', 'total', 'quantity'},
            'algebra_equations': {'equation', 'substitution', 'inverse', 'composition', 'variable'},
            'geometry_space': {'center', 'distance', 'perpendicular', 'vector', 'boundary', 'coordinate'},
            'calculus_change': {'delta', 'velocity', 'acceleration', 'finite', 'rate', 'curvature'},
            'probability_uncertainty': {'probability', 'uncertainty', 'frequency', 'weak', 'calibration'},
            'logic_proof': {'proof', 'counterexample', 'predicate', 'falsif', 'domain'},
            'discrete_structures': {'graph', 'path', 'set', 'relation', 'state'},
            'symmetry_invariance': {'symmetry', 'invariant', 'rotation', 'perpendicular', 'transform'},
            'optimization_extrema': {'optimization', 'extremum', 'minimum', 'maximum', 'error'},
            'dynamics_systems': {'residual', 'field', 'transition', 'periodic', 'phase', 'force'},
            'information_computation': {'information', 'encoding', 'compression', 'algorithm', 'hidden'},
            'higher_dimensions': {'dimension', 'projection', 'latent', 'basis', 'lift'},
        }
        keywords = set(explicit.get(str(domain.get('key')), set()))
        for field_name in (
            'key',
            'name',
            'primitive_targets',
            'equation_families',
            'expected_discoveries',
        ):
            value = domain.get(field_name)
            if isinstance(value, list):
                words = value
            else:
                words = [value]
            for word in words:
                for token in str(word).lower().replace('-', '_').split('_'):
                    if len(token) >= 5:
                        keywords.add(token)
        return keywords

    def _family_search_text(self, family: TheoryFamily) -> str:
        parts = [
            family.theory_kind,
            *sorted(family.operator_kinds),
            *sorted(family.concept_kinds),
        ]
        for example in family.examples:
            parts.extend(
                str(example.get(field, ''))
                for field in ('claim', 'target', 'expression', 'theory_kind')
            )
        return ' '.join(parts).lower()

    def _domain_bridge_count(self, domain_key: str) -> int:
        return sum(
            1 for bridge in MATH_DOMAIN_TRANSFER_BRIDGES
            if bridge['source_domain'] == domain_key or bridge['target_domain'] == domain_key
        )

    def _domain_next_pressure(self, domain: dict[str, Any], status: str) -> str:
        if status == 'transfer_ready':
            return 'force this domain to transfer across a bridge and search for counterexamples'
        if status == 'active':
            return 'repeat on another seed or bridge domain before treating it as reusable'
        tasks = list(domain.get('observation_tasks') or [])
        return tasks[0] if tasks else 'create an observation world for this domain'

    def _domain_bridge_world_seed(self, bridge: dict[str, Any]) -> dict[str, Any]:
        source = self._domain_by_key(str(bridge.get('source_domain')))
        target = self._domain_by_key(str(bridge.get('target_domain')))
        return {
            'source_world_seed': source.get('world_seed'),
            'target_world_seed': target.get('world_seed'),
            'combined_pressure': (
                f"{source.get('name', bridge.get('source_domain'))} -> "
                f"{target.get('name', bridge.get('target_domain'))}"
            ),
        }

    def _domain_by_key(self, domain_key: str) -> dict[str, Any]:
        for domain in MATH_DOMAIN_CURRICULUM:
            if domain['key'] == domain_key:
                return domain
        return {'key': domain_key, 'name': domain_key, 'world_seed': 'unknown'}

    def representation_agenda(self, limit: int = 5) -> list[dict]:
        """
        Propose new internal measurements or operators from accumulated theory.

        Next-experiment planning asks where to look. This agenda asks what the
        agent should add to its mathematical language before it looks there.
        """
        proposals: dict[str, dict[str, Any]] = {}
        for experiment in self.disagreement_experiments(limit=limit * 3):
            proposal = self._representation_from_disagreement(experiment)
            if proposal:
                proposals[proposal['key']] = proposal
        for revision in self.domain_revisions():
            proposal = self._representation_from_domain_revision(revision)
            if proposal:
                proposals[proposal['key']] = proposal
        for anomaly in self.operator_prior_anomalies(limit=limit * 3):
            proposal = self._representation_from_operator_prior_anomaly(anomaly)
            if proposal:
                proposals[proposal['key']] = proposal
        for family in self.families.values():
            proposal = self._representation_from_family(family)
            if proposal:
                proposals.setdefault(proposal['key'], proposal)

        ranked = sorted(
            proposals.values(),
            key=lambda item: (
                item['priority'],
                item['evidence'].get('support_count', 0),
                item['key'],
            ),
            reverse=True,
        )
        return ranked[:limit]

    def generated_operator_priors(
        self,
        limit: int = 8,
        context: str | None = None,
    ) -> list[dict]:
        """Convert representation agenda items into workbench-search priors."""
        limit = max(0, int(limit or 0))
        if limit <= 0:
            return []
        candidates = self._generated_operator_prior_candidates(
            limit=limit,
            context=context,
        )
        return self._bounded_operator_priors(candidates, limit=limit)

    def operator_prior_budget_report(
        self,
        limit: int = 8,
        context: str | None = None,
    ) -> dict[str, Any]:
        limit = max(0, int(limit or 0))
        candidates = self._generated_operator_prior_candidates(
            limit=limit,
            context=context,
        )
        selected = self._bounded_operator_priors(candidates, limit=limit)
        signature_counts: dict[str, int] = {}
        for prior in selected:
            signature = self._operator_prior_budget_signature(prior)
            signature_key = '|'.join(signature)
            signature_counts[signature_key] = signature_counts.get(signature_key, 0) + 1
        return {
            'limit': limit,
            'context': context,
            'candidate_count': len(candidates),
            'selected_count': len(selected),
            'max_per_signature': self._operator_prior_signature_cap(limit),
            'signature_counts': signature_counts,
            'selected_keys': [item.get('key') for item in selected],
        }

    def _generated_operator_prior_candidates(
        self,
        limit: int,
        context: str | None,
    ) -> list[dict[str, Any]]:
        candidate_limit = max(limit * 4, 16)
        priors: dict[str, dict[str, Any]] = {}
        for record in reversed(self.disagreement_records):
            for prior in self._operator_priors_from_disagreement(record):
                adjusted = self._apply_operator_prior_feedback(prior)
                adjusted = self._annotate_operator_prior_with_first_principles(adjusted)
                adjusted = self._annotate_operator_prior_with_algebraic_foundation(adjusted)
                if not self._operator_prior_allowed_in_context(adjusted, context):
                    continue
                priors[str(adjusted.get('key'))] = adjusted
                if len(priors) >= candidate_limit:
                    break
            if len(priors) >= candidate_limit:
                break
        for claim in self.operator_prior_discovery_claims(limit=max(5, candidate_limit)):
            prior = self._operator_prior_from_claim(claim)
            if not prior:
                continue
            adjusted = self._apply_operator_prior_feedback(prior)
            adjusted = self._annotate_operator_prior_with_first_principles(adjusted)
            adjusted = self._annotate_operator_prior_with_algebraic_foundation(adjusted)
            if not self._operator_prior_allowed_in_context(adjusted, context):
                continue
            priors.setdefault(str(adjusted.get('key')), adjusted)
            if len(priors) >= candidate_limit:
                break
        ranked = sorted(
            priors.values(),
            key=lambda item: (
                item.get('usefulness', 0.0),
                item.get('key', ''),
            ),
            reverse=True,
        )
        return ranked

    def _bounded_operator_priors(
        self,
        candidates: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        max_per_signature = self._operator_prior_signature_cap(limit)
        max_per_kind = max(2, limit)
        selected: list[dict[str, Any]] = []
        selected_keys: set[str] = set()
        signature_counts: dict[tuple[str, str, str, str], int] = {}
        kind_counts: dict[str, int] = {}

        for prior in candidates:
            signature = self._operator_prior_budget_signature(prior)
            kind = str(prior.get('operator_kind', 'unknown'))
            if signature_counts.get(signature, 0) >= max_per_signature:
                continue
            if kind_counts.get(kind, 0) >= max_per_kind:
                continue
            self._select_operator_prior(
                prior,
                selected,
                selected_keys,
                signature_counts,
                kind_counts,
                signature,
                kind,
            )
            if len(selected) >= limit:
                return selected

        for prior in candidates:
            key = str(prior.get('key', ''))
            if key in selected_keys:
                continue
            signature = self._operator_prior_budget_signature(prior)
            if signature_counts.get(signature, 0) >= max_per_signature + 1:
                continue
            kind = str(prior.get('operator_kind', 'unknown'))
            self._select_operator_prior(
                prior,
                selected,
                selected_keys,
                signature_counts,
                kind_counts,
                signature,
                kind,
            )
            if len(selected) >= limit:
                return selected
        return selected

    def _select_operator_prior(
        self,
        prior: dict[str, Any],
        selected: list[dict[str, Any]],
        selected_keys: set[str],
        signature_counts: dict[tuple[str, str, str, str], int],
        kind_counts: dict[str, int],
        signature: tuple[str, str, str, str],
        kind: str,
    ):
        selected.append(prior)
        selected_keys.add(str(prior.get('key', '')))
        signature_counts[signature] = signature_counts.get(signature, 0) + 1
        kind_counts[kind] = kind_counts.get(kind, 0) + 1

    def _operator_prior_signature_cap(self, limit: int) -> int:
        return 2 if limit >= 5 else 3

    def _operator_prior_budget_signature(
        self,
        prior: dict[str, Any],
    ) -> tuple[str, str, str, str]:
        parameters = dict(prior.get('parameters') or {})
        return (
            str(prior.get('operator_kind', 'unknown')),
            str(parameters.get('relation', 'unknown')),
            str(parameters.get('source_context') or parameters.get('context') or 'unknown'),
            self._operator_prior_window_kind(prior),
        )

    def _operator_prior_window_kind(self, prior: dict[str, Any]) -> str:
        operator_kind = str(prior.get('operator_kind', ''))
        expression = str(prior.get('expression', ''))
        if operator_kind == 'localized_tapered_power' or 'taper' in expression or 'max(0' in expression:
            return 'tapered'
        if operator_kind == 'localized_cutoff_window' or 'inside(' in expression:
            return 'windowed'
        return 'global'

    def operator_prior_invariant_consolidations(
        self,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Summarize repeated headline equations into cross-seed law families."""
        groups: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
        for record in self.equation_case_records:
            expression = str(record.get('expression') or '')
            if not expression:
                continue
            context = str(record.get('context') or 'unknown')
            target = str(record.get('target') or 'unknown')
            law_family = self._equation_law_family(record)
            vector_basis = self._equation_vector_basis(expression)
            window_kind = self._equation_window_kind(expression)
            key = (context, target, law_family, vector_basis, window_kind)
            group = groups.setdefault(
                key,
                {
                    'context': context,
                    'target': target,
                    'law_family': law_family,
                    'vector_basis': vector_basis,
                    'window_kind': window_kind,
                    'records': [],
                    'support_seeds': set(),
                    'scores': [],
                    'leak_count': 0,
                    'pass_count': 0,
                    'expressions': {},
                    'exponents': {},
                    'parameter_records': [],
                },
            )
            group['records'].append(record)
            group['parameter_records'].append(dict(record.get('parameters') or {}))
            if isinstance(record.get('seed'), int):
                group['support_seeds'].add(int(record['seed']))
            group['scores'].append(float(record.get('score', 0.0) or 0.0))
            group['leak_count'] += int(record.get('label_leak_count', 0) or 0)
            if record.get('passed'):
                group['pass_count'] += 1
            group['expressions'][expression] = group['expressions'].get(expression, 0) + 1
            exponent = self._equation_distance_exponent(record)
            if exponent is not None:
                group['exponents'][exponent] = group['exponents'].get(exponent, 0) + 1

        consolidations = []
        for key, group in groups.items():
            support_count = len(group['records'])
            mean_score = (
                sum(group['scores']) / len(group['scores'])
                if group['scores']
                else 0.0
            )
            parameter_candidates = [
                {
                    'name': 'distance_exponent',
                    'value': exponent,
                    'support': count,
                }
                for exponent, count in sorted(
                    group['exponents'].items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ]
            dominant_expression, dominant_support = max(
                group['expressions'].items(),
                key=lambda item: (item[1], item[0]),
            )
            status = self._equation_invariant_status(
                support_count=support_count,
                parameter_candidate_count=len(parameter_candidates),
            )
            item = {
                'key': ':'.join(key),
                'context': group['context'],
                'target': group['target'],
                'law_family': group['law_family'],
                'vector_basis': group['vector_basis'],
                'window_kind': group['window_kind'],
                'status': status,
                'support_count': support_count,
                'support_seeds': sorted(group['support_seeds']),
                'mean_score': round(mean_score, 3),
                'pass_count': group['pass_count'],
                'leak_count': group['leak_count'],
                'dominant_expression': dominant_expression,
                'dominant_expression_support': dominant_support,
                'dominant_parameters': self._equation_invariant_dominant_parameters(
                    list(group.get('parameter_records') or [])
                ),
                'parameter_candidates': parameter_candidates,
            }
            item['robust_claim'] = self._equation_invariant_claim(item)
            item['next_experiment'] = self._equation_invariant_next_experiment(item)
            item['falsifies_if'] = self._equation_invariant_falsifier(item)
            consolidations.append(item)

        status_priority = {
            'robust_law': 4,
            'robust_family_parameter_unresolved': 3,
            'candidate_family': 2,
            'local_candidate': 1,
        }
        consolidations.sort(
            key=lambda item: (
                status_priority.get(item['status'], 0),
                item['support_count'],
                item['mean_score'],
                -item['leak_count'],
                item['key'],
            ),
            reverse=True,
        )
        return consolidations[:limit]

    def _equation_law_family(self, record: dict[str, Any]) -> str:
        expression = str(record.get('expression') or '')
        role = str(record.get('role') or '')
        window_kind = self._equation_window_kind(expression)
        if 'separation^' in expression:
            if window_kind == 'tapered':
                return 'localized_tapered_power'
            return 'inverse_separation_power'
        if window_kind == 'windowed':
            return 'localized_window'
        if 'vx * dt' in expression or 'vy * dt' in expression:
            return 'linear_transition'
        if role:
            return role
        return 'equation'

    def _equation_invariant_dominant_parameters(
        self,
        parameter_records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        dominant = {}
        for name in (
            'center_x',
            'center_y',
            'cutoff_radius',
            'distance_exponent',
        ):
            value = self._dominant_numeric_parameter_from_dicts(
                parameter_records,
                name,
            )
            if value is not None:
                dominant[name] = value
        return _rounded_dict(dominant)

    def _equation_vector_basis(self, expression: str) -> str:
        for basis in (
            'unit_anchor_vector',
            'unit_generated_center_vector',
            'unit_local_inferred_vector',
        ):
            if basis in expression:
                return basis
        if 'perpendicular' in expression:
            return 'perpendicular_unit_center_vector'
        if 'unit(center - position)' in expression:
            return 'unit_center_position_vector'
        if 'unit(' in expression:
            return 'unit_vector'
        return 'scalar_or_unknown'

    def _equation_window_kind(self, expression: str) -> str:
        if 'taper' in expression or 'max(0' in expression:
            return 'tapered'
        if 'inside(' in expression or 'separation <=' in expression:
            return 'windowed'
        return 'global'

    def _equation_distance_exponent(self, record: dict[str, Any]) -> float | None:
        parameters = dict(record.get('parameters') or {})
        value = parameters.get('distance_exponent')
        if isinstance(value, (int, float)):
            return round(float(value), 3)
        expression = str(record.get('expression') or '')
        if 'separation^' not in expression:
            return None
        token = expression.split('separation^', 1)[1].strip().split()[0]
        token = token.split('*')[0].split('/')[0].rstrip('),.;')
        cleaned = ''.join(
            char for char in token
            if char.isdigit() or char in {'-', '.', '_'}
        ).replace('_', '.')
        try:
            return round(float(cleaned), 3)
        except ValueError:
            return None

    def _equation_invariant_status(
        self,
        support_count: int,
        parameter_candidate_count: int,
    ) -> str:
        if support_count >= 3 and parameter_candidate_count > 1:
            return 'robust_family_parameter_unresolved'
        if support_count >= 3:
            return 'robust_law'
        if support_count >= 2:
            return 'candidate_family'
        return 'local_candidate'

    def _equation_invariant_claim(self, item: dict[str, Any]) -> str:
        parameters = item.get('parameter_candidates') or []
        if item['status'] == 'robust_family_parameter_unresolved':
            values = ', '.join(str(parameter['value']) for parameter in parameters)
            return (
                f"{item['context']} repeatedly fits {item['law_family']} "
                f"using {item['vector_basis']}; exponent remains unresolved "
                f"among {values}"
            )
        if item['status'] == 'robust_law':
            return (
                f"{item['context']} repeatedly fits {item['law_family']} "
                f"using {item['vector_basis']}"
            )
        return (
            f"{item['context']} has a candidate {item['law_family']} "
            f"using {item['vector_basis']}"
        )

    def _equation_invariant_next_experiment(self, item: dict[str, Any]) -> str:
        if len(item.get('parameter_candidates') or []) > 1:
            return (
                'run matched near/far residual probes to choose the distance '
                'exponent without changing the vector basis'
            )
        if item.get('leak_count', 0) > 0:
            return 'rerun the same law family on leak-clean held-out observations'
        if item['status'] == 'robust_law':
            return 'test the same invariant on hidden and off-center holdout worlds'
        return 'repeat on another seed before treating this as reusable theory'

    def _equation_invariant_falsifier(self, item: dict[str, Any]) -> str:
        if item['law_family'] in {
            'inverse_separation_power',
            'localized_tapered_power',
        }:
            return (
                'held-out residuals prefer a different vector basis, window, '
                'or no inverse-separation improvement'
            )
        return 'held-out equations fail to reproduce the repeated residual pattern'

    def equation_invariant_resolution_experiments(
        self,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        recommendations = []
        for invariant in self.operator_prior_invariant_consolidations(
            limit=max(5, len(self.equation_case_records)),
        ):
            if invariant.get('status') != 'robust_family_parameter_unresolved':
                continue
            if invariant.get('law_family') not in {
                'inverse_separation_power',
                'localized_tapered_power',
            }:
                continue
            candidates = list(invariant.get('parameter_candidates') or [])
            if len(candidates) < 2:
                continue
            outcome_stats = self._invariant_resolution_outcome_stats(
                str(invariant.get('key', 'unknown')),
            )
            if outcome_stats['selected_count'] > 0:
                continue
            signature = self._equation_invariant_resolution_signature(
                invariant,
                refinement_level=outcome_stats['still_open_count'],
            )
            probe_points = list(signature.get('probe_points') or [])
            probe_action = {}
            if probe_points:
                first_point = probe_points[0]
                probe_action = {
                    'type': 'spawn',
                    'x': first_point.get('x'),
                    'y': first_point.get('y'),
                    'vx': 0.0,
                    'vy': 0.0,
                    'source': 'planned_equation_invariant_resolution',
                    'probe_label': first_point.get('label'),
                    'invariant_key': invariant.get('key'),
                }
            priority = max(
                0.35,
                min(
                    0.98,
                    0.91
                    + min(0.05, 0.012 * int(invariant.get('support_count', 0) or 0))
                    + min(0.03, 0.04 * float(invariant.get('mean_score', 0.0) or 0.0))
                    - min(0.24, 0.08 * outcome_stats['attempt_count']),
                ),
            )
            theory_kind = self._equation_invariant_theory_kind(invariant)
            labels = self._equation_invariant_variant_labels(invariant, theory_kind)
            recommendations.append({
                'theory_kind': theory_kind,
                'experiment_kind': 'equation_invariant_exponent_resolution',
                'priority': round(priority, 3),
                'family_status': str(invariant.get('status', 'unknown')),
                'target_context': 'invariant_source_context',
                'source_context': invariant.get('context'),
                'avoid_contexts': [],
                'reason': (
                    'resolve robust equation invariant exponent: '
                    f"{invariant.get('robust_claim')}"
                ),
                'expected_result': (
                    'near, mid, and far residual magnitudes should select one '
                    'distance exponent while keeping the same vector basis'
                ),
                'falsifies_if': invariant.get('falsifies_if'),
                'proof_evidence': {
                    'support_count': int(invariant.get('support_count', 0) or 0),
                    'context_count': 1,
                    'proof_rate': 0.0,
                    'mean_score': float(invariant.get('mean_score', 0.0) or 0.0),
                    'leak_count': int(invariant.get('leak_count', 0) or 0),
                    'parameter_candidate_count': len(candidates),
                    'attempt_count': outcome_stats['attempt_count'],
                    'still_open_count': outcome_stats['still_open_count'],
                },
                'invariant_key': invariant.get('key'),
                'equation_invariant': invariant,
                'disagreement_signature': signature,
                'primary_theory_label': labels[0] if labels else None,
                'rival_theory_kinds': [],
                'rival_theory_labels': labels[1:],
                'probe_action': _rounded_dict(probe_action),
                'suggested_campaign': {
                    'command_family': 'equation_campaign',
                    'world_selection': 'invariant_source_context',
                    'enable_equation_probes': True,
                },
            })
        recommendations.sort(
            key=lambda item: (
                item['priority'],
                item['proof_evidence']['support_count'],
                item.get('invariant_key', ''),
            ),
            reverse=True,
        )
        return recommendations[:limit]

    def _invariant_resolution_outcome_stats(self, invariant_key: str) -> dict[str, int]:
        outcomes = [
            outcome for outcome in self.planned_outcomes
            if outcome.get('experiment_kind') == 'equation_invariant_exponent_resolution'
            and outcome.get('invariant_key') == invariant_key
        ]
        return {
            'attempt_count': len(outcomes),
            'still_open_count': sum(
                1 for outcome in outcomes
                if outcome.get('outcome') == 'invariant_resolution_still_unresolved'
            ),
            'selected_count': sum(
                1 for outcome in outcomes
                if outcome.get('outcome') in {
                    'invariant_exponent_selected',
                    'invariant_rival_exponent_selected',
                }
            ),
        }

    def _equation_invariant_resolution_signature(
        self,
        invariant: dict[str, Any],
        refinement_level: int = 0,
    ) -> dict[str, Any]:
        parameters = dict(invariant.get('dominant_parameters') or {})
        center_x = _coerce_float(parameters.get('center_x'), 10.0)
        center_y = _coerce_float(parameters.get('center_y'), 10.0)
        near_distance, mid_distance, far_distance = (
            self._equation_invariant_probe_distances(invariant)
        )
        probe_points = [
            self._invariant_radial_probe_point(
                'near_exponent_ratio',
                center_x,
                center_y,
                near_distance,
            ),
            self._invariant_radial_probe_point(
                'mid_log_slope',
                center_x,
                center_y,
                mid_distance,
            ),
            self._invariant_radial_probe_point(
                'far_exponent_ratio',
                center_x,
                center_y,
                far_distance,
            ),
        ]
        if refinement_level > 0:
            refined = self._refine_probe_points(
                'distance_exponent_race',
                probe_points,
                refinement_level,
            )
            if refined:
                probe_points = refined
        exponents = [
            float(candidate['value'])
            for candidate in list(invariant.get('parameter_candidates') or [])
            if isinstance(candidate.get('value'), (int, float))
        ]
        ratio_denominator = max(far_distance, 1e-6)
        expected_ratios = {
            str(exponent): round((near_distance / ratio_denominator) ** exponent, 6)
            for exponent in exponents
        }
        return {
            'mode': 'distance_exponent_race',
            'resolution_source': 'robust_equation_invariant',
            'invariant_key': invariant.get('key'),
            'question': (
                'Which candidate distance exponent preserves the near/mid/far '
                'residual magnitude ratios for this repeated invariant?'
            ),
            'candidate_exponents': exponents,
            'expected_far_over_near_ratio_by_exponent': expected_ratios,
            'probe_points': probe_points,
            'rival_predictions': [
                self._equation_invariant_prediction_signature(
                    invariant,
                    exponent,
                    expected_ratios.get(str(exponent), 0.0),
                )
                for exponent in exponents[:4]
            ],
            'refinement_level': int(refinement_level),
            'refinement_strategy': (
                self._refinement_strategy('distance_exponent_race')
                if refinement_level > 0
                else 'near/mid/far samples magnify exponent-ratio differences'
            ),
        }

    def _equation_invariant_probe_distances(
        self,
        invariant: dict[str, Any],
    ) -> tuple[float, float, float]:
        parameters = dict(invariant.get('dominant_parameters') or {})
        radius = parameters.get('cutoff_radius')
        if isinstance(radius, (int, float)) and radius > 1.5:
            near = max(0.75, min(float(radius) * 0.22, 2.0))
            mid = max(near * 1.8, min(float(radius) * 0.55, 4.5))
            far = max(mid * 1.3, min(float(radius) * 0.9, 8.0))
            return (round(near, 6), round(mid, 6), round(far, 6))
        return (1.25, 3.5, 7.5)

    def _invariant_radial_probe_point(
        self,
        label: str,
        center_x: float,
        center_y: float,
        distance: float,
        world_width: float = 20.0,
        world_height: float = 20.0,
    ) -> dict[str, Any]:
        x = min(max(center_x + distance, 1.0), world_width - 1.0)
        y = min(max(center_y, 1.0), world_height - 1.0)
        return {
            'label': label,
            'x': round(x, 6),
            'y': round(y, 6),
            'distance_from_center': round(abs(x - center_x), 6),
        }

    def _equation_invariant_prediction_signature(
        self,
        invariant: dict[str, Any],
        exponent: float,
        expected_ratio: float,
    ) -> dict[str, Any]:
        theory_kind = self._equation_invariant_theory_kind(invariant)
        return {
            'theory_key': (
                f"{invariant.get('key', 'equation_invariant')}:"
                f"separation^-{exponent}"
            ),
            'theory_kind': theory_kind,
            'score': invariant.get('mean_score', 0.0),
            'distance_exponent': exponent,
            'prediction': (
                f"near/far residual ratio should follow separation^-{exponent} "
                f"(expected far-over-near {expected_ratio})"
            ),
            'falsified_if': (
                f"observed log-slope rejects exponent {exponent} while another "
                'candidate preserves the vector-basis residuals'
            ),
            'mode': 'distance_exponent_race',
        }

    def _equation_invariant_theory_kind(self, invariant: dict[str, Any]) -> str:
        vector_basis = str(invariant.get('vector_basis') or '')
        law_family = str(invariant.get('law_family') or '')
        relation = (
            'perpendicular'
            if 'perpendicular' in vector_basis
            else 'direction'
        )
        if law_family == 'localized_tapered_power':
            return f'tapered_distance_{relation}_residual'
        return f'distance_scaled_{relation}_residual'

    def _equation_invariant_variant_labels(
        self,
        invariant: dict[str, Any],
        theory_kind: str,
    ) -> list[str]:
        family_kind = self._family_key(theory_kind)
        labels = []
        for candidate in list(invariant.get('parameter_candidates') or []):
            exponent = candidate.get('value')
            if isinstance(exponent, (int, float)):
                labels.append(f'{family_kind}/separation^-{exponent}')
        return labels

    def proof_certificates(self, limit: int = 5) -> list[dict]:
        status_priority = {
            'domain_limited': 5,
            'needs_counterexample': 4,
            'local': 3,
            'reusable': 2,
            'established': 1,
            'provisional': 0,
        }
        certificates = [
            family.proof_certificate
            for family in self.families.values()
        ]
        certificates.sort(
            key=lambda item: (
                status_priority.get(item['status'], 0),
                item['support']['support_count'],
                item['support']['proof_rate'],
                item['support']['mean_score'],
                item['theory_kind'],
            ),
            reverse=True,
        )
        return certificates[:limit]

    def disagreement_experiments(self, limit: int = 5) -> list[dict]:
        groups: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
        for record in self.disagreement_records:
            families = tuple(record.get('family_kinds', []))
            labels = tuple(record.get('rival_labels', []))
            key = (str(record.get('mode', 'unknown')), families if len(families) > 1 else labels)
            group = groups.setdefault(
                key,
                {
                    'latest': record,
                    'occurrences': 0,
                    'contexts': set(),
                },
            )
            group['latest'] = record
            group['occurrences'] += 1
            context = record.get('context')
            if context:
                group['contexts'].add(str(context))

        recommendations = []
        for group in groups.values():
            record = group['latest']
            family_kinds = list(record.get('family_kinds', []))
            rival_labels = list(record.get('rival_labels', []))
            primary = family_kinds[0] if family_kinds else 'model_disagreement'
            mode = str(record.get('mode', 'unknown'))
            outcome_stats = self._disagreement_outcome_stats(
                primary=primary,
                mode=mode,
                primary_label=rival_labels[0] if rival_labels else primary,
            )
            base_priority = {
                'taper_shape_vs_hard_boundary': 0.96,
                'cutoff_boundary_vs_smooth_falloff': 0.95,
                'distance_exponent_race': 0.93,
                'vector_direction_disagreement': 0.86,
            }.get(mode, 0.82)
            occurrence_bonus = min(0.04, 0.015 * group['occurrences'])
            context_bonus = min(0.03, 0.015 * len(group['contexts']))
            attempted_penalty = min(0.28, 0.08 * outcome_stats['attempt_count'])
            unresolved_penalty = min(0.32, 0.16 * outcome_stats['still_open_count'])
            rival_penalty = min(0.22, 0.22 * outcome_stats['rival_confirmed_count'])
            target_penalty = min(0.12, 0.12 * outcome_stats['target_confirmed_count'])
            priority = max(
                0.05,
                min(
                    1.0,
                    base_priority
                    + occurrence_bonus
                    + context_bonus
                    - attempted_penalty
                    - unresolved_penalty
                    - rival_penalty
                    - target_penalty,
                ),
            )
            rival_predictions = list(record.get('rival_predictions', []))
            falsifies_if = '; '.join(
                str(item.get('falsified_if'))
                for item in rival_predictions[:3]
                if item.get('falsified_if')
            )
            if not falsifies_if:
                falsifies_if = 'one rival explains the next samples while another loses predictive power'
            disagreement_signature = self._refined_disagreement_signature(
                record,
                outcome_stats,
            )
            if outcome_stats['still_open_count'] > 0:
                reason = (
                    f"refine unresolved {mode}: "
                    f"{record.get('question', 'rival theories still disagree')}"
                )
                stagnation_status = 'needs_refinement'
            elif outcome_stats['rival_confirmed_count'] > 0:
                reason = (
                    f"recheck narrowed {mode} only after domain revision: "
                    f"{record.get('question', 'a rival has already won once')}"
                )
                stagnation_status = 'rival_recently_confirmed'
            else:
                reason = f"resolve {mode}: {record.get('question', 'rival theories disagree')}"
                stagnation_status = 'fresh'
            recommendations.append({
                'theory_kind': primary,
                'experiment_kind': 'model_disagreement_probe',
                'priority': round(priority, 3),
                'family_status': 'disagreement',
                'target_context': 'recorded_disagreement_context',
                'source_context': record.get('context'),
                'source_contexts': sorted(group['contexts']),
                'avoid_contexts': [],
                'reason': reason,
                'expected_result': record.get('expected_contrast'),
                'falsifies_if': falsifies_if,
                'proof_evidence': {
                    'support_count': group['occurrences'],
                    'context_count': len(group['contexts']),
                    'proof_rate': 0.0,
                    'mean_score': round(float(record.get('mean_rival_score', 0.0) or 0.0), 3),
                    'attempt_count': outcome_stats['attempt_count'],
                    'still_open_count': outcome_stats['still_open_count'],
                    'rival_confirmed_count': outcome_stats['rival_confirmed_count'],
                    'target_confirmed_count': outcome_stats['target_confirmed_count'],
                },
                'stagnation_status': stagnation_status,
                'disagreement_signature': disagreement_signature,
                'primary_theory_label': rival_labels[0] if rival_labels else primary,
                'rival_theory_kinds': family_kinds[1:],
                'rival_theory_labels': rival_labels[1:],
                'probe_action': record.get('action', {}),
            })
        recommendations.sort(
            key=lambda item: (
                item['priority'],
                item['proof_evidence']['support_count'],
                item['theory_kind'],
            ),
            reverse=True,
        )
        return self._diversified_experiments(recommendations, limit=limit)

    def _diversified_experiments(
        self,
        recommendations: list[dict[str, Any]],
        *,
        limit: int,
        max_per_bucket: int = 2,
    ) -> list[dict[str, Any]]:
        """Keep one unresolved question from monopolizing the next-run queue."""
        if limit <= 0:
            return []
        selected = []
        selected_ids = set()
        bucket_counts: dict[tuple[str, ...], int] = {}
        for recommendation in recommendations:
            bucket = self._experiment_diversity_bucket(recommendation)
            if bucket_counts.get(bucket, 0) >= max_per_bucket:
                continue
            selected.append(recommendation)
            selected_ids.add(id(recommendation))
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
            if len(selected) >= limit:
                return selected
        for recommendation in recommendations:
            if id(recommendation) in selected_ids:
                continue
            selected.append(recommendation)
            if len(selected) >= limit:
                break
        return selected

    def _experiment_diversity_bucket(
        self,
        recommendation: dict[str, Any],
    ) -> tuple[str, ...]:
        experiment_kind = str(recommendation.get('experiment_kind', 'unknown'))
        operator_key = str(recommendation.get('operator_prior_key') or '')
        if operator_key:
            return (experiment_kind, 'operator_prior', operator_key)
        signature = dict(recommendation.get('disagreement_signature') or {})
        mode = str(signature.get('mode') or '')
        if experiment_kind == 'model_disagreement_probe' and mode:
            return (experiment_kind, mode)
        invariant_key = str(recommendation.get('invariant_key') or '')
        if invariant_key:
            return (experiment_kind, 'equation_invariant', invariant_key)
        return (
            experiment_kind,
            str(recommendation.get('theory_kind', 'unknown')),
            str(recommendation.get('target_context', 'unknown')),
        )

    def next_experiments(self, limit: int = 5) -> list[dict]:
        recommendations = [
            family.experiment_recommendation()
            for family in self.families.values()
        ]
        recommendations.extend(self.disagreement_experiments(limit=limit * 2))
        recommendations.extend(
            self.equation_invariant_resolution_experiments(limit=limit * 2)
        )
        recommendations.extend(self.operator_prior_claim_experiments(limit=limit * 2))
        recommendations.extend(self.operator_prior_repair_experiments(limit=limit * 2))
        recommendations.extend(self.operator_prior_validation_experiments(limit=limit * 2))
        recommendations.sort(
            key=lambda item: (
                item['priority'],
                item['proof_evidence']['support_count'],
                item['theory_kind'],
            ),
            reverse=True,
        )
        return recommendations[:limit]

    def planned_experiments(
        self,
        world_types: list[str],
        object_counts: list[int],
        steps: int,
        seed_start: int = 0,
        limit: int = 5,
    ) -> list[dict]:
        plans = []
        used_cases = set()
        for recommendation in self.next_experiments(limit=limit * 2):
            context = self._select_plan_context(recommendation, world_types)
            object_count = object_counts[0] if object_counts else 5
            seed = self._next_seed_for_context(context, seed_start, used_cases)
            plan = {
                'theory_kind': recommendation['theory_kind'],
                'experiment_kind': recommendation['experiment_kind'],
                'priority': recommendation['priority'],
                'world_type': context,
                'seed': seed,
                'object_count': object_count,
                'steps': steps,
                'hidden_holdout': context == 'hidden_procedural',
                'reason': recommendation['reason'],
                'expected_result': recommendation['expected_result'],
                'falsifies_if': recommendation['falsifies_if'],
                'source_status': recommendation['family_status'],
            }
            if recommendation.get('experiment_kind') == 'model_disagreement_probe':
                plan['disagreement_signature'] = recommendation.get(
                    'disagreement_signature',
                    {},
                )
                plan['primary_theory_label'] = recommendation.get('primary_theory_label')
                plan['rival_theory_kinds'] = recommendation.get('rival_theory_kinds', [])
                plan['rival_theory_labels'] = recommendation.get('rival_theory_labels', [])
                plan['probe_action'] = recommendation.get('probe_action', {})
            if recommendation.get('experiment_kind') == 'equation_invariant_exponent_resolution':
                plan['invariant_key'] = recommendation.get('invariant_key')
                plan['equation_invariant'] = recommendation.get(
                    'equation_invariant',
                    {},
                )
                plan['disagreement_signature'] = recommendation.get(
                    'disagreement_signature',
                    {},
                )
                plan['primary_theory_label'] = recommendation.get('primary_theory_label')
                plan['rival_theory_kinds'] = recommendation.get('rival_theory_kinds', [])
                plan['rival_theory_labels'] = recommendation.get('rival_theory_labels', [])
                plan['probe_action'] = recommendation.get('probe_action', {})
            if recommendation.get('probe_action'):
                plan['probe_action'] = _rounded_dict(
                    dict(recommendation.get('probe_action') or {})
                )
            if recommendation.get('experiment_kind') in {
                'operator_prior_refinement_validation',
                'operator_prior_domain_repair',
                'operator_prior_hidden_holdout_counterexample',
                'operator_prior_domain_predicate_validation',
            }:
                plan['operator_prior_key'] = recommendation.get('operator_prior_key')
                plan['operator_prior_kind'] = recommendation.get('operator_prior_kind')
                plan['operator_prior_parameters'] = recommendation.get(
                    'operator_prior_parameters',
                    {},
                )
                plan['operator_prior_domain'] = recommendation.get(
                    'operator_prior_domain',
                    {},
                )
                if recommendation.get('failure_context'):
                    plan['failure_context'] = recommendation.get('failure_context')
                if recommendation.get('operator_prior_anomaly'):
                    plan['operator_prior_anomaly'] = recommendation.get(
                        'operator_prior_anomaly',
                        {},
                    )
                if recommendation.get('operator_prior_claim'):
                    plan['operator_prior_claim'] = recommendation.get(
                        'operator_prior_claim',
                        {},
                    )
            case_key = (plan['world_type'], plan['seed'], plan['object_count'])
            if case_key in used_cases:
                continue
            used_cases.add(case_key)
            plans.append(plan)
            if len(plans) >= limit:
                break
        return plans

    def resource_efficiency_report(
        self,
        recommended_record_window: int = 96,
        recommended_operator_window: int = 192,
    ) -> dict[str, Any]:
        """Report how close memory is to bounded long-run storage."""
        report = resource_efficiency_report(
            records=self.records,
            operator_prior_outcomes=self.operator_prior_outcomes,
            compressed_shards=self.compressed_experience_shards,
            recommended_record_window=recommended_record_window,
            recommended_operator_window=recommended_operator_window,
        )
        report['canonical_law_compression'] = self.canonical_law_compression_report()
        return report

    def canonical_law_compression_report(self) -> dict[str, Any]:
        """Report reusable-law compression status separately from raw event shards."""
        return canonical_law_compression_report(
            canonical_law_shards=self.canonical_law_shards,
        )

    def compact_canonical_laws(
        self,
        *,
        source: str = 'manual_canonical_law_compaction',
    ) -> dict[str, Any]:
        """Persist compact canonical summaries of repeated equations and laws."""
        shard = build_canonical_law_shard(
            domain_world_records=self.domain_world_records,
            autonomous_scientist_records=self.autonomous_scientist_records,
            arithmetic_rediscovery_records=self.arithmetic_rediscovery_records,
            operator_prior_outcomes=self.operator_prior_outcomes,
            source=source,
        )
        if shard.get('canonical_laws'):
            seen = {
                existing.get('shard_id')
                for existing in self.canonical_law_shards
            }
            if shard.get('shard_id') not in seen:
                self.canonical_law_shards.append(shard)
        return self.canonical_law_compression_report()

    def compact_experience(
        self,
        *,
        keep_recent_records: int = 96,
        keep_recent_operator_outcomes: int = 192,
        max_operator_anchors: int = 2,
        source: str = 'manual_compaction',
        force_summary: bool = False,
    ) -> dict[str, Any]:
        """
        Move older evidence into quantized shards while retaining recent detail.

        Operator prior outcomes keep a tiny set of raw anchor examples per
        operator so later claim/repair logic still has concrete successes and
        failures to inspect.
        """
        keep_recent_records = max(0, keep_recent_records)
        keep_recent_operator_outcomes = max(0, keep_recent_operator_outcomes)
        older_records = (
            self.records[:-keep_recent_records]
            if keep_recent_records
            else list(self.records)
        )
        recent_records = (
            self.records[-keep_recent_records:]
            if keep_recent_records
            else []
        )
        retained_anchor_outcomes = [
            outcome for outcome in self.operator_prior_outcomes
            if outcome.get('retention_role') == 'operator_anchor'
        ]
        compactable_outcomes = [
            outcome for outcome in self.operator_prior_outcomes
            if outcome.get('retention_role') != 'operator_anchor'
        ]
        older_outcomes = (
            compactable_outcomes[:-keep_recent_operator_outcomes]
            if keep_recent_operator_outcomes
            else list(compactable_outcomes)
        )
        recent_outcomes = (
            compactable_outcomes[-keep_recent_operator_outcomes:]
            if keep_recent_operator_outcomes
            else []
        )
        summary_records = list(older_records)
        summary_outcomes = list(older_outcomes)
        summary_only = False
        if force_summary and not summary_records and not summary_outcomes:
            summary_records = list(self.records)
            summary_outcomes = list(compactable_outcomes)
            summary_only = True
        if summary_records or summary_outcomes:
            shard = build_compressed_experience_shard(
                records=list(summary_records),
                operator_prior_outcomes=list(summary_outcomes),
                source=source,
            )
            if summary_only:
                shard['summary_only'] = True
            seen = {
                existing.get('shard_id')
                for existing in self.compressed_experience_shards
            }
            if shard.get('shard_id') not in seen:
                self.compressed_experience_shards.append(shard)
        anchor_indexes = operator_outcome_anchor_indexes(
            list(older_outcomes),
            max_per_operator=max_operator_anchors,
        )
        anchor_outcomes = [
            {**outcome, 'retention_role': 'operator_anchor'}
            for index, outcome in enumerate(older_outcomes)
            if index in anchor_indexes
        ]
        self.records = list(recent_records)
        self.operator_prior_outcomes = self._dedupe_dict_rows(
            [*retained_anchor_outcomes, *anchor_outcomes, *recent_outcomes]
        )
        return self.resource_efficiency_report(
            recommended_record_window=keep_recent_records,
            recommended_operator_window=(
                keep_recent_operator_outcomes
                + max(0, len(anchor_outcomes))
            ),
        )

    def to_dict(self) -> dict:
        return {
            'version': 1,
            'records': list(self.records),
            'planned_outcomes': list(self.planned_outcomes),
            'disagreement_records': list(self.disagreement_records),
            'operator_prior_outcomes': list(self.operator_prior_outcomes),
            'equation_case_records': list(self.equation_case_records),
            'domain_world_records': list(self.domain_world_records),
            'autonomous_scientist_records': list(self.autonomous_scientist_records),
            'arithmetic_rediscovery_records': list(
                self.arithmetic_rediscovery_records
            ),
            'compressed_experience_shards': list(self.compressed_experience_shards),
            'canonical_law_shards': list(self.canonical_law_shards),
            'families': {
                key: family.to_dict()
                for key, family in self.families.items()
            },
            'resource_efficiency': self.resource_efficiency_report(),
            'canonical_law_compression': self.canonical_law_compression_report(),
            'reusable_families': self.reusable_families(),
            'family_evaluations': self.family_evaluations(),
            'proof_gaps': self.proof_gaps(),
            'proof_certificates': self.proof_certificates(),
            'self_authored_equations': self.self_authored_equations(),
            'discovery_readiness': self.discovery_readiness_report(),
            'discovery_evidence_dossier': self.discovery_evidence_dossier(),
            'first_principles_basis': self.first_principles_basis(),
            'adaptive_dimension_agenda': self.adaptive_dimension_agenda(),
            'algebraic_foundation_baseline': self.algebraic_foundation_baseline(),
            'algebraic_expression_agenda': self.algebraic_expression_agenda(),
            'math_domain_curriculum': self.math_domain_curriculum(),
            'domain_curriculum_agenda': self.domain_curriculum_agenda(),
            'domain_world_blueprints': self.domain_world_blueprints(),
            'domain_world_discoveries': self.domain_world_discovery_reports(),
            'domain_world_transfer_evidence': self.domain_world_transfer_evidence(),
            'domain_transfer_experiments': self.domain_transfer_experiments(),
            'autonomous_scientist_evidence': self.autonomous_scientist_evidence(),
            'arithmetic_rediscovery_evidence': self.arithmetic_rediscovery_evidence(),
            'latest_autonomous_scientist_report': self.latest_autonomous_scientist_report(),
            'representation_agenda': self.representation_agenda(),
            'generated_operator_priors': self.generated_operator_priors(),
            'operator_prior_budget_report': self.operator_prior_budget_report(),
            'operator_prior_invariant_consolidations': (
                self.operator_prior_invariant_consolidations()
            ),
            'equation_invariant_resolution_experiments': (
                self.equation_invariant_resolution_experiments()
            ),
            'operator_prior_feedback': self.operator_prior_feedback(),
            'operator_prior_domains': self.operator_prior_domains(),
            'operator_prior_anomalies': self.operator_prior_anomalies(),
            'operator_prior_discovery_claims': self.operator_prior_discovery_claims(),
            'operator_prior_discovery_chains': self.operator_prior_discovery_chains(),
            'operator_prior_claim_experiments': self.operator_prior_claim_experiments(),
            'operator_prior_repair_experiments': self.operator_prior_repair_experiments(),
            'operator_prior_validation_experiments': self.operator_prior_validation_experiments(),
            'disagreement_experiments': self.disagreement_experiments(),
            'generalization_gaps': self.generalization_gaps(),
            'domain_revisions': self.domain_revisions(),
            'next_experiments': self.next_experiments(),
            'planned_experiments': self.planned_experiments(
                world_types=[
                    'standard',
                    'sideways_wind',
                    'vortex',
                    'inverse_square_repulsion',
                    'localized_gravity',
                    'time_varying',
                ],
                object_counts=[5],
                steps=240,
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'CumulativeTheoryMemory':
        memory = cls()
        memory.records = list(data.get('records', []))
        memory.planned_outcomes = list(data.get('planned_outcomes', []))
        memory.disagreement_records = list(data.get('disagreement_records', []))
        memory.operator_prior_outcomes = list(data.get('operator_prior_outcomes', []))
        memory.equation_case_records = list(data.get('equation_case_records', []))
        memory.domain_world_records = list(data.get('domain_world_records', []))
        memory.autonomous_scientist_records = list(
            data.get('autonomous_scientist_records', [])
        )
        memory.arithmetic_rediscovery_records = list(
            data.get('arithmetic_rediscovery_records', [])
        )
        memory.compressed_experience_shards = list(
            data.get('compressed_experience_shards', [])
        )
        memory.canonical_law_shards = list(data.get('canonical_law_shards', []))
        memory.families = {
            str(key): TheoryFamily.from_dict(family_data)
            for key, family_data in data.get('families', {}).items()
        }
        return memory

    @classmethod
    def load(cls, path: str | Path) -> 'CumulativeTheoryMemory':
        memory_path = Path(path)
        if not memory_path.exists():
            return cls()
        if memory_path.stat().st_size == 0:
            return cls()
        with memory_path.open('r', encoding='utf-8') as handle:
            return cls.from_dict(json.load(handle))

    def save(self, path: str | Path):
        memory_path = Path(path)
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        with memory_path.open('w', encoding='utf-8') as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)

    @staticmethod
    def _dedupe_dict_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[str] = set()
        deduped = []
        for row in rows:
            key = json.dumps(row, sort_keys=True, separators=(',', ':'), default=str)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)
        return deduped

    def _disagreement_record(
        self,
        context: str,
        seed: int,
        probe_plan: dict,
    ) -> dict[str, Any] | None:
        signature = dict(probe_plan.get('disagreement_signature') or {})
        rival_predictions = list(signature.get('rival_predictions') or [])
        if not signature.get('mode') or len(rival_predictions) < 2:
            return None
        family_kinds = []
        rival_labels = []
        for prediction in rival_predictions:
            kind = self._family_key(str(prediction.get('theory_kind', 'unknown')))
            if kind not in family_kinds:
                family_kinds.append(kind)
            label = self._rival_label(prediction)
            if label not in rival_labels:
                rival_labels.append(label)
        scores = [
            float(prediction.get('score', 0.0) or 0.0)
            for prediction in rival_predictions
            if isinstance(prediction.get('score', 0.0), (int, float))
        ]
        mean_score = sum(scores) / len(scores) if scores else 0.0
        return {
            'context': context,
            'seed': seed,
            'mode': signature.get('mode'),
            'question': signature.get('question'),
            'family_kinds': family_kinds,
            'rival_labels': rival_labels,
            'theory_keys': list(probe_plan.get('theory_keys') or []),
            'reason': probe_plan.get('reason'),
            'expected_contrast': probe_plan.get('expected_contrast'),
            'action': _rounded_dict(dict(probe_plan.get('action') or {})),
            'probe_points': _rounded_value(signature.get('probe_points') or []),
            'rival_predictions': _rounded_value(rival_predictions),
            'mean_rival_score': round(mean_score, 3),
            'disagreement_signature': _rounded_dict(signature),
        }

    def _disagreement_outcome_stats(
        self,
        primary: str,
        mode: str,
        primary_label: str,
    ) -> dict[str, int]:
        matching = [
            outcome for outcome in self.planned_outcomes
            if outcome.get('experiment_kind') == 'model_disagreement_probe'
            and outcome.get('theory_kind') == primary
            and outcome.get('disagreement_mode') == mode
            and (
                not outcome.get('primary_theory_label')
                or outcome.get('primary_theory_label') == primary_label
            )
        ]
        return {
            'attempt_count': len(matching),
            'still_open_count': sum(
                1 for outcome in matching
                if outcome.get('outcome') == 'disagreement_still_open'
            ),
            'rival_confirmed_count': sum(
                1 for outcome in matching
                if outcome.get('outcome') == 'rival_confirmed'
            ),
            'target_confirmed_count': sum(
                1 for outcome in matching
                if outcome.get('outcome') == 'target_confirmed'
            ),
        }

    def _refined_disagreement_signature(
        self,
        record: dict,
        outcome_stats: dict[str, int],
    ) -> dict[str, Any]:
        signature = _rounded_dict(dict(record.get('disagreement_signature') or {}))
        refinement_level = int(outcome_stats.get('still_open_count', 0) or 0)
        signature['refinement_level'] = refinement_level
        if refinement_level <= 0:
            return signature

        mode = str(signature.get('mode', 'unknown'))
        points = list(signature.get('probe_points') or [])
        refined_points = self._refine_probe_points(mode, points, refinement_level)
        if refined_points:
            signature['probe_points'] = refined_points
            signature['refinement_strategy'] = self._refinement_strategy(mode)
        return signature

    def _refine_probe_points(
        self,
        mode: str,
        points: list[dict],
        refinement_level: int,
    ) -> list[dict]:
        if not points:
            return []
        radial_points = [
            point for point in points
            if isinstance(point.get('distance_from_center'), (int, float))
            and isinstance(point.get('x'), (int, float))
            and isinstance(point.get('y'), (int, float))
        ]
        if not radial_points:
            return points

        center_x = float(radial_points[0]['x']) - float(radial_points[0]['distance_from_center'])
        center_y = float(radial_points[0]['y'])
        distances = sorted(float(point['distance_from_center']) for point in radial_points)
        min_distance = distances[0]
        max_distance = distances[-1]
        expansion = 1.0 + min(0.6, 0.2 * refinement_level)

        if mode == 'distance_exponent_race':
            targets = [
                ('very_near_center', max(0.35, min_distance * 0.55)),
                ('mid_log_check', max(min_distance * 1.4, (min_distance + max_distance) * 0.5)),
                ('very_far_from_center', max_distance * expansion),
            ]
        elif mode == 'taper_shape_vs_hard_boundary':
            targets = [
                ('deep_inside_region', max(0.35, min_distance * 0.65)),
                ('half_strength_region', max(min_distance, max_distance * 0.55)),
                ('boundary_outer_tail', max_distance * expansion),
            ]
        elif mode == 'cutoff_boundary_vs_smooth_falloff':
            targets = [
                ('inside_margin', max(0.35, min_distance * 0.75)),
                ('boundary_margin', max_distance),
                ('outside_tail', max_distance * expansion),
            ]
        else:
            return points
        return [
            {
                'label': label,
                'x': round(center_x + distance, 6),
                'y': round(center_y, 6),
                'distance_from_center': round(distance, 6),
                'refined_from': mode,
            }
            for label, distance in targets
        ]

    def _refinement_strategy(self, mode: str) -> str:
        strategies = {
            'distance_exponent_race': (
                'spread near/mid/far samples to magnify exponent-ratio differences'
            ),
            'taper_shape_vs_hard_boundary': (
                'sample deep inside, half-strength, and beyond-boundary regions'
            ),
            'cutoff_boundary_vs_smooth_falloff': (
                'sample inside margin, boundary margin, and outside tail'
            ),
        }
        return strategies.get(mode, 'reuse disagreement probes with a sharper contrast')

    def _rival_label(self, prediction: dict) -> str:
        family_kind = self._family_key(str(prediction.get('theory_kind', 'unknown')))
        prediction_text = str(prediction.get('prediction', ''))
        if 'separation^-' in prediction_text:
            exponent = prediction_text.split('separation^-', 1)[1].split()[0].rstrip('.,;')
            return f'{family_kind}/separation^-{exponent}'
        if 'graded residual' in prediction_text:
            return f'{family_kind}/graded'
        if 'flat local residual' in prediction_text:
            return f'{family_kind}/hard_boundary'
        if 'smooth nonzero falloff' in prediction_text:
            return f'{family_kind}/smooth_falloff'
        if 'quarter turn' in prediction_text:
            return f'{family_kind}/quarter_turn'
        if 'aligns with the center vector' in prediction_text:
            return f'{family_kind}/center_aligned'
        return family_kind

    def _select_plan_context(self, recommendation: dict, world_types: list[str]) -> str:
        target = recommendation.get('target_context')
        avoid = set(recommendation.get('avoid_contexts', []))
        candidates = list(world_types or [])
        if target == 'recorded_disagreement_context':
            source_context = recommendation.get('source_context')
            if source_context:
                return str(source_context)
            return candidates[0] if candidates else 'standard'
        if target == 'invariant_source_context':
            source_context = recommendation.get('source_context')
            if source_context:
                return str(source_context)
            invariant = dict(recommendation.get('equation_invariant') or {})
            context = invariant.get('context')
            if context:
                return str(context)
            return candidates[0] if candidates else 'standard'
        if target == 'operator_prior_unseen_context':
            for world_type in candidates:
                if world_type not in avoid:
                    return world_type
            return 'hidden_procedural'
        if target == 'operator_prior_failure_context':
            failure_context = recommendation.get('failure_context')
            if failure_context:
                return str(failure_context)
            anomaly = dict(recommendation.get('operator_prior_anomaly') or {})
            failure_context = anomaly.get('failure_context')
            if failure_context:
                return str(failure_context)
            return candidates[0] if candidates else 'standard'
        if target in {'unseen_world_context', 'rival_or_hidden_context'}:
            for world_type in candidates:
                if world_type not in avoid:
                    return world_type
            return 'hidden_procedural'
        if target in {'hidden_holdout', 'new_seed_or_hidden_holdout'}:
            return 'hidden_procedural'
        if target == 'known_success_and_failure_contexts':
            if avoid:
                return sorted(avoid)[0]
            return candidates[0] if candidates else 'standard'
        if target == 'same_or_similar_context':
            if avoid:
                return sorted(avoid)[0]
            return candidates[0] if candidates else 'standard'
        return candidates[0] if candidates else 'standard'

    def _planned_outcome_label(
        self,
        experiment_kind: str,
        found_family: bool,
        proof_passed: bool,
        new_context: bool,
    ) -> str:
        if experiment_kind == 'transfer_test':
            if found_family and proof_passed and new_context:
                return 'transfer_confirmed'
            if not found_family:
                return 'transfer_absent'
            return 'transfer_weak'
        if experiment_kind == 'disagreement_counterexample':
            if not found_family or not proof_passed:
                return 'counterexample_found'
            return 'counterexample_not_found'
        if experiment_kind == 'hidden_holdout_counterexample':
            if found_family and proof_passed:
                return 'holdout_survived'
            return 'counterexample_found'
        if experiment_kind in {'replication_seed', 'replication_or_holdout'}:
            if found_family and proof_passed:
                return 'replication_confirmed'
            return 'replication_failed'
        if experiment_kind == 'operator_prior_refinement_validation':
            return 'operator_prior_validation_deferred_to_prior_feedback'
        if experiment_kind == 'operator_prior_domain_repair':
            return 'operator_prior_repair_deferred_to_prior_feedback'
        if experiment_kind == 'operator_prior_hidden_holdout_counterexample':
            return 'operator_prior_holdout_deferred_to_prior_feedback'
        if experiment_kind == 'operator_prior_domain_predicate_validation':
            return 'operator_prior_domain_predicate_deferred_to_prior_feedback'
        if found_family and proof_passed:
            return 'evidence_confirmed'
        if found_family:
            return 'evidence_weak'
        return 'evidence_absent'

    def _model_disagreement_outcome_label(
        self,
        target_found: bool,
        target_proof_passed: bool,
        rival_found: bool,
        rival_proof_passed: bool,
    ) -> str:
        if target_found and target_proof_passed and rival_found and rival_proof_passed:
            return 'disagreement_still_open'
        if target_found and target_proof_passed:
            return 'target_confirmed'
        if rival_found and rival_proof_passed:
            return 'rival_confirmed'
        if target_found:
            return 'target_weak'
        if rival_found:
            return 'rival_weak'
        return 'evidence_absent'

    def _equation_invariant_resolution_outcome_label(
        self,
        target_found: bool,
        target_proof_passed: bool,
        rival_found: bool,
        rival_proof_passed: bool,
    ) -> str:
        if target_found and target_proof_passed and rival_found and rival_proof_passed:
            return 'invariant_resolution_still_unresolved'
        if target_found and target_proof_passed:
            return 'invariant_exponent_selected'
        if rival_found and rival_proof_passed:
            return 'invariant_rival_exponent_selected'
        if target_found or rival_found:
            return 'invariant_resolution_weak'
        return 'invariant_resolution_absent'

    def _matches_plan_target(self, plan: dict, theory: dict) -> bool:
        family = self._family_key(str(theory.get('theory_kind', '')))
        primary_label = plan.get('primary_theory_label')
        if primary_label:
            return self._theory_variant_label(theory) == primary_label
        return family == str(plan.get('theory_kind', 'unknown'))

    def _plan_target_scope(self, plan: dict, context: str) -> str:
        primary_label = plan.get('primary_theory_label')
        if primary_label:
            return f'{context}/{primary_label}'
        return context

    def _plan_rival_scope(self, plan: dict, context: str) -> str | None:
        labels = list(plan.get('rival_theory_labels') or [])
        families = list(plan.get('rival_theory_kinds') or [])
        if labels:
            return f"{context}/{labels[0]}"
        if families:
            return f"{context}/{families[0]}"
        return None

    def _matches_plan_rival(self, plan: dict, theory: dict) -> bool:
        family = self._family_key(str(theory.get('theory_kind', '')))
        label = self._theory_variant_label(theory)
        rival_families = set(plan.get('rival_theory_kinds') or [])
        rival_labels = set(plan.get('rival_theory_labels') or [])
        return family in rival_families or label in rival_labels

    def _theory_variant_label(self, theory: dict) -> str:
        theory_kind = str(theory.get('theory_kind', 'unknown'))
        family_kind = self._family_key(theory_kind)
        parameters = dict(theory.get('parameters') or {})
        if 'distance_scaled' in theory_kind and 'distance_exponent' in parameters:
            return f"{family_kind}/separation^-{parameters['distance_exponent']}"
        if 'tapered_distance' in theory_kind:
            return f'{family_kind}/graded'
        if 'cutoff' in theory_kind:
            return f'{family_kind}/hard_boundary'
        if 'perpendicular' in theory_kind:
            return f'{family_kind}/quarter_turn'
        if 'direction' in theory_kind:
            return f'{family_kind}/center_aligned'
        return family_kind

    def _next_seed_for_context(
        self,
        context: str,
        seed_start: int,
        used_cases: set[tuple[str, int, int]],
    ) -> int:
        used_seeds = {
            int(record.get('seed', seed_start))
            for record in self.records
            if record.get('context') == context
            and isinstance(record.get('seed'), int)
        }
        seed = seed_start
        while seed in used_seeds or any(case[0] == context and case[1] == seed for case in used_cases):
            seed += 1
        return seed

    def _report_dict(self, report) -> dict:
        return report.to_dict() if hasattr(report, 'to_dict') else dict(report or {})

    def _family_key(self, theory_kind: str) -> str:
        if theory_kind == 'generated_periodic_residual':
            return 'periodic_residual'
        if theory_kind.startswith('generated_tapered_distance_'):
            return theory_kind.replace('generated_tapered_distance_', 'tapered_distance_', 1)
        if theory_kind.startswith('generated_cutoff_'):
            return theory_kind.replace('generated_cutoff_', 'cutoff_', 1)
        if theory_kind.startswith('generated_distance_scaled_'):
            return theory_kind.replace('generated_distance_scaled_', 'distance_scaled_', 1)
        if theory_kind.startswith('local_distance_scaled_'):
            return theory_kind.replace('local_distance_scaled_', 'distance_scaled_', 1)
        if theory_kind.startswith('local_') and theory_kind.endswith('_residual'):
            return theory_kind.replace('local_', '', 1)
        return theory_kind

    def _relevant_concepts(self, theory: dict, concept_proposals: list[dict]) -> list[dict]:
        concept_keys = set(theory.get('concept_keys', []))
        return [
            concept for concept in concept_proposals
            if concept.get('key') in concept_keys
        ]

    def _relevant_operators(self, theory: dict, operator_proposals: list[dict]) -> list[dict]:
        concept_keys = set(theory.get('concept_keys', []))
        relevant = []
        for operator in operator_proposals:
            origin = str(operator.get('generated_from', ''))
            if origin in concept_keys:
                relevant.append(operator)
                continue
            if any(key.startswith(origin) for key in concept_keys):
                relevant.append(operator)
                continue
            if 'inferred' in origin and any('inferred' in key for key in concept_keys):
                relevant.append(operator)
                continue
            if (
                operator.get('operator_kind') == 'rotate_quarter_turn'
                and any('perpendicular' in key for key in concept_keys)
            ):
                relevant.append(operator)
                continue
            if (
                operator.get('operator_kind') == 'normalize_vector'
                and any('direction' in key or 'field_axis' in key for key in concept_keys)
            ):
                relevant.append(operator)
        return relevant

    def _relevant_checks(self, theory: dict, proof_checks: list[dict]) -> list[dict]:
        theory_key = str(theory.get('key', ''))
        return [
            check for check in proof_checks
            if str(check.get('key', '')).startswith(f'proof:{theory_key}:')
        ]

    def summary(self, limit: int = 5) -> str:
        lines = [
            "Cumulative theory memory:",
            f"  Runs observed: {len(self.records)}",
            f"  Theory families: {len(self.families)}",
        ]
        families = self.reusable_families()[:limit]
        if families:
            lines.append("  Reusable families:")
            for family in families:
                contexts = ','.join(family['contexts'])
                lines.append(
                    f"    {family['theory_kind']}: support={family['support_count']}, "
                    f"generalization={family['generalization_score']:.2f}, "
                    f"proof={family['proof_rate']:.2f}, "
                    f"status={family['generalization_status']}, contexts={contexts}"
                )
                lines.append(
                    f"      next: {family['next_proof_obligation']}"
                )
        certificates = self.proof_certificates(limit=limit)
        if certificates:
            lines.append("  Proof certificates:")
            for certificate in certificates:
                support = certificate['support']
                lines.append(
                    f"    {certificate['theory_kind']}: status={certificate['status']}, "
                    f"support={support['support_count']}, proof={support['proof_rate']:.2f}"
                )
                lines.append(
                    f"      accepted: {certificate['accepted_because'][0]}"
                )
                if certificate['not_universal_because']:
                    lines.append(
                        f"      limit: {certificate['not_universal_because'][0]}"
                    )
        readiness = self.discovery_readiness_report()
        lines.append(
            "  Discovery readiness: "
            f"{readiness['readiness_score']:.0%} "
            f"status={readiness['status']} "
            f"gates={readiness['passed_gate_count']}/{readiness['gate_count']}"
        )
        resource = self.resource_efficiency_report()
        lines.append(
            "  Resource efficiency: "
            f"shards={resource['compressed_shard_count']} "
            f"detail_ratio={resource['detail_reduction_ratio']:.2f} "
            f"bytes_ratio={resource['estimated_compression_ratio']:.2f} "
            f"long_run_ready={resource['long_run_ready']}"
        )
        canonical = self.canonical_law_compression_report()
        lines.append(
            "  Canonical law compression: "
            f"shards={canonical['canonical_law_shard_count']} "
            f"laws={canonical['canonical_law_count']} "
            f"robust={canonical['robust_law_count']} "
            f"bytes_ratio={canonical['estimated_law_compression_ratio']:.2f}"
        )
        arithmetic = self.arithmetic_rediscovery_evidence()
        lines.append(
            "  Arithmetic rediscovery: "
            f"coverage={arithmetic['best_coverage']:.0%} "
            f"targets={arithmetic['discovered_target_count']} "
            f"leaks={arithmetic['leaked_manifest_count']}"
        )
        if readiness['missing_gates']:
            lines.append(
                "      missing: " + ','.join(readiness['missing_gates'][:4])
            )
        dossier = dict(readiness.get('evidence_dossier') or {})
        if any(dossier.get(key) for key in ('chains', 'claims', 'planned_tests')):
            lines.append("  Discovery evidence dossier:")
            for chain in dossier.get('chains', [])[:limit]:
                lines.append(
                    f"    chain {chain['operator_kind']}: status={chain['status']}, "
                    f"steps={chain['step_count']}, next={chain['next_obligation']}"
                )
            for claim in dossier.get('claims', [])[:limit]:
                lines.append(
                    f"    claim {claim['operator_kind']}: status={claim['status']}, "
                    f"best={claim['best_score']:.2f}"
                )
            for plan in dossier.get('planned_tests', [])[:limit]:
                lines.append(
                    f"    planned {plan['experiment_kind']}: "
                    f"{plan['world_type']} seed={plan['seed']}"
                )
        dimensions = self.adaptive_dimension_agenda(limit=limit)
        if dimensions:
            lines.append("  Adaptive dimensions:")
            for dimension in dimensions:
                primitives = ','.join(dimension.get('first_principles', [])[:3])
                lines.append(
                    f"    {dimension['name']}: kind={dimension['dimension_kind']}, "
                    f"priority={dimension['priority']:.2f}, primitives={primitives}"
                )
        algebraic_foundation = self.algebraic_foundation_baseline()
        lines.append(
            "  Algebraic foundation: "
            f"families={algebraic_foundation['expression_family_count']}, "
            f"structures={algebraic_foundation['structure_count']}, "
            f"proof_obligations={algebraic_foundation['proof_obligation_count']}"
        )
        algebraic_agenda = self.algebraic_expression_agenda(limit=limit)
        if algebraic_agenda:
            lines.append("  Algebraic expression agenda:")
            for item in algebraic_agenda:
                families = ','.join(item.get('expression_families', [])[:3])
                obligations = ','.join(item.get('proof_obligations', [])[:3])
                lines.append(
                    f"    {item['signal']}: families={families}, "
                    f"priority={item['priority']:.2f}, obligations={obligations}"
                )
        authored_equations = self.self_authored_equations(limit=limit)
        if authored_equations:
            lines.append("  Self-authored equations:")
            for equation in authored_equations:
                lines.append(
                    f"    {equation['equation_kind']}: status={equation['status']}, "
                    f"support={equation['support_count']}, "
                    f"confidence={equation['confidence']:.2f}"
                )
                lines.append(f"      expression: {equation['expression']}")
        domain_curriculum = self.math_domain_curriculum()
        lines.append(
            "  Math domain curriculum: "
            f"domains={domain_curriculum['domain_count']}, "
            f"bridges={domain_curriculum['transfer_bridge_count']}, "
            f"active={domain_curriculum['coverage']['active_domain_count']}"
        )
        domain_agenda = self.domain_curriculum_agenda(limit=limit)
        if domain_agenda:
            lines.append("  Domain curriculum agenda:")
            for item in domain_agenda:
                lines.append(
                    f"    {item['domain_key']}: status={item['status']}, "
                    f"priority={item['priority']:.2f}, support={item['support_count']}"
                )
                lines.append(f"      next: {item['next_pressure']}")
        domain_worlds = self.domain_world_blueprints(limit=limit)
        if domain_worlds:
            lines.append("  Domain world blueprints:")
            for item in domain_worlds:
                lines.append(
                    f"    {item['domain_key']}: samples={item['sample_count']}, "
                    f"falsifiers={item['falsifier_count']}, "
                    f"leaks={item['leaky_observation_count']}"
                )
                targets = ','.join(item.get('transfer_targets', [])[:3]) or 'none'
                lines.append(f"      transfer: {targets}")
        domain_discoveries = self.domain_world_discovery_reports(limit=limit)
        if domain_discoveries:
            lines.append("  Domain world discoveries:")
            for item in domain_discoveries:
                equations = list(item.get('self_authored_equations') or [])
                expression = equations[0].get('expression') if equations else 'none'
                lines.append(
                    f"    {item['domain_key']}: candidates={item['candidate_count']}, "
                    f"coverage={item['benchmark_coverage']:.0%}, "
                    f"falsifiers={item['falsification_test_count']}"
                )
                lines.append(f"      expression: {expression}")
        domain_transfer_evidence = self.domain_world_transfer_evidence(limit=limit)
        if domain_transfer_evidence:
            lines.append("  Domain world transfer evidence:")
            for item in domain_transfer_evidence:
                lines.append(
                    f"    {item['source_domain']}->{item['target_domain']}: "
                    f"status={item['status']}"
                )
                source = ','.join(item.get('source_matches', [])[:3]) or 'none'
                target = ','.join(item.get('target_matches', [])[:3]) or 'none'
                lines.append(f"      basis: {source} -> {target}")
        scientist = self.latest_autonomous_scientist_report()
        if scientist:
            coverage = dict(scientist.get('coverage') or {})
            lines.append("  Autonomous scientist loop:")
            lines.append(
                f"    status={scientist.get('status')}, "
                f"invariants={coverage.get('invariant_count', 0)}, "
                f"residual_probes={coverage.get('residual_experiment_count', 0)}, "
                f"stress_worlds={coverage.get('stress_world_count', 0)}, "
                f"equations={coverage.get('authored_equation_extension_count', 0)}"
            )
            for event in list(scientist.get('live_events') or [])[:2]:
                lines.append(
                    f"      event: {event.get('event')} "
                    f"{event.get('relation_kind') or event.get('key') or ''}"
                )
        domain_transfers = self.domain_transfer_experiments(limit=limit)
        if domain_transfers:
            lines.append("  Domain transfer probes:")
            for item in domain_transfers:
                lines.append(
                    f"    {item['source_domain']}->{item['target_domain']}: "
                    f"priority={item['priority']:.2f}"
                )
                lines.append(f"      question: {item['transfer_question']}")
        disagreements = self.disagreement_experiments(limit=limit)
        if disagreements:
            lines.append("  Disagreement probes:")
            for experiment in disagreements:
                primary = experiment.get('primary_theory_label') or experiment['theory_kind']
                rivals = (
                    ','.join(experiment['rival_theory_kinds'])
                    or ','.join(experiment.get('rival_theory_labels', []))
                    or 'none'
                )
                lines.append(
                    f"    {primary} vs {rivals}: "
                    f"mode={experiment['disagreement_signature'].get('mode')}, "
                    f"priority={experiment['priority']:.2f}"
                )
        gaps = self.proof_gaps()[:limit]
        if gaps:
            lines.append("  Proof gaps:")
            for gap in gaps:
                lines.append(
                    f"    {gap['theory_kind']}: status={gap['status']}, "
                    f"proof={gap['proof_rate']:.2f}, support={gap['support_count']}"
                )
        revisions = self.domain_revisions()[:limit]
        if revisions:
            lines.append("  Domain revisions:")
            for revision in revisions:
                domain = revision['domain_hypothesis']
                included = ','.join(domain['included_contexts']) or 'none'
                excluded = ','.join(domain['excluded_contexts']) or 'none'
                lines.append(
                    f"    {revision['theory_kind']}: include={included}; exclude={excluded}"
                )
        agenda = self.representation_agenda(limit=limit)
        if agenda:
            lines.append("  Representation agenda:")
            for proposal in agenda:
                lines.append(
                    f"    {proposal['proposal_kind']}:{proposal['name']} "
                    f"for {proposal['theory_kind']} priority={proposal['priority']:.2f}"
                )
        invariants = self.operator_prior_invariant_consolidations(limit=limit)
        if invariants:
            lines.append("  Robust equation invariants:")
            for invariant in invariants:
                lines.append(
                    f"    {invariant['law_family']}:{invariant['context']} "
                    f"status={invariant['status']}, "
                    f"support={invariant['support_count']}, "
                    f"score={invariant['mean_score']:.2f}"
                )
                lines.append(f"      next: {invariant['next_experiment']}")
        invariant_resolution = self.equation_invariant_resolution_experiments(
            limit=limit,
        )
        if invariant_resolution:
            lines.append("  Equation invariant resolution:")
            for experiment in invariant_resolution:
                signature = dict(experiment.get('disagreement_signature') or {})
                exponents = ','.join(
                    str(value)
                    for value in signature.get('candidate_exponents', [])[:4]
                )
                lines.append(
                    f"    {experiment['theory_kind']}: priority="
                    f"{experiment['priority']:.2f}, exponents={exponents}"
                )
                lines.append(f"      next: {experiment['reason']}")
        feedback = self.operator_prior_feedback(limit=limit)
        if feedback:
            lines.append("  Operator prior feedback:")
            for item in feedback:
                lines.append(
                    f"    {item['operator_kind']}:{item['outcome']} "
                    f"score={item['best_score']:.2f} in {item['context']}"
                )
        claims = self.operator_prior_discovery_claims(limit=limit)
        if claims:
            lines.append("  Operator prior discovery claims:")
            for claim in claims:
                lines.append(
                    f"    {claim['operator_kind']}: status={claim['status']}, "
                    f"support={claim['proof_evidence']['confirmed_count']}, "
                    f"best={claim['proof_evidence']['best_score']:.2f}"
                )
        chains = self.operator_prior_discovery_chains(limit=limit)
        if chains:
            lines.append("  Operator prior discovery chains:")
            for chain in chains:
                lines.append(
                    f"    {chain['operator_kind']}: status={chain['status']}, "
                    f"steps={len(chain['steps'])}, next={chain['next_obligation']}"
                )
        claim_experiments = self.operator_prior_claim_experiments(limit=limit)
        if claim_experiments:
            lines.append("  Operator prior claim experiments:")
            for experiment in claim_experiments:
                lines.append(
                    f"    {experiment['experiment_kind']} for "
                    f"{experiment['operator_prior_kind']} priority="
                    f"{experiment['priority']:.2f}"
                )
        anomalies = self.operator_prior_anomalies(limit=limit)
        if anomalies:
            lines.append("  Operator prior anomalies:")
            for anomaly in anomalies:
                lines.append(
                    f"    {anomaly['operator_kind']}:{anomaly['anomaly_kind']} "
                    f"in {anomaly['failure_context']} severity={anomaly['severity']:.2f}"
                )
        domains = self.operator_prior_domains(limit=limit)
        if domains:
            lines.append("  Operator prior domains:")
            for domain in domains:
                hypothesis = domain['domain_hypothesis']
                lines.append(
                    f"    {domain['operator_kind']}: include="
                    f"{','.join(hypothesis['included_contexts']) or 'none'}; "
                    f"exclude={','.join(hypothesis['excluded_contexts']) or 'none'}"
                )
        repairs = self.operator_prior_repair_experiments(limit=limit)
        if repairs:
            lines.append("  Operator prior repair:")
            for repair in repairs:
                lines.append(
                    f"    {repair['operator_prior_kind']} priority="
                    f"{repair['priority']:.2f}, target={repair['target_context']}"
                )
        validations = self.operator_prior_validation_experiments(limit=limit)
        if validations:
            lines.append("  Operator prior validation:")
            for validation in validations:
                lines.append(
                    f"    {validation['operator_prior_kind']} priority="
                    f"{validation['priority']:.2f}, target={validation['target_context']}"
                )
        experiments = self.next_experiments(limit=limit)
        if experiments:
            lines.append("  Next experiments:")
            for experiment in experiments:
                lines.append(
                    f"    {experiment['experiment_kind']} for {experiment['theory_kind']}: "
                    f"priority={experiment['priority']:.2f}, target={experiment['target_context']}"
                )
        return "\n".join(lines)

    def operator_prior_feedback(self, limit: int = 5) -> list[dict[str, Any]]:
        feedback = list(self.operator_prior_outcomes)
        feedback.sort(
            key=lambda item: (
                item.get('best_score', 0.0),
                item.get('matching_equation_count', 0),
                item.get('operator_key', ''),
            ),
            reverse=True,
        )
        return feedback[:limit]

    def operator_prior_discovery_chains(self, limit: int = 5) -> list[dict[str, Any]]:
        """Explain the evidence chain for each invented operator prior."""
        claims = {
            claim['operator_key']: claim
            for claim in self.operator_prior_discovery_claims(
                limit=max(5, len(self.operator_prior_outcomes)),
            )
        }
        anomalies_by_key: dict[str, list[dict[str, Any]]] = {}
        for anomaly in self.operator_prior_anomalies(
            limit=max(5, len(self.operator_prior_outcomes)),
        ):
            anomalies_by_key.setdefault(str(anomaly.get('operator_key', 'unknown')), []).append(anomaly)
        outcomes_by_key: dict[str, list[dict[str, Any]]] = {}
        for outcome in self.operator_prior_outcomes:
            outcomes_by_key.setdefault(str(outcome.get('operator_key', 'unknown')), []).append(outcome)
        planned_by_key: dict[str, list[dict[str, Any]]] = {}
        for outcome in self.planned_outcomes:
            key = outcome.get('operator_prior_key')
            if key:
                planned_by_key.setdefault(str(key), []).append(outcome)
        recommendations_by_key: dict[str, list[dict[str, Any]]] = {}
        for recommendation in (
            self.operator_prior_claim_experiments(limit=max(5, len(outcomes_by_key) + 2))
            + self.operator_prior_repair_experiments(limit=max(5, len(outcomes_by_key) + 2))
            + self.operator_prior_validation_experiments(limit=max(5, len(outcomes_by_key) + 2))
        ):
            key = recommendation.get('operator_prior_key')
            if key:
                recommendations_by_key.setdefault(str(key), []).append(recommendation)

        chains = []
        all_keys = set(outcomes_by_key) | set(claims) | set(planned_by_key)
        for operator_key in all_keys:
            outcomes = outcomes_by_key.get(operator_key, [])
            if not outcomes:
                continue
            operator_kind = str(outcomes[0].get('operator_kind', 'unknown'))
            claim = claims.get(operator_key)
            status = claim.get('status') if claim else self._operator_chain_status(outcomes)
            parameters = (
                dict(claim.get('parameters') or {})
                if claim
                else self._best_refined_parameters_for_prior(operator_key)
                or dict(outcomes[0].get('parameters') or {})
            )
            steps = []
            first = outcomes[0]
            steps.append({
                'step_kind': 'operator_prior_invented',
                'context': first.get('context'),
                'seed': first.get('seed'),
                'summary': (
                    f"{operator_kind} prior entered the workbench from cumulative memory"
                ),
                'parameters': _rounded_dict(parameters),
            })
            for outcome in outcomes:
                steps.append({
                    'step_kind': 'operator_prior_tested',
                    'context': outcome.get('context'),
                    'seed': outcome.get('seed'),
                    'outcome': outcome.get('outcome'),
                    'score': outcome.get('best_score', 0.0),
                    'summary': (
                        f"feedback was {outcome.get('outcome', 'unknown')} "
                        f"with score {float(outcome.get('best_score', 0.0) or 0.0):.2f}"
                    ),
                })
            for anomaly in anomalies_by_key.get(operator_key, []):
                steps.append({
                    'step_kind': 'anomaly_detected',
                    'context': anomaly.get('failure_context'),
                    'outcome': anomaly.get('failure_outcome'),
                    'summary': anomaly.get('question'),
                    'severity': anomaly.get('severity'),
                })
            for planned in planned_by_key.get(operator_key, []):
                experiment_kind = str(planned.get('experiment_kind', 'planned_operator_prior'))
                steps.append({
                    'step_kind': 'planned_outcome_recorded',
                    'context': planned.get('context'),
                    'experiment_kind': experiment_kind,
                    'outcome': planned.get('outcome'),
                    'summary': (
                        f"{experiment_kind} produced {planned.get('outcome', 'unknown')}"
                    ),
                    'refined_parameters': _rounded_dict(
                        dict(planned.get('operator_prior_refined_parameters') or {})
                    ),
                })
            if claim:
                steps.append({
                    'step_kind': 'discovery_claim_synthesized',
                    'summary': claim['claim'],
                    'status': claim['status'],
                    'next_obligation': claim['next_obligation'],
                })
            recommendations = recommendations_by_key.get(operator_key, [])
            if recommendations:
                recommendation = recommendations[0]
                steps.append({
                    'step_kind': 'next_experiment_selected',
                    'experiment_kind': recommendation.get('experiment_kind'),
                    'target_context': recommendation.get('target_context'),
                    'summary': recommendation.get('reason'),
                })
            chains.append({
                'operator_key': operator_key,
                'operator_kind': operator_kind,
                'status': status,
                'claim': claim.get('claim') if claim else None,
                'parameters': _rounded_dict(parameters),
                'support_contexts': sorted({
                    str(outcome.get('context', 'unknown'))
                    for outcome in outcomes
                    if outcome.get('outcome') == 'confirmed'
                }),
                'failure_contexts': sorted({
                    str(outcome.get('context', 'unknown'))
                    for outcome in outcomes
                    if outcome.get('outcome') in {'weak', 'unmatched'}
                }),
                'next_obligation': (
                    claim.get('next_obligation')
                    if claim
                    else 'collect enough evidence to synthesize an operator discovery claim'
                ),
                'steps': steps,
                'proof_evidence': (
                    dict(claim.get('proof_evidence') or {})
                    if claim
                    else {
                        'confirmed_count': sum(1 for item in outcomes if item.get('outcome') == 'confirmed'),
                        'weak_count': sum(1 for item in outcomes if item.get('outcome') == 'weak'),
                        'unmatched_count': sum(1 for item in outcomes if item.get('outcome') == 'unmatched'),
                        'best_score': max(
                            (float(item.get('best_score', 0.0) or 0.0) for item in outcomes),
                            default=0.0,
                        ),
                    }
                ),
            })
        status_rank = {
            'validated': 5,
            'repaired': 4,
            'supported': 3,
            'domain_limited': 2,
            'weak_candidate': 1,
            'unmatched': 0,
        }
        chains.sort(
            key=lambda item: (
                status_rank.get(str(item.get('status')), 0),
                len(item['steps']),
                item['proof_evidence'].get('best_score', 0.0),
                item['operator_key'],
            ),
            reverse=True,
        )
        return chains[:limit]

    def _operator_chain_status(self, outcomes: list[dict[str, Any]]) -> str:
        if any(outcome.get('outcome') == 'confirmed' for outcome in outcomes):
            return 'supported'
        if any(outcome.get('outcome') == 'weak' for outcome in outcomes):
            return 'weak_candidate'
        return 'unmatched'

    def operator_prior_discovery_claims(self, limit: int = 5) -> list[dict[str, Any]]:
        """Summarize invented-operator evidence as proof-like discovery claims."""
        groups: dict[str, dict[str, Any]] = {}
        for outcome in self.operator_prior_outcomes:
            key = str(outcome.get('operator_key', 'unknown'))
            group = groups.setdefault(
                key,
                {
                    'operator_key': key,
                    'operator_kind': outcome.get('operator_kind', 'unknown'),
                    'outcomes': [],
                },
            )
            group['outcomes'].append(outcome)

        claims = []
        for group in groups.values():
            outcomes = list(group['outcomes'])
            confirmed = [
                item for item in outcomes
                if item.get('outcome') == 'confirmed'
            ]
            weak = [
                item for item in outcomes
                if item.get('outcome') == 'weak'
            ]
            unmatched = [
                item for item in outcomes
                if item.get('outcome') == 'unmatched'
            ]
            if not confirmed and not weak and not unmatched:
                continue
            best = max(
                confirmed or weak or unmatched,
                key=lambda item: (
                    float(item.get('best_score', 0.0) or 0.0),
                    int(item.get('matching_equation_count', 0) or 0),
                ),
            )
            operator_key = group['operator_key']
            operator_kind = str(group['operator_kind'])
            related_outcomes = [
                outcome for outcome in self.planned_outcomes
                if outcome.get('operator_prior_key') == operator_key
            ]
            planned_refined_parameters = {}
            for outcome in reversed(related_outcomes):
                refined = dict(outcome.get('operator_prior_refined_parameters') or {})
                if refined:
                    planned_refined_parameters = refined
                    break
            parameters = (
                planned_refined_parameters
                or self._best_refined_parameters_for_prior(operator_key)
                or dict(best.get('refined_parameters') or {})
                or dict(best.get('parameters') or {})
            )
            expression = self._operator_expression(operator_kind, parameters)
            domain = self._operator_prior_domain_for_key(operator_key)
            included = list(domain.get('included_contexts') or [])
            weak_contexts = list(domain.get('weak_contexts') or [])
            excluded = list(domain.get('excluded_contexts') or [])
            repair_confirmed = sum(
                1 for outcome in related_outcomes
                if outcome.get('outcome') == 'operator_prior_repair_confirmed'
            )
            validation_confirmed = sum(
                1 for outcome in related_outcomes
                if outcome.get('outcome') == 'operator_prior_validation_confirmed'
            )
            failed_followups = sum(
                1 for outcome in related_outcomes
                if str(outcome.get('outcome', '')).endswith('_failed')
            )
            if validation_confirmed and not excluded:
                status = 'validated'
            elif repair_confirmed:
                status = 'repaired'
            elif unmatched and not confirmed and not weak:
                status = 'needs_repair'
            elif excluded:
                status = 'domain_limited'
            elif confirmed:
                status = 'supported'
            else:
                status = 'weak_candidate'

            accepted_because = []
            if confirmed:
                accepted_because.append(
                    f"confirmed in {len(set(item.get('context') for item in confirmed))} context(s)"
                )
            if weak and not confirmed:
                accepted_because.append(
                    f"weakly signaled in {len(set(item.get('context') for item in weak))} context(s)"
                )
            if unmatched and not confirmed and not weak:
                accepted_because.append(
                    'failed cleanly enough to define a repair target instead of disappearing'
                )
            if parameters:
                accepted_because.append('has an executable generated-operator expression')
            if any(item.get('refined_parameters') for item in outcomes):
                accepted_because.append('parameters were refined by held-out feedback')
            if repair_confirmed:
                accepted_because.append('a planned repair probe confirmed the revised prior')
            if validation_confirmed:
                accepted_because.append('an unseen-context validation probe confirmed the prior')
            if not accepted_because:
                accepted_because.append('recorded as an invented-operator candidate')

            not_universal_because = []
            if excluded:
                not_universal_because.append(
                    'failed or remained unmatched in: ' + ', '.join(excluded)
                )
            if unmatched and not repair_confirmed:
                not_universal_because.append('unrepaired unmatched feedback is still present')
            if not validation_confirmed:
                not_universal_because.append('no confirmed unseen-context validation yet')
            if failed_followups:
                not_universal_because.append(
                    f'{failed_followups} planned operator follow-up(s) failed'
                )

            if status == 'validated':
                next_obligation = 'seek a hidden holdout that should break the invented operator if it is overgeneralized'
            elif status == 'repaired':
                next_obligation = 'validate the repaired operator in an unseen context'
            elif status == 'domain_limited':
                next_obligation = 'invent or test a domain predicate that separates success from failure contexts'
            elif status == 'supported':
                next_obligation = 'try the invented operator in a new context or seed'
            elif status == 'needs_repair':
                next_obligation = 'run a targeted repair probe in the strongest unmatched context'
            else:
                next_obligation = 'collect a stronger residual case before promoting the operator'

            contexts = included or weak_contexts or sorted({
                str(item.get('context', 'unknown'))
                for item in confirmed + weak + unmatched
            })
            claims.append({
                'operator_key': operator_key,
                'operator_kind': operator_kind,
                'status': status,
                'claim': (
                    f"{operator_kind} {expression} explains residual structure "
                    f"in {', '.join(contexts) if contexts else 'candidate contexts'}"
                ),
                'expression': expression,
                'parameters': _rounded_dict(parameters),
                'domain_hypothesis': domain,
                'accepted_because': accepted_because,
                'not_universal_because': not_universal_because,
                'would_break_if': (
                    'future repair or validation probes become unmatched, or the '
                    'operator wins only by adding context-specific exceptions'
                ),
                'next_obligation': next_obligation,
                'proof_evidence': {
                    'confirmed_count': len(confirmed),
                    'weak_count': len(weak),
                    'unmatched_count': len(unmatched),
                    'best_score': round(float(best.get('best_score', 0.0) or 0.0), 3),
                    'repair_confirmed_count': repair_confirmed,
                    'validation_confirmed_count': validation_confirmed,
                    'failed_followup_count': failed_followups,
                    'context_count': len(set(item.get('context') for item in confirmed + weak + unmatched)),
                },
                'recent_outcomes': related_outcomes[-3:],
            })
        status_rank = {
            'validated': 5,
            'repaired': 4,
            'supported': 3,
            'domain_limited': 2,
            'weak_candidate': 1,
            'needs_repair': 1,
        }
        claims.sort(
            key=lambda item: (
                status_rank.get(item['status'], 0),
                item['proof_evidence']['best_score'],
                item['proof_evidence']['confirmed_count'],
                item['operator_key'],
            ),
            reverse=True,
        )
        return claims[:limit]

    def operator_prior_claim_experiments(self, limit: int = 5) -> list[dict[str, Any]]:
        """Use proof-like invented-operator claims to choose their next test."""
        recommendations = []
        for claim in self.operator_prior_discovery_claims(
            limit=max(5, len(self.operator_prior_outcomes)),
        ):
            evidence = dict(claim.get('proof_evidence') or {})
            domain = dict(claim.get('domain_hypothesis') or {})
            status = str(claim.get('status', 'unknown'))
            operator_key = claim.get('operator_key')
            operator_kind = claim.get('operator_kind')
            parameters = dict(claim.get('parameters') or {})
            if status == 'needs_repair':
                excluded = list(domain.get('excluded_contexts') or [])
                weak_contexts = list(domain.get('weak_contexts') or [])
                failure_context = excluded[0] if excluded else (
                    weak_contexts[0] if weak_contexts else None
                )
                if any(
                    outcome.get('operator_prior_key') == operator_key
                    and outcome.get('experiment_kind') == 'operator_prior_domain_repair'
                    and outcome.get('context') == failure_context
                    for outcome in self.planned_outcomes
                ):
                    continue
                pseudo_anomaly = {
                    'operator_key': operator_key,
                    'operator_kind': operator_kind,
                    'failure_context': failure_context or 'standard',
                    'parameters': parameters,
                }
                priority = min(
                    0.9,
                    0.66
                    + 0.03 * int(evidence.get('unmatched_count', 0) or 0)
                    + 0.02 * int(evidence.get('weak_count', 0) or 0),
                )
                recommendations.append({
                    'theory_kind': f'operator_prior:{operator_kind}',
                    'operator_prior_key': operator_key,
                    'operator_prior_kind': operator_kind,
                    'operator_prior_parameters': _rounded_dict(parameters),
                    'operator_prior_domain': domain,
                    'operator_prior_claim': claim,
                    'experiment_kind': 'operator_prior_domain_repair',
                    'priority': round(max(0.2, priority), 3),
                    'family_status': 'operator_prior_claim_needs_repair',
                    'target_context': 'operator_prior_failure_context',
                    'failure_context': failure_context,
                    'avoid_contexts': [],
                    'reason': (
                        'repair provisional invented-operator claim: '
                        f"{claim['next_obligation']}"
                    ),
                    'expected_result': (
                        'the failed context should reveal a boundary condition, '
                        'missing factor, or narrower operator expression'
                    ),
                    'falsifies_if': (
                        'the prior remains unmatched and no generated repair '
                        'improves residual error in the target context'
                    ),
                    'proof_evidence': {
                        'support_count': 0,
                        'weak_count': int(evidence.get('weak_count', 0) or 0),
                        'unmatched_count': int(evidence.get('unmatched_count', 0) or 0),
                        'best_score': evidence.get('best_score', 0.0),
                        'claim_status': status,
                    },
                    'probe_action': self._operator_prior_repair_action(pseudo_anomaly),
                    'suggested_campaign': {
                        'command_family': 'equation_campaign',
                        'world_selection': 'operator_prior_failure_context',
                        'enable_equation_probes': True,
                    },
                })
            elif status in {'repaired', 'supported'}:
                if int(evidence.get('validation_confirmed_count', 0) or 0) > 0:
                    continue
                priority = 0.79 if status == 'supported' else 0.88
                priority += min(0.05, 0.02 * int(evidence.get('confirmed_count', 0) or 0))
                recommendations.append({
                    'theory_kind': f'operator_prior:{operator_kind}',
                    'operator_prior_key': operator_key,
                    'operator_prior_kind': operator_kind,
                    'operator_prior_parameters': _rounded_dict(parameters),
                    'operator_prior_domain': domain,
                    'operator_prior_claim': claim,
                    'experiment_kind': 'operator_prior_refinement_validation',
                    'priority': round(min(0.94, priority), 3),
                    'family_status': f'operator_prior_claim_{status}',
                    'target_context': 'operator_prior_unseen_context',
                    'avoid_contexts': (
                        list(domain.get('included_contexts') or [])
                        + list(domain.get('excluded_contexts') or [])
                    ),
                    'reason': f"validate invented-operator claim: {claim['next_obligation']}",
                    'expected_result': (
                        'the claim-backed operator prior should be confirmed or '
                        'weakly supported outside its current evidence contexts'
                    ),
                    'falsifies_if': (
                        'the claim-backed prior is unmatched in an unseen context '
                        'or a simpler residual law wins without it'
                    ),
                    'proof_evidence': {
                        'support_count': int(evidence.get('confirmed_count', 0) or 0),
                        'weak_count': int(evidence.get('weak_count', 0) or 0),
                        'unmatched_count': int(evidence.get('unmatched_count', 0) or 0),
                        'best_score': evidence.get('best_score', 0.0),
                        'claim_status': status,
                    },
                    'suggested_campaign': {
                        'command_family': 'equation_campaign',
                        'world_selection': 'operator_prior_unseen_context',
                        'enable_equation_probes': True,
                    },
                })
            elif status == 'domain_limited':
                if any(
                    outcome.get('operator_prior_key') == operator_key
                    and outcome.get('experiment_kind') == 'operator_prior_domain_predicate_validation'
                    and outcome.get('outcome') == 'operator_prior_domain_predicate_confirmed'
                    for outcome in self.planned_outcomes
                ):
                    continue
                included = list(domain.get('included_contexts') or [])
                excluded = list(domain.get('excluded_contexts') or [])
                if not included and not excluded:
                    continue
                failure_context = excluded[0] if excluded else None
                priority = min(
                    0.93,
                    0.74
                    + 0.04 * int(evidence.get('confirmed_count', 0) or 0)
                    + 0.03 * int(evidence.get('unmatched_count', 0) or 0)
                    + (0.04 if failure_context else 0.0),
                )
                recommendations.append({
                    'theory_kind': f'operator_prior:{operator_kind}',
                    'operator_prior_key': operator_key,
                    'operator_prior_kind': operator_kind,
                    'operator_prior_parameters': _rounded_dict(parameters),
                    'operator_prior_domain': domain,
                    'operator_prior_claim': claim,
                    'experiment_kind': 'operator_prior_domain_predicate_validation',
                    'priority': round(max(0.2, priority), 3),
                    'family_status': 'operator_prior_claim_domain_limited',
                    'target_context': (
                        'operator_prior_failure_context'
                        if failure_context
                        else 'operator_prior_unseen_context'
                    ),
                    'failure_context': failure_context,
                    'avoid_contexts': [],
                    'reason': (
                        'test invented-operator domain predicate: '
                        f"{claim['next_obligation']}"
                    ),
                    'expected_result': (
                        'the prior should be confirmed inside included contexts '
                        'and weak or unmatched inside excluded contexts'
                    ),
                    'falsifies_if': (
                        'the prior is confirmed in an excluded context, or becomes '
                        'unmatched in an included context, without a sharper domain predicate'
                    ),
                    'proof_evidence': {
                        'support_count': int(evidence.get('confirmed_count', 0) or 0),
                        'weak_count': int(evidence.get('weak_count', 0) or 0),
                        'unmatched_count': int(evidence.get('unmatched_count', 0) or 0),
                        'included_context_count': len(included),
                        'excluded_context_count': len(excluded),
                        'best_score': evidence.get('best_score', 0.0),
                        'claim_status': status,
                    },
                    'suggested_campaign': {
                        'command_family': 'equation_campaign',
                        'world_selection': (
                            'operator_prior_failure_context'
                            if failure_context
                            else 'operator_prior_unseen_context'
                        ),
                        'enable_equation_probes': True,
                    },
                })
            elif status == 'validated':
                if any(
                    outcome.get('operator_prior_key') == operator_key
                    and outcome.get('experiment_kind') == 'operator_prior_hidden_holdout_counterexample'
                    and outcome.get('outcome') == 'operator_prior_holdout_confirmed'
                    for outcome in self.planned_outcomes
                ):
                    continue
                recommendations.append({
                    'theory_kind': f'operator_prior:{operator_kind}',
                    'operator_prior_key': operator_key,
                    'operator_prior_kind': operator_kind,
                    'operator_prior_parameters': _rounded_dict(parameters),
                    'operator_prior_domain': domain,
                    'operator_prior_claim': claim,
                    'experiment_kind': 'operator_prior_hidden_holdout_counterexample',
                    'priority': round(min(0.82, 0.68 + 0.03 * int(evidence.get('confirmed_count', 0) or 0)), 3),
                    'family_status': 'operator_prior_claim_validated',
                    'target_context': 'hidden_holdout',
                    'avoid_contexts': [],
                    'reason': f"try to break validated invented-operator claim: {claim['next_obligation']}",
                    'expected_result': (
                        'a validated operator should survive a blind holdout or '
                        'fail in a way that sharpens its domain'
                    ),
                    'falsifies_if': (
                        'the operator prior is unmatched on a hidden holdout and '
                        'no domain predicate explains the failure'
                    ),
                    'proof_evidence': {
                        'support_count': int(evidence.get('confirmed_count', 0) or 0),
                        'validation_confirmed_count': int(
                            evidence.get('validation_confirmed_count', 0) or 0
                        ),
                        'best_score': evidence.get('best_score', 0.0),
                        'claim_status': status,
                    },
                    'suggested_campaign': {
                        'command_family': 'equation_campaign',
                        'world_selection': 'hidden_holdout',
                        'enable_equation_probes': True,
                    },
                })
        recommendations.sort(
            key=lambda item: (
                item['priority'],
                item['proof_evidence']['support_count'],
                item['proof_evidence'].get('best_score', 0.0),
                item['operator_prior_key'],
            ),
            reverse=True,
        )
        return recommendations[:limit]

    def operator_prior_anomalies(self, limit: int = 5) -> list[dict[str, Any]]:
        """Find places where a remembered operator prior broke its own domain story."""
        domain_by_key = {
            domain['operator_key']: domain
            for domain in self.operator_prior_domains(
                limit=max(5, len(self.operator_prior_outcomes)),
            )
        }
        groups: dict[str, dict[str, Any]] = {}
        for outcome in self.operator_prior_outcomes:
            key = str(outcome.get('operator_key', 'unknown'))
            group = groups.setdefault(
                key,
                {
                    'operator_key': key,
                    'operator_kind': outcome.get('operator_kind', 'unknown'),
                    'outcomes': [],
                },
            )
            group['outcomes'].append(outcome)

        anomalies = []
        for group in groups.values():
            outcomes = list(group['outcomes'])
            supports = [
                outcome for outcome in outcomes
                if outcome.get('outcome') == 'confirmed'
            ]
            if not supports:
                continue
            failures = [
                outcome for outcome in outcomes
                if outcome.get('outcome') in {'weak', 'unmatched'}
            ]
            if not failures:
                continue
            best_support = max(
                supports,
                key=lambda outcome: (
                    float(outcome.get('best_score', 0.0) or 0.0),
                    int(outcome.get('matching_equation_count', 0) or 0),
                ),
            )
            support_score = float(best_support.get('best_score', 0.0) or 0.0)
            support_contexts = sorted({
                str(outcome.get('context', 'unknown'))
                for outcome in supports
            })
            weak_contexts = sorted({
                str(outcome.get('context', 'unknown'))
                for outcome in outcomes
                if outcome.get('outcome') == 'weak'
            })
            parameters = (
                self._best_refined_parameters_for_prior(group['operator_key'])
                or dict(best_support.get('refined_parameters') or {})
                or dict(best_support.get('parameters') or {})
            )
            domain = dict(domain_by_key.get(group['operator_key'], {}))
            domain_hypothesis = dict(domain.get('domain_hypothesis') or {})
            for failure in failures:
                failure_context = str(failure.get('context', 'unknown'))
                failure_outcome = str(failure.get('outcome', 'unknown'))
                failure_score = float(failure.get('best_score', 0.0) or 0.0)
                score_gap = max(0.0, support_score - failure_score)
                anomaly_kind = (
                    'replication_break'
                    if failure_context in support_contexts
                    else 'domain_break'
                )
                if failure_outcome == 'weak':
                    anomaly_kind = (
                        'partial_replication_break'
                        if failure_context in support_contexts
                        else 'domain_weakening'
                    )
                severity = 0.58 + min(0.28, score_gap * 0.22)
                if failure_outcome == 'unmatched':
                    severity += 0.08
                if failure_context in support_contexts:
                    severity += 0.06
                anomalies.append({
                    'operator_key': group['operator_key'],
                    'operator_kind': group['operator_kind'],
                    'anomaly_kind': anomaly_kind,
                    'severity': round(min(0.98, severity), 3),
                    'support_contexts': support_contexts,
                    'weak_contexts': weak_contexts,
                    'failure_context': failure_context,
                    'failure_outcome': failure_outcome,
                    'support_score': round(support_score, 3),
                    'failure_score': round(failure_score, 3),
                    'score_gap': round(score_gap, 3),
                    'parameters': _rounded_dict(parameters),
                    'domain_hypothesis': domain_hypothesis,
                    'question': (
                        f"why did {group['operator_kind']} work in "
                        f"{', '.join(support_contexts)} but become "
                        f"{failure_outcome} in {failure_context}?"
                    ),
                    'repair_hint': (
                        'invent a narrower domain predicate or missing factor '
                        'that separates the success and failure contexts'
                    ),
                })
        anomalies.sort(
            key=lambda item: (
                item['severity'],
                item['score_gap'],
                item['support_score'],
                item['operator_key'],
            ),
            reverse=True,
        )
        return anomalies[:limit]

    def operator_prior_repair_experiments(self, limit: int = 5) -> list[dict[str, Any]]:
        repairs = []
        for anomaly in self.operator_prior_anomalies(
            limit=max(5, len(self.operator_prior_outcomes)),
        ):
            if self._operator_prior_repair_resolved(
                str(anomaly.get('operator_key', '')),
                str(anomaly.get('failure_context', '')),
            ):
                continue
            parameters = dict(anomaly.get('parameters') or {})
            priority = min(
                0.94,
                0.62
                + 0.24 * float(anomaly.get('severity', 0.0) or 0.0)
                + min(0.08, 0.04 * len(anomaly.get('support_contexts') or [])),
            )
            repairs.append({
                'theory_kind': f"operator_prior:{anomaly['operator_kind']}",
                'operator_prior_key': anomaly['operator_key'],
                'operator_prior_kind': anomaly['operator_kind'],
                'operator_prior_parameters': _rounded_dict(parameters),
                'operator_prior_domain': dict(anomaly.get('domain_hypothesis') or {}),
                'operator_prior_anomaly': anomaly,
                'experiment_kind': 'operator_prior_domain_repair',
                'priority': round(max(0.2, priority), 3),
                'family_status': 'operator_prior_anomaly',
                'target_context': 'operator_prior_failure_context',
                'failure_context': anomaly['failure_context'],
                'avoid_contexts': [],
                'reason': anomaly['question'],
                'expected_result': (
                    'the failed context should reveal a boundary condition, '
                    'missing factor, or revised operator that explains the anomaly'
                ),
                'falsifies_if': (
                    'the same prior remains unmatched and no narrower generated '
                    'operator improves residual error in the failure context'
                ),
                'proof_evidence': {
                    'support_count': len(anomaly.get('support_contexts') or []),
                    'weak_count': len(anomaly.get('weak_contexts') or []),
                    'failure_count': 1,
                    'support_score': anomaly.get('support_score', 0.0),
                    'failure_score': anomaly.get('failure_score', 0.0),
                    'severity': anomaly.get('severity', 0.0),
                },
                'probe_action': self._operator_prior_repair_action(anomaly),
                'suggested_campaign': {
                    'command_family': 'equation_campaign',
                    'world_selection': 'operator_prior_failure_context',
                    'enable_equation_probes': True,
                },
            })
        repairs.sort(
            key=lambda item: (
                item['priority'],
                item['proof_evidence']['severity'],
                item['operator_prior_key'],
            ),
            reverse=True,
        )
        return repairs[:limit]

    def _operator_prior_repair_resolved(
        self,
        operator_key: str,
        failure_context: str,
    ) -> bool:
        for outcome in self.planned_outcomes:
            if outcome.get('experiment_kind') != 'operator_prior_domain_repair':
                continue
            if outcome.get('operator_prior_key') != operator_key:
                continue
            if outcome.get('context') != failure_context:
                continue
            if outcome.get('outcome') == 'operator_prior_repair_confirmed':
                return True
        return False

    def operator_prior_validation_experiments(self, limit: int = 5) -> list[dict[str, Any]]:
        prior_by_key = {
            prior.get('key'): prior
            for prior in self.generated_operator_priors(
                limit=max(12, len(self.operator_prior_outcomes) + 4),
            )
        }
        recommendations = []
        for domain in self.operator_prior_domains(
            limit=max(5, len(self.operator_prior_outcomes)),
        ):
            prior = prior_by_key.get(domain['operator_key'])
            if not prior or not prior.get('refined_from_operator_key'):
                continue
            hypothesis = dict(domain.get('domain_hypothesis') or {})
            refined_parameters = dict(prior.get('parameters') or {})
            support_count = int(domain.get('confirmed_count', 0) or 0)
            weak_count = int(domain.get('weak_count', 0) or 0)
            unmatched_count = int(domain.get('unmatched_count', 0) or 0)
            priority = min(
                0.91,
                0.68
                + 0.07 * max(1, support_count)
                + 0.03 * weak_count
                - 0.04 * unmatched_count,
            )
            recommendations.append({
                'theory_kind': f"operator_prior:{domain['operator_kind']}",
                'operator_prior_key': domain['operator_key'],
                'operator_prior_kind': domain['operator_kind'],
                'operator_prior_parameters': _rounded_dict(refined_parameters),
                'operator_prior_domain': hypothesis,
                'experiment_kind': 'operator_prior_refinement_validation',
                'priority': round(max(0.2, priority), 3),
                'family_status': 'operator_prior_refined',
                'target_context': 'operator_prior_unseen_context',
                'avoid_contexts': (
                    list(hypothesis.get('included_contexts') or [])
                    + list(hypothesis.get('excluded_contexts') or [])
                ),
                'reason': (
                    'validate a refined generated operator parameter in a context '
                    'outside its current domain evidence'
                ),
                'expected_result': (
                    'the refined prior should produce a confirmed or weak '
                    'operator-prior feedback result'
                ),
                'falsifies_if': (
                    'the refined prior is unmatched in the validation context '
                    'or a rival generated operator wins with lower residual error'
                ),
                'proof_evidence': {
                    'support_count': support_count,
                    'weak_count': weak_count,
                    'unmatched_count': unmatched_count,
                    'best_score': domain.get('best_score', 0.0),
                    'parameter_count': len(refined_parameters),
                },
                'suggested_campaign': {
                    'command_family': 'equation_campaign',
                    'world_selection': 'operator_prior_unseen_context',
                    'enable_equation_probes': True,
                },
            })
        recommendations.sort(
            key=lambda item: (
                item['priority'],
                item['proof_evidence']['support_count'],
                item['proof_evidence']['best_score'],
                item['operator_prior_key'],
            ),
            reverse=True,
        )
        return recommendations[:limit]

    def operator_prior_domains(self, limit: int = 5) -> list[dict[str, Any]]:
        groups: dict[str, dict[str, Any]] = {}
        for outcome in self.operator_prior_outcomes:
            key = str(outcome.get('operator_key', 'unknown'))
            group = groups.setdefault(
                key,
                {
                    'operator_key': key,
                    'operator_kind': outcome.get('operator_kind', 'unknown'),
                    'outcomes': [],
                    'confirmed_contexts': set(),
                    'weak_contexts': set(),
                    'unmatched_contexts': set(),
                    'best_score': 0.0,
                },
            )
            group['outcomes'].append(outcome)
            context = str(outcome.get('context', 'unknown'))
            best_score = float(outcome.get('best_score', 0.0) or 0.0)
            group['best_score'] = max(group['best_score'], best_score)
            if outcome.get('outcome') == 'confirmed':
                group['confirmed_contexts'].add(context)
            elif outcome.get('outcome') == 'weak':
                group['weak_contexts'].add(context)
            elif outcome.get('outcome') == 'unmatched':
                group['unmatched_contexts'].add(context)

        domains = []
        for group in groups.values():
            included = sorted(group['confirmed_contexts'])
            weak = sorted(group['weak_contexts'] - group['confirmed_contexts'])
            excluded = sorted(
                group['unmatched_contexts']
                - group['confirmed_contexts']
                - group['weak_contexts']
            )
            if included:
                claim = (
                    f"{group['operator_kind']} prior is supported in "
                    f"{', '.join(included)}"
                )
            elif weak:
                claim = (
                    f"{group['operator_kind']} prior is weakly signaled in "
                    f"{', '.join(weak)}"
                )
            else:
                claim = (
                    f"{group['operator_kind']} prior has not found a supporting context yet"
                )
            if excluded:
                claim += f" and should avoid {', '.join(excluded)} until revised"
            domains.append({
                'operator_key': group['operator_key'],
                'operator_kind': group['operator_kind'],
                'attempt_count': len(group['outcomes']),
                'confirmed_count': len(group['confirmed_contexts']),
                'weak_count': len(group['weak_contexts']),
                'unmatched_count': len(group['unmatched_contexts']),
                'best_score': round(group['best_score'], 3),
                'domain_hypothesis': {
                    'claim': claim,
                    'included_contexts': included,
                    'weak_contexts': weak,
                    'excluded_contexts': excluded,
                    'revision_needed': bool(excluded and not included),
                    'next_test': (
                        'try the prior in an unseen context'
                        if not included
                        else 'verify the prior outside confirmed contexts'
                    ),
                },
            })
        domains.sort(
            key=lambda item: (
                item['confirmed_count'],
                item['best_score'],
                -item['unmatched_count'],
                item['operator_key'],
            ),
            reverse=True,
        )
        return domains[:limit]

    def _operator_prior_repair_action(self, anomaly: dict[str, Any]) -> dict[str, Any]:
        parameters = dict(anomaly.get('parameters') or {})
        center_x = float(parameters.get('center_x', 10.0) or 10.0)
        center_y = float(parameters.get('center_y', 10.0) or 10.0)
        radius = parameters.get('cutoff_radius')
        if isinstance(radius, (int, float)) and radius > 0:
            distance = float(radius) * 1.05
            label = 'operator_prior_boundary_repair'
        else:
            exponent = parameters.get('distance_exponent', 2.0)
            distance = 4.0
            if isinstance(exponent, (int, float)):
                distance = max(1.25, min(7.5, 2.5 + float(exponent)))
            label = 'operator_prior_domain_repair'
        return {
            'type': 'spawn',
            'x': round(center_x + distance, 6),
            'y': round(center_y, 6),
            'vx': 0.0,
            'vy': 0.0,
            'source': 'planned_operator_prior_repair',
            'probe_label': label,
            'operator_prior_key': anomaly.get('operator_key'),
            'operator_prior_kind': anomaly.get('operator_kind'),
            'failure_context': anomaly.get('failure_context'),
        }

    def _apply_operator_prior_feedback(self, prior: dict[str, Any]) -> dict[str, Any]:
        adjusted = dict(prior)
        stats = self._operator_prior_feedback_stats(str(prior.get('key', '')))
        usefulness = float(adjusted.get('usefulness', 0.0) or 0.0)
        usefulness += 0.04 * stats['confirmed_count']
        usefulness -= 0.03 * stats['weak_count']
        usefulness -= 0.08 * stats['unmatched_count']
        adjusted['usefulness'] = round(max(0.05, min(0.98, usefulness)), 3)
        if stats['attempt_count']:
            adjusted['feedback'] = stats
            adjusted['domain_hypothesis'] = self._operator_prior_domain_for_key(
                str(prior.get('key', ''))
            )
            refined_parameters = self._best_refined_parameters_for_prior(
                str(prior.get('key', ''))
            )
            if refined_parameters:
                parameters = {
                    **dict(adjusted.get('parameters') or {}),
                    **refined_parameters,
                }
                adjusted['parameters'] = _rounded_dict(parameters)
                adjusted['expression'] = self._operator_expression(
                    str(adjusted.get('operator_kind', 'unknown')),
                    parameters,
                )
                adjusted['refined_from_operator_key'] = prior.get('key')
        return adjusted

    def _best_refined_parameters_for_prior(self, operator_key: str) -> dict[str, Any]:
        for outcome in reversed(self.planned_outcomes):
            if outcome.get('operator_prior_key') != operator_key:
                continue
            refined = dict(outcome.get('operator_prior_refined_parameters') or {})
            if refined:
                return refined
        candidates = [
            outcome for outcome in self.operator_prior_outcomes
            if outcome.get('operator_key') == operator_key
            and outcome.get('outcome') in {'confirmed', 'weak'}
            and outcome.get('refined_parameters')
        ]
        if not candidates:
            return {}
        best = max(
            candidates,
            key=lambda outcome: (
                float(outcome.get('best_score', 0.0) or 0.0),
                int(outcome.get('matching_equation_count', 0) or 0),
            ),
        )
        return dict(best.get('refined_parameters') or {})

    def _operator_expression(
        self,
        operator_kind: str,
        parameters: dict[str, Any],
    ) -> str:
        relation = str(parameters.get('relation', 'direction'))
        base = self._base_vector_expression(relation)
        if operator_kind == 'inverse_separation_power':
            exponent = parameters.get('distance_exponent', '?')
            return f'{base} / separation^{exponent}'
        if operator_kind == 'localized_cutoff_window':
            radius = parameters.get('cutoff_radius', '?')
            return f'inside(separation <= {radius}) * {base}'
        if operator_kind == 'localized_tapered_power':
            radius = parameters.get('cutoff_radius', '?')
            exponent = parameters.get('distance_exponent', '?')
            return (
                f'inside(separation <= {radius}) * '
                f'max(0, 1 - separation/{radius}) * '
                f'{base} / separation^{exponent}'
            )
        return str(parameters.get('expression', operator_kind))

    def _operator_prior_from_claim(self, claim: dict[str, Any]) -> dict[str, Any] | None:
        operator_key = claim.get('operator_key')
        operator_kind = str(claim.get('operator_kind', 'unknown'))
        parameters = dict(claim.get('parameters') or {})
        if not operator_key or not parameters:
            return None
        input_map = {
            'inverse_separation_power': ['center', 'position', 'distance_exponent'],
            'localized_cutoff_window': ['center', 'position', 'cutoff_radius'],
            'localized_tapered_power': [
                'center',
                'position',
                'cutoff_radius',
                'distance_exponent',
            ],
        }
        usefulness_by_status = {
            'validated': 0.78,
            'repaired': 0.72,
            'supported': 0.66,
            'domain_limited': 0.56,
            'weak_candidate': 0.42,
            'needs_repair': 0.34,
        }
        status = str(claim.get('status', 'supported'))
        return {
            'key': operator_key,
            'operator_kind': operator_kind,
            'inputs': input_map.get(operator_kind, ['generated_operator_inputs']),
            'expression': self._operator_expression(operator_kind, parameters),
            'generated_from': 'operator_prior_discovery_claim',
            'source': 'theory_memory_prior',
            'usefulness': usefulness_by_status.get(status, 0.5),
            'test_hint': (
                'retest the proof-like invented-operator claim on held-out residuals'
            ),
            'parameters': _rounded_dict(parameters),
            'domain_hypothesis': dict(claim.get('domain_hypothesis') or {}),
            'claim_status': status,
        }

    def _operator_prior_allowed_in_context(
        self,
        prior: dict[str, Any],
        context: str | None,
    ) -> bool:
        if not context:
            return True
        domain = dict(prior.get('domain_hypothesis') or {})
        excluded = set(domain.get('excluded_contexts') or [])
        included = set(domain.get('included_contexts') or [])
        if context in excluded and context not in included:
            return False
        return True

    def _annotate_operator_prior_with_first_principles(
        self,
        prior: dict[str, Any],
    ) -> dict[str, Any]:
        annotated = dict(prior)
        operator_kind = str(annotated.get('operator_kind', 'unknown'))
        if operator_kind == 'inverse_separation_power':
            primitives = [
                'order_metric',
                'inverse_operation',
                'composition',
                'dimension_lift',
            ]
            dimensions = [
                'separation',
                'residual_strength_exponent',
                'signed_radial_projection',
            ]
        elif operator_kind == 'localized_cutoff_window':
            primitives = [
                'order_metric',
                'domain_partition',
                'composition',
                'dimension_lift',
            ]
            dimensions = [
                'separation',
                'signed_boundary_margin',
                'inside_domain_indicator',
            ]
        elif operator_kind == 'localized_tapered_power':
            primitives = [
                'order_metric',
                'inverse_operation',
                'domain_partition',
                'composition',
                'dimension_lift',
            ]
            dimensions = [
                'separation',
                'local_taper_fraction',
                'residual_strength_exponent',
                'inside_domain_indicator',
            ]
        elif operator_kind == 'phase_basis':
            primitives = [
                'recursion_iteration',
                'symmetry_transform',
                'composition',
                'dimension_lift',
            ]
            dimensions = ['phase_sine', 'phase_cosine', 'cyclic_time']
        else:
            primitives = ['identity_equality', 'composition', 'dimension_lift']
            dimensions = ['operator_context_axis']
        annotated['first_principles'] = primitives
        annotated['adaptive_dimensions'] = dimensions
        annotated['dimension_policy'] = (
            'dimensions are retained only while they improve held-out residual '
            'fit, separate rival theories, or explain a domain-limited claim'
        )
        return annotated

    def _annotate_operator_prior_with_algebraic_foundation(
        self,
        prior: dict[str, Any],
    ) -> dict[str, Any]:
        annotated = dict(prior)
        profile = self._algebraic_profile_for_operator_kind(
            str(annotated.get('operator_kind', 'unknown'))
        )
        annotated['algebraic_families'] = profile['expression_families']
        annotated['algebraic_structures'] = profile['algebraic_structures']
        annotated['algebraic_proof_obligations'] = profile['proof_obligations']
        annotated['algebraic_search_controls'] = profile['search_controls']
        return annotated

    def _algebraic_agenda_item_from_signal(
        self,
        key: str,
        source: str,
        signal: dict[str, Any],
        priority: float,
    ) -> dict[str, Any] | None:
        family_keys = self._algebraic_family_keys_from_signal(signal)
        if not family_keys:
            return None
        structures = self._algebraic_structure_keys_for_families(family_keys)
        obligations = self._algebraic_obligation_keys_for_families(
            family_keys,
            structures,
        )
        family_index = {
            family['key']: family
            for family in ALGEBRAIC_EXPRESSION_FAMILIES
        }
        expression_templates = [
            family_index[family_key]['expression_schema'][0]
            for family_key in family_keys
            if family_key in family_index
        ]
        signal_name = (
            signal.get('name')
            or signal.get('operator_kind')
            or signal.get('theory_kind')
            or signal.get('key')
            or source
        )
        return {
            'key': key,
            'source': source,
            'signal': str(signal_name),
            'proposal_kind': signal.get('proposal_kind') or signal.get('dimension_kind') or source,
            'theory_kind': signal.get('theory_kind'),
            'priority': round(max(0.0, min(1.0, priority)), 3),
            'expression_families': family_keys,
            'algebraic_structures': structures,
            'proof_obligations': obligations,
            'template_seeds': expression_templates[:4],
            'search_controls': self._algebraic_search_controls_for_families(family_keys),
            'reason': self._algebraic_reason_for_signal(signal, family_keys),
        }

    def _algebraic_family_keys_from_signal(
        self,
        signal: dict[str, Any],
    ) -> list[str]:
        parts = []
        for field_name in (
            'key',
            'name',
            'expression',
            'theory_kind',
            'operator_kind',
            'anomaly_kind',
            'dimension_kind',
            'proposal_kind',
            'source',
            'claim',
            'question',
            'reason',
        ):
            value = signal.get(field_name)
            if value is not None:
                parts.append(str(value))
        evidence = signal.get('evidence')
        if isinstance(evidence, dict):
            parts.extend(str(value) for value in evidence.values())
        parameters = signal.get('parameters')
        if isinstance(parameters, dict):
            parts.extend(str(value) for value in parameters.values())
        text = ' '.join(parts).lower()
        keys: list[str] = []

        def add(*family_keys: str):
            for family_key in family_keys:
                if family_key not in keys:
                    keys.append(family_key)

        if any(term in text for term in ('separation', 'distance', 'inverse', 'exponent', 'power', 'radial')):
            add('power_law', 'rational_ratio', 'logarithmic', 'vector_projection_norm')
        if any(term in text for term in ('boundary', 'cutoff', 'inside', 'outside', 'domain', 'predicate', 'localized')):
            add('piecewise_predicate', 'set_relation_cardinality', 'optimization_extremum')
        if 'taper' in text:
            add('piecewise_predicate', 'power_law', 'rational_ratio')
        if any(term in text for term in ('phase', 'periodic', 'cycle', 'cyclic', 'time_varying')):
            add('sinusoidal_phase', 'recurrence_iteration', 'symmetry_invariant')
        if any(term in text for term in ('direction', 'projection', 'field_axis', 'perpendicular', 'center', 'orthogonal', 'vector')):
            add('vector_projection_norm', 'matrix_linear_transform', 'optimization_extremum')
        if any(term in text for term in ('transition', 'motion', 'delta', 'velocity', 'acceleration', 'step')):
            add('finite_difference_calculus', 'recurrence_iteration', 'accumulation_integral')
        if any(term in text for term in ('count', 'cardinality', 'membership', 'ordering', 'order')):
            add('set_relation_cardinality', 'affine_linear', 'constant_identity')
        if any(term in text for term in ('weak', 'unmatched', 'counterexample', 'calibration', 'support')):
            add('probability_statistics', 'piecewise_predicate', 'optimization_extremum')
        if not keys:
            add('affine_linear', 'polynomial_basis', 'piecewise_predicate')
        return keys[:6]

    def _algebraic_structure_keys_for_families(
        self,
        family_keys: list[str],
    ) -> list[str]:
        structure_map = {
            'constant_identity': ['monoid', 'ordered_set'],
            'affine_linear': ['field', 'vector_space'],
            'polynomial_basis': ['ring', 'field'],
            'rational_ratio': ['field', 'ordered_set'],
            'power_law': ['field', 'metric_space', 'ordered_set'],
            'logarithmic': ['field', 'ordered_set'],
            'exponential': ['monoid', 'field', 'ordered_set'],
            'sinusoidal_phase': ['group', 'vector_space'],
            'piecewise_predicate': ['lattice_boolean_algebra', 'ordered_set', 'topological_neighborhood'],
            'recurrence_iteration': ['semigroup', 'monoid'],
            'finite_difference_calculus': ['vector_space', 'metric_space'],
            'accumulation_integral': ['monoid', 'vector_space'],
            'vector_projection_norm': ['vector_space', 'metric_space'],
            'matrix_linear_transform': ['vector_space', 'group'],
            'set_relation_cardinality': ['lattice_boolean_algebra', 'ordered_set'],
            'graph_relation_path': ['graph', 'semigroup'],
            'probability_statistics': ['probability_space', 'ordered_set'],
            'optimization_extremum': ['ordered_set', 'metric_space'],
            'symmetry_invariant': ['group', 'metric_space'],
        }
        structures: list[str] = []
        for family_key in family_keys:
            for structure in structure_map.get(family_key, []):
                if structure not in structures:
                    structures.append(structure)
        return structures[:6]

    def _algebraic_obligation_keys_for_families(
        self,
        family_keys: list[str],
        structure_keys: list[str],
    ) -> list[str]:
        targets = set(family_keys) | set(structure_keys) | {'all_equation_families'}
        obligations = []
        for obligation in ALGEBRAIC_PROOF_OBLIGATIONS:
            applies_to = set(obligation.get('applies_to') or [])
            if targets & applies_to:
                obligations.append(str(obligation['key']))
        for required in ('dimensional_consistency', 'heldout_counterexample'):
            if required not in obligations:
                obligations.append(required)
        required_tail = [
            item for item in ('dimensional_consistency', 'heldout_counterexample')
            if item in obligations
        ]
        optional = [
            item for item in obligations
            if item not in required_tail
        ]
        return (optional[: max(0, 7 - len(required_tail))] + required_tail)[:7]

    def _algebraic_search_controls_for_families(
        self,
        family_keys: list[str],
    ) -> dict[str, Any]:
        family_index = {
            family['key']: family
            for family in ALGEBRAIC_EXPRESSION_FAMILIES
        }
        max_cost = max(
            (
                int(family_index[key].get('complexity_cost', 1))
                for key in family_keys
                if key in family_index
            ),
            default=1,
        )
        return {
            'baseline_version': ALGEBRAIC_SEARCH_CONTROLS['baseline_version'],
            'max_complexity_cost': max_cost,
            'require_heldout_score': True,
            'require_falsifier': True,
            'expand_when': ALGEBRAIC_SEARCH_CONTROLS['complexity_budget']['expand_when'],
        }

    def _algebraic_profile_for_operator_kind(
        self,
        operator_kind: str,
    ) -> dict[str, Any]:
        profiles = {
            'inverse_separation_power': [
                'power_law',
                'rational_ratio',
                'logarithmic',
                'vector_projection_norm',
            ],
            'localized_cutoff_window': [
                'piecewise_predicate',
                'set_relation_cardinality',
                'optimization_extremum',
                'vector_projection_norm',
            ],
            'localized_tapered_power': [
                'piecewise_predicate',
                'power_law',
                'rational_ratio',
                'vector_projection_norm',
            ],
            'phase_basis': [
                'sinusoidal_phase',
                'recurrence_iteration',
                'symmetry_invariant',
                'vector_projection_norm',
            ],
        }
        family_keys = profiles.get(
            operator_kind,
            ['affine_linear', 'polynomial_basis', 'piecewise_predicate'],
        )
        structures = self._algebraic_structure_keys_for_families(family_keys)
        return {
            'expression_families': family_keys,
            'algebraic_structures': structures,
            'proof_obligations': self._algebraic_obligation_keys_for_families(
                family_keys,
                structures,
            ),
            'search_controls': self._algebraic_search_controls_for_families(family_keys),
        }

    def _algebraic_reason_for_signal(
        self,
        signal: dict[str, Any],
        family_keys: list[str],
    ) -> str:
        name = (
            signal.get('name')
            or signal.get('operator_kind')
            or signal.get('theory_kind')
            or signal.get('key')
            or 'current residual signal'
        )
        families = ', '.join(family_keys[:3])
        return f"{name} should first search {families} before escalating complexity"

    def _operator_prior_domain_for_key(self, operator_key: str) -> dict[str, Any]:
        for domain in self.operator_prior_domains(limit=max(5, len(self.operator_prior_outcomes))):
            if domain.get('operator_key') == operator_key:
                return domain['domain_hypothesis']
        return {
            'claim': 'operator prior has not been tested yet',
            'included_contexts': [],
            'weak_contexts': [],
            'excluded_contexts': [],
            'revision_needed': False,
            'next_test': 'try the prior in a first context',
        }

    def _operator_prior_feedback_stats(self, operator_key: str) -> dict[str, Any]:
        matching = [
            outcome for outcome in self.operator_prior_outcomes
            if outcome.get('operator_key') == operator_key
        ]
        return {
            'attempt_count': len(matching),
            'confirmed_count': sum(1 for item in matching if item.get('outcome') == 'confirmed'),
            'weak_count': sum(1 for item in matching if item.get('outcome') == 'weak'),
            'unmatched_count': sum(1 for item in matching if item.get('outcome') == 'unmatched'),
            'best_score': round(
                max((float(item.get('best_score', 0.0) or 0.0) for item in matching), default=0.0),
                3,
            ),
        }

    def _representation_from_disagreement(
        self,
        experiment: dict[str, Any],
    ) -> dict[str, Any] | None:
        signature = dict(experiment.get('disagreement_signature') or {})
        mode = str(signature.get('mode', 'unknown'))
        theory_kind = str(experiment.get('theory_kind', 'model_disagreement'))
        priority = min(1.0, float(experiment.get('priority', 0.5) or 0.5) + 0.03)
        evidence = dict(experiment.get('proof_evidence') or {})
        if mode == 'distance_exponent_race':
            name = 'separation_exponent_from_log_ratio'
            expression = (
                '-log(abs(residual_near) / abs(residual_far)) '
                '/ log(separation_near / separation_far)'
            )
            reason = (
                'competing inverse-distance laws need a scale-free exponent '
                'measurement instead of trying one exponent at a time'
            )
            expected_use = (
                'estimate the residual-strength exponent from matched near/far probes'
            )
            proposal_kind = 'derived_variable'
        elif mode == 'cutoff_boundary_vs_smooth_falloff':
            name = 'signed_boundary_margin'
            expression = 'separation - cutoff_radius'
            reason = (
                'cutoff and smooth-falloff rivals disagree most around the '
                'inside/outside boundary'
            )
            expected_use = (
                'bucket residuals by inside, boundary, and outside margin'
            )
            proposal_kind = 'derived_variable'
        elif mode == 'taper_shape_vs_hard_boundary':
            name = 'local_taper_fraction'
            expression = 'max(0, 1 - separation / cutoff_radius)'
            reason = (
                'tapered and hard-boundary rivals need a continuous boundary '
                'coordinate to compare shape'
            )
            expected_use = (
                'test whether residual strength changes smoothly before the boundary'
            )
            proposal_kind = 'operator_prior'
        elif mode == 'vector_direction_disagreement':
            name = 'signed_residual_alignment_pair'
            expression = (
                '(dot(residual, unit(center - position)), '
                'dot(residual, perpendicular(unit(center - position))))'
            )
            reason = (
                'direction and perpendicular residual rivals need matched signed '
                'alignment coordinates'
            )
            expected_use = (
                'separate center-aligned and quarter-turn residual explanations'
            )
            proposal_kind = 'derived_variable'
        else:
            return None
        return {
            'key': f'representation:{mode}:{name}',
            'name': name,
            'proposal_kind': proposal_kind,
            'source': 'model_disagreement',
            'theory_kind': theory_kind,
            'priority': round(priority, 3),
            'expression': expression,
            'reason': reason,
            'expected_use': expected_use,
            'evidence': {
                'mode': mode,
                'support_count': evidence.get('support_count', 0),
                'attempt_count': evidence.get('attempt_count', 0),
                'still_open_count': evidence.get('still_open_count', 0),
                'rival_confirmed_count': evidence.get('rival_confirmed_count', 0),
                'target_confirmed_count': evidence.get('target_confirmed_count', 0),
            },
        }

    def _representation_from_domain_revision(
        self,
        revision: dict[str, Any],
    ) -> dict[str, Any] | None:
        domain = dict(revision.get('domain_hypothesis') or {})
        excluded = list(domain.get('excluded_contexts') or [])
        included = list(domain.get('included_contexts') or [])
        if not excluded:
            return None
        theory_kind = str(revision.get('theory_kind', 'unknown'))
        evidence = dict(revision.get('proof_evidence') or {})
        return {
            'key': f'representation:domain_predicate:{theory_kind}',
            'name': 'learned_domain_predicate',
            'proposal_kind': 'domain_predicate',
            'source': 'domain_revision',
            'theory_kind': theory_kind,
            'priority': round(min(1.0, 0.84 + 0.03 * len(excluded)), 3),
            'expression': 'applies_if(context in included and context not in excluded)',
            'reason': (
                'a counterexample means the theory needs a learned domain guard '
                'instead of a universal claim'
            ),
            'expected_use': (
                'reuse the family only where the narrowed domain predicts it should apply'
            ),
            'evidence': {
                'support_count': evidence.get('support_count', 0),
                'counterexample_count': evidence.get('counterexample_count', 0),
                'included_contexts': included,
                'excluded_contexts': excluded,
            },
        }

    def _representation_from_operator_prior_anomaly(
        self,
        anomaly: dict[str, Any],
    ) -> dict[str, Any] | None:
        failure_context = str(anomaly.get('failure_context', 'unknown'))
        support_contexts = list(anomaly.get('support_contexts') or [])
        if not support_contexts or not failure_context:
            return None
        operator_key = str(anomaly.get('operator_key', 'unknown'))
        operator_kind = str(anomaly.get('operator_kind', 'unknown'))
        severity = float(anomaly.get('severity', 0.0) or 0.0)
        score_gap = float(anomaly.get('score_gap', 0.0) or 0.0)
        parameters = dict(anomaly.get('parameters') or {})
        priority = min(1.0, 0.82 + 0.10 * severity + min(0.05, score_gap * 0.04))
        expression = (
            'applies_if('
            'operator_residual_signature_matches_success_contexts '
            'and context not in failure_contexts)'
        )
        if 'cutoff_radius' in parameters:
            expression = (
                'applies_if(boundary_margin and residual_inside_outside_contrast '
                'matches success_contexts)'
            )
        elif 'distance_exponent' in parameters:
            expression = (
                'applies_if(residual_strength_ratio_matches_refined_exponent '
                'and context not in failure_contexts)'
            )
        return {
            'key': (
                f'representation:operator_prior_domain_predicate:'
                f'{operator_key}:{failure_context}'
            ),
            'name': 'operator_prior_domain_predicate',
            'proposal_kind': 'domain_predicate',
            'source': 'operator_prior_anomaly',
            'theory_kind': f'operator_prior:{operator_kind}',
            'priority': round(priority, 3),
            'expression': expression,
            'reason': (
                'a generated operator succeeded in one context and failed in '
                'another, so the agent needs a domain variable that predicts '
                'when the invented operator applies'
            ),
            'expected_use': (
                'split future generated-operator tests by the learned success '
                'and failure conditions instead of retrying the prior universally'
            ),
            'evidence': {
                'operator_key': operator_key,
                'operator_kind': operator_kind,
                'anomaly_kind': anomaly.get('anomaly_kind'),
                'support_count': len(support_contexts),
                'support_contexts': support_contexts,
                'failure_context': failure_context,
                'failure_outcome': anomaly.get('failure_outcome'),
                'support_score': anomaly.get('support_score', 0.0),
                'failure_score': anomaly.get('failure_score', 0.0),
                'score_gap': anomaly.get('score_gap', 0.0),
                'parameters': _rounded_dict(parameters),
            },
        }

    def _representation_from_family(
        self,
        family: TheoryFamily,
    ) -> dict[str, Any] | None:
        if family.support_count < 2:
            return None
        if family.counterexample_count > 0:
            return None
        status = family.generalization_status
        if status not in {'local', 'reusable', 'established'}:
            return None
        operators = sorted(family.operator_kinds)
        if not operators:
            return None
        operator_name = operators[0]
        return {
            'key': f'representation:promote_operator:{family.theory_kind}:{operator_name}',
            'name': f'promote_{operator_name}',
            'proposal_kind': 'operator_prior',
            'source': 'family_generalization',
            'theory_kind': family.theory_kind,
            'priority': round(min(1.0, 0.58 + 0.05 * family.support_count), 3),
            'expression': f'reuse {operator_name} when residual evidence matches this family',
            'reason': (
                'repeated supporting runs make this operator worth trying early '
                'in future searches'
            ),
            'expected_use': (
                'seed future equation searches with the reusable operator family'
            ),
            'evidence': family.proof_evidence,
        }

    def _adaptive_dimension_from_representation(
        self,
        proposal: dict[str, Any],
    ) -> dict[str, Any] | None:
        name = str(proposal.get('name', 'unknown'))
        evidence = dict(proposal.get('evidence') or {})
        priority = min(1.0, float(proposal.get('priority', 0.5) or 0.5) + 0.02)
        base = {
            'source': proposal.get('source', 'representation_agenda'),
            'theory_kind': proposal.get('theory_kind', 'unknown'),
            'priority': round(priority, 3),
            'reason': proposal.get('reason'),
            'expected_use': proposal.get('expected_use'),
            'evidence': {
                **evidence,
                'representation_key': proposal.get('key'),
            },
            'expansion_policy': (
                'keep this dimension only while it improves held-out residuals '
                'or separates a success context from a failure context'
            ),
        }
        if name == 'separation_exponent_from_log_ratio':
            return {
                **base,
                'key': 'adaptive_dimension:residual_strength_exponent',
                'name': 'residual_strength_exponent',
                'dimension_kind': 'scale_free_metric_axis',
                'expression': proposal.get('expression'),
                'first_principles': [
                    'order_metric',
                    'inverse_operation',
                    'composition',
                    'dimension_lift',
                ],
            }
        if name == 'signed_boundary_margin':
            return {
                **base,
                'key': 'adaptive_dimension:signed_boundary_margin',
                'name': 'signed_boundary_margin',
                'dimension_kind': 'domain_margin_axis',
                'expression': proposal.get('expression'),
                'first_principles': [
                    'order_metric',
                    'domain_partition',
                    'dimension_lift',
                ],
            }
        if name == 'local_taper_fraction':
            return {
                **base,
                'key': 'adaptive_dimension:local_taper_fraction',
                'name': 'local_taper_fraction',
                'dimension_kind': 'continuous_domain_coordinate',
                'expression': proposal.get('expression'),
                'first_principles': [
                    'order_metric',
                    'composition',
                    'domain_partition',
                    'dimension_lift',
                ],
            }
        if name == 'signed_residual_alignment_pair':
            return {
                **base,
                'key': 'adaptive_dimension:signed_residual_alignment_pair',
                'name': 'signed_residual_alignment_pair',
                'dimension_kind': 'projection_basis',
                'expression': proposal.get('expression'),
                'first_principles': [
                    'symmetry_transform',
                    'composition',
                    'dimension_lift',
                ],
            }
        if 'domain_predicate' in name:
            return {
                **base,
                'key': f"adaptive_dimension:domain_indicator:{proposal.get('key')}",
                'name': 'learned_domain_indicator',
                'dimension_kind': 'domain_indicator',
                'expression': proposal.get('expression'),
                'first_principles': [
                    'identity_equality',
                    'domain_partition',
                    'dimension_lift',
                ],
            }
        if str(proposal.get('proposal_kind')) == 'operator_prior':
            return {
                **base,
                'key': f"adaptive_dimension:operator_reuse_axis:{proposal.get('key')}",
                'name': 'operator_reuse_axis',
                'dimension_kind': 'operator_context_axis',
                'expression': proposal.get('expression'),
                'first_principles': [
                    'identity_equality',
                    'composition',
                    'dimension_lift',
                ],
            }
        return None

    def _adaptive_dimension_from_operator_anomaly(
        self,
        anomaly: dict[str, Any],
    ) -> dict[str, Any] | None:
        operator_key = str(anomaly.get('operator_key', 'unknown'))
        failure_context = str(anomaly.get('failure_context', 'unknown'))
        parameters = dict(anomaly.get('parameters') or {})
        expression = 'domain_success_score(operator_signature, context, residual_shape)'
        first_principles = ['domain_partition', 'identity_equality', 'dimension_lift']
        if 'cutoff_radius' in parameters:
            expression = 'boundary_margin = 1 - separation / cutoff_radius'
            first_principles = ['order_metric', 'domain_partition', 'dimension_lift']
        elif 'distance_exponent' in parameters:
            expression = 'residual_strength_log_ratio_error(distance_exponent)'
            first_principles = [
                'order_metric',
                'inverse_operation',
                'composition',
                'dimension_lift',
            ]
        severity = float(anomaly.get('severity', 0.0) or 0.0)
        return {
            'key': f'adaptive_dimension:operator_anomaly:{operator_key}:{failure_context}',
            'name': 'operator_anomaly_separator',
            'dimension_kind': 'failure_separator_axis',
            'source': 'operator_prior_anomaly',
            'theory_kind': f"operator_prior:{anomaly.get('operator_kind', 'unknown')}",
            'priority': round(min(1.0, 0.83 + 0.12 * severity), 3),
            'expression': expression,
            'first_principles': first_principles,
            'reason': (
                'the same invented operator works in one context and fails in '
                'another, so add a coordinate that predicts the split'
            ),
            'expected_use': (
                'gate future operator tests by a learned success/failure axis '
                'instead of assuming the current visible dimensions are complete'
            ),
            'evidence': {
                'operator_key': operator_key,
                'operator_kind': anomaly.get('operator_kind'),
                'support_contexts': list(anomaly.get('support_contexts') or []),
                'failure_context': failure_context,
                'support_count': len(anomaly.get('support_contexts') or []),
                'score_gap': anomaly.get('score_gap', 0.0),
                'parameters': _rounded_dict(parameters),
            },
            'expansion_policy': (
                'keep only if it predicts at least one failure context without '
                'removing known success contexts'
            ),
        }

    def _adaptive_dimension_from_generalization_gap(
        self,
        gap: dict[str, Any],
    ) -> dict[str, Any] | None:
        status = str(gap.get('status', 'unknown'))
        if status not in {'local', 'domain_limited', 'needs_counterexample'}:
            return None
        theory_kind = str(gap.get('theory_kind', 'unknown'))
        evidence = dict(gap.get('proof_evidence') or {})
        return {
            'key': f'adaptive_dimension:generalization_gap:{theory_kind}',
            'name': 'generalization_context_axis',
            'dimension_kind': 'context_transfer_axis',
            'source': 'generalization_gap',
            'theory_kind': theory_kind,
            'priority': round(0.72 + min(0.18, 0.03 * int(evidence.get('support_count', 0) or 0)), 3),
            'expression': 'context_embedding that preserves successes and exposes transfer failures',
            'first_principles': [
                'identity_equality',
                'domain_partition',
                'symmetry_transform',
                'dimension_lift',
            ],
            'reason': 'a local or narrowed family needs a coordinate for when it transfers',
            'expected_use': 'choose transfer tests and hidden holdouts by structural context, not labels',
            'evidence': {
                **evidence,
                'status': status,
            },
            'expansion_policy': (
                'retain only if it improves transfer prediction or generates a falsifying holdout'
            ),
        }

    def _operator_priors_from_disagreement(
        self,
        record: dict[str, Any],
    ) -> list[dict[str, Any]]:
        mode = str(record.get('mode', 'unknown'))
        center = self._center_from_probe_points(list(record.get('probe_points') or []))
        if center is None:
            return []
        relation = self._relation_from_record(record)
        if mode == 'distance_exponent_race':
            return self._distance_operator_priors_from_record(record, center, relation)
        if mode == 'cutoff_boundary_vs_smooth_falloff':
            return self._cutoff_operator_priors_from_record(record, center, relation)
        if mode == 'taper_shape_vs_hard_boundary':
            return self._taper_operator_priors_from_record(record, center, relation)
        return []

    def _distance_operator_priors_from_record(
        self,
        record: dict[str, Any],
        center: tuple[float, float],
        relation: str,
    ) -> list[dict[str, Any]]:
        exponents = self._exponents_from_record(record)
        if not exponents:
            return []
        cx, cy = center
        priors = []
        for exponent in exponents:
            priors.append({
                'key': (
                    f"operator:memory_prior:inverse_separation_power:"
                    f"{exponent}:{relation}:{record.get('context', 'unknown')}:{record.get('seed', '?')}"
                ),
                'operator_kind': 'inverse_separation_power',
                'inputs': ['center', 'position', 'distance_exponent'],
                'expression': (
                    f"{self._base_vector_expression(relation)} / separation^{exponent}"
                ),
                'generated_from': 'representation:separation_exponent_from_log_ratio',
                'usefulness': self._operator_prior_usefulness(record),
                'test_hint': 'score remembered exponent rivals on new held-out residuals',
                'parameters': {
                    'center_x': cx,
                    'center_y': cy,
                    'distance_exponent': exponent,
                    'relation': relation,
                    'source_context': record.get('context'),
                },
            })
        return priors

    def _cutoff_operator_priors_from_record(
        self,
        record: dict[str, Any],
        center: tuple[float, float],
        relation: str,
    ) -> list[dict[str, Any]]:
        radius = self._cutoff_radius_from_record(record)
        if radius is None:
            return []
        cx, cy = center
        return [{
            'key': (
                f"operator:memory_prior:localized_cutoff_window:"
                f"{radius}:{relation}:{record.get('context', 'unknown')}:{record.get('seed', '?')}"
            ),
            'operator_kind': 'localized_cutoff_window',
            'inputs': ['center', 'position', 'cutoff_radius'],
            'expression': (
                f"inside(separation <= {radius}) * {self._base_vector_expression(relation)}"
            ),
            'generated_from': 'representation:signed_boundary_margin',
            'usefulness': self._operator_prior_usefulness(record),
            'test_hint': 'score remembered inside/outside boundary on held-out residuals',
            'parameters': {
                'center_x': cx,
                'center_y': cy,
                'cutoff_radius': radius,
                'relation': relation,
                'source_context': record.get('context'),
            },
        }]

    def _taper_operator_priors_from_record(
        self,
        record: dict[str, Any],
        center: tuple[float, float],
        relation: str,
    ) -> list[dict[str, Any]]:
        radius = self._cutoff_radius_from_record(record)
        if radius is None:
            return []
        cx, cy = center
        exponents = self._exponents_from_record(record) or [1.0, 2.0]
        priors = []
        for exponent in exponents[:3]:
            priors.append({
                'key': (
                    f"operator:memory_prior:localized_tapered_power:"
                    f"{radius}:{exponent}:{relation}:"
                    f"{record.get('context', 'unknown')}:{record.get('seed', '?')}"
                ),
                'operator_kind': 'localized_tapered_power',
                'inputs': ['center', 'position', 'cutoff_radius', 'distance_exponent'],
                'expression': (
                    f"inside(separation <= {radius}) * "
                    f"max(0, 1 - separation/{radius}) * "
                    f"{self._base_vector_expression(relation)} / separation^{exponent}"
                ),
                'generated_from': 'representation:local_taper_fraction',
                'usefulness': max(0.0, self._operator_prior_usefulness(record) - 0.01),
                'test_hint': 'score remembered continuous taper on center, mid, and boundary residuals',
                'parameters': {
                    'center_x': cx,
                    'center_y': cy,
                    'cutoff_radius': radius,
                    'distance_exponent': exponent,
                    'relation': relation,
                    'source_context': record.get('context'),
                },
            })
        return priors

    def _center_from_probe_points(
        self,
        points: list[dict[str, Any]],
    ) -> tuple[float, float] | None:
        radial_points = [
            point for point in points
            if isinstance(point.get('x'), (int, float))
            and isinstance(point.get('y'), (int, float))
            and isinstance(point.get('distance_from_center'), (int, float))
        ]
        if not radial_points:
            return None
        center_x_values = [
            float(point['x']) - float(point['distance_from_center'])
            for point in radial_points
        ]
        center_y_values = [float(point['y']) for point in radial_points]
        cx = sum(center_x_values) / len(center_x_values)
        cy = sum(center_y_values) / len(center_y_values)
        return (round(cx, 6), round(cy, 6))

    def _relation_from_record(self, record: dict[str, Any]) -> str:
        text = ' '.join(
            str(value)
            for value in (
                list(record.get('family_kinds') or [])
                + list(record.get('rival_labels') or [])
            )
        )
        return 'perpendicular' if 'perpendicular' in text else 'direction'

    def _exponents_from_record(self, record: dict[str, Any]) -> list[float]:
        exponents = set()
        texts = list(record.get('rival_labels') or [])
        texts.extend(
            str(item.get('prediction', ''))
            for item in list(record.get('rival_predictions') or [])
        )
        for text in texts:
            for chunk in str(text).split('separation^-')[1:]:
                token = chunk.split()[0].split('/')[0].rstrip('.,;:)')
                try:
                    exponents.add(round(float(token), 3))
                except ValueError:
                    continue
        return sorted(exponents)

    def _cutoff_radius_from_record(self, record: dict[str, Any]) -> float | None:
        points = [
            point for point in list(record.get('probe_points') or [])
            if isinstance(point.get('distance_from_center'), (int, float))
        ]
        if not points:
            return None
        boundary_points = [
            point for point in points
            if 'boundary' in str(point.get('label', ''))
        ]
        source = boundary_points or points
        distances = sorted(float(point['distance_from_center']) for point in source)
        if not distances:
            return None
        radius = (distances[0] + distances[-1]) / 2.0
        return round(max(0.5, radius), 3)

    def _base_vector_expression(self, relation: str) -> str:
        if relation == 'perpendicular':
            return 'perpendicular(unit(center - position))'
        return 'unit(center - position)'

    def _operator_prior_usefulness(self, record: dict[str, Any]) -> float:
        score = float(record.get('mean_rival_score', 0.0) or 0.0)
        return round(max(0.25, min(0.95, score)), 3)


class AutonomousDiscoveryLoop:
    """
    Convert fitted equations into an explicit mini scientific method.

    The loop is deliberately label-free: it does not say gravity, vortex, or
    force. It works in terms of residuals, centers, phase, direction, and
    competing explanations.
    """

    def __init__(self, min_theory_score: float = 0.18):
        self.min_theory_score = min_theory_score

    def build_report(
        self,
        equations: list,
        step: int,
        current_count: int = 0,
        world_width: float = 20.0,
        world_height: float = 20.0,
    ) -> DiscoveryCycleReport:
        theories = self._build_theories(equations)
        concept_proposals = self._concept_proposals(theories)
        operator_proposals = self._operator_proposals(theories, concept_proposals)
        proof_checks = self._proof_checks(theories, operator_proposals)
        probe_plan = self._choose_probe(
            theories,
            current_count=current_count,
            world_width=world_width,
            world_height=world_height,
        )
        return DiscoveryCycleReport(
            step=step,
            theories=theories,
            concept_proposals=concept_proposals,
            operator_proposals=operator_proposals,
            proof_checks=proof_checks,
            probe_plan=probe_plan,
            open_questions=self._open_questions(theories, probe_plan),
        )

    def _build_theories(self, equations: list) -> list[TheoryRecord]:
        records = []
        for equation in equations:
            role = _get(equation, 'role', '')
            score = float(_get(equation, 'score', 0.0) or 0.0)
            if score < self.min_theory_score:
                continue
            theory = self._theory_from_equation(equation, role, score)
            if theory is not None:
                records.append(theory)
        records.sort(
            key=lambda item: (
                self._theory_priority(item.theory_kind),
                item.score,
                -item.uncertainty,
            ),
            reverse=True,
        )
        return records[:8]

    def _theory_from_equation(
        self,
        equation,
        role: str,
        score: float,
    ) -> TheoryRecord | None:
        key = str(_get(equation, 'key', 'unknown'))
        target = str(_get(equation, 'target', 'unknown_target'))
        expression = str(_get(equation, 'expression', 'unknown_expression'))
        mse = float(_get(equation, 'mse', 0.0) or 0.0)
        baseline_mse = float(_get(equation, 'baseline_mse', 0.0) or 0.0)
        parameters = dict(_get(equation, 'parameters', {}) or {})
        failures = self._failure_notes(mse, baseline_mse)
        status = 'promising' if score >= 0.62 else 'tentative'
        uncertainty = max(0.0, min(1.0, 1.0 - score))

        if role in {'residual_periodic_equation', 'generated_operator_periodic_equation'}:
            period = parameters.get('period_steps', '?')
            concept_keys = [
                f'concept:phase_from_step:{period}',
                f'concept:residual_axis:{target}',
            ]
            return TheoryRecord(
                key=f'theory:{key}',
                theory_kind='generated_periodic_residual'
                if role == 'generated_operator_periodic_equation'
                else 'periodic_residual',
                source_equation_key=key,
                claim=f'{target} has a residual component that repeats by step phase',
                explains=[
                    'simple scalar drift misses a signed repeating remainder',
                    f'phase template {expression} compresses the remainder',
                ],
                failures=failures,
                score=score,
                uncertainty=uncertainty,
                concept_keys=concept_keys,
                next_experiment='wait_to_compare_later_phase',
                status=status,
                target=target,
                expression=expression,
                parameters=parameters,
            )

        if role in {
            'generated_operator_tapered_distance_direction_equation',
            'generated_operator_tapered_distance_perpendicular_equation',
        }:
            relation = 'perpendicular' if 'perpendicular' in role else 'direction'
            cutoff_radius = parameters.get('cutoff_radius', '?')
            exponent = parameters.get('distance_exponent', '?')
            concept_keys = [
                'concept:baseline_adjusted_residual',
                'concept:inferred_local_center',
                f'concept:{relation}_field_axis',
                f'concept:localized_cutoff_region:{cutoff_radius}',
                f'concept:distance_strength_exponent:{exponent}',
                f'concept:boundary_taper:{cutoff_radius}',
            ]
            return TheoryRecord(
                key=f'theory:{key}',
                theory_kind=f'generated_tapered_distance_{relation}_residual',
                source_equation_key=key,
                claim=(
                    f'{target} is better explained by a local {relation} residual '
                    f'with separation exponent {exponent} that tapers to zero near radius {cutoff_radius}'
                ),
                explains=[
                    'hard cutoff and smooth global falloff each leave a shape error behind',
                    f'{expression} compresses direction, strength, and boundary behavior together',
                ],
                failures=failures,
                score=score,
                uncertainty=uncertainty,
                concept_keys=concept_keys,
                next_experiment='sample_center_mid_boundary_and_outside_the_local_region',
                status=status,
                target=target,
                expression=expression,
                parameters=parameters,
            )

        if role in {
            'generated_operator_cutoff_direction_equation',
            'generated_operator_cutoff_perpendicular_equation',
        }:
            relation = 'perpendicular' if 'perpendicular' in role else 'direction'
            cutoff_radius = parameters.get('cutoff_radius', '?')
            concept_keys = [
                'concept:baseline_adjusted_residual',
                'concept:inferred_local_center',
                f'concept:{relation}_field_axis',
                f'concept:localized_cutoff_region:{cutoff_radius}',
            ]
            return TheoryRecord(
                key=f'theory:{key}',
                theory_kind=f'generated_cutoff_{relation}_residual',
                source_equation_key=key,
                claim=(
                    f'{target} is better explained by a {relation} residual '
                    f'that only applies inside radius {cutoff_radius}'
                ),
                explains=[
                    'a global residual direction overpredicts samples outside the active region',
                    f'{expression} compresses the residual by adding a domain condition',
                ],
                failures=failures,
                score=score,
                uncertainty=uncertainty,
                concept_keys=concept_keys,
                next_experiment='sample_inside_and_outside_the_inferred_cutoff',
                status=status,
                target=target,
                expression=expression,
                parameters=parameters,
            )

        if role in {
            'local_residual_distance_scaled_direction_equation',
            'local_residual_distance_scaled_perpendicular_equation',
            'residual_distance_scaled_direction_equation',
            'residual_distance_scaled_perpendicular_equation',
            'generated_operator_distance_scaled_direction_equation',
            'generated_operator_distance_scaled_perpendicular_equation',
        }:
            relation = 'perpendicular' if 'perpendicular' in role else 'direction'
            local_prefix = 'local_' if role.startswith('local_') else ''
            generated_prefix = 'generated_' if role.startswith('generated_operator_') else ''
            exponent = parameters.get('distance_exponent', '?')
            concept_keys = [
                'concept:baseline_adjusted_residual',
                'concept:inferred_local_center' if local_prefix else 'concept:inferred_center',
                f'concept:{relation}_field_axis',
                f'concept:distance_strength_exponent:{exponent}',
            ]
            return TheoryRecord(
                key=f'theory:{key}',
                theory_kind=f'{generated_prefix}{local_prefix}distance_scaled_{relation}_residual',
                source_equation_key=key,
                claim=(
                    f'{target} is better explained when residual strength changes '
                    f'with separation exponent {exponent}'
                ),
                explains=[
                    'plain residual direction leaves strength error behind',
                    f'{expression} compresses both residual direction and residual size',
                ],
                failures=failures,
                score=score,
                uncertainty=uncertainty,
                concept_keys=concept_keys,
                next_experiment='compare_near_and_far_samples_from_inferred_center',
                status=status,
                target=target,
                expression=expression,
                parameters=parameters,
            )

        if role in {'local_residual_direction_equation', 'local_residual_perpendicular_equation'}:
            relation = 'perpendicular' if 'perpendicular' in role else 'direction'
            concept_keys = [
                'concept:local_high_change_region',
                'concept:inferred_local_center',
                f'concept:{relation}_residual_axis',
            ]
            return TheoryRecord(
                key=f'theory:{key}',
                theory_kind=f'local_{relation}_residual',
                source_equation_key=key,
                claim=(
                    f'{target} is better explained near an inferred local center '
                    f'by a {relation} residual'
                ),
                explains=[
                    'residuals are concentrated in a local high-change subset',
                    f'{expression} compresses paired channel changes after baseline removal',
                ],
                failures=failures,
                score=score,
                uncertainty=uncertainty,
                concept_keys=concept_keys,
                next_experiment='spawn_near_and_away_from_inferred_center',
                status=status,
                target=target,
                expression=expression,
                parameters=parameters,
            )

        if role in {'residual_direction_equation', 'residual_perpendicular_equation'}:
            relation = 'perpendicular' if 'perpendicular' in role else 'direction'
            concept_keys = [
                'concept:baseline_adjusted_residual',
                'concept:inferred_center',
                f'concept:{relation}_field_axis',
            ]
            return TheoryRecord(
                key=f'theory:{key}',
                theory_kind=f'{relation}_residual',
                source_equation_key=key,
                claim=f'{target} contains a {relation} residual after simple drift is removed',
                explains=[
                    'baseline-adjusted residuals align with a geometric vector',
                    f'{expression} compresses paired channel changes',
                ],
                failures=failures,
                score=score,
                uncertainty=uncertainty,
                concept_keys=concept_keys,
                next_experiment='sample_new_locations_around_inferred_center',
                status=status,
                target=target,
                expression=expression,
                parameters=parameters,
            )

        if role in {'vector_direction_equation', 'vector_perpendicular_equation'}:
            relation = 'perpendicular' if 'perpendicular' in role else 'direction'
            return TheoryRecord(
                key=f'theory:{key}',
                theory_kind=f'{relation}_vector_relation',
                source_equation_key=key,
                claim=f'{target} changes with a reusable {relation} vector relation',
                explains=[f'{expression} predicts paired channel changes'],
                failures=failures,
                score=score,
                uncertainty=uncertainty,
                concept_keys=[f'concept:{relation}_vector_relation'],
                next_experiment='sample_same_relation_at_new_locations',
                status=status,
                target=target,
                expression=expression,
                parameters=parameters,
            )

        if role in {'position_update_equation', 'constant_change_equation'}:
            return TheoryRecord(
                key=f'theory:{key}',
                theory_kind='simple_transition',
                source_equation_key=key,
                claim=f'{target} follows a compact simple-transition rule',
                explains=[f'{expression} predicts the repeated transition'],
                failures=failures,
                score=score,
                uncertainty=uncertainty,
                concept_keys=['concept:simple_transition_baseline'],
                next_experiment='look_for_residuals_after_simple_transition',
                status=status,
                target=target,
                expression=expression,
                parameters=parameters,
            )

        return None

    def _concept_proposals(self, theories: list[TheoryRecord]) -> list[ConceptProposal]:
        proposals: dict[str, ConceptProposal] = {}
        for theory in theories:
            for concept_key in theory.concept_keys:
                proposals[concept_key] = self._proposal_from_concept_key(concept_key, theory)
        ranked = sorted(
            proposals.values(),
            key=lambda item: (item.usefulness, item.key),
            reverse=True,
        )
        return ranked[:10]

    def _proposal_from_concept_key(
        self,
        concept_key: str,
        theory: TheoryRecord,
    ) -> ConceptProposal:
        parameters = dict(theory.parameters)
        usefulness = max(0.0, min(1.0, theory.score - 0.04 * theory.uncertainty))
        if 'phase_from_step' in concept_key:
            period = parameters.get('period_steps', '?')
            return ConceptProposal(
                key=concept_key,
                concept_kind='phase',
                basis='residuals recur at similar step offsets',
                expression_seed=f'(step mod {period}) / {period}',
                usefulness=usefulness,
                parameters={'period_steps': period},
            )
        if 'distance_strength_exponent' in concept_key:
            exponent = parameters.get('distance_exponent', '?')
            return ConceptProposal(
                key=concept_key,
                concept_kind='distance_strength_law',
                basis='direction-only residuals still miss how large the change should be',
                expression_seed=f'residual magnitude ~ 1 / separation^{exponent}',
                usefulness=usefulness,
                parameters={
                    'distance_exponent': exponent,
                    'distance_mse_improvement': parameters.get('distance_mse_improvement'),
                },
            )
        if 'localized_cutoff_region' in concept_key:
            cutoff_radius = parameters.get('cutoff_radius', '?')
            return ConceptProposal(
                key=concept_key,
                concept_kind='localized_cutoff_region',
                basis='residuals improve when the field has an inside/outside domain',
                expression_seed=f'inside(separation <= {cutoff_radius})',
                usefulness=usefulness,
                parameters={
                    'cutoff_radius': cutoff_radius,
                    'cutoff_mse_improvement': parameters.get('cutoff_mse_improvement'),
                },
            )
        if 'boundary_taper' in concept_key:
            cutoff_radius = parameters.get('cutoff_radius', '?')
            return ConceptProposal(
                key=concept_key,
                concept_kind='boundary_taper',
                basis='residual strength fades inside a local domain instead of switching abruptly',
                expression_seed=f'max(0, 1 - separation / {cutoff_radius})',
                usefulness=usefulness,
                parameters={
                    'cutoff_radius': cutoff_radius,
                    'tapered_vs_cutoff_improvement': parameters.get('tapered_vs_cutoff_improvement'),
                },
            )
        if 'inferred_local_center' in concept_key:
            return ConceptProposal(
                key=concept_key,
                concept_kind='local_center',
                basis='largest residuals point toward a shared off-center location',
                expression_seed='argmin perpendicular residual-line distance',
                usefulness=usefulness,
                parameters={
                    'center_x': parameters.get('center_x'),
                    'center_y': parameters.get('center_y'),
                },
            )
        if 'local_high_change_region' in concept_key:
            return ConceptProposal(
                key=concept_key,
                concept_kind='residual_cluster',
                basis='the strongest residuals form a reusable subset',
                expression_seed='top residual magnitude quantile',
                usefulness=usefulness,
                parameters={},
            )
        if 'perpendicular' in concept_key:
            return ConceptProposal(
                key=concept_key,
                concept_kind='orthogonal_direction',
                basis='paired residual channels rotate relative to the center vector',
                expression_seed='(-unit_y, unit_x)',
                usefulness=usefulness,
                parameters={},
            )
        if 'direction' in concept_key or 'field_axis' in concept_key:
            return ConceptProposal(
                key=concept_key,
                concept_kind='direction_axis',
                basis='paired residual channels align with a center vector',
                expression_seed='unit(center - position)',
                usefulness=usefulness,
                parameters={},
            )
        if 'residual_axis' in concept_key:
            return ConceptProposal(
                key=concept_key,
                concept_kind='residual_channel',
                basis='baseline-adjusted error becomes a modeled quantity',
                expression_seed='observed_delta - simple_baseline_delta',
                usefulness=usefulness,
                parameters={},
            )
        return ConceptProposal(
            key=concept_key,
            concept_kind='transition_abstraction',
            basis='compact rule explains repeated transitions',
            expression_seed='next_state - current_state',
            usefulness=usefulness,
            parameters={},
        )

    def _operator_proposals(
        self,
        theories: list[TheoryRecord],
        concepts: list[ConceptProposal],
    ) -> list[OperatorProposal]:
        proposals = {}
        concepts_by_key = {concept.key: concept for concept in concepts}
        for theory in theories:
            usefulness = max(0.0, min(1.0, theory.score - 0.03 * theory.uncertainty))
            if theory.theory_kind in {'periodic_residual', 'generated_periodic_residual'}:
                period = theory.parameters.get('period_steps', '?')
                proposals[f'operator:phase_basis:{period}'] = OperatorProposal(
                    key=f'operator:phase_basis:{period}',
                    operator_kind='phase_basis',
                    inputs=['step', 'period'],
                    expression=f'(sin(2*pi*step/{period}), cos(2*pi*step/{period}))',
                    generated_from='concept:phase_from_step',
                    usefulness=usefulness,
                    test_hint='hold position variables similar and compare residual sign across phases',
                    parameters={'period_steps': period},
                )
            if 'distance_scaled' in theory.theory_kind:
                exponent = theory.parameters.get('distance_exponent', '?')
                relation = 'perpendicular' if 'perpendicular' in theory.theory_kind else 'direction'
                base_vector = (
                    'perpendicular(unit(center - position))'
                    if relation == 'perpendicular'
                    else 'unit(center - position)'
                )
                proposals[f'operator:inverse_separation_power:{exponent}:{relation}'] = OperatorProposal(
                    key=f'operator:inverse_separation_power:{exponent}:{relation}',
                    operator_kind='inverse_separation_power',
                    inputs=['center', 'position', 'distance_exponent'],
                    expression=f'{base_vector} / separation^{exponent}',
                    generated_from=f'concept:distance_strength_exponent:{exponent}',
                    usefulness=usefulness,
                    test_hint='compare residual magnitudes at matched directions but different separations',
                    parameters={
                        'distance_exponent': exponent,
                        'relation': relation,
                        'center_x': theory.parameters.get('center_x'),
                        'center_y': theory.parameters.get('center_y'),
                    },
                )
            proposes_local_window = theory.theory_kind in {
                'local_direction_residual',
                'local_perpendicular_residual',
            }
            proposes_taper = (
                proposes_local_window
                or theory.theory_kind.startswith('generated_cutoff_')
            )
            if proposes_local_window or proposes_taper:
                relation = 'perpendicular' if 'perpendicular' in theory.theory_kind else 'direction'
                cutoff_radius = theory.parameters.get('cutoff_radius')
                base_vector = (
                    'perpendicular(unit(center - position))'
                    if relation == 'perpendicular'
                    else 'unit(center - position)'
                )
                radius_label = cutoff_radius if cutoff_radius is not None else '?'
                if proposes_local_window:
                    proposal_key = (
                        f'operator:localized_cutoff_window:{theory.source_equation_key}:{relation}'
                    )
                    proposals[proposal_key] = OperatorProposal(
                        key=proposal_key,
                        operator_kind='localized_cutoff_window',
                        inputs=['center', 'position', 'cutoff_radius'],
                        expression=f'inside(separation <= {radius_label}) * {base_vector}',
                        generated_from=(
                            f'concept:localized_cutoff_region:{cutoff_radius}'
                            if cutoff_radius is not None
                            else 'concept:local_high_change_region'
                        ),
                        usefulness=usefulness,
                        test_hint='compare matched samples just inside and just outside the inferred radius',
                        parameters={
                            'cutoff_radius': cutoff_radius,
                            'relation': relation,
                            'center_x': theory.parameters.get('center_x'),
                            'center_y': theory.parameters.get('center_y'),
                        },
                    )
                exponent = theory.parameters.get('distance_exponent')
                exponent_label = exponent if exponent is not None else '?'
                tapered_key = (
                    f'operator:localized_tapered_power:{theory.source_equation_key}:{relation}'
                )
                proposals[tapered_key] = OperatorProposal(
                    key=tapered_key,
                    operator_kind='localized_tapered_power',
                    inputs=['center', 'position', 'cutoff_radius', 'distance_exponent'],
                    expression=(
                        f'inside(separation <= {radius_label}) * '
                        f'max(0, 1 - separation/{radius_label}) * '
                        f'{base_vector} / separation^{exponent_label}'
                    ),
                    generated_from=(
                        f'concept:boundary_taper:{cutoff_radius}'
                        if cutoff_radius is not None
                        else 'concept:local_high_change_region'
                    ),
                    usefulness=max(0.0, usefulness - 0.01),
                    test_hint='compare center, middle, boundary, and outside samples from the same inferred center',
                    parameters={
                        'cutoff_radius': cutoff_radius,
                        'distance_exponent': exponent,
                        'relation': relation,
                        'center_x': theory.parameters.get('center_x'),
                        'center_y': theory.parameters.get('center_y'),
                    },
                )
            if 'direction' in theory.theory_kind and 'distance_scaled' not in theory.theory_kind:
                proposals[f'operator:normalize_vector:{theory.source_equation_key}'] = OperatorProposal(
                    key=f'operator:normalize_vector:{theory.source_equation_key}',
                    operator_kind='normalize_vector',
                    inputs=['center', 'position'],
                    expression='unit(center - position)',
                    generated_from='concept:direction_axis',
                    usefulness=usefulness,
                    test_hint='move the probe while preserving direction and check residual alignment',
                    parameters={},
                )
            if 'perpendicular' in theory.theory_kind and 'distance_scaled' not in theory.theory_kind:
                proposals[f'operator:rotate_quarter_turn:{theory.source_equation_key}'] = OperatorProposal(
                    key=f'operator:rotate_quarter_turn:{theory.source_equation_key}',
                    operator_kind='rotate_quarter_turn',
                    inputs=['direction_axis'],
                    expression='(-unit_y, unit_x)',
                    generated_from='concept:orthogonal_direction',
                    usefulness=usefulness,
                    test_hint='sample around the center and check whether residuals rotate with position',
                    parameters={},
                )
            if any(key in concepts_by_key for key in theory.concept_keys if 'inferred' in key):
                proposals[f'operator:center_from_residual_lines:{theory.source_equation_key}'] = OperatorProposal(
                    key=f'operator:center_from_residual_lines:{theory.source_equation_key}',
                    operator_kind='center_from_residual_lines',
                    inputs=['position', 'residual_vector'],
                    expression='argmin distance from candidate center to residual-aligned lines',
                    generated_from='concept:inferred_center',
                    usefulness=usefulness,
                    test_hint='new probes near the inferred center should change residual magnitude or direction',
                    parameters={
                        'center_x': theory.parameters.get('center_x'),
                        'center_y': theory.parameters.get('center_y'),
                    },
                )
        ranked = sorted(
            proposals.values(),
            key=lambda item: (
                self._operator_priority(item.operator_kind),
                item.usefulness,
                item.key,
            ),
            reverse=True,
        )
        return ranked[:10]

    def _operator_priority(self, operator_kind: str) -> float:
        priorities = {
            'phase_basis': 4.0,
            'inverse_separation_power': 3.8,
            'localized_tapered_power': 3.75,
            'localized_cutoff_window': 3.7,
            'center_from_residual_lines': 2.5,
            'rotate_quarter_turn': 2.0,
            'normalize_vector': 1.8,
        }
        return priorities.get(operator_kind, 0.0)

    def _proof_checks(
        self,
        theories: list[TheoryRecord],
        operators: list[OperatorProposal],
    ) -> list[ProofCheck]:
        checks = []
        operator_kinds = {operator.operator_kind for operator in operators}
        for theory in theories[:5]:
            if 'tapered_distance' in theory.theory_kind:
                cutoff_radius = theory.parameters.get('cutoff_radius')
                exponent = theory.parameters.get('distance_exponent')
                smooth_improvement = theory.parameters.get('tapered_vs_smooth_improvement', 0.0)
                cutoff_improvement = theory.parameters.get('tapered_vs_cutoff_improvement', 0.0)
                passed = (
                    isinstance(cutoff_radius, (int, float))
                    and cutoff_radius > 0.0
                    and isinstance(exponent, (int, float))
                    and smooth_improvement is not None
                    and float(smooth_improvement) >= 0.03
                    and cutoff_improvement is not None
                    and float(cutoff_improvement) >= 0.03
                )
                checks.append(ProofCheck(
                    key=f'proof:{theory.key}:beats_cutoff_and_smooth_falloff',
                    check_kind='shape_contrast',
                    status='passed' if passed else 'open',
                    statement='tapered local operator must beat hard cutoff and global smooth falloff residual laws',
                    evidence={
                        'cutoff_radius': cutoff_radius,
                        'distance_exponent': exponent,
                        'tapered_vs_smooth_improvement': smooth_improvement,
                        'tapered_vs_cutoff_improvement': cutoff_improvement,
                    },
                ))
            elif 'cutoff' in theory.theory_kind:
                cutoff_radius = theory.parameters.get('cutoff_radius')
                improvement = theory.parameters.get('cutoff_mse_improvement', 0.0)
                smooth_improvement = theory.parameters.get('cutoff_vs_smooth_improvement', 0.0)
                passed = (
                    isinstance(cutoff_radius, (int, float))
                    and cutoff_radius > 0.0
                    and improvement is not None
                    and float(improvement) >= 0.04
                    and smooth_improvement is not None
                    and float(smooth_improvement) >= 0.03
                )
                checks.append(ProofCheck(
                    key=f'proof:{theory.key}:near_far_cutoff_contrast',
                    check_kind='near_far_contrast',
                    status='passed' if passed else 'open',
                    statement='localized operator must beat global and smooth-falloff residual laws on inside/outside samples',
                    evidence={
                        'cutoff_radius': cutoff_radius,
                        'cutoff_mse_improvement': improvement,
                        'cutoff_vs_smooth_improvement': smooth_improvement,
                    },
                ))
            elif 'distance_scaled' in theory.theory_kind:
                improvement = theory.parameters.get('distance_mse_improvement', 0.0)
                exponent = theory.parameters.get('distance_exponent')
                passed = (
                    isinstance(exponent, (int, float))
                    and improvement is not None
                    and float(improvement) >= 0.04
                    and 'inverse_separation_power' in operator_kinds
                )
                checks.append(ProofCheck(
                    key=f'proof:{theory.key}:beats_direction_only',
                    check_kind='simpler_model_contrast',
                    status='passed' if passed else 'open',
                    statement='distance-strength operator must beat the simpler direction-only residual',
                    evidence={
                        'distance_exponent': exponent,
                        'distance_mse_improvement': improvement,
                    },
                ))
            elif theory.theory_kind in {'periodic_residual', 'generated_periodic_residual'}:
                period = theory.parameters.get('period_steps')
                passed = (
                    isinstance(period, (int, float))
                    and float(period) > 0.0
                    and 'phase_basis' in operator_kinds
                )
                checks.append(ProofCheck(
                    key=f'proof:{theory.key}:phase_operator_valid',
                    check_kind='operator_domain',
                    status='passed' if passed else 'open',
                    statement='phase operator must have a positive period and paired cyclic basis',
                    evidence={'period_steps': period},
                ))
            elif 'residual' in theory.theory_kind:
                center_x = theory.parameters.get('center_x')
                center_y = theory.parameters.get('center_y')
                passed = isinstance(center_x, (int, float)) and isinstance(center_y, (int, float))
                checks.append(ProofCheck(
                    key=f'proof:{theory.key}:center_finite',
                    check_kind='operator_domain',
                    status='passed' if passed else 'open',
                    statement='center-based residual theories must carry a finite inferred center',
                    evidence={'center_x': center_x, 'center_y': center_y},
                ))
        return checks

    def _choose_probe(
        self,
        theories: list[TheoryRecord],
        current_count: int,
        world_width: float,
        world_height: float,
    ) -> DiscoveryProbePlan | None:
        if not theories:
            return None
        competitors = self._competing_theories(theories)
        top = competitors[0] if competitors else theories[0]
        if top.score < 0.25:
            return None

        if top.theory_kind in {'periodic_residual', 'generated_periodic_residual'}:
            period = top.parameters.get('period_steps')
            return DiscoveryProbePlan(
                action={
                    'type': 'wait',
                    'source': 'discovery_loop_probe',
                    'theory_key': top.key,
                    'period_steps': period,
                },
                theory_keys=[theory.key for theory in competitors[:2]] or [top.key],
                reason='wait because the next observation should change by phase, not location',
                expected_contrast='same simple baseline, different residual sign or amplitude by phase',
            )

        if 'residual' in top.theory_kind or 'vector_relation' in top.theory_kind:
            if current_count >= 12:
                return DiscoveryProbePlan(
                    action={
                        'type': 'wait',
                        'source': 'discovery_loop_probe',
                        'theory_key': top.key,
                    },
                    theory_keys=[theory.key for theory in competitors[:2]] or [top.key],
                    reason='wait because the world is already crowded enough to test residual behavior',
                    expected_contrast='new passive samples should strengthen one residual geometry',
                    disagreement_signature=self._passive_disagreement_signature(competitors),
                )
            action = self._spatial_probe_action(top, world_width, world_height)
            return DiscoveryProbePlan(
                action=action,
                theory_keys=[theory.key for theory in competitors[:2]] or [top.key],
                reason=self._spatial_probe_reason(top, competitors),
                expected_contrast=self._spatial_expected_contrast(top, competitors),
                disagreement_signature=self._spatial_disagreement_signature(
                    top,
                    competitors,
                    world_width,
                    world_height,
                ),
            )

        if top.theory_kind == 'simple_transition':
            return DiscoveryProbePlan(
                action={
                    'type': 'wait',
                    'source': 'discovery_loop_probe',
                    'theory_key': top.key,
                },
                theory_keys=[top.key],
                reason='collect more residuals because only a simple transition is explained so far',
                expected_contrast='unexplained error should either shrink or cluster into a new concept',
            )

        return None

    def _spatial_probe_action(
        self,
        theory: TheoryRecord,
        world_width: float,
        world_height: float,
    ) -> dict:
        cx = _coerce_float(theory.parameters.get('center_x'), world_width / 2.0)
        cy = _coerce_float(theory.parameters.get('center_y'), world_height / 2.0)
        radius = min(world_width, world_height) * 0.18
        if 'tapered_distance' in theory.theory_kind:
            cutoff_radius = _coerce_float(theory.parameters.get('cutoff_radius'), radius)
            offset = (max(radius * 0.45, cutoff_radius * 0.72), 0.0)
        elif 'cutoff' in theory.theory_kind:
            cutoff_radius = _coerce_float(theory.parameters.get('cutoff_radius'), radius)
            offset = (max(radius * 0.45, cutoff_radius * 1.08), 0.0)
        elif 'perpendicular' in theory.theory_kind:
            offset = (0.0, radius)
        elif 'distance_scaled' in theory.theory_kind:
            offset = (radius * 0.45, 0.0)
        else:
            offset = (radius, 0.0)
        x = min(max(cx + offset[0], 1.0), world_width - 1.0)
        y = min(max(cy + offset[1], 1.0), world_height - 1.0)
        return {
            'type': 'spawn',
            'x': x,
            'y': y,
            'vx': 0.0,
            'vy': 0.0,
            'source': 'discovery_loop_probe',
            'theory_key': theory.key,
        }

    def _spatial_disagreement_signature(
        self,
        top: TheoryRecord,
        competitors: list[TheoryRecord],
        world_width: float,
        world_height: float,
    ) -> dict[str, Any]:
        if len(competitors) < 2:
            return {}
        mode = self._disagreement_mode(competitors)
        cx = _coerce_float(top.parameters.get('center_x'), world_width / 2.0)
        cy = _coerce_float(top.parameters.get('center_y'), world_height / 2.0)
        cutoff_radius = self._first_numeric_parameter(competitors, 'cutoff_radius')
        if cutoff_radius is None:
            cutoff_radius = min(world_width, world_height) * 0.18
        if mode == 'taper_shape_vs_hard_boundary':
            distances = [
                ('near_center', cutoff_radius * 0.30),
                ('mid_region', cutoff_radius * 0.68),
                ('just_inside_boundary', cutoff_radius * 0.92),
                ('just_outside_boundary', cutoff_radius * 1.08),
            ]
        elif mode == 'cutoff_boundary_vs_smooth_falloff':
            distances = [
                ('inside_boundary', cutoff_radius * 0.82),
                ('just_outside_boundary', cutoff_radius * 1.08),
            ]
        elif mode == 'distance_exponent_race':
            distances = [
                ('near_center', cutoff_radius * 0.45),
                ('far_from_center', cutoff_radius * 1.65),
            ]
            probe_points = [
                self._radial_probe_point(label, cx, cy, distance, world_width, world_height)
                for label, distance in distances
            ]
        else:
            offset = min(world_width, world_height) * 0.18
            probe_points = [
                self._offset_probe_point('direction_test_east', cx, cy, offset, 0.0, world_width, world_height),
                self._offset_probe_point('direction_test_north', cx, cy, 0.0, offset, world_width, world_height),
            ]
            distances = []
        if mode in {
            'taper_shape_vs_hard_boundary',
            'cutoff_boundary_vs_smooth_falloff',
        }:
            probe_points = [
                self._radial_probe_point(label, cx, cy, distance, world_width, world_height)
                for label, distance in distances
            ]
        return {
            'mode': mode,
            'question': self._disagreement_question(mode),
            'probe_points': probe_points,
            'rival_predictions': [
                self._theory_prediction_signature(theory, mode)
                for theory in competitors[:3]
            ],
        }

    def _passive_disagreement_signature(
        self,
        competitors: list[TheoryRecord],
    ) -> dict[str, Any]:
        if len(competitors) < 2:
            return {}
        mode = self._disagreement_mode(competitors)
        return {
            'mode': mode,
            'question': self._disagreement_question(mode),
            'probe_points': [],
            'rival_predictions': [
                self._theory_prediction_signature(theory, mode)
                for theory in competitors[:3]
            ],
        }

    def _disagreement_mode(self, competitors: list[TheoryRecord]) -> str:
        kinds = {theory.theory_kind for theory in competitors}
        if any('tapered_distance' in kind for kind in kinds):
            return 'taper_shape_vs_hard_boundary'
        if (
            any('cutoff' in kind for kind in kinds)
            and any('distance_scaled' in kind for kind in kinds)
        ):
            return 'cutoff_boundary_vs_smooth_falloff'
        distance_exponents = {
            theory.parameters.get('distance_exponent')
            for theory in competitors
            if 'distance_scaled' in theory.theory_kind
        }
        if len(distance_exponents) >= 2:
            return 'distance_exponent_race'
        return 'vector_direction_disagreement'

    def _disagreement_question(self, mode: str) -> str:
        questions = {
            'taper_shape_vs_hard_boundary': (
                'Does residual strength fade through the local region or stay flat until a hard boundary?'
            ),
            'cutoff_boundary_vs_smooth_falloff': (
                'Does the residual switch off outside a finite domain or continue as a smooth falloff?'
            ),
            'distance_exponent_race': (
                'Which separation exponent keeps near and far residual magnitudes consistent?'
            ),
            'vector_direction_disagreement': (
                'Which residual vector direction survives at the same sampled location?'
            ),
        }
        return questions.get(mode, 'Which rival theory loses predictive power under the next probe?')

    def _theory_prediction_signature(
        self,
        theory: TheoryRecord,
        mode: str,
    ) -> dict[str, Any]:
        kind = theory.theory_kind
        if 'tapered_distance' in kind:
            cutoff_radius = theory.parameters.get('cutoff_radius', '?')
            exponent = theory.parameters.get('distance_exponent', '?')
            prediction = (
                f'graded residual: separation^{exponent} strength fades toward radius {cutoff_radius}'
            )
            falsified_if = 'mid-region or boundary samples behave like a flat hard cutoff or smooth global law'
        elif 'cutoff' in kind:
            cutoff_radius = theory.parameters.get('cutoff_radius', '?')
            prediction = f'flat local residual inside radius {cutoff_radius}, near-zero outside it'
            falsified_if = 'outside-boundary samples keep a coherent residual or inside strength varies smoothly with distance'
        elif 'distance_scaled' in kind:
            exponent = theory.parameters.get('distance_exponent', '?')
            prediction = f'smooth nonzero falloff proportional to separation^-{exponent}'
            falsified_if = 'residual collapses abruptly at a finite boundary or near/far ratio rejects the exponent'
        elif 'perpendicular' in kind:
            prediction = 'residual vector rotates a quarter turn from the center vector'
            falsified_if = 'new samples align with the center vector instead of its perpendicular'
        elif 'direction' in kind:
            prediction = 'residual vector aligns with the center vector'
            falsified_if = 'new samples consistently rotate away from the center vector'
        else:
            prediction = 'residual behavior should preserve this theory family'
            falsified_if = 'a rival theory explains the new samples with lower residual error'
        return {
            'theory_key': theory.key,
            'theory_kind': kind,
            'score': theory.score,
            'prediction': prediction,
            'falsified_if': falsified_if,
            'mode': mode,
        }

    def _radial_probe_point(
        self,
        label: str,
        center_x: float,
        center_y: float,
        distance: float,
        world_width: float,
        world_height: float,
    ) -> dict[str, Any]:
        x = min(max(center_x + distance, 1.0), world_width - 1.0)
        y = min(max(center_y, 1.0), world_height - 1.0)
        return {
            'label': label,
            'x': x,
            'y': y,
            'distance_from_center': abs(x - center_x),
        }

    def _offset_probe_point(
        self,
        label: str,
        center_x: float,
        center_y: float,
        dx: float,
        dy: float,
        world_width: float,
        world_height: float,
    ) -> dict[str, Any]:
        x = min(max(center_x + dx, 1.0), world_width - 1.0)
        y = min(max(center_y + dy, 1.0), world_height - 1.0)
        return {
            'label': label,
            'x': x,
            'y': y,
            'distance_from_center': ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5,
        }

    def _first_numeric_parameter(
        self,
        theories: list[TheoryRecord],
        name: str,
    ) -> float | None:
        for theory in theories:
            value = theory.parameters.get(name)
            if isinstance(value, (int, float)):
                return float(value)
        return None

    def _competing_theories(self, theories: list[TheoryRecord]) -> list[TheoryRecord]:
        residuals = [theory for theory in theories if 'residual' in theory.theory_kind]
        if len(residuals) >= 2:
            top = residuals[0]
            close = [
                theory for theory in residuals
                if top.score - theory.score <= 0.18
            ]
            if len({theory.theory_kind for theory in close}) >= 2:
                return close
            exponents = {
                theory.parameters.get('distance_exponent')
                for theory in close
                if 'distance_scaled' in theory.theory_kind
            }
            if len(exponents) >= 2:
                return close
        return [theories[0]]

    def _spatial_probe_reason(
        self,
        top: TheoryRecord,
        competitors: list[TheoryRecord],
    ) -> str:
        if len(competitors) >= 2:
            if any('tapered_distance' in theory.theory_kind for theory in competitors):
                return 'spawn inside the inferred region to separate tapered local shape from simpler residual laws'
            if (
                any('cutoff' in theory.theory_kind for theory in competitors)
                and any('distance_scaled' in theory.theory_kind for theory in competitors)
            ):
                return 'spawn near the inferred boundary to separate cutoff locality from smooth falloff'
            if all('distance_scaled' in theory.theory_kind for theory in competitors):
                return 'spawn where competing distance exponents predict different residual strengths'
            return 'spawn where competing residual theories predict different vector directions'
        if 'tapered_distance' in top.theory_kind:
            return 'spawn inside the inferred local region to test the taper shape'
        if 'cutoff' in top.theory_kind:
            return 'spawn near the inferred boundary to test whether the residual switches off'
        if 'distance_scaled' in top.theory_kind:
            return 'spawn near the inferred center to test the residual strength exponent'
        if top.theory_kind.startswith('local_'):
            return 'spawn near the inferred local center to test whether residuals intensify there'
        return 'spawn at a fresh location to test whether the vector relation transfers'

    def _spatial_expected_contrast(
        self,
        top: TheoryRecord,
        competitors: list[TheoryRecord],
    ) -> str:
        if len(competitors) >= 2:
            if any('tapered_distance' in theory.theory_kind for theory in competitors):
                return 'center, mid-region, and boundary samples should show a graded residual shape'
            if (
                any('cutoff' in theory.theory_kind for theory in competitors)
                and any('distance_scaled' in theory.theory_kind for theory in competitors)
            ):
                return 'inside-boundary samples should keep residual strength while outside samples should collapse toward baseline'
            if all('distance_scaled' in theory.theory_kind for theory in competitors):
                exponents = sorted({
                    str(theory.parameters.get('distance_exponent', '?'))
                    for theory in competitors
                })
                return f'near and far samples should select between exponents {", ".join(exponents)}'
            kinds = ', '.join(theory.theory_kind for theory in competitors[:2])
            return f'one of these theories should improve while the other weakens: {kinds}'
        if 'tapered_distance' in top.theory_kind:
            cutoff_radius = top.parameters.get('cutoff_radius', '?')
            exponent = top.parameters.get('distance_exponent', '?')
            return f'residuals should follow separation^{exponent} while tapering toward radius {cutoff_radius}'
        if 'cutoff' in top.theory_kind:
            cutoff_radius = top.parameters.get('cutoff_radius', '?')
            return f'samples beyond radius {cutoff_radius} should lose the residual rather than merely weaken'
        if 'distance_scaled' in top.theory_kind:
            exponent = top.parameters.get('distance_exponent', '?')
            return f'near and far samples should scale by roughly separation^{exponent}'
        if top.theory_kind.startswith('local_'):
            return 'near-center samples should have stronger baseline-adjusted residuals than far samples'
        return 'new samples should preserve the same direction or perpendicular alignment'

    def _failure_notes(self, mse: float, baseline_mse: float) -> list[str]:
        if baseline_mse <= 1e-12:
            return ['no meaningful residual remains after the simpler baseline']
        unexplained = mse / max(baseline_mse, 1e-12)
        notes = []
        if unexplained > 0.35:
            notes.append('large residual error remains after this explanation')
        elif unexplained > 0.12:
            notes.append('some residual error remains after this explanation')
        return notes

    def _open_questions(
        self,
        theories: list[TheoryRecord],
        probe_plan: DiscoveryProbePlan | None,
    ) -> list[str]:
        if not theories:
            return ['Which simple transition leaves a residual worth studying?']
        questions = []
        if probe_plan is None:
            questions.append('Is there enough evidence to choose an active falsification probe?')
        if not any(theory.theory_kind.startswith('local_') for theory in theories):
            questions.append('Do residuals concentrate around an unknown location?')
        if not any(
            theory.theory_kind in {'periodic_residual', 'generated_periodic_residual'}
            for theory in theories
        ):
            questions.append('Do residuals repeat by hidden phase or elapsed step?')
        if not any('cutoff' in theory.theory_kind for theory in theories):
            questions.append('Do residuals have a finite inside/outside domain?')
        if not any('tapered_distance' in theory.theory_kind for theory in theories):
            questions.append('Does local residual strength taper before it disappears?')
        if len({theory.theory_kind for theory in theories if 'residual' in theory.theory_kind}) >= 2:
            questions.append('Which residual theory fails fastest under a targeted probe?')
        return questions[:4]

    def _theory_priority(self, theory_kind: str) -> float:
        priorities = {
            'periodic_residual': 5.4,
            'generated_periodic_residual': 5.45,
            'generated_tapered_distance_direction_residual': 5.95,
            'generated_tapered_distance_perpendicular_residual': 5.95,
            'generated_cutoff_direction_residual': 5.85,
            'generated_cutoff_perpendicular_residual': 5.85,
            'generated_distance_scaled_direction_residual': 5.7,
            'generated_distance_scaled_perpendicular_residual': 5.7,
            'local_distance_scaled_direction_residual': 5.6,
            'local_distance_scaled_perpendicular_residual': 5.6,
            'distance_scaled_direction_residual': 5.5,
            'distance_scaled_perpendicular_residual': 5.5,
            'local_direction_residual': 5.3,
            'local_perpendicular_residual': 5.3,
            'direction_residual': 5.0,
            'perpendicular_residual': 5.0,
            'direction_vector_relation': 4.0,
            'perpendicular_vector_relation': 4.0,
            'simple_transition': 1.0,
        }
        return priorities.get(theory_kind, 0.0)


def _get(equation, name: str, default=None):
    if isinstance(equation, dict):
        return equation.get(name, default)
    return getattr(equation, name, default)


def _target_from_claim(claim: str) -> str:
    for marker in (
        ' is better explained',
        ' contains ',
        ' changes ',
        ' follows ',
        ' has ',
    ):
        if marker in claim:
            return claim.split(marker, 1)[0].strip()
    return ''


def _coerce_float(value, fallback: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return fallback


def _rounded_dict(values: dict) -> dict:
    return {
        key: _rounded_value(value)
        for key, value in values.items()
    }


def _rounded_value(value):
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return _rounded_dict(value)
    if isinstance(value, list):
        return [_rounded_value(item) for item in value]
    return value


def _format_number(value) -> str:
    if isinstance(value, (int, float)):
        rounded = round(float(value), 6)
        if rounded.is_integer():
            return str(int(rounded))
        return str(rounded).rstrip('0').rstrip('.')
    return str(value)
