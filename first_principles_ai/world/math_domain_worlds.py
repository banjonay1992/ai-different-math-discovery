from __future__ import annotations

"""
Deterministic math-domain worlds.

The manifest is benchmark-side truth. Observations are deliberately generic:
they expose objects, readings, events, and held-out contrasts, but not the
domain key, expected discovery labels, or falsifiers that scoring uses later.
"""

from dataclasses import dataclass, field
import math
import random
from typing import Any, Callable


FORBIDDEN_OBSERVATION_KEYS = {
    'answer',
    'benchmark',
    'benchmark_answer',
    'benchmark_truth',
    'comparison_hits',
    'domain_key',
    'domain_name',
    'expected',
    'expected_discoveries',
    'expected_discovery',
    'falsifier',
    'falsifiers',
    'hidden_rule',
    'label',
    'labels',
    'truth',
    'manifest',
    'missing_comparison_tags',
}


DOMAIN_TRANSFER_TARGETS: dict[str, tuple[str, ...]] = {
    'arithmetic_quantity': ('algebra_equations',),
    'algebra_equations': ('geometry_space',),
    'geometry_space': ('calculus_change',),
    'calculus_change': ('dynamics_systems',),
    'probability_uncertainty': ('information_computation',),
    'logic_proof': ('algebra_equations',),
    'discrete_structures': ('algebra_equations',),
    'symmetry_invariance': ('geometry_space',),
    'optimization_extrema': ('calculus_change',),
    'dynamics_systems': ('probability_uncertainty',),
    'information_computation': ('logic_proof',),
    'higher_dimensions': ('dynamics_systems',),
}


PRIMITIVE_CONTRASTS: dict[str, dict[str, Any]] = {
    'arithmetic_quantity': {
        'contrast_kind': 'cardinality_balance',
        'public_primitives': ['count', 'group', 'order', 'equality'],
        'observed_examples': [
            {'view': 'grouped', 'tokens': [['a0', 'a1'], ['b0']]},
            {'view': 'regrouped', 'tokens': [['c0'], ['c1', 'c2']]},
        ],
        'heldout_probe': {'view': 'permuted', 'ask': 'same extent after relabeling'},
        'control_question': 'Which changes alter extent, and which only rename it?',
        'hidden_rule': 'finite extent survives relabeling and regrouping',
        'expected_discoveries': (
            'count invariance',
            'conservation of total under regrouping',
        ),
        'falsifier': 'regrouping or relabeling changes the inferred total',
    },
    'algebra_equations': {
        'contrast_kind': 'reversible_substitution',
        'public_primitives': ['variable', 'inverse', 'composition', 'equality'],
        'observed_examples': [
            {'slot': 2, 'route': ['double', 'shift'], 'reading': 7},
            {'slot': 3, 'route': ['double', 'shift'], 'reading': 9},
        ],
        'heldout_probe': {'reading': 11, 'ask': 'recover slot without trying every value'},
        'control_question': 'Which operation sequence can be undone?',
        'hidden_rule': 'inverse operations recover a hidden slot from a balanced relation',
        'expected_discoveries': ('symbolic substitution', 'equation balance'),
        'falsifier': 'inverse replay returns a slot that fails direct substitution',
    },
    'geometry_space': {
        'contrast_kind': 'coordinate_measurement',
        'public_primitives': ['point', 'distance', 'angle', 'frame'],
        'observed_examples': [
            {'frame': 'a', 'marks': [(0, 0), (3, 4)], 'reading': 5.0},
            {'frame': 'shifted', 'marks': [(2, -1), (5, 3)], 'reading': 5.0},
        ],
        'heldout_probe': {'transform': 'rotate_or_shift', 'ask': 'which readings survive'},
        'control_question': 'What stays fixed when coordinates change?',
        'hidden_rule': 'metric readings can be frame-independent',
        'expected_discoveries': ('metric distance', 'coordinate transform'),
        'falsifier': 'a shared frame change alters the claimed metric',
    },
    'calculus_change': {
        'contrast_kind': 'finite_difference_accumulation',
        'public_primitives': ['rate', 'difference', 'accumulation', 'limit'],
        'observed_examples': [
            {'t': 0, 'value': 1.0},
            {'t': 1, 'value': 1.5},
            {'t': 2, 'value': 2.2},
        ],
        'heldout_probe': {'refine_between': [1, 2], 'ask': 'sum local changes'},
        'control_question': 'When does a local step predict total change?',
        'hidden_rule': 'neighbor differences accumulate into endpoint change',
        'expected_discoveries': (
            'derivative-like rate',
            'integral-like accumulation',
        ),
        'falsifier': 'refining the interval changes the accumulated endpoint',
    },
    'probability_uncertainty': {
        'contrast_kind': 'sample_noise_split',
        'public_primitives': ['sample', 'frequency', 'condition', 'expectation'],
        'observed_examples': [
            {'window': 0, 'draws': ['a', 'b', 'b', 'c']},
            {'window': 1, 'draws': ['b', 'a', 'b', 'c']},
        ],
        'heldout_probe': {'condition': 'flagged', 'ask': 'does the candidate set narrow'},
        'control_question': 'Which variation is stable signal rather than sample noise?',
        'hidden_rule': 'conditioning and repeated samples separate signal from noise',
        'expected_discoveries': ('frequency convergence', 'conditional split'),
        'falsifier': 'conditioning fails to change the best prediction',
    },
    'logic_proof': {
        'contrast_kind': 'counterexample_search',
        'public_primitives': ['predicate', 'implication', 'counterexample', 'proof'],
        'observed_examples': [
            {'case': 2, 'p': True, 'q': True},
            {'case': 3, 'p': False, 'q': False},
        ],
        'heldout_probe': {'ask': 'smallest case that breaks the implication'},
        'control_question': 'What would make this rule false?',
        'hidden_rule': 'proof pressure requires domains and counterexamples',
        'expected_discoveries': ('falsification', 'domain restriction'),
        'falsifier': 'the claim cannot name a breaking case',
    },
    'discrete_structures': {
        'contrast_kind': 'path_composition',
        'public_primitives': ['set', 'relation', 'path', 'state'],
        'observed_examples': [
            {'links': [('a', 'b'), ('b', 'c')], 'path': ['a', 'b', 'c']},
            {'links': [('c', 'd')], 'path': ['a', 'b', 'c', 'd']},
        ],
        'heldout_probe': {'ask': 'compose links or find missing relation'},
        'control_question': 'Which local relations can be chained?',
        'hidden_rule': 'local relations compose into finite paths or transitions',
        'expected_discoveries': ('path composition', 'connectivity'),
        'falsifier': 'two valid local links cannot be composed',
    },
    'symmetry_invariance': {
        'contrast_kind': 'transform_invariance',
        'public_primitives': ['transform', 'orbit', 'invariant', 'equivalence'],
        'observed_examples': [
            {'transform': 'shift', 'reading_changed': False},
            {'transform': 'reflect', 'reading_changed': False},
        ],
        'heldout_probe': {'transform': 'compose_two', 'ask': 'same invariant'},
        'control_question': 'Which quantities survive allowed transforms?',
        'hidden_rule': 'some readings are coordinate-free under transformations',
        'expected_discoveries': ('invariant quantity', 'coordinate-free law'),
        'falsifier': 'the invariant only works in the first view',
    },
    'optimization_extrema': {
        'contrast_kind': 'objective_comparison',
        'public_primitives': ['objective', 'constraint', 'minimum', 'gradient'],
        'observed_examples': [
            {'choice': -1, 'cost': 4.0},
            {'choice': 0, 'cost': 1.0},
            {'choice': 1, 'cost': 2.0},
        ],
        'heldout_probe': {'direction': 'toward_lower_cost', 'ask': 'does cost improve'},
        'control_question': 'When does a local move improve the objective?',
        'hidden_rule': 'nearby comparisons reveal extrema and active constraints',
        'expected_discoveries': ('least-error fit', 'tradeoff curve'),
        'falsifier': 'following the predicted improving direction raises cost',
    },
    'dynamics_systems': {
        'contrast_kind': 'transition_rollout',
        'public_primitives': ['state', 'transition', 'field', 'conservation'],
        'observed_examples': [
            {'step': 0, 'state': {'x': 0.0, 'v': 1.0}},
            {'step': 1, 'state': {'x': 1.2, 'v': 1.2}},
        ],
        'heldout_probe': {'horizon': 4, 'ask': 'roll out same transition'},
        'control_question': 'Does a one-step residual become a reusable update?',
        'hidden_rule': 'state transitions compose across longer horizons',
        'expected_discoveries': ('state update law', 'residual field'),
        'falsifier': 'a one-step update fails a longer trajectory',
    },
    'information_computation': {
        'contrast_kind': 'compression_holdout',
        'public_primitives': ['encoding', 'description', 'algorithm', 'memory'],
        'observed_examples': [
            {'method': 'literal', 'units': 8},
            {'method': 'repeat_rule', 'units': 4},
        ],
        'heldout_probe': {'length': 'longer', 'ask': 'decode and replay'},
        'control_question': 'Which shorter description still predicts?',
        'hidden_rule': 'compression only counts when it reconstructs holdouts',
        'expected_discoveries': ('compression preference', 'algorithmic recurrence'),
        'falsifier': 'the shorter description cannot replay the held-out sequence',
    },
    'higher_dimensions': {
        'contrast_kind': 'projection_residual',
        'public_primitives': ['dimension', 'projection', 'basis', 'latent coordinate'],
        'observed_examples': [
            {'view': 'xy', 'residual': 0.4},
            {'view': 'xz', 'residual': 0.0},
        ],
        'heldout_probe': {'view': 'new_projection', 'ask': 'does latent coordinate help'},
        'control_question': 'When should the system invent another axis?',
        'hidden_rule': 'latent coordinates can explain projection residuals',
        'expected_discoveries': ('latent axis', 'projection invariance'),
        'falsifier': 'the latent coordinate helps one projection but fails another',
    },
}


@dataclass(frozen=True)
class MathDomainSample:
    """One observation sample plus benchmark-only interpretation."""

    sample_id: str
    observation_kind: str
    public: dict[str, Any] = field(default_factory=dict)
    hidden_rule: str = ''
    expected_discoveries: tuple[str, ...] = ()
    falsifier: str = ''
    transfer_targets: tuple[str, ...] = ()

    def observation(self) -> dict[str, Any]:
        """Return the learner-facing sample with benchmark truth removed."""
        return {
            'sample_id': self.sample_id,
            'observation_kind': self.observation_kind,
            **_json_copy(self.public),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the benchmark-side manifest view."""
        return {
            'sample_id': self.sample_id,
            'observation_kind': self.observation_kind,
            'public': _json_copy(self.public),
            'hidden_rule': self.hidden_rule,
            'expected_discoveries': list(self.expected_discoveries),
            'falsifier': self.falsifier,
            'transfer_targets': list(self.transfer_targets),
        }


@dataclass(frozen=True)
class MathDomainWorldManifest:
    """Benchmark-side truth for a generated math-domain world."""

    domain_key: str
    domain_name: str
    seed: int
    variant: int
    samples: tuple[MathDomainSample, ...]

    @property
    def expected_discoveries(self) -> tuple[str, ...]:
        return tuple(sorted({
            discovery
            for sample in self.samples
            for discovery in sample.expected_discoveries
        }))

    @property
    def falsifiers(self) -> tuple[str, ...]:
        return tuple(
            sample.falsifier
            for sample in self.samples
            if sample.falsifier
        )

    @property
    def transfer_targets(self) -> tuple[str, ...]:
        return tuple(sorted({
            target
            for sample in self.samples
            for target in sample.transfer_targets
        }))

    def observations(self) -> list[dict[str, Any]]:
        return [sample.observation() for sample in self.samples]

    def to_dict(self) -> dict[str, Any]:
        return {
            'domain_key': self.domain_key,
            'domain_name': self.domain_name,
            'seed': self.seed,
            'variant': self.variant,
            'sample_count': len(self.samples),
            'samples': [sample.to_dict() for sample in self.samples],
            'expected_discoveries': list(self.expected_discoveries),
            'falsifiers': list(self.falsifiers),
            'transfer_targets': list(self.transfer_targets),
            'observation_schema': self.observation_schema(),
        }

    def observation_schema(self) -> dict[str, list[str]]:
        keys_by_kind: dict[str, set[str]] = {}
        for observation in self.observations():
            kind = str(observation.get('observation_kind', 'unknown'))
            keys_by_kind.setdefault(kind, set()).update(observation.keys())
        return {
            kind: sorted(keys)
            for kind, keys in sorted(keys_by_kind.items())
        }


def generate_math_domain_world_manifest(
    domain_key: str,
    seed: int = 0,
    variant: int = 0,
) -> MathDomainWorldManifest:
    """Build a deterministic benchmark manifest for one math domain."""
    try:
        name, builder = DOMAIN_WORLD_GENERATORS[domain_key]
    except KeyError as exc:
        raise ValueError(f"Unknown math domain: {domain_key}") from exc
    rng = random.Random((seed + 1) * 7919 + (variant + 1) * 104729 + _domain_offset(domain_key))
    samples = tuple(builder(rng, seed, variant))
    return MathDomainWorldManifest(
        domain_key=domain_key,
        domain_name=name,
        seed=seed,
        variant=variant,
        samples=samples,
    )


def generate_all_math_domain_world_manifests(
    seed: int = 0,
    variant: int = 0,
) -> list[MathDomainWorldManifest]:
    return [
        generate_math_domain_world_manifest(domain_key, seed=seed, variant=variant)
        for domain_key in DOMAIN_WORLD_GENERATORS
    ]


def math_domain_manifest_from_observation(observation: Any) -> bool:
    """Return True when an observation leaks benchmark-only manifest fields."""
    if isinstance(observation, dict):
        for key, value in observation.items():
            if str(key) in FORBIDDEN_OBSERVATION_KEYS:
                return True
            if math_domain_manifest_from_observation(value):
                return True
    elif isinstance(observation, (list, tuple)):
        return any(math_domain_manifest_from_observation(item) for item in observation)
    return False


def _arithmetic_quantity_samples(
    rng: random.Random,
    seed: int,
    variant: int,
) -> list[MathDomainSample]:
    left = rng.randint(2, 5)
    right = rng.randint(1, 4)
    total = left + right
    shuffled = [f"u{i}" for i in range(total)]
    rng.shuffle(shuffled)
    return [
        _sample(
            'arithmetic_quantity',
            0,
            'collection_event',
            {
                'before': {'group_a': _tokens('a', left), 'group_b': _tokens('b', right)},
                'event': {'move': 'join', 'labels_scrambled': False},
                'after': {'group_c': _tokens('c', total)},
                'heldout_view': {'group_c': shuffled},
            },
            'combined extent is stable under regrouping and relabeling',
            ('count invariance', 'conservation of total under regrouping'),
            'the total changes when token names are permuted',
            DOMAIN_TRANSFER_TARGETS['arithmetic_quantity'],
        ),
        _sample(
            'arithmetic_quantity',
            1,
            'collection_event',
            {
                'before': {'group': _tokens('q', total)},
                'event': {'move': 'remove_one', 'removed': 'q0'},
                'after': {'group': _tokens('q', total - 1)},
                'contrast': {'event': 'add_one', 'result_size_hint': total + 1},
            },
            'one local appearance or disappearance shifts extent by one',
            ('successor arithmetic',),
            'one-item events produce jumps larger than one',
            DOMAIN_TRANSFER_TARGETS['arithmetic_quantity'],
        ),
        _primitive_contrast_sample('arithmetic_quantity', 2, rng),
    ]


def _algebra_equation_samples(
    rng: random.Random,
    seed: int,
    variant: int,
) -> list[MathDomainSample]:
    hidden = rng.randint(2, 9)
    scale = rng.randint(2, 5)
    offset = rng.randint(1, 7)
    result = scale * hidden + offset
    probe = hidden + rng.choice([1, 2])
    return [
        _sample(
            'algebra_equations',
            0,
            'balanced_machine',
            {
                'machine': [
                    {'step': 'input_slot', 'value': '?'},
                    {'step': 'stretch', 'amount': scale},
                    {'step': 'shift', 'amount': offset},
                ],
                'seen_pairs': [
                    {'input': hidden - 1, 'output': scale * (hidden - 1) + offset},
                    {'input': hidden, 'output': result},
                    {'input': hidden + 1, 'output': scale * (hidden + 1) + offset},
                ],
                'query': {'output': result, 'missing_input': '?'},
            },
            'a reversible chain preserves balance between slot and output',
            ('symbolic substitution', 'equation balance'),
            'inverse steps recover a different slot value than direct testing',
            DOMAIN_TRANSFER_TARGETS['algebra_equations'],
        ),
        _sample(
            'algebra_equations',
            1,
            'rewrite_contrast',
            {
                'forms': [
                    {'form_id': 'r0', 'steps': ['stretch', 'shift']},
                    {'form_id': 'r1', 'steps': ['shift_repeated', 'stretch_remainder']},
                ],
                'checks': [
                    {'slot': hidden, 'left': result, 'right': result},
                    {'slot': probe, 'left': scale * probe + offset, 'right': scale * probe + offset},
                ],
                'holdout_slot': probe + 1,
            },
            'different operation descriptions can define one reusable relation',
            ('factor/rewrite equivalence', 'symbolic substitution'),
            'two descriptions agree on examples but split on the held-out slot',
            DOMAIN_TRANSFER_TARGETS['algebra_equations'],
        ),
        _primitive_contrast_sample('algebra_equations', 2, rng),
    ]


def _geometry_space_samples(
    rng: random.Random,
    seed: int,
    variant: int,
) -> list[MathDomainSample]:
    point_a = (_round(rng.uniform(-3.0, 1.0)), _round(rng.uniform(-2.0, 2.0)))
    point_b = (_round(point_a[0] + rng.uniform(2.0, 5.0)), _round(point_a[1] + rng.uniform(1.0, 4.0)))
    shift = (_round(rng.uniform(1.0, 3.0)), _round(rng.uniform(-2.0, 2.0)))
    center = (_round((point_a[0] + point_b[0]) / 2), _round((point_a[1] + point_b[1]) / 2))
    return [
        _sample(
            'geometry_space',
            0,
            'frame_change',
            {
                'view_a': {'p': point_a, 'q': point_b},
                'view_b': {
                    'p': (_round(point_a[0] + shift[0]), _round(point_a[1] + shift[1])),
                    'q': (_round(point_b[0] + shift[0]), _round(point_b[1] + shift[1])),
                },
                'paired_readings': [
                    {'view': 'a', 'separation_reading': _separation(point_a, point_b)},
                    {'view': 'b', 'separation_reading': _separation(point_a, point_b)},
                ],
            },
            'separation remains stable under a shared frame shift',
            ('metric distance', 'coordinate transform'),
            'a shared frame shift changes the measured separation',
            DOMAIN_TRANSFER_TARGETS['geometry_space'],
        ),
        _sample(
            'geometry_space',
            1,
            'boundary_probe',
            {
                'marks': {'p': point_a, 'q': point_b, 'mid': center},
                'membership_checks': [
                    {'mark': 'p', 'side': 'low'},
                    {'mark': 'mid', 'side': 'boundary'},
                    {'mark': 'q', 'side': 'high'},
                ],
                'alternate_view': {'origin_shift': shift},
            },
            'local marks can define a boundary that survives re-description',
            ('local/global shape distinction', 'coordinate transform'),
            'a boundary claim holds only in the original frame',
            DOMAIN_TRANSFER_TARGETS['geometry_space'],
        ),
        _primitive_contrast_sample('geometry_space', 2, rng),
    ]


def _calculus_change_samples(
    rng: random.Random,
    seed: int,
    variant: int,
) -> list[MathDomainSample]:
    start = rng.uniform(-2.0, 2.0)
    slope = rng.uniform(0.4, 1.4)
    curve = rng.uniform(0.05, 0.18)
    series = [
        {'t': t, 'reading': _round(start + slope * t + curve * t * t)}
        for t in range(6)
    ]
    increments = [
        _round(series[i + 1]['reading'] - series[i]['reading'])
        for i in range(len(series) - 1)
    ]
    return [
        _sample(
            'calculus_change',
            0,
            'refined_series',
            {
                'coarse_trace': series[::2],
                'fine_trace': series,
                'neighbor_changes': increments,
                'endpoint_change': _round(series[-1]['reading'] - series[0]['reading']),
            },
            'local changes accumulate into endpoint change',
            ('derivative-like rate', 'integral-like accumulation'),
            'neighbor changes sum to a different endpoint change',
            DOMAIN_TRANSFER_TARGETS['calculus_change'],
        ),
        _sample(
            'calculus_change',
            1,
            'local_prediction',
            {
                'anchor': series[2],
                'nearby': [series[1], series[3]],
                'far': series[5],
                'local_step_reading': _round((increments[1] + increments[2]) / 2),
            },
            'small neighborhoods give better local prediction than distant reuse',
            ('local linear approximation', 'derivative-like rate'),
            'a local estimate predicts distant samples equally well',
            DOMAIN_TRANSFER_TARGETS['calculus_change'],
        ),
        _primitive_contrast_sample('calculus_change', 2, rng),
    ]


def _probability_uncertainty_samples(
    rng: random.Random,
    seed: int,
    variant: int,
) -> list[MathDomainSample]:
    weights = [rng.randint(1, 4), rng.randint(2, 6), rng.randint(1, 3)]
    bag = ['a'] * weights[0] + ['b'] * weights[1] + ['c'] * weights[2]
    trials = [rng.choice(bag) for _ in range(36)]
    tagged = [
        {'draw': i, 'symbol': symbol, 'flag': symbol in {'b', 'c'}}
        for i, symbol in enumerate(trials)
    ]
    return [
        _sample(
            'probability_uncertainty',
            0,
            'repeated_draws',
            {
                'trial_log': tagged,
                'windows': [
                    {'start': 0, 'end': 12, 'symbols': trials[:12]},
                    {'start': 12, 'end': 24, 'symbols': trials[12:24]},
                    {'start': 24, 'end': 36, 'symbols': trials[24:]},
                ],
            },
            'longer repeated samples stabilize relative frequencies',
            ('frequency convergence', 'conditional split'),
            'later windows systematically reverse the same source mix',
            DOMAIN_TRANSFER_TARGETS['probability_uncertainty'],
        ),
        _sample(
            'probability_uncertainty',
            1,
            'evidence_split',
            {
                'cases': tagged,
                'observed_flag': True,
                'candidate_symbols_after_flag': sorted({item['symbol'] for item in tagged if item['flag']}),
            },
            'conditioning on evidence narrows the relevant sample space',
            ('conditional split', 'expected error minimization'),
            'conditioning does not change the best next-symbol prediction',
            DOMAIN_TRANSFER_TARGETS['probability_uncertainty'],
        ),
        _primitive_contrast_sample('probability_uncertainty', 2, rng),
    ]


def _logic_proof_samples(
    rng: random.Random,
    seed: int,
    variant: int,
) -> list[MathDomainSample]:
    values = list(range(1, 10))
    rng.shuffle(values)
    cases = [
        {
            'case': value,
            'marks': {
                'p': value % 2 == 0,
                'q': (value + value) % 2 == 0,
                'r': value > 5,
            },
        }
        for value in values[:7]
    ]
    return [
        _sample(
            'logic_proof',
            0,
            'claim_checks',
            {
                'cases': cases,
                'candidate_claims': [
                    {'claim_id': 'c0', 'when': 'p', 'then': 'q'},
                    {'claim_id': 'c1', 'when': 'r', 'then': 'p'},
                ],
                'search_instruction': {'look_for': 'smallest_breaking_case'},
            },
            'a rule needs supporting cases and a counterexample search',
            ('falsification', 'domain restriction'),
            'a single supporting example is enough to promote a universal rule',
            DOMAIN_TRANSFER_TARGETS['logic_proof'],
        ),
        _sample(
            'logic_proof',
            1,
            'partition_checks',
            {
                'cases': cases,
                'bins': [
                    {'bin': 'p_true', 'members': [item['case'] for item in cases if item['marks']['p']]},
                    {'bin': 'p_false', 'members': [item['case'] for item in cases if not item['marks']['p']]},
                ],
            },
            'predicates define domains where a claim can be tested',
            ('domain restriction', 'proof by repeated invariant check'),
            'a domain split overlaps or misses observed cases',
            DOMAIN_TRANSFER_TARGETS['logic_proof'],
        ),
        _primitive_contrast_sample('logic_proof', 2, rng),
    ]


def _discrete_structure_samples(
    rng: random.Random,
    seed: int,
    variant: int,
) -> list[MathDomainSample]:
    nodes = ['n0', 'n1', 'n2', 'n3', 'n4']
    edges = [('n0', 'n1'), ('n1', 'n2'), ('n2', 'n4'), ('n0', 'n3'), ('n3', 'n4')]
    if rng.random() > 0.5:
        edges.append(('n1', 'n3'))
    return [
        _sample(
            'discrete_structures',
            0,
            'link_walks',
            {
                'items': nodes,
                'links': [{'from': a, 'to': b} for a, b in edges],
                'walks': [
                    ['n0', 'n1', 'n2', 'n4'],
                    ['n0', 'n3', 'n4'],
                ],
                'blocked_walk': ['n4', 'n2', 'n1'],
            },
            'local links compose into reachable paths',
            ('path composition', 'connectivity'),
            'two valid local links cannot be chained into a valid path',
            DOMAIN_TRANSFER_TARGETS['discrete_structures'],
        ),
        _sample(
            'discrete_structures',
            1,
            'state_steps',
            {
                'states': ['s0', 's1', 's2'],
                'steps': [
                    {'from': 's0', 'move': 'a', 'to': 's1'},
                    {'from': 's1', 'move': 'b', 'to': 's2'},
                    {'from': 's0', 'move': 'a_then_b', 'to': 's2'},
                ],
            },
            'repeated finite transitions form reusable composition rules',
            ('finite transition algebra', 'path composition'),
            'a composed move reaches a different state than its pieces',
            DOMAIN_TRANSFER_TARGETS['discrete_structures'],
        ),
        _primitive_contrast_sample('discrete_structures', 2, rng),
    ]


def _symmetry_invariance_samples(
    rng: random.Random,
    seed: int,
    variant: int,
) -> list[MathDomainSample]:
    shape = [(0, 0), (2, 0), (1, 2)]
    shifted = [(x + 3, y - 1) for x, y in shape]
    reflected = [(-x, y) for x, y in shape]
    return [
        _sample(
            'symmetry_invariance',
            0,
            'transform_family',
            {
                'views': [
                    {'view': 'base', 'marks': shape},
                    {'view': 'shifted', 'marks': shifted},
                    {'view': 'reflected', 'marks': reflected},
                ],
                'readings': [
                    {'view': 'base', 'edge_pattern': [2.0, _round(math.sqrt(5)), _round(math.sqrt(5))]},
                    {'view': 'shifted', 'edge_pattern': [2.0, _round(math.sqrt(5)), _round(math.sqrt(5))]},
                    {'view': 'reflected', 'edge_pattern': [2.0, _round(math.sqrt(5)), _round(math.sqrt(5))]},
                ],
            },
            'allowed transforms preserve some readings while changing coordinates',
            ('invariant quantity', 'coordinate-free law'),
            'the preserved reading succeeds only before transform',
            DOMAIN_TRANSFER_TARGETS['symmetry_invariance'],
        ),
        _sample(
            'symmetry_invariance',
            1,
            'operation_order',
            {
                'start_marks': shape,
                'routes': [
                    {'route': ['shift', 'reflect'], 'marks': [(-x - 3, y - 1) for x, y in shape]},
                    {'route': ['reflect', 'shift'], 'marks': [(-x + 3, y - 1) for x, y in shape]},
                ],
            },
            'transform composition may preserve structure while order still matters',
            ('group-like composition', 'coordinate-free law'),
            'order-sensitive transforms are treated as interchangeable',
            DOMAIN_TRANSFER_TARGETS['symmetry_invariance'],
        ),
        _primitive_contrast_sample('symmetry_invariance', 2, rng),
    ]


def _optimization_extrema_samples(
    rng: random.Random,
    seed: int,
    variant: int,
) -> list[MathDomainSample]:
    optimum = rng.uniform(-1.5, 2.5)
    candidates = [
        _round(optimum + delta)
        for delta in (-3.0, -1.0, 0.0, 1.0, 2.5)
    ]
    scored = [
        {'choice': x, 'cost_reading': _round((x - optimum) ** 2 + 0.4)}
        for x in candidates
    ]
    return [
        _sample(
            'optimization_extrema',
            0,
            'choice_scores',
            {
                'candidates': scored,
                'neighbor_tests': [
                    {'from': scored[1]['choice'], 'direction': 'up', 'change': _round(scored[2]['cost_reading'] - scored[1]['cost_reading'])},
                    {'from': scored[3]['choice'], 'direction': 'down', 'change': _round(scored[2]['cost_reading'] - scored[3]['cost_reading'])},
                ],
            },
            'best choices sit where nearby changes stop improving',
            ('least-error fit', 'tradeoff curve'),
            'moving in the predicted improving direction raises held-out cost',
            DOMAIN_TRANSFER_TARGETS['optimization_extrema'],
        ),
        _sample(
            'optimization_extrema',
            1,
            'constraint_scores',
            {
                'allowed_interval': [_round(optimum - 1.0), _round(optimum + 0.5)],
                'candidates': scored,
                'boundary_checks': [
                    {'choice': _round(optimum - 1.0), 'allowed': True},
                    {'choice': _round(optimum + 1.0), 'allowed': False},
                ],
            },
            'constraints can move the best available choice to a boundary',
            ('constraint boundary', 'least-error fit'),
            'an unconstrained best remains valid when it violates the boundary',
            DOMAIN_TRANSFER_TARGETS['optimization_extrema'],
        ),
        _primitive_contrast_sample('optimization_extrema', 2, rng),
    ]


def _dynamics_system_samples(
    rng: random.Random,
    seed: int,
    variant: int,
) -> list[MathDomainSample]:
    state = {'x': _round(rng.uniform(-2.0, 2.0)), 'v': _round(rng.uniform(0.2, 1.0))}
    force = _round(rng.uniform(-0.2, 0.4))
    trace = []
    x = state['x']
    v = state['v']
    for step in range(6):
        trace.append({'step': step, 'x': _round(x), 'v': _round(v)})
        v += force
        x += v
    return [
        _sample(
            'dynamics_systems',
            0,
            'state_trace',
            {
                'trace': trace,
                'baseline': 'repeat_last_step',
                'residual_readings': [
                    _round(trace[i + 1]['v'] - trace[i]['v'])
                    for i in range(len(trace) - 1)
                ],
            },
            'state updates reuse a transition rule across time',
            ('state update law', 'residual field'),
            'a one-step rule works but fails the same trace at longer horizons',
            DOMAIN_TRANSFER_TARGETS['dynamics_systems'],
        ),
        _sample(
            'dynamics_systems',
            1,
            'cycle_trace',
            {
                'phase_marks': [
                    {'step': step, 'reading': _round(math.sin(step * math.pi / 2))}
                    for step in range(8)
                ],
                'repeat_gap': 4,
            },
            'repeated phases can predict later state without seeing every step',
            ('phase or conservation law', 'state update law'),
            'the phase repeat fails on a held-out cycle',
            DOMAIN_TRANSFER_TARGETS['dynamics_systems'],
        ),
        _primitive_contrast_sample('dynamics_systems', 2, rng),
    ]


def _information_computation_samples(
    rng: random.Random,
    seed: int,
    variant: int,
) -> list[MathDomainSample]:
    message = ''.join(rng.choice(['a', 'a', 'b', 'c']) for _ in range(18))
    runs = _run_lengths(message)
    return [
        _sample(
            'information_computation',
            0,
            'message_codes',
            {
                'message': message,
                'descriptions': [
                    {'method': 'raw', 'units': len(message)},
                    {'method': 'runs', 'units': len(runs) * 2, 'code': runs},
                ],
                'holdout_prefix': message[:9],
            },
            'shorter descriptions that predict holdouts are preferred',
            ('compression preference', 'algorithmic recurrence'),
            'a shorter description fails to reconstruct held-out observations',
            DOMAIN_TRANSFER_TARGETS['information_computation'],
        ),
        _sample(
            'information_computation',
            1,
            'hidden_state_machine',
            {
                'observations': [
                    {'input': 'tick', 'visible': i % 3, 'output': 'pulse' if i % 3 == 0 else 'quiet'}
                    for i in range(9)
                ],
                'candidate_memory_sizes': [1, 2, 3],
            },
            'small internal state can explain recurring visible outputs',
            ('hidden-state inference', 'algorithmic recurrence'),
            'the inferred state predicts seen cycles but not a longer one',
            DOMAIN_TRANSFER_TARGETS['information_computation'],
        ),
        _primitive_contrast_sample('information_computation', 2, rng),
    ]


def _higher_dimension_samples(
    rng: random.Random,
    seed: int,
    variant: int,
) -> list[MathDomainSample]:
    points = [
        (_round(rng.uniform(-2.0, 2.0)), _round(rng.uniform(-2.0, 2.0)), _round(rng.uniform(-1.0, 1.0)))
        for _ in range(4)
    ]
    visible = [(x, y) for x, y, _z in points]
    lifted_readings = [
        {'visible': (x, y), 'extra_reading': z, 'combined_reading': _round(x + y + z)}
        for x, y, z in points
    ]
    return [
        _sample(
            'higher_dimensions',
            0,
            'projection_pairs',
            {
                'visible_marks': visible,
                'paired_readings': lifted_readings,
                'projection_views': [
                    {'view': 'xy', 'marks': visible},
                    {'view': 'xz', 'marks': [(x, z) for x, _y, z in points]},
                ],
            },
            'an extra coordinate can explain residuals invisible in a projection',
            ('latent axis', 'projection invariance'),
            'the extra coordinate helps one projection but fails another same-structure view',
            DOMAIN_TRANSFER_TARGETS['higher_dimensions'],
        ),
        _sample(
            'higher_dimensions',
            1,
            'dimension_change',
            {
                'families': [
                    {'visible_axes': 2, 'sample': [list(mark) for mark in visible]},
                    {'visible_axes': 3, 'sample': [list(mark) for mark in points]},
                ],
                'same_rule_query': {'axes': 'variable', 'holdout_axes': 4},
            },
            'some laws should preserve form when the number of axes changes',
            ('dimension-independent law', 'projection invariance'),
            'a rule tied to one axis count is promoted as dimension-independent',
            DOMAIN_TRANSFER_TARGETS['higher_dimensions'],
        ),
        _primitive_contrast_sample('higher_dimensions', 2, rng),
    ]


DomainBuilder = Callable[[random.Random, int, int], list[MathDomainSample]]


DOMAIN_WORLD_GENERATORS: dict[str, tuple[str, DomainBuilder]] = {
    'arithmetic_quantity': ('arithmetic and quantity', _arithmetic_quantity_samples),
    'algebra_equations': ('algebra and symbolic equations', _algebra_equation_samples),
    'geometry_space': ('geometry and measurement', _geometry_space_samples),
    'calculus_change': ('calculus and change', _calculus_change_samples),
    'probability_uncertainty': ('probability and uncertainty', _probability_uncertainty_samples),
    'logic_proof': ('logic and proof', _logic_proof_samples),
    'discrete_structures': ('discrete structures and graphs', _discrete_structure_samples),
    'symmetry_invariance': ('symmetry and invariance', _symmetry_invariance_samples),
    'optimization_extrema': ('optimization and extrema', _optimization_extrema_samples),
    'dynamics_systems': ('dynamics and systems', _dynamics_system_samples),
    'information_computation': ('information and computation', _information_computation_samples),
    'higher_dimensions': ('higher-dimensional worlds', _higher_dimension_samples),
}


def _sample(
    domain_key: str,
    index: int,
    observation_kind: str,
    public: dict[str, Any],
    hidden_rule: str,
    expected_discoveries: tuple[str, ...],
    falsifier: str,
    transfer_targets: tuple[str, ...],
) -> MathDomainSample:
    return MathDomainSample(
        sample_id=f"s{index:02d}",
        observation_kind=observation_kind,
        public=public,
        hidden_rule=hidden_rule,
        expected_discoveries=expected_discoveries,
        falsifier=falsifier,
        transfer_targets=transfer_targets,
    )


def _primitive_contrast_sample(
    domain_key: str,
    index: int,
    rng: random.Random,
) -> MathDomainSample:
    config = PRIMITIVE_CONTRASTS[domain_key]
    public = {
        'contrast_kind': config['contrast_kind'],
        'public_primitives': list(config['public_primitives']),
        'observed_examples': _json_copy(config['observed_examples']),
        'heldout_probe': _json_copy(config['heldout_probe']),
        'control_question': config['control_question'],
        'contrast_id': rng.randint(1000, 9999),
    }
    return _sample(
        domain_key,
        index,
        'primitive_contrast',
        public,
        config['hidden_rule'],
        tuple(config['expected_discoveries']),
        config['falsifier'],
        DOMAIN_TRANSFER_TARGETS[domain_key],
    )


def _domain_offset(domain_key: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(domain_key))


def _tokens(prefix: str, count: int) -> list[str]:
    return [f"{prefix}{index}" for index in range(count)]


def _round(value: float, digits: int = 3) -> float:
    return round(float(value), digits)


def _separation(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    return _round(math.hypot(point_b[0] - point_a[0], point_b[1] - point_a[1]))


def _run_lengths(message: str) -> list[dict[str, Any]]:
    if not message:
        return []
    runs = []
    current = message[0]
    length = 1
    for char in message[1:]:
        if char == current:
            length += 1
        else:
            runs.append({'symbol': current, 'length': length})
            current = char
            length = 1
    runs.append({'symbol': current, 'length': length})
    return runs


def _json_copy(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_copy(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_copy(item) for item in value]
    if isinstance(value, list):
        return [_json_copy(item) for item in value]
    return value
