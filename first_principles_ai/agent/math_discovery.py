from __future__ import annotations

"""
Raw mathematical structure discovery.

This module is intentionally split into two layers:

1. The agent-facing layer records internal transition patterns from raw object
   observations. It does not store labels like "addition", "equality", or
   "number" in the discovered concepts.
2. The comparison layer maps those internal structures to human mathematical
   ideas after the fact, so we can test where the agent's invented structures
   land relative to real math.
"""

from dataclasses import dataclass, field

from agent.representation import ConceptType, KnowledgeBase, RuleStatus


@dataclass
class RawMathPattern:
    """One internally discovered transition pattern."""
    key: str
    kind: str
    description: str
    first_step: int
    evidence: int = 0
    last_step: int = 0
    properties: dict = field(default_factory=dict)
    concept_name: str | None = None
    rule_name: str | None = None

    def add_evidence(self, step: int, amount: int = 1):
        self.evidence += amount
        self.last_step = step

    def to_dict(self) -> dict:
        return {
            'key': self.key,
            'kind': self.kind,
            'description': self.description,
            'first_step': self.first_step,
            'last_step': self.last_step,
            'evidence': self.evidence,
            'properties': dict(self.properties),
            'concept_name': self.concept_name,
            'rule_name': self.rule_name,
        }


@dataclass
class HumanMathComparison:
    """Post-hoc mapping from an internal pattern to human math."""
    internal_key: str
    human_concept: str
    confidence: float
    evidence: int
    basis: str

    def to_dict(self) -> dict:
        return {
            'internal_key': self.internal_key,
            'human_concept': self.human_concept,
            'confidence': round(self.confidence, 3),
            'evidence': self.evidence,
            'basis': self.basis,
        }


class EmergentMathDiscovery:
    """
    Discovers math-like structure from raw transitions.

    The discovery layer watches:
    - whether object tokens persist across adjacent observations
    - whether object attributes remain stable through motion
    - whether the raw collection extent changes by discrete deltas
    - whether interventions reliably produce repeatable deltas
    - whether scalar channels induce stable rankings
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase | None = None,
        concept_evidence_threshold: int = 8,
        operation_evidence_threshold: int = 2,
        tolerance: float = 1e-6,
    ):
        self.knowledge_base = knowledge_base
        self.concept_evidence_threshold = concept_evidence_threshold
        self.operation_evidence_threshold = operation_evidence_threshold
        self.tolerance = tolerance
        self.patterns: dict[str, RawMathPattern] = {}
        self._action_deltas: dict[str, list[int]] = {}
        self._last_extent_transform: dict | None = None
        self._sequence_signatures_by_net: dict[int, set[tuple]] = {}
        self._last_velocity_signs: dict[int, dict[str, int]] = {}
        self._sign_switch_counts: dict[str, int] = {}
        self._action_outcomes: dict[tuple[str, int], int] = {}

    def observe_transition(self, before: dict, after: dict, action: dict | None, step: int):
        before_snapshot = self._snapshot(before)
        after_snapshot = self._snapshot(after)
        action_type = (action or {'type': 'wait'}).get('type', 'wait')

        self._observe_token_continuity(before_snapshot, after_snapshot, step)
        self._observe_stable_attributes(before_snapshot, after_snapshot, step)
        self._observe_collection_extent(before_snapshot, after_snapshot, action_type, step)
        self._observe_metric_and_scale_relations(after_snapshot, step)
        self._observe_channel_symmetry(after_snapshot, step)
        self._observe_channel_recurrence(after_snapshot, step)
        self._observe_rank_channels(after_snapshot, step)
        self._maybe_install_patterns(step)

    def discovered_patterns(self) -> list[RawMathPattern]:
        return sorted(
            self.patterns.values(),
            key=lambda pattern: (-pattern.evidence, pattern.key),
        )

    def compare_to_human_math(self) -> list[HumanMathComparison]:
        comparisons = []
        for pattern in self.discovered_patterns():
            mapping = self._human_mapping(pattern)
            if mapping is None:
                continue
            human_concept, basis = mapping
            comparisons.append(HumanMathComparison(
                internal_key=pattern.key,
                human_concept=human_concept,
                confidence=min(1.0, pattern.evidence / max(self.concept_evidence_threshold, 1)),
                evidence=pattern.evidence,
                basis=basis,
            ))
        return comparisons

    def summary(self) -> str:
        lines = [
            "Emergent math discovery:",
            f"  Internal patterns: {len(self.patterns)}",
        ]
        installed = [pattern for pattern in self.patterns.values() if pattern.concept_name]
        if installed:
            lines.append("  Agent-internal structures:")
            for pattern in sorted(installed, key=lambda item: item.concept_name or item.key):
                symbol = ''
                if self.knowledge_base and pattern.concept_name:
                    concept = self.knowledge_base.concepts.get(pattern.concept_name)
                    if concept and concept.notation:
                        symbol = f", symbol={concept.notation}"
                lines.append(
                    f"    {pattern.concept_name}: {pattern.key} "
                    f"(evidence={pattern.evidence}{symbol})"
                )

        comparisons = self.compare_to_human_math()
        if comparisons:
            lines.append("  Human-math comparison candidates:")
            for comparison in comparisons:
                lines.append(
                    f"    {comparison.internal_key} -> {comparison.human_concept} "
                    f"(confidence={comparison.confidence:.2f}, evidence={comparison.evidence})"
                )
        return "\n".join(lines)

    def _observe_token_continuity(self, before: dict, after: dict, step: int):
        shared_tokens = before['ids'] & after['ids']
        if shared_tokens:
            self._record(
                key='raw_pattern:token_continuity',
                kind='relation',
                description='Recurring token continuity across adjacent observations.',
                step=step,
                amount=len(shared_tokens),
                properties={'structural_role': 'persistence'},
            )

    def _observe_stable_attributes(self, before: dict, after: dict, step: int):
        shared_tokens = before['ids'] & after['ids']
        for channel in ('mass', 'radius'):
            stable = 0
            for token_id in shared_tokens:
                before_value = before['objects'][token_id][channel]
                after_value = after['objects'][token_id][channel]
                if abs(before_value - after_value) <= self.tolerance:
                    stable += 1
            if stable:
                self._record(
                    key=f'raw_pattern:stable_channel:{channel}',
                    kind='pattern',
                    description=f"Channel '{channel}' remains unchanged across token transitions.",
                    step=step,
                    amount=stable,
                    properties={'structural_role': 'invariant', 'channel': channel},
                )

    def _observe_collection_extent(
        self,
        before: dict,
        after: dict,
        action_type: str,
        step: int,
    ):
        delta = after['extent'] - before['extent']
        self._record(
            key='raw_pattern:collection_extent',
            kind='quantity_like',
            description='Raw collection extent is repeatedly observable across transitions.',
            step=step,
            properties={'structural_role': 'discrete_extent'},
        )
        if delta == 0:
            return

        self._record(
            key=f'raw_pattern:collection_extent_delta:{delta}',
            kind='transform',
            description=f"Raw collection extent repeatedly shifts by {delta}.",
            step=step,
            properties={'structural_role': 'discrete_delta', 'delta': delta},
        )

        transform_key = f'raw_operator:intervention:{action_type}:extent_delta:{delta}'
        self._action_deltas.setdefault(action_type, []).append(delta)
        self._record(
            key=transform_key,
            kind='operation',
            description=f"Intervention '{action_type}' repeatedly maps collection extent by {delta}.",
            step=step,
            properties={
                'structural_role': 'intervention_transform',
                'action_type': action_type,
                'delta': delta,
            },
        )
        self._observe_inverse_and_repeatable_transforms(action_type, delta, step)
        self._observe_adjacent_transform_sequence(action_type, delta, step)
        self._observe_conditional_transition(action_type, delta, step)
        self._last_extent_transform = {
            'action_type': action_type,
            'delta': delta,
            'step': step,
        }

    def _observe_conditional_transition(self, action_type: str, delta: int, step: int):
        outcome_key = (action_type, delta)
        self._action_outcomes[outcome_key] = self._action_outcomes.get(outcome_key, 0) + 1
        self._record(
            key=f'raw_relation:conditional_transition:{action_type}:extent_delta:{delta}',
            kind='relation',
            description=(
                f"Intervention signature '{action_type}' recurs with raw "
                f"extent shift {delta}."
            ),
            step=step,
            properties={
                'structural_role': 'conditional_transition',
                'action_type': action_type,
                'delta': delta,
                'occurrences': self._action_outcomes[outcome_key],
            },
        )
        if self._action_outcomes[outcome_key] >= 2:
            self._record(
                key=f'raw_relation:input_output_signature:{action_type}:extent_delta:{delta}',
                kind='relation',
                description=(
                    f"Input signature '{action_type}' repeatedly maps to "
                    f"raw extent shift {delta}."
                ),
                step=step,
                properties={
                    'structural_role': 'input_output_mapping',
                    'action_type': action_type,
                    'delta': delta,
                },
            )

    def _observe_inverse_and_repeatable_transforms(self, action_type: str, delta: int, step: int):
        same_delta_count = self._action_deltas[action_type].count(delta)
        if same_delta_count >= 2:
            self._record(
                key=f'raw_operator:repeatable:{action_type}:extent_delta:{delta}',
                kind='operation',
                description=f"Intervention '{action_type}' produces the same extent transform repeatedly.",
                step=step,
                properties={
                    'structural_role': 'repeatable_transform',
                    'action_type': action_type,
                    'delta': delta,
                    'repeat_count': same_delta_count,
                },
            )

        for other_action, deltas in self._action_deltas.items():
            if other_action == action_type:
                continue
            if -delta in deltas:
                ordered = sorted((action_type, other_action))
                self._record(
                    key=f'raw_operator:paired_transforms:{ordered[0]}:{ordered[1]}',
                    kind='operation',
                    description='Two intervention signatures produce opposite extent transforms.',
                    step=step,
                    properties={
                        'structural_role': 'opposed_transforms',
                        'actions': tuple(ordered),
                        'delta_magnitude': abs(delta),
                    },
                )

    def _observe_adjacent_transform_sequence(self, action_type: str, delta: int, step: int):
        previous = self._last_extent_transform
        if previous is None or previous['step'] != step - 1:
            return

        first_action = previous['action_type']
        first_delta = previous['delta']
        net_delta = first_delta + delta
        self._record(
            key=(
                f"raw_operator:paired_sequence:{first_action}:{action_type}:"
                f"extent_delta:{first_delta}_then_{delta}:net_{net_delta}"
            ),
            kind='operation',
            description=(
                "Two adjacent extent transforms jointly produce "
                f"net extent shift {net_delta}."
            ),
            step=step,
            properties={
                'structural_role': 'paired_step_transform',
                'actions': (first_action, action_type),
                'deltas': (first_delta, delta),
                'net_delta': net_delta,
            },
        )
        if net_delta == 0:
            self._record(
                key=(
                    f"raw_operator:returning_sequence:{first_action}:{action_type}:"
                    f"extent_delta:{first_delta}_then_{delta}"
                ),
                kind='operation',
                description=(
                    "Two adjacent extent transforms return raw collection "
                    "extent to its earlier level."
                ),
                step=step,
                properties={
                    'structural_role': 'returning_sequence',
                    'actions': (first_action, action_type),
                    'deltas': (first_delta, delta),
                    'net_delta': net_delta,
                },
            )
        self._observe_same_net_sequences(first_action, action_type, first_delta, delta, net_delta, step)

    def _observe_same_net_sequences(
        self,
        first_action: str,
        second_action: str,
        first_delta: int,
        second_delta: int,
        net_delta: int,
        step: int,
    ):
        signature = (first_action, second_action, first_delta, second_delta)
        reverse_signature = (second_action, first_action, second_delta, first_delta)
        signatures = self._sequence_signatures_by_net.setdefault(net_delta, set())
        had_reverse = reverse_signature in signatures and reverse_signature != signature
        signatures.add(signature)

        if len(signatures) >= 2:
            self._record(
                key=f"raw_operator:same_net_sequence_set:net_{net_delta}",
                kind='operation',
                description=(
                    "Different adjacent extent-transform sequences produce "
                    f"the same net extent shift {net_delta}."
                ),
                step=step,
                properties={
                    'structural_role': 'same_net_sequence_set',
                    'net_delta': net_delta,
                    'sequence_count': len(signatures),
                    'sequences': [self._format_sequence_signature(item) for item in sorted(signatures)],
                },
            )

        if had_reverse:
            ordered = sorted((signature, reverse_signature))
            self._record(
                key=(
                    f"raw_operator:swapped_sequence_same_net:"
                    f"{ordered[0][0]}:{ordered[0][1]}__{ordered[1][0]}:{ordered[1][1]}:"
                    f"net_{net_delta}"
                ),
                kind='operation',
                description=(
                    "Two adjacent extent-transform sequences with swapped "
                    f"order produce the same net extent shift {net_delta}."
                ),
                step=step,
                properties={
                    'structural_role': 'swapped_sequence_same_net',
                    'net_delta': net_delta,
                    'sequences': [self._format_sequence_signature(item) for item in ordered],
                },
            )

    def _observe_rank_channels(self, snapshot: dict, step: int):
        if snapshot['extent'] < 2:
            return
        for channel in ('position_x', 'position_y'):
            values = [
                obj[channel]
                for obj in snapshot['objects'].values()
            ]
            unique_values = self._unique_rounded(values)
            if len(unique_values) < 2:
                continue
            self._record(
                key=f'raw_relation:channel_rank:{channel}',
                kind='relation',
                description=f"Channel '{channel}' induces a repeatable ranking among tokens.",
                step=step,
                properties={'structural_role': 'ordering', 'channel': channel},
            )

    def _observe_metric_and_scale_relations(self, snapshot: dict, step: int):
        if snapshot['extent'] < 2:
            return

        objects = list(snapshot['objects'].values())
        separations = []
        for index, first in enumerate(objects):
            for second in objects[index + 1:]:
                dx = first['position_x'] - second['position_x']
                dy = first['position_y'] - second['position_y']
                separations.append((dx * dx + dy * dy) ** 0.5)

        positive = [value for value in separations if value > self.tolerance]
        if positive:
            self._record(
                key='raw_relation:pairwise_separation_magnitude',
                kind='relation',
                description='Token pairs expose repeatable separation magnitudes.',
                step=step,
                amount=min(len(positive), 4),
                properties={'structural_role': 'metric_separation'},
            )

        scale_values = []
        for obj in objects:
            radius = obj['radius']
            if abs(radius) > self.tolerance:
                scale_values.append(obj['mass'] / radius)
        if len(scale_values) >= 2:
            self._record(
                key='raw_relation:channel_scale:mass_per_radius',
                kind='relation',
                description='Two stable scalar channels expose repeatable scale quotients.',
                step=step,
                amount=min(len(scale_values), 4),
                properties={'structural_role': 'scale_quotient'},
            )

    def _observe_channel_symmetry(self, snapshot: dict, step: int):
        if snapshot['extent'] < 3:
            return

        for channel in ('position_x', 'position_y'):
            values = [obj[channel] for obj in snapshot['objects'].values()]
            center = sum(values) / len(values)
            below = any(value < center - self.tolerance for value in values)
            above = any(value > center + self.tolerance for value in values)
            if below and above:
                self._record(
                    key=f'raw_relation:balanced_channel:{channel}',
                    kind='relation',
                    description=f"Channel '{channel}' recurs on both sides of its aggregate level.",
                    step=step,
                    properties={'structural_role': 'balanced_symmetry', 'channel': channel},
                )

    def _observe_channel_recurrence(self, snapshot: dict, step: int):
        for token_id, obj in snapshot['objects'].items():
            previous = self._last_velocity_signs.setdefault(token_id, {})
            for channel in ('velocity_x', 'velocity_y'):
                sign = self._sign(obj[channel])
                if sign == 0:
                    continue
                prior = previous.get(channel)
                if prior is not None and prior != sign:
                    self._sign_switch_counts[channel] = self._sign_switch_counts.get(channel, 0) + 1
                    if self._sign_switch_counts[channel] >= 2:
                        self._record(
                            key=f'raw_pattern:alternating_channel:{channel}',
                            kind='pattern',
                            description=f"Channel '{channel}' revisits alternating direction states.",
                            step=step,
                            properties={
                                'structural_role': 'cyclic_recurrence',
                                'channel': channel,
                                'switch_count': self._sign_switch_counts[channel],
                            },
                        )
                previous[channel] = sign

    def _record(
        self,
        key: str,
        kind: str,
        description: str,
        step: int,
        amount: int = 1,
        properties: dict | None = None,
    ) -> RawMathPattern:
        if key not in self.patterns:
            self.patterns[key] = RawMathPattern(
                key=key,
                kind=kind,
                description=description,
                first_step=step,
                last_step=step,
                properties=dict(properties or {}),
            )
        pattern = self.patterns[key]
        pattern.add_evidence(step, amount=amount)
        if properties:
            pattern.properties.update(properties)
        return pattern

    def _maybe_install_patterns(self, step: int):
        if self.knowledge_base is None:
            return

        for pattern in self.patterns.values():
            threshold = (
                self.operation_evidence_threshold
                if pattern.kind == 'operation'
                else self.concept_evidence_threshold
            )
            if pattern.evidence < threshold:
                continue
            if pattern.concept_name is None:
                pattern.concept_name = self._install_concept(pattern, step).internal_name
            if pattern.kind == 'operation' and pattern.rule_name:
                rule = self.knowledge_base.rules.get(pattern.rule_name)
                if rule is not None:
                    rule.evidence_for = pattern.evidence
                    rule.confidence = 1.0
            if (
                pattern.kind == 'operation'
                and pattern.rule_name is None
                and 'action_type' in pattern.properties
                and 'delta' in pattern.properties
            ):
                pattern.rule_name = self._install_operation_rule(pattern, step)

    def _install_concept(self, pattern: RawMathPattern, step: int):
        concept_type = {
            'relation': ConceptType.RELATION,
            'operation': ConceptType.OPERATION,
            'quantity_like': ConceptType.PATTERN,
            'transform': ConceptType.PATTERN,
            'pattern': ConceptType.PATTERN,
        }.get(pattern.kind, ConceptType.PATTERN)

        notation = None
        if pattern.kind == 'operation':
            notation = self.knowledge_base.notation_system.generate_operation_symbol(pattern.key)

        return self.knowledge_base.add_concept(
            concept_type=concept_type,
            description=pattern.description,
            feature_key=pattern.key,
            step=step,
            properties={
                'source': 'emergent_math',
                'pattern_key': pattern.key,
                'pattern_kind': pattern.kind,
                'evidence': pattern.evidence,
                **pattern.properties,
            },
            notation=notation,
        )

    def _install_operation_rule(self, pattern: RawMathPattern, step: int) -> str:
        action_type = pattern.properties.get('action_type', 'unknown')
        delta = pattern.properties.get('delta')
        rule = self.knowledge_base.add_hypothesis(
            conditions=f"intervention_signature={action_type}",
            prediction=f"raw_collection_extent_delta={delta}",
            feature_key=pattern.key,
            step=step,
            properties={
                'source': 'emergent_math',
                'hypothesis_type': 'emergent_math_operation',
                'pattern_key': pattern.key,
                'delta': delta,
                'action_type': action_type,
            },
        )
        rule.evidence_for = pattern.evidence
        rule.evidence_against = 0
        rule.confidence = 1.0
        rule.status = RuleStatus.CONFIRMED
        rule.confirmed_at_step = step
        self.knowledge_base.discovery_log.append({
            'type': 'discovery',
            'name': rule.internal_name,
            'conditions': rule.conditions,
            'prediction': rule.prediction,
            'step': step,
            'confidence': rule.confidence,
            'evidence_for': rule.evidence_for,
            'evidence_against': rule.evidence_against,
        })
        return rule.internal_name

    def _human_mapping(self, pattern: RawMathPattern) -> tuple[str, str] | None:
        role = pattern.properties.get('structural_role')
        if role == 'persistence':
            return (
                'identity / object permanence',
                'same raw token survives multiple transitions',
            )
        if role == 'invariant':
            return (
                'equality-like invariant',
                'a channel remains unchanged while other channels vary',
            )
        if role == 'discrete_extent':
            return (
                'discrete quantity / cardinality',
                'the collection has a repeatably measurable extent',
            )
        if role == 'discrete_delta':
            return (
                'integer-like change',
                'collection extent shifts by a whole-token delta',
            )
        if role == 'intervention_transform':
            delta = pattern.properties.get('delta')
            if delta == 1:
                return (
                    'successor-like operation',
                    'one intervention repeatedly increases discrete extent by one',
                )
            if delta == -1:
                return (
                    'predecessor-like operation',
                    'one intervention repeatedly decreases discrete extent by one',
                )
            return (
                'translation-like operation',
                'one intervention repeatedly shifts discrete extent',
            )
        if role == 'repeatable_transform':
            return (
                'iteration / composition',
                'the same transform can be applied repeatedly',
            )
        if role == 'paired_step_transform':
            return (
                'operation composition',
                'two adjacent transforms have a combined net effect',
            )
        if role == 'returning_sequence':
            return (
                'identity-like cancellation',
                'two adjacent transforms return the measured extent to its prior level',
            )
        if role == 'same_net_sequence_set':
            return (
                'operation equivalence class',
                'different adjacent operation sequences produce the same net transform',
            )
        if role == 'swapped_sequence_same_net':
            return (
                'commutativity-like operation behavior',
                'swapping adjacent transform order preserves the measured net effect',
            )
        if role == 'opposed_transforms':
            return (
                'inverse operation pair',
                'two transforms undo each other at the extent level',
            )
        if role == 'ordering':
            return (
                'order relation',
                'a scalar channel ranks observed tokens',
            )
        if role == 'metric_separation':
            return (
                'metric / distance structure',
                'token pairs expose repeatable magnitudes between positions',
            )
        if role == 'scale_quotient':
            return (
                'ratio-like relation',
                'two scalar channels can be compared by repeatable quotients',
            )
        if role == 'balanced_symmetry':
            return (
                'symmetry-like balance',
                'values appear on both sides of an aggregate reference level',
            )
        if role == 'cyclic_recurrence':
            return (
                'periodicity-like recurrence',
                'a channel alternates through recurring direction states',
            )
        if role == 'input_output_mapping':
            return (
                'function-like mapping',
                'a repeated input signature maps to a repeated output signature',
            )
        if role == 'conditional_transition':
            return (
                'conditional rule',
                'an observed condition is followed by a repeatable transition',
            )
        return None

    def _snapshot(self, raw_state: dict) -> dict:
        objects = {}
        for obj in raw_state.get('objects', []):
            token_id = obj['id']
            position = obj.get('position', (0.0, 0.0))
            velocity = obj.get('velocity', (0.0, 0.0))
            objects[token_id] = {
                'position_x': float(position[0]),
                'position_y': float(position[1]),
                'velocity_x': float(velocity[0]),
                'velocity_y': float(velocity[1]),
                'mass': float(obj.get('mass', 0.0)),
                'radius': float(obj.get('radius', 0.0)),
            }
        return {
            'ids': set(objects.keys()),
            'objects': objects,
            'extent': len(objects),
        }

    def _unique_rounded(self, values: list[float]) -> set[float]:
        return {round(value / self.tolerance) * self.tolerance for value in values}

    def _format_sequence_signature(self, signature: tuple) -> str:
        first_action, second_action, first_delta, second_delta = signature
        return f"{first_action}:{first_delta}->{second_action}:{second_delta}"

    def _sign(self, value: float) -> int:
        if value > self.tolerance:
            return 1
        if value < -self.tolerance:
            return -1
        return 0
