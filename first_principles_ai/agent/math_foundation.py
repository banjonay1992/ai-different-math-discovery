from __future__ import annotations

"""
Math foundation readiness layer.

This layer is deliberately downstream of raw discovery. It does not teach
finished math facts. It inspects what the agent already found, compresses those
findings into readiness gates, builds small proof/check traces, and proposes
math-specific probes for the next run.
"""

from dataclasses import dataclass, field

from agent.representation import ConceptType, KnowledgeBase, RuleStatus


FORBIDDEN_FOUNDATION_LABELS = {
    'addition',
    'subtraction',
    'commutativity',
    'equivalence',
    'theorem',
    'proof',
}


@dataclass
class FoundationArtifact:
    """One readiness artifact made from prior raw discoveries."""
    key: str
    artifact_type: str
    description: str
    evidence: int
    properties: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'key': self.key,
            'artifact_type': self.artifact_type,
            'description': self.description,
            'evidence': self.evidence,
            'properties': dict(self.properties),
        }


@dataclass
class FoundationProbe:
    """A math-specific question the next run can test."""
    key: str
    question: str
    target: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'key': self.key,
            'question': self.question,
            'target': dict(self.target),
        }


@dataclass
class FoundationProofTrace:
    """Compact check trace for a readiness artifact."""
    key: str
    claim: str
    supports: list[str] = field(default_factory=list)
    exceptions: list[str] = field(default_factory=list)
    transfer_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'key': self.key,
            'claim': self.claim,
            'supports': list(self.supports),
            'exceptions': list(self.exceptions),
            'transfer_notes': list(self.transfer_notes),
        }


@dataclass
class MathFoundationReport:
    """Readiness report before the final discovery run."""
    artifacts: list[FoundationArtifact]
    proof_traces: list[FoundationProofTrace]
    probes: list[FoundationProbe]
    gates: dict[str, bool]

    @property
    def readiness_score(self) -> float:
        if not self.gates:
            return 0.0
        return sum(1 for ready in self.gates.values() if ready) / len(self.gates)

    @property
    def missing_gates(self) -> list[str]:
        return [
            gate for gate, ready in self.gates.items()
            if not ready
        ]

    def to_dict(self) -> dict:
        return {
            'readiness_score': round(self.readiness_score, 3),
            'gates': dict(self.gates),
            'missing_gates': self.missing_gates,
            'artifacts': [artifact.to_dict() for artifact in self.artifacts],
            'proof_traces': [trace.to_dict() for trace in self.proof_traces],
            'probes': [probe.to_dict() for probe in self.probes],
        }

    def summary(self, limit: int = 8) -> str:
        lines = [
            "Math foundation readiness:",
            f"  Readiness score: {self.readiness_score:.1%}",
            f"  Missing gates: {', '.join(self.missing_gates) if self.missing_gates else 'none'}",
        ]
        lines.append("  Gates:")
        for gate, ready in self.gates.items():
            lines.append(f"    {gate}: {'ready' if ready else 'needs work'}")
        if self.artifacts:
            lines.append("  Foundation artifacts:")
            for artifact in self.artifacts[:limit]:
                lines.append(
                    f"    {artifact.key}: {artifact.description} "
                    f"(evidence={artifact.evidence})"
                )
        if self.proof_traces:
            lines.append("  Check traces:")
            for trace in self.proof_traces[:min(3, limit)]:
                lines.append(
                    f"    {trace.key}: supports={len(trace.supports)}, "
                    f"exceptions={len(trace.exceptions)}, transfers={len(trace.transfer_notes)}"
                )
        if self.probes:
            lines.append("  Next math probes:")
            for probe in self.probes[:min(3, limit)]:
                lines.append(f"    {probe.key}: {probe.question}")
        return "\n".join(lines)


class MathFoundationWorkbench:
    """Build readiness artifacts from existing math and equation discoveries."""

    def __init__(
        self,
        knowledge_base: KnowledgeBase | None = None,
        math_discovery=None,
        equation_workbench=None,
    ):
        self.knowledge_base = knowledge_base
        self.math_discovery = math_discovery
        self.equation_workbench = equation_workbench

    def evaluate(self, install: bool = False) -> MathFoundationReport:
        patterns = self._patterns()
        equations = self._equations()
        artifacts = []
        artifacts.extend(self._number_artifacts(patterns))
        artifacts.extend(self._equation_template_artifacts(equations))
        artifacts.extend(self._composition_artifacts(patterns))
        artifacts.extend(self._geometry_artifacts(patterns, equations))

        proof_traces = self._proof_traces(artifacts, equations, patterns)
        gates = self._gates(artifacts, proof_traces)
        probes = self._probes(gates, artifacts)
        report = MathFoundationReport(
            artifacts=artifacts,
            proof_traces=proof_traces,
            probes=probes,
            gates=gates,
        )
        if install and self.knowledge_base is not None:
            self._install_report(report)
        return report

    def label_leaks(self, report: MathFoundationReport) -> list[dict]:
        leaks = []
        for artifact in report.artifacts:
            text = f"{artifact.key} {artifact.description}".lower()
            found = sorted(label for label in FORBIDDEN_FOUNDATION_LABELS if label in text)
            if found:
                leaks.append({
                    'artifact': artifact.key,
                    'labels': found,
                    'description': artifact.description,
                })
        return leaks

    def _number_artifacts(self, patterns: list) -> list[FoundationArtifact]:
        roles = self._roles(patterns)
        delta_patterns = [
            pattern for pattern in patterns
            if pattern.properties.get('structural_role') == 'discrete_delta'
        ]
        has_positive = any(pattern.properties.get('delta') == 1 for pattern in delta_patterns)
        has_negative = any(pattern.properties.get('delta') == -1 for pattern in delta_patterns)
        sequence_roles = {
            'repeatable_transform',
            'same_net_sequence_set',
            'swapped_sequence_same_net',
            'returning_sequence',
        }
        sequence_evidence = sum(
            pattern.evidence for pattern in patterns
            if pattern.properties.get('structural_role') in sequence_roles
        )
        artifacts = []
        if 'discrete_extent' in roles and delta_patterns:
            artifacts.append(FoundationArtifact(
                key='raw_foundation:extent_delta_ladder',
                artifact_type='number_system',
                description='Raw collection extent, one-step changes, and repeated shifts share a reusable ladder.',
                evidence=sum(pattern.evidence for pattern in delta_patterns),
                properties={
                    'has_positive_unit_shift': has_positive,
                    'has_negative_unit_shift': has_negative,
                    'delta_values': sorted({
                        pattern.properties.get('delta')
                        for pattern in delta_patterns
                    }),
                },
            ))
        if sequence_evidence:
            artifacts.append(FoundationArtifact(
                key='raw_foundation:path_net_result_system',
                artifact_type='number_system',
                description='Different action paths can be compared by their net raw extent result.',
                evidence=sequence_evidence,
                properties={
                    'sequence_roles': sorted(sequence_roles & roles),
                },
            ))
        return artifacts

    def _equation_template_artifacts(self, equations: list) -> list[FoundationArtifact]:
        equations_by_key = {equation.key: equation for equation in equations}
        artifacts = []
        if (
            'raw_eq:next_x_from_velocity' in equations_by_key
            and 'raw_eq:next_y_from_velocity' in equations_by_key
        ):
            x_eq = equations_by_key['raw_eq:next_x_from_velocity']
            y_eq = equations_by_key['raw_eq:next_y_from_velocity']
            artifacts.append(FoundationArtifact(
                key='raw_foundation:channel_step_template',
                artifact_type='equation_template',
                description='Two position channels share a next-channel update template with rate and elapsed step.',
                evidence=int((x_eq.score + y_eq.score) * 100),
                properties={
                    'source_equations': [x_eq.key, y_eq.key],
                    'template': 'channel_next ~= channel + rate * dt',
                    'min_score': min(x_eq.score, y_eq.score),
                },
            ))
        if (
            'raw_eq:mass_persistence' in equations_by_key
            and 'raw_eq:radius_persistence' in equations_by_key
        ):
            artifacts.append(FoundationArtifact(
                key='raw_foundation:stable_channel_template',
                artifact_type='equation_template',
                description='Stable scalar channels share a next-channel persistence template.',
                evidence=2,
                properties={
                    'source_equations': [
                        'raw_eq:mass_persistence',
                        'raw_eq:radius_persistence',
                    ],
                    'template': 'channel_next ~= channel',
                },
            ))
        if (
            'raw_eq:delta_count_from_action' in equations_by_key
            and 'raw_eq:next_count_from_action' in equations_by_key
        ):
            artifacts.append(FoundationArtifact(
                key='raw_foundation:extent_action_template',
                artifact_type='equation_template',
                description='Raw action indicators and prior extent form a reusable next-extent template.',
                evidence=2,
                properties={
                    'source_equations': [
                        'raw_eq:delta_count_from_action',
                        'raw_eq:next_count_from_action',
                    ],
                    'template': 'extent_next ~= extent + positive_indicator - negative_indicator',
                },
            ))
        return artifacts

    def _composition_artifacts(self, patterns: list) -> list[FoundationArtifact]:
        roles = self._roles(patterns)
        artifacts = []
        if 'returning_sequence' in roles or 'opposed_transforms' in roles:
            evidence = sum(
                pattern.evidence for pattern in patterns
                if pattern.properties.get('structural_role') in {'returning_sequence', 'opposed_transforms'}
            )
            artifacts.append(FoundationArtifact(
                key='raw_foundation:return_path_planner',
                artifact_type='composition_inverse',
                description='Observed path pairs can return a measured extent to its prior level.',
                evidence=evidence,
                properties={'roles': sorted(roles & {'returning_sequence', 'opposed_transforms'})},
            ))
        if 'same_net_sequence_set' in roles or 'swapped_sequence_same_net' in roles:
            evidence = sum(
                pattern.evidence for pattern in patterns
                if pattern.properties.get('structural_role') in {'same_net_sequence_set', 'swapped_sequence_same_net'}
            )
            artifacts.append(FoundationArtifact(
                key='raw_foundation:same_result_path_planner',
                artifact_type='composition_inverse',
                description='Observed paths can be grouped when they land on the same measured result.',
                evidence=evidence,
                properties={'roles': sorted(roles & {'same_net_sequence_set', 'swapped_sequence_same_net'})},
            ))
        return artifacts

    def _geometry_artifacts(self, patterns: list, equations: list) -> list[FoundationArtifact]:
        roles = self._roles(patterns)
        equation_roles = {equation.role for equation in equations}
        artifacts = []
        if 'metric_separation' in roles and 'ordering' in roles:
            artifacts.append(FoundationArtifact(
                key='raw_foundation:metric_order_space',
                artifact_type='geometry',
                description='Separation magnitudes and scalar rankings form a reusable spatial comparison basis.',
                evidence=sum(
                    pattern.evidence for pattern in patterns
                    if pattern.properties.get('structural_role') in {'metric_separation', 'ordering'}
                ),
                properties={'roles': sorted(roles & {'metric_separation', 'ordering'})},
            ))
        direction_roles = {
            'vector_direction_equation',
            'vector_perpendicular_equation',
            'residual_direction_equation',
            'residual_perpendicular_equation',
            'local_residual_direction_equation',
            'local_residual_perpendicular_equation',
            'residual_distance_scaled_direction_equation',
            'residual_distance_scaled_perpendicular_equation',
            'local_residual_distance_scaled_direction_equation',
            'local_residual_distance_scaled_perpendicular_equation',
            'generated_operator_distance_scaled_direction_equation',
            'generated_operator_distance_scaled_perpendicular_equation',
            'generated_operator_cutoff_direction_equation',
            'generated_operator_cutoff_perpendicular_equation',
            'generated_operator_tapered_distance_direction_equation',
            'generated_operator_tapered_distance_perpendicular_equation',
        }
        if equation_roles & direction_roles:
            artifacts.append(FoundationArtifact(
                key='raw_foundation:direction_vector_space',
                artifact_type='geometry',
                description='Direction and perpendicular channels can be tested as vector-style templates.',
                evidence=len(equation_roles & direction_roles),
                properties={'equation_roles': sorted(equation_roles)},
            ))
        if 'balanced_symmetry' in roles or 'cyclic_recurrence' in roles:
            artifacts.append(FoundationArtifact(
                key='raw_foundation:shape_recurrence_space',
                artifact_type='geometry',
                description='Balanced channel placement or recurring direction states support shape-like checks.',
                evidence=sum(
                    pattern.evidence for pattern in patterns
                    if pattern.properties.get('structural_role') in {'balanced_symmetry', 'cyclic_recurrence'}
                ),
                properties={'roles': sorted(roles & {'balanced_symmetry', 'cyclic_recurrence'})},
            ))
        return artifacts

    def _proof_traces(
        self,
        artifacts: list[FoundationArtifact],
        equations: list,
        patterns: list,
    ) -> list[FoundationProofTrace]:
        traces = []
        equation_by_key = {equation.key: equation for equation in equations}
        pattern_by_role = {}
        for pattern in patterns:
            pattern_by_role.setdefault(pattern.properties.get('structural_role'), []).append(pattern)
        for artifact in artifacts:
            supports = []
            exceptions = []
            transfer_notes = []
            for source_key in artifact.properties.get('source_equations', []):
                equation = equation_by_key.get(source_key)
                if equation is None:
                    continue
                supports.append(
                    f"{source_key} held out with score {equation.score:.2f} over {equation.sample_count} samples"
                )
                if equation.score < 0.62:
                    exceptions.append(f"{source_key} needs stronger held-out evidence")
            for role in artifact.properties.get('roles', []):
                role_patterns = pattern_by_role.get(role, [])
                supports.append(f"{role} observed in {len(role_patterns)} raw pattern(s)")
            if artifact.artifact_type == 'equation_template':
                transfer_notes.append('template spans multiple channels or action-state views')
            if artifact.artifact_type == 'composition_inverse':
                transfer_notes.append('path result checks can guide future intervention sequences')
            if artifact.artifact_type == 'geometry':
                transfer_notes.append('spatial checks can be retested at new probe locations')
            if not supports:
                supports.append(f"{artifact.key} evidence count {artifact.evidence}")
            traces.append(FoundationProofTrace(
                key=f"raw_check:{artifact.key}",
                claim=artifact.description,
                supports=supports,
                exceptions=exceptions,
                transfer_notes=transfer_notes,
            ))
        return traces

    def _gates(self, artifacts: list[FoundationArtifact], proof_traces: list[FoundationProofTrace]) -> dict[str, bool]:
        artifact_types = {artifact.artifact_type for artifact in artifacts}
        return {
            'number_system_stability': 'number_system' in artifact_types,
            'equation_templates': 'equation_template' in artifact_types,
            'composition_inverse_planning': 'composition_inverse' in artifact_types,
            'check_traces': bool(proof_traces),
            'geometry_basis': 'geometry' in artifact_types,
            'self_directed_math_probes': True,
        }

    def _probes(self, gates: dict[str, bool], artifacts: list[FoundationArtifact]) -> list[FoundationProbe]:
        probes = []
        if not gates.get('composition_inverse_planning'):
            probes.append(FoundationProbe(
                key='raw_probe:path_return_sequence',
                question='Try opposing action paths and check whether measured extent returns to its prior level.',
                target={'action_sequence': ['spawn', 'remove']},
            ))
        if not gates.get('geometry_basis'):
            probes.append(FoundationProbe(
                key='raw_probe:spatial_direction_template',
                question='Spawn low-motion probes at several offsets and compare direction/perpendicular channel changes.',
                target={'locations': 'center_offsets'},
            ))
        if not gates.get('equation_templates'):
            probes.append(FoundationProbe(
                key='raw_probe:channel_template_transfer',
                question='Retest whether a next-channel equation transfers across two observed channels.',
                target={'channels': ['x', 'y']},
            ))
        if gates.get('composition_inverse_planning') and gates.get('geometry_basis'):
            probes.append(FoundationProbe(
                key='raw_probe:compare_path_and_space_templates',
                question='Compare whether a path result relation holds after moving probes to a new spatial region.',
                target={'combine': ['path_result', 'spatial_offsets']},
            ))
        if not probes:
            probes.append(FoundationProbe(
                key='raw_probe:final_discovery_ready',
                question='Run the final held-out discovery campaign and inspect invented structures manually.',
                target={'status': 'ready_for_user_watched_run'},
            ))
        return probes

    def _install_report(self, report: MathFoundationReport):
        existing = {
            concept.feature_key
            for concept in self.knowledge_base.get_all_concepts()
            if concept.properties.get('source') == 'math_foundation'
        }
        for artifact in report.artifacts:
            if artifact.key in existing:
                continue
            self.knowledge_base.add_concept(
                ConceptType.META_CONCEPT,
                artifact.description,
                artifact.key,
                step=artifact.evidence,
                properties={
                    'source': 'math_foundation',
                    'artifact_type': artifact.artifact_type,
                    'readiness_gate': artifact.artifact_type,
                    **artifact.properties,
                },
            )
        for trace in report.proof_traces:
            if trace.key in self.knowledge_base.rules:
                continue
            rule = self.knowledge_base.add_hypothesis(
                conditions='foundation artifact is reviewed',
                prediction=trace.claim,
                feature_key=trace.key,
                step=len(trace.supports),
                properties={
                    'source': 'math_foundation',
                    'hypothesis_type': 'foundation_check_trace',
                    'supports': list(trace.supports),
                    'exceptions': list(trace.exceptions),
                    'transfer_notes': list(trace.transfer_notes),
                },
            )
            rule.evidence_for = max(1, len(trace.supports))
            rule.evidence_against = len(trace.exceptions)
            total = rule.evidence_for + rule.evidence_against
            rule.confidence = rule.evidence_for / max(total, 1)
            if not trace.exceptions:
                rule.status = RuleStatus.CONFIRMED

    def _patterns(self) -> list:
        if self.math_discovery is None:
            return []
        return list(self.math_discovery.discovered_patterns())

    def _equations(self) -> list:
        if self.equation_workbench is None:
            return []
        return list(self.equation_workbench.discovered_equations())

    def _roles(self, patterns: list) -> set[str]:
        return {
            pattern.properties.get('structural_role')
            for pattern in patterns
            if pattern.properties.get('structural_role')
        }
