from __future__ import annotations

"""
Equation discovery starter kit.

This gives the agent useful mathematical building blocks without handing it
finished physics labels. It can combine primitive variables with simple
operators, score candidate equations on held-out observations, and install
compact equations that earn evidence.
"""

from dataclasses import dataclass, field
import math
from statistics import mean
from typing import Callable

from agent.discovery_loop import AutonomousDiscoveryLoop
from agent.representation import KnowledgeBase, RuleStatus


FORBIDDEN_EQUATION_LABELS = {
    'gravity',
    'newton',
    'vortex',
    'centripetal',
    'force',
    'momentum',
    'energy',
}


@dataclass
class PrimitiveEquation:
    """A scored equation candidate."""
    key: str
    target: str
    expression: str
    description: str
    score: float
    mse: float
    baseline_mse: float
    complexity: int
    sample_count: int
    parameters: dict = field(default_factory=dict)
    role: str = 'starter_equation'
    rule_name: str | None = None

    def to_dict(self) -> dict:
        return {
            'key': self.key,
            'target': self.target,
            'expression': self.expression,
            'description': self.description,
            'score': round(self.score, 3),
            'mse': round(self.mse, 6),
            'baseline_mse': round(self.baseline_mse, 6),
            'complexity': self.complexity,
            'sample_count': self.sample_count,
            'parameters': {
                key: round(value, 6) if isinstance(value, float) else value
                for key, value in self.parameters.items()
            },
            'role': self.role,
            'rule_name': self.rule_name,
        }


class EquationWorkbench:
    """
    Build and test compact equations from primitive observations.

    The workbench is intentionally small. It does not know domain labels such
    as gravity or vortex; it only sees state variables, action indicators, and
    simple operators like +, -, *, /, square, and sqrt.
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase | None = None,
        min_samples: int = 16,
        install_score_threshold: float = 0.62,
        max_installed: int = 8,
        generated_operator_priors: list[dict] | None = None,
        max_operator_feedback_rows: int = 384,
        max_operator_feedback_operators: int = 5,
    ):
        self.knowledge_base = knowledge_base
        self.min_samples = min_samples
        self.install_score_threshold = install_score_threshold
        self.max_installed = max_installed
        self.max_operator_feedback_rows = max(
            min_samples * 2,
            int(max_operator_feedback_rows),
        )
        self.max_operator_feedback_operators = max(
            1,
            int(max_operator_feedback_operators),
        )
        self.aggregate_rows: list[dict] = []
        self.object_rows: list[dict] = []
        self.equations: dict[str, PrimitiveEquation] = {}
        self.probe_suggestions: list[dict] = []
        self.last_probe_step: int = -999
        self.probe_index: int = 0
        self.discovery_loop = AutonomousDiscoveryLoop()
        self.generated_operator_bank: dict[str, dict] = {}
        self.generated_operator_prior_count = 0
        self.install_generated_operator_priors(generated_operator_priors or [])

    def install_generated_operator_priors(self, priors: list[dict]):
        for prior in priors:
            key = prior.get('key')
            kind = prior.get('operator_kind')
            if not key or not kind:
                continue
            item = dict(prior)
            item.setdefault('source', 'theory_memory_prior')
            item.setdefault('usefulness', 0.3)
            self.generated_operator_bank[str(key)] = item
        self._refresh_operator_prior_count()

    def _refresh_operator_prior_count(self):
        self.generated_operator_prior_count = sum(
            1 for item in self.generated_operator_bank.values()
            if item.get('source') == 'theory_memory_prior'
            or str(item.get('key', '')).startswith('operator:memory_prior:')
        )

    def observe_transition(self, before: dict, after: dict, action: dict | None, step: int):
        action = action or {'type': 'wait'}
        dt = max(float(after.get('time', 0.0)) - float(before.get('time', 0.0)), 1e-9)
        before_objects = self._object_map(before)
        after_objects = self._object_map(after)
        action_type = action.get('type', 'wait')

        self.aggregate_rows.append({
            'step': step,
            'time': float(before.get('time', 0.0)),
            'dt': dt,
            'count_before': len(before_objects),
            'count_after': len(after_objects),
            'delta_count': len(after_objects) - len(before_objects),
            'action_spawn': 1.0 if action_type == 'spawn' else 0.0,
            'action_remove': 1.0 if action_type == 'remove' else 0.0,
            'action_push': 1.0 if action_type == 'push' else 0.0,
            'action_move': 1.0 if action_type == 'move' else 0.0,
            'action_freeze': 1.0 if action_type == 'freeze' else 0.0,
            'action_duplicate': 1.0 if action_type == 'duplicate' else 0.0,
            'action_wait': 1.0 if action_type == 'wait' else 0.0,
        })

        world_size = after.get('world_size') or before.get('world_size') or (20.0, 20.0)
        anchor_x = float(world_size[0]) / 2.0
        anchor_y = float(world_size[1]) / 2.0
        for token_id in sorted(before_objects.keys() & after_objects.keys()):
            row = self._object_row(
                before_objects[token_id],
                after_objects[token_id],
                dt,
                step,
                anchor_x,
                anchor_y,
                action,
            )
            if row is not None:
                self.object_rows.append(row)

    def discover(self, step: int | None = None) -> list[PrimitiveEquation]:
        candidates = []
        candidates.extend(self._aggregate_equations())
        candidates.extend(self._object_equations())
        self.equations = self._merge_candidate_equations(candidates)

        operator_keys_before = set(self.generated_operator_bank)
        self._refresh_generated_operator_bank(step or self._latest_step())
        if set(self.generated_operator_bank) - operator_keys_before:
            feedback_candidates = self._operator_feedback_equations(self.object_rows)
            if feedback_candidates:
                self.equations = self._merge_candidate_equations(
                    list(self.equations.values()) + feedback_candidates
                )
        if self.knowledge_base is not None:
            self._install_equations(step or self._latest_step())
        return self.discovered_equations()

    def _merge_candidate_equations(
        self,
        candidates: list[PrimitiveEquation],
    ) -> dict[str, PrimitiveEquation]:
        candidates.sort(key=lambda item: (item.score, -item.complexity), reverse=True)
        old_equations = self.equations
        merged_equations = {}
        for equation in candidates:
            old_equation = old_equations.get(equation.key)
            if old_equation is not None:
                equation.rule_name = old_equation.rule_name
                if self._should_retain_best_equation(old_equation, equation):
                    merged_equations[equation.key] = old_equation
                    continue
            merged_equations[equation.key] = equation
        return merged_equations

    def _should_retain_best_equation(
        self,
        old_equation: PrimitiveEquation,
        new_equation: PrimitiveEquation,
    ) -> bool:
        retainable_roles = {
            'local_residual_direction_equation',
            'local_residual_perpendicular_equation',
            'local_residual_distance_scaled_direction_equation',
            'local_residual_distance_scaled_perpendicular_equation',
            'residual_distance_scaled_direction_equation',
            'residual_distance_scaled_perpendicular_equation',
            'generated_operator_distance_scaled_direction_equation',
            'generated_operator_distance_scaled_perpendicular_equation',
            'generated_operator_cutoff_direction_equation',
            'generated_operator_cutoff_perpendicular_equation',
            'generated_operator_tapered_distance_direction_equation',
            'generated_operator_tapered_distance_perpendicular_equation',
            'generated_operator_periodic_equation',
            'residual_periodic_equation',
        }
        return (
            old_equation.role in retainable_roles
            and old_equation.score > new_equation.score
            and old_equation.score >= self._probe_score_threshold(old_equation.role)
            and old_equation.baseline_mse >= 1e-3
        )

    def discovered_equations(self) -> list[PrimitiveEquation]:
        return sorted(
            self.equations.values(),
            key=lambda item: (item.score, -item.complexity),
            reverse=True,
        )

    def label_leaks(self) -> list[dict]:
        leaks = []
        for equation in self.discovered_equations():
            text = f"{equation.key} {equation.description} {equation.expression}".lower()
            found = sorted(label for label in FORBIDDEN_EQUATION_LABELS if label in text)
            if found:
                leaks.append({
                    'equation': equation.key,
                    'labels': found,
                    'description': equation.description,
                    'expression': equation.expression,
                })
        return leaks

    def review_pack(self, limit: int = 8) -> dict:
        equations = [equation.to_dict() for equation in self.discovered_equations()]
        categories = self._categorized_equations(equations, limit=limit)
        interesting = self._interesting_equations(equations, limit=limit)
        discovery_report = self.discovery_report(limit=limit)
        misses = [
            item for item in equations
            if 0.25 <= item['score'] < self.install_score_threshold
        ][:limit]
        return {
            'equation_count': len(equations),
            'installed_count': sum(1 for item in equations if item.get('rule_name')),
            'top_equations': equations[:limit],
            'interesting_equations': interesting,
            'categories': categories,
            'interesting_misses': misses,
            'probe_suggestions': list(self.probe_suggestions),
            'discovery_loop': discovery_report,
            'generated_operator_count': len(self.generated_operator_bank),
            'generated_operator_prior_count': self.generated_operator_prior_count,
            'operator_prior_results': self._operator_prior_results(equations),
            'generated_operators': list(self.generated_operator_bank.values()),
            'label_leaks': self.label_leaks(),
        }

    def _operator_prior_results(self, equations: list[dict]) -> list[dict]:
        results = []
        for operator in self.generated_operator_bank.values():
            key = operator.get('key')
            is_prior = (
                operator.get('source') == 'theory_memory_prior'
                or str(key).startswith('operator:memory_prior:')
            )
            if not key or not is_prior:
                continue
            matching = [
                equation for equation in equations
                if equation.get('parameters', {}).get('operator_key') == key
            ]
            best_equation = max(
                matching,
                key=lambda equation: float(equation.get('score', 0.0) or 0.0),
                default={},
            )
            best = float(best_equation.get('score', 0.0) or 0.0)
            if best >= self.install_score_threshold:
                outcome = 'confirmed'
            elif best >= 0.25:
                outcome = 'weak'
            else:
                outcome = 'unmatched'
            results.append({
                'operator_key': key,
                'operator_kind': operator.get('operator_kind'),
                'outcome': outcome,
                'best_score': round(best, 3),
                'matching_equation_count': len(matching),
                'parameters': dict(operator.get('parameters') or {}),
                'best_equation': {
                    'key': best_equation.get('key'),
                    'role': best_equation.get('role'),
                    'target': best_equation.get('target'),
                    'expression': best_equation.get('expression'),
                    'score': round(best, 3),
                    'parameters': dict(best_equation.get('parameters') or {}),
                } if best_equation else {},
            })
        results.sort(
            key=lambda item: (
                item['best_score'],
                item['matching_equation_count'],
                item['operator_key'],
            ),
            reverse=True,
        )
        return results

    def discovery_report(
        self,
        limit: int = 8,
        current_count: int = 0,
        world_width: float = 20.0,
        world_height: float = 20.0,
        step: int | None = None,
    ) -> dict:
        report = self.discovery_loop.build_report(
            self.discovered_equations()[:limit * 2],
            step=step if step is not None else self._latest_step(),
            current_count=current_count,
            world_width=world_width,
            world_height=world_height,
        )
        return report.to_dict()

    def summary(self, limit: int = 8) -> str:
        pack = self.review_pack(limit=limit)
        lines = [
            "Equation workbench:",
            f"  Candidate equations: {pack['equation_count']}",
            f"  Installed equations: {pack['installed_count']}",
            f"  Generated operators: {pack['generated_operator_count']}",
            f"  Label leaks: {len(pack['label_leaks'])}",
        ]
        if pack['interesting_equations']:
            lines.append("  Interesting equations:")
            for equation in pack['interesting_equations']:
                lines.append(
                    f"    {equation['key']}: {equation['target']} ~= "
                    f"{equation['expression']} "
                    f"(score={equation['score']:.2f}, mse={equation['mse']:.4f})"
                )
        if pack['categories']:
            lines.append("  Categories:")
            for category, items in pack['categories'].items():
                if not items:
                    continue
                top = items[0]
                lines.append(
                    f"    {category}: {len(items)} "
                    f"(top={top['target']} ~= {top['expression']}, "
                    f"score={top['score']:.2f})"
                )
        if pack['interesting_misses']:
            lines.append("  Interesting misses:")
            for equation in pack['interesting_misses'][:3]:
                lines.append(
                    f"    {equation['key']}: score={equation['score']:.2f}, "
                    f"expression={equation['expression']}"
                )
        discovery_loop = pack.get('discovery_loop') or {}
        if discovery_loop.get('phase') != 'collect_more_observations':
            lines.append(
                f"  Discovery loop: {discovery_loop.get('phase')} "
                f"with {len(discovery_loop.get('theories', []))} theories"
            )
            probe_plan = discovery_loop.get('probe_plan')
            if probe_plan:
                lines.append(f"    Next probe: {probe_plan['reason']}")
                signature = probe_plan.get('disagreement_signature') or {}
                if signature.get('mode'):
                    lines.append(f"    Falsification mode: {signature['mode']}")
        return "\n".join(lines)

    def suggest_probe_action(
        self,
        current_count: int,
        world_width: float,
        world_height: float,
        step: int,
    ) -> dict | None:
        """
        Let discovered equations request a simple probe.

        This is opt-in from the experiment runner. It is meant for equation
        campaigns where we want the current equations to ask for cleaner
        evidence, without perturbing normal benchmark behavior.
        """
        if step - self.last_probe_step < 30:
            return None
        equation = self._best_probe_equation()
        if equation is None:
            return None
        if (
            current_count >= 12
            and equation.role not in {'residual_periodic_equation', 'generated_operator_periodic_equation'}
        ):
            return None
        discovery_report = self.discovery_report(
            current_count=current_count,
            world_width=world_width,
            world_height=world_height,
            step=step,
        )
        discovery_plan = discovery_report.get('probe_plan')

        if equation.role in {'residual_periodic_equation', 'generated_operator_periodic_equation'}:
            self.last_probe_step = step
            suggestion = {
                'equation_key': equation.key,
                'role': equation.role,
                'step': step,
                'period_steps': equation.parameters.get('period_steps'),
                'question': self._probe_question(equation),
                'discovery_phase': discovery_report.get('phase'),
                'discovery_probe_plan': discovery_plan,
            }
            self.probe_suggestions.append(suggestion)
            return {
                'type': 'wait',
                'source': 'equation_workbench_probe',
                'equation_key': equation.key,
            }

        center_x = float(equation.parameters.get('center_x', world_width / 2))
        center_y = float(equation.parameters.get('center_y', world_height / 2))
        radius = min(world_width, world_height) * 0.22
        if 'tapered_distance' in equation.role:
            radius = max(
                min(world_width, world_height) * 0.08,
                float(equation.parameters.get('cutoff_radius', radius)) * 0.72,
            )
        elif 'cutoff' in equation.role:
            radius = max(
                min(world_width, world_height) * 0.08,
                float(equation.parameters.get('cutoff_radius', radius)) * 1.08,
            )
        offsets = [
            (radius, 0.0),
            (-radius, 0.0),
            (0.0, radius),
            (0.0, -radius),
            (radius * 0.7, radius * 0.7),
            (-radius * 0.7, radius * 0.7),
        ]
        dx, dy = offsets[self.probe_index % len(offsets)]
        self.probe_index += 1
        self.last_probe_step = step
        x = min(max(center_x + dx, 1.0), world_width - 1.0)
        y = min(max(center_y + dy, 1.0), world_height - 1.0)
        suggestion = {
            'equation_key': equation.key,
            'role': equation.role,
            'step': step,
            'x': round(x, 3),
            'y': round(y, 3),
            'question': self._probe_question(equation),
            'discovery_phase': discovery_report.get('phase'),
            'discovery_probe_plan': discovery_plan,
        }
        self.probe_suggestions.append(suggestion)
        return {
            'type': 'spawn',
            'x': x,
            'y': y,
            'vx': 0.0,
            'vy': 0.0,
            'source': 'equation_workbench_probe',
            'equation_key': equation.key,
        }

    def _aggregate_equations(self) -> list[PrimitiveEquation]:
        rows = self.aggregate_rows
        if len(rows) < self.min_samples:
            return []
        specs = [
            _DirectSpec(
                key='raw_eq:delta_count_from_action',
                target='delta_count',
                expression='action_spawn - action_remove',
                description='A raw action signature predicts collection extent change.',
                complexity=3,
                fn=lambda row: row['action_spawn'] - row['action_remove'],
                role='action_extent_mapping',
            ),
            _DirectSpec(
                key='raw_eq:next_count_from_action',
                target='count_after',
                expression='count_before + action_spawn - action_remove',
                description='Previous collection extent and action signature predict next extent.',
                complexity=4,
                fn=lambda row: row['count_before'] + row['action_spawn'] - row['action_remove'],
                role='state_transition_equation',
            ),
        ]
        return [
            equation for equation in (self._score_direct(rows, spec) for spec in specs)
            if equation is not None
        ]

    def _object_equations(self) -> list[PrimitiveEquation]:
        rows = self.object_rows
        if len(rows) < self.min_samples:
            return []
        direct_specs = [
            _DirectSpec(
                key='raw_eq:next_x_from_velocity',
                target='next_x',
                expression='x + vx * dt',
                description='Position channel advances by velocity scaled by elapsed step.',
                complexity=3,
                fn=lambda row: row['x'] + row['vx'] * row['dt'],
                role='position_update_equation',
            ),
            _DirectSpec(
                key='raw_eq:next_y_from_velocity',
                target='next_y',
                expression='y + vy * dt',
                description='Position channel advances by velocity scaled by elapsed step.',
                complexity=3,
                fn=lambda row: row['y'] + row['vy'] * row['dt'],
                role='position_update_equation',
            ),
            _DirectSpec(
                key='raw_eq:mass_persistence',
                target='next_mass',
                expression='mass',
                description='A stable scalar channel predicts its next observed value.',
                complexity=1,
                fn=lambda row: row['mass'],
                role='invariant_equation',
            ),
            _DirectSpec(
                key='raw_eq:radius_persistence',
                target='next_radius',
                expression='radius',
                description='A stable scalar channel predicts its next observed value.',
                complexity=1,
                fn=lambda row: row['radius'],
                role='invariant_equation',
            ),
            _DirectSpec(
                key='raw_eq:speed_square_from_components',
                target='speed_sq',
                expression='vx*vx + vy*vy',
                description='Two signed channels combine into a squared magnitude.',
                complexity=4,
                fn=lambda row: row['vx'] * row['vx'] + row['vy'] * row['vy'],
                role='derived_magnitude_equation',
            ),
            _DirectSpec(
                key='raw_eq:anchor_distance_from_components',
                target='anchor_distance',
                expression='sqrt(anchor_dx*anchor_dx + anchor_dy*anchor_dy)',
                description='Two offset channels combine into a separation magnitude.',
                complexity=5,
                fn=lambda row: math.sqrt(row['anchor_dx'] ** 2 + row['anchor_dy'] ** 2),
                role='derived_magnitude_equation',
            ),
        ]
        equations = [
            equation for equation in (self._score_direct(rows, spec) for spec in direct_specs)
            if equation is not None
        ]

        fit_specs = [
            _FitSpec(
                key='raw_eq:constant_delta_vx',
                target='dvx',
                expression='k',
                description='A learned scalar parameter predicts a repeated channel change.',
                complexity=1,
                feature=lambda row: 1.0,
                role='constant_change_equation',
            ),
            _FitSpec(
                key='raw_eq:constant_delta_vy',
                target='dvy',
                expression='k',
                description='A learned scalar parameter predicts a repeated channel change.',
                complexity=1,
                feature=lambda row: 1.0,
                role='constant_change_equation',
            ),
            _FitSpec(
                key='raw_eq:anchor_radial_delta_vx',
                target='dvx',
                expression='k * unit_anchor_x',
                description='A scaled direction channel predicts horizontal change.',
                complexity=3,
                feature=lambda row: row['unit_anchor_x'],
                role='direction_scaled_equation',
            ),
            _FitSpec(
                key='raw_eq:anchor_radial_delta_vy',
                target='dvy',
                expression='k * unit_anchor_y',
                description='A scaled direction channel predicts vertical change.',
                complexity=3,
                feature=lambda row: row['unit_anchor_y'],
                role='direction_scaled_equation',
            ),
            _FitSpec(
                key='raw_eq:anchor_tangent_delta_vx',
                target='dvx',
                expression='k * (-unit_anchor_y)',
                description='A scaled perpendicular channel predicts horizontal change.',
                complexity=4,
                feature=lambda row: -row['unit_anchor_y'],
                role='perpendicular_scaled_equation',
            ),
            _FitSpec(
                key='raw_eq:anchor_tangent_delta_vy',
                target='dvy',
                expression='k * unit_anchor_x',
                description='A scaled perpendicular channel predicts vertical change.',
                complexity=4,
                feature=lambda row: row['unit_anchor_x'],
                role='perpendicular_scaled_equation',
            ),
        ]
        equations.extend(
            equation for equation in (self._score_fitted_scale(rows, spec) for spec in fit_specs)
            if equation is not None
        )
        vector_equations = [
            self._score_vector_scale(
                rows,
                key='raw_eq:anchor_radial_delta_vector',
                target='delta_velocity',
                expression='k * unit_anchor_vector',
                description='A scaled direction vector predicts paired channel changes.',
                x_feature=lambda row: row['unit_anchor_x'],
                y_feature=lambda row: row['unit_anchor_y'],
                complexity=4,
                role='vector_direction_equation',
            ),
            self._score_vector_scale(
                rows,
                key='raw_eq:anchor_tangent_delta_vector',
                target='delta_velocity',
                expression='k * perpendicular(unit_anchor_vector)',
                description='A scaled perpendicular vector predicts paired channel changes.',
                x_feature=lambda row: -row['unit_anchor_y'],
                y_feature=lambda row: row['unit_anchor_x'],
                complexity=5,
                role='vector_perpendicular_equation',
            ),
        ]
        equations.extend(equation for equation in vector_equations if equation is not None)
        equations.extend(self._residual_vector_equations(rows))
        equations.extend(self._temporal_residual_equations(rows))
        equations.extend(self._operator_feedback_equations(rows))
        return equations

    def _score_direct(self, rows: list[dict], spec: '_DirectSpec') -> PrimitiveEquation | None:
        train, valid = self._split_rows(rows)
        if not valid:
            return None
        predictions = [spec.fn(row) for row in valid]
        targets = [row[spec.target] for row in valid]
        return self._build_equation(
            key=spec.key,
            target=spec.target,
            expression=spec.expression,
            description=spec.description,
            predictions=predictions,
            targets=targets,
            sample_count=len(rows),
            complexity=spec.complexity,
            parameters={},
            role=spec.role,
        )

    def _score_fitted_scale(self, rows: list[dict], spec: '_FitSpec') -> PrimitiveEquation | None:
        train, valid = self._split_rows(rows)
        if not train or not valid:
            return None
        numerator = sum(row[spec.target] * spec.feature(row) for row in train)
        denominator = sum(spec.feature(row) ** 2 for row in train)
        if abs(denominator) < 1e-12:
            return None
        scale = numerator / denominator
        predictions = [scale * spec.feature(row) for row in valid]
        targets = [row[spec.target] for row in valid]
        return self._build_equation(
            key=spec.key,
            target=spec.target,
            expression=spec.expression,
            description=spec.description,
            predictions=predictions,
            targets=targets,
            sample_count=len(rows),
            complexity=spec.complexity,
            parameters={'k': scale},
            role=spec.role,
        )

    def _score_vector_scale(
        self,
        rows: list[dict],
        key: str,
        target: str,
        expression: str,
        description: str,
        x_feature: Callable[[dict], float],
        y_feature: Callable[[dict], float],
        complexity: int,
        role: str,
    ) -> PrimitiveEquation | None:
        train, valid = self._split_rows(rows)
        if not train or not valid:
            return None
        numerator = sum(
            row['dvx'] * x_feature(row) + row['dvy'] * y_feature(row)
            for row in train
        )
        denominator = sum(
            x_feature(row) ** 2 + y_feature(row) ** 2
            for row in train
        )
        if abs(denominator) < 1e-12:
            return None
        scale = numerator / denominator
        predictions = []
        targets = []
        for row in valid:
            predictions.extend([scale * x_feature(row), scale * y_feature(row)])
            targets.extend([row['dvx'], row['dvy']])
        return self._build_equation(
            key=key,
            target=target,
            expression=expression,
            description=description,
            predictions=predictions,
            targets=targets,
            sample_count=len(rows),
            complexity=complexity,
            parameters={'k': scale},
            role=role,
        )

    def _residual_vector_equations(self, rows: list[dict]) -> list[PrimitiveEquation]:
        rows = self._residual_candidate_rows(rows)
        if len(rows) < self.min_samples:
            return []
        train, valid = self._split_rows(rows)
        if not train or not valid:
            return []
        baseline = self._fit_axis_baseline(train)
        if baseline is None:
            return []

        anchor_center = self._average_anchor_center(rows)
        residuals = self._residual_rows(train, baseline, significant_only=True)
        center_specs = [
            ('anchor', anchor_center, 'anchor', rows, 'residual'),
        ]
        radial_center = self._infer_residual_center(residuals, mode='radial')
        if radial_center is not None:
            center_specs.append(('inferred_direction', radial_center, 'inferred', rows, 'residual'))
        tangent_center = self._infer_residual_center(residuals, mode='tangent')
        if tangent_center is not None and tangent_center != radial_center:
            center_specs.append(('inferred_perpendicular', tangent_center, 'inferred', rows, 'residual'))

        local_residuals = self._strong_residual_subset(residuals)
        local_rows = [row for row, _, _, _ in local_residuals]
        if len(local_rows) >= self.min_samples:
            local_radial_center = self._infer_residual_center(local_residuals, mode='radial')
            if local_radial_center is not None:
                center_specs.append((
                    'local_direction',
                    local_radial_center,
                    'local_inferred',
                    local_rows,
                    'local_residual',
                ))
            local_tangent_center = self._infer_residual_center(local_residuals, mode='tangent')
            if local_tangent_center is not None and local_tangent_center != local_radial_center:
                center_specs.append((
                    'local_perpendicular',
                    local_tangent_center,
                    'local_inferred',
                    local_rows,
                    'local_residual',
                ))

        equations = []
        seen = set()
        for prefix, center, center_label, score_rows, role_prefix in center_specs:
            cx, cy = center
            rounded_center = (round(cx, 3), round(cy, 3))
            if (prefix, rounded_center) in seen:
                continue
            seen.add((prefix, rounded_center))
            is_local = role_prefix == 'local_residual'
            direction_equation = self._score_residual_vector_scale(
                score_rows,
                baseline=baseline,
                center=(cx, cy),
                key=f'raw_eq:residual_{prefix}_direction_vector',
                target='baseline_adjusted_delta_velocity',
                expression=f'k * unit_{center_label}_vector',
                description=(
                    'After subtracting repeated scalar drift, a direction vector predicts paired channel changes.'
                    if not is_local
                    else 'After subtracting repeated scalar drift, high-change observations align with a direction vector.'
                ),
                vector_feature=lambda row, center=(cx, cy): self._unit_to_center(row, center),
                complexity=5,
                role='local_residual_direction_equation' if is_local else 'residual_direction_equation',
            )
            equations.append(direction_equation)
            distance_rows = rows if is_local else score_rows
            equations.extend(self._distance_scaled_residual_vector_equations(
                distance_rows,
                baseline=baseline,
                center=(cx, cy),
                center_label=center_label,
                prefix=prefix,
                relation='direction',
                reference_equation=direction_equation,
                is_local=is_local,
            ))
            perpendicular_equation = self._score_residual_vector_scale(
                score_rows,
                baseline=baseline,
                center=(cx, cy),
                key=f'raw_eq:residual_{prefix}_perpendicular_vector',
                target='baseline_adjusted_delta_velocity',
                expression=f'k * perpendicular(unit_{center_label}_vector)',
                description=(
                    'After subtracting repeated scalar drift, a perpendicular vector predicts paired channel changes.'
                    if not is_local
                    else 'After subtracting repeated scalar drift, high-change observations align with a perpendicular vector.'
                ),
                vector_feature=lambda row, center=(cx, cy): self._perpendicular_to_center(row, center),
                complexity=6,
                role='local_residual_perpendicular_equation' if is_local else 'residual_perpendicular_equation',
            )
            equations.append(perpendicular_equation)
            equations.extend(self._distance_scaled_residual_vector_equations(
                distance_rows,
                baseline=baseline,
                center=(cx, cy),
                center_label=center_label,
                prefix=prefix,
                relation='perpendicular',
                reference_equation=perpendicular_equation,
                is_local=is_local,
            ))
        return [equation for equation in equations if equation is not None]

    def _distance_scaled_residual_vector_equations(
        self,
        rows: list[dict],
        baseline: dict[str, tuple[float, float]],
        center: tuple[float, float],
        center_label: str,
        prefix: str,
        relation: str,
        reference_equation: PrimitiveEquation | None,
        is_local: bool,
    ) -> list[PrimitiveEquation]:
        best: PrimitiveEquation | None = None
        for exponent in self._candidate_distance_exponents():
            equation = self._score_residual_distance_vector_scale(
                rows,
                baseline=baseline,
                center=center,
                center_label=center_label,
                prefix=prefix,
                relation=relation,
                exponent=exponent,
                reference_equation=reference_equation,
                is_local=is_local,
            )
            if equation is None:
                continue
            if best is None or equation.score > best.score:
                best = equation
        return [best] if best is not None else []

    def _score_residual_distance_vector_scale(
        self,
        rows: list[dict],
        baseline: dict[str, tuple[float, float]],
        center: tuple[float, float],
        center_label: str,
        prefix: str,
        relation: str,
        exponent: float,
        reference_equation: PrimitiveEquation | None,
        is_local: bool,
    ) -> PrimitiveEquation | None:
        train, valid = self._split_rows(rows)
        if not train or not valid:
            return None
        numerator = 0.0
        denominator = 0.0
        for row in train:
            rx, ry = self._baseline_residual(row, baseline)
            fx, fy = self._distance_scaled_vector(row, center, relation, exponent)
            numerator += rx * fx + ry * fy
            denominator += fx * fx + fy * fy
        if abs(denominator) < 1e-12:
            return None
        scale = numerator / denominator
        predictions = []
        targets = []
        for row in valid:
            rx, ry = self._baseline_residual(row, baseline)
            fx, fy = self._distance_scaled_vector(row, center, relation, exponent)
            predictions.extend([scale * fx, scale * fy])
            targets.extend([rx, ry])

        exponent_label = self._exponent_label(exponent)
        vector_expression = (
            f'perpendicular(unit_{center_label}_vector)'
            if relation == 'perpendicular'
            else f'unit_{center_label}_vector'
        )
        role_prefix = 'local_residual' if is_local else 'residual'
        role = f'{role_prefix}_distance_scaled_{relation}_equation'
        equation = self._build_equation(
            key=f'raw_eq:residual_{prefix}_distance_{relation}_{exponent_label}',
            target='baseline_adjusted_delta_velocity',
            expression=f'k * {vector_expression} / separation^{exponent_label}',
            description=(
                'After subtracting repeated scalar drift, residual strength changes with separation.'
                if not is_local
                else 'After subtracting repeated scalar drift, local high-change residual strength changes with separation.'
            ),
            predictions=predictions,
            targets=targets,
            sample_count=len(rows),
            complexity=7,
            parameters={
                'k': scale,
                'center_x': center[0],
                'center_y': center[1],
                'distance_exponent': exponent,
                'baseline_x_intercept': baseline['x'][0],
                'baseline_x_velocity_scale': baseline['x'][1],
                'baseline_y_intercept': baseline['y'][0],
                'baseline_y_velocity_scale': baseline['y'][1],
            },
            role=role,
        )
        if equation is None:
            return None
        reference_mse = self._residual_vector_reference_mse(
            rows,
            baseline=baseline,
            center=center,
            relation=relation,
        )
        if reference_mse is not None and reference_mse > 1e-12:
            improvement = 1.0 - equation.mse / reference_mse
        elif reference_equation is not None and reference_equation.mse > 1e-12:
            reference_mse = reference_equation.mse
            improvement = 1.0 - equation.mse / reference_mse
        else:
            improvement = 0.0
        equation.parameters['reference_mse'] = reference_mse
        equation.parameters['distance_mse_improvement'] = improvement
        if not self._distance_scaled_residual_passes_quality_gate(equation):
            return None
        return equation

    def _residual_vector_reference_mse(
        self,
        rows: list[dict],
        baseline: dict[str, tuple[float, float]],
        center: tuple[float, float],
        relation: str,
    ) -> float | None:
        train, valid = self._split_rows(rows)
        if not train or not valid:
            return None
        numerator = 0.0
        denominator = 0.0
        for row in train:
            rx, ry = self._baseline_residual(row, baseline)
            if relation == 'perpendicular':
                fx, fy = self._perpendicular_to_center(row, center)
            else:
                fx, fy = self._unit_to_center(row, center)
            numerator += rx * fx + ry * fy
            denominator += fx * fx + fy * fy
        if abs(denominator) < 1e-12:
            return None
        scale = numerator / denominator
        predictions = []
        targets = []
        for row in valid:
            rx, ry = self._baseline_residual(row, baseline)
            if relation == 'perpendicular':
                fx, fy = self._perpendicular_to_center(row, center)
            else:
                fx, fy = self._unit_to_center(row, center)
            predictions.extend([scale * fx, scale * fy])
            targets.extend([rx, ry])
        return self._mse(predictions, targets)

    def _distance_scaled_reference_mse(
        self,
        rows: list[dict],
        baseline: dict[str, tuple[float, float]],
        center: tuple[float, float],
        relation: str,
    ) -> float | None:
        best_mse = None
        for exponent in self._candidate_distance_exponents():
            mse = self._distance_scaled_reference_mse_for_exponent(
                rows,
                baseline=baseline,
                center=center,
                relation=relation,
                exponent=exponent,
            )
            if mse is None:
                continue
            if best_mse is None or mse < best_mse:
                best_mse = mse
        return best_mse

    def _distance_scaled_reference_mse_for_exponent(
        self,
        rows: list[dict],
        baseline: dict[str, tuple[float, float]],
        center: tuple[float, float],
        relation: str,
        exponent: float,
    ) -> float | None:
        train, valid = self._split_rows(rows)
        if not train or not valid:
            return None
        numerator = 0.0
        denominator = 0.0
        for row in train:
            rx, ry = self._baseline_residual(row, baseline)
            fx, fy = self._distance_scaled_vector(row, center, relation, exponent)
            numerator += rx * fx + ry * fy
            denominator += fx * fx + fy * fy
        if abs(denominator) < 1e-12:
            return None
        scale = numerator / denominator
        predictions = []
        targets = []
        for row in valid:
            rx, ry = self._baseline_residual(row, baseline)
            fx, fy = self._distance_scaled_vector(row, center, relation, exponent)
            predictions.extend([scale * fx, scale * fy])
            targets.extend([rx, ry])
        return self._mse(predictions, targets)

    def _cutoff_reference_mse(
        self,
        rows: list[dict],
        baseline: dict[str, tuple[float, float]],
        center: tuple[float, float],
        relation: str,
        cutoff_radius: float,
    ) -> float | None:
        train, valid = self._split_rows(rows)
        if not train or not valid:
            return None
        numerator = 0.0
        denominator = 0.0
        for row in train:
            rx, ry = self._baseline_residual(row, baseline)
            fx, fy = self._cutoff_window_vector(row, center, relation, cutoff_radius)
            numerator += rx * fx + ry * fy
            denominator += fx * fx + fy * fy
        if abs(denominator) < 1e-12:
            return None
        scale = numerator / denominator
        predictions = []
        targets = []
        for row in valid:
            rx, ry = self._baseline_residual(row, baseline)
            fx, fy = self._cutoff_window_vector(row, center, relation, cutoff_radius)
            predictions.extend([scale * fx, scale * fy])
            targets.extend([rx, ry])
        return self._mse(predictions, targets)

    def _score_residual_vector_scale(
        self,
        rows: list[dict],
        baseline: dict[str, tuple[float, float]],
        center: tuple[float, float],
        key: str,
        target: str,
        expression: str,
        description: str,
        vector_feature: Callable[[dict], tuple[float, float]],
        complexity: int,
        role: str,
    ) -> PrimitiveEquation | None:
        train, valid = self._split_rows(rows)
        if not train or not valid:
            return None
        numerator = 0.0
        denominator = 0.0
        for row in train:
            rx, ry = self._baseline_residual(row, baseline)
            fx, fy = vector_feature(row)
            numerator += rx * fx + ry * fy
            denominator += fx * fx + fy * fy
        if abs(denominator) < 1e-12:
            return None
        scale = numerator / denominator
        predictions = []
        targets = []
        for row in valid:
            rx, ry = self._baseline_residual(row, baseline)
            fx, fy = vector_feature(row)
            predictions.extend([scale * fx, scale * fy])
            targets.extend([rx, ry])
        parameters = {
            'k': scale,
            'center_x': center[0],
            'center_y': center[1],
            'baseline_x_intercept': baseline['x'][0],
            'baseline_x_velocity_scale': baseline['x'][1],
            'baseline_y_intercept': baseline['y'][0],
            'baseline_y_velocity_scale': baseline['y'][1],
        }
        equation = self._build_equation(
            key=key,
            target=target,
            expression=expression,
            description=description,
            predictions=predictions,
            targets=targets,
            sample_count=len(rows),
            complexity=complexity,
            parameters=parameters,
            role=role,
        )
        if equation is None:
            return None
        if equation.baseline_mse < 1e-8:
            equation.parameters['directional_alignment'] = 0.0
            equation.score = 0.0
            return equation
        if role in {'local_residual_direction_equation', 'local_residual_perpendicular_equation'}:
            equation.parameters['mse_improvement'] = self._equation_mse_improvement(equation)
            if not self._local_residual_passes_quality_gate(equation):
                equation.parameters['directional_alignment'] = 0.0
                equation.score = 0.0
                return equation
        alignment = self._directional_alignment_score(valid, baseline, vector_feature, scale)
        equation.parameters['directional_alignment'] = alignment
        alignment_score = max(0.0, alignment - 0.01 * max(0, complexity - 1))
        equation.score = max(equation.score, min(1.0, alignment_score))
        return equation

    def _equation_mse_improvement(self, equation: PrimitiveEquation) -> float:
        return 1.0 - (equation.mse / max(equation.baseline_mse, 1e-12))

    def _local_residual_passes_quality_gate(self, equation: PrimitiveEquation) -> bool:
        return (
            equation.baseline_mse >= 1e-3
            and self._equation_mse_improvement(equation) >= 0.005
        )

    def _distance_scaled_residual_passes_quality_gate(self, equation: PrimitiveEquation) -> bool:
        return (
            equation.baseline_mse >= 1e-3
            and equation.parameters.get('distance_mse_improvement', 0.0) >= 0.04
            and self._equation_mse_improvement(equation) >= 0.08
        )

    def _cutoff_window_passes_quality_gate(self, equation: PrimitiveEquation) -> bool:
        return (
            equation.baseline_mse >= 1e-3
            and equation.parameters.get('cutoff_mse_improvement', 0.0) >= 0.04
            and equation.parameters.get('cutoff_vs_smooth_improvement', 0.0) >= 0.03
            and equation.parameters.get('mse_improvement', 0.0) >= 0.08
            and equation.parameters.get('outside_residual_fraction', 1.0) <= 0.16
            and equation.parameters.get('inside_projection_cv', 1.0) <= 0.35
            and equation.parameters.get('inside_valid_count', 0) >= 3
            and equation.parameters.get('outside_valid_count', 0) >= 3
        )

    def _tapered_distance_passes_quality_gate(self, equation: PrimitiveEquation) -> bool:
        return (
            equation.baseline_mse >= 1e-3
            and equation.parameters.get('tapered_mse_improvement', 0.0) >= 0.04
            and equation.parameters.get('tapered_vs_smooth_improvement', 0.0) >= 0.03
            and equation.parameters.get('tapered_vs_cutoff_improvement', 0.0) >= 0.03
            and equation.parameters.get('mse_improvement', 0.0) >= 0.08
            and equation.parameters.get('inside_valid_count', 0) >= 3
            and equation.parameters.get('outside_valid_count', 0) >= 3
            and equation.parameters.get('mid_valid_count', 0) >= 2
        )

    def _temporal_residual_equations(self, rows: list[dict]) -> list[PrimitiveEquation]:
        rows = self._residual_candidate_rows(rows)
        if len(rows) < max(self.min_samples, 40):
            return []
        train, valid = self._split_rows(rows)
        if not train or not valid:
            return []
        baseline = self._fit_axis_baseline(train)
        if baseline is None:
            return []

        equations = []
        for axis in ('x', 'y'):
            target = 'dvx' if axis == 'x' else 'dvy'
            velocity = 'vx' if axis == 'x' else 'vy'
            best: PrimitiveEquation | None = None
            for period in self._candidate_period_steps(rows):
                equation = self._score_periodic_residual_axis(
                    rows,
                    baseline=baseline,
                    axis=axis,
                    target=target,
                    velocity=velocity,
                    period_steps=period,
                )
                if equation is None:
                    continue
                if best is None or equation.score > best.score:
                    best = equation
            if best is not None:
                equations.append(best)
        return equations

    def _operator_feedback_equations(self, rows: list[dict]) -> list[PrimitiveEquation]:
        if not self.generated_operator_bank:
            return []
        rows = self._residual_candidate_rows(rows)
        rows = self._bounded_operator_feedback_rows(rows)
        if len(rows) < self.min_samples:
            return []
        train, valid = self._split_rows(rows)
        if not train or not valid:
            return []
        baseline = self._fit_axis_baseline(train)
        if baseline is None:
            return []

        equations = []
        for operator in self._ranked_operator_feedback_items():
            kind = operator.get('operator_kind')
            if kind == 'inverse_separation_power':
                equations.extend(self._operator_feedback_distance_equations(
                    rows,
                    baseline=baseline,
                    operator=operator,
                ))
            elif kind == 'localized_cutoff_window':
                equations.extend(self._operator_feedback_cutoff_equations(
                    rows,
                    baseline=baseline,
                    operator=operator,
                ))
            elif kind == 'localized_tapered_power':
                equations.extend(self._operator_feedback_tapered_distance_equations(
                    rows,
                    baseline=baseline,
                    operator=operator,
                ))
            elif kind == 'phase_basis':
                equations.extend(self._operator_feedback_periodic_equations(
                    rows,
                    operator=operator,
                ))
        return equations

    def _ranked_operator_feedback_items(self) -> list[dict]:
        scorable_kinds = {
            'inverse_separation_power',
            'localized_cutoff_window',
            'localized_tapered_power',
            'phase_basis',
        }
        return sorted(
            [
                item for item in self.generated_operator_bank.values()
                if item.get('operator_kind') in scorable_kinds
            ],
            key=lambda item: (
                float(item.get('usefulness', 0.0) or 0.0),
                str(item.get('key', '')),
            ),
            reverse=True,
        )[:self.max_operator_feedback_operators]

    def _bounded_operator_feedback_rows(self, rows: list[dict]) -> list[dict]:
        max_rows = self.max_operator_feedback_rows
        if len(rows) <= max_rows:
            return list(rows)
        if max_rows <= 1:
            return [rows[0]]
        stride = (len(rows) - 1) / (max_rows - 1)
        return [
            rows[min(len(rows) - 1, int(round(index * stride)))]
            for index in range(max_rows)
        ]

    def _operator_feedback_distance_equations(
        self,
        rows: list[dict],
        baseline: dict[str, tuple[float, float]],
        operator: dict,
    ) -> list[PrimitiveEquation]:
        parameters = operator.get('parameters', {})
        center_x = parameters.get('center_x')
        center_y = parameters.get('center_y')
        exponent = parameters.get('distance_exponent')
        relation = parameters.get('relation', 'direction')
        if not isinstance(center_x, (int, float)) or not isinstance(center_y, (int, float)):
            return []
        if not isinstance(exponent, (int, float)):
            return []
        if relation not in {'direction', 'perpendicular'}:
            return []

        best: PrimitiveEquation | None = None
        for candidate_exponent in self._generated_distance_exponent_variants(float(exponent)):
            equation = self._score_generated_operator_distance_vector_scale(
                rows,
                baseline=baseline,
                center=(float(center_x), float(center_y)),
                relation=relation,
                exponent=candidate_exponent,
                operator=operator,
            )
            if equation is None:
                continue
            if best is None or equation.score > best.score:
                best = equation
        return [best] if best is not None else []

    def _score_generated_operator_distance_vector_scale(
        self,
        rows: list[dict],
        baseline: dict[str, tuple[float, float]],
        center: tuple[float, float],
        relation: str,
        exponent: float,
        operator: dict,
    ) -> PrimitiveEquation | None:
        train, valid = self._split_rows(rows)
        if not train or not valid:
            return None
        numerator = 0.0
        denominator = 0.0
        for row in train:
            rx, ry = self._baseline_residual(row, baseline)
            fx, fy = self._distance_scaled_vector(row, center, relation, exponent)
            numerator += rx * fx + ry * fy
            denominator += fx * fx + fy * fy
        if abs(denominator) < 1e-12:
            return None
        scale = numerator / denominator
        predictions = []
        targets = []
        for row in valid:
            rx, ry = self._baseline_residual(row, baseline)
            fx, fy = self._distance_scaled_vector(row, center, relation, exponent)
            predictions.extend([scale * fx, scale * fy])
            targets.extend([rx, ry])

        exponent_label = self._exponent_label(exponent)
        vector_expression = (
            'perpendicular(unit_generated_center_vector)'
            if relation == 'perpendicular'
            else 'unit_generated_center_vector'
        )
        equation = self._build_equation(
            key=f"raw_eq:generated_operator_{operator.get('key', 'unknown')}_{exponent_label}",
            target='baseline_adjusted_delta_velocity',
            expression=f'k * {vector_expression} / separation^{exponent_label}',
            description='A generated operator from an earlier residual theory predicts residual strength.',
            predictions=predictions,
            targets=targets,
            sample_count=len(rows),
            complexity=8,
            parameters={
                'k': scale,
                'center_x': center[0],
                'center_y': center[1],
                'distance_exponent': exponent,
                'operator_key': operator.get('key'),
                'operator_kind': operator.get('operator_kind'),
                'relation': relation,
                'baseline_x_intercept': baseline['x'][0],
                'baseline_x_velocity_scale': baseline['x'][1],
                'baseline_y_intercept': baseline['y'][0],
                'baseline_y_velocity_scale': baseline['y'][1],
            },
            role=f'generated_operator_distance_scaled_{relation}_equation',
        )
        if equation is None:
            return None
        reference_mse = self._residual_vector_reference_mse(
            rows,
            baseline=baseline,
            center=center,
            relation=relation,
        )
        if reference_mse is None or reference_mse <= 1e-12:
            return None
        equation.parameters['reference_mse'] = reference_mse
        equation.parameters['distance_mse_improvement'] = 1.0 - equation.mse / reference_mse
        if not self._distance_scaled_residual_passes_quality_gate(equation):
            return None
        return equation

    def _operator_feedback_cutoff_equations(
        self,
        rows: list[dict],
        baseline: dict[str, tuple[float, float]],
        operator: dict,
    ) -> list[PrimitiveEquation]:
        parameters = operator.get('parameters', {})
        center_x = parameters.get('center_x')
        center_y = parameters.get('center_y')
        cutoff_radius = parameters.get('cutoff_radius')
        relation = parameters.get('relation', 'direction')
        if not isinstance(center_x, (int, float)) or not isinstance(center_y, (int, float)):
            return []
        if cutoff_radius is not None and not isinstance(cutoff_radius, (int, float)):
            return []
        if relation not in {'direction', 'perpendicular'}:
            return []

        best: PrimitiveEquation | None = None
        for candidate_center in self._generated_center_variants((float(center_x), float(center_y))):
            for candidate_radius in self._generated_cutoff_radius_variants(
                cutoff_radius,
                rows=rows,
                center=candidate_center,
            ):
                equation = self._score_generated_operator_cutoff_vector_scale(
                    rows,
                    baseline=baseline,
                    center=candidate_center,
                    relation=relation,
                    cutoff_radius=candidate_radius,
                    operator=operator,
                )
                if equation is None:
                    continue
                if best is None or equation.score > best.score:
                    best = equation
        return [best] if best is not None else []

    def _score_generated_operator_cutoff_vector_scale(
        self,
        rows: list[dict],
        baseline: dict[str, tuple[float, float]],
        center: tuple[float, float],
        relation: str,
        cutoff_radius: float,
        operator: dict,
    ) -> PrimitiveEquation | None:
        rows = self._local_contrast_rows(rows, center, cutoff_radius)
        if len(rows) < self.min_samples:
            return None
        train, valid = self._split_rows(rows)
        if not train or not valid:
            return None
        min_bucket = max(3, min(8, self.min_samples // 4))
        train_inside, train_outside = self._cutoff_bucket_counts(train, center, cutoff_radius)
        valid_inside, valid_outside = self._cutoff_bucket_counts(valid, center, cutoff_radius)
        if (
            train_inside < min_bucket
            or train_outside < min_bucket
            or valid_inside < min_bucket
            or valid_outside < min_bucket
        ):
            return None

        numerator = 0.0
        denominator = 0.0
        for row in train:
            rx, ry = self._baseline_residual(row, baseline)
            fx, fy = self._cutoff_window_vector(row, center, relation, cutoff_radius)
            numerator += rx * fx + ry * fy
            denominator += fx * fx + fy * fy
        if abs(denominator) < 1e-12:
            return None
        scale = numerator / denominator
        predictions = []
        targets = []
        for row in valid:
            rx, ry = self._baseline_residual(row, baseline)
            fx, fy = self._cutoff_window_vector(row, center, relation, cutoff_radius)
            predictions.extend([scale * fx, scale * fy])
            targets.extend([rx, ry])

        radius_label = self._scalar_label(cutoff_radius)
        vector_expression = (
            'perpendicular(unit_generated_center_vector)'
            if relation == 'perpendicular'
            else 'unit_generated_center_vector'
        )
        equation = self._build_equation(
            key=f"raw_eq:generated_operator_{operator.get('key', 'unknown')}_cutoff_{radius_label}",
            target='baseline_adjusted_delta_velocity',
            expression=(
                f'k * inside(separation <= {radius_label}) * {vector_expression}'
            ),
            description='A generated cutoff-window operator predicts a residual only inside an inferred region.',
            predictions=predictions,
            targets=targets,
            sample_count=len(rows),
            complexity=8,
            parameters={
                'k': scale,
                'center_x': center[0],
                'center_y': center[1],
                'cutoff_radius': cutoff_radius,
                'operator_key': operator.get('key'),
                'operator_kind': operator.get('operator_kind'),
                'relation': relation,
                'inside_train_count': train_inside,
                'outside_train_count': train_outside,
                'inside_valid_count': valid_inside,
                'outside_valid_count': valid_outside,
                'inside_fraction': (train_inside + valid_inside) / max(len(rows), 1),
                'baseline_x_intercept': baseline['x'][0],
                'baseline_x_velocity_scale': baseline['x'][1],
                'baseline_y_intercept': baseline['y'][0],
                'baseline_y_velocity_scale': baseline['y'][1],
            },
            role=f'generated_operator_cutoff_{relation}_equation',
        )
        if equation is None:
            return None
        reference_mse = self._residual_vector_reference_mse(
            rows,
            baseline=baseline,
            center=center,
            relation=relation,
        )
        if reference_mse is None or reference_mse <= 1e-12:
            return None
        equation.parameters['reference_mse'] = reference_mse
        equation.parameters['cutoff_mse_improvement'] = 1.0 - equation.mse / reference_mse
        smooth_reference_mse = self._distance_scaled_reference_mse(
            rows,
            baseline=baseline,
            center=center,
            relation=relation,
        )
        if smooth_reference_mse is None or smooth_reference_mse <= 1e-12:
            return None
        equation.parameters['smooth_reference_mse'] = smooth_reference_mse
        equation.parameters['cutoff_vs_smooth_improvement'] = (
            1.0 - equation.mse / smooth_reference_mse
        )
        equation.parameters['mse_improvement'] = self._equation_mse_improvement(equation)
        inside_energy = 0.0
        outside_energy = 0.0
        inside_projection_values = []
        for row in valid:
            rx, ry = self._baseline_residual(row, baseline)
            energy = rx * rx + ry * ry
            if self._distance_to_center(row, center) <= cutoff_radius:
                inside_energy += energy
                fx, fy = self._cutoff_window_vector(row, center, relation, cutoff_radius)
                inside_projection_values.append(rx * fx + ry * fy)
            else:
                outside_energy += energy
        inside_mean = inside_energy / max(valid_inside, 1)
        outside_mean = outside_energy / max(valid_outside, 1)
        equation.parameters['outside_residual_fraction'] = (
            outside_mean / max(inside_mean, 1e-12)
        )
        projection_mean = sum(inside_projection_values) / max(len(inside_projection_values), 1)
        projection_variance = sum(
            (value - projection_mean) ** 2 for value in inside_projection_values
        ) / max(len(inside_projection_values), 1)
        projection_stdev = math.sqrt(projection_variance)
        equation.parameters['inside_projection_cv'] = (
            projection_stdev / max(abs(projection_mean), 1e-12)
        )
        if not self._cutoff_window_passes_quality_gate(equation):
            return None
        alignment = self._directional_alignment_score(
            valid,
            baseline,
            lambda row, center=center, relation=relation, cutoff_radius=cutoff_radius: (
                self._cutoff_window_vector(row, center, relation, cutoff_radius)
            ),
            scale,
        )
        equation.parameters['directional_alignment'] = alignment
        equation.score = max(equation.score, min(1.0, alignment - 0.06))
        return equation

    def _operator_feedback_tapered_distance_equations(
        self,
        rows: list[dict],
        baseline: dict[str, tuple[float, float]],
        operator: dict,
    ) -> list[PrimitiveEquation]:
        parameters = operator.get('parameters', {})
        center_x = parameters.get('center_x')
        center_y = parameters.get('center_y')
        cutoff_radius = parameters.get('cutoff_radius')
        exponent = parameters.get('distance_exponent')
        relation = parameters.get('relation', 'direction')
        if not isinstance(center_x, (int, float)) or not isinstance(center_y, (int, float)):
            return []
        if cutoff_radius is not None and not isinstance(cutoff_radius, (int, float)):
            return []
        if exponent is not None and not isinstance(exponent, (int, float)):
            return []
        if relation not in {'direction', 'perpendicular'}:
            return []

        exponent_candidates = (
            self._generated_distance_exponent_variants(float(exponent))
            if isinstance(exponent, (int, float))
            else self._candidate_distance_exponents()
        )
        best: PrimitiveEquation | None = None
        for candidate_center in self._generated_center_variants((float(center_x), float(center_y))):
            radius_candidates = self._generated_cutoff_radius_variants(
                cutoff_radius,
                rows=rows,
                center=candidate_center,
            )
            for candidate_radius in radius_candidates:
                for candidate_exponent in exponent_candidates:
                    equation = self._score_generated_operator_tapered_distance_vector_scale(
                        rows,
                        baseline=baseline,
                        center=candidate_center,
                        relation=relation,
                        cutoff_radius=candidate_radius,
                        exponent=candidate_exponent,
                        operator=operator,
                    )
                    if equation is None:
                        continue
                    if best is None or equation.score > best.score:
                        best = equation
        return [best] if best is not None else []

    def _score_generated_operator_tapered_distance_vector_scale(
        self,
        rows: list[dict],
        baseline: dict[str, tuple[float, float]],
        center: tuple[float, float],
        relation: str,
        cutoff_radius: float,
        exponent: float,
        operator: dict,
    ) -> PrimitiveEquation | None:
        rows = self._local_contrast_rows(rows, center, cutoff_radius)
        if len(rows) < self.min_samples:
            return None
        train, valid = self._split_rows(rows)
        if not train or not valid:
            return None
        min_bucket = max(3, min(8, self.min_samples // 4))
        train_inside, train_outside = self._cutoff_bucket_counts(train, center, cutoff_radius)
        valid_inside, valid_outside = self._cutoff_bucket_counts(valid, center, cutoff_radius)
        mid_valid = self._taper_mid_count(valid, center, cutoff_radius)
        if (
            train_inside < min_bucket
            or train_outside < min_bucket
            or valid_inside < min_bucket
            or valid_outside < min_bucket
            or mid_valid < 2
        ):
            return None

        numerator = 0.0
        denominator = 0.0
        for row in train:
            rx, ry = self._baseline_residual(row, baseline)
            fx, fy = self._tapered_distance_vector(
                row,
                center,
                relation,
                cutoff_radius,
                exponent,
            )
            numerator += rx * fx + ry * fy
            denominator += fx * fx + fy * fy
        if abs(denominator) < 1e-12:
            return None
        scale = numerator / denominator
        predictions = []
        targets = []
        for row in valid:
            rx, ry = self._baseline_residual(row, baseline)
            fx, fy = self._tapered_distance_vector(
                row,
                center,
                relation,
                cutoff_radius,
                exponent,
            )
            predictions.extend([scale * fx, scale * fy])
            targets.extend([rx, ry])

        radius_label = self._scalar_label(cutoff_radius)
        exponent_label = self._exponent_label(exponent)
        vector_expression = (
            'perpendicular(unit_generated_center_vector)'
            if relation == 'perpendicular'
            else 'unit_generated_center_vector'
        )
        equation = self._build_equation(
            key=(
                f"raw_eq:generated_operator_{operator.get('key', 'unknown')}"
                f"_taper_{radius_label}_{exponent_label}"
            ),
            target='baseline_adjusted_delta_velocity',
            expression=(
                f'k * taper(separation, {radius_label}) * '
                f'{vector_expression} / separation^{exponent_label}'
            ),
            description='A generated local taper operator predicts residual strength fading to zero at an inferred boundary.',
            predictions=predictions,
            targets=targets,
            sample_count=len(rows),
            complexity=10,
            parameters={
                'k': scale,
                'center_x': center[0],
                'center_y': center[1],
                'cutoff_radius': cutoff_radius,
                'distance_exponent': exponent,
                'operator_key': operator.get('key'),
                'operator_kind': operator.get('operator_kind'),
                'relation': relation,
                'inside_train_count': train_inside,
                'outside_train_count': train_outside,
                'inside_valid_count': valid_inside,
                'outside_valid_count': valid_outside,
                'mid_valid_count': mid_valid,
                'inside_fraction': (train_inside + valid_inside) / max(len(rows), 1),
                'baseline_x_intercept': baseline['x'][0],
                'baseline_x_velocity_scale': baseline['x'][1],
                'baseline_y_intercept': baseline['y'][0],
                'baseline_y_velocity_scale': baseline['y'][1],
            },
            role=f'generated_operator_tapered_distance_{relation}_equation',
        )
        if equation is None:
            return None
        direction_reference_mse = self._residual_vector_reference_mse(
            rows,
            baseline=baseline,
            center=center,
            relation=relation,
        )
        smooth_reference_mse = self._distance_scaled_reference_mse(
            rows,
            baseline=baseline,
            center=center,
            relation=relation,
        )
        cutoff_reference_mse = self._cutoff_reference_mse(
            rows,
            baseline=baseline,
            center=center,
            relation=relation,
            cutoff_radius=cutoff_radius,
        )
        if (
            direction_reference_mse is None
            or smooth_reference_mse is None
            or cutoff_reference_mse is None
            or direction_reference_mse <= 1e-12
            or smooth_reference_mse <= 1e-12
            or cutoff_reference_mse <= 1e-12
        ):
            return None
        equation.parameters['reference_mse'] = direction_reference_mse
        equation.parameters['smooth_reference_mse'] = smooth_reference_mse
        equation.parameters['cutoff_reference_mse'] = cutoff_reference_mse
        equation.parameters['tapered_mse_improvement'] = (
            1.0 - equation.mse / direction_reference_mse
        )
        equation.parameters['tapered_vs_smooth_improvement'] = (
            1.0 - equation.mse / smooth_reference_mse
        )
        equation.parameters['tapered_vs_cutoff_improvement'] = (
            1.0 - equation.mse / cutoff_reference_mse
        )
        equation.parameters['mse_improvement'] = self._equation_mse_improvement(equation)
        if not self._tapered_distance_passes_quality_gate(equation):
            return None
        alignment = self._directional_alignment_score(
            valid,
            baseline,
            lambda row, center=center, relation=relation, cutoff_radius=cutoff_radius, exponent=exponent: (
                self._tapered_distance_vector(row, center, relation, cutoff_radius, exponent)
            ),
            scale,
        )
        equation.parameters['directional_alignment'] = alignment
        equation.score = max(equation.score, min(1.0, alignment - 0.08))
        return equation

    def _operator_feedback_periodic_equations(
        self,
        rows: list[dict],
        operator: dict,
    ) -> list[PrimitiveEquation]:
        parameters = operator.get('parameters', {})
        period = parameters.get('period_steps')
        if not isinstance(period, (int, float)):
            return []
        candidates = []
        baseline_train, _ = self._split_rows(rows)
        baseline = self._fit_axis_baseline(baseline_train)
        if baseline is None:
            return []
        for candidate_period in self._generated_period_variants(int(round(period))):
            for axis in ('x', 'y'):
                equation = self._score_periodic_residual_axis(
                    rows,
                    baseline=baseline,
                    axis=axis,
                    target='dvx' if axis == 'x' else 'dvy',
                    velocity='vx' if axis == 'x' else 'vy',
                    period_steps=candidate_period,
                )
                if equation is None:
                    continue
                equation.key = (
                    f"raw_eq:generated_operator_{operator.get('key', 'unknown')}_{axis}_{candidate_period}"
                )
                equation.description = 'A generated phase-basis operator predicts repeating residual change.'
                equation.role = 'generated_operator_periodic_equation'
                equation.complexity = max(equation.complexity, 7)
                equation.parameters['operator_key'] = operator.get('key')
                equation.parameters['operator_kind'] = operator.get('operator_kind')
                candidates.append(equation)
        if not candidates:
            return []
        candidates.sort(key=lambda equation: equation.score, reverse=True)
        return candidates[:2]

    def _refresh_generated_operator_bank(self, step: int):
        if not self.equations:
            return
        report = self.discovery_loop.build_report(
            self.discovered_equations()[:16],
            step=step,
        )
        for operator in report.operator_proposals:
            item = operator.to_dict()
            if item.get('usefulness', 0.0) < 0.25:
                continue
            self.generated_operator_bank[item['key']] = item
        if len(self.generated_operator_bank) > 24:
            ranked = sorted(
                self.generated_operator_bank.values(),
                key=lambda item: item.get('usefulness', 0.0),
                reverse=True,
            )
            self.generated_operator_bank = {
                item['key']: item for item in ranked[:24]
            }
            self._refresh_operator_prior_count()

    def _generated_distance_exponent_variants(self, exponent: float) -> list[float]:
        candidates = {
            exponent,
            exponent - 0.5,
            exponent - 0.25,
            exponent + 0.25,
            exponent + 0.5,
        }
        return sorted(
            round(value, 2)
            for value in candidates
            if 0.25 <= value <= 4.0
        )

    def _generated_cutoff_radius_variants(
        self,
        radius: float | None,
        rows: list[dict],
        center: tuple[float, float],
    ) -> list[float]:
        candidates = set()
        if isinstance(radius, (int, float)) and radius > 0.0:
            candidates.update({
                float(radius) * 0.65,
                float(radius) * 0.8,
                float(radius),
                float(radius) * 1.2,
                float(radius) * 1.4,
            })
        distances = sorted(
            self._distance_to_center(row, center)
            for row in rows
            if self._distance_to_center(row, center) > 1e-9
        )
        if distances:
            for fraction in (0.3, 0.4, 0.5, 0.6, 0.7):
                index = min(len(distances) - 1, max(0, int(len(distances) * fraction)))
                candidates.add(distances[index])
        return sorted(
            round(value, 3)
            for value in candidates
            if 0.5 <= value <= 50.0
        )

    def _generated_center_variants(
        self,
        center: tuple[float, float],
    ) -> list[tuple[float, float]]:
        cx, cy = center
        offsets = [
            (0.0, 0.0),
            (0.0, 2.25),
            (0.0, -2.25),
            (2.25, 0.0),
            (-2.25, 0.0),
        ]
        return [
            (round(cx + dx, 3), round(cy + dy, 3))
            for dx, dy in offsets
        ]

    def _generated_period_variants(self, period: int) -> list[int]:
        candidates = {
            period,
            period - 4,
            period - 2,
            period + 2,
            period + 4,
        }
        return sorted(value for value in candidates if value >= 8)

    def _score_periodic_residual_axis(
        self,
        rows: list[dict],
        baseline: dict[str, tuple[float, float]],
        axis: str,
        target: str,
        velocity: str,
        period_steps: int,
    ) -> PrimitiveEquation | None:
        train, valid = self._split_rows(rows)
        if not train or not valid:
            return None
        coeffs = self._fit_periodic_coefficients(
            train,
            baseline=baseline,
            axis=axis,
            period_steps=period_steps,
        )
        if coeffs is None:
            return None
        sine_weight, cosine_weight = coeffs
        predictions = []
        targets = []
        for row in valid:
            residual = self._axis_residual(row, baseline, axis)
            sine, cosine = self._periodic_features(row['step'], period_steps)
            predictions.append(sine_weight * sine + cosine_weight * cosine)
            targets.append(residual)
        equation = self._build_equation(
            key=f'raw_eq:residual_periodic_{axis}_{period_steps}',
            target=f'baseline_adjusted_delta_v{axis}',
            expression=f'a * sin(step/{period_steps}) + b * cos(step/{period_steps})',
            description='After subtracting repeated scalar drift, a repeating step template predicts signed channel change.',
            predictions=predictions,
            targets=targets,
            sample_count=len(rows),
            complexity=6,
            parameters={
                'period_steps': period_steps,
                'sine_weight': sine_weight,
                'cosine_weight': cosine_weight,
                'amplitude': math.sqrt(sine_weight * sine_weight + cosine_weight * cosine_weight),
                'baseline_intercept': baseline[axis][0],
                'baseline_velocity_scale': baseline[axis][1],
            },
            role='residual_periodic_equation',
        )
        if equation is None:
            return None
        if equation.baseline_mse < 1e-8:
            equation.score = 0.0
        return equation

    def _fit_periodic_coefficients(
        self,
        rows: list[dict],
        baseline: dict[str, tuple[float, float]],
        axis: str,
        period_steps: int,
    ) -> tuple[float, float] | None:
        ss = 0.0
        sc = 0.0
        cc = 0.0
        sy = 0.0
        cy = 0.0
        for row in rows:
            residual = self._axis_residual(row, baseline, axis)
            sine, cosine = self._periodic_features(row['step'], period_steps)
            ss += sine * sine
            sc += sine * cosine
            cc += cosine * cosine
            sy += sine * residual
            cy += cosine * residual
        det = ss * cc - sc * sc
        if abs(det) < 1e-12:
            return None
        sine_weight = (sy * cc - cy * sc) / det
        cosine_weight = (ss * cy - sc * sy) / det
        return (sine_weight, cosine_weight)

    def _candidate_period_steps(self, rows: list[dict]) -> list[int]:
        if not rows:
            return []
        min_step = min(int(row['step']) for row in rows)
        max_step = max(int(row['step']) for row in rows)
        span = max(1, max_step - min_step)
        lower = max(12, int(span / 12))
        upper = max(lower + 4, int(span / 2))
        seeded = {24, 32, 40, 48, 56, 60, 64, 72, 80, 88, 96, 104, 112, 120}
        sampled = set(range(lower, min(upper, 160) + 1, 4))
        candidates = sorted(period for period in seeded | sampled if 8 <= period <= max(160, span))
        return candidates[:40]

    def _periodic_features(self, step: float, period_steps: int) -> tuple[float, float]:
        angle = (2.0 * math.pi * float(step)) / max(float(period_steps), 1.0)
        return (math.sin(angle), math.cos(angle))

    def _axis_residual(
        self,
        row: dict,
        baseline: dict[str, tuple[float, float]],
        axis: str,
    ) -> float:
        rx, ry = self._baseline_residual(row, baseline)
        return rx if axis == 'x' else ry

    def _directional_alignment_score(
        self,
        rows: list[dict],
        baseline: dict[str, tuple[float, float]],
        vector_feature: Callable[[dict], tuple[float, float]],
        scale: float,
    ) -> float:
        weighted_alignment = 0.0
        total_weight = 0.0
        for row in rows:
            rx, ry = self._baseline_residual(row, baseline)
            fx, fy = vector_feature(row)
            px = scale * fx
            py = scale * fy
            predicted_mag = math.sqrt(px * px + py * py)
            target_mag = math.sqrt(rx * rx + ry * ry)
            if predicted_mag <= 1e-9 or target_mag <= 1e-9:
                continue
            weight = target_mag
            weighted_alignment += weight * (
                (px * rx + py * ry) / (predicted_mag * target_mag)
            )
            total_weight += weight
        if total_weight <= 1e-12:
            return 0.0
        return max(0.0, min(1.0, weighted_alignment / total_weight))

    def _build_equation(
        self,
        key: str,
        target: str,
        expression: str,
        description: str,
        predictions: list[float],
        targets: list[float],
        sample_count: int,
        complexity: int,
        parameters: dict,
        role: str,
    ) -> PrimitiveEquation | None:
        if not predictions or len(predictions) != len(targets):
            return None
        mse = self._mse(predictions, targets)
        baseline = self._baseline_mse(targets)
        fit = 1.0 if baseline <= 1e-12 and mse <= 1e-12 else max(0.0, 1.0 - (mse / max(baseline, 1e-12)))
        score = max(0.0, min(1.0, fit - 0.015 * max(0, complexity - 1)))
        return PrimitiveEquation(
            key=key,
            target=target,
            expression=expression,
            description=description,
            score=score,
            mse=mse,
            baseline_mse=baseline,
            complexity=complexity,
            sample_count=sample_count,
            parameters=parameters,
            role=role,
        )

    def _fit_axis_baseline(self, rows: list[dict]) -> dict[str, tuple[float, float]] | None:
        x_model = self._fit_intercept_and_velocity(rows, target='dvx', velocity='vx')
        y_model = self._fit_intercept_and_velocity(rows, target='dvy', velocity='vy')
        if x_model is None or y_model is None:
            return None
        return {'x': x_model, 'y': y_model}

    def _fit_intercept_and_velocity(
        self,
        rows: list[dict],
        target: str,
        velocity: str,
    ) -> tuple[float, float] | None:
        if not rows:
            return None
        n = float(len(rows))
        sum_v = sum(row[velocity] for row in rows)
        sum_y = sum(row[target] for row in rows)
        sum_vv = sum(row[velocity] * row[velocity] for row in rows)
        sum_vy = sum(row[velocity] * row[target] for row in rows)
        det = n * sum_vv - sum_v * sum_v
        if abs(det) < 1e-12:
            return (sum_y / n, 0.0)
        intercept = (sum_y * sum_vv - sum_v * sum_vy) / det
        velocity_scale = (n * sum_vy - sum_v * sum_y) / det
        return (intercept, velocity_scale)

    def _baseline_residual(
        self,
        row: dict,
        baseline: dict[str, tuple[float, float]],
    ) -> tuple[float, float]:
        x_intercept, x_velocity_scale = baseline['x']
        y_intercept, y_velocity_scale = baseline['y']
        rx = row['dvx'] - (x_intercept + x_velocity_scale * row['vx'])
        ry = row['dvy'] - (y_intercept + y_velocity_scale * row['vy'])
        return (rx, ry)

    def _residual_rows(
        self,
        rows: list[dict],
        baseline: dict[str, tuple[float, float]],
        significant_only: bool,
    ) -> list[tuple[dict, float, float, float]]:
        residuals = []
        for row in rows:
            rx, ry = self._baseline_residual(row, baseline)
            mag = math.sqrt(rx * rx + ry * ry)
            if significant_only and mag < 1e-6:
                continue
            residuals.append((row, rx, ry, mag))
        return residuals

    def _strong_residual_subset(
        self,
        residuals: list[tuple[dict, float, float, float]],
    ) -> list[tuple[dict, float, float, float]]:
        if len(residuals) < self.min_samples:
            return []
        magnitudes = sorted(mag for _, _, _, mag in residuals)
        cutoff = magnitudes[max(0, int(len(magnitudes) * 0.65) - 1)]
        strong = [
            item for item in residuals
            if item[3] >= cutoff and item[3] > 1e-8
        ]
        return strong if len(strong) >= self.min_samples else []

    def _infer_residual_center(
        self,
        residuals: list[tuple[dict, float, float, float]],
        mode: str,
    ) -> tuple[float, float] | None:
        if len(residuals) < self.min_samples:
            return None

        a00 = 0.0
        a01 = 0.0
        a11 = 0.0
        b0 = 0.0
        b1 = 0.0
        for row, rx, ry, mag in residuals:
            if mag < 1e-9:
                continue
            ux = rx / mag
            uy = ry / mag
            if mode == 'tangent':
                ux, uy = -uy, ux

            m00 = 1.0 - ux * ux
            m01 = -ux * uy
            m11 = 1.0 - uy * uy
            a00 += m00
            a01 += m01
            a11 += m11
            b0 += m00 * row['x'] + m01 * row['y']
            b1 += m01 * row['x'] + m11 * row['y']

        det = a00 * a11 - a01 * a01
        if abs(det) < 1e-9:
            return None

        cx = (b0 * a11 - b1 * a01) / det
        cy = (a00 * b1 - a01 * b0) / det
        min_x = min(row['x'] for row, _, _, _ in residuals) - 10.0
        max_x = max(row['x'] for row, _, _, _ in residuals) + 10.0
        min_y = min(row['y'] for row, _, _, _ in residuals) - 10.0
        max_y = max(row['y'] for row, _, _, _ in residuals) + 10.0
        if not (min_x <= cx <= max_x and min_y <= cy <= max_y):
            return None
        return (round(cx, 3), round(cy, 3))

    def _average_anchor_center(self, rows: list[dict]) -> tuple[float, float]:
        if not rows:
            return (0.0, 0.0)
        return (
            mean(row['anchor_x'] for row in rows),
            mean(row['anchor_y'] for row in rows),
        )

    def _unit_to_center(self, row: dict, center: tuple[float, float]) -> tuple[float, float]:
        cx, cy = center
        dx = cx - row['x']
        dy = cy - row['y']
        dist = math.sqrt(dx * dx + dy * dy)
        if dist <= 1e-9:
            return (0.0, 0.0)
        return (dx / dist, dy / dist)

    def _perpendicular_to_center(
        self,
        row: dict,
        center: tuple[float, float],
    ) -> tuple[float, float]:
        ux, uy = self._unit_to_center(row, center)
        return (-uy, ux)

    def _distance_scaled_vector(
        self,
        row: dict,
        center: tuple[float, float],
        relation: str,
        exponent: float,
    ) -> tuple[float, float]:
        cx, cy = center
        dx = cx - row['x']
        dy = cy - row['y']
        distance = max(math.sqrt(dx * dx + dy * dy), 0.5)
        if relation == 'perpendicular':
            ux, uy = self._perpendicular_to_center(row, center)
        else:
            ux, uy = self._unit_to_center(row, center)
        scale = 1.0 / (distance ** exponent)
        return (ux * scale, uy * scale)

    def _cutoff_window_vector(
        self,
        row: dict,
        center: tuple[float, float],
        relation: str,
        cutoff_radius: float,
    ) -> tuple[float, float]:
        if self._distance_to_center(row, center) > cutoff_radius:
            return (0.0, 0.0)
        if relation == 'perpendicular':
            return self._perpendicular_to_center(row, center)
        return self._unit_to_center(row, center)

    def _tapered_distance_vector(
        self,
        row: dict,
        center: tuple[float, float],
        relation: str,
        cutoff_radius: float,
        exponent: float,
    ) -> tuple[float, float]:
        distance = self._distance_to_center(row, center)
        if distance > cutoff_radius:
            return (0.0, 0.0)
        if relation == 'perpendicular':
            ux, uy = self._perpendicular_to_center(row, center)
        else:
            ux, uy = self._unit_to_center(row, center)
        safe_distance = max(distance, 0.5)
        taper = max(0.0, 1.0 - distance / max(cutoff_radius, 1e-9))
        scale = taper / (safe_distance ** exponent)
        return (ux * scale, uy * scale)

    def _cutoff_bucket_counts(
        self,
        rows: list[dict],
        center: tuple[float, float],
        cutoff_radius: float,
    ) -> tuple[int, int]:
        inside = sum(
            1 for row in rows
            if self._distance_to_center(row, center) <= cutoff_radius
        )
        return (inside, len(rows) - inside)

    def _taper_mid_count(
        self,
        rows: list[dict],
        center: tuple[float, float],
        cutoff_radius: float,
    ) -> int:
        lower = cutoff_radius * 0.25
        upper = cutoff_radius * 0.85
        return sum(
            1 for row in rows
            if lower <= self._distance_to_center(row, center) <= upper
        )

    def _local_contrast_rows(
        self,
        rows: list[dict],
        center: tuple[float, float],
        cutoff_radius: float,
    ) -> list[dict]:
        inside = [
            row for row in rows
            if self._distance_to_center(row, center) <= cutoff_radius
        ]
        if len(inside) < max(4, self.min_samples // 4):
            return []
        outside_shell = [
            row for row in rows
            if cutoff_radius < self._distance_to_center(row, center) <= cutoff_radius * 1.8
        ]
        if len(outside_shell) < max(4, self.min_samples // 4):
            outside_shell = [
                row for row in rows
                if self._distance_to_center(row, center) > cutoff_radius
            ]
        outside_shell.sort(
            key=lambda row: abs(self._distance_to_center(row, center) - cutoff_radius)
        )
        outside_limit = max(self.min_samples, len(inside))
        selected = inside + outside_shell[:outside_limit]
        selected.sort(key=lambda row: (int(row.get('step', 0)), row.get('id', 0)))
        return selected

    def _distance_to_center(self, row: dict, center: tuple[float, float]) -> float:
        cx, cy = center
        dx = cx - row['x']
        dy = cy - row['y']
        return math.sqrt(dx * dx + dy * dy)

    def _candidate_distance_exponents(self) -> list[float]:
        return [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    def _exponent_label(self, exponent: float) -> str:
        if abs(exponent - round(exponent)) <= 1e-9:
            return str(int(round(exponent)))
        return str(exponent).replace('.', '_')

    def _scalar_label(self, value: float) -> str:
        if abs(value - round(value)) <= 1e-9:
            return str(int(round(value)))
        return str(round(value, 3)).replace('.', '_')

    def _residual_candidate_rows(self, rows: list[dict]) -> list[dict]:
        quiet_rows = [
            row for row in rows
            if not row.get('action_target_push')
        ]
        if len(quiet_rows) < self.min_samples:
            quiet_rows = rows
        magnitudes = sorted(
            math.sqrt(row['dvx'] * row['dvx'] + row['dvy'] * row['dvy'])
            for row in quiet_rows
        )
        if len(magnitudes) < self.min_samples:
            return quiet_rows
        cutoff = magnitudes[min(len(magnitudes) - 1, int(len(magnitudes) * 0.95))]
        filtered = [
            row for row in quiet_rows
            if math.sqrt(row['dvx'] * row['dvx'] + row['dvy'] * row['dvy']) <= cutoff
        ]
        return filtered if len(filtered) >= self.min_samples else quiet_rows

    def _install_equations(self, step: int):
        installed = 0
        for equation in self._installable_equations():
            if installed >= self.max_installed:
                break
            if equation.score < self.install_score_threshold:
                continue
            if equation.rule_name is not None:
                continue
            rule = self.knowledge_base.add_hypothesis(
                conditions='primitive variables are observed',
                prediction=f"{equation.target} ~= {equation.expression}",
                feature_key=equation.target,
                step=step,
                properties={
                    'source': 'equation_workbench',
                    'hypothesis_type': 'starter_equation',
                    'equation_key': equation.key,
                    'expression': equation.expression,
                    'score': round(equation.score, 3),
                    'mse': round(equation.mse, 6),
                    'baseline_mse': round(equation.baseline_mse, 6),
                    'complexity': equation.complexity,
                    'sample_count': equation.sample_count,
                    'parameters': dict(equation.parameters),
                    'role': equation.role,
                },
            )
            evidence_for = max(10, int(equation.score * 24))
            rule.evidence_for = evidence_for
            rule.evidence_against = 0
            rule.confidence = min(1.0, equation.score)
            rule.status = RuleStatus.CONFIRMED
            rule.confirmed_at_step = step
            equation.rule_name = rule.internal_name
            installed += 1

    def _installable_equations(self) -> list[PrimitiveEquation]:
        equations = self.discovered_equations()
        equations.sort(
            key=lambda equation: (
                self._interesting_role_priority(equation.role),
                equation.score,
                -equation.complexity,
            ),
            reverse=True,
        )
        return equations

    def _categorized_equations(self, equations: list[dict], limit: int) -> dict[str, list[dict]]:
        categories = {
            'foundation_invariants': [],
            'state_transitions': [],
            'motion_updates': [],
            'direction_vectors': [],
            'residual_dynamics': [],
            'residual_strength': [],
            'residual_periodic': [],
            'derived_magnitudes': [],
            'other': [],
        }
        for equation in equations:
            category = self._category_for_role(equation.get('role', ''))
            categories.setdefault(category, []).append(equation)
        return {
            category: items[:limit]
            for category, items in categories.items()
            if items
        }

    def _interesting_equations(self, equations: list[dict], limit: int) -> list[dict]:
        boring_roles = {
            'invariant_equation',
            'action_extent_mapping',
            'state_transition_equation',
        }
        interesting = [
            equation for equation in equations
            if equation.get('role') not in boring_roles
        ]
        interesting.sort(key=self._interesting_sort_key, reverse=True)
        return interesting[:limit]

    def _interesting_sort_key(self, equation: dict) -> tuple[float, float, int]:
        score = float(equation.get('score', 0.0))
        role = equation.get('role', '')
        priority = self._interesting_role_priority(role)
        if role in {'local_residual_direction_equation', 'local_residual_perpendicular_equation'}:
            minimum_score = 0.23
        elif role in {
            'local_residual_distance_scaled_direction_equation',
            'local_residual_distance_scaled_perpendicular_equation',
            'residual_distance_scaled_direction_equation',
            'residual_distance_scaled_perpendicular_equation',
            'generated_operator_distance_scaled_direction_equation',
            'generated_operator_distance_scaled_perpendicular_equation',
            'generated_operator_cutoff_direction_equation',
            'generated_operator_cutoff_perpendicular_equation',
            'generated_operator_tapered_distance_direction_equation',
            'generated_operator_tapered_distance_perpendicular_equation',
        }:
            minimum_score = 0.42
        elif role in {
            'residual_direction_equation',
            'residual_perpendicular_equation',
            'residual_periodic_equation',
            'generated_operator_periodic_equation',
        }:
            minimum_score = 0.45
        else:
            minimum_score = 0.15
        if priority >= 3.0 and score < minimum_score:
            priority = 0.0
        return (priority, score, -int(equation.get('complexity', 0)))

    def _interesting_role_priority(self, role: str) -> float:
        priorities = {
            'generated_operator_tapered_distance_direction_equation': 5.95,
            'generated_operator_tapered_distance_perpendicular_equation': 5.95,
            'generated_operator_cutoff_direction_equation': 5.85,
            'generated_operator_cutoff_perpendicular_equation': 5.85,
            'generated_operator_distance_scaled_direction_equation': 5.75,
            'generated_operator_distance_scaled_perpendicular_equation': 5.75,
            'generated_operator_periodic_equation': 5.45,
            'local_residual_distance_scaled_direction_equation': 5.5,
            'local_residual_distance_scaled_perpendicular_equation': 5.5,
            'residual_distance_scaled_direction_equation': 5.4,
            'residual_distance_scaled_perpendicular_equation': 5.4,
            'local_residual_direction_equation': 5.2,
            'local_residual_perpendicular_equation': 5.2,
            'residual_direction_equation': 5.0,
            'residual_perpendicular_equation': 5.0,
            'residual_periodic_equation': 4.8,
            'vector_direction_equation': 4.0,
            'vector_perpendicular_equation': 4.0,
            'direction_scaled_equation': 3.5,
            'perpendicular_scaled_equation': 3.5,
            'constant_change_equation': 3.0,
            'position_update_equation': 2.0,
            'derived_magnitude_equation': 1.0,
        }
        return priorities.get(role, 0.0)

    def _category_for_role(self, role: str) -> str:
        if role == 'invariant_equation':
            return 'foundation_invariants'
        if role in {'action_extent_mapping', 'state_transition_equation'}:
            return 'state_transitions'
        if role in {'position_update_equation', 'constant_change_equation'}:
            return 'motion_updates'
        if role in {
            'direction_scaled_equation',
            'perpendicular_scaled_equation',
            'vector_direction_equation',
            'vector_perpendicular_equation',
        }:
            return 'direction_vectors'
        if role in {
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
        }:
            return 'residual_strength'
        if role in {
            'residual_direction_equation',
            'residual_perpendicular_equation',
            'local_residual_direction_equation',
            'local_residual_perpendicular_equation',
        }:
            return 'residual_dynamics'
        if role in {'residual_periodic_equation', 'generated_operator_periodic_equation'}:
            return 'residual_periodic'
        if role == 'derived_magnitude_equation':
            return 'derived_magnitudes'
        return 'other'

    def _best_probe_equation(self) -> PrimitiveEquation | None:
        probe_roles = {
            'direction_scaled_equation',
            'perpendicular_scaled_equation',
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
            'generated_operator_periodic_equation',
            'residual_periodic_equation',
            'constant_change_equation',
        }
        candidates = [
            equation for equation in self.discovered_equations()
            if (
                equation.role in probe_roles
                and equation.score >= self._probe_score_threshold(equation.role)
            )
        ]
        if not candidates:
            return None
        candidates.sort(
            key=lambda equation: (
                self._interesting_role_priority(equation.role),
                equation.score,
                -equation.complexity,
            ),
            reverse=True,
        )
        return candidates[0]

    def _probe_score_threshold(self, role: str) -> float:
        if role in {'local_residual_direction_equation', 'local_residual_perpendicular_equation'}:
            return 0.23
        if role in {
            'local_residual_distance_scaled_direction_equation',
            'local_residual_distance_scaled_perpendicular_equation',
            'residual_distance_scaled_direction_equation',
            'residual_distance_scaled_perpendicular_equation',
            'generated_operator_distance_scaled_direction_equation',
            'generated_operator_distance_scaled_perpendicular_equation',
            'generated_operator_cutoff_direction_equation',
            'generated_operator_cutoff_perpendicular_equation',
            'generated_operator_tapered_distance_direction_equation',
            'generated_operator_tapered_distance_perpendicular_equation',
        }:
            return 0.42
        if role == 'generated_operator_periodic_equation':
            return 0.45
        if role in {
            'residual_direction_equation',
            'residual_perpendicular_equation',
            'residual_periodic_equation',
        }:
            return 0.45
        return 0.25

    def _probe_question(self, equation: PrimitiveEquation) -> str:
        if equation.role in {'residual_periodic_equation', 'generated_operator_periodic_equation'}:
            return 'wait across another cycle and compare the repeating step template'
        if 'tapered_distance' in equation.role:
            return 'test whether residual strength tapers through the local region and fades at the boundary'
        if 'cutoff' in equation.role:
            return 'test just inside and just outside the inferred residual boundary'
        if 'distance_scaled' in equation.role:
            return 'test whether residual strength changes with separation at a new location'
        if 'perpendicular' in equation.role:
            return 'test whether perpendicular direction scaling remains stable at a new location'
        if 'direction' in equation.role:
            return 'test whether direction scaling remains stable at a new location'
        return 'test whether repeated channel change remains stable at a new location'

    def _object_map(self, raw_state: dict) -> dict[int, dict]:
        return {
            obj['id']: obj
            for obj in raw_state.get('objects', [])
        }

    def _object_row(
        self,
        before: dict,
        after: dict,
        dt: float,
        step: int,
        anchor_x: float,
        anchor_y: float,
        action: dict,
    ) -> dict | None:
        x, y = before.get('position', (0.0, 0.0))
        next_x, next_y = after.get('position', (0.0, 0.0))
        vx, vy = before.get('velocity', (0.0, 0.0))
        next_vx, next_vy = after.get('velocity', (0.0, 0.0))
        action_type = action.get('type', 'wait')
        action_target_push = (
            action_type == 'push'
            and action.get('object_id') == before.get('id')
        )
        anchor_dx = anchor_x - float(x)
        anchor_dy = anchor_y - float(y)
        anchor_distance = math.sqrt(anchor_dx * anchor_dx + anchor_dy * anchor_dy)
        if anchor_distance <= 1e-9:
            unit_x = 0.0
            unit_y = 0.0
        else:
            unit_x = anchor_dx / anchor_distance
            unit_y = anchor_dy / anchor_distance
        speed_sq = float(vx) * float(vx) + float(vy) * float(vy)
        return {
            'step': step,
            'dt': dt,
            'x': float(x),
            'y': float(y),
            'vx': float(vx),
            'vy': float(vy),
            'next_x': float(next_x),
            'next_y': float(next_y),
            'next_vx': float(next_vx),
            'next_vy': float(next_vy),
            'dvx': float(next_vx) - float(vx),
            'dvy': float(next_vy) - float(vy),
            'mass': float(before.get('mass', 0.0)),
            'radius': float(before.get('radius', 0.0)),
            'next_mass': float(after.get('mass', 0.0)),
            'next_radius': float(after.get('radius', 0.0)),
            'speed_sq': speed_sq,
            'anchor_x': anchor_x,
            'anchor_y': anchor_y,
            'action_push': 1.0 if action_type == 'push' else 0.0,
            'action_target_push': 1.0 if action_target_push else 0.0,
            'anchor_dx': anchor_dx,
            'anchor_dy': anchor_dy,
            'anchor_distance': anchor_distance,
            'unit_anchor_x': unit_x,
            'unit_anchor_y': unit_y,
        }

    def _split_rows(self, rows: list[dict]) -> tuple[list[dict], list[dict]]:
        if len(rows) < 2:
            return [], []
        valid = [
            row for index, row in enumerate(rows)
            if index % 5 == 4
        ]
        train = [
            row for index, row in enumerate(rows)
            if index % 5 != 4
        ]
        if not valid:
            valid = [rows[-1]]
            train = rows[:-1]
        return train, valid

    def _mse(self, predictions: list[float], targets: list[float]) -> float:
        return sum((pred - target) ** 2 for pred, target in zip(predictions, targets)) / len(targets)

    def _baseline_mse(self, targets: list[float]) -> float:
        if not targets:
            return 0.0
        baseline = mean(targets)
        return sum((value - baseline) ** 2 for value in targets) / len(targets)

    def _latest_step(self) -> int:
        rows = self.aggregate_rows + self.object_rows
        if not rows:
            return 0
        return max(int(row.get('step', 0)) for row in rows)


@dataclass(frozen=True)
class _DirectSpec:
    key: str
    target: str
    expression: str
    description: str
    complexity: int
    fn: Callable[[dict], float]
    role: str


@dataclass(frozen=True)
class _FitSpec:
    key: str
    target: str
    expression: str
    description: str
    complexity: int
    feature: Callable[[dict], float]
    role: str
