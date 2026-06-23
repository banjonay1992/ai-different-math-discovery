from __future__ import annotations

"""
General dynamics law learning.

This is an early step toward first-principles AI: instead of asking only
hand-written detectors whether a known force type exists, the agent compares
small candidate equations by how well they predict observed velocity changes.
"""

import math
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class MotionSample:
    """One observed transition for one object."""
    step: int
    x: float
    y: float
    vx: float
    vy: float
    dvx: float
    dvy: float


@dataclass
class VectorTerm:
    """A candidate vector-valued term in a dynamics equation."""
    name: str
    vector_fn: Callable[[MotionSample], tuple[float, float]]
    description: str
    origin: str = 'seeded'
    properties: dict = field(default_factory=dict)


@dataclass
class LearnedLaw:
    """A compact, inspectable dynamics rule learned from observation."""
    name: str
    law_type: str
    description: str
    confidence: float
    mse: float
    baseline_mse: float
    improvement: float
    sample_count: int
    coefficients: dict[str, float] = field(default_factory=dict)
    properties: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'law_type': self.law_type,
            'description': self.description,
            'confidence': round(self.confidence, 3),
            'mse': round(self.mse, 6),
            'baseline_mse': round(self.baseline_mse, 6),
            'improvement': round(self.improvement, 3),
            'sample_count': self.sample_count,
            'coefficients': {
                key: round(value, 6)
                for key, value in self.coefficients.items()
            },
            'properties': dict(self.properties),
        }


@dataclass
class _FittedModel:
    terms: list[VectorTerm]
    coefficients: list[float]
    mse: float

    @property
    def coefficient_map(self) -> dict[str, float]:
        return {
            term.name: coeff
            for term, coeff in zip(self.terms, self.coefficients)
        }


@dataclass
class _CandidateGroup:
    name: str
    law_type: str
    terms: list[VectorTerm]
    description: str
    properties: dict = field(default_factory=dict)


class DynamicsLawLearner:
    """
    Searches for simple predictive equations over observed motion.

    The learner is intentionally modest: it fits linear combinations of vector
    terms, but those terms are generated from primitive observations
    (position, velocity, time, and relative direction to candidate centers).
    """

    def __init__(self, min_samples: int = 80, law_priors: list[dict] | None = None):
        self.min_samples = min_samples
        self.center_values = [6.0, 8.0, 10.0, 12.0, 14.0]
        self.period_values = [60, 80, 100]
        self.law_priors = law_priors or []

    def discover(
        self,
        samples: list[MotionSample],
        max_laws: int = 5,
        law_priors: list[dict] | None = None,
    ) -> list[LearnedLaw]:
        if len(samples) < self.min_samples:
            return []
        active_priors = law_priors if law_priors is not None else self.law_priors

        baseline_terms = self._baseline_terms()
        baseline = self._fit(samples, baseline_terms)
        if baseline is None:
            return []

        invented = self._invent_terms(samples, baseline)
        laws = []
        uniform = self._uniform_law(baseline, len(samples))
        if uniform is not None:
            laws.append(uniform)

        candidate_laws = []
        candidate_laws.extend(self._best_radial_laws(
            samples, baseline_terms, baseline, invented['radial_centers'], active_priors
        ))
        candidate_laws.extend(self._best_tangential_laws(
            samples, baseline_terms, baseline, invented['tangent_centers'], active_priors
        ))
        candidate_laws.extend(self._best_time_laws(
            samples, baseline_terms, baseline, invented['periods'], active_priors
        ))
        composed = self._compose_law(samples, baseline_terms, baseline, invented, active_priors)
        if composed is not None:
            candidate_laws.append(composed)

        candidate_laws.sort(key=self._law_elegance_score, reverse=True)
        for law in candidate_laws:
            if law.confidence >= 0.35:
                laws.append(law)

        deduped = []
        seen_types = set()
        for law in sorted(laws, key=self._law_elegance_score, reverse=True):
            key = law.properties.get('signature', law.law_type)
            if key in seen_types:
                continue
            deduped.append(law)
            seen_types.add(key)
            if len(deduped) >= max_laws:
                break

        return deduped

    def _baseline_terms(self) -> list[VectorTerm]:
        return [
            VectorTerm('constant_x', lambda s: (1.0, 0.0), 'constant horizontal acceleration'),
            VectorTerm('constant_y', lambda s: (0.0, 1.0), 'constant vertical acceleration'),
            VectorTerm('damping_x', lambda s: (s.vx, 0.0), 'horizontal velocity damping'),
            VectorTerm('damping_y', lambda s: (0.0, s.vy), 'vertical velocity damping'),
        ]

    def _uniform_law(self, baseline: _FittedModel, sample_count: int) -> LearnedLaw | None:
        coeffs = baseline.coefficient_map
        ax = coeffs.get('constant_x', 0.0)
        ay = coeffs.get('constant_y', 0.0)
        magnitude = math.sqrt(ax * ax + ay * ay)
        if magnitude < 0.04:
            return None

        confidence = min(1.0, magnitude / 0.12)
        if abs(ax) > abs(ay):
            direction = 'rightward' if ax > 0 else 'leftward'
            axis = 'x'
        else:
            direction = 'upward' if ay > 0 else 'downward'
            axis = 'y'

        return LearnedLaw(
            name='learned_uniform_acceleration',
            law_type='uniform_acceleration',
            description=f"Velocity change includes a persistent {direction} acceleration component",
            confidence=confidence,
            mse=baseline.mse,
            baseline_mse=baseline.mse,
            improvement=0.0,
            sample_count=sample_count,
            coefficients=coeffs,
            properties={
                'axis': axis,
                'direction': direction,
                'signature': f'uniform_{axis}_{direction}',
            },
        )

    def _best_radial_laws(
        self,
        samples: list[MotionSample],
        baseline_terms: list[VectorTerm],
        baseline: _FittedModel,
        invented_centers: list[tuple[float, float]],
        law_priors: list[dict],
    ) -> list[LearnedLaw]:
        best_radial = None
        best_inverse = None

        memory_centers = self._prior_centers(law_priors, {
            'radial_field',
            'inverse_square_radial_field',
            'composed_dynamics',
        })
        for cx, cy, origin in self._candidate_centers(invented_centers, memory_centers):
            radial_term = self._radial_term(cx, cy, inverse_square=False)
            radial_model = self._fit(samples, baseline_terms + [radial_term])
            radial_law = self._radial_law_from_model(
                radial_model, baseline, cx, cy, inverse_square=False,
                sample_count=len(samples), term_origin=origin
            )
            if radial_law and (best_radial is None or radial_law.confidence > best_radial.confidence):
                best_radial = radial_law

            inverse_term = self._radial_term(cx, cy, inverse_square=True)
            inverse_model = self._fit(samples, baseline_terms + [inverse_term])
            inverse_law = self._radial_law_from_model(
                inverse_model, baseline, cx, cy, inverse_square=True,
                sample_count=len(samples), term_origin=origin
            )
            if inverse_law and (best_inverse is None or inverse_law.confidence > best_inverse.confidence):
                best_inverse = inverse_law

        return [law for law in (best_radial, best_inverse) if law is not None]

    def _radial_law_from_model(
        self,
        model: _FittedModel | None,
        baseline: _FittedModel,
        cx: float,
        cy: float,
        inverse_square: bool,
        sample_count: int,
        term_origin: str,
    ) -> LearnedLaw | None:
        if model is None:
            return None

        term_name = 'inverse_square_radial' if inverse_square else 'unit_radial'
        coeff = model.coefficient_map.get(term_name, 0.0)
        improvement = self._improvement(baseline.mse, model.mse)
        strength_score = min(1.0, abs(coeff) / (0.035 if inverse_square else 0.12))
        confidence = max(0.0, min(1.0, 0.80 * improvement + 0.20 * strength_score))
        if confidence < 0.10:
            return None

        outward = coeff < 0
        direction = 'repulsive' if outward else 'attractive'
        law_type = 'inverse_square_radial_field' if inverse_square else 'radial_field'
        name = f"learned_{law_type}"
        description = (
            f"Velocity change is better predicted by a {direction} "
            f"{'inverse-square ' if inverse_square else ''}radial field near ({cx:.1f}, {cy:.1f})"
        )
        return LearnedLaw(
            name=name,
            law_type=law_type,
            description=description,
            confidence=confidence,
            mse=model.mse,
            baseline_mse=baseline.mse,
            improvement=improvement,
            sample_count=sample_count,
            coefficients=model.coefficient_map,
            properties={
                'center_x': cx,
                'center_y': cy,
                'direction': direction,
                'coefficient': coeff,
                'term_origin': term_origin,
                'signature': f'{law_type}_{direction}',
            },
        )

    def _best_tangential_laws(
        self,
        samples: list[MotionSample],
        baseline_terms: list[VectorTerm],
        baseline: _FittedModel,
        invented_centers: list[tuple[float, float]],
        law_priors: list[dict],
    ) -> list[LearnedLaw]:
        best = None
        memory_centers = self._prior_centers(law_priors, {'tangential_field', 'composed_dynamics'})
        for cx, cy, origin in self._candidate_centers(invented_centers, memory_centers):
            term = self._tangential_term(cx, cy)
            model = self._fit(samples, baseline_terms + [term])
            if model is None:
                continue
            coeff = model.coefficient_map.get('tangent_ccw', 0.0)
            improvement = self._improvement(baseline.mse, model.mse)
            strength_score = min(1.0, abs(coeff) / 0.12)
            confidence = max(0.0, min(1.0, 0.80 * improvement + 0.20 * strength_score))
            if confidence < 0.10:
                continue
            spin = 'counterclockwise' if coeff > 0 else 'clockwise'
            law = LearnedLaw(
                name='learned_tangential_field',
                law_type='tangential_field',
                description=(
                    f"Velocity change is better predicted by a {spin} tangential field "
                    f"near ({cx:.1f}, {cy:.1f})"
                ),
                confidence=confidence,
                mse=model.mse,
                baseline_mse=baseline.mse,
                improvement=improvement,
                sample_count=len(samples),
                coefficients=model.coefficient_map,
                properties={
                    'center_x': cx,
                    'center_y': cy,
                    'spin': spin,
                    'coefficient': coeff,
                    'term_origin': origin,
                    'signature': 'tangential_field',
                },
            )
            if best is None or law.confidence > best.confidence:
                best = law
        return [best] if best is not None else []

    def _best_time_laws(
        self,
        samples: list[MotionSample],
        baseline_terms: list[VectorTerm],
        baseline: _FittedModel,
        invented_periods: list[int],
        law_priors: list[dict],
    ) -> list[LearnedLaw]:
        best = None
        for axis in ('x', 'y'):
            for period, origin in self._candidate_periods(
                invented_periods,
                self._prior_periods(law_priors),
            ):
                terms = baseline_terms + self._time_terms(period, axis, origin=origin)
                model = self._fit(samples, terms)
                if model is None:
                    continue
                coeffs = model.coefficient_map
                sin_coeff = coeffs.get(f'sin_t_{axis}_{period}', 0.0)
                cos_coeff = coeffs.get(f'cos_t_{axis}_{period}', 0.0)
                amplitude = math.sqrt(sin_coeff * sin_coeff + cos_coeff * cos_coeff)
                improvement = self._improvement(baseline.mse, model.mse)
                confidence = max(0.0, min(1.0, 0.80 * improvement + 0.20 * min(1.0, amplitude / 0.10)))
                if confidence < 0.10:
                    continue
                law = LearnedLaw(
                    name='learned_time_varying_field',
                    law_type='time_varying_field',
                    description=(
                        f"Velocity change is better predicted by a periodic {axis}-axis "
                        f"acceleration with period about {period} steps"
                    ),
                    confidence=confidence,
                    mse=model.mse,
                    baseline_mse=baseline.mse,
                    improvement=improvement,
                    sample_count=len(samples),
                    coefficients=coeffs,
                    properties={
                        'axis': axis,
                        'period_steps': period,
                        'amplitude': amplitude,
                        'term_origin': origin,
                        'signature': f'time_varying_{axis}',
                    },
                )
                if best is None or law.confidence > best.confidence:
                    best = law
        return [best] if best is not None else []

    def _invent_terms(self, samples: list[MotionSample], baseline: _FittedModel) -> dict:
        residuals = self._residuals(samples, baseline, significant_only=True)
        all_residuals = self._residuals(samples, baseline, significant_only=False)
        radial_center = self._infer_center_from_residuals(residuals, mode='radial')
        tangent_center = self._infer_center_from_residuals(residuals, mode='tangent')
        periods = self._invent_periods(all_residuals)

        return {
            'radial_centers': [radial_center] if radial_center is not None else [],
            'tangent_centers': [tangent_center] if tangent_center is not None else [],
            'periods': periods,
        }

    def _compose_law(
        self,
        samples: list[MotionSample],
        baseline_terms: list[VectorTerm],
        baseline: _FittedModel,
        invented: dict,
        law_priors: list[dict],
    ) -> LearnedLaw | None:
        """Build a compact multi-term dynamics law by prediction improvement."""
        candidates = self._composition_candidates(invented, law_priors)
        if not candidates:
            return None

        selected: list[_CandidateGroup] = []
        current_model = baseline
        current_score = 0.0
        min_score_gain = 0.025

        for _ in range(4):
            best_group = None
            best_model = None
            best_score_gain = 0.0
            best_score = current_score
            for group in candidates:
                if any(group is selected_group for selected_group in selected):
                    continue
                trial_terms = baseline_terms + self._group_terms(selected + [group])
                trial_model = self._fit(samples, trial_terms)
                if trial_model is None:
                    continue
                trial_score = self._composition_score(baseline, trial_model, selected + [group])
                score_gain = trial_score - current_score
                if score_gain > best_score_gain:
                    best_score_gain = score_gain
                    best_score = trial_score
                    best_group = group
                    best_model = trial_model

            if best_group is None or best_model is None or best_score_gain < min_score_gain:
                break
            selected.append(best_group)
            current_model = best_model
            current_score = best_score

        if len(selected) < 2:
            return None

        selected, current_model = self._prune_composed_groups(
            samples, baseline_terms, baseline, selected, current_model
        )
        if len(selected) < 2:
            return None

        total_improvement = self._improvement(baseline.mse, current_model.mse)
        if total_improvement < 0.15:
            return None

        component_descriptions = [group.description for group in selected]
        component_types = [group.law_type for group in selected]
        if len(set(component_types)) < 2:
            return None

        complexity_penalty = self._composition_penalty(selected)
        elegance_score = max(0.0, total_improvement - complexity_penalty)
        confidence = max(0.0, min(1.0, elegance_score))

        return LearnedLaw(
            name='learned_composed_dynamics',
            law_type='composed_dynamics',
            description="Velocity change is best predicted by a composed law: "
                        + " + ".join(component_descriptions),
            confidence=confidence,
            mse=current_model.mse,
            baseline_mse=baseline.mse,
            improvement=total_improvement,
            sample_count=len(samples),
            coefficients=current_model.coefficient_map,
            properties={
                'components': component_types,
                'component_descriptions': component_descriptions,
                'component_count': len(selected),
                'term_origins': [group.properties.get('origin', 'seeded') for group in selected],
                'complexity_penalty': round(complexity_penalty, 3),
                'elegance_score': round(elegance_score, 3),
                'signature': 'composed_dynamics_' + '_'.join(component_types),
            },
        )

    def _composition_candidates(self, invented: dict, law_priors: list[dict]) -> list[_CandidateGroup]:
        groups: list[_CandidateGroup] = []

        for cx, cy, origin in self._candidate_centers(
            invented['radial_centers'],
            self._prior_centers(law_priors, {
                'radial_field',
                'inverse_square_radial_field',
                'composed_dynamics',
            }),
        )[:8]:
            groups.append(_CandidateGroup(
                name=f'radial_{cx:.2f}_{cy:.2f}',
                law_type='radial_field',
                terms=[self._radial_term(
                    cx, cy, inverse_square=False,
                    name=f'radial_unit_{cx:.2f}_{cy:.2f}',
                    origin=origin,
                )],
                description=f"radial field near ({cx:.1f}, {cy:.1f})",
                properties={'center_x': cx, 'center_y': cy, 'origin': origin},
            ))
            groups.append(_CandidateGroup(
                name=f'inverse_radial_{cx:.2f}_{cy:.2f}',
                law_type='inverse_square_radial_field',
                terms=[self._radial_term(
                    cx, cy, inverse_square=True,
                    name=f'radial_inv_sq_{cx:.2f}_{cy:.2f}',
                    origin=origin,
                )],
                description=f"inverse-square radial field near ({cx:.1f}, {cy:.1f})",
                properties={'center_x': cx, 'center_y': cy, 'origin': origin},
            ))

        for cx, cy, origin in self._candidate_centers(
            invented['tangent_centers'],
            self._prior_centers(law_priors, {'tangential_field', 'composed_dynamics'}),
        )[:8]:
            groups.append(_CandidateGroup(
                name=f'tangent_{cx:.2f}_{cy:.2f}',
                law_type='tangential_field',
                terms=[self._tangential_term(
                    cx, cy,
                    name=f'tangent_ccw_{cx:.2f}_{cy:.2f}',
                    origin=origin,
                )],
                description=f"tangential field near ({cx:.1f}, {cy:.1f})",
                properties={'center_x': cx, 'center_y': cy, 'origin': origin},
            ))

        for period, origin in self._candidate_periods(
            invented['periods'],
            self._prior_periods(law_priors),
        )[:5]:
            for axis in ('x', 'y'):
                groups.append(_CandidateGroup(
                    name=f'time_{axis}_{period}',
                    law_type='time_varying_field',
                    terms=self._time_terms(period, axis, origin=origin),
                    description=f"periodic {axis}-axis field with period {period}",
                    properties={'axis': axis, 'period_steps': period, 'origin': origin},
                ))

        return groups

    def _group_terms(self, groups: list[_CandidateGroup]) -> list[VectorTerm]:
        terms = []
        for group in groups:
            terms.extend(group.terms)
        return terms

    def _composition_score(
        self,
        baseline: _FittedModel,
        model: _FittedModel,
        groups: list[_CandidateGroup],
    ) -> float:
        return self._improvement(baseline.mse, model.mse) - self._composition_penalty(groups)

    def _composition_penalty(self, groups: list[_CandidateGroup]) -> float:
        component_count = len(groups)
        term_count = sum(len(group.terms) for group in groups)
        family_counts: dict[str, int] = {}
        seeded_count = 0
        for group in groups:
            family_counts[group.law_type] = family_counts.get(group.law_type, 0) + 1
            if group.properties.get('origin', 'seeded') == 'seeded':
                seeded_count += 1

        repeated_family_count = sum(max(0, count - 1) for count in family_counts.values())
        return (
            0.035 * max(0, component_count - 1)
            + 0.012 * max(0, term_count - component_count)
            + 0.060 * repeated_family_count
            + 0.010 * seeded_count
        )

    def _law_elegance_score(self, law: LearnedLaw) -> float:
        prior_bonus = 0.0
        if law.properties.get('term_origin') == 'memory':
            prior_bonus = 0.015
        if 'memory' in law.properties.get('term_origins', []):
            prior_bonus = max(prior_bonus, 0.015)
        if 'elegance_score' in law.properties:
            return float(law.properties['elegance_score']) + prior_bonus
        return law.confidence - float(law.properties.get('complexity_penalty', 0.0)) + prior_bonus

    def _prune_composed_groups(
        self,
        samples: list[MotionSample],
        baseline_terms: list[VectorTerm],
        baseline: _FittedModel,
        selected: list[_CandidateGroup],
        current_model: _FittedModel,
    ) -> tuple[list[_CandidateGroup], _FittedModel]:
        pruned = list(selected)
        changed = True
        while changed and len(pruned) > 1:
            changed = False
            current_score = self._composition_score(baseline, current_model, pruned)
            for group in list(pruned):
                trial = [candidate for candidate in pruned if candidate is not group]
                trial_model = self._fit(samples, baseline_terms + self._group_terms(trial))
                if trial_model is None:
                    continue
                trial_score = self._composition_score(baseline, trial_model, trial)
                if current_score - trial_score < 0.025:
                    pruned = trial
                    current_model = trial_model
                    changed = True
                    break
        return pruned, current_model

    def _residuals(
        self,
        samples: list[MotionSample],
        model: _FittedModel,
        significant_only: bool = True,
    ) -> list[tuple[MotionSample, float, float, float]]:
        residuals = []
        magnitudes = []
        raw = []
        for sample in samples:
            pred_x, pred_y = self._predict_delta(sample, model)
            rx = sample.dvx - pred_x
            ry = sample.dvy - pred_y
            mag = math.sqrt(rx * rx + ry * ry)
            raw.append((sample, rx, ry, mag))
            magnitudes.append(mag)

        if not magnitudes:
            return []

        if not significant_only:
            return raw

        threshold = max(0.015, self._percentile(magnitudes, 0.35))
        for sample, rx, ry, mag in raw:
            if mag >= threshold:
                residuals.append((sample, rx, ry, mag))
        return residuals

    def _predict_delta(self, sample: MotionSample, model: _FittedModel) -> tuple[float, float]:
        pred_x = 0.0
        pred_y = 0.0
        for coeff, term in zip(model.coefficients, model.terms):
            tx, ty = term.vector_fn(sample)
            pred_x += coeff * tx
            pred_y += coeff * ty
        return (pred_x, pred_y)

    def _infer_center_from_residuals(
        self,
        residuals: list[tuple[MotionSample, float, float, float]],
        mode: str,
    ) -> tuple[float, float] | None:
        if len(residuals) < 20:
            return None

        a00 = 0.0
        a01 = 0.0
        a11 = 0.0
        b0 = 0.0
        b1 = 0.0

        for sample, rx, ry, mag in residuals:
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
            b0 += m00 * sample.x + m01 * sample.y
            b1 += m01 * sample.x + m11 * sample.y

        det = a00 * a11 - a01 * a01
        if abs(det) < 1e-9:
            return None

        cx = (b0 * a11 - b1 * a01) / det
        cy = (a00 * b1 - a01 * b0) / det
        if not (-5.0 <= cx <= 25.0 and -5.0 <= cy <= 25.0):
            return None
        return (round(cx, 3), round(cy, 3))

    def _invent_periods(self, residuals: list[tuple[MotionSample, float, float, float]]) -> list[int]:
        candidates = []
        for axis in ('x', 'y'):
            series = self._residual_series(residuals, axis)
            if len(series) < 60:
                continue
            candidates.extend(self._periods_from_zero_crossings(series))
            candidates.extend(self._periods_from_autocorrelation(series))

        deduped = []
        seen = set()
        for period in sorted(candidates, key=lambda item: item[1], reverse=True):
            p = int(round(period[0]))
            if p < 20 or p > 160:
                continue
            if any(abs(p - existing) <= 2 for existing in seen):
                continue
            deduped.append(p)
            seen.add(p)
            if len(deduped) >= 3:
                break
        return deduped

    def _residual_series(
        self,
        residuals: list[tuple[MotionSample, float, float, float]],
        axis: str,
    ) -> list[float]:
        grouped: dict[int, list[float]] = {}
        for sample, rx, ry, _ in residuals:
            grouped.setdefault(sample.step, []).append(rx if axis == 'x' else ry)
        return [
            sum(values) / len(values)
            for _, values in sorted(grouped.items())
            if values
        ]

    def _periods_from_zero_crossings(self, series: list[float]) -> list[tuple[int, float]]:
        amplitude = (max(series) - min(series)) / 2
        if amplitude < 0.02:
            return []

        threshold = amplitude * 0.15
        signs = []
        for value in series:
            if value > threshold:
                signs.append(1)
            elif value < -threshold:
                signs.append(-1)
            else:
                signs.append(0)

        crossings = []
        prev_sign = 0
        for idx, sign in enumerate(signs):
            if sign == 0:
                continue
            if prev_sign and sign != prev_sign:
                crossings.append(idx)
            prev_sign = sign

        if len(crossings) < 3:
            return []

        gaps = [
            crossings[i] - crossings[i - 1]
            for i in range(1, len(crossings))
            if crossings[i] > crossings[i - 1]
        ]
        if not gaps:
            return []

        half_period = sum(gaps) / len(gaps)
        period = int(round(2 * half_period))
        consistency = 1.0 / (1.0 + self._std(gaps))
        return [(period, consistency)]

    def _periods_from_autocorrelation(self, series: list[float]) -> list[tuple[int, float]]:
        max_lag = min(160, len(series) // 2)
        scored = []
        for lag in range(20, max_lag + 1):
            corr = self._lag_correlation(series, lag)
            if corr > 0.35:
                scored.append((lag, corr))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:3]

    def _prior_centers(
        self,
        law_priors: list[dict],
        law_types: set[str],
    ) -> list[tuple[float, float]]:
        centers = []
        for prior in law_priors:
            if prior.get('law_type') not in law_types:
                continue
            ranges = prior.get('parameter_ranges', {})
            if 'center_x' not in ranges or 'center_y' not in ranges:
                continue
            x_low, x_high = ranges['center_x']
            y_low, y_high = ranges['center_y']
            centers.append(((x_low + x_high) / 2, (y_low + y_high) / 2))
        return centers

    def _prior_periods(self, law_priors: list[dict]) -> list[int]:
        periods = []
        for prior in law_priors:
            if prior.get('law_type') not in {'time_varying_field', 'composed_dynamics'}:
                continue
            ranges = prior.get('parameter_ranges', {})
            if 'period_steps' not in ranges:
                continue
            low, high = ranges['period_steps']
            periods.append(int(round((low + high) / 2)))
        return periods

    def _candidate_centers(
        self,
        invented_centers: list[tuple[float, float]],
        memory_centers: list[tuple[float, float]] | None = None,
    ) -> list[tuple[float, float, str]]:
        candidates = []
        for cx, cy in memory_centers or []:
            candidates.append((cx, cy, 'memory'))
        for cx, cy in invented_centers:
            candidates.append((cx, cy, 'invented'))
        for cx in self.center_values:
            for cy in self.center_values:
                candidates.append((cx, cy, 'seeded'))

        deduped = []
        seen = set()
        for cx, cy, origin in candidates:
            key = (round(cx, 1), round(cy, 1))
            if key in seen:
                continue
            deduped.append((cx, cy, origin))
            seen.add(key)
        return deduped

    def _candidate_periods(
        self,
        invented_periods: list[int],
        memory_periods: list[int] | None = None,
    ) -> list[tuple[int, str]]:
        candidates = [(period, 'memory') for period in (memory_periods or [])]
        candidates.extend((period, 'invented') for period in invented_periods)
        candidates.extend((period, 'seeded') for period in self.period_values)

        deduped = []
        seen = set()
        for period, origin in candidates:
            if any(abs(period - existing) <= 2 for existing in seen):
                continue
            deduped.append((period, origin))
            seen.add(period)
        return deduped

    def _radial_term(
        self,
        cx: float,
        cy: float,
        inverse_square: bool,
        name: str | None = None,
        origin: str = 'seeded',
    ) -> VectorTerm:
        term_name = name or ('inverse_square_radial' if inverse_square else 'unit_radial')

        def vector(sample: MotionSample) -> tuple[float, float]:
            dx = cx - sample.x
            dy = cy - sample.y
            dist_sq = max(dx * dx + dy * dy, 0.25)
            dist = math.sqrt(dist_sq)
            scale = 1.0 / dist_sq if inverse_square else 1.0
            return ((dx / dist) * scale, (dy / dist) * scale)

        description = 'radial inverse-square direction' if inverse_square else 'radial direction'
        return VectorTerm(
            term_name,
            vector,
            description,
            origin=origin,
            properties={'center_x': cx, 'center_y': cy, 'inverse_square': inverse_square},
        )

    def _tangential_term(
        self,
        cx: float,
        cy: float,
        name: str = 'tangent_ccw',
        origin: str = 'seeded',
    ) -> VectorTerm:
        def vector(sample: MotionSample) -> tuple[float, float]:
            dx = sample.x - cx
            dy = sample.y - cy
            dist = max(math.sqrt(dx * dx + dy * dy), 0.5)
            return (-dy / dist, dx / dist)

        return VectorTerm(
            name,
            vector,
            'counterclockwise tangent direction',
            origin=origin,
            properties={'center_x': cx, 'center_y': cy},
        )

    def _time_terms(self, period: int, axis: str, origin: str = 'seeded') -> list[VectorTerm]:
        def sin_vector(sample: MotionSample) -> tuple[float, float]:
            value = math.sin((2 * math.pi * sample.step) / period)
            return (value, 0.0) if axis == 'x' else (0.0, value)

        def cos_vector(sample: MotionSample) -> tuple[float, float]:
            value = math.cos((2 * math.pi * sample.step) / period)
            return (value, 0.0) if axis == 'x' else (0.0, value)

        return [
            VectorTerm(
                f'sin_t_{axis}_{period}',
                sin_vector,
                f'sinusoidal {axis}-axis acceleration',
                origin=origin,
                properties={'axis': axis, 'period_steps': period},
            ),
            VectorTerm(
                f'cos_t_{axis}_{period}',
                cos_vector,
                f'cosine {axis}-axis acceleration',
                origin=origin,
                properties={'axis': axis, 'period_steps': period},
            ),
        ]

    def _fit(self, samples: list[MotionSample], terms: list[VectorTerm]) -> _FittedModel | None:
        rows = []
        targets = []
        for sample in samples:
            x_row = []
            y_row = []
            for term in terms:
                tx, ty = term.vector_fn(sample)
                x_row.append(tx)
                y_row.append(ty)
            rows.append(x_row)
            targets.append(sample.dvx)
            rows.append(y_row)
            targets.append(sample.dvy)

        coefficients = self._least_squares(rows, targets)
        if coefficients is None:
            return None

        mse = 0.0
        for row, target in zip(rows, targets):
            predicted = sum(coeff * value for coeff, value in zip(coefficients, row))
            error = target - predicted
            mse += error * error
        mse /= max(len(targets), 1)
        return _FittedModel(terms=terms, coefficients=coefficients, mse=mse)

    def _least_squares(self, rows: list[list[float]], targets: list[float]) -> list[float] | None:
        if not rows:
            return None
        n_terms = len(rows[0])
        if n_terms == 0:
            return None

        matrix = [[0.0 for _ in range(n_terms)] for _ in range(n_terms)]
        vector = [0.0 for _ in range(n_terms)]
        ridge = 1e-6
        for row, target in zip(rows, targets):
            for i in range(n_terms):
                vector[i] += row[i] * target
                for j in range(n_terms):
                    matrix[i][j] += row[i] * row[j]
        for i in range(n_terms):
            matrix[i][i] += ridge

        return self._solve_linear_system(matrix, vector)

    def _solve_linear_system(self, matrix: list[list[float]], vector: list[float]) -> list[float] | None:
        n = len(vector)
        aug = [row[:] + [value] for row, value in zip(matrix, vector)]

        for col in range(n):
            pivot_row = max(range(col, n), key=lambda row_idx: abs(aug[row_idx][col]))
            if abs(aug[pivot_row][col]) < 1e-12:
                return None
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

            pivot = aug[col][col]
            for j in range(col, n + 1):
                aug[col][j] /= pivot

            for row_idx in range(n):
                if row_idx == col:
                    continue
                factor = aug[row_idx][col]
                if abs(factor) < 1e-12:
                    continue
                for j in range(col, n + 1):
                    aug[row_idx][j] -= factor * aug[col][j]

        return [aug[i][n] for i in range(n)]

    @staticmethod
    def _improvement(baseline_mse: float, model_mse: float) -> float:
        if baseline_mse <= 1e-12:
            return 0.0
        return max(0.0, 1.0 - (model_mse / baseline_mse))

    @staticmethod
    def _percentile(values: list[float], fraction: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        idx = int((len(ordered) - 1) * fraction)
        return ordered[idx]

    @staticmethod
    def _std(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))

    @staticmethod
    def _lag_correlation(series: list[float], lag: int) -> float:
        if lag <= 0 or lag >= len(series):
            return 0.0

        left = series[:-lag]
        right = series[lag:]
        if len(left) < 2:
            return 0.0

        mean_left = sum(left) / len(left)
        mean_right = sum(right) / len(right)
        var_left = sum((value - mean_left) ** 2 for value in left)
        var_right = sum((value - mean_right) ** 2 for value in right)
        if var_left < 1e-12 or var_right < 1e-12:
            return 0.0

        covariance = sum(
            (a - mean_left) * (b - mean_right)
            for a, b in zip(left, right)
        )
        return covariance / math.sqrt(var_left * var_right)
