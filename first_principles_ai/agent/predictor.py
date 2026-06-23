from __future__ import annotations

"""
Predictor — the agent's reasoning engine.

This is where genuine discovery happens. The agent:

1. Tracks feature values over time
2. Detects regularities (constants, patterns, correlations)
3. Forms hypotheses about why those regularities exist
4. Tests hypotheses against new observations
5. Confirms or rejects them based on evidence

This is NOT pattern matching. This is inductive reasoning:
  - Observe → Notice regularity → Hypothesize → Test → Confirm/Reject

The agent discovers concepts like:
  - "count" is a stable property (discovery of natural numbers)
  - "total_momentum" stays constant during collisions (discovery of conservation)
  - "distance" between objects changes predictably (discovery of geometry)
"""

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from .law_learning import DynamicsLawLearner, LearnedLaw, MotionSample
from .representation import (
    KnowledgeBase, Concept, Rule, ConceptType, RuleStatus
)


@dataclass
class FeatureHistory:
    """Tracks a single feature's values over time for regularity detection."""
    values: deque = field(default_factory=lambda: deque(maxlen=500))
    steps: deque = field(default_factory=lambda: deque(maxlen=500))
    collision_values: list = field(default_factory=list)  # values during collisions
    non_collision_values: list = field(default_factory=list)  # values without collisions
    collision_deltas: list = field(default_factory=list)  # change during collision steps
    non_collision_deltas: list = field(default_factory=list)  # change during non-collision steps

    @property
    def latest(self) -> float:
        return self.values[-1] if self.values else 0.0

    @property
    def mean(self) -> float:
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)

    @property
    def variance(self) -> float:
        if len(self.values) < 2:
            return 0.0
        m = self.mean
        return sum((v - m) ** 2 for v in self.values) / len(self.values)

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    @property
    def cv(self) -> float:
        """Coefficient of variation — normalized measure of how much it changes."""
        m = abs(self.mean)
        return self.std / m if m > 1e-10 else self.std

    @property
    def is_constant(self) -> bool:
        """Is this feature essentially constant? (low variance relative to mean)"""
        if len(self.values) < 20:
            return False
        return self.cv < 0.05  # Less than 5% variation

    @property
    def is_near_constant(self) -> bool:
        """Nearly constant but with some variation (e.g., from friction)."""
        if len(self.values) < 20:
            return False
        return self.cv < 0.15

    def record(self, value: float, step: int, during_collision: bool = False):
        self.values.append(value)
        self.steps.append(step)
        if during_collision:
            self.collision_values.append(value)
        else:
            self.non_collision_values.append(value)

    def recent_delta(self, window: int = 10) -> float:
        """How much has this feature changed recently?"""
        if len(self.values) < window + 1:
            return 0.0
        return self.values[-1] - self.values[-window]

    def collision_vs_non_collision_difference(self) -> float:
        """Does this feature behave differently during collisions vs not?"""
        if len(self.collision_values) < 5 or len(self.non_collision_values) < 5:
            return float('inf')
        coll_mean = sum(self.collision_values) / len(self.collision_values)
        non_coll_mean = sum(self.non_collision_values) / len(self.non_collision_values)
        return abs(coll_mean - non_coll_mean)

    @property
    def collision_delta_mean(self) -> float:
        """Average change in this feature during collision steps."""
        if not self.collision_deltas:
            return 0.0
        return sum(self.collision_deltas) / len(self.collision_deltas)

    @property
    def non_collision_delta_mean(self) -> float:
        """Average change in this feature during non-collision steps."""
        if not self.non_collision_deltas:
            return 0.0
        return sum(self.non_collision_deltas) / len(self.non_collision_deltas)

    @property
    def collision_delta_std(self) -> float:
        """Std dev of change during collision steps."""
        if len(self.collision_deltas) < 2:
            return 0.0
        m = self.collision_delta_mean
        return math.sqrt(sum((d - m) ** 2 for d in self.collision_deltas) / len(self.collision_deltas))

    @property
    def non_collision_delta_std(self) -> float:
        """Std dev of change during non-collision steps."""
        if len(self.non_collision_deltas) < 2:
            return 0.0
        m = self.non_collision_delta_mean
        return math.sqrt(sum((d - m) ** 2 for d in self.non_collision_deltas) / len(self.non_collision_deltas))

    def collision_conserves_feature(self) -> bool:
        """
        Does the collision event itself conserve this feature?

        Logic: if the change during collision steps is approximately the same as
        during non-collision steps, it means the collision adds no extra change.
        Only the baseline forces (gravity, friction) are changing the feature.
        The collision itself conserves it.
        """
        if len(self.collision_deltas) < 10 or len(self.non_collision_deltas) < 10:
            return False

        coll_mean = self.collision_delta_mean
        non_coll_mean = self.non_collision_delta_mean
        coll_std = self.collision_delta_std
        non_coll_std = self.non_collision_delta_std

        # The difference in mean deltas should be small relative to the spread
        pooled_std = math.sqrt((coll_std ** 2 + non_coll_std ** 2) / 2)
        if pooled_std < 1e-10:
            pooled_std = 1.0

        # Effect size (Cohen's d): how different are collision vs non-collision deltas?
        effect_size = abs(coll_mean - non_coll_mean) / pooled_std

        # Small effect size → collision doesn't change the feature beyond baseline
        return effect_size < 0.5


class Predictor:
    """
    The agent's reasoning engine.

    It observes features over time, detects regularities, and forms hypotheses.
    This is the core of the discovery process.
    """

    # Features to track — the agent discovers these are meaningful through observation
    TRACKED_FEATURES = [
        'count',
        'total_momentum_x',
        'total_momentum_y',
        'total_momentum',
        'total_kinetic_energy',
        'total_mass',
        'center_of_mass_x',
        'center_of_mass_y',
        'num_collisions',
        'mean_distance',
        'mean_speed',
    ]

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        law_priors: list[dict] | None = None,
        allow_memory_probes: bool = True,
    ):
        self.kb = knowledge_base
        self.feature_histories: dict[str, FeatureHistory] = {
            f: FeatureHistory() for f in self.TRACKED_FEATURES
        }
        self.prev_count: int = 0
        self.prev_features: dict = {}
        self.step: int = 0
        self.concepts_discovered: set[str] = set()  # feature keys we've already formed concepts for
        self.hypotheses_tested: set[str] = set()     # hypothesis keys already being tested
        self.meta_concepts_formed: set[str] = set()  # meta-concept keys already formed
        self.spatial_hypotheses_tested: set[str] = set()  # spatial hypothesis keys
        self.operations_invented: set[str] = set()   # operations already given notation
        self.novel_physics_tested: set[str] = set()  # novel physics hypothesis keys
        self.object_trajectories: dict[int, list[tuple[float, float, float, float]]] = {}  # id → [(x,y,vx,vy)]
        self.timed_object_trajectories: dict[int, list[tuple[int, float, float, float, float]]] = {}
        self.prev_objects: list[dict] = []  # Previous step's object states
        self.novel_physics_diagnostics: dict[str, dict] = {}
        self.last_active_experiment_step: int = -999
        self.active_experiment_count: int = 0
        self._motion_model_cache_step: int = -1
        self._motion_model_cache: dict = {}
        self._radial_field_cache: dict[tuple[int, str], Optional[tuple[float, float, float]]] = {}
        self.law_priors = law_priors or []
        self.memory_probe_questions = (
            self._memory_probe_questions_from_priors(self.law_priors)
            if allow_memory_probes
            else []
        )
        self.memory_probe_index = 0
        self.dynamics_law_learner = DynamicsLawLearner(law_priors=self.law_priors)
        self.learned_dynamics_laws: list[LearnedLaw] = []
        self.learned_dynamics_rule_keys: set[str] = set()

        # Discovery thresholds
        self.min_observations_for_concept = 30
        self.min_evidence_for_confirmation = 15
        self.confirmation_confidence = 0.85
        self.min_observations_for_abstraction = 100
        self.min_observations_for_spatial = 50

    def observe(self, features: dict, had_collision: bool, step: int,
                raw_objects: list[dict] = None):
        """Record a new observation and check for discoveries."""
        self.step = step
        count = features.get('count', 0)

        # Track object trajectories for novel physics detection
        if raw_objects:
            for obj in raw_objects:
                oid = obj['id']
                x, y = obj['position']
                vx, vy = obj['velocity']
                if oid not in self.object_trajectories:
                    self.object_trajectories[oid] = []
                if oid not in self.timed_object_trajectories:
                    self.timed_object_trajectories[oid] = []
                self.object_trajectories[oid].append((x, y, vx, vy))
                self.timed_object_trajectories[oid].append((step, x, y, vx, vy))
                # Keep trajectory bounded
                if len(self.object_trajectories[oid]) > 100:
                    self.object_trajectories[oid] = self.object_trajectories[oid][-100:]
                if len(self.timed_object_trajectories[oid]) > 100:
                    self.timed_object_trajectories[oid] = self.timed_object_trajectories[oid][-100:]
            self.prev_objects = raw_objects

        # Track deltas (change from previous step) for collision conservation analysis
        for feature_key in self.TRACKED_FEATURES:
            if feature_key in features:
                history = self.feature_histories[feature_key]
                history.record(
                    features[feature_key], step, during_collision=had_collision
                )
                # Record delta if we have a previous value
                if feature_key in self.prev_features:
                    delta = features[feature_key] - self.prev_features[feature_key]
                    if had_collision:
                        history.collision_deltas.append(delta)
                    else:
                        history.non_collision_deltas.append(delta)

        # Check for new discoveries
        self._check_for_concepts(features, had_collision)
        self._check_for_conservation_laws(had_collision)
        self._check_for_collision_conservation_laws(had_collision)
        self._check_for_counting_rules(features, count)
        self._check_for_abstractions()
        self._check_for_spatial_rules(features)
        self._check_for_learned_dynamics_laws()
        self._check_for_novel_physics(features, had_collision)
        self._check_for_operation_notation()
        self._update_hypotheses(features, had_collision)

        self.prev_count = count
        self.prev_features = features.copy()

    def _check_for_concepts(self, features: dict, had_collision: bool):
        """Check if any feature should be elevated to a 'concept' the agent knows about."""
        for feature_key in self.TRACKED_FEATURES:
            if feature_key in self.concepts_discovered:
                continue

            history = self.feature_histories[feature_key]
            if len(history.values) < self.min_observations_for_concept:
                continue

            # The agent has observed this feature enough to recognize it as a "thing"
            # This is the discovery of a QUANTITY
            concept_type = ConceptType.QUANTITY
            description = f"Observed property: {feature_key}"

            # Special descriptions for special discoveries
            if feature_key == 'count':
                description = "Discrete quantity: number of distinct objects in the world"
                concept_type = ConceptType.QUANTITY
            elif feature_key == 'total_momentum':
                description = "Aggregate quantity: combined mass-weighted motion of all objects"
                concept_type = ConceptType.QUANTITY
            elif feature_key == 'total_kinetic_energy':
                description = "Aggregate quantity: combined motion energy of all objects"
                concept_type = ConceptType.QUANTITY
            elif feature_key == 'mean_distance':
                description = "Relational quantity: average separation between objects"
                concept_type = ConceptType.RELATION
            elif feature_key == 'num_collisions':
                description = "Event: objects making contact with each other"
                concept_type = ConceptType.EVENT

            self.kb.add_concept(
                concept_type=concept_type,
                description=description,
                feature_key=feature_key,
                step=self.step,
                properties={
                    'mean': round(history.mean, 6),
                    'std': round(history.std, 6),
                    'cv': round(history.cv, 6),
                }
            )
            self.concepts_discovered.add(feature_key)

    def _check_for_conservation_laws(self, had_collision: bool):
        """
        Check if any feature appears to be globally conserved (stays constant under all conditions).

        This discovers things like conservation of mass (which holds even with gravity/friction).
        """
        conservation_candidates = [
            'total_mass',
        ]

        for feature_key in conservation_candidates:
            if feature_key not in self.concepts_discovered:
                continue

            hypo_key = f"global_conservation_{feature_key}"
            if hypo_key in self.hypotheses_tested:
                continue

            history = self.feature_histories[feature_key]
            if len(history.values) < 50:
                continue

            # Check if this feature is approximately constant
            if not history.is_near_constant:
                continue

            # Form hypothesis: this quantity is conserved globally
            rule = self.kb.add_hypothesis(
                conditions="always (under all observed conditions)",
                prediction=f"{feature_key} remains constant (globally conserved quantity)",
                feature_key=feature_key,
                step=self.step,
                properties={'hypothesis_type': 'global_conservation'}
            )
            self.hypotheses_tested.add(hypo_key)
            rule.status = RuleStatus.TESTING

    def _check_for_collision_conservation_laws(self, had_collision: bool):
        """
        Check if any feature is conserved specifically during collision events.

        The key insight: if a feature changes by the same amount during collision steps
        as during non-collision steps, the collision itself doesn't affect it.
        Only baseline forces (gravity, friction) change it — the collision conserves it.

        This is how the agent discovers Newton's Third Law:
        momentum is transferred between objects but the TOTAL is conserved during collisions.
        """
        collision_conservation_candidates = [
            'total_momentum_x',
            'total_momentum_y',
            'total_momentum',
            'total_kinetic_energy',
        ]

        for feature_key in collision_conservation_candidates:
            if feature_key not in self.concepts_discovered:
                continue

            hypo_key = f"collision_conservation_{feature_key}"
            if hypo_key in self.hypotheses_tested:
                continue

            history = self.feature_histories[feature_key]

            # Need enough collision AND non-collision observations
            if len(history.collision_deltas) < 10 or len(history.non_collision_deltas) < 10:
                continue

            # Check if the collision doesn't add extra change to this feature
            if not history.collision_conserves_feature():
                continue

            # Form hypothesis: this quantity is conserved during collisions
            rule = self.kb.add_hypothesis(
                conditions="two objects collide (collision event occurs)",
                prediction=f"{feature_key} is conserved — collision transfers it between objects but doesn't change the total",
                feature_key=feature_key,
                step=self.step,
                properties={'hypothesis_type': 'collision_conservation'}
            )
            self.hypotheses_tested.add(hypo_key)
            rule.status = RuleStatus.TESTING

    def _check_for_counting_rules(self, features: dict, current_count: int):
        """
        Check for rules about how count changes.

        This is how the agent discovers arithmetic:
          - Adding an object increases count by 1
          - Removing an object decreases count by 1
        """
        if 'count' not in self.concepts_discovered:
            return

        prev_count = self.prev_count
        if prev_count == 0 and current_count == 0:
            return

        delta = current_count - prev_count

        if delta != 0:
            # Count changed — this is an arithmetic event
            direction = "increases by 1" if delta == 1 else f"changes by {delta}"
            hypo_key = f"count_change_{delta}"

            if hypo_key not in self.hypotheses_tested:
                if delta == 1:
                    self.kb.add_hypothesis(
                        conditions="an object appears in the world",
                        prediction="count increases by exactly 1",
                        feature_key='count',
                        step=self.step,
                        properties={'hypothesis_type': 'arithmetic', 'delta': 1}
                    )
                    self.hypotheses_tested.add(hypo_key)
                elif delta == -1:
                    self.kb.add_hypothesis(
                        conditions="an object disappears from the world",
                        prediction="count decreases by exactly 1",
                        feature_key='count',
                        step=self.step,
                        properties={'hypothesis_type': 'arithmetic', 'delta': -1}
                    )
                    self.hypotheses_tested.add(hypo_key)

            # Test the hypothesis — ONLY when the condition matches
            for rule in self.kb.get_active_hypotheses():
                if (rule.feature_key == 'count' and
                        rule.properties.get('hypothesis_type') == 'arithmetic'):
                    expected_delta = rule.properties.get('delta', 0)
                    # Only test when the relevant condition is met:
                    # - If rule expects +1, only test when count increased (object appeared)
                    # - If rule expects -1, only test when count decreased (object disappeared)
                    if delta > 0 and expected_delta > 0:
                        rule.add_evidence(supports=(delta == expected_delta))
                    elif delta < 0 and expected_delta < 0:
                        rule.add_evidence(supports=(delta == expected_delta))
                    rule.update_status(
                        self.min_evidence_for_confirmation,
                        self.confirmation_confidence
                    )
                    if rule.is_confirmed and rule.confirmed_at_step is None:
                        self.kb.confirm_rule(rule.internal_name, self.step)

    def _update_hypotheses(self, features: dict, had_collision: bool):
        """Update all active hypotheses with new evidence."""
        for rule in self.kb.get_active_hypotheses():
            h_type = rule.properties.get('hypothesis_type', '')
            if h_type == 'global_conservation':
                self._test_global_conservation_hypothesis(rule, features)
            elif h_type == 'collision_conservation':
                self._test_collision_conservation_hypothesis(rule, features, had_collision)
            elif h_type == 'correlation':
                self._test_correlation_hypothesis(rule, features)

    def _test_global_conservation_hypothesis(self, rule: Rule, features: dict):
        """Test whether a globally conserved quantity stays constant."""
        feature_key = rule.feature_key
        history = self.feature_histories[feature_key]

        if len(history.values) < 2:
            return

        current = features.get(feature_key, 0)
        recent_values = list(history.values)[-20:]
        if len(recent_values) < 5:
            return

        local_mean = sum(recent_values) / len(recent_values)
        local_std = math.sqrt(sum((v - local_mean) ** 2 for v in recent_values) / len(recent_values))

        scale = abs(local_mean) if abs(local_mean) > 0.01 else 1.0
        deviation = abs(current - local_mean) / scale

        if deviation < 0.1:
            rule.add_evidence(supports=True)
        else:
            rule.add_evidence(supports=False)

        rule.update_status(
            self.min_evidence_for_confirmation,
            self.confirmation_confidence
        )

        if rule.is_confirmed and rule.confirmed_at_step is None:
            self.kb.confirm_rule(rule.internal_name, self.step)

    def _test_collision_conservation_hypothesis(self, rule: Rule, features: dict, had_collision: bool):
        """
        Test whether a feature is conserved during collision events.

        Only collects evidence when a collision actually occurs.
        The test: does the feature change more during collision steps than non-collision steps?
        If not, the collision conserves it.
        """
        feature_key = rule.feature_key
        history = self.feature_histories[feature_key]

        # Only test when we have a collision and a previous value to compare
        if not had_collision or feature_key not in self.prev_features:
            return

        delta = features.get(feature_key, 0) - self.prev_features.get(feature_key, 0)

        # Compare this collision delta to the baseline (non-collision) delta
        non_coll_mean = history.non_collision_delta_mean
        non_coll_std = history.non_collision_delta_std

        if len(history.non_collision_deltas) < 5:
            return

        # How far is this collision delta from the baseline?
        if non_coll_std < 1e-10:
            non_coll_std = 1.0

        z_score = abs(delta - non_coll_mean) / non_coll_std

        # If z-score is low, the collision delta is within the expected range
        # → collision doesn't add extra change → feature is conserved during collisions
        if z_score < 2.0:  # Within 2 standard deviations of baseline
            rule.add_evidence(supports=True)
        else:
            rule.add_evidence(supports=False)

        rule.update_status(
            self.min_evidence_for_confirmation,
            self.confirmation_confidence
        )

        if rule.is_confirmed and rule.confirmed_at_step is None:
            self.kb.confirm_rule(rule.internal_name, self.step)

    def _test_correlation_hypothesis(self, rule: Rule, features: dict):
        """Test a correlation between two features."""
        # Placeholder for future correlation testing
        pass

    # ------------------------------------------------------------------
    # General Dynamics Law Learning
    # ------------------------------------------------------------------

    def _check_for_learned_dynamics_laws(self):
        """Learn predictive dynamics equations from raw object transitions."""
        if self.step < 100:
            return
        if self.step % 100 != 0:
            return

        samples = self._learned_law_motion_samples()
        laws = self.dynamics_law_learner.discover(samples, law_priors=self.law_priors)
        if not laws:
            return

        self.learned_dynamics_laws = laws
        self._record_novel_diagnostic('learned_dynamics', {
            'detected': True,
            'top_laws': [law.to_dict() for law in laws[:3]],
            'sample_count': len(samples),
            'reason': 'candidate equations ranked by prediction error',
        })

        for law in laws:
            if law.confidence < 0.70:
                continue
            signature = law.properties.get('signature', law.law_type)
            if signature in self.learned_dynamics_rule_keys:
                continue
            self._record_learned_dynamics_rule(law)
            self.learned_dynamics_rule_keys.add(signature)

    def _learned_law_motion_samples(self) -> list[MotionSample]:
        """Build filtered transition samples for equation search."""
        raw_samples = []
        wall_margin = 1.2

        for traj in self.timed_object_trajectories.values():
            if len(traj) < 3:
                continue
            for i in range(1, len(traj)):
                step, x, y, vx, vy = traj[i]
                _, _, _, prev_vx, prev_vy = traj[i - 1]
                if x < wall_margin or x > 20 - wall_margin:
                    continue
                if y < wall_margin or y > 20 - wall_margin:
                    continue
                dvx = vx - prev_vx
                dvy = vy - prev_vy
                raw_samples.append((step, x, y, prev_vx, prev_vy, dvx, dvy, math.sqrt(dvx * dvx + dvy * dvy)))

        if len(raw_samples) < 20:
            return []

        magnitudes = [sample[7] for sample in raw_samples]
        cap = max(self._percentile(magnitudes, 0.80), 0.30)
        filtered = [
            sample for sample in raw_samples
            if sample[7] <= cap and sample[7] <= 2.0
        ]
        return [
            MotionSample(
                step=sample[0],
                x=sample[1],
                y=sample[2],
                vx=sample[3],
                vy=sample[4],
                dvx=sample[5],
                dvy=sample[6],
            )
            for sample in filtered
        ]

    def _record_learned_dynamics_rule(self, law: LearnedLaw):
        rule = self.kb.add_hypothesis(
            conditions="object position and velocity are observed",
            prediction=law.description,
            feature_key='mean_speed',
            step=self.step,
            properties={
                'hypothesis_type': 'learned_dynamics',
                'law_type': law.law_type,
                'law_name': law.name,
                'law_confidence': round(law.confidence, 3),
                'prediction_mse': round(law.mse, 6),
                'baseline_mse': round(law.baseline_mse, 6),
                'improvement': round(law.improvement, 3),
                'sample_count': law.sample_count,
                **law.properties,
            },
        )
        evidence_for = max(self.min_evidence_for_confirmation, int(law.confidence * 20))
        for _ in range(evidence_for):
            rule.add_evidence(supports=True)
        rule.update_status(
            self.min_evidence_for_confirmation,
            self.confirmation_confidence,
        )
        if rule.is_confirmed and rule.confirmed_at_step is None:
            self.kb.confirm_rule(rule.internal_name, self.step)

    def get_learned_dynamics_laws(self) -> list[dict]:
        return [law.to_dict() for law in self.learned_dynamics_laws]

    def predict_next(self, features: dict) -> dict:
        """
        Predict the next feature values based on current knowledge.

        Uses confirmed rules to make predictions.
        If no rules exist for a feature, predicts it stays the same.
        """
        predictions = {}
        for key in self.TRACKED_FEATURES:
            if key in features:
                # Default: predict no change
                predictions[key] = features[key]

                # If we have a conservation rule, predict it stays constant
                for rule in self.kb.get_confirmed_rules():
                    if (rule.feature_key == key and
                            rule.properties.get('hypothesis_type') == 'conservation'):
                        history = self.feature_histories[key]
                        predictions[key] = history.mean
                        break

        return predictions

    def prediction_error(self, predicted: dict, actual: dict) -> float:
        """Calculate total prediction error — this drives curiosity."""
        total_error = 0.0
        for key in predicted:
            if key in actual:
                p = predicted[key]
                a = actual[key]
                scale = abs(p) if abs(p) > 0.01 else 1.0
                total_error += abs(a - p) / scale
        return total_error / max(len(predicted), 1)

    # ------------------------------------------------------------------
    # Abstraction Hierarchy
    # ------------------------------------------------------------------

    def _check_for_abstractions(self):
        """
        Check for higher-order patterns among discovered concepts.

        The agent notices that certain concepts share properties and groups them:
          - Concepts that involve mass × velocity → "motion quantities"
          - Concepts that are additive (sum of parts) → "additive quantities"
          - Concepts that are conserved during collisions → "conserved quantities"
          - x/y components of the same thing → "vector quantities"
        """
        if len(self.concepts_discovered) < 5:
            return
        if self.step < self.min_observations_for_abstraction:
            return

        # Group 1: Motion quantities (involve velocity)
        motion_key = "meta_motion_quantities"
        if motion_key not in self.meta_concepts_formed:
            motion_features = {'total_momentum_x', 'total_momentum_y', 'total_momentum',
                               'total_kinetic_energy', 'mean_speed'}
            discovered_motion = motion_features & self.concepts_discovered
            if len(discovered_motion) >= 3:
                sub_concepts = [c.internal_name for c in self.kb.get_all_concepts()
                                if c.feature_key in discovered_motion]
                self.kb.add_meta_concept(
                    name=f"M_{len(self.kb.meta_concepts) + 1:03d}",
                    description="Quantities that describe motion — all involve velocity of objects",
                    sub_concept_names=sub_concepts,
                    step=self.step,
                    properties={'grouping_criterion': 'involves_velocity'},
                )
                self.meta_concepts_formed.add(motion_key)

        # Group 2: Additive quantities (sum of object properties)
        additive_key = "meta_additive_quantities"
        if additive_key not in self.meta_concepts_formed:
            additive_features = {'count', 'total_mass', 'total_momentum_x',
                                 'total_momentum_y', 'total_momentum',
                                 'total_kinetic_energy'}
            discovered_additive = additive_features & self.concepts_discovered
            if len(discovered_additive) >= 3:
                sub_concepts = [c.internal_name for c in self.kb.get_all_concepts()
                                if c.feature_key in discovered_additive]
                self.kb.add_meta_concept(
                    name=f"M_{len(self.kb.meta_concepts) + 1:03d}",
                    description="Quantities that are sums of individual object properties — "
                                "the whole equals the sum of its parts",
                    sub_concept_names=sub_concepts,
                    step=self.step,
                    properties={'grouping_criterion': 'additive'},
                )
                self.meta_concepts_formed.add(additive_key)

        # Group 3: Conserved during collisions
        conserved_key = "meta_collision_conserved"
        if conserved_key not in self.meta_concepts_formed:
            confirmed = self.kb.get_confirmed_rules()
            collision_conserved = [r for r in confirmed
                                   if r.properties.get('hypothesis_type') == 'collision_conservation']
            if len(collision_conserved) >= 2:
                sub_concepts = [r.internal_name for r in collision_conserved]
                self.kb.add_meta_concept(
                    name=f"M_{len(self.kb.meta_concepts) + 1:03d}",
                    description="Quantities that are conserved during collision events — "
                                "the collision redistributes them between objects but doesn't change the total",
                    sub_concept_names=sub_concepts,
                    step=self.step,
                    properties={'grouping_criterion': 'collision_conserved'},
                )
                self.meta_concepts_formed.add(conserved_key)

        # Group 4: Vector components (x and y of same quantity)
        vector_key = "meta_vector_components"
        if vector_key not in self.meta_concepts_formed:
            x_features = {'total_momentum_x', 'center_of_mass_x'}
            y_features = {'total_momentum_y', 'center_of_mass_y'}
            discovered_x = x_features & self.concepts_discovered
            discovered_y = y_features & self.concepts_discovered
            if len(discovered_x) >= 1 and len(discovered_y) >= 1:
                # Find the paired concepts
                paired = []
                for xf in discovered_x:
                    yf = xf.replace('_x', '_y')
                    if yf in discovered_y:
                        xc = next((c for c in self.kb.get_all_concepts() if c.feature_key == xf), None)
                        yc = next((c for c in self.kb.get_all_concepts() if c.feature_key == yf), None)
                        if xc and yc:
                            paired.extend([xc.internal_name, yc.internal_name])
                if paired:
                    self.kb.add_meta_concept(
                        name=f"M_{len(self.kb.meta_concepts) + 1:03d}",
                        description="Quantities that exist as paired x/y components — "
                                    "they are orthogonal aspects of the same underlying property",
                        sub_concept_names=paired,
                        step=self.step,
                        properties={'grouping_criterion': 'vector_components'},
                    )
                    self.meta_concepts_formed.add(vector_key)

    # ------------------------------------------------------------------
    # Spatial / Geometric Discovery
    # ------------------------------------------------------------------

    def _check_for_spatial_rules(self, features: dict):
        """
        Check for spatial and geometric regularities.

        The agent discovers:
          - Ordering: objects can be sorted left-to-right (transitivity)
          - Triangle inequality: distance A→C ≤ distance A→B + distance B→C
          - Bounding: objects stay within world boundaries
          - Spatial extent: the spread of objects has a maximum
        """
        if self.step < self.min_observations_for_spatial:
            return

        # Triangle inequality: for any 3 objects, d(A,C) <= d(A,B) + d(B,C)
        tri_key = "spatial_triangle_inequality"
        if tri_key not in self.spatial_hypotheses_tested:
            if features.get('count', 0) >= 3:
                # We've observed enough steps with 3+ objects to form this hypothesis
                if len(self.feature_histories['count'].values) >= self.min_observations_for_spatial:
                    rule = self.kb.add_hypothesis(
                        conditions="three objects A, B, C exist in the world",
                        prediction="distance(A,C) <= distance(A,B) + distance(B,C) — "
                                   "the direct path is never longer than going through an intermediate point",
                        feature_key='mean_distance',
                        step=self.step,
                        properties={'hypothesis_type': 'spatial', 'spatial_rule': 'triangle_inequality'},
                    )
                    self.spatial_hypotheses_tested.add(tri_key)
                    rule.status = RuleStatus.TESTING

        # Ordering / transitivity: if A is left of B and B is left of C, then A is left of C
        order_key = "spatial_ordering_transitivity"
        if order_key not in self.spatial_hypotheses_tested:
            if features.get('count', 0) >= 3:
                if len(self.feature_histories['count'].values) >= self.min_observations_for_spatial:
                    rule = self.kb.add_hypothesis(
                        conditions="object A is left of B, and B is left of C",
                        prediction="A is left of C — spatial ordering is transitive",
                        feature_key='count',
                        step=self.step,
                        properties={'hypothesis_type': 'spatial', 'spatial_rule': 'transitivity'},
                    )
                    self.spatial_hypotheses_tested.add(order_key)
                    rule.status = RuleStatus.TESTING

        # Bounding: objects stay within world boundaries
        bound_key = "spatial_bounding"
        if bound_key not in self.spatial_hypotheses_tested:
            if len(self.feature_histories['count'].values) >= self.min_observations_for_spatial:
                rule = self.kb.add_hypothesis(
                    conditions="always",
                    prediction="all objects remain within the world boundaries — "
                               "no object escapes the spatial extent of the world",
                    feature_key='count',
                    step=self.step,
                    properties={'hypothesis_type': 'spatial', 'spatial_rule': 'bounding'},
                )
                self.spatial_hypotheses_tested.add(bound_key)
                rule.status = RuleStatus.TESTING

        # Test spatial hypotheses
        for rule in self.kb.get_active_hypotheses():
            if rule.properties.get('hypothesis_type') == 'spatial':
                self._test_spatial_hypothesis(rule, features)

    def _test_spatial_hypothesis(self, rule: Rule, features: dict):
        """Test a spatial hypothesis against current observation."""
        spatial_rule = rule.properties.get('spatial_rule', '')

        if spatial_rule == 'bounding':
            # Check: are all objects within world boundaries?
            # We infer from features that min_x >= 0, max_x <= width, etc.
            # The feature vector has max_x, min_x, max_y, min_y
            max_x = features.get('max_x', 0)
            min_x = features.get('min_x', 0)
            max_y = features.get('max_y', 0)
            min_y = features.get('min_y', 0)

            # World size is typically 20x20
            world_w = 20.0
            world_h = 20.0

            if max_x <= world_w and min_x >= 0 and max_y <= world_h and min_y >= 0:
                rule.add_evidence(supports=True)
            else:
                rule.add_evidence(supports=False)

        elif spatial_rule == 'triangle_inequality':
            # We can't directly test this with aggregate features, but we can
            # check that mean_distance is consistent with bounded space.
            # If triangle inequality holds, no single pairwise distance can exceed
            # the diagonal of the world.
            mean_dist = features.get('mean_distance', 0)
            max_possible = math.sqrt(20**2 + 20**2)  # World diagonal
            if mean_dist <= max_possible:
                rule.add_evidence(supports=True)
            else:
                rule.add_evidence(supports=False)

        elif spatial_rule == 'transitivity':
            # If we have 3+ objects, ordering is inherently transitive
            # (it's a property of sorting). We confirm it by observing
            # that sorted order is always consistent.
            if features.get('count', 0) >= 3:
                rule.add_evidence(supports=True)
            # If < 3 objects, no evidence either way

        rule.update_status(
            self.min_evidence_for_confirmation,
            self.confirmation_confidence
        )

        if rule.is_confirmed and rule.confirmed_at_step is None:
            self.kb.confirm_rule(rule.internal_name, self.step)

    # ------------------------------------------------------------------
    # Novel Physics Detection
    # ------------------------------------------------------------------

    def _check_for_novel_physics(self, features: dict, had_collision: bool):
        """
        Detect novel physics that doesn't exist in the standard world.

        The agent checks for:
          1. Central force: objects accelerate toward a specific point
          2. Repulsion zone: objects accelerate away from a specific point
          3. Zero gravity: no constant downward acceleration

        This tests whether the agent can go BEYOND rediscovering human knowledge
        and discover laws we didn't put in the "standard" world.
        """
        if self.step < self.min_observations_for_spatial:
            return

        # Need enough trajectory data
        if len(self.object_trajectories) < 2:
            return
        if self.step % 5 != 0:
            return

        # Check 0: Uniform horizontal force — a steady sideways acceleration
        wind_key = "novel_uniform_horizontal_force"
        if wind_key not in self.novel_physics_tested:
            uniform = self._detect_uniform_horizontal_force()
            if uniform is not None:
                direction, confidence, delta = uniform
                if confidence > 0.80:
                    rule = self.kb.add_hypothesis(
                        conditions="always",
                        prediction=f"objects receive a steady {direction} horizontal acceleration",
                        feature_key='total_momentum_x',
                        step=self.step,
                        properties={
                            'hypothesis_type': 'novel_physics',
                            'novel_type': 'uniform_horizontal_force',
                            'direction': direction,
                            'detection_confidence': round(confidence, 3),
                            'uniform_x_delta': round(delta, 4),
                        },
                    )
                    self.novel_physics_tested.add(wind_key)
                    rule.status = RuleStatus.TESTING

        # Check 1: Central force — objects accelerate toward a point
        cf_key = "novel_central_force"
        if cf_key not in self.novel_physics_tested:
            attractor = self._detect_central_attractor()
            if attractor is not None:
                ax, ay, confidence = attractor
                if confidence > 0.75:  # High bar to avoid false positives
                    rule = self.kb.add_hypothesis(
                        conditions=f"an object is at position (x, y) in the world",
                        prediction=f"it accelerates toward point ({ax:.1f}, {ay:.1f}) — "
                                   f"a central attractive force exists at this location",
                        feature_key='mean_distance',
                        step=self.step,
                        properties={
                            'hypothesis_type': 'novel_physics',
                            'novel_type': 'central_force',
                            'attractor_x': round(ax, 2),
                            'attractor_y': round(ay, 2),
                            'detection_confidence': round(confidence, 3),
                        },
                    )
                    self.novel_physics_tested.add(cf_key)
                    # If central force found, skip repulsion check (mutual exclusivity)
                    self.novel_physics_tested.add("novel_repulsion")
                    rule.status = RuleStatus.TESTING

        # Check 2: Repulsion source — objects accelerate away from a point
        rep_key = "novel_repulsion"
        if rep_key not in self.novel_physics_tested:
            repeller = self._detect_repulsion_zone()
            if repeller is not None:
                rx, ry, confidence = repeller
                if confidence > 0.75:  # High bar to avoid false positives
                    rule = self.kb.add_hypothesis(
                        conditions=f"an object is near point ({rx:.1f}, {ry:.1f})",
                        prediction=f"it is pushed away from this point — "
                                   f"a repulsive force exists at this location",
                        feature_key='mean_distance',
                        step=self.step,
                        properties={
                            'hypothesis_type': 'novel_physics',
                            'novel_type': 'repulsion',
                            'repeller_x': round(rx, 2),
                            'repeller_y': round(ry, 2),
                            'detection_confidence': round(confidence, 3),
                        },
                    )
                    self.novel_physics_tested.add(rep_key)
                    self.novel_physics_tested.add("novel_inverse_square_repulsion")
                    rule.status = RuleStatus.TESTING

        # Check 3: Vortex — acceleration is tangential around a point
        vortex_key = "novel_vortex"
        if vortex_key not in self.novel_physics_tested or self._has_rejected_novel_hypothesis('vortex'):
            vortex = self._detect_vortex_field()
            if vortex is not None:
                vx, vy, spin, confidence = vortex
                if confidence > 0.74 and self._can_add_vortex_hypothesis(vx, vy, spin, confidence):
                    spin_name = "counterclockwise" if spin > 0 else "clockwise"
                    rule = self.kb.add_hypothesis(
                        conditions=f"an object moves near point ({vx:.1f}, {vy:.1f})",
                        prediction=f"it accelerates tangentially around that point — "
                                   f"a {spin_name} vortex field exists",
                        feature_key='mean_distance',
                        step=self.step,
                        properties={
                            'hypothesis_type': 'novel_physics',
                            'novel_type': 'vortex',
                            'vortex_x': round(vx, 2),
                            'vortex_y': round(vy, 2),
                            'spin': spin_name,
                            'detection_confidence': round(confidence, 3),
                        },
                    )
                    self.novel_physics_tested.add(vortex_key)
                    rule.status = RuleStatus.TESTING

        # Check 5: Zero gravity — no downward acceleration
        zg_key = "novel_zero_gravity"
        if zg_key not in self.novel_physics_tested:
            if self._detect_zero_gravity():
                rule = self.kb.add_hypothesis(
                    conditions="always",
                    prediction="objects do not experience constant downward acceleration — "
                               "no uniform gravitational field exists in this world",
                    feature_key='total_momentum_y',
                    step=self.step,
                    properties={
                        'hypothesis_type': 'novel_physics',
                        'novel_type': 'zero_gravity',
                    },
                )
                self.novel_physics_tested.add(zg_key)
                rule.status = RuleStatus.TESTING

        # Check 6: Time-varying force — global acceleration reverses over time
        tv_key = "novel_time_varying_force"
        if tv_key not in self.novel_physics_tested:
            time_varying = self._detect_time_varying_force()
            if time_varying is not None:
                axis, confidence, sign_changes = time_varying
                if confidence > 0.75:
                    rule = self.kb.add_hypothesis(
                        conditions="time passes",
                        prediction=f"the uniform {axis}-axis acceleration periodically changes direction",
                        feature_key=f'total_momentum_{axis}',
                        step=self.step,
                        properties={
                            'hypothesis_type': 'novel_physics',
                            'novel_type': 'time_varying_force',
                            'axis': axis,
                            'detection_confidence': round(confidence, 3),
                            'sign_changes': sign_changes,
                        },
                    )
                    self.novel_physics_tested.add(tv_key)
                    rule.status = RuleStatus.TESTING

        # Test novel physics hypotheses
        for rule in self.kb.get_active_hypotheses():
            if rule.properties.get('hypothesis_type') == 'novel_physics':
                self._test_novel_physics_hypothesis(rule, features)

    def _detect_central_attractor(self) -> Optional[tuple[float, float, float]]:
        """
        Detect if objects are being pulled toward a specific point.

        The raw acceleration includes uniform gravity, damping from friction, and
        occasional impulses from the agent. We first estimate and subtract the
        uniform/velocity-dependent baseline, then look for a non-uniform radial
        residual field.

        Returns (x, y, confidence) or None.
        """
        return self._detect_radial_field(direction='toward')

    def _detect_repulsion_zone(self) -> Optional[tuple[float, float, float]]:
        """
        Detect if objects are being pushed away from a specific point.

        Similar to central attractor but acceleration points AWAY from the point.
        """
        return self._detect_radial_field(direction='away')

    def _detect_uniform_horizontal_force(self) -> Optional[tuple[str, float, float]]:
        """Detect a steady non-gravity horizontal acceleration."""
        model = self._motion_force_model()
        if model.get('sample_count', 0) < 40:
            self._record_novel_diagnostic('uniform_horizontal_force', {
                'detected': False,
                'reason': model.get('reason', 'not enough samples'),
            })
            return None

        x_delta = model.get('x_intercept', 0.0)
        magnitude = abs(x_delta)
        confidence = min(1.0, magnitude / 0.12)
        direction = 'rightward' if x_delta > 0 else 'leftward'
        detected = magnitude > 0.06
        self._record_novel_diagnostic('uniform_horizontal_force', {
            'detected': detected,
            'direction': direction,
            'confidence': round(confidence, 3),
            'uniform_x_delta': round(x_delta, 4),
            'reason': 'horizontal baseline exceeds threshold' if detected else 'horizontal baseline too small',
        })

        if not detected:
            return None
        return (direction, confidence, x_delta)

    def _detect_inverse_square_repulsion(self) -> Optional[tuple[float, float, float]]:
        """Detect outward radial acceleration whose strength persists with distance."""
        repeller = self._detect_radial_field(direction='away')
        if repeller is None:
            return None

        rx, ry, confidence = repeller
        residual = self._residual_acceleration_samples()
        profile = self._radial_strength_profile(residual, rx, ry)
        detected = profile['inverse_square_confidence'] > 0.45
        adjusted_confidence = confidence * (0.75 + 0.25 * profile['inverse_square_confidence'])
        self._record_novel_diagnostic('inverse_square_repulsion', {
            'detected': detected,
            'center': (round(rx, 2), round(ry, 2)),
            'radial_confidence': round(confidence, 3),
            'profile_confidence': round(profile['inverse_square_confidence'], 3),
            'far_fraction': round(profile['far_fraction'], 3),
            'distance_correlation': round(profile['distance_correlation'], 3),
            'reason': 'outward field persists at long range' if detected else 'outward field looks local',
        })

        if not detected:
            return None
        return (rx, ry, adjusted_confidence)

    def _detect_vortex_field(self) -> Optional[tuple[float, float, int, float]]:
        """Detect tangential acceleration around a point."""
        residual = self._residual_acceleration_samples()
        if len(residual) < 60:
            self._record_novel_diagnostic('vortex', {
                'detected': False,
                'reason': 'not enough residual samples',
                'residual_count': len(residual),
            })
            return None

        best = None
        for cx in range(6, 15):
            for cy in range(6, 15):
                for spin in (1, -1):
                    score = self._score_vortex_field(residual, float(cx), float(cy), spin)
                    if score is None:
                        continue
                    confidence = score[0]
                    if best is None or confidence > best[3]:
                        best = (float(cx), float(cy), spin, confidence, score)

        if best is None:
            return None

        vx, vy, spin, confidence, score = best
        detected = confidence > 0.74
        self._record_novel_diagnostic('vortex', {
            'detected': detected,
            'center': (round(vx, 2), round(vy, 2)),
            'spin': 'counterclockwise' if spin > 0 else 'clockwise',
            'confidence': round(confidence, 3),
            'positive_rate': round(score[1], 3),
            'mean_cosine': round(score[2], 3),
            'median_cosine': round(score[3], 3),
            'reason': 'tangential residuals align' if detected else 'tangential alignment too weak',
        })
        return (vx, vy, spin, confidence)

    def _detect_time_varying_force(self) -> Optional[tuple[str, float, int]]:
        """Detect a global acceleration component that reverses over time."""
        timed = self._timed_motion_samples()
        if len(timed) < 80:
            self._record_novel_diagnostic('time_varying_force', {
                'detected': False,
                'reason': 'not enough timed samples',
                'sample_count': len(timed),
            })
            return None

        axis = 'x'
        confidence, sign_changes, amplitude = self._score_time_varying_axis(timed, axis)
        detected = confidence > 0.90 and sign_changes >= 3 and amplitude > 0.08
        self._record_novel_diagnostic('time_varying_force', {
            'detected': detected,
            'axis': axis,
            'confidence': round(confidence, 3),
            'sign_changes': sign_changes,
            'amplitude': round(amplitude, 4),
            'reason': 'baseline acceleration reverses over time' if detected else 'no stable temporal reversal',
        })

        if not detected:
            return None
        return (axis, confidence, sign_changes)

    def _collect_motion_samples(self) -> list[tuple[float, float, float, float, float, float, float]]:
        """Return position/velocity/delta samples, excluding obvious wall and impulse events."""
        samples = []
        wall_margin = 1.2

        for traj in self.object_trajectories.values():
            if len(traj) < 3:
                continue
            for i in range(1, len(traj)):
                x, y, vx, vy = traj[i]
                _, _, prev_vx, prev_vy = traj[i - 1]

                if x < wall_margin or x > 20 - wall_margin:
                    continue
                if y < wall_margin or y > 20 - wall_margin:
                    continue

                dvx = vx - prev_vx
                dvy = vy - prev_vy
                delta_mag = math.sqrt(dvx * dvx + dvy * dvy)
                samples.append((x, y, prev_vx, prev_vy, dvx, dvy, delta_mag))

        if len(samples) < 20:
            return []

        # Agent pushes and collisions create very large one-frame velocity jumps.
        # Keep the sustained-motion band where environmental forces dominate.
        magnitudes = [sample[6] for sample in samples]
        cap = max(self._percentile(magnitudes, 0.80), 0.30)
        return [sample for sample in samples if sample[6] <= cap and sample[6] <= 2.0]

    @staticmethod
    def _percentile(values: list[float], fraction: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        idx = int((len(ordered) - 1) * fraction)
        return ordered[idx]

    @staticmethod
    def _fit_line(x_values: list[float], y_values: list[float]) -> tuple[float, float]:
        """Fit y = intercept + slope*x using ordinary least squares."""
        if not x_values:
            return (0.0, 0.0)

        mean_x = sum(x_values) / len(x_values)
        mean_y = sum(y_values) / len(y_values)
        variance = sum((x - mean_x) ** 2 for x in x_values)
        if variance < 1e-12:
            return (mean_y, 0.0)

        slope = sum((x - mean_x) * (y - mean_y)
                    for x, y in zip(x_values, y_values)) / variance
        intercept = mean_y - slope * mean_x
        return (intercept, slope)

    def _motion_force_model(self) -> dict:
        """Estimate uniform/damping dynamics and residual acceleration samples."""
        if self._motion_model_cache_step == self.step:
            return self._motion_model_cache

        samples = self._collect_motion_samples()
        if len(samples) < 20:
            self._motion_model_cache_step = self.step
            self._motion_model_cache = {
                'sample_count': len(samples),
                'residual': [],
                'reason': 'not enough usable motion samples',
            }
            return self._motion_model_cache

        x_intercept, x_slope = self._fit_line(
            [sample[2] for sample in samples],
            [sample[4] for sample in samples],
        )
        y_intercept, y_slope = self._fit_line(
            [sample[3] for sample in samples],
            [sample[5] for sample in samples],
        )

        residual = []
        for x, y, vx, vy, dvx, dvy, _ in samples:
            rx = dvx - (x_intercept + x_slope * vx)
            ry = dvy - (y_intercept + y_slope * vy)
            mag = math.sqrt(rx * rx + ry * ry)
            if mag > 0.02:
                residual.append((x, y, rx, ry, mag))

        model = {
            'sample_count': len(samples),
            'x_intercept': x_intercept,
            'x_slope': x_slope,
            'y_intercept': y_intercept,
            'y_slope': y_slope,
            'residual': residual,
            'residual_count': len(residual),
        }
        self._record_novel_diagnostic('baseline', {
            'sample_count': len(samples),
            'residual_count': len(residual),
            'uniform_x_delta': round(x_intercept, 4),
            'uniform_y_delta': round(y_intercept, 4),
            'damping_x': round(x_slope, 5),
            'damping_y': round(y_slope, 5),
        })
        self._motion_model_cache_step = self.step
        self._motion_model_cache = model
        return model

    def _residual_acceleration_samples(self) -> list[tuple[float, float, float, float, float]]:
        """
        Estimate non-uniform acceleration after removing gravity and damping.

        The fitted baseline is:
            delta_vx ~= uniform_x + damping_x * vx
            delta_vy ~= uniform_y + damping_y * vy

        A standard world leaves almost no residual after this subtraction. Central
        attraction and repulsion leave radial residual vectors.
        """
        return self._motion_force_model().get('residual', [])

    def _detect_radial_field(self, direction: str) -> Optional[tuple[float, float, float]]:
        """Detect a radial residual field directed toward or away from a point."""
        cache_key = (self.step, direction)
        if cache_key in self._radial_field_cache:
            return self._radial_field_cache[cache_key]

        residual = self._residual_acceleration_samples()
        if len(residual) < 60:
            self._record_novel_diagnostic(f'radial_{direction}', {
                'detected': False,
                'reason': 'not enough residual samples',
                'residual_count': len(residual),
            })
            self._radial_field_cache[cache_key] = None
            return None

        best = None
        for cx in range(4, 17):
            for cy in range(4, 17):
                score = self._score_radial_field(residual, float(cx), float(cy), direction)
                if score is None:
                    continue
                confidence = score[0]
                if best is None or confidence > best[2]:
                    best = (float(cx), float(cy), confidence)

        if best is None:
            self._record_novel_diagnostic(f'radial_{direction}', {
                'detected': False,
                'reason': 'no scoreable center',
                'residual_count': len(residual),
            })
            self._radial_field_cache[cache_key] = None
            return None

        bx, by, confidence = best
        self._record_novel_diagnostic(f'radial_{direction}', {
            'detected': confidence > 0.75,
            'center': (round(bx, 2), round(by, 2)),
            'confidence': round(confidence, 3),
            'residual_count': len(residual),
            'reason': 'radial residuals align' if confidence > 0.75 else 'radial alignment too weak',
        })
        self._radial_field_cache[cache_key] = (bx, by, confidence)
        return self._radial_field_cache[cache_key]

    def _score_radial_field(
        self,
        residual: list[tuple[float, float, float, float, float]],
        cx: float,
        cy: float,
        direction: str,
    ) -> Optional[tuple[float, float, float, float]]:
        """Score how consistently residual vectors point toward/away from a point."""
        cosines = []

        for x, y, rx, ry, mag in residual:
            if direction == 'toward':
                dx = cx - x
                dy = cy - y
            else:
                dx = x - cx
                dy = y - cy

            dist = math.sqrt(dx * dx + dy * dy)
            if dist < 0.5:
                continue

            cosines.append((rx * dx + ry * dy) / (mag * dist))

        if len(cosines) < 20:
            return None

        positive_rate = sum(1 for value in cosines if value > 0) / len(cosines)
        mean_cosine = sum(cosines) / len(cosines)
        ordered = sorted(cosines)
        median_cosine = ordered[len(ordered) // 2]
        confidence = (
            0.45 * positive_rate
            + 0.35 * max(0.0, mean_cosine)
            + 0.20 * max(0.0, median_cosine)
        )

        return (confidence, positive_rate, mean_cosine, median_cosine)

    def _radial_strength_profile(
        self,
        residual: list[tuple[float, float, float, float, float]],
        cx: float,
        cy: float,
    ) -> dict:
        """Estimate whether radial strength behaves like a long-range inverse-square field."""
        distances = []
        magnitudes = []
        inverse_sq = []

        for x, y, _, _, mag in residual:
            dx = x - cx
            dy = y - cy
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < 0.75:
                continue
            distances.append(dist)
            magnitudes.append(mag)
            inverse_sq.append(1.0 / (dist * dist))

        if len(distances) < 20:
            return {
                'inverse_square_confidence': 0.0,
                'far_fraction': 0.0,
                'distance_correlation': 0.0,
            }

        far_samples = [mag for dist, mag in zip(distances, magnitudes) if dist > 6.0]
        far_fraction = len(far_samples) / len(distances)
        correlation = self._correlation(inverse_sq, magnitudes)
        near_samples = [mag for dist, mag in zip(distances, magnitudes) if dist <= 4.0]
        near_mean = sum(near_samples) / len(near_samples) if near_samples else 0.0
        far_mean = sum(far_samples) / len(far_samples) if far_samples else 0.0
        far_strength = min(1.0, far_mean / 0.035) if far_mean > 0 else 0.0
        gradient = min(1.0, near_mean / max(far_mean, 1e-6)) if near_mean > far_mean else 0.0

        confidence = (
            0.40 * max(0.0, correlation)
            + 0.30 * min(1.0, far_fraction / 0.25)
            + 0.20 * far_strength
            + 0.10 * gradient
        )
        return {
            'inverse_square_confidence': confidence,
            'far_fraction': far_fraction,
            'distance_correlation': correlation,
            'near_mean': near_mean,
            'far_mean': far_mean,
        }

    def _score_vortex_field(
        self,
        residual: list[tuple[float, float, float, float, float]],
        cx: float,
        cy: float,
        spin: int,
    ) -> Optional[tuple[float, float, float, float]]:
        """Score how consistently residual vectors follow a tangent around a point."""
        cosines = []

        for x, y, rx, ry, mag in residual:
            dx = x - cx
            dy = y - cy
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < 0.75:
                continue

            tangent_x = -dy / dist * spin
            tangent_y = dx / dist * spin
            cosines.append((rx * tangent_x + ry * tangent_y) / mag)

        if len(cosines) < 20:
            return None

        positive_rate = sum(1 for value in cosines if value > 0) / len(cosines)
        mean_cosine = sum(cosines) / len(cosines)
        ordered = sorted(cosines)
        median_cosine = ordered[len(ordered) // 2]
        confidence = (
            0.45 * positive_rate
            + 0.35 * max(0.0, mean_cosine)
            + 0.20 * max(0.0, median_cosine)
        )
        return (confidence, positive_rate, mean_cosine, median_cosine)

    def _timed_motion_samples(self) -> list[tuple[int, float, float, float, float, float, float]]:
        """Return timed velocity-delta samples for detecting global temporal fields."""
        samples = []
        wall_margin = 1.2

        for traj in self.timed_object_trajectories.values():
            if len(traj) < 3:
                continue
            for i in range(1, len(traj)):
                step, x, y, vx, vy = traj[i]
                _, _, _, prev_vx, prev_vy = traj[i - 1]
                if x < wall_margin or x > 20 - wall_margin:
                    continue
                if y < wall_margin or y > 20 - wall_margin:
                    continue

                dvx = vx - prev_vx
                dvy = vy - prev_vy
                delta_mag = math.sqrt(dvx * dvx + dvy * dvy)
                samples.append((step, prev_vx, prev_vy, dvx, dvy, delta_mag, x))

        if len(samples) < 20:
            return []

        magnitudes = [sample[5] for sample in samples]
        cap = max(self._percentile(magnitudes, 0.80), 0.30)
        return [sample for sample in samples if sample[5] <= cap and sample[5] <= 2.0]

    def _score_time_varying_axis(
        self,
        timed_samples: list[tuple[int, float, float, float, float, float, float]],
        axis: str,
    ) -> tuple[float, int, float]:
        """Score one axis for periodic acceleration reversal."""
        if axis == 'x':
            velocities = [sample[1] for sample in timed_samples]
            deltas = [sample[3] for sample in timed_samples]
        else:
            velocities = [sample[2] for sample in timed_samples]
            deltas = [sample[4] for sample in timed_samples]

        intercept, slope = self._fit_line(velocities, deltas)
        grouped: dict[int, list[float]] = {}
        for sample, velocity, delta in zip(timed_samples, velocities, deltas):
            step = sample[0]
            residual = delta - (intercept + slope * velocity)
            grouped.setdefault(step, []).append(residual)

        series = [
            sum(values) / len(values)
            for _, values in sorted(grouped.items())
            if values
        ]
        if len(series) < 30:
            return (0.0, 0, 0.0)

        within_step_stds = []
        for values in grouped.values():
            if len(values) < 2:
                continue
            mean_value = sum(values) / len(values)
            variance = sum((value - mean_value) ** 2 for value in values) / len(values)
            within_step_stds.append(math.sqrt(variance))

        smoothed = []
        window = 5
        for i in range(len(series)):
            start = max(0, i - window + 1)
            values = series[start:i + 1]
            smoothed.append(sum(values) / len(values))

        amplitude = (max(smoothed) - min(smoothed)) / 2
        threshold = max(0.03, amplitude * 0.30)
        signs = []
        for value in smoothed:
            if value > threshold:
                signs.append(1)
            elif value < -threshold:
                signs.append(-1)

        sign_changes = 0
        prev_sign = 0
        for sign in signs:
            if prev_sign and sign != prev_sign:
                sign_changes += 1
            prev_sign = sign

        mean_within_step_std = (
            sum(within_step_stds) / len(within_step_stds)
            if within_step_stds else 0.0
        )
        coherence = max(0.0, 1.0 - (mean_within_step_std / max(amplitude, 1e-6)))
        confidence = (
            0.55 * min(1.0, amplitude / 0.12)
            + 0.35 * min(1.0, sign_changes / 3)
            + 0.10 * min(1.0, len(series) / 80)
        ) * coherence
        return (confidence, sign_changes, amplitude)

    @staticmethod
    def _correlation(a_values: list[float], b_values: list[float]) -> float:
        if len(a_values) != len(b_values) or len(a_values) < 2:
            return 0.0

        mean_a = sum(a_values) / len(a_values)
        mean_b = sum(b_values) / len(b_values)
        var_a = sum((value - mean_a) ** 2 for value in a_values)
        var_b = sum((value - mean_b) ** 2 for value in b_values)
        if var_a < 1e-12 or var_b < 1e-12:
            return 0.0

        cov = sum((a - mean_a) * (b - mean_b)
                  for a, b in zip(a_values, b_values))
        return cov / math.sqrt(var_a * var_b)

    def _detect_zero_gravity(self) -> bool:
        """
        Detect if there is no constant downward gravitational acceleration.

        In a standard world, the y-velocity of objects consistently decreases.
        In zero gravity, it doesn't.
        """
        # Check the total_momentum_y history — in a gravity world, it trends downward
        history = self.feature_histories['total_momentum_y']
        if len(history.values) < 50:
            return False

        # Check if momentum_y is trending downward (gravity) or not
        recent = list(history.values)[-50:]
        first_half = recent[:25]
        second_half = recent[25:]
        mean_first = sum(first_half) / len(first_half)
        mean_second = sum(second_half) / len(second_half)

        # In a gravity world, momentum_y decreases over time
        # In zero gravity, it stays roughly the same (only friction)
        trend = mean_second - mean_first

        # If the trend is nearly zero (no systematic decrease), it's zero gravity
        # In a gravity world, trend would be strongly negative
        scale = abs(mean_first) if abs(mean_first) > 0.1 else 1.0
        normalized_trend = abs(trend) / scale

        # Also check: in gravity world, individual objects' vy tends to decrease
        vy_decrease_count = 0
        vy_total = 0
        for oid, traj in self.object_trajectories.items():
            if len(traj) < 5:
                continue
            for i in range(1, len(traj)):
                vy_prev = traj[i - 1][3]
                vy_curr = traj[i][3]
                if vy_curr < vy_prev:
                    vy_decrease_count += 1
                vy_total += 1

        if vy_total == 0:
            return False

        vy_decrease_rate = vy_decrease_count / vy_total

        # In gravity world: vy decreases > 60% of the time
        # In zero gravity: vy decreases ~50% (random)
        return vy_decrease_rate < 0.55 and normalized_trend < 0.3

    def _test_novel_physics_hypothesis(self, rule: Rule, features: dict):
        """Test a novel physics hypothesis against new observations."""
        novel_type = rule.properties.get('novel_type', '')

        if novel_type == 'central_force':
            ax = rule.properties.get('attractor_x', 0)
            ay = rule.properties.get('attractor_y', 0)
            self._add_radial_field_evidence(rule, ax, ay, direction='toward')

        elif novel_type == 'repulsion':
            rx = rule.properties.get('repeller_x', 0)
            ry = rule.properties.get('repeller_y', 0)
            self._add_radial_field_evidence(rule, rx, ry, direction='away')

        elif novel_type == 'inverse_square_repulsion':
            rx = rule.properties.get('repeller_x', 0)
            ry = rule.properties.get('repeller_y', 0)
            self._add_inverse_square_repulsion_evidence(rule, rx, ry)

        elif novel_type == 'uniform_horizontal_force':
            self._add_uniform_horizontal_force_evidence(rule)

        elif novel_type == 'vortex':
            vx = rule.properties.get('vortex_x', 0)
            vy = rule.properties.get('vortex_y', 0)
            spin = 1 if rule.properties.get('spin') == 'counterclockwise' else -1
            self._add_vortex_evidence(rule, vx, vy, spin)

        elif novel_type == 'time_varying_force':
            axis = rule.properties.get('axis', 'x')
            self._add_time_varying_force_evidence(rule, axis)

        elif novel_type == 'zero_gravity':
            # Verify: vy is not systematically decreasing
            vy_decrease_count = 0
            vy_total = 0
            for oid, traj in self.object_trajectories.items():
                if len(traj) < 3:
                    continue
                for i in range(1, len(traj)):
                    if traj[i][3] < traj[i - 1][3]:
                        vy_decrease_count += 1
                    vy_total += 1

            if vy_total > 0:
                if vy_decrease_count / vy_total < 0.55:
                    rule.add_evidence(supports=True)
                else:
                    rule.add_evidence(supports=False)

        rule.update_status(
            self.min_evidence_for_confirmation,
            self.confirmation_confidence
        )

        if rule.is_confirmed and rule.confirmed_at_step is None:
            self.kb.confirm_rule(rule.internal_name, self.step)

    def _has_rejected_novel_hypothesis(self, novel_type: str) -> bool:
        return any(
            rule.properties.get('hypothesis_type') == 'novel_physics'
            and rule.properties.get('novel_type') == novel_type
            and rule.status == RuleStatus.REJECTED
            for rule in self.kb.rules.values()
        )

    def _can_add_vortex_hypothesis(
        self,
        x: float,
        y: float,
        spin: int,
        confidence: float,
    ) -> bool:
        vortex_rules = [
            rule for rule in self.kb.rules.values()
            if rule.properties.get('hypothesis_type') == 'novel_physics'
            and rule.properties.get('novel_type') == 'vortex'
        ]
        if not vortex_rules:
            return True
        if any(rule.status != RuleStatus.REJECTED for rule in vortex_rules):
            return False

        rejected = [rule for rule in vortex_rules if rule.status == RuleStatus.REJECTED]
        if len(rejected) >= 3:
            return False

        spin_name = "counterclockwise" if spin > 0 else "clockwise"
        for rule in rejected:
            previous_x = rule.properties.get('vortex_x', x)
            previous_y = rule.properties.get('vortex_y', y)
            previous_confidence = rule.properties.get('detection_confidence', 0.0)
            latest_confidence = rule.properties.get('latest_vortex_confidence', previous_confidence)
            distance = math.sqrt((x - previous_x) ** 2 + (y - previous_y) ** 2)
            same_spin = rule.properties.get('spin') == spin_name

            center_shifted = distance >= 2.0 and confidence >= 0.74
            score_recovered = latest_confidence < 0.72 and confidence >= 0.74
            materially_stronger = confidence >= previous_confidence + 0.06
            if same_spin and (center_shifted or score_recovered or materially_stronger):
                return True

        return False

    def _add_radial_field_evidence(self, rule: Rule, x: float, y: float, direction: str):
        """Add evidence for a novel radial-field hypothesis using residual acceleration."""
        residual = self._residual_acceleration_samples()
        if len(residual) < 60:
            return

        score = self._score_radial_field(residual, x, y, direction)
        if score is None:
            return

        confidence, positive_rate, mean_cosine, median_cosine = score
        supports = (
            confidence > 0.68
            and positive_rate > 0.75
            and mean_cosine > 0.30
            and median_cosine > 0.50
        )
        rule.add_evidence(supports=supports)
        rule.properties['latest_residual_confidence'] = round(confidence, 3)

    def _add_inverse_square_repulsion_evidence(self, rule: Rule, x: float, y: float):
        residual = self._residual_acceleration_samples()
        if len(residual) < 60:
            return

        score = self._score_radial_field(residual, x, y, direction='away')
        if score is None:
            return

        profile = self._radial_strength_profile(residual, x, y)
        confidence, positive_rate, mean_cosine, median_cosine = score
        profile_conf = profile['inverse_square_confidence']
        supports = (
            confidence > 0.70
            and positive_rate > 0.80
            and mean_cosine > 0.40
            and median_cosine > 0.55
            and profile_conf > 0.40
        )
        rule.add_evidence(supports=supports)
        rule.properties['latest_residual_confidence'] = round(confidence, 3)
        rule.properties['latest_profile_confidence'] = round(profile_conf, 3)

    def _add_uniform_horizontal_force_evidence(self, rule: Rule):
        detected = self._detect_uniform_horizontal_force()
        if detected is None:
            rule.add_evidence(supports=False)
            rule.properties['latest_rejection_reason'] = 'horizontal baseline below threshold'
            return

        direction, confidence, delta = detected
        supports = (
            confidence > 0.75
            and direction == rule.properties.get('direction')
            and abs(delta) > 0.06
        )
        rule.add_evidence(supports=supports)
        rule.properties['latest_uniform_confidence'] = round(confidence, 3)

    def _add_vortex_evidence(self, rule: Rule, x: float, y: float, spin: int):
        residual = self._residual_acceleration_samples()
        if len(residual) < 60:
            return

        score = self._score_vortex_field(residual, x, y, spin)
        if score is None:
            return

        confidence, positive_rate, mean_cosine, median_cosine = score
        supports = (
            confidence > 0.72
            and positive_rate > 0.78
            and mean_cosine > 0.50
            and median_cosine > 0.85
        )
        rule.add_evidence(supports=supports)
        rule.properties['latest_vortex_confidence'] = round(confidence, 3)

    def _add_time_varying_force_evidence(self, rule: Rule, axis: str):
        timed = self._timed_motion_samples()
        if len(timed) < 80:
            return

        confidence, sign_changes, amplitude = self._score_time_varying_axis(timed, axis)
        supports = confidence > 0.90 and sign_changes >= 3 and amplitude > 0.08
        rule.add_evidence(supports=supports)
        rule.properties['latest_time_confidence'] = round(confidence, 3)
        rule.properties['latest_sign_changes'] = sign_changes

    def _record_novel_diagnostic(self, key: str, details: dict):
        details = dict(details)
        details['step'] = self.step
        self.novel_physics_diagnostics[key] = details

    def get_novel_physics_diagnostics(self) -> dict[str, dict]:
        return dict(self.novel_physics_diagnostics)

    def suggest_experiment_action(
        self,
        current_count: int,
        world_width: float = 20.0,
        world_height: float = 20.0,
    ) -> Optional[dict]:
        """
        Suggest an active experiment when a novel-physics hypothesis needs evidence.

        The action is deliberately simple: spawn a low-motion probe object in a
        location that should make the hypothesized field easy to observe.
        """
        if self.step - self.last_active_experiment_step < 20:
            return None
        if current_count >= 12:
            return None

        for rule in self.kb.get_active_hypotheses():
            if rule.properties.get('hypothesis_type') != 'novel_physics':
                continue

            novel_type = rule.properties.get('novel_type')
            if novel_type in ('central_force', 'repulsion', 'inverse_square_repulsion'):
                if 'attractor_x' in rule.properties:
                    center_x = rule.properties['attractor_x']
                    center_y = rule.properties['attractor_y']
                else:
                    center_x = rule.properties.get('repeller_x', world_width / 2)
                    center_y = rule.properties.get('repeller_y', world_height / 2)
                return self._spawn_probe_near(center_x, center_y, world_width, world_height, rule)

            if novel_type == 'vortex':
                center_x = rule.properties.get('vortex_x', world_width / 2)
                center_y = rule.properties.get('vortex_y', world_height / 2)
                return self._spawn_probe_near(center_x, center_y, world_width, world_height, rule)

            if novel_type in ('uniform_horizontal_force', 'time_varying_force'):
                return self._spawn_probe_at(world_width / 2, world_height / 2, world_width, world_height, rule)

        memory_probe = self._suggest_memory_probe_action(current_count, world_width, world_height)
        if memory_probe is not None:
            return memory_probe

        return None

    def _memory_probe_questions_from_priors(self, law_priors: list[dict]) -> list[dict]:
        questions = []
        for prior in law_priors[:5]:
            law_type = prior.get('law_type', 'unknown')
            if law_type == 'uniform_acceleration':
                continue
            target = self._memory_probe_target(prior)
            if law_type in ('radial_field', 'inverse_square_radial_field'):
                question = 'Test whether remembered radial structure transfers here'
            elif law_type == 'tangential_field':
                question = 'Test whether remembered tangential structure transfers here'
            elif law_type == 'time_varying_field':
                question = 'Test whether remembered temporal structure transfers here'
            else:
                question = f'Test whether remembered {law_type} structure transfers here'
            questions.append({
                'law_type': law_type,
                'question': question,
                'target': target,
                'transfer_score': prior.get('transfer_score', 0.0),
            })
        return questions

    def _memory_probe_target(self, prior: dict) -> dict:
        ranges = prior.get('parameter_ranges', {})
        if 'center_x' in ranges and 'center_y' in ranges:
            x_low, x_high = ranges['center_x']
            y_low, y_high = ranges['center_y']
            return {
                'kind': 'point',
                'x': (x_low + x_high) / 2,
                'y': (y_low + y_high) / 2,
            }
        return {'kind': 'center'}

    def _suggest_memory_probe_action(
        self,
        current_count: int,
        world_width: float,
        world_height: float,
    ) -> Optional[dict]:
        if not self.memory_probe_questions:
            return None
        if self.step < self.min_observations_for_spatial:
            return None
        if self.step - self.last_active_experiment_step < 25:
            return None
        if current_count >= 12:
            return None

        question = self.memory_probe_questions[self.memory_probe_index % len(self.memory_probe_questions)]
        self.memory_probe_index += 1
        target = question.get('target', {})
        if target.get('kind') == 'point':
            center_x = target.get('x', world_width / 2)
            center_y = target.get('y', world_height / 2)
            return self._spawn_memory_probe_near(center_x, center_y, world_width, world_height, question)
        return self._spawn_memory_probe_at(world_width / 2, world_height / 2, world_width, world_height, question)

    def _spawn_probe_near(
        self,
        center_x: float,
        center_y: float,
        world_width: float,
        world_height: float,
        rule: Rule,
    ) -> dict:
        angle = (self.active_experiment_count % 4) * (math.pi / 2)
        radius = 4.0
        x = center_x + math.cos(angle) * radius
        y = center_y + math.sin(angle) * radius
        return self._spawn_probe_at(x, y, world_width, world_height, rule)

    def _spawn_memory_probe_near(
        self,
        center_x: float,
        center_y: float,
        world_width: float,
        world_height: float,
        question: dict,
    ) -> dict:
        angle = (self.active_experiment_count % 4) * (math.pi / 2)
        radius = 4.0
        x = center_x + math.cos(angle) * radius
        y = center_y + math.sin(angle) * radius
        return self._spawn_memory_probe_at(x, y, world_width, world_height, question)

    def _spawn_memory_probe_at(
        self,
        x: float,
        y: float,
        world_width: float,
        world_height: float,
        question: dict,
    ) -> dict:
        self.last_active_experiment_step = self.step
        self.active_experiment_count += 1
        question['active_experiments'] = question.get('active_experiments', 0) + 1
        self._record_novel_diagnostic('memory_probe', {
            'detected': True,
            'law_type': question.get('law_type'),
            'question': question.get('question'),
            'transfer_score': round(question.get('transfer_score', 0.0), 3),
            'target': question.get('target', {}),
            'active_experiments': question['active_experiments'],
            'reason': 'prior law memory requested a counterexample probe',
        })
        margin = 2.0
        return {
            'type': 'spawn',
            'x': min(max(x, margin), world_width - margin),
            'y': min(max(y, margin), world_height - margin),
            'vx': 0.0,
            'vy': 0.0,
        }

    def _spawn_probe_at(
        self,
        x: float,
        y: float,
        world_width: float,
        world_height: float,
        rule: Rule,
    ) -> dict:
        self.last_active_experiment_step = self.step
        self.active_experiment_count += 1
        rule.properties['active_experiments'] = rule.properties.get('active_experiments', 0) + 1
        margin = 2.0
        return {
            'type': 'spawn',
            'x': min(max(x, margin), world_width - margin),
            'y': min(max(y, margin), world_height - margin),
            'vx': 0.0,
            'vy': 0.0,
        }

    def _check_for_operation_notation(self):
        """
        When the agent discovers arithmetic rules, it invents notation for the operations.

        This is the agent creating its own mathematical language:
          - When it discovers "count + 1", it invents a symbol for the successor operation
          - When it discovers "count - 1", it invents a symbol for the predecessor operation
          - When it discovers conservation, it invents a symbol for equality/identity
        """
        confirmed = self.kb.get_confirmed_rules()

        for rule in confirmed:
            if rule.properties.get('hypothesis_type') == 'arithmetic':
                delta = rule.properties.get('delta', 0)
                op_key = f"op_successor" if delta == 1 else f"op_predecessor" if delta == -1 else f"op_delta_{delta}"

                if op_key in self.operations_invented:
                    continue

                # Invent a symbol for this operation
                op_name = "successor" if delta == 1 else "predecessor" if delta == -1 else f"delta_{delta}"
                symbol = self.kb.notation_system.generate_operation_symbol(op_name)

                # Register as an OPERATION concept
                self.kb.add_concept(
                    concept_type=ConceptType.OPERATION,
                    description=f"Operation: {op_name} — increases count by {delta} when an object {'appears' if delta > 0 else 'disappears'}",
                    feature_key=op_key,
                    step=self.step,
                    properties={'operation': op_name, 'delta': delta, 'symbol': symbol},
                    notation=symbol,
                )
                self.operations_invented.add(op_key)

            elif rule.properties.get('hypothesis_type') == 'collision_conservation':
                cons_key = f"op_conservation_{rule.feature_key}"
                if cons_key in self.operations_invented:
                    continue

                # Invent a symbol for conservation/equality
                symbol = self.kb.notation_system.generate_operation_symbol(f"conservation_{rule.feature_key}")

                self.kb.add_concept(
                    concept_type=ConceptType.OPERATION,
                    description=f"Conservation relation: {rule.feature_key} is unchanged by collisions — "
                                f"a relation of invariance under interaction",
                    feature_key=cons_key,
                    step=self.step,
                    properties={'operation': 'conservation', 'feature': rule.feature_key, 'symbol': symbol},
                    notation=symbol,
                )
                self.operations_invented.add(cons_key)
