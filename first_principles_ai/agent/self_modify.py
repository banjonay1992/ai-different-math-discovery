from __future__ import annotations

"""
Self-Modification — the agent inspects and improves its own reasoning process.

This is genuine self-improvement — not just "more data," but better METHODS.

The agent can:
  1. INSPECT its own reasoning: what hypotheses has it tested? How long did they take?
  2. DETECT inefficiencies: "I'm testing too many hypotheses at once" or "I'm not exploring enough"
  3. MODIFY its own parameters: adjust thresholds, change exploration rates
  4. EVALUATE the modification: did it improve discovery rate?

This is meta-learning: learning how to learn better.

Key insight: the agent's reasoning process is itself a system that can be
observed, hypothesized about, and improved — just like the physical world.
"""

import math
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict


@dataclass
class ReasoningMetric:
    """A metric about the agent's reasoning process."""
    name: str
    value: float
    step: int
    description: str = ""


@dataclass
class SelfModification:
    """A modification the agent made to its own reasoning process."""
    step: int
    parameter: str           # What was changed
    old_value: float         # Previous value
    new_value: float         # New value
    rationale: str           # Why the agent changed it
    discovery_rate_before: float  # Discoveries per step before change
    discovery_rate_after: float   # Discoveries per step after change
    improvement: float = 0.0      # Net improvement


class SelfModifier:
    """
    The agent's self-improvement engine.

    It monitors the agent's own reasoning process and makes adjustments
    to improve discovery efficiency.

    Parameters it can modify:
      - min_observations_for_concept: How many observations before forming a concept
      - min_evidence_for_confirmation: How much evidence to confirm a rule
      - confirmation_confidence: Confidence threshold for confirmation
      - exploration_rate (in curiosity): How much to explore vs exploit
      - min_observations_for_abstraction: When to start forming abstractions

    The agent uses a simple principle:
      - If discovery rate is too low → loosen thresholds (be more willing to form hypotheses)
      - If false positive rate is too high → tighten thresholds
      - If exploration isn't yielding discoveries → increase exploration
      - If exploration is wasting time → decrease exploration
    """

    def __init__(self, predictor, curiosity=None):
        self.predictor = predictor
        self.curiosity = curiosity

        # Track reasoning metrics over time
        self.metrics_history: list[list[ReasoningMetric]] = []
        self.modifications: list[SelfModification] = []

        # Discovery tracking
        self.discoveries_by_step: list[int] = []  # Steps at which discoveries were made
        self.last_check_step: int = 0
        self.last_check_count: int = 0

        # Modification cooldown — don't modify too frequently
        self.min_steps_between_modifications = 200
        self.last_modification_step: int = 0

        # Evaluation window
        self.evaluation_window = 200  # Steps to evaluate before/after a modification

    def record_discovery(self, step: int):
        """Record that a discovery was made at this step."""
        self.discoveries_by_step.append(step)

    def check_and_modify(self, step: int):
        """
        Inspect the reasoning process and make modifications if needed.

        Called periodically during the experiment.
        """
        # Record current metrics
        metrics = self._collect_metrics(step)
        self.metrics_history.append(metrics)

        # Don't modify too frequently
        if step - self.last_modification_step < self.min_steps_between_modifications:
            return

        # Need enough history to evaluate
        if len(self.discoveries_by_step) < 5:
            return

        # Calculate discovery rate in recent window
        recent_discoveries = [s for s in self.discoveries_by_step
                              if s > step - self.evaluation_window]
        discovery_rate = len(recent_discoveries) / self.evaluation_window

        # Calculate discovery rate in the previous window
        older_discoveries = [s for s in self.discoveries_by_step
                             if step - 2 * self.evaluation_window < s <= step - self.evaluation_window]
        older_rate = len(older_discoveries) / self.evaluation_window

        # Decision logic: what should we modify?

        # 1. If discovery rate has dropped, loosen thresholds
        if discovery_rate < older_rate * 0.5 and discovery_rate < 0.01:
            self._try_loosen_thresholds(step, discovery_rate, older_rate)

        # 2. If we have many rejected hypotheses, tighten thresholds
        rejected = sum(1 for r in self.predictor.kb.rules.values()
                       if r.status == self.predictor.kb.rules[list(self.predictor.kb.rules.keys())[0]].status.REJECTED
                       ) if self.predictor.kb.rules else 0
        total_rules = len(self.predictor.kb.rules)
        if total_rules > 10 and rejected / total_rules > 0.5:
            self._try_tighten_thresholds(step, discovery_rate)

        # 3. If exploration isn't yielding discoveries, adjust curiosity
        if self.curiosity and discovery_rate < 0.005 and step > 500:
            self._try_adjust_curiosity(step, discovery_rate)

        # 4. If we haven't formed abstractions yet, lower the abstraction threshold
        if (len(self.predictor.kb.get_meta_concepts()) == 0
                and step > self.predictor.min_observations_for_abstraction * 1.5):
            self._try_lower_abstraction_threshold(step)

    def _collect_metrics(self, step: int) -> list[ReasoningMetric]:
        """Collect metrics about the current reasoning process."""
        metrics = [
            ReasoningMetric("discovery_count", len(self.predictor.kb.discovery_log), step,
                            "Total discoveries made"),
            ReasoningMetric("concept_count", len(self.predictor.kb.concepts), step,
                            "Concepts discovered"),
            ReasoningMetric("confirmed_rules", len(self.predictor.kb.get_confirmed_rules()), step,
                            "Confirmed rules"),
            ReasoningMetric("active_hypotheses", len(self.predictor.kb.get_active_hypotheses()), step,
                            "Active hypotheses being tested"),
            ReasoningMetric("min_obs_for_concept", self.predictor.min_observations_for_concept, step,
                            "Threshold: observations needed to form a concept"),
            ReasoningMetric("min_evidence", self.predictor.min_evidence_for_confirmation, step,
                            "Threshold: evidence needed to confirm a rule"),
            ReasoningMetric("confirmation_conf", self.predictor.confirmation_confidence, step,
                            "Threshold: confidence needed to confirm"),
        ]

        if self.curiosity:
            metrics.append(ReasoningMetric("exploration_rate", self.curiosity.exploration_rate, step,
                                           "Current exploration rate"))

        return metrics

    def _try_loosen_thresholds(self, step: int, current_rate: float, older_rate: float):
        """Loosen discovery thresholds to encourage more hypothesis formation."""
        # Lower the minimum observations for concept formation
        old_val = self.predictor.min_observations_for_concept
        new_val = max(10, int(old_val * 0.7))

        if new_val != old_val:
            mod = SelfModification(
                step=step,
                parameter="min_observations_for_concept",
                old_value=old_val,
                new_value=new_val,
                rationale=f"Discovery rate dropped ({older_rate:.4f} → {current_rate:.4f}); "
                          f"loosening concept formation threshold to encourage more discoveries",
                discovery_rate_before=older_rate,
                discovery_rate_after=current_rate,
            )
            self.predictor.min_observations_for_concept = new_val
            self.modifications.append(mod)
            self.last_modification_step = step

    def _try_tighten_thresholds(self, step: int, discovery_rate: float):
        """Tighten thresholds to reduce false positives."""
        old_val = self.predictor.confirmation_confidence
        new_val = min(0.95, old_val + 0.05)

        if new_val != old_val:
            mod = SelfModification(
                step=step,
                parameter="confirmation_confidence",
                old_value=old_val,
                new_value=new_val,
                rationale=f"High rejection rate; tightening confirmation threshold "
                          f"to reduce false positives",
                discovery_rate_before=discovery_rate,
                discovery_rate_after=discovery_rate,
            )
            self.predictor.confirmation_confidence = new_val
            self.modifications.append(mod)
            self.last_modification_step = step

    def _try_adjust_curiosity(self, step: int, discovery_rate: float):
        """Adjust exploration rate if curiosity isn't yielding discoveries."""
        old_val = self.curiosity.exploration_rate
        # Increase exploration to try new things
        new_val = min(1.0, old_val * 1.3)

        if new_val != old_val and new_val > old_val + 0.01:
            mod = SelfModification(
                step=step,
                parameter="exploration_rate",
                old_value=old_val,
                new_value=new_val,
                rationale=f"Low discovery rate ({discovery_rate:.4f}); "
                          f"increasing exploration to seek novel situations",
                discovery_rate_before=discovery_rate,
                discovery_rate_after=discovery_rate,
            )
            self.curiosity.exploration_rate = new_val
            self.modifications.append(mod)
            self.last_modification_step = step

    def _try_lower_abstraction_threshold(self, step: int):
        """Lower the threshold for forming abstractions."""
        old_val = self.predictor.min_observations_for_abstraction
        new_val = max(30, int(old_val * 0.5))

        if new_val != old_val:
            mod = SelfModification(
                step=step,
                parameter="min_observations_for_abstraction",
                old_value=old_val,
                new_value=new_val,
                rationale=f"No abstractions formed yet after {step} steps; "
                          f"lowering threshold to encourage meta-concept formation",
                discovery_rate_before=0,
                discovery_rate_after=0,
            )
            self.predictor.min_observations_for_abstraction = new_val
            self.modifications.append(mod)
            self.last_modification_step = step

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"SELF-MODIFICATION REPORT",
            f"{'='*60}",
            f"Total self-modifications: {len(self.modifications)}",
            f"Reasoning metrics collected: {len(self.metrics_history)}",
        ]

        if self.modifications:
            lines.append(f"\nModifications Made:")
            for mod in self.modifications:
                lines.append(f"\n  Step {mod.step}: {mod.parameter}")
                lines.append(f"    {mod.old_value:.4f} → {mod.new_value:.4f}")
                lines.append(f"    Rationale: {mod.rationale}")

            # Evaluate improvement
            if len(self.modifications) >= 2:
                lines.append(f"\nSelf-Improvement Analysis:")
                lines.append(f"  The agent modified its own reasoning process {len(self.modifications)} times.")
                lines.append(f"  This is meta-learning: learning how to learn better.")
        else:
            lines.append(f"\n  No self-modifications were needed — reasoning parameters were well-tuned.")

        # Show current reasoning parameters
        lines.append(f"\nCurrent Reasoning Parameters:")
        lines.append(f"  min_observations_for_concept: {self.predictor.min_observations_for_concept}")
        lines.append(f"  min_evidence_for_confirmation: {self.predictor.min_evidence_for_confirmation}")
        lines.append(f"  confirmation_confidence: {self.predictor.confirmation_confidence}")
        lines.append(f"  min_observations_for_abstraction: {self.predictor.min_observations_for_abstraction}")
        if self.curiosity:
            lines.append(f"  exploration_rate: {self.curiosity.exploration_rate:.4f}")

        return "\n".join(lines)
