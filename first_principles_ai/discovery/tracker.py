from __future__ import annotations

"""
Discovery Tracker — logs and reports what the agent discovers.

Each discovery is recorded with:
  - What was discovered (the agent's internal name)
  - When it was discovered (step number)
  - What evidence supports it
  - What human concept it maps to (if any)

This is the convergence test: does the agent independently arrive at
the same concepts humans did?
"""

from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class Discovery:
    """A single discovery made by the agent."""
    agent_name: str              # The agent's internal name (e.g., "C_003")
    discovery_type: str          # "concept" or "rule"
    description: str             # What the agent found
    step: int                    # When it was found
    timestamp: float = field(default_factory=time.time)
    human_mapping: Optional[str] = None    # Which human concept this maps to
    human_description: Optional[str] = None  # Description of the human concept
    convergence: bool = False    # Does it converge with a human concept?
    evidence: dict = field(default_factory=dict)


class DiscoveryTracker:
    """Tracks all discoveries and maps them to human concepts."""

    def __init__(self):
        self.discoveries: list[Discovery] = []
        self.convergence_count: int = 0
        self.total_discoveries: int = 0

    def record_concept(self, concept, human_mapping: str = None,
                       human_description: str = None):
        """Record a concept discovery."""
        discovery = Discovery(
            agent_name=concept.internal_name,
            discovery_type='concept',
            description=concept.description,
            step=concept.discovered_at_step,
            human_mapping=human_mapping,
            human_description=human_description,
            convergence=human_mapping is not None,
            evidence=concept.properties,
        )
        self.discoveries.append(discovery)
        self.total_discoveries += 1
        if discovery.convergence:
            self.convergence_count += 1

    def record_rule(self, rule, human_mapping: str = None,
                    human_description: str = None):
        """Record a confirmed rule/law discovery."""
        discovery = Discovery(
            agent_name=rule.internal_name,
            discovery_type='rule',
            description=f"WHEN {rule.conditions}, THEN {rule.prediction}",
            step=rule.confirmed_at_step or rule.discovered_at_step,
            human_mapping=human_mapping,
            human_description=human_description,
            convergence=human_mapping is not None,
            evidence={
                'confidence': rule.confidence,
                'evidence_for': rule.evidence_for,
                'evidence_against': rule.evidence_against,
            },
        )
        self.discoveries.append(discovery)
        self.total_discoveries += 1
        if discovery.convergence:
            self.convergence_count += 1

    def record_meta_concept(self, meta, human_mapping: str = None,
                            human_description: str = None):
        """Record a meta-concept (higher-order abstraction) discovery."""
        discovery = Discovery(
            agent_name=meta.internal_name,
            discovery_type='meta_concept',
            description=meta.description,
            step=meta.discovered_at_step,
            human_mapping=human_mapping,
            human_description=human_description,
            convergence=human_mapping is not None,
            evidence={'sub_concepts': meta.sub_concepts},
        )
        self.discoveries.append(discovery)
        self.total_discoveries += 1
        if discovery.convergence:
            self.convergence_count += 1

    def get_convergence_rate(self) -> float:
        """What fraction of discoveries map to human concepts?"""
        if self.total_discoveries == 0:
            return 0.0
        return self.convergence_count / self.total_discoveries

    def get_timeline(self) -> list[Discovery]:
        """Return discoveries in chronological order."""
        return sorted(self.discoveries, key=lambda d: d.step)

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"DISCOVERY SUMMARY",
            f"{'='*60}",
            f"Total discoveries: {self.total_discoveries}",
            f"Convergent with human concepts: {self.convergence_count}",
            f"Convergence rate: {self.get_convergence_rate():.1%}",
            f"\nTimeline:",
        ]

        for d in self.get_timeline():
            status = "CONVERGENT" if d.convergence else "NOVEL"
            human = f" → {d.human_mapping}" if d.human_mapping else ""
            lines.append(
                f"  Step {d.step:5d} | {d.discovery_type:8s} | {d.agent_name:8s} | "
                f"{status:10s}{human}"
            )
            lines.append(f"           {d.description}")

        lines.append(f"\n{'='*60}")
        lines.append(f"CONVERGENCE ANALYSIS")
        lines.append(f"{'='*60}")

        if self.convergence_count > 0:
            lines.append(
                f"\nThe agent independently discovered {self.convergence_count} concepts "
                f"that map directly to human mathematical/physical knowledge."
            )
            lines.append(
                f"Convergence rate: {self.get_convergence_rate():.1%}"
            )
            lines.append(
                "\nThis suggests these concepts are NOT human inventions\n"
                "but properties of the physical world itself.\n"
                "The agent discovered them from observation alone —\n"
                "no human data, no training set, no examples."
            )
        else:
            lines.append("\nNo convergent discoveries yet. The agent is still learning.")

        return "\n".join(lines)
