from __future__ import annotations

"""
Autonomous experiment scheduling for the first-principles agent.

The planner does not know which physics worlds are "supposed" to be novel.
It only sees what memory has already covered and asks for runs that are likely
to improve coverage, test transfer, or expose reusable dynamics laws.
"""

from collections import Counter
from dataclasses import dataclass, field
import hashlib


@dataclass(frozen=True)
class ExperimentCandidate:
    """A scored candidate world run."""
    world_type: str
    seed: int
    object_count: int
    steps: int
    score: float
    reasons: dict[str, float] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, int, int, int]:
        return (self.world_type, self.seed, self.object_count, self.steps)

    def to_dict(self) -> dict:
        return {
            'world_type': self.world_type,
            'seed': self.seed,
            'object_count': self.object_count,
            'steps': self.steps,
            'score': round(self.score, 4),
            'reasons': {
                key: round(value, 4)
                for key, value in self.reasons.items()
            },
        }


class ExplorationPlanner:
    """
    Proposes many possible experiments and chooses the most promising next one.

    Scoring favors:
    - worlds and object counts with little prior coverage
    - worlds where existing law schemas have not yet transferred or failed
    - seed diversity
    - slightly richer object counts when available
    """

    def __init__(
        self,
        world_types: list[str],
        object_counts: list[int],
        steps: int,
        seed_start: int = 0,
        seed_span: int = 20,
    ):
        self.world_types = list(world_types)
        self.object_counts = list(object_counts)
        self.steps = steps
        self.seed_start = seed_start
        self.seed_span = max(1, seed_span)

    def propose(
        self,
        law_memory,
        completed_keys: set[tuple[str, int, int, int]] | None = None,
        limit: int | None = None,
    ) -> list[ExperimentCandidate]:
        completed_keys = set(completed_keys or set())
        completed_keys.update(self._memory_keys(law_memory))

        world_counts = Counter(episode.world_type for episode in law_memory.episodes)
        object_counts = Counter(
            (episode.world_type, episode.object_count)
            for episode in law_memory.episodes
        )
        seed_counts = Counter(episode.seed for episode in law_memory.episodes)

        max_object_count = max(self.object_counts) if self.object_counts else 1
        candidates = []
        for world_type in self.world_types:
            transfer_gap = self._transfer_gap(law_memory, world_type)
            for object_count in self.object_counts:
                for seed in range(self.seed_start, self.seed_start + self.seed_span):
                    key = (world_type, seed, object_count, self.steps)
                    if key in completed_keys:
                        continue
                    reasons = {
                        'under_tested_world': 1.0 / (1.0 + world_counts[world_type]),
                        'under_tested_object_count': 1.0 / (
                            1.0 + object_counts[(world_type, object_count)]
                        ),
                        'seed_diversity': 1.0 / (1.0 + seed_counts[seed]),
                        'transfer_gap': transfer_gap,
                        'interaction_richness': object_count / max(max_object_count, 1),
                        'jitter': self._stable_jitter(key),
                    }
                    score = (
                        0.34 * reasons['under_tested_world']
                        + 0.18 * reasons['under_tested_object_count']
                        + 0.14 * reasons['seed_diversity']
                        + 0.24 * reasons['transfer_gap']
                        + 0.08 * reasons['interaction_richness']
                        + reasons['jitter']
                    )
                    candidates.append(ExperimentCandidate(
                        world_type=world_type,
                        seed=seed,
                        object_count=object_count,
                        steps=self.steps,
                        score=score,
                        reasons=reasons,
                    ))

        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        if limit is not None:
            return candidates[:limit]
        return candidates

    def _transfer_gap(self, law_memory, world_type: str) -> float:
        if not law_memory.schemas:
            return 0.5
        untested = 0
        for schema in law_memory.schemas.values():
            if world_type not in schema.source_worlds and world_type not in schema.absent_worlds:
                untested += 1
        return untested / max(len(law_memory.schemas), 1)

    def _memory_keys(self, law_memory) -> set[tuple[str, int, int, int]]:
        return {
            (
                episode.world_type,
                episode.seed,
                episode.object_count,
                episode.step_count,
            )
            for episode in law_memory.episodes
        }

    def _stable_jitter(self, key: tuple[str, int, int, int]) -> float:
        digest = hashlib.sha256(repr(key).encode('utf-8')).hexdigest()
        return (int(digest[:8], 16) / 0xFFFFFFFF) * 0.01


def score_exploration_outcome(metrics: dict) -> float:
    """Summarize how much useful information an experiment produced."""
    learned_score = min(1.0, metrics.get('learned_rule_count', 0) / 3.0)
    novel_score = min(1.0, len(metrics.get('detected', set())) / 2.0)

    transfer_report = metrics.get('memory_transfer') or {}
    matched_score = min(1.0, len(transfer_report.get('matched_priors', [])) / 3.0)
    falsified_score = min(1.0, len(transfer_report.get('missing_priors', [])) / 3.0)

    first_step = metrics.get('first_learned_step')
    if first_step is None:
        speed_score = 0.0
    else:
        speed_score = max(0.0, 1.0 - first_step / max(metrics.get('steps', 1), 1))

    return round(
        0.34 * learned_score
        + 0.24 * novel_score
        + 0.18 * matched_score
        + 0.16 * falsified_score
        + 0.08 * speed_score,
        4,
    )


def explain_exploration_outcome(metrics: dict) -> str:
    """Give a short human-readable reason for why a run mattered."""
    if metrics.get('learned_rule_count', 0) > 0:
        laws = ', '.join(metrics.get('learned_law_types', [])) or 'dynamics laws'
        return f"learned reusable law families: {laws}"
    detected = sorted(metrics.get('detected', set()))
    if detected:
        return f"confirmed novel physics: {', '.join(detected)}"
    transfer_report = metrics.get('memory_transfer') or {}
    if transfer_report.get('matched_priors'):
        return "confirmed that prior law schemas transfer here"
    if transfer_report.get('missing_priors'):
        return "found counterexamples where prior law schemas did not transfer"
    return "expanded coverage for a low-surprise baseline"
