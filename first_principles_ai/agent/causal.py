from __future__ import annotations

"""
Causal Reasoning — the agent's causal inference system.

This moves the agent from "what happens" to "why it happens."

The agent tracks:
  - INTERVENTIONS: Actions it takes (push, spawn, remove)
  - EVENTS: Things that happen in the world (collisions, object appears/disappears)
  - CHANGES: Feature deltas between steps
  - CAUSAL CHAINS: Action → Event → Change sequences

The agent builds a causal graph:
  push(A) → A.velocity_changed → A.moved → A_collided_with_B → B.velocity_changed → B.moved

This enables:
  - "What caused this change?" — trace back the causal chain
  - "What happens if I push A?" — forward simulate the causal graph
  - "Would B have moved if I hadn't pushed A?" — counterfactual reasoning
"""

import math
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict


@dataclass
class CausalEvent:
    """A single event in a causal chain."""
    step: int
    event_type: str           # 'action', 'collision', 'appearance', 'disappearance', 'motion'
    description: str          # Human-readable description
    actor_id: Optional[int] = None    # Which object was involved
    target_id: Optional[int] = None   # Secondary object (e.g., collision partner)
    properties: dict = field(default_factory=dict)


@dataclass
class CausalLink:
    """A directed causal relationship between two events."""
    cause: CausalEvent
    effect: CausalEvent
    strength: float = 1.0     # How consistently this cause produces this effect
    observed_count: int = 0   # How many times this link has been observed


class CausalGraph:
    """
    The agent's causal model of the world.

    Nodes are event types. Edges are causal relationships.
    The agent can query this graph to answer:
      - "What usually happens after I push an object?"
      - "What caused this collision?"
      - "If I do X, what will happen?"
    """

    def __init__(self):
        # Adjacency: cause_type → list of (effect_type, count)
        self.edges: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # Event history for tracing
        self.event_history: list[CausalEvent] = []
        # Per-object event chains
        self.object_chains: dict[int, list[CausalEvent]] = defaultdict(list)
        # Total observations per event type
        self.event_type_counts: dict[str, int] = defaultdict(int)

    def add_event(self, event: CausalEvent):
        """Record an event and update causal relationships."""
        self.event_history.append(event)
        self.event_type_counts[event.event_type] += 1

        if event.actor_id is not None:
            chain = self.object_chains[event.actor_id]

            # Link to previous event for this object (if within recent window)
            if chain:
                prev_event = chain[-1]
                # Only link if events are close in time (within 5 steps)
                if event.step - prev_event.step <= 5:
                    self.edges[prev_event.event_type][event.event_type] += 1

            chain.append(event)

    def get_causal_chain(self, event_type: str, max_depth: int = 5) -> list[list[str]]:
        """
        Find all causal chains starting from an event type.
        Returns list of chains, each a sequence of event types.
        """
        chains = []
        self._dfs_chain(event_type, [], chains, max_depth)
        return chains

    def _dfs_chain(self, current: str, path: list[str], chains: list[list[str]], max_depth: int):
        path = path + [current]
        chains.append(path)

        if len(path) >= max_depth:
            return

        # Get next events sorted by frequency
        next_events = sorted(
            self.edges[current].items(),
            key=lambda x: x[1],
            reverse=True
        )

        for next_type, count in next_events[:3]:  # Top 3 most common effects
            # Avoid cycles
            if next_type not in path:
                self._dfs_chain(next_type, path, chains, max_depth)

    def get_most_common_effect(self, cause_type: str) -> Optional[tuple[str, float]]:
        """What most commonly follows this cause? Returns (effect_type, probability)."""
        if cause_type not in self.edges:
            return None

        effects = self.edges[cause_type]
        total = sum(effects.values())
        if total == 0:
            return None

        most_common = max(effects.items(), key=lambda x: x[1])
        probability = most_common[1] / total
        return (most_common[0], probability)

    def get_cause_probability(self, cause_type: str, effect_type: str) -> float:
        """P(effect | cause) — how likely is this effect given this cause?"""
        if cause_type not in self.edges:
            return 0.0
        total = sum(self.edges[cause_type].values())
        if total == 0:
            return 0.0
        return self.edges[cause_type].get(effect_type, 0) / total

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"CAUSAL GRAPH SUMMARY",
            f"{'='*60}",
            f"Total events recorded: {len(self.event_history)}",
            f"Event types: {dict(self.event_type_counts)}",
            f"\nCausal relationships (cause → effect):",
        ]

        for cause, effects in sorted(self.edges.items()):
            total = sum(effects.values())
            for effect, count in sorted(effects.items(), key=lambda x: x[1], reverse=True):
                prob = count / total
                lines.append(f"  {cause:20s} → {effect:20s}  "
                           f"(count={count}, P={prob:.2f})")

        lines.append(f"\nMost common causal chains:")
        for event_type in self.event_type_counts:
            chains = self.get_causal_chain(event_type, max_depth=4)
            for chain in chains[:2]:  # Top 2 chains per event type
                lines.append(f"  {' → '.join(chain)}")

        return "\n".join(lines)


class CausalReasoner:
    """
    The agent's causal reasoning engine.

    It observes actions and events, builds a causal graph,
    and can answer causal questions.
    """

    def __init__(self):
        self.graph = CausalGraph()
        self.prev_action: Optional[dict] = None
        self.prev_step: int = 0

    def observe_step(self, action: dict, observation_before: dict, observation_after: dict,
                     step: int, collisions: list):
        """
        Record one step of experience and update the causal model.

        Args:
            action: What the agent did this step
            observation_before: World state before the action
            observation_after: World state after the action
            step: Current step number
            collisions: List of collision events that occurred
        """
        # Record the action as a causal event
        if action and action.get('type') != 'wait':
            action_event = CausalEvent(
                step=step,
                event_type=f"action_{action['type']}",
                description=f"Agent performed {action['type']}",
                actor_id=action.get('object_id'),
                properties=action,
            )
            self.graph.add_event(action_event)

        # Record collision events
        for coll in collisions:
            coll_event = CausalEvent(
                step=step,
                event_type='collision',
                description=f"Objects {coll['object_a']} and {coll['object_b']} collided",
                actor_id=coll['object_a'],
                target_id=coll['object_b'],
                properties=coll,
            )
            self.graph.add_event(coll_event)

        # Detect appearances and disappearances
        before_ids = set(obj['id'] for obj in observation_before.get('objects', []))
        after_ids = set(obj['id'] for obj in observation_after.get('objects', []))

        for new_id in after_ids - before_ids:
            appear_event = CausalEvent(
                step=step,
                event_type='appearance',
                description=f"Object {new_id} appeared",
                actor_id=new_id,
            )
            self.graph.add_event(appear_event)

        for gone_id in before_ids - after_ids:
            disappear_event = CausalEvent(
                step=step,
                event_type='disappearance',
                description=f"Object {gone_id} disappeared",
                actor_id=gone_id,
            )
            self.graph.add_event(disappear_event)

        # Detect significant motion changes for objects that were pushed
        if action and action.get('type') == 'push':
            oid = action.get('object_id')
            if oid is not None:
                # Find the object in after-observation
                for obj in observation_after.get('objects', []):
                    if obj['id'] == oid:
                        speed = math.sqrt(obj['velocity'][0]**2 + obj['velocity'][1]**2)
                        if speed > 0.5:
                            motion_event = CausalEvent(
                                step=step,
                                event_type='motion',
                                description=f"Object {oid} moved at speed {speed:.2f}",
                                actor_id=oid,
                                properties={'speed': round(speed, 4)},
                            )
                            self.graph.add_event(motion_event)
                        break

        self.prev_action = action
        self.prev_step = step

    def answer_what_happens_if(self, action_type: str) -> Optional[tuple[str, float]]:
        """
        Predict what will happen if the agent takes a certain action.
        Returns (most_likely_effect, probability).
        """
        cause = f"action_{action_type}"
        return self.graph.get_most_common_effect(cause)

    def answer_what_caused(self, event_type: str) -> list[tuple[str, float]]:
        """
        What are the most likely causes of an event type?
        Returns list of (cause_type, probability) sorted by likelihood.
        """
        causes = []
        for cause, effects in self.graph.edges.items():
            total = sum(effects.values())
            if total > 0 and event_type in effects:
                prob = effects[event_type] / total
                if prob > 0.1:
                    causes.append((cause, prob))
        causes.sort(key=lambda x: x[1], reverse=True)
        return causes

    def counterfactual(self, action_type: str, observed_effect: str) -> str:
        """
        Simple counterfactual reasoning:
        'If I hadn't done X, would Y have happened?'

        Uses the causal graph to estimate whether the effect
        commonly occurs without the action.
        """
        cause = f"action_{action_type}"

        # P(effect | action)
        p_with = self.graph.get_cause_probability(cause, observed_effect)

        # P(effect | other causes) — baseline rate
        total_other = 0
        count_other = 0
        for c, effects in self.graph.edges.items():
            if c != cause:
                for e, count in effects.items():
                    total_other += count
                    if e == observed_effect:
                        count_other += count

        p_without = count_other / total_other if total_other > 0 else 0.0

        if p_with > 0.3 and p_without < 0.1:
            return (f"If the agent had NOT performed {action_type}, "
                    f"{observed_effect} would likely NOT have occurred "
                    f"(P|action={p_with:.2f}, P|no action={p_without:.2f})")
        elif p_with > 0.3 and p_without > 0.3:
            return (f"{observed_effect} commonly occurs regardless of {action_type} "
                    f"(P|action={p_with:.2f}, P|no action={p_without:.2f}) — "
                    f"the action is not the primary cause")
        else:
            return (f"Insufficient evidence to determine if {action_type} "
                    f"caused {observed_effect}")
