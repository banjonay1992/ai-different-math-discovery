from __future__ import annotations

"""
Multi-Agent Language Emergence — agents invent their own communication system.

Multiple agents share the same world. They need to cooperate to solve problems
(like moving objects to specific positions). They can:
  - OBSERVE the world
  - ACT on the world (push, spawn, remove)
  - SIGNAL to other agents (send a symbol)
  - RECEIVE signals from other agents

They start with NO shared language. They must invent one from scratch.

We then analyze:
  - Does their language develop structure?
  - Does it develop compositionality (signals made of parts)?
  - Does it develop convention (consistent symbol→meaning mappings)?
  - Does it converge on the same structure as human language?

This tests whether language structure is a fundamental property of
communication, or a human-specific invention.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict


@dataclass
class Signal:
    """A signal sent from one agent to another."""
    sender_id: int
    receiver_id: int
    symbol: str           # The signal's symbol (from the agent's notation system)
    meaning: str          # Sender-private intended meaning, not shown to receiver learning
    step: int
    context: dict = field(default_factory=dict)  # World state when sent


@dataclass
class SignalMeaning:
    """A learned mapping between a signal and its meaning."""
    symbol: str
    meaning: str
    count: int = 0          # How many times this mapping has been observed
    success_count: int = 0  # How many times the receiver acted correctly


class CommunicationSystem:
    """
    An agent's internal communication system.

    Each agent has its own signal→meaning mapping. When two agents
    communicate, they must align their mappings through interaction.

    This is like how children learn language: they hear a word,
    infer its meaning from context, and test that inference.
    """

    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        # symbol → list of (meaning, count) — what this agent thinks each symbol means
        self.symbol_to_meanings: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        # meaning → preferred symbol — what this agent uses to express a meaning
        self.meaning_to_symbol: dict[str, str] = {}
        # All signals sent and received
        self.signals_sent: list[Signal] = []
        self.signals_received: list[Signal] = []
        self.inferred_received_count = 0
        self.ambiguous_received_count = 0
        self.successful_inference_count = 0
        self.failed_inference_count = 0
        self.action_useful_inference_count = 0
        # Signal alphabet (shared starting point — just indices)
        self._next_symbol_idx = 0

    def generate_signal(self, meaning: str, receiver_id: int, step: int,
                        context: dict) -> Signal:
        """Generate a signal for a given meaning."""
        # If we have a preferred symbol for this meaning, use it
        if meaning in self.meaning_to_symbol:
            symbol = self.meaning_to_symbol[meaning]
        else:
            # Generate a new symbol
            symbol = f"s{self._next_symbol_idx}"
            self._next_symbol_idx += 1
            self.meaning_to_symbol[meaning] = symbol

        signal = Signal(
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            symbol=symbol,
            meaning=meaning,
            step=step,
            context=context,
        )
        self.signals_sent.append(signal)
        return signal

    def receive_signal(self, signal: Signal, candidate_meanings: list[str] | None = None) -> Optional[str]:
        """Process a received signal and infer its meaning from shared context."""
        self.signals_received.append(signal)

        candidates = self._candidate_meanings(signal, candidate_meanings)
        if not candidates:
            return None

        self.inferred_received_count += 1
        if len(candidates) > 1:
            self.ambiguous_received_count += 1

        # Cross-situational inference: every plausible context candidate gets
        # some evidence. No update uses signal.meaning, which is sender-private.
        weight = 1.0 / len(candidates)
        for meaning in candidates:
            self.symbol_to_meanings[signal.symbol][meaning] += weight

        inferred = self._best_supported_candidate(signal.symbol, candidates)
        if inferred and inferred not in self.meaning_to_symbol:
            self.meaning_to_symbol[inferred] = signal.symbol
        signal.context['_receiver_inferred_meaning'] = inferred
        signal.context['_receiver_candidate_count'] = len(candidates)
        return inferred

    def reinforce_inference(
        self,
        signal: Signal,
        active_meanings: set[str],
        action_relevant_meanings: set[str] | None = None,
    ) -> bool:
        """Strengthen a symbol mapping when receiver inference matched the transition."""
        inferred = signal.context.get('_receiver_inferred_meaning')
        if not inferred:
            return False

        action_relevant_meanings = action_relevant_meanings or set()
        if inferred not in active_meanings:
            self.failed_inference_count += 1
            signal.context['_receiver_feedback'] = 'failed'
            return False

        self.successful_inference_count += 1
        self.symbol_to_meanings[signal.symbol][inferred] += 1.0
        self.meaning_to_symbol[inferred] = signal.symbol
        signal.context['_receiver_feedback'] = 'success'

        if inferred in action_relevant_meanings:
            self.action_useful_inference_count += 1
            self.symbol_to_meanings[signal.symbol][inferred] += 1.0
            signal.context['_receiver_action_useful'] = True
        else:
            signal.context['_receiver_action_useful'] = False
        return True

    def _candidate_meanings(self, signal: Signal, candidate_meanings: list[str] | None) -> list[str]:
        candidates = list(candidate_meanings or signal.context.get('_meaning_candidates', []))
        seen = set()
        unique = []
        for meaning in candidates:
            if meaning in seen:
                continue
            seen.add(meaning)
            unique.append(meaning)
        return unique

    def _best_supported_candidate(self, symbol: str, candidates: list[str]) -> Optional[str]:
        if not candidates:
            return None
        meanings = self.symbol_to_meanings.get(symbol, {})
        return max(candidates, key=lambda meaning: (meanings.get(meaning, 0.0), meaning))

    def interpret_signal(self, symbol: str) -> Optional[str]:
        """What does this symbol mean to me?"""
        if symbol not in self.symbol_to_meanings:
            return None
        # Return the most common meaning for this symbol
        meanings = self.symbol_to_meanings[symbol]
        if not meanings:
            return None
        return max(meanings.items(), key=lambda x: x[1])[0]

    def alignment_score(self, other: 'CommunicationSystem') -> float:
        """
        How aligned are our communication systems?
        1.0 = perfect agreement, 0.0 = no overlap.
        """
        # Check how many meaning→symbol mappings we share
        shared = 0
        total = 0
        all_meanings = set(self.meaning_to_symbol.keys()) | set(other.meaning_to_symbol.keys())
        for meaning in all_meanings:
            my_sym = self.meaning_to_symbol.get(meaning)
            their_sym = other.meaning_to_symbol.get(meaning)
            if my_sym and their_sym:
                total += 1
                if my_sym == their_sym:
                    shared += 1
            elif my_sym or their_sym:
                total += 1
        return shared / total if total > 0 else 0.0

    def vocabulary_size(self) -> int:
        return len(self.meaning_to_symbol)

    def summary(self) -> str:
        lines = [
            f"Agent {self.agent_id} Communication System:",
            f"  Vocabulary size: {self.vocabulary_size()}",
            f"  Signals sent: {len(self.signals_sent)}",
            f"  Signals received: {len(self.signals_received)}",
            f"  Meanings inferred from context: {self.inferred_received_count}",
            f"  Ambiguous received contexts: {self.ambiguous_received_count}",
            f"  Successful grounded inferences: {self.successful_inference_count}",
            f"  Failed grounded inferences: {self.failed_inference_count}",
            f"  Action-useful inferences: {self.action_useful_inference_count}",
            f"  Meaning → Symbol mappings:",
        ]
        for meaning, symbol in sorted(self.meaning_to_symbol.items()):
            lines.append(f"    {meaning:30s} → {symbol}")
        return "\n".join(lines)


class MultiAgentExperiment:
    """
    Run a multi-agent language emergence experiment.

    Multiple agents share a world. They take turns:
      1. Observing the world
      2. Optionally signaling to another agent
      3. Acting

    The key question: do they develop a shared, structured communication system?
    """

    # Cold-start fallback meanings used only before internal math concepts exist.
    FEATURE_FALLBACK_MEANINGS = [
        "object_moving",
        "object_stationary",
        "collision_happened",
        "object_appeared",
        "object_disappeared",
        "many_objects",
        "few_objects",
        "high_energy",
        "low_energy",
        "push_needed",
        "wait_needed",
    ]

    def __init__(self, num_agents: int = 2, seed: int = 42, max_grounded_meanings: int = 8):
        random.seed(seed)
        self.num_agents = num_agents
        self.max_grounded_meanings = max_grounded_meanings
        self.comms = [CommunicationSystem(i) for i in range(num_agents)]
        self.all_signals: list[Signal] = []
        self.step = 0
        self.math_grounded_signal_count = 0
        self.feature_fallback_signal_count = 0
        self.successful_grounding_count = 0
        self.failed_grounding_count = 0
        self.action_useful_signal_count = 0
        self.math_meaning_sources: dict[str, dict] = {}

        # Track language development metrics over time
        self.alignment_history: list[tuple[int, float]] = []
        self.vocab_history: list[tuple[int, int]] = []

    def step_agents(
        self,
        features: dict,
        had_collision: bool,
        step: int,
        math_patterns: list | None = None,
    ) -> list[Signal]:
        """
        One step of multi-agent interaction.

        Each agent observes the world, optionally signals, and we track
        whether they develop a shared language.
        """
        self.step = step
        signals_this_step = []

        # Prefer agent-internal math structures. Feature labels are only a
        # cold-start fallback before the math substrate has installed concepts.
        meanings = self._extract_grounded_math_meanings(math_patterns or [])
        meaning_source = 'emergent_math'
        if not meanings:
            meanings = self._extract_feature_fallback_meanings(features, had_collision)
            meaning_source = 'feature_fallback'
        active_math_meanings = self._active_math_meanings(math_patterns or [], step)
        action_relevant_meanings = self._action_relevant_math_meanings(math_patterns or [], step)

        # Each agent signals to a random other agent
        for i in range(self.num_agents):
            if not meanings:
                continue

            # Pick a meaning to communicate
            meaning = random.choice(meanings)
            receiver = (i + 1) % self.num_agents  # Signal to next agent
            math_groundings = {
                candidate: self.math_meaning_sources.get(candidate)
                for candidate in meanings
                if candidate.startswith('math:')
            }

            signal = self.comms[i].generate_signal(
                meaning=meaning,
                receiver_id=receiver,
                step=step,
                context={
                    **features,
                    '_meaning_source': meaning_source,
                    '_meaning_candidates': list(meanings),
                    '_math_groundings': math_groundings,
                },
            )
            signals_this_step.append(signal)
            if meaning_source == 'emergent_math':
                self.math_grounded_signal_count += 1
            else:
                self.feature_fallback_signal_count += 1

            # Receiver processes the signal
            self.comms[receiver].receive_signal(signal, candidate_meanings=meanings)
            success = self.comms[receiver].reinforce_inference(
                signal,
                active_meanings=active_math_meanings if meaning_source == 'emergent_math' else set(meanings),
                action_relevant_meanings=action_relevant_meanings,
            )
            if success:
                self.successful_grounding_count += 1
            else:
                self.failed_grounding_count += 1
            if signal.context.get('_receiver_action_useful'):
                self.action_useful_signal_count += 1

        self.all_signals.extend(signals_this_step)

        # Track alignment every 100 steps
        if step % 100 == 0 and self.num_agents >= 2:
            alignment = self.comms[0].alignment_score(self.comms[1])
            self.alignment_history.append((step, alignment))
            total_vocab = sum(c.vocabulary_size() for c in self.comms)
            self.vocab_history.append((step, total_vocab))

        return signals_this_step

    def _extract_grounded_math_meanings(self, math_patterns: list) -> list[str]:
        """Extract communicable meanings from installed emergent math concepts."""
        ranked = sorted(
            math_patterns,
            key=lambda pattern: (
                -self._pattern_value(pattern, 'evidence', 0),
                str(self._pattern_value(pattern, 'key', '')),
            ),
        )
        meanings = []
        for pattern in ranked:
            concept_name = self._pattern_value(pattern, 'concept_name')
            if not concept_name:
                continue
            meaning = f"math:{concept_name}"
            if meaning not in meanings:
                meanings.append(meaning)
            self.math_meaning_sources[meaning] = {
                'pattern_key': self._pattern_value(pattern, 'key'),
                'pattern_kind': self._pattern_value(pattern, 'kind'),
                'evidence': self._pattern_value(pattern, 'evidence', 0),
                'last_step': self._pattern_value(pattern, 'last_step', 0),
                'properties': dict(self._pattern_value(pattern, 'properties', {}) or {}),
            }
            if len(meanings) >= self.max_grounded_meanings:
                break
        return meanings

    def _active_math_meanings(self, math_patterns: list, step: int) -> set[str]:
        return {
            f"math:{concept_name}"
            for pattern in math_patterns
            for concept_name in [self._pattern_value(pattern, 'concept_name')]
            if concept_name and self._pattern_value(pattern, 'last_step') == step
        }

    def _action_relevant_math_meanings(self, math_patterns: list, step: int) -> set[str]:
        action_roles = {
            'discrete_delta',
            'intervention_transform',
            'repeatable_transform',
            'opposed_transforms',
        }
        meanings = set()
        for pattern in math_patterns:
            concept_name = self._pattern_value(pattern, 'concept_name')
            if not concept_name or self._pattern_value(pattern, 'last_step') != step:
                continue
            properties = self._pattern_value(pattern, 'properties', {}) or {}
            role = properties.get('structural_role')
            kind = self._pattern_value(pattern, 'kind')
            if kind in ('operation', 'transform') or role in action_roles:
                meanings.add(f"math:{concept_name}")
        return meanings

    def _pattern_value(self, pattern, key: str, default=None):
        if isinstance(pattern, dict):
            return pattern.get(key, default)
        return getattr(pattern, key, default)

    def _extract_feature_fallback_meanings(self, features: dict, had_collision: bool) -> list[str]:
        """Extract cold-start fallback meanings from engineered world features."""
        meanings = []

        if features.get('count', 0) > 3:
            meanings.append("many_objects")
        elif features.get('count', 0) <= 2 and features.get('count', 0) > 0:
            meanings.append("few_objects")

        if features.get('mean_speed', 0) > 1.0:
            meanings.append("object_moving")
        elif features.get('mean_speed', 0) < 0.1:
            meanings.append("object_stationary")

        if had_collision:
            meanings.append("collision_happened")

        if features.get('total_kinetic_energy', 0) > 10.0:
            meanings.append("high_energy")
        elif features.get('total_kinetic_energy', 0) < 1.0:
            meanings.append("low_energy")

        return meanings

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"MULTI-AGENT LANGUAGE EMERGENCE REPORT",
            f"{'='*60}",
            f"Agents: {self.num_agents}",
            f"Total signals exchanged: {len(self.all_signals)}",
        ]

        # Alignment over time
        if self.alignment_history:
            lines.append(f"\nLanguage Alignment Over Time:")
            for step, align in self.alignment_history:
                bar = "█" * int(align * 20)
                lines.append(f"  Step {step:5d}: {align:.2f} {bar}")

            final_align = self.alignment_history[-1][1]
            lines.append(f"\nFinal alignment: {final_align:.1%}")
            if final_align > 0.7:
                lines.append("  → Agents developed a SHARED communication system!")
            elif final_align > 0.3:
                lines.append("  → Partial communication alignment — language is emerging")
            else:
                lines.append("  → Agents have not yet aligned their communication")

        # Vocabulary growth
        if self.vocab_history:
            lines.append(f"\nVocabulary Growth:")
            for step, vocab in self.vocab_history:
                lines.append(f"  Step {step:5d}: {vocab} symbols")

        # Each agent's communication system
        for comm in self.comms:
            lines.append(f"\n{comm.summary()}")

        # Analyze language structure
        lines.append(f"\nLanguage Structure Analysis:")
        all_meanings = set()
        for comm in self.comms:
            all_meanings.update(comm.meaning_to_symbol.keys())

        # Check for compositionality: do similar meanings share symbol parts?
        # Check for convention: is the same meaning mapped to the same symbol across agents?
        convention_count = 0
        convention_total = 0
        for meaning in all_meanings:
            symbols = []
            for comm in self.comms:
                if meaning in comm.meaning_to_symbol:
                    symbols.append(comm.meaning_to_symbol[meaning])
            if len(symbols) >= 2:
                convention_total += 1
                if len(set(symbols)) == 1:
                    convention_count += 1

        if convention_total > 0:
            convention_rate = convention_count / convention_total
            lines.append(f"  Convention rate (same meaning → same symbol): {convention_rate:.1%}")
            if convention_rate > 0.5:
                lines.append("  → Language shows CONVENTIONALITY — agents agree on symbol meanings")

        math_meanings = sorted(m for m in all_meanings if m.startswith('math:'))
        if math_meanings:
            lines.append(f"  Math-grounded meanings: {len(math_meanings)}")
            lines.append(f"  Math-grounded signals: {self.math_grounded_signal_count}")
            lines.append(f"  Feature-fallback signals: {self.feature_fallback_signal_count}")
            lines.append(f"  Successful grounded inferences: {self.successful_grounding_count}")
            lines.append(f"  Failed grounded inferences: {self.failed_grounding_count}")
            lines.append(f"  Action-useful signal inferences: {self.action_useful_signal_count}")
            for meaning in math_meanings[:8]:
                source = self.math_meaning_sources.get(meaning, {})
                pattern_key = source.get('pattern_key', 'unknown')
                lines.append(f"    {meaning} grounded in {pattern_key}")

        # Check for category structure in any cold-start fallback vocabulary.
        motion_meanings = {"object_moving", "object_stationary", "collision_happened", "high_energy", "low_energy"}
        quantity_meanings = {"many_objects", "few_objects", "object_appeared", "object_disappeared"}

        for comm in self.comms:
            motion_syms = set(comm.meaning_to_symbol.get(m) for m in motion_meanings if m in comm.meaning_to_symbol)
            quantity_syms = set(comm.meaning_to_symbol.get(m) for m in quantity_meanings if m in comm.meaning_to_symbol)
            overlap = motion_syms & quantity_syms
            if motion_syms and quantity_syms:
                lines.append(f"  Agent {comm.agent_id}: motion symbols={motion_syms}, "
                           f"quantity symbols={quantity_syms}, overlap={overlap}")

        return "\n".join(lines)
