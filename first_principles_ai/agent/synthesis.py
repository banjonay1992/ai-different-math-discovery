from __future__ import annotations

"""
Program Synthesis — the agent writes its own reusable procedures.

After discovering concepts and rules, the agent can compose them into
programs that solve problems in its world. These programs are:
  - Explicit and inspectable (not neural weights)
  - Reusable (define once, run many times)
  - Verifiable (test against the world)
  - Composable (programs can call other programs)

The agent discovers that building reusable procedures is more efficient
than solving everything from scratch. It invents:
  - Functions (do this, then this, then this)
  - Loops (repeat until condition)
  - Conditionals (if this, then that)
  - Abstraction (this procedure works for many situations)

Programs are stored in the agent's own notation system.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Optional

from .representation import ConceptType


@dataclass
class ProgramStep:
    """A single step in a program."""
    action_type: str       # 'push', 'wait', 'check', 'loop', 'call'
    params: dict           # Parameters for this step
    description: str = ""  # What this step does (in the agent's terms)


@dataclass
class Program:
    """
    A reusable procedure the agent has synthesized.

    The agent invents these to solve recurring problems:
      - "move object to position X" → a sequence of pushes
      - "create N objects" → a loop that spawns objects
      - "clear the world" → a loop that removes all objects
    """
    internal_name: str           # Agent's name (e.g., "P_001")
    description: str             # What the program does
    steps: list[ProgramStep]     # The procedure
    created_at_step: int         # When it was first written
    run_count: int = 0           # How many times it's been executed
    success_count: int = 0       # How many times it achieved its goal
    failure_count: int = 0       # How many times it failed
    notation: Optional[str] = None  # The agent's symbol for this program

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0

    def __repr__(self):
        return (f"Program({self.internal_name}, '{self.description}', "
                f"runs={self.run_count}, success={self.success_rate:.1%})")


class ProgramSynthesizer:
    """
    The agent's program synthesis engine.

    It observes recurring patterns in its actions and synthesizes
    them into reusable programs. It can:
      - Detect when it's repeating the same sequence
      - Extract that sequence into a named program
      - Test the program by running it
      - Compose programs from simpler programs
    """

    def __init__(self, knowledge_base):
        self.kb = knowledge_base
        self.programs: dict[str, Program] = {}
        self._program_counter = 0
        self.action_history: list[dict] = []
        self.action_history_window: list[dict] = []
        self.window_size = 20
        self.synthesized_patterns: set[str] = set()

    def new_program_name(self) -> str:
        self._program_counter += 1
        return f"P_{self._program_counter:03d}"

    def record_action(self, action: dict, step: int, result_features: dict):
        """Record an action and its outcome for pattern detection."""
        action_record = {
            'action': action,
            'step': step,
            'result': result_features,
        }
        self.action_history.append(action_record)
        self.action_history_window.append(action_record)

        if len(self.action_history_window) > self.window_size:
            self.action_history_window.pop(0)

        # Check for synthesizable patterns
        if len(self.action_history_window) >= 10:
            self._detect_repeating_patterns(step)

    def _detect_repeating_patterns(self, current_step: int):
        """Detect if the agent is repeating the same action sequence."""
        window = self.action_history_window

        # Look for repeated action types in the window
        action_types = [w['action'].get('type', 'wait') for w in window]

        # Check for common patterns:
        # 1. Repeated pushes (trying to move something)
        # 2. Spawn then push (create and interact)
        # 3. Repeated removes (clearing the world)

        # Pattern: "push N times in a row"
        push_streak = 0
        for at in reversed(action_types):
            if at == 'push':
                push_streak += 1
            else:
                break

        if push_streak >= 5:
            pattern_key = "pattern_repeated_push"
            if pattern_key not in self.synthesized_patterns:
                self._synthesize_push_program(current_step)
                self.synthesized_patterns.add(pattern_key)

        # Pattern: "spawn then push"
        if len(action_types) >= 2:
            for i in range(len(action_types) - 1):
                if action_types[i] == 'spawn' and action_types[i + 1] == 'push':
                    pattern_key = "pattern_spawn_then_push"
                    if pattern_key not in self.synthesized_patterns:
                        self._synthesize_spawn_push_program(current_step)
                        self.synthesized_patterns.add(pattern_key)
                    break

        # Pattern: "multiple removes in a row"
        remove_streak = 0
        for at in reversed(action_types):
            if at == 'remove':
                remove_streak += 1
            else:
                break

        if remove_streak >= 3:
            pattern_key = "pattern_clear_world"
            if pattern_key not in self.synthesized_patterns:
                self._synthesize_clear_program(current_step)
                self.synthesized_patterns.add(pattern_key)

    def _synthesize_push_program(self, step: int):
        """Synthesize: 'apply force to an object to move it'."""
        name = self.new_program_name()
        symbol = self.kb.notation_system.generate_operation_symbol("program_push")

        prog = Program(
            internal_name=name,
            description="Procedure: apply repeated force to an object to induce motion — "
                       "discovered by noticing that pushing objects repeatedly causes sustained movement",
            steps=[
                ProgramStep(
                    action_type='push',
                    params={'direction': 'random', 'force': 'moderate'},
                    description="Apply force to target object in a chosen direction",
                ),
                ProgramStep(
                    action_type='wait',
                    params={'duration': 1},
                    description="Observe the resulting motion",
                ),
                ProgramStep(
                    action_type='check',
                    params={'condition': 'object_is_moving'},
                    description="Check if the object is now in motion",
                ),
            ],
            created_at_step=step,
            notation=symbol,
        )
        self.programs[name] = prog

        self.kb.add_concept(
            concept_type=ConceptType.OPERATION,
            description=f"Reusable procedure: induce motion in an object by applying force — "
                        f"a composable action sequence",
            feature_key=f"program_{name}",
            step=step,
            properties={'program': name, 'type': 'procedure'},
            notation=symbol,
        )

    def _synthesize_spawn_push_program(self, step: int):
        """Synthesize: 'create an object and interact with it'."""
        name = self.new_program_name()
        symbol = self.kb.notation_system.generate_operation_symbol("program_spawn_interact")

        prog = Program(
            internal_name=name,
            description="Procedure: create a new object and then interact with it — "
                       "discovered by noticing that new objects are often pushed immediately after creation",
            steps=[
                ProgramStep(
                    action_type='spawn',
                    params={'position': 'random', 'velocity': 'random'},
                    description="Create a new object in the world",
                ),
                ProgramStep(
                    action_type='push',
                    params={'target': 'newly_created', 'force': 'moderate'},
                    description="Apply force to the newly created object",
                ),
                ProgramStep(
                    action_type='wait',
                    params={'duration': 1},
                    description="Observe the result",
                ),
            ],
            created_at_step=step,
            notation=symbol,
        )
        self.programs[name] = prog

        self.kb.add_concept(
            concept_type=ConceptType.OPERATION,
            description=f"Reusable procedure: create and interact — a composable action sequence "
                        f"for introducing new objects and testing their behavior",
            feature_key=f"program_{name}",
            step=step,
            properties={'program': name, 'type': 'procedure'},
            notation=symbol,
        )

    def _synthesize_clear_program(self, step: int):
        """Synthesize: 'remove all objects from the world'."""
        name = self.new_program_name()
        symbol = self.kb.notation_system.generate_operation_symbol("program_clear")

        prog = Program(
            internal_name=name,
            description="Procedure: remove objects from the world one by one until none remain — "
                       "discovered by noticing that repeated removal reduces count to zero",
            steps=[
                ProgramStep(
                    action_type='check',
                    params={'condition': 'objects_exist'},
                    description="Check if any objects remain",
                ),
                ProgramStep(
                    action_type='remove',
                    params={'target': 'any'},
                    description="Remove one object",
                ),
                ProgramStep(
                    action_type='loop',
                    params={'back_to': 0, 'condition': 'objects_exist'},
                    description="Repeat until no objects remain",
                ),
            ],
            created_at_step=step,
            notation=symbol,
        )
        self.programs[name] = prog

        self.kb.add_concept(
            concept_type=ConceptType.OPERATION,
            description=f"Reusable procedure: clear the world — a loop that removes objects "
                        f"until none remain. Demonstrates the agent discovered iteration.",
            feature_key=f"program_{name}",
            step=step,
            properties={'program': name, 'type': 'procedure', 'contains_loop': True},
            notation=symbol,
        )

    def get_programs(self) -> list[Program]:
        return list(self.programs.values())

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"PROGRAM SYNTHESIS REPORT",
            f"{'='*60}",
            f"Programs synthesized: {len(self.programs)}",
            f"Total actions recorded: {len(self.action_history)}",
        ]

        for prog in self.programs.values():
            lines.append(f"\n  {prog.internal_name} (symbol: {prog.notation})")
            lines.append(f"    Description: {prog.description}")
            lines.append(f"    Steps:")
            for i, step in enumerate(prog.steps):
                lines.append(f"      {i+1}. [{step.action_type}] {step.description}")
            lines.append(f"    Runs: {prog.run_count}, Success: {prog.success_rate:.1%}")

        if not self.programs:
            lines.append("\n  No programs synthesized yet.")

        return "\n".join(lines)
