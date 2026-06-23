from __future__ import annotations

"""
Curiosity — the agent's exploration strategy.

The agent is NOT rewarded for achieving goals.
It is rewarded for LEARNING — for reducing its own uncertainty about the world.

Curiosity = prediction error × novelty

When the agent's predictions are wrong, it gets curious.
It seeks out situations where it doesn't know what will happen.
This is the same drive that makes human children learn.

Action selection:
  1. Generate candidate actions (push random object, spawn object, wait, etc.)
  2. For each, predict what would happen (using current knowledge)
  3. Estimate uncertainty (how confident is the prediction?)
  4. Choose the action with highest expected information gain
"""

import math
import random
from typing import Optional


class Curiosity:
    """Curiosity-driven action selection."""

    def __init__(self, environment):
        self.env = environment
        self.action_history: list[dict] = []
        self.exploration_rate = 0.9  # Start highly exploratory
        self.min_exploration = 0.3   # Always explore at least 30% of the time

    def decay_exploration(self, decay: float = 0.999):
        """Gradually shift from exploration to exploitation as knowledge grows."""
        self.exploration_rate = max(self.min_exploration, self.exploration_rate * decay)

    def select_action(self, predictor, current_features: dict) -> dict:
        """
        Select the next action based on curiosity.

        The agent prefers actions that it predicts will lead to the most
        uncertain outcomes — because those are where it can learn the most.
        """
        candidate_actions = self._generate_candidate_actions(current_features)

        if not candidate_actions:
            return {'type': 'wait'}

        # Sometimes just explore randomly (epsilon-greedy style)
        if random.random() < self.exploration_rate:
            # Score each action by expected information gain
            scored = []
            for action in candidate_actions:
                score = self._estimate_information_gain(action, predictor, current_features)
                scored.append((score, action))

            scored.sort(key=lambda x: x[0], reverse=True)
            best = scored[0][1]
        else:
            best = random.choice(candidate_actions)

        self.action_history.append(best)
        return best

    def _generate_candidate_actions(self, features: dict) -> list[dict]:
        """Generate a set of possible actions to consider."""
        actions = [{'type': 'wait'}]

        obj_ids = self.env.get_object_ids()
        if not obj_ids:
            return actions

        # Push a random object in a random direction
        for _ in range(3):
            oid = random.choice(obj_ids)
            angle = random.uniform(0, 2 * math.pi)
            force = random.uniform(2, 8)
            actions.append({
                'type': 'push',
                'object_id': oid,
                'fx': force * math.cos(angle),
                'fy': force * math.sin(angle),
            })

        # Spawn a new object
        actions.append({
            'type': 'spawn',
            'x': random.uniform(2, self.env.world.width - 2),
            'y': random.uniform(2, self.env.world.height - 2),
            'vx': random.uniform(-3, 3),
            'vy': random.uniform(-3, 3),
        })

        # Remove an object (if there are enough)
        if len(obj_ids) > 2:
            actions.append({
                'type': 'remove',
                'object_id': random.choice(obj_ids),
            })

        return actions

    def _estimate_information_gain(self, action: dict, predictor,
                                    current_features: dict) -> float:
        """
        Estimate how much an action would teach the agent.

        Heuristic: actions that change the world state more are more informative
        (for an agent with little knowledge). As knowledge grows, the agent
        becomes more selective.

        Also: actions the agent hasn't tried much yet are more interesting.
        """
        score = 0.0

        # Novelty: how often has this type of action been taken?
        action_type = action['type']
        type_count = sum(1 for a in self.action_history if a['type'] == action_type)
        novelty = 1.0 / (1.0 + type_count * 0.1)
        score += novelty * 2.0

        # State-change potential: actions that change the world are more interesting
        if action_type == 'push':
            # Pushing objects creates collisions → more to learn about
            score += 1.5
            # If we know little about momentum, pushing is extra interesting
            if not predictor.kb.has_concept_for_feature('total_momentum'):
                score += 2.0
        elif action_type == 'spawn':
            # Spawning changes count → tests arithmetic hypotheses
            score += 1.0
            if not predictor.kb.has_concept_for_feature('count'):
                score += 1.5
        elif action_type == 'remove':
            # Removing changes count → tests subtraction
            score += 0.8
        elif action_type == 'wait':
            # Waiting is useful for observing natural dynamics
            score += 0.3
            # But if there's not much movement, waiting is boring
            mean_speed = current_features.get('mean_speed', 0)
            if mean_speed < 0.5:
                score -= 0.5

        # Add some randomness to break ties
        score += random.uniform(-0.1, 0.1)

        return score
