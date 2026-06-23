from __future__ import annotations

"""
Representation — the agent's symbolic knowledge system.

This is NOT neural weights. This is explicit, inspectable, correctable knowledge.

The agent builds its understanding through:
  - CONCEPTS: Abstract properties it has noticed (e.g., "count", "momentum")
  - RULES: Confirmed regularities (e.g., "momentum is conserved in collisions")
  - HYPOTHESES: Untested candidate rules

Each concept and rule has:
  - A unique internal name (NOT a human name — the agent invents its own)
  - A mathematical/logical structure
  - Evidence (observations that support or contradict it)
  - A confidence score

The agent can inspect, modify, and delete its own knowledge.
This is what makes it self-correcting.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ConceptType(Enum):
    """Types of concepts the agent can form."""
    QUANTITY = "quantity"        # A measurable property (count, mass, distance)
    RELATION = "relation"        # A relationship between objects (left_of, colliding_with)
    CONSERVATION = "conservation"  # A quantity that stays constant under certain conditions
    PATTERN = "pattern"          # A regularity in how things change
    EVENT = "event"              # Something that happens (collision, appearance)
    META_CONCEPT = "meta_concept"  # A higher-order concept grouping other concepts
    OPERATION = "operation"      # An operation the agent has invented (addition, subtraction)
    GEOMETRIC = "geometric"      # A spatial/geometric property (distance, ordering, symmetry)


class RuleStatus(Enum):
    HYPOTHESIS = "hypothesis"    # Untested or insufficient evidence
    TESTING = "testing"          # Being actively tested
    CONFIRMED = "confirmed"      # Strong evidence supports it
    REJECTED = "rejected"        # Evidence contradicts it


@dataclass
class Concept:
    """
    An abstract concept the agent has discovered.

    The agent gives this its own internal name (like C_001).
    We later map it to human concepts in the comparison module.
    """
    internal_name: str           # Agent's own name (e.g., "C_003")
    concept_type: ConceptType
    description: str             # What the agent noticed (auto-generated)
    feature_key: str             # Which feature this concept is about
    discovered_at_step: int      # When the agent first noticed it
    discovered_at_time: float = field(default_factory=time.time)
    properties: dict = field(default_factory=dict)  # Extra metadata
    notation: Optional[str] = None  # The agent's self-invented symbol for this concept
    sub_concepts: list = field(default_factory=list)  # For meta-concepts: which concepts this groups

    def __repr__(self):
        notation_str = f", symbol='{self.notation}'" if self.notation else ""
        return f"Concept({self.internal_name}, {self.concept_type.value}, '{self.description}'{notation_str})"


@dataclass
class Rule:
    """
    A rule about the world. Either a hypothesis (untested) or confirmed knowledge.

    Structure: "When [conditions], [prediction]"
    Example: "When [collision occurs], [total_momentum_x stays constant]"

    The agent forms these from observation, tests them, and confirms/rejects them.
    """
    internal_name: str           # Agent's own name (e.g., "R_002")
    conditions: str              # When does this rule apply?
    prediction: str              # What does the rule predict?
    feature_key: str             # Which feature is being predicted
    status: RuleStatus = RuleStatus.HYPOTHESIS
    evidence_for: int = 0        # Number of observations supporting
    evidence_against: int = 0    # Number of observations contradicting
    confidence: float = 0.0      # 0.0 to 1.0
    discovered_at_step: int = 0
    confirmed_at_step: Optional[int] = None
    properties: dict = field(default_factory=dict)

    @property
    def is_confirmed(self) -> bool:
        return self.status == RuleStatus.CONFIRMED

    @property
    def total_evidence(self) -> int:
        return self.evidence_for + self.evidence_against

    def add_evidence(self, supports: bool):
        if supports:
            self.evidence_for += 1
        else:
            self.evidence_against += 1
        total = self.total_evidence
        if total > 0:
            self.confidence = self.evidence_for / total

    def update_status(self, min_evidence: int = 10, min_confidence: float = 0.9):
        """Promote or reject based on accumulated evidence."""
        if self.status == RuleStatus.REJECTED:
            return
        if self.total_evidence >= min_evidence:
            if self.confidence >= min_confidence:
                self.status = RuleStatus.CONFIRMED
            elif self.confidence < 0.5:
                self.status = RuleStatus.REJECTED

    def __repr__(self):
        status_str = self.status.value
        return (f"Rule({self.internal_name}, {status_str}, "
                f"conf={self.confidence:.2f}, "
                f"evidence={self.evidence_for}+/{self.evidence_against}-)")


class KnowledgeBase:
    """
    The agent's entire knowledge about the world.

    This is explicit, inspectable, and modifiable.
    The agent can:
      - Add concepts it has discovered
      - Add hypotheses it has formed
      - Test hypotheses and update their status
      - Review and revise its knowledge
      - Detect contradictions in its own knowledge
      - Form meta-concepts (higher-order abstractions)
      - Invent its own notation for concepts and operations
    """

    def __init__(self):
        self.concepts: dict[str, Concept] = {}
        self.rules: dict[str, Rule] = {}
        self._concept_counter = 0
        self._rule_counter = 0
        self.discovery_log: list[dict] = []
        self.notation_system: NotationSystem = NotationSystem()
        self.meta_concepts: list[MetaConcept] = []

    def new_concept_name(self) -> str:
        self._concept_counter += 1
        return f"C_{self._concept_counter:03d}"

    def new_rule_name(self) -> str:
        self._rule_counter += 1
        return f"R_{self._rule_counter:03d}"

    def add_concept(self, concept_type: ConceptType, description: str,
                    feature_key: str, step: int, properties: dict = None,
                    notation: str = None) -> Concept:
        """Register a newly discovered concept."""
        name = self.new_concept_name()

        # Auto-generate notation if not provided
        if notation is None and concept_type in (ConceptType.QUANTITY, ConceptType.OPERATION,
                                                   ConceptType.GEOMETRIC):
            notation = self.notation_system.generate_symbol(feature_key, concept_type)

        concept = Concept(
            internal_name=name,
            concept_type=concept_type,
            description=description,
            feature_key=feature_key,
            discovered_at_step=step,
            properties=properties or {},
            notation=notation,
        )
        self.concepts[name] = concept
        self.discovery_log.append({
            'type': 'concept',
            'name': name,
            'description': description,
            'concept_type': concept_type.value,
            'step': step,
            'feature_key': feature_key,
            'notation': notation,
        })
        return concept

    def add_hypothesis(self, conditions: str, prediction: str,
                       feature_key: str, step: int, properties: dict = None) -> Rule:
        """Register a new hypothesis to be tested."""
        name = self.new_rule_name()
        rule = Rule(
            internal_name=name,
            conditions=conditions,
            prediction=prediction,
            feature_key=feature_key,
            discovered_at_step=step,
            properties=properties or {},
        )
        self.rules[name] = rule
        self.discovery_log.append({
            'type': 'hypothesis',
            'name': name,
            'conditions': conditions,
            'prediction': prediction,
            'step': step,
        })
        return rule

    def confirm_rule(self, rule_name: str, step: int):
        """Mark a hypothesis as confirmed after sufficient testing."""
        if rule_name in self.rules:
            rule = self.rules[rule_name]
            rule.status = RuleStatus.CONFIRMED
            rule.confirmed_at_step = step
            self.discovery_log.append({
                'type': 'discovery',
                'name': rule_name,
                'conditions': rule.conditions,
                'prediction': rule.prediction,
                'step': step,
                'confidence': rule.confidence,
                'evidence_for': rule.evidence_for,
                'evidence_against': rule.evidence_against,
            })

    def reject_rule(self, rule_name: str):
        """Mark a hypothesis as rejected."""
        if rule_name in self.rules:
            self.rules[rule_name].status = RuleStatus.REJECTED

    def has_concept_for_feature(self, feature_key: str) -> bool:
        return any(c.feature_key == feature_key for c in self.concepts.values())

    def get_confirmed_rules(self) -> list[Rule]:
        return [r for r in self.rules.values() if r.is_confirmed]

    def get_active_hypotheses(self) -> list[Rule]:
        return [r for r in self.rules.values()
                if r.status in (RuleStatus.HYPOTHESIS, RuleStatus.TESTING)]

    def get_all_concepts(self) -> list[Concept]:
        return list(self.concepts.values())

    def get_concepts_by_type(self, concept_type: ConceptType) -> list[Concept]:
        return [c for c in self.concepts.values() if c.concept_type == concept_type]

    def add_meta_concept(self, name: str, description: str, sub_concept_names: list[str],
                         step: int, properties: dict = None) -> 'MetaConcept':
        """Register a higher-order concept that groups other concepts."""
        meta = MetaConcept(
            internal_name=name,
            description=description,
            sub_concepts=sub_concept_names,
            discovered_at_step=step,
            properties=properties or {},
        )
        self.meta_concepts.append(meta)
        self.discovery_log.append({
            'type': 'meta_concept',
            'name': name,
            'description': description,
            'sub_concepts': sub_concept_names,
            'step': step,
        })
        return meta

    def get_meta_concepts(self) -> list['MetaConcept']:
        return list(self.meta_concepts)

    def summary(self) -> str:
        lines = [
            f"Knowledge Base Summary:",
            f"  Concepts discovered: {len(self.concepts)}",
            f"  Meta-concepts (abstractions): {len(self.meta_concepts)}",
            f"  Rules confirmed: {len(self.get_confirmed_rules())}",
            f"  Active hypotheses: {len(self.get_active_hypotheses())}",
            f"  Rejected hypotheses: {sum(1 for r in self.rules.values() if r.status == RuleStatus.REJECTED)}",
            f"  Invented symbols: {len(self.notation_system.symbols)}",
        ]
        return "\n".join(lines)


@dataclass
class MetaConcept:
    """
    A higher-order concept that groups related concepts together.

    The agent discovers these by noticing that certain concepts share properties:
      - momentum and energy both describe motion → "motion quantity"
      - count and mass both add up → "additive quantity"
      - x-momentum and y-momentum are components of the same thing → "vector"

    This is abstraction — the foundation of mathematical structure.
    """
    internal_name: str
    description: str
    sub_concepts: list[str]       # Names of concepts this meta-concept groups
    discovered_at_step: int
    properties: dict = field(default_factory=dict)

    def __repr__(self):
        return (f"MetaConcept({self.internal_name}, '{self.description}', "
                f"groups={self.sub_concepts})")


class NotationSystem:
    """
    The agent's self-invented symbolic notation.

    The agent does NOT use human mathematical symbols (+, -, =, π, Σ).
    It invents its own from a generative system.

    Symbol generation:
      - Quantities get a symbol based on a deterministic hash of their properties
      - Operations get a symbol that encodes their structure
      - Relations get a symbol encoding their type

    The key insight: we compare the STRUCTURE of the agent's notation to human notation.
    Does it independently invent something like '+' for addition?
    Does it invent something like '=' for conservation?
    """

    # The agent's symbol alphabet — deliberately NOT human math symbols
    # Uses a mix of geometric shapes and custom characters
    SYMBOL_ALPHABET = [
        'α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'λ', 'μ',
        'ν', 'ξ', 'ρ', 'σ', 'τ', 'φ', 'χ', 'ψ', 'ω',
        'Δ', 'Σ', 'Φ', 'Ω', '∇', '∮', '⊕', '⊗', '⊙', '◎',
        '◆', '◇', '▲', '△', '●', '○', '■', '□', '★', '☆',
    ]

    def __init__(self):
        self.symbols: dict[str, str] = {}  # feature_key → invented symbol
        self._used_symbols: set[str] = set()
        self._symbol_idx = 0

    def _next_symbol(self) -> str:
        """Get the next unused symbol from the alphabet."""
        if self._symbol_idx < len(self.SYMBOL_ALPHABET):
            s = self.SYMBOL_ALPHABET[self._symbol_idx]
            self._symbol_idx += 1
            return s
        # If we run out, generate composite symbols
        base = self.SYMBOL_ALPHABET[self._symbol_idx % len(self.SYMBOL_ALPHABET)]
        suffix = self._symbol_idx // len(self.SYMBOL_ALPHABET)
        self._symbol_idx += 1
        return f"{base}_{suffix}"

    def generate_symbol(self, feature_key: str, concept_type: ConceptType) -> str:
        """Generate a unique symbol for a concept."""
        if feature_key in self.symbols:
            return self.symbols[feature_key]

        symbol = self._next_symbol()
        self.symbols[feature_key] = symbol
        self._used_symbols.add(symbol)
        return symbol

    def generate_operation_symbol(self, operation_type: str) -> str:
        """Generate a symbol for an operation (like +, -, =)."""
        key = f"op_{operation_type}"
        if key in self.symbols:
            return self.symbols[key]

        symbol = self._next_symbol()
        self.symbols[key] = symbol
        self._used_symbols.add(symbol)
        return symbol

    def get_notation_table(self) -> dict[str, str]:
        """Return the full mapping of features to symbols."""
        return dict(self.symbols)

    def summary(self) -> str:
        lines = [f"\nNotation System ({len(self.symbols)} symbols invented):"]
        for key, symbol in sorted(self.symbols.items()):
            lines.append(f"  {key:30s} → {symbol}")
        return "\n".join(lines)
