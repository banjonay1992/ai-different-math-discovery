from __future__ import annotations

"""
Comparison — maps the agent's independently discovered concepts to human concepts.

This is the convergence test. If the agent, starting from nothing,
discovers the same mathematical/physical laws that humans did,
then those laws are properties of reality — not human conventions.

Mapping is done by examining:
  - Which feature the concept is about
  - What type of regularity was discovered
  - The structure of the rule

We do NOT tell the agent about these mappings.
They are applied AFTER discovery, for evaluation only.
"""

from .tracker import DiscoveryTracker
from agent.representation import KnowledgeBase, ConceptType, RuleStatus


# Mapping from feature keys to human mathematical/physical concepts
FEATURE_TO_HUMAN = {
    'count': {
        'concept': 'Natural Numbers (Counting)',
        'description': 'The set of non-negative integers. The most fundamental mathematical concept.',
        'field': 'Mathematics',
    },
    'total_momentum_x': {
        'concept': 'Linear Momentum (x-component)',
        'description': 'p = Σ(mᵢ·vᵢ). A conserved vector quantity in physics.',
        'field': 'Physics',
    },
    'total_momentum_y': {
        'concept': 'Linear Momentum (y-component)',
        'description': 'p = Σ(mᵢ·vᵢ). A conserved vector quantity in physics.',
        'field': 'Physics',
    },
    'total_momentum': {
        'concept': 'Linear Momentum (magnitude)',
        'description': '|p| = |Σ(mᵢ·vᵢ)|. The total momentum of a system.',
        'field': 'Physics',
    },
    'total_kinetic_energy': {
        'concept': 'Kinetic Energy',
        'description': 'KE = Σ(½·mᵢ·vᵢ²). The energy of motion.',
        'field': 'Physics',
    },
    'total_mass': {
        'concept': 'Mass (Additivity)',
        'description': 'M = Σ(mᵢ). Mass is an additive, conserved quantity.',
        'field': 'Physics',
    },
    'center_of_mass_x': {
        'concept': 'Center of Mass (x)',
        'description': 'x_cm = Σ(mᵢ·xᵢ)/M. The mass-weighted average position.',
        'field': 'Physics',
    },
    'center_of_mass_y': {
        'concept': 'Center of Mass (y)',
        'description': 'y_cm = Σ(mᵢ·yᵢ)/M. The mass-weighted average position.',
        'field': 'Physics',
    },
    'mean_distance': {
        'concept': 'Euclidean Distance / Metric',
        'description': 'd = √((x₁-x₂)² + (y₁-y₂)²). The spatial separation between objects.',
        'field': 'Geometry',
    },
    'mean_speed': {
        'concept': 'Average Speed',
        'description': 'v̄ = Σ|vᵢ|/n. The mean magnitude of velocity.',
        'field': 'Physics',
    },
    'num_collisions': {
        'concept': 'Collision / Interaction Events',
        'description': 'Discrete events where objects make contact and exchange momentum.',
        'field': 'Physics',
    },
}

# Mapping for global conservation rules
GLOBAL_CONSERVATION_TO_HUMAN = {
    'total_mass': {
        'concept': "Conservation of Mass",
        'description': "Mass is neither created nor destroyed in physical interactions.",
        'field': 'Physics',
    },
}

# Mapping for collision-specific conservation rules
COLLISION_CONSERVATION_TO_HUMAN = {
    'total_momentum_x': {
        'concept': "Conservation of Momentum (x-component) during Collisions",
        'description': "Newton's Third Law: during collisions, momentum is transferred between objects "
                       "but the total x-momentum is conserved. The collision itself doesn't change it.",
        'field': 'Physics',
    },
    'total_momentum_y': {
        'concept': "Conservation of Momentum (y-component) during Collisions",
        'description': "Newton's Third Law: during collisions, momentum is transferred between objects "
                       "but the total y-momentum is conserved. The collision itself doesn't change it.",
        'field': 'Physics',
    },
    'total_momentum': {
        'concept': "Conservation of Momentum during Collisions",
        'description': "Newton's Third Law: during collisions, total momentum is conserved. "
                       "Momentum is transferred between objects but the total doesn't change. "
                       "One of the most fundamental laws of physics, discovered independently.",
        'field': 'Physics',
    },
    'total_kinetic_energy': {
        'concept': "Conservation of Kinetic Energy during Collisions (elastic)",
        'description': "In perfectly elastic collisions, total kinetic energy is conserved.",
        'field': 'Physics',
    },
}

# Mapping for arithmetic rules
ARITHMETIC_TO_HUMAN = {
    1: {
        'concept': "Successor Function / Addition by One",
        'description': "The Peano axiom: n + 1 = S(n). Adding one object increases count by one.",
        'field': 'Mathematics',
    },
    -1: {
        'concept': "Predecessor Function / Subtraction by One",
        'description': "n - 1 = P(n). Removing one object decreases count by one.",
        'field': 'Mathematics',
    },
}

# Mapping for spatial/geometric rules
SPATIAL_TO_HUMAN = {
    'triangle_inequality': {
        'concept': "Triangle Inequality",
        'description': "For any three points, the direct distance is never longer than "
                       "going through an intermediate point. A fundamental property of metric spaces.",
        'field': 'Geometry',
    },
    'transitivity': {
        'concept': "Transitivity of Spatial Ordering",
        'description': "If A is left of B and B is left of C, then A is left of C. "
                       "Spatial ordering forms a total order — a fundamental property of 1D arrangement.",
        'field': 'Mathematics',
    },
    'bounding': {
        'concept': "Spatial Bounding / Compactness",
        'description': "Objects are confined to a bounded region. The world has finite spatial extent. "
                       "A fundamental property of physical space.",
        'field': 'Geometry',
    },
}

# Mapping for meta-concepts (abstractions)
META_TO_HUMAN = {
    'involves_velocity': {
        'concept': "Kinematic Quantities (Class of Motion-Related Measures)",
        'description': "The recognition that momentum, energy, and speed are all quantities "
                       "that characterize motion. This is the concept of a physical quantity class.",
        'field': 'Physics',
    },
    'additive': {
        'concept': "Additive / Extensive Quantities",
        'description': "Quantities where the whole equals the sum of parts. "
                       "This is the concept of additivity — central to both mathematics and physics.",
        'field': 'Mathematics',
    },
    'collision_conserved': {
        'concept': "Conserved Quantities (Class of Invariants)",
        'description': "The recognition that multiple quantities share the property of being "
                       "conserved during interactions. This is the concept of a conservation law class.",
        'field': 'Physics',
    },
    'vector_components': {
        'concept': "Vector Quantities (Orthogonal Components)",
        'description': "The recognition that some quantities exist as paired x/y components — "
                       "orthogonal aspects of a single underlying vector. This is the concept of a vector.",
        'field': 'Mathematics',
    },
}

# Mapping for operation notation
OPERATION_TO_HUMAN = {
    'successor': {
        'concept': "Addition Operator (+)",
        'description': "The agent invented its own symbol for the successor operation — "
                       "the fundamental operation of arithmetic.",
        'field': 'Mathematics',
    },
    'predecessor': {
        'concept': "Subtraction Operator (-)",
        'description': "The agent invented its own symbol for the predecessor operation.",
        'field': 'Mathematics',
    },
    'conservation': {
        'concept': "Equality / Invariance Operator (=)",
        'description': "The agent invented its own symbol for conservation — "
                       "the concept that a quantity remains unchanged under transformation.",
        'field': 'Mathematics',
    },
}


def map_discoveries(kb: KnowledgeBase, tracker: DiscoveryTracker):
    """
    Map all discoveries in the knowledge base to human concepts.
    Called after the experiment to evaluate convergence.
    """
    # Map concepts (including operations with self-generated notation)
    for concept in kb.get_all_concepts():
        feature_key = concept.feature_key
        human = FEATURE_TO_HUMAN.get(feature_key)

        if human:
            tracker.record_concept(
                concept=concept,
                human_mapping=human['concept'],
                human_description=human['description'],
            )
        elif concept.concept_type == ConceptType.OPERATION:
            # Map operation notation to human operators
            op_name = concept.properties.get('operation', '')
            if op_name in ('successor', 'predecessor'):
                human = OPERATION_TO_HUMAN.get(op_name)
                if human:
                    tracker.record_concept(
                        concept=concept,
                        human_mapping=human['concept'],
                        human_description=human['description'],
                    )
                else:
                    tracker.record_concept(concept=concept)
            elif op_name == 'conservation':
                human = OPERATION_TO_HUMAN.get('conservation')
                if human:
                    tracker.record_concept(
                        concept=concept,
                        human_mapping=human['concept'],
                        human_description=human['description'],
                    )
                else:
                    tracker.record_concept(concept=concept)
            else:
                tracker.record_concept(concept=concept)
        else:
            tracker.record_concept(concept=concept)

    # Map meta-concepts (abstractions)
    for meta in kb.get_meta_concepts():
        criterion = meta.properties.get('grouping_criterion', '')
        human = META_TO_HUMAN.get(criterion)
        if human:
            tracker.record_meta_concept(
                meta=meta,
                human_mapping=human['concept'],
                human_description=human['description'],
            )
        else:
            tracker.record_meta_concept(meta=meta)

    # Map confirmed rules
    for rule in kb.get_confirmed_rules():
        hypo_type = rule.properties.get('hypothesis_type', '')

        if hypo_type == 'global_conservation':
            human = GLOBAL_CONSERVATION_TO_HUMAN.get(rule.feature_key)
            if human:
                tracker.record_rule(
                    rule=rule,
                    human_mapping=human['concept'],
                    human_description=human['description'],
                )
            else:
                tracker.record_rule(rule=rule)

        elif hypo_type == 'collision_conservation':
            human = COLLISION_CONSERVATION_TO_HUMAN.get(rule.feature_key)
            if human:
                tracker.record_rule(
                    rule=rule,
                    human_mapping=human['concept'],
                    human_description=human['description'],
                )
            else:
                tracker.record_rule(rule=rule)

        elif hypo_type == 'arithmetic':
            delta = rule.properties.get('delta', 0)
            human = ARITHMETIC_TO_HUMAN.get(delta)
            if human:
                tracker.record_rule(
                    rule=rule,
                    human_mapping=human['concept'],
                    human_description=human['description'],
                )
            else:
                tracker.record_rule(rule=rule)

        elif hypo_type == 'spatial':
            spatial_rule = rule.properties.get('spatial_rule', '')
            human = SPATIAL_TO_HUMAN.get(spatial_rule)
            if human:
                tracker.record_rule(
                    rule=rule,
                    human_mapping=human['concept'],
                    human_description=human['description'],
                )
            else:
                tracker.record_rule(rule=rule)

        elif hypo_type == 'novel_physics':
            # Novel physics discoveries are NOT in human knowledge — they're genuinely new
            novel_type = rule.properties.get('novel_type', 'unknown')
            if novel_type == 'central_force':
                tracker.record_rule(
                    rule=rule,
                    human_mapping="Central Force / Gravitational Well (Novel Discovery)",
                    human_description="The agent discovered an attractive central force — "
                                      "analogous to gravity from a point mass. This is a novel "
                                      "discovery in the context of this world.",
                )
            elif novel_type == 'repulsion':
                tracker.record_rule(
                    rule=rule,
                    human_mapping="Repulsion Zone (Novel Discovery)",
                    human_description="The agent discovered a repulsive force field — "
                                      "a region that pushes objects away. Novel physics not "
                                      "present in the standard world.",
                )
            elif novel_type == 'zero_gravity':
                tracker.record_rule(
                    rule=rule,
                    human_mapping="Absence of Uniform Gravitational Field (Novel Discovery)",
                    human_description="The agent detected the ABSENCE of gravity — "
                                      "recognizing that a expected force is missing. "
                                      "This is detecting a null result, which is fundamental "
                                      "to scientific reasoning.",
                )
            elif novel_type == 'uniform_horizontal_force':
                tracker.record_rule(
                    rule=rule,
                    human_mapping="Uniform Horizontal Force Field (Novel Discovery)",
                    human_description="The agent discovered a steady sideways acceleration "
                                      "layered on top of ordinary gravity.",
                )
            elif novel_type == 'inverse_square_repulsion':
                tracker.record_rule(
                    rule=rule,
                    human_mapping="Inverse-Square Repulsive Source (Novel Discovery)",
                    human_description="The agent discovered a point source that pushes objects "
                                      "outward with strength decreasing by distance.",
                )
            elif novel_type == 'vortex':
                tracker.record_rule(
                    rule=rule,
                    human_mapping="Vortex / Tangential Force Field (Novel Discovery)",
                    human_description="The agent discovered a rotational field where objects "
                                      "accelerate tangentially around a point.",
                )
            elif novel_type == 'time_varying_force':
                tracker.record_rule(
                    rule=rule,
                    human_mapping="Time-Varying Force Field (Novel Discovery)",
                    human_description="The agent discovered a global acceleration that changes "
                                      "direction over time.",
                )
            else:
                tracker.record_rule(rule=rule)
        else:
            tracker.record_rule(rule=rule)
