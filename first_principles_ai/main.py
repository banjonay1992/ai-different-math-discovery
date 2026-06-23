"""
First Principles AI — Discovery from Scratch

An agent starts with ZERO knowledge about mathematics or physics.
It observes a 2D physics world and must discover concepts independently:
  - Counting (natural numbers)
  - Conservation of momentum
  - Conservation of mass
  - Arithmetic (addition/subtraction by one)
  - Spatial relationships
  - Energy concepts

After the experiment, we compare the agent's discoveries to human concepts.
If they converge, those concepts are properties of reality — not human inventions.

Run: python main.py
"""

import sys
import os
import contextlib
import io
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from world.environment import Environment
from world.physics import PhysicsObject
from world.hidden_worlds import (
    HiddenWorldManifest,
    generate_hidden_world_manifest,
    hidden_manifest_from_observation,
)
from agent.perception import Perception
from agent.representation import KnowledgeBase, RuleStatus, ConceptType
from agent.predictor import Predictor
from agent.curiosity import Curiosity
from agent.causal import CausalReasoner
from agent.synthesis import ProgramSynthesizer
from agent.language import MultiAgentExperiment
from agent.self_modify import SelfModifier
from agent.law_memory import LawMemory
from agent.math_discovery import EmergentMathDiscovery
from agent.equation_workbench import EquationWorkbench
from agent.compute_budget import plan_adaptive_compute_budget
from agent.discovery_loop import CumulativeTheoryMemory
from agent.math_foundation import MathFoundationWorkbench
from agent.explorer import (
    ExplorationPlanner,
    explain_exploration_outcome,
    score_exploration_outcome,
)
from discovery.tracker import DiscoveryTracker
from discovery.comparison import map_discoveries


WORLD_TYPES = [
    'standard',
    'central_force',
    'repulsion',
    'zero_gravity',
    'sideways_wind',
    'vortex',
    'inverse_square_repulsion',
    'localized_gravity',
    'time_varying',
]

HIDDEN_DISCOVERY_TARGETS = {
    'uniform_component',
    'radial_component',
    'repulsive_component',
    'tangential_component',
    'time_varying_component',
    'composed_law',
}

EXPECTED_NOVEL_BY_WORLD = {
    'standard': set(),
    'central_force': {'central_force'},
    'repulsion': {'repulsion'},
    'zero_gravity': {'zero_gravity'},
    'sideways_wind': {'uniform_horizontal_force'},
    'vortex': {'vortex'},
    'inverse_square_repulsion': {'repulsion'},
    'localized_gravity': {'central_force'},
    'time_varying': {'time_varying_force'},
}

MATH_BENCHMARK_TARGETS = {
    'identity / object permanence',
    'equality-like invariant',
    'discrete quantity / cardinality',
    'integer-like change',
    'successor-like operation',
    'predecessor-like operation',
    'iteration / composition',
    'inverse operation pair',
    'operation composition',
    'identity-like cancellation',
    'operation equivalence class',
    'commutativity-like operation behavior',
    'order relation',
    'metric / distance structure',
    'ratio-like relation',
    'symmetry-like balance',
    'periodicity-like recurrence',
    'function-like mapping',
    'conditional rule',
}

FORBIDDEN_AGENT_MATH_LABELS = {
    'addition',
    'subtraction',
    'successor',
    'predecessor',
    'composition',
    'identity',
    'equivalence',
    'commutativity',
    'equality',
}


def run_experiment(
    num_steps: int = 2000,
    num_initial_objects: int = 5,
    seed: int = 42,
    verbose: bool = True,
    report_interval: int = 100,
    world_type: str = 'standard',
    num_agents: int = 2,
    law_memory: LawMemory = None,
    hidden_manifest: HiddenWorldManifest | None = None,
    allow_memory_probes: bool = True,
    enable_equation_probes: bool = False,
    planned_actions: list[dict] | None = None,
    equation_operator_priors: list[dict] | None = None,
    equation_max_operator_feedback_rows: int = 384,
    equation_max_operator_feedback_operators: int = 5,
):
    """
    Run the first-principles discovery experiment.

    Args:
        num_steps: How many timesteps to run
        num_initial_objects: Starting objects in the world
        seed: Random seed for reproducibility
        verbose: Print progress
        report_interval: How often to print status updates
    """
    print("=" * 70)
    print("FIRST PRINCIPLES AI — Discovery from Scratch")
    print("=" * 70)
    print()
    print("An agent with ZERO prior knowledge observes a physics world.")
    print("It must discover mathematical and physical concepts independently.")
    print("No training data. No human examples. Just observation and reasoning.")
    print()
    print(f"Configuration:")
    print(f"  Steps: {num_steps}")
    print(f"  Initial objects: {num_initial_objects}")
    print(f"  Seed: {seed}")
    print(f"  World type: {world_type if hidden_manifest is None else 'hidden_procedural'}")
    print(f"  Agents: {num_agents}")
    print()
    print("-" * 70)
    print("EXPERIMENT STARTING...")
    print("-" * 70)

    # Initialize
    PhysicsObject._next_id = 0
    env = Environment(
        num_initial_objects=num_initial_objects,
        seed=seed,
        world_type=world_type,
        hidden_manifest=hidden_manifest,
    )
    kb = KnowledgeBase()
    law_priors = law_memory.suggest_priors() if law_memory is not None else []
    predictor = Predictor(
        kb,
        law_priors=law_priors,
        allow_memory_probes=allow_memory_probes,
    )
    curiosity = Curiosity(env)
    causal = CausalReasoner()
    synthesizer = ProgramSynthesizer(kb)
    tracker = DiscoveryTracker()
    multi_agent = MultiAgentExperiment(num_agents=num_agents, seed=seed)
    self_modifier = SelfModifier(predictor, curiosity)
    math_discovery = EmergentMathDiscovery(kb)
    kb.emergent_math_discovery = math_discovery
    equation_workbench = EquationWorkbench(
        kb,
        generated_operator_priors=equation_operator_priors,
        max_operator_feedback_rows=equation_max_operator_feedback_rows,
        max_operator_feedback_operators=equation_max_operator_feedback_operators,
    )
    kb.equation_workbench = equation_workbench

    # Track what we've already reported to avoid duplicate prints
    reported_concepts = set()
    reported_rules = set()

    prev_discovery_count = 0

    for step in range(num_steps):
        # 1. OBSERVE the world
        raw_state = env.observe()
        observation = Perception.perceive(raw_state)
        features = observation.get_feature_vector()
        had_collision = len(observation.collisions) > 0

        # 2. PREDICT what happens next (using current knowledge)
        predicted = predictor.predict_next(features)

        # 3. SELECT an action (planned falsification, active experiment, curiosity)
        action = None
        if planned_actions and step < len(planned_actions):
            action = dict(planned_actions[step])
        if action is None:
            action = predictor.suggest_experiment_action(
                current_count=features.get('count', 0),
                world_width=observation.world_width or 20.0,
                world_height=observation.world_height or 20.0,
            )
        if action is None and enable_equation_probes:
            if step > 0 and step % 40 == 0:
                equation_workbench.discover(step=step)
            action = equation_workbench.suggest_probe_action(
                current_count=features.get('count', 0),
                world_width=observation.world_width or 20.0,
                world_height=observation.world_height or 20.0,
                step=step,
            )
        if action is None:
            action = curiosity.select_action(predictor, features)

        # 4. ACT and observe the outcome
        state_before = raw_state
        raw_state = env.step(action)
        math_discovery.observe_transition(state_before, raw_state, action, step + 1)
        equation_workbench.observe_transition(state_before, raw_state, action, step + 1)
        observation = Perception.perceive(raw_state)
        new_features = observation.get_feature_vector()
        had_collision = len(observation.collisions) > 0

        # 5. LEARN — record observations and check for discoveries
        predictor.observe(new_features, had_collision, step + 1, raw_objects=raw_state.get('objects', []))

        # 6. CAUSAL REASONING — track actions and effects
        causal.observe_step(action, state_before, raw_state, step + 1, observation.collisions)

        # 7. PROGRAM SYNTHESIS — record actions for pattern detection
        synthesizer.record_action(action, step + 1, new_features)

        # 8. MULTI-AGENT LANGUAGE — agents communicate about world state
        multi_agent.step_agents(
            new_features,
            had_collision,
            step + 1,
            math_patterns=math_discovery.discovered_patterns(),
        )

        # 9. SELF-MODIFICATION — agent inspects and improves its own reasoning
        if kb.discovery_log and len(kb.discovery_log) > prev_discovery_count:
            self_modifier.record_discovery(step + 1)
        self_modifier.check_and_modify(step + 1)

        # 10. Calculate prediction error (drives future curiosity)
        error = predictor.prediction_error(predicted, new_features)

        # 11. Decay exploration over time
        curiosity.decay_exploration()

        # 9. Report new discoveries in real-time
        if verbose:
            new_discoveries = len(kb.discovery_log) - prev_discovery_count
            if new_discoveries > 0:
                for log_entry in kb.discovery_log[prev_discovery_count:]:
                    _report_discovery(log_entry, step + 1)
                prev_discovery_count = len(kb.discovery_log)

            # Periodic status
            if (step + 1) % report_interval == 0:
                confirmed = len(kb.get_confirmed_rules())
                concepts = len(kb.get_all_concepts())
                hypotheses = len(kb.get_active_hypotheses())
                causal_events = len(causal.graph.event_history)
                print(
                    f"  [Step {step + 1:5d}] "
                    f"Concepts: {concepts} | "
                    f"Confirmed: {confirmed} | "
                    f"Hypotheses: {hypotheses} | "
                    f"Causal events: {causal_events} | "
                    f"Objects: {new_features.get('count', 0)} | "
                    f"Error: {error:.3f}"
                )

    # Final: map discoveries to human concepts
    equation_workbench.discover(step=num_steps)
    math_foundation = MathFoundationWorkbench(
        kb,
        math_discovery=math_discovery,
        equation_workbench=equation_workbench,
    )
    foundation_report = math_foundation.evaluate(install=True)
    kb.math_foundation_report = foundation_report

    print()
    print("-" * 70)
    print("MAPPING DISCOVERIES TO HUMAN CONCEPTS...")
    print("-" * 70)

    map_discoveries(kb, tracker)

    # Print final summary
    print(tracker.summary())

    print()
    print("-" * 70)
    print("KNOWLEDGE BASE DETAILS")
    print("-" * 70)
    print(kb.summary())

    print("\nConfirmed Rules:")
    for rule in kb.get_confirmed_rules():
        print(f"  {rule.internal_name}: WHEN {rule.conditions}, THEN {rule.prediction}")
        print(f"    Confidence: {rule.confidence:.2f} "
              f"(Evidence: {rule.evidence_for}+ / {rule.evidence_against}-)")

    print("\nAll Concepts:")
    for concept in kb.get_all_concepts():
        print(f"  {concept.internal_name} ({concept.concept_type.value}): {concept.description}")

    # Causal reasoning report
    print()
    print("-" * 70)
    print("CAUSAL REASONING REPORT")
    print("-" * 70)
    print(causal.graph.summary())

    # Causal queries
    print("\nCausal Queries:")
    for action_type in ['push', 'spawn', 'remove']:
        result = causal.answer_what_happens_if(action_type)
        if result:
            effect, prob = result
            print(f"  What happens if agent does '{action_type}'? → {effect} (P={prob:.2f})")

    for event_type in ['collision', 'motion', 'appearance']:
        causes = causal.answer_what_caused(event_type)
        if causes:
            top_cause, prob = causes[0]
            print(f"  What causes '{event_type}'? → {top_cause} (P={prob:.2f})")

    # Counterfactual example
    print("\nCounterfactual Reasoning:")
    cf = causal.counterfactual('push', 'collision')
    print(f"  {cf}")
    cf2 = causal.counterfactual('spawn', 'appearance')
    print(f"  {cf2}")

    # Abstraction hierarchy report
    print()
    print("-" * 70)
    print("ABSTRACTION HIERARCHY REPORT")
    print("-" * 70)
    meta_concepts = kb.get_meta_concepts()
    if meta_concepts:
        print(f"\nMeta-concepts discovered: {len(meta_concepts)}")
        for mc in meta_concepts:
            print(f"\n  {mc.internal_name}: {mc.description}")
            print(f"    Groups: {', '.join(mc.sub_concepts)}")
    else:
        print("\n  No meta-concepts discovered yet.")

    # Self-generated notation report
    print()
    print("-" * 70)
    print("SELF-GENERATED NOTATION REPORT")
    print("-" * 70)
    print(kb.notation_system.summary())

    # Show operations with their invented symbols
    operations = kb.get_concepts_by_type(ConceptType.OPERATION)
    if operations:
        print(f"\nInvented Operations ({len(operations)}):")
        for op in operations:
            print(f"  {op.internal_name} (symbol: {op.notation}): {op.description}")

    # Raw mathematical structure report
    print()
    print("-" * 70)
    print("EMERGENT MATH DISCOVERY REPORT")
    print("-" * 70)
    print(math_discovery.summary())

    # Equation workbench report
    print()
    print("-" * 70)
    print("EQUATION WORKBENCH REPORT")
    print("-" * 70)
    print(equation_workbench.summary())

    # Math foundation readiness report
    print()
    print("-" * 70)
    print("MATH FOUNDATION READINESS REPORT")
    print("-" * 70)
    print(foundation_report.summary())

    # Program synthesis report
    print()
    print("-" * 70)
    print("PROGRAM SYNTHESIS REPORT")
    print("-" * 70)
    print(synthesizer.summary())

    # Novel physics report
    novel_rules = [r for r in kb.get_confirmed_rules()
                   if r.properties.get('hypothesis_type') == 'novel_physics']
    print()
    print("-" * 70)
    print("NOVEL PHYSICS DISCOVERY REPORT")
    print("-" * 70)
    if novel_rules:
        print(f"\nNovel physics rules confirmed: {len(novel_rules)}")
        for rule in novel_rules:
            novel_type = rule.properties.get('novel_type', 'unknown')
            print(f"\n  {rule.internal_name} ({novel_type})")
            print(f"    WHEN: {rule.conditions}")
            print(f"    THEN: {rule.prediction}")
            print(f"    Confidence: {rule.confidence:.2f} "
                  f"(Evidence: {rule.evidence_for}+ / {rule.evidence_against}-)")
            if 'attractor_x' in rule.properties:
                print(f"    Detected attractor at: ({rule.properties['attractor_x']}, {rule.properties['attractor_y']})")
            if 'repeller_x' in rule.properties:
                print(f"    Detected repeller at: ({rule.properties['repeller_x']}, {rule.properties['repeller_y']})")
            if 'vortex_x' in rule.properties:
                print(f"    Detected vortex at: ({rule.properties['vortex_x']}, {rule.properties['vortex_y']})")
                print(f"    Spin: {rule.properties.get('spin', 'unknown')}")
            if 'direction' in rule.properties:
                print(f"    Direction: {rule.properties['direction']}")
            if 'axis' in rule.properties:
                print(f"    Axis: {rule.properties['axis']}")
            for key in (
                'detection_confidence',
                'latest_residual_confidence',
                'latest_profile_confidence',
                'latest_uniform_confidence',
                'latest_vortex_confidence',
                'latest_time_confidence',
            ):
                if key in rule.properties:
                    label = key.replace('_', ' ').title()
                    print(f"    {label}: {rule.properties[key]}")
            if rule.properties.get('active_experiments'):
                print(f"    Active experiments run: {rule.properties['active_experiments']}")
        print(f"\n  → The agent discovered physics laws that DON'T exist in the standard world!")
        print(f"  → This is genuine novel discovery, not rediscovery of human knowledge.")
    else:
        print("\n  No novel physics discovered (standard world — as expected)")
    _print_detector_diagnostics(predictor)
    _print_learned_dynamics_laws(predictor)
    if law_memory is not None:
        law_memory.record_experiment(
            world_type=world_type,
            seed=seed,
            object_count=num_initial_objects,
            step_count=num_steps,
            learned_laws=predictor.get_learned_dynamics_laws(),
            confirmed_novel_types=_confirmed_novel_types(kb),
            diagnostics=predictor.get_novel_physics_diagnostics(),
        )
        law_memory.install_principles(kb, step=num_steps)

    # Multi-agent language emergence report
    print()
    print("-" * 70)
    print("MULTI-AGENT LANGUAGE EMERGENCE REPORT")
    print("-" * 70)
    print(multi_agent.summary())

    # Self-modification report
    print()
    print("-" * 70)
    print("SELF-MODIFICATION REPORT")
    print("-" * 70)
    print(self_modifier.summary())

    return tracker, kb, causal


def _print_detector_diagnostics(predictor: Predictor):
    diagnostics = predictor.get_novel_physics_diagnostics()
    if not diagnostics:
        return

    print("\nDetector diagnostics:")
    for key, details in sorted(diagnostics.items()):
        detected = details.get('detected')
        status = "detected" if detected else "not detected"
        confidence = details.get('confidence')
        conf_text = f", confidence={confidence}" if confidence is not None else ""
        reason = details.get('reason', '')
        print(f"  {key}: {status}{conf_text} — {reason}")


def _print_learned_dynamics_laws(predictor: Predictor):
    learned_laws = predictor.get_learned_dynamics_laws()
    if not learned_laws:
        return

    print("\nLearned dynamics laws:")
    for law in learned_laws[:5]:
        print(
            f"  {law['name']}: confidence={law['confidence']:.2f}, "
            f"improvement={law['improvement']:.2f}"
        )
        print(f"    {law['description']}")


def _confirmed_novel_rules(kb: KnowledgeBase) -> list:
    return [
        rule for rule in kb.get_confirmed_rules()
        if rule.properties.get('hypothesis_type') == 'novel_physics'
    ]


def _confirmed_novel_types(kb: KnowledgeBase) -> set[str]:
    return {
        rule.properties.get('novel_type')
        for rule in _confirmed_novel_rules(kb)
    }


def _novel_types_from_learned_laws(learned_laws: list[dict]) -> set[str]:
    """Map strong general dynamics laws to benchmark-level physics discoveries."""
    discovered = set()
    for law in learned_laws:
        confidence = float(law.get('confidence', 0.0))
        if confidence < 0.45:
            continue
        law_type = law.get('law_type')
        properties = law.get('properties', {})
        if law_type == 'tangential_field':
            discovered.add('vortex')
        elif law_type == 'time_varying_field':
            discovered.add('time_varying_force')
        elif law_type in {'radial_field', 'inverse_square_radial_field'}:
            if properties.get('direction') == 'repulsive':
                discovered.add('repulsion')
            elif properties.get('direction') == 'attractive':
                discovered.add('central_force')
    return discovered


def _novel_confidence(kb: KnowledgeBase, expected: set[str]) -> float:
    rules = _confirmed_novel_rules(kb)
    if expected:
        rules = [rule for rule in rules if rule.properties.get('novel_type') in expected]
    if not rules:
        return 0.0
    return max(rule.confidence for rule in rules)


def run_benchmark(
    seeds: int = 5,
    steps: int = 800,
    object_counts: list[int] = None,
    world_types: list[str] = None,
    num_agents: int = 2,
    law_memory: LawMemory = None,
) -> list[dict]:
    """Run a seed/object-count sweep over all configured worlds."""
    object_counts = object_counts or [5]
    world_types = world_types or WORLD_TYPES
    results = []
    law_memory = law_memory or LawMemory()

    print("=" * 70)
    print("NOVEL PHYSICS BENCHMARK")
    print("=" * 70)
    print(f"Worlds: {', '.join(world_types)}")
    print(f"Seeds: 0..{seeds - 1}")
    print(f"Object counts: {', '.join(str(count) for count in object_counts)}")
    print(f"Steps per run: {steps}")
    print()
    print(f"{'World':24s} {'Seed':>4s} {'Obj':>3s} {'Expected':28s} {'Detected':28s} {'Conf':>5s} Result")
    print("-" * 110)

    for world_type in world_types:
        expected = EXPECTED_NOVEL_BY_WORLD.get(world_type, set())
        for object_count in object_counts:
            for seed in range(seeds):
                with contextlib.redirect_stdout(io.StringIO()):
                    _, kb, _ = run_experiment(
                        num_steps=steps,
                        num_initial_objects=object_count,
                        seed=seed,
                        verbose=False,
                        report_interval=max(steps, 1),
                        world_type=world_type,
                        num_agents=num_agents,
                        law_memory=law_memory,
                    )

                learned_laws = (
                    law_memory.episodes[-1].learned_laws
                    if law_memory.episodes
                    else []
                )
                learned_detected = _novel_types_from_learned_laws(learned_laws)
                detected = _confirmed_novel_types(kb) | learned_detected
                confidence = _novel_confidence(kb, expected)
                if expected and confidence == 0.0 and (detected & expected):
                    confidence = max(
                        (
                            float(law.get('confidence', 0.0))
                            for law in learned_laws
                            if _novel_types_from_learned_laws([law]) & expected
                        ),
                        default=0.0,
                    )
                passed = detected == expected
                result = {
                    'world_type': world_type,
                    'seed': seed,
                    'objects': object_count,
                    'expected': expected,
                    'detected': detected,
                    'confidence': confidence,
                    'passed': passed,
                    'memory_transfer': law_memory.episodes[-1].transfer_report,
                }
                results.append(result)
                expected_text = ','.join(sorted(expected)) or 'none'
                detected_text = ','.join(sorted(detected)) or 'none'
                print(
                    f"{world_type:24s} {seed:4d} {object_count:3d} "
                    f"{expected_text:28s} {detected_text:28s} "
                    f"{confidence:5.2f} {'PASS' if passed else 'FAIL'}"
                )

    passes = sum(1 for result in results if result['passed'])
    total = len(results)
    print("-" * 110)
    print(f"Passed: {passes}/{total} ({passes / max(total, 1):.1%})")
    print()
    print(law_memory.summary())
    return results


def run_transfer_benchmark(
    seeds: int = 3,
    steps: int = 500,
    object_counts: list[int] = None,
    world_types: list[str] = None,
    num_agents: int = 2,
    law_memory: LawMemory = None,
) -> list[dict]:
    """Compare cold runs against warm runs that can use law memory priors."""
    object_counts = object_counts or [5]
    world_types = world_types or WORLD_TYPES
    warm_memory = law_memory or LawMemory()
    results = []

    print("=" * 70)
    print("COLD VS WARM LAW MEMORY BENCHMARK")
    print("=" * 70)
    print(f"Worlds: {', '.join(world_types)}")
    print(f"Seeds: 0..{seeds - 1}")
    print(f"Object counts: {', '.join(str(count) for count in object_counts)}")
    print(f"Steps per run: {steps}")
    print()
    print(
        f"{'World':24s} {'Seed':>4s} {'Obj':>3s} "
        f"{'ColdStep':>8s} {'WarmStep':>8s} {'ColdN':>5s} {'WarmN':>5s} Result"
    )
    print("-" * 84)

    for world_type in world_types:
        for object_count in object_counts:
            for seed in range(seeds):
                cold = _run_experiment_metrics(
                    world_type, seed, object_count, steps, num_agents, law_memory=None
                )
                warm = _run_experiment_metrics(
                    world_type, seed, object_count, steps, num_agents, law_memory=warm_memory
                )
                cold_step = cold['first_learned_step']
                warm_step = warm['first_learned_step']
                helped = (
                    warm['learned_rule_count'] > cold['learned_rule_count']
                    or (
                        cold_step is not None
                        and warm_step is not None
                        and warm_step <= cold_step
                    )
                    or (
                        cold_step is None
                        and warm_step is not None
                    )
                )
                result = {
                    'world_type': world_type,
                    'seed': seed,
                    'objects': object_count,
                    'cold': cold,
                    'warm': warm,
                    'helped_or_matched': helped,
                    'transfer_report': warm_memory.episodes[-1].transfer_report,
                }
                results.append(result)
                print(
                    f"{world_type:24s} {seed:4d} {object_count:3d} "
                    f"{_fmt_metric_step(cold_step):>8s} {_fmt_metric_step(warm_step):>8s} "
                    f"{cold['learned_rule_count']:5d} {warm['learned_rule_count']:5d} "
                    f"{'HELP/MATCH' if helped else 'NO HELP'}"
                )

    print("-" * 84)
    helped_count = sum(1 for result in results if result['helped_or_matched'])
    print(f"Warm memory helped or matched cold: {helped_count}/{len(results)}")
    print()
    print(warm_memory.summary())
    return results


def run_autonomous_exploration(
    budget: int = 12,
    steps: int = 500,
    object_counts: list[int] = None,
    world_types: list[str] = None,
    num_agents: int = 2,
    law_memory: LawMemory = None,
    seed_start: int = 0,
    seed_span: int = 20,
) -> list[dict]:
    """Let the agent choose a sequence of experiments from a large candidate pool."""
    object_counts = object_counts or [5]
    world_types = world_types or WORLD_TYPES
    law_memory = law_memory or LawMemory()
    planner = ExplorationPlanner(
        world_types=world_types,
        object_counts=object_counts,
        steps=steps,
        seed_start=seed_start,
        seed_span=max(seed_span, budget),
    )
    completed_keys = set()
    results = []

    print("=" * 70)
    print("AUTONOMOUS EXPLORATION LOOP")
    print("=" * 70)
    print(f"Budget: {budget} experiments")
    print(f"Candidate worlds: {', '.join(world_types)}")
    print(f"Candidate object counts: {', '.join(str(count) for count in object_counts)}")
    print(f"Seed pool: {seed_start}..{seed_start + max(seed_span, budget) - 1}")
    print(f"Steps per run: {steps}")
    print()
    print(
        f"{'Iter':>4s} {'World':24s} {'Seed':>4s} {'Obj':>3s} "
        f"{'Plan':>6s} {'Value':>6s} {'Learned':>7s} {'Novel':22s} Reason"
    )
    print("-" * 116)

    for iteration in range(1, budget + 1):
        candidates = planner.propose(law_memory, completed_keys=completed_keys, limit=1)
        if not candidates:
            break

        candidate = candidates[0]
        completed_keys.add(candidate.key)
        metrics = _run_experiment_metrics(
            world_type=candidate.world_type,
            seed=candidate.seed,
            object_count=candidate.object_count,
            steps=candidate.steps,
            num_agents=num_agents,
            law_memory=law_memory,
        )
        metrics['steps'] = candidate.steps
        value = score_exploration_outcome(metrics)
        reason = explain_exploration_outcome(metrics)
        result = {
            'iteration': iteration,
            'candidate': candidate.to_dict(),
            'metrics': metrics,
            'outcome_score': value,
            'reason': reason,
        }
        results.append(result)

        novel_text = ','.join(sorted(metrics['detected'])) or 'none'
        print(
            f"{iteration:4d} {candidate.world_type:24s} {candidate.seed:4d} "
            f"{candidate.object_count:3d} {candidate.score:6.2f} {value:6.2f} "
            f"{metrics['learned_rule_count']:7d} {novel_text:22s} {reason}"
        )

    print("-" * 116)
    print(f"Completed: {len(results)}/{budget} experiments")
    if results:
        best = max(results, key=lambda item: item['outcome_score'])
        best_candidate = best['candidate']
        print(
            "Best run: "
            f"{best_candidate['world_type']} seed={best_candidate['seed']} "
            f"objects={best_candidate['object_count']} "
            f"value={best['outcome_score']:.2f}"
        )
    print()
    print(law_memory.summary())
    return results


def run_hidden_holdout_benchmark(
    train_worlds: int = 3,
    holdout_worlds: int = 3,
    steps: int = 500,
    object_count: int = 5,
    num_agents: int = 2,
    law_memory: LawMemory = None,
    seed_start: int = 0,
) -> list[dict]:
    """Train memory on hidden worlds, then compare cold/warm holdout discovery."""
    law_memory = law_memory or LawMemory()
    results = []

    print("=" * 70)
    print("HIDDEN PROCEDURAL HOLDOUT BENCHMARK")
    print("=" * 70)
    print(f"Training hidden worlds: {train_worlds}")
    print(f"Holdout hidden worlds: {holdout_worlds}")
    print(f"Objects per run: {object_count}")
    print(f"Steps per run: {steps}")
    print()

    for index in range(train_worlds):
        manifest = generate_hidden_world_manifest(seed_start + index, variant=index)
        metrics = _run_hidden_experiment_metrics(
            manifest=manifest,
            seed=seed_start + index,
            object_count=object_count,
            steps=steps,
            num_agents=num_agents,
            law_memory=law_memory,
            allow_memory_probes=True,
        )
        results.append({
            'phase': 'train',
            'manifest': manifest.to_dict(),
            'metrics': metrics,
        })

    print(
        f"{'Phase':8s} {'Hidden':14s} {'Expected':42s} "
        f"{'Cold':>6s} {'Warm':>6s} {'Leaks':>5s} Result"
    )
    print("-" * 104)
    for index in range(holdout_worlds):
        variant = train_worlds + index
        seed = seed_start + train_worlds + index
        manifest = generate_hidden_world_manifest(seed, variant=variant)
        cold = _run_hidden_experiment_metrics(
            manifest=manifest,
            seed=seed,
            object_count=object_count,
            steps=steps,
            num_agents=num_agents,
            law_memory=None,
            allow_memory_probes=False,
        )
        warm = _run_hidden_experiment_metrics(
            manifest=manifest,
            seed=seed,
            object_count=object_count,
            steps=steps,
            num_agents=num_agents,
            law_memory=law_memory,
            allow_memory_probes=False,
        )
        helped_or_matched = (
            warm['score'] >= cold['score']
            and warm['passed']
        )
        result = {
            'phase': 'holdout',
            'manifest': manifest.to_dict(),
            'cold': cold,
            'warm': warm,
            'helped_or_matched': helped_or_matched,
        }
        results.append(result)
        expected_text = ','.join(sorted(manifest.expected_discoveries))
        print(
            f"{'holdout':8s} {manifest.hidden_id:14s} {expected_text:42s} "
            f"{cold['score']:6.2f} {warm['score']:6.2f} "
            f"{int(warm['observation_leak']):5d} "
            f"{'PASS' if helped_or_matched else 'CHECK'}"
        )

    print("-" * 104)
    holdouts = [result for result in results if result['phase'] == 'holdout']
    passes = sum(1 for result in holdouts if result['helped_or_matched'])
    print(f"Holdout warm memory helped or matched cold: {passes}/{len(holdouts)}")
    print()
    print(law_memory.summary())
    return results


def run_hidden_autonomous_exploration(
    budget: int = 12,
    steps: int = 500,
    object_count: int = 5,
    num_agents: int = 2,
    law_memory: LawMemory = None,
    seed_start: int = 0,
) -> list[dict]:
    """Run a budgeted campaign over generated hidden worlds."""
    law_memory = law_memory or LawMemory()
    results = []
    print("=" * 70)
    print("HIDDEN PROCEDURAL AUTONOMOUS CAMPAIGN")
    print("=" * 70)
    print(f"Budget: {budget} hidden experiments")
    print(f"Objects per run: {object_count}")
    print(f"Steps per run: {steps}")
    print()
    print(
        f"{'Iter':>4s} {'Hidden':14s} {'Expected':42s} "
        f"{'Score':>6s} {'Learned':>7s} {'Proposals':>9s}"
    )
    print("-" * 96)

    for iteration in range(1, budget + 1):
        seed = seed_start + iteration - 1
        manifest = generate_hidden_world_manifest(seed, variant=iteration - 1)
        metrics = _run_hidden_experiment_metrics(
            manifest=manifest,
            seed=seed,
            object_count=object_count,
            steps=steps,
            num_agents=num_agents,
            law_memory=law_memory,
            allow_memory_probes=True,
        )
        result = {
            'iteration': iteration,
            'manifest': manifest.to_dict(),
            'metrics': metrics,
        }
        results.append(result)
        print(
            f"{iteration:4d} {manifest.hidden_id:14s} "
            f"{','.join(sorted(manifest.expected_discoveries)):42s} "
            f"{metrics['score']:6.2f} {metrics['learned_rule_count']:7d} "
            f"{len(metrics['experiment_proposals']):9d}"
        )

    print("-" * 96)
    print(f"Completed: {len(results)}/{budget} hidden experiments")
    print()
    print(law_memory.summary())
    return results


def run_equation_campaign(
    seeds: int = 1,
    steps: int = 220,
    object_counts: list[int] = None,
    world_types: list[str] = None,
    hidden_worlds: int = 0,
    num_agents: int = 2,
    enable_equation_probes: bool = True,
    followup_budget: int = 0,
    theory_memory: CumulativeTheoryMemory | None = None,
) -> list[dict]:
    """Run worlds and collect equation-review packs for manual inspection."""
    object_counts = object_counts or [5]
    world_types = world_types or ['standard', 'sideways_wind', 'vortex']
    results = []
    theory_memory = theory_memory or CumulativeTheoryMemory()

    print("=" * 70)
    print("EQUATION DISCOVERY CAMPAIGN")
    print("=" * 70)
    print(f"Worlds: {', '.join(world_types)}")
    print(f"Hidden generated worlds: {hidden_worlds}")
    print(f"Seeds: 0..{seeds - 1}")
    print(f"Object counts: {', '.join(str(count) for count in object_counts)}")
    print(f"Steps per run: {steps}")
    print()
    print(
        f"{'Context':24s} {'Seed':>4s} {'Obj':>3s} "
        f"{'Eqns':>5s} {'Inst':>5s} {'Leaks':>5s} {'Probe':>5s} "
        f"{'DynScore':>8s} Interesting equation"
    )
    print("-" * 112)

    for world_type in world_types:
        for object_count in object_counts:
            for seed in range(seeds):
                result = _run_equation_campaign_case(
                    context=world_type,
                    seed=seed,
                    object_count=object_count,
                    steps=steps,
                    num_agents=num_agents,
                    world_type=world_type,
                    enable_equation_probes=enable_equation_probes,
                    equation_operator_priors=theory_memory.generated_operator_priors(
                        context=world_type,
                    ),
                )
                results.append(result)
                theory_memory.record_result(
                    result['context'],
                    result['seed'],
                    result.get('discovery_loop', {}),
                )
                theory_memory.record_operator_prior_results(
                    result['context'],
                    result['seed'],
                    result,
                )
                _print_equation_campaign_row(result)

    for index in range(hidden_worlds):
        manifest = generate_hidden_world_manifest(index, variant=index)
        result = _run_equation_campaign_case(
            context=manifest.hidden_id,
            seed=index,
            object_count=object_counts[0],
            steps=steps,
            num_agents=num_agents,
            world_type='hidden_procedural',
            hidden_manifest=manifest,
            enable_equation_probes=enable_equation_probes,
            equation_operator_priors=theory_memory.generated_operator_priors(
                context=manifest.hidden_id,
            ),
        )
        result['manifest'] = manifest.to_dict()
        results.append(result)
        theory_memory.record_result(
            result['context'],
            result['seed'],
            result.get('discovery_loop', {}),
        )
        theory_memory.record_operator_prior_results(
            result['context'],
            result['seed'],
            result,
        )
        _print_equation_campaign_row(result)

    followups = _run_equation_followup_cases(
        theory_memory=theory_memory,
        world_types=world_types,
        object_counts=object_counts,
        steps=steps,
        num_agents=num_agents,
        limit=followup_budget,
        enable_equation_probes=enable_equation_probes,
        progress_fn=_print_equation_followup_progress,
    )
    if followups:
        print("Autonomous follow-up probes:")
        for result in followups:
            results.append(result)
            _print_equation_campaign_row(result)
            outcome = result.get('planned_experiment_outcome', {})
            plan = result.get('planned_experiment', {})
            print(
                f"    outcome: {outcome.get('outcome', 'unknown')} "
                f"for {plan.get('experiment_kind', 'planned_probe')}"
            )

    print("-" * 112)
    total = len(results)
    passed = sum(1 for result in results if result['passed'])
    print(f"Review packs without leaks and with equations: {passed}/{total}")
    _print_equation_category_review(results)
    _print_cumulative_theory_review(theory_memory)
    if results:
        best = max(
            results,
            key=_interesting_result_rank,
        )
        print(
            "Best interesting equation: "
            f"{best['context']} seed={best['seed']} "
            f"{best['interesting_equation'].get('target', '?')} ~= "
            f"{best['interesting_equation'].get('expression', '?')} "
            f"(score={best['interesting_score']:.2f})"
        )
    return results


def _run_equation_followup_cases(
    theory_memory: CumulativeTheoryMemory,
    world_types: list[str],
    object_counts: list[int],
    steps: int,
    num_agents: int,
    limit: int,
    enable_equation_probes: bool = True,
    run_case_fn=None,
    progress_fn=None,
) -> list[dict]:
    if limit <= 0:
        return []
    run_case_fn = run_case_fn or _run_equation_campaign_case
    followups = []
    for iteration in range(limit):
        plans = theory_memory.planned_experiments(
            world_types=world_types,
            object_counts=object_counts,
            steps=steps,
            seed_start=0,
            limit=1,
        )
        if not plans:
            break
        plan = plans[0]
        hidden_manifest = None
        context = plan['world_type']
        if plan['world_type'] == 'hidden_procedural':
            hidden_manifest = generate_hidden_world_manifest(
                plan['seed'],
                variant=plan['seed'],
            )
            context = hidden_manifest.hidden_id
        if progress_fn:
            progress_fn(
                'start',
                {
                    'iteration': iteration + 1,
                    'limit': limit,
                    'plan': dict(plan),
                    'context': context,
                },
            )
        result = run_case_fn(
            context=context,
            seed=plan['seed'],
            object_count=plan['object_count'],
            steps=plan['steps'],
            num_agents=num_agents,
            world_type=plan['world_type'],
            hidden_manifest=hidden_manifest,
            enable_equation_probes=enable_equation_probes,
            planned_actions=_planned_probe_actions(plan),
            equation_operator_priors=theory_memory.generated_operator_priors(
                context=context,
            ),
        )
        result['phase'] = 'autonomous_followup'
        result['followup_iteration'] = iteration + 1
        result['planned_experiment'] = dict(plan)
        outcome = theory_memory.record_planned_result(
            plan,
            context=result['context'],
            seed=result['seed'],
            report=result.get('discovery_loop', {}),
            operator_prior_result=result,
        )
        result['planned_experiment_outcome'] = outcome
        followups.append(result)
        if progress_fn:
            progress_fn(
                'finish',
                {
                    'iteration': iteration + 1,
                    'limit': limit,
                    'plan': dict(plan),
                    'context': result['context'],
                    'result': result,
                    'outcome': outcome,
                },
            )
    return followups


def _print_equation_followup_progress(event: str, payload: dict):
    iteration = int(payload.get('iteration', 0) or 0)
    limit = int(payload.get('limit', 0) or 0)
    plan = dict(payload.get('plan') or {})
    context = payload.get('context') or plan.get('world_type') or 'unknown'
    experiment_kind = plan.get('experiment_kind', 'planned_probe')
    theory_kind = plan.get('theory_kind', 'unknown')
    if event == 'start':
        if iteration == 1:
            print("Autonomous follow-up progress:", flush=True)
        print(
            f"  [{iteration}/{limit}] running {experiment_kind} "
            f"for {theory_kind} in {context}",
            flush=True,
        )
        reason = str(plan.get('reason') or '')
        if reason:
            print(f"      why: {reason}", flush=True)
    elif event == 'finish':
        outcome = dict(payload.get('outcome') or {})
        result = dict(payload.get('result') or {})
        print(
            f"      done: {outcome.get('outcome', 'unknown')} "
            f"eqns={result.get('equation_count', 0)} "
            f"score={float(result.get('interesting_score', 0.0) or 0.0):.2f}",
            flush=True,
        )


def _planned_probe_actions(plan: dict, max_actions: int = 4) -> list[dict]:
    signature = dict(plan.get('disagreement_signature') or {})
    actions = []
    for point in list(signature.get('probe_points') or [])[:max_actions]:
        if not {'x', 'y'} <= set(point):
            continue
        actions.append({
            'type': 'spawn',
            'x': float(point['x']),
            'y': float(point['y']),
            'vx': 0.0,
            'vy': 0.0,
            'source': 'planned_model_disagreement_probe',
            'probe_label': point.get('label'),
            'disagreement_mode': signature.get('mode'),
        })
    if actions:
        return actions
    action = dict(plan.get('probe_action') or {})
    if action.get('type') in {'spawn', 'wait', 'push', 'remove'}:
        action.setdefault('source', 'planned_theory_probe')
        return [action]
    return []


def _interesting_result_rank(result: dict) -> tuple[float, float]:
    equation = result.get('interesting_equation') or {}
    role = equation.get('role', '')
    priorities = {
        'generated_operator_tapered_distance_direction_equation': 6.0,
        'generated_operator_tapered_distance_perpendicular_equation': 6.0,
        'generated_operator_cutoff_direction_equation': 5.9,
        'generated_operator_cutoff_perpendicular_equation': 5.9,
        'generated_operator_distance_scaled_direction_equation': 5.8,
        'generated_operator_distance_scaled_perpendicular_equation': 5.8,
        'generated_operator_periodic_equation': 5.45,
        'local_residual_distance_scaled_direction_equation': 5.6,
        'local_residual_distance_scaled_perpendicular_equation': 5.6,
        'residual_distance_scaled_direction_equation': 5.5,
        'residual_distance_scaled_perpendicular_equation': 5.5,
        'residual_periodic_equation': 5.3,
        'local_residual_direction_equation': 5.2,
        'local_residual_perpendicular_equation': 5.2,
        'residual_direction_equation': 5.0,
        'residual_perpendicular_equation': 5.0,
        'vector_direction_equation': 4.0,
        'vector_perpendicular_equation': 4.0,
        'direction_scaled_equation': 3.5,
        'perpendicular_scaled_equation': 3.5,
        'constant_change_equation': 3.0,
        'position_update_equation': 2.0,
        'derived_magnitude_equation': 1.0,
    }
    return (priorities.get(role, 0.0), float(result.get('interesting_score', 0.0)))


def run_math_foundation_prep(
    seeds: int = 1,
    steps: int = 220,
    object_counts: list[int] = None,
    world_types: list[str] = None,
    hidden_worlds: int = 1,
    num_agents: int = 2,
    theory_memory: CumulativeTheoryMemory | None = None,
) -> list[dict]:
    """
    Check readiness gates before the user-watched final discovery run.

    This is a prep/readiness command, not the final campaign.
    """
    object_counts = object_counts or [5]
    world_types = world_types or ['standard', 'sideways_wind', 'vortex']
    results = []
    theory_memory = theory_memory or CumulativeTheoryMemory()

    print("=" * 70)
    print("MATH FOUNDATION PREP CHECK")
    print("=" * 70)
    print(f"Worlds: {', '.join(world_types)}")
    print(f"Hidden generated worlds: {hidden_worlds}")
    print(f"Seeds: 0..{seeds - 1}")
    print(f"Object counts: {', '.join(str(count) for count in object_counts)}")
    print(f"Steps per run: {steps}")
    print("Final discovery run: held for user-watched session")
    print()
    print(
        f"{'Context':24s} {'Seed':>4s} {'Obj':>3s} "
        f"{'Ready':>7s} {'Missing':36s} {'Probes':>6s}"
    )
    print("-" * 92)

    for world_type in world_types:
        for object_count in object_counts:
            for seed in range(seeds):
                result = _run_math_foundation_prep_case(
                    context=world_type,
                    seed=seed,
                    object_count=object_count,
                    steps=steps,
                    num_agents=num_agents,
                    world_type=world_type,
                    equation_operator_priors=theory_memory.generated_operator_priors(
                        context=world_type,
                    ),
                )
                results.append(result)
                theory_memory.record_result(
                    result['context'],
                    result['seed'],
                    result.get('discovery_loop', {}),
                )
                theory_memory.record_operator_prior_results(
                    result['context'],
                    result['seed'],
                    result,
                )
                _print_math_foundation_prep_row(result)

    for index in range(hidden_worlds):
        manifest = generate_hidden_world_manifest(index, variant=index)
        result = _run_math_foundation_prep_case(
            context=manifest.hidden_id,
            seed=index,
            object_count=object_counts[0],
            steps=steps,
            num_agents=num_agents,
            world_type='hidden_procedural',
            hidden_manifest=manifest,
            equation_operator_priors=theory_memory.generated_operator_priors(
                context=manifest.hidden_id,
            ),
        )
        result['manifest'] = manifest.to_dict()
        results.append(result)
        theory_memory.record_result(
            result['context'],
            result['seed'],
            result.get('discovery_loop', {}),
        )
        theory_memory.record_operator_prior_results(
            result['context'],
            result['seed'],
            result,
        )
        _print_math_foundation_prep_row(result)

    print("-" * 92)
    passed = sum(1 for result in results if result['ready_for_final'])
    print(f"Prep contexts ready: {passed}/{len(results)}")
    _print_cumulative_theory_review(theory_memory)
    print("Final command is ready, but not run:")
    print(
        "  python3 first_principles_ai/main.py --math-final-discovery "
        "--benchmark-steps 600 --object-counts 5 --equation-hidden-worlds 3 "
        "--theory-memory-file tmp/theory-memory.json"
    )
    return results


def run_discovery_readiness_audit(
    theory_memory: CumulativeTheoryMemory | None = None,
) -> dict:
    """Print the non-final discovery-loop readiness audit."""
    theory_memory = theory_memory or CumulativeTheoryMemory()
    report = theory_memory.discovery_readiness_report()

    print("=" * 70)
    print("DISCOVERY READINESS AUDIT")
    print("=" * 70)
    print("Final discovery run: held for user-watched session")
    print(
        f"Score: {report['readiness_score']:.0%} "
        f"status={report['status']} "
        f"gates={report['passed_gate_count']}/{report['gate_count']}"
    )
    print()
    print(f"{'Gate':34s} {'State':>6s} Evidence")
    print("-" * 92)
    for key, gate in report['gates'].items():
        state = 'PASS' if gate['passed'] else 'CHECK'
        evidence = ', '.join(
            f"{name}={value}"
            for name, value in gate.get('evidence', {}).items()
        )
        if len(evidence) > 44:
            evidence = evidence[:41] + '...'
        print(f"{key:34s} {state:>6s} {evidence}")
    print()
    _print_discovery_evidence_dossier(report.get('evidence_dossier', {}))
    if report['recommended_actions']:
        print()
        print("Recommended non-final next actions:")
        for action in report['recommended_actions']:
            label = 'FINAL' if action.get('runs_final') else 'SAFE'
            print(f"  [{label}] {action['action_kind']}: {action['reason']}")
            print(f"    {action['command']}")
    else:
        print()
        print("Recommended non-final next actions: none")
    if report['ready_for_watched_final']:
        print()
        print("Watched final command is ready, but not run by this audit.")
    return report


def run_memory_efficiency_review(
    theory_memory: CumulativeTheoryMemory | None = None,
    compact: bool = False,
    keep_recent_records: int = 96,
    keep_recent_operator_outcomes: int = 192,
) -> dict:
    """Print the bounded-memory and quantized-summary state."""
    theory_memory = theory_memory or CumulativeTheoryMemory()
    if compact:
        report = theory_memory.compact_experience(
            keep_recent_records=keep_recent_records,
            keep_recent_operator_outcomes=keep_recent_operator_outcomes,
            source='cli_compaction',
        )
    else:
        report = theory_memory.resource_efficiency_report(
            recommended_record_window=keep_recent_records,
            recommended_operator_window=keep_recent_operator_outcomes,
        )

    print("=" * 70)
    print("MEMORY EFFICIENCY REVIEW")
    print("=" * 70)
    print(f"Compacted this run: {compact}")
    print(
        "Raw windows: "
        f"records={report['raw_record_count']} "
        f"operator_outcomes={report['raw_operator_prior_outcome_count']}"
    )
    print(
        "Compacted totals: "
        f"records={report['compacted_record_count']} "
        f"operator_outcomes={report['compacted_operator_prior_outcome_count']} "
        f"shards={report['compressed_shard_count']}"
    )
    print(
        "Estimated retained size: "
        f"{report['estimated_retained_bytes']} bytes "
        f"(uncompressed estimate {report['estimated_uncompressed_bytes']} bytes)"
    )
    print(
        "Reduction: "
        f"detail={report['detail_reduction_ratio']:.2f}x "
        f"bytes={report['estimated_compression_ratio']:.2f}x"
    )
    print(f"Long-run ready: {report['long_run_ready']}")
    if report['recommended_actions']:
        print("Recommended actions:")
        for action in report['recommended_actions']:
            print(f"  {action['action_kind']}: {action['reason']}")
    else:
        print("Recommended actions: none")
    return report


def run_domain_curriculum_preview(
    theory_memory: CumulativeTheoryMemory | None = None,
    limit: int = 12,
) -> list[dict]:
    """Print generated domain-world blueprints without running simulations."""
    theory_memory = theory_memory or CumulativeTheoryMemory()
    blueprints = theory_memory.domain_world_blueprints(limit=limit)
    discoveries = {
        item['domain_key']: item
        for item in theory_memory.domain_world_discovery_reports(limit=limit)
    }

    print("=" * 70)
    print("DOMAIN CURRICULUM WORLD PREVIEW")
    print("=" * 70)
    print("Final discovery run: not run")
    print()
    print(
        f"{'Domain':28s} {'Samples':>7s} {'Cand':>5s} "
        f"{'Cov':>5s} {'Fals':>5s} {'Leaks':>5s} Transfer"
    )
    print("-" * 104)
    for item in blueprints:
        discovery = discoveries.get(item['domain_key'], {})
        transfers = ','.join(item.get('transfer_targets', [])[:3]) or 'none'
        print(
            f"{item['domain_key']:28s} "
            f"{item['sample_count']:7d} "
            f"{int(discovery.get('candidate_count', 0) or 0):5d} "
            f"{float(discovery.get('benchmark_coverage', 0.0) or 0.0):5.0%} "
            f"{item['falsifier_count']:5d} "
            f"{item['leaky_observation_count']:5d} "
            f"{transfers}"
        )
    return blueprints


def run_domain_world_discovery_ingest(
    theory_memory: CumulativeTheoryMemory | None = None,
    seed: int = 0,
    variant: int = 0,
) -> list[dict]:
    """Record generated domain-world discoveries into theory memory."""
    theory_memory = theory_memory or CumulativeTheoryMemory()
    records = theory_memory.record_domain_world_discoveries(
        seed=seed,
        variant=variant,
    )

    print("=" * 70)
    print("DOMAIN WORLD DISCOVERY INGEST")
    print("=" * 70)
    print("Final discovery run: not run")
    print()
    print(f"{'Domain':28s} {'Cand':>5s} {'Cov':>5s} {'Fals':>5s} Transfer basis")
    print("-" * 104)
    for record in records:
        basis = ','.join(record.get('transfer_basis', [])[:4]) or 'none'
        print(
            f"{str(record.get('domain_key')):28s} "
            f"{int(record.get('candidate_count', 0) or 0):5d} "
            f"{float(record.get('benchmark_coverage', 0.0) or 0.0):5.0%} "
            f"{int(record.get('falsification_test_count', 0) or 0):5d} "
            f"{basis}"
        )
    return records


def run_autonomous_scientist_loop(
    theory_memory: CumulativeTheoryMemory | None = None,
    seed_start: int = 0,
    seed_count: int = 3,
    variants: list[int] | None = None,
    live: bool = False,
    event_limit: int = 80,
) -> dict:
    """Run and persist the non-final scientist loop over domain worlds."""
    theory_memory = theory_memory or CumulativeTheoryMemory()
    variants = variants or [0]
    report = theory_memory.record_autonomous_scientist_loop(
        seed_start=seed_start,
        seed_count=seed_count,
        variants=variants,
        event_limit=event_limit,
    )
    coverage = dict(report.get('coverage') or {})

    print("=" * 70)
    print("AUTONOMOUS SCIENTIST LOOP")
    print("=" * 70)
    print("Final discovery run: not run")
    print(
        f"Seeds: {seed_start}..{seed_start + max(1, seed_count) - 1} | "
        f"Variants: {', '.join(str(item) for item in variants)}"
    )
    print(
        f"Invariants: {coverage.get('invariant_count', 0)} "
        f"(robust={coverage.get('robust_invariant_count', 0)}) | "
        f"Residual probes: {coverage.get('residual_experiment_count', 0)} | "
        f"Stress worlds: {coverage.get('stress_world_count', 0)} | "
        f"Equation rewrites: {coverage.get('authored_equation_extension_count', 0)}"
    )
    print()
    print(f"{'Invariant':32s} {'Status':>18s} {'Support':>7s} Expression")
    print("-" * 104)
    for item in list(report.get('invariant_consolidations') or [])[:8]:
        expression = str(item.get('law_expression', ''))
        if len(expression) > 42:
            expression = expression[:39] + '...'
        print(
            f"{str(item.get('relation_kind')):32s} "
            f"{str(item.get('status')):>18s} "
            f"{int(item.get('support_count', 0) or 0):7d} "
            f"{expression}"
        )
    print()
    print("Top residual-driven experiments:")
    for item in list(report.get('residual_experiments') or [])[:5]:
        print(
            f"  {item.get('domain_key')}:{item.get('relation_kind')} "
            f"priority={float(item.get('priority', 0.0) or 0.0):.2f}"
        )
        print(f"    next: {item.get('designed_next_experiment')}")
    print()
    print("Harder non-final stress worlds:")
    for item in list(report.get('harder_stress_worlds') or [])[:5]:
        print(
            f"  {item.get('key')}: world={item.get('world_type')} "
            f"priority={float(item.get('priority', 0.0) or 0.0):.2f}"
        )
        print(f"    pressure: {item.get('pressure')}")
    if live:
        for event in list(report.get('live_events') or []):
            print(
                "SCIENTIST_EVENT "
                + json.dumps(event, sort_keys=True),
                flush=True,
            )
    return report


def run_hf_non_final_campaign(
    theory_memory: CumulativeTheoryMemory | None = None,
    seeds: int = 1,
    steps: int = 80,
    object_counts: list[int] = None,
    world_types: list[str] = None,
    hidden_worlds: int = 0,
    num_agents: int = 2,
    output_file: str | None = None,
    domain_seed: int = 0,
    domain_variant: int = 0,
    domain_seed_count: int = 1,
    domain_variants: list[int] | None = None,
    scientist_seed_count: int = 3,
    scientist_variants: list[int] | None = None,
    live_scientist: bool = True,
    include_prep: bool = True,
    auto_compact: bool = True,
    compact_keep_records: int = 96,
    compact_keep_operator_outcomes: int = 192,
    adaptive_compute: bool = True,
    max_adaptive_steps: int | None = None,
    max_adaptive_seeds: int | None = None,
    max_adaptive_hidden_worlds: int | None = None,
    prep_runner=None,
) -> dict:
    """
    HF-friendly non-final run.

    It records domain-world discoveries, optionally runs a small foundation prep
    campaign, writes a JSON artifact, and never runs the watched final command.
    """
    theory_memory = theory_memory or CumulativeTheoryMemory()
    object_counts = object_counts or [3]
    world_types = world_types or ['standard']
    domain_variants = domain_variants or [domain_variant]
    scientist_variants = scientist_variants or domain_variants
    prep_runner = prep_runner or run_math_foundation_prep
    compaction_events = []

    _emit_hf_progress('start', {
        'runs_final': False,
        'world_types': world_types,
        'seeds': seeds,
        'steps': steps,
        'object_counts': object_counts,
        'hidden_worlds': hidden_worlds,
        'domain_seed_count': domain_seed_count,
        'domain_variants': domain_variants,
        'scientist_seed_count': scientist_seed_count,
        'scientist_variants': scientist_variants,
        'auto_compact': auto_compact,
        'adaptive_compute': adaptive_compute,
    })
    domain_records = []
    for variant in domain_variants:
        for seed_offset in range(max(1, int(domain_seed_count or 1))):
            domain_records.extend(theory_memory.record_domain_world_discoveries(
                seed=domain_seed + seed_offset,
                variant=variant,
            ))
    _emit_hf_progress('domain_world_discoveries_recorded', {
        'record_count': len(domain_records),
        'covered_domains': [
            record.get('domain_key')
            for record in domain_records
            if float(record.get('benchmark_coverage', 0.0) or 0.0) >= 1.0
        ],
    })
    compaction_events.append(_hf_compact_theory_memory(
        theory_memory,
        phase='after_domain_worlds',
        enabled=auto_compact,
        keep_recent_records=compact_keep_records,
        keep_recent_operator_outcomes=compact_keep_operator_outcomes,
    ))
    scientist_report = theory_memory.record_autonomous_scientist_loop(
        seed_start=domain_seed,
        seed_count=scientist_seed_count,
        variants=scientist_variants,
        event_limit=80,
    )
    scientist_coverage = dict(scientist_report.get('coverage') or {})
    _emit_hf_progress('autonomous_scientist_finish', {
        'status': scientist_report.get('status'),
        'invariant_count': scientist_coverage.get('invariant_count'),
        'robust_invariant_count': scientist_coverage.get('robust_invariant_count'),
        'residual_experiment_count': scientist_coverage.get('residual_experiment_count'),
        'stress_world_count': scientist_coverage.get('stress_world_count'),
        'authored_equation_extension_count': scientist_coverage.get(
            'authored_equation_extension_count'
        ),
        'live_event_count': scientist_coverage.get('live_event_count'),
    })
    if live_scientist:
        for event in list(scientist_report.get('live_events') or []):
            print(
                "SCIENTIST_EVENT "
                + json.dumps(event, sort_keys=True),
                flush=True,
            )

    compaction_events.append(_hf_compact_theory_memory(
        theory_memory,
        phase='after_autonomous_scientist',
        enabled=auto_compact,
        keep_recent_records=compact_keep_records,
        keep_recent_operator_outcomes=compact_keep_operator_outcomes,
    ))
    readiness_before_prep = theory_memory.discovery_readiness_report()
    resource_before_prep = theory_memory.resource_efficiency_report(
        recommended_record_window=compact_keep_records,
        recommended_operator_window=compact_keep_operator_outcomes,
    )
    compute_budget_plan = plan_adaptive_compute_budget(
        readiness=readiness_before_prep,
        scientist_report=scientist_report,
        resource_report=resource_before_prep,
        requested_steps=steps,
        requested_seeds=seeds,
        requested_hidden_worlds=hidden_worlds,
        max_steps=max_adaptive_steps or max(steps, steps * 2),
        max_seeds=max_adaptive_seeds or max(seeds, seeds + 1),
        max_hidden_worlds=(
            max_adaptive_hidden_worlds
            if max_adaptive_hidden_worlds is not None
            else max(hidden_worlds, hidden_worlds + 1)
        ),
        enabled=adaptive_compute,
    )
    _emit_hf_progress('compute_budget_plan', compute_budget_plan)

    prep_results = []
    if include_prep:
        effective = dict(compute_budget_plan.get('effective') or {})
        effective_steps = int(effective.get('steps', steps) or steps)
        effective_seeds = int(effective.get('seeds', seeds) or seeds)
        effective_hidden_worlds = int(
            effective.get('hidden_worlds', hidden_worlds) or 0
        )
        _emit_hf_progress('foundation_prep_start', {
            'world_types': world_types,
            'requested_seeds': seeds,
            'requested_steps': steps,
            'requested_hidden_worlds': hidden_worlds,
            'seeds': effective_seeds,
            'steps': effective_steps,
            'hidden_worlds': effective_hidden_worlds,
            'compute_expanded': compute_budget_plan['expanded'],
        })
        prep_results = prep_runner(
            seeds=effective_seeds,
            steps=effective_steps,
            object_counts=object_counts,
            world_types=world_types,
            hidden_worlds=effective_hidden_worlds,
            num_agents=num_agents,
            theory_memory=theory_memory,
        )
        _emit_hf_progress('foundation_prep_finish', {
            'result_count': len(prep_results),
            'ready_count': sum(
                1 for result in prep_results
                if result.get('ready_for_final')
            ),
        })
        compaction_events.append(_hf_compact_theory_memory(
            theory_memory,
            phase='after_foundation_prep',
            enabled=auto_compact,
            keep_recent_records=compact_keep_records,
            keep_recent_operator_outcomes=compact_keep_operator_outcomes,
        ))

    readiness = theory_memory.discovery_readiness_report()
    resource_efficiency = theory_memory.resource_efficiency_report(
        recommended_record_window=compact_keep_records,
        recommended_operator_window=compact_keep_operator_outcomes,
    )
    result = {
        'run_kind': 'hf_non_final_campaign',
        'runs_final': False,
        'domain_world_record_count': len(domain_records),
        'prep_result_count': len(prep_results),
        'readiness': readiness,
        'readiness_before_prep': readiness_before_prep,
        'compute_budget_plan': compute_budget_plan,
        'compaction_events': compaction_events,
        'resource_efficiency': resource_efficiency,
        'prep_results': prep_results,
        'autonomous_scientist_report': scientist_report,
        'theory_memory': theory_memory.to_dict(),
    }
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', encoding='utf-8') as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
        _emit_hf_progress('artifact_written', {
            'output_file': str(output_path),
        })
    _emit_hf_progress('finish', {
        'readiness_score': readiness['readiness_score'],
        'status': readiness['status'],
        'passed_gate_count': readiness['passed_gate_count'],
        'gate_count': readiness['gate_count'],
        'compute_expanded': compute_budget_plan['expanded'],
        'compressed_shards': resource_efficiency['compressed_shard_count'],
        'long_run_ready': resource_efficiency['long_run_ready'],
    })
    return result


def _emit_hf_progress(event: str, payload: dict):
    print(
        "HF_PROGRESS "
        + json.dumps({'event': event, **payload}, sort_keys=True),
        flush=True,
    )


def _hf_compact_theory_memory(
    theory_memory: CumulativeTheoryMemory,
    *,
    phase: str,
    enabled: bool,
    keep_recent_records: int,
    keep_recent_operator_outcomes: int,
) -> dict:
    before = theory_memory.resource_efficiency_report(
        recommended_record_window=keep_recent_records,
        recommended_operator_window=keep_recent_operator_outcomes,
    )
    if enabled:
        after = theory_memory.compact_experience(
            keep_recent_records=keep_recent_records,
            keep_recent_operator_outcomes=keep_recent_operator_outcomes,
            source=f'hf_batch:{phase}',
        )
    else:
        after = before
    event = {
        'phase': phase,
        'enabled': enabled,
        'raw_records_before': before['raw_record_count'],
        'raw_operator_outcomes_before': before['raw_operator_prior_outcome_count'],
        'raw_records_after': after['raw_record_count'],
        'raw_operator_outcomes_after': after['raw_operator_prior_outcome_count'],
        'compressed_shards': after['compressed_shard_count'],
        'detail_reduction_ratio': after['detail_reduction_ratio'],
        'long_run_ready': after['long_run_ready'],
    }
    _emit_hf_progress('memory_compaction_checkpoint', event)
    return event


def run_math_final_discovery(
    seeds: int = 1,
    steps: int = 600,
    object_counts: list[int] = None,
    world_types: list[str] = None,
    hidden_worlds: int = 3,
    num_agents: int = 2,
    theory_memory: CumulativeTheoryMemory | None = None,
) -> list[dict]:
    """Run the watched discovery campaign and report live performance metrics."""
    object_counts = object_counts or [5]
    world_types = world_types or WORLD_TYPES
    results = []
    theory_memory = theory_memory or CumulativeTheoryMemory()

    print("=" * 70, flush=True)
    print("FINAL WATCHED MATH DISCOVERY CAMPAIGN", flush=True)
    print("=" * 70, flush=True)
    print(f"Worlds: {', '.join(world_types)}", flush=True)
    print(f"Hidden generated worlds: {hidden_worlds}", flush=True)
    print(f"Seeds: 0..{seeds - 1}", flush=True)
    print(f"Object counts: {', '.join(str(count) for count in object_counts)}", flush=True)
    print(f"Steps per run: {steps}", flush=True)
    print(flush=True)
    print(
        f"{'Context':24s} {'Seed':>4s} {'Obj':>3s} "
        f"{'Ready':>7s} {'FPrb':>4s} {'Eqns':>5s} {'Inst':>5s} "
        f"{'Leaks':>5s} {'EPrb':>4s} {'EqScore':>7s} {'Result':>6s} "
        f"Interesting equation",
        flush=True,
    )
    print("-" * 132, flush=True)

    for world_type in world_types:
        for object_count in object_counts:
            for seed in range(seeds):
                print(
                    f"Running final case: {world_type} seed={seed} "
                    f"objects={object_count} steps={steps}",
                    flush=True,
                )
                result = _run_math_final_discovery_case(
                    context=world_type,
                    seed=seed,
                    object_count=object_count,
                    steps=steps,
                    num_agents=num_agents,
                    world_type=world_type,
                    equation_operator_priors=theory_memory.generated_operator_priors(
                        context=world_type,
                    ),
                )
                results.append(result)
                theory_memory.record_result(
                    result['context'],
                    result['seed'],
                    result.get('discovery_loop', {}),
                )
                theory_memory.record_operator_prior_results(
                    result['context'],
                    result['seed'],
                    result,
                )
                _print_math_final_discovery_row(result)

    for index in range(hidden_worlds):
        manifest = generate_hidden_world_manifest(index, variant=index)
        print(
            f"Running final case: {manifest.hidden_id} seed={index} "
            f"objects={object_counts[0]} steps={steps}",
            flush=True,
        )
        result = _run_math_final_discovery_case(
            context=manifest.hidden_id,
            seed=index,
            object_count=object_counts[0],
            steps=steps,
            num_agents=num_agents,
            world_type='hidden_procedural',
            hidden_manifest=manifest,
            equation_operator_priors=theory_memory.generated_operator_priors(
                context=manifest.hidden_id,
            ),
        )
        result['manifest'] = manifest.to_dict()
        results.append(result)
        theory_memory.record_result(
            result['context'],
            result['seed'],
            result.get('discovery_loop', {}),
        )
        theory_memory.record_operator_prior_results(
            result['context'],
            result['seed'],
            result,
        )
        _print_math_final_discovery_row(result)

    print("-" * 132, flush=True)
    total = len(results)
    ready = sum(1 for result in results if result['ready_for_final'])
    clean_equations = sum(1 for result in results if result['equation_passed'])
    passed = sum(1 for result in results if result['passed'])
    print(f"Foundation ready: {ready}/{total}", flush=True)
    print(f"Equation packs clean: {clean_equations}/{total}", flush=True)
    print(f"Final contexts passed: {passed}/{total}", flush=True)
    _print_equation_category_review(results)
    _print_cumulative_theory_review(theory_memory)
    if results:
        best = max(
            results,
            key=_interesting_result_rank,
        )
        print(
            "Best interesting equation: "
            f"{best['context']} seed={best['seed']} "
            f"{best['interesting_equation'].get('target', '?')} ~= "
            f"{best['interesting_equation'].get('expression', '?')} "
            f"(score={best['interesting_score']:.2f})",
            flush=True,
        )
    return results


def run_math_benchmark(
    seeds: int = 3,
    steps: int = 160,
    object_counts: list[int] = None,
    world_types: list[str] = None,
    num_agents: int = 2,
    required_concepts: set[str] | None = None,
) -> list[dict]:
    """Measure emergent math comparison coverage across repeated runs."""
    object_counts = object_counts or [5]
    world_types = world_types or ['standard']
    required_concepts = set(required_concepts or MATH_BENCHMARK_TARGETS)
    results = []

    print("=" * 70)
    print("EMERGENT MATH BENCHMARK")
    print("=" * 70)
    print(f"Worlds: {', '.join(world_types)}")
    print(f"Seeds: 0..{seeds - 1}")
    print(f"Object counts: {', '.join(str(count) for count in object_counts)}")
    print(f"Steps per run: {steps}")
    print(f"Required comparison concepts: {len(required_concepts)}")
    print()
    print(
        f"{'World':24s} {'Seed':>4s} {'Obj':>3s} "
        f"{'Patterns':>8s} {'Concepts':>8s} {'Coverage':>8s} {'Leaks':>5s} Result"
    )
    print("-" * 90)

    for world_type in world_types:
        for object_count in object_counts:
            for seed in range(seeds):
                with contextlib.redirect_stdout(io.StringIO()):
                    _, kb, _ = run_experiment(
                        num_steps=steps,
                        num_initial_objects=object_count,
                        seed=seed,
                        verbose=False,
                        report_interval=max(steps, 1),
                        world_type=world_type,
                        num_agents=num_agents,
                    )

                metrics = _math_metrics_from_knowledge(kb, required_concepts)
                result = {
                    'world_type': world_type,
                    'seed': seed,
                    'objects': object_count,
                    **metrics,
                }
                results.append(result)
                print(
                    f"{world_type:24s} {seed:4d} {object_count:3d} "
                    f"{metrics['pattern_count']:8d} {len(metrics['human_concepts']):8d} "
                    f"{metrics['coverage']:8.1%} {len(metrics['label_leaks']):5d} "
                    f"{'PASS' if metrics['passed'] else 'CHECK'}"
                )

    passes = sum(1 for result in results if result['passed'])
    total = len(results)
    combined = sorted(set().union(*(result['human_concepts'] for result in results))) if results else []
    print("-" * 90)
    print(f"Passed: {passes}/{total} ({passes / max(total, 1):.1%})")
    print(f"Combined human-math comparison concepts: {len(combined)}")
    for concept in combined:
        print(f"  - {concept}")
    return results


def _math_metrics_from_knowledge(kb: KnowledgeBase, required_concepts: set[str]) -> dict:
    math_discovery = getattr(kb, 'emergent_math_discovery', None)
    if math_discovery is None:
        return {
            'pattern_count': 0,
            'human_concepts': set(),
            'missing_concepts': set(required_concepts),
            'coverage': 0.0,
            'label_leaks': [],
            'passed': False,
            'comparisons': [],
        }

    comparisons = [comparison.to_dict() for comparison in math_discovery.compare_to_human_math()]
    human_concepts = {comparison['human_concept'] for comparison in comparisons}
    missing = set(required_concepts) - human_concepts
    label_leaks = _emergent_math_label_leaks(kb)
    coverage = len(required_concepts & human_concepts) / max(len(required_concepts), 1)
    return {
        'pattern_count': len(math_discovery.discovered_patterns()),
        'human_concepts': human_concepts,
        'missing_concepts': missing,
        'coverage': coverage,
        'label_leaks': label_leaks,
        'passed': not missing and not label_leaks,
        'comparisons': comparisons,
    }


def _run_hidden_experiment_metrics(
    manifest: HiddenWorldManifest,
    seed: int,
    object_count: int,
    steps: int,
    num_agents: int,
    law_memory: LawMemory | None,
    allow_memory_probes: bool = True,
) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        _, kb, _ = run_experiment(
            num_steps=steps,
            num_initial_objects=object_count,
            seed=seed,
            verbose=False,
            report_interval=max(steps, 1),
            world_type='hidden_procedural',
            num_agents=num_agents,
            law_memory=law_memory,
            hidden_manifest=manifest,
            allow_memory_probes=allow_memory_probes,
        )

    observation_probe = Environment(
        num_initial_objects=1,
        seed=seed,
        world_type='hidden_procedural',
        hidden_manifest=manifest,
    ).observe()
    discovered = _blind_hidden_discoveries(kb)
    expected = set(manifest.expected_discoveries)
    matched = discovered & expected
    false_discoveries = discovered - HIDDEN_DISCOVERY_TARGETS
    score = len(matched) / max(len(expected), 1)
    learned_rules = [
        rule for rule in kb.get_confirmed_rules()
        if rule.properties.get('hypothesis_type') == 'learned_dynamics'
    ]
    return {
        'expected_discoveries': expected,
        'discovered': discovered,
        'matched': matched,
        'missing': expected - discovered,
        'false_discoveries': false_discoveries,
        'score': score,
        'passed': score >= 0.5 and not hidden_manifest_from_observation(observation_probe),
        'observation_leak': hidden_manifest_from_observation(observation_probe),
        'learned_rule_count': len(learned_rules),
        'learned_law_types': sorted({
            rule.properties.get('law_type', 'unknown')
            for rule in learned_rules
        }),
        'detected_novel_types': _confirmed_novel_types(kb),
        'experiment_proposals': _experiment_proposals_from_knowledge(kb),
        'label_leaks': _emergent_math_label_leaks(kb),
        'memory_transfer': (
            law_memory.episodes[-1].transfer_report
            if law_memory is not None and law_memory.episodes
            else {'observed': [], 'matched_priors': [], 'missing_priors': []}
        ),
    }


def _run_equation_campaign_case(
    context: str,
    seed: int,
    object_count: int,
    steps: int,
    num_agents: int,
    world_type: str,
    hidden_manifest: HiddenWorldManifest | None = None,
    enable_equation_probes: bool = True,
    planned_actions: list[dict] | None = None,
    equation_operator_priors: list[dict] | None = None,
) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        _, kb, _ = run_experiment(
            num_steps=steps,
            num_initial_objects=object_count,
            seed=seed,
            verbose=False,
            report_interval=max(steps, 1),
            world_type=world_type,
            num_agents=num_agents,
            hidden_manifest=hidden_manifest,
            allow_memory_probes=False,
            enable_equation_probes=enable_equation_probes,
            planned_actions=planned_actions,
            equation_operator_priors=equation_operator_priors,
        )
    metrics = _equation_metrics_from_knowledge(kb)
    return {
        'context': context,
        'seed': seed,
        'objects': object_count,
        'steps': steps,
        **metrics,
    }


def _equation_metrics_from_knowledge(kb: KnowledgeBase) -> dict:
    workbench = getattr(kb, 'equation_workbench', None)
    if workbench is None:
        return {
            'equation_count': 0,
            'installed_count': 0,
            'generated_operator_count': 0,
            'generated_operator_prior_count': 0,
            'operator_prior_results': [],
            'top_equation': {},
            'top_score': 0.0,
            'interesting_equation': {},
            'interesting_score': 0.0,
            'categories': {},
            'interesting_misses': [],
            'probe_suggestions': [],
            'discovery_loop': {},
            'label_leaks': [],
            'passed': False,
        }
    pack = workbench.review_pack()
    top_equation = pack['top_equations'][0] if pack['top_equations'] else {}
    interesting_equation = (
        pack['interesting_equations'][0]
        if pack['interesting_equations']
        else top_equation
    )
    top_score = float(top_equation.get('score', 0.0)) if top_equation else 0.0
    interesting_score = (
        float(interesting_equation.get('score', 0.0))
        if interesting_equation
        else 0.0
    )
    return {
        'equation_count': pack['equation_count'],
        'installed_count': pack['installed_count'],
        'top_equation': top_equation,
        'top_score': top_score,
        'interesting_equation': interesting_equation,
        'interesting_score': interesting_score,
        'categories': pack['categories'],
        'interesting_misses': pack['interesting_misses'],
        'probe_suggestions': pack['probe_suggestions'],
        'discovery_loop': pack['discovery_loop'],
        'generated_operator_count': pack['generated_operator_count'],
        'generated_operator_prior_count': pack['generated_operator_prior_count'],
        'operator_prior_results': pack['operator_prior_results'],
        'label_leaks': pack['label_leaks'],
        'passed': pack['equation_count'] > 0 and not pack['label_leaks'],
    }


def _run_math_foundation_prep_case(
    context: str,
    seed: int,
    object_count: int,
    steps: int,
    num_agents: int,
    world_type: str,
    hidden_manifest: HiddenWorldManifest | None = None,
    equation_operator_priors: list[dict] | None = None,
) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        _, kb, _ = run_experiment(
            num_steps=steps,
            num_initial_objects=object_count,
            seed=seed,
            verbose=False,
            report_interval=max(steps, 1),
            world_type=world_type,
            num_agents=num_agents,
            hidden_manifest=hidden_manifest,
            allow_memory_probes=False,
            enable_equation_probes=True,
            equation_operator_priors=equation_operator_priors,
            equation_max_operator_feedback_rows=96,
            equation_max_operator_feedback_operators=2,
        )
    metrics = _foundation_metrics_from_knowledge(kb)
    equations = _equation_metrics_from_knowledge(kb)
    return {
        'context': context,
        'seed': seed,
        'objects': object_count,
        'steps': steps,
        **metrics,
        'equation_count': equations['equation_count'],
        'installed_count': equations['installed_count'],
        'interesting_equation': equations['interesting_equation'],
        'interesting_score': equations['interesting_score'],
        'discovery_loop': equations['discovery_loop'],
        'generated_operator_prior_count': equations['generated_operator_prior_count'],
        'equation_passed': equations['passed'],
    }


def _run_math_final_discovery_case(
    context: str,
    seed: int,
    object_count: int,
    steps: int,
    num_agents: int,
    world_type: str,
    hidden_manifest: HiddenWorldManifest | None = None,
    equation_operator_priors: list[dict] | None = None,
) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        _, kb, _ = run_experiment(
            num_steps=steps,
            num_initial_objects=object_count,
            seed=seed,
            verbose=False,
            report_interval=max(steps, 1),
            world_type=world_type,
            num_agents=num_agents,
            hidden_manifest=hidden_manifest,
            allow_memory_probes=False,
            enable_equation_probes=True,
            equation_operator_priors=equation_operator_priors,
            equation_max_operator_feedback_rows=96,
            equation_max_operator_feedback_operators=2,
        )
    foundation = _foundation_metrics_from_knowledge(kb)
    equations = _equation_metrics_from_knowledge(kb)
    return {
        'context': context,
        'seed': seed,
        'objects': object_count,
        'steps': steps,
        **foundation,
        **equations,
        'equation_passed': equations['passed'],
        'passed': foundation['ready_for_final'] and equations['passed'],
    }


def _foundation_metrics_from_knowledge(kb: KnowledgeBase) -> dict:
    report = getattr(kb, 'math_foundation_report', None)
    if report is None:
        return {
            'readiness_score': 0.0,
            'missing_gates': [
                'number_system_stability',
                'equation_templates',
                'composition_inverse_planning',
                'check_traces',
                'geometry_basis',
                'self_directed_math_probes',
            ],
            'gates': {},
            'artifact_count': 0,
            'proof_trace_count': 0,
            'probe_count': 0,
            'ready_for_final': False,
            'probes': [],
        }
    return {
        'readiness_score': report.readiness_score,
        'missing_gates': report.missing_gates,
        'gates': dict(report.gates),
        'artifact_count': len(report.artifacts),
        'proof_trace_count': len(report.proof_traces),
        'probe_count': len(report.probes),
        'ready_for_final': report.readiness_score >= 0.84,
        'probes': [probe.to_dict() for probe in report.probes],
    }


def _print_math_foundation_prep_row(result: dict):
    missing = ','.join(result['missing_gates']) or 'none'
    if len(missing) > 34:
        missing = missing[:31] + '...'
    print(
        f"{result['context']:24s} {result['seed']:4d} {result['objects']:3d} "
        f"{result['readiness_score']:7.1%} {missing:36s} {result['probe_count']:6d}"
    )


def _print_equation_campaign_row(result: dict):
    interesting = result.get('interesting_equation') or {}
    equation_text = (
        f"{interesting.get('target', '?')} ~= {interesting.get('expression', '?')}"
        if interesting
        else 'none'
    )
    print(
        f"{result['context']:24s} {result['seed']:4d} {result['objects']:3d} "
        f"{result['equation_count']:5d} {result['installed_count']:5d} "
        f"{len(result['label_leaks']):5d} {len(result['probe_suggestions']):5d} "
        f"{result['interesting_score']:8.2f} {equation_text}"
    )


def _print_math_final_discovery_row(result: dict):
    interesting = result.get('interesting_equation') or {}
    equation_text = (
        f"{interesting.get('target', '?')} ~= {interesting.get('expression', '?')}"
        if interesting
        else 'none'
    )
    status = 'PASS' if result['passed'] else 'CHECK'
    print(
        f"{result['context']:24s} {result['seed']:4d} {result['objects']:3d} "
        f"{result['readiness_score']:7.1%} {result['probe_count']:4d} "
        f"{result['equation_count']:5d} {result['installed_count']:5d} "
        f"{len(result['label_leaks']):5d} {len(result['probe_suggestions']):4d} "
        f"{result['interesting_score']:7.2f} {status:>6s} {equation_text}",
        flush=True,
    )


def _print_equation_category_review(results: list[dict]):
    if not results:
        return
    print()
    print("Category leaders:")
    for result in results:
        categories = result.get('categories', {})
        leaders = []
        for category in (
            'state_transitions',
            'motion_updates',
            'direction_vectors',
            'residual_dynamics',
            'residual_strength',
            'residual_periodic',
        ):
            items = categories.get(category, [])
            if not items:
                continue
            top = items[0]
            if float(top.get('score', 0.0)) < 0.05:
                continue
            leaders.append(
                f"{category}={top['target']}~{top['expression']}({top['score']:.2f})"
            )
        if leaders:
            print(f"  {result['context']}: " + "; ".join(leaders))


def _dossier_has_entries(dossier: dict) -> bool:
    return any(
        dossier.get(key)
        for key in (
            'chains',
            'claims',
            'planned_tests',
            'next_experiments',
            'proof_certificates',
            'disagreement_probes',
            'self_authored_equations',
            'domain_world_blueprints',
            'domain_world_discoveries',
            'domain_world_transfer_evidence',
            'domain_transfer_probes',
        )
    )


def _print_discovery_evidence_dossier(
    dossier: dict,
    title: str = "Evidence dossier:",
):
    print(title)
    if not _dossier_has_entries(dossier):
        print("  none yet")
        return
    for chain in dossier.get('chains', [])[:3]:
        print(
            f"  chain {chain['operator_kind']}: status={chain['status']} "
            f"steps={chain['step_count']} best={chain['best_score']:.2f}"
        )
        print(f"    next: {chain['next_obligation']}")
    for claim in dossier.get('claims', [])[:3]:
        print(
            f"  claim {claim['operator_kind']}: status={claim['status']} "
            f"best={claim['best_score']:.2f}"
        )
        print(f"    next: {claim['next_obligation']}")
    for plan in dossier.get('planned_tests', [])[:3]:
        print(
            f"  planned {plan['experiment_kind']}: {plan['world_type']} "
            f"seed={plan['seed']} priority={plan['priority']:.2f}"
        )
        print(f"    why: {plan['reason']}")
    for experiment in dossier.get('next_experiments', [])[:3]:
        print(
            f"  next {experiment['experiment_kind']}: "
            f"{experiment['theory_kind']} priority={experiment['priority']:.2f}"
        )
    for probe in dossier.get('disagreement_probes', [])[:2]:
        print(
            f"  disagreement {probe['theory_kind']}: "
            f"mode={probe.get('mode')} priority={probe['priority']:.2f}"
        )
    for equation in dossier.get('self_authored_equations', [])[:2]:
        print(
            f"  authored {equation['equation_kind']}: "
            f"status={equation['status']} confidence={equation['confidence']:.2f}"
        )
        print(f"    expression: {equation['expression']}")
    for blueprint in dossier.get('domain_world_blueprints', [])[:2]:
        print(
            f"  domain world {blueprint['domain_key']}: "
            f"samples={blueprint['sample_count']} "
            f"falsifiers={blueprint['falsifier_count']} "
            f"leaks={int(blueprint['leaks_benchmark_truth'])}"
        )
        print(f"    next: {blueprint['next_pressure']}")
    for discovery in dossier.get('domain_world_discoveries', [])[:2]:
        print(
            f"  domain discovery {discovery['domain_key']}: "
            f"candidates={discovery['candidate_count']} "
            f"coverage={discovery['benchmark_coverage']:.0%} "
            f"falsifiers={discovery['falsification_test_count']}"
        )
        if discovery.get('top_expression'):
            print(f"    expression: {discovery['top_expression']}")
    for item in dossier.get('domain_world_transfer_evidence', [])[:2]:
        print(
            f"  domain transfer evidence "
            f"{item['source_domain']}->{item['target_domain']}: "
            f"status={item['status']}"
        )
        print(
            "    basis: "
            + ','.join(item.get('source_matches') or ['none'])
            + " -> "
            + ','.join(item.get('target_matches') or ['none'])
        )
    for transfer in dossier.get('domain_transfer_probes', [])[:2]:
        print(
            f"  domain transfer {transfer['source_domain']}->{transfer['target_domain']}: "
            f"priority={transfer['priority']:.2f}"
        )
        print(f"    question: {transfer['transfer_question']}")


def _print_cumulative_theory_review(theory_memory: CumulativeTheoryMemory):
    families = theory_memory.reusable_families()
    certificates = theory_memory.proof_certificates(limit=3)
    disagreements = theory_memory.disagreement_experiments(limit=3)
    gaps = theory_memory.proof_gaps()
    generalization_gaps = theory_memory.generalization_gaps()
    domain_revisions = theory_memory.domain_revisions()
    discovery_readiness = theory_memory.discovery_readiness_report()
    adaptive_dimensions = theory_memory.adaptive_dimension_agenda(limit=3)
    algebraic_foundation = theory_memory.algebraic_foundation_baseline()
    algebraic_agenda = theory_memory.algebraic_expression_agenda(limit=3)
    self_authored_equations = theory_memory.self_authored_equations(limit=3)
    math_domain_curriculum = theory_memory.math_domain_curriculum()
    domain_curriculum_agenda = theory_memory.domain_curriculum_agenda(limit=3)
    domain_world_blueprints = theory_memory.domain_world_blueprints(limit=3)
    domain_world_discoveries = theory_memory.domain_world_discovery_reports(limit=3)
    domain_world_transfer_evidence = theory_memory.domain_world_transfer_evidence(limit=3)
    domain_transfer_experiments = theory_memory.domain_transfer_experiments(limit=3)
    representation_agenda = theory_memory.representation_agenda(limit=3)
    generated_operator_priors = theory_memory.generated_operator_priors(limit=3)
    operator_prior_feedback = theory_memory.operator_prior_feedback(limit=3)
    operator_prior_domains = theory_memory.operator_prior_domains(limit=3)
    operator_prior_anomalies = theory_memory.operator_prior_anomalies(limit=3)
    operator_prior_claims = theory_memory.operator_prior_discovery_claims(limit=3)
    operator_prior_chains = theory_memory.operator_prior_discovery_chains(limit=3)
    operator_prior_claim_experiments = theory_memory.operator_prior_claim_experiments(limit=3)
    operator_prior_repairs = theory_memory.operator_prior_repair_experiments(limit=3)
    next_experiments = theory_memory.next_experiments(limit=3)
    planned = theory_memory.planned_experiments(
        world_types=[
            'standard',
            'sideways_wind',
            'vortex',
            'inverse_square_repulsion',
            'localized_gravity',
            'time_varying',
        ],
        object_counts=[5],
        steps=240,
        limit=3,
    )
    if not any([
        families,
        certificates,
        disagreements,
        gaps,
        generalization_gaps,
        domain_revisions,
        discovery_readiness,
        adaptive_dimensions,
        algebraic_foundation,
        algebraic_agenda,
        self_authored_equations,
        math_domain_curriculum,
        domain_curriculum_agenda,
        domain_world_blueprints,
        domain_world_discoveries,
        domain_world_transfer_evidence,
        domain_transfer_experiments,
        representation_agenda,
        generated_operator_priors,
        operator_prior_feedback,
        operator_prior_domains,
        operator_prior_anomalies,
        operator_prior_claims,
        operator_prior_chains,
        operator_prior_claim_experiments,
        operator_prior_repairs,
        next_experiments,
        planned,
    ]):
        return
    print()
    if families:
        print("Reusable theory families:")
        for family in families[:6]:
            contexts = ','.join(family['contexts'])
            if len(contexts) > 54:
                contexts = contexts[:51] + '...'
            operators = ','.join(family['operator_kinds'][:3]) or 'none'
            print(
                f"  {family['theory_kind']}: support={family['support_count']}, "
                f"gen={family['generalization_score']:.2f}, "
                f"proof={family['proof_rate']:.2f}, "
                f"status={family['generalization_status']}, "
                f"ops={operators}, contexts={contexts}"
            )
            print(f"    next: {family['next_proof_obligation']}")
    if certificates:
        print("Theory proof certificates:")
        for certificate in certificates:
            support = certificate['support']
            print(
                f"  {certificate['theory_kind']}: status={certificate['status']}, "
                f"support={support['support_count']}, proof={support['proof_rate']:.2f}"
            )
            print(f"    accepts: {certificate['accepted_because'][0]}")
            if certificate['not_universal_because']:
                print(f"    limit: {certificate['not_universal_because'][0]}")
            print(f"    breaks if: {certificate['would_break_if']}")
    if disagreements:
        print("Theory disagreement probes:")
        for experiment in disagreements:
            primary = experiment.get('primary_theory_label') or experiment['theory_kind']
            rivals = (
                ','.join(experiment['rival_theory_kinds'])
                or ','.join(experiment.get('rival_theory_labels', []))
                or 'none'
            )
            signature = experiment['disagreement_signature']
            print(
                f"  {primary} vs {rivals}: "
                f"mode={signature.get('mode')}, priority={experiment['priority']:.2f}"
            )
            print(f"    question: {signature.get('question')}")
    if gaps:
        print("Theory proof gaps:")
        for gap in gaps[:3]:
            print(
                f"  {gap['theory_kind']}: status={gap['status']}, "
                f"proof={gap['proof_rate']:.2f}, support={gap['support_count']}"
            )
    if generalization_gaps:
        print("Theory generalization gaps:")
        for gap in generalization_gaps[:3]:
            evidence = gap['proof_evidence']
            print(
                f"  {gap['theory_kind']}: status={gap['status']}, "
                f"contexts={evidence['context_count']}, "
                f"support={evidence['support_count']}"
            )
    if domain_revisions:
        print("Theory domain revisions:")
        for revision in domain_revisions[:3]:
            domain = revision['domain_hypothesis']
            included = ','.join(domain['included_contexts']) or 'none'
            excluded = ','.join(domain['excluded_contexts']) or 'none'
            print(
                f"  {revision['theory_kind']}: include={included}; "
                f"exclude={excluded}"
            )
            print(f"    claim: {domain['claim']}")
    print("Theory discovery readiness:")
    print(
        f"  score={discovery_readiness['readiness_score']:.0%} "
        f"status={discovery_readiness['status']} "
        f"gates={discovery_readiness['passed_gate_count']}/"
        f"{discovery_readiness['gate_count']}"
    )
    if discovery_readiness['missing_gates']:
        print(
            "    missing: "
            + ', '.join(discovery_readiness['missing_gates'][:4])
        )
    dossier = discovery_readiness.get('evidence_dossier', {})
    if _dossier_has_entries(dossier):
        _print_discovery_evidence_dossier(
            dossier,
            title="Theory discovery evidence dossier:",
        )
    if adaptive_dimensions:
        print("Theory adaptive dimensions:")
        for dimension in adaptive_dimensions:
            primitives = ','.join(dimension.get('first_principles', [])[:3])
            print(
                f"  {dimension['name']}: kind={dimension['dimension_kind']} "
                f"priority={dimension['priority']:.2f}"
            )
            print(f"    primitives: {primitives}")
            print(f"    expression: {dimension['expression']}")
    if algebraic_foundation:
        print("Theory algebraic foundation:")
        print(
            f"  families={algebraic_foundation['expression_family_count']} "
            f"structures={algebraic_foundation['structure_count']} "
            f"proof_obligations={algebraic_foundation['proof_obligation_count']}"
        )
    if algebraic_agenda:
        print("Theory algebraic expression agenda:")
        for item in algebraic_agenda:
            families = ','.join(item.get('expression_families', [])[:3])
            obligations = ','.join(item.get('proof_obligations', [])[:3])
            print(
                f"  {item['signal']}: families={families} "
                f"priority={item['priority']:.2f}"
            )
            print(f"    obligations: {obligations}")
    if self_authored_equations:
        print("Theory self-authored equations:")
        for equation in self_authored_equations:
            print(
                f"  {equation['equation_kind']}: status={equation['status']} "
                f"support={equation['support_count']} "
                f"confidence={equation['confidence']:.2f}"
            )
            print(f"    expression: {equation['expression']}")
            if equation.get('approximation_notes'):
                print(f"    note: {equation['approximation_notes'][0]}")
    if math_domain_curriculum:
        coverage = math_domain_curriculum['coverage']
        print("Theory math domain curriculum:")
        print(
            f"  domains={math_domain_curriculum['domain_count']} "
            f"bridges={math_domain_curriculum['transfer_bridge_count']} "
            f"active={coverage['active_domain_count']}"
        )
    if domain_curriculum_agenda:
        print("Theory domain curriculum agenda:")
        for item in domain_curriculum_agenda:
            print(
                f"  {item['domain_key']}: status={item['status']} "
                f"priority={item['priority']:.2f} support={item['support_count']}"
            )
            print(f"    next: {item['next_pressure']}")
    if domain_world_blueprints:
        print("Theory domain world blueprints:")
        for item in domain_world_blueprints:
            print(
                f"  {item['domain_key']}: samples={item['sample_count']} "
                f"falsifiers={item['falsifier_count']} "
                f"leaks={item['leaky_observation_count']}"
            )
            targets = ','.join(item.get('transfer_targets', [])[:3]) or 'none'
            print(f"    transfer: {targets}")
    if domain_world_discoveries:
        print("Theory domain world discoveries:")
        for item in domain_world_discoveries:
            equations = list(item.get('self_authored_equations') or [])
            expression = equations[0].get('expression') if equations else 'none'
            print(
                f"  {item['domain_key']}: candidates={item['candidate_count']} "
                f"coverage={item['benchmark_coverage']:.0%} "
                f"falsifiers={item['falsification_test_count']}"
            )
            print(f"    expression: {expression}")
    if domain_world_transfer_evidence:
        print("Theory domain world transfer evidence:")
        for item in domain_world_transfer_evidence:
            print(
                f"  {item['source_domain']} -> {item['target_domain']}: "
                f"status={item['status']}"
            )
            source = ','.join(item.get('source_matches', [])[:3]) or 'none'
            target = ','.join(item.get('target_matches', [])[:3]) or 'none'
            print(f"    basis: {source} -> {target}")
    if domain_transfer_experiments:
        print("Theory domain transfer probes:")
        for item in domain_transfer_experiments:
            print(
                f"  {item['source_domain']} -> {item['target_domain']}: "
                f"priority={item['priority']:.2f}"
            )
            print(f"    question: {item['transfer_question']}")
    if representation_agenda:
        print("Theory representation agenda:")
        for proposal in representation_agenda:
            print(
                f"  {proposal['proposal_kind']}: {proposal['name']} "
                f"for {proposal['theory_kind']} priority={proposal['priority']:.2f}"
            )
            print(f"    expression: {proposal['expression']}")
    if generated_operator_priors:
        print("Theory generated operator priors:")
        for operator in generated_operator_priors:
            parameters = operator.get('parameters', {})
            print(
                f"  {operator['operator_kind']}: {operator['key']} "
                f"usefulness={operator.get('usefulness', 0.0):.2f}"
            )
            print(f"    parameters: {parameters}")
    if operator_prior_feedback:
        print("Theory operator prior feedback:")
        for item in operator_prior_feedback:
            print(
                f"  {item['operator_kind']}: {item['outcome']} "
                f"score={item['best_score']:.2f} in {item['context']}"
            )
    if operator_prior_claims:
        print("Theory operator prior discovery claims:")
        for item in operator_prior_claims:
            evidence = item['proof_evidence']
            print(
                f"  {item['operator_kind']}: status={item['status']} "
                f"support={evidence['confirmed_count']} "
                f"best={evidence['best_score']:.2f}"
            )
            print(f"    claim: {item['claim']}")
    if operator_prior_chains:
        print("Theory operator prior discovery chains:")
        for item in operator_prior_chains:
            print(
                f"  {item['operator_kind']}: status={item['status']} "
                f"steps={len(item['steps'])}"
            )
            if item['steps']:
                print(f"    latest: {item['steps'][-1]['summary']}")
    if operator_prior_claim_experiments:
        print("Theory operator prior claim experiments:")
        for item in operator_prior_claim_experiments:
            print(
                f"  {item['experiment_kind']}: {item['operator_prior_kind']} "
                f"priority={item['priority']:.2f}"
            )
            print(f"    why: {item['reason']}")
    if operator_prior_anomalies:
        print("Theory operator prior anomalies:")
        for item in operator_prior_anomalies:
            print(
                f"  {item['operator_kind']}: {item['anomaly_kind']} "
                f"in {item['failure_context']} severity={item['severity']:.2f}"
            )
            print(f"    question: {item['question']}")
    if operator_prior_domains:
        print("Theory operator prior domains:")
        for item in operator_prior_domains:
            domain = item['domain_hypothesis']
            included = ','.join(domain['included_contexts']) or 'none'
            excluded = ','.join(domain['excluded_contexts']) or 'none'
            print(
                f"  {item['operator_kind']}: include={included}; "
                f"exclude={excluded}"
            )
            print(f"    claim: {domain['claim']}")
    if operator_prior_repairs:
        print("Theory operator prior repair probes:")
        for item in operator_prior_repairs:
            print(
                f"  {item['operator_prior_kind']}: priority={item['priority']:.2f}, "
                f"target={item['target_context']}"
            )
            print(f"    why: {item['reason']}")
    if next_experiments:
        print("Theory next experiments:")
        for experiment in next_experiments:
            print(
                f"  {experiment['experiment_kind']}: {experiment['theory_kind']} "
                f"priority={experiment['priority']:.2f}, "
                f"target={experiment['target_context']}"
            )
            print(f"    why: {experiment['reason']}")
    if planned:
        print("Suggested concrete campaign cases:")
        for plan in planned:
            print(
                f"  {plan['world_type']} seed={plan['seed']} "
                f"objects={plan['object_count']} steps={plan['steps']} "
                f"for {plan['theory_kind']}"
            )
            if plan.get('experiment_kind') == 'model_disagreement_probe':
                mode = plan.get('disagreement_signature', {}).get('mode')
                print(f"    disagreement: {mode}")
            if plan.get('experiment_kind') == 'operator_prior_domain_repair':
                anomaly = plan.get('operator_prior_anomaly', {})
                print(f"    repair: {anomaly.get('anomaly_kind', 'operator_prior_anomaly')}")


def _blind_hidden_discoveries(kb: KnowledgeBase) -> set[str]:
    discoveries = set()
    learned_rules = [
        rule for rule in kb.get_confirmed_rules()
        if rule.properties.get('hypothesis_type') == 'learned_dynamics'
    ]
    law_types = {
        rule.properties.get('law_type')
        for rule in learned_rules
    }
    novel_types = _confirmed_novel_types(kb)

    if 'uniform_acceleration' in law_types or 'uniform_horizontal_force' in novel_types:
        discoveries.add('uniform_component')
    if (
        'radial_field' in law_types
        or 'inverse_square_radial_field' in law_types
        or 'central_force' in novel_types
    ):
        directions = {
            rule.properties.get('direction')
            for rule in learned_rules
            if rule.properties.get('law_type') in {'radial_field', 'inverse_square_radial_field'}
        }
        if 'repulsive' in directions or 'repulsion' in novel_types:
            discoveries.add('repulsive_component')
        if 'attractive' in directions or 'central_force' in novel_types or not directions:
            discoveries.add('radial_component')
    if 'repulsion' in novel_types:
        discoveries.add('repulsive_component')
    if 'tangential_field' in law_types or 'vortex' in novel_types:
        discoveries.add('tangential_component')
    if 'time_varying_field' in law_types or 'time_varying_force' in novel_types:
        discoveries.add('time_varying_component')
    if 'composed_dynamics' in law_types:
        discoveries.add('composed_law')
    return discoveries


def _experiment_proposals_from_knowledge(kb: KnowledgeBase) -> list[dict]:
    proposals = []
    for rule in kb.get_active_hypotheses():
        if rule.properties.get('hypothesis_type') != 'novel_physics':
            continue
        novel_type = rule.properties.get('novel_type')
        if novel_type in ('central_force', 'repulsion'):
            proposals.append({
                'proposal': 'spawn_probe_near_candidate_center',
                'novel_type': novel_type,
            })
        elif novel_type == 'vortex':
            proposals.append({
                'proposal': 'spawn_probe_near_candidate_tangential_center',
                'novel_type': novel_type,
            })
        elif novel_type == 'time_varying_force':
            proposals.append({
                'proposal': 'wait_and_compare_later_acceleration_sign',
                'novel_type': novel_type,
            })
        elif novel_type == 'uniform_horizontal_force':
            proposals.append({
                'proposal': 'spawn_low_motion_probe_at_center',
                'novel_type': novel_type,
            })

    math_discovery = getattr(kb, 'emergent_math_discovery', None)
    if math_discovery is not None:
        for comparison in math_discovery.compare_to_human_math():
            if comparison.human_concept in {'function-like mapping', 'conditional rule'}:
                proposals.append({
                    'proposal': 'repeat_intervention_to_test_mapping',
                    'basis': comparison.internal_key,
                })
                break
    return proposals


def _emergent_math_label_leaks(kb: KnowledgeBase) -> list[dict]:
    leaks = []
    for concept in kb.get_all_concepts():
        if concept.properties.get('source') != 'emergent_math':
            continue
        description = concept.description.lower()
        found = sorted(label for label in FORBIDDEN_AGENT_MATH_LABELS if label in description)
        if found:
            leaks.append({
                'concept': concept.internal_name,
                'feature_key': concept.feature_key,
                'labels': found,
                'description': concept.description,
            })
    return leaks


def _run_experiment_metrics(
    world_type: str,
    seed: int,
    object_count: int,
    steps: int,
    num_agents: int,
    law_memory: LawMemory | None,
) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        _, kb, _ = run_experiment(
            num_steps=steps,
            num_initial_objects=object_count,
            seed=seed,
            verbose=False,
            report_interval=max(steps, 1),
            world_type=world_type,
            num_agents=num_agents,
            law_memory=law_memory,
        )

    learned_rules = [
        rule for rule in kb.get_confirmed_rules()
        if rule.properties.get('hypothesis_type') == 'learned_dynamics'
    ]
    learned_steps = [
        rule.confirmed_at_step or rule.discovered_at_step
        for rule in learned_rules
    ]
    sample_counts = [
        rule.properties.get('sample_count')
        for rule in learned_rules
        if isinstance(rule.properties.get('sample_count'), int)
    ]
    return {
        'detected': _confirmed_novel_types(kb),
        'learned_rule_count': len(learned_rules),
        'learned_law_types': sorted({
            rule.properties.get('law_type', 'unknown')
            for rule in learned_rules
        }),
        'first_learned_step': min(learned_steps) if learned_steps else None,
        'min_sample_count': min(sample_counts) if sample_counts else None,
        'memory_transfer': (
            law_memory.episodes[-1].transfer_report
            if law_memory is not None and law_memory.episodes
            else {'observed': [], 'matched_priors': [], 'missing_priors': []}
        ),
    }


def _fmt_metric_step(value: int | None) -> str:
    return str(value) if value is not None else '-'


def _parse_csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(',') if part.strip()]


def _parse_csv_worlds(value: str) -> list[str]:
    worlds = [part.strip() for part in value.split(',') if part.strip()]
    unknown = [world for world in worlds if world not in WORLD_TYPES]
    if unknown:
        raise ValueError(f"Unknown world type(s): {', '.join(unknown)}")
    return worlds


def _report_discovery(log_entry: dict, step: int):
    """Print a discovery notification in real-time."""
    dtype = log_entry['type']

    if dtype == 'concept':
        print(f"\n  *** NEW CONCEPT DISCOVERED at step {log_entry['step']} ***")
        print(f"      Agent name: {log_entry['name']}")
        print(f"      Description: {log_entry['description']}")
        print(f"      Type: {log_entry['concept_type']}")
        print()
    elif dtype == 'hypothesis':
        print(f"\n  --- HYPOTHESIS FORMED at step {log_entry['step']} ---")
        print(f"      Agent name: {log_entry['name']}")
        print(f"      WHEN: {log_entry['conditions']}")
        print(f"      THEN: {log_entry['prediction']}")
        print()
    elif dtype == 'discovery':
        print(f"\n  !!! RULE CONFIRMED at step {log_entry['step']} !!!")
        print(f"      Agent name: {log_entry['name']}")
        print(f"      WHEN: {log_entry['conditions']}")
        print(f"      THEN: {log_entry['prediction']}")
        print(f"      Confidence: {log_entry['confidence']:.2f} "
              f"(Evidence: {log_entry['evidence_for']}+ / {log_entry['evidence_against']}-)")
        print()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='First Principles AI — Discovery from Scratch'
    )
    parser.add_argument('--steps', type=int, default=2000,
                        help='Number of simulation steps (default: 2000)')
    parser.add_argument('--objects', type=int, default=5,
                        help='Initial number of objects (default: 5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress real-time discovery prints')
    parser.add_argument('--report-interval', type=int, default=100,
                        help='Status report interval in steps (default: 100)')
    parser.add_argument('--world-type', type=str, default='standard',
                        choices=WORLD_TYPES,
                        help='Physics world type (default: standard)')
    parser.add_argument('--agents', type=int, default=2,
                        help='Number of agents for language emergence (default: 2)')
    parser.add_argument('--benchmark-worlds', action='store_true',
                        help='Run a seed/object-count benchmark across physics worlds')
    parser.add_argument('--transfer-benchmark', action='store_true',
                        help='Compare cold runs with warm law-memory-guided runs')
    parser.add_argument('--math-benchmark', action='store_true',
                        help='Benchmark emergent math comparison coverage across repeated runs')
    parser.add_argument('--equation-campaign', action='store_true',
                        help='Run an equation-discovery review campaign')
    parser.add_argument('--math-foundation-prep', action='store_true',
                        help='Run readiness checks before the final watched math discovery run')
    parser.add_argument('--discovery-readiness', action='store_true',
                        help='Print a non-final readiness audit from cumulative theory memory')
    parser.add_argument('--memory-efficiency-review', action='store_true',
                        help='Print bounded-memory and quantized-summary status')
    parser.add_argument('--compact-theory-memory', action='store_true',
                        help='Compact old theory memory evidence into quantized shards')
    parser.add_argument('--memory-keep-records', type=int, default=96,
                        help='Recent raw discovery records to keep after compaction')
    parser.add_argument('--memory-keep-operator-outcomes', type=int, default=192,
                        help='Recent raw operator outcomes to keep after compaction')
    parser.add_argument('--domain-curriculum-preview', action='store_true',
                        help='Preview generated math-domain worlds without running final discovery')
    parser.add_argument('--domain-world-discovery-ingest', action='store_true',
                        help='Record generated domain-world discoveries into theory memory')
    parser.add_argument('--autonomous-scientist-loop', action='store_true',
                        help='Run the non-final scientist loop over repeated domain-world discoveries')
    parser.add_argument('--hf-non-final-campaign', action='store_true',
                        help='Run a Hugging Face friendly non-final discovery campaign')
    parser.add_argument('--math-final-discovery', action='store_true',
                        help='Run the final math discovery campaign when the user is ready to watch')
    parser.add_argument('--equation-hidden-worlds', type=int, default=0,
                        help='Generated hidden worlds to include in equation campaign')
    parser.add_argument('--equation-followup-budget', type=int, default=0,
                        help='Autonomous planned follow-up probes to run after the initial equation campaign')
    parser.add_argument('--explore-worlds', action='store_true',
                        help='Run an autonomous experiment campaign chosen by the planner')
    parser.add_argument('--hidden-holdout-benchmark', action='store_true',
                        help='Run train/holdout benchmark over generated hidden worlds')
    parser.add_argument('--explore-hidden-worlds', action='store_true',
                        help='Run an autonomous campaign over generated hidden worlds')
    parser.add_argument('--hidden-train-worlds', type=int, default=3,
                        help='Hidden worlds used to warm memory before holdout (default: 3)')
    parser.add_argument('--hidden-holdout-worlds', type=int, default=3,
                        help='Hidden holdout worlds to evaluate (default: 3)')
    parser.add_argument('--exploration-budget', type=int, default=12,
                        help='Number of autonomous experiments to run (default: 12)')
    parser.add_argument('--exploration-seed-start', type=int, default=0,
                        help='First seed in the autonomous exploration pool (default: 0)')
    parser.add_argument('--domain-world-seed', type=int, default=0,
                        help='Seed offset for generated math-domain world discovery')
    parser.add_argument('--domain-world-seed-count', type=int, default=1,
                        help='Number of domain-world seeds to record in HF/non-final campaigns')
    parser.add_argument('--domain-world-variant', type=int, default=0,
                        help='Variant for generated math-domain world discovery')
    parser.add_argument('--domain-world-variants', type=str, default=None,
                        help='Comma-separated generated math-domain variants for larger campaigns')
    parser.add_argument('--scientist-seed-start', type=int, default=0,
                        help='First seed for the autonomous scientist loop')
    parser.add_argument('--scientist-seed-count', type=int, default=3,
                        help='Number of seeds for the autonomous scientist loop')
    parser.add_argument('--scientist-variants', type=str, default='0',
                        help='Comma-separated variants for the autonomous scientist loop')
    parser.add_argument('--scientist-live', action='store_true',
                        help='Print SCIENTIST_EVENT JSON lines during scientist loop commands')
    parser.add_argument('--scientist-event-limit', type=int, default=80,
                        help='Maximum live scientist events to retain/print')
    parser.add_argument('--seeds', type=int, default=5,
                        help='Number of seeds to run in benchmark mode (default: 5)')
    parser.add_argument('--benchmark-steps', type=int, default=800,
                        help='Steps per run in benchmark mode (default: 800)')
    parser.add_argument('--object-counts', type=str, default='5',
                        help='Comma-separated object counts for benchmark mode (default: 5)')
    parser.add_argument('--world-types', type=str, default=','.join(WORLD_TYPES),
                        help='Comma-separated world types for benchmark mode')
    parser.add_argument('--memory-file', type=str, default=None,
                        help='Optional JSON file for persistent law memory')
    parser.add_argument('--theory-memory-file', type=str, default=None,
                        help='Optional JSON file for persistent cumulative theory memory')
    parser.add_argument('--hf-output-file', type=str, default=None,
                        help='Optional JSON artifact path for HF non-final campaign output')
    parser.add_argument('--hf-skip-prep', action='store_true',
                        help='HF non-final campaign records domain worlds without running prep cases')
    parser.add_argument('--hf-no-live-scientist', action='store_true',
                        help='Suppress SCIENTIST_EVENT lines in HF non-final campaign logs')
    parser.add_argument('--hf-no-auto-compact', action='store_true',
                        help='Disable automatic theory-memory compaction between HF batches')
    parser.add_argument('--hf-no-adaptive-compute', action='store_true',
                        help='Disable residual-driven compute-budget expansion in HF campaigns')
    parser.add_argument('--hf-max-adaptive-steps', type=int, default=None,
                        help='Maximum steps after adaptive HF compute expansion')
    parser.add_argument('--hf-max-adaptive-seeds', type=int, default=None,
                        help='Maximum seeds after adaptive HF compute expansion')
    parser.add_argument('--hf-max-adaptive-hidden-worlds', type=int, default=None,
                        help='Maximum hidden worlds after adaptive HF compute expansion')
    parser.add_argument('--no-save-memory', action='store_true',
                        help='Load memory without writing updates back to disk')
    parser.add_argument('--no-save-theory-memory', action='store_true',
                        help='Load theory memory without writing updates back to disk')

    args = parser.parse_args()
    law_memory = LawMemory.load(args.memory_file) if args.memory_file else None
    theory_memory = (
        CumulativeTheoryMemory.load(args.theory_memory_file)
        if args.theory_memory_file
        else None
    )

    if args.benchmark_worlds:
        run_benchmark(
            seeds=args.seeds,
            steps=args.benchmark_steps,
            object_counts=_parse_csv_ints(args.object_counts),
            world_types=_parse_csv_worlds(args.world_types),
            num_agents=args.agents,
            law_memory=law_memory,
        )
        if args.memory_file and law_memory is not None and not args.no_save_memory:
            law_memory.save(args.memory_file)
        raise SystemExit(0)

    if args.transfer_benchmark:
        run_transfer_benchmark(
            seeds=args.seeds,
            steps=args.benchmark_steps,
            object_counts=_parse_csv_ints(args.object_counts),
            world_types=_parse_csv_worlds(args.world_types),
            num_agents=args.agents,
            law_memory=law_memory,
        )
        if args.memory_file and law_memory is not None and not args.no_save_memory:
            law_memory.save(args.memory_file)
        raise SystemExit(0)

    if args.math_benchmark:
        run_math_benchmark(
            seeds=args.seeds,
            steps=args.benchmark_steps,
            object_counts=_parse_csv_ints(args.object_counts),
            world_types=_parse_csv_worlds(args.world_types),
            num_agents=args.agents,
        )
        raise SystemExit(0)

    if args.discovery_readiness:
        run_discovery_readiness_audit(theory_memory=theory_memory)
        raise SystemExit(0)

    if args.memory_efficiency_review or args.compact_theory_memory:
        theory_memory = theory_memory or CumulativeTheoryMemory()
        run_memory_efficiency_review(
            theory_memory=theory_memory,
            compact=args.compact_theory_memory,
            keep_recent_records=args.memory_keep_records,
            keep_recent_operator_outcomes=args.memory_keep_operator_outcomes,
        )
        if (
            args.compact_theory_memory
            and args.theory_memory_file
            and theory_memory is not None
            and not args.no_save_theory_memory
        ):
            theory_memory.save(args.theory_memory_file)
        raise SystemExit(0)

    if args.domain_curriculum_preview:
        run_domain_curriculum_preview(theory_memory=theory_memory)
        raise SystemExit(0)

    if args.domain_world_discovery_ingest:
        theory_memory = theory_memory or CumulativeTheoryMemory()
        run_domain_world_discovery_ingest(
            theory_memory=theory_memory,
            seed=args.domain_world_seed,
            variant=args.domain_world_variant,
        )
        if (
            args.theory_memory_file
            and theory_memory is not None
            and not args.no_save_theory_memory
        ):
            theory_memory.save(args.theory_memory_file)
        raise SystemExit(0)

    if args.autonomous_scientist_loop:
        theory_memory = theory_memory or CumulativeTheoryMemory()
        run_autonomous_scientist_loop(
            theory_memory=theory_memory,
            seed_start=args.scientist_seed_start,
            seed_count=args.scientist_seed_count,
            variants=_parse_csv_ints(args.scientist_variants),
            live=args.scientist_live,
            event_limit=args.scientist_event_limit,
        )
        if (
            args.theory_memory_file
            and theory_memory is not None
            and not args.no_save_theory_memory
        ):
            theory_memory.save(args.theory_memory_file)
        raise SystemExit(0)

    if args.hf_non_final_campaign:
        theory_memory = theory_memory or CumulativeTheoryMemory()
        run_hf_non_final_campaign(
            theory_memory=theory_memory,
            seeds=args.seeds,
            steps=args.benchmark_steps,
            object_counts=_parse_csv_ints(args.object_counts),
            world_types=_parse_csv_worlds(args.world_types),
            hidden_worlds=args.equation_hidden_worlds,
            num_agents=args.agents,
            output_file=args.hf_output_file,
            domain_seed=args.domain_world_seed,
            domain_variant=args.domain_world_variant,
            domain_seed_count=args.domain_world_seed_count,
            domain_variants=(
                _parse_csv_ints(args.domain_world_variants)
                if args.domain_world_variants
                else [args.domain_world_variant]
            ),
            scientist_seed_count=args.scientist_seed_count,
            scientist_variants=_parse_csv_ints(args.scientist_variants),
            live_scientist=not args.hf_no_live_scientist,
            include_prep=not args.hf_skip_prep,
            auto_compact=not args.hf_no_auto_compact,
            compact_keep_records=args.memory_keep_records,
            compact_keep_operator_outcomes=args.memory_keep_operator_outcomes,
            adaptive_compute=not args.hf_no_adaptive_compute,
            max_adaptive_steps=args.hf_max_adaptive_steps,
            max_adaptive_seeds=args.hf_max_adaptive_seeds,
            max_adaptive_hidden_worlds=args.hf_max_adaptive_hidden_worlds,
        )
        if (
            args.theory_memory_file
            and theory_memory is not None
            and not args.no_save_theory_memory
        ):
            theory_memory.save(args.theory_memory_file)
        raise SystemExit(0)

    if args.equation_campaign:
        run_equation_campaign(
            seeds=args.seeds,
            steps=args.benchmark_steps,
            object_counts=_parse_csv_ints(args.object_counts),
            world_types=_parse_csv_worlds(args.world_types),
            hidden_worlds=args.equation_hidden_worlds,
            num_agents=args.agents,
            followup_budget=args.equation_followup_budget,
            theory_memory=theory_memory,
        )
        if (
            args.theory_memory_file
            and theory_memory is not None
            and not args.no_save_theory_memory
        ):
            theory_memory.save(args.theory_memory_file)
        raise SystemExit(0)

    if args.math_foundation_prep:
        run_math_foundation_prep(
            seeds=args.seeds,
            steps=args.benchmark_steps,
            object_counts=_parse_csv_ints(args.object_counts),
            world_types=_parse_csv_worlds(args.world_types),
            hidden_worlds=args.equation_hidden_worlds,
            num_agents=args.agents,
            theory_memory=theory_memory,
        )
        if (
            args.theory_memory_file
            and theory_memory is not None
            and not args.no_save_theory_memory
        ):
            theory_memory.save(args.theory_memory_file)
        raise SystemExit(0)

    if args.math_final_discovery:
        run_math_final_discovery(
            seeds=args.seeds,
            steps=args.benchmark_steps,
            object_counts=_parse_csv_ints(args.object_counts),
            world_types=_parse_csv_worlds(args.world_types),
            hidden_worlds=args.equation_hidden_worlds,
            num_agents=args.agents,
            theory_memory=theory_memory,
        )
        if (
            args.theory_memory_file
            and theory_memory is not None
            and not args.no_save_theory_memory
        ):
            theory_memory.save(args.theory_memory_file)
        raise SystemExit(0)

    if args.explore_worlds:
        run_autonomous_exploration(
            budget=args.exploration_budget,
            steps=args.benchmark_steps,
            object_counts=_parse_csv_ints(args.object_counts),
            world_types=_parse_csv_worlds(args.world_types),
            num_agents=args.agents,
            law_memory=law_memory,
            seed_start=args.exploration_seed_start,
            seed_span=args.seeds,
        )
        if args.memory_file and law_memory is not None and not args.no_save_memory:
            law_memory.save(args.memory_file)
        raise SystemExit(0)

    if args.hidden_holdout_benchmark:
        run_hidden_holdout_benchmark(
            train_worlds=args.hidden_train_worlds,
            holdout_worlds=args.hidden_holdout_worlds,
            steps=args.benchmark_steps,
            object_count=_parse_csv_ints(args.object_counts)[0],
            num_agents=args.agents,
            law_memory=law_memory,
            seed_start=args.exploration_seed_start,
        )
        if args.memory_file and law_memory is not None and not args.no_save_memory:
            law_memory.save(args.memory_file)
        raise SystemExit(0)

    if args.explore_hidden_worlds:
        run_hidden_autonomous_exploration(
            budget=args.exploration_budget,
            steps=args.benchmark_steps,
            object_count=_parse_csv_ints(args.object_counts)[0],
            num_agents=args.agents,
            law_memory=law_memory,
            seed_start=args.exploration_seed_start,
        )
        if args.memory_file and law_memory is not None and not args.no_save_memory:
            law_memory.save(args.memory_file)
        raise SystemExit(0)

    if law_memory is None and args.memory_file:
        law_memory = LawMemory()

    run_experiment(
        num_steps=args.steps,
        num_initial_objects=args.objects,
        seed=args.seed,
        verbose=not args.quiet,
        report_interval=args.report_interval,
        world_type=args.world_type,
        num_agents=args.agents,
        law_memory=law_memory,
    )
    if args.memory_file and law_memory is not None and not args.no_save_memory:
        law_memory.save(args.memory_file)
