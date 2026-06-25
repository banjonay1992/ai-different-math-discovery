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
import concurrent.futures
import contextlib
import hashlib
import io
import json
import base64
import zlib
import time
from collections import Counter
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from world.environment import Environment
from world.physics import PhysicsObject
from world.tensor_backend import (
    available_force_backends,
    compute_external_force_deltas,
    resolve_force_backend,
)
from world.hidden_worlds import (
    HiddenWorldManifest,
    generate_hidden_world_manifest,
    generate_self_authored_hidden_world_manifest,
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
from agent.equation_tensor_backend import available_equation_scoring_backends
from agent.compute_budget import plan_adaptive_compute_budget
from agent.discovery_loop import AutonomousDiscoveryLoop, CumulativeTheoryMemory
from agent.math_foundation import MathFoundationWorkbench
from agent.resource_efficiency import estimate_json_bytes
from agent.super_system import (
    build_experiment_design_cockpit,
    build_super_system_report,
    experiment_design_from_plan,
    format_intervention_action,
    planned_probe_actions,
    super_system_summary_lines,
    theory_beliefs_from_plan,
)
from agent.status_capsule import (
    build_ai_different_status_capsule,
    git_check_ignore_for_path,
    git_status_for_path,
    load_capsule_memory_data,
)
from agent.family_outcome_evaluator import (
    append_outcome_evaluator_memory,
    build_outcome_evaluator_ledger,
    export_outcome_evaluator_message,
    load_outcome_evaluator_memory,
    load_response_ledgers,
    write_outcome_evaluator_ledger,
    write_outcome_evaluator_memory,
)
from agent.experiment_contracts import (
    load_experiment_contract_ledger,
    read_family_bus_messages,
    update_contract_ledger,
    validate_evaluator_ledger,
    write_contract_outbox_jsonl,
    write_experiment_contract_ledger,
)
from agent.cross_module_adjudicator import (
    build_adjudication,
    load_adjudicator_ledger,
    load_plain_json,
    read_family_transcript,
    write_adjudication_outbox_jsonl,
    write_adjudicator_ledger,
)
from agent.experiment_agenda import (
    build_experiment_agenda,
    load_experiment_agenda_ledger,
    load_plain_json as load_agenda_plain_json,
    read_agenda_transcript,
    write_agenda_outbox_jsonl,
    write_experiment_agenda_ledger,
)
from agent.hypothesis_lifecycle import (
    build_hypothesis_lifecycle,
    load_hypothesis_lifecycle_ledger,
    load_plain_json as load_lifecycle_plain_json,
    read_lifecycle_transcript,
    write_hypothesis_lifecycle_ledger,
    write_lifecycle_outbox_jsonl,
)
from agent.evidence_scorecard import (
    build_evidence_scorecard,
    load_plain_json as load_scorecard_plain_json,
    load_scorecard_ledger,
    read_scorecard_transcript,
    write_scorecard_ledger,
    write_scorecard_outbox_jsonl,
)
from agent.campaign_planner import (
    build_experiment_campaign,
    load_campaign_ledger,
    load_plain_json as load_campaign_plain_json,
    read_campaign_transcript,
    write_campaign_ledger,
    write_campaign_outbox_jsonl,
)
from agent.campaign_outcome_assessor import (
    build_campaign_outcome_assessment,
    load_campaign_outcome_ledger,
    load_plain_json as load_campaign_outcome_plain_json,
    read_campaign_outcome_transcript,
    write_campaign_outcome_ledger,
    write_campaign_outcome_outbox_jsonl,
)
from agent.science_benefit_evaluator import (
    build_science_benefit_evaluation,
    load_plain_json as load_science_benefit_plain_json,
    load_science_benefit_ledger,
    read_science_benefit_transcript,
    write_science_benefit_ledger,
    write_science_benefit_outbox_jsonl,
)
from agent.science_campaign_action_planner import (
    build_science_campaign_action_plan,
    load_plain_json as load_science_action_plain_json,
    load_science_action_ledger,
    read_science_action_transcript,
    write_science_action_ledger,
    write_science_action_outbox_jsonl,
)
from agent.science_action_outcome_assessor import (
    build_science_action_outcome_assessment,
    load_plain_json as load_science_action_outcome_plain_json,
    load_science_action_outcome_ledger,
    read_science_action_outcome_transcript,
    write_science_action_outcome_ledger,
    write_science_action_outcome_outbox_jsonl,
)
from agent.science_theory_frontier import (
    build_science_theory_frontier_plan,
    load_plain_json as load_science_theory_frontier_plain_json,
    load_science_theory_frontier_ledger,
    read_science_theory_frontier_transcript,
    write_science_theory_frontier_ledger,
    write_science_theory_frontier_outbox_jsonl,
)
from agent.science_theory_frontier_outcome import (
    build_science_theory_frontier_outcome_assessment,
    load_plain_json as load_science_theory_frontier_outcome_plain_json,
    load_science_theory_frontier_outcome_ledger,
    read_science_theory_frontier_outcome_transcript,
    write_science_theory_frontier_outcome_ledger,
    write_science_theory_frontier_outcome_outbox_jsonl,
)
from agent.module_chat_adapter import (
    append_rolling_family_record,
    build_module_family_response_ledger,
    choose_module_family_followup,
    choose_module_family_recipient,
    export_chat_driven_response_message,
    export_capsule_chat_message,
    export_module_family_response_message,
    load_rolling_family_memory,
    module_chat_summary_from_messages,
    read_module_chat_inbox,
    read_module_chat_log,
    rolling_unprocessed_inbound_messages,
    write_response_ledger,
    write_rolling_family_memory,
)
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
    'zero_gravity',
    'damping_component',
    'piecewise_component',
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


def _record_stage_timing(profile: dict | None, stage: str, started: float):
    if profile is None:
        return
    elapsed = time.perf_counter() - started
    stages = profile.setdefault('stages', {})
    item = stages.setdefault(stage, {'seconds': 0.0, 'count': 0})
    item['seconds'] += elapsed
    item['count'] += 1


def _experiment_runtime_profile_summary(
    profile: dict | None,
    *,
    elapsed_seconds: float,
    steps: int,
    force_backend: str,
    equation_scoring_backend: str = 'python',
) -> dict:
    if profile is None:
        return {'enabled': False}
    stages = []
    for stage, item in dict(profile.get('stages') or {}).items():
        seconds = float(item.get('seconds', 0.0) or 0.0)
        count = int(item.get('count', 0) or 0)
        stages.append({
            'stage': stage,
            'seconds': round(seconds, 6),
            'count': count,
            'avg_seconds': round(seconds / max(count, 1), 6),
            'share_of_elapsed': round(seconds / max(elapsed_seconds, 1e-9), 4),
        })
    stages.sort(key=lambda item: item['seconds'], reverse=True)
    return {
        'enabled': True,
        'steps': int(steps),
        'force_backend': force_backend,
        'equation_scoring_backend': equation_scoring_backend,
        'elapsed_seconds': round(elapsed_seconds, 6),
        'profiled_stage_seconds': round(
            sum(float(item['seconds']) for item in stages),
            6,
        ),
        'hot_stage': stages[0]['stage'] if stages else None,
        'stages': stages,
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
    force_backend: str = 'python',
    equation_scoring_backend: str = 'python',
    profile_timings: bool = False,
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
    print(f"  Force backend: {force_backend}")
    print(f"  Equation scoring backend: {equation_scoring_backend}")
    print(f"  Timing profile: {profile_timings}")
    print()
    print("-" * 70)
    print("EXPERIMENT STARTING...")
    print("-" * 70)

    # Initialize
    run_started = time.perf_counter()
    runtime_profile = {'stages': {}} if profile_timings else None
    PhysicsObject._next_id = 0
    started = time.perf_counter()
    env = Environment(
        num_initial_objects=num_initial_objects,
        seed=seed,
        world_type=world_type,
        hidden_manifest=hidden_manifest,
        force_backend=force_backend,
    )
    _record_stage_timing(runtime_profile, 'environment_init', started)
    started = time.perf_counter()
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
        scoring_backend=equation_scoring_backend,
    )
    kb.equation_workbench = equation_workbench
    _record_stage_timing(runtime_profile, 'agent_init', started)

    # Track what we've already reported to avoid duplicate prints
    reported_concepts = set()
    reported_rules = set()

    prev_discovery_count = 0
    started = time.perf_counter()
    current_raw_state = env.observe()
    _record_stage_timing(runtime_profile, 'environment_observe', started)
    started = time.perf_counter()
    current_observation = Perception.perceive(current_raw_state)
    _record_stage_timing(runtime_profile, 'perception', started)
    started = time.perf_counter()
    current_features = current_observation.get_feature_vector()
    _record_stage_timing(runtime_profile, 'feature_vector', started)

    for step in range(num_steps):
        # 1. OBSERVE the world
        raw_state = current_raw_state
        observation = current_observation
        features = current_features
        had_collision = len(observation.collisions) > 0

        # 2. PREDICT what happens next (using current knowledge)
        started = time.perf_counter()
        predicted = predictor.predict_next(features)
        _record_stage_timing(runtime_profile, 'predict_next', started)

        # 3. SELECT an action (planned falsification, active experiment, curiosity)
        action = None
        if planned_actions and step < len(planned_actions):
            action = dict(planned_actions[step])
        if action is None:
            started = time.perf_counter()
            action = predictor.suggest_experiment_action(
                current_count=features.get('count', 0),
                world_width=observation.world_width or 20.0,
                world_height=observation.world_height or 20.0,
            )
            _record_stage_timing(runtime_profile, 'predictor_action_select', started)
        if action is None and enable_equation_probes:
            if step > 0 and step % 40 == 0:
                started = time.perf_counter()
                equation_workbench.discover(step=step)
                _record_stage_timing(runtime_profile, 'equation_mid_discover', started)
            started = time.perf_counter()
            action = equation_workbench.suggest_probe_action(
                current_count=features.get('count', 0),
                world_width=observation.world_width or 20.0,
                world_height=observation.world_height or 20.0,
                step=step,
            )
            _record_stage_timing(runtime_profile, 'equation_probe_select', started)
        if action is None:
            started = time.perf_counter()
            action = curiosity.select_action(predictor, features)
            _record_stage_timing(runtime_profile, 'curiosity_action_select', started)

        # 4. ACT and observe the outcome
        state_before = raw_state
        started = time.perf_counter()
        raw_state = env.step(action)
        _record_stage_timing(runtime_profile, 'environment_step', started)
        started = time.perf_counter()
        math_discovery.observe_transition(state_before, raw_state, action, step + 1)
        _record_stage_timing(runtime_profile, 'math_transition_observe', started)
        started = time.perf_counter()
        equation_workbench.observe_transition(state_before, raw_state, action, step + 1)
        _record_stage_timing(runtime_profile, 'equation_transition_observe', started)
        started = time.perf_counter()
        observation = Perception.perceive(raw_state)
        _record_stage_timing(runtime_profile, 'perception', started)
        started = time.perf_counter()
        new_features = observation.get_feature_vector()
        _record_stage_timing(runtime_profile, 'feature_vector', started)
        had_collision = len(observation.collisions) > 0
        current_raw_state = raw_state
        current_observation = observation
        current_features = new_features

        # 5. LEARN — record observations and check for discoveries
        started = time.perf_counter()
        predictor.observe(new_features, had_collision, step + 1, raw_objects=raw_state.get('objects', []))
        _record_stage_timing(runtime_profile, 'predictor_observe', started)

        # 6. CAUSAL REASONING — track actions and effects
        started = time.perf_counter()
        causal.observe_step(action, state_before, raw_state, step + 1, observation.collisions)
        _record_stage_timing(runtime_profile, 'causal_observe', started)

        # 7. PROGRAM SYNTHESIS — record actions for pattern detection
        started = time.perf_counter()
        synthesizer.record_action(action, step + 1, new_features)
        _record_stage_timing(runtime_profile, 'synthesis_record', started)

        # 8. MULTI-AGENT LANGUAGE — agents communicate about world state
        started = time.perf_counter()
        multi_agent.step_agents(
            new_features,
            had_collision,
            step + 1,
            math_patterns=math_discovery.discovered_patterns(),
        )
        _record_stage_timing(runtime_profile, 'language_step', started)

        # 9. SELF-MODIFICATION — agent inspects and improves its own reasoning
        started = time.perf_counter()
        if kb.discovery_log and len(kb.discovery_log) > prev_discovery_count:
            self_modifier.record_discovery(step + 1)
        self_modifier.check_and_modify(step + 1)
        _record_stage_timing(runtime_profile, 'self_modify', started)

        # 10. Calculate prediction error (drives future curiosity)
        started = time.perf_counter()
        error = predictor.prediction_error(predicted, new_features)

        # 11. Decay exploration over time
        curiosity.decay_exploration()
        _record_stage_timing(runtime_profile, 'prediction_error_curiosity_decay', started)

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
    started = time.perf_counter()
    equation_workbench.discover(step=num_steps)
    _record_stage_timing(runtime_profile, 'equation_final_discover', started)
    math_foundation = MathFoundationWorkbench(
        kb,
        math_discovery=math_discovery,
        equation_workbench=equation_workbench,
    )
    started = time.perf_counter()
    foundation_report = math_foundation.evaluate(install=True)
    _record_stage_timing(runtime_profile, 'math_foundation_evaluate', started)
    kb.math_foundation_report = foundation_report

    print()
    print("-" * 70)
    print("MAPPING DISCOVERIES TO HUMAN CONCEPTS...")
    print("-" * 70)

    started = time.perf_counter()
    map_discoveries(kb, tracker)
    _record_stage_timing(runtime_profile, 'map_discoveries', started)

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
        started = time.perf_counter()
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
        _record_stage_timing(runtime_profile, 'law_memory_record', started)

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

    kb.runtime_profile = _experiment_runtime_profile_summary(
        runtime_profile,
        elapsed_seconds=time.perf_counter() - run_started,
        steps=num_steps,
        force_backend=force_backend,
        equation_scoring_backend=equation_scoring_backend,
    )
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
    force_backend: str = 'python',
    equation_scoring_backend: str = 'python',
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
                        force_backend=force_backend,
                        equation_scoring_backend=equation_scoring_backend,
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
    force_backend: str = 'python',
    equation_scoring_backend: str = 'python',
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
                    world_type, seed, object_count, steps, num_agents,
                    law_memory=None, force_backend=force_backend,
                    equation_scoring_backend=equation_scoring_backend,
                )
                warm = _run_experiment_metrics(
                    world_type, seed, object_count, steps, num_agents,
                    law_memory=warm_memory, force_backend=force_backend,
                    equation_scoring_backend=equation_scoring_backend,
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
    force_backend: str = 'python',
    equation_scoring_backend: str = 'python',
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
            force_backend=force_backend,
            equation_scoring_backend=equation_scoring_backend,
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
    force_backend: str = 'python',
    equation_scoring_backend: str = 'python',
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
            force_backend=force_backend,
            equation_scoring_backend=equation_scoring_backend,
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
            force_backend=force_backend,
            equation_scoring_backend=equation_scoring_backend,
        )
        warm = _run_hidden_experiment_metrics(
            manifest=manifest,
            seed=seed,
            object_count=object_count,
            steps=steps,
            num_agents=num_agents,
            law_memory=law_memory,
            allow_memory_probes=False,
            force_backend=force_backend,
            equation_scoring_backend=equation_scoring_backend,
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
    force_backend: str = 'python',
    equation_scoring_backend: str = 'python',
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
            force_backend=force_backend,
            equation_scoring_backend=equation_scoring_backend,
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
    emit_hf_artifact_summary: bool = False,
    force_backend: str = 'python',
    equation_scoring_backend: str = 'python',
) -> list[dict]:
    """Run worlds and collect equation-review packs for manual inspection."""
    object_counts = object_counts or [5]
    world_types = world_types or ['standard', 'sideways_wind', 'vortex']
    results = []
    theory_memory = theory_memory or CumulativeTheoryMemory()
    starting_memory_summary = theory_memory.memory_checkpoint_summary()

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
                    force_backend=force_backend,
                    equation_scoring_backend=equation_scoring_backend,
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
                theory_memory.record_equation_case_result(
                    result['context'],
                    result['seed'],
                    result,
                    phase='equation_campaign',
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
            force_backend=force_backend,
            equation_scoring_backend=equation_scoring_backend,
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
        theory_memory.record_equation_case_result(
            result['context'],
            result['seed'],
            result,
            phase='equation_campaign',
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
        force_backend=force_backend,
        equation_scoring_backend=equation_scoring_backend,
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
    memory_delta = theory_memory.memory_delta_since(starting_memory_summary)
    print(
        "Theory memory delta: "
        f"records={memory_delta['new_records']}, "
        f"equation_cases={memory_delta['new_equation_cases']}, "
        f"planned_outcomes={memory_delta['new_planned_outcomes']}, "
        f"readiness_delta={memory_delta['readiness_score_delta']:.3f}"
    )
    if emit_hf_artifact_summary:
        print(
            "HF_ARTIFACT_SUMMARY "
            + json.dumps(
                _equation_campaign_artifact_summary(
                    results,
                    theory_memory,
                    starting_memory_summary=starting_memory_summary,
                ),
                sort_keys=True,
            ),
            flush=True,
        )
    return results


def _equation_campaign_artifact_summary(
    results: list[dict],
    theory_memory: CumulativeTheoryMemory,
    starting_memory_summary: dict | None = None,
) -> dict:
    ending_memory_summary = theory_memory.memory_checkpoint_summary()
    return {
        'run_kind': 'equation_campaign',
        'runs_final': False,
        'starting_memory': dict(starting_memory_summary or {}),
        'ending_memory': ending_memory_summary,
        'memory_delta': theory_memory.memory_delta_since(starting_memory_summary or {}),
        'result_count': len(results),
        'passed_count': sum(1 for result in results if result.get('passed')),
        'rows': [
            {
                'context': result.get('context'),
                'seed': result.get('seed'),
                'steps': result.get('steps'),
                'equation_count': result.get('equation_count', 0),
                'label_leak_count': len(result.get('label_leaks') or []),
                'interesting_score': round(
                    float(result.get('interesting_score', 0.0) or 0.0),
                    3,
                ),
                'interesting_expression': dict(
                    result.get('interesting_equation') or {}
                ).get('expression'),
                'planned_experiment_kind': dict(
                    result.get('planned_experiment') or {}
                ).get('experiment_kind'),
                'planned_outcome': dict(
                    result.get('planned_experiment_outcome') or {}
                ).get('outcome'),
            }
            for result in results
        ],
        'readiness': theory_memory.discovery_readiness_report(),
        'theorem_consolidations': theory_memory.theorem_consolidations(limit=5),
        'domain_predicate_learning_agenda': (
            theory_memory.domain_predicate_learning_agenda(limit=5)
        ),
        'selected_law_conflict_experiments': (
            theory_memory.selected_law_conflict_experiments(limit=5)
        ),
        'model_disagreement_domain_split_experiments': (
            theory_memory.model_disagreement_domain_split_experiments(limit=5)
        ),
        'localized_gravity_structure_experiments': (
            theory_memory.localized_gravity_structure_experiments(limit=5)
        ),
        'blind_holdout_validation_experiments': (
            theory_memory.blind_holdout_validation_experiments(limit=5)
        ),
        'law_domain_split_hypotheses': theory_memory.law_domain_split_hypotheses(
            limit=5
        ),
        'planned_outcomes_tail': list(theory_memory.planned_outcomes[-12:]),
    }


def _abstraction_equation(
    *,
    key: str,
    role: str,
    score: float = 0.84,
    target: str = 'baseline_adjusted_delta_velocity',
    expression: str = 'candidate_expression',
    parameters: dict | None = None,
    mse: float = 0.02,
    baseline_mse: float = 0.30,
) -> dict:
    return {
        'key': key,
        'target': target,
        'expression': expression,
        'description': 'label-clean abstraction campaign candidate',
        'score': score,
        'mse': mse,
        'baseline_mse': baseline_mse,
        'complexity': 5,
        'sample_count': 120,
        'parameters': dict(parameters or {}),
        'role': role,
    }


def _abstraction_transfer_source_cases(seed_start: int = 0) -> list[dict]:
    return [
        {
            'context': 'surface_form_local_domain',
            'seed': int(seed_start),
            'step': 180,
            'current_count': 4,
            'equations': [
                _abstraction_equation(
                    key='abs_source_local_direction',
                    role='local_residual_direction_equation',
                    score=0.80,
                    expression='unit(candidate_center - position)',
                    parameters={'center_x': 8.0, 'center_y': 12.0, 'k': 0.2},
                ),
                _abstraction_equation(
                    key='abs_source_cutoff_direction',
                    role='generated_operator_cutoff_direction_equation',
                    score=0.84,
                    expression='inside(separation <= r) * unit_vector',
                    parameters={
                        'center_x': 8.0,
                        'center_y': 12.0,
                        'cutoff_radius': 6.0,
                        'cutoff_mse_improvement': 0.12,
                        'cutoff_vs_smooth_improvement': 0.08,
                    },
                ),
                _abstraction_equation(
                    key='abs_source_tapered_direction',
                    role='generated_operator_tapered_distance_direction_equation',
                    score=0.90,
                    expression='domain_weight * unit_vector / separation^2',
                    parameters={
                        'center_x': 8.0,
                        'center_y': 12.0,
                        'cutoff_radius': 6.0,
                        'distance_exponent': 2.0,
                        'distance_mse_improvement': 0.16,
                        'tapered_vs_cutoff_improvement': 0.06,
                        'tapered_vs_smooth_improvement': 0.10,
                    },
                ),
            ],
        },
        {
            'context': 'surface_form_cyclic_residual',
            'seed': int(seed_start) + 1,
            'step': 181,
            'current_count': 4,
            'equations': [
                _abstraction_equation(
                    key='abs_source_periodic_residual',
                    role='residual_periodic_equation',
                    score=0.82,
                    target='baseline_adjusted_delta_vx',
                    expression='a * sin(step / period) + b * cos(step / period)',
                    parameters={'period_steps': 40, 'amplitude': 0.18},
                ),
                _abstraction_equation(
                    key='abs_source_generated_periodic_residual',
                    role='generated_operator_periodic_equation',
                    score=0.86,
                    target='baseline_adjusted_delta_vy',
                    expression='phase_basis(step, period) dot residual_channel',
                    parameters={'period_steps': 40, 'amplitude': 0.16},
                ),
            ],
        },
    ]


def _target_family_for_abstraction_plan(plan: dict) -> str:
    families = list(plan.get('target_theory_kinds') or [])
    preferred = [
        'tapered_distance_direction_residual',
        'cutoff_direction_residual',
        'distance_scaled_direction_residual',
        'periodic_residual',
        'direction_residual',
    ]
    for family in preferred:
        if family in families:
            return family
    return str(families[0]) if families else 'distance_scaled_direction_residual'


def _abstraction_transfer_target_equation(
    plan: dict,
    outcome_mode: str,
) -> dict:
    if outcome_mode == 'absent':
        return _abstraction_equation(
            key='abs_target_absent_transition',
            role='constant_change_equation',
            score=0.72,
            target='delta_position',
            expression='constant_delta',
            parameters={'dx': 0.1, 'dy': 0.0},
        )

    family = _target_family_for_abstraction_plan(plan)
    strong = outcome_mode == 'confirmed'
    if family == 'periodic_residual':
        period = 48 if strong else None
        return _abstraction_equation(
            key='abs_target_periodic_transfer',
            role='generated_operator_periodic_equation',
            score=0.84 if strong else 0.58,
            target='baseline_adjusted_delta_vx',
            expression='phase_basis(step, period) dot residual_channel',
            parameters={'period_steps': period, 'amplitude': 0.13},
        )
    if family == 'cutoff_direction_residual':
        return _abstraction_equation(
            key='abs_target_cutoff_transfer',
            role='generated_operator_cutoff_direction_equation',
            score=0.84 if strong else 0.61,
            expression='inside(separation <= r) * unit_vector',
            parameters={
                'center_x': 10.0,
                'center_y': 10.0,
                'cutoff_radius': 5.0,
                'cutoff_mse_improvement': 0.10 if strong else 0.01,
                'cutoff_vs_smooth_improvement': 0.08 if strong else 0.0,
            },
        )
    if family == 'direction_residual':
        return _abstraction_equation(
            key='abs_target_direction_transfer',
            role='local_residual_direction_equation',
            score=0.80 if strong else 0.55,
            expression='unit(candidate_center - position)',
            parameters={
                'center_x': 10.0 if strong else None,
                'center_y': 10.0 if strong else None,
            },
        )
    if family == 'distance_scaled_direction_residual':
        return _abstraction_equation(
            key='abs_target_distance_transfer',
            role='generated_operator_distance_scaled_direction_equation',
            score=0.85 if strong else 0.60,
            expression='unit_vector / separation^2',
            parameters={
                'center_x': 10.0,
                'center_y': 10.0,
                'distance_exponent': 2.0,
                'distance_mse_improvement': 0.13 if strong else 0.01,
            },
        )
    return _abstraction_equation(
        key='abs_target_tapered_transfer',
        role='generated_operator_tapered_distance_direction_equation',
        score=0.88 if strong else 0.62,
        expression='domain_weight * unit_vector / separation^2',
        parameters={
            'center_x': 10.0,
            'center_y': 10.0,
            'cutoff_radius': 5.0,
            'distance_exponent': 2.0,
            'distance_mse_improvement': 0.14 if strong else 0.01,
            'tapered_vs_cutoff_improvement': 0.06 if strong else 0.0,
            'tapered_vs_smooth_improvement': 0.09 if strong else 0.0,
        },
    )


def _abstraction_transfer_campaign_plan(
    theory_memory: CumulativeTheoryMemory,
    *,
    world_types: list[str],
    object_count: int,
    steps: int,
    seed_start: int,
) -> dict:
    plans = theory_memory.planned_experiments(
        world_types=world_types,
        object_counts=[object_count],
        steps=steps,
        seed_start=seed_start,
        limit=10,
    )
    for plan in plans:
        if plan.get('experiment_kind') == 'abstraction_transfer_probe':
            return plan
    experiments = theory_memory.abstraction_transfer_experiments(limit=1)
    if not experiments:
        return {}
    recommendation = experiments[0]
    context = theory_memory._select_plan_context(recommendation, world_types)
    return {
        'theory_kind': recommendation['theory_kind'],
        'experiment_kind': recommendation['experiment_kind'],
        'priority': recommendation['priority'],
        'world_type': context,
        'seed': seed_start,
        'object_count': object_count,
        'steps': steps,
        'hidden_holdout': context == 'hidden_procedural',
        'reason': recommendation['reason'],
        'expected_result': recommendation['expected_result'],
        'falsifies_if': recommendation['falsifies_if'],
        'source_status': recommendation.get('family_status'),
        'target_context': recommendation.get('target_context'),
        'abstraction_key': recommendation.get('abstraction_key'),
        'abstraction_kind': recommendation.get('abstraction_kind'),
        'canonical_name': recommendation.get('canonical_name'),
        'compressed_expression': recommendation.get('compressed_expression'),
        'compression_rule': recommendation.get('compression_rule'),
        'transfer_target': recommendation.get('transfer_target'),
        'unrelated_world': recommendation.get('unrelated_world'),
        'solve_hint': recommendation.get('solve_hint'),
        'source_contexts': list(recommendation.get('source_contexts') or []),
        'target_theory_kinds': list(recommendation.get('target_theory_kinds') or []),
        'abstraction_bridge': dict(recommendation.get('abstraction_bridge') or {}),
    }


def run_abstraction_transfer_campaign(
    theory_memory: CumulativeTheoryMemory | None = None,
    *,
    seed_start: int = 0,
    steps: int = 120,
    object_count: int = 5,
    target_world_types: list[str] | None = None,
    outcome_mode: str = 'confirmed',
    emit_hf_artifact_summary: bool = False,
) -> dict:
    """Run a tiny non-final abstraction-transfer campaign."""
    if outcome_mode not in {'confirmed', 'weak', 'absent'}:
        raise ValueError('outcome_mode must be confirmed, weak, or absent')
    theory_memory = theory_memory or CumulativeTheoryMemory()
    target_world_types = target_world_types or [
        'standard',
        'time_varying',
        'inverse_square_repulsion',
        'localized_gravity',
        'hidden_procedural',
    ]
    starting_memory_summary = theory_memory.memory_checkpoint_summary()
    loop = AutonomousDiscoveryLoop()
    source_results = []

    print("=" * 70)
    print("ABSTRACTION TRANSFER CAMPAIGN")
    print("=" * 70)
    print(f"Outcome mode: {outcome_mode}")
    print(f"Target worlds: {', '.join(target_world_types)}")
    print()

    for case in _abstraction_transfer_source_cases(seed_start=seed_start):
        report = loop.build_report(
            case['equations'],
            step=case['step'],
            current_count=case['current_count'],
        )
        theory_memory.record_result(case['context'], case['seed'], report)
        packed_report = report.to_dict()
        source_results.append({
            'context': case['context'],
            'seed': case['seed'],
            'bridge_count': len(packed_report.get('abstraction_bridges') or []),
            'bridge_kinds': [
                bridge.get('abstraction_kind')
                for bridge in packed_report.get('abstraction_bridges') or []
            ],
            'concept_kinds': [
                concept.get('concept_kind')
                for concept in packed_report.get('concept_proposals') or []
            ],
        })
        print(
            f"{case['context']:28s} seed={case['seed']:3d} "
            f"bridges={source_results[-1]['bridge_count']:2d} "
            f"kinds={','.join(source_results[-1]['bridge_kinds'][:4])}"
        )

    bridges = theory_memory.abstraction_bridges(limit=8)
    plan = _abstraction_transfer_campaign_plan(
        theory_memory,
        world_types=target_world_types,
        object_count=object_count,
        steps=steps,
        seed_start=seed_start + 100,
    )
    transfer_result = {}
    if plan:
        target_context = str(plan.get('world_type') or 'hidden_procedural')
        target_equation = _abstraction_transfer_target_equation(plan, outcome_mode)
        target_report = loop.build_report(
            [target_equation],
            step=steps,
            current_count=object_count,
        )
        outcome = theory_memory.record_planned_result(
            plan,
            context=target_context,
            seed=int(plan.get('seed', seed_start + 100) or seed_start + 100),
            report=target_report,
        )
        transfer_result = {
            'context': target_context,
            'seed': int(plan.get('seed', seed_start + 100) or seed_start + 100),
            'target_equation_role': target_equation.get('role'),
            'plan': dict(plan),
            'outcome': outcome,
            'report': target_report.to_dict(),
        }
        print()
        print(
            "Transfer probe: "
            f"{plan.get('abstraction_kind')} -> {target_context} "
            f"outcome={outcome.get('outcome')}"
        )
    else:
        print()
        print("Transfer probe: no abstraction transfer plan was available")

    memory_delta = theory_memory.memory_delta_since(starting_memory_summary)
    summary = {
        'run_kind': 'abstraction_transfer_campaign',
        'runs_final': False,
        'outcome_mode': outcome_mode,
        'source_results': source_results,
        'bridge_count': len(bridges),
        'bridges': bridges,
        'selected_plan': dict(plan or {}),
        'transfer_result': transfer_result,
        'abstraction_discovery_evidence': theory_memory.abstraction_discovery_evidence(),
        'readiness': theory_memory.discovery_readiness_report(),
        'rediscovery_goal_progress': theory_memory.rediscovery_goal_progress_report(),
        'starting_memory': dict(starting_memory_summary or {}),
        'ending_memory': theory_memory.memory_checkpoint_summary(),
        'memory_delta': memory_delta,
    }
    print()
    evidence = summary['abstraction_discovery_evidence']
    print(
        "Abstraction transfer evidence: "
        f"bridges={evidence['bridge_count']} "
        f"outcomes={evidence['transfer_outcome_count']} "
        f"confirmed={evidence['transfer_confirmed_count']} "
        f"weak={evidence['transfer_weak_count']} "
        f"absent={evidence['transfer_absent_count']}"
    )
    print(
        "Theory memory delta: "
        f"records={memory_delta['new_records']}, "
        f"planned_outcomes={memory_delta['new_planned_outcomes']}, "
        f"readiness_delta={memory_delta['readiness_score_delta']:.3f}"
    )
    if emit_hf_artifact_summary:
        print(
            "HF_ARTIFACT_SUMMARY "
            + json.dumps(summary, sort_keys=True),
            flush=True,
        )
    return summary


def _hf_upload_requires_create_pr(error: Exception) -> bool:
    message = str(error).lower()
    if '403' in message or 'forbidden' in message:
        return True
    return (
        'create_pr' in message
        and 'pull request' in message
        and ('403' in message or 'forbidden' in message)
    )


def _hf_upload_fallback_reason(error: Exception) -> str:
    message = str(error).lower()
    if 'create_pr' in message or 'pull request' in message:
        return 'create_pr_required'
    if '403' in message or 'forbidden' in message:
        return 'forbidden_retry_create_pr'
    return 'create_pr_retry'


def upload_hf_artifact_file(
    api,
    *,
    path_or_fileobj,
    path_in_repo: str,
    repo_id: str,
    repo_type: str = 'dataset',
    create_pr: bool = False,
    create_pr_on_forbidden: bool = True,
    **kwargs,
) -> dict:
    upload_kwargs = {
        'path_or_fileobj': path_or_fileobj,
        'path_in_repo': path_in_repo,
        'repo_id': repo_id,
        'repo_type': repo_type,
        'create_pr': create_pr,
    }
    upload_kwargs.update(kwargs)
    try:
        uploaded = api.upload_file(**upload_kwargs)
        return {
            'status': 'uploaded_via_pr' if create_pr else 'uploaded',
            'path_in_repo': path_in_repo,
            'repo_id': repo_id,
            'repo_type': repo_type,
            'create_pr': create_pr,
            'url': str(uploaded),
        }
    except Exception as error:
        if not create_pr and create_pr_on_forbidden:
            if _hf_upload_requires_create_pr(error):
                retry_kwargs = dict(upload_kwargs)
                retry_kwargs['create_pr'] = True
                uploaded = api.upload_file(**retry_kwargs)
                return {
                    'status': 'uploaded_via_pr',
                    'path_in_repo': path_in_repo,
                    'repo_id': repo_id,
                    'repo_type': repo_type,
                    'create_pr': True,
                    'fallback_reason': _hf_upload_fallback_reason(error),
                    'url': str(uploaded),
                }
        raise


def _run_equation_followup_cases(
    theory_memory: CumulativeTheoryMemory,
    world_types: list[str],
    object_counts: list[int],
    steps: int,
    num_agents: int,
    limit: int,
    enable_equation_probes: bool = True,
    force_backend: str = 'python',
    equation_scoring_backend: str = 'python',
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
            residual_first=bool(plan.get('residual_first')),
            force_backend=force_backend,
            equation_scoring_backend=equation_scoring_backend,
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
        theory_memory.record_equation_case_result(
            result['context'],
            result['seed'],
            result,
            phase='equation_followup',
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
    return planned_probe_actions(plan, max_actions=max_actions)


def _interesting_result_rank(result: dict) -> tuple[float, float]:
    equation = result.get('interesting_equation') or {}
    role = equation.get('role', '')
    return (
        _interesting_equation_priority(role),
        float(result.get('interesting_score', 0.0)),
    )


def run_math_foundation_prep(
    seeds: int = 1,
    steps: int = 220,
    object_counts: list[int] = None,
    world_types: list[str] = None,
    hidden_worlds: int = 1,
    num_agents: int = 2,
    theory_memory: CumulativeTheoryMemory | None = None,
    force_backend: str = 'python',
    equation_scoring_backend: str = 'python',
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
                    force_backend=force_backend,
                    equation_scoring_backend=equation_scoring_backend,
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
                theory_memory.record_equation_case_result(
                    result['context'],
                    result['seed'],
                    result,
                    phase='math_foundation_prep',
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
            force_backend=force_backend,
            equation_scoring_backend=equation_scoring_backend,
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
        theory_memory.record_equation_case_result(
            result['context'],
            result['seed'],
            result,
            phase='math_foundation_prep',
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
        "--self-authored-worlds 2 "
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


def run_super_system_audit(
    theory_memory: CumulativeTheoryMemory | None = None,
    *,
    output_file: str | Path | None = None,
    latest_artifact: dict | None = None,
    world_types: list[str] | None = None,
    object_counts: list[int] | None = None,
    steps: int = 240,
    limit: int = 5,
) -> dict:
    """Print and optionally persist the connected non-final system audit."""
    theory_memory = theory_memory or CumulativeTheoryMemory()
    report = build_super_system_report(
        theory_memory,
        world_types=world_types or [
            'standard',
            'sideways_wind',
            'vortex',
            'inverse_square_repulsion',
            'localized_gravity',
            'time_varying',
        ],
        object_counts=object_counts or [5],
        steps=steps,
        limit=limit,
        latest_artifact=latest_artifact,
    )
    for line in super_system_summary_lines(report):
        print(line)
    if output_file:
        artifact_path = _write_json_artifact(output_file, report)
        print()
        print(f"Super-system artifact: {artifact_path}")
    return report


def run_rediscovery_goal_progress_audit(
    theory_memory: CumulativeTheoryMemory | None = None,
) -> dict:
    """Print the stricter non-final progress estimate for the 85% goal."""
    theory_memory = theory_memory or CumulativeTheoryMemory()
    report = theory_memory.rediscovery_goal_progress_report()

    print("=" * 70)
    print("REDISCOVERY GOAL PROGRESS")
    print("=" * 70)
    print("Watched final run: not run by this audit")
    print(
        f"Progress: {report['progress_percent']:.1f}% "
        f"target={report['target_percent']:.1f}% "
        f"status={report['status']}"
    )
    print()
    print(f"{'Gate':34s} {'Score':>6s} {'State':>6s} Evidence")
    print("-" * 100)
    for key, gate in report['gates'].items():
        state = 'PASS' if gate['passed'] else 'WORK'
        evidence = ', '.join(
            f"{name}={value}"
            for name, value in gate.get('evidence', {}).items()
        )
        if len(evidence) > 46:
            evidence = evidence[:43] + '...'
        print(
            f"{key:34s} {gate['score']:6.0%} {state:>6s} {evidence}"
        )
    if report['recommended_next_steps']:
        print()
        print("Recommended non-final next steps:")
        for step in report['recommended_next_steps']:
            print(f"  - {step}")
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
    force_backend: str = 'python',
    equation_scoring_backend: str = 'python',
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
        'force_backend': force_backend,
        'equation_scoring_backend': equation_scoring_backend,
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
    arithmetic_report = theory_memory.record_arithmetic_rediscovery(
        seed_start=domain_seed,
        seed_count=max(1, min(2, int(scientist_seed_count or 1))),
        variants=scientist_variants,
    )
    _emit_hf_progress('arithmetic_rediscovery_finish', {
        'status': arithmetic_report.get('status'),
        'coverage': arithmetic_report.get('coverage'),
        'discovered_target_count': arithmetic_report.get('discovered_target_count'),
        'target_count': arithmetic_report.get('target_count'),
        'leaked_manifest': arithmetic_report.get('leaked_manifest'),
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
        requested_world_types=world_types,
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
    targeting_plan = dict(compute_budget_plan.get('targeting_plan') or {})
    prep_world_types = list(
        targeting_plan.get('effective_world_types')
        or world_types
    )
    prep_execution_plan = {
        'requested_world_types': list(world_types),
        'effective_world_types': prep_world_types,
        'object_counts': list(object_counts),
        'requested_steps': steps,
        'requested_seeds': seeds,
        'requested_hidden_worlds': hidden_worlds,
        'targeted': bool(targeting_plan.get('focused')),
        'targeting_reasons': list(targeting_plan.get('reasons') or []),
    }
    if include_prep:
        effective = dict(compute_budget_plan.get('effective') or {})
        effective_steps = int(effective.get('steps', steps) or steps)
        effective_seeds = int(effective.get('seeds', seeds) or seeds)
        effective_hidden_worlds = int(
            effective.get('hidden_worlds', hidden_worlds) or 0
        )
        prep_execution_plan.update({
            'effective_steps': effective_steps,
            'effective_seeds': effective_seeds,
            'effective_hidden_worlds': effective_hidden_worlds,
        })
        prep_execution_plan['estimated_requested_compute_units'] = (
            steps
            * max(1, seeds)
            * max(1, len(world_types) * max(1, len(object_counts)) + hidden_worlds)
        )
        prep_execution_plan['estimated_effective_compute_units'] = (
            effective_steps
            * max(1, effective_seeds)
            * max(
                1,
                len(prep_world_types) * max(1, len(object_counts))
                + effective_hidden_worlds,
            )
        )
        _emit_hf_progress('foundation_prep_start', {
            'world_types': prep_world_types,
            'requested_world_types': world_types,
            'requested_seeds': seeds,
            'requested_steps': steps,
            'requested_hidden_worlds': hidden_worlds,
            'seeds': effective_seeds,
            'steps': effective_steps,
            'hidden_worlds': effective_hidden_worlds,
            'compute_expanded': compute_budget_plan['expanded'],
            'compute_targeted': prep_execution_plan['targeted'],
            'estimated_requested_compute_units': prep_execution_plan[
                'estimated_requested_compute_units'
            ],
            'estimated_effective_compute_units': prep_execution_plan[
                'estimated_effective_compute_units'
            ],
        })
        prep_results = prep_runner(
            seeds=effective_seeds,
            steps=effective_steps,
            object_counts=object_counts,
            world_types=prep_world_types,
            hidden_worlds=effective_hidden_worlds,
            num_agents=num_agents,
            theory_memory=theory_memory,
            force_backend=force_backend,
            equation_scoring_backend=equation_scoring_backend,
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
        'prep_execution_plan': prep_execution_plan,
        'compaction_events': compaction_events,
        'resource_efficiency': resource_efficiency,
        'prep_results': prep_results,
        'arithmetic_rediscovery_report': arithmetic_report,
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


def run_hf_adaptive_comparison(
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
    max_adaptive_steps: int | None = None,
    max_adaptive_seeds: int | None = None,
    max_adaptive_hidden_worlds: int | None = None,
    force_backend: str = 'python',
    equation_scoring_backend: str = 'python',
    prep_runner=None,
) -> dict:
    """Run fixed and adaptive non-final campaigns from the same memory snapshot."""
    base_memory = theory_memory or CumulativeTheoryMemory()
    base_snapshot = base_memory.to_dict()
    object_counts = object_counts or [3]
    world_types = world_types or ['standard']
    domain_variants = domain_variants or [domain_variant]
    scientist_variants = scientist_variants or domain_variants

    _emit_hf_progress('adaptive_comparison_start', {
        'runs_final': False,
        'world_types': world_types,
        'seeds': seeds,
        'steps': steps,
        'hidden_worlds': hidden_worlds,
        'domain_seed_count': domain_seed_count,
        'domain_variants': domain_variants,
        'scientist_seed_count': scientist_seed_count,
        'scientist_variants': scientist_variants,
        'force_backend': force_backend,
        'equation_scoring_backend': equation_scoring_backend,
    })
    variants = []
    for name, adaptive_enabled in (
        ('fixed_budget', False),
        ('adaptive_budget', True),
    ):
        variant_memory = CumulativeTheoryMemory.from_dict(base_snapshot)
        _emit_hf_progress('adaptive_comparison_variant_start', {
            'variant': name,
            'adaptive_compute': adaptive_enabled,
        })
        started = time.perf_counter()
        result = run_hf_non_final_campaign(
            theory_memory=variant_memory,
            seeds=seeds,
            steps=steps,
            object_counts=object_counts,
            world_types=world_types,
            hidden_worlds=hidden_worlds,
            num_agents=num_agents,
            output_file=None,
            domain_seed=domain_seed,
            domain_variant=domain_variant,
            domain_seed_count=domain_seed_count,
            domain_variants=domain_variants,
            scientist_seed_count=scientist_seed_count,
            scientist_variants=scientist_variants,
            live_scientist=live_scientist,
            include_prep=include_prep,
            auto_compact=auto_compact,
            compact_keep_records=compact_keep_records,
            compact_keep_operator_outcomes=compact_keep_operator_outcomes,
            adaptive_compute=adaptive_enabled,
            max_adaptive_steps=max_adaptive_steps,
            max_adaptive_seeds=max_adaptive_seeds,
            max_adaptive_hidden_worlds=max_adaptive_hidden_worlds,
            force_backend=force_backend,
            equation_scoring_backend=equation_scoring_backend,
            prep_runner=prep_runner,
        )
        elapsed_seconds = time.perf_counter() - started
        telemetry = _hf_campaign_telemetry(
            name,
            result,
            elapsed_seconds=elapsed_seconds,
        )
        variants.append({
            'variant': name,
            'adaptive_compute': adaptive_enabled,
            'telemetry': telemetry,
            'result': result,
        })
        _emit_hf_progress('adaptive_comparison_variant_finish', telemetry)

    comparison = _compare_hf_variant_telemetry(
        fixed=variants[0]['telemetry'],
        adaptive=variants[1]['telemetry'],
    )
    report = {
        'run_kind': 'hf_adaptive_comparison',
        'runs_final': False,
        'variant_count': len(variants),
        'comparison': comparison,
        'variants': variants,
        'starting_memory': {
            'records': len(base_memory.records),
            'operator_prior_outcomes': len(base_memory.operator_prior_outcomes),
            'domain_world_records': len(base_memory.domain_world_records),
            'autonomous_scientist_records': len(base_memory.autonomous_scientist_records),
            'arithmetic_rediscovery_records': len(
                base_memory.arithmetic_rediscovery_records
            ),
            'compressed_experience_shards': len(base_memory.compressed_experience_shards),
            'canonical_law_shards': len(base_memory.canonical_law_shards),
        },
    }
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', encoding='utf-8') as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        _emit_hf_progress('artifact_written', {
            'output_file': str(output_path),
        })
    _emit_hf_progress('adaptive_comparison_finish', comparison)
    return report


def _hf_campaign_telemetry(
    variant: str,
    result: dict,
    *,
    elapsed_seconds: float,
) -> dict:
    readiness = dict(result.get('readiness') or {})
    plan = dict(result.get('compute_budget_plan') or {})
    requested = dict(plan.get('requested') or {})
    effective = dict(plan.get('effective') or requested)
    prep_execution_plan = dict(result.get('prep_execution_plan') or {})
    resource = dict(result.get('resource_efficiency') or {})
    canonical = dict(resource.get('canonical_law_compression') or {})
    arithmetic = dict(result.get('arithmetic_rediscovery_report') or {})
    memory_bytes = estimate_json_bytes(result.get('theory_memory') or {})
    artifact_bytes = estimate_json_bytes(result)
    compute_units = _hf_compute_units(
        effective,
        prep_results=list(result.get('prep_results') or []),
        prep_execution_plan=prep_execution_plan,
    )
    requested_compute_units = _hf_requested_compute_units(
        requested,
        prep_execution_plan=prep_execution_plan,
    )
    readiness_score = float(readiness.get('readiness_score', 0.0) or 0.0)
    elapsed_seconds = max(0.001, float(elapsed_seconds or 0.001))
    return {
        'variant': variant,
        'elapsed_seconds': round(elapsed_seconds, 3),
        'readiness_score': round(readiness_score, 3),
        'readiness_status': readiness.get('status'),
        'passed_gate_count': readiness.get('passed_gate_count', 0),
        'gate_count': readiness.get('gate_count', 0),
        'prep_result_count': result.get('prep_result_count', 0),
        'prep_ready_count': sum(
            1 for item in result.get('prep_results') or []
            if item.get('ready_for_final')
        ),
        'compute_requested': requested,
        'compute_effective': effective,
        'compute_units': compute_units,
        'requested_compute_units': requested_compute_units,
        'compute_unit_savings': max(0, requested_compute_units - compute_units),
        'compute_unit_savings_ratio': round(
            max(0, requested_compute_units - compute_units)
            / max(1, requested_compute_units),
            3,
        ),
        'compute_expanded': bool(plan.get('expanded')),
        'compute_targeted': bool(plan.get('targeted')),
        'prep_execution_plan': prep_execution_plan,
        'residual_pressure_score': (
            dict(plan.get('residual_pressure') or {}).get('score', 0.0)
        ),
        'artifact_bytes': artifact_bytes,
        'memory_bytes': memory_bytes,
        'memory_kb': round(memory_bytes / 1024, 3),
        'readiness_per_second': round(readiness_score / elapsed_seconds, 5),
        'readiness_per_compute_unit': round(
            readiness_score / max(1, compute_units),
            7,
        ),
        'readiness_per_memory_kb': round(
            readiness_score / max(0.001, memory_bytes / 1024),
            5,
        ),
        'compressed_shards': resource.get('compressed_shard_count', 0),
        'long_run_ready': resource.get('long_run_ready', False),
        'canonical_law_shards': canonical.get('canonical_law_shard_count', 0),
        'canonical_law_count': canonical.get('canonical_law_count', 0),
        'canonical_law_ready': canonical.get('long_run_law_ready', False),
        'arithmetic_coverage': arithmetic.get('coverage', 0.0),
        'arithmetic_status': arithmetic.get('status'),
    }


def _compare_hf_variant_telemetry(
    *,
    fixed: dict,
    adaptive: dict,
) -> dict:
    readiness_delta = round(
        float(adaptive.get('readiness_score', 0.0) or 0.0)
        - float(fixed.get('readiness_score', 0.0) or 0.0),
        3,
    )
    compute_ratio = _safe_float_ratio(
        adaptive.get('compute_units', 0),
        fixed.get('compute_units', 0),
    )
    memory_ratio = _safe_float_ratio(
        adaptive.get('memory_bytes', 0),
        fixed.get('memory_bytes', 0),
    )
    readiness_per_compute_delta = round(
        float(adaptive.get('readiness_per_compute_unit', 0.0) or 0.0)
        - float(fixed.get('readiness_per_compute_unit', 0.0) or 0.0),
        7,
    )
    adaptive_improved_efficiency = readiness_per_compute_delta > 0
    adaptive_kept_quality = (
        float(adaptive.get('readiness_score', 0.0) or 0.0)
        >= float(fixed.get('readiness_score', 0.0) or 0.0)
        and int(adaptive.get('passed_gate_count', 0) or 0)
        >= int(fixed.get('passed_gate_count', 0) or 0)
    )
    adaptive_kept_compression = (
        bool(adaptive.get('canonical_law_ready'))
        and bool(adaptive.get('long_run_ready'))
    )
    if adaptive_kept_quality and adaptive_kept_compression and adaptive_improved_efficiency:
        recommendation = 'keep_efficiency_targeted_adaptive_budgeting'
    elif adaptive_kept_quality and adaptive_improved_efficiency:
        recommendation = 'keep_targeted_adaptive_and_compact_memory_next'
    elif adaptive_kept_quality and adaptive_kept_compression:
        recommendation = 'keep_adaptive_for_quality_but_improve_efficiency'
    else:
        recommendation = 'prefer_fixed_until_adaptive_quality_recovers'
    return {
        'fixed_variant': fixed['variant'],
        'adaptive_variant': adaptive['variant'],
        'readiness_delta': readiness_delta,
        'adaptive_kept_or_improved_quality': adaptive_kept_quality,
        'adaptive_kept_compression_ready': adaptive_kept_compression,
        'adaptive_improved_readiness_per_compute': adaptive_improved_efficiency,
        'compute_unit_ratio_adaptive_to_fixed': compute_ratio,
        'memory_byte_ratio_adaptive_to_fixed': memory_ratio,
        'readiness_per_compute_unit_delta': readiness_per_compute_delta,
        'fixed_compute_units': fixed.get('compute_units', 0),
        'adaptive_compute_units': adaptive.get('compute_units', 0),
        'fixed_requested_compute_units': fixed.get('requested_compute_units', 0),
        'adaptive_requested_compute_units': adaptive.get('requested_compute_units', 0),
        'adaptive_compute_unit_savings': adaptive.get('compute_unit_savings', 0),
        'adaptive_compute_unit_savings_ratio': adaptive.get(
            'compute_unit_savings_ratio',
            0.0,
        ),
        'fixed_memory_bytes': fixed.get('memory_bytes', 0),
        'adaptive_memory_bytes': adaptive.get('memory_bytes', 0),
        'fixed_readiness': fixed.get('readiness_score', 0.0),
        'adaptive_readiness': adaptive.get('readiness_score', 0.0),
        'recommendation': recommendation,
    }


def _hf_compute_units(
    effective: dict,
    *,
    prep_results: list[dict] | None = None,
    prep_execution_plan: dict | None = None,
) -> int:
    if prep_results:
        return sum(
            max(1, int(result.get('steps', effective.get('steps', 1)) or 1))
            for result in prep_results
        )
    prep_execution_plan = prep_execution_plan or {}
    steps = max(1, int(effective.get('steps', 1) or 1))
    seeds = max(1, int(effective.get('seeds', 1) or 1))
    hidden_worlds = max(0, int(effective.get('hidden_worlds', 0) or 0))
    world_types = list(prep_execution_plan.get('effective_world_types') or [])
    object_counts = list(prep_execution_plan.get('object_counts') or [1])
    case_count = len(world_types) * max(1, len(object_counts)) + hidden_worlds
    if case_count <= 0:
        case_count = hidden_worlds + 1
    return steps * seeds * max(1, case_count)


def _hf_requested_compute_units(
    requested: dict,
    *,
    prep_execution_plan: dict | None = None,
) -> int:
    prep_execution_plan = prep_execution_plan or {}
    steps = max(1, int(requested.get('steps', 1) or 1))
    seeds = max(1, int(requested.get('seeds', 1) or 1))
    hidden_worlds = max(
        0,
        int(
            prep_execution_plan.get(
                'requested_hidden_worlds',
                requested.get('hidden_worlds', 0),
            )
            or 0
        ),
    )
    world_types = list(prep_execution_plan.get('requested_world_types') or [])
    object_counts = list(prep_execution_plan.get('object_counts') or [1])
    case_count = len(world_types) * max(1, len(object_counts)) + hidden_worlds
    if case_count <= 0:
        case_count = hidden_worlds + 1
    return steps * seeds * max(1, case_count)


def _safe_float_ratio(numerator, denominator) -> float:
    try:
        denominator = float(denominator)
        if denominator <= 0:
            return 0.0
        return round(float(numerator) / denominator, 3)
    except (TypeError, ValueError):
        return 0.0


def _emit_hf_progress(event: str, payload: dict):
    print(
        "HF_PROGRESS "
        + json.dumps({'event': event, **payload}, sort_keys=True),
        flush=True,
    )


def parse_live_progress_line(line: str) -> dict | None:
    """Parse one live progress line from a local or HF run log."""
    text = str(line).strip()
    prefixes = {
        'HF_PROGRESS ': 'hf_progress',
        'SCIENTIST_EVENT ': 'scientist_event',
        'HF_ARTIFACT ': 'hf_artifact',
        'HF_ARTIFACT_SUMMARY ': 'hf_artifact_summary',
        'HF_ARTIFACT_CHUNK ': 'hf_artifact_chunk',
    }
    for prefix, stream in prefixes.items():
        if not text.startswith(prefix):
            continue
        try:
            payload = json.loads(text[len(prefix):])
        except json.JSONDecodeError:
            return {
                'stream': stream,
                'event': 'parse_error',
                'raw': text,
            }
        if not isinstance(payload, dict):
            payload = {'payload': payload}
        payload = dict(payload)
        payload.setdefault('event', stream)
        payload['stream'] = stream
        return payload
    return None


def run_live_progress_viewer(
    progress_file: str | None = None,
    *,
    follow: bool = False,
    max_events: int = 0,
    poll_seconds: float = 1.0,
) -> dict:
    """Print a compact live view of HF/scientist progress events."""
    counts = {
        'hf_progress': 0,
        'scientist_event': 0,
        'hf_artifact': 0,
        'hf_artifact_summary': 0,
        'hf_artifact_chunk': 0,
        'parse_error': 0,
    }
    last_hf = {}
    last_scientist = {}
    artifacts = {}
    artifact_summary = {}
    printed = 0

    print("=" * 70, flush=True)
    print("LIVE DISCOVERY PROGRESS VIEW", flush=True)
    print("=" * 70, flush=True)

    def consume_line(line: str) -> bool:
        nonlocal printed, last_hf, last_scientist, artifacts, artifact_summary
        event = parse_live_progress_line(line)
        if event is None:
            return True
        if event.get('event') == 'parse_error':
            counts['parse_error'] += 1
            print("PARSE_ERROR unable to decode progress JSON", flush=True)
        elif event['stream'] == 'hf_progress':
            counts['hf_progress'] += 1
            last_hf = event
            print(_format_hf_progress_event(event), flush=True)
        elif event['stream'] == 'scientist_event':
            counts['scientist_event'] += 1
            last_scientist = event
            print(_format_scientist_event(event), flush=True)
        elif event['stream'] == 'hf_artifact':
            counts['hf_artifact'] += 1
            artifacts = event
            print(_format_hf_artifact_event(event), flush=True)
        elif event['stream'] == 'hf_artifact_summary':
            counts['hf_artifact_summary'] += 1
            artifact_summary = event
            print(_format_hf_artifact_summary_event(event), flush=True)
        elif event['stream'] == 'hf_artifact_chunk':
            counts['hf_artifact_chunk'] += 1
            print(_format_hf_artifact_chunk_event(event), flush=True)
        printed += 1
        return max_events <= 0 or printed < max_events

    if progress_file:
        path = Path(progress_file)
        with path.open('r', encoding='utf-8') as handle:
            for line in handle:
                if not consume_line(line):
                    break
            if follow and (max_events <= 0 or printed < max_events):
                while True:
                    line = handle.readline()
                    if line:
                        if not consume_line(line):
                            break
                    else:
                        time.sleep(max(0.1, poll_seconds))
    else:
        for line in sys.stdin:
            if not consume_line(line):
                break

    summary = {
        'counts': counts,
        'last_hf_event': last_hf,
        'last_scientist_event': last_scientist,
        'artifacts': artifacts,
        'artifact_summary': artifact_summary,
    }
    print("-" * 70, flush=True)
    print(
        "Summary: "
        f"hf={counts['hf_progress']} "
        f"scientist={counts['scientist_event']} "
        f"artifacts={counts['hf_artifact']} "
        f"artifact_summaries={counts['hf_artifact_summary']} "
        f"artifact_chunks={counts['hf_artifact_chunk']} "
        f"parse_errors={counts['parse_error']}",
        flush=True,
    )
    return summary


def _format_hf_progress_event(event: dict) -> str:
    name = str(event.get('event', 'unknown'))
    pieces = [f"HF {name}"]
    for key in (
        'variant',
        'readiness_score',
        'readiness_status',
        'passed_gate_count',
        'gate_count',
        'compute_units',
        'compute_targeted',
        'long_run_ready',
        'recommendation',
    ):
        if key in event:
            pieces.append(f"{key}={event[key]}")
    return ' | '.join(pieces)


def _format_scientist_event(event: dict) -> str:
    name = str(event.get('event', 'unknown'))
    pieces = [f"SCIENTIST {name}"]
    for key in (
        'domain_key',
        'relation_kind',
        'status',
        'priority',
        'expression',
        'next_experiment',
    ):
        if key in event:
            pieces.append(f"{key}={event[key]}")
    return ' | '.join(pieces)


def _format_hf_artifact_event(event: dict) -> str:
    pieces = ["HF_ARTIFACT"]
    for key in ('run_id', 'summary', 'report', 'memory', 'log'):
        if key in event:
            pieces.append(f"{key}={event[key]}")
    return ' | '.join(pieces)


def _format_hf_artifact_summary_event(event: dict) -> str:
    pieces = ["HF_ARTIFACT_SUMMARY"]
    for key in (
        'run_kind',
        'result_count',
        'passed_count',
        'runs_final',
    ):
        if key in event:
            pieces.append(f"{key}={event[key]}")
    readiness = dict(event.get('readiness') or {})
    if readiness:
        pieces.append(f"readiness_score={readiness.get('readiness_score')}")
        pieces.append(f"readiness_status={readiness.get('status')}")
    memory_delta = dict(event.get('memory_delta') or {})
    if memory_delta:
        pieces.append(f"new_equation_cases={memory_delta.get('new_equation_cases')}")
        pieces.append(f"new_planned_outcomes={memory_delta.get('new_planned_outcomes')}")
    return ' | '.join(pieces)


def _format_hf_artifact_chunk_event(event: dict) -> str:
    return (
        "HF_ARTIFACT_CHUNK "
        f"run_id={event.get('run_id')} "
        f"part={event.get('index')}/{event.get('total')} "
        f"encoding={event.get('encoding')}"
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
        canonical = theory_memory.compact_canonical_laws(
            source=f'hf_batch:{phase}',
        )
        after = theory_memory.compact_experience(
            keep_recent_records=keep_recent_records,
            keep_recent_operator_outcomes=keep_recent_operator_outcomes,
            source=f'hf_batch:{phase}',
            force_summary=True,
        )
    else:
        after = before
        canonical = theory_memory.canonical_law_compression_report()
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
        'canonical_law_shards': canonical['canonical_law_shard_count'],
        'canonical_law_count': canonical['canonical_law_count'],
        'canonical_law_ready': canonical['long_run_law_ready'],
    }
    _emit_hf_progress('memory_compaction_checkpoint', event)
    return event


def run_math_final_discovery(
    seeds: int = 1,
    steps: int = 600,
    object_counts: list[int] = None,
    world_types: list[str] = None,
    hidden_worlds: int = 3,
    hidden_world_start: int = 0,
    self_authored_worlds: int = 0,
    num_agents: int = 2,
    section_study_cycles: int = 1,
    theory_memory: CumulativeTheoryMemory | None = None,
    theory_memory_checkpoint_file: str | Path | None = None,
    artifact_output_file: str | Path | None = None,
    hf_output_repo: str | None = None,
    run_id: str | None = None,
    parallel_cases: int = 1,
    profile_final_run: bool = False,
    force_backend: str = 'python',
    equation_scoring_backend: str = 'python',
) -> list[dict]:
    """Run the watched discovery campaign and report live performance metrics."""
    final_started = time.perf_counter()
    starting_memory_summary = (
        theory_memory.memory_checkpoint_summary()
        if theory_memory is not None
        else CumulativeTheoryMemory().memory_checkpoint_summary()
    )
    object_counts = object_counts or [5]
    world_types = WORLD_TYPES if world_types is None else list(world_types)
    hidden_worlds = max(0, int(hidden_worlds or 0))
    hidden_world_start = max(0, int(hidden_world_start or 0))
    self_authored_worlds = max(0, int(self_authored_worlds or 0))
    section_study_cycles = max(1, int(section_study_cycles or 1))
    parallel_cases = max(1, int(parallel_cases or 1))
    results = []
    theory_memory = theory_memory or CumulativeTheoryMemory()
    runtime_profile_events: list[dict] | None = [] if profile_final_run else None

    print("=" * 70, flush=True)
    print("FINAL WATCHED MATH DISCOVERY CAMPAIGN", flush=True)
    print("=" * 70, flush=True)
    print(f"Worlds: {', '.join(world_types) if world_types else '(none)'}", flush=True)
    print(f"Hidden generated worlds: {hidden_worlds}", flush=True)
    print(f"Hidden generated world start: {hidden_world_start}", flush=True)
    print(f"Self-authored hidden worlds: {self_authored_worlds}", flush=True)
    print(f"Seeds: 0..{seeds - 1}", flush=True)
    print(f"Object counts: {', '.join(str(count) for count in object_counts)}", flush=True)
    print(f"Steps per run: {steps}", flush=True)
    print(f"Section study cycles: {section_study_cycles}", flush=True)
    print(f"Parallel case workers: {parallel_cases}", flush=True)
    print(f"Force backend: {force_backend}", flush=True)
    print(f"Equation scoring backend: {equation_scoring_backend}", flush=True)
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
        section_results = []
        for cycle in range(section_study_cycles):
            if section_study_cycles > 1:
                print(
                    f"Section study cycle {cycle + 1}/{section_study_cycles}: "
                    f"{world_type}",
                    flush=True,
                )
            cycle_results = _run_math_final_section_cycle(
                context=world_type,
                theory_memory=theory_memory,
                object_counts=object_counts,
                steps=steps,
                seeds=seeds,
                cycle=cycle,
                num_agents=num_agents,
                world_type=world_type,
                parallel_cases=parallel_cases,
                runtime_profile_events=runtime_profile_events,
                force_backend=force_backend,
                equation_scoring_backend=equation_scoring_backend,
            )
            results.extend(cycle_results)
            section_results.extend(cycle_results)
            if section_study_cycles > 1:
                _print_section_study_summary(
                    world_type,
                    section_results,
                    theory_memory,
                    object_counts=object_counts,
                    steps=steps,
                    cycle=cycle + 1,
                    total_cycles=section_study_cycles,
                )
            _checkpoint_theory_memory(
                theory_memory,
                theory_memory_checkpoint_file,
                label=f"{world_type} cycle {cycle + 1}/{section_study_cycles}",
            )

    for offset in range(hidden_worlds):
        index = hidden_world_start + offset
        manifest = generate_hidden_world_manifest(index, variant=index)
        _run_hidden_manifest_final_section(
            manifest,
            results=results,
            theory_memory=theory_memory,
            object_counts=object_counts,
            steps=steps,
            seeds=seeds,
            num_agents=num_agents,
            section_study_cycles=section_study_cycles,
            manifest_source='generated',
            theory_memory_checkpoint_file=theory_memory_checkpoint_file,
            parallel_cases=parallel_cases,
            runtime_profile_events=runtime_profile_events,
            force_backend=force_backend,
            equation_scoring_backend=equation_scoring_backend,
        )

    if self_authored_worlds:
        authored_designs = theory_memory.autonomous_experiment_design_agenda(
            limit=self_authored_worlds,
        )
        if not authored_designs:
            print(
                "Self-authored hidden worlds requested, but no autonomous "
                "designs are available yet.",
                flush=True,
            )
        for index, design in enumerate(authored_designs):
            manifest = generate_self_authored_hidden_world_manifest(
                design,
                seed=index,
                variant=index,
            )
            if _hidden_manifest_observation_leaks(
                manifest,
                seed=0,
                object_count=object_counts[0],
            ):
                print(
                    f"Skipping self-authored hidden world {manifest.hidden_id}: "
                    "manifest leaked through observation",
                    flush=True,
                )
                continue
            question = str(
                design.get('question')
                or design.get('reason')
                or design.get('design_key')
                or 'autonomous_design'
            )
            if len(question) > 72:
                question = question[:69] + '...'
            print(
                f"Self-authored hidden world: {manifest.hidden_id} "
                f"source={design.get('source', 'unknown')} question={question}",
                flush=True,
            )
            _run_hidden_manifest_final_section(
                manifest,
                results=results,
                theory_memory=theory_memory,
                object_counts=object_counts,
                steps=steps,
                seeds=seeds,
                num_agents=num_agents,
                section_study_cycles=section_study_cycles,
                manifest_source='self_authored',
                self_authored_world_design=design,
                theory_memory_checkpoint_file=theory_memory_checkpoint_file,
                parallel_cases=parallel_cases,
                runtime_profile_events=runtime_profile_events,
                force_backend=force_backend,
                equation_scoring_backend=equation_scoring_backend,
            )

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
    if (
        artifact_output_file
        or hf_output_repo
        or os.environ.get('HF_OUTPUT_REPO')
        or os.environ.get('HF_RUN_REPO')
    ):
        _persist_math_final_artifact(
            results,
            theory_memory,
            artifact_output_file=artifact_output_file,
            hf_output_repo=hf_output_repo,
            run_id=run_id,
            run_config={
                'seeds': seeds,
                'steps': steps,
                'object_counts': list(object_counts),
                'world_types': list(world_types),
                'hidden_worlds': hidden_worlds,
                'hidden_world_start': hidden_world_start,
                'self_authored_worlds': self_authored_worlds,
                'num_agents': num_agents,
                'section_study_cycles': section_study_cycles,
                'parallel_cases': parallel_cases,
                'profile_final_run': bool(profile_final_run),
            },
            starting_memory_summary=starting_memory_summary,
            runtime_profile_events=(
                [
                    *(runtime_profile_events or []),
                    {
                        'event': 'final_run_profile',
                        'elapsed_seconds': round(
                            time.perf_counter() - final_started,
                            3,
                        ),
                        'result_count': len(results),
                    },
                ]
                if profile_final_run
                else None
            ),
        )
    return results


def _run_hidden_manifest_final_section(
    manifest: HiddenWorldManifest,
    *,
    results: list[dict],
    theory_memory: CumulativeTheoryMemory,
    object_counts: list[int],
    steps: int,
    seeds: int,
    num_agents: int,
    section_study_cycles: int,
    manifest_source: str,
    self_authored_world_design: dict | None = None,
    theory_memory_checkpoint_file: str | Path | None = None,
    parallel_cases: int = 1,
    runtime_profile_events: list[dict] | None = None,
    force_backend: str = 'python',
    equation_scoring_backend: str = 'python',
):
    section_results = []
    for cycle in range(section_study_cycles):
        if section_study_cycles > 1:
            print(
                f"Section study cycle {cycle + 1}/{section_study_cycles}: "
                f"{manifest.hidden_id}",
                flush=True,
            )
        cycle_results = _run_math_final_section_cycle(
            context=manifest.hidden_id,
            theory_memory=theory_memory,
            object_counts=object_counts,
            steps=steps,
            seeds=seeds,
            cycle=cycle,
            num_agents=num_agents,
            world_type='hidden_procedural',
            hidden_manifest=manifest,
            result_metadata={
                'manifest': manifest.to_dict(),
                'manifest_source': manifest_source,
                **(
                    {'self_authored_world_design': dict(self_authored_world_design)}
                    if self_authored_world_design is not None
                    else {}
                ),
            },
            parallel_cases=parallel_cases,
            runtime_profile_events=runtime_profile_events,
            force_backend=force_backend,
            equation_scoring_backend=equation_scoring_backend,
        )
        results.extend(cycle_results)
        section_results.extend(cycle_results)
        if section_study_cycles > 1:
            _print_section_study_summary(
                manifest.hidden_id,
                section_results,
                theory_memory,
                object_counts=object_counts,
                steps=steps,
                cycle=cycle + 1,
                total_cycles=section_study_cycles,
            )
        _checkpoint_theory_memory(
            theory_memory,
            theory_memory_checkpoint_file,
            label=f"{manifest.hidden_id} cycle {cycle + 1}/{section_study_cycles}",
        )


def _execute_math_final_case_payload(payload: dict) -> dict:
    started = time.perf_counter()
    result = _run_math_final_discovery_case(**payload)
    result['case_elapsed_seconds'] = round(time.perf_counter() - started, 3)
    return result


def _run_math_final_section_cycle(
    *,
    context: str,
    theory_memory: CumulativeTheoryMemory,
    object_counts: list[int],
    steps: int,
    seeds: int,
    cycle: int,
    num_agents: int,
    world_type: str,
    hidden_manifest: HiddenWorldManifest | None = None,
    result_metadata: dict | None = None,
    parallel_cases: int = 1,
    runtime_profile_events: list[dict] | None = None,
    force_backend: str = 'python',
    equation_scoring_backend: str = 'python',
) -> list[dict]:
    cycle_started = time.perf_counter()
    descriptors = []
    for index, case in enumerate(
        _section_cycle_cases(
            theory_memory,
            context,
            object_counts=object_counts,
            steps=steps,
            seeds=seeds,
            cycle=cycle,
        )
    ):
        plan = case.get('plan')
        actual_seed = int(case['seed'])
        object_count = int(case['object_count'])
        case_steps = int(case.get('steps', steps) or steps)
        plan_text = f" plan={plan['experiment_kind']}" if plan else ''
        print(
            f"Running final case: {context} seed={actual_seed} "
            f"objects={object_count} steps={case_steps}{plan_text}",
            flush=True,
        )
        descriptors.append({
            'index': index,
            'plan': dict(plan) if plan else None,
            'payload': {
                'context': context,
                'seed': actual_seed,
                'object_count': object_count,
                'steps': case_steps,
                'num_agents': num_agents,
                'world_type': world_type,
                'hidden_manifest': hidden_manifest,
                'planned_actions': _planned_probe_actions(plan) if plan else None,
                'residual_first': (
                    bool(plan and plan.get('residual_first'))
                    or _should_force_residual_first(theory_memory, context)
                ),
                'equation_operator_priors': theory_memory.generated_operator_priors(
                    context=context,
                ),
                'force_backend': force_backend,
                'equation_scoring_backend': equation_scoring_backend,
            },
        })

    worker_count = min(max(1, int(parallel_cases or 1)), len(descriptors) or 1)
    if worker_count <= 1:
        completed = [
            _finalize_math_final_case_result(
                _execute_math_final_case_payload(descriptor['payload']),
                descriptor,
                theory_memory,
                result_metadata,
            )
            for descriptor in descriptors
        ]
        _record_section_cycle_profile(
            runtime_profile_events,
            context=context,
            cycle=cycle,
            worker_count=worker_count,
            completed=completed,
            started=cycle_started,
        )
        return completed

    completed = []
    pending: dict[int, dict] = {}
    next_index = 0
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=worker_count,
    ) as executor:
        futures = {
            executor.submit(
                _execute_math_final_case_payload,
                descriptor['payload'],
            ): descriptor['index']
            for descriptor in descriptors
        }
        for future in concurrent.futures.as_completed(futures):
            index = futures[future]
            pending[index] = future.result()
            while next_index in pending:
                result = pending.pop(next_index)
                completed.append(
                    _finalize_math_final_case_result(
                        result,
                        descriptors[next_index],
                        theory_memory,
                        result_metadata,
                    )
                )
                next_index += 1
    _record_section_cycle_profile(
        runtime_profile_events,
        context=context,
        cycle=cycle,
        worker_count=worker_count,
        completed=completed,
        started=cycle_started,
    )
    return completed


def _record_section_cycle_profile(
    runtime_profile_events: list[dict] | None,
    *,
    context: str,
    cycle: int,
    worker_count: int,
    completed: list[dict],
    started: float,
):
    if runtime_profile_events is None:
        return
    elapsed = time.perf_counter() - started
    case_seconds = [
        float(result.get('case_elapsed_seconds', 0.0) or 0.0)
        for result in completed
    ]
    event = {
        'event': 'section_cycle_profile',
        'context': context,
        'cycle': cycle + 1,
        'worker_count': worker_count,
        'case_count': len(completed),
        'elapsed_seconds': round(elapsed, 3),
        'case_seconds_sum': round(sum(case_seconds), 3),
        'case_seconds_max': round(max(case_seconds) if case_seconds else 0.0, 3),
        'parallel_efficiency_estimate': round(
            sum(case_seconds) / max(0.001, elapsed * max(1, worker_count)),
            3,
        ),
    }
    runtime_profile_events.append(event)
    print(
        "PROFILE_EVENT " + json.dumps(event, sort_keys=True),
        flush=True,
    )


def _finalize_math_final_case_result(
    result: dict,
    descriptor: dict,
    theory_memory: CumulativeTheoryMemory,
    result_metadata: dict | None,
) -> dict:
    finalized = dict(result)
    if result_metadata:
        finalized.update(result_metadata)
    plan = descriptor.get('plan')
    if plan:
        finalized['phase'] = 'math_final_discovery_followup'
        finalized['planned_experiment'] = dict(plan)
        finalized['planned_experiment_outcome'] = (
            theory_memory.record_planned_result(
                plan,
                context=finalized['context'],
                seed=finalized['seed'],
                report=finalized.get('discovery_loop', {}),
                operator_prior_result=finalized,
            )
        )
    else:
        theory_memory.record_result(
            finalized['context'],
            finalized['seed'],
            finalized.get('discovery_loop', {}),
        )
        theory_memory.record_operator_prior_results(
            finalized['context'],
            finalized['seed'],
            finalized,
        )
    theory_memory.record_equation_case_result(
        finalized['context'],
        finalized['seed'],
        finalized,
        phase=(
            'math_final_discovery_followup'
            if plan
            else 'math_final_discovery'
        ),
    )
    _print_math_final_discovery_row(finalized)
    return finalized


def _checkpoint_theory_memory(
    theory_memory: CumulativeTheoryMemory,
    checkpoint_file: str | Path | None,
    *,
    label: str,
):
    if not checkpoint_file:
        return
    theory_memory.save(checkpoint_file)
    print(
        f"Theory memory checkpoint saved: {checkpoint_file} after {label}",
        flush=True,
    )


def _hidden_manifest_observation_leaks(
    manifest: HiddenWorldManifest,
    *,
    seed: int,
    object_count: int,
) -> bool:
    observation = Environment(
        num_initial_objects=object_count,
        seed=seed,
        world_type='hidden_procedural',
        hidden_manifest=manifest,
    ).observe()
    return hidden_manifest_from_observation(observation)


def run_math_benchmark(
    seeds: int = 3,
    steps: int = 160,
    object_counts: list[int] = None,
    world_types: list[str] = None,
    num_agents: int = 2,
    required_concepts: set[str] | None = None,
    force_backend: str = 'python',
    equation_scoring_backend: str = 'python',
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
                        force_backend=force_backend,
                        equation_scoring_backend=equation_scoring_backend,
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
    force_backend: str = 'python',
    equation_scoring_backend: str = 'python',
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
            force_backend=force_backend,
            equation_scoring_backend=equation_scoring_backend,
        )

    observation_probe = Environment(
        num_initial_objects=1,
        seed=seed,
        world_type='hidden_procedural',
        hidden_manifest=manifest,
        force_backend=force_backend,
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
        'force_backend': force_backend,
        'equation_scoring_backend': equation_scoring_backend,
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
    residual_first: bool = False,
    force_backend: str = 'python',
    equation_scoring_backend: str = 'python',
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
            force_backend=force_backend,
            equation_scoring_backend=equation_scoring_backend,
        )
    metrics = _equation_metrics_from_knowledge(kb, residual_first=residual_first)
    return {
        'context': context,
        'seed': seed,
        'objects': object_count,
        'steps': steps,
        'force_backend': force_backend,
        'equation_scoring_backend': equation_scoring_backend,
        **metrics,
    }


def _equation_metrics_from_knowledge(
    kb: KnowledgeBase,
    *,
    residual_first: bool = False,
) -> dict:
    return _equation_metrics_from_pack(
        getattr(kb, 'equation_workbench', None),
        residual_first=residual_first,
    )


def _equation_metrics_from_pack(
    workbench,
    *,
    residual_first: bool = False,
) -> dict:
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
    interesting_equation = _select_interesting_equation(
        pack,
        residual_first=residual_first,
    ) or top_equation
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


def _select_interesting_equation(
    pack: dict,
    *,
    residual_first: bool = False,
) -> dict:
    top_equation = (
        pack.get('top_equations', [{}])[0]
        if pack.get('top_equations')
        else {}
    )
    default = (
        pack.get('interesting_equations', [{}])[0]
        if pack.get('interesting_equations')
        else top_equation
    )
    if not residual_first:
        return dict(default or {})
    residual = _best_residual_first_equation(pack)
    if not residual:
        return dict(default or {})
    default_role = str((default or {}).get('role') or '')
    if default_role in {'position_update_equation', 'constant_change_equation'}:
        return residual
    if _interesting_equation_priority(residual.get('role', '')) > _interesting_equation_priority(default_role):
        return residual
    return dict(default or {})


def _best_residual_first_equation(pack: dict) -> dict:
    categories = dict(pack.get('categories') or {})
    candidates = []
    for category in ('residual_strength', 'residual_dynamics', 'residual_periodic'):
        candidates.extend(dict(item) for item in categories.get(category, []))
    candidates = [
        candidate for candidate in candidates
        if float(candidate.get('score', 0.0) or 0.0) >= 0.20
    ]
    if not candidates:
        return {}
    candidates.sort(
        key=lambda item: (
            _interesting_equation_priority(str(item.get('role') or '')),
            float(item.get('score', 0.0) or 0.0),
            -int(item.get('complexity', 0) or 0),
        ),
        reverse=True,
    )
    return candidates[0]


def _interesting_equation_priority(role: str) -> float:
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
    return priorities.get(role, 0.0)


def _run_math_foundation_prep_case(
    context: str,
    seed: int,
    object_count: int,
    steps: int,
    num_agents: int,
    world_type: str,
    hidden_manifest: HiddenWorldManifest | None = None,
    equation_operator_priors: list[dict] | None = None,
    force_backend: str = 'python',
    equation_scoring_backend: str = 'python',
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
            force_backend=force_backend,
            equation_scoring_backend=equation_scoring_backend,
        )
    metrics = _foundation_metrics_from_knowledge(kb)
    equations = _equation_metrics_from_knowledge(kb)
    return {
        'context': context,
        'seed': seed,
        'objects': object_count,
        'steps': steps,
        'force_backend': force_backend,
        'equation_scoring_backend': equation_scoring_backend,
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
    planned_actions: list[dict] | None = None,
    residual_first: bool = False,
    force_backend: str = 'python',
    equation_scoring_backend: str = 'python',
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
            planned_actions=planned_actions,
            equation_operator_priors=equation_operator_priors,
            equation_max_operator_feedback_rows=96,
            equation_max_operator_feedback_operators=2,
            force_backend=force_backend,
            equation_scoring_backend=equation_scoring_backend,
        )
    foundation = _foundation_metrics_from_knowledge(kb)
    equations = _equation_metrics_from_knowledge(
        kb,
        residual_first=residual_first,
    )
    return {
        'context': context,
        'seed': seed,
        'objects': object_count,
        'steps': steps,
        'force_backend': force_backend,
        'equation_scoring_backend': equation_scoring_backend,
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


def _counter_preview(counter: Counter, limit: int = 4) -> str:
    if not counter:
        return 'none'
    return ', '.join(
        f"{key}:{count}" for key, count in counter.most_common(limit)
    )


def _interesting_equation_family(result: dict) -> tuple[str, str]:
    interesting = dict(result.get('interesting_equation') or {})
    parameters = dict(interesting.get('parameters') or {})
    expression = str(interesting.get('expression') or '')
    family = (
        parameters.get('operator_kind')
        or interesting.get('role')
        or 'unknown'
    )
    exponent = parameters.get('distance_exponent')
    if exponent is None and 'separation^' in expression:
        exponent = expression.rsplit('separation^', 1)[-1].split()[0]
    exponent_label = 'none' if exponent is None else str(exponent)
    return str(family), exponent_label


def _is_motion_update_family(family: str) -> bool:
    return family in {
        'position_update_equation',
        'constant_change_equation',
    }


def _clean_equation_results(section_results: list[dict]) -> list[dict]:
    return [
        result for result in section_results
        if result.get('interesting_equation')
        and not result.get('label_leaks')
        and (result.get('passed') or result.get('equation_passed'))
    ]


def _should_force_residual_first(
    theory_memory: CumulativeTheoryMemory,
    context: str,
) -> bool:
    if context in {'standard', 'zero_gravity'}:
        return False
    records = [
        record for record in getattr(theory_memory, 'equation_case_records', [])
        if str(record.get('context') or '') == context
        and bool(record.get('passed'))
        and int(record.get('label_leak_count', 0) or 0) == 0
    ]
    for record in records:
        family = (
            dict(record.get('parameters') or {}).get('operator_kind')
            or record.get('role')
            or 'unknown'
        )
        if not _is_motion_update_family(str(family)):
            return True
    return False


def _section_contexts_for_plan(plan: dict) -> set[str]:
    contexts = set()
    for key in ('world_type', 'source_context'):
        value = plan.get(key)
        if value:
            contexts.add(str(value))
    for key in ('original_record', 'learned_invariant', 'equation_invariant'):
        value = dict(plan.get(key) or {})
        context = value.get('context')
        if context:
            contexts.add(str(context))
    hypothesis = dict(plan.get('domain_split_hypothesis') or {})
    if hypothesis.get('source_context'):
        contexts.add(str(hypothesis['source_context']))
    contexts.update(str(item) for item in hypothesis.get('conflict_contexts') or [])
    return contexts


def _section_followup_plans(
    theory_memory: CumulativeTheoryMemory,
    context: str,
    *,
    object_counts: list[int],
    steps: int,
    limit: int = 2,
) -> list[dict]:
    candidate_limit = max(12, limit * 8)
    candidates = theory_memory.planned_experiments(
        world_types=[context],
        object_counts=object_counts,
        steps=steps,
        limit=candidate_limit,
    )
    followups = [
        plan for plan in candidates
        if str(plan.get('world_type') or '') == context
        and context in _section_contexts_for_plan(plan)
    ]
    return followups[:limit]


def _section_cycle_cases(
    theory_memory: CumulativeTheoryMemory,
    context: str,
    *,
    object_counts: list[int],
    steps: int,
    seeds: int,
    cycle: int,
) -> list[dict]:
    target_count = max(1, seeds) * max(1, len(object_counts))
    if cycle <= 0:
        return [
            {
                'seed': seed,
                'object_count': object_count,
                'steps': steps,
                'plan': None,
            }
            for object_count in object_counts
            for seed in range(seeds)
        ]

    cases = []
    seen_plans = set()
    followups = _section_followup_plans(
        theory_memory,
        context,
        object_counts=object_counts,
        steps=steps,
        limit=target_count,
    )
    allowed_counts = set(object_counts)
    for plan in followups:
        object_count = int(plan.get('object_count') or object_counts[0])
        if object_count not in allowed_counts:
            object_count = object_counts[0]
        plan_key = (
            plan.get('experiment_kind'),
            plan.get('replay_key'),
            int(plan.get('seed', 0) or 0),
            object_count,
        )
        if plan_key in seen_plans:
            continue
        seen_plans.add(plan_key)
        cases.append({
            'seed': int(plan.get('seed', 0) or 0),
            'object_count': object_count,
            'steps': int(plan.get('steps', steps) or steps),
            'plan': dict(plan),
        })
        if len(cases) >= target_count:
            return cases

    fresh_seed = cycle * max(1, seeds)
    fresh_seen = {
        (case['seed'], case['object_count'])
        for case in cases
    }
    while len(cases) < target_count:
        for object_count in object_counts:
            if len(cases) >= target_count:
                break
            while (fresh_seed, object_count) in fresh_seen:
                fresh_seed += 1
            cases.append({
                'seed': fresh_seed,
                'object_count': object_count,
                'steps': steps,
                'plan': None,
            })
            fresh_seen.add((fresh_seed, object_count))
            fresh_seed += 1
    return cases


def _section_best_result(context: str, section_results: list[dict]) -> dict:
    if context in {'standard', 'zero_gravity'}:
        return max(
            section_results,
            key=lambda item: float(item.get('interesting_score', 0.0) or 0.0),
        )
    clean_results = _clean_equation_results(section_results)
    non_motion_results = [
        result for result in clean_results
        if not _is_motion_update_family(_interesting_equation_family(result)[0])
    ]
    if not non_motion_results:
        return max(section_results, key=_interesting_result_rank)

    signature_counts = Counter(
        _interesting_equation_family(result)
        for result in non_motion_results
    )
    signature_scores = {}
    for signature in signature_counts:
        matching = [
            result for result in non_motion_results
            if _interesting_equation_family(result) == signature
        ]
        signature_scores[signature] = sum(
            float(result.get('interesting_score', 0.0) or 0.0)
            for result in matching
        ) / max(1, len(matching))
    dominant_signature = max(
        signature_counts,
        key=lambda signature: (
            signature_counts[signature],
            _interesting_equation_priority(signature[0]),
            signature_scores[signature],
        ),
    )
    signature_results = [
        result for result in non_motion_results
        if _interesting_equation_family(result) == dominant_signature
    ]
    return max(signature_results, key=_interesting_result_rank)


def _print_section_study_summary(
    context: str,
    section_results: list[dict],
    theory_memory: CumulativeTheoryMemory,
    *,
    object_counts: list[int],
    steps: int,
    cycle: int,
    total_cycles: int,
):
    if not section_results:
        return
    passed = sum(1 for result in section_results if result.get('passed'))
    leaks = sum(len(result.get('label_leaks') or []) for result in section_results)
    family_counts = Counter()
    exponent_counts = Counter()
    for result in section_results:
        family, exponent = _interesting_equation_family(result)
        family_counts[family] += 1
        exponent_counts[exponent] += 1
    best = _section_best_result(context, section_results)
    best_equation = dict(best.get('interesting_equation') or {})
    best_text = (
        f"{best_equation.get('target', '?')} ~= "
        f"{best_equation.get('expression', '?')}"
    )
    print(
        f"Section study summary: {context} cycle={cycle}/{total_cycles} "
        f"rows={len(section_results)} passed={passed}/{len(section_results)} "
        f"leaks={leaks}",
        flush=True,
    )
    print(
        f"  Families: {_counter_preview(family_counts)}",
        flush=True,
    )
    print(
        f"  Distance exponents: {_counter_preview(exponent_counts)}",
        flush=True,
    )
    print(
        f"  Best so far: {best_text} "
        f"(score={float(best.get('interesting_score', 0.0) or 0.0):.2f})",
        flush=True,
    )
    consolidation = _section_parameter_consolidation(context, section_results)
    if consolidation.get('dominant_family'):
        parts = [
            f"family={consolidation['dominant_family']}",
            f"support={consolidation['support_count']}/{consolidation['eligible_count']}",
        ]
        exponent = consolidation.get('selected_distance_exponent')
        if exponent is not None:
            parts.append(
                f"exponent={exponent} "
                f"({consolidation['distance_exponent_confidence']:.0%})"
            )
        radius = consolidation.get('selected_cutoff_radius')
        if radius is not None:
            parts.append(
                f"radius~{radius} "
                f"({consolidation['cutoff_radius_confidence']:.0%})"
            )
        print(f"  Robust law: {', '.join(parts)}", flush=True)
    leak_diagnosis = _section_leak_diagnosis(context, section_results)
    if leak_diagnosis.get('leak_count'):
        print(
            "  Leak diagnosis: "
            f"labels={_counter_preview(Counter(leak_diagnosis['label_counts']))} "
            f"rows={leak_diagnosis['affected_row_count']}",
            flush=True,
        )
    decomposition = _section_composite_decomposition(context, section_results)
    if decomposition.get('inferred_component_count', 0) > 1:
        component_text = ', '.join(
            f"{item['component']}:{item['support_count']}"
            for item in decomposition['inferred_components'][:4]
        )
        print(f"  Component hypotheses: {component_text}", flush=True)
    followups = _section_followup_plans(
        theory_memory,
        context,
        object_counts=object_counts,
        steps=steps,
        limit=2,
    )
    if followups:
        print("  Section follow-up probes:", flush=True)
        for plan in followups:
            print(
                f"    {plan['experiment_kind']} {plan['world_type']} "
                f"seed={plan['seed']} "
                f"priority={plan['priority']:.2f}: {plan['reason']}",
                flush=True,
            )


def _section_groups(results: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for result in results:
        groups.setdefault(str(result.get('context') or 'unknown'), []).append(result)
    return groups


def _numeric_parameter_from_result(result: dict, key: str) -> float | None:
    parameters = dict((result.get('interesting_equation') or {}).get('parameters') or {})
    value = parameters.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _rounded_number(value: float | None, digits: int = 3) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _section_parameter_consolidation(
    context: str,
    section_results: list[dict],
) -> dict:
    clean_results = _clean_equation_results(section_results)
    non_motion = [
        result for result in clean_results
        if not _is_motion_update_family(_interesting_equation_family(result)[0])
    ]
    if not non_motion:
        return {
            'context': context,
            'eligible_count': 0,
            'dominant_family': None,
        }
    family_counts = Counter(
        _interesting_equation_family(result)[0]
        for result in non_motion
    )
    dominant_family, support_count = family_counts.most_common(1)[0]
    family_results = [
        result for result in non_motion
        if _interesting_equation_family(result)[0] == dominant_family
    ]

    exponents = [
        _numeric_parameter_from_result(result, 'distance_exponent')
        for result in family_results
    ]
    exponents = [value for value in exponents if value is not None]
    exponent_counts = Counter(
        round(float(value), 2)
        for value in exponents
    )
    selected_exponent = None
    exponent_confidence = 0.0
    if exponent_counts:
        selected_exponent, exponent_support = exponent_counts.most_common(1)[0]
        exponent_confidence = exponent_support / max(1, len(exponents))

    radii = [
        _numeric_parameter_from_result(result, 'cutoff_radius')
        for result in family_results
    ]
    radii = [value for value in radii if value is not None]
    radius_clusters = Counter(round(float(value)) for value in radii)
    selected_radius = None
    radius_confidence = 0.0
    radius_spread = None
    if radius_clusters:
        cluster, radius_support = radius_clusters.most_common(1)[0]
        cluster_values = [
            value for value in radii
            if round(float(value)) == cluster
        ]
        selected_radius = _median(cluster_values)
        radius_confidence = radius_support / max(1, len(radii))
        radius_spread = (
            round(max(cluster_values) - min(cluster_values), 3)
            if len(cluster_values) > 1
            else 0.0
        )

    relation_counts = Counter()
    for result in family_results:
        parameters = dict((result.get('interesting_equation') or {}).get('parameters') or {})
        relation = parameters.get('relation')
        if relation:
            relation_counts[str(relation)] += 1

    return {
        'context': context,
        'eligible_count': len(non_motion),
        'dominant_family': dominant_family,
        'family_counts': dict(family_counts),
        'support_count': support_count,
        'support_fraction': round(support_count / max(1, len(non_motion)), 3),
        'selected_distance_exponent': _rounded_number(selected_exponent),
        'distance_exponent_counts': {
            str(key): count for key, count in exponent_counts.items()
        },
        'distance_exponent_confidence': round(exponent_confidence, 3),
        'selected_cutoff_radius': _rounded_number(selected_radius),
        'cutoff_radius_clusters': {
            str(key): count for key, count in radius_clusters.items()
        },
        'cutoff_radius_confidence': round(radius_confidence, 3),
        'cutoff_radius_cluster_spread': radius_spread,
        'relation_counts': dict(relation_counts),
    }


def _section_leak_diagnosis(context: str, section_results: list[dict]) -> dict:
    leak_rows = []
    label_counts = Counter()
    expression_counts = Counter()
    for result in section_results:
        leaks = list(result.get('label_leaks') or [])
        if not leaks:
            continue
        labels = Counter()
        for leak in leaks:
            for label in leak.get('labels') or []:
                label_counts[str(label)] += 1
                labels[str(label)] += 1
            expression = leak.get('expression') or leak.get('description')
            if expression:
                expression_counts[str(expression)] += 1
        leak_rows.append({
            'context': result.get('context'),
            'seed': result.get('seed'),
            'phase': result.get('phase', 'math_final_discovery'),
            'interesting_role': (
                (result.get('interesting_equation') or {}).get('role')
            ),
            'interesting_expression': (
                (result.get('interesting_equation') or {}).get('expression')
            ),
            'leak_count': len(leaks),
            'labels': dict(labels),
        })
    if not leak_rows:
        return {
            'context': context,
            'leak_count': 0,
            'affected_row_count': 0,
            'label_counts': {},
            'rows': [],
            'recommendation': None,
        }
    recommendation = (
        'block leaked rows from robust-law selection and inspect forbidden '
        'label source before trusting localized/hidden claims'
    )
    return {
        'context': context,
        'leak_count': sum(row['leak_count'] for row in leak_rows),
        'affected_row_count': len(leak_rows),
        'label_counts': dict(label_counts),
        'top_expressions': dict(expression_counts.most_common(5)),
        'rows': leak_rows,
        'recommendation': recommendation,
    }


def _component_hypothesis_from_result(result: dict) -> str | None:
    family, _exponent = _interesting_equation_family(result)
    equation = dict(result.get('interesting_equation') or {})
    expression = str(equation.get('expression') or '').lower()
    role = str(equation.get('role') or '')
    if _is_motion_update_family(family):
        return 'baseline_motion_or_uniform_component'
    if family == 'phase_basis' or 'periodic' in role or 'sin(step' in expression:
        return 'time_varying_component'
    if 'perpendicular' in expression or 'perpendicular' in role:
        return 'tangential_component'
    if family in {
        'localized_tapered_power',
        'localized_cutoff_window',
        'local_residual_direction_equation',
        'local_residual_distance_scaled_direction_equation',
    }:
        return 'localized_radial_component'
    if 'unit_inferred_vector' in expression or 'unit_local_inferred_vector' in expression:
        return 'radial_or_uniform_component'
    return None


def _section_composite_decomposition(
    context: str,
    section_results: list[dict],
) -> dict:
    component_counts = Counter()
    family_counts = Counter()
    manifest_components = Counter()
    seen_manifests = set()
    for result in _clean_equation_results(section_results):
        family, _exponent = _interesting_equation_family(result)
        family_counts[family] += 1
        component = _component_hypothesis_from_result(result)
        if component:
            component_counts[component] += 1
        manifest = dict(result.get('manifest') or {})
        if manifest:
            manifest_key = json.dumps(
                manifest,
                sort_keys=True,
                separators=(',', ':'),
                default=str,
            )
            if manifest_key not in seen_manifests:
                seen_manifests.add(manifest_key)
                for component_data in manifest.get('components') or []:
                    component_type = component_data.get('type')
                    if component_type:
                        manifest_components[str(component_type)] += 1
    inferred_components = [
        {
            'component': component,
            'support_count': count,
            'support_fraction': round(count / max(1, sum(component_counts.values())), 3),
        }
        for component, count in component_counts.most_common()
    ]
    status = (
        'composite_hypothesis'
        if len(component_counts) > 1
        else 'single_component_or_unresolved'
    )
    return {
        'context': context,
        'status': status,
        'inferred_component_count': len(component_counts),
        'inferred_components': inferred_components,
        'family_counts': dict(family_counts),
        'benchmark_manifest_components': dict(manifest_components),
    }


def _math_final_rows_for_artifact(results: list[dict]) -> list[dict]:
    rows = []
    for result in results:
        equation = dict(result.get('interesting_equation') or {})
        rows.append({
            'context': result.get('context'),
            'seed': result.get('seed'),
            'objects': result.get('objects'),
            'steps': result.get('steps'),
            'phase': result.get('phase', 'math_final_discovery'),
            'passed': bool(result.get('passed')),
            'ready_for_final': bool(result.get('ready_for_final')),
            'equation_passed': bool(result.get('equation_passed')),
            'leak_count': len(result.get('label_leaks') or []),
            'interesting_score': result.get('interesting_score'),
            'interesting_role': equation.get('role'),
            'interesting_target': equation.get('target'),
            'interesting_expression': equation.get('expression'),
            'interesting_parameters': dict(equation.get('parameters') or {}),
            'case_elapsed_seconds': result.get('case_elapsed_seconds'),
            'planned_experiment_kind': (
                dict(result.get('planned_experiment') or {}).get('experiment_kind')
            ),
            'planned_outcome': (
                dict(result.get('planned_experiment_outcome') or {}).get('outcome')
            ),
        })
    return rows


def _theory_beliefs_from_plan(plan: dict) -> list[dict]:
    return theory_beliefs_from_plan(plan)


def _format_intervention_action(action: dict) -> str:
    return format_intervention_action(action)


def _experiment_design_from_plan(plan: dict) -> dict:
    return experiment_design_from_plan(plan)


def _experiment_design_cockpit(
    theory_memory: CumulativeTheoryMemory,
    *,
    world_types: list[str] | None = None,
    object_counts: list[int] | None = None,
    steps: int = 240,
    limit: int = 5,
) -> list[dict]:
    return build_experiment_design_cockpit(
        theory_memory,
        world_types=world_types or WORLD_TYPES,
        object_counts=object_counts or [5],
        steps=steps,
        limit=limit,
    )


def _weak_case_reasons(result: dict) -> list[str]:
    reasons = []
    if not result.get('ready_for_final'):
        reasons.append('foundation_not_ready')
    if not result.get('equation_passed'):
        reasons.append('equation_not_clean')
    if result.get('label_leaks'):
        reasons.append('label_leak')
    if (
        result.get('interesting_equation')
        and float(result.get('interesting_score', 0.0) or 0.0) < 0.25
    ):
        reasons.append('low_equation_score')
    outcome = dict(result.get('planned_experiment_outcome') or {})
    outcome_name = str(outcome.get('outcome') or '')
    if any(token in outcome_name for token in ('conflict', 'absent', 'needs_repair')):
        reasons.append('planned_holdout_or_repair_conflict')
    if not result.get('passed') and not reasons:
        reasons.append('failed_without_specific_reason')
    return reasons


def _weak_case_diagnostics(results: list[dict]) -> dict:
    rows = []
    reason_counts = Counter()
    context_counts = Counter()
    for result in results:
        reasons = _weak_case_reasons(result)
        if result.get('passed') and not reasons:
            continue
        context = str(result.get('context') or 'unknown')
        for reason in reasons:
            reason_counts[reason] += 1
        context_counts[context] += 1
        rows.append({
            'context': context,
            'seed': result.get('seed'),
            'phase': result.get('phase', 'math_final_discovery'),
            'passed': bool(result.get('passed')),
            'ready_for_final': bool(result.get('ready_for_final')),
            'equation_passed': bool(result.get('equation_passed')),
            'leak_count': len(result.get('label_leaks') or []),
            'interesting_score': round(
                float(result.get('interesting_score', 0.0) or 0.0),
                3,
            ),
            'interesting_role': (
                dict(result.get('interesting_equation') or {}).get('role')
            ),
            'planned_experiment_kind': (
                dict(result.get('planned_experiment') or {}).get('experiment_kind')
            ),
            'planned_outcome': (
                dict(result.get('planned_experiment_outcome') or {}).get('outcome')
            ),
            'reasons': reasons,
        })
    status = 'all_clean' if not rows else 'needs_diagnosis'
    next_actions = []
    if reason_counts.get('label_leak'):
        next_actions.append('inspect and block leaked rows before theorem selection')
    if reason_counts.get('planned_holdout_or_repair_conflict'):
        next_actions.append('run targeted replay/repair probes for conflicted theories')
    if reason_counts.get('equation_not_clean') or reason_counts.get('low_equation_score'):
        next_actions.append('increase local residual/operator search pressure for weak contexts')
    if reason_counts.get('foundation_not_ready'):
        next_actions.append('rerun foundation probes for weak rows before trusting equations')
    return {
        'status': status,
        'weak_case_count': len(rows),
        'reason_counts': dict(reason_counts),
        'context_counts': dict(context_counts),
        'rows': rows,
        'next_actions': next_actions,
    }


def _runtime_profile_summary(
    results: list[dict],
    events: list[dict] | None = None,
) -> dict:
    case_rows = []
    context_totals: dict[str, dict[str, float]] = {}
    for result in results:
        elapsed = result.get('case_elapsed_seconds')
        if not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool):
            continue
        context = str(result.get('context') or 'unknown')
        elapsed = max(0.0, float(elapsed))
        case_rows.append({
            'context': context,
            'seed': result.get('seed'),
            'phase': result.get('phase', 'math_final_discovery'),
            'elapsed_seconds': round(elapsed, 3),
            'passed': bool(result.get('passed')),
            'equation_count': int(result.get('equation_count', 0) or 0),
            'interesting_score': round(
                float(result.get('interesting_score', 0.0) or 0.0),
                3,
            ),
        })
        bucket = context_totals.setdefault(context, {
            'case_count': 0.0,
            'total_case_seconds': 0.0,
            'max_case_seconds': 0.0,
        })
        bucket['case_count'] += 1.0
        bucket['total_case_seconds'] += elapsed
        bucket['max_case_seconds'] = max(bucket['max_case_seconds'], elapsed)

    by_context = []
    for context, bucket in sorted(context_totals.items()):
        count = int(bucket['case_count'])
        total = bucket['total_case_seconds']
        by_context.append({
            'context': context,
            'case_count': count,
            'total_case_seconds': round(total, 3),
            'avg_case_seconds': round(total / max(1, count), 3),
            'max_case_seconds': round(bucket['max_case_seconds'], 3),
        })
    by_context.sort(
        key=lambda item: (
            item['total_case_seconds'],
            item['max_case_seconds'],
        ),
        reverse=True,
    )
    total_case_seconds = sum(item['elapsed_seconds'] for item in case_rows)
    slowest = sorted(
        case_rows,
        key=lambda item: item['elapsed_seconds'],
        reverse=True,
    )[:8]
    section_events = [
        dict(event)
        for event in list(events or [])
        if event.get('event') == 'section_cycle_profile'
    ]
    return {
        'enabled': bool(case_rows or section_events),
        'case_count': len(case_rows),
        'total_case_seconds': round(total_case_seconds, 3),
        'avg_case_seconds': round(
            total_case_seconds / max(1, len(case_rows)),
            3,
        ),
        'slowest_cases': slowest,
        'by_context': by_context,
        'section_events': section_events,
        'event_count': len(events or []),
        'shard_recommendation': (
            'split the slowest contexts into separate HF jobs and merge artifacts'
            if len(by_context) >= 3
            else 'collect more profiled sections before sharding'
        ),
    }


def _artifact_status_needs_log_fallback(upload: dict) -> bool:
    return upload.get('status') not in {'uploaded', 'uploaded_via_pr'}


def _artifact_log_chunks(
    artifact: dict,
    *,
    run_id: str,
    max_chars: int = 12000,
) -> list[dict]:
    raw = json.dumps(artifact, sort_keys=True, separators=(',', ':')).encode('utf-8')
    encoded = base64.b64encode(zlib.compress(raw, level=9)).decode('ascii')
    max_chars = max(1024, int(max_chars or 12000))
    chunks = [
        encoded[index:index + max_chars]
        for index in range(0, len(encoded), max_chars)
    ] or ['']
    checksum = sum(raw) % 1000000007
    return [
        {
            'run_id': run_id,
            'index': index + 1,
            'total': len(chunks),
            'encoding': 'zlib+base64+json',
            'raw_bytes': len(raw),
            'compressed_chars': len(encoded),
            'checksum_mod': checksum,
            'data': chunk,
        }
        for index, chunk in enumerate(chunks)
    ]


def _emit_artifact_log_chunks(
    artifact: dict,
    *,
    run_id: str,
) -> list[dict]:
    chunks = _artifact_log_chunks(artifact, run_id=run_id)
    for chunk in chunks:
        print(
            "HF_ARTIFACT_CHUNK " + json.dumps(chunk, sort_keys=True),
            flush=True,
        )
    return chunks


def _compact_artifact_summary_for_log(artifact: dict) -> dict:
    upload = dict(artifact.get('hf_upload') or {})
    return {
        'run_kind': artifact.get('run_kind'),
        'runs_final': bool(artifact.get('runs_final')),
        'run_id': artifact.get('run_id'),
        'result_count': artifact.get('result_count'),
        'passed_count': artifact.get('passed_count'),
        'ready_count': artifact.get('ready_count'),
        'equation_clean_count': artifact.get('equation_clean_count'),
        'section_count': artifact.get('section_count'),
        'readiness': artifact.get('readiness'),
        'memory_delta': artifact.get('memory_delta'),
        'artifact_path': artifact.get('artifact_path'),
        'hf_upload': {
            'status': upload.get('status'),
            'repo_id': upload.get('repo_id'),
            'path_in_repo': upload.get('path_in_repo'),
            'reason': upload.get('reason'),
            'error': upload.get('error'),
            'create_repo_error': upload.get('create_repo_error'),
        },
        'experiment_design_cockpit': list(
            artifact.get('experiment_design_cockpit') or []
        )[:3],
        'weak_case_diagnostics': {
            'status': dict(artifact.get('weak_case_diagnostics') or {}).get('status'),
            'weak_case_count': dict(
                artifact.get('weak_case_diagnostics') or {}
            ).get('weak_case_count'),
            'reason_counts': dict(
                dict(artifact.get('weak_case_diagnostics') or {}).get(
                    'reason_counts'
                )
                or {}
            ),
        },
        'runtime_profile': {
            'enabled': dict(artifact.get('runtime_profile') or {}).get('enabled'),
            'case_count': dict(artifact.get('runtime_profile') or {}).get('case_count'),
            'total_case_seconds': dict(
                artifact.get('runtime_profile') or {}
            ).get('total_case_seconds'),
            'artifact_persist_seconds': dict(
                artifact.get('runtime_profile') or {}
            ).get('artifact_persist_seconds'),
        },
    }


def _math_final_artifact_summary(
    results: list[dict],
    theory_memory: CumulativeTheoryMemory,
    *,
    run_id: str,
    run_config: dict,
    starting_memory_summary: dict,
    runtime_profile_events: list[dict] | None = None,
) -> dict:
    section_groups = _section_groups(results)
    section_consolidations = [
        _section_parameter_consolidation(context, rows)
        for context, rows in sorted(section_groups.items())
    ]
    leak_diagnostics = [
        diagnosis for diagnosis in (
            _section_leak_diagnosis(context, rows)
            for context, rows in sorted(section_groups.items())
        )
        if diagnosis.get('leak_count')
    ]
    composite_decompositions = [
        decomposition for decomposition in (
            _section_composite_decomposition(context, rows)
            for context, rows in sorted(section_groups.items())
        )
        if (
            decomposition.get('inferred_component_count', 0) > 1
            or decomposition.get('benchmark_manifest_components')
        )
    ]
    weak_case_diagnostics = _weak_case_diagnostics(results)
    resource_efficiency = theory_memory.resource_efficiency_report()
    super_system_snapshot = build_super_system_report(
        theory_memory,
        world_types=list(run_config.get('world_types') or WORLD_TYPES),
        object_counts=list(run_config.get('object_counts') or [5]),
        steps=int(run_config.get('steps', 240) or 240),
        limit=5,
    )
    return {
        'run_kind': 'math_final_discovery',
        'runs_final': True,
        'run_id': run_id,
        'run_config': dict(run_config),
        'result_count': len(results),
        'passed_count': sum(1 for result in results if result.get('passed')),
        'equation_clean_count': sum(
            1 for result in results
            if result.get('equation_passed') and not result.get('label_leaks')
        ),
        'ready_count': sum(1 for result in results if result.get('ready_for_final')),
        'section_count': len(section_groups),
        'rows': _math_final_rows_for_artifact(results),
        'section_consolidations': section_consolidations,
        'leak_diagnostics': leak_diagnostics,
        'weak_case_diagnostics': weak_case_diagnostics,
        'composite_decompositions': composite_decompositions,
        'runtime_profile': _runtime_profile_summary(
            results,
            runtime_profile_events,
        ),
        'readiness': theory_memory.discovery_readiness_report(),
        'resource_efficiency': resource_efficiency,
        'canonical_law_compression': (
            resource_efficiency.get('canonical_law_compression')
            or theory_memory.canonical_law_compression_report()
        ),
        'memory_delta': theory_memory.memory_delta_since(starting_memory_summary),
        'experiment_design_cockpit': _experiment_design_cockpit(
            theory_memory,
            world_types=list(run_config.get('world_types') or WORLD_TYPES),
            object_counts=list(run_config.get('object_counts') or [5]),
            steps=int(run_config.get('steps', 240) or 240),
            limit=5,
        ),
        'super_system_snapshot': super_system_snapshot,
        'theory_memory': theory_memory.to_dict(),
    }


def _result_from_artifact_row(row: dict) -> dict:
    leak_count = max(0, int(row.get('leak_count', 0) or 0))
    return {
        'context': row.get('context'),
        'seed': row.get('seed'),
        'objects': row.get('objects'),
        'steps': row.get('steps'),
        'phase': row.get('phase', 'math_final_discovery'),
        'passed': bool(row.get('passed')),
        'ready_for_final': bool(row.get('ready_for_final')),
        'equation_passed': bool(row.get('equation_passed')),
        'label_leaks': [
            {'labels': ['unknown'], 'description': 'leak preserved from shard row'}
            for _ in range(leak_count)
        ],
        'interesting_score': row.get('interesting_score') or 0.0,
        'interesting_equation': {
            'role': row.get('interesting_role'),
            'target': row.get('interesting_target'),
            'expression': row.get('interesting_expression'),
            'score': row.get('interesting_score') or 0.0,
            'parameters': dict(row.get('interesting_parameters') or {}),
        },
        'case_elapsed_seconds': row.get('case_elapsed_seconds'),
        **(
            {
                'planned_experiment': {
                    'experiment_kind': row.get('planned_experiment_kind'),
                }
            }
            if row.get('planned_experiment_kind')
            else {}
        ),
        **(
            {
                'planned_experiment_outcome': {
                    'outcome': row.get('planned_outcome'),
                }
            }
            if row.get('planned_outcome')
            else {}
        ),
    }


def merge_final_artifacts(
    artifact_files: list[str | Path],
    *,
    output_file: str | Path | None = None,
    run_id: str | None = None,
) -> dict:
    """Merge sharded final-run summaries without rerunning discovery."""
    artifacts = []
    for path in artifact_files:
        artifact_path = Path(path)
        with artifact_path.open('r', encoding='utf-8') as handle:
            artifact = json.load(handle)
        artifact['_source_path'] = str(artifact_path)
        artifacts.append(artifact)

    rows = []
    source_artifacts = []
    runtime_events = []
    for artifact in artifacts:
        source_artifacts.append({
            'path': artifact.get('_source_path'),
            'run_id': artifact.get('run_id'),
            'run_kind': artifact.get('run_kind'),
            'result_count': artifact.get('result_count'),
            'passed_count': artifact.get('passed_count'),
            'weak_case_count': (
                dict(artifact.get('weak_case_diagnostics') or {})
                .get('weak_case_count')
            ),
            'upload_status': (
                dict(artifact.get('hf_upload') or {}).get('status')
            ),
        })
        rows.extend(dict(row) for row in artifact.get('rows') or [])
        profile = dict(artifact.get('runtime_profile') or {})
        for event in list(profile.get('section_events') or []):
            runtime_events.append({
                **dict(event),
                'source_run_id': artifact.get('run_id'),
            })

    results = [_result_from_artifact_row(row) for row in rows]
    section_groups = _section_groups(results)
    merged_run_id = run_id or _default_hf_run_id(prefix='math-final-merged')
    merged = {
        'run_kind': 'math_final_shard_merge',
        'runs_final': True,
        'run_id': merged_run_id,
        'shard_count': len(artifacts),
        'source_artifacts': source_artifacts,
        'result_count': len(results),
        'passed_count': sum(1 for result in results if result.get('passed')),
        'equation_clean_count': sum(
            1 for result in results
            if result.get('equation_passed') and not result.get('label_leaks')
        ),
        'ready_count': sum(1 for result in results if result.get('ready_for_final')),
        'section_count': len(section_groups),
        'rows': rows,
        'section_consolidations': [
            _section_parameter_consolidation(context, group_rows)
            for context, group_rows in sorted(section_groups.items())
        ],
        'leak_diagnostics': [
            diagnosis for diagnosis in (
                _section_leak_diagnosis(context, group_rows)
                for context, group_rows in sorted(section_groups.items())
            )
            if diagnosis.get('leak_count')
        ],
        'weak_case_diagnostics': _weak_case_diagnostics(results),
        'composite_decompositions': [
            decomposition for decomposition in (
                _section_composite_decomposition(context, group_rows)
                for context, group_rows in sorted(section_groups.items())
            )
            if (
                decomposition.get('inferred_component_count', 0) > 1
                or decomposition.get('benchmark_manifest_components')
            )
        ],
        'runtime_profile': _runtime_profile_summary(results, runtime_events),
    }
    if output_file:
        merged['artifact_path'] = str(_write_json_artifact(output_file, merged))
    return merged


def run_merge_final_artifacts(
    artifact_files: list[str | Path],
    *,
    output_file: str | Path | None = None,
    run_id: str | None = None,
) -> dict:
    report = merge_final_artifacts(
        artifact_files,
        output_file=output_file,
        run_id=run_id,
    )
    print("=" * 70)
    print("FINAL ARTIFACT SHARD MERGE")
    print("=" * 70)
    print(f"Shards: {report['shard_count']}")
    print(
        f"Rows: {report['passed_count']}/{report['result_count']} passed, "
        f"weak_cases={report['weak_case_diagnostics']['weak_case_count']}"
    )
    profile = dict(report.get('runtime_profile') or {})
    if profile.get('enabled'):
        print(
            "Runtime profile: "
            f"cases={profile['case_count']} "
            f"total_case_seconds={profile['total_case_seconds']}"
        )
        for item in list(profile.get('by_context') or [])[:5]:
            print(
                f"  {item['context']}: total={item['total_case_seconds']}s "
                f"avg={item['avg_case_seconds']}s"
            )
    if output_file:
        print(f"Merged artifact: {output_file}")
    return report


def _physics_force_kernel_fixture(sample_count: int) -> dict:
    positions = [
        (
            2.0 + (index % 113) * 0.137,
            2.5 + (index % 109) * 0.129,
        )
        for index in range(sample_count)
    ]
    return {
        'positions': positions,
        'dt': 0.016,
        'time_value': 0.016,
        'gravity': 9.8,
        'uniform_force': {'x': 8.0, 'y': -0.25},
        'central_force': {'x': 10.0, 'y': 10.0, 'strength': 200.0},
        'gravity_wells': [{
            'x': 7.0,
            'y': 13.0,
            'strength': 220.0,
            'radius': 8.5,
        }],
        'repulsion_zones': [{
            'x': 13.0,
            'y': 8.0,
            'strength': 80.0,
            'radius': 6.0,
        }],
        'inverse_square_repulsions': [{
            'x': 5.0,
            'y': 5.0,
            'strength': 130.0,
        }],
        'vortex_fields': [{
            'x': 10.0,
            'y': 10.0,
            'strength': 140.0,
            'radius': 9.0,
            'direction': 1.0,
        }],
        'time_varying_force': {
            'axis': 'x',
            'amplitude': 12.0,
            'period': 1.28,
            'phase': 0.0,
        },
        'force_components': [
            {
                'type': 'cutoff_radial',
                'params': {
                    'x': 11.0,
                    'y': 6.0,
                    'strength': 155.0,
                    'radius': 6.0,
                    'exponent': 2.0,
                    'direction': -1.0,
                },
            },
            {
                'type': 'piecewise_radial',
                'params': {
                    'x': 9.0,
                    'y': 12.0,
                    'split_radius': 6.0,
                    'inner': {
                        'strength': 120.0,
                        'exponent': 1.0,
                        'direction': 1.0,
                    },
                    'outer': {
                        'strength': 70.0,
                        'exponent': 1.5,
                        'direction': -1.0,
                    },
                },
            },
            {
                'type': 'periodic_regime_uniform',
                'params': {
                    'period': 1.1,
                    'phase': 0.0,
                    'duty_cycle': 0.6,
                    'on_force': {'x': 5.0, 'y': 0.0},
                    'off_force': {'x': -2.0, 'y': 0.0},
                },
            },
        ],
    }


def _python_force_kernel_deltas(fixture: dict) -> list[tuple[float, float]]:
    from world.physics import PhysicsWorld, Vec2

    world = PhysicsWorld(force_backend='python')
    world.gravity = fixture['gravity']
    world.uniform_force = fixture['uniform_force']
    world.central_force = fixture['central_force']
    world.gravity_wells = [dict(item) for item in fixture['gravity_wells']]
    world.repulsion_zones = [dict(item) for item in fixture['repulsion_zones']]
    world.inverse_square_repulsions = [
        dict(item) for item in fixture['inverse_square_repulsions']
    ]
    world.vortex_fields = [dict(item) for item in fixture['vortex_fields']]
    world.time_varying_force = fixture['time_varying_force']
    world.force_components = [
        {
            'type': component['type'],
            'params': dict(component['params']),
        }
        for component in fixture['force_components']
    ]
    world.time = fixture['time_value']
    for index, (x, y) in enumerate(fixture['positions']):
        world.add_object(
            PhysicsObject(
                position=Vec2(x, y),
                velocity=Vec2(0.0, 0.0),
                mass=1.0,
                radius=0.1,
                object_id=index,
            )
        )
    world._apply_external_forces_python(fixture['dt'])
    return [
        (obj.velocity.x, obj.velocity.y)
        for obj in world.objects
    ]


def _max_delta_error(
    reference: list[tuple[float, float]],
    candidate: list[tuple[float, float]],
) -> float:
    return max(
        (
            max(abs(rx - cx), abs(ry - cy))
            for (rx, ry), (cx, cy) in zip(reference, candidate)
        ),
        default=0.0,
    )


def run_gpu_feasibility_benchmark(
    *,
    sample_count: int = 50000,
    repeats: int = 3,
    prefer_cuda: bool = True,
    force_backend: str | None = None,
    output_file: str | Path | None = None,
) -> dict:
    """Tiny tensor-style benchmark for deciding whether a GPU port is worth it."""
    sample_count = max(128, int(sample_count or 128))
    repeats = max(1, int(repeats or 1))

    xs = [((index % 997) - 498) / 31.0 for index in range(sample_count)]
    ys = [((index % 991) - 495) / 29.0 for index in range(sample_count)]
    cpu_started = time.perf_counter()
    checksum = 0.0
    for _ in range(repeats):
        total = 0.0
        for x, y in zip(xs, ys):
            separation = (x * x + y * y) ** 0.5 + 1e-6
            total += x / (separation * separation)
            total += y / (separation ** 1.5)
        checksum = total
    python_cpu_seconds = time.perf_counter() - cpu_started

    torch_status = 'not_checked'
    torch_cpu_seconds = None
    cuda_seconds = None
    cuda_available = False
    device_name = None
    try:
        import torch  # type: ignore

        torch_status = 'available'
        tx = torch.tensor(xs, dtype=torch.float32)
        ty = torch.tensor(ys, dtype=torch.float32)
        started = time.perf_counter()
        torch_checksum = None
        for _ in range(repeats):
            separation = torch.sqrt(tx * tx + ty * ty) + 1e-6
            torch_checksum = (
                tx / (separation * separation)
                + ty / torch.pow(separation, 1.5)
            ).sum()
        torch_cpu_seconds = time.perf_counter() - started
        if torch_checksum is not None:
            checksum = float(torch_checksum.item())
        cuda_available = bool(torch.cuda.is_available())
        if prefer_cuda and cuda_available:
            device = torch.device('cuda')
            device_name = torch.cuda.get_device_name(0)
            gx = tx.to(device)
            gy = ty.to(device)
            torch.cuda.synchronize()
            started = time.perf_counter()
            gpu_checksum = None
            for _ in range(repeats):
                separation = torch.sqrt(gx * gx + gy * gy) + 1e-6
                gpu_checksum = (
                    gx / (separation * separation)
                    + gy / torch.pow(separation, 1.5)
                ).sum()
            torch.cuda.synchronize()
            cuda_seconds = time.perf_counter() - started
            if gpu_checksum is not None:
                checksum = float(gpu_checksum.item())
    except Exception as error:
        torch_status = f'unavailable:{error.__class__.__name__}'

    fixture = _physics_force_kernel_fixture(sample_count)
    physics_reference = []
    started = time.perf_counter()
    for _ in range(repeats):
        physics_reference = _python_force_kernel_deltas(fixture)
    python_force_kernel_seconds = time.perf_counter() - started
    requested_force_backend = force_backend or ('cuda' if prefer_cuda else 'numpy')
    force_backend_status = resolve_force_backend(requested_force_backend)
    physics_backend_result = []
    started = time.perf_counter()
    if force_backend_status['backend'] == 'python':
        physics_backend_result = physics_reference
    else:
        for _ in range(repeats):
            physics_backend_result = compute_external_force_deltas(
                positions=fixture['positions'],
                dt=fixture['dt'],
                time_value=fixture['time_value'],
                gravity=fixture['gravity'],
                uniform_force=fixture['uniform_force'],
                central_force=fixture['central_force'],
                gravity_wells=fixture['gravity_wells'],
                repulsion_zones=fixture['repulsion_zones'],
                inverse_square_repulsions=fixture['inverse_square_repulsions'],
                vortex_fields=fixture['vortex_fields'],
                time_varying_force=fixture['time_varying_force'],
                force_components=fixture['force_components'],
                backend=force_backend_status['backend'],
            )
    physics_backend_seconds = time.perf_counter() - started
    physics_force_speedup = (
        python_force_kernel_seconds / physics_backend_seconds
        if physics_backend_seconds > 0
        else 0.0
    )
    physics_force_max_abs_error = _max_delta_error(
        physics_reference,
        physics_backend_result,
    )
    physics_force_parity_passed = physics_force_max_abs_error <= 1e-8

    reference_cpu_seconds = (
        torch_cpu_seconds
        if torch_cpu_seconds is not None
        else python_cpu_seconds
    )
    gpu_speedup = (
        reference_cpu_seconds / cuda_seconds
        if cuda_seconds and cuda_seconds > 0
        else 0.0
    )
    if not physics_force_parity_passed:
        recommendation = 'fix_tensor_physics_parity_before_gpu_runs'
    elif (
        force_backend_status['backend'] == 'cuda'
        and physics_force_speedup >= 3.0
    ):
        recommendation = 'use_cuda_force_backend_for_large_force_batches'
    elif (
        force_backend_status['backend'] != 'python'
        and physics_force_speedup >= 1.25
    ):
        recommendation = 'use_tensor_force_backend_for_force_heavy_shards'
    elif cuda_seconds:
        recommendation = 'prefer_cpu_sharding_until_gpu_kernel_is_larger'
    else:
        recommendation = 'prefer_cpu_sharding_until_tensor_backend_exists'
    report = {
        'run_kind': 'gpu_feasibility_benchmark',
        'runs_final': False,
        'sample_count': sample_count,
        'repeats': repeats,
        'python_cpu_seconds': round(python_cpu_seconds, 6),
        'torch_status': torch_status,
        'torch_cpu_seconds': (
            round(torch_cpu_seconds, 6)
            if torch_cpu_seconds is not None
            else None
        ),
        'cuda_available': cuda_available,
        'cuda_seconds': round(cuda_seconds, 6) if cuda_seconds is not None else None,
        'cuda_device': device_name,
        'gpu_speedup_vs_reference_cpu': round(gpu_speedup, 3),
        'available_force_backends': available_force_backends(),
        'physics_force_backend_request': requested_force_backend,
        'physics_force_backend': force_backend_status['backend'],
        'physics_force_backend_fallback_reason': (
            force_backend_status.get('fallback_reason')
        ),
        'python_force_kernel_seconds': round(python_force_kernel_seconds, 6),
        'physics_force_backend_seconds': round(physics_backend_seconds, 6),
        'physics_force_speedup_vs_python': round(physics_force_speedup, 3),
        'physics_force_max_abs_error': round(physics_force_max_abs_error, 12),
        'physics_force_parity_passed': physics_force_parity_passed,
        'recommendation': recommendation,
        'checksum': round(float(checksum), 6),
    }
    if output_file:
        report['artifact_path'] = str(_write_json_artifact(output_file, report))
    print("=" * 70)
    print("GPU FEASIBILITY BENCHMARK")
    print("=" * 70)
    print(
        f"samples={sample_count} repeats={repeats} "
        f"python_cpu={report['python_cpu_seconds']}s "
        f"torch_cpu={report['torch_cpu_seconds']} "
        f"cuda={report['cuda_seconds']}"
    )
    print(
        "physics_force_kernel="
        f"{report['physics_force_backend']} "
        f"python={report['python_force_kernel_seconds']}s "
        f"backend={report['physics_force_backend_seconds']}s "
        f"speedup={report['physics_force_speedup_vs_python']} "
        f"parity={report['physics_force_parity_passed']}"
    )
    print(f"recommendation={recommendation}")
    return report


def _backend_profile_match(reference: dict, candidate: dict) -> dict:
    checks = {
        'ready_for_final': reference.get('ready_for_final') == candidate.get('ready_for_final'),
        'equation_passed': reference.get('equation_passed') == candidate.get('equation_passed'),
        'equation_count': reference.get('equation_count') == candidate.get('equation_count'),
        'installed_count': reference.get('installed_count') == candidate.get('installed_count'),
        'interesting_role': (
            reference.get('interesting_role') == candidate.get('interesting_role')
        ),
        'leak_count': reference.get('leak_count') == candidate.get('leak_count'),
    }
    return {
        'matches_reference': all(checks.values()),
        'checks': checks,
    }


def _backend_profile_summary(
    rows: list[dict],
    reference_backend: str,
    reference_equation_scoring_backend: str = 'python',
) -> list[dict]:
    summaries = []
    reference_elapsed_by_case = {
        (
            row['context'],
            row['seed'],
            row['objects'],
            row['steps'],
        ): float(row.get('elapsed_seconds', 0.0) or 0.0)
        for row in rows
        if (
            row.get('force_backend') == reference_backend
            and row.get('equation_scoring_backend') == reference_equation_scoring_backend
        )
    }
    backend_pairs = sorted({
        (row['force_backend'], row.get('equation_scoring_backend', 'python'))
        for row in rows
    })
    for backend, equation_backend in backend_pairs:
        backend_rows = [
            row for row in rows
            if row['force_backend'] == backend
            and row.get('equation_scoring_backend', 'python') == equation_backend
        ]
        elapsed = [
            float(row.get('elapsed_seconds', 0.0) or 0.0)
            for row in backend_rows
        ]
        reference_elapsed = []
        for row in backend_rows:
            key = (row['context'], row['seed'], row['objects'], row['steps'])
            if key in reference_elapsed_by_case:
                reference_elapsed.append(reference_elapsed_by_case[key])
        hot_stages = Counter(row.get('hot_stage') for row in backend_rows)
        hot_stages.pop(None, None)
        avg_elapsed = sum(elapsed) / max(len(elapsed), 1)
        avg_reference_elapsed = sum(reference_elapsed) / max(len(reference_elapsed), 1)
        summaries.append({
            'force_backend': backend,
            'equation_scoring_backend': equation_backend,
            'case_count': len(backend_rows),
            'avg_elapsed_seconds': round(avg_elapsed, 6),
            'speedup_vs_reference_backend': round(
                avg_reference_elapsed / avg_elapsed
                if avg_elapsed > 0 and avg_reference_elapsed > 0
                else 0.0,
                3,
            ),
            'metric_match_count': sum(
                1 for row in backend_rows
                if row.get('matches_reference', True)
            ),
            'hot_stages': [
                {'stage': stage, 'count': count}
                for stage, count in hot_stages.most_common(5)
            ],
        })
    return summaries


def run_backend_profile_comparison(
    *,
    backends: list[str] | None = None,
    equation_scoring_backends: list[str] | None = None,
    seeds: int = 1,
    steps: int = 80,
    object_counts: list[int] | None = None,
    world_types: list[str] | None = None,
    num_agents: int = 2,
    output_file: str | Path | None = None,
) -> dict:
    """Run a small non-final discovery profile across simulator backends."""
    backends = backends or ['python', 'numpy']
    equation_scoring_backends = equation_scoring_backends or ['python']
    object_counts = object_counts or [3]
    world_types = world_types or ['standard', 'sideways_wind']
    rows = []
    reference_backend = backends[0]
    reference_rows = {}

    print("=" * 70)
    print("BACKEND PROFILE COMPARISON")
    print("=" * 70)
    print(f"Backends: {', '.join(backends)}")
    print(f"Equation scoring backends: {', '.join(equation_scoring_backends)}")
    print(f"Worlds: {', '.join(world_types)}")
    print(f"Seeds: 0..{seeds - 1}")
    print(f"Object counts: {', '.join(str(count) for count in object_counts)}")
    print(f"Steps per run: {steps}")
    print("Runs final: False")
    print()
    print(
        f"{'Backend':10s} {'EqBackend':10s} {'World':22s} {'Seed':>4s} {'Obj':>3s} "
        f"{'Elapsed':>8s} {'Hot stage':28s} {'Match':>5s}"
    )
    print("-" * 92)

    for backend in backends:
        for equation_backend in equation_scoring_backends:
            for world_type in world_types:
                for object_count in object_counts:
                    for seed in range(seeds):
                        started = time.perf_counter()
                        with contextlib.redirect_stdout(io.StringIO()):
                            _, kb, _ = run_experiment(
                                num_steps=steps,
                                num_initial_objects=object_count,
                                seed=seed,
                                verbose=False,
                                report_interval=max(steps, 1),
                                world_type=world_type,
                                num_agents=num_agents,
                                enable_equation_probes=True,
                                force_backend=backend,
                                equation_scoring_backend=equation_backend,
                                profile_timings=True,
                            )
                        elapsed = time.perf_counter() - started
                        foundation = _foundation_metrics_from_knowledge(kb)
                        equations = _equation_metrics_from_knowledge(kb)
                        interesting = equations.get('interesting_equation') or {}
                        profile = dict(getattr(kb, 'runtime_profile', {}) or {})
                        row = {
                            'context': world_type,
                            'seed': seed,
                            'objects': object_count,
                            'steps': steps,
                            'force_backend': backend,
                            'equation_scoring_backend': equation_backend,
                            'elapsed_seconds': round(elapsed, 6),
                            'profile': profile,
                            'hot_stage': profile.get('hot_stage'),
                            'top_stages': list(profile.get('stages') or [])[:6],
                            'ready_for_final': foundation['ready_for_final'],
                            'readiness_score': foundation['readiness_score'],
                            'equation_passed': equations['passed'],
                            'equation_count': equations['equation_count'],
                            'installed_count': equations['installed_count'],
                            'interesting_role': interesting.get('role'),
                            'interesting_score': equations['interesting_score'],
                            'leak_count': len(equations.get('label_leaks') or []),
                        }
                        case_key = (world_type, seed, object_count, steps)
                        if (
                            backend == reference_backend
                            and equation_backend == equation_scoring_backends[0]
                        ):
                            row['reference_backend'] = True
                            row['matches_reference'] = True
                            row['reference_checks'] = {}
                            reference_rows[case_key] = row
                        else:
                            match = _backend_profile_match(reference_rows[case_key], row)
                            row['reference_backend'] = False
                            row['matches_reference'] = match['matches_reference']
                            row['reference_checks'] = match['checks']
                        rows.append(row)
                        print(
                            f"{backend:10s} {equation_backend:10s} "
                            f"{world_type:22s} {seed:4d} "
                            f"{object_count:3d} {elapsed:8.3f} "
                            f"{str(row.get('hot_stage') or 'none')[:28]:28s} "
                            f"{'YES' if row['matches_reference'] else 'NO':>5s}",
                            flush=True,
                        )

    summaries = _backend_profile_summary(
        rows,
        reference_backend,
        equation_scoring_backends[0],
    )
    report = {
        'run_kind': 'backend_profile_comparison',
        'runs_final': False,
        'reference_backend': reference_backend,
        'reference_equation_scoring_backend': equation_scoring_backends[0],
        'backends': list(backends),
        'equation_scoring_backends': list(equation_scoring_backends),
        'world_types': list(world_types),
        'seeds': int(seeds),
        'steps': int(steps),
        'object_counts': list(object_counts),
        'rows': rows,
        'backend_summaries': summaries,
        'all_metric_matches': all(row.get('matches_reference') for row in rows),
        'available_force_backends': available_force_backends(),
        'available_equation_scoring_backends': available_equation_scoring_backends(),
    }
    if output_file:
        report['artifact_path'] = str(_write_json_artifact(output_file, report))
    print("-" * 92)
    for summary in summaries:
        print(
            f"{summary['force_backend']}: "
            f"equation={summary['equation_scoring_backend']} "
            f"avg={summary['avg_elapsed_seconds']}s "
            f"speedup_vs_{reference_backend}="
            f"{summary['speedup_vs_reference_backend']} "
            f"matches={summary['metric_match_count']}/{summary['case_count']}"
        )
    print(f"All backend metrics match reference: {report['all_metric_matches']}")
    return report


def _default_hf_run_id(prefix: str = 'math-final') -> str:
    return f"{prefix}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"


def _write_json_artifact(path: str | Path, artifact: dict):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as handle:
        json.dump(artifact, handle, indent=2, sort_keys=True)
    return output_path


def _file_sha256(path: str | Path) -> str | None:
    file_path = Path(path)
    if not file_path.exists():
        return None
    digest = hashlib.sha256()
    with file_path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_memory_hash_state(
    path: str | Path,
    before_hash: str | None,
) -> dict:
    after_hash = _file_sha256(path)
    return {
        'path': str(path),
        'exists': Path(path).exists(),
        'before_hash': before_hash,
        'after_hash': after_hash,
        'unchanged': before_hash == after_hash,
    }


def run_status_capsule(
    theory_memory: CumulativeTheoryMemory | None = None,
    *,
    theory_memory_file: str | Path | None = None,
    runtime_memory_path: str = 'tmp/theory-memory.json',
    output_file: str | Path | None = None,
    memory_data: dict | None = None,
    git_status_text: str | None = None,
    git_ignored_text: str | None = None,
) -> dict:
    """Print an orchestrator-facing status capsule without running discovery."""
    if memory_data is not None:
        loaded_memory = dict(memory_data)
    elif theory_memory is not None:
        loaded_memory = theory_memory.to_dict()
    else:
        loaded_memory = load_capsule_memory_data(theory_memory_file)
    status_text = (
        git_status_text
        if git_status_text is not None
        else git_status_for_path(runtime_memory_path)
    )
    ignored_text = (
        git_ignored_text
        if git_ignored_text is not None
        else git_check_ignore_for_path(runtime_memory_path)
    )
    capsule = build_ai_different_status_capsule(
        loaded_memory,
        git_status_text=status_text,
        git_ignored_text=ignored_text,
        runtime_memory_path=runtime_memory_path,
    )
    if output_file:
        capsule['artifact_path'] = str(Path(output_file))
        _write_json_artifact(output_file, capsule)
    print(
        "AI_DIFFERENT_STATUS_CAPSULE "
        + json.dumps(capsule, sort_keys=True),
        flush=True,
    )
    return capsule


def run_module_chat_export(
    theory_memory: CumulativeTheoryMemory | None = None,
    *,
    theory_memory_file: str | Path | None = None,
    runtime_memory_path: str = 'tmp/theory-memory.json',
    recipient: str = 'orchestrator',
    topic: str = 'ai_different.status_capsule',
    inbox_file: str | Path | None = None,
    output_file: str | Path | None = None,
    memory_data: dict | None = None,
    git_status_text: str | None = None,
    git_ignored_text: str | None = None,
) -> dict:
    """Export the status capsule as a plain module-chat JSON message."""
    if memory_data is not None:
        loaded_memory = dict(memory_data)
    elif theory_memory is not None:
        loaded_memory = theory_memory.to_dict()
    else:
        loaded_memory = load_capsule_memory_data(theory_memory_file)
    status_text = (
        git_status_text
        if git_status_text is not None
        else git_status_for_path(runtime_memory_path)
    )
    ignored_text = (
        git_ignored_text
        if git_ignored_text is not None
        else git_check_ignore_for_path(runtime_memory_path)
    )
    capsule = build_ai_different_status_capsule(
        loaded_memory,
        git_status_text=status_text,
        git_ignored_text=ignored_text,
        runtime_memory_path=runtime_memory_path,
    )
    inbox_summary = read_module_chat_inbox(inbox_file, participant='ai_different')
    message = export_capsule_chat_message(
        capsule,
        recipient=recipient,
        topic=topic,
        inbox_summary=inbox_summary,
    )
    if output_file:
        _write_json_artifact(output_file, message)
    print(
        "AI_DIFFERENT_MODULE_CHAT_MESSAGE "
        + json.dumps(message, sort_keys=True),
        flush=True,
    )
    return message


def run_module_chat_response_loop(
    theory_memory: CumulativeTheoryMemory | None = None,
    *,
    theory_memory_file: str | Path | None = None,
    runtime_memory_path: str = 'tmp/theory-memory.json',
    recipient: str = 'orchestrator',
    topic: str = 'ai_different.abstraction_transfer_response',
    inbox_file: str | Path | None = None,
    output_file: str | Path | None = None,
    response_mode: str = 'plan',
    fallback_outcome_mode: str = 'confirmed',
    memory_data: dict | None = None,
    git_status_text: str | None = None,
    git_ignored_text: str | None = None,
) -> dict:
    """Read module-chat inbox and emit an abstraction-transfer response message."""
    if response_mode not in {'plan', 'run'}:
        raise ValueError('response_mode must be plan or run')
    if fallback_outcome_mode not in {'confirmed', 'weak', 'absent'}:
        raise ValueError('fallback_outcome_mode must be confirmed, weak, or absent')
    if memory_data is not None:
        loaded_memory = dict(memory_data)
        working_memory = CumulativeTheoryMemory.from_dict(loaded_memory)
    elif theory_memory is not None:
        loaded_memory = theory_memory.to_dict()
        working_memory = CumulativeTheoryMemory.from_dict(loaded_memory)
    else:
        loaded_memory = load_capsule_memory_data(theory_memory_file)
        working_memory = CumulativeTheoryMemory.from_dict(loaded_memory)
    status_text = (
        git_status_text
        if git_status_text is not None
        else git_status_for_path(runtime_memory_path)
    )
    ignored_text = (
        git_ignored_text
        if git_ignored_text is not None
        else git_check_ignore_for_path(runtime_memory_path)
    )
    capsule = build_ai_different_status_capsule(
        loaded_memory,
        git_status_text=status_text,
        git_ignored_text=ignored_text,
        runtime_memory_path=runtime_memory_path,
    )
    inbox_summary = read_module_chat_inbox(inbox_file, participant='ai_different')
    request_preview = export_chat_driven_response_message(
        capsule,
        inbox_summary,
        recipient=recipient,
        topic=topic,
    )['body']['selected_chat_request']
    campaign_summary = None
    ran_campaign = False
    if response_mode == 'run':
        outcome_mode = (
            request_preview.get('outcome_mode')
            if request_preview.get('outcome_mode') in {'confirmed', 'weak', 'absent'}
            else fallback_outcome_mode
        )
        with contextlib.redirect_stdout(io.StringIO()):
            campaign_summary = run_abstraction_transfer_campaign(
                theory_memory=working_memory,
                seed_start=0,
                steps=90,
                object_count=5,
                target_world_types=[
                    'standard',
                    'time_varying',
                    'hidden_procedural',
                ],
                outcome_mode=outcome_mode,
                emit_hf_artifact_summary=False,
            )
        capsule = build_ai_different_status_capsule(
            working_memory.to_dict(),
            git_status_text=status_text,
            git_ignored_text=ignored_text,
            runtime_memory_path=runtime_memory_path,
        )
        ran_campaign = True
    message = export_chat_driven_response_message(
        capsule,
        inbox_summary,
        campaign_summary=campaign_summary,
        ran_campaign=ran_campaign,
        recipient=recipient,
        topic=topic,
    )
    if output_file:
        _write_json_artifact(output_file, message)
    print(
        "AI_DIFFERENT_MODULE_CHAT_RESPONSE "
        + json.dumps(message, sort_keys=True),
        flush=True,
    )
    return message


def run_module_chat_family_response(
    theory_memory: CumulativeTheoryMemory | None = None,
    *,
    theory_memory_file: str | Path | None = None,
    runtime_memory_path: str = 'tmp/theory-memory.json',
    recipient: str = 'auto',
    topic: str = 'ai_different.module_family_response',
    inbox_file: str | Path | None = None,
    output_file: str | Path | None = None,
    ledger_file: str | Path = 'tmp/module-chat-response-ledger.json',
    response_mode: str = 'plan',
    fallback_outcome_mode: str = 'confirmed',
    memory_data: dict | None = None,
    git_status_text: str | None = None,
    git_ignored_text: str | None = None,
) -> dict:
    """Run the richer three-module coordination response workflow."""
    if response_mode not in {'plan', 'run'}:
        raise ValueError('response_mode must be plan or run')
    if fallback_outcome_mode not in {'confirmed', 'weak', 'absent'}:
        raise ValueError('fallback_outcome_mode must be confirmed, weak, or absent')
    if memory_data is not None:
        loaded_memory = dict(memory_data)
        working_memory = CumulativeTheoryMemory.from_dict(loaded_memory)
    elif theory_memory is not None:
        loaded_memory = theory_memory.to_dict()
        working_memory = CumulativeTheoryMemory.from_dict(loaded_memory)
    else:
        loaded_memory = load_capsule_memory_data(theory_memory_file)
        working_memory = CumulativeTheoryMemory.from_dict(loaded_memory)
    status_text = (
        git_status_text
        if git_status_text is not None
        else git_status_for_path(runtime_memory_path)
    )
    ignored_text = (
        git_ignored_text
        if git_ignored_text is not None
        else git_check_ignore_for_path(runtime_memory_path)
    )
    capsule = build_ai_different_status_capsule(
        loaded_memory,
        git_status_text=status_text,
        git_ignored_text=ignored_text,
        runtime_memory_path=runtime_memory_path,
    )
    inbox_summary = read_module_chat_inbox(inbox_file, participant='ai_different')
    selected = choose_module_family_followup(capsule, inbox_summary)
    selected_recipient = choose_module_family_recipient(
        inbox_summary,
        selected,
        requested_recipient=recipient,
    )
    before_hash = _file_sha256(runtime_memory_path)
    initial_hash_state = _runtime_memory_hash_state(runtime_memory_path, before_hash)
    preview_ledger = build_module_family_response_ledger(
        capsule,
        inbox_summary,
        selected_recipient=selected_recipient,
        response_mode=response_mode,
        runtime_memory_hash_state=initial_hash_state,
        ledger_path=ledger_file,
    )
    campaign_summary = None
    ran_campaign = False
    if preview_ledger['run_decision']['should_run_no_save_campaign']:
        outcome_mode = (
            selected.get('outcome_mode')
            if selected.get('outcome_mode') in {'confirmed', 'weak', 'absent'}
            else fallback_outcome_mode
        )
        with contextlib.redirect_stdout(io.StringIO()):
            campaign_summary = run_abstraction_transfer_campaign(
                theory_memory=working_memory,
                seed_start=0,
                steps=90,
                object_count=5,
                target_world_types=[
                    'standard',
                    'time_varying',
                    'hidden_procedural',
                ],
                outcome_mode=outcome_mode,
                emit_hf_artifact_summary=False,
            )
        capsule = build_ai_different_status_capsule(
            working_memory.to_dict(),
            git_status_text=status_text,
            git_ignored_text=ignored_text,
            runtime_memory_path=runtime_memory_path,
        )
        ran_campaign = True
    runtime_hash_state = _runtime_memory_hash_state(runtime_memory_path, before_hash)
    ledger = build_module_family_response_ledger(
        capsule,
        inbox_summary,
        selected_recipient=selected_recipient,
        response_mode=response_mode,
        campaign_summary=campaign_summary,
        ran_campaign=ran_campaign,
        runtime_memory_hash_state=runtime_hash_state,
        ledger_path=ledger_file,
    )
    write_response_ledger(ledger_file, ledger)
    message = export_module_family_response_message(
        ledger,
        recipient=selected_recipient,
        topic=topic,
    )
    if output_file:
        _write_json_artifact(output_file, message)
    print(
        "AI_DIFFERENT_MODULE_FAMILY_RESPONSE "
        + json.dumps(message, sort_keys=True),
        flush=True,
    )
    return message


def run_module_chat_rolling_family_response(
    theory_memory: CumulativeTheoryMemory | None = None,
    *,
    theory_memory_file: str | Path | None = None,
    runtime_memory_path: str = 'tmp/theory-memory.json',
    recipient: str = 'auto',
    topic: str = 'ai_different.module_family_response',
    chat_log_file: str | Path | None = None,
    output_file: str | Path | None = None,
    ledger_file: str | Path = 'tmp/module-chat-response-ledger.json',
    rolling_memory_file: str | Path = 'tmp/module-chat-family-memory.json',
    response_mode: str = 'plan',
    fallback_outcome_mode: str = 'confirmed',
    memory_data: dict | None = None,
    git_status_text: str | None = None,
    git_ignored_text: str | None = None,
) -> dict:
    """Run an idempotent rolling module-family response pass over a bus log."""
    if response_mode not in {'plan', 'run'}:
        raise ValueError('response_mode must be plan or run')
    if fallback_outcome_mode not in {'confirmed', 'weak', 'absent'}:
        raise ValueError('fallback_outcome_mode must be confirmed, weak, or absent')
    if memory_data is not None:
        loaded_memory = dict(memory_data)
        working_memory = CumulativeTheoryMemory.from_dict(loaded_memory)
    elif theory_memory is not None:
        loaded_memory = theory_memory.to_dict()
        working_memory = CumulativeTheoryMemory.from_dict(loaded_memory)
    else:
        loaded_memory = load_capsule_memory_data(theory_memory_file)
        working_memory = CumulativeTheoryMemory.from_dict(loaded_memory)
    status_text = (
        git_status_text
        if git_status_text is not None
        else git_status_for_path(runtime_memory_path)
    )
    ignored_text = (
        git_ignored_text
        if git_ignored_text is not None
        else git_check_ignore_for_path(runtime_memory_path)
    )
    capsule = build_ai_different_status_capsule(
        loaded_memory,
        git_status_text=status_text,
        git_ignored_text=ignored_text,
        runtime_memory_path=runtime_memory_path,
    )
    rolling_memory = load_rolling_family_memory(rolling_memory_file)
    log_summary = read_module_chat_log(chat_log_file, participant='ai_different')
    unprocessed = rolling_unprocessed_inbound_messages(
        log_summary,
        rolling_memory,
        participant='ai_different',
    )
    before_hash = _file_sha256(runtime_memory_path)
    runtime_hash_state = _runtime_memory_hash_state(runtime_memory_path, before_hash)
    new_messages = list(unprocessed['new_messages'])
    skipped_messages = list(unprocessed['skipped_messages'])
    message = None
    ledger = None
    if new_messages:
        round_summary = module_chat_summary_from_messages(
            new_messages,
            path=log_summary.get('path'),
            invalid_messages=log_summary.get('invalid_messages') or [],
        )
        selected = choose_module_family_followup(capsule, round_summary)
        selected_recipient = choose_module_family_recipient(
            round_summary,
            selected,
            requested_recipient=recipient,
        )
        preview_ledger = build_module_family_response_ledger(
            capsule,
            round_summary,
            selected_recipient=selected_recipient,
            response_mode=response_mode,
            runtime_memory_hash_state=runtime_hash_state,
            ledger_path=ledger_file,
        )
        campaign_summary = None
        ran_campaign = False
        if preview_ledger['run_decision']['should_run_no_save_campaign']:
            outcome_mode = (
                selected.get('outcome_mode')
                if selected.get('outcome_mode') in {'confirmed', 'weak', 'absent'}
                else fallback_outcome_mode
            )
            with contextlib.redirect_stdout(io.StringIO()):
                campaign_summary = run_abstraction_transfer_campaign(
                    theory_memory=working_memory,
                    seed_start=0,
                    steps=90,
                    object_count=5,
                    target_world_types=[
                        'standard',
                        'time_varying',
                        'hidden_procedural',
                    ],
                    outcome_mode=outcome_mode,
                    emit_hf_artifact_summary=False,
                )
            capsule = build_ai_different_status_capsule(
                working_memory.to_dict(),
                git_status_text=status_text,
                git_ignored_text=ignored_text,
                runtime_memory_path=runtime_memory_path,
            )
            ran_campaign = True
        runtime_hash_state = _runtime_memory_hash_state(runtime_memory_path, before_hash)
        ledger = build_module_family_response_ledger(
            capsule,
            round_summary,
            selected_recipient=selected_recipient,
            response_mode=response_mode,
            campaign_summary=campaign_summary,
            ran_campaign=ran_campaign,
            runtime_memory_hash_state=runtime_hash_state,
            ledger_path=ledger_file,
        )
        write_response_ledger(ledger_file, ledger)
        message = export_module_family_response_message(
            ledger,
            recipient=selected_recipient,
            topic=topic,
        )
        rolling_memory = append_rolling_family_record(
            rolling_memory,
            processed_messages=new_messages,
            ledger=ledger,
            response_message=message,
            skipped_count=len(skipped_messages),
            runtime_memory_hash_state=runtime_hash_state,
            observed_outgoing_response_ids=unprocessed.get('prior_outgoing_response_ids'),
        )
        write_rolling_family_memory(rolling_memory_file, rolling_memory)
    latest = dict(rolling_memory.get('latest') or {})
    result = {
        'rolling_family_response_available': True,
        'source_log_path': log_summary.get('path'),
        'rolling_memory_path': str(rolling_memory_file),
        'processed_message_count': len(rolling_memory.get('processed_message_ids') or []),
        'skipped_message_count': len(skipped_messages),
        'new_message_count': len(new_messages),
        'new_message_ids': list(unprocessed.get('new_message_ids') or []),
        'skipped_message_ids': list(unprocessed.get('skipped_message_ids') or []),
        'prior_outgoing_response_ids': list(
            unprocessed.get('prior_outgoing_response_ids') or []
        ),
        'outgoing_response_ids': list(rolling_memory.get('outgoing_response_ids') or []),
        'selected_recipient': (
            message.get('recipient') if message else latest.get('selected_recipient')
        ),
        'evidence_counts_by_sender': (
            ledger.get('evidence_counts_by_sender')
            if ledger
            else latest.get('evidence_counts_by_sender', {})
        ),
        'outcome_or_plan': (
            ledger.get('outcome_or_plan')
            if ledger
            else latest.get('outcome_or_plan', {})
        ),
        'ledger_id': ledger.get('ledger_id') if ledger else latest.get('response_ledger_id'),
        'ledger_hash': (
            ledger.get('ledger_hash') if ledger else latest.get('response_ledger_hash')
        ),
        'ledger_path': (
            ledger.get('artifact_path') if ledger else latest.get('response_ledger_path')
        ),
        'response_message': message,
        'label_clean': (
            bool(ledger.get('label_clean')) if ledger else bool(latest.get('label_clean', True))
        ),
        'project_owned_boundary': dict(capsule.get('project_owned_boundary') or {}),
        'runtime_memory_hash_state': runtime_hash_state,
        'runtime_memory_mutated': not bool(runtime_hash_state.get('unchanged', True)),
        'third_party_checkpoint_used': bool(
            (capsule.get('project_owned_boundary') or {}).get(
                'third_party_checkpoint_used'
            )
        ),
        'invalid_message_count': len(log_summary.get('invalid_messages') or []),
        'memory_hash': rolling_memory.get('memory_hash'),
    }
    if output_file:
        _write_json_artifact(output_file, result)
    print(
        "AI_DIFFERENT_ROLLING_FAMILY_RESPONSE "
        + json.dumps(result, sort_keys=True),
        flush=True,
    )
    return result


def run_family_outcome_evaluator(
    theory_memory: CumulativeTheoryMemory | None = None,
    *,
    theory_memory_file: str | Path | None = None,
    runtime_memory_path: str = 'tmp/theory-memory.json',
    recipient: str = 'orchestrator',
    topic: str = 'ai_different.family_outcome_evaluation',
    rolling_memory_file: str | Path = 'tmp/module-chat-family-memory.json',
    response_ledger_files: list[str | Path] | None = None,
    output_file: str | Path | None = None,
    evaluator_ledger_file: str | Path = 'tmp/module-chat-outcome-evaluator-ledger.json',
    evaluator_memory_file: str | Path = 'tmp/module-chat-outcome-evaluator-memory.json',
    memory_data: dict | None = None,
    git_status_text: str | None = None,
    git_ignored_text: str | None = None,
) -> dict:
    """Evaluate accumulated module-family evidence into the next science step."""
    if memory_data is not None:
        loaded_memory = dict(memory_data)
    elif theory_memory is not None:
        loaded_memory = theory_memory.to_dict()
    else:
        loaded_memory = load_capsule_memory_data(theory_memory_file)
    status_text = (
        git_status_text
        if git_status_text is not None
        else git_status_for_path(runtime_memory_path)
    )
    ignored_text = (
        git_ignored_text
        if git_ignored_text is not None
        else git_check_ignore_for_path(runtime_memory_path)
    )
    capsule = build_ai_different_status_capsule(
        loaded_memory,
        git_status_text=status_text,
        git_ignored_text=ignored_text,
        runtime_memory_path=runtime_memory_path,
    )
    rolling_memory = load_rolling_family_memory(rolling_memory_file)
    evaluator_memory = load_outcome_evaluator_memory(evaluator_memory_file)
    response_ledgers = load_response_ledgers(list(response_ledger_files or []))
    before_hash = _file_sha256(runtime_memory_path)
    runtime_hash_state = _runtime_memory_hash_state(runtime_memory_path, before_hash)
    ledger = build_outcome_evaluator_ledger(
        rolling_memory=rolling_memory,
        response_ledgers=response_ledgers,
        evaluator_memory=evaluator_memory,
        runtime_memory_hash_state=runtime_hash_state,
        project_owned_boundary=dict(capsule.get('project_owned_boundary') or {}),
        ledger_path=evaluator_ledger_file,
    )
    if ledger.get('decision', {}).get('decision_kind') != 'no_op':
        ledger['outgoing_module_chat_response_ids'] = [str(ledger.get('ledger_id'))]
        ledger['ledger_hash'] = hashlib.sha256(
            json.dumps(ledger, sort_keys=True, separators=(',', ':')).encode('utf-8')
        ).hexdigest()
    message = export_outcome_evaluator_message(
        ledger,
        recipient=recipient,
        topic=topic,
    )
    write_outcome_evaluator_ledger(evaluator_ledger_file, ledger)
    evaluator_memory = append_outcome_evaluator_memory(
        evaluator_memory,
        ledger,
        message,
    )
    write_outcome_evaluator_memory(evaluator_memory_file, evaluator_memory)
    result = {
        'family_outcome_evaluator_available': True,
        'rolling_memory_path': str(rolling_memory_file),
        'response_ledger_count': len(response_ledgers),
        'evaluator_memory_path': str(evaluator_memory_file),
        'evaluator_ledger_path': str(evaluator_ledger_file),
        'evaluator_ledger_id': ledger.get('ledger_id'),
        'evaluator_ledger_hash': ledger.get('ledger_hash'),
        'processed_ledger_ids': list(ledger.get('processed_ledger_ids') or []),
        'processed_evidence_ids': list(ledger.get('processed_evidence_ids') or []),
        'classification_counts': dict(ledger.get('classification_counts') or {}),
        'decision': dict(ledger.get('decision') or {}),
        'selected_experiment': dict(ledger.get('selected_experiment') or {}),
        'expected_transfer_signal': ledger.get('expected_transfer_signal'),
        'unresolved_blockers': list(ledger.get('unresolved_blockers') or []),
        'response_message': message,
        'outgoing_response_ids': list(
            evaluator_memory.get('outgoing_response_ids') or []
        ),
        'label_clean': bool(ledger.get('label_clean')),
        'project_owned_boundary': dict(capsule.get('project_owned_boundary') or {}),
        'third_party_checkpoint_used': bool(ledger.get('third_party_checkpoint_used')),
        'runtime_memory_hash_state': runtime_hash_state,
        'runtime_memory_mutated': not bool(runtime_hash_state.get('unchanged', True)),
    }
    if output_file:
        _write_json_artifact(output_file, result)
    print(
        "AI_DIFFERENT_FAMILY_OUTCOME_EVALUATOR "
        + json.dumps(result, sort_keys=True),
        flush=True,
    )
    return result


def run_experiment_contract_loop(
    theory_memory: CumulativeTheoryMemory | None = None,
    *,
    theory_memory_file: str | Path | None = None,
    runtime_memory_path: str = 'tmp/theory-memory.json',
    evaluator_ledger_file: str | Path = 'tmp/module-chat-outcome-evaluator-ledger.json',
    family_bus_file: str | Path | None = None,
    contract_ledger_file: str | Path = 'tmp/module-chat-experiment-contract-ledger.json',
    outbox_file: str | Path | None = None,
    output_file: str | Path | None = None,
    target_recipient: str = 'broadcast',
    repair_recipient: str = 'orchestrator',
    memory_data: dict | None = None,
    git_status_text: str | None = None,
    git_ignored_text: str | None = None,
) -> dict:
    """Emit/resolve AI Different experiment contracts from evaluator evidence."""
    if memory_data is not None:
        loaded_memory = dict(memory_data)
    elif theory_memory is not None:
        loaded_memory = theory_memory.to_dict()
    else:
        loaded_memory = load_capsule_memory_data(theory_memory_file)
    status_text = (
        git_status_text
        if git_status_text is not None
        else git_status_for_path(runtime_memory_path)
    )
    ignored_text = (
        git_ignored_text
        if git_ignored_text is not None
        else git_check_ignore_for_path(runtime_memory_path)
    )
    capsule = build_ai_different_status_capsule(
        loaded_memory,
        git_status_text=status_text,
        git_ignored_text=ignored_text,
        runtime_memory_path=runtime_memory_path,
    )
    with Path(evaluator_ledger_file).open('r', encoding='utf-8') as handle:
        evaluator_ledger = validate_evaluator_ledger(json.load(handle))
    contract_ledger = load_experiment_contract_ledger(contract_ledger_file)
    bus_summary = read_family_bus_messages(family_bus_file)
    before_hash = _file_sha256(runtime_memory_path)
    runtime_hash_state = _runtime_memory_hash_state(runtime_memory_path, before_hash)
    updated_ledger, message = update_contract_ledger(
        contract_ledger,
        evaluator_ledger,
        list(bus_summary.get('messages') or []),
        runtime_memory_hash_state=runtime_hash_state,
        project_owned_boundary=dict(capsule.get('project_owned_boundary') or {}),
        target_recipient=target_recipient,
        repair_recipient=repair_recipient,
        artifact_path=contract_ledger_file,
    )
    write_experiment_contract_ledger(contract_ledger_file, updated_ledger)
    if outbox_file:
        write_contract_outbox_jsonl(outbox_file, message)
    latest = dict(updated_ledger.get('latest') or {})
    result = {
        'experiment_contract_capability': True,
        'contract_ledger_path': str(contract_ledger_file),
        'contract_ledger_hash': updated_ledger.get('ledger_hash'),
        'new_contract_count': int(latest.get('new_contract_count', 0) or 0),
        'skipped_contract_count': int(latest.get('skipped_contract_count', 0) or 0),
        'resolved_contract_count': int(latest.get('resolved_contract_count', 0) or 0),
        'blocked_contract_count': int(latest.get('blocked_contract_count', 0) or 0),
        'chosen_recipient': latest.get('chosen_recipient'),
        'chosen_action': latest.get('chosen_action'),
        'outbox_count': int(latest.get('outbox_count', 0) or 0),
        'outbox_file': str(outbox_file) if outbox_file else None,
        'response_message': message,
        'open_contract_count': sum(
            1 for contract in updated_ledger.get('contracts') or []
            if contract.get('status') == 'open'
        ),
        'resolved_total': sum(
            1 for contract in updated_ledger.get('contracts') or []
            if contract.get('status') == 'resolved'
        ),
        'blocked_total': sum(
            1 for contract in updated_ledger.get('contracts') or []
            if contract.get('status') == 'blocked'
        ),
        'runtime_memory_hash_state': runtime_hash_state,
        'runtime_memory_mutated': not bool(runtime_hash_state.get('unchanged', True)),
        'label_leaks': list(latest.get('label_leaks') or []),
        'project_owned_boundary': dict(capsule.get('project_owned_boundary') or {}),
        'third_party_checkpoint_used': bool(
            (capsule.get('project_owned_boundary') or {}).get(
                'third_party_checkpoint_used'
            )
        ),
        'invalid_message_count': len(bus_summary.get('invalid_messages') or []),
    }
    if output_file:
        _write_json_artifact(output_file, result)
    print(
        "AI_DIFFERENT_EXPERIMENT_CONTRACT "
        + json.dumps(result, sort_keys=True),
        flush=True,
    )
    return result


def run_cross_module_adjudicator(
    theory_memory: CumulativeTheoryMemory | None = None,
    *,
    theory_memory_file: str | Path | None = None,
    runtime_memory_path: str = 'tmp/theory-memory.json',
    transcript_file: str | Path | None = None,
    evaluator_ledger_file: str | Path = 'tmp/module-chat-outcome-evaluator-ledger.json',
    contract_ledger_file: str | Path = 'tmp/module-chat-experiment-contract-ledger.json',
    adjudicator_ledger_file: str | Path = 'tmp/module-chat-cross-module-adjudicator-ledger.json',
    outbox_file: str | Path | None = None,
    output_file: str | Path | None = None,
    memory_data: dict | None = None,
    git_status_text: str | None = None,
    git_ignored_text: str | None = None,
) -> dict:
    """Adjudicate a family transcript against AI Different experiment contracts."""
    if memory_data is not None:
        loaded_memory = dict(memory_data)
    elif theory_memory is not None:
        loaded_memory = theory_memory.to_dict()
    else:
        loaded_memory = load_capsule_memory_data(theory_memory_file)
    status_text = (
        git_status_text
        if git_status_text is not None
        else git_status_for_path(runtime_memory_path)
    )
    ignored_text = (
        git_ignored_text
        if git_ignored_text is not None
        else git_check_ignore_for_path(runtime_memory_path)
    )
    capsule = build_ai_different_status_capsule(
        loaded_memory,
        git_status_text=status_text,
        git_ignored_text=ignored_text,
        runtime_memory_path=runtime_memory_path,
    )
    transcript = read_family_transcript(transcript_file)
    evaluator_ledger = load_plain_json(evaluator_ledger_file)
    contract_ledger = load_plain_json(contract_ledger_file)
    adjudicator_ledger = load_adjudicator_ledger(adjudicator_ledger_file)
    before_hash = _file_sha256(runtime_memory_path)
    runtime_hash_state = _runtime_memory_hash_state(runtime_memory_path, before_hash)
    updated_ledger, message = build_adjudication(
        transcript_messages=list(transcript.get('messages') or []),
        adjudicator_ledger=adjudicator_ledger,
        evaluator_ledger=evaluator_ledger,
        contract_ledger=contract_ledger,
        runtime_memory_hash_state=runtime_hash_state,
        project_owned_boundary=dict(capsule.get('project_owned_boundary') or {}),
        artifact_path=adjudicator_ledger_file,
    )
    write_adjudicator_ledger(adjudicator_ledger_file, updated_ledger)
    if outbox_file:
        write_adjudication_outbox_jsonl(outbox_file, message)
    latest = dict(updated_ledger.get('latest') or {})
    result = {
        'cross_module_adjudicator_available': True,
        'adjudicator_ledger_path': str(adjudicator_ledger_file),
        'adjudicator_ledger_hash': updated_ledger.get('ledger_hash'),
        'processed_message_count': len(updated_ledger.get('processed_message_ids') or []),
        'new_message_count': int(latest.get('new_message_count', 0) or 0),
        'skipped_message_count': int(latest.get('skipped_message_count', 0) or 0),
        'open_contract_count': int(latest.get('open_contract_count', 0) or 0),
        'resolved_contract_count': int(latest.get('resolved_contract_count', 0) or 0),
        'blocked_contract_count': int(latest.get('blocked_contract_count', 0) or 0),
        'selected_action': latest.get('selected_action'),
        'chosen_recipient': latest.get('chosen_recipient'),
        'outbox_count': int(latest.get('outbox_count', 0) or 0),
        'outbox_file': str(outbox_file) if outbox_file else None,
        'response_message': message,
        'runtime_memory_hash_state': runtime_hash_state,
        'runtime_memory_mutated': not bool(runtime_hash_state.get('unchanged', True)),
        'label_leaks': list(latest.get('label_leaks') or []),
        'label_leaks_count': len(latest.get('label_leaks') or []),
        'project_owned_boundary': dict(capsule.get('project_owned_boundary') or {}),
        'third_party_checkpoint_used': bool(
            (capsule.get('project_owned_boundary') or {}).get(
                'third_party_checkpoint_used'
            )
        ),
        'invalid_message_count': len(transcript.get('invalid_messages') or []),
        'no_sibling_imports': True,
        'project_owned_checkpoint_claimed': bool(
            (capsule.get('project_owned_boundary') or {}).get(
                'project_owned_checkpoint_verified'
            )
        ),
    }
    if output_file:
        _write_json_artifact(output_file, result)
    print(
        "AI_DIFFERENT_CROSS_MODULE_ADJUDICATOR "
        + json.dumps(result, sort_keys=True),
        flush=True,
    )
    return result


def run_experiment_agenda_scheduler(
    theory_memory: CumulativeTheoryMemory | None = None,
    *,
    theory_memory_file: str | Path | None = None,
    runtime_memory_path: str = 'tmp/theory-memory.json',
    transcript_file: str | Path | None = None,
    evaluator_ledger_file: str | Path = 'tmp/module-chat-outcome-evaluator-ledger.json',
    outcome_ledger_file: str | Path | None = None,
    contract_ledger_file: str | Path = 'tmp/module-chat-experiment-contract-ledger.json',
    adjudicator_ledger_file: str | Path = 'tmp/module-chat-cross-module-adjudicator-ledger.json',
    agenda_ledger_file: str | Path = 'tmp/module-chat-experiment-agenda-ledger.json',
    outbox_file: str | Path | None = None,
    output_file: str | Path | None = None,
    memory_data: dict | None = None,
    git_status_text: str | None = None,
    git_ignored_text: str | None = None,
) -> dict:
    """Schedule the next safe AI Different experiment agenda step."""
    if memory_data is not None:
        loaded_memory = dict(memory_data)
    elif theory_memory is not None:
        loaded_memory = theory_memory.to_dict()
    else:
        loaded_memory = load_capsule_memory_data(theory_memory_file)
    status_text = (
        git_status_text
        if git_status_text is not None
        else git_status_for_path(runtime_memory_path)
    )
    ignored_text = (
        git_ignored_text
        if git_ignored_text is not None
        else git_check_ignore_for_path(runtime_memory_path)
    )
    capsule = build_ai_different_status_capsule(
        loaded_memory,
        git_status_text=status_text,
        git_ignored_text=ignored_text,
        runtime_memory_path=runtime_memory_path,
    )
    transcript = read_agenda_transcript(transcript_file)
    evaluator_ledger = load_agenda_plain_json(evaluator_ledger_file)
    outcome_ledger = load_agenda_plain_json(outcome_ledger_file) if outcome_ledger_file else {}
    contract_ledger = load_agenda_plain_json(contract_ledger_file)
    adjudicator_ledger = load_agenda_plain_json(adjudicator_ledger_file)
    agenda_ledger = load_experiment_agenda_ledger(agenda_ledger_file)
    before_hash = _file_sha256(runtime_memory_path)
    runtime_hash_state = _runtime_memory_hash_state(runtime_memory_path, before_hash)
    updated_ledger, message = build_experiment_agenda(
        transcript_messages=list(transcript.get('messages') or []),
        agenda_ledger=agenda_ledger,
        evaluator_ledger=evaluator_ledger,
        outcome_ledger=outcome_ledger,
        contract_ledger=contract_ledger,
        adjudicator_ledger=adjudicator_ledger,
        runtime_memory_data=loaded_memory,
        runtime_memory_hash_state=runtime_hash_state,
        project_owned_boundary=dict(capsule.get('project_owned_boundary') or {}),
        artifact_path=agenda_ledger_file,
    )
    write_experiment_agenda_ledger(agenda_ledger_file, updated_ledger)
    if outbox_file:
        write_agenda_outbox_jsonl(outbox_file, message)
    latest = dict(updated_ledger.get('latest') or {})
    result = {
        'experiment_agenda_capability': True,
        'agenda_ledger_path': str(agenda_ledger_file),
        'agenda_ledger_hash': updated_ledger.get('ledger_hash'),
        'agenda_id': latest.get('agenda_id'),
        'processed_message_count': len(updated_ledger.get('processed_message_ids') or []),
        'new_message_count': int(latest.get('new_message_count', 0) or 0),
        'skipped_message_count': int(latest.get('skipped_message_count', 0) or 0),
        'open_gate_count': int(latest.get('open_gate_count', 0) or 0),
        'resolved_gate_count': int(latest.get('resolved_gate_count', 0) or 0),
        'blocked_gate_count': int(latest.get('blocked_gate_count', 0) or 0),
        'selected_action': latest.get('selected_action'),
        'chosen_recipient': latest.get('chosen_recipient'),
        'outbox_count': int(latest.get('outbox_count', 0) or 0),
        'outbox_file': str(outbox_file) if outbox_file else None,
        'response_message': message,
        'runtime_memory_hash_state': runtime_hash_state,
        'runtime_memory_mutated': not bool(runtime_hash_state.get('unchanged', True)),
        'label_leaks': list(latest.get('label_leaks') or []),
        'label_leaks_count': len(latest.get('label_leaks') or []),
        'project_owned_boundary': dict(capsule.get('project_owned_boundary') or {}),
        'third_party_checkpoint_used': bool(
            (capsule.get('project_owned_boundary') or {}).get(
                'third_party_checkpoint_used'
            )
        ),
        'invalid_message_count': len(transcript.get('invalid_messages') or []),
        'no_sibling_imports': True,
        'project_owned_checkpoint_claimed': bool(
            (capsule.get('project_owned_boundary') or {}).get(
                'project_owned_checkpoint_verified'
            )
        ),
    }
    if output_file:
        _write_json_artifact(output_file, result)
    print(
        "AI_DIFFERENT_EXPERIMENT_AGENDA "
        + json.dumps(result, sort_keys=True),
        flush=True,
    )
    return result


def run_hypothesis_lifecycle_curator(
    theory_memory: CumulativeTheoryMemory | None = None,
    *,
    theory_memory_file: str | Path | None = None,
    runtime_memory_path: str = 'tmp/theory-memory.json',
    transcript_file: str | Path | None = None,
    evaluator_ledger_file: str | Path = 'tmp/module-chat-outcome-evaluator-ledger.json',
    outcome_ledger_file: str | Path | None = None,
    contract_ledger_file: str | Path = 'tmp/module-chat-experiment-contract-ledger.json',
    adjudicator_ledger_file: str | Path = 'tmp/module-chat-cross-module-adjudicator-ledger.json',
    agenda_ledger_file: str | Path = 'tmp/module-chat-experiment-agenda-ledger.json',
    lifecycle_ledger_file: str | Path = 'tmp/module-chat-hypothesis-lifecycle-ledger.json',
    outbox_file: str | Path | None = None,
    output_file: str | Path | None = None,
    memory_data: dict | None = None,
    git_status_text: str | None = None,
    git_ignored_text: str | None = None,
) -> dict:
    """Curate science-side hypothesis lifecycle memory from module-family evidence."""
    if memory_data is not None:
        loaded_memory = dict(memory_data)
    elif theory_memory is not None:
        loaded_memory = theory_memory.to_dict()
    else:
        loaded_memory = load_capsule_memory_data(theory_memory_file)
    status_text = (
        git_status_text
        if git_status_text is not None
        else git_status_for_path(runtime_memory_path)
    )
    ignored_text = (
        git_ignored_text
        if git_ignored_text is not None
        else git_check_ignore_for_path(runtime_memory_path)
    )
    capsule = build_ai_different_status_capsule(
        loaded_memory,
        git_status_text=status_text,
        git_ignored_text=ignored_text,
        runtime_memory_path=runtime_memory_path,
    )
    transcript = read_lifecycle_transcript(transcript_file)
    evaluator_ledger = load_lifecycle_plain_json(evaluator_ledger_file)
    outcome_ledger = load_lifecycle_plain_json(outcome_ledger_file) if outcome_ledger_file else {}
    contract_ledger = load_lifecycle_plain_json(contract_ledger_file)
    adjudicator_ledger = load_lifecycle_plain_json(adjudicator_ledger_file)
    agenda_ledger = load_lifecycle_plain_json(agenda_ledger_file)
    lifecycle_ledger = load_hypothesis_lifecycle_ledger(lifecycle_ledger_file)
    before_hash = _file_sha256(runtime_memory_path)
    runtime_hash_state = _runtime_memory_hash_state(runtime_memory_path, before_hash)
    updated_ledger, message = build_hypothesis_lifecycle(
        transcript_messages=list(transcript.get('messages') or []),
        lifecycle_ledger=lifecycle_ledger,
        evaluator_ledger=evaluator_ledger,
        outcome_ledger=outcome_ledger,
        contract_ledger=contract_ledger,
        adjudicator_ledger=adjudicator_ledger,
        agenda_ledger=agenda_ledger,
        runtime_memory_data=loaded_memory,
        runtime_memory_hash_state=runtime_hash_state,
        project_owned_boundary=dict(capsule.get('project_owned_boundary') or {}),
        artifact_path=lifecycle_ledger_file,
    )
    write_hypothesis_lifecycle_ledger(lifecycle_ledger_file, updated_ledger)
    if outbox_file:
        write_lifecycle_outbox_jsonl(outbox_file, message)
    latest = dict(updated_ledger.get('latest') or {})
    state_counts = dict(latest.get('state_counts') or {})
    result = {
        'hypothesis_lifecycle_capability': True,
        'lifecycle_ledger_path': str(lifecycle_ledger_file),
        'lifecycle_ledger_hash': updated_ledger.get('ledger_hash'),
        'lifecycle_id': latest.get('lifecycle_id'),
        'processed_message_count': len(updated_ledger.get('processed_message_ids') or []),
        'new_message_count': int(latest.get('new_message_count', 0) or 0),
        'skipped_message_count': int(latest.get('skipped_message_count', 0) or 0),
        'proposed_hypothesis_count': int(state_counts.get('proposed', 0) or 0),
        'waiting_hypothesis_count': int(state_counts.get('waiting_evidence', 0) or 0),
        'blocked_hypothesis_count': int(state_counts.get('blocked', 0) or 0),
        'resolved_hypothesis_count': int(state_counts.get('resolved', 0) or 0),
        'retired_hypothesis_count': int(state_counts.get('retired', 0) or 0),
        'refine_next_hypothesis_count': int(state_counts.get('refine_next', 0) or 0),
        'selected_action': latest.get('selected_action'),
        'chosen_recipient': latest.get('chosen_recipient'),
        'outbox_count': int(latest.get('outbox_count', 0) or 0),
        'outbox_file': str(outbox_file) if outbox_file else None,
        'response_message': message,
        'runtime_memory_hash_state': runtime_hash_state,
        'runtime_memory_mutated': not bool(runtime_hash_state.get('unchanged', True)),
        'label_leaks': list(latest.get('label_leaks') or []),
        'label_leaks_count': len(latest.get('label_leaks') or []),
        'project_owned_boundary': dict(capsule.get('project_owned_boundary') or {}),
        'third_party_checkpoint_used': bool(
            (capsule.get('project_owned_boundary') or {}).get(
                'third_party_checkpoint_used'
            )
        ),
        'invalid_message_count': len(transcript.get('invalid_messages') or []),
        'no_sibling_imports': True,
        'project_owned_checkpoint_claimed': bool(
            (capsule.get('project_owned_boundary') or {}).get(
                'project_owned_checkpoint_verified'
            )
        ),
    }
    if output_file:
        _write_json_artifact(output_file, result)
    print(
        "AI_DIFFERENT_HYPOTHESIS_LIFECYCLE "
        + json.dumps(result, sort_keys=True),
        flush=True,
    )
    return result


def run_evidence_scorecard_runner(
    theory_memory: CumulativeTheoryMemory | None = None,
    *,
    theory_memory_file: str | Path | None = None,
    runtime_memory_path: str = 'tmp/theory-memory.json',
    transcript_file: str | Path | None = None,
    evaluator_ledger_file: str | Path = 'tmp/module-chat-outcome-evaluator-ledger.json',
    outcome_ledger_file: str | Path | None = None,
    contract_ledger_file: str | Path = 'tmp/module-chat-experiment-contract-ledger.json',
    adjudicator_ledger_file: str | Path = 'tmp/module-chat-cross-module-adjudicator-ledger.json',
    agenda_ledger_file: str | Path = 'tmp/module-chat-experiment-agenda-ledger.json',
    lifecycle_ledger_file: str | Path = 'tmp/module-chat-hypothesis-lifecycle-ledger.json',
    scorecard_ledger_file: str | Path = 'tmp/module-chat-evidence-scorecard-ledger.json',
    outbox_file: str | Path | None = None,
    output_file: str | Path | None = None,
    memory_data: dict | None = None,
    git_status_text: str | None = None,
    git_ignored_text: str | None = None,
) -> dict:
    """Score hypothesis evidence gates and choose one refinement/repair action."""
    if memory_data is not None:
        loaded_memory = dict(memory_data)
    elif theory_memory is not None:
        loaded_memory = theory_memory.to_dict()
    else:
        loaded_memory = load_capsule_memory_data(theory_memory_file)
    status_text = (
        git_status_text
        if git_status_text is not None
        else git_status_for_path(runtime_memory_path)
    )
    ignored_text = (
        git_ignored_text
        if git_ignored_text is not None
        else git_check_ignore_for_path(runtime_memory_path)
    )
    capsule = build_ai_different_status_capsule(
        loaded_memory,
        git_status_text=status_text,
        git_ignored_text=ignored_text,
        runtime_memory_path=runtime_memory_path,
    )
    transcript = read_scorecard_transcript(transcript_file)
    evaluator_ledger = load_scorecard_plain_json(evaluator_ledger_file)
    outcome_ledger = load_scorecard_plain_json(outcome_ledger_file) if outcome_ledger_file else {}
    contract_ledger = load_scorecard_plain_json(contract_ledger_file)
    adjudicator_ledger = load_scorecard_plain_json(adjudicator_ledger_file)
    agenda_ledger = load_scorecard_plain_json(agenda_ledger_file)
    lifecycle_ledger = load_scorecard_plain_json(lifecycle_ledger_file)
    scorecard_ledger = load_scorecard_ledger(scorecard_ledger_file)
    before_hash = _file_sha256(runtime_memory_path)
    runtime_hash_state = _runtime_memory_hash_state(runtime_memory_path, before_hash)
    updated_ledger, message = build_evidence_scorecard(
        transcript_messages=list(transcript.get('messages') or []),
        scorecard_ledger=scorecard_ledger,
        evaluator_ledger=evaluator_ledger,
        outcome_ledger=outcome_ledger,
        contract_ledger=contract_ledger,
        adjudicator_ledger=adjudicator_ledger,
        agenda_ledger=agenda_ledger,
        lifecycle_ledger=lifecycle_ledger,
        runtime_memory_data=loaded_memory,
        runtime_memory_hash_state=runtime_hash_state,
        project_owned_boundary=dict(capsule.get('project_owned_boundary') or {}),
        artifact_path=scorecard_ledger_file,
    )
    write_scorecard_ledger(scorecard_ledger_file, updated_ledger)
    if outbox_file:
        write_scorecard_outbox_jsonl(outbox_file, message)
    latest = dict(updated_ledger.get('latest') or {})
    readiness_counts = dict(latest.get('readiness_counts') or {})
    result = {
        'evidence_scorecard_capability': True,
        'scorecard_ledger_path': str(scorecard_ledger_file),
        'scorecard_ledger_hash': updated_ledger.get('ledger_hash'),
        'scorecard_id': latest.get('scorecard_id'),
        'processed_message_count': len(updated_ledger.get('processed_message_ids') or []),
        'new_message_count': int(latest.get('new_message_count', 0) or 0),
        'skipped_message_count': int(latest.get('skipped_message_count', 0) or 0),
        'waiting_scorecard_count': int(readiness_counts.get('waiting', 0) or 0),
        'resolved_scorecard_count': int(readiness_counts.get('resolved', 0) or 0),
        'retired_scorecard_count': int(readiness_counts.get('retired', 0) or 0),
        'refine_scorecard_count': int(readiness_counts.get('refine', 0) or 0),
        'repair_scorecard_count': int(readiness_counts.get('repair', 0) or 0),
        'selected_action': latest.get('selected_action'),
        'chosen_recipient': latest.get('chosen_recipient'),
        'outbox_count': int(latest.get('outbox_count', 0) or 0),
        'outbox_file': str(outbox_file) if outbox_file else None,
        'response_message': message,
        'runtime_memory_hash_state': runtime_hash_state,
        'runtime_memory_mutated': not bool(runtime_hash_state.get('unchanged', True)),
        'label_leaks': list(latest.get('label_leaks') or []),
        'label_leaks_count': len(latest.get('label_leaks') or []),
        'project_owned_boundary': dict(capsule.get('project_owned_boundary') or {}),
        'third_party_checkpoint_used': bool(
            (capsule.get('project_owned_boundary') or {}).get(
                'third_party_checkpoint_used'
            )
        ),
        'invalid_message_count': len(transcript.get('invalid_messages') or []),
        'no_sibling_imports': True,
        'project_owned_checkpoint_claimed': bool(
            (capsule.get('project_owned_boundary') or {}).get(
                'project_owned_checkpoint_verified'
            )
        ),
    }
    if output_file:
        _write_json_artifact(output_file, result)
    print(
        "AI_DIFFERENT_EVIDENCE_SCORECARD "
        + json.dumps(result, sort_keys=True),
        flush=True,
    )
    return result


def run_experiment_campaign_planner(
    theory_memory: CumulativeTheoryMemory | None = None,
    *,
    theory_memory_file: str | Path | None = None,
    runtime_memory_path: str = 'tmp/theory-memory.json',
    transcript_file: str | Path | None = None,
    evaluator_ledger_file: str | Path = 'tmp/module-chat-outcome-evaluator-ledger.json',
    outcome_ledger_file: str | Path | None = None,
    contract_ledger_file: str | Path = 'tmp/module-chat-experiment-contract-ledger.json',
    adjudicator_ledger_file: str | Path = 'tmp/module-chat-cross-module-adjudicator-ledger.json',
    agenda_ledger_file: str | Path = 'tmp/module-chat-experiment-agenda-ledger.json',
    lifecycle_ledger_file: str | Path = 'tmp/module-chat-hypothesis-lifecycle-ledger.json',
    scorecard_ledger_file: str | Path = 'tmp/module-chat-evidence-scorecard-ledger.json',
    campaign_ledger_file: str | Path = 'tmp/module-chat-experiment-campaign-ledger.json',
    outbox_file: str | Path | None = None,
    output_file: str | Path | None = None,
    memory_data: dict | None = None,
    git_status_text: str | None = None,
    git_ignored_text: str | None = None,
) -> dict:
    """Plan the next symbolic campaign or acceptance bundle."""
    if memory_data is not None:
        loaded_memory = dict(memory_data)
    elif theory_memory is not None:
        loaded_memory = theory_memory.to_dict()
    else:
        loaded_memory = load_capsule_memory_data(theory_memory_file)
    status_text = (
        git_status_text
        if git_status_text is not None
        else git_status_for_path(runtime_memory_path)
    )
    ignored_text = (
        git_ignored_text
        if git_ignored_text is not None
        else git_check_ignore_for_path(runtime_memory_path)
    )
    capsule = build_ai_different_status_capsule(
        loaded_memory,
        git_status_text=status_text,
        git_ignored_text=ignored_text,
        runtime_memory_path=runtime_memory_path,
    )
    transcript = read_campaign_transcript(transcript_file)
    evaluator_ledger = load_campaign_plain_json(evaluator_ledger_file)
    outcome_ledger = load_campaign_plain_json(outcome_ledger_file) if outcome_ledger_file else {}
    contract_ledger = load_campaign_plain_json(contract_ledger_file)
    adjudicator_ledger = load_campaign_plain_json(adjudicator_ledger_file)
    agenda_ledger = load_campaign_plain_json(agenda_ledger_file)
    lifecycle_ledger = load_campaign_plain_json(lifecycle_ledger_file)
    scorecard_ledger = load_campaign_plain_json(scorecard_ledger_file)
    campaign_ledger = load_campaign_ledger(campaign_ledger_file)
    before_hash = _file_sha256(runtime_memory_path)
    runtime_hash_state = _runtime_memory_hash_state(runtime_memory_path, before_hash)
    updated_ledger, message = build_experiment_campaign(
        transcript_messages=list(transcript.get('messages') or []),
        campaign_ledger=campaign_ledger,
        evaluator_ledger=evaluator_ledger,
        outcome_ledger=outcome_ledger,
        contract_ledger=contract_ledger,
        adjudicator_ledger=adjudicator_ledger,
        agenda_ledger=agenda_ledger,
        lifecycle_ledger=lifecycle_ledger,
        scorecard_ledger=scorecard_ledger,
        runtime_memory_data=loaded_memory,
        runtime_memory_hash_state=runtime_hash_state,
        project_owned_boundary=dict(capsule.get('project_owned_boundary') or {}),
        artifact_path=campaign_ledger_file,
    )
    write_campaign_ledger(campaign_ledger_file, updated_ledger)
    if outbox_file:
        write_campaign_outbox_jsonl(outbox_file, message)
    latest = dict(updated_ledger.get('latest') or {})
    type_counts = dict(latest.get('campaign_type_counts') or {})
    result = {
        'campaign_planner_capability': True,
        'campaign_ledger_path': str(campaign_ledger_file),
        'campaign_ledger_hash': updated_ledger.get('ledger_hash'),
        'campaign_id': latest.get('campaign_id'),
        'processed_message_count': len(updated_ledger.get('processed_message_ids') or []),
        'new_message_count': int(latest.get('new_message_count', 0) or 0),
        'skipped_message_count': int(latest.get('skipped_message_count', 0) or 0),
        'campaign_type_counts': type_counts,
        'selected_action': latest.get('selected_action'),
        'chosen_recipient': latest.get('chosen_recipient'),
        'outbox_count': int(latest.get('outbox_count', 0) or 0),
        'outbox_file': str(outbox_file) if outbox_file else None,
        'response_message': message,
        'runtime_memory_hash_state': runtime_hash_state,
        'runtime_memory_mutated': not bool(runtime_hash_state.get('unchanged', True)),
        'label_leaks': list(latest.get('label_leaks') or []),
        'label_leaks_count': len(latest.get('label_leaks') or []),
        'project_owned_boundary': dict(capsule.get('project_owned_boundary') or {}),
        'third_party_checkpoint_used': bool(
            (capsule.get('project_owned_boundary') or {}).get(
                'third_party_checkpoint_used'
            )
        ),
        'invalid_message_count': len(transcript.get('invalid_messages') or []),
        'no_sibling_imports': True,
        'project_owned_checkpoint_claimed': bool(
            (capsule.get('project_owned_boundary') or {}).get(
                'project_owned_checkpoint_verified'
            )
        ),
    }
    if output_file:
        _write_json_artifact(output_file, result)
    print(
        "AI_DIFFERENT_EXPERIMENT_CAMPAIGN "
        + json.dumps(result, sort_keys=True),
        flush=True,
    )
    return result


def run_campaign_outcome_assessor(
    theory_memory: CumulativeTheoryMemory | None = None,
    *,
    theory_memory_file: str | Path | None = None,
    runtime_memory_path: str = 'tmp/theory-memory.json',
    transcript_file: str | Path | None = None,
    evaluator_ledger_file: str | Path = 'tmp/module-chat-outcome-evaluator-ledger.json',
    outcome_ledger_file: str | Path | None = None,
    contract_ledger_file: str | Path = 'tmp/module-chat-experiment-contract-ledger.json',
    adjudicator_ledger_file: str | Path = 'tmp/module-chat-cross-module-adjudicator-ledger.json',
    agenda_ledger_file: str | Path = 'tmp/module-chat-experiment-agenda-ledger.json',
    lifecycle_ledger_file: str | Path = 'tmp/module-chat-hypothesis-lifecycle-ledger.json',
    scorecard_ledger_file: str | Path = 'tmp/module-chat-evidence-scorecard-ledger.json',
    campaign_ledger_file: str | Path = 'tmp/module-chat-experiment-campaign-ledger.json',
    campaign_outcome_ledger_file: str | Path = 'tmp/module-chat-experiment-campaign-outcome-ledger.json',
    outbox_file: str | Path | None = None,
    output_file: str | Path | None = None,
    memory_data: dict | None = None,
    git_status_text: str | None = None,
    git_ignored_text: str | None = None,
) -> dict:
    """Assess campaign return evidence and plan a symbolic theory update."""
    if memory_data is not None:
        loaded_memory = dict(memory_data)
    elif theory_memory is not None:
        loaded_memory = theory_memory.to_dict()
    else:
        loaded_memory = load_capsule_memory_data(theory_memory_file)
    status_text = (
        git_status_text
        if git_status_text is not None
        else git_status_for_path(runtime_memory_path)
    )
    ignored_text = (
        git_ignored_text
        if git_ignored_text is not None
        else git_check_ignore_for_path(runtime_memory_path)
    )
    capsule = build_ai_different_status_capsule(
        loaded_memory,
        git_status_text=status_text,
        git_ignored_text=ignored_text,
        runtime_memory_path=runtime_memory_path,
    )
    transcript = read_campaign_outcome_transcript(transcript_file)
    evaluator_ledger = load_campaign_outcome_plain_json(evaluator_ledger_file)
    outcome_ledger = load_campaign_outcome_plain_json(outcome_ledger_file) if outcome_ledger_file else {}
    contract_ledger = load_campaign_outcome_plain_json(contract_ledger_file)
    adjudicator_ledger = load_campaign_outcome_plain_json(adjudicator_ledger_file)
    agenda_ledger = load_campaign_outcome_plain_json(agenda_ledger_file)
    lifecycle_ledger = load_campaign_outcome_plain_json(lifecycle_ledger_file)
    scorecard_ledger = load_campaign_outcome_plain_json(scorecard_ledger_file)
    campaign_ledger = load_campaign_outcome_plain_json(campaign_ledger_file)
    campaign_outcome_ledger = load_campaign_outcome_ledger(campaign_outcome_ledger_file)
    before_hash = _file_sha256(runtime_memory_path)
    runtime_hash_state = _runtime_memory_hash_state(runtime_memory_path, before_hash)
    updated_ledger, message = build_campaign_outcome_assessment(
        transcript_messages=list(transcript.get('messages') or []),
        campaign_outcome_ledger=campaign_outcome_ledger,
        evaluator_ledger=evaluator_ledger,
        outcome_ledger=outcome_ledger,
        contract_ledger=contract_ledger,
        adjudicator_ledger=adjudicator_ledger,
        agenda_ledger=agenda_ledger,
        lifecycle_ledger=lifecycle_ledger,
        scorecard_ledger=scorecard_ledger,
        campaign_ledger=campaign_ledger,
        runtime_memory_data=loaded_memory,
        runtime_memory_hash_state=runtime_hash_state,
        project_owned_boundary=dict(capsule.get('project_owned_boundary') or {}),
        artifact_path=campaign_outcome_ledger_file,
    )
    write_campaign_outcome_ledger(campaign_outcome_ledger_file, updated_ledger)
    if outbox_file:
        write_campaign_outcome_outbox_jsonl(outbox_file, message)
    latest = dict(updated_ledger.get('latest') or {})
    status_counts = dict(latest.get('status_counts') or {})
    result = {
        'campaign_outcome_capability': True,
        'campaign_outcome_ledger_path': str(campaign_outcome_ledger_file),
        'campaign_outcome_ledger_hash': updated_ledger.get('ledger_hash'),
        'outcome_id': latest.get('outcome_id'),
        'processed_message_count': len(updated_ledger.get('processed_message_ids') or []),
        'new_message_count': int(latest.get('new_message_count', 0) or 0),
        'skipped_message_count': int(latest.get('skipped_message_count', 0) or 0),
        'accepted_count': int(status_counts.get('accepted', 0) or 0),
        'request_count': int(status_counts.get('request', 0) or 0),
        'failed_count': int(status_counts.get('failed', 0) or 0),
        'refined_count': int(status_counts.get('refined', 0) or 0),
        'retired_count': int(status_counts.get('retired', 0) or 0),
        'repair_count': int(status_counts.get('repair', 0) or 0),
        'selected_action': latest.get('selected_action'),
        'chosen_recipient': latest.get('chosen_recipient'),
        'outbox_count': int(latest.get('outbox_count', 0) or 0),
        'outbox_file': str(outbox_file) if outbox_file else None,
        'response_message': message,
        'runtime_memory_hash_state': runtime_hash_state,
        'runtime_memory_mutated': not bool(runtime_hash_state.get('unchanged', True)),
        'label_leaks': list(latest.get('label_leaks') or []),
        'label_leaks_count': len(latest.get('label_leaks') or []),
        'project_owned_boundary': dict(capsule.get('project_owned_boundary') or {}),
        'third_party_checkpoint_used': bool(
            (capsule.get('project_owned_boundary') or {}).get(
                'third_party_checkpoint_used'
            )
        ),
        'invalid_message_count': len(transcript.get('invalid_messages') or []),
        'no_sibling_imports': True,
        'project_owned_checkpoint_claimed': bool(
            (capsule.get('project_owned_boundary') or {}).get(
                'project_owned_checkpoint_verified'
            )
        ),
    }
    if output_file:
        _write_json_artifact(output_file, result)
    print(
        "AI_DIFFERENT_CAMPAIGN_OUTCOME "
        + json.dumps(result, sort_keys=True),
        flush=True,
    )
    return result


def run_science_benefit_evaluator(
    theory_memory: CumulativeTheoryMemory | None = None,
    *,
    theory_memory_file: str | Path | None = None,
    runtime_memory_path: str = 'tmp/theory-memory.json',
    transcript_file: str | Path | None = None,
    evaluator_ledger_file: str | Path = 'tmp/module-chat-outcome-evaluator-ledger.json',
    outcome_ledger_file: str | Path | None = None,
    contract_ledger_file: str | Path = 'tmp/module-chat-experiment-contract-ledger.json',
    adjudicator_ledger_file: str | Path = 'tmp/module-chat-cross-module-adjudicator-ledger.json',
    agenda_ledger_file: str | Path = 'tmp/module-chat-experiment-agenda-ledger.json',
    lifecycle_ledger_file: str | Path = 'tmp/module-chat-hypothesis-lifecycle-ledger.json',
    scorecard_ledger_file: str | Path = 'tmp/module-chat-evidence-scorecard-ledger.json',
    campaign_ledger_file: str | Path = 'tmp/module-chat-experiment-campaign-ledger.json',
    campaign_outcome_ledger_file: str | Path = 'tmp/module-chat-experiment-campaign-outcome-ledger.json',
    benefit_ledger_file: str | Path = 'tmp/module-chat-science-benefit-ledger.json',
    prior_benefit_ledger_file: str | Path | None = None,
    outbox_file: str | Path | None = None,
    output_file: str | Path | None = None,
    memory_data: dict | None = None,
    git_status_text: str | None = None,
    git_ignored_text: str | None = None,
) -> dict:
    """Compare isolated and connected symbolic campaign-management evidence."""
    if memory_data is not None:
        loaded_memory = dict(memory_data)
    elif theory_memory is not None:
        loaded_memory = theory_memory.to_dict()
    else:
        loaded_memory = load_capsule_memory_data(theory_memory_file)
    status_text = (
        git_status_text
        if git_status_text is not None
        else git_status_for_path(runtime_memory_path)
    )
    ignored_text = (
        git_ignored_text
        if git_ignored_text is not None
        else git_check_ignore_for_path(runtime_memory_path)
    )
    capsule = build_ai_different_status_capsule(
        loaded_memory,
        git_status_text=status_text,
        git_ignored_text=ignored_text,
        runtime_memory_path=runtime_memory_path,
    )
    transcript = read_science_benefit_transcript(transcript_file)
    evaluator_ledger = load_science_benefit_plain_json(evaluator_ledger_file)
    outcome_ledger = load_science_benefit_plain_json(outcome_ledger_file) if outcome_ledger_file else {}
    contract_ledger = load_science_benefit_plain_json(contract_ledger_file)
    adjudicator_ledger = load_science_benefit_plain_json(adjudicator_ledger_file)
    agenda_ledger = load_science_benefit_plain_json(agenda_ledger_file)
    lifecycle_ledger = load_science_benefit_plain_json(lifecycle_ledger_file)
    scorecard_ledger = load_science_benefit_plain_json(scorecard_ledger_file)
    campaign_ledger = load_science_benefit_plain_json(campaign_ledger_file)
    campaign_outcome_ledger = load_science_benefit_plain_json(campaign_outcome_ledger_file)
    benefit_ledger = load_science_benefit_ledger(benefit_ledger_file)
    prior_benefit_ledger = (
        load_science_benefit_plain_json(prior_benefit_ledger_file)
        if prior_benefit_ledger_file
        else {}
    )
    before_hash = _file_sha256(runtime_memory_path)
    runtime_hash_state = _runtime_memory_hash_state(runtime_memory_path, before_hash)
    updated_ledger, message = build_science_benefit_evaluation(
        transcript_messages=list(transcript.get('messages') or []),
        benefit_ledger=benefit_ledger,
        evaluator_ledger=evaluator_ledger,
        outcome_ledger=outcome_ledger,
        contract_ledger=contract_ledger,
        adjudicator_ledger=adjudicator_ledger,
        agenda_ledger=agenda_ledger,
        lifecycle_ledger=lifecycle_ledger,
        scorecard_ledger=scorecard_ledger,
        campaign_ledger=campaign_ledger,
        campaign_outcome_ledger=campaign_outcome_ledger,
        prior_benefit_ledger=prior_benefit_ledger,
        runtime_memory_data=loaded_memory,
        runtime_memory_hash_state=runtime_hash_state,
        project_owned_boundary=dict(capsule.get('project_owned_boundary') or {}),
        artifact_path=benefit_ledger_file,
    )
    write_science_benefit_ledger(benefit_ledger_file, updated_ledger)
    if outbox_file:
        write_science_benefit_outbox_jsonl(outbox_file, message)
    latest = dict(updated_ledger.get('latest') or {})
    status_counts = dict(latest.get('status_counts') or {})
    result = {
        'science_benefit_capability': True,
        'benefit_ledger_path': str(benefit_ledger_file),
        'benefit_ledger_hash': updated_ledger.get('ledger_hash'),
        'benefit_id': latest.get('benefit_id'),
        'processed_message_count': len(updated_ledger.get('processed_message_ids') or []),
        'new_message_count': int(latest.get('new_message_count', 0) or 0),
        'skipped_message_count': int(latest.get('skipped_message_count', 0) or 0),
        'benefit_count': int(status_counts.get('benefit', 0) or 0),
        'no_benefit_count': int(status_counts.get('no_benefit', 0) or 0),
        'insufficient_count': int(status_counts.get('insufficient', 0) or 0),
        'repair_count': int(status_counts.get('repair', 0) or 0),
        'benefit_classification': latest.get('benefit_classification'),
        'selected_action': latest.get('selected_action'),
        'chosen_recipient': latest.get('chosen_recipient'),
        'outbox_count': int(latest.get('outbox_count', 0) or 0),
        'outbox_file': str(outbox_file) if outbox_file else None,
        'response_message': message,
        'runtime_memory_hash_state': runtime_hash_state,
        'runtime_memory_mutated': not bool(runtime_hash_state.get('unchanged', True)),
        'label_leaks': list(latest.get('label_leaks') or []),
        'label_leaks_count': len(latest.get('label_leaks') or []),
        'project_owned_boundary': dict(capsule.get('project_owned_boundary') or {}),
        'third_party_checkpoint_used': bool(
            (capsule.get('project_owned_boundary') or {}).get(
                'third_party_checkpoint_used'
            )
        ),
        'invalid_message_count': len(transcript.get('invalid_messages') or []),
        'no_sibling_imports': True,
        'project_owned_checkpoint_claimed': bool(
            (capsule.get('project_owned_boundary') or {}).get(
                'project_owned_checkpoint_verified'
            )
        ),
    }
    if output_file:
        _write_json_artifact(output_file, result)
    print(
        "AI_DIFFERENT_SCIENCE_BENEFIT "
        + json.dumps(result, sort_keys=True),
        flush=True,
    )
    return result


def run_science_campaign_action_planner(
    theory_memory: CumulativeTheoryMemory | None = None,
    *,
    theory_memory_file: str | Path | None = None,
    runtime_memory_path: str = 'tmp/theory-memory.json',
    transcript_file: str | Path | None = None,
    benefit_ledger_file: str | Path = 'tmp/module-chat-science-benefit-ledger.json',
    evaluator_ledger_file: str | Path = 'tmp/module-chat-outcome-evaluator-ledger.json',
    outcome_ledger_file: str | Path | None = None,
    contract_ledger_file: str | Path = 'tmp/module-chat-experiment-contract-ledger.json',
    adjudicator_ledger_file: str | Path = 'tmp/module-chat-cross-module-adjudicator-ledger.json',
    agenda_ledger_file: str | Path = 'tmp/module-chat-experiment-agenda-ledger.json',
    lifecycle_ledger_file: str | Path = 'tmp/module-chat-hypothesis-lifecycle-ledger.json',
    scorecard_ledger_file: str | Path = 'tmp/module-chat-evidence-scorecard-ledger.json',
    campaign_ledger_file: str | Path = 'tmp/module-chat-experiment-campaign-ledger.json',
    campaign_outcome_ledger_file: str | Path = 'tmp/module-chat-experiment-campaign-outcome-ledger.json',
    action_ledger_file: str | Path = 'tmp/module-chat-science-campaign-action-ledger.json',
    prior_action_ledger_file: str | Path | None = None,
    outbox_file: str | Path | None = None,
    output_file: str | Path | None = None,
    memory_data: dict | None = None,
    git_status_text: str | None = None,
    git_ignored_text: str | None = None,
) -> dict:
    """Turn science-benefit records into one safe next campaign action."""
    if memory_data is not None:
        loaded_memory = dict(memory_data)
    elif theory_memory is not None:
        loaded_memory = theory_memory.to_dict()
    else:
        loaded_memory = load_capsule_memory_data(theory_memory_file)
    status_text = (
        git_status_text
        if git_status_text is not None
        else git_status_for_path(runtime_memory_path)
    )
    ignored_text = (
        git_ignored_text
        if git_ignored_text is not None
        else git_check_ignore_for_path(runtime_memory_path)
    )
    capsule = build_ai_different_status_capsule(
        loaded_memory,
        git_status_text=status_text,
        git_ignored_text=ignored_text,
        runtime_memory_path=runtime_memory_path,
    )
    transcript = read_science_action_transcript(transcript_file)
    benefit_ledger = load_science_action_plain_json(benefit_ledger_file)
    evaluator_ledger = load_science_action_plain_json(evaluator_ledger_file)
    outcome_ledger = load_science_action_plain_json(outcome_ledger_file) if outcome_ledger_file else {}
    contract_ledger = load_science_action_plain_json(contract_ledger_file)
    adjudicator_ledger = load_science_action_plain_json(adjudicator_ledger_file)
    agenda_ledger = load_science_action_plain_json(agenda_ledger_file)
    lifecycle_ledger = load_science_action_plain_json(lifecycle_ledger_file)
    scorecard_ledger = load_science_action_plain_json(scorecard_ledger_file)
    campaign_ledger = load_science_action_plain_json(campaign_ledger_file)
    campaign_outcome_ledger = load_science_action_plain_json(campaign_outcome_ledger_file)
    action_ledger = load_science_action_ledger(action_ledger_file)
    prior_action_ledger = (
        load_science_action_plain_json(prior_action_ledger_file)
        if prior_action_ledger_file
        else {}
    )
    before_hash = _file_sha256(runtime_memory_path)
    runtime_hash_state = _runtime_memory_hash_state(runtime_memory_path, before_hash)
    updated_ledger, message = build_science_campaign_action_plan(
        transcript_messages=list(transcript.get('messages') or []),
        action_ledger=action_ledger,
        benefit_ledger=benefit_ledger,
        evaluator_ledger=evaluator_ledger,
        outcome_ledger=outcome_ledger,
        contract_ledger=contract_ledger,
        adjudicator_ledger=adjudicator_ledger,
        agenda_ledger=agenda_ledger,
        lifecycle_ledger=lifecycle_ledger,
        scorecard_ledger=scorecard_ledger,
        campaign_ledger=campaign_ledger,
        campaign_outcome_ledger=campaign_outcome_ledger,
        prior_action_ledger=prior_action_ledger,
        runtime_memory_data=loaded_memory,
        runtime_memory_hash_state=runtime_hash_state,
        project_owned_boundary=dict(capsule.get('project_owned_boundary') or {}),
        artifact_path=action_ledger_file,
    )
    write_science_action_ledger(action_ledger_file, updated_ledger)
    if outbox_file:
        write_science_action_outbox_jsonl(outbox_file, message)
    latest = dict(updated_ledger.get('latest') or {})
    state_counts = dict(latest.get('state_counts') or {})
    result = {
        'science_campaign_action_capability': True,
        'action_ledger_path': str(action_ledger_file),
        'action_ledger_hash': updated_ledger.get('ledger_hash'),
        'action_id': latest.get('action_id'),
        'processed_message_count': len(updated_ledger.get('processed_message_ids') or []),
        'new_message_count': int(latest.get('new_message_count', 0) or 0),
        'skipped_message_count': int(latest.get('skipped_message_count', 0) or 0),
        'repair_count': int(state_counts.get('repair', 0) or 0),
        'retire_count': int(state_counts.get('retire', 0) or 0),
        'math_count': int(state_counts.get('math', 0) or 0),
        'code_count': int(state_counts.get('code', 0) or 0),
        'language_count': int(state_counts.get('language', 0) or 0),
        'refine_count': int(state_counts.get('refine', 0) or 0),
        'next_count': int(state_counts.get('next', 0) or 0),
        'no_benefit_count': int(state_counts.get('no_benefit', 0) or 0),
        'selected_action': latest.get('selected_action'),
        'chosen_recipient': latest.get('chosen_recipient'),
        'outbox_count': int(latest.get('outbox_count', 0) or 0),
        'outbox_file': str(outbox_file) if outbox_file else None,
        'response_message': message,
        'runtime_memory_hash_state': runtime_hash_state,
        'runtime_memory_mutated': not bool(runtime_hash_state.get('unchanged', True)),
        'label_leaks': list(latest.get('label_leaks') or []),
        'label_leaks_count': len(latest.get('label_leaks') or []),
        'project_owned_boundary': dict(capsule.get('project_owned_boundary') or {}),
        'third_party_checkpoint_used': bool(
            (capsule.get('project_owned_boundary') or {}).get(
                'third_party_checkpoint_used'
            )
        ),
        'invalid_message_count': len(transcript.get('invalid_messages') or []),
        'no_sibling_imports': True,
        'project_owned_checkpoint_claimed': bool(
            (capsule.get('project_owned_boundary') or {}).get(
                'project_owned_checkpoint_verified'
            )
        ),
    }
    if output_file:
        _write_json_artifact(output_file, result)
    print(
        "AI_DIFFERENT_SCIENCE_ACTION "
        + json.dumps(result, sort_keys=True),
        flush=True,
    )
    return result


def run_science_action_outcome_assessor(
    theory_memory: CumulativeTheoryMemory | None = None,
    *,
    theory_memory_file: str | Path | None = None,
    runtime_memory_path: str = 'tmp/theory-memory.json',
    transcript_file: str | Path | None = None,
    action_ledger_file: str | Path = 'tmp/module-chat-science-campaign-action-ledger.json',
    benefit_ledger_file: str | Path = 'tmp/module-chat-science-benefit-ledger.json',
    evaluator_ledger_file: str | Path = 'tmp/module-chat-outcome-evaluator-ledger.json',
    outcome_ledger_file: str | Path | None = None,
    contract_ledger_file: str | Path = 'tmp/module-chat-experiment-contract-ledger.json',
    adjudicator_ledger_file: str | Path = 'tmp/module-chat-cross-module-adjudicator-ledger.json',
    agenda_ledger_file: str | Path = 'tmp/module-chat-experiment-agenda-ledger.json',
    lifecycle_ledger_file: str | Path = 'tmp/module-chat-hypothesis-lifecycle-ledger.json',
    scorecard_ledger_file: str | Path = 'tmp/module-chat-evidence-scorecard-ledger.json',
    campaign_ledger_file: str | Path = 'tmp/module-chat-experiment-campaign-ledger.json',
    campaign_outcome_ledger_file: str | Path = 'tmp/module-chat-experiment-campaign-outcome-ledger.json',
    action_outcome_ledger_file: str | Path = 'tmp/module-chat-science-action-outcome-ledger.json',
    prior_action_outcome_ledger_file: str | Path | None = None,
    outbox_file: str | Path | None = None,
    output_file: str | Path | None = None,
    memory_data: dict | None = None,
    git_status_text: str | None = None,
    git_ignored_text: str | None = None,
) -> dict:
    """Assess whether science campaign actions received useful evidence."""
    if memory_data is not None:
        loaded_memory = dict(memory_data)
    elif theory_memory is not None:
        loaded_memory = theory_memory.to_dict()
    else:
        loaded_memory = load_capsule_memory_data(theory_memory_file)
    status_text = (
        git_status_text
        if git_status_text is not None
        else git_status_for_path(runtime_memory_path)
    )
    ignored_text = (
        git_ignored_text
        if git_ignored_text is not None
        else git_check_ignore_for_path(runtime_memory_path)
    )
    capsule = build_ai_different_status_capsule(
        loaded_memory,
        git_status_text=status_text,
        git_ignored_text=ignored_text,
        runtime_memory_path=runtime_memory_path,
    )
    transcript = read_science_action_outcome_transcript(transcript_file)
    action_ledger = load_science_action_outcome_plain_json(action_ledger_file)
    benefit_ledger = load_science_action_outcome_plain_json(benefit_ledger_file)
    evaluator_ledger = load_science_action_outcome_plain_json(evaluator_ledger_file)
    outcome_ledger = load_science_action_outcome_plain_json(outcome_ledger_file) if outcome_ledger_file else {}
    contract_ledger = load_science_action_outcome_plain_json(contract_ledger_file)
    adjudicator_ledger = load_science_action_outcome_plain_json(adjudicator_ledger_file)
    agenda_ledger = load_science_action_outcome_plain_json(agenda_ledger_file)
    lifecycle_ledger = load_science_action_outcome_plain_json(lifecycle_ledger_file)
    scorecard_ledger = load_science_action_outcome_plain_json(scorecard_ledger_file)
    campaign_ledger = load_science_action_outcome_plain_json(campaign_ledger_file)
    campaign_outcome_ledger = load_science_action_outcome_plain_json(campaign_outcome_ledger_file)
    action_outcome_ledger = load_science_action_outcome_ledger(action_outcome_ledger_file)
    prior_action_outcome_ledger = (
        load_science_action_outcome_plain_json(prior_action_outcome_ledger_file)
        if prior_action_outcome_ledger_file
        else {}
    )
    before_hash = _file_sha256(runtime_memory_path)
    runtime_hash_state = _runtime_memory_hash_state(runtime_memory_path, before_hash)
    updated_ledger, message = build_science_action_outcome_assessment(
        transcript_messages=list(transcript.get('messages') or []),
        action_outcome_ledger=action_outcome_ledger,
        action_ledger=action_ledger,
        benefit_ledger=benefit_ledger,
        evaluator_ledger=evaluator_ledger,
        outcome_ledger=outcome_ledger,
        contract_ledger=contract_ledger,
        adjudicator_ledger=adjudicator_ledger,
        agenda_ledger=agenda_ledger,
        lifecycle_ledger=lifecycle_ledger,
        scorecard_ledger=scorecard_ledger,
        campaign_ledger=campaign_ledger,
        campaign_outcome_ledger=campaign_outcome_ledger,
        prior_action_outcome_ledger=prior_action_outcome_ledger,
        runtime_memory_data=loaded_memory,
        runtime_memory_hash_state=runtime_hash_state,
        project_owned_boundary=dict(capsule.get('project_owned_boundary') or {}),
        artifact_path=action_outcome_ledger_file,
    )
    write_science_action_outcome_ledger(action_outcome_ledger_file, updated_ledger)
    if outbox_file:
        write_science_action_outcome_outbox_jsonl(outbox_file, message)
    latest = dict(updated_ledger.get('latest') or {})
    state_counts = dict(latest.get('state_counts') or {})
    result = {
        'science_action_outcome_capability': True,
        'action_outcome_ledger_path': str(action_outcome_ledger_file),
        'action_outcome_ledger_hash': updated_ledger.get('ledger_hash'),
        'outcome_id': latest.get('outcome_id'),
        'processed_message_count': len(updated_ledger.get('processed_message_ids') or []),
        'new_message_count': int(latest.get('new_message_count', 0) or 0),
        'skipped_message_count': int(latest.get('skipped_message_count', 0) or 0),
        'math_count': int(state_counts.get('math', 0) or 0),
        'code_count': int(state_counts.get('code', 0) or 0),
        'language_count': int(state_counts.get('language', 0) or 0),
        'refine_count': int(state_counts.get('refine', 0) or 0),
        'retire_count': int(state_counts.get('retire', 0) or 0),
        'waiting_count': int(state_counts.get('waiting', 0) or 0),
        'no_gain_count': int(state_counts.get('no_gain', 0) or 0),
        'repair_count': int(state_counts.get('repair', 0) or 0),
        'selected_outcome': latest.get('selected_outcome'),
        'selected_action': latest.get('selected_action'),
        'chosen_recipient': latest.get('chosen_recipient'),
        'outbox_count': int(latest.get('outbox_count', 0) or 0),
        'outbox_file': str(outbox_file) if outbox_file else None,
        'response_message': message,
        'runtime_memory_hash_state': runtime_hash_state,
        'runtime_memory_mutated': not bool(runtime_hash_state.get('unchanged', True)),
        'label_leaks': list(latest.get('label_leaks') or []),
        'label_leaks_count': len(latest.get('label_leaks') or []),
        'project_owned_boundary': dict(capsule.get('project_owned_boundary') or {}),
        'third_party_checkpoint_used': bool(
            (capsule.get('project_owned_boundary') or {}).get(
                'third_party_checkpoint_used'
            )
        ),
        'invalid_message_count': len(transcript.get('invalid_messages') or []),
        'no_sibling_imports': True,
        'project_owned_checkpoint_claimed': bool(
            (capsule.get('project_owned_boundary') or {}).get(
                'project_owned_checkpoint_verified'
            )
        ),
    }
    if output_file:
        _write_json_artifact(output_file, result)
    print(
        "AI_DIFFERENT_SCIENCE_ACTION_OUTCOME "
        + json.dumps(result, sort_keys=True),
        flush=True,
    )
    return result


def run_science_theory_frontier_planner(
    theory_memory: CumulativeTheoryMemory | None = None,
    *,
    theory_memory_file: str | Path | None = None,
    runtime_memory_path: str = 'tmp/theory-memory.json',
    transcript_file: str | Path | None = None,
    action_outcome_ledger_file: str | Path = 'tmp/module-chat-science-action-outcome-ledger.json',
    action_ledger_file: str | Path = 'tmp/module-chat-science-campaign-action-ledger.json',
    benefit_ledger_file: str | Path = 'tmp/module-chat-science-benefit-ledger.json',
    evaluator_ledger_file: str | Path = 'tmp/module-chat-outcome-evaluator-ledger.json',
    outcome_ledger_file: str | Path | None = None,
    contract_ledger_file: str | Path = 'tmp/module-chat-experiment-contract-ledger.json',
    adjudicator_ledger_file: str | Path = 'tmp/module-chat-cross-module-adjudicator-ledger.json',
    agenda_ledger_file: str | Path = 'tmp/module-chat-experiment-agenda-ledger.json',
    lifecycle_ledger_file: str | Path = 'tmp/module-chat-hypothesis-lifecycle-ledger.json',
    scorecard_ledger_file: str | Path = 'tmp/module-chat-evidence-scorecard-ledger.json',
    campaign_ledger_file: str | Path = 'tmp/module-chat-experiment-campaign-ledger.json',
    campaign_outcome_ledger_file: str | Path = 'tmp/module-chat-experiment-campaign-outcome-ledger.json',
    frontier_ledger_file: str | Path = 'tmp/module-chat-science-theory-frontier-ledger.json',
    prior_frontier_ledger_file: str | Path | None = None,
    sibling_outcome_ledgers_file: str | Path | None = None,
    outbox_file: str | Path | None = None,
    output_file: str | Path | None = None,
    memory_data: dict | None = None,
    git_status_text: str | None = None,
    git_ignored_text: str | None = None,
) -> dict:
    """Plan a durable theory-frontier move from assessed science outcomes."""
    if memory_data is not None:
        loaded_memory = dict(memory_data)
    elif theory_memory is not None:
        loaded_memory = theory_memory.to_dict()
    else:
        loaded_memory = load_capsule_memory_data(theory_memory_file)
    status_text = (
        git_status_text
        if git_status_text is not None
        else git_status_for_path(runtime_memory_path)
    )
    ignored_text = (
        git_ignored_text
        if git_ignored_text is not None
        else git_check_ignore_for_path(runtime_memory_path)
    )
    capsule = build_ai_different_status_capsule(
        loaded_memory,
        git_status_text=status_text,
        git_ignored_text=ignored_text,
        runtime_memory_path=str(runtime_memory_path),
    )
    transcript = read_science_theory_frontier_transcript(transcript_file)
    action_outcome_ledger = load_science_theory_frontier_plain_json(action_outcome_ledger_file)
    action_ledger = load_science_theory_frontier_plain_json(action_ledger_file)
    benefit_ledger = load_science_theory_frontier_plain_json(benefit_ledger_file)
    evaluator_ledger = load_science_theory_frontier_plain_json(evaluator_ledger_file)
    outcome_ledger = load_science_theory_frontier_plain_json(outcome_ledger_file) if outcome_ledger_file else {}
    contract_ledger = load_science_theory_frontier_plain_json(contract_ledger_file)
    adjudicator_ledger = load_science_theory_frontier_plain_json(adjudicator_ledger_file)
    agenda_ledger = load_science_theory_frontier_plain_json(agenda_ledger_file)
    lifecycle_ledger = load_science_theory_frontier_plain_json(lifecycle_ledger_file)
    scorecard_ledger = load_science_theory_frontier_plain_json(scorecard_ledger_file)
    campaign_ledger = load_science_theory_frontier_plain_json(campaign_ledger_file)
    campaign_outcome_ledger = load_science_theory_frontier_plain_json(campaign_outcome_ledger_file)
    frontier_ledger = load_science_theory_frontier_ledger(frontier_ledger_file)
    prior_frontier_ledger = (
        load_science_theory_frontier_plain_json(prior_frontier_ledger_file)
        if prior_frontier_ledger_file
        else {}
    )
    sibling_outcome_ledgers = (
        load_science_theory_frontier_plain_json(sibling_outcome_ledgers_file)
        if sibling_outcome_ledgers_file
        else {}
    )
    before_hash = _file_sha256(runtime_memory_path)
    runtime_hash_state = _runtime_memory_hash_state(runtime_memory_path, before_hash)
    updated_ledger, message = build_science_theory_frontier_plan(
        transcript_messages=list(transcript.get('messages') or []),
        frontier_ledger=frontier_ledger,
        action_outcome_ledger=action_outcome_ledger,
        action_ledger=action_ledger,
        benefit_ledger=benefit_ledger,
        evaluator_ledger=evaluator_ledger,
        outcome_ledger=outcome_ledger,
        contract_ledger=contract_ledger,
        adjudicator_ledger=adjudicator_ledger,
        agenda_ledger=agenda_ledger,
        lifecycle_ledger=lifecycle_ledger,
        scorecard_ledger=scorecard_ledger,
        campaign_ledger=campaign_ledger,
        campaign_outcome_ledger=campaign_outcome_ledger,
        prior_frontier_ledger=prior_frontier_ledger,
        sibling_outcome_ledgers=sibling_outcome_ledgers,
        runtime_memory_data=loaded_memory,
        runtime_memory_hash_state=runtime_hash_state,
        project_owned_boundary=dict(capsule.get('project_owned_boundary') or {}),
        artifact_path=frontier_ledger_file,
    )
    write_science_theory_frontier_ledger(frontier_ledger_file, updated_ledger)
    if outbox_file:
        write_science_theory_frontier_outbox_jsonl(outbox_file, message)
    latest = dict(updated_ledger.get('latest') or {})
    state_counts = dict(latest.get('state_counts') or {})
    result = {
        'science_theory_frontier_capability': True,
        'frontier_ledger_path': str(frontier_ledger_file),
        'frontier_ledger_hash': updated_ledger.get('ledger_hash'),
        'frontier_id': latest.get('frontier_id'),
        'processed_message_count': len(updated_ledger.get('processed_message_ids') or []),
        'new_message_count': int(latest.get('new_message_count', 0) or 0),
        'skipped_message_count': int(latest.get('skipped_message_count', 0) or 0),
        'promote_count': int(state_counts.get('promote', 0) or 0),
        'retire_count': int(state_counts.get('retire', 0) or 0),
        'block_count': int(state_counts.get('block', 0) or 0),
        'funfun_count': int(state_counts.get('funfun', 0) or 0),
        'code_count': int(state_counts.get('code', 0) or 0),
        'language_count': int(state_counts.get('language', 0) or 0),
        'refine_count': int(state_counts.get('refine', 0) or 0),
        'frontier_count': int(state_counts.get('frontier', 0) or 0),
        'waiting_count': int(state_counts.get('waiting', 0) or 0),
        'no_gain_count': int(state_counts.get('no_gain', 0) or 0),
        'repair_count': int(state_counts.get('repair', 0) or 0),
        'selected_theory_move': latest.get('selected_theory_move'),
        'selected_action': latest.get('selected_action'),
        'chosen_recipient': latest.get('chosen_recipient'),
        'outbox_count': int(latest.get('outbox_count', 0) or 0),
        'outbox_file': str(outbox_file) if outbox_file else None,
        'response_message': message,
        'runtime_memory_hash_state': runtime_hash_state,
        'runtime_memory_mutated': not bool(runtime_hash_state.get('unchanged', True)),
        'label_leaks': list(latest.get('label_leaks') or []),
        'label_leaks_count': len(latest.get('label_leaks') or []),
        'project_owned_boundary': dict(capsule.get('project_owned_boundary') or {}),
        'third_party_checkpoint_used': bool(
            (capsule.get('project_owned_boundary') or {}).get(
                'third_party_checkpoint_used'
            )
        ),
        'invalid_message_count': len(transcript.get('invalid_messages') or []),
        'no_sibling_imports': True,
        'project_owned_checkpoint_claimed': bool(
            (capsule.get('project_owned_boundary') or {}).get(
                'project_owned_checkpoint_verified'
            )
        ),
    }
    if output_file:
        _write_json_artifact(output_file, result)
    print(
        "AI_DIFFERENT_SCIENCE_THEORY_FRONTIER "
        + json.dumps(result, sort_keys=True),
        flush=True,
    )
    return result


def run_science_theory_frontier_outcome_assessor(
    theory_memory: CumulativeTheoryMemory | None = None,
    *,
    theory_memory_file: str | Path | None = None,
    runtime_memory_path: str = 'tmp/theory-memory.json',
    transcript_file: str | Path | None = None,
    frontier_ledger_file: str | Path = 'tmp/module-chat-science-theory-frontier-ledger.json',
    action_outcome_ledger_file: str | Path = 'tmp/module-chat-science-action-outcome-ledger.json',
    action_ledger_file: str | Path = 'tmp/module-chat-science-campaign-action-ledger.json',
    benefit_ledger_file: str | Path = 'tmp/module-chat-science-benefit-ledger.json',
    evaluator_ledger_file: str | Path = 'tmp/module-chat-outcome-evaluator-ledger.json',
    outcome_ledger_file: str | Path | None = None,
    contract_ledger_file: str | Path = 'tmp/module-chat-experiment-contract-ledger.json',
    adjudicator_ledger_file: str | Path = 'tmp/module-chat-cross-module-adjudicator-ledger.json',
    agenda_ledger_file: str | Path = 'tmp/module-chat-experiment-agenda-ledger.json',
    lifecycle_ledger_file: str | Path = 'tmp/module-chat-hypothesis-lifecycle-ledger.json',
    scorecard_ledger_file: str | Path = 'tmp/module-chat-evidence-scorecard-ledger.json',
    campaign_ledger_file: str | Path = 'tmp/module-chat-experiment-campaign-ledger.json',
    campaign_outcome_ledger_file: str | Path = 'tmp/module-chat-experiment-campaign-outcome-ledger.json',
    theory_memory_ledger_file: str | Path | None = None,
    frontier_outcome_ledger_file: str | Path = 'tmp/module-chat-science-theory-frontier-outcome-ledger.json',
    prior_frontier_outcome_ledger_file: str | Path | None = None,
    sibling_frontier_outcome_ledgers_file: str | Path | None = None,
    outbox_file: str | Path | None = None,
    output_file: str | Path | None = None,
    memory_data: dict | None = None,
    git_status_text: str | None = None,
    git_ignored_text: str | None = None,
) -> dict:
    """Assess whether a planned theory-frontier move changed symbolic state."""
    if memory_data is not None:
        loaded_memory = dict(memory_data)
    elif theory_memory is not None:
        loaded_memory = theory_memory.to_dict()
    else:
        loaded_memory = load_capsule_memory_data(theory_memory_file)
    status_text = (
        git_status_text
        if git_status_text is not None
        else git_status_for_path(runtime_memory_path)
    )
    ignored_text = (
        git_ignored_text
        if git_ignored_text is not None
        else git_check_ignore_for_path(runtime_memory_path)
    )
    capsule = build_ai_different_status_capsule(
        loaded_memory,
        git_status_text=status_text,
        git_ignored_text=ignored_text,
        runtime_memory_path=str(runtime_memory_path),
    )
    transcript = read_science_theory_frontier_outcome_transcript(transcript_file)
    frontier_ledger = load_science_theory_frontier_outcome_plain_json(frontier_ledger_file)
    action_outcome_ledger = load_science_theory_frontier_outcome_plain_json(action_outcome_ledger_file)
    action_ledger = load_science_theory_frontier_outcome_plain_json(action_ledger_file)
    benefit_ledger = load_science_theory_frontier_outcome_plain_json(benefit_ledger_file)
    evaluator_ledger = load_science_theory_frontier_outcome_plain_json(evaluator_ledger_file)
    outcome_ledger = load_science_theory_frontier_outcome_plain_json(outcome_ledger_file) if outcome_ledger_file else {}
    contract_ledger = load_science_theory_frontier_outcome_plain_json(contract_ledger_file)
    adjudicator_ledger = load_science_theory_frontier_outcome_plain_json(adjudicator_ledger_file)
    agenda_ledger = load_science_theory_frontier_outcome_plain_json(agenda_ledger_file)
    lifecycle_ledger = load_science_theory_frontier_outcome_plain_json(lifecycle_ledger_file)
    scorecard_ledger = load_science_theory_frontier_outcome_plain_json(scorecard_ledger_file)
    campaign_ledger = load_science_theory_frontier_outcome_plain_json(campaign_ledger_file)
    campaign_outcome_ledger = load_science_theory_frontier_outcome_plain_json(campaign_outcome_ledger_file)
    theory_memory_ledger = (
        load_science_theory_frontier_outcome_plain_json(theory_memory_ledger_file)
        if theory_memory_ledger_file
        else {}
    )
    frontier_outcome_ledger = load_science_theory_frontier_outcome_ledger(frontier_outcome_ledger_file)
    prior_frontier_outcome_ledger = (
        load_science_theory_frontier_outcome_plain_json(prior_frontier_outcome_ledger_file)
        if prior_frontier_outcome_ledger_file
        else {}
    )
    sibling_frontier_outcome_ledgers = (
        load_science_theory_frontier_outcome_plain_json(sibling_frontier_outcome_ledgers_file)
        if sibling_frontier_outcome_ledgers_file
        else {}
    )
    before_hash = _file_sha256(runtime_memory_path)
    runtime_hash_state = _runtime_memory_hash_state(runtime_memory_path, before_hash)
    updated_ledger, message = build_science_theory_frontier_outcome_assessment(
        transcript_messages=list(transcript.get('messages') or []),
        frontier_outcome_ledger=frontier_outcome_ledger,
        frontier_ledger=frontier_ledger,
        action_outcome_ledger=action_outcome_ledger,
        action_ledger=action_ledger,
        benefit_ledger=benefit_ledger,
        evaluator_ledger=evaluator_ledger,
        outcome_ledger=outcome_ledger,
        contract_ledger=contract_ledger,
        adjudicator_ledger=adjudicator_ledger,
        agenda_ledger=agenda_ledger,
        lifecycle_ledger=lifecycle_ledger,
        scorecard_ledger=scorecard_ledger,
        campaign_ledger=campaign_ledger,
        campaign_outcome_ledger=campaign_outcome_ledger,
        theory_memory_ledger=theory_memory_ledger,
        prior_frontier_outcome_ledger=prior_frontier_outcome_ledger,
        sibling_frontier_outcome_ledgers=sibling_frontier_outcome_ledgers,
        runtime_memory_data=loaded_memory,
        runtime_memory_hash_state=runtime_hash_state,
        project_owned_boundary=dict(capsule.get('project_owned_boundary') or {}),
        artifact_path=frontier_outcome_ledger_file,
    )
    write_science_theory_frontier_outcome_ledger(frontier_outcome_ledger_file, updated_ledger)
    if outbox_file:
        write_science_theory_frontier_outcome_outbox_jsonl(outbox_file, message)
    latest = dict(updated_ledger.get('latest') or {})
    state_counts = dict(latest.get('state_counts') or {})
    result = {
        'science_theory_frontier_outcome_capability': True,
        'frontier_outcome_ledger_path': str(frontier_outcome_ledger_file),
        'frontier_outcome_ledger_hash': updated_ledger.get('ledger_hash'),
        'frontier_outcome_id': latest.get('frontier_outcome_id'),
        'processed_message_count': len(updated_ledger.get('processed_message_ids') or []),
        'new_message_count': int(latest.get('new_message_count', 0) or 0),
        'skipped_message_count': int(latest.get('skipped_message_count', 0) or 0),
        'memory_count': int(state_counts.get('memory', 0) or 0),
        'promote_count': int(state_counts.get('promote', 0) or 0),
        'retire_count': int(state_counts.get('retire', 0) or 0),
        'block_count': int(state_counts.get('block', 0) or 0),
        'funfun_count': int(state_counts.get('funfun', 0) or 0),
        'code_count': int(state_counts.get('code', 0) or 0),
        'language_count': int(state_counts.get('language', 0) or 0),
        'refine_count': int(state_counts.get('refine', 0) or 0),
        'frontier_count': int(state_counts.get('frontier', 0) or 0),
        'waiting_count': int(state_counts.get('waiting', 0) or 0),
        'no_gain_count': int(state_counts.get('no_gain', 0) or 0),
        'repair_count': int(state_counts.get('repair', 0) or 0),
        'selected_outcome': latest.get('selected_outcome'),
        'selected_action': latest.get('selected_action'),
        'chosen_recipient': latest.get('chosen_recipient'),
        'outbox_count': int(latest.get('outbox_count', 0) or 0),
        'outbox_file': str(outbox_file) if outbox_file else None,
        'response_message': message,
        'runtime_memory_hash_state': runtime_hash_state,
        'runtime_memory_mutated': not bool(runtime_hash_state.get('unchanged', True)),
        'label_leaks': list(latest.get('label_leaks') or []),
        'label_leaks_count': len(latest.get('label_leaks') or []),
        'project_owned_boundary': dict(capsule.get('project_owned_boundary') or {}),
        'third_party_checkpoint_used': bool(
            (capsule.get('project_owned_boundary') or {}).get(
                'third_party_checkpoint_used'
            )
        ),
        'invalid_message_count': len(transcript.get('invalid_messages') or []),
        'no_sibling_imports': True,
        'project_owned_checkpoint_claimed': bool(
            (capsule.get('project_owned_boundary') or {}).get(
                'project_owned_checkpoint_verified'
            )
        ),
    }
    if output_file:
        _write_json_artifact(output_file, result)
    print(
        "AI_DIFFERENT_SCIENCE_THEORY_FRONTIER_OUTCOME "
        + json.dumps(result, sort_keys=True),
        flush=True,
    )
    return result


def _upload_math_final_artifact(
    artifact_path: Path,
    *,
    hf_output_repo: str | None,
    run_id: str,
) -> dict:
    repo_id = (
        hf_output_repo
        or os.environ.get('HF_OUTPUT_REPO')
        or os.environ.get('HF_RUN_REPO')
    )
    if not repo_id:
        return {
            'status': 'skipped',
            'reason': 'no_hf_output_repo',
        }
    token = os.environ.get('HF_TOKEN')
    if not token:
        return {
            'status': 'skipped',
            'repo_id': repo_id,
            'reason': 'missing_HF_TOKEN',
        }
    try:
        from huggingface_hub import HfApi
    except Exception as error:
        return {
            'status': 'failed',
            'repo_id': repo_id,
            'reason': 'missing_huggingface_hub',
            'error': str(error),
        }
    api = HfApi(token=token)
    create_error = None
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type='dataset',
            exist_ok=True,
            private=False,
        )
    except Exception as error:
        create_error = str(error)
    path_in_repo = f"runs/{run_id}/summary.json"
    try:
        upload = upload_hf_artifact_file(
            api,
            path_or_fileobj=str(artifact_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type='dataset',
        )
        if create_error:
            upload['create_repo_warning'] = create_error
        return upload
    except Exception as error:
        return {
            'status': 'failed',
            'repo_id': repo_id,
            'repo_type': 'dataset',
            'path_in_repo': path_in_repo,
            'reason': 'upload_failed',
            'create_repo_error': create_error,
            'error': str(error),
        }


def _persist_math_final_artifact(
    results: list[dict],
    theory_memory: CumulativeTheoryMemory,
    *,
    artifact_output_file: str | Path | None,
    hf_output_repo: str | None,
    run_id: str | None,
    run_config: dict,
    starting_memory_summary: dict,
    runtime_profile_events: list[dict] | None = None,
) -> dict:
    resolved_run_id = run_id or os.environ.get('HF_RUN_ID') or _default_hf_run_id()
    output_file = (
        Path(artifact_output_file)
        if artifact_output_file
        else Path('tmp') / f'{resolved_run_id}-summary.json'
    )
    artifact = _math_final_artifact_summary(
        results,
        theory_memory,
        run_id=resolved_run_id,
        run_config=run_config,
        starting_memory_summary=starting_memory_summary,
        runtime_profile_events=runtime_profile_events,
    )
    persist_started = time.perf_counter()
    artifact_path = _write_json_artifact(output_file, artifact)
    upload = _upload_math_final_artifact(
        artifact_path,
        hf_output_repo=hf_output_repo,
        run_id=resolved_run_id,
    )
    artifact['artifact_path'] = str(artifact_path)
    artifact['hf_upload'] = upload
    _write_json_artifact(artifact_path, artifact)
    if upload.get('status') in {'uploaded', 'uploaded_via_pr'}:
        second_upload = _upload_math_final_artifact(
            artifact_path,
            hf_output_repo=hf_output_repo,
            run_id=resolved_run_id,
        )
        artifact['hf_upload'] = second_upload
        _write_json_artifact(artifact_path, artifact)
    if _artifact_status_needs_log_fallback(artifact['hf_upload']):
        artifact['log_artifact_chunks'] = _emit_artifact_log_chunks(
            artifact,
            run_id=resolved_run_id,
        )
        _write_json_artifact(artifact_path, artifact)
    artifact.setdefault('runtime_profile', {})['artifact_persist_seconds'] = round(
        time.perf_counter() - persist_started,
        3,
    )
    _write_json_artifact(artifact_path, artifact)
    print(
        "HF_ARTIFACT "
        + json.dumps({
            'run_id': resolved_run_id,
            'summary': str(artifact_path),
            'repo': (
                hf_output_repo
                or os.environ.get('HF_OUTPUT_REPO')
                or os.environ.get('HF_RUN_REPO')
            ),
            'upload_status': artifact['hf_upload'].get('status'),
            'path_in_repo': artifact['hf_upload'].get('path_in_repo'),
            'reason': artifact['hf_upload'].get('reason'),
            'error': artifact['hf_upload'].get('error'),
            'create_repo_error': artifact['hf_upload'].get('create_repo_error'),
        }, sort_keys=True),
        flush=True,
    )
    print(
        "HF_ARTIFACT_SUMMARY "
        + json.dumps(_compact_artifact_summary_for_log(artifact), sort_keys=True),
        flush=True,
    )
    return artifact


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
    domain_rediscovery_experiments = theory_memory.domain_rediscovery_experiments(limit=3)
    autonomous_designs = theory_memory.autonomous_experiment_design_agenda(limit=3)
    theorem_memory = theory_memory.theorem_memory(limit=3)
    theorem_consolidations = theory_memory.theorem_consolidations(limit=3)
    blind_holdout_benchmark = theory_memory.blind_holdout_benchmark_report(limit=3)
    representation_agenda = theory_memory.representation_agenda(limit=3)
    generated_operator_priors = theory_memory.generated_operator_priors(limit=3)
    operator_prior_invariants = theory_memory.operator_prior_invariant_consolidations(limit=3)
    invariant_resolution_experiments = (
        theory_memory.equation_invariant_resolution_experiments(limit=3)
    )
    selected_law_replay_agenda = theory_memory.selected_law_replay_agenda(limit=3)
    selected_law_conflicts = theory_memory.selected_law_conflict_experiments(limit=3)
    model_domain_splits = theory_memory.model_disagreement_domain_split_experiments(
        limit=3
    )
    localized_gravity_probes = theory_memory.localized_gravity_structure_experiments(
        limit=3
    )
    law_domain_splits = theory_memory.law_domain_split_hypotheses(limit=3)
    domain_predicate_agenda = theory_memory.domain_predicate_learning_agenda(limit=3)
    blind_holdout_validations = theory_memory.blind_holdout_validation_experiments(limit=3)
    post_run_replay_agenda = theory_memory.post_run_replay_agenda(limit=3)
    operator_prior_feedback = theory_memory.operator_prior_feedback(limit=3)
    operator_prior_domains = theory_memory.operator_prior_domains(limit=3)
    operator_prior_anomalies = theory_memory.operator_prior_anomalies(limit=3)
    operator_prior_claims = theory_memory.operator_prior_discovery_claims(limit=3)
    operator_prior_chains = theory_memory.operator_prior_discovery_chains(limit=3)
    operator_prior_claim_experiments = theory_memory.operator_prior_claim_experiments(limit=3)
    operator_prior_repairs = theory_memory.operator_prior_repair_experiments(limit=3)
    next_experiments = theory_memory.next_experiments(limit=3)
    experiment_design_cockpit = _experiment_design_cockpit(
        theory_memory,
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
        domain_rediscovery_experiments,
        autonomous_designs,
        theorem_memory,
        theorem_consolidations,
        blind_holdout_benchmark.get('plan'),
        representation_agenda,
        generated_operator_priors,
        operator_prior_invariants,
        invariant_resolution_experiments,
        selected_law_replay_agenda,
        selected_law_conflicts,
        model_domain_splits,
        localized_gravity_probes,
        law_domain_splits,
        domain_predicate_agenda,
        blind_holdout_validations,
        post_run_replay_agenda,
        operator_prior_feedback,
        operator_prior_domains,
        operator_prior_anomalies,
        operator_prior_claims,
        operator_prior_chains,
        operator_prior_claim_experiments,
        operator_prior_repairs,
        next_experiments,
        experiment_design_cockpit,
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
    if operator_prior_invariants:
        print("Theory robust equation invariants:")
        for item in operator_prior_invariants:
            print(
                f"  {item['law_family']}: {item['context']} "
                f"status={item['status']} support={item['support_count']} "
                f"score={item['mean_score']:.2f}"
            )
            print(f"    claim: {item['robust_claim']}")
            print(f"    next: {item['next_experiment']}")
    if invariant_resolution_experiments:
        print("Theory equation invariant resolution probes:")
        for item in invariant_resolution_experiments:
            signature = item.get('disagreement_signature', {})
            exponents = ','.join(
                str(value)
                for value in signature.get('candidate_exponents', [])[:4]
            )
            print(
                f"  {item['theory_kind']}: priority={item['priority']:.2f} "
                f"context={item.get('source_context')} exponents={exponents}"
            )
            print(f"    next: {item['expected_result']}")
    if selected_law_replay_agenda:
        print("Theory selected-law replay agenda:")
        for item in selected_law_replay_agenda:
            print(
                f"  {item['primary_theory_label']}: "
                f"context={item.get('source_context')} priority={item['priority']:.2f}"
            )
            print(f"    next: {item['expected_result']}")
    if selected_law_conflicts:
        print("Theory selected-law conflict resolution:")
        for item in selected_law_conflicts:
            rivals = ','.join(item.get('rival_theory_labels', [])[:3]) or 'none'
            print(
                f"  {item['primary_theory_label']} vs {rivals}: "
                f"context={item.get('source_context')} priority={item['priority']:.2f}"
            )
            print(f"    next: {item['expected_result']}")
    if model_domain_splits:
        print("Theory model-disagreement domain splits:")
        for item in model_domain_splits:
            signature = item.get('disagreement_signature', {})
            print(
                f"  {item['theory_kind']}: mode={signature.get('mode')} "
                f"priority={item['priority']:.2f}"
            )
            print(f"    next: {item['expected_result']}")
    if localized_gravity_probes:
        print("Theory localized-gravity structure probes:")
        for item in localized_gravity_probes:
            print(
                f"  {item['theory_kind']}: priority={item['priority']:.2f} "
                f"quick={bool(item.get('quick_probe'))}"
            )
            print(f"    next: {item['expected_result']}")
    if law_domain_splits:
        print("Theory law domain split hypotheses:")
        for item in law_domain_splits:
            contexts = (
                ','.join(item.get('conflict_contexts', [])[:3])
                or ','.join(item.get('source_contexts', [])[:3])
                or 'none'
            )
            split_key = (
                item.get('invariant_key')
                or item.get('key')
                or item.get('theory_kind')
                or 'domain_split'
            )
            print(
                f"  {split_key}: status={item.get('status', 'unknown')} "
                f"kind={item.get('split_kind', 'unknown')} contexts={contexts}"
            )
            print(f"    question: {item.get('question', 'unknown')}")
    if domain_predicate_agenda:
        print("Theory domain predicate learning agenda:")
        for item in domain_predicate_agenda:
            predicate = item.get('candidate_predicate', {})
            print(
                f"  {predicate.get('predicate_kind', 'predicate')}: "
                f"priority={item['priority']:.2f} "
                f"confidence={predicate.get('confidence', 0.0)}"
            )
            print(f"    question: {predicate.get('question')}")
    if blind_holdout_validations:
        print("Theory blind holdout validation agenda:")
        for item in blind_holdout_validations:
            print(
                f"  {item['primary_theory_label']}: "
                f"priority={item['priority']:.2f}"
            )
            print(f"    next: {item['expected_result']}")
    if post_run_replay_agenda:
        print("Theory post-run replay agenda:")
        for item in post_run_replay_agenda:
            print(
                f"  {item['replay_issue']}: {item['source_context']} "
                f"seed={item['replay_seed']} priority={item['priority']:.2f}"
            )
            print(f"    why: {item['reason']}")
            print(f"    expected: {item['expected_result']}")
    if theorem_memory:
        print("Theory theorem memory:")
        for item in theorem_memory:
            evidence = item.get('evidence', {})
            print(
                f"  {item['theorem_kind']}: status={item['status']} "
                f"support={evidence.get('support_count', 0)}"
            )
            print(f"    statement: {item.get('statement')}")
    if theorem_consolidations:
        print("Theory theorem consolidations:")
        for item in theorem_consolidations:
            evidence = item.get('evidence', {})
            print(
                f"  {item['law_family']}: status={item['status']} "
                f"support={evidence.get('support_count', 0)}"
            )
            print(f"    statement: {item.get('statement')}")
            print(f"    next: {item.get('next_obligation')}")
    if domain_rediscovery_experiments:
        print("Theory domain rediscovery agenda:")
        for item in domain_rediscovery_experiments:
            print(
                f"  {item['domain_key']}: status={item['family_status']} "
                f"priority={item['priority']:.2f}"
            )
            print(f"    why: {item['reason']}")
    if autonomous_designs:
        print("Theory autonomous experiment designs:")
        for item in autonomous_designs:
            print(
                f"  {item['source']}:{item['experiment_kind']} "
                f"priority={item['priority']:.2f}"
            )
            print(f"    falsifier: {item['falsifies_if']}")
    if experiment_design_cockpit:
        print("Theory experiment design cockpit:")
        for design in experiment_design_cockpit:
            print(
                f"  Current theories disagree: {design['question']} "
                f"(priority={design['priority']:.2f})"
            )
            belief_text = ', '.join(
                f"{belief['theory']}={belief['belief']:.0%}"
                for belief in design.get('beliefs', [])[:3]
            )
            print(f"    beliefs: {belief_text or 'none'}")
            print(f"    proposed intervention: {design['intervention_text']}")
            print(f"    expected: {design.get('expected_result')}")
            print(f"    falsifier: {design.get('falsifies_if')}")
    if blind_holdout_benchmark.get('plan'):
        print(
            "Theory blind holdout benchmark: "
            f"cases={blind_holdout_benchmark['benchmark_count']} "
            f"ready={blind_holdout_benchmark['ready_for_blind_run']} "
            f"leak_blockers={blind_holdout_benchmark['leak_blocker_count']}"
        )
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
            if plan.get('experiment_kind') == 'post_run_replay_revision':
                print(f"    replay: {plan.get('replay_issue', 'post_run_revision')}")
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
    if 'zero_gravity' in novel_types:
        discoveries.add('zero_gravity')
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
    force_backend: str = 'python',
    equation_scoring_backend: str = 'python',
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
            force_backend=force_backend,
            equation_scoring_backend=equation_scoring_backend,
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
        'force_backend': force_backend,
        'equation_scoring_backend': equation_scoring_backend,
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


def _parse_abstraction_transfer_worlds(value: str) -> list[str]:
    worlds = [part.strip() for part in value.split(',') if part.strip()]
    allowed = set(WORLD_TYPES) | {'hidden_procedural'}
    unknown = [world for world in worlds if world not in allowed]
    if unknown:
        raise ValueError(f"Unknown abstraction target world(s): {', '.join(unknown)}")
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
    parser.add_argument('--abstraction-transfer-campaign', action='store_true',
                        help='Run a tiny non-final abstraction transfer campaign')
    parser.add_argument('--abstraction-transfer-outcome', type=str, default='confirmed',
                        choices=['confirmed', 'weak', 'absent'],
                        help='Simulated transfer outcome for the non-final abstraction campaign')
    parser.add_argument('--math-foundation-prep', action='store_true',
                        help='Run readiness checks before the final watched math discovery run')
    parser.add_argument('--discovery-readiness', action='store_true',
                        help='Print a non-final readiness audit from cumulative theory memory')
    parser.add_argument('--rediscovery-goal-progress', action='store_true',
                        help='Print stricter non-final progress toward the 85% rediscovery goal')
    parser.add_argument('--super-system-audit', action='store_true',
                        help='Print a connected non-final audit across every discovery subsystem')
    parser.add_argument('--super-system-output-file', type=str, default=None,
                        help='Optional JSON path for the connected super-system audit')
    parser.add_argument('--super-system-limit', type=int, default=5,
                        help='Rows per subsystem to include in the super-system audit')
    parser.add_argument('--status-capsule', action='store_true',
                        help='Print an orchestrator-facing AI Different status capsule')
    parser.add_argument('--status-capsule-output-file', type=str, default=None,
                        help='Optional JSON path for the status capsule')
    parser.add_argument('--module-chat-export', action='store_true',
                        help='Export the AI Different capsule as a module-chat JSON message')
    parser.add_argument('--module-chat-response-loop', action='store_true',
                        help='Read module-chat inbox and emit an abstraction-transfer response message')
    parser.add_argument('--module-chat-family-response', action='store_true',
                        help='Read a richer module-family inbox, persist a ledger, and emit a response')
    parser.add_argument('--module-chat-rolling-family-response', action='store_true',
                        help='Run idempotent rolling module-family response over a chat JSONL log')
    parser.add_argument('--module-chat-outcome-evaluator', action='store_true',
                        help='Evaluate rolling family evidence and choose the next science experiment')
    parser.add_argument('--module-chat-experiment-contract', action='store_true',
                        help='Emit or resolve plain-data experiment contracts from evaluator decisions')
    parser.add_argument('--module-chat-cross-module-adjudicator', action='store_true',
                        help='Adjudicate a family transcript against experiment-contract evidence')
    parser.add_argument('--module-chat-experiment-agenda', action='store_true',
                        help='Schedule the next safe experiment contract or repair from adjudicated evidence')
    parser.add_argument('--module-chat-hypothesis-lifecycle', action='store_true',
                        help='Curate hypothesis lifecycle memory from module-family evidence')
    parser.add_argument('--module-chat-evidence-scorecard', action='store_true',
                        help='Score hypothesis evidence gates and choose a refinement/repair action')
    parser.add_argument('--module-chat-experiment-campaign', action='store_true',
                        help='Plan the next symbolic experiment campaign or acceptance bundle')
    parser.add_argument('--module-chat-campaign-outcome', action='store_true',
                        help='Assess campaign return evidence and plan a theory update')
    parser.add_argument('--module-chat-science-benefit', action='store_true',
                        help='Compare isolated versus connected symbolic campaign evidence')
    parser.add_argument('--module-chat-science-action', action='store_true',
                        help='Plan the next campaign action from science-benefit records')
    parser.add_argument('--module-chat-science-action-outcome', action='store_true',
                        help='Assess whether science campaign actions received useful evidence')
    parser.add_argument('--module-chat-science-theory-frontier', action='store_true',
                        help='Plan one durable theory-frontier move from assessed science outcomes')
    parser.add_argument('--module-chat-science-theory-frontier-outcome', action='store_true',
                        help='Assess whether theory-frontier moves changed symbolic theory state')
    parser.add_argument('--module-chat-response-mode', type=str, default='plan',
                        choices=['plan', 'run'],
                        help='Plan or run the cheap no-save abstraction-transfer response')
    parser.add_argument('--module-chat-recipient', type=str, default='orchestrator',
                        help='Recipient participant for module-chat export')
    parser.add_argument('--module-chat-topic', type=str, default='ai_different.status_capsule',
                        help='Topic for module-chat export')
    parser.add_argument('--module-chat-inbox', type=str, default=None,
                        help='Optional module-chat inbox JSONL file to read')
    parser.add_argument('--module-chat-output-file', type=str, default=None,
                        help='Optional JSON path for the exported module-chat message')
    parser.add_argument('--module-chat-ledger-file', type=str,
                        default='tmp/module-chat-response-ledger.json',
                        help='JSON path for the persisted module-chat response ledger')
    parser.add_argument('--module-chat-rolling-memory-file', type=str,
                        default='tmp/module-chat-family-memory.json',
                        help='JSON path for rolling module-family response memory')
    parser.add_argument('--module-chat-evaluator-ledger-files', type=str, default=None,
                        help='Comma-separated response ledger JSON files for the outcome evaluator')
    parser.add_argument('--module-chat-outcome-ledger-file', type=str,
                        default='tmp/module-chat-outcome-evaluator-ledger.json',
                        help='JSON path for the persisted outcome-evaluator ledger')
    parser.add_argument('--module-chat-outcome-memory-file', type=str,
                        default='tmp/module-chat-outcome-evaluator-memory.json',
                        help='JSON path for durable outcome-evaluator memory')
    parser.add_argument('--module-chat-contract-ledger-file', type=str,
                        default='tmp/module-chat-experiment-contract-ledger.json',
                        help='JSON path for experiment-contract ledger state')
    parser.add_argument('--module-chat-contract-outbox-file', type=str,
                        default='tmp/module-chat-experiment-contract-outbox.jsonl',
                        help='JSONL path for at most one emitted contract/repair message')
    parser.add_argument('--module-chat-adjudicator-ledger-file', type=str,
                        default='tmp/module-chat-cross-module-adjudicator-ledger.json',
                        help='JSON path for cross-module adjudicator state')
    parser.add_argument('--module-chat-adjudicator-outbox-file', type=str,
                        default='tmp/module-chat-cross-module-adjudicator-outbox.jsonl',
                        help='JSONL path for at most one adjudication response message')
    parser.add_argument('--module-chat-agenda-outcome-ledger-file', type=str,
                        default=None,
                        help='Optional extra outcome ledger JSON path for the experiment agenda scheduler')
    parser.add_argument('--module-chat-agenda-ledger-file', type=str,
                        default='tmp/module-chat-experiment-agenda-ledger.json',
                        help='JSON path for experiment agenda scheduler state')
    parser.add_argument('--module-chat-agenda-outbox-file', type=str,
                        default='tmp/module-chat-experiment-agenda-outbox.jsonl',
                        help='JSONL path for at most one experiment agenda response message')
    parser.add_argument('--module-chat-lifecycle-ledger-file', type=str,
                        default='tmp/module-chat-hypothesis-lifecycle-ledger.json',
                        help='JSON path for hypothesis lifecycle memory state')
    parser.add_argument('--module-chat-lifecycle-outbox-file', type=str,
                        default='tmp/module-chat-hypothesis-lifecycle-outbox.jsonl',
                        help='JSONL path for at most one hypothesis lifecycle response message')
    parser.add_argument('--module-chat-scorecard-ledger-file', type=str,
                        default='tmp/module-chat-evidence-scorecard-ledger.json',
                        help='JSON path for experiment evidence scorecard state')
    parser.add_argument('--module-chat-scorecard-outbox-file', type=str,
                        default='tmp/module-chat-evidence-scorecard-outbox.jsonl',
                        help='JSONL path for at most one evidence scorecard response message')
    parser.add_argument('--module-chat-campaign-ledger-file', type=str,
                        default='tmp/module-chat-experiment-campaign-ledger.json',
                        help='JSON path for experiment campaign planner state')
    parser.add_argument('--module-chat-campaign-outbox-file', type=str,
                        default='tmp/module-chat-experiment-campaign-outbox.jsonl',
                        help='JSONL path for at most one experiment campaign response message')
    parser.add_argument('--module-chat-campaign-outcome-ledger-file', type=str,
                        default='tmp/module-chat-experiment-campaign-outcome-ledger.json',
                        help='JSON path for campaign outcome assessment state')
    parser.add_argument('--module-chat-campaign-outcome-outbox-file', type=str,
                        default='tmp/module-chat-experiment-campaign-outcome-outbox.jsonl',
                        help='JSONL path for at most one campaign outcome response message')
    parser.add_argument('--module-chat-benefit-ledger-file', type=str,
                        default='tmp/module-chat-science-benefit-ledger.json',
                        help='JSON path for connected-vs-isolated science benefit state')
    parser.add_argument('--module-chat-prior-benefit-ledger-file', type=str,
                        default=None,
                        help='Optional prior science benefit ledger JSON path to compare against')
    parser.add_argument('--module-chat-benefit-outbox-file', type=str,
                        default='tmp/module-chat-science-benefit-outbox.jsonl',
                        help='JSONL path for at most one science benefit response message')
    parser.add_argument('--module-chat-action-ledger-file', type=str,
                        default='tmp/module-chat-science-campaign-action-ledger.json',
                        help='JSON path for science-campaign action planner state')
    parser.add_argument('--module-chat-prior-action-ledger-file', type=str,
                        default=None,
                        help='Optional prior science-campaign action ledger JSON path')
    parser.add_argument('--module-chat-action-outbox-file', type=str,
                        default='tmp/module-chat-science-campaign-action-outbox.jsonl',
                        help='JSONL path for at most one science action response message')
    parser.add_argument('--module-chat-action-outcome-ledger-file', type=str,
                        default='tmp/module-chat-science-action-outcome-ledger.json',
                        help='JSON path for science action outcome assessor state')
    parser.add_argument('--module-chat-prior-action-outcome-ledger-file', type=str,
                        default=None,
                        help='Optional prior science action outcome ledger JSON path')
    parser.add_argument('--module-chat-action-outcome-outbox-file', type=str,
                        default='tmp/module-chat-science-action-outcome-outbox.jsonl',
                        help='JSONL path for at most one science action outcome response')
    parser.add_argument('--module-chat-frontier-ledger-file', type=str,
                        default='tmp/module-chat-science-theory-frontier-ledger.json',
                        help='JSON path for science theory-frontier planner state')
    parser.add_argument('--module-chat-prior-frontier-ledger-file', type=str,
                        default=None,
                        help='Optional prior science theory-frontier ledger JSON path')
    parser.add_argument('--module-chat-sibling-outcome-ledgers-file', type=str,
                        default=None,
                        help='Optional plain JSON bundle of sibling outcome ledgers')
    parser.add_argument('--module-chat-frontier-outbox-file', type=str,
                        default='tmp/module-chat-science-theory-frontier-outbox.jsonl',
                        help='JSONL path for at most one science theory-frontier response')
    parser.add_argument('--module-chat-frontier-outcome-ledger-file', type=str,
                        default='tmp/module-chat-science-theory-frontier-outcome-ledger.json',
                        help='JSON path for science theory-frontier outcome assessment state')
    parser.add_argument('--module-chat-prior-frontier-outcome-ledger-file', type=str,
                        default=None,
                        help='Optional prior science theory-frontier outcome ledger JSON path')
    parser.add_argument('--module-chat-sibling-frontier-outcome-ledgers-file', type=str,
                        default=None,
                        help='Optional plain JSON bundle of sibling frontier/outcome ledgers')
    parser.add_argument('--module-chat-theory-memory-ledger-file', type=str,
                        default=None,
                        help='Optional plain JSON theory-memory evidence ledger to read without mutation')
    parser.add_argument('--module-chat-frontier-outcome-outbox-file', type=str,
                        default='tmp/module-chat-science-theory-frontier-outcome-outbox.jsonl',
                        help='JSONL path for at most one science theory-frontier outcome response')
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
    parser.add_argument('--hf-adaptive-comparison', action='store_true',
                        help='Compare fixed and adaptive HF non-final campaigns from one memory snapshot')
    parser.add_argument('--live-progress-view', action='store_true',
                        help='View HF_PROGRESS, SCIENTIST_EVENT, and HF_ARTIFACT lines from a log')
    parser.add_argument('--live-progress-file', type=str, default=None,
                        help='Progress log file to read or follow')
    parser.add_argument('--live-progress-follow', action='store_true',
                        help='Keep following the progress file after existing lines are read')
    parser.add_argument('--live-progress-max-events', type=int, default=0,
                        help='Stop after this many parsed progress events (0 means no limit)')
    parser.add_argument('--merge-final-artifacts', action='store_true',
                        help='Merge sharded final-discovery JSON artifacts without rerunning discovery')
    parser.add_argument('--merge-artifact-files', type=str, default=None,
                        help='Comma-separated JSON artifact files to merge')
    parser.add_argument('--merge-output-file', type=str, default=None,
                        help='Optional JSON path for the merged final artifact')
    parser.add_argument('--gpu-feasibility-benchmark', action='store_true',
                        help='Run a tiny non-final CPU/GPU tensor feasibility benchmark')
    parser.add_argument('--gpu-sample-count', type=int, default=50000,
                        help='Samples for the tiny GPU feasibility benchmark')
    parser.add_argument('--gpu-repeats', type=int, default=3,
                        help='Repeats for the tiny GPU feasibility benchmark')
    parser.add_argument('--gpu-output-file', type=str, default=None,
                        help='Optional JSON path for the GPU feasibility report')
    parser.add_argument('--physics-force-backend', type=str, default='python',
                        choices=['python', 'numpy', 'torch', 'cuda', 'auto'],
                        help='Simulator force backend for physics runs (default: python)')
    parser.add_argument('--equation-scoring-backend', type=str, default='python',
                        choices=['python', 'numpy', 'torch', 'cuda', 'auto'],
                        help='Numeric backend for equation scoring reductions (default: python)')
    parser.add_argument('--backend-profile-comparison', action='store_true',
                        help='Run a non-final timing comparison across physics backends')
    parser.add_argument('--backend-profile-backends', type=str, default='python,numpy',
                        help='Comma-separated backends for timing comparison')
    parser.add_argument('--backend-profile-equation-backends', type=str, default='python',
                        help='Comma-separated equation scoring backends for timing comparison')
    parser.add_argument('--backend-profile-output-file', type=str, default=None,
                        help='Optional JSON path for backend profile comparison output')
    parser.add_argument('--math-final-discovery', action='store_true',
                        help='Run the final math discovery campaign when the user is ready to watch')
    parser.add_argument('--section-study-cycles', type=int, default=1,
                        help='Repeat and consolidate each final world section before moving on')
    parser.add_argument('--parallel-cases', type=int, default=1,
                        help='Independent final cases to run concurrently inside each section cycle')
    parser.add_argument('--profile-final-run', action='store_true',
                        help='Record per-case and per-section runtime profile data in final artifacts')
    parser.add_argument('--equation-hidden-worlds', type=int, default=0,
                        help='Generated hidden worlds to include in equation campaign')
    parser.add_argument('--equation-hidden-start', type=int, default=0,
                        help='First generated hidden-world index to run in final campaign')
    parser.add_argument('--math-final-skip-known-worlds', action='store_true',
                        help='Skip named/known world sections in the final campaign')
    parser.add_argument('--self-authored-worlds', type=int, default=0,
                        help='Autonomous hidden worlds to create from theory-memory designs')
    parser.add_argument('--equation-followup-budget', type=int, default=0,
                        help='Autonomous planned follow-up probes to run after the initial equation campaign')
    parser.add_argument('--hf-log-artifact-summary', action='store_true',
                        help='Print a compact HF_ARTIFACT_SUMMARY JSON line for log-based persistence')
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
                        help='Optional JSON artifact path for HF campaign output')
    parser.add_argument('--hf-output-repo', type=str, default=None,
                        help='Optional HF dataset repo for campaign artifact upload')
    parser.add_argument('--hf-run-id', type=str, default=None,
                        help='Stable run id for HF artifact paths')
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

    if args.status_capsule:
        run_status_capsule(
            theory_memory=theory_memory,
            theory_memory_file=args.theory_memory_file,
            output_file=args.status_capsule_output_file or args.hf_output_file,
        )
        raise SystemExit(0)

    if args.module_chat_export:
        run_module_chat_export(
            theory_memory=theory_memory,
            theory_memory_file=args.theory_memory_file,
            recipient=args.module_chat_recipient,
            topic=args.module_chat_topic,
            inbox_file=args.module_chat_inbox,
            output_file=args.module_chat_output_file or args.hf_output_file,
        )
        raise SystemExit(0)

    if args.module_chat_response_loop:
        response_topic = (
            args.module_chat_topic
            if args.module_chat_topic != 'ai_different.status_capsule'
            else 'ai_different.abstraction_transfer_response'
        )
        run_module_chat_response_loop(
            theory_memory=theory_memory,
            theory_memory_file=args.theory_memory_file,
            recipient=args.module_chat_recipient,
            topic=response_topic,
            inbox_file=args.module_chat_inbox,
            output_file=args.module_chat_output_file or args.hf_output_file,
            response_mode=args.module_chat_response_mode,
            fallback_outcome_mode=args.abstraction_transfer_outcome,
        )
        raise SystemExit(0)

    if args.module_chat_family_response:
        response_topic = (
            args.module_chat_topic
            if args.module_chat_topic != 'ai_different.status_capsule'
            else 'ai_different.module_family_response'
        )
        recipient = (
            'auto'
            if args.module_chat_recipient == 'orchestrator'
            else args.module_chat_recipient
        )
        run_module_chat_family_response(
            theory_memory=theory_memory,
            theory_memory_file=args.theory_memory_file,
            recipient=recipient,
            topic=response_topic,
            inbox_file=args.module_chat_inbox,
            output_file=args.module_chat_output_file or args.hf_output_file,
            ledger_file=args.module_chat_ledger_file,
            response_mode=args.module_chat_response_mode,
            fallback_outcome_mode=args.abstraction_transfer_outcome,
        )
        raise SystemExit(0)

    if args.module_chat_rolling_family_response:
        response_topic = (
            args.module_chat_topic
            if args.module_chat_topic != 'ai_different.status_capsule'
            else 'ai_different.module_family_response'
        )
        recipient = (
            'auto'
            if args.module_chat_recipient == 'orchestrator'
            else args.module_chat_recipient
        )
        run_module_chat_rolling_family_response(
            theory_memory=theory_memory,
            theory_memory_file=args.theory_memory_file,
            recipient=recipient,
            topic=response_topic,
            chat_log_file=args.module_chat_inbox,
            output_file=args.module_chat_output_file or args.hf_output_file,
            ledger_file=args.module_chat_ledger_file,
            rolling_memory_file=args.module_chat_rolling_memory_file,
            response_mode=args.module_chat_response_mode,
            fallback_outcome_mode=args.abstraction_transfer_outcome,
        )
        raise SystemExit(0)

    if args.module_chat_outcome_evaluator:
        response_topic = (
            args.module_chat_topic
            if args.module_chat_topic != 'ai_different.status_capsule'
            else 'ai_different.family_outcome_evaluation'
        )
        ledger_files = (
            [
                item.strip()
                for item in args.module_chat_evaluator_ledger_files.split(',')
                if item.strip()
            ]
            if args.module_chat_evaluator_ledger_files
            else [args.module_chat_ledger_file]
        )
        run_family_outcome_evaluator(
            theory_memory=theory_memory,
            theory_memory_file=args.theory_memory_file,
            recipient=args.module_chat_recipient,
            topic=response_topic,
            rolling_memory_file=args.module_chat_rolling_memory_file,
            response_ledger_files=ledger_files,
            output_file=args.module_chat_output_file or args.hf_output_file,
            evaluator_ledger_file=args.module_chat_outcome_ledger_file,
            evaluator_memory_file=args.module_chat_outcome_memory_file,
        )
        raise SystemExit(0)

    if args.module_chat_experiment_contract:
        run_experiment_contract_loop(
            theory_memory=theory_memory,
            theory_memory_file=args.theory_memory_file,
            evaluator_ledger_file=args.module_chat_outcome_ledger_file,
            family_bus_file=args.module_chat_inbox,
            contract_ledger_file=args.module_chat_contract_ledger_file,
            outbox_file=args.module_chat_contract_outbox_file,
            output_file=args.module_chat_output_file or args.hf_output_file,
            target_recipient=args.module_chat_recipient
            if args.module_chat_recipient != 'orchestrator'
            else 'broadcast',
        )
        raise SystemExit(0)

    if args.module_chat_cross_module_adjudicator:
        run_cross_module_adjudicator(
            theory_memory=theory_memory,
            theory_memory_file=args.theory_memory_file,
            transcript_file=args.module_chat_inbox,
            evaluator_ledger_file=args.module_chat_outcome_ledger_file,
            contract_ledger_file=args.module_chat_contract_ledger_file,
            adjudicator_ledger_file=args.module_chat_adjudicator_ledger_file,
            outbox_file=args.module_chat_adjudicator_outbox_file,
            output_file=args.module_chat_output_file or args.hf_output_file,
        )
        raise SystemExit(0)

    if args.module_chat_experiment_agenda:
        run_experiment_agenda_scheduler(
            theory_memory=theory_memory,
            theory_memory_file=args.theory_memory_file,
            transcript_file=args.module_chat_inbox,
            evaluator_ledger_file=args.module_chat_outcome_ledger_file,
            outcome_ledger_file=args.module_chat_agenda_outcome_ledger_file,
            contract_ledger_file=args.module_chat_contract_ledger_file,
            adjudicator_ledger_file=args.module_chat_adjudicator_ledger_file,
            agenda_ledger_file=args.module_chat_agenda_ledger_file,
            outbox_file=args.module_chat_agenda_outbox_file,
            output_file=args.module_chat_output_file or args.hf_output_file,
        )
        raise SystemExit(0)

    if args.module_chat_hypothesis_lifecycle:
        run_hypothesis_lifecycle_curator(
            theory_memory=theory_memory,
            theory_memory_file=args.theory_memory_file,
            transcript_file=args.module_chat_inbox,
            evaluator_ledger_file=args.module_chat_outcome_ledger_file,
            outcome_ledger_file=args.module_chat_agenda_outcome_ledger_file,
            contract_ledger_file=args.module_chat_contract_ledger_file,
            adjudicator_ledger_file=args.module_chat_adjudicator_ledger_file,
            agenda_ledger_file=args.module_chat_agenda_ledger_file,
            lifecycle_ledger_file=args.module_chat_lifecycle_ledger_file,
            outbox_file=args.module_chat_lifecycle_outbox_file,
            output_file=args.module_chat_output_file or args.hf_output_file,
        )
        raise SystemExit(0)

    if args.module_chat_evidence_scorecard:
        run_evidence_scorecard_runner(
            theory_memory=theory_memory,
            theory_memory_file=args.theory_memory_file,
            transcript_file=args.module_chat_inbox,
            evaluator_ledger_file=args.module_chat_outcome_ledger_file,
            outcome_ledger_file=args.module_chat_agenda_outcome_ledger_file,
            contract_ledger_file=args.module_chat_contract_ledger_file,
            adjudicator_ledger_file=args.module_chat_adjudicator_ledger_file,
            agenda_ledger_file=args.module_chat_agenda_ledger_file,
            lifecycle_ledger_file=args.module_chat_lifecycle_ledger_file,
            scorecard_ledger_file=args.module_chat_scorecard_ledger_file,
            outbox_file=args.module_chat_scorecard_outbox_file,
            output_file=args.module_chat_output_file or args.hf_output_file,
        )
        raise SystemExit(0)

    if args.module_chat_experiment_campaign:
        run_experiment_campaign_planner(
            theory_memory=theory_memory,
            theory_memory_file=args.theory_memory_file,
            transcript_file=args.module_chat_inbox,
            evaluator_ledger_file=args.module_chat_outcome_ledger_file,
            outcome_ledger_file=args.module_chat_agenda_outcome_ledger_file,
            contract_ledger_file=args.module_chat_contract_ledger_file,
            adjudicator_ledger_file=args.module_chat_adjudicator_ledger_file,
            agenda_ledger_file=args.module_chat_agenda_ledger_file,
            lifecycle_ledger_file=args.module_chat_lifecycle_ledger_file,
            scorecard_ledger_file=args.module_chat_scorecard_ledger_file,
            campaign_ledger_file=args.module_chat_campaign_ledger_file,
            outbox_file=args.module_chat_campaign_outbox_file,
            output_file=args.module_chat_output_file or args.hf_output_file,
        )
        raise SystemExit(0)

    if args.module_chat_campaign_outcome:
        run_campaign_outcome_assessor(
            theory_memory=theory_memory,
            theory_memory_file=args.theory_memory_file,
            transcript_file=args.module_chat_inbox,
            evaluator_ledger_file=args.module_chat_outcome_ledger_file,
            outcome_ledger_file=args.module_chat_agenda_outcome_ledger_file,
            contract_ledger_file=args.module_chat_contract_ledger_file,
            adjudicator_ledger_file=args.module_chat_adjudicator_ledger_file,
            agenda_ledger_file=args.module_chat_agenda_ledger_file,
            lifecycle_ledger_file=args.module_chat_lifecycle_ledger_file,
            scorecard_ledger_file=args.module_chat_scorecard_ledger_file,
            campaign_ledger_file=args.module_chat_campaign_ledger_file,
            campaign_outcome_ledger_file=args.module_chat_campaign_outcome_ledger_file,
            outbox_file=args.module_chat_campaign_outcome_outbox_file,
            output_file=args.module_chat_output_file or args.hf_output_file,
        )
        raise SystemExit(0)

    if args.module_chat_science_benefit:
        run_science_benefit_evaluator(
            theory_memory=theory_memory,
            theory_memory_file=args.theory_memory_file,
            transcript_file=args.module_chat_inbox,
            evaluator_ledger_file=args.module_chat_outcome_ledger_file,
            outcome_ledger_file=args.module_chat_agenda_outcome_ledger_file,
            contract_ledger_file=args.module_chat_contract_ledger_file,
            adjudicator_ledger_file=args.module_chat_adjudicator_ledger_file,
            agenda_ledger_file=args.module_chat_agenda_ledger_file,
            lifecycle_ledger_file=args.module_chat_lifecycle_ledger_file,
            scorecard_ledger_file=args.module_chat_scorecard_ledger_file,
            campaign_ledger_file=args.module_chat_campaign_ledger_file,
            campaign_outcome_ledger_file=args.module_chat_campaign_outcome_ledger_file,
            benefit_ledger_file=args.module_chat_benefit_ledger_file,
            prior_benefit_ledger_file=args.module_chat_prior_benefit_ledger_file,
            outbox_file=args.module_chat_benefit_outbox_file,
            output_file=args.module_chat_output_file or args.hf_output_file,
        )
        raise SystemExit(0)

    if args.module_chat_science_action:
        run_science_campaign_action_planner(
            theory_memory=theory_memory,
            theory_memory_file=args.theory_memory_file,
            transcript_file=args.module_chat_inbox,
            benefit_ledger_file=args.module_chat_benefit_ledger_file,
            evaluator_ledger_file=args.module_chat_outcome_ledger_file,
            outcome_ledger_file=args.module_chat_agenda_outcome_ledger_file,
            contract_ledger_file=args.module_chat_contract_ledger_file,
            adjudicator_ledger_file=args.module_chat_adjudicator_ledger_file,
            agenda_ledger_file=args.module_chat_agenda_ledger_file,
            lifecycle_ledger_file=args.module_chat_lifecycle_ledger_file,
            scorecard_ledger_file=args.module_chat_scorecard_ledger_file,
            campaign_ledger_file=args.module_chat_campaign_ledger_file,
            campaign_outcome_ledger_file=args.module_chat_campaign_outcome_ledger_file,
            action_ledger_file=args.module_chat_action_ledger_file,
            prior_action_ledger_file=args.module_chat_prior_action_ledger_file,
            outbox_file=args.module_chat_action_outbox_file,
            output_file=args.module_chat_output_file or args.hf_output_file,
        )
        raise SystemExit(0)

    if args.module_chat_science_action_outcome:
        run_science_action_outcome_assessor(
            theory_memory=theory_memory,
            theory_memory_file=args.theory_memory_file,
            transcript_file=args.module_chat_inbox,
            action_ledger_file=args.module_chat_action_ledger_file,
            benefit_ledger_file=args.module_chat_benefit_ledger_file,
            evaluator_ledger_file=args.module_chat_outcome_ledger_file,
            outcome_ledger_file=args.module_chat_agenda_outcome_ledger_file,
            contract_ledger_file=args.module_chat_contract_ledger_file,
            adjudicator_ledger_file=args.module_chat_adjudicator_ledger_file,
            agenda_ledger_file=args.module_chat_agenda_ledger_file,
            lifecycle_ledger_file=args.module_chat_lifecycle_ledger_file,
            scorecard_ledger_file=args.module_chat_scorecard_ledger_file,
            campaign_ledger_file=args.module_chat_campaign_ledger_file,
            campaign_outcome_ledger_file=args.module_chat_campaign_outcome_ledger_file,
            action_outcome_ledger_file=args.module_chat_action_outcome_ledger_file,
            prior_action_outcome_ledger_file=args.module_chat_prior_action_outcome_ledger_file,
            outbox_file=args.module_chat_action_outcome_outbox_file,
            output_file=args.module_chat_output_file or args.hf_output_file,
        )
        raise SystemExit(0)

    if args.module_chat_science_theory_frontier:
        run_science_theory_frontier_planner(
            theory_memory=theory_memory,
            theory_memory_file=args.theory_memory_file,
            transcript_file=args.module_chat_inbox,
            action_outcome_ledger_file=args.module_chat_action_outcome_ledger_file,
            action_ledger_file=args.module_chat_action_ledger_file,
            benefit_ledger_file=args.module_chat_benefit_ledger_file,
            evaluator_ledger_file=args.module_chat_outcome_ledger_file,
            outcome_ledger_file=args.module_chat_agenda_outcome_ledger_file,
            contract_ledger_file=args.module_chat_contract_ledger_file,
            adjudicator_ledger_file=args.module_chat_adjudicator_ledger_file,
            agenda_ledger_file=args.module_chat_agenda_ledger_file,
            lifecycle_ledger_file=args.module_chat_lifecycle_ledger_file,
            scorecard_ledger_file=args.module_chat_scorecard_ledger_file,
            campaign_ledger_file=args.module_chat_campaign_ledger_file,
            campaign_outcome_ledger_file=args.module_chat_campaign_outcome_ledger_file,
            frontier_ledger_file=args.module_chat_frontier_ledger_file,
            prior_frontier_ledger_file=args.module_chat_prior_frontier_ledger_file,
            sibling_outcome_ledgers_file=args.module_chat_sibling_outcome_ledgers_file,
            outbox_file=args.module_chat_frontier_outbox_file,
            output_file=args.module_chat_output_file or args.hf_output_file,
        )
        raise SystemExit(0)

    if args.module_chat_science_theory_frontier_outcome:
        run_science_theory_frontier_outcome_assessor(
            theory_memory=theory_memory,
            theory_memory_file=args.theory_memory_file,
            transcript_file=args.module_chat_inbox,
            frontier_ledger_file=args.module_chat_frontier_ledger_file,
            action_outcome_ledger_file=args.module_chat_action_outcome_ledger_file,
            action_ledger_file=args.module_chat_action_ledger_file,
            benefit_ledger_file=args.module_chat_benefit_ledger_file,
            evaluator_ledger_file=args.module_chat_outcome_ledger_file,
            outcome_ledger_file=args.module_chat_agenda_outcome_ledger_file,
            contract_ledger_file=args.module_chat_contract_ledger_file,
            adjudicator_ledger_file=args.module_chat_adjudicator_ledger_file,
            agenda_ledger_file=args.module_chat_agenda_ledger_file,
            lifecycle_ledger_file=args.module_chat_lifecycle_ledger_file,
            scorecard_ledger_file=args.module_chat_scorecard_ledger_file,
            campaign_ledger_file=args.module_chat_campaign_ledger_file,
            campaign_outcome_ledger_file=args.module_chat_campaign_outcome_ledger_file,
            theory_memory_ledger_file=args.module_chat_theory_memory_ledger_file,
            frontier_outcome_ledger_file=args.module_chat_frontier_outcome_ledger_file,
            prior_frontier_outcome_ledger_file=args.module_chat_prior_frontier_outcome_ledger_file,
            sibling_frontier_outcome_ledgers_file=args.module_chat_sibling_frontier_outcome_ledgers_file,
            outbox_file=args.module_chat_frontier_outcome_outbox_file,
            output_file=args.module_chat_output_file or args.hf_output_file,
        )
        raise SystemExit(0)

    if args.live_progress_view:
        run_live_progress_viewer(
            progress_file=args.live_progress_file,
            follow=args.live_progress_follow,
            max_events=args.live_progress_max_events,
        )
        raise SystemExit(0)

    if args.merge_final_artifacts:
        if not args.merge_artifact_files:
            raise SystemExit('--merge-artifact-files is required')
        run_merge_final_artifacts(
            [
                item.strip()
                for item in args.merge_artifact_files.split(',')
                if item.strip()
            ],
            output_file=args.merge_output_file or args.hf_output_file,
            run_id=args.hf_run_id,
        )
        raise SystemExit(0)

    if args.gpu_feasibility_benchmark:
        run_gpu_feasibility_benchmark(
            sample_count=args.gpu_sample_count,
            repeats=args.gpu_repeats,
            force_backend=(
                None
                if args.physics_force_backend == 'python'
                else args.physics_force_backend
            ),
            output_file=args.gpu_output_file or args.hf_output_file,
        )
        raise SystemExit(0)

    if args.backend_profile_comparison:
        run_backend_profile_comparison(
            backends=[
                item.strip()
                for item in args.backend_profile_backends.split(',')
                if item.strip()
            ],
            equation_scoring_backends=[
                item.strip()
                for item in args.backend_profile_equation_backends.split(',')
                if item.strip()
            ],
            seeds=args.seeds,
            steps=args.benchmark_steps,
            object_counts=_parse_csv_ints(args.object_counts),
            world_types=_parse_csv_worlds(args.world_types),
            num_agents=args.agents,
            output_file=(
                args.backend_profile_output_file
                or args.hf_output_file
            ),
        )
        raise SystemExit(0)

    if args.benchmark_worlds:
        run_benchmark(
            seeds=args.seeds,
            steps=args.benchmark_steps,
            object_counts=_parse_csv_ints(args.object_counts),
            world_types=_parse_csv_worlds(args.world_types),
            num_agents=args.agents,
            law_memory=law_memory,
            force_backend=args.physics_force_backend,
            equation_scoring_backend=args.equation_scoring_backend,
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
            force_backend=args.physics_force_backend,
            equation_scoring_backend=args.equation_scoring_backend,
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
            force_backend=args.physics_force_backend,
            equation_scoring_backend=args.equation_scoring_backend,
        )
        raise SystemExit(0)

    if args.discovery_readiness:
        run_discovery_readiness_audit(theory_memory=theory_memory)
        raise SystemExit(0)

    if args.rediscovery_goal_progress:
        run_rediscovery_goal_progress_audit(theory_memory=theory_memory)
        raise SystemExit(0)

    if args.super_system_audit:
        run_super_system_audit(
            theory_memory=theory_memory,
            output_file=args.super_system_output_file,
            world_types=_parse_csv_worlds(args.world_types),
            object_counts=_parse_csv_ints(args.object_counts),
            steps=args.benchmark_steps,
            limit=args.super_system_limit,
        )
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

    if args.hf_adaptive_comparison:
        theory_memory = theory_memory or CumulativeTheoryMemory()
        comparison_report = run_hf_adaptive_comparison(
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
            max_adaptive_steps=args.hf_max_adaptive_steps,
            max_adaptive_seeds=args.hf_max_adaptive_seeds,
            max_adaptive_hidden_worlds=args.hf_max_adaptive_hidden_worlds,
            force_backend=args.physics_force_backend,
            equation_scoring_backend=args.equation_scoring_backend,
        )
        if (
            args.theory_memory_file
            and not args.no_save_theory_memory
            and comparison_report.get('variants')
        ):
            adaptive_memory = comparison_report['variants'][-1]['result'].get(
                'theory_memory',
                {},
            )
            CumulativeTheoryMemory.from_dict(adaptive_memory).save(
                args.theory_memory_file
            )
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
            force_backend=args.physics_force_backend,
            equation_scoring_backend=args.equation_scoring_backend,
        )
        if (
            args.theory_memory_file
            and theory_memory is not None
            and not args.no_save_theory_memory
        ):
            theory_memory.save(args.theory_memory_file)
        raise SystemExit(0)

    if args.abstraction_transfer_campaign:
        run_abstraction_transfer_campaign(
            theory_memory=theory_memory,
            seed_start=args.seed,
            steps=args.benchmark_steps,
            object_count=_parse_csv_ints(args.object_counts)[0],
            target_world_types=_parse_abstraction_transfer_worlds(args.world_types),
            outcome_mode=args.abstraction_transfer_outcome,
            emit_hf_artifact_summary=args.hf_log_artifact_summary,
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
            emit_hf_artifact_summary=args.hf_log_artifact_summary,
            force_backend=args.physics_force_backend,
            equation_scoring_backend=args.equation_scoring_backend,
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
            force_backend=args.physics_force_backend,
            equation_scoring_backend=args.equation_scoring_backend,
        )
        if (
            args.theory_memory_file
            and theory_memory is not None
            and not args.no_save_theory_memory
        ):
            theory_memory.save(args.theory_memory_file)
        raise SystemExit(0)

    if args.math_final_discovery:
        final_world_types = (
            []
            if args.math_final_skip_known_worlds
            else _parse_csv_worlds(args.world_types)
        )
        run_math_final_discovery(
            seeds=args.seeds,
            steps=args.benchmark_steps,
            object_counts=_parse_csv_ints(args.object_counts),
            world_types=final_world_types,
            hidden_worlds=args.equation_hidden_worlds,
            hidden_world_start=args.equation_hidden_start,
            self_authored_worlds=args.self_authored_worlds,
            num_agents=args.agents,
            section_study_cycles=args.section_study_cycles,
            theory_memory=theory_memory,
            theory_memory_checkpoint_file=(
                args.theory_memory_file
                if args.theory_memory_file and not args.no_save_theory_memory
                else None
            ),
            artifact_output_file=args.hf_output_file,
            hf_output_repo=args.hf_output_repo,
            run_id=args.hf_run_id,
            parallel_cases=args.parallel_cases,
            profile_final_run=args.profile_final_run,
            force_backend=args.physics_force_backend,
            equation_scoring_backend=args.equation_scoring_backend,
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
            force_backend=args.physics_force_backend,
            equation_scoring_backend=args.equation_scoring_backend,
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
            force_backend=args.physics_force_backend,
            equation_scoring_backend=args.equation_scoring_backend,
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
            force_backend=args.physics_force_backend,
            equation_scoring_backend=args.equation_scoring_backend,
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
        force_backend=args.physics_force_backend,
        equation_scoring_backend=args.equation_scoring_backend,
    )
    if args.memory_file and law_memory is not None and not args.no_save_memory:
        law_memory.save(args.memory_file)
