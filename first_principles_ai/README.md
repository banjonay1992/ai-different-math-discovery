# First Principles AI — Discovery from Scratch

An agent starts with **zero knowledge** about mathematics or physics. It observes a 2D physics world and must discover concepts independently — no training data, no human examples, just observation and reasoning.

After the experiment, we compare the agent's discoveries to human concepts. If they converge, those concepts are properties of reality — not human inventions.

## The Idea

Current AI trains on human output and learns to produce human-like output. We never test whether it *understands* anything. This project takes a different approach:

1. Give an agent a world to observe (with real physics: gravity, collisions, friction)
2. Let it form its own concepts from observation
3. Let it hypothesize and test rules about how the world works
4. Compare its discoveries to human mathematical/physical concepts
5. If they match — those concepts are real, not cultural artifacts

This is **convergent epistemology**: if two independent systems arrive at the same truth, that truth is likely real.

## What the Agent Discovers

Running `python3 main.py --steps 2000` produces:

- **11 concepts** (100% convergence with human knowledge):
  - Natural Numbers (Counting)
  - Linear Momentum (x, y, magnitude)
  - Kinetic Energy
  - Mass (Additivity)
  - Center of Mass (x, y)
  - Collision/Interaction Events
  - Euclidean Distance
  - Average Speed

- **2 confirmed rules** (100% convergence):
  - Successor Function: "When an object appears, count increases by exactly 1" (Peano axiom)
  - Predecessor Function: "When an object disappears, count decreases by exactly 1"

- **Conservation hypotheses correctly rejected** — the agent noticed that momentum and energy are NOT conserved in a world with gravity and friction. This is correct physics.

## Architecture

```
first_principles_ai/
├── world/
│   ├── physics.py       # 2D physics engine (gravity, collisions, friction)
│   └── environment.py   # Agent's interface to the world (observe, push, spawn, remove)
├── agent/
│   ├── perception.py    # Converts raw state into structured observations
│   ├── representation.py  # Explicit, inspectable knowledge base (concepts, rules, hypotheses)
│   ├── predictor.py     # Reasoning engine: observe → detect regularities → hypothesize → test
│   └── curiosity.py     # Curiosity-driven action selection (seek maximum information gain)
├── discovery/
│   ├── tracker.py       # Logs all discoveries with timestamps and evidence
│   └── comparison.py    # Maps agent discoveries to human concepts (convergence test)
├── main.py              # Entry point — runs the experiment
└── requirements.txt
```

## Key Design Decisions

### Knowledge is explicit, not neural weights
The agent's knowledge is stored as inspectable `Concept` and `Rule` objects. You can see what it knows, when it learned it, and what evidence supports it. This makes the system **correctable** — you can inspect and fix individual beliefs.

### Curiosity drives exploration
The agent has no external reward. It is driven by **prediction error** — when its predictions are wrong, it gets curious. It seeks situations where it doesn't know what will happen. This is the same drive that makes children learn.

### Hypothesis testing, not pattern matching
The agent doesn't learn by correlating inputs with outputs. It:
1. Observes features over time
2. Detects regularities (constants, patterns)
3. Forms explicit hypotheses ("momentum is conserved")
4. Tests them against new observations
5. Confirms or rejects based on evidence

### Self-correcting
Because knowledge is built from first principles, the agent can:
- Trace back through its reasoning chain
- Detect contradictions in its own knowledge
- Reject hypotheses that don't match reality
- Re-derive anything from observation

## Usage

```bash
# Install dependencies
pip3 install -r first_principles_ai/requirements.txt

# Optional GPU path (if you plan to run CUDA-feasibility or tensor backends on GPU)
pip3 install -r first_principles_ai/requirements-gpu.txt

# Run the experiment (default: 2000 steps)
python3 main.py

# Custom configuration
python3 main.py --steps 5000 --objects 10 --seed 123

# Try alternate physics worlds
python3 main.py --world-type vortex --steps 1000

# Benchmark detection across worlds and seeds
python3 main.py --benchmark-worlds --seeds 20 --benchmark-steps 800 --object-counts 4,5,8

# Let the agent choose many experiments from a candidate pool
python3 main.py --explore-worlds --exploration-budget 24 --benchmark-steps 500 --seeds 12 --object-counts 4,5,8

# Benchmark emergent math coverage across repeated runs
python3 main.py --math-benchmark --seeds 3 --benchmark-steps 160 --world-types standard --object-counts 5

# Build equation review packs from a starter-kit equation workbench
python3 main.py --equation-campaign --seeds 1 --benchmark-steps 220 --world-types standard,sideways_wind,vortex --equation-hidden-worlds 2

# Check whether the math-foundation gates are ready before the final watched run
python3 main.py --math-foundation-prep --seeds 1 --benchmark-steps 220 --world-types standard,sideways_wind,vortex --equation-hidden-worlds 2

# Inspect the saved discovery notebook without running simulations or the final
python3 main.py --discovery-readiness --theory-memory-file tmp/theory-memory.json

# Export an orchestrator-facing status capsule as a plain module-chat message
python3 main.py --module-chat-export --module-chat-inbox tmp/module-chat-inbox.jsonl --module-chat-output-file tmp/ai-different-module-message.json --theory-memory-file tmp/theory-memory.json

# Process a richer Language/Code/funfun exchange once, persisting a compact response ledger
python3 main.py --module-chat-family-response --module-chat-response-mode run --module-chat-inbox tmp/module-chat-log.jsonl --module-chat-ledger-file tmp/module-chat-response-ledger.json --theory-memory-file tmp/theory-memory.json

# Process a rolling module-chat log idempotently, skipping old inbound messages
python3 main.py --module-chat-rolling-family-response --module-chat-response-mode run --module-chat-inbox tmp/module-chat-log.jsonl --module-chat-rolling-memory-file tmp/module-chat-family-memory.json --module-chat-ledger-file tmp/module-chat-response-ledger.json --theory-memory-file tmp/theory-memory.json

# Evaluate accumulated rolling family evidence into one next science experiment/defer decision
python3 main.py --module-chat-outcome-evaluator --module-chat-rolling-memory-file tmp/module-chat-family-memory.json --module-chat-evaluator-ledger-files tmp/module-chat-response-ledger.json --module-chat-outcome-memory-file tmp/module-chat-outcome-evaluator-memory.json --module-chat-outcome-ledger-file tmp/module-chat-outcome-evaluator-ledger.json --theory-memory-file tmp/theory-memory.json

# Emit/resolve an AI Different experiment contract from the evaluator ledger and family bus evidence
python3 main.py --module-chat-experiment-contract --module-chat-outcome-ledger-file tmp/module-chat-outcome-evaluator-ledger.json --module-chat-inbox tmp/module-chat-log.jsonl --module-chat-contract-ledger-file tmp/module-chat-experiment-contract-ledger.json --module-chat-contract-outbox-file tmp/module-chat-experiment-contract-outbox.jsonl --theory-memory-file tmp/theory-memory.json

# Adjudicate a full family turn and request repair, resolve a contract, or emit the next safe contract
python3 main.py --module-chat-cross-module-adjudicator --module-chat-inbox tmp/module-chat-log.jsonl --module-chat-outcome-ledger-file tmp/module-chat-outcome-evaluator-ledger.json --module-chat-contract-ledger-file tmp/module-chat-experiment-contract-ledger.json --module-chat-adjudicator-ledger-file tmp/module-chat-cross-module-adjudicator-ledger.json --module-chat-adjudicator-outbox-file tmp/module-chat-cross-module-adjudicator-outbox.jsonl --theory-memory-file tmp/theory-memory.json

# Schedule the next safe experiment agenda item from adjudicated module-family evidence
python3 main.py --module-chat-experiment-agenda --module-chat-inbox tmp/module-chat-log.jsonl --module-chat-outcome-ledger-file tmp/module-chat-outcome-evaluator-ledger.json --module-chat-contract-ledger-file tmp/module-chat-experiment-contract-ledger.json --module-chat-adjudicator-ledger-file tmp/module-chat-cross-module-adjudicator-ledger.json --module-chat-agenda-ledger-file tmp/module-chat-experiment-agenda-ledger.json --module-chat-agenda-outbox-file tmp/module-chat-experiment-agenda-outbox.jsonl --theory-memory-file tmp/theory-memory.json

# Curate hypothesis lifecycle memory from agendas, contracts, adjudication, and sibling evidence
python3 main.py --module-chat-hypothesis-lifecycle --module-chat-inbox tmp/module-chat-log.jsonl --module-chat-outcome-ledger-file tmp/module-chat-outcome-evaluator-ledger.json --module-chat-contract-ledger-file tmp/module-chat-experiment-contract-ledger.json --module-chat-adjudicator-ledger-file tmp/module-chat-cross-module-adjudicator-ledger.json --module-chat-agenda-ledger-file tmp/module-chat-experiment-agenda-ledger.json --module-chat-lifecycle-ledger-file tmp/module-chat-hypothesis-lifecycle-ledger.json --module-chat-lifecycle-outbox-file tmp/module-chat-hypothesis-lifecycle-outbox.jsonl --theory-memory-file tmp/theory-memory.json

# Score hypothesis evidence gates and choose resolve, repair, retire, or one refinement
python3 main.py --module-chat-evidence-scorecard --module-chat-inbox tmp/module-chat-log.jsonl --module-chat-outcome-ledger-file tmp/module-chat-outcome-evaluator-ledger.json --module-chat-contract-ledger-file tmp/module-chat-experiment-contract-ledger.json --module-chat-adjudicator-ledger-file tmp/module-chat-cross-module-adjudicator-ledger.json --module-chat-agenda-ledger-file tmp/module-chat-experiment-agenda-ledger.json --module-chat-lifecycle-ledger-file tmp/module-chat-hypothesis-lifecycle-ledger.json --module-chat-scorecard-ledger-file tmp/module-chat-evidence-scorecard-ledger.json --module-chat-scorecard-outbox-file tmp/module-chat-evidence-scorecard-outbox.jsonl --theory-memory-file tmp/theory-memory.json

# Plan the next symbolic campaign or acceptance bundle from scored hypothesis evidence
python3 main.py --module-chat-experiment-campaign --module-chat-inbox tmp/module-chat-log.jsonl --module-chat-outcome-ledger-file tmp/module-chat-outcome-evaluator-ledger.json --module-chat-contract-ledger-file tmp/module-chat-experiment-contract-ledger.json --module-chat-adjudicator-ledger-file tmp/module-chat-cross-module-adjudicator-ledger.json --module-chat-agenda-ledger-file tmp/module-chat-experiment-agenda-ledger.json --module-chat-lifecycle-ledger-file tmp/module-chat-hypothesis-lifecycle-ledger.json --module-chat-scorecard-ledger-file tmp/module-chat-evidence-scorecard-ledger.json --module-chat-campaign-ledger-file tmp/module-chat-experiment-campaign-ledger.json --module-chat-campaign-outbox-file tmp/module-chat-experiment-campaign-outbox.jsonl --theory-memory-file tmp/theory-memory.json

# Assess campaign return evidence and plan accept, repair, refine, retire, or more evidence
python3 main.py --module-chat-campaign-outcome --module-chat-inbox tmp/module-chat-log.jsonl --module-chat-outcome-ledger-file tmp/module-chat-outcome-evaluator-ledger.json --module-chat-contract-ledger-file tmp/module-chat-experiment-contract-ledger.json --module-chat-adjudicator-ledger-file tmp/module-chat-cross-module-adjudicator-ledger.json --module-chat-agenda-ledger-file tmp/module-chat-experiment-agenda-ledger.json --module-chat-lifecycle-ledger-file tmp/module-chat-hypothesis-lifecycle-ledger.json --module-chat-scorecard-ledger-file tmp/module-chat-evidence-scorecard-ledger.json --module-chat-campaign-ledger-file tmp/module-chat-experiment-campaign-ledger.json --module-chat-campaign-outcome-ledger-file tmp/module-chat-experiment-campaign-outcome-ledger.json --module-chat-campaign-outcome-outbox-file tmp/module-chat-experiment-campaign-outcome-outbox.jsonl --theory-memory-file tmp/theory-memory.json

# Compare isolated and connected campaign evidence for symbolic benefit
python3 main.py --module-chat-science-benefit --module-chat-inbox tmp/module-chat-log.jsonl --module-chat-outcome-ledger-file tmp/module-chat-outcome-evaluator-ledger.json --module-chat-contract-ledger-file tmp/module-chat-experiment-contract-ledger.json --module-chat-adjudicator-ledger-file tmp/module-chat-cross-module-adjudicator-ledger.json --module-chat-agenda-ledger-file tmp/module-chat-experiment-agenda-ledger.json --module-chat-lifecycle-ledger-file tmp/module-chat-hypothesis-lifecycle-ledger.json --module-chat-scorecard-ledger-file tmp/module-chat-evidence-scorecard-ledger.json --module-chat-campaign-ledger-file tmp/module-chat-experiment-campaign-ledger.json --module-chat-campaign-outcome-ledger-file tmp/module-chat-experiment-campaign-outcome-ledger.json --module-chat-benefit-ledger-file tmp/module-chat-science-benefit-ledger.json --module-chat-benefit-outbox-file tmp/module-chat-science-benefit-outbox.jsonl --theory-memory-file tmp/theory-memory.json

# Plan one safe campaign action from science-benefit evidence
python3 main.py --module-chat-science-action --module-chat-inbox tmp/module-chat-log.jsonl --module-chat-benefit-ledger-file tmp/module-chat-science-benefit-ledger.json --module-chat-outcome-ledger-file tmp/module-chat-outcome-evaluator-ledger.json --module-chat-contract-ledger-file tmp/module-chat-experiment-contract-ledger.json --module-chat-adjudicator-ledger-file tmp/module-chat-cross-module-adjudicator-ledger.json --module-chat-agenda-ledger-file tmp/module-chat-experiment-agenda-ledger.json --module-chat-lifecycle-ledger-file tmp/module-chat-hypothesis-lifecycle-ledger.json --module-chat-scorecard-ledger-file tmp/module-chat-evidence-scorecard-ledger.json --module-chat-campaign-ledger-file tmp/module-chat-experiment-campaign-ledger.json --module-chat-campaign-outcome-ledger-file tmp/module-chat-experiment-campaign-outcome-ledger.json --module-chat-action-ledger-file tmp/module-chat-science-campaign-action-ledger.json --module-chat-action-outbox-file tmp/module-chat-science-campaign-action-outbox.jsonl --theory-memory-file tmp/theory-memory.json

# Assess whether planned science-campaign actions received useful evidence
python3 main.py --module-chat-science-action-outcome --module-chat-inbox tmp/module-chat-log.jsonl --module-chat-action-ledger-file tmp/module-chat-science-campaign-action-ledger.json --module-chat-benefit-ledger-file tmp/module-chat-science-benefit-ledger.json --module-chat-outcome-ledger-file tmp/module-chat-outcome-evaluator-ledger.json --module-chat-contract-ledger-file tmp/module-chat-experiment-contract-ledger.json --module-chat-adjudicator-ledger-file tmp/module-chat-cross-module-adjudicator-ledger.json --module-chat-agenda-ledger-file tmp/module-chat-experiment-agenda-ledger.json --module-chat-lifecycle-ledger-file tmp/module-chat-hypothesis-lifecycle-ledger.json --module-chat-scorecard-ledger-file tmp/module-chat-evidence-scorecard-ledger.json --module-chat-campaign-ledger-file tmp/module-chat-experiment-campaign-ledger.json --module-chat-campaign-outcome-ledger-file tmp/module-chat-experiment-campaign-outcome-ledger.json --module-chat-action-outcome-ledger-file tmp/module-chat-science-action-outcome-ledger.json --module-chat-action-outcome-outbox-file tmp/module-chat-science-action-outcome-outbox.jsonl --theory-memory-file tmp/theory-memory.json

# Plan the next durable theory-frontier move from assessed science outcomes
python3 main.py --module-chat-science-theory-frontier --module-chat-inbox tmp/module-chat-log.jsonl --module-chat-action-outcome-ledger-file tmp/module-chat-science-action-outcome-ledger.json --module-chat-action-ledger-file tmp/module-chat-science-campaign-action-ledger.json --module-chat-benefit-ledger-file tmp/module-chat-science-benefit-ledger.json --module-chat-outcome-ledger-file tmp/module-chat-outcome-evaluator-ledger.json --module-chat-contract-ledger-file tmp/module-chat-experiment-contract-ledger.json --module-chat-adjudicator-ledger-file tmp/module-chat-cross-module-adjudicator-ledger.json --module-chat-agenda-ledger-file tmp/module-chat-experiment-agenda-ledger.json --module-chat-lifecycle-ledger-file tmp/module-chat-hypothesis-lifecycle-ledger.json --module-chat-scorecard-ledger-file tmp/module-chat-evidence-scorecard-ledger.json --module-chat-campaign-ledger-file tmp/module-chat-experiment-campaign-ledger.json --module-chat-campaign-outcome-ledger-file tmp/module-chat-experiment-campaign-outcome-ledger.json --module-chat-frontier-ledger-file tmp/module-chat-science-theory-frontier-ledger.json --module-chat-frontier-outbox-file tmp/module-chat-science-theory-frontier-outbox.jsonl --theory-memory-file tmp/theory-memory.json

# Assess whether theory-frontier moves changed symbolic theory state
python3 main.py --module-chat-science-theory-frontier-outcome --module-chat-inbox tmp/module-chat-log.jsonl --module-chat-frontier-ledger-file tmp/module-chat-science-theory-frontier-ledger.json --module-chat-action-outcome-ledger-file tmp/module-chat-science-action-outcome-ledger.json --module-chat-action-ledger-file tmp/module-chat-science-campaign-action-ledger.json --module-chat-benefit-ledger-file tmp/module-chat-science-benefit-ledger.json --module-chat-outcome-ledger-file tmp/module-chat-outcome-evaluator-ledger.json --module-chat-contract-ledger-file tmp/module-chat-experiment-contract-ledger.json --module-chat-adjudicator-ledger-file tmp/module-chat-cross-module-adjudicator-ledger.json --module-chat-agenda-ledger-file tmp/module-chat-experiment-agenda-ledger.json --module-chat-lifecycle-ledger-file tmp/module-chat-hypothesis-lifecycle-ledger.json --module-chat-scorecard-ledger-file tmp/module-chat-evidence-scorecard-ledger.json --module-chat-campaign-ledger-file tmp/module-chat-experiment-campaign-ledger.json --module-chat-campaign-outcome-ledger-file tmp/module-chat-experiment-campaign-outcome-ledger.json --module-chat-frontier-outcome-ledger-file tmp/module-chat-science-theory-frontier-outcome-ledger.json --module-chat-frontier-outcome-outbox-file tmp/module-chat-science-theory-frontier-outcome-outbox.jsonl --theory-memory-file tmp/theory-memory.json

# Plan the next durable science campaign strategy from frontier outcomes
python3 main.py --module-chat-science-campaign-strategy --module-chat-inbox tmp/module-chat-log.jsonl --module-chat-frontier-outcome-ledger-file tmp/module-chat-science-theory-frontier-outcome-ledger.json --module-chat-frontier-ledger-file tmp/module-chat-science-theory-frontier-ledger.json --module-chat-campaign-ledger-file tmp/module-chat-experiment-campaign-ledger.json --module-chat-campaign-strategy-ledger-file tmp/module-chat-science-campaign-strategy-ledger.json --module-chat-campaign-strategy-outbox-file tmp/module-chat-science-campaign-strategy-outbox.jsonl --theory-memory-file tmp/theory-memory.json

# Assess whether science campaign strategies changed symbolic campaign state
python3 main.py --module-chat-science-campaign-strategy-outcome --module-chat-inbox tmp/module-chat-log.jsonl --module-chat-campaign-strategy-ledger-file tmp/module-chat-science-campaign-strategy-ledger.json --module-chat-frontier-outcome-ledger-file tmp/module-chat-science-theory-frontier-outcome-ledger.json --module-chat-frontier-ledger-file tmp/module-chat-science-theory-frontier-ledger.json --module-chat-campaign-ledger-file tmp/module-chat-experiment-campaign-ledger.json --module-chat-campaign-strategy-outcome-ledger-file tmp/module-chat-science-campaign-strategy-outcome-ledger.json --module-chat-campaign-strategy-outcome-outbox-file tmp/module-chat-science-campaign-strategy-outcome-outbox.jsonl --theory-memory-file tmp/theory-memory.json

# Plan the next higher-level science campaign cycle strategy from strategy outcomes
python3 main.py --module-chat-science-campaign-cycle-strategy --module-chat-inbox tmp/module-chat-log.jsonl --module-chat-campaign-strategy-outcome-ledger-file tmp/module-chat-science-campaign-strategy-outcome-ledger.json --module-chat-campaign-strategy-ledger-file tmp/module-chat-science-campaign-strategy-ledger.json --module-chat-frontier-outcome-ledger-file tmp/module-chat-science-theory-frontier-outcome-ledger.json --module-chat-frontier-ledger-file tmp/module-chat-science-theory-frontier-ledger.json --module-chat-campaign-ledger-file tmp/module-chat-experiment-campaign-ledger.json --module-chat-campaign-cycle-strategy-ledger-file tmp/module-chat-science-campaign-cycle-strategy-ledger.json --module-chat-campaign-cycle-strategy-outbox-file tmp/module-chat-science-campaign-cycle-strategy-outbox.jsonl --theory-memory-file tmp/theory-memory.json

# Assess whether science campaign cycle strategies changed symbolic campaign state
python3 main.py --module-chat-science-campaign-cycle-strategy-outcome --module-chat-inbox tmp/module-chat-log.jsonl --module-chat-campaign-cycle-strategy-ledger-file tmp/module-chat-science-campaign-cycle-strategy-ledger.json --module-chat-campaign-strategy-outcome-ledger-file tmp/module-chat-science-campaign-strategy-outcome-ledger.json --module-chat-campaign-strategy-ledger-file tmp/module-chat-science-campaign-strategy-ledger.json --module-chat-frontier-outcome-ledger-file tmp/module-chat-science-theory-frontier-outcome-ledger.json --module-chat-frontier-ledger-file tmp/module-chat-science-theory-frontier-ledger.json --module-chat-campaign-ledger-file tmp/module-chat-experiment-campaign-ledger.json --module-chat-campaign-cycle-strategy-outcome-ledger-file tmp/module-chat-science-campaign-cycle-strategy-outcome-ledger.json --module-chat-campaign-cycle-strategy-outcome-outbox-file tmp/module-chat-science-campaign-cycle-strategy-outcome-outbox.jsonl --theory-memory-file tmp/theory-memory.json

# Export local science coordination payoff records for the history participant
python3 main.py --module-chat-science-coordination-history --module-chat-inbox tmp/module-chat-log.jsonl --module-chat-campaign-cycle-strategy-outcome-ledger-file tmp/module-chat-science-campaign-cycle-strategy-outcome-ledger.json --module-chat-campaign-cycle-strategy-ledger-file tmp/module-chat-science-campaign-cycle-strategy-ledger.json --module-chat-campaign-strategy-outcome-ledger-file tmp/module-chat-science-campaign-strategy-outcome-ledger.json --module-chat-campaign-strategy-ledger-file tmp/module-chat-science-campaign-strategy-ledger.json --module-chat-frontier-outcome-ledger-file tmp/module-chat-science-theory-frontier-outcome-ledger.json --module-chat-frontier-ledger-file tmp/module-chat-science-theory-frontier-ledger.json --module-chat-science-history-ledger-file tmp/module-chat-science-coordination-history-ledger.json --module-chat-science-history-outbox-file tmp/module-chat-science-coordination-history-outbox.jsonl --theory-memory-file tmp/theory-memory.json

# Recommend a cautious science-side coordination policy from local history records
python3 main.py --module-chat-science-coordination-policy --module-chat-inbox tmp/module-chat-log.jsonl --module-chat-science-history-ledger-file tmp/module-chat-science-coordination-history-ledger.json --module-chat-campaign-cycle-strategy-outcome-ledger-file tmp/module-chat-science-campaign-cycle-strategy-outcome-ledger.json --module-chat-campaign-cycle-strategy-ledger-file tmp/module-chat-science-campaign-cycle-strategy-ledger.json --module-chat-campaign-strategy-outcome-ledger-file tmp/module-chat-science-campaign-strategy-outcome-ledger.json --module-chat-campaign-strategy-ledger-file tmp/module-chat-science-campaign-strategy-ledger.json --module-chat-frontier-outcome-ledger-file tmp/module-chat-science-theory-frontier-outcome-ledger.json --module-chat-science-policy-ledger-file tmp/module-chat-science-coordination-policy-ledger.json --module-chat-science-policy-outbox-file tmp/module-chat-science-coordination-policy-outbox.jsonl --theory-memory-file tmp/theory-memory.json

# Assess whether science coordination policies changed symbolic campaign outcomes
python3 main.py --module-chat-science-coordination-policy-outcome --module-chat-inbox tmp/module-chat-log.jsonl --module-chat-science-policy-ledger-file tmp/module-chat-science-coordination-policy-ledger.json --module-chat-science-history-ledger-file tmp/module-chat-science-coordination-history-ledger.json --module-chat-campaign-cycle-strategy-outcome-ledger-file tmp/module-chat-science-campaign-cycle-strategy-outcome-ledger.json --module-chat-campaign-cycle-strategy-ledger-file tmp/module-chat-science-campaign-cycle-strategy-ledger.json --module-chat-campaign-strategy-outcome-ledger-file tmp/module-chat-science-campaign-strategy-outcome-ledger.json --module-chat-frontier-outcome-ledger-file tmp/module-chat-science-theory-frontier-outcome-ledger.json --module-chat-science-policy-outcome-ledger-file tmp/module-chat-science-coordination-policy-outcome-ledger.json --module-chat-science-policy-outcome-outbox-file tmp/module-chat-science-coordination-policy-outcome-outbox.jsonl --theory-memory-file tmp/theory-memory.json

# Aggregate repeated science coordination policy outcomes into a cautious local scorecard
python3 main.py --module-chat-science-coordination-policy-scorecard --module-chat-inbox tmp/module-chat-log.jsonl --module-chat-science-policy-outcome-ledger-file tmp/module-chat-science-coordination-policy-outcome-ledger.json --module-chat-science-policy-ledger-file tmp/module-chat-science-coordination-policy-ledger.json --module-chat-science-history-ledger-file tmp/module-chat-science-coordination-history-ledger.json --module-chat-campaign-cycle-strategy-outcome-ledger-file tmp/module-chat-science-campaign-cycle-strategy-outcome-ledger.json --module-chat-science-policy-scorecard-ledger-file tmp/module-chat-science-coordination-policy-scorecard-ledger.json --module-chat-science-policy-scorecard-outbox-file tmp/module-chat-science-coordination-policy-scorecard-outbox.jsonl --theory-memory-file tmp/theory-memory.json

# Plan a bounded science coordination A/B probe from local scorecards
python3 main.py --module-chat-science-coordination-ab-probe --module-chat-inbox tmp/module-chat-log.jsonl --module-chat-science-policy-scorecard-ledger-file tmp/module-chat-science-coordination-policy-scorecard-ledger.json --module-chat-science-policy-outcome-ledger-file tmp/module-chat-science-coordination-policy-outcome-ledger.json --module-chat-science-policy-ledger-file tmp/module-chat-science-coordination-policy-ledger.json --module-chat-science-history-ledger-file tmp/module-chat-science-coordination-history-ledger.json --module-chat-campaign-cycle-strategy-outcome-ledger-file tmp/module-chat-science-campaign-cycle-strategy-outcome-ledger.json --module-chat-science-ab-probe-ledger-file tmp/module-chat-science-coordination-ab-probe-ledger.json --module-chat-science-ab-probe-outbox-file tmp/module-chat-science-coordination-ab-probe-outbox.jsonl --theory-memory-file tmp/theory-memory.json

# Assess whether a science coordination A/B probe produced useful symbolic evidence
python3 main.py --module-chat-science-coordination-ab-probe-outcome --module-chat-inbox tmp/module-chat-log.jsonl --module-chat-science-ab-probe-ledger-file tmp/module-chat-science-coordination-ab-probe-ledger.json --module-chat-science-policy-scorecard-ledger-file tmp/module-chat-science-coordination-policy-scorecard-ledger.json --module-chat-science-policy-outcome-ledger-file tmp/module-chat-science-coordination-policy-outcome-ledger.json --module-chat-science-history-ledger-file tmp/module-chat-science-coordination-history-ledger.json --module-chat-campaign-cycle-strategy-outcome-ledger-file tmp/module-chat-science-campaign-cycle-strategy-outcome-ledger.json --module-chat-science-ab-probe-outcome-ledger-file tmp/module-chat-science-coordination-ab-probe-outcome-ledger.json --module-chat-science-ab-probe-outcome-outbox-file tmp/module-chat-science-coordination-ab-probe-outcome-outbox.jsonl --theory-memory-file tmp/theory-memory.json

# Aggregate repeated science coordination A/B outcomes into interaction-history memory
python3 main.py --module-chat-science-coordination-interaction-history --module-chat-inbox tmp/module-chat-log.jsonl --module-chat-science-ab-probe-outcome-ledger-file tmp/module-chat-science-coordination-ab-probe-outcome-ledger.json --module-chat-science-ab-probe-ledger-file tmp/module-chat-science-coordination-ab-probe-ledger.json --module-chat-science-policy-scorecard-ledger-file tmp/module-chat-science-coordination-policy-scorecard-ledger.json --module-chat-science-policy-outcome-ledger-file tmp/module-chat-science-coordination-policy-outcome-ledger.json --module-chat-science-history-ledger-file tmp/module-chat-science-coordination-history-ledger.json --module-chat-campaign-cycle-strategy-outcome-ledger-file tmp/module-chat-science-campaign-cycle-strategy-outcome-ledger.json --module-chat-science-interaction-history-ledger-file tmp/module-chat-science-coordination-interaction-history-ledger.json --module-chat-science-interaction-history-outbox-file tmp/module-chat-science-coordination-interaction-history-outbox.jsonl --theory-memory-file tmp/theory-memory.json

# Plan the next science coordination policy from local interaction-history evidence
python3 main.py --module-chat-science-history-guided-policy --module-chat-inbox tmp/module-chat-log.jsonl --module-chat-science-interaction-history-ledger-file tmp/module-chat-science-coordination-interaction-history-ledger.json --module-chat-science-ab-probe-outcome-ledger-file tmp/module-chat-science-coordination-ab-probe-outcome-ledger.json --module-chat-science-ab-probe-ledger-file tmp/module-chat-science-coordination-ab-probe-ledger.json --module-chat-science-policy-scorecard-ledger-file tmp/module-chat-science-coordination-policy-scorecard-ledger.json --module-chat-science-policy-outcome-ledger-file tmp/module-chat-science-coordination-policy-outcome-ledger.json --module-chat-science-history-ledger-file tmp/module-chat-science-coordination-history-ledger.json --module-chat-campaign-cycle-strategy-outcome-ledger-file tmp/module-chat-science-campaign-cycle-strategy-outcome-ledger.json --module-chat-science-history-guided-policy-ledger-file tmp/module-chat-science-history-guided-policy-ledger.json --module-chat-science-history-guided-policy-outbox-file tmp/module-chat-science-history-guided-policy-outbox.jsonl --theory-memory-file tmp/theory-memory.json

# Assess whether a science history-guided policy changed symbolic outcomes
python3 main.py --module-chat-science-history-guided-policy-outcome --module-chat-inbox tmp/module-chat-log.jsonl --module-chat-science-history-guided-policy-ledger-file tmp/module-chat-science-history-guided-policy-ledger.json --module-chat-science-interaction-history-ledger-file tmp/module-chat-science-coordination-interaction-history-ledger.json --module-chat-science-ab-probe-outcome-ledger-file tmp/module-chat-science-coordination-ab-probe-outcome-ledger.json --module-chat-science-ab-probe-ledger-file tmp/module-chat-science-coordination-ab-probe-ledger.json --module-chat-science-history-guided-policy-outcome-ledger-file tmp/module-chat-science-history-guided-policy-outcome-ledger.json --module-chat-science-history-guided-policy-outcome-outbox-file tmp/module-chat-science-history-guided-policy-outcome-outbox.jsonl --theory-memory-file tmp/theory-memory.json

# Aggregate repeated science history-guided policy outcomes into retention memory
python3 main.py --module-chat-science-history-guided-policy-retention --module-chat-inbox tmp/module-chat-log.jsonl --module-chat-science-history-guided-policy-outcome-ledger-file tmp/module-chat-science-history-guided-policy-outcome-ledger.json --module-chat-science-history-guided-policy-ledger-file tmp/module-chat-science-history-guided-policy-ledger.json --module-chat-science-interaction-history-ledger-file tmp/module-chat-science-coordination-interaction-history-ledger.json --module-chat-science-ab-probe-outcome-ledger-file tmp/module-chat-science-coordination-ab-probe-outcome-ledger.json --module-chat-science-ab-probe-ledger-file tmp/module-chat-science-coordination-ab-probe-ledger.json --module-chat-science-history-guided-policy-retention-ledger-file tmp/module-chat-science-history-guided-policy-retention-ledger.json --module-chat-science-history-guided-policy-retention-outbox-file tmp/module-chat-science-history-guided-policy-retention-outbox.jsonl --theory-memory-file tmp/theory-memory.json

# Inspect long-run memory growth and quantized compression state
python3 main.py --memory-efficiency-review --theory-memory-file tmp/theory-memory.json

# Compact older theory evidence into bounded quantized shards while keeping recent raw detail
python3 main.py --compact-theory-memory --memory-keep-records 96 --memory-keep-operator-outcomes 192 --theory-memory-file tmp/theory-memory.json

# Preview generated math-domain worlds without running the final
python3 main.py --domain-curriculum-preview --theory-memory-file tmp/theory-memory.json

# Persist generated domain-world discoveries into the theory notebook
python3 main.py --domain-world-discovery-ingest --theory-memory-file tmp/theory-memory.json

# Consolidate domain discoveries into invariants, residual probes, harder worlds, and live events
python3 main.py --autonomous-scientist-loop --scientist-seed-count 3 --scientist-variants 0,1 --scientist-live --theory-memory-file tmp/theory-memory.json

# HF-friendly non-final run: emits HF_PROGRESS lines and writes a JSON artifact
python3 main.py --hf-non-final-campaign --benchmark-steps 80 --world-types standard --object-counts 3 --equation-hidden-worlds 0 --theory-memory-file tmp/theory-memory.json --hf-output-file tmp/hf-non-final-report.json

# Bigger HF non-final run with multi-seed domain discovery and live scientist trace
python3 main.py --hf-non-final-campaign --benchmark-steps 160 --seeds 2 --world-types standard,sideways_wind,vortex,localized_gravity,time_varying,inverse_square_repulsion --object-counts 3,4 --equation-hidden-worlds 2 --domain-world-seed-count 4 --domain-world-variants 0,1 --scientist-seed-count 4 --scientist-variants 0,1 --theory-memory-file tmp/theory-memory.json --hf-output-file tmp/hf-non-final-report.json

# Long HF runs auto-compact theory memory and adapt compute only under residual pressure
python3 main.py --hf-non-final-campaign --benchmark-steps 160 --hf-max-adaptive-steps 320 --hf-max-adaptive-seeds 3 --memory-keep-records 96 --memory-keep-operator-outcomes 192 --theory-memory-file tmp/theory-memory.json --hf-output-file tmp/hf-non-final-report.json

# Compare fixed versus adaptive non-final compute from the same memory snapshot.
# Adaptive prep targets high-value worlds first, so the report can prove readiness per compute.
python3 main.py --hf-adaptive-comparison --benchmark-steps 80 --seeds 1 --world-types standard,localized_gravity,time_varying --object-counts 3 --equation-hidden-worlds 1 --hf-max-adaptive-steps 160 --hf-max-adaptive-seeds 2 --hf-max-adaptive-hidden-worlds 2 --theory-memory-file tmp/theory-memory.json --hf-output-file tmp/hf-adaptive-comparison-report.json

# Heavy/final campaigns should be launched on Hugging Face compute where practical
# so local preview and tests stay lightweight.

# Run the watched final math/equation discovery performance campaign
python3 main.py --math-final-discovery --benchmark-steps 600 --object-counts 5 --equation-hidden-worlds 3 --theory-memory-file tmp/theory-memory.json

# Train on generated hidden worlds, then evaluate unseen holdouts
python3 main.py --hidden-holdout-benchmark --hidden-train-worlds 6 --hidden-holdout-worlds 6 --benchmark-steps 600

# Let the agent spend a long budget on hidden generated worlds
python3 main.py --explore-hidden-worlds --exploration-budget 100 --benchmark-steps 800 --memory-file hidden_law_memory.json

# Quiet mode (no real-time discovery prints)
python3 main.py --quiet
```

Available world types:

- `standard`
- `zero_gravity`
- `central_force`
- `repulsion`
- `sideways_wind`
- `vortex`
- `inverse_square_repulsion`
- `localized_gravity`
- `time_varying`

## What This Proves

When the agent independently discovers:
- **Counting** — it means natural numbers are a property of discrete reality, not a human invention
- **Momentum** — it means momentum is a real conserved quantity in physics, not just a useful calculation
- **Arithmetic rules** — it means addition/subtraction by one is a fundamental property of discrete objects

The agent arrived at these concepts from **observation alone** — no human data, no training set, no examples. The convergence with human knowledge suggests these concepts are discovered, not invented.

## First-Principles Direction

The system now includes an early general dynamics learner. In addition to
specialized detectors, it fits small candidate equations to observed object
motion and keeps laws that reduce prediction error. This is the first step
toward replacing hand-written discovery checks with agent-derived models.

It also now invents some of its own equation terms from what the current model
fails to explain: likely radial/tangential centers from residual directions and
candidate oscillation periods from residual time series. It can also propose
localized cutoff windows when residuals appear to have a finite region where a
law applies, and local tapered-power terms when residual strength fades across
that region instead of switching on/off abruptly.

The learner can now compose several discovered terms into one compact dynamics
law, for example combining a radial field with a periodic field when neither
single explanation is enough.

Repeated discoveries can now be consolidated into self-authored equation
templates. These are generalized hypotheses such as a residual vector field
with a dominant separation exponent or a perpendicular/tapered local field,
with observed variants marked as approximations until holdout probes confirm
or narrow the form.

It also scores law elegance: extra components, repeated law families, and
seeded grid terms carry a small cost, so the agent prefers the simplest
explanation that preserves predictive accuracy.

The benchmark path now carries cross-experiment law memory. Each run is stored
as an episode, repeated law families are consolidated into semantic schemas,
and new runs report which prior schemas transferred or failed to appear. This
keeps the agent's long-term memory auditable while moving toward reusable
principles such as radial fields, tangential fields, temporal forces, and
composed dynamics.

The memory design follows four research-inspired constraints:

1. Keep fast episodic memory separate from slower semantic abstraction.
2. Use replay/transfer checks so old discoveries can be compared against new worlds.
3. Prefer compressed explanations over overfit collections of terms.
4. Store enough context to notice when a law is local to some worlds rather than universal.

Memory is now active, not just reflective. Learned schemas can guide future law
search by proposing remembered centers and periods, and spatial/temporal priors
can request counterexample probes when no stronger hypothesis is already being
tested. Uniform acceleration remains a search prior only, because active probes
can interfere with absence-of-gravity evidence.

The system now also has a raw math-discovery substrate. It watches object-level
transitions directly, invents internal patterns for persistence, stable
channels, collection extent changes, repeatable transforms, opposed transforms,
adjacent transform sequences, returning sequences, same-net sequence sets,
swapped-order sequence matches, and channel rankings, then compares those
internal discoveries to human math only after the fact. This keeps the
experiment honest: the agent-facing layer does not store labels such as
addition, subtraction, successor, predecessor, composition, identity,
equivalence, commutativity, or equality as taught concepts.

Multi-agent communication is now grounded in those internal math structures
when they exist. Agents exchange symbols for handles such as `math:C_001`
rather than starting from human-ish labels like `many_objects` or
`object_moving`; the older feature labels remain only as a cold-start fallback
before emergent math concepts have been installed. Receivers are not handed the
sender's intended meaning; they infer plausible symbol mappings from the shared
context candidates. Those inferred mappings are then reinforced only when the
receiver's inferred meaning matches a math structure that was actually active
in the observed transition, with extra credit for intervention/transform
patterns that make the signal useful for action.

The CLI can compare cold and warm agents:

```bash
python3 main.py --transfer-benchmark --seeds 3 --benchmark-steps 500
```

It can directly benchmark emergent math coverage. This runs repeated worlds,
collects post-hoc human-math comparison concepts, checks for missing target
concepts, and flags any taught-label leakage in the agent-facing emergent math
descriptions:

```bash
python3 main.py --math-benchmark --seeds 3 --benchmark-steps 160 --world-types standard
```

## GPU feasibility migration runbook

Install optional GPU dependencies before running tensor-heavy checks:

```bash
cd first_principles_ai
python3 -m pip install -r requirements-gpu.txt
```

Run the mini feasibility benchmark:

```bash
python3 main.py --gpu-feasibility-benchmark --gpu-sample-count 50000 --gpu-repeats 4 --physics-force-backend cuda --gpu-output-file tmp/gpu-feasibility.json
```

Report fields to check in the printed/JSON output:

- `torch_status`: `available` means `torch` imported; otherwise includes the import failure reason.
- `torch_cpu_seconds` and `cuda_seconds`: timing fields for torch CPU and CUDA kernels (if measured).
- `available_force_backends`: availability snapshot for numpy/torch/cuda on this machine.
- `physics_force_backend_request`: backend requested by CLI (`auto`/`numpy`/`torch`/`cuda`).
- `physics_force_backend`: actual backend selected after resolution.
- `physics_force_backend_fallback_reason`: why a requested backend could not be honored.
- `physics_force_max_abs_error`: max absolute delta difference against the python baseline.
- `physics_force_parity_passed`: must be true (default threshold `<= 1e-8`) before relying on GPU/torch parity.
- `physics_force_speedup_vs_python`: speedup ratio for force kernel.
- `recommendation`: one of:
  - `fix_tensor_physics_parity_before_gpu_runs`
  - `use_cuda_force_backend_for_large_force_batches`
  - `use_tensor_force_backend_for_force_heavy_shards`
  - `prefer_cpu_sharding_until_gpu_kernel_is_larger`
  - `prefer_cpu_sharding_until_tensor_backend_exists`

Suggested interpretation:

- If `physics_force_parity_passed` is false, fix parity before changing production backend.
- If parity passes and speedup is strong with request resolved to `cuda`, use `--physics-force-backend cuda` for heavy force workloads.
- If parity passes but fallback is `torch`/`numpy`, treat this as CPU tensor acceleration or fallback mode, not full GPU compute.
- Always confirm the artifact JSON includes `recommendation`, `fallback_reason`, and timing fields before rerunning longer campaigns.

The equation workbench is the starter-kit layer for manual research loops. It
gives the agent primitive variables, simple operators, held-out scoring, and
compact equation installation without giving domain labels such as gravity,
Newton, or vortex. The campaign command prints review packs with top equations,
categorized dynamics equations, interesting misses, equation-driven probe
counts, and leak checks so we can inspect what clicked and then amend the
starter kit deliberately:

The workbench also scores baseline-adjusted residual equations. It first
removes repeated scalar drift from velocity changes, then checks whether the
remaining change aligns with direction vectors, perpendicular-vector templates,
local high-change centres, distance-scaled strength templates, or repeating
step templates. Generated cutoff-window operators can then test whether the
residual has an inside/outside domain rather than a smooth global falloff; a
cutoff law must beat both global direction and smooth distance-scaled
references before it is accepted. Generated tapered-power operators go one
step further by testing whether residual strength follows a distance exponent
while tapering toward a local boundary; they must beat global smooth falloff
and hard-cutoff references. Weak residuals stay visible only in category
review, while
stronger residual equations can become the headline interesting equation and
request extra probes. Spatial residuals spawn around their inferred centre;
distance-strength residuals compare near and far samples; repeating residuals
ask the agent to wait and compare the next cycle; cutoff residuals probe near
the inferred boundary.

On top of the fitted equations, the autonomous discovery loop now builds a
small theory ledger. Each strong equation is converted into an explanation,
the residual concept it had to invent, generated operator proposals, proof-like
checks, known failure notes, and a next experiment. When two residual
explanations are close, the loop chooses a probe that should make one theory
stronger and the other weaker. Probe plans now carry an explicit disagreement
signature: the falsification mode, concrete probe points, each rival theory's
prediction, and what observation would count against it. Campaigns also
consolidate theory families across worlds and remember those disagreement
signatures as research leads, so repeated discoveries become reusable ideas
with proof gaps and rival-model probes instead of isolated fits. This is the
bridge from "the template fits" toward "this residual suggests a theory; what
test would falsify it?"
The ledger also writes self-authored equation templates from repeated evidence:
it picks dominant parameters, records rough variants as approximations, and
attaches proof obligations plus falsification tests before treating the template
as more than a working hypothesis.

The notebook now also carries a broad math-domain curriculum. It names the
pressure sources the system should experience to rediscover core mathematics:
quantity/arithmetic, algebra, geometry, calculus/change, probability,
logic/proof, discrete structures, symmetry, optimization, dynamics,
information/computation, and higher-dimensional worlds. Each domain lists the
observations to create, primitive concepts to rediscover, equation families to
try, proof pressure, and expected discoveries.
The curriculum includes explicit transfer bridges too, such as quantity to
algebra, algebra to geometry, geometry to calculus, calculus to dynamics,
probability to information, and higher dimensions back into dynamics. Bridge
probes carry a transfer question and a falsifier so the system can test whether
a rule is truly reusable across domains instead of just successful in one
world.

Cumulative theory memory now evaluates those families instead of only ranking
them. Each repeated family is marked as provisional, local, reusable,
established, domain-limited, or needing a counterexample, with evidence counts
and the next proof obligation. That keeps "generalizes" separate from "fit
twice in one context" and gives the next experiment a reason beyond collecting
more data.
It also emits a ranked next-experiment queue, such as transfer tests for local
families, disagreement-counterexample probes for weak-proof families, and
hidden-holdout challenges for established families. The queue is translated
into concrete suggested campaign cases with a world type, seed, object count,
step budget, expected result, and falsification condition; these are printed
for review instead of automatically starting the final watched run.
Recorded rival-model disagreements can also become concrete
model-disagreement probes with their original probe points, rival predictions,
and falsification clauses attached.
Equation campaigns can optionally spend a small follow-up budget on those
planned probes. Model-disagreement follow-ups replay the concrete probe points
from the signature as forced spawn interventions, then feed the observed result
back into cumulative memory as a confirmed transfer, counterexample, weak
transfer, holdout survival, or replication failure; model-disagreement probes
can also mark the target confirmed, a rival confirmed, or the disagreement
still open. Follow-up budgets are adaptive: after each observed outcome, the
next probe is replanned from the revised theory memory instead of using a stale
batch of old suggestions. Repeated still-open or rival-confirmed disagreements
are downweighted so the loop can refine the question or switch to another proof
obligation rather than cycling on the same stale probe. Still-open
disagreements also refine their probe geometry, such as widening exponent-race
samples into near/mid/far checks or pushing boundary probes into inside,
half-strength, and outside-tail regions. The default remains review-only, so
the final watched run is still a separate user-started command.
The cumulative theory ledger can also be persisted with
`--theory-memory-file`, giving the system a durable cross-run notebook of
families, proof certificates, planned outcomes, disagreement history, and
refined follow-up probes. Use `--no-save-theory-memory` to inspect a loaded
notebook without writing new results back. The same notebook is available to
equation campaigns, foundation prep, and the eventual watched final-discovery
command, so the final session can start from accumulated theory memory rather
than a blank ledger.
It also emits a non-final discovery-readiness audit: gates for residual-to-theory
formation, proof-like evaluation, model disagreement planning, representation
agenda, executable operator priors, operator feedback, anomaly repair, discovery
claims, claim-driven planning, first-principles adaptive dimensions, a broad
algebraic/equation foundation baseline, self-authored equation synthesis, and
the broad domain curriculum plus cross-domain transfer loop. This audit is
printed during prep/review commands so the final watched run can stay manual.
The audit now carries an evidence dossier too: compact summaries of the
strongest invented-operator chains, proof-like claims, concrete planned tests,
open disagreement probes, self-authored equations, domain-transfer probes, and
proof certificates behind the readiness score.
It can also be run by itself with `--discovery-readiness`; that command only
loads the theory notebook, prints the gate audit, evidence dossier, and
recommended non-final next actions, and never starts the watched final campaign.
When a planned case is later run, memory can record the outcome as transfer
confirmed, transfer absent, counterexample found, holdout survived, or
replication failed. Counterexamples are fed back into the family ledger, so a
formerly broad theory can be revised into a domain-limited one instead of being
left as an overgeneralized success. Domain-limited families carry an explicit
domain hypothesis: contexts where the family still applies, contexts where it
failed, and the next test for the narrowed claim. Campaign summaries print
those domain revisions alongside proof gaps and next experiments.
Families also emit proof certificates: a compact claim, the evidence for
accepting it so far, the reasons it is not universal yet, what observation
would break it, and the next proof obligation. This gives the autonomous loop a
machine-readable argument it can inspect when choosing follow-up experiments,
instead of treating proof status as just another score.
The same memory now produces a representation agenda: proposed new
measurements, operators, or domain predicates that should be added to the
agent's mathematical language. For example, exponent disagreements can produce
a log-ratio residual-strength variable, boundary disagreements can produce a
signed boundary-margin variable, taper disagreements can promote a continuous
taper operator, and counterexamples can promote a learned domain predicate.
Those proposals are now grounded in a small first-principles basis instead of a
fixed 2D coordinate assumption: identity/equality, composition, inverse
operations, order/metric comparison, symmetry transforms, recursion/iteration,
conservation/balance, domain partitioning, and dimension lifting. Residual
failures can therefore propose adaptive dimensions such as a scale-free
strength exponent, signed boundary margin, local taper coordinate, signed
alignment projection pair, domain indicator, or context-transfer axis. A
generated operator records which primitive moves and lifted dimensions it used,
so future reviews can ask whether the system needed a genuinely new coordinate
or merely refit an old 2D feature.
The memory also carries a much wider algebraic foundation baseline. It includes
expression families such as affine, polynomial, rational, power-law,
logarithmic, exponential, sinusoidal, piecewise, recurrence,
finite-difference/integral, vector, matrix, set, graph, probability, extremum,
and symmetry-invariant forms. It also records algebraic structures such as
semigroups, monoids, groups, rings, fields, vector spaces, ordered and metric
spaces, boolean/domain lattices, neighborhoods, graphs, and probability spaces.
These are not answer templates. They are a search grammar: the agent must still
earn each equation through held-out residual scoring, domain checks, proof-like
obligations, and named falsifiers. Generated operator priors now carry the
algebraic families, structures, proof obligations, and complexity controls that
justify trying them.
Executable parts of that agenda are converted into generated operator priors
for later workbench runs, where they still have to win held-out residual
scoring before becoming installed equations. The notebook records whether each
memory-generated operator was confirmed, weak, or unmatched, and feeds that
evidence back into future prior usefulness instead of retrying every invented
term equally. It also derives a small domain hypothesis for each invented
operator, so a prior that failed in one context can be avoided there while
remaining available for unseen worlds or contexts where it was confirmed. If a
held-out fit finds better parameters for the same invented operator, such as a
nearby separation exponent, those refined parameters are carried forward into
the next generated prior. Refined priors also create validation experiments
that look for an unseen context where the amended operator should either earn
fresh support or be marked as too narrow. When a prior is confirmed in one
context but becomes weak or unmatched in another, memory now records that as an
operator-prior anomaly and schedules a repair probe in the failure context. That
repair probe asks what missing domain predicate, boundary condition, or extra
factor would explain why the invented operator broke. These anomalies also feed
back into the representation agenda as candidate domain predicates, so the next
round can invent a measurement that separates the success context from the
failure context before retrying the operator. Follow-up repair and validation
runs are then judged from their actual operator-prior feedback: confirmed, weak,
or failed, with any newly refined parameters attached to the planned outcome.
Once enough evidence exists, memory emits an operator-prior discovery claim: a
proof-like summary of the invented expression, where it worked, where it failed,
which repair or validation probes confirmed it, and what obligation comes next.
A confirmed repair also suppresses repeating the same repair probe. These claims
can choose their own next experiments: repaired or merely supported invented
operators ask for unseen-context validation, domain-limited claims ask for a
domain-predicate validation in a success or failure context, and validated
operators ask for a hidden holdout that could break or narrow the claim.
Domain-predicate validation treats an unmatched prior in an explicitly excluded
context as evidence for the narrowed domain, while a confirmed prior in that
excluded context falsifies the domain predicate. Campaign summaries also print
operator-prior discovery chains: invention, feedback, anomaly, repair, claim,
and next experiment in one reviewable timeline.

Generated operators feed back into later discovery cycles. For example, once
the loop proposes an inverse-separation operator, phase-basis operator, or
localized cutoff-window/tapered-power operator, the workbench stores it in an
operator bank and scores new equations built from it on later held-out
observations. This lets the agent refine candidates such as
`unit_generated_center_vector / separation^2.25` or
`inside(separation <= r) * taper(separation, r) * unit_generated_center_vector / separation^p`
even when that exact term was not in the original starter grid.

```bash
python3 main.py --equation-campaign --seeds 1 --benchmark-steps 220 --world-types standard,sideways_wind,vortex --equation-hidden-worlds 2
```

During equation campaigns only, strong direction/perpendicular equations can
request low-motion probe objects at new locations. Normal physics and hidden
holdout benchmarks keep that behavior disabled so their scores remain
comparable.

The math foundation prep layer is the final gate before a user-watched
discovery run. It checks that the agent has number-like extent structure,
equation templates, path composition/inverse checks, proof/check traces,
geometry bases, self-directed math probes, and the cumulative discovery-loop
readiness gates. The final run performs the
watched math/equation discovery campaign and reports readiness, equation
counts, installed equations, leak checks, probe suggestions, and the most
interesting equation from each context:

```bash
python3 main.py --math-final-discovery --benchmark-steps 600 --object-counts 5 --equation-hidden-worlds 3 --theory-memory-file tmp/theory-memory.json
```

It can also run an autonomous exploration campaign. The planner scores
candidate worlds, object counts, and seeds by coverage gaps, transfer-test
opportunities, seed diversity, and interaction richness, then spends its budget
on the next most useful experiment:

```bash
python3 main.py --explore-worlds --exploration-budget 24 --benchmark-steps 500 --seeds 12
```

The next proof gate is hidden procedural worlds. These worlds are generated
from mixtures of uniform, radial, repulsive, tangential, temporal, and localized
components. The benchmark keeps the component manifest for scoring, but raw
observations expose only `hidden_procedural`, not the hidden recipe. The blind
score asks whether learned law families recover reusable structure on unseen
holdouts and whether warm memory helps without increasing leaks.

It can also persist cross-run law memory:

```bash
python3 main.py --benchmark-worlds --memory-file law_memory.json
python3 main.py --world-type vortex --memory-file law_memory.json
```

## Limitations & Future Directions

This is a prototype. To extend:

1. **Collision-specific conservation** — test momentum conservation only during collision events (where it holds), not globally (where gravity breaks it)
2. **Spatial reasoning** — let the agent discover geometric relationships (triangle inequality, angle sums)
3. **Multi-agent communication** — let agents invent their own language and compare its structure to human language
4. **Program synthesis** — let the agent write code to solve problems in its world
5. **3D physics** — add a dimension and see if the agent rediscovers 3D-specific concepts
6. **Symbolic notation** — let the agent invent its own mathematical notation and compare to human notation
