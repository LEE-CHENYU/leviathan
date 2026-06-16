# PathWise: Planning through World Model for Automated Heuristic Design via Self-Evolving LLMs

- **arXiv**: 2601.20539v1
- **Published**: 2026-01-28T12:34:50+00:00
- **Authors**: Oguzhan Gungordu, Siheng Xiong, Faramarz Fekri
- **Categories**: cs.AI, cs.CL
- **Relevance score**: 8.47
- **PDF**: https://arxiv.org/pdf/2601.20539v1
- **Link**: http://arxiv.org/abs/2601.20539v1

## Summary
# PathWise Summary for Leviathan

## Core Contribution & Method

PathWise introduces a multi-agent reasoning framework that transforms automated heuristic design (AHD) from trial-and-error evolution into **state-aware planning**. The system formulates heuristic generation as a sequential Markov Decision Process over an **entailment graph**—a compact, stateful memory capturing how heuristics are derived across generations. Three coordinated LLM agents drive the process:

- **Policy Agent**: Acts as high-level planner, selecting parent heuristics and generating derivation rationales
- **World Model Agent**: Executes actions by generating heuristic rollouts (code-level synthesis)
- **Critic Agents**: Provide routed reflections summarizing lessons from prior steps, enabling self-evolution without parameter updates

The entailment graph encodes derivation history, parent metadata, and performance, allowing the system to carry forward past decisions and reuse or avoid derivation paths across generations.

## Key Findings

Experiments across diverse combinatorial optimization problems (TSP, CVRP, KP, MKP, OP, BPP) show PathWise:
- **Converges faster** with fewer evaluations (achieves stronger performance at 500 evaluations vs. 1000 for baselines)
- **Generalizes across LLM backbones** (works with GPT-4o-mini and GPT-5-nano at different reasoning levels)
- **Scales to larger problem sizes** while maintaining performance
- Outperforms population-based methods (ReEvo, HSEvo) and tree-based methods (MCTS-AHD)

## Why It Matters for Leviathan

This paper directly addresses Leviathan's core themes:

**Self-Improving Strategies**: The critic agents create a feedback loop where policy and world model adapt based on performance—pure emergent self-improvement without gradient updates.

**Diversity**: Two mechanisms stand out: (1) **Prompt-level perturbation** with exploration rate decay, and (2) **State shuffling** to eliminate positional bias. These ensure exploration early, exploitation later.

**Coordination**: The multi-agent architecture demonstrates how specialized roles (planning, execution, critique) can coordinate through structured state representations rather than unstructured conversation.

**Evaluation**: The entailment graph provides interpretable traces of search trajectories—valuable for understanding emergent behaviors.

## Experiments to Try

1. **Replace the LLM backbone** with Claude or open-source models to test generalization
2. **Apply the entailment graph framework** to non-optimization tasks (code generation, theorem proving)
3. **Vary the diversity mechanisms** to understand when exploration vs. exploitation dominates
4. **Inject adversarial examples** into the graph to test robustness of critic feedback
5. **Scale Imax and Np** to test computational scaling laws

## Risks & Limitations

- **Black-box decisions**: LLM reasoning traces are opaque; hard to predict or audit emergent strategies
- **Computational overhead**: Multi-agent orchestration requires significant LLM calls per evaluation
- **Domain dependence**: Results validated on COPs—unclear transfer to open-ended tasks
- **Evaluation bottleneck**: Heuristic evaluation on real problems remains expensive; limits search depth

## Abstract
Large Language Models (LLMs) have enabled automated heuristic design (AHD) for combinatorial optimization problems (COPs), but existing frameworks' reliance on fixed evolutionary rules and static prompt templates often leads to myopic heuristic generation, redundant evaluations, and limited reasoning about how new heuristics should be derived. We propose a novel multi-agent reasoning framework, referred to as Planning through World Model for Automated Heuristic Design via Self-Evolving LLMs (PathWise), which formulates heuristic generation as a sequential decision process over an entailment graph serving as a compact, stateful memory of the search trajectory. This approach allows the system to carry forward past decisions and reuse or avoid derivation information across generations. A policy agent plans evolutionary actions, a world model agent generates heuristic rollouts conditioned on those actions, and critic agents provide routed reflections summarizing lessons from prior steps, shifting LLM-based AHD from trial-and-error evolution toward state-aware planning through reasoning. Experiments across diverse COPs show that PathWise converges faster to better heuristics, generalizes across different LLM backbones, and scales to larger problem sizes.
