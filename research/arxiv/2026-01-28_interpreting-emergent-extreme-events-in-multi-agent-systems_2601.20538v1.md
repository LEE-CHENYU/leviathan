# Interpreting Emergent Extreme Events in Multi-Agent Systems

- **arXiv**: 2601.20538v1
- **Published**: 2026-01-28T12:32:16+00:00
- **Authors**: Ling Tang, Jilin Mei, Dongrui Liu, Chen Qian, Dawei Cheng, Jing Shao, Xia Hu
- **Categories**: cs.MA, cs.AI
- **Relevance score**: 9.47
- **PDF**: https://arxiv.org/pdf/2601.20538v1
- **Link**: http://arxiv.org/abs/2601.20538v1

## Summary
# Summary: Interpreting Emergent Extreme Events in MAS

## Core Contribution and Method

This paper introduces the first framework for explaining emergent extreme events ("Black Swans") in LLM-powered multi-agent systems. The core contribution is a Shapley-value-based attribution method that answers three questions: *when* does an extreme event originate, *who* drives it, and *what* behaviors contribute to it.

The method works by:
1. Computing attribution scores for each agent action using Shapley values (via Monte Carlo sampling for tractability)
2. Aggregating scores across time, agents, and behavior types
3. Deriving five quantitative metrics: relative risk latency, agent risk concentration, risk-instability correlation, agent risk synchronization, and behavior risk concentration

## Key Findings/Claims

Five generalizable insights emerge across economic, financial, and social scenarios:

1. **Temporal duality**: Extreme events either stem from early dormant risks (long buildup) or immediate shocks (sudden onset)
2. **Agent concentration**: A small subset (~20%) of agents drives the majority of risk
3. **Risk-instability correlation**: High-risk agents tend to be highly unstable in their actions
4. **Synchronization**: Agents tend to increase or decrease risk synchronously
5. **Behavioral concentration**: A few behavior patterns account for most risk contribution

## Why It Matters for Leviathan

For the Leviathan project, this framework directly addresses critical safety and interpretability challenges:

- **Self-improving strategies**: The method can identify which agent actions lead to dangerous capability gains or goal drift, enabling targeted intervention before Black Swan events
- **Diversity**: By quantifying agent risk concentration, we can assess whether our diversity mechanisms are preventing single-point failures
- **Coordination**: The synchronization metric reveals when agents are overly aligned in risky behaviors—potentially signaling emergent collusive strategies
- **Mechanisms**: The behavioral attribution can guide mechanism design (e.g., restricting high-risk action patterns)
- **Evaluation**: Provides a rigorous framework for evaluating emergent strategy safety beyond simple performance metrics

## Concrete Experiments to Try

1. **Apply to Leviathan's emergent strategies**: Run the attribution framework on Leviathan's training trajectories to identify high-risk emergent behaviors
2. **Ablate diversity mechanisms**: Remove diversity constraints and test whether risk concentration increases as predicted
3. **Early warning system**: Use relative risk latency to detect dormant risks before they manifest
4. **Agent-level interventions**: Remove or constrain high-risk agents (as identified by attribution) and measure system stability
5. **Cross-architecture comparison**: Test whether different LLM backbones exhibit different risk profiles using these metrics

## Risks/Limitations

- **Computational cost**: Exact Shapley values require O(2^NT) evaluations; Monte Carlo approximation may miss rare but critical action combinations
- **Threshold sensitivity**: The definition of "extreme event" (

## Abstract
Large language model-powered multi-agent systems have emerged as powerful tools for simulating complex human-like systems. The interactions within these systems often lead to extreme events whose origins remain obscured by the black box of emergence. Interpreting these events is critical for system safety. This paper proposes the first framework for explaining emergent extreme events in multi-agent systems, aiming to answer three fundamental questions: When does the event originate? Who drives it? And what behaviors contribute to it? Specifically, we adapt the Shapley value to faithfully attribute the occurrence of extreme events to each action taken by agents at different time steps, i.e., assigning an attribution score to the action to measure its influence on the event. We then aggregate the attribution scores along the dimensions of time, agent, and behavior to quantify the risk contribution of each dimension. Finally, we design a set of metrics based on these contribution scores to characterize the features of extreme events. Experiments across diverse multi-agent system scenarios (economic, financial, and social) demonstrate the effectiveness of our framework and provide general insights into the emergence of extreme phenomena.
