# JADE: Bridging the Strategic-Operational Gap in Dynamic Agentic RAG

- **arXiv**: 2601.21916v1
- **Published**: 2026-01-29T16:06:44+00:00
- **Authors**: Yiqun Chen, Erhan Zhang, Tianyi Hu, Shijie Wang, Zixuan Yang, Meizhi Zhong, Xiaochi Wei, Yan Gao, Yi Wu, Yao Hu, Jiaxin Mao
- **Categories**: cs.AI, cs.CL, cs.IR
- **Relevance score**: 7.50
- **PDF**: https://arxiv.org/pdf/2601.21916v1
- **Link**: http://arxiv.org/abs/2601.21916v1

## Summary
## JADE: Strategic-Operational Alignment in Multi-Agent RAG

### Core Contribution & Method

JADE addresses a fundamental problem in agentic RAG systems: the "strategic-operational mismatch" where sophisticated planners fail because executors are treated as frozen black boxes. The framework unifies planning and execution into a **cooperative multi-agent system** with three key innovations:

1. **Shared-backbone architecture**: A single LLM backbone serves both Planner and Executors, with roles instantiated via specialized prompts
2. **Unified experience replay**: All agent interactions (workflow decisions, document filtering, answer generation) are flattened into atomic transition tuples ⟨observation, action, reward⟩
3. **Co-adaptive optimization**: PPO training with global rewards enables the Planner to learn within executor capability boundaries while executors align with strategic intent

### Key Findings

JADE achieves state-of-the-art performance across seven benchmarks, with a 7B model outperforming GPT-4o-based decoupled systems. The dynamic workflow orchestration enables flexible efficiency-effectiveness tradeoffs. Critically, joint optimization transforms disjoint modules into a synergistic team—solving the credit assignment problem in long-horizon reasoning.

### Why It Matters for Leviathan

JADE directly addresses Leviathan's core themes: **co-adaptation** mirrors how Leviathan agents could evolve coordination strategies, **parameter sharing** provides a blueprint for unified multi-agent backbones, and the **global reward structure** offers a template for aligning emergent behaviors. The work demonstrates that collaborative synergy can outperform raw scale—highly relevant for Levi's self-improving agent ecosystems.

### Concrete Experiments to Try

1. Replace JADE's single-backbone setup with Levi's heterogeneous agent architectures to test cross-architecture coordination
2. Apply JADE's reward design (outcome-based + local penalties) to Levi's coordination mechanisms
3. Test the "workflow planning → execution" loop on Levi's complex, multi-step tasks
4. Scale the parameter-sharing approach to Levi's diverse agent populations

### Risks & Limitations

- PPO training requires careful reward tuning; sparse global rewards may cause instability
- Role definitions depend heavily on prompt engineering—brittleness under distribution shift
- The framework assumes cooperative goals; may not extend to competitive or mixed-motive scenarios
- Performance gains require non-trivial computational overhead for joint training

## Abstract
The evolution of Retrieval-Augmented Generation (RAG) has shifted from static retrieval pipelines to dynamic, agentic workflows where a central planner orchestrates multi-turn reasoning. However, existing paradigms face a critical dichotomy: they either optimize modules jointly within rigid, fixed-graph architectures, or empower dynamic planning while treating executors as frozen, black-box tools. We identify that this \textit{decoupled optimization} creates a ``strategic-operational mismatch,'' where sophisticated planning strategies fail to materialize due to unadapted local executors, often leading to negative performance gains despite increased system complexity. In this paper, we propose \textbf{JADE} (\textbf{J}oint \textbf{A}gentic \textbf{D}ynamic \textbf{E}xecution), a unified framework for the joint optimization of planning and execution within dynamic, multi-turn workflows. By modeling the system as a cooperative multi-agent team unified under a single shared backbone, JADE enables end-to-end learning driven by outcome-based rewards. This approach facilitates \textit{co-adaptation}: the planner learns to operate within the capability boundaries of the executors, while the executors evolve to align with high-level strategic intent. Empirical results demonstrate that JADE transforms disjoint modules into a synergistic system, yielding remarkable performance improvements via joint optimization and enabling a flexible balance between efficiency and effectiveness through dynamic workflow orchestration.
