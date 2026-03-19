# MemCtrl: Using MLLMs as Active Memory Controllers on Embodied Agents

- **arXiv**: 2601.20831v1
- **Published**: 2026-01-28T18:31:17+00:00
- **Authors**: Vishnu Sashank Dorbala, Dinesh Manocha
- **Categories**: cs.AI, cs.RO
- **Relevance score**: 5.47
- **PDF**: https://arxiv.org/pdf/2601.20831v1
- **Link**: http://arxiv.org/abs/2601.20831v1

## Summary
## MemCtrl: Summary for Leviathan

**Core Contribution and Method**

MemCtrl introduces a lightweight, trainable memory head (μ) that attaches to Multimodal Large Language Models (MLLMs) to enable active, real-time memory filtering during embodied task execution. Rather than treating memory as a large offline storage space (as in RAG systems), μ learns to gate which observations to retain, update, or discard on-the-fly. The memory head is trained via two approaches: (1) offline supervised learning from a high-performing expert model (GPT-4o), and (2) online RL with sparse task rewards and dense action validity rewards. Crucially, μ is model-agnostic and detachable, requiring no finetuning of the underlying MLLM backbone.

**Key Findings/Claims**

Augmenting low-performing MLLMs (Qwen2.5-VL-7B, Gemma-3-12B) with MemCtrl yields approximately 16% average improvement on EmbodiedBench, with over 20% gains on complex and long-horizon instruction subsets. The method proves particularly effective on navigation-heavy tasks (Habitat) versus manipulation tasks (ALFRED). Qualitative analysis reveals distinct behavioral profiles: RL-trained heads exhibit exploratory behavior, while expert-trained heads show exploitative repetition—suggesting the training approach fundamentally shapes agent strategy.

**Why It Matters for Leviathan**

This work directly supports Leviathan's focus on self-improving strategies: memory heads learn which observations matter, enabling agents to improve memory efficiency and decision quality without architectural changes. The detachability mechanism offers a concrete design pattern for modular agent components that could transfer across diverse embodied setups. The emergence of distinct behavioral strategies (exploratory vs. exploitative) from different training regimes connects to coordination and diversity concerns—different agents could use different memory heads to contribute complementary capabilities. Finally, the evaluation framework (measuring task success, memory efficiency, and behavioral patterns) provides a template for assessing memory mechanisms in multi-agent contexts.

**Experiments to Try**

- Test μ-trained agents in multi-agent coordination tasks to measure communication efficiency gains
- Vary the reward structures in online RL training to shape different emergent strategies
- Transfer memory heads between agent types to assess cross-agent generalization
- Apply active memory filtering to multi-agent communication protocols, measuring bandwidth reduction
- Extend the framework to include inter-agent memory sharing, where agents learn what to communicate versus retain

**Risks and Limitations**

The supervised approach requires high-quality expert demonstrations, which may be costly or unavailable for novel domains. RL variants suffer from sparse reward signal issues, limiting training efficiency. Performance gains diminish on short-horizon tasks where memory filtering provides limited benefit. Current

## Abstract
Foundation models rely on in-context learning for personalized decision making. The limited size of this context window necessitates memory compression and retrieval systems like RAG. These systems however often treat memory as large offline storage spaces, which is unfavorable for embodied agents that are expected to operate under strict memory and compute constraints, online. In this work, we propose MemCtrl, a novel framework that uses Multimodal Large Language Models (MLLMs) for pruning memory online. MemCtrl augments MLLMs with a trainable memory head μthat acts as a gate to determine which observations or reflections to retain, update, or discard during exploration. We evaluate with training two types of μ, 1) via an offline expert, and 2) via online RL, and observe significant improvement in overall embodied task completion ability on μ-augmented MLLMs. In particular, on augmenting two low performing MLLMs with MemCtrl on multiple subsets of the EmbodiedBench benchmark, we observe that μ-augmented MLLMs show an improvement of around 16% on average, with over 20% on specific instruction subsets. Finally, we present a qualitative analysis on the memory fragments collected by μ, noting the superior performance of μaugmented MLLMs on long and complex instruction types.
