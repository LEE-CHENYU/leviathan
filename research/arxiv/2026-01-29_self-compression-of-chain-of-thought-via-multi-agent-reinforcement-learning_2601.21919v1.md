# Self-Compression of Chain-of-Thought via Multi-Agent Reinforcement Learning

- **arXiv**: 2601.21919v1
- **Published**: 2026-01-29T16:13:10+00:00
- **Authors**: Yiqun Chen, Jinyuan Feng, Wei Yang, Meizhi Zhong, Zhengliang Shi, Rui Li, Xiaochi Wei, Yan Gao, Yi Wu, Yao Hu, Zhiqiang Pu, Jiaxin Mao
- **Categories**: cs.AI, cs.CL
- **Relevance score**: 6.50
- **PDF**: https://arxiv.org/pdf/2601.21919v1
- **Link**: http://arxiv.org/abs/2601.21919v1

## Summary
# SCMA: Self-Compression via Multi-Agent RL
**Decision-Oriented Summary for Leviathan**

---

## 1. Core Contribution and Method

SCMA addresses inefficient reasoning in Large Reasoning Models (LRMs) by reframing Chain-of-Thought compression as a collaborative multi-agent problem. Rather than applying blanket length penalties (which sacrifice accuracy), SCMA deploys three specialized agents that jointly optimize for *importance-weighted conciseness*:

- **Reasoning Agent**: Generates reasoning paths
- **Segmentation Agent**: Decomposes reasoning into logical chunks
- **Scoring Agent**: Quantifies each chunk's contribution to the final answer

The key insight: penalties are applied *discriminatively*—high-importance chunks are exempted, redundant chunks are aggressively compressed. All agents share parameters from a single LLM, enabling efficient co-evolution via Multi-Agent GRPO. Critically, the Segmentation and Scoring agents are training-only; deployment uses only the Reasoning Agent, introducing zero inference overhead.

---

## 2. Key Findings/Claims

- **Compression + Accuracy**: SCMA reduces response length by 11.1–39.0% *while* improving accuracy by 4.33–10.02% across multiple benchmarks
- **Scalability**: Benefits increase with model scale—Qwen3-8B achieves 39% length reduction on MATH500 with 89.2% accuracy
- **Emergent Coordination**: Ablations show MARL joint optimization significantly outperforms frozen baseline agents, indicating genuine collaborative capability
- **Generalization**: Trained only on GSM8K, SCMA transfers effectively to unseen challenging problems (AIME, AMC23)

---

## 3. Why It Matters for Leviathan

This paper directly addresses Leviathan's core themes:

- **Self-Improving Strategies**: The framework demonstrates *internal self-regulation*—agents learn to critique and compress their own reasoning without external supervision
- **Diversity & Coordination**: Three agents with distinct roles collaborate toward a shared objective, modeling how specialized components can harmonize
- **Mechanisms**: The importance-weighted penalty provides a template for *granular reward shaping*—applicable to other multi-agent coordination problems
- **Evaluation**: Demonstrates that compressed reasoning can be *more* accurate, challenging the assumption that longer chains are necessary

---

## 4. Concrete Experiments for Leviathan

1. **Extend Agent Diversity**: Add verification or revision agents to the MARL framework; test whether additional specialization improves coordination
2. **Emergent Behavior Study**: Track whether Segmentation/Scoring agents develop consistent "importance" heuristics across tasks—can we interpret their learned representations?
3. **Cross-Task Transfer**: Apply SCMA's compression mechanism to non-mathematical domains (code generation, strategic planning) to test domain generality
4. **Coordination Dynamics**: Vary the reward-sharing structure; test if competitive or hierarchical objectives yield different compression behaviors

---

## 5. Risks/Limitations

- **Training Instability**: Multi-agent co-evolution risks misalignment—agents may collude to "game" rewards rather than genuinely improve reasoning
- **Over-Compression Risk**: Aggressive α values risk eliminating *apparently redundant but actually critical* steps (e.g., verification loops)
- **Task-Specific Tuning**: Optimal α varies across model scales and datasets; may require extensive hyperparameter search
- **Evaluation Gap**: Length reduction metrics may not capture reasoning *quality*—need deeper analysis of whether compressed chains maintain robustness on adversarial cases

## Abstract
The inference overhead induced by redundant reasoning undermines the interactive experience and severely bottlenecks the deployment of Large Reasoning Models. Existing reinforcement learning (RL)-based solutions tackle this problem by coupling a length penalty with outcome-based rewards. This simplistic reward weighting struggles to reconcile brevity with accuracy, as enforcing brevity may compromise critical reasoning logic. In this work, we address this limitation by proposing a multi-agent RL framework that selectively penalizes redundant chunks, while preserving essential reasoning logic. Our framework, Self-Compression via MARL (SCMA), instantiates redundancy detection and evaluation through two specialized agents: \textbf{a Segmentation Agent} for decomposing the reasoning process into logical chunks, and \textbf{a Scoring Agent} for quantifying the significance of each chunk. The Segmentation and Scoring agents collaboratively define an importance-weighted length penalty during training, incentivizing \textbf{a Reasoning Agent} to prioritize essential logic without introducing inference overhead during deployment. Empirical evaluations across model scales demonstrate that SCMA reduces response length by 11.1\% to 39.0\% while boosting accuracy by 4.33\% to 10.02\%. Furthermore, ablation studies and qualitative analysis validate that the synergistic optimization within the MARL framework fosters emergent behaviors, yielding more powerful LRMs compared to vanilla RL paradigms.
