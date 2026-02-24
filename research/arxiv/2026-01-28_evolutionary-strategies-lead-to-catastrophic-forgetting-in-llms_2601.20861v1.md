# Evolutionary Strategies lead to Catastrophic Forgetting in LLMs

- **arXiv**: 2601.20861v1
- **Published**: 2026-01-28T18:59:34+00:00
- **Authors**: Immanuel Abdi, Akshat Gupta, Micah Mok, Alexander Lu, Nicholas Lee, Gopala Anumanchipalli
- **Categories**: cs.LG, cs.AI, cs.CL
- **Relevance score**: 6.47
- **PDF**: https://arxiv.org/pdf/2601.20861v1
- **Link**: http://arxiv.org/abs/2601.20861v1

## Summary
# Summary: Evolutionary Strategies and Catastrophic Forgetting in LLMs

## Core Contribution and Method

This paper empirically evaluates Evolutionary Strategies (ES) as a gradient-free alternative to gradient-based fine-tuning (GRPO) for LLMs. The authors compare ES against GRPO across math and reasoning benchmarks (GSM8K, MATH, OlympiadBench, Countdown) using Qwen-2.5-1.5B and Llama-3.2-1B models. They analyze forgetting curves by tracking prior-task performance (HellaSwag) while training on new tasks, and dissect update characteristics (norm and sparsity) to explain observed behaviors.

## Key Findings

ES achieves performance within 3-4 percentage points of GRPO on downstream tasks with comparable compute. However, ES causes severe catastrophic forgetting: prior-task accuracy drops ~10% after continued training, while GRPO maintains stable prior performance. The root cause is structural: ES updates have Frobenius norms three orders of magnitude larger than GRPO updates after 500 iterations, and ES updates exhibit only ~5-20% sparsity versus GRPO's ~95% sparsity. ES produces dense, high-magnitude parameter shifts across all layers, while GRPO concentrates changes in task-relevant subspaces.

## Why It Matters for Leviathan

This directly impacts Leviathan's self-improving strategy design. If agents use ES-style gradient-free updates for online adaptation, they risk eroding previously learned coordination capabilities. The sparsity finding is particularly relevant: gradient-based methods naturally constrain updates to relevant subspaces, preserving unrelated skills. For multi-agent diversity, this suggests ES may inadvertently homogenize agent capabilities over time rather than preserving distributed specialization. The KL regularization in GRPO serves as a coordination mechanism—keeping agents tethered to reference behaviors—which ES lacks. This implies any gradient-free self-improvement scheme needs explicit mechanisms to prevent capability drift.

## Concrete Experiments to Try

First, implement KL-like regularization or projection constraints for ES updates to force sparsity and limit norm growth. Second, test "capsule" approaches where ES operates only on specific parameter subsets (e.g., last-layer adapters) to bound interference. Third, evaluate whether population-based selection in ES can be modified to favor updates that preserve prior-task performance—treating stability as an explicit fitness component. Fourth, run the forgetting curve experiments within Leviathan's actual coordination tasks to quantify drift in emergent strategies.

## Risks and Limitations

ES exhibits high variance across runs due to its stochastic nature; larger population sizes (tested up to 30) reduce but don't eliminate this. The forgetting measurements use a single prior task, potentially underestimating multi-faceted capability loss. Crucially, the paper shows ES is poorly suited for any scenario requiring retained capabilities during adaptation—exactly the regime Leviathan operates in. Gradient-free approaches remain memory-efficient but sacrifice the natural stability mechanisms embedded in gradient-based optimization.

## Abstract
One of the biggest missing capabilities in current AI systems is the ability to learn continuously after deployment. Implementing such continually learning systems have several challenges, one of which is the large memory requirement of gradient-based algorithms that are used to train state-of-the-art LLMs. Evolutionary Strategies (ES) have recently re-emerged as a gradient-free alternative to traditional learning algorithms and have shown encouraging performance on specific tasks in LLMs. In this paper, we perform a comprehensive analysis of ES and specifically evaluate its forgetting curves when training for an increasing number of update steps. We first find that ES is able to reach performance numbers close to GRPO for math and reasoning tasks with a comparable compute budget. However, and most importantly for continual learning, the performance gains in ES is accompanied by significant forgetting of prior abilities, limiting its applicability for training models online. We also explore the reason behind this behavior and show that the updates made using ES are much less sparse and have orders of magnitude larger $\ell_2$ norm compared to corresponding GRPO updates, explaining the contrasting forgetting curves between the two algorithms. With this study, we aim to highlight the issue of forgetting in gradient-free algorithms like ES and hope to inspire future work to mitigate these issues.
