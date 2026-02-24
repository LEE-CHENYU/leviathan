# SokoBench: Evaluating Long-Horizon Planning and Reasoning in Large Language Models

- **arXiv**: 2601.20856v1
- **Published**: 2026-01-28T18:56:00+00:00
- **Authors**: Sebastiano Monti, Carlo Nicolini, Gianni Pellegrini, Jacopo Staiano, Bruno Lepri
- **Categories**: cs.AI
- **Relevance score**: 5.47
- **PDF**: https://arxiv.org/pdf/2601.20856v1
- **Link**: http://arxiv.org/abs/2601.20856v1

## Summary
# SokoBench Summary for Leviathan

## Core Contribution and Method
The paper introduces **SokoBench**, a simplified Sokoban puzzle benchmark designed to isolate long-horizon planning from state persistence. Rather than using complex spatial layouts, the authors generate linear corridor puzzles where a player must push a box to a goal across varying distances (5-100 steps). This minimal environment—with single boxes, linear paths, and controlled geometry—allows precise measurement of planning degradation without confounding factors. They test three Large Reasoning Models (LRMs): DeepSeek R1, GPT-5-mini, and GPT-oss 120B, under two conditions: direct 1-shot inference and an LLM-modulo pipeline with external PDDL planning tools.

## Key Findings/Claims
1. **Fundamental planning threshold**: Performance degrades sharply beyond ~25 moves, suggesting a hard limit on forward planning capacity regardless of test-time scaling
2. **State tracking failure**: The primary bottleneck is internal counting and state representation—models "wander" through solution spaces rather than systematically exploring, often looping until token limits
3. **Tool augmentation has modest returns**: Equipping LRMs with PDDL parsers and solvers yields only marginal improvements, indicating architectural rather than methodological limitations
4. **Error patterns**: Incorrect solutions show high token variance (multiple failure modes), while correct solutions remain concise—likely reflecting GRPO training biases favoring shorter reasoning traces

## Why It Matters for Leviathan
- **Multi-agent coordination**: If individual agents cannot reliably plan beyond ~25 steps, emergent group strategies will inherit this brittleness. Long-horizon coordination (e.g., multi-turn negotiations, distributed resource allocation) may be fundamentally constrained
- **Self-improvement loops**: Agents attempting recursive self-improvement or extended reasoning chains will hit planning ceilings, limiting the depth of reflective reasoning we can extract
- **System design**: Leviathan's architecture must account for planning limits—perhaps through hierarchical decomposition, external memory, or role specialization—to bypass individual agent constraints
- **Evaluation**: SokoBench provides a clean, reproducible test for long-horizon capabilities that could supplement existing Leviathan evaluation frameworks

## Concrete Experiments to Try
1. **Hierarchical decomposition test**: Break long corridor puzzles into subgoals; evaluate whether multi-agent teams with specialized planners outperform single LRMs
2. **External memory integration**: Add a simple visited-state tracking mechanism to LRMs via tool use; measure improvement in loop prevention
3. **Role diversity experiment**: Assign different models to planning vs. validation roles; test whether error detection/correction extends effective horizon
4. **Curriculum learning**: Train on progressively longer corridors; assess whether planning capacity can be expanded through adaptation

## Risks and Limitations
- **Domain specificity**: Sokoban's deterministic, fully observable nature may not transfer to partially observed, stochastic environments Leviathan faces
- **Computational cost**: LLM-modulo pipelines require significant resources (~75 minutes per evaluation), limiting rapid iteration
- **Memorization concerns**: Anomalous accuracy peaks (e.g., GPT-5-mini at ℓ=50) suggest potential training data contamination, complicating interpretation
- **Single-box simplification**: Real planning tasks involve multiple concurrent objectives; this benchmark may underestimate multi-objective planning failures

## Abstract
Although the capabilities of large language models have been increasingly tested on complex reasoning tasks, their long-horizon planning abilities have not yet been extensively investigated. In this work, we provide a systematic assessment of the planning and long-horizon reasoning capabilities of state-of-the-art Large Reasoning Models (LRMs). We propose a novel benchmark based on Sokoban puzzles, intentionally simplified to isolate long-horizon planning from state persistence. Our findings reveal a consistent degradation in planning performance when more than 25 moves are required to reach the solution, suggesting a fundamental constraint on forward planning capacity. We show that equipping LRMs with Planning Domain Definition Language (PDDL) parsing, validation, and solving tools allows for modest improvements, suggesting inherent architectural limitations which might not be overcome by test-time scaling approaches alone.
