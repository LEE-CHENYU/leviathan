# CAR-bench: Evaluating the Consistency and Limit-Awareness of LLM Agents under Real-World Uncertainty

- **arXiv**: 2601.22027v1
- **Published**: 2026-01-29T17:33:42+00:00
- **Authors**: Johannes Kirmayr, Lukas Stappen, Elisabeth André
- **Categories**: cs.AI
- **Relevance score**: 6.50
- **PDF**: https://arxiv.org/pdf/2601.22027v1
- **Link**: http://arxiv.org/abs/2601.22027v1

## Summary
# CAR-bench: Summary for Leviathan Project

## 1. Core Contribution and Method

CAR-bench is a benchmark for evaluating LLM agent reliability under real-world uncertainty, specifically in an in-car assistant domain. The framework features an LLM-simulated user, 58 interconnected tools, 19 domain policies, and comprehensive environmental databases (navigation, productivity, weather, etc.). The key methodological innovation is the introduction of two novel task types: **Hallucination tasks** (agents must acknowledge missing capabilities rather than fabricating) and **Disambiguation tasks** (agents must resolve uncertainty through clarification or internal information gathering before acting). Success is measured via Pass^k (consistent success across k trials) alongside Pass@k (at least one success), explicitly capturing the consistency gap between potential and reliable performance.

## 2. Key Findings/Claims

Baseline results reveal significant reliability gaps across all task types. Frontier reasoning models (GPT-5, Claude-Opus-4) achieve only ~54% average Pass^3, with Disambiguation being the hardest category—**no model exceeds 50% consistent success** due to premature actions. Hallucination tasks expose systematic fabrication tendencies: models prioritize satisfying user requests over acknowledging limitations, violating policies or inventing information. The gap between Pass@3 and Pass^3 is substantial (e.g., GPT-5 drops from 68% to 36% on Disambiguation), indicating latent competence that current agents cannot access consistently. Thinking models outperform non-thinking variants, but the advantage widens with complexity rather than eliminating failures.

## 3. Why It Matters for Leviathan

CAR-bench directly addresses Leviathan's core concerns about **reliable emergent coordination**. The benchmark operationalizes limit-awareness and uncertainty resolution—capabilities essential when multiple agents must coordinate without central oversight. The Pass^k metric provides a principled way to evaluate whether diverse strategies actually improve reliability or merely increase variance. The Hallucination task type specifically tests whether agents will "fake it" rather than admit uncertainty, a failure mode that could cascade catastrophically in multi-agent systems. Disambiguation tasks model scenarios where agents must choose between costly clarification and risky assumptions—precisely the coordination problems emergent strategies must navigate.

## 4. Concrete Experiments to Try

Run Leviathan agent ensembles through CAR-bench to test whether **diversity of approaches** improves consistency on Disambiguation tasks (do different agent variants catch each other's premature actions?). Test whether **coordination mechanisms** (e.g., agents queuing disambiguation requests) reduce policy violations in Hallucination scenarios. Evaluate whether **self-improving strategies** (iterative refinement based on failure traces) can narrow the Pass@3→Pass^3 gap. Finally, benchmark whether **multi-agent voting** on ambiguous instructions improves disambiguation success rates compared to single-agent baselines.

## 5. Risks/Limitations

CAR-bench focuses on **single-agent performance** in a constrained

## Abstract
Existing benchmarks for Large Language Model (LLM) agents focus on task completion under idealistic settings but overlook reliability in real-world, user-facing applications. In domains, such as in-car voice assistants, users often issue incomplete or ambiguous requests, creating intrinsic uncertainty that agents must manage through dialogue, tool use, and policy adherence. We introduce CAR-bench, a benchmark for evaluating consistency, uncertainty handling, and capability awareness in multi-turn, tool-using LLM agents in an in-car assistant domain. The environment features an LLM-simulated user, domain policies, and 58 interconnected tools spanning navigation, productivity, charging, and vehicle control. Beyond standard task completion, CAR-bench introduces Hallucination tasks that test agents' limit-awareness under missing tools or information, and Disambiguation tasks that require resolving uncertainty through clarification or internal information gathering. Baseline results reveal large gaps between occasional and consistent success on all task types. Even frontier reasoning LLMs achieve less than 50% consistent pass rate on Disambiguation tasks due to premature actions, and frequently violate policies or fabricate information to satisfy user requests in Hallucination tasks, underscoring the need for more reliable and self-aware LLM agents in real-world settings.
