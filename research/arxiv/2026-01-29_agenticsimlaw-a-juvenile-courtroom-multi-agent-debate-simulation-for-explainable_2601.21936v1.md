# AgenticSimLaw: A Juvenile Courtroom Multi-Agent Debate Simulation for Explainable High-Stakes Tabular Decision Making

- **arXiv**: 2601.21936v1
- **Published**: 2026-01-29T16:26:10+00:00
- **Authors**: Jon Chun, Kathrine Elkins, Yong Suk Lee
- **Categories**: cs.AI
- **Relevance score**: 9.50
- **PDF**: https://arxiv.org/pdf/2601.21936v1
- **Link**: http://arxiv.org/abs/2601.21936v1

## Summary
# AgenticSimLaw Summary for Leviathan

## Core Contribution and Method
AgenticSimLaw introduces a **structured multi-agent debate framework** for transparent, high-stakes decision-making. The system implements a courtroom-style simulation with three role-structured agents (prosecutor, defense, judge) following a **7-turn interaction protocol**. Each agent has private reasoning strategies before public utterances, and all interactions are logged for complete auditability. The framework was benchmarked on recidivism prediction using the NLSY97 dataset across ~90 model-strategy combinations.

## Key Findings
Structured multi-agent debate outperforms single-agent reasoning on **stability and generalizability**, with stronger accuracy-F1 correlation. The approach provides fine-grained control over reasoning steps and generates complete interaction transcripts for explainability. Notably, older, lower-ranked models showed higher F1 scores under the debate framework, suggesting structured interaction compensates for model limitations. However, the approach requires ~9,100 tokens per run versus 500-800 for single-shot prompting.

## Why It Matters for Leviathan

**Self-Improving Strategies**: The adversarial debate format enables iterative refinement through explicit critique and rebuttal cycles. Agents must counter opposing arguments, creating natural pressure for strategy improvement.

**Diversity**: Role-structured agents with private reasoning represent complementary perspectives. The prosecutor emphasizes risk factors while defense highlights protective factors—diverse heuristics that could prevent narrow local optima.

**Coordination**: The 7-turn protocol provides a **repeatable coordination mechanism** (opening statements → rebuttals → closing arguments → verdict). This structured turn-taking could generalize to other multi-agent coordination problems.

**Mechanisms**: The framework demonstrates how to externalize internal reasoning into observable interactions, a critical mechanism for debugging emergent behaviors in multi-agent systems.

**Evaluation**: Complete transcripts enable systematic profiling of agent behaviors, failure modes, and bias patterns—essential for evaluating whether emergent strategies remain aligned with intended goals.

## Concrete Experiments to Try
1. **Vary role structure**: Test 2-agent (advocate/arbiter) and 4-agent (adds witness/expert) configurations to identify minimal effective structure
2. **Dynamic turn protocols**: Allow agents to signal when debate should terminate early versus completing all 7 turns
3. **Cross-domain transfer**: Apply the debate framework to non-legal tasks (resource allocation, collaborative planning) to test generalizability
4. **Emergence tracking**: Instrument the framework to measure whether novel strategies emerge through adversarial interaction over multiple runs

## Risks and Limitations
The **high compute cost** limits scalability for large agent populations. The paper explicitly notes transcripts provide **plausible rather than faithful explanations**—agents may rationalize post-hoc rather than reveal true reasoning. The framework was designed for research and benchmarking, not operational deployment in sensitive domains. Finally, performance rankings under debate don't correlate with standard benchmark rankings, suggesting transfer learning challenges.

## Abstract
We introduce AgenticSimLaw, a role-structured, multi-agent debate framework that provides transparent and controllable test-time reasoning for high-stakes tabular decision-making tasks. Unlike black-box approaches, our courtroom-style orchestration explicitly defines agent roles (prosecutor, defense, judge), interaction protocols (7-turn structured debate), and private reasoning strategies, creating a fully auditable decision-making process. We benchmark this framework on young adult recidivism prediction using the NLSY97 dataset, comparing it against traditional chain-of-thought (CoT) prompting across almost 90 unique combinations of models and strategies. Our results demonstrate that structured multi-agent debate provides more stable and generalizable performance compared to single-agent reasoning, with stronger correlation between accuracy and F1-score metrics. Beyond performance improvements, AgenticSimLaw offers fine-grained control over reasoning steps, generates complete interaction transcripts for explainability, and enables systematic profiling of agent behaviors. While we instantiate this framework in the criminal justice domain to stress-test reasoning under ethical complexity, the approach generalizes to any deliberative, high-stakes decision task requiring transparency and human oversight. This work addresses key LLM-based multi-agent system challenges: organization through structured roles, observability through logged interactions, and responsibility through explicit non-deployment constraints for sensitive domains. Data, results, and code will be available on github.com under the MIT license.
