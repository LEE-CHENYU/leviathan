# Post-Disaster Resource Redistribution and Cooperation Evolution Based on Two-Layer Network Evolutionary Games

- **arXiv**: 2601.22021v1
- **Published**: 2026-01-29T17:29:34+00:00
- **Authors**: Yu Chen, Genjiu Xu, Sinan Feng, Chaoqian Wang
- **Categories**: physics.soc-ph, cs.GT
- **Relevance score**: 6.50
- **PDF**: https://arxiv.org/pdf/2601.22021v1
- **Link**: http://arxiv.org/abs/2601.22021v1

## Summary
# Summary: Two-Layer Network Games for Post-Disaster Cooperation

## 1. Core Contribution and Method

This paper develops a **two-layer network evolutionary game model** that couples organizational and individual behavior during post-disaster resource redistribution. The upper layer models shelter-level interactions using a continuous-strategy public goods game, while the lower layer models victim-level cooperation through a binary evolutionary game on lattice networks. The model captures asymmetric cross-layer feedback where shelter cooperation influences victim behavior and vice versa. The authors use Monte Carlo simulations on both synthetic scale-free networks (Barabási-Albert) and a real-world network derived from Beijing shelter coordinates.

## 2. Key Findings/Claims

The study identifies critical **threshold effects** in institutional mechanisms:

- **Incentives backfire at high levels**: Moderate enhancement factors (r ≈ 1.1-1.3) and subsidies promote cooperation, but excessive values induce free-riding and reduce both shelter inputs and victim cooperation.
- **Punishment must be credible and comprehensive**: Single punishment measures are ineffective. Only the combination of baseline punishment (γ₀), execution assurance (α), and marginal impact penalties (γ₁) sustains high cooperation. Coverage matters—punishing 80%+ of shelters approaches saturation.
- **Structural targeting outperforms random enforcement**: Directed punishment expanding from hub shelters achieves higher cooperation at lower cost than random coverage, leveraging network centrality for efficient diffusion.
- **Cross-layer dynamics are asymmetric**: Shelter cooperation has a stronger, non-monotonic effect on victims (initial suppression followed by promotion), while victim feedback is weaker and more delayed.

## 3. Why It Matters for Leviathan

These findings directly address Leviathan's core challenges:

- **Mechanism design**: The threshold effects demonstrate that institutional incentives require careful calibration—more is not always better. This warns against naive intensification of coordination mechanisms.
- **Emergent strategy**: The model shows how local rules (punishment, subsidies) propagate through network structure to produce system-level cooperation, relevant for self-improving systems that must bootstrap coordination.
- **Evaluation metrics**: The cooperation-coordination trade-offs suggest Leviathan should track marginal returns on coordination efforts rather than absolute intensity.
- **Structural diversity**: Real-world network topology (Beijing case) showed amplified effects compared to synthetic scale-free networks, implying Leviathan should validate mechanisms across diverse structural contexts.

## 4. Concrete Experiments to Try

1. **Vary incentive intensity systematically** to map cooperation thresholds in Leviathan's coordination games
2. **Test targeted vs. random punishment** on Leviathan's network topology to validate structural targeting effects
3. **Implement asymmetric cross-layer coupling** where different agent classes use different game structures
4. **Run Beijing-style geographic network simulations** using Leviathan's actual deployment topology
5. **Measure marginal returns** on cooperation as a function of punishment coverage to find optimal intervention points

## 5. Risks/Limitations

- **Assumes homogeneous victim populations** within shelters; real systems have heterogeneous agents
- **Binary victim strategies** may not capture nuanced cooperation levels in complex systems
- **Parameter sensitivity**: Results depend heavily on punishment credibility and execution—unreliable enforcement could invert effects
- **Scalability**: Lattice-based victim networks may not reflect large-scale emergent patterns
- **No explicit self-modification**: The model studies exogenous mechanism changes, not systems that autonomously redesign their own coordination rules

## Abstract
In the aftermath of large-scale disasters, the scarcity of resources and the paralysis of infrastructure raise severe challenges to effective post-disaster recovery. Efficient coordination between shelters and victims plays a crucial role in building community resilience, yet the evolution of two-layer behavioral feedback between these two groups through network coupling remains insufficiently understood. Here, this study develops a two-layer network to capture the cross-layer coupling between shelters and victims. The upper layer uses a post-disaster emergency resource redistribution model within the framework of the public goods game, while the lower layer adopts a cooperative evolutionary game to describe internal victim interactions. Monte Carlo simulations on scale-free networks reveal threshold effects of incentives: moderate public goods enhancement and subsidies promote cooperation, whereas excessive incentives induce free-riding. In contrast, credible and well-executed punishment effectively suppresses defection. Targeted punishment of highly connected shelters significantly enhances cooperation under resource constraints. A comparative analysis using a network generated from the actual coordinates of Beijing shelters confirms the model's generality and practical applicability. The findings highlight the importance of calibrated incentives, enforceable sanctions, and structural targeting in fostering robust cooperation across organizational and individual levels in post-disaster environments.
