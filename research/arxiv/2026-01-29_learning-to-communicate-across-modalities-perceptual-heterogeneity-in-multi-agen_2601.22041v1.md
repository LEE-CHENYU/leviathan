# Learning to Communicate Across Modalities: Perceptual Heterogeneity in Multi-Agent Systems

- **arXiv**: 2601.22041v1
- **Published**: 2026-01-29T17:45:41+00:00
- **Authors**: Naomi Pitzer, Daniela Mihai
- **Categories**: cs.MA, cs.AI, cs.CV, cs.LG
- **Relevance score**: 5.50
- **PDF**: https://arxiv.org/pdf/2601.22041v1
- **Link**: http://arxiv.org/abs/2601.22041v1

## Summary
# Summary: Learning to Communicate Across Modalities

## 1. Core Contribution and Method

This paper investigates how emergent communication protocols develop when agents have misaligned perceptual representations—a scenario largely overlooked in prior work. The authors extend the multi-step referential game framework (Evtimova et al., 2018) to compare unimodal (audio→audio) and multimodal (audio→image) systems. Agents are trained jointly using REINFORCE with baseline rewards, entropy regularization, and classification loss. The Sender maps its private input plus the Receiver's message to binary outputs; the Receiver uses GRU-based state updates to make termination decisions and generate responses.

## 2. Key Findings/Claims

Three core results emerge. First, **perceptual heterogeneity degrades efficiency**: multimodal systems require longer messages and show higher classification entropy under compression, while unimodal systems maintain accuracy with fewer bits. Second, **meaning is distributed rather than compositional**: bit perturbation experiments reveal that constant bits carry class-discriminative information, but their contributions depend on surrounding patterns—no fixed bit-to-meaning mapping exists. Third, **cross-system interoperability is learnable but not zero-shot**: agents trained in different perceptual worlds fail to communicate initially, but just 15 epochs of fine-tuning enables successful adaptation between previously incompatible systems.

## 3. Why It Matters for Leviathan

This work directly advances Leviathan's interests in **self-improving strategies** and **coordination mechanisms**. The interoperability experiments demonstrate that agents can rapidly adapt their communication protocols when paired with novel partners—a blueprint for how heterogeneous Leviathan agents might improve through cross-population transfer. The finding that perceptual misalignment doesn't prevent coordination supports designing systems with intentionally diverse perceptual configurations. The distributional encoding insight suggests Leviathan should evaluate meaning at the pattern level rather than attributing fixed semantics to individual message components.

## 4. Concrete Experiments to Try

Leviathan should test: **(1)** dynamic modality switching where agents choose optimal perceptual modes per task; **(2)** population-level experiments with more than two modalities (tri-modal audio-vision-text systems); **(3)** adversarial perturbation of inter-agent perceptual alignment to stress-test coordination; **(4)** measurement of "protocol drift" over extended multi-generational training; **(5)** fine-grained analysis of how bit-distributional patterns map to strategic rather than just class-level meaning.

## 5. Risks/Limitations

The synthetic and controlled natural datasets may not capture real-world perceptual noise. The paper focuses on binary communication with limited message length; richer protocol spaces might show different behavior. Interoperability requires fine-tuning—the zero-shot case fails—which raises concerns about sample efficiency in large-scale deployments. The architectures use relatively simple networks (no transformers, limited memory); results may not transfer to more sophisticated agent designs.

## Abstract
Emergent communication offers insight into how agents develop shared structured representations, yet most research assumes homogeneous modalities or aligned representational spaces, overlooking the perceptual heterogeneity of real-world settings. We study a heterogeneous multi-step binary communication game where agents differ in modality and lack perceptual grounding. Despite perceptual misalignment, multimodal systems converge to class-consistent messages grounded in perceptual input. Unimodal systems communicate more efficiently, using fewer bits and achieving lower classification entropy, while multimodal agents require greater information exchange and exhibit higher uncertainty. Bit perturbation experiments provide strong evidence that meaning is encoded in a distributional rather than compositional manner, as each bit's contribution depends on its surrounding pattern. Finally, interoperability analyses show that systems trained in different perceptual worlds fail to directly communicate, but limited fine-tuning enables successful cross-system communication. This work positions emergent communication as a framework for studying how agents adapt and transfer representations across heterogeneous modalities, opening new directions for both theory and experimentation.
