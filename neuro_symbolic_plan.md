# Neuro-Symbolic Research Plan (No-Mock Version)

## Current Verified Status

What is implemented and measured in this repo:
- Real bridge from controlled English templates to Narsese truth-valued statements.
- Real ONA execution via subprocess (`./NAR shell < file.nal`).
- Explicit ablations:
1. exact lexical baseline
2. embedding-only bridge
3. ONA direct propagation
4. ONA revision under conflicting evidence
5. ONA multi-hop + revision
6. neural baseline (offline MLP)
- Held-out noun-pair evaluation with confidence intervals and McNemar tests.
- Multi-split sweep (seeds 0..4) showing stable separation.
- External offline WSC273 integration with local cached models and a learned LOPO bridge on an executable causal subset.
- Full-WSC273 stratified CV learned bridge (non-handcrafted on external set) with paired McNemar comparison against strongest neural anchor.

What is not yet achieved:
- External benchmark integration exists (WSC273 offline), but ONA still does not exceed strongest full-WSC neural anchor.
- No end-to-end neural training through ONA.
- No top-tier-ready evidence yet.

## Short-Term Objective (Next 4-8 weeks)

Move from strong internal evidence to external, publication-defensible evidence.

### Workstream A: External Dataset Improvement
- Upgrade from initial WSC273 integration to stronger transfer results.
- Keep exact no-mock policy: every prediction must come from executable code path.
- Preserve paired comparisons on identical examples and report failure clusters.
- Current checkpoint (April 26, 2026): RoBERTa-large MLM anchor \(0.689\); learned bridge/ONA direct \(0.670\) on full-WSC 5-fold CV.

### Workstream B: Stronger Neural Baselines
- Add offline-capable transformer baseline if local checkpoints are available.
- Keep train/dev/test splits fixed and versioned.
- Report seed variance and calibration metrics.

### Workstream C: Statistical Rigor
- Continue bootstrap CIs and McNemar tests.
- Add multiple-hypothesis correction for many comparisons.
- Add preregistered-style evaluation spec in-repo (`evaluation_protocol.md`).

## Mid-Term Objective (2-4 months)

Demonstrate bridge value under more realistic language variation.
- Expand parser front-end beyond one template while keeping explicit failure accounting.
- Add adversarial perturbation sets (word order, distractor clauses, negation).
- Track explanation fidelity (derived chain should causally support chosen answer).

## Hard Blockers for ICLR/NeurIPS-Level Claim

1. External benchmark performance is currently weak relative to stronger neural baselines.
2. Semantic grounding still partly hand-calibrated.
3. No learned feedback loop from symbolic outcomes to neural parameters.

## Acceptance Criteria for Next Paper Draft

Only claim readiness when all are true:
- At least one accepted external benchmark with full reproducibility and competitive metrics.
- No mocked tie-breakers or oracle fallbacks in any evaluation path.
- Comparable or better performance than strong neural baselines on targeted subsets.
- Transparent error analysis with released prediction files.

## Immediate Next Implementation Tasks

1. Add `evaluation_protocol.md` documenting split generation, seeds, significance tests, and model snapshot paths.
2. Add deterministic export of per-example reasoning traces for full-WSC learned ONA predictions.
3. Add calibration analysis (ECE/Brier) for bridge probabilities before and after ONA conversion.
