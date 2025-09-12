# SANS Implementation Notes

## Energy Convention in This Codebase

**CRITICAL**: This model uses an energy-based formulation where:
- **LOWER energy = BETTER** (more likely to be a real/positive sample)
- **HIGHER energy = WORSE** (less likely to be real)

This is confirmed by the baseline loss computation which uses `-energy` in cross-entropy (line 757 in denoising_diffusion_pytorch_1d.py).

## What Are "Hard" Negatives?

In the context of Self-Adversarial Negative Sampling (SANS):

- **Hard negative**: A negative sample with **LOW energy** (close to real samples)
  - These are difficult for the model to distinguish from positives
  - They provide the most learning signal
  - Example: A corrupted sample that is only slightly different from a real one

- **Easy negative**: A negative sample with **HIGH energy** (far from real samples)
  - These are easy for the model to reject
  - They provide less learning signal
  - Example: Random noise that looks nothing like real data

## SANS Weight Computation

The SANS implementation computes weights to focus on hard negatives:

```python
neg_logits = -energy_negs  # Convert energies to logits
# Lower energy → Higher neg_logit → Higher softmax weight
w = softmax(neg_logits * alpha).detach()
```

This ensures that negatives with LOWER energy (harder to distinguish) get HIGHER weights in the loss.

## Key Implementation Details

1. **Weight Detachment**: Weights must be detached AFTER softmax computation to prevent gradients flowing through the sampling distribution.

2. **Temperature Schedule**: Alpha (temperature) DECREASES from base_α to 0.1 over diffusion timesteps:
   - Early timesteps (t≈0): High alpha → strong focus on hardest negatives (critical denoising)
   - Late timesteps (t≈T-1): Low alpha → more uniform sampling (noisy data)
   - This is reversed from RotatE but appropriate for diffusion where t=0 is clean data

3. **Number of Negatives (K)**: Should be ≥16 for effective adversarial sampling. Default is now K=16.

## Common Pitfalls

1. **Wrong sign**: If your model uses a score where HIGHER is better, you'd need `w = softmax(score * alpha)` instead.

2. **Detaching too early**: Detaching logits before softmax can still allow gradient flow through the original tensors.

3. **Wrong temperature direction**: Decreasing alpha over time reduces focus on hard negatives when the model needs it most.

## References

The SANS technique originates from:
- Sun et al., "RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space" (ICLR 2019)

In RotatE, SANS helps the model learn by focusing training on the negative samples that are most likely to be confused with positive samples.