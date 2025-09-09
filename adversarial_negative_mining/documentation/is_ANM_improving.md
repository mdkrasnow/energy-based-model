Here’s a tight, **checklist-first** plan to prove your ANM is (a) actually running, (b) producing *hard* negatives, and (c) improving the learned energy landscape beyond base IRED. I’m giving you concrete quantities, formulas, and tiny code probes you can drop in and then report.

# A) “Is ANM doing anything?” (hardness & margin)

**Key objects** (per contrastive/NCE form):

* **Margin**: $m = E_\theta(x,\tilde y^-;k) - E_\theta(x,\tilde y^+;k)$
* **Contrastive loss (2-way InfoNCE)**:
  $L_{\text{con}} = \log\!\big(1 + e^{-m}\big)$
  Gradients: $\frac{\partial L_{\text{con}}}{\partial E^+} = \sigma(-m),\quad \frac{\partial L_{\text{con}}}{\partial E^-} = -\sigma(-m)$
  where $\sigma(z)=1/(1+e^{-z})$.

**You should see, over training (with ANM on):**

1. **Hardness at mining time (pre-update)**: fraction $p_{\text{hard}}=\Pr[E(\tilde y^-)\!<\!E(\tilde y^+)]$ should be **high early** (ANM finds “plausible but wrong”) and **drop** across epochs as the model pushes those negatives up.

   * Log: `p_hard_batch = mean((E_neg < E_pos).float())`.

2. **Margin growth**: batch mean margin $\bar m$ should **increase** with epochs; histogram shifts right.

   * Track: `mean_m = (E_neg - E_pos).mean()`, and the 25/50/75 percentiles.

3. **Loss tightening**: $L_{\text{con}}$ should **decrease** faster vs baseline corruption-only negatives.

   * Compare two runs: ANM vs vanilla corruption, same seed.

# B) Inner-loop convergence (the miner itself)

Treat mining as constrained minimization of
$\tilde E(y)=E_{\bar\theta}(x,y,k) - \lambda\|y-y^+\|_2^2$ over $y\in\mathcal C(y^+)$ (e.g., $\ell_2$ or $\ell_\infty$ ball; discrete: masked simplex).

**What to ensure and log each mined example (or every N steps):**

1. **Monotone decrease (Armijo)**:
   Accept step size $\alpha_t$ only if
   $\tilde E(y_{t+1}) \le \tilde E(y_t) - c\,\alpha_t \|\nabla \tilde E(y_t)\|^2$ with $c\in(0,1)$.

   * Log backtrack count; if you hit `max_backtracks` often, lower `alpha` or `eps`.

2. **Projected gradient mapping norm** (stationarity proxy):
   $\|\,y - \Pi_{\mathcal C}(y - \tfrac{1}{L}\nabla \tilde E(y))\,\| \rightarrow 0$.

   * Log `gm_norm`; early-stop inner loop when `< tol` (e.g., 1e-5 \~ 1e-4).

3. **Feasibility residual**:
   $r = \|y^- - \Pi_{\mathcal C}(y^-)\|$ (should be \~0). If not, your projection is wrong.

4. **Distance-from-truth “wrongness”**:
   $d=\|y^- - y^+\|$. Enforce $d \ge \delta$ (e.g., $\delta=0.1\cdot \text{eps}$). If too small, reinit.

5. **EMA stability**: mine with $\bar\theta$ (EMA) and compute the loss with $\theta$. Confirm miner uses eval mode (no dropout) and **no grads** flow into $\bar\theta$.

# C) Variance control & the notorious renoising bug

IRED requires **shared noise** $\epsilon$ between positive and negative:

$$
\tilde y^\pm = \sqrt{1-\sigma_k^2}\,y^\pm + \sigma_k\,\epsilon,\quad \textbf{same }\epsilon.
$$

**Assertions to add:**

* **Shared-epsilon equality test:**
  `(y_tilde_pos - sqrt1m_sigma2*y_pos) - (y_tilde_neg - sqrt1m_sigma2*y_neg)` must be **exactly zero** (up to fp tol). If not, you re-sampled noise.
* **No second q\_sample:** grep/trace that there is **exactly one** noising of `y_neg` after mining. Never “re-q\_sample” the adversary.

# D) Gradient contribution balance (don’t let MSE drown ANM)

If $L_{\text{total}} = L_{\text{mse}} + \lambda_{\text{con}} L_{\text{con}}$, measure gradient norm shares:

* $g_{\text{mse}} = \|\nabla_\theta L_{\text{mse}}\|,\;\; g_{\text{con}} = \|\nabla_\theta (\lambda_{\text{con}} L_{\text{con}})\|$
* **Signal ratio**: $r = \frac{g_{\text{con}}}{g_{\text{con}} + g_{\text{mse}}}$.
  Target **0.3–0.7** early in training. If $r \to 0$, increase $\lambda_{\text{con}}$ or make negatives harder (bigger `eps`, more steps) until r is healthy.

# E) Energy landscape shaping (local geometry tests)

You want ANM to **steepen** the landscape around $y^+$ and raise energy along wrong directions.

1. **Line scan** along the mined direction:
   $\phi(t) = E_\theta\!\big(x,(1-t)\tilde y^+ + t \tilde y^-,k\big),\; t\in[0,1]$.
   Over epochs, you should see:

   * **Barrier increase**: $\max_t \phi(t) - \phi(0)$ goes **up**.
   * **Local curvature at truth**: Use finite diff near $t=0$:
     $\phi''(0) \approx \frac{\phi(\delta)-2\phi(0)+\phi(-\delta)}{\delta^2}$ (or directional Hessian via autograd). Should **increase**.

2. **Directional derivative at truth** (toward the mined neg):
   $g = \nabla_y E_\theta(x,\tilde y^+,k) \cdot \frac{(\tilde y^- - \tilde y^+)}{\| \tilde y^- - \tilde y^+\|}$.
   Desire $g \uparrow 0^+$ (less downhill toward the wrong mode).

# F) Miner effectiveness vs trivial negatives

Run an **A/B within the same training** (e.g., 50/50 batches):

* **Vanilla negatives** $y^-_{\text{rand}}=c(y)$ vs **ANM negatives** $y^-_{\text{anm}}$.
  For each batch:

  * Compare **pre-update margins** $m_{\text{rand}}$ vs $m_{\text{anm}}$ (ANM should have **smaller** margins → harder).
  * Compare **post-update margin increase** $\Delta m = m^{(t+1)} - m^{(t)}$ (ANM should yield **larger** $\Delta m$).
  * Compare **loss decrease** $\Delta L_{\text{con}}$ (ANM should decrease more).

# G) Discrete-task feasibility (e.g., Sudoku/graph constraints)

If you use masked-simplex projections:

* **Entropy of mined logits**: $H(p) = -\sum p_i \log p_i$. If $H\to 0$ too fast, negatives are too one-hot (may become trivial); if too high, they’re near-uniform (too easy). Keep a middle band early.
* **Mask compliance**: Sum of invalid mass should be \~0 after projection; log it.
* **Constraint-violation rate**: (task-defined) should **decrease** with epochs; measure on **held-out** eps/k.

# H) Bilevel time-scale & overfitting checks

* **Inner steps**: 2–5 with Armijo + projection; more steps can destabilize (miner over-optimizes against a moving target).
* **EMA for mining**: confirm you mine with $\bar\theta$ and **detach** (no grads through miner).
* **Outer LR**: ANM on → reduce base LR by 1.5–2× and verify training remains smooth (no oscillatory margins).
* **Generalization of margins**: Compute $\bar m$ on **dev** (held-out $k,\epsilon$) not seen during mining; ANM should yield **higher** dev margins vs baseline.

# I) Quick code probes to add (tiny, safe)

**1) Shared-noise invariant (per batch):**

```python
# after forming y_tilde_pos/neg
lhs = y_tilde_pos - sqrt1m_sigma2[k] * y_pos
rhs = y_tilde_neg - sqrt1m_sigma2[k] * y_neg
assert torch.allclose(lhs, rhs, atol=1e-6), "Shared-noise violated (re-noising bug)."
```

**2) No-grad through miner (bilevel hygiene):**

```python
for p in model.parameters():
    assert p.grad is None  # before backward
# run miner (should be in torch.no_grad() except for grad wrt y_neg)
loss = (L_mse + L_contrast).mean()
loss.backward()
# Verify miner didn't create param grads before backward; and after backward they exist (normal).
```

**3) Gradient share ratio r:**

```python
for p in model.parameters(): p.grad = None
(L_mse).backward(retain_graph=True)
g_mse = torch.sqrt(sum((p.grad.detach()**2).sum() for p in model.parameters() if p.grad is not None))
for p in model.parameters(): p.grad = None
(lambda_con * L_contrast).backward()
g_con = torch.sqrt(sum((p.grad.detach()**2).sum() for p in model.parameters() if p.grad is not None))
r = g_con / (g_con + g_mse + 1e-12)
log({"grad_share_contrast": r.item()})
```

**4) Miner monotonicity & backtracks:**

```python
# inside your miner loop
if E_try <= E - armijo_c * step * (grad**2).mean():
    accepted += 1
else:
    backtracks += 1
# log {"miner_accepted": accepted, "miner_backtracks": backtracks}
```

# J) Outcome metrics that *should* move if ANM helps

Pick your task’s native metric (e.g., constraint satisfaction, accuracy, objective gap). You should see with ANM:

* **Faster early improvement** (steeper learning curve in first 10–20% of steps).
* **Higher final dev margins** ($\bar m_{\text{dev}}$) and **lower violation rate**.
* **Energy profile barriers** (Section E) consistently higher.

# K) Failure signatures & immediate fixes

* **Flat margins & tiny r**: Increase $\lambda_{\text{con}}$; raise `eps`; 1–2 more inner steps; ensure EMA mining and shared ε.
* **Unstable miner (many backtracks / NaNs)**: Lower `alpha`; lower `eps`; enable grad clipping; Armijo on.
* **ANM finds *too* easy negatives (E\_neg » E\_pos)**: Reduce `eps`; add distance penalty $\lambda$; fewer inner steps.
* **Model “cheats” by uniformly raising E**: Track `E_pos` trend; it should **decrease** while `E_neg` **increases**. If both rise, regularize baseline (weight decay) and increase positive supervision (MSE weight).

---

If you instrument the above, you’ll have a clear picture: **ANM is useful iff it increases dev-set margins, steepens the barrier along mined directions, contributes a healthy gradient share, and your miner is a bona fide projected descent (monotone) with shared-noise evaluation**. This is the signature of “actually improving over base IRED,” not just “different negatives, same outcome.”
