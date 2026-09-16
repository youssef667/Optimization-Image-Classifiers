# Major Task 1 — Nonlinear System Optimization: Gradient Descent, Newton-Raphson & Line Search

This project solves a nonlinear system of three equations by reformulating it as an unconstrained
least-squares minimization problem, then solving that problem with three different numerical
optimization strategies: **fixed-step Gradient Descent**, **Newton-Raphson**, and **Gradient
Descent with an optimal line search (Brent's method)**. Symbolic differentiation (via SymPy) is
used first to derive and verify the closed-form Gradient vector and Hessian matrix that the
numerical methods rely on.

## Problem Formulation

The system consists of three nonlinear equations in `x1, x2, x3`:

```
g1(x1,x2,x3) = 3*x1 - cos(x2*x3) - 0.5
g2(x1,x2,x3) = x1^2 - 81*(x2 + 0.1)^2 + sin(x3) + 1.06
g3(x1,x2,x3) = exp(-x1*x2) + 20*x3 + (10*pi - 3)/3
```

These are combined into a single scalar objective function:

```
F(x1,x2,x3) = 0.5*g1^2 + 0.5*g2^2 + 0.5*g3^2
```

Driving `F` to its minimum (ideally `F ≈ 0`) drives `g1`, `g2`, and `g3` toward zero simultaneously,
which is exactly the solution of the original nonlinear system.

## Files

| File | Purpose |
|---|---|
| `Major_Task1_A&B.py` | **Parts A & B** — Symbolic derivation. Uses SymPy to compute the 3×1 Gradient vector `∇F` and the 3×3 Hessian matrix `H` of `F` analytically, and prints their shapes. |
| `Major_Task1_C.py` | **Part C** — Fixed-step-size **Gradient Descent**. Numerically evaluates the (hand-derived, closed-form) gradient each iteration and updates `X ← X − α·∇F(X)` with a constant learning rate. |
| `Major_Task1_D.py` | **Part D** — **Newton-Raphson Method**. Computes both the gradient and the Hessian at every iteration and updates `X ← X − H⁻¹·∇F(X)`, using the Hessian inverse in place of a fixed learning rate. |
| `Major_Task1_E.py` | **Part E** — Gradient Descent with an **optimal line search**. At each iteration, instead of using a fixed or matrix-based step, the optimal step size `α` along the current gradient direction is found via 1-D minimization (`scipy.optimize.brent`) of `φ(α) = F(X − α·∇F(X))`. |

## Methods

### Part A & B — Symbolic Gradient & Hessian
Uses `sympy.diff` to differentiate `F` with respect to `x1, x2, x3` to build the gradient, then
differentiates each gradient component again to build the full Hessian matrix. This provides the
exact analytical expressions used (and hard-coded) in Parts C–E, and serves as a correctness check
for those manually derived formulas.

### Part C — Gradient Descent (fixed step)

```
x_(k+1) = x_k − α · ∇F(x_k)
```

- Learning rate: `α = 0.0027`
- Initial point: `X0 = [0.071, -0.2, 0.06]`
- Convergence tolerance: `ε = 1e-12` on the gradient magnitude `‖∇F‖`
- Max iterations: `100`

### Part D — Newton-Raphson Method

```
x_(k+1) = x_k − H(x_k)⁻¹ · ∇F(x_k)
```

- Initial point: `X0 = [0, 0, 0]`
- Convergence tolerance: `ε = 1e-12`
- The Hessian is recomputed and re-inverted (`np.linalg.inv`) at every iteration
- Max iterations: `100`

### Part E — Gradient Descent with Optimal Line Search

```
x_(k+1) = x_k − α*_k · ∇F(x_k),   where α*_k = argmin_α F(x_k − α·∇F(x_k))
```

- Initial point: `X0 = [0, 0, 0]`
- Convergence tolerance: `ε = 1e-12`
- At each step, `α*_k` is found with `scipy.optimize.brent`, avoiding the need to hand-tune a
  learning rate
- Loop runs until `‖∇F‖ < ε` (no fixed iteration cap)

## Gradient Descent vs. Newton-Raphson vs. Line Search

| Method | Step Size | Info Used per Iteration | Cost per Iteration | Notes |
|---|---|---|---|---|
| Gradient Descent (C) | Fixed `α` | Gradient only | Low | Sensitive to the choice of `α`; can be slow or diverge |
| Newton-Raphson (D) | `H⁻¹` | Gradient + Hessian | High (matrix inversion) | Fast local convergence, but costly and sensitive to initial point |
| GD + Line Search (E) | Optimal `α*` per step | Gradient + 1-D search | Medium | No manual tuning of `α`; adapts step size automatically |

## Requirements

```bash
pip install numpy sympy scipy matplotlib
```

## Running the Scripts

```bash
python "Major_Task1_A&B.py"   # Symbolic gradient & Hessian
python "Major_Task1_C.py"     # Gradient Descent (fixed step)
python "Major_Task1_D.py"     # Newton-Raphson
python "Major_Task1_E.py"     # Gradient Descent with line search
```

Each numerical script prints the gradient magnitude and objective value at every iteration, prints
the final solution `[x1, x2, x3]`, and produces a convergence plot (gradient magnitude and
objective function value vs. iteration number) using Matplotlib.

## Notes

- The Newton-Raphson implementation explicitly computes `H⁻¹` via `np.linalg.inv`. For larger
  systems, solving the linear system `H·Δx = ∇F` directly (e.g. `np.linalg.solve`) is generally
  preferable to explicitly inverting the Hessian.
- Parts C and D use hand-derived closed-form gradient (and, for D, Hessian) expressions rather than
  calling into SymPy at runtime, for numerical performance; Part A & B exist to derive/verify those
  expressions symbolically.
- This project is intended as an educational implementation comparing first-order (gradient-based)
  and second-order (Hessian-based) numerical optimization techniques, along with the effect of a
  fixed vs. adaptively-chosen step size.

## Authors

- Youssef Malak
- Bavly Ehab
- Bassam Sobhy