# Controller Methodology & Mathematical Formulations

## 1. Physical Plant Formulation: 2-DOF EV Quarter-Car Model

The vehicle suspension system is represented as a Two-Degree-of-Freedom (2-DOF) quarter-car heave model parameterized for modern electric vehicles (with elevated chassis sprung mass $m_s$ representing the structural battery pack).

```
                      ┌──────────────────────┐
                      │    Sprung Mass ms    │  (Chassis / Battery)
                      └──────────┬───────────┘
                                 │
                 ┌───────────────┼───────────────┐
                 │               │               │
               ┌─┴─┐           ┌─┴─┐           ┌─┴─┐
               │ks │           │cs │           │ u │ (Active Actuator)
               └─┬─┘           └─┬─┘           └─┬─┘
                 │               │               │
                 └───────────────┼───────────────┘
                                 │
                      ┌──────────┴───────────┐
                      │   Unsprung Mass mu   │  (Wheel / Hub / Brake)
                      └──────────┬───────────┘
                                 │
                               ┌─┴─┐
                               │kt │ (Tyre Radial Stiffness)
                               └─┬─┘
                                 │
                            Road z_r(t)
```

### 1.1 Equations of Motion
The continuous-time coupled differential equations governing the vertical motion are:

$$m_s \ddot{z}_s(t) = -k_s (z_s(t) - z_u(t)) - c_s (\dot{z}_s(t) - \dot{z}_u(t)) + u(t)$$

$$m_u \ddot{z}_u(t) = k_s (z_s(t) - z_u(t)) + c_s (\dot{z}_s(t) - \dot{z}_u(t)) - k_t (z_u(t) - z_r(t)) - u(t)$$

Where:
- $z_s, \dot{z}_s, \ddot{z}_s$: Sprung mass vertical displacement [m], velocity [m/s], and acceleration [m/s²].
- $z_u, \dot{z}_u, \ddot{z}_u$: Unsprung mass vertical displacement [m], velocity [m/s], and acceleration [m/s²].
- $z_r$: Road surface vertical displacement [m].
- $u$: Actuator control force [N] (+u exerts tension pushing $m_s$ up and $m_u$ down).

### 1.2 Continuous State-Space Representation
Defining state vector $x = [z_s, \dot{z}_s, z_u, \dot{z}_u]^T \in \mathbb{R}^4$:

$$\dot{x}(t) = A x(t) + B u(t) + E z_r(t)$$

$$A = \begin{bmatrix}
0 & 1 & 0 & 0 \\
-\frac{k_s}{m_s} & -\frac{c_s}{m_s} & \frac{k_s}{m_s} & \frac{c_s}{m_s} \\
0 & 0 & 0 & 1 \\
\frac{k_s}{m_u} & \frac{c_s}{m_u} & -\frac{k_s + k_t}{m_u} & -\frac{c_s}{m_u}
\end{bmatrix}, \quad
B = \begin{bmatrix} 0 \\ \frac{1}{m_s} \\ 0 \\ -\frac{1}{m_u} \end{bmatrix}, \quad
E = \begin{bmatrix} 0 \\ 0 \\ 0 \\ \frac{k_t}{m_u} \end{bmatrix}$$

### 1.3 Exact Zero-Order Hold (ZOH) Discretization
To formulate discrete-time controllers (MPC), the plant is discretized over sampling time $T_s = \Delta t = 0.002\text{ s}$ using the exact block matrix exponential:

$$M = \begin{bmatrix} A & B & E \\ 0_{1\times 4} & 0 & 0 \\ 0_{1\times 4} & 0 & 0 \end{bmatrix} \in \mathbb{R}^{6 \times 6}$$

$$\exp(M T_s) = \begin{bmatrix} A_d & B_d & E_d \\ 0_{1\times 4} & 1 & 0 \\ 0_{1\times 4} & 0 & 1 \end{bmatrix}$$

$$x_{k+1} = A_d x_k + B_d u_k + E_d z_{r,k}$$

---

## 2. Actuator Physical Constraints & Energy Accounting

In real automotive applications, suspension actuators cannot deliver infinite or instantaneous forces. The actuator model enforces:

1. **Amplitude Force Saturation:**
   $$u_{\min} \le u(t) \le u_{\max} \quad (u_{\min} = -1800\text{ N}, \ u_{\max} = +1800\text{ N})$$

2. **Slew-Rate Rate Limiting:**
   $$\left|\frac{du}{dt}\right| \le \dot{u}_{\max} = 18000\text{ N/s} \implies |u_k - u_{k-1}| \le \dot{u}_{\max} \cdot \Delta t = 36\text{ N}$$

3. **Electrical Energy & Regeneration Model:**
   Mechanical power executed on suspension: $P_{mech} = u_{applied} \cdot (\dot{z}_s - \dot{z}_u)$ [W].
   - **Motoring Mode ($P_{mech} \ge 0$):** $P_{elec} = \frac{P_{mech}}{\eta_m} + P_{standby}$ with motoring efficiency $\eta_m = 0.85$ and standby power $P_{standby} = 25\text{ W}$.
   - **Regenerative Mode ($P_{mech} < 0$):** $P_{elec} = P_{mech} \cdot \eta_r + P_{standby}$ with regenerative damping efficiency $\eta_r = 0.50$.
   - **Total Electrical Energy:** $E_{elec} = \int_0^T P_{elec}(t) dt$ [Joules].

---

## 3. Baseline Controllers

### 3.1 Baseline 1: Passive Suspension
The reference baseline operates with the active actuator unpowered ($u \equiv 0$). The performance reflects the classical fixed passive trade-off: damping high body bounce at low frequencies conflicts with isolating wheel-hop resonance vibrations at high frequencies.

### 3.2 Baseline 2: Continuous Linear Quadratic Regulator (LQR)
To penalize passenger discomfort, the quadratic cost functional penalizes sprung mass acceleration $\ddot{z}_s$:

$$J_{LQR} = \int_0^\infty \left[ q_1 \ddot{z}_s(t)^2 + q_2 (z_s(t) - z_u(t))^2 + q_3 (z_u(t) - z_r(t))^2 + r_0 u(t)^2 \right] dt$$

#### Rigorous State-Input Cross-Coupling Derivation:
Because $\ddot{z}_s = C_a x + D_a u$ where $C_a = [-\frac{k_s}{m_s}, -\frac{c_s}{m_s}, \frac{k_s}{m_s}, \frac{c_s}{m_s}]$ and $D_a = \frac{1}{m_s}$, squaring acceleration yields:

$$\ddot{z}_s^2 = x^T (C_a^T C_a) x + 2 x^T (C_a^T D_a) u + u^T (D_a^2) u$$

This produces a non-zero cross-coupling matrix $N$:

$$J_{LQR} = \int_0^\infty \left[ x^T Q x + 2 x^T N u + u^T R u \right] dt$$

$$Q = q_1 C_a^T C_a + q_2 C_{sd}^T C_{sd} + q_3 C_{td}^T C_{td}$$

$$N = q_1 C_a^T D_a \in \mathbb{R}^{4 \times 1}$$

$$R = r_0 + q_1 D_a^2 \in \mathbb{R}_{>0}$$

#### Modified Continuous Algebraic Riccati Equation (CARE):
Eliminating $N$ via transformation:

$$A^* = A - B R^{-1} N^T, \quad Q^* = Q - N R^{-1} N^T \ge 0$$

$$A^{*T} P + P A^* - P B R^{-1} B^T P + Q^* = 0$$

The optimal state feedback gain is:

$$K = R^{-1} (B^T P + N^T) \in \mathbb{R}^{1 \times 4}, \quad u(t) = -K x(t)$$

Closed-loop stability is verified by ensuring all eigenvalues of $A_{cl} = A - B K$ satisfy $\text{Re}(\lambda_i) < 0$.

---

## 4. Proposed Primary Controller: Constrained Model Predictive Control (MPC)

Model Predictive Control solves a receding-horizon constrained optimization problem at each sampling interval $k$. Unlike LQR, MPC:
1. Anticipates future road elevations over a prediction horizon $N_p$.
2. Directly enforces physical actuator force saturation $u \in [u_{\min}, u_{\max}]$ and slew-rate limits inside the optimization problem.

### 4.1 Cost Function Formulation
Over a prediction horizon $N_p$ steps:

$$J_{MPC} = \sum_{j=0}^{N_p - 1} \left[ q_1 \ddot{z}_{s, k+j}^2 + q_2 \delta_{s, k+j}^2 + q_3 \delta_{t, k+j}^2 + r_0 u_{k+j}^2 + r_{\Delta} (u_{k+j} - u_{k+j-1})^2 \right] + x_{k+N_p}^T P_f x_{k+N_p}$$

Where:
- $\delta_s = z_s - z_u = C_{sd} x$: Suspension travel.
- $\delta_t = z_u - z_r = C_{td} x - z_r$: Tyre dynamic deflection.
- $P_f$: Terminal cost matrix obtained from the discrete Algebraic Riccati Equation (DARE).

### 4.2 Condensed Prediction Formulation
Stacking future control inputs $U = [u_0, u_1, \dots, u_{N_p-1}]^T \in \mathbb{R}^{N_p}$ and future states $X = [x_1^T, \dots, x_{N_p}^T]^T \in \mathbb{R}^{4 N_p}$:

$$X = \Phi_x x_0 + \Gamma_u U + \Gamma_w W$$

Where $W = [z_{r,0}, z_{r,1}, \dots, z_{r,N_p-1}]^T$ is the road disturbance vector, and:

$$\Phi_x = \begin{bmatrix} A_d \\ A_d^2 \\ \vdots \\ A_d^{N_p} \end{bmatrix}, \quad
\Gamma_u = \begin{bmatrix}
B_d & 0 & \dots & 0 \\
A_d B_d & B_d & \dots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
A_d^{N_p-1} B_d & A_d^{N_p-2} B_d & \dots & B_d
\end{bmatrix}, \quad
\Gamma_w = \begin{bmatrix}
E_d & 0 & \dots & 0 \\
A_d E_d & E_d & \dots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
A_d^{N_p-1} E_d & A_d^{N_p-2} E_d & \dots & E_d
\end{bmatrix}$$

### 4.3 Exact Quadratic Program (QP) Derivation
Substituting $X$ into the quadratic stage cost reduces the multi-step problem to a standard convex QP:

$$\min_{U \in \mathbb{R}^{N_p}} \frac{1}{2} U^T H U + f^T U$$

#### Precomputed Constant Hessian Matrix $H$:
$$H = 2 \left[ \Gamma_u^T \bar{Q} \Gamma_u + \Gamma_u^T \bar{N} + \bar{N}^T \Gamma_u + R I_{N_p} + r_{\Delta} D^T D \right]$$

Because $\Gamma_u, \bar{Q}, \bar{N}, R,$ and $D$ depend solely on plant parameters and weights, **$H$ is constant and strictly positive definite**. It is precomputed once during initialization along with its inverse $H^{-1}$.

#### Dynamic Linear Gradient Vector $f$:
$$f = 2 \Gamma_u^T \bar{Q} (\Phi_x x_0 + \Gamma_w W) + 2 \bar{N}^T (\Phi_x x_0 + \Gamma_w W) + 2 e_0 (N^T x_0) - 2 \Gamma_u^T \bar{S}_w - 2 r_{\Delta} u_{-1} D^T d_{prev}$$

### 4.4 Constraints Formulation
1. **Actuator Force Box Bounds:**
   $$u_{\min} \le U \le u_{\max}$$

2. **Actuator Slew-Rate Linear Inequality Constraints:**
   $$-\Delta u_{\max} \mathbf{1} + d_{prev} u_{-1} \le D U \le \Delta u_{\max} \mathbf{1} + d_{prev} u_{-1}$$

### 4.5 Two-Stage Ultra-Fast Real-Time Solver Architecture
To achieve high-speed computation (0.3–1.0 ms per step at 500 Hz):
1. **Stage 1 (Unconstrained Solution Check):** Compute $U^* = -H^{-1} f$ via precomputed matrix multiplication ($\approx 5\ \mu\text{s}$). If $U^*$ satisfies force bounds and slew-rate inequalities, by convexity it is the global constrained optimum!
2. **Stage 2 (Constrained QP Optimization):** If any constraint is active, solve the constrained QP using `scipy.optimize.minimize(method='SLSQP')` with analytical gradient $H U + f$ and warm-started from clipped $U^*$.
3. **Receding Horizon Action:** Apply first command $u_{0}^*$ to actuator and shift window.

---

## 5. Proposed AI-Assisted Adaptive LQR Architecture

Complementing the MPC benchmark, the project incorporates the machine-learning-driven adaptive LQR architecture:
1. **Rolling-Window Feature Extraction:** 13 time- and frequency-domain features extracted from unsprung and sprung accelerometer signals.
2. **Intelligent Road Classification:** Random Forest classifier trained offline to identify road classes (`smooth`, `bumpy`, `rough`) with zero data leakage.
3. **PSO-Optimized Weighting:** Offline Particle Swarm Optimization tuning weighting sets $(q_1, q_2, q_3, r_0)$ for comfort, balanced, and road-holding modes.
4. **Bumpless Transfer & Dwell-Time Filter:** Ensures force continuity during controller switching without high-frequency chattering.
