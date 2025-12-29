# Actor–Critic Contract Code Map

## A. Teacher–student critic + reward
- Formula: $Q_w(s,a)=\frac{w^T\phi(s,a)}{\sqrt{N}}$, $r(s,a)=\frac{w_R^T\phi(s,a)}{\sqrt{N}}$
- Code locations:
  - Critic feature map $\phi(s,a)$: `tdrl_unfixed_ac/features/critic_features.py:12`
  - Teacher reward vector $w_R$: `tdrl_unfixed_ac/features/teacher.py:12`
  - Reward uses teacher: `tdrl_unfixed_ac/envs/torus_gg.py:156`
  - Critic value $Q_w$: `tdrl_unfixed_ac/algos/unfixed_ac.py:120`
  - Training enforces teacher reward: `tdrl_unfixed_ac/algos/train_unfixed_ac.py:245`
- Variable map:
  - $\phi(s,a)$ -> `info["phi"]` (from env step)
  - $w$ -> `w`
  - $w_R$ -> `env.teacher_reward.w_R`
  - $r(s,a)$ -> `reward`
- Implementation notes:
  - $\phi$ is fixed random features; $w_R$ is initialized once and never updated.
  - Reward is computed from the same $\phi(s,a)$ returned in `info`.

## B. Time scales (outer vs inner loops)
- Formula: outer loop $n$ updates $w,\theta^\pi,\theta^\mu$; inner loop $t=1..T$ uses fixed policies and critic.
- Code locations:
  - Outer loop: `tdrl_unfixed_ac/algos/train_unfixed_ac.py:319`
  - Inner loops over trajectories $b$ and steps $t$: `tdrl_unfixed_ac/algos/train_unfixed_ac.py:352`
  - Updates after sampling: `tdrl_unfixed_ac/algos/train_unfixed_ac.py:418`
- Variable map:
  - $B$ -> `trajectories`
  - $T$ -> `horizon`
  - $n$ -> `iter` in logs
- Implementation notes:
  - No parameter updates occur inside the trajectory loops.

## C. Continuous action + Linear–Gaussian policies
- Formula: $\mu(a|s)=\mathcal{N}(m_\mu(s),\sigma_\mu^2I)$, $\pi(a|s)=\mathcal{N}(m_\pi(s),\sigma_\pi^2I)$, $m(s)=\theta^T\psi(s)/\sqrt{N_{act}}$.
- Code locations:
  - Mean function: `tdrl_unfixed_ac/algos/unfixed_ac.py:11`
  - Linear-Gaussian policy: `tdrl_unfixed_ac/algos/unfixed_ac.py:40`
  - Actor feature map $\psi(s)$: `tdrl_unfixed_ac/features/actor_features.py:10`
  - Policy instantiation in training: `tdrl_unfixed_ac/algos/train_unfixed_ac.py:319`
- Variable map:
  - $\theta^\pi$ -> `theta_pi`
  - $\theta^\mu$ -> `theta_mu`
  - $\psi(s)$ -> `info["psi"]`
  - $\sigma_\pi$ -> `sigma_pi`, $\sigma_\mu$ -> `sigma_mu`
- Implementation notes:
  - Default `squash_action=False` gives unsquashed Gaussian actions; optional squashing exists but is disabled for contract alignment.

## D. Score / REINFORCE factor
- Formula: $g(s,a)=\nabla_{\theta^\pi}\log\pi(a|s)=\frac{(a-m_\pi(s))\,\psi(s)}{\sigma_\pi^2\sqrt{N_{act}}}$
- Code locations:
  - Score: `tdrl_unfixed_ac/algos/unfixed_ac.py:89`
- Variable map:
  - $a$ -> `action` (unsquashed when `squash_action=False`)
  - $m_\pi(s)$ -> `mean`
  - $\psi(s)$ -> `psi`
- Implementation notes:
  - `pre_squash` is identity when `squash_action=False`, so the formula matches the paper.

## E. Off-policy correction
- Formula: $\rho=\pi(a|s)/\mu(a|s)$, report $\mathbb{E}[\rho^2]$ and tail stats.
- Code locations:
  - $\rho$ computation: `tdrl_unfixed_ac/algos/train_unfixed_ac.py:371`
  - Optional rho clip: `tdrl_unfixed_ac/algos/unfixed_ac.py:105`
  - Mean/p95 of $\rho^2$: `tdrl_unfixed_ac/algos/train_unfixed_ac.py:427`
  - Sigma condition check: `tdrl_unfixed_ac/reporting/run_report.py:561`
- Variable map:
  - $\rho$ -> `rho`
  - $\rho^2$ stats -> `mean_rho2`, `p95_rho2`
- Implementation notes:
  - Run reports alert if $\sigma_\pi^2 \ge 2\sigma_\mu^2$.

## F. TD error (bootstrapping with $\pi$)
- Formula: $\Delta=\frac{(w_R-w)^T\phi}{\sqrt{N}}+\gamma\frac{w^T\bar\phi_\pi}{\sqrt{N}}$ with $\bar\phi_\pi(s')=\mathbb{E}_{a'\sim\pi}[\phi(s',a')]$.
- Code locations:
  - $\bar\phi_\pi$ (MC): `tdrl_unfixed_ac/algos/train_unfixed_ac.py:388`
  - TD error: `tdrl_unfixed_ac/algos/train_unfixed_ac.py:391`
  - Critic value scaling: `tdrl_unfixed_ac/algos/unfixed_ac.py:120`
- Variable map:
  - $\bar\phi_\pi$ -> `bar_phi`
  - $\Delta$ -> `delta`
  - $\gamma$ -> `gamma`
- Implementation notes:
  - Trajectories are sampled with $\mu$, bootstrapping uses $\pi$.

## G. Updates with DMFT scaling
- Formula:
  - $w_{n+1}=w_n+\frac{\eta^{(w)}}{\sqrt{B}T}\sum_{b,t}\rho\,\Delta\,\phi$
  - $\theta^\pi_{n+1}=\theta^\pi_n+\frac{\eta^{(\pi)}}{\sqrt{B}T}\sum_{b,t}\rho\,g\,\Delta$
  - $\theta^\mu_{n+1}=\theta^\mu_n+\beta(\theta^\pi_n-\theta^\mu_n)$
- Code locations:
  - Batch scale: `tdrl_unfixed_ac/algos/unfixed_ac.py:127`
  - Gradient accumulators: `tdrl_unfixed_ac/algos/train_unfixed_ac.py:399`
  - Updates: `tdrl_unfixed_ac/algos/train_unfixed_ac.py:418`
- Variable map:
  - $\eta^{(w)}$ -> `alpha_w`
  - $\eta^{(\pi)}$ -> `alpha_pi`
  - $B$ -> `trajectories`
  - $T$ -> `horizon`
- Implementation notes:
  - Scaling uses $1/(\sqrt{B}T)$ via `batch_step_scale`.

## H. Semi-gradient
- Requirement: bootstrapped target does not backprop through $w$.
- Code locations:
  - Manual update (numpy, no autograd): `tdrl_unfixed_ac/algos/train_unfixed_ac.py:391`
- Implementation notes:
  - Updates are explicit numpy assignments; no autograd graph is constructed.
