# Actor–Critic Contract Compliance Report

## A. Teacher–student critic + reward — PASS
Evidence (teacher reward + fixed w_R):
```python
# tdrl_unfixed_ac/features/teacher.py:12-40
class TeacherReward:
    def __init__(...):
        ...
        self.w_R = self._init_teacher_vector()

    def __call__(self, phi: np.ndarray) -> float:
        ...
        return float(np.dot(self.w_R, phi) / scale)
```
Evidence (env uses teacher reward with same phi):
```python
# tdrl_unfixed_ac/envs/torus_gg.py:156-162
obs_vec_t, psi_t, phi_t = self._compute_features(obs_t, clipped_action)
reward = 0.0
if self.use_teacher_reward:
    reward = float(self.teacher_reward(phi_t))
```
Evidence (training enforces teacher reward):
```python
# tdrl_unfixed_ac/algos/train_unfixed_ac.py:245-247
env = TorusGobletGhostEnv(config=env_cfg)
if require_teacher_reward and not env.use_teacher_reward:
    raise ValueError("Teacher reward required: set env.use_teacher_reward=True.")
```
Why PASS: reward is computed as $w_R^T\phi/\sqrt{N}$ with fixed $w_R$, and training refuses non-teacher reward.

## B. Time scales (outer vs inner loops) — PASS
Evidence (outer loop + inner loops + updates after sampling):
```python
# tdrl_unfixed_ac/algos/train_unfixed_ac.py:319-420
for n in range(start_iter, outer_iters):
    ...
    for traj_idx in range(trajectories):
        ...
        for t_idx in range(horizon):
            ...
            grad_w += rho * delta * phi
            grad_theta += rho * delta * g
    w = w + alpha_w * step_scale * grad_w
    theta_pi = theta_pi + alpha_pi * step_scale * grad_theta
    theta_mu = (1.0 - beta) * theta_mu + beta * theta_pi
```
Why PASS: parameters update only after collecting all trajectories; trajectories are sampled with fixed $(\mu_n,\pi_n)$.

## C. Continuous action + Linear–Gaussian policies — PASS
Evidence (linear NTK mean + Gaussian sampling):
```python
# tdrl_unfixed_ac/algos/unfixed_ac.py:11-66
def policy_mean(theta, psi):
    return (theta.T @ psi) / np.sqrt(theta.shape[0])

class LinearGaussianPolicy:
    def sample_action(...):
        u = rng.normal(loc=mean, scale=self.sigma, size=self.action_dim)
        if self.squash_action:
            return _squash_action(u, self.v_max)
        return u
```
Evidence (fixed actor feature map):
```python
# tdrl_unfixed_ac/features/actor_features.py:10-30
self.W = self.rng.normal(...)
self.b = self.rng.normal(...)
psi = np.tanh(self.W @ obs_vec + self.b)
```
Why PASS: policy mean is linear with $1/\sqrt{N_{act}}$ scaling, and actions are Gaussian when `squash_action=False` (default for base-check).

## D. Score / REINFORCE factor — PASS
Evidence (score matches analytic formula):
```python
# tdrl_unfixed_ac/algos/unfixed_ac.py:89-97
mean = self.mean(psi)
u = self.pre_squash(action)
diff = u - mean
scale = 1.0 / (self.sigma * self.sigma * np.sqrt(self.actor_dim))
return np.outer(psi, diff) * scale
```
Why PASS: with `squash_action=False`, $u=a$ and the score equals $(a-m_\pi)\psi/(\sigma_\pi^2\sqrt{N_{act}})$.

## E. Off-policy correction — PASS
Evidence (rho + stats):
```python
# tdrl_unfixed_ac/algos/train_unfixed_ac.py:371-439
logp_pi_exec = pi_policy.log_prob(a_exec, psi)
logp_mu_exec = mu_policy.log_prob(a_exec, psi)
rho_exec = float(np.exp(logp_pi_exec - logp_mu_exec))
rho = apply_rho_clip(rho_exec, rho_clip, disable=disable_rho_clip)
...
mean_rho2 = float(np.mean(rho2_arr))
p95_rho2 = float(np.quantile(rho2_arr, 0.95))
```
Evidence (sigma condition check in report):
```python
# tdrl_unfixed_ac/reporting/run_report.py:561-592
lhs = sigma_pi_val * sigma_pi_val
rhs = 2.0 * sigma_mu_val * sigma_mu_val
passed = lhs < rhs
```
Why PASS: rho is computed as $\pi/\mu$ per step; mean/p95 rho^2 are logged; reports flag $\sigma_\pi^2 \ge 2\sigma_\mu^2$.

## F. TD error with $\pi$ bootstrapping — PASS
Evidence (bar_phi from pi + TD error):
```python
# tdrl_unfixed_ac/algos/train_unfixed_ac.py:388-394
bar_phi = _mc_bar_phi(env, pi_policy, psi_next, rollout_rng, k_mc=k_mc)
q_sa = critic_value(w, phi)
q_next = critic_value(w, bar_phi)
delta = float(reward + gamma * q_next - q_sa)
```
Why PASS: bootstrapped feature expectation uses $\pi$; reward uses teacher $w_R$; TD error matches the paper formula.

## G. Updates with DMFT scaling — PASS
Evidence (batch scaling + updates):
```python
# tdrl_unfixed_ac/algos/unfixed_ac.py:127-131
return 1.0 / (np.sqrt(b_val) * t_val)
```
```python
# tdrl_unfixed_ac/algos/train_unfixed_ac.py:418-420
w = w + alpha_w * step_scale * grad_w
theta_pi = theta_pi + alpha_pi * step_scale * grad_theta
theta_mu = (1.0 - beta) * theta_mu + beta * theta_pi
```
Why PASS: updates use $1/(\sqrt{B}T)$ scaling and the tracking rule matches $\theta^\mu_{n+1}=\theta^\mu_n+\beta(\theta^\pi_n-\theta^\mu_n)$.

## H. Semi-gradient — PASS
Evidence (manual update, no autograd graph):
```python
# tdrl_unfixed_ac/algos/train_unfixed_ac.py:391-420
q_next = critic_value(w, bar_phi)
delta = float(reward + gamma * q_next - q_sa)
...
w = w + alpha_w * step_scale * grad_w
```
Why PASS: critic updates are numpy assignments; no autograd graph for bootstrapped target.
