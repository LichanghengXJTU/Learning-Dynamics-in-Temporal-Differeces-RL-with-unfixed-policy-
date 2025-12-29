# Step C metric map

This maps Step C CSV metrics to definitions, code locations, dependencies, and whether they are empirical or theory-aligned.

## Shared definitions
- Delta(s_t,a_t) = r_t + gamma * Q_w(s_{t+1}, a'~pi) - Q_w(s_t,a_t). Code: `tdrl_unfixed_ac/algos/train_unfixed_ac.py:330`.
- Q_w(s,a) = (w dot phi(s,a)) / sqrt(d). Code: `tdrl_unfixed_ac/algos/unfixed_ac.py:79`.
- bar_phi(s_{t+1}) = E_{a'~pi}[phi(s_{t+1},a')] estimated by MC. Code: `tdrl_unfixed_ac/algos/train_unfixed_ac.py:327`.
- rho = exp(logpi - logmu), computed on clipped actions and optionally clipped by rho_clip. Code: `tdrl_unfixed_ac/algos/train_unfixed_ac.py:315`, `tdrl_unfixed_ac/algos/unfixed_ac.py:64`.
- td_loss (training) = mean(Delta^2) across all steps (no 1/2 factor). Code: `tdrl_unfixed_ac/algos/train_unfixed_ac.py:360`.
- Q_hat(t,t') = mean_b Delta_cache[b,t] * Delta_cache[b,t'] from cached TD errors. Code: `tdrl_unfixed_ac/probes/q_kernel_probe.py:26`.

## learning_curves.csv (training loop)
| Metric | Definition | Code location | Dependencies | Clip/normalize/mask/stopgrad | Empirical vs theory |
| --- | --- | --- | --- | --- | --- |
| iter | Outer iteration index n (0-based). | `tdrl_unfixed_ac/algos/train_unfixed_ac.py:402` | n | none | empirical (counter) |
| td_loss | mean(Delta^2) over all sampled steps. | `tdrl_unfixed_ac/algos/train_unfixed_ac.py:360` | Delta, gamma, Q_w, phi, bar_phi, reward | no clip; Q_w uses 1/sqrt(d) normalization | empirical (sample average) |
| critic_teacher_error | (1/d) * ||w - w_teacher||^2. | `tdrl_unfixed_ac/algos/train_unfixed_ac.py:397` | w, teacher_w, feature_dim | normalized by feature_dim | deterministic parameter metric |
| tracking_gap | (1/actor_dim) * ||theta_pi - theta_mu||^2. | `tdrl_unfixed_ac/algos/train_unfixed_ac.py:398` | theta_pi, theta_mu, actor_dim | normalized by actor_dim | deterministic parameter metric |
| w_norm | ||w||_2. | `tdrl_unfixed_ac/algos/train_unfixed_ac.py:399` | w | none | deterministic parameter metric |
| mean_rho | mean(rho) over sampled steps. | `tdrl_unfixed_ac/algos/train_unfixed_ac.py:364` | logpi, logmu, rho_clip | action clipped; rho optionally clipped | empirical |
| mean_rho2 | mean(rho^2). | `tdrl_unfixed_ac/algos/train_unfixed_ac.py:365` | rho | rho uses clip | empirical |
| min_rho | min(rho). | `tdrl_unfixed_ac/algos/train_unfixed_ac.py:366` | rho | rho uses clip | empirical |
| max_rho | max(rho). | `tdrl_unfixed_ac/algos/train_unfixed_ac.py:367` | rho | rho uses clip | empirical |
| p95_rho | 95th percentile of rho. | `tdrl_unfixed_ac/algos/train_unfixed_ac.py:368` | rho | rho uses clip | empirical |
| p99_rho | 99th percentile of rho. | `tdrl_unfixed_ac/algos/train_unfixed_ac.py:369` | rho | rho uses clip | empirical |
| p95_rho2 | 95th percentile of rho^2. | `tdrl_unfixed_ac/algos/train_unfixed_ac.py:370` | rho | rho uses clip | empirical |
| p99_rho2 | 99th percentile of rho^2. | `tdrl_unfixed_ac/algos/train_unfixed_ac.py:371` | rho | rho uses clip | empirical |
| max_rho2 | max(rho^2). | `tdrl_unfixed_ac/algos/train_unfixed_ac.py:372` | rho | rho uses clip | empirical |
| delta_mean | mean(Delta). | `tdrl_unfixed_ac/algos/train_unfixed_ac.py:386` | Delta | no clip | empirical |
| delta_std | std(Delta). | `tdrl_unfixed_ac/algos/train_unfixed_ac.py:387` | Delta | no clip | empirical |
| delta_p95 | 95th percentile of Delta. | `tdrl_unfixed_ac/algos/train_unfixed_ac.py:388` | Delta | no clip | empirical |
| delta_p99 | 99th percentile of Delta. | `tdrl_unfixed_ac/algos/train_unfixed_ac.py:389` | Delta | no clip | empirical |
| delta_max | max(Delta). | `tdrl_unfixed_ac/algos/train_unfixed_ac.py:390` | Delta | no clip | empirical |
| fixed_point_gap | ||w - w_sharp|| from fixed-point probe. | `tdrl_unfixed_ac/probes/manager.py:156` | w, w_sharp | none | empirical estimate (probe) |
| fixed_point_drift | ||w_sharp - w_sharp_prev|| from probe. | `tdrl_unfixed_ac/probes/manager.py:158` | w_sharp | none | empirical estimate (probe) |
| fixed_point_drift_defined | 1 if drift is defined, else 0. | `tdrl_unfixed_ac/probes/manager.py:159` | prior probe state | none | empirical flag |
| stability_proxy | mean spectral proxy from stability probe. | `tdrl_unfixed_ac/probes/stability_probe.py:82` | rho, phi, bar_phi, gamma | uses train_step_scale, power_iters | theory proxy (linearized) |
| dist_mmd2 | MMD^2 between obs features under mu and pi. | `tdrl_unfixed_ac/probes/distribution_probe.py:197` | obs_vec samples | RBF kernel with median bandwidth | empirical |
| dist_mean_l2 | ||mean(obs_mu) - mean(obs_pi)||_2. | `tdrl_unfixed_ac/probes/distribution_probe.py:198` | obs_vec samples | none | empirical |
| dist_action_kl | mean KL(N(mu_pi,sigma_pi) || N(mu_mu,sigma_mu)). | `tdrl_unfixed_ac/probes/distribution_probe.py:203` | policy means, sigmas | analytic Gaussian KL | theory-aligned (closed form) |
| dist_action_tv | sample-based TV distance between actions. | `tdrl_unfixed_ac/probes/distribution_probe.py:205` | sampled actions | log-ratio clipped to +/-50 | empirical |
| td_loss_from_Q | (1/(2T_cache)) * sum_t Q_hat(t,t). | `tdrl_unfixed_ac/probes/q_kernel_probe.py:41` | Delta_cache | uses finite mask if NaN | empirical estimate of theory Q_hat |
| td_loss_from_Q_abs_diff | |td_loss_from_Q - td_loss|. | `tdrl_unfixed_ac/probes/q_kernel_probe.py:45` | td_loss, td_loss_from_Q | NaN if td_loss is 0 or non-finite | empirical diagnostic |
| td_loss_from_Q_rel_diff | abs_diff / |td_loss|. | `tdrl_unfixed_ac/probes/q_kernel_probe.py:46` | td_loss, td_loss_from_Q | NaN if td_loss is 0 or non-finite | empirical diagnostic |

## probes/fixed_point_probe.csv
| Metric | Definition | Code location | Dependencies | Clip/normalize/mask/stopgrad | Empirical vs theory |
| --- | --- | --- | --- | --- | --- |
| iter | Training iteration when probe ran. | `tdrl_unfixed_ac/probes/manager.py:171` | iteration | none | empirical (counter) |
| w_gap | ||w - w_sharp||. | `tdrl_unfixed_ac/probes/manager.py:157` | w, w_sharp | none | empirical estimate |
| w_sharp_drift | ||w_sharp - w_sharp_prev||. | `tdrl_unfixed_ac/probes/manager.py:158` | w_sharp | none | empirical estimate |
| w_sharp_drift_defined | 1 if drift computed, else 0. | `tdrl_unfixed_ac/probes/manager.py:159` | prior probe state | none | empirical flag |
| converged | 1 if fixed-point iterations converged. | `tdrl_unfixed_ac/probes/fixed_point_probe.py:57` | iterative solver | none | empirical flag |
| num_iters | Iterations used to converge. | `tdrl_unfixed_ac/probes/fixed_point_probe.py:73` | iterative solver | none | empirical |
| batch_size | Probe batch size. | `tdrl_unfixed_ac/probes/fixed_point_probe.py:74` | config | none | metadata |
| tol | Convergence tolerance. | `tdrl_unfixed_ac/probes/fixed_point_probe.py:75` | config | none | metadata |
| rho_mean | mean(rho) in probe batch. | `tdrl_unfixed_ac/probes/common.py:113` | logpi, logmu, rho_clip | action clipped; rho optionally clipped | empirical |
| rho2_mean | mean(rho^2) in probe batch. | `tdrl_unfixed_ac/probes/common.py:115` | rho | rho uses clip | empirical |
| rho_min | min(rho) in probe batch. | `tdrl_unfixed_ac/probes/common.py:116` | rho | rho uses clip | empirical |
| rho_max | max(rho) in probe batch. | `tdrl_unfixed_ac/probes/common.py:117` | rho | rho uses clip | empirical |
| rho_p95 | 95th percentile of rho. | `tdrl_unfixed_ac/probes/common.py:118` | rho | rho uses clip | empirical |
| rho_p99 | 99th percentile of rho. | `tdrl_unfixed_ac/probes/common.py:119` | rho | rho uses clip | empirical |
| rho_clip | rho clip upper bound (NaN if None). | `tdrl_unfixed_ac/probes/common.py:123` | config | none | metadata |
| rho_clip_active | 1 if clip active, else 0. | `tdrl_unfixed_ac/probes/common.py:124` | config | none | metadata |

## probes/stability_probe.csv
| Metric | Definition | Code location | Dependencies | Clip/normalize/mask/stopgrad | Empirical vs theory |
| --- | --- | --- | --- | --- | --- |
| iter | Training iteration when probe ran. | `tdrl_unfixed_ac/probes/manager.py:221` | iteration | none | empirical (counter) |
| stability_proxy | Mean spectral proxy from power iteration. | `tdrl_unfixed_ac/probes/stability_probe.py:82` | rho, phi, bar_phi, gamma | uses train_step_scale, power_iters | theory proxy |
| stability_proxy_mean | Same as stability_proxy. | `tdrl_unfixed_ac/probes/stability_probe.py:83` | same as above | same as above | theory proxy |
| stability_proxy_std | Std over k_mc repeats. | `tdrl_unfixed_ac/probes/stability_probe.py:84` | same as above | same as above | empirical |
| power_iters | Power-iteration steps. | `tdrl_unfixed_ac/probes/stability_probe.py:50` | config | none | metadata |
| batch_size | Probe batch size. | `tdrl_unfixed_ac/probes/stability_probe.py:94` | config | none | metadata |
| stability_probe_step_scale | Step scale used in probe. | `tdrl_unfixed_ac/probes/stability_probe.py:95` | alpha_w, trajectories, horizon | none | metadata |
| rho_mean | mean(rho) in probe batch. | `tdrl_unfixed_ac/probes/common.py:113` | logpi, logmu, rho_clip | action clipped; rho optionally clipped | empirical |
| rho2_mean | mean(rho^2) in probe batch. | `tdrl_unfixed_ac/probes/common.py:115` | rho | rho uses clip | empirical |
| rho_min | min(rho) in probe batch. | `tdrl_unfixed_ac/probes/common.py:116` | rho | rho uses clip | empirical |
| rho_max | max(rho) in probe batch. | `tdrl_unfixed_ac/probes/common.py:117` | rho | rho uses clip | empirical |
| rho_p95 | 95th percentile of rho. | `tdrl_unfixed_ac/probes/common.py:118` | rho | rho uses clip | empirical |
| rho_p99 | 99th percentile of rho. | `tdrl_unfixed_ac/probes/common.py:119` | rho | rho uses clip | empirical |
| rho_clip | rho clip upper bound (NaN if None). | `tdrl_unfixed_ac/probes/common.py:123` | config | none | metadata |
| rho_clip_active | 1 if clip active, else 0. | `tdrl_unfixed_ac/probes/common.py:124` | config | none | metadata |

## probes/distribution_probe.csv
| Metric | Definition | Code location | Dependencies | Clip/normalize/mask/stopgrad | Empirical vs theory |
| --- | --- | --- | --- | --- | --- |
| iter | Training iteration when probe ran. | `tdrl_unfixed_ac/probes/manager.py:260` | iteration | none | empirical (counter) |
| mmd2 | MMD^2 between obs under mu/pi. | `tdrl_unfixed_ac/probes/distribution_probe.py:197` | obs_vec samples | RBF kernel with median bandwidth | empirical |
| mmd_sigma | RBF bandwidth used for MMD. | `tdrl_unfixed_ac/probes/distribution_probe.py:152` | obs_vec samples | median heuristic | empirical |
| mean_l2 | ||mean(obs_mu) - mean(obs_pi)||_2. | `tdrl_unfixed_ac/probes/distribution_probe.py:198` | obs_vec samples | none | empirical |
| num_samples | Number of state samples. | `tdrl_unfixed_ac/probes/distribution_probe.py:229` | config | none | metadata |
| dist_action_kl | mean KL(N(mu_pi,sigma_pi) || N(mu_mu,sigma_mu)). | `tdrl_unfixed_ac/probes/distribution_probe.py:203` | policy means, sigmas | analytic Gaussian KL | theory-aligned |
| dist_action_tv | sample-based TV distance. | `tdrl_unfixed_ac/probes/distribution_probe.py:205` | sampled actions | log-ratio clipped to +/-50 | empirical |
| action_samples | Number of action samples per state. | `tdrl_unfixed_ac/probes/distribution_probe.py:232` | config | none | metadata |
| rho_mean | mean(rho) in rho-sample batch. | `tdrl_unfixed_ac/probes/distribution_probe.py:222` | logpi, logmu, rho_clip | action clipped; rho optionally clipped | empirical |
| rho2_mean | mean(rho^2) in rho-sample batch. | `tdrl_unfixed_ac/probes/common.py:115` | rho | rho uses clip | empirical |
| rho_min | min(rho) in rho-sample batch. | `tdrl_unfixed_ac/probes/common.py:116` | rho | rho uses clip | empirical |
| rho_max | max(rho) in rho-sample batch. | `tdrl_unfixed_ac/probes/common.py:117` | rho | rho uses clip | empirical |
| rho_p95 | 95th percentile of rho. | `tdrl_unfixed_ac/probes/common.py:118` | rho | rho uses clip | empirical |
| rho_p99 | 99th percentile of rho. | `tdrl_unfixed_ac/probes/common.py:119` | rho | rho uses clip | empirical |
| rho_clip | rho clip upper bound (NaN if None). | `tdrl_unfixed_ac/probes/common.py:123` | config | none | metadata |
| rho_clip_active | 1 if clip active, else 0. | `tdrl_unfixed_ac/probes/common.py:124` | config | none | metadata |

## probes/q_kernel_probe.csv
| Metric | Definition | Code location | Dependencies | Clip/normalize/mask/stopgrad | Empirical vs theory |
| --- | --- | --- | --- | --- | --- |
| iter | Training iteration when probe ran. | `tdrl_unfixed_ac/probes/q_kernel_probe.py:52` | iteration | none | empirical (counter) |
| td_loss | Logged training td_loss for that iter. | `tdrl_unfixed_ac/probes/q_kernel_probe.py:53` | td_loss | none | empirical |
| td_loss_from_Q | (1/(2T_cache)) * sum_t Q_hat(t,t). | `tdrl_unfixed_ac/probes/q_kernel_probe.py:41` | Delta_cache | finite-mask if NaN | empirical estimate |
| td_loss_from_Q_abs_diff | |td_loss_from_Q - td_loss|. | `tdrl_unfixed_ac/probes/q_kernel_probe.py:45` | td_loss, td_loss_from_Q | NaN if td_loss is 0 or non-finite | empirical diagnostic |
| td_loss_from_Q_rel_diff | abs_diff / |td_loss|. | `tdrl_unfixed_ac/probes/q_kernel_probe.py:46` | td_loss, td_loss_from_Q | NaN if td_loss is 0 or non-finite | empirical diagnostic |
| cache_batch_size | B_cache used for Delta_cache. | `tdrl_unfixed_ac/probes/q_kernel_probe.py:57` | config | none | metadata |
| cache_horizon | T_cache used for Delta_cache. | `tdrl_unfixed_ac/probes/q_kernel_probe.py:58` | config | none | metadata |
| cache_valid_t | Count of finite diag entries in Q_hat. | `tdrl_unfixed_ac/probes/q_kernel_probe.py:59` | Delta_cache | finite-mask | metadata |
