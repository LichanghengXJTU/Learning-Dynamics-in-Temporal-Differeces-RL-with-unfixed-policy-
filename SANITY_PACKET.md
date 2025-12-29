# 0. Executive Summary (one screen)
- Four sanity checks are defined by `scripts/run_sanity_suite.py` (on_policy, no_bootstrap, fixed_pi, full_triad_short); these are the only four templates in the suite.
- Latest suite at `outputs/sanity_suite/20251227_162149` shows all four checks FAIL due to `no_nan_inf` (NaN in `fixed_point_drift`), and on_policy also fails `on_policy_expected` because `dist_mmd2` ~0.02035 > 1e-3.
- Failures are metric/health-check based (no Python exceptions); root cause is the first probe drift being NaN and a strict health check that treats any NaN as failure.
- Current environment info: commit `a84b364b797739f6a6669e4a8308245a452d798f`, python `3.14.2`, os `macOS-15.3-arm64-arm-64bit-Mach-O`, numpy `not_installed: ModuleNotFoundError`, torch `not_installed: ModuleNotFoundError`.
- I did not rerun the suite in this environment (numpy/torch missing); evidence below is from existing outputs plus minimal repro commands.

# 1. Repository Map for Sanity Checks
- Suite runner: `scripts/run_sanity_suite.py` (main), wrapper `tools/run_sanity_suite.py`.
- Sanity templates: `configs/sanity/on_policy.yaml`, `configs/sanity/no_bootstrap.yaml`, `configs/sanity/fixed_pi.yaml`, `configs/sanity/full_triad_short.yaml`.
- Base sanity config: `configs/train_sanity.yaml` (also duplicated at `tdrl_unfixed_ac/configs/train_sanity.yaml`).
- Preflight smoke: `scripts/smoke_rollout.py`, tests `tests/test_env_smoke.py`.
- Reporting/diagnostics: `tdrl_unfixed_ac/reporting/run_report.py`, `scripts/check_run_report.py`, `scripts/aggregate_reports.py`.
- Probes: `tdrl_unfixed_ac/probes/fixed_point_probe.py`, `tdrl_unfixed_ac/probes/stability_probe.py`, `tdrl_unfixed_ac/probes/distribution_probe.py`, manager `tdrl_unfixed_ac/probes/manager.py`.
- Outputs: `outputs/sanity_suite/20251227_162149/*`, `outputs/sanity_suite/20251227_162149/SUMMARY.md`, `outputs/preflight_train/20251227_162033/run_report.{json,md}`.

# 2. The Four Sanity Checks (design + pass criteria)
All four checks are defined in `scripts/run_sanity_suite.py` via `CHECK_TEMPLATES` and run with base config `configs/train_sanity.yaml` plus per-check overrides.

## SC1: on_policy
- Name/entry: on_policy via `scripts/run_sanity_suite.py::_run_check` (check_name=on_policy).
- Hypothesis/invariant: mu == pi (beta=1.0, sigma_mu==sigma_pi) so mean_rho2 ~= 1 and dist_mmd2 ~= 0.
- Pass/Fail criteria: no_nan_inf pass; monotone_time pass; rho_sane pass; on_policy_expected: |mean_rho2-1|<=0.1 and dist_mmd2<=1e-3.
- How to run: `python3 tools/run_sanity_suite.py --base configs/train_sanity.yaml --out_root outputs/sanity_suite`
- Config dependency: base `configs/train_sanity.yaml` + overrides `configs/sanity/on_policy.yaml` + _apply_required_overrides(beta=1.0, sigma_mu=sigma_pi).
- Outputs: `outputs/sanity_suite/20251227_162149/on_policy/learning_curves.csv`, `outputs/sanity_suite/20251227_162149/on_policy/probes/*.csv`, `outputs/sanity_suite/20251227_162149/on_policy/run_report.json`, `outputs/sanity_suite/20251227_162149/on_policy/run_report.md`, `outputs/sanity_suite/20251227_162149/on_policy/stdout.log`.
- Call chain:
```text
scripts/run_sanity_suite.py::main
  -> _run_check (sets check_name + output_dir)
    -> tdrl_unfixed_ac.algos.train_unfixed_ac.train_unfixed_ac
      -> TorusGobletGhostEnv.reset/step (env transition + reward)
        -> ActorFeatureMap / CriticFeatureMap / build_observation_vector
      -> ProbeManager.maybe_run
        -> run_fixed_point_probe / run_stability_probe / run_distribution_probe
      -> generate_run_report -> _health_checks (no_nan_inf/on_policy_expected/etc.)
```

## SC2: no_bootstrap
- Name/entry: no_bootstrap via `scripts/run_sanity_suite.py::_run_check` (check_name=no_bootstrap).
- Hypothesis/invariant: gamma=0 removes bootstrap; TD update uses reward-only target.
- Pass/Fail criteria: no_nan_inf pass; monotone_time pass; rho_sane pass; no_bootstrap_expected: gamma==0.
- How to run: `python3 tools/run_sanity_suite.py --base configs/train_sanity.yaml --out_root outputs/sanity_suite`
- Config dependency: base `configs/train_sanity.yaml` + overrides `configs/sanity/no_bootstrap.yaml` + _apply_required_overrides(gamma=0.0).
- Outputs: `outputs/sanity_suite/20251227_162149/no_bootstrap/learning_curves.csv`, `outputs/sanity_suite/20251227_162149/no_bootstrap/probes/*.csv`, `outputs/sanity_suite/20251227_162149/no_bootstrap/run_report.json`, `outputs/sanity_suite/20251227_162149/no_bootstrap/run_report.md`, `outputs/sanity_suite/20251227_162149/no_bootstrap/stdout.log`.
- Call chain:
```text
scripts/run_sanity_suite.py::main
  -> _run_check (sets check_name + output_dir)
    -> tdrl_unfixed_ac.algos.train_unfixed_ac.train_unfixed_ac
      -> TorusGobletGhostEnv.reset/step (env transition + reward)
        -> ActorFeatureMap / CriticFeatureMap / build_observation_vector
      -> ProbeManager.maybe_run
        -> run_fixed_point_probe / run_stability_probe / run_distribution_probe
      -> generate_run_report -> _health_checks (no_nan_inf/on_policy_expected/etc.)
```

## SC3: fixed_pi
- Name/entry: fixed_pi via `scripts/run_sanity_suite.py::_run_check` (check_name=fixed_pi).
- Hypothesis/invariant: alpha_pi=0 disables actor updates; theta_pi should not change.
- Pass/Fail criteria: no_nan_inf pass; monotone_time pass; rho_sane pass; fixed_pi_expected: alpha_pi==0.
- How to run: `python3 tools/run_sanity_suite.py --base configs/train_sanity.yaml --out_root outputs/sanity_suite`
- Config dependency: base `configs/train_sanity.yaml` + overrides `configs/sanity/fixed_pi.yaml` + _apply_required_overrides(alpha_pi=0.0).
- Outputs: `outputs/sanity_suite/20251227_162149/fixed_pi/learning_curves.csv`, `outputs/sanity_suite/20251227_162149/fixed_pi/probes/*.csv`, `outputs/sanity_suite/20251227_162149/fixed_pi/run_report.json`, `outputs/sanity_suite/20251227_162149/fixed_pi/run_report.md`, `outputs/sanity_suite/20251227_162149/fixed_pi/stdout.log`.
- Call chain:
```text
scripts/run_sanity_suite.py::main
  -> _run_check (sets check_name + output_dir)
    -> tdrl_unfixed_ac.algos.train_unfixed_ac.train_unfixed_ac
      -> TorusGobletGhostEnv.reset/step (env transition + reward)
        -> ActorFeatureMap / CriticFeatureMap / build_observation_vector
      -> ProbeManager.maybe_run
        -> run_fixed_point_probe / run_stability_probe / run_distribution_probe
      -> generate_run_report -> _health_checks (no_nan_inf/on_policy_expected/etc.)
```

## SC4: full_triad_short
- Name/entry: full_triad_short via `scripts/run_sanity_suite.py::_run_check` (check_name=full_triad_short).
- Hypothesis/invariant: Short end-to-end integration of critic + actor + probes.
- Pass/Fail criteria: no_nan_inf pass; monotone_time pass; rho_sane pass (no extra check_name constraints).
- How to run: `python3 tools/run_sanity_suite.py --base configs/train_sanity.yaml --out_root outputs/sanity_suite`
- Config dependency: base `configs/train_sanity.yaml` + overrides `configs/sanity/full_triad_short.yaml` (outer_iters=8).
- Outputs: `outputs/sanity_suite/20251227_162149/full_triad_short/learning_curves.csv`, `outputs/sanity_suite/20251227_162149/full_triad_short/probes/*.csv`, `outputs/sanity_suite/20251227_162149/full_triad_short/run_report.json`, `outputs/sanity_suite/20251227_162149/full_triad_short/run_report.md`, `outputs/sanity_suite/20251227_162149/full_triad_short/stdout.log`.
- Call chain:
```text
scripts/run_sanity_suite.py::main
  -> _run_check (sets check_name + output_dir)
    -> tdrl_unfixed_ac.algos.train_unfixed_ac.train_unfixed_ac
      -> TorusGobletGhostEnv.reset/step (env transition + reward)
        -> ActorFeatureMap / CriticFeatureMap / build_observation_vector
      -> ProbeManager.maybe_run
        -> run_fixed_point_probe / run_stability_probe / run_distribution_probe
      -> generate_run_report -> _health_checks (no_nan_inf/on_policy_expected/etc.)
```

# 3. Implementation Evidence (key code excerpts)
All four checks share the same training and probe code paths. The excerpts below apply to SC1-SC4; each excerpt lists file, function(s), line range, and why it matters for the observed failures.

## EX1
- file: `scripts/run_sanity_suite.py`
- function: _log_config_summary/_apply_required_overrides/_resolve_run_config
- lines: L114-L177
- why relevant: Defines per-check overrides (on_policy/no_bootstrap/fixed_pi) and sets check_name/output_dir used by health checks.
```python
 114	def _log_config_summary(name: str, base_path: Path, template_path: Optional[Path], cfg: Dict[str, Any]) -> None:
 115	    print(f"== Sanity check: {name} ==")
 116	    print(f"Base config: {base_path}")
 117	    print(f"Template config: {template_path if template_path else 'None'}")
 118	    print("Key hyperparameters:")
 119	    keys = [
 120	        "seed",
 121	        "outer_iters",
 122	        "trajectories",
 123	        "horizon",
 124	        "gamma",
 125	        "alpha_w",
 126	        "alpha_pi",
 127	        "beta",
 128	        "sigma_mu",
 129	        "sigma_pi",
 130	        "K_mc",
 131	        "theta_radius",
 132	        "checkpoint_every",
 133	        "log_every",
 134	    ]
 135	    for key in keys:
 136	        if key in cfg:
 137	            print(f"  {key}: {cfg[key]}")
 138	    print(f"env_config_path: {cfg.get('env_config_path')}")
 139	    env_overrides = cfg.get("env", {})
 140	    if env_overrides:
 141	        print(f"env overrides: {env_overrides}")
 142	    probes_cfg = cfg.get("probes", {})
 143	    if probes_cfg:
 144	        print(f"probes.enabled: {probes_cfg.get('enabled', False)}")
 145	        print(f"probes.every: {probes_cfg.get('every', 0)}")
 146	    if name == "on_policy":
 147	        print("Expected: rho ~= 1 (mu=pi, sigma_mu=sigma_pi).")
 148	    if name == "no_bootstrap":
 149	        print("Expected: gamma=0 (bootstrap disabled).")
 150	    if name == "fixed_pi":
 151	        print("Expected: alpha_pi=0 (actor update disabled).")
 152	
 153	
 154	def _apply_required_overrides(name: str, cfg: Dict[str, Any]) -> None:
 155	    if name == "on_policy":
 156	        cfg["beta"] = 1.0
 157	        sigma_pi = float(cfg.get("sigma_pi", 1.0))
 158	        cfg["sigma_mu"] = sigma_pi
 159	    elif name == "no_bootstrap":
 160	        cfg["gamma"] = 0.0
 161	    elif name == "fixed_pi":
 162	        cfg["alpha_pi"] = 0.0
 163	
 164	
 165	def _resolve_run_config(base_path: Path, template_path: Path, run_dir: Path, name: str) -> Dict[str, Any]:
 166	    cfg = load_train_config(str(base_path))
 167	    if template_path.exists():
 168	        overrides = _load_payload(template_path)
 169	        if overrides:
 170	            _deep_update(cfg, overrides)
 171	    _apply_required_overrides(name, cfg)
 172	    cfg["check_name"] = name
 173	    cfg["output_dir"] = str(run_dir)
 174	    env_path = cfg.get("env_config_path")
 175	    if env_path:
 176	        cfg["env_config_path"] = _resolve_relative(str(env_path), base_path.parent)
 177	    return cfg
```

## EX2
- file: `scripts/run_sanity_suite.py`
- function: _run_preflight/_run_check
- lines: L180-L231
- why relevant: Executes each sanity run, captures stdout, and generates run_report.{json,md}.
```python
 180	def _run_preflight(suite_dir: Path, env_config_path: Optional[str]) -> None:
 181	    preflight_dir = suite_dir / "preflight_smoke"
 182	    preflight_dir.mkdir(parents=True, exist_ok=True)
 183	    log_path = preflight_dir / "stdout.log"
 184	    smoke_script = ROOT / "scripts" / "smoke_rollout.py"
 185	
 186	    with log_path.open("w") as log:
 187	        log.write("== Preflight: compileall ==\n")
 188	        try:
 189	            subprocess.run(
 190	                [sys.executable, "-m", "compileall", "."],
 191	                cwd=str(ROOT),
 192	                stdout=log,
 193	                stderr=log,
 194	                check=False,
 195	            )
 196	        except Exception:
 197	            traceback.print_exc(file=log)
 198	
 199	        log.write("\n== Preflight: smoke_rollout ==\n")
 200	        if smoke_script.exists():
 201	            cmd = [sys.executable, str(smoke_script), "--steps", "300"]
 202	            if env_config_path:
 203	                cmd.extend(["--config", env_config_path])
 204	            try:
 205	                subprocess.run(cmd, cwd=str(ROOT), stdout=log, stderr=log, check=False)
 206	            except Exception:
 207	                traceback.print_exc(file=log)
 208	        else:
 209	            log.write("smoke_rollout.py not found; skipping.\n")
 210	
 211	
 212	def _run_check(name: str, base_path: Path, template_path: Path, suite_dir: Path) -> None:
 213	    run_dir = suite_dir / name
 214	    run_dir.mkdir(parents=True, exist_ok=True)
 215	
 216	    log_path = run_dir / "stdout.log"
 217	    with log_path.open("w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
 218	        cfg: Optional[Dict[str, Any]] = None
 219	        exception: Optional[str] = None
 220	        try:
 221	            cfg = _resolve_run_config(base_path, template_path, run_dir, name)
 222	            _dump_config(run_dir / "config_resolved.yaml", cfg)
 223	            _log_config_summary(name, base_path, template_path, cfg)
 224	            result = train_unfixed_ac(cfg)
 225	            print(f"Training complete. Logs at {result['csv_path']}")
 226	        except Exception:
 227	            print("Run failed with exception:")
 228	            exception = traceback.format_exc()
 229	            print(exception)
 230	        finally:
 231	            _ensure_run_reports(run_dir, cfg, exception)
```

## EX3
- file: `tdrl_unfixed_ac/algos/train_unfixed_ac.py`
- function: load_train_config
- lines: L81-L166
- why relevant: Shows default config merge logic used by all checks.
```python
  81	def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
  82	    for key, value in updates.items():
  83	        if isinstance(value, dict) and isinstance(base.get(key), dict):
  84	            _deep_update(base[key], value)
  85	        else:
  86	            base[key] = value
  87	    return base
  88	
  89	
  90	def load_train_config(path: Optional[str] = None) -> Dict[str, Any]:
  91	    """Load training config from JSON/YAML file with defaults."""
  92	    config = deepcopy(DEFAULT_TRAIN_CONFIG)
  93	    if path is None and DEFAULT_TRAIN_CONFIG_PATH.exists():
  94	        path = str(DEFAULT_TRAIN_CONFIG_PATH)
  95	    if path is None:
  96	        return config
  97	    text = Path(path).read_text()
  98	    payload = yaml.safe_load(text) if yaml is not None else json.loads(text)
  99	    if payload:
 100	        _deep_update(config, payload)
 101	    return config
 102	
 103	
 104	def _clip_action(action: np.ndarray, v_max: float) -> np.ndarray:
 105	    norm = float(np.linalg.norm(action))
 106	    if norm > v_max and norm > 0.0:
 107	        return action / norm * v_max
 108	    return action
 109	
 110	
 111	def _mc_bar_phi(
 112	    env: TorusGobletGhostEnv,
 113	    policy: LinearGaussianPolicy,
 114	    psi: np.ndarray,
 115	    rng: np.random.Generator,
 116	    k_mc: int,
 117	) -> np.ndarray:
 118	    if k_mc <= 0:
 119	        return env.compute_features(np.zeros(policy.action_dim, dtype=float))["phi"]
 120	    phis = []
 121	    for _ in range(k_mc):
 122	        action = policy.sample_action(psi, rng)
 123	        action = _clip_action(action, env.v_max)
 124	        phi = env.compute_features(action)["phi"]
 125	        phis.append(phi)
 126	    return np.mean(np.stack(phis, axis=0), axis=0)
 127	
 128	
 129	def _json_ready(obj: Any) -> Any:
 130	    if isinstance(obj, np.ndarray):
 131	        return obj.tolist()
 132	    if isinstance(obj, (np.integer, np.floating)):
 133	        return obj.item()
 134	    return obj
 135	
 136	
 137	def _save_checkpoint(path: Path, **payload: Any) -> None:
 138	    path.parent.mkdir(parents=True, exist_ok=True)
 139	    with path.open("wb") as handle:
 140	        np.savez(handle, **payload)
 141	
 142	
 143	def _load_checkpoint(path: Path) -> Dict[str, Any]:
 144	    with np.load(path, allow_pickle=True) as data:
 145	        return {key: data[key] for key in data.files}
 146	
 147	
 148	def _load_existing_logs(csv_path: Path) -> list[Dict[str, Any]]:
 149	    if not csv_path.exists():
 150	        return []
 151	    with csv_path.open("r", newline="") as handle:
 152	        reader = csv.DictReader(handle)
 153	        return list(reader)
 154	
 155	
 156	def _last_logged_iter(logs: list[Dict[str, Any]]) -> Optional[int]:
 157	    for row in reversed(logs):
 158	        raw = row.get("iter")
 159	        if raw is None or str(raw).strip() == "":
 160	            continue
 161	        try:
 162	            return int(float(raw))
 163	        except ValueError:
 164	            continue
 165	    return None
 166	
```

## EX4
- file: `tdrl_unfixed_ac/algos/train_unfixed_ac.py`
- function: train_unfixed_ac (loop + logging)
- lines: L272-L360
- why relevant: Computes td_loss/mean_rho2 and writes probe outputs into learning_curves.csv.
```python
 272	        for n in range(start_iter, outer_iters):
 273	            mu_policy = LinearGaussianPolicy(theta=theta_mu, sigma=sigma_mu)
 274	            pi_policy = LinearGaussianPolicy(theta=theta_pi, sigma=sigma_pi)
 275	
 276	            grad_w = np.zeros_like(w)
 277	            grad_theta = np.zeros_like(theta_pi)
 278	            td_errors = []
 279	            rho_sq = []
 280	
 281	            for _ in range(trajectories):
 282	                env.reset(seed=int(rollout_rng.integers(0, seed_max)))
 283	                zero_action = np.zeros(2, dtype=float)
 284	                feat0 = env.compute_features(zero_action)
 285	                psi = feat0["psi"]
 286	                for _ in range(horizon):
 287	                    # ---- sample action from behavior policy mu ----
 288	                    a = mu_policy.sample_action(psi, rollout_rng)
 289	                    a = _clip_action(a, env.v_max)  # keep consistent with env._clip_action
 290	
 291	                    # ---- importance ratio rho = pi(a|s) / mu(a|s) ----
 292	                    logp_pi = pi_policy.log_prob(a, psi)
 293	                    logp_mu = mu_policy.log_prob(a, psi)
 294	                    rho = float(np.exp(logp_pi - logp_mu))
 295	
 296	                    # ---- step env (reward + phi are consistent via info["phi"]) ----
 297	                    obs, reward, terminated, truncated, info = env.step(a)
 298	
 299	                    phi = info["phi"]  # phi(s_t, a_t) used for reward + TD
 300	                    psi_next = info["psi_next"]  # psi(s_{t+1})
 301	
 302	                    # ---- compute bar_phi(s_{t+1}) = E_{a'~pi}[phi(s_{t+1},a')] ----
 303	                    bar_phi = _mc_bar_phi(env, pi_policy, psi_next, rollout_rng, k_mc=k_mc)
 304	
 305	                    # ---- TD error ----
 306	                    q_sa = critic_value(w, phi)
 307	                    q_next = critic_value(w, bar_phi)
 308	                    delta = float(reward + gamma * q_next - q_sa)
 309	
 310	                    # ---- actor score grad (target policy) ----
 311	                    g = pi_policy.score(a, psi)  # grad_theta log pi(a|s)
 312	
 313	                    # ---- accumulate gradients ----
 314	                    grad_w += rho * delta * phi
 315	                    grad_theta += rho * delta * g
 316	
 317	                    td_errors.append(delta)
 318	                    rho_sq.append(rho * rho)
 319	
 320	                    # advance
 321	                    psi = psi_next
 322	                    if terminated or truncated:
 323	                        break
 324	
 325	            scale = 1.0 / total_steps
 326	            w = w + alpha_w * scale * grad_w
 327	            theta_pi = theta_pi + alpha_pi * scale * grad_theta
 328	            theta_mu = (1.0 - beta) * theta_mu + beta * theta_pi
 329	
 330	            theta_pi = project_to_ball(theta_pi, theta_radius)
 331	            theta_mu = project_to_ball(theta_mu, theta_radius)
 332	
 333	            td_loss = float(np.mean(np.square(td_errors))) if td_errors else float("nan")
 334	            critic_teacher_error = float(np.dot(w - teacher_w, w - teacher_w) / feature_dim)
 335	            tracking_gap = float(np.linalg.norm(theta_pi - theta_mu) ** 2 / actor_dim)
 336	            mean_rho2 = float(np.mean(rho_sq)) if rho_sq else float("nan")
 337	            w_norm = float(np.linalg.norm(w))
 338	
 339	            log_row = {
 340	                "iter": n,
 341	                "td_loss": td_loss,
 342	                "critic_teacher_error": critic_teacher_error,
 343	                "tracking_gap": tracking_gap,
 344	                "mean_rho2": mean_rho2,
 345	                "w_norm": w_norm,
 346	                **probe_defaults,
 347	            }
 348	            probe_updates = probe_manager.maybe_run(
 349	                iteration=n, td_loss=td_loss, w=w, theta_mu=theta_mu, theta_pi=theta_pi
 350	            )
 351	            if probe_updates:
 352	                log_row.update(probe_updates)
 353	            logs.append(log_row)
 354	            if csv_writer is None:
 355	                csv_fieldnames = list(log_row.keys())
 356	                csv_handle = csv_path.open("w", newline="")
 357	                csv_writer = csv.DictWriter(csv_handle, fieldnames=csv_fieldnames)
 358	                csv_writer.writeheader()
 359	            csv_writer.writerow(log_row)
 360	            csv_handle.flush()
```

## EX5
- file: `tdrl_unfixed_ac/probes/manager.py`
- function: ProbeManager.log_defaults/ProbeManager.maybe_run
- lines: L69-L154
- why relevant: Introduces NaN defaults and sets fixed_point_drift to NaN on the first probe, triggering no_nan_inf.
```python
  69	    def log_defaults(self) -> Dict[str, float]:
  70	        defaults: Dict[str, float] = {}
  71	        if not self.enabled:
  72	            return defaults
  73	        if self.fixed_enabled:
  74	            defaults["fixed_point_gap"] = float("nan")
  75	            defaults["fixed_point_drift"] = float("nan")
  76	        if self.stability_enabled:
  77	            defaults["stability_proxy"] = float("nan")
  78	        if self.dist_enabled:
  79	            defaults["dist_mmd2"] = float("nan")
  80	            defaults["dist_mean_l2"] = float("nan")
  81	        return defaults
  82	
  83	    def maybe_run(
  84	        self,
  85	        *,
  86	        iteration: int,
  87	        td_loss: float,
  88	        w: np.ndarray,
  89	        theta_mu: np.ndarray,
  90	        theta_pi: np.ndarray,
  91	    ) -> Dict[str, float]:
  92	        if not self.enabled:
  93	            return {}
  94	
  95	        self._td_history.append(float(td_loss))
  96	        if self._last_probe_iter == iteration:
  97	            return {}
  98	
  99	        should_run = self._should_run(iteration)
 100	        if not should_run:
 101	            return {}
 102	
 103	        self._last_probe_iter = iteration
 104	        results: Dict[str, float] = {}
 105	
 106	        if self.fixed_enabled:
 107	            fixed_cfg = self._with_defaults(
 108	                self.fixed_cfg,
 109	                {
 110	                    "batch_size": 4096,
 111	                    "max_iters": 200,
 112	                    "tol": 1e-4,
 113	                    "alpha_w": self.alpha_w,
 114	                    "gamma": self.gamma,
 115	                    "k_mc": self.k_mc,
 116	                },
 117	            )
 118	            fixed_out = run_fixed_point_probe(
 119	                env_config=self.env_config,
 120	                theta_mu=theta_mu,
 121	                theta_pi=theta_pi,
 122	                w_init=w,
 123	                sigma_mu=self.sigma_mu,
 124	                sigma_pi=self.sigma_pi,
 125	                alpha_w=float(fixed_cfg["alpha_w"]),
 126	                gamma=float(fixed_cfg["gamma"]),
 127	                k_mc=int(fixed_cfg["k_mc"]),
 128	                batch_size=int(fixed_cfg["batch_size"]),
 129	                max_iters=int(fixed_cfg["max_iters"]),
 130	                tol=float(fixed_cfg["tol"]),
 131	                seed=self._seed_for(iteration, 1),
 132	            )
 133	            w_sharp = fixed_out["w_sharp"]
 134	            gap = float(np.linalg.norm(w - w_sharp))
 135	            drift = (
 136	                float(np.linalg.norm(w_sharp - self._prev_w_sharp))
 137	                if self._prev_w_sharp is not None
 138	                else float("nan")
 139	            )
 140	            self._prev_w_sharp = np.array(w_sharp, copy=True)
 141	            results["fixed_point_gap"] = gap
 142	            results["fixed_point_drift"] = drift
 143	            self._append_probe(
 144	                "fixed_point_probe",
 145	                {
 146	                    "iter": iteration,
 147	                    "w_gap": gap,
 148	                    "w_sharp_drift": drift,
 149	                    "converged": fixed_out["converged"],
 150	                    "num_iters": fixed_out["num_iters"],
 151	                    "batch_size": fixed_out["batch_size"],
 152	                    "tol": fixed_out["tol"],
 153	                },
 154	            )
```

## EX6
- file: `tdrl_unfixed_ac/reporting/run_report.py`
- function: _health_checks
- lines: L249-L368
- why relevant: Implements no_nan_inf and on_policy_expected thresholds that fail current runs.
```python
 249	    if not rows:
 250	        checks["no_nan_inf"] = {
 251	            "pass": False,
 252	            "reason": "learning_curves.csv missing or empty",
 253	            "observed": {"nan_count": None, "inf_count": None},
 254	            "applicable": True,
 255	        }
 256	    else:
 257	        nan_cols: List[str] = []
 258	        inf_cols: List[str] = []
 259	        nan_count = 0
 260	        inf_count = 0
 261	        for col in numeric_cols:
 262	            for row in rows:
 263	                value = row.get(col)
 264	                if isinstance(value, float) and math.isnan(value):
 265	                    nan_count += 1
 266	                    if col not in nan_cols:
 267	                        nan_cols.append(col)
 268	                if isinstance(value, float) and math.isinf(value):
 269	                    inf_count += 1
 270	                    if col not in inf_cols:
 271	                        inf_cols.append(col)
 272	        passed = nan_count == 0 and inf_count == 0
 273	        reason = "all numeric columns finite" if passed else "found NaN/Inf in numeric columns"
 274	        checks["no_nan_inf"] = {
 275	            "pass": passed,
 276	            "reason": reason,
 277	            "observed": {"nan_count": nan_count, "inf_count": inf_count, "nan_cols": nan_cols, "inf_cols": inf_cols},
 278	            "applicable": True,
 279	        }
 280	
 281	    # monotone_time
 282	    time_col = None
 283	    if "step" in numeric_cols:
 284	        time_col = "step"
 285	    elif "iter" in numeric_cols:
 286	        time_col = "iter"
 287	
 288	    if not rows or time_col is None:
 289	        checks["monotone_time"] = {
 290	            "pass": False,
 291	            "reason": "time column missing or no data",
 292	            "observed": {"time_col": time_col, "num_rows": len(rows)},
 293	            "applicable": True,
 294	        }
 295	    else:
 296	        values = [row.get(time_col) for row in rows]
 297	        finite = _iter_finite(values)
 298	        if len(finite) < 2:
 299	            checks["monotone_time"] = {
 300	                "pass": True,
 301	                "reason": "not enough rows to check monotonicity",
 302	                "observed": {"time_col": time_col, "num_rows": len(rows)},
 303	                "applicable": True,
 304	            }
 305	        else:
 306	            passed = True
 307	            first_bad = None
 308	            prev = finite[0]
 309	            for idx, value in enumerate(finite[1:], start=1):
 310	                if value <= prev:
 311	                    passed = False
 312	                    first_bad = idx
 313	                    break
 314	                prev = value
 315	            reason = "time is strictly increasing" if passed else "time column not strictly increasing"
 316	            checks["monotone_time"] = {
 317	                "pass": passed,
 318	                "reason": reason,
 319	                "observed": {"time_col": time_col, "first_violation_index": first_bad},
 320	                "applicable": True,
 321	            }
 322	
 323	    # rho_sane
 324	    rho_col = None
 325	    if "mean_rho2" in numeric_cols:
 326	        rho_col = "mean_rho2"
 327	    else:
 328	        for col in sorted(numeric_cols):
 329	            if "rho" in col:
 330	                rho_col = col
 331	                break
 332	    if rho_col is None:
 333	        checks["rho_sane"] = {
 334	            "pass": True,
 335	            "reason": "rho statistics not found",
 336	            "observed": {"rho_column": None, "threshold": rho_threshold},
 337	            "applicable": False,
 338	        }
 339	    else:
 340	        values = _iter_finite([row.get(rho_col) for row in rows])
 341	        max_val = max(values) if values else None
 342	        passed = max_val is not None and max_val <= rho_threshold
 343	        reason = "rho within threshold" if passed else "rho exceeds threshold"
 344	        checks["rho_sane"] = {
 345	            "pass": passed,
 346	            "reason": reason,
 347	            "observed": {"rho_column": rho_col, "max": max_val, "threshold": rho_threshold},
 348	            "applicable": True,
 349	        }
 350	
 351	    # on_policy_expected
 352	    check_name = cfg.get("check_name")
 353	    applicable = check_name == "on_policy"
 354	    mean_rho2 = None
 355	    if "mean_rho2" in numeric_cols:
 356	        mean_rho2 = _last_finite([row.get("mean_rho2") for row in rows])
 357	    rho_ok = mean_rho2 is not None and abs(mean_rho2 - 1.0) <= 0.1
 358	    mmd_ok = dist_mmd2_value is not None and dist_mmd2_value <= 1e-3
 359	    if applicable:
 360	        passed = rho_ok and mmd_ok if dist_mmd2_value is not None else rho_ok
 361	        if dist_mmd2_value is None:
 362	            reason = "mean_rho2 close to 1; dist_mmd2 unavailable"
 363	        else:
 364	            reason = "mean_rho2 close to 1 and dist_mmd2 close to 0" if passed else "on-policy expectations not met"
 365	        checks["on_policy_expected"] = {
 366	            "pass": passed,
 367	            "reason": reason,
 368	            "observed": {"mean_rho2": mean_rho2, "dist_mmd2": dist_mmd2_value},
```

## EX7
- file: `tdrl_unfixed_ac/envs/torus_gg.py`
- function: TorusGobletGhostEnv.reset/step
- lines: L120-L225
- why relevant: Defines env transition and reward/feature semantics used by sanity metrics.
```python
 120	    # API ----------------------------------------------------------------- #
 121	    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
 122	        if seed is not None:
 123	            self.rng = self.seeder.reseed(seed)
 124	        self._step_count = 0
 125	        self._sample_state(resample_types=True)
 126	        self.last_events = {
 127	            "caught": False,
 128	            "picked": False,
 129	            "picked_type": 0.0,
 130	            "restart": False,
 131	        }
 132	        return self._get_obs(), {"seed": seed}
 133	        
 134	    def step(self, action):
 135	        """
 136	        Gym-style step.
 137	
 138	        IMPORTANT SEMANTICS (for DMFT / teacher-student consistency):
 139	        - reward is computed from the SAME feature vector phi(s_t, a_t) returned in info["phi"].
 140	        - events (caught/picked/restart) describe what happened during THIS transition and
 141	          are written into self.last_events for the NEXT observation.
 142	        - if restart happens (p-mix), we ignore caught/picked from the regular dynamics step.
 143	        """
 144	        # -------------------------
 145	        # 0) sanitize + clip action
 146	        # -------------------------
 147	        action = np.asarray(action, dtype=float).reshape(-1)
 148	        if action.shape[0] != 2:
 149	            raise ValueError(f"Expected action dim 2, got shape {action.shape}")
 150	        clipped_action = self._clip_action(action)
 151	
 152	        # ---------------------------------------------------------
 153	        # 1) compute features on CURRENT state (s_t) for TD + reward
 154	        # ---------------------------------------------------------
 155	        obs_t = self._get_obs()
 156	        obs_vec_t, psi_t, phi_t = self._compute_features(obs_t, clipped_action)
 157	
 158	        reward = 0.0
 159	        if self.use_teacher_reward:
 160	            reward = float(self.teacher_reward(phi_t))
 161	
 162	        # -------------------------
 163	        # 2) advance the environment
 164	        # -------------------------
 165	        # adv dynamics
 166	        env_noise = self.rng.normal(loc=0.0, scale=self.sigma_env, size=(2,))
 167	        self.adventurer = self._wrap(self.adventurer + self.dt * clipped_action + env_noise)
 168	
 169	        # ghost dynamics
 170	        self._ghost_step()
 171	
 172	        # events under regular dynamics
 173	        caught = self._check_caught()
 174	        picked, picked_type = self._check_picked()
 175	
 176	        # p-mix restart
 177	        restart = False
 178	        if self.rng.random() < self.p_mix:
 179	            self._sample_state(resample_types=True)
 180	            restart = True
 181	            # By definition of restart ~ nu, ignore events from the regular transition
 182	            caught = False
 183	            picked = False
 184	            picked_type = 0.0
 185	
 186	        # bookkeeping + write events into NEXT obs
 187	        self._step_count += 1
 188	        self.last_events = {
 189	            "caught": bool(caught),
 190	            "picked": bool(picked),
 191	            "picked_type": float(picked_type),
 192	            "restart": bool(restart),
 193	        }
 194	
 195	        # ---------------------------------------------------------
 196	        # 3) build next observation (s_{t+1}) + next-state features
 197	        # ---------------------------------------------------------
 198	        obs_next = self._get_obs()
 199	        obs_vec_next, psi_next, phi_next = self._compute_features(obs_next, clipped_action)
 200	
 201	        terminated = False
 202	        truncated = False
 203	
 204	        info = {
 205	            # transition events (THIS step)
 206	            "caught": bool(caught),
 207	            "picked": bool(picked),
 208	            "picked_type": float(picked_type),
 209	            "restart": bool(restart),
 210	            "step": int(self._step_count),
 211	
 212	            # features for CURRENT (s_t, a_t) used for reward + TD
 213	            "obs_vec": obs_vec_t,
 214	            "psi": psi_t,
 215	            "phi": phi_t,
 216	            "clipped_action": clipped_action,
 217	            "reward_teacher": float(reward),
 218	
 219	            # convenience: next-state features (s_{t+1})
 220	            "obs_vec_next": obs_vec_next,
 221	            "psi_next": psi_next,
 222	            "phi_next": phi_next,
 223	        }
 224	
 225	        return obs_next, reward, terminated, truncated, info
```

## EX8
- file: `tdrl_unfixed_ac/features/observations.py`
- function: build_event_flags/build_observation_vector
- lines: L12-L78
- why relevant: Defines observation vector and event flags used in features and probes.
```python
  12	def _encode_position(pos: np.ndarray, torus_size: float) -> np.ndarray:
  13	    """Encode a 2D position with sin/cos to remove torus discontinuity."""
  14	    pos = np.asarray(pos, dtype=float).reshape(2)
  15	    angle = 2.0 * np.pi * pos / torus_size
  16	    return np.concatenate([np.sin(angle), np.cos(angle)], axis=0)
  17	
  18	
  19	def build_event_flags(raw_obs: Dict[str, Any]) -> np.ndarray:
  20	    """Return event one-hots: caught, picked_pos, picked_neg, restart."""
  21	    picked_type = float(raw_obs.get("picked_type", 0.0))
  22	    picked = bool(raw_obs.get("picked", False))
  23	    return np.array(
  24	        [
  25	            float(raw_obs.get("caught", False)),
  26	            float(picked and picked_type > 0.0),
  27	            float(picked and picked_type < 0.0),
  28	            float(raw_obs.get("restart", False)),
  29	        ],
  30	        dtype=float,
  31	    )
  32	
  33	
  34	def build_observation_vector(raw_obs: Dict[str, Any], torus_size: float) -> np.ndarray:
  35	    """
  36	    Build a low-dimensional observation vector o(s) from raw env observation.
  37	
  38	    Components (all continuous and fixed-size):
  39	        - sin/cos of adventurer position (4)
  40	        - sin/cos of ghost position (4)
  41	        - relative vector to ghost (2, scaled by torus size)
  42	        - relative vector to nearest goblet (2, scaled by torus size)
  43	        - nearest goblet type (1) and mean goblet type (1)
  44	        - event flags caught/pick_pos/pick_neg/restart (4)
  45	    """
  46	    adventurer = np.asarray(raw_obs["adventurer"], dtype=float).reshape(2)
  47	    ghost = np.asarray(raw_obs["ghost"], dtype=float).reshape(2)
  48	    goblets_pos = np.asarray(raw_obs["goblets_pos"], dtype=float)
  49	    goblets_type = np.asarray(raw_obs["goblets_type"], dtype=float)
  50	
  51	    rel_ghost = torus_delta(adventurer, ghost, torus_size) / torus_size
  52	
  53	    if goblets_pos.size > 0:
  54	        deltas = torus_delta(adventurer, goblets_pos, torus_size)
  55	        dists = np.linalg.norm(deltas, axis=1)
  56	        nearest_idx = int(np.argmin(dists))
  57	        nearest_delta = deltas[nearest_idx] / torus_size
  58	        nearest_type = float(goblets_type[nearest_idx])
  59	        mean_type = float(np.mean(goblets_type))
  60	    else:
  61	        nearest_delta = np.zeros(2, dtype=float)
  62	        nearest_type = 0.0
  63	        mean_type = 0.0
  64	
  65	    events = build_event_flags(raw_obs)
  66	
  67	    obs_vec = np.concatenate(
  68	        [
  69	            _encode_position(adventurer, torus_size),
  70	            _encode_position(ghost, torus_size),
  71	            rel_ghost,
  72	            nearest_delta,
  73	            np.array([nearest_type, mean_type], dtype=float),
  74	            events,
  75	        ],
  76	        axis=0,
  77	    )
  78	    return obs_vec
```

## EX9
- file: `tdrl_unfixed_ac/features/actor_features.py`
- function: ActorFeatureMap.__call__
- lines: L1-L30
- why relevant: Defines actor feature dimension and psi clipping.
```python
   1	"""Actor feature map producing bounded psi(s)."""
   2	
   3	from __future__ import annotations
   4	
   5	from typing import Optional
   6	
   7	import numpy as np
   8	
   9	
  10	class ActorFeatureMap:
  11	    """Fixed random projection with norm clipping to bound psi(s)."""
  12	
  13	    def __init__(self, obs_dim: int, dim: int, c_psi: float = 1.0, rng: Optional[np.random.Generator] = None) -> None:
  14	        self.obs_dim = int(obs_dim)
  15	        self.dim = int(dim)
  16	        self.c_psi = float(c_psi)
  17	        self.rng = rng if rng is not None else np.random.default_rng()
  18	
  19	        scale = 1.0 / max(np.sqrt(self.obs_dim), 1.0)
  20	        self.W = self.rng.normal(loc=0.0, scale=scale, size=(self.dim, self.obs_dim))
  21	        self.b = self.rng.normal(loc=0.0, scale=0.1, size=self.dim)
  22	
  23	    def __call__(self, obs_vec: np.ndarray) -> np.ndarray:
  24	        obs_vec = np.asarray(obs_vec, dtype=float).reshape(self.obs_dim)
  25	        z = self.W @ obs_vec + self.b
  26	        psi = np.tanh(z)
  27	        norm = float(np.linalg.norm(psi))
  28	        if norm > self.c_psi:
  29	            psi = psi * (self.c_psi / (norm + 1e-12))
  30	        return psi
```

## EX10
- file: `tdrl_unfixed_ac/features/critic_features.py`
- function: CriticFeatureMap.__call__
- lines: L12-L50
- why relevant: Defines critic feature dimension and event features used by reward/TD.
```python
  12	class CriticFeatureMap:
  13	    """Compute phi(s, a) with random Fourier features and appended event features."""
  14	
  15	    def __init__(
  16	        self,
  17	        obs_dim: int,
  18	        dim: int,
  19	        *,
  20	        action_dim: int = 2,
  21	        sigma: float = 1.0,
  22	        rng: Optional[np.random.Generator] = None,
  23	    ) -> None:
  24	        self.obs_dim = int(obs_dim)
  25	        self.dim = int(dim)
  26	        self.action_dim = int(action_dim)
  27	        self.event_dim = 4  # caught, pick_pos, pick_neg, action_penalty
  28	        self.base_dim = max(self.dim - self.event_dim, 1)
  29	        self.input_dim = self.obs_dim + self.action_dim + 1  # +1 for bias term in x
  30	        self.rng = rng if rng is not None else np.random.default_rng()
  31	
  32	        self.W = self.rng.normal(loc=0.0, scale=sigma, size=(self.base_dim, self.input_dim))
  33	        self.b = self.rng.uniform(low=0.0, high=2.0 * np.pi, size=self.base_dim)
  34	
  35	    def __call__(self, obs_vec: np.ndarray, action: np.ndarray, raw_obs: Dict[str, Any]) -> np.ndarray:
  36	        obs_vec = np.asarray(obs_vec, dtype=float).reshape(self.obs_dim)
  37	        action = np.asarray(action, dtype=float).reshape(self.action_dim)
  38	        x = np.concatenate([obs_vec, action, np.array([1.0], dtype=float)], axis=0)
  39	
  40	        proj = self.W @ x + self.b
  41	        base_features = np.sqrt(2.0 / self.base_dim) * np.cos(proj)
  42	
  43	        events = build_event_flags(raw_obs)
  44	        action_penalty = float(np.dot(action, action))
  45	        event_features = np.concatenate([events[:3], np.array([action_penalty], dtype=float)])
  46	
  47	        phi_full = np.concatenate([base_features, event_features], axis=0)
  48	        if phi_full.shape[0] < self.dim:
  49	            phi_full = np.pad(phi_full, (0, self.dim - phi_full.shape[0]))
  50	        return phi_full[: self.dim]
```

## EX11
- file: `tdrl_unfixed_ac/utils/seeding.py`
- function: Seeder
- lines: L1-L31
- why relevant: Defines deterministic seed spawning used by env and probes.
```python
   1	"""Deterministic seeding utilities."""
   2	
   3	from __future__ import annotations
   4	
   5	from dataclasses import dataclass
   6	from typing import Optional
   7	
   8	import numpy as np
   9	
  10	
  11	@dataclass
  12	class Seeder:
  13	    """Wrapper around numpy SeedSequence to simplify reproducible RNG."""
  14	
  15	    seed: Optional[int] = None
  16	
  17	    def __post_init__(self) -> None:
  18	        self.seed_sequence = np.random.SeedSequence(self.seed)
  19	        self.rng = np.random.default_rng(self.seed_sequence)
  20	
  21	    def spawn(self) -> np.random.Generator:
  22	        """Spawn a child generator deterministically."""
  23	        child_seq = self.seed_sequence.spawn(1)[0]
  24	        return np.random.default_rng(child_seq)
  25	
  26	    def reseed(self, seed: Optional[int]) -> np.random.Generator:
  27	        """Reset to a new base seed and return the generator."""
  28	        self.seed = seed
  29	        self.seed_sequence = np.random.SeedSequence(self.seed)
  30	        self.rng = np.random.default_rng(self.seed_sequence)
  31	        return self.rng
```

## EX12
- file: `tdrl_unfixed_ac/probes/distribution_probe.py`
- function: _collect_obs_vectors/_mmd_rbf/run_distribution_probe
- lines: L14-L101
- why relevant: Computes dist_mmd2; on_policy_expected compares it to 1e-3.
```python
  14	def _collect_obs_vectors(
  15	    env: TorusGobletGhostEnv,
  16	    policy: LinearGaussianPolicy,
  17	    rng: np.random.Generator,
  18	    num_samples: int,
  19	) -> np.ndarray:
  20	    action_dim = int(policy.action_dim)
  21	    zero_action = np.zeros(action_dim, dtype=float)
  22	    seed_max = np.iinfo(np.int32).max
  23	
  24	    env.reset(seed=int(rng.integers(0, seed_max)))
  25	    obs_vecs = []
  26	    for _ in range(num_samples):
  27	        features = env.compute_features(zero_action)
  28	        obs_vecs.append(features["obs_vec"])
  29	
  30	        psi = features["psi"]
  31	        action = policy.sample_action(psi, rng)
  32	        action = clip_action(action, env.v_max)
  33	        _, _, terminated, truncated, _ = env.step(action)
  34	        if terminated or truncated:
  35	            raise RuntimeError("Environment should be continuing but returned a terminal flag.")
  36	
  37	    return np.stack(obs_vecs, axis=0)
  38	
  39	
  40	def _median_bandwidth(x: np.ndarray, max_samples: int, rng: np.random.Generator) -> float:
  41	    if x.shape[0] > max_samples:
  42	        idx = rng.choice(x.shape[0], size=max_samples, replace=False)
  43	        x = x[idx]
  44	    diffs = x[:, None, :] - x[None, :, :]
  45	    dist_sq = np.sum(diffs * diffs, axis=-1)
  46	    median = float(np.median(dist_sq))
  47	    if median <= 1e-12:
  48	        return 1.0
  49	    return np.sqrt(0.5 * median)
  50	
  51	
  52	def _mmd_rbf(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> Tuple[float, float]:
  53	    combined = np.concatenate([x, y], axis=0)
  54	    sigma = _median_bandwidth(combined, max_samples=300, rng=rng)
  55	    denom = 2.0 * sigma * sigma
  56	
  57	    diff_xx = x[:, None, :] - x[None, :, :]
  58	    diff_yy = y[:, None, :] - y[None, :, :]
  59	    diff_xy = x[:, None, :] - y[None, :, :]
  60	
  61	    k_xx = np.exp(-np.sum(diff_xx * diff_xx, axis=-1) / denom)
  62	    k_yy = np.exp(-np.sum(diff_yy * diff_yy, axis=-1) / denom)
  63	    k_xy = np.exp(-np.sum(diff_xy * diff_xy, axis=-1) / denom)
  64	
  65	    mmd2 = float(np.mean(k_xx) + np.mean(k_yy) - 2.0 * np.mean(k_xy))
  66	    return mmd2, sigma
  67	
  68	
  69	def run_distribution_probe(
  70	    *,
  71	    env_config: Dict[str, Any],
  72	    theta_mu: np.ndarray,
  73	    theta_pi: np.ndarray,
  74	    sigma_mu: float,
  75	    sigma_pi: float,
  76	    num_samples: int,
  77	    seed: Optional[int],
  78	) -> Dict[str, Any]:
  79	    """Compare visitation distributions using MMD over observation vectors."""
  80	    base_seed = int(seed) if seed is not None else 0
  81	    rng_mu = np.random.default_rng(base_seed + 1)
  82	    rng_pi = np.random.default_rng(base_seed + 2)
  83	
  84	    env_mu = TorusGobletGhostEnv(config=env_config, rng=np.random.default_rng(base_seed + 11))
  85	    env_pi = TorusGobletGhostEnv(config=env_config, rng=np.random.default_rng(base_seed + 13))
  86	
  87	    mu_policy = LinearGaussianPolicy(theta=np.array(theta_mu, copy=True), sigma=float(sigma_mu))
  88	    pi_policy = LinearGaussianPolicy(theta=np.array(theta_pi, copy=True), sigma=float(sigma_pi))
  89	
  90	    obs_mu = _collect_obs_vectors(env_mu, mu_policy, rng_mu, num_samples)
  91	    obs_pi = _collect_obs_vectors(env_pi, pi_policy, rng_pi, num_samples)
  92	
  93	    mmd2, sigma = _mmd_rbf(obs_mu, obs_pi, rng_mu)
  94	    mean_l2 = float(np.linalg.norm(obs_mu.mean(axis=0) - obs_pi.mean(axis=0)))
  95	
  96	    return {
  97	        "mmd2": float(mmd2),
  98	        "mmd_sigma": float(sigma),
  99	        "mean_l2": float(mean_l2),
 100	        "num_samples": int(num_samples),
 101	    }
```

## EX13
- file: `tdrl_unfixed_ac/probes/fixed_point_probe.py`
- function: run_fixed_point_probe
- lines: L14-L63
- why relevant: Defines fixed-point iteration that produces w_sharp and drift.
```python
  14	def run_fixed_point_probe(
  15	    *,
  16	    env_config: Dict[str, Any],
  17	    theta_mu: np.ndarray,
  18	    theta_pi: np.ndarray,
  19	    w_init: np.ndarray,
  20	    sigma_mu: float,
  21	    sigma_pi: float,
  22	    alpha_w: float,
  23	    gamma: float,
  24	    k_mc: int,
  25	    batch_size: int,
  26	    max_iters: int,
  27	    tol: float,
  28	    seed: Optional[int],
  29	) -> Dict[str, Any]:
  30	    """Estimate the TD fixed point w_sharp for frozen (mu, pi)."""
  31	    rng = np.random.default_rng(seed)
  32	    env = TorusGobletGhostEnv(config=env_config, rng=rng)
  33	
  34	    mu_policy = LinearGaussianPolicy(theta=np.array(theta_mu, copy=True), sigma=float(sigma_mu))
  35	    pi_policy = LinearGaussianPolicy(theta=np.array(theta_pi, copy=True), sigma=float(sigma_pi))
  36	
  37	    batch = collect_critic_batch(env, mu_policy, pi_policy, rng, batch_size, k_mc)
  38	
  39	    w = np.array(w_init, copy=True)
  40	    feature_dim = w.shape[0]
  41	    scale = np.sqrt(feature_dim)
  42	    diff = gamma * batch["bar_phi"] - batch["phi"]
  43	
  44	    converged = False
  45	    steps = 0
  46	    for step in range(int(max_iters)):
  47	        delta = batch["reward"] + (diff @ w) / scale
  48	        grad = (batch["rho"] * delta)[:, None] * batch["phi"]
  49	        w_next = w + alpha_w * grad.mean(axis=0)
  50	        steps = step + 1
  51	        if float(np.linalg.norm(w_next - w)) <= tol:
  52	            converged = True
  53	            w = w_next
  54	            break
  55	        w = w_next
  56	
  57	    return {
  58	        "w_sharp": w,
  59	        "converged": converged,
  60	        "num_iters": steps,
  61	        "batch_size": int(batch_size),
  62	        "tol": float(tol),
  63	    }
```

## EX14
- file: `tdrl_unfixed_ac/probes/stability_probe.py`
- function: run_stability_probe
- lines: L14-L67
- why relevant: Defines stability_proxy computed in probes.
```python
  14	def run_stability_probe(
  15	    *,
  16	    env_config: Dict[str, Any],
  17	    theta_mu: np.ndarray,
  18	    theta_pi: np.ndarray,
  19	    sigma_mu: float,
  20	    sigma_pi: float,
  21	    alpha_w: float,
  22	    gamma: float,
  23	    k_mc: int,
  24	    batch_size: int,
  25	    power_iters: int,
  26	    seed: Optional[int],
  27	) -> Dict[str, Any]:
  28	    """Estimate local amplification (spectral radius proxy) for critic updates."""
  29	    rng = np.random.default_rng(seed)
  30	    env = TorusGobletGhostEnv(config=env_config, rng=rng)
  31	
  32	    mu_policy = LinearGaussianPolicy(theta=np.array(theta_mu, copy=True), sigma=float(sigma_mu))
  33	    pi_policy = LinearGaussianPolicy(theta=np.array(theta_pi, copy=True), sigma=float(sigma_pi))
  34	
  35	    batch = collect_critic_batch(env, mu_policy, pi_policy, rng, batch_size, k_mc)
  36	
  37	    phi = batch["phi"]
  38	    diff = gamma * batch["bar_phi"] - phi
  39	    rho = batch["rho"]
  40	    feature_dim = phi.shape[1]
  41	    scale = np.sqrt(feature_dim)
  42	
  43	    v = rng.normal(size=feature_dim)
  44	    v_norm = float(np.linalg.norm(v))
  45	    if v_norm <= 1e-12:
  46	        v = np.ones(feature_dim, dtype=float) / np.sqrt(feature_dim)
  47	    else:
  48	        v = v / v_norm
  49	
  50	    spectral = float("nan")
  51	    for _ in range(int(power_iters)):
  52	        dot_term = diff @ v
  53	        update = (rho * dot_term)[:, None] * phi
  54	        v_next = v + (alpha_w / scale) * update.mean(axis=0)
  55	        norm_next = float(np.linalg.norm(v_next))
  56	        if norm_next <= 1e-12:
  57	            spectral = 0.0
  58	            v = v_next
  59	            break
  60	        spectral = norm_next
  61	        v = v_next / norm_next
  62	
  63	    return {
  64	        "stability_proxy": float(spectral),
  65	        "power_iters": int(power_iters),
  66	        "batch_size": int(batch_size),
  67	    }
```

# 4. Failure Evidence (commands + logs + traceback + key metrics)
## Suite summary table (from `outputs/sanity_suite/20251227_162149/SUMMARY.md`)
```markdown
# Summary

| run | label | status | mean_rho2 | stability_proxy | fixed_point_drift | dist_mmd2 | reasons | run_dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_pi | fixed_pi | FAIL | 1.48 | 1 | 0.01721 | 0.01805 | no_nan_inf: found NaN/Inf in numeric columns | /Users/enhuili/Desktop/Learning Dynamics in Temporal Differences Reinforcement Learning with Unfixed Policy/Learning-Dynamics-in-Temporal-Differeces-RL-with-unfixed-policy-/outputs/sanity_suite/20251227_162149/fixed_pi |
| full_triad_short | full_triad_short | FAIL | 1.67 | 1 | 0.02569 | 0.03616 | no_nan_inf: found NaN/Inf in numeric columns | /Users/enhuili/Desktop/Learning Dynamics in Temporal Differences Reinforcement Learning with Unfixed Policy/Learning-Dynamics-in-Temporal-Differeces-RL-with-unfixed-policy-/outputs/sanity_suite/20251227_162149/full_triad_short |
| no_bootstrap | no_bootstrap | FAIL | 1.48 | 1 | 0.02956 | 0.01805 | no_nan_inf: found NaN/Inf in numeric columns | /Users/enhuili/Desktop/Learning Dynamics in Temporal Differences Reinforcement Learning with Unfixed Policy/Learning-Dynamics-in-Temporal-Differeces-RL-with-unfixed-policy-/outputs/sanity_suite/20251227_162149/no_bootstrap |
| on_policy | on_policy | FAIL | 1 | 1 | 0.01524 | 0.02035 | no_nan_inf: found NaN/Inf in numeric columns; on_policy_expected: on-policy expectations not met | /Users/enhuili/Desktop/Learning Dynamics in Temporal Differences Reinforcement Learning with Unfixed Policy/Learning-Dynamics-in-Temporal-Differeces-RL-with-unfixed-policy-/outputs/sanity_suite/20251227_162149/on_policy |
```

## SC1: on_policy failure evidence
- Run dir: `outputs/sanity_suite/20251227_162149/on_policy`
- Command used: `python3 tools/run_sanity_suite.py --base configs/train_sanity.yaml --out_root outputs/sanity_suite`
- Environment: os `macOS-15.3-arm64-arm-64bit-Mach-O`, python `3.14.2`, numpy `not_installed: ModuleNotFoundError`, torch `not_installed: ModuleNotFoundError`, git `a84b364b797739f6a6669e4a8308245a452d798f`
- Failure summary: status FAIL; reasons: no_nan_inf: found NaN/Inf in numeric columns; on_policy_expected: on-policy expectations not met
- Failure layer: reporting/health checks in `tdrl_unfixed_ac/reporting/run_report.py` (no_nan_inf + on_policy_expected).
- Traceback excerpt: none (run_report.exception is null).
- run_report.json: `outputs/sanity_suite/20251227_162149/on_policy/run_report.json`; run_report.md: `outputs/sanity_suite/20251227_162149/on_policy/run_report.md`
- Key metrics (last values):
```text
td_loss_last: 0.00017556309856472368
w_norm_last: 1.5482531723208302
mean_rho2_last: 1.0
tracking_gap_last: 0.0
critic_teacher_error_last: 0.02238142231481026
stability_proxy_last: 0.9999934827737278
fixed_point_drift_last: 0.015239748699403508
fixed_point_gap_last: 0.025008419192704683
dist_mmd2_last: 0.020349567022197146
dist_mean_l2_last: 0.346025674224485
```
- Health check evidence:
```json
{
  "no_nan_inf": {
    "pass": false,
    "reason": "found NaN/Inf in numeric columns",
    "observed": {
      "nan_count": 1,
      "inf_count": 0,
      "nan_cols": [
        "fixed_point_drift"
      ],
      "inf_cols": []
    },
    "applicable": true
  },
  "on_policy_expected": {
    "pass": false,
    "reason": "on-policy expectations not met",
    "observed": {
      "mean_rho2": 1.0,
      "dist_mmd2": 0.020349567022197146
    },
    "applicable": true
  },
  "no_bootstrap_expected": {
    "pass": true,
    "reason": "not applicable",
    "observed": {
      "gamma": 0.95
    },
    "applicable": false
  },
  "fixed_pi_expected": {
    "pass": true,
    "reason": "not applicable",
    "observed": {
      "alpha_pi": 0.05
    },
    "applicable": false
  }
}
```
- stdout.log excerpt:
```text
== Sanity check: on_policy ==
Base config: /Users/enhuili/Desktop/Learning Dynamics in Temporal Differences Reinforcement Learning with Unfixed Policy/Learning-Dynamics-in-Temporal-Differeces-RL-with-unfixed-policy-/configs/train_sanity.yaml
Template config: /Users/enhuili/Desktop/Learning Dynamics in Temporal Differences Reinforcement Learning with Unfixed Policy/Learning-Dynamics-in-Temporal-Differeces-RL-with-unfixed-policy-/configs/sanity/on_policy.yaml
Key hyperparameters:
  seed: 0
  outer_iters: 5
  trajectories: 2
  horizon: 50
  gamma: 0.95
  alpha_w: 0.1
  alpha_pi: 0.05
  beta: 1.0
  sigma_mu: 0.2
  sigma_pi: 0.2
  K_mc: 2
  theta_radius: 5.0
  checkpoint_every: 1
  log_every: 1
env_config_path: /Users/enhuili/Desktop/Learning Dynamics in Temporal Differences Reinforcement Learning with Unfixed Policy/Learning-Dynamics-in-Temporal-Differeces-RL-with-unfixed-policy-/configs/default.yaml
env overrides: {'feature_dim': 256, 'actor_feature_dim': 64, 'p_mix': 0.1}
probes.enabled: True
probes.every: 1
Expected: rho ~= 1 (mu=pi, sigma_mu=sigma_pi).
iter 000 | td_loss 0.0000 | teacher_err 0.0224 | gap 0.0000 | rho2 1.0000 | w_norm 1.548
iter 001 | td_loss 0.0001 | teacher_err 0.0224 | gap 0.0000 | rho2 1.0000 | w_norm 1.548
iter 002 | td_loss 0.0001 | teacher_err 0.0224 | gap 0.0000 | rho2 1.0000 | w_norm 1.548
iter 003 | td_loss 0.0000 | teacher_err 0.0224 | gap 0.0000 | rho2 1.0000 | w_norm 1.548
iter 004 | td_loss 0.0002 | teacher_err 0.0224 | gap 0.0000 | rho2 1.0000 | w_norm 1.548
Training complete. Logs at /Users/enhuili/Desktop/Learning Dynamics in Temporal Differences Reinforcement Learning with Unfixed Policy/Learning-Dynamics-in-Temporal-Differeces-RL-with-unfixed-policy-/outputs/sanity_suite/20251227_162149/on_policy/learning_curves.csv
```
- Evidence for NaN in fixed_point_drift (learning_curves.csv + fixed_point_probe.csv):
```text
iter,td_loss,critic_teacher_error,tracking_gap,mean_rho2,w_norm,fixed_point_gap,fixed_point_drift,stability_proxy,dist_mmd2,dist_mean_l2
0,3.742315070264404e-05,0.022386097718609874,0.0,1.0,1.5482847288091095,0.020518126968341065,nan,0.9999925544361736,0.06437985762221576,0.8864182630741991
1,9.133093379136516e-05,0.022385113226645742,0.0,1.0,1.5482844797364614,0.021557675175344997,0.015383231404630208,0.9999930626767822,0.03638746121735248,0.5973546651773209

iter,w_gap,w_sharp_drift,converged,num_iters,batch_size,tol
0,0.020518126968341065,nan,False,200,512,1e-06
1,0.021557675175344997,0.015383231404630208,False,200,512,1e-06
```
- Evidence for dist_mmd2 > 1e-3 (distribution_probe.csv):
```text
iter,mmd2,mmd_sigma,mean_l2,num_samples
0,0.06437985762221576,2.35935393757647,0.8864182630741991,512
1,0.03638746121735248,2.359675755551282,0.5973546651773209,512
```

## SC2: no_bootstrap failure evidence
- Run dir: `outputs/sanity_suite/20251227_162149/no_bootstrap`
- Command used: `python3 tools/run_sanity_suite.py --base configs/train_sanity.yaml --out_root outputs/sanity_suite`
- Environment: os `macOS-15.3-arm64-arm-64bit-Mach-O`, python `3.14.2`, numpy `not_installed: ModuleNotFoundError`, torch `not_installed: ModuleNotFoundError`, git `a84b364b797739f6a6669e4a8308245a452d798f`
- Failure summary: status FAIL; reasons: no_nan_inf: found NaN/Inf in numeric columns
- Failure layer: reporting/health checks in `tdrl_unfixed_ac/reporting/run_report.py` (no_nan_inf + on_policy_expected).
- Traceback excerpt: none (run_report.exception is null).
- run_report.json: `outputs/sanity_suite/20251227_162149/no_bootstrap/run_report.json`; run_report.md: `outputs/sanity_suite/20251227_162149/no_bootstrap/run_report.md`
- Key metrics (last values):
```text
td_loss_last: 0.00016474806376779515
w_norm_last: 1.5481190955806512
mean_rho2_last: 1.4800007583294021
tracking_gap_last: 8.49851299569964e-11
critic_teacher_error_last: 0.02237930953111923
stability_proxy_last: 0.9999720584644873
fixed_point_drift_last: 0.02955709482694678
fixed_point_gap_last: 0.032348436892866615
dist_mmd2_last: 0.018054714257991655
dist_mean_l2_last: 0.32049132267537256
```
- Health check evidence:
```json
{
  "no_nan_inf": {
    "pass": false,
    "reason": "found NaN/Inf in numeric columns",
    "observed": {
      "nan_count": 1,
      "inf_count": 0,
      "nan_cols": [
        "fixed_point_drift"
      ],
      "inf_cols": []
    },
    "applicable": true
  },
  "on_policy_expected": {
    "pass": true,
    "reason": "not applicable",
    "observed": {
      "mean_rho2": 1.4800007583294021,
      "dist_mmd2": 0.018054714257991655
    },
    "applicable": false
  },
  "no_bootstrap_expected": {
    "pass": true,
    "reason": "gamma=0 confirmed",
    "observed": {
      "gamma": 0.0
    },
    "applicable": true
  },
  "fixed_pi_expected": {
    "pass": true,
    "reason": "not applicable",
    "observed": {
      "alpha_pi": 0.05
    },
    "applicable": false
  }
}
```
- stdout.log excerpt:
```text
== Sanity check: no_bootstrap ==
Base config: /Users/enhuili/Desktop/Learning Dynamics in Temporal Differences Reinforcement Learning with Unfixed Policy/Learning-Dynamics-in-Temporal-Differeces-RL-with-unfixed-policy-/configs/train_sanity.yaml
Template config: /Users/enhuili/Desktop/Learning Dynamics in Temporal Differences Reinforcement Learning with Unfixed Policy/Learning-Dynamics-in-Temporal-Differeces-RL-with-unfixed-policy-/configs/sanity/no_bootstrap.yaml
Key hyperparameters:
  seed: 0
  outer_iters: 5
  trajectories: 2
  horizon: 50
  gamma: 0.0
  alpha_w: 0.1
  alpha_pi: 0.05
  beta: 0.2
  sigma_mu: 0.3
  sigma_pi: 0.2
  K_mc: 2
  theta_radius: 5.0
  checkpoint_every: 1
  log_every: 1
env_config_path: /Users/enhuili/Desktop/Learning Dynamics in Temporal Differences Reinforcement Learning with Unfixed Policy/Learning-Dynamics-in-Temporal-Differeces-RL-with-unfixed-policy-/configs/default.yaml
env overrides: {'feature_dim': 256, 'actor_feature_dim': 64, 'p_mix': 0.1}
probes.enabled: True
probes.every: 1
Expected: gamma=0 (bootstrap disabled).
iter 000 | td_loss 0.0001 | teacher_err 0.0224 | gap 0.0000 | rho2 1.7800 | w_norm 1.548
iter 001 | td_loss 0.0001 | teacher_err 0.0224 | gap 0.0000 | rho2 1.4295 | w_norm 1.548
iter 002 | td_loss 0.0001 | teacher_err 0.0224 | gap 0.0000 | rho2 1.7604 | w_norm 1.548
iter 003 | td_loss 0.0001 | teacher_err 0.0224 | gap 0.0000 | rho2 1.5848 | w_norm 1.548
iter 004 | td_loss 0.0002 | teacher_err 0.0224 | gap 0.0000 | rho2 1.4800 | w_norm 1.548
Training complete. Logs at /Users/enhuili/Desktop/Learning Dynamics in Temporal Differences Reinforcement Learning with Unfixed Policy/Learning-Dynamics-in-Temporal-Differeces-RL-with-unfixed-policy-/outputs/sanity_suite/20251227_162149/no_bootstrap/learning_curves.csv
```
- Evidence for NaN in fixed_point_drift (learning_curves.csv + fixed_point_probe.csv):
```text
iter,td_loss,critic_teacher_error,tracking_gap,mean_rho2,w_norm,fixed_point_gap,fixed_point_drift,stability_proxy,dist_mmd2,dist_mean_l2
0,6.760460593485862e-05,0.022385445633137005,2.8706581648728122e-11,1.77998782662024,1.5482436843438367,0.02906376748582679,nan,0.99997843203927,0.052785594865999985,0.7840526686956741
1,9.724655341431956e-05,0.022384144361504578,4.7188747576719035e-11,1.429471463936628,1.5482282181051938,0.0355655145914621,0.029309863337798205,0.9999757302423025,0.04284739125479142,0.668417668906293

iter,w_gap,w_sharp_drift,converged,num_iters,batch_size,tol
0,0.02906376748582679,nan,False,200,512,1e-06
1,0.0355655145914621,0.029309863337798205,False,200,512,1e-06
```

## SC3: fixed_pi failure evidence
- Run dir: `outputs/sanity_suite/20251227_162149/fixed_pi`
- Command used: `python3 tools/run_sanity_suite.py --base configs/train_sanity.yaml --out_root outputs/sanity_suite`
- Environment: os `macOS-15.3-arm64-arm-64bit-Mach-O`, python `3.14.2`, numpy `not_installed: ModuleNotFoundError`, torch `not_installed: ModuleNotFoundError`, git `a84b364b797739f6a6669e4a8308245a452d798f`
- Failure summary: status FAIL; reasons: no_nan_inf: found NaN/Inf in numeric columns
- Failure layer: reporting/health checks in `tdrl_unfixed_ac/reporting/run_report.py` (no_nan_inf + on_policy_expected).
- Traceback excerpt: none (run_report.exception is null).
- run_report.json: `outputs/sanity_suite/20251227_162149/fixed_pi/run_report.json`; run_report.md: `outputs/sanity_suite/20251227_162149/fixed_pi/run_report.md`
- Key metrics (last values):
```text
td_loss_last: 0.00015153027026911734
w_norm_last: 1.5482396714326339
mean_rho2_last: 1.480000678331054
tracking_gap_last: 2.058643256595072e-34
critic_teacher_error_last: 0.022380879533493984
stability_proxy_last: 0.9999917340584825
fixed_point_drift_last: 0.017212841740933394
fixed_point_gap_last: 0.021609671234043917
dist_mmd2_last: 0.018054739395543806
dist_mean_l2_last: 0.3204931453720982
```
- Health check evidence:
```json
{
  "no_nan_inf": {
    "pass": false,
    "reason": "found NaN/Inf in numeric columns",
    "observed": {
      "nan_count": 1,
      "inf_count": 0,
      "nan_cols": [
        "fixed_point_drift"
      ],
      "inf_cols": []
    },
    "applicable": true
  },
  "on_policy_expected": {
    "pass": true,
    "reason": "not applicable",
    "observed": {
      "mean_rho2": 1.480000678331054,
      "dist_mmd2": 0.018054739395543806
    },
    "applicable": false
  },
  "no_bootstrap_expected": {
    "pass": true,
    "reason": "not applicable",
    "observed": {
      "gamma": 0.95
    },
    "applicable": false
  },
  "fixed_pi_expected": {
    "pass": true,
    "reason": "alpha_pi=0 confirmed",
    "observed": {
      "alpha_pi": 0.0
    },
    "applicable": true
  }
}
```
- stdout.log excerpt:
```text
== Sanity check: fixed_pi ==
Base config: /Users/enhuili/Desktop/Learning Dynamics in Temporal Differences Reinforcement Learning with Unfixed Policy/Learning-Dynamics-in-Temporal-Differeces-RL-with-unfixed-policy-/configs/train_sanity.yaml
Template config: /Users/enhuili/Desktop/Learning Dynamics in Temporal Differences Reinforcement Learning with Unfixed Policy/Learning-Dynamics-in-Temporal-Differeces-RL-with-unfixed-policy-/configs/sanity/fixed_pi.yaml
Key hyperparameters:
  seed: 0
  outer_iters: 5
  trajectories: 2
  horizon: 50
  gamma: 0.95
  alpha_w: 0.1
  alpha_pi: 0.0
  beta: 0.2
  sigma_mu: 0.3
  sigma_pi: 0.2
  K_mc: 2
  theta_radius: 5.0
  checkpoint_every: 1
  log_every: 1
env_config_path: /Users/enhuili/Desktop/Learning Dynamics in Temporal Differences Reinforcement Learning with Unfixed Policy/Learning-Dynamics-in-Temporal-Differeces-RL-with-unfixed-policy-/configs/default.yaml
env overrides: {'feature_dim': 256, 'actor_feature_dim': 64, 'p_mix': 0.1}
probes.enabled: True
probes.every: 1
Expected: alpha_pi=0 (actor update disabled).
iter 000 | td_loss 0.0000 | teacher_err 0.0224 | gap 0.0000 | rho2 1.7800 | w_norm 1.548
iter 001 | td_loss 0.0001 | teacher_err 0.0224 | gap 0.0000 | rho2 1.4295 | w_norm 1.548
iter 002 | td_loss 0.0001 | teacher_err 0.0224 | gap 0.0000 | rho2 1.7604 | w_norm 1.548
iter 003 | td_loss 0.0001 | teacher_err 0.0224 | gap 0.0000 | rho2 1.5848 | w_norm 1.548
iter 004 | td_loss 0.0002 | teacher_err 0.0224 | gap 0.0000 | rho2 1.4800 | w_norm 1.548
Training complete. Logs at /Users/enhuili/Desktop/Learning Dynamics in Temporal Differences Reinforcement Learning with Unfixed Policy/Learning-Dynamics-in-Temporal-Differeces-RL-with-unfixed-policy-/outputs/sanity_suite/20251227_162149/fixed_pi/learning_curves.csv
```
- Evidence for NaN in fixed_point_drift (learning_curves.csv + fixed_point_probe.csv):
```text
iter,td_loss,critic_teacher_error,tracking_gap,mean_rho2,w_norm,fixed_point_gap,fixed_point_drift,stability_proxy,dist_mmd2,dist_mean_l2
0,3.9874858102734524e-05,0.02238589076269836,1.7045843581273991e-34,1.77998782662024,1.5482714919684342,0.024988697368814428,nan,0.9999913415688266,0.05278545895675513,0.7840519994032467
1,0.0001023249789170584,0.022384662169519145,2.058643256595072e-34,1.4294663970661148,1.548266876507248,0.02680930707890348,0.019816430091339574,0.9999904661514252,0.04284736726642224,0.6684174386189262

iter,w_gap,w_sharp_drift,converged,num_iters,batch_size,tol
0,0.024988697368814428,nan,False,200,512,1e-06
1,0.02680930707890348,0.019816430091339574,False,200,512,1e-06
```

## SC4: full_triad_short failure evidence
- Run dir: `outputs/sanity_suite/20251227_162149/full_triad_short`
- Command used: `python3 tools/run_sanity_suite.py --base configs/train_sanity.yaml --out_root outputs/sanity_suite`
- Environment: os `macOS-15.3-arm64-arm-64bit-Mach-O`, python `3.14.2`, numpy `not_installed: ModuleNotFoundError`, torch `not_installed: ModuleNotFoundError`, git `a84b364b797739f6a6669e4a8308245a452d798f`
- Failure summary: status FAIL; reasons: no_nan_inf: found NaN/Inf in numeric columns
- Failure layer: reporting/health checks in `tdrl_unfixed_ac/reporting/run_report.py` (no_nan_inf + on_policy_expected).
- Traceback excerpt: none (run_report.exception is null).
- run_report.json: `outputs/sanity_suite/20251227_162149/full_triad_short/run_report.json`; run_report.md: `outputs/sanity_suite/20251227_162149/full_triad_short/run_report.md`
- Key metrics (last values):
```text
td_loss_last: 8.23466270435367e-05
w_norm_last: 1.5482158794631429
mean_rho2_last: 1.6703710201030901
tracking_gap_last: 6.381568606009765e-11
critic_teacher_error_last: 0.022377953431216045
stability_proxy_last: 0.9999917568193094
fixed_point_drift_last: 0.02569435520355139
fixed_point_gap_last: 0.029224662734367277
dist_mmd2_last: 0.036157684334715445
dist_mean_l2_last: 0.636098673202821
```
- Health check evidence:
```json
{
  "no_nan_inf": {
    "pass": false,
    "reason": "found NaN/Inf in numeric columns",
    "observed": {
      "nan_count": 1,
      "inf_count": 0,
      "nan_cols": [
        "fixed_point_drift"
      ],
      "inf_cols": []
    },
    "applicable": true
  },
  "on_policy_expected": {
    "pass": true,
    "reason": "not applicable",
    "observed": {
      "mean_rho2": 1.6703710201030901,
      "dist_mmd2": 0.036157684334715445
    },
    "applicable": false
  },
  "no_bootstrap_expected": {
    "pass": true,
    "reason": "not applicable",
    "observed": {
      "gamma": 0.95
    },
    "applicable": false
  },
  "fixed_pi_expected": {
    "pass": true,
    "reason": "not applicable",
    "observed": {
      "alpha_pi": 0.05
    },
    "applicable": false
  }
}
```
- stdout.log excerpt:
```text
== Sanity check: full_triad_short ==
Base config: /Users/enhuili/Desktop/Learning Dynamics in Temporal Differences Reinforcement Learning with Unfixed Policy/Learning-Dynamics-in-Temporal-Differeces-RL-with-unfixed-policy-/configs/train_sanity.yaml
Template config: /Users/enhuili/Desktop/Learning Dynamics in Temporal Differences Reinforcement Learning with Unfixed Policy/Learning-Dynamics-in-Temporal-Differeces-RL-with-unfixed-policy-/configs/sanity/full_triad_short.yaml
Key hyperparameters:
  seed: 0
  outer_iters: 8
  trajectories: 2
  horizon: 50
  gamma: 0.95
  alpha_w: 0.1
  alpha_pi: 0.05
  beta: 0.2
  sigma_mu: 0.3
  sigma_pi: 0.2
  K_mc: 2
  theta_radius: 5.0
  checkpoint_every: 1
  log_every: 1
env_config_path: /Users/enhuili/Desktop/Learning Dynamics in Temporal Differences Reinforcement Learning with Unfixed Policy/Learning-Dynamics-in-Temporal-Differeces-RL-with-unfixed-policy-/configs/default.yaml
env overrides: {'feature_dim': 256, 'actor_feature_dim': 64, 'p_mix': 0.1}
probes.enabled: True
probes.every: 1
iter 000 | td_loss 0.0000 | teacher_err 0.0224 | gap 0.0000 | rho2 1.7800 | w_norm 1.548
iter 001 | td_loss 0.0001 | teacher_err 0.0224 | gap 0.0000 | rho2 1.4295 | w_norm 1.548
iter 002 | td_loss 0.0001 | teacher_err 0.0224 | gap 0.0000 | rho2 1.7604 | w_norm 1.548
iter 003 | td_loss 0.0001 | teacher_err 0.0224 | gap 0.0000 | rho2 1.5848 | w_norm 1.548
iter 004 | td_loss 0.0002 | teacher_err 0.0224 | gap 0.0000 | rho2 1.4800 | w_norm 1.548
iter 005 | td_loss 0.0000 | teacher_err 0.0224 | gap 0.0000 | rho2 1.5374 | w_norm 1.548
iter 006 | td_loss 0.0001 | teacher_err 0.0224 | gap 0.0000 | rho2 1.5813 | w_norm 1.548
iter 007 | td_loss 0.0001 | teacher_err 0.0224 | gap 0.0000 | rho2 1.6704 | w_norm 1.548
```
- Evidence for NaN in fixed_point_drift (learning_curves.csv + fixed_point_probe.csv):
```text
iter,td_loss,critic_teacher_error,tracking_gap,mean_rho2,w_norm,fixed_point_gap,fixed_point_drift,stability_proxy,dist_mmd2,dist_mean_l2
0,3.9874858102734524e-05,0.02238589076269836,1.0435084527070561e-11,1.77998782662024,1.5482714919684342,0.024988720242623413,nan,0.9999913415722261,0.0527855180525828,0.7840523726976157
1,0.00010232496298692331,0.022384662166282095,4.1057419508854366e-11,1.4294693439412458,1.548266876531246,0.026809323500101696,0.019816425872568783,0.9999904661397808,0.042847332214031475,0.6684170891378501

iter,w_gap,w_sharp_drift,converged,num_iters,batch_size,tol
0,0.024988720242623413,nan,False,200,512,1e-06
1,0.026809323500101696,0.019816425872568783,False,200,512,1e-06
```

## Related preflight evidence (not part of the four checks)
- run_dir: `outputs/preflight_train/20251227_162033`
- run_report.json: `outputs/preflight_train/20251227_162033/run_report.json`
- run_report.md: `outputs/preflight_train/20251227_162033/run_report.md`
- status: FAIL; reasons: no_nan_inf: found NaN/Inf in numeric columns
```json
{
  "no_nan_inf": {
    "pass": false,
    "reason": "found NaN/Inf in numeric columns",
    "observed": {
      "nan_count": 1,
      "inf_count": 0,
      "nan_cols": [
        "fixed_point_drift"
      ],
      "inf_cols": []
    },
    "applicable": true
  }
}
```

# 5. Minimal Repro Guide
1) Install deps: `python3 -m pip install numpy pyyaml` (torch not required for these checks).
2) Run suite: `python3 tools/run_sanity_suite.py --base configs/train_sanity.yaml --out_root outputs/sanity_suite`
3) Aggregate summary: `python3 scripts/aggregate_reports.py --root outputs/sanity_suite/<timestamp> --out outputs/sanity_suite/<timestamp>/SUMMARY.md --out-csv outputs/sanity_suite/<timestamp>/SUMMARY.csv`
4) Expected outputs per check: `learning_curves.csv`, `probes/*.csv`, `run_report.json`, `run_report.md`, `stdout.log` under `outputs/sanity_suite/<timestamp>/<check>/`.
5) Confirm same failure:
   - `run_report.json.health_checks.no_nan_inf.observed.nan_cols` includes `fixed_point_drift` for each check.
   - For on_policy, `run_report.json.health_checks.on_policy_expected.observed.dist_mmd2` > 1e-3.

# 6. Suspected Root Causes (ranked, with pointers)
1) `fixed_point_drift` is NaN on the first probe iteration, and `no_nan_inf` treats any NaN as failure.
   - Evidence: `outputs/sanity_suite/20251227_162149/*/learning_curves.csv` row 0 shows `fixed_point_drift=nan`; `outputs/sanity_suite/20251227_162149/*/probes/fixed_point_probe.csv` row 0 shows `w_sharp_drift=nan`.
   - Code pointers: `tdrl_unfixed_ac/probes/manager.py` L69-L139 (EX5) sets NaN drift when `_prev_w_sharp` is None; `tdrl_unfixed_ac/reporting/run_report.py` L249-L277 (EX6) fails on any NaN.
2) `on_policy_expected` threshold for dist_mmd2 is likely too strict for 512-sample MMD.
   - Evidence: on_policy `dist_mmd2` ~0.02035 (run_report.json, distribution_probe.csv) while `mean_rho2` ~= 1.0.
   - Code pointers: `tdrl_unfixed_ac/reporting/run_report.py` L351-L368 (EX6) uses dist_mmd2 <= 1e-3; `tdrl_unfixed_ac/probes/distribution_probe.py` L52-L66 (EX12) computes MMD over finite samples.
3) Sanity config typo for stability probe parameter name (`n_power_iters` vs `power_iters`) means overrides may be ignored.
   - Evidence: `configs/train_sanity.yaml` L20-L25 uses `n_power_iters`; `tdrl_unfixed_ac/probes/manager.py` expects `power_iters` (EX5).

# 7. Impact Assessment (does it invalidate data? what must be fixed before long runs?)
- Current FAIL status in the sanity suite is dominated by a bookkeeping issue (NaN drift on first probe). This makes PASS/FAIL unreliable until either NaN handling is relaxed or the drift is defined for iter 0.
- The on_policy check failing dist_mmd2 suggests a distribution mismatch or a too-strict threshold; if the mismatch is real, off-policy corrections may be unreliable and long runs could be biased.
- Before long runs, fix no_nan_inf vs probe NaNs and recalibrate on_policy_expected (threshold or probe design). The environment/feature pipeline itself shows no runtime exceptions in these runs.
