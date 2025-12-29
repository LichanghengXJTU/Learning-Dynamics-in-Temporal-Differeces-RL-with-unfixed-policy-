#!/usr/bin/env python3
"""Run contract unit tests for the unfixed actor-critic implementation."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tdrl_unfixed_ac.algos.train_unfixed_ac import train_unfixed_ac
from tdrl_unfixed_ac.algos.unfixed_ac import (
    LinearGaussianPolicy,
    batch_step_scale,
    critic_value,
    importance_ratio,
)


class Node:
    """Minimal reverse-mode autodiff node for numpy arrays."""

    def __init__(
        self,
        value: np.ndarray,
        parents: Optional[List[Tuple["Node", Any]]] = None,
    ) -> None:
        self.value = np.asarray(value, dtype=float)
        self.parents = parents or []
        self.grad: Optional[np.ndarray] = None

    def __add__(self, other: Any) -> "Node":
        other_val, other_node = _to_value(other)
        out = Node(self.value + other_val)
        parents = [(self, lambda g: g)]
        if other_node is not None:
            parents.append((other_node, lambda g: g))
        out.parents = parents
        return out

    def __radd__(self, other: Any) -> "Node":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "Node":
        other_val, other_node = _to_value(other)
        out = Node(self.value - other_val)
        parents = [(self, lambda g: g)]
        if other_node is not None:
            parents.append((other_node, lambda g: -g))
        out.parents = parents
        return out

    def __rsub__(self, other: Any) -> "Node":
        other_val, other_node = _to_value(other)
        out = Node(other_val - self.value)
        parents = [(self, lambda g: -g)]
        if other_node is not None:
            parents.append((other_node, lambda g: g))
        out.parents = parents
        return out

    def __mul__(self, other: Any) -> "Node":
        other_val, other_node = _to_value(other)
        out = Node(self.value * other_val)
        parents = [(self, lambda g: g * other_val)]
        if other_node is not None:
            parents.append((other_node, lambda g: g * self.value))
        out.parents = parents
        return out

    def __rmul__(self, other: Any) -> "Node":
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> "Node":
        other_val, other_node = _to_value(other)
        out = Node(self.value / other_val)
        parents = [(self, lambda g: g / other_val)]
        if other_node is not None:
            parents.append((other_node, lambda g: -g * self.value / (other_val * other_val)))
        out.parents = parents
        return out

    def __rtruediv__(self, other: Any) -> "Node":
        other_val, other_node = _to_value(other)
        out = Node(other_val / self.value)
        parents = [(self, lambda g: -g * other_val / (self.value * self.value))]
        if other_node is not None:
            parents.append((other_node, lambda g: g / self.value))
        out.parents = parents
        return out

    def __matmul__(self, other: Any) -> "Node":
        other_val, other_node = _to_value(other)
        if other_node is not None:
            raise ValueError("Node @ Node not supported in this minimal autodiff.")
        out = Node(self.value @ other_val)

        def _grad_matmul(g: np.ndarray) -> np.ndarray:
            g = np.asarray(g, dtype=float)
            return np.outer(g, other_val)

        out.parents = [(self, _grad_matmul)]
        return out

    @property
    def T(self) -> "Node":
        out = Node(self.value.T)
        out.parents = [(self, lambda g: np.asarray(g, dtype=float).T)]
        return out

    def sum(self) -> "Node":
        out = Node(np.array(self.value.sum(), dtype=float))
        out.parents = [(self, lambda g: np.ones_like(self.value) * g)]
        return out

    def backward(self, grad: Optional[np.ndarray] = None) -> None:
        topo: List[Node] = []
        visited: set[int] = set()

        def _build(node: Node) -> None:
            node_id = id(node)
            if node_id in visited:
                return
            visited.add(node_id)
            for parent, _ in node.parents:
                _build(parent)
            topo.append(node)

        _build(self)
        for node in topo:
            node.grad = np.zeros_like(node.value, dtype=float)

        if grad is None:
            self.grad = np.ones_like(self.value, dtype=float)
        else:
            self.grad = np.asarray(grad, dtype=float)

        for node in reversed(topo):
            if node.grad is None:
                continue
            for parent, grad_fn in node.parents:
                parent.grad = parent.grad + grad_fn(node.grad)


def _to_value(obj: Any) -> Tuple[np.ndarray, Optional[Node]]:
    if isinstance(obj, Node):
        return obj.value, obj
    return np.asarray(obj, dtype=float), None


def _autograd_log_prob_grad(
    theta: np.ndarray,
    psi: np.ndarray,
    action: np.ndarray,
    sigma: float,
) -> Tuple[float, np.ndarray]:
    theta_node = Node(theta)
    action_node = Node(action)
    mean = (theta_node.T @ psi) / math.sqrt(theta.shape[0])
    diff = action_node - mean
    sq = (diff * diff).sum()
    var = float(sigma) ** 2
    log_norm = -0.5 * action.size * math.log(2.0 * math.pi * var)
    log_prob = log_norm - 0.5 * sq / var
    log_prob.backward()
    log_prob_val = float(np.asarray(log_prob.value, dtype=float).reshape(()))
    return log_prob_val, np.asarray(theta_node.grad, dtype=float)


def _write_report(path: Path, results: List[Dict[str, Any]]) -> None:
    lines = ["# Contract Tests Report", ""]
    lines.append("| test | status | error | threshold | details |")
    lines.append("| --- | --- | --- | --- | --- |")
    for result in results:
        status = "PASS" if result["pass"] else "FAIL"
        error = _fmt(result.get("error"))
        threshold = _fmt(result.get("threshold"))
        details = result.get("details", "")
        lines.append(f"| {result['name']} | {status} | {error} | {threshold} | {details} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, (float, np.floating)):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return f"{value:.4g}"
    return str(value)


def test_score_autograd(seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    actor_dim = 5
    action_dim = 3
    theta = rng.normal(size=(actor_dim, action_dim))
    psi = rng.normal(size=(actor_dim,))
    action = rng.normal(size=(action_dim,))
    sigma = 0.7

    policy = LinearGaussianPolicy(theta=theta, sigma=sigma, v_max=1.0, squash_action=False)
    mean = policy.mean(psi)
    g_analytic = np.outer(psi, action - mean) / (sigma * sigma * math.sqrt(actor_dim))

    _, g_auto = _autograd_log_prob_grad(theta, psi, action, sigma)
    err = float(np.max(np.abs(g_analytic - g_auto)))
    threshold = 1e-6
    return {
        "name": "T1_score_autograd",
        "pass": err < threshold,
        "error": err,
        "threshold": threshold,
        "details": f"actor_dim={actor_dim}, action_dim={action_dim}",
    }


def test_rho_ratio(seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed + 11)
    actor_dim = 4
    action_dim = 2
    theta_mu = rng.normal(size=(actor_dim, action_dim))
    theta_pi = rng.normal(size=(actor_dim, action_dim))
    psi = rng.normal(size=(actor_dim,))
    action = rng.normal(size=(action_dim,))
    sigma_mu = 0.5
    sigma_pi = 0.3

    mu_policy = LinearGaussianPolicy(theta=theta_mu, sigma=sigma_mu, v_max=1.0, squash_action=False)
    pi_policy = LinearGaussianPolicy(theta=theta_pi, sigma=sigma_pi, v_max=1.0, squash_action=False)
    logp_pi = pi_policy.log_prob(action, psi)
    logp_mu = mu_policy.log_prob(action, psi)
    rho_analytic = math.exp(logp_pi - logp_mu)
    rho_code = importance_ratio(logp_pi, logp_mu)
    rel_err = abs(rho_code - rho_analytic) / max(1.0, abs(rho_analytic))
    threshold = 1e-6
    return {
        "name": "T2_rho_ratio",
        "pass": rel_err < threshold,
        "error": rel_err,
        "threshold": threshold,
        "details": f"rho={rho_code:.4g}, log_rho={(logp_pi - logp_mu):.4g}",
    }


def test_td_error(seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed + 23)
    feature_dim = 7
    w = rng.normal(size=(feature_dim,))
    w_r = rng.normal(size=(feature_dim,))
    phi = rng.normal(size=(feature_dim,))
    bar_phi = rng.normal(size=(feature_dim,))
    gamma = 0.9

    reward = float(np.dot(w_r, phi) / math.sqrt(feature_dim))
    delta_code = reward + gamma * critic_value(w, bar_phi) - critic_value(w, phi)
    delta_manual = float((np.dot(w_r - w, phi) + gamma * np.dot(w, bar_phi)) / math.sqrt(feature_dim))
    err = float(abs(delta_code - delta_manual))
    threshold = 1e-7
    return {
        "name": "T3_td_error",
        "pass": err < threshold,
        "error": err,
        "threshold": threshold,
        "details": f"delta_code={delta_code:.4g}, delta_manual={delta_manual:.4g}",
    }


def test_critic_update_scaling(seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed + 37)
    b_val = 4
    t_val = 3
    feature_dim = 6
    w = rng.normal(size=(feature_dim,))
    phi = rng.normal(size=(b_val * t_val, feature_dim))
    rho = rng.normal(size=(b_val * t_val,))
    delta = rng.normal(size=(b_val * t_val,))
    alpha_w = 0.2

    grad = (rho * delta)[:, None] * phi
    grad_sum = np.sum(grad, axis=0)
    manual_scale = 1.0 / (math.sqrt(b_val) * t_val)
    code_scale = batch_step_scale(b_val, t_val)
    w_next_manual = w + alpha_w * manual_scale * grad_sum
    w_next_code = w + alpha_w * code_scale * grad_sum
    err = float(np.max(np.abs(w_next_manual - w_next_code)))
    threshold = 1e-7

    scale_b1 = batch_step_scale(1, t_val)
    scale_b4 = batch_step_scale(4, t_val)
    ratio = scale_b4 / scale_b1 if scale_b1 > 0 else float("nan")
    ratio_err = float(abs(ratio - 0.5))
    ratio_ok = ratio_err < 1e-12

    return {
        "name": "T4_critic_update_scaling",
        "pass": err < threshold and ratio_ok,
        "error": err,
        "threshold": threshold,
        "details": f"scale_ratio={ratio:.4g}, ratio_err={ratio_err:.4g}",
    }


def test_semi_gradient() -> Dict[str, Any]:
    src = inspect.getsource(train_unfixed_ac)
    uses_autograd = ("torch" in src) or ("jax" in src)
    passed = not uses_autograd
    return {
        "name": "T5_semi_gradient",
        "pass": passed,
        "error": 0.0 if passed else 1.0,
        "threshold": 0.0,
        "details": "numpy update; no autograd frameworks imported" if passed else "autograd detected in training loop",
    }


def run(base_dir: Path, seed: int) -> int:
    tests_dir = base_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    log_path = tests_dir / "contract_tests.log"
    results_path = tests_dir / "contract_tests_results.json"
    report_path = tests_dir / "contract_tests_report.md"

    results = [
        test_score_autograd(seed),
        test_rho_ratio(seed),
        test_td_error(seed),
        test_critic_update_scaling(seed),
        test_semi_gradient(),
    ]

    with log_path.open("w", encoding="utf-8") as handle:
        for result in results:
            status = "PASS" if result["pass"] else "FAIL"
            handle.write(f"{result['name']}: {status} (error={_fmt(result.get('error'))})\n")

    results_payload = {
        "seed": seed,
        "tests": results,
        "summary": {
            "passed": all(r["pass"] for r in results),
            "num_tests": len(results),
        },
    }
    results_path.write_text(json.dumps(results_payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    _write_report(report_path, results)
    return 0 if results_payload["summary"]["passed"] else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Run contract unit tests.")
    parser.add_argument("--base-dir", type=str, required=True, help="Base output dir (outputs/base_check/<TS>).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for deterministic tests.")
    args = parser.parse_args()
    base_dir = Path(args.base_dir)
    raise SystemExit(run(base_dir, args.seed))


if __name__ == "__main__":
    main()
