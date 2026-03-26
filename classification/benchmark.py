"""Train each classification model over N seeds and report average Brier score and ECE."""

import argparse

import jax
from tqdm import tqdm
import jax.random as jr
import equinox as eqx
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray

from utils.models import MLP, Ensemble, VariationalBNN

from utils.config import Config
from utils.data import gen_data
from utils.train import (
    train_mlp,
    train_ensemble,
    train_repulsive_ensemble,
    train_mfvi,
    run_hmc,
    predict_hmc,
)
from utils.diagnostics import brier_score, expected_calibration_error

cfg = Config()


def evaluate_seed(seed: int) -> dict[str, dict[str, float]]:
    """Train all models with the given seed and return Brier/ECE on a held-out test set."""
    key = jr.PRNGKey(seed)

    data = gen_data(cfg.n_obs, cfg.noise, seed)
    mean, std = data["X"].mean(0), data["X"].std(0)

    test_data = gen_data(n_obs=1000, noise=cfg.noise, seed=seed + 10_000)
    X_test = test_data["X"]
    y_test = test_data["label"]
    X_test_norm = (X_test - mean) / (std + 1e-6)

    results = {}

    # --- Deterministic MLP ---
    key, subkey = jr.split(key)
    det_net = train_mlp(data, cfg.n_steps, subkey, cfg)

    @eqx.filter_jit
    def predict_det(model: MLP, X: Float[Array, "n 2"]) -> Float[Array, "n"]:
        return jax.nn.sigmoid(jax.vmap(model)(X))

    p = predict_det(det_net, X_test_norm)
    results["Deterministic"] = {
        "brier": float(brier_score(p, y_test)),
        "ece": float(expected_calibration_error(p, y_test, n_bins=20)),
    }

    # --- Ensemble ---
    key, subkey = jr.split(key)
    ens = train_ensemble(data, cfg.n_steps, subkey, cfg, n_member=cfg.n_member)

    @eqx.filter_jit
    def predict_ens(model: Ensemble, X: Float[Array, "n 2"]) -> Float[Array, "n"]:
        return jax.nn.sigmoid(jax.vmap(model)(X)).mean(1)

    p = predict_ens(ens, X_test_norm)
    results["Ensemble"] = {
        "brier": float(brier_score(p, y_test)),
        "ece": float(expected_calibration_error(p, y_test, n_bins=20)),
    }

    # --- Repulsive Ensemble ---
    key, subkey = jr.split(key)
    rep_ens = train_repulsive_ensemble(
        data, cfg.n_steps, subkey, cfg, n_member=cfg.n_member
    )
    p = predict_ens(rep_ens, X_test_norm)
    results["Rep. Ensemble"] = {
        "brier": float(brier_score(p, y_test)),
        "ece": float(expected_calibration_error(p, y_test, n_bins=20)),
    }

    # --- Mean-field VI ---
    key, subkey = jr.split(key)
    vi_model = train_mfvi(data, 2 * cfg.n_steps, subkey, cfg)
    vi_keys = jr.split(jr.PRNGKey(seed + 20_000), 500)

    @eqx.filter_jit
    def predict_vi(
        model: VariationalBNN, X: Float[Array, "n 2"], keys: PRNGKeyArray
    ) -> Float[Array, "n"]:
        return jax.nn.sigmoid(
            jax.vmap(lambda x: jax.vmap(lambda k: model(x, key=k))(keys))(X)
        ).mean(1)

    p = predict_vi(vi_model, X_test_norm, vi_keys)
    results["Mean-field VI"] = {
        "brier": float(brier_score(p, y_test)),
        "ece": float(expected_calibration_error(p, y_test, n_bins=20)),
    }

    # --- HMC ---
    key, subkey = jr.split(key)
    hmc_samples, hmc_static = run_hmc(data, subkey, cfg)
    p = predict_hmc(X_test, hmc_samples, hmc_static).mean(0)
    results["HMC"] = {
        "brier": float(brier_score(p, y_test)),
        "ece": float(expected_calibration_error(p, y_test, n_bins=20)),
    }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark classification models over multiple seeds."
    )
    parser.add_argument("--n_seeds", type=int, default=5, help="Number of random seeds")
    parser.add_argument("--start_seed", type=int, default=0, help="Starting seed value")
    parser.add_argument(
        "--n_member", type=int, default=cfg.n_member, help="Number of ensemble members"
    )
    args = parser.parse_args()

    cfg.n_member = args.n_member

    seeds = list(range(args.start_seed, args.start_seed + args.n_seeds))
    method_names = [
        "Deterministic",
        "Ensemble",
        "Rep. Ensemble",
        "Mean-field VI",
        "HMC",
    ]

    all_results = {name: {"brier": [], "ece": []} for name in method_names}

    for seed in tqdm(seeds, desc="Seeds"):
        res = evaluate_seed(seed)
        for name in method_names:
            all_results[name]["brier"].append(res[name]["brier"])
            all_results[name]["ece"].append(res[name]["ece"])

    print(f"\n{'='*60}")
    print(
        f"Results over {args.n_seeds} seeds ({args.start_seed}..{args.start_seed + args.n_seeds - 1})"
    )
    print(f"{'='*60}")
    print(f"{'Method':20s}  {'Brier':>14s}  {'ECE':>14s}")
    print("-" * 54)
    for name in method_names:
        bs = np.array(all_results[name]["brier"])
        ece = np.array(all_results[name]["ece"])
        print(
            f"{name:20s}  {bs.mean():.4f} ± {bs.std():.4f}  {ece.mean():.4f} ± {ece.std():.4f}"
        )


if __name__ == "__main__":
    main()
