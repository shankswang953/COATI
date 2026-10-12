"""Compare Gaussian endpoint transport with synchronization enabled/disabled.

The known map supplies secondary geometry. Both flows share initial weights,
observations, architecture and training budget; no intermediate path is supplied.
"""
import argparse
from pathlib import Path
import uuid
import numpy as np
import torch
import matplotlib.pyplot as plt
import coati as ct
from coati.toy_comparison import (
    make_shared_initial_checkpoint, plot_gaussian_sync,
    plot_bump_height, compare_gaussian_runs,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="A fresh comparison directory")
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--secondary-weight", type=float, default=.7)
    args = parser.parse_args()
    SEED = args.seed
    N_CELLS = 128
    ITERATIONS = args.iterations
    SECONDARY_WEIGHT = args.secondary_weight
    HEIGHT, WIDTH, CENTER = 1., .08, (.5, .5)
    data = ct.datasets.gaussian_sync(n=N_CELLS, seed=SEED, height=HEIGHT, width=WIDTH, center=CENTER)
    T = ct.datasets.gaussian_bump_map(height=HEIGHT, width=WIDTH, center=CENTER)
    print("Observed times:", data.times)
    print("Primary / secondary:", data.primary[0].shape, data.secondary[0].shape)

    base = ct.Config(
        multimodal=False, balanced=True, iterations=ITERATIONS,
        batch_size=64, learning_rate=.003, hidden_dim=64, n_layers=3,
        activation="tanh", matching_weight=300., energy_weight=1.,
        manifold_weight=0., dt=.05, seed=SEED,
    )
    configs = {
        "Without sync": base,
        "With sync": base.with_parameters(multimodal=True, secondary_weight=SECONDARY_WEIGHT),
    }

    run_dir = Path(args.output)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError("Choose a fresh comparison directory.")
    initial_checkpoint = make_shared_initial_checkpoint(base, run_dir / "initial.pt")
    results = {}
    for name, config in configs.items():
        results[name] = ct.train_vector_field(
            data, config, output_dir=run_dir / ("with_sync" if config.multimodal else "without_sync"),
            T_model=T if config.multimodal else None,
            warm_start_path=initial_checkpoint,
        )
        print(name, "trained for", ITERATIONS, "iterations")

    predictions = {}
    for name, result in results.items():
        prediction = result.predict_trajectory(times=np.linspace(0., 1., 41))
        with torch.no_grad():
            prediction["secondary"] = T(torch.tensor(prediction["primary"])).numpy()
        assert np.isfinite(prediction["primary"]).all()
        np.testing.assert_array_equal(prediction["secondary"][..., :2], prediction["primary"])
        predictions[name] = prediction

    figure = plot_gaussian_sync(data, T, results, predictions)
    figure.savefig(run_dir / "comparison.png", dpi=180, bbox_inches="tight")
    figure.savefig(run_dir / "comparison.pdf", bbox_inches="tight")
    figure = plot_bump_height(predictions)
    figure.savefig(run_dir / "bump_height.png", dpi=180, bbox_inches="tight")
    scores = compare_gaussian_runs(results, predictions)
    scores.to_csv(run_dir / "comparison.csv")
    print(scores.round(5).to_string())
    print("Results:", run_dir.resolve())


if __name__ == "__main__":
    main()
