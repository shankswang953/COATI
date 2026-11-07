"""Compare Gaussian transport with and without secondary synchronization.

Default geometry and training settings are shared with the tutorial through
coati's packaged Gaussian example configuration.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import coati as ct
from coati.toy_comparison import (
    gaussian_sync_example, make_shared_initial_checkpoint,
    plot_gaussian_sync, plot_bump_height, compare_gaussian_runs,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="A fresh comparison directory")
    parser.add_argument("--iterations", type=int, help="Refinement iterations per model")
    parser.add_argument("--warmup-iterations", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--secondary-weight", type=float)
    parser.add_argument("--mass-weight", type=float, help="Optional unbalanced mass penalty override")
    parser.add_argument("--unbalanced", action="store_true")
    args = parser.parse_args()
    example = gaussian_sync_example(
        balanced=not args.unbalanced, iterations=args.iterations,
        warmup_iterations=args.warmup_iterations, seed=args.seed,
        secondary_weight=args.secondary_weight,
    )
    if args.mass_weight is not None:
        example.warmup_config = example.warmup_config.with_parameters(mass_weight=args.mass_weight)
        example.configs = {name: config.with_parameters(mass_weight=args.mass_weight)
                           for name, config in example.configs.items()}
    data, T = example.data, example.mapping
    run_dir = Path(args.output)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError("Choose a fresh comparison directory.")
    initial = make_shared_initial_checkpoint(example.warmup_config, run_dir / "initial.pt")
    warmup = ct.train_vector_field(data, example.warmup_config,
                                  output_dir=run_dir / "common_warmup", warm_start_path=initial)
    common = warmup.output_dir / "checkpoints" / "path_param.pth"
    results, predictions = {}, {}
    for name, config in example.configs.items():
        result = ct.train_vector_field(
            data, config, output_dir=run_dir / ("with_sync" if config.multimodal else "without_sync"),
            T_model=T if config.multimodal else None, warm_start_path=common,
        )
        prediction = result.predict_trajectory(times=example.times)
        with torch.no_grad():
            prediction["secondary"] = T(torch.tensor(prediction["primary"])).numpy()
        assert all(np.isfinite(value).all() for value in prediction.values())
        np.testing.assert_array_equal(prediction["secondary"][..., :2], prediction["primary"])
        results[name], predictions[name] = result, prediction
        print(name, "refined after the common warm-up")
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
