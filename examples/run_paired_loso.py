"""Train the map and synchronized flow without the middle snapshot, then query it."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")

import argparse
import json
import numpy as np
import coati as ct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--map-type", choices=["film", "mlp"], default="film")
    parser.add_argument("--map-epochs", type=int, default=300)
    parser.add_argument("--niters", type=int, default=200)
    args = parser.parse_args()
    observed = ct.datasets.paired(n=128, seed=0)
    training = observed.without_times([.5])
    result = ct.fit(training, ct.Config(niters=args.niters, sync_loss=True, sync_weight=.35, map_type=args.map_type),
                    map_config=ct.MapConfig(epochs=args.map_epochs), output_dir=args.output)
    prediction = result.predict(times=[0., .5, 1.])
    secondary = result.project_secondary(prediction)
    np.savez(result.output_dir / "heldout_prediction.npz", times=prediction["times"],
             primary=prediction["primary"], secondary=secondary)
    restored = ct.load_result(result.output_dir)
    np.testing.assert_array_equal(secondary, restored.project_secondary(restored.predict(times=[0., .5, 1.])))
    print(json.dumps(result.map_report, indent=2))
    print(f"Map report: {result.output_dir / 'map/report.txt'}")
    print("The omitted stage was used only as a prediction time, not to fit either model.")


if __name__ == "__main__":
    main()
