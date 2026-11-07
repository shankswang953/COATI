"""Run the original GaussToy BalancedSync configuration or replay its checkpoint."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import coati as ct
from coati.gaussian_reference import load_example, replay_reference, plot_reference, _asset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="A new result directory")
    parser.add_argument("--replay", action="store_true", help="Replay original weights without fitting")
    parser.add_argument("--iterations", type=int, help="Optional short smoke run; omit for original settings")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Choose a fresh result directory.")
    if args.replay and args.iterations is not None:
        parser.error("--iterations is only used when fitting")
    example = load_example()
    if args.replay:
        example, prediction, errors = replay_reference()
        args.output.mkdir(parents=True, exist_ok=True)
        print("Checkpoint replay maximum errors:", json.dumps(errors))
    else:
        config = example.config if args.iterations is None else example.config.with_parameters(iterations=args.iterations)
        result = ct.train_vector_field(example.data, config, output_dir=args.output,
                                       T_model=example.mapping)
        prediction = result.predict_trajectory(times=example.times, method=example.settings['prediction']['method'])
        with torch.no_grad():
            prediction['secondary'] = example.mapping(torch.tensor(prediction['primary'])).numpy()
    assert all(np.isfinite(value).all() for value in prediction.values())
    for space in ('primary', 'secondary'):
        torch.save(torch.tensor(prediction[space]), args.output / (space+'_trajectory.pt'))
    figure = plot_reference(example, prediction)
    figure.savefig(args.output/'trajectory_primary_secondary.png', bbox_inches='tight', facecolor='white')
    figure.savefig(args.output/'trajectory_primary_secondary.pdf', bbox_inches='tight', facecolor='white')
    plt.close(figure)
    (args.output/'archived_original_figure.png').write_bytes(_asset('reference_figure.png').read_bytes())
    (args.output/'reference_settings.json').write_text(json.dumps(example.settings,indent=2)+'\n')
    print('Figure:',args.output/'trajectory_primary_secondary.png')


if __name__=='__main__':
    main()
