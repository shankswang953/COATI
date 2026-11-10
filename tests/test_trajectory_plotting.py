import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from coati.downstream import plot_trajectory_embedding


class FirstTwoCoordinates:
    def transform(self, values):
        return np.asarray(values)[:, :2]


def example_inputs():
    rng = np.random.default_rng(7)
    reference = rng.normal(size=(20, 3))
    trajectory = np.stack([reference[:8] + step for step in (0.0, 0.1, 0.2, 0.3)])
    labels = np.array(["A"] * 10 + ["B"] * 10)
    return reference, trajectory, labels


def test_global_plot_transforms_and_records_coordinates():
    reference, trajectory, _ = example_inputs()
    ax = plot_trajectory_embedding(
        trajectory,
        reference=reference,
        transform=FirstTwoCoordinates(),
        times=[0.0, 1.0, 2.0, 3.0],
        max_paths=5,
        random_state=3,
    )
    assert ax.coati_trajectory_embedding.shape == (4, 8, 2)
    assert len(ax.coati_path_indices) == 5
    assert len(ax.collections) >= 4
    plt.close(ax.figure)


def test_local_plot_combines_label_and_explicit_selections():
    reference, trajectory, labels = example_inputs()
    ax = plot_trajectory_embedding(
        trajectory[:, :, :2],
        reference_embedding=reference[:, :2],
        labels=labels,
        focus_labels="A",
        reference_subset=np.arange(12),
        trajectory_subset=[1, 3, 5],
        view="local",
        path_color="black",
    )
    assert np.array_equal(ax.coati_path_indices, [1, 3, 5])
    assert len(ax.figure.axes) == 1
    plt.close(ax.figure)


def test_high_dimensional_input_requires_existing_transform():
    _, trajectory, _ = example_inputs()
    with pytest.raises(ValueError, match="existing embedding transform"):
        plot_trajectory_embedding(trajectory)
