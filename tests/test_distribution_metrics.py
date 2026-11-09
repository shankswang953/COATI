import numpy as np
import pytest

from coati.downstream.distribution_metrics import (
    composition_similarity,
    evaluate_distribution,
    normalize_particle_weights,
    sinkhorn_w2,
    support_recall,
)


def _two_clusters(seed=4):
    rng = np.random.default_rng(seed)
    left = rng.normal((-1.0, 0.0), 0.08, size=(30, 2))
    right = rng.normal((1.0, 0.0), 0.08, size=(30, 2))
    points = np.vstack([left, right])
    labels = np.asarray(["left"] * 30 + ["right"] * 30)
    return points, labels


def test_log_weight_normalization_is_shift_invariant():
    log_weights = np.asarray([-3.0, -1.0, 2.0])
    expected = normalize_particle_weights(3, log_weights=log_weights)
    actual = normalize_particle_weights(3, log_weights=log_weights + 1000.0)
    np.testing.assert_allclose(actual, expected)
    assert actual.sum() == pytest.approx(1.0)


def test_identical_distribution_has_zero_w2_and_perfect_composition():
    points, labels = _two_clusters()
    score, predicted_unknown, observed_unknown = composition_similarity(
        points, points, labels
    )
    assert score == pytest.approx(1.0)
    assert predicted_unknown == 0.0
    assert observed_unknown == 0.0
    assert sinkhorn_w2(points, points, max_clusters=20) == 0.0


def test_missing_cluster_reduces_support_and_composition():
    reference, labels = _two_clusters()
    complete = evaluate_distribution(
        reference, reference, labels, support_k=5, max_clusters=20
    )
    missing = evaluate_distribution(
        reference[:30], reference, labels, support_k=5, max_clusters=20
    )
    assert complete.support_recall > 0.95
    assert missing.support_recall < complete.support_recall
    assert missing.composition_similarity < complete.composition_similarity
    assert missing.sinkhorn_w2 > complete.sinkhorn_w2


def test_support_recall_uses_effective_particle_count():
    reference, _ = _two_clusters()
    prediction = reference[::3]
    uniform = support_recall(prediction, reference, k=5)
    concentrated = np.zeros(len(prediction))
    concentrated[0] = 1.0
    concentrated_score = support_recall(
        prediction, reference, weights=concentrated, k=5
    )
    assert concentrated_score < uniform


def test_unknown_is_excluded_before_composition_jsd():
    reference = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    labels = np.asarray(["A", "A", "B", "unknown"])
    score, predicted_unknown, observed_unknown = composition_similarity(
        reference[:3], reference, labels, unknown_label="Unknown"
    )
    assert score == pytest.approx(1.0)
    assert predicted_unknown == 0.0
    assert observed_unknown == pytest.approx(0.25)


def test_invalid_weight_arguments_are_rejected():
    with pytest.raises(ValueError, match="not both"):
        normalize_particle_weights(2, weights=np.ones(2), log_weights=np.zeros(2))
