import hashlib
import json
import numpy as np
from synthetic.gaussian_reference import load_example, replay_reference, _asset


def test_original_artifacts_are_byte_identical_and_observations_are_not_resampled():
    manifest = json.loads(_asset('manifest.json').read_text())
    for name, entry in manifest.items():
        assert hashlib.sha256(_asset(name).read_bytes()).hexdigest() == entry['sha256']
    example = load_example()
    assert [len(x) for x in example.data.primary] == [1000, 2000]
    assert example.data.labels == ['time0', 'time1']
    with _asset('primary.npz').open('rb') as stream, np.load(stream, allow_pickle=False) as saved:
        for actual, key in zip(example.data.primary, ['time0', 'time1']):
            np.testing.assert_array_equal(actual, saved[key])
    with _asset('secondary.npz').open('rb') as stream, np.load(stream, allow_pickle=False) as saved:
        for actual, key in zip(example.data.secondary, ['time0', 'time1']):
            np.testing.assert_array_equal(actual, saved[key])
            np.testing.assert_array_equal(actual[:, 2], 0.)
    assert example.config.balanced and example.config.multimodal
    assert not any(parameter.requires_grad for parameter in example.mapping.parameters())


def test_original_checkpoint_replays_the_archived_trajectory_in_coati():
    example, prediction, error = replay_reference()
    assert prediction['primary'].shape == (51, 1000, 2)
    assert prediction['secondary'].shape == (51, 1000, 3)
    assert max(error.values()) < 1e-5
    np.testing.assert_array_equal(prediction['primary'][0], example.data.primary[0])
