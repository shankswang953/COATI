import json
import pytest
from coati import Config


def test_old_disabled_mmd_config_loads(tmp_path):
    legacy = Config(niters=3).to_dict()
    legacy.update(mmd_loss=False, sync_mmd_loss=False)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(legacy))
    current = Config.load(path).to_dict()
    assert current == Config(niters=3).to_dict()
    assert "mmd_loss" not in current and "sync_mmd_loss" not in current


@pytest.mark.parametrize("flag", ["mmd_loss", "sync_mmd_loss"])
@pytest.mark.parametrize("via_extra", [False, True])
def test_removed_matching_cannot_be_silently_enabled(flag, via_extra):
    kwargs = {"extra": {flag: True}} if via_extra else {flag: True}
    with pytest.raises(ValueError, match="removed.*W2"):
        Config(**kwargs)
