import json
from pathlib import Path

import pytest

from pytutor.config import (
    DEFAULT_SETTINGS,
    SettingsError,
    load_settings,
    prepare_user_space,
    pytutor_home,
)


@pytest.fixture
def tmp_pytutor_home(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTUTOR_HOME", str(tmp_path))
    monkeypatch.setattr("pytutor.config.HOME_DIR", tmp_path)
    monkeypatch.setattr("pytutor.config.DATA_DIR", tmp_path / "data")
    monkeypatch.setattr("pytutor.config.STORAGE_DIR", tmp_path / "storage")
    monkeypatch.setattr("pytutor.config.SETTINGS_PATH", tmp_path / "settings.json")
    return tmp_path


@pytest.mark.parametrize(
    "env_value, expected",
    [
        pytest.param(None, Path.home() / ".pytutor", id="unset-uses-default"),
        pytest.param("/custom/path", Path("/custom/path"), id="absolute-path"),
        pytest.param("relative", Path("relative"), id="relative-path"),
    ],
)
def test_pytutor_home(env_value, monkeypatch, expected):
    if env_value is None:
        monkeypatch.delenv("PYTUTOR_HOME", raising=False)
    else:
        monkeypatch.setenv("PYTUTOR_HOME", env_value)
    assert pytutor_home() == expected


def test_load_settings_returns_defaults_when_no_file(tmp_pytutor_home):
    assert load_settings() == DEFAULT_SETTINGS


@pytest.mark.parametrize(
    "overrides, expected_keys",
    [
        pytest.param(
            {"chat_model": "custom-model"},
            {"chat_model": "custom-model"},
            id="single-override",
        ),
        pytest.param(
            {"chat_model": "m", "top_k": 1},
            {"chat_model": "m", "top_k": 1},
            id="multiple-overrides",
        ),
        pytest.param(
            {},
            {},
            id="empty-overrides-keeps-defaults",
        ),
    ],
)
def test_load_settings_partial_overrides(tmp_pytutor_home, overrides, expected_keys):
    settings_path = tmp_pytutor_home / "settings.json"
    settings_path.write_text(json.dumps(overrides))
    result = load_settings()
    for k, v in expected_keys.items():
        assert result[k] == v
    for k in DEFAULT_SETTINGS:
        if k not in expected_keys:
            assert result[k] == DEFAULT_SETTINGS[k]


def test_load_settings_extra_keys_preserved(tmp_pytutor_home):
    custom = {**DEFAULT_SETTINGS, "extra_key": "value"}
    (tmp_pytutor_home / "settings.json").write_text(json.dumps(custom))
    result = load_settings()
    assert result["extra_key"] == "value"


@pytest.mark.parametrize(
    "content, should_error",
    [
        pytest.param("not json", True, id="plain-text"),
        pytest.param("{unclosed", True, id="truncated-json"),
        pytest.param("null", True, id="null-json"),
        pytest.param("{}", False, id="empty-object"),
        pytest.param('{"a": 1}', False, id="valid-object"),
    ],
)
def test_load_settings_invalid_json(tmp_pytutor_home, content, should_error):
    (tmp_pytutor_home / "settings.json").write_text(content)
    if should_error:
        with pytest.raises(SettingsError, match="Failed to parse"):
            load_settings()
    else:
        result = load_settings()
        assert isinstance(result, dict)


def test_prepare_user_space_creates_home_dir(tmp_pytutor_home):
    prepare_user_space()
    assert tmp_pytutor_home.exists()


def test_prepare_user_space_creates_nested_path(tmp_path, monkeypatch):
    nested = tmp_path / "a" / "b" / "c"
    monkeypatch.setenv("PYTUTOR_HOME", str(nested))
    monkeypatch.setattr("pytutor.config.HOME_DIR", nested)
    monkeypatch.setattr("pytutor.config.DATA_DIR", nested / "data")
    monkeypatch.setattr("pytutor.config.STORAGE_DIR", nested / "storage")
    monkeypatch.setattr("pytutor.config.SETTINGS_PATH", nested / "settings.json")
    prepare_user_space()
    assert nested.exists()


def test_prepare_user_space_writes_default_settings(tmp_pytutor_home):
    prepare_user_space()
    settings_path = tmp_pytutor_home / "settings.json"
    assert settings_path.exists()
    assert json.loads(settings_path.read_text()) == DEFAULT_SETTINGS


def test_prepare_user_space_does_not_overwrite(tmp_pytutor_home):
    existing = {"chat_model": "my-model"}
    (tmp_pytutor_home / "settings.json").write_text(json.dumps(existing))
    prepare_user_space()
    result = json.loads((tmp_pytutor_home / "settings.json").read_text())
    assert result == existing


@pytest.mark.parametrize(
    "legacy_files, expect_data, expect_storage",
    [
        pytest.param(
            {"data": True, "storage": True},
            True,
            True,
            id="all-legacy",
        ),
        pytest.param(
            {"data": True},
            True,
            False,
            id="only-data",
        ),
        pytest.param(
            {"storage": True},
            False,
            True,
            id="only-storage",
        ),
        pytest.param(
            {},
            False,
            False,
            id="no-legacy",
        ),
    ],
)
def test_prepare_user_space_migrates_dirs(
    tmp_pytutor_home, monkeypatch, legacy_files, expect_data, expect_storage
):
    monkeypatch.setattr("pytutor.config.LEGACY_ROOT", tmp_pytutor_home)
    for name in legacy_files:
        (tmp_pytutor_home / name).mkdir()

    prepare_user_space()

    assert (tmp_pytutor_home / "data").exists() == expect_data
    assert (tmp_pytutor_home / "storage").exists() == expect_storage


def test_prepare_user_space_migrates_settings(tmp_pytutor_home, monkeypatch):
    monkeypatch.setattr("pytutor.config.LEGACY_ROOT", tmp_pytutor_home)
    legacy = {"chat_model": "legacy-model"}
    (tmp_pytutor_home / "settings.json").write_text(json.dumps(legacy))

    prepare_user_space()

    migrated = json.loads((tmp_pytutor_home / "settings.json").read_text())
    assert migrated["chat_model"] == "legacy-model"


def test_prepare_user_space_idempotent(tmp_pytutor_home):
    prepare_user_space()
    prepare_user_space()
    assert (
        json.loads((tmp_pytutor_home / "settings.json").read_text()) == DEFAULT_SETTINGS
    )


def test_prepare_user_space_existing_dirs_not_overwritten(
    tmp_pytutor_home, monkeypatch
):
    monkeypatch.setattr("pytutor.config.LEGACY_ROOT", tmp_pytutor_home)
    (tmp_pytutor_home / "data").mkdir()
    (tmp_pytutor_home / "data" / "file.txt").write_text("old")
    (tmp_pytutor_home / "storage").mkdir()

    prepare_user_space()

    assert (tmp_pytutor_home / "data" / "file.txt").read_text() == "old"
