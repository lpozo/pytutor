import json
import os
import shutil
from pathlib import Path

DEFAULT_SETTINGS = {
    "embedding_model": "nomic-embed-text",
    "chat_model": "llama3.2",
    "top_k": 5,
    "embed_batch_size": 512,
    "ollama_host": "http://localhost:11434",
}

LEGACY_ROOT = Path(__file__).parent.parent


def pytutor_home() -> Path:
    """User-space app directory, overridable via the ``PYTUTOR_HOME`` env var."""
    override = os.environ.get("PYTUTOR_HOME")
    return Path(override) if override else Path.home() / ".pytutor"


HOME_DIR = pytutor_home()
DATA_DIR = HOME_DIR / "data"
STORAGE_DIR = HOME_DIR / "storage"
SETTINGS_PATH = HOME_DIR / "settings.json"


class SettingsError(Exception):
    pass


def load_settings() -> dict:
    data = dict(DEFAULT_SETTINGS)
    if SETTINGS_PATH.exists():
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            try:
                data.update(json.load(f))
            except Exception as e:
                raise SettingsError(f"Failed to parse settings.json: {e}")
    return data


def prepare_user_space() -> None:
    """Create the user-space directory and prepare it for use.

    Migrates a legacy ``data/``, ``storage/`` and ``settings.json`` from the
    package root into the user space on first run, and writes a default
    ``settings.json`` when none exists.
    """
    HOME_DIR.mkdir(parents=True, exist_ok=True)
    _migrate_legacy_files()
    if not SETTINGS_PATH.exists():
        _write_default_settings()


def _migrate_legacy_files() -> None:
    for name in ("data", "storage"):
        target = HOME_DIR / name
        source = LEGACY_ROOT / name
        if not target.exists() and source.exists():
            shutil.move(str(source), str(target))
    settings_target = SETTINGS_PATH
    settings_source = LEGACY_ROOT / "settings.json"
    if not settings_target.exists() and settings_source.exists():
        shutil.move(str(settings_source), str(settings_target))


def _write_default_settings() -> None:
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_SETTINGS, f, indent=2)
