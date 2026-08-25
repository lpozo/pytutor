import json
from pathlib import Path

import pytest


@pytest.fixture
def tmp_pytutor_home(tmp_path, monkeypatch):
    """Set PYTUTOR_HOME to a temp directory and return it."""
    monkeypatch.setenv("PYTUTOR_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture
def sample_chunks():
    return [
        {
            "chunk_id": "1",
            "content": "A list comprehension creates a list from an iterable.",
            "source_path": "library/stdtypes.txt",
            "score": 0.85,
        },
        {
            "chunk_id": "2",
            "content": "List comprehensions provide a concise way to create lists.",
            "source_path": "tutorial/datastructures.txt",
            "score": 0.72,
        },
    ]
