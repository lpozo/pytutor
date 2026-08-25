import json

import pytest

from pytutor.index import META_FILE, VECTOR_STORE_BACKEND, _load_meta, _write_meta


def test_load_meta_returns_none_when_missing(tmp_path):
    assert _load_meta(tmp_path) is None


def test_load_meta_valid(tmp_path):
    meta = {"embedding_model": "test", "docs_version": "3.14"}
    (tmp_path / META_FILE).write_text(json.dumps(meta))
    assert _load_meta(tmp_path) == meta


@pytest.mark.parametrize(
    "content",
    [
        pytest.param("{invalid json", id="invalid-json"),
    ],
)
def test_load_meta_raises_on_parse_error(tmp_path, content):
    (tmp_path / META_FILE).write_text(content)
    with pytest.raises(json.JSONDecodeError):
        _load_meta(tmp_path)


@pytest.mark.parametrize(
    "content, expected",
    [
        pytest.param("null", None, id="null"),
        pytest.param("123", 123, id="number"),
        pytest.param('["a", "b"]', ["a", "b"], id="array"),
    ],
)
def test_load_meta_valid_json(tmp_path, content, expected):
    (tmp_path / META_FILE).write_text(content)
    assert _load_meta(tmp_path) == expected


def test_write_meta_creates_file(tmp_path):
    _write_meta(tmp_path, "nomic-embed-text", "3.14", "faiss")
    assert (tmp_path / META_FILE).exists()


def test_write_meta_correct_content(tmp_path):
    _write_meta(tmp_path, "model-a", "3.13", "faiss")
    result = json.loads((tmp_path / META_FILE).read_text())
    assert result == {
        "embedding_model": "model-a",
        "docs_version": "3.13",
        "vector_store": "faiss",
    }


def test_write_meta_overwrites_existing(tmp_path):
    _write_meta(tmp_path, "old-model", "3.12", "json")
    _write_meta(tmp_path, "new-model", "3.14", "faiss")
    result = json.loads((tmp_path / META_FILE).read_text())
    assert result["embedding_model"] == "new-model"


@pytest.mark.parametrize(
    "embedding_model, docs_version, vector_store",
    [
        pytest.param("nomic-embed-text", "3.14", "faiss", id="defaults"),
        pytest.param("", "", "", id="empty-strings"),
        pytest.param("model-v2", None, "custom", id="none-version"),
    ],
)
def test_write_meta_various_inputs(
    tmp_path, embedding_model, docs_version, vector_store
):
    _write_meta(tmp_path, embedding_model, docs_version, vector_store)
    result = json.loads((tmp_path / META_FILE).read_text())
    assert result["embedding_model"] == embedding_model
    assert result["docs_version"] == docs_version
    assert result["vector_store"] == vector_store


def test_meta_file_name_constant():
    assert META_FILE == "meta.json"


def test_vector_store_backend_constant():
    assert VECTOR_STORE_BACKEND == "faiss"
