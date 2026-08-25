import pytest

from pytutor.config import DATA_DIR
from pytutor.retrieval import RetrievalService


@pytest.mark.parametrize(
    "source_path, expected",
    [
        pytest.param(
            str(DATA_DIR / "library" / "stdtypes.txt"),
            "library/stdtypes.txt",
            id="absolute-under-data-dir",
        ),
        pytest.param(
            "library/stdtypes.txt",
            "library/stdtypes.txt",
            id="relative-path",
        ),
        pytest.param(
            str(DATA_DIR / "tutorial" / "datastructures.txt"),
            "tutorial/datastructures.txt",
            id="nested-under-data-dir",
        ),
        pytest.param(
            "/completely/unrelated/path.txt",
            "/completely/unrelated/path.txt",
            id="unrelated-absolute",
        ),
        pytest.param(
            "/home/user/project/data/library/file.txt",
            "library/file.txt",
            id="data-in-middle-of-path",
        ),
        pytest.param(
            "/foo/bar/baz.txt",
            "/foo/bar/baz.txt",
            id="no-data-in-parts",
        ),
        pytest.param(
            "",
            ".",
            id="empty-string",
        ),
    ],
)
def test_relative_source(source_path, expected):
    assert RetrievalService._relative_source(source_path) == expected


def test_relative_source_just_data_dir():
    result = RetrievalService._relative_source(str(DATA_DIR))
    assert result == "."
