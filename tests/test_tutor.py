import pytest

from pytutor.tutor import MAX_HISTORY_TURNS, QUERY_CONTEXT_CHARS, Tutor


@pytest.fixture
def tutor():
    return Tutor(
        {
            "chat_model": "test-model",
            "top_k": 3,
            "ollama_host": "http://localhost:11434/",
        },
        retrieval=None,
    )


@pytest.fixture
def chunks():
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


@pytest.mark.parametrize(
    "settings, expected_host",
    [
        pytest.param(
            {"chat_model": "m", "ollama_host": "http://host:11434/"},
            "http://host:11434",
            id="trailing-slash-stripped",
        ),
        pytest.param(
            {"chat_model": "m", "ollama_host": "http://host:11434"},
            "http://host:11434",
            id="no-trailing-slash",
        ),
        pytest.param(
            {"chat_model": "m"},
            "http://localhost:11434",
            id="default-host",
        ),
    ],
)
def test_tutor_host_parsing(settings, expected_host):
    t = Tutor(settings, retrieval=None)
    assert t.host == expected_host


@pytest.mark.parametrize(
    "settings, expected_top_k",
    [
        pytest.param({"chat_model": "m", "top_k": 1}, 1, id="explicit-top-k"),
        pytest.param({"chat_model": "m"}, 5, id="default-top-k"),
        pytest.param({"chat_model": "m", "top_k": "7"}, 7, id="string-top-k"),
    ],
)
def test_tutor_top_k_parsing(settings, expected_top_k):
    t = Tutor(settings, retrieval=None)
    assert t.top_k == expected_top_k


@pytest.mark.parametrize(
    "question, history, expected",
    [
        pytest.param(
            "What is a list?",
            [],
            "What is a list?",
            id="empty-history",
        ),
        pytest.param(
            "What is a list?",
            [{"role": "user", "content": "Hi"}],
            "Hi What is a list?",
            id="single-turn",
        ),
        pytest.param(
            "What is a list?",
            [
                {"role": "user", "content": "Tell me about lists"},
                {"role": "assistant", "content": "Lists are ordered."},
            ],
            "Tell me about lists Lists are ordered. What is a list?",
            id="two-turns",
        ),
    ],
)
def test_retrieval_query_building(tutor, question, history, expected):
    assert tutor._retrieval_query(question, history) == expected


def test_retrieval_query_ignores_old_history(tutor):
    history = [
        {"role": "user", "content": "ancient question"},
        {"role": "assistant", "content": "ancient answer"},
        {"role": "user", "content": "recent question"},
        {"role": "assistant", "content": "recent answer"},
    ]
    result = tutor._retrieval_query("follow up?", history)
    assert "ancient question" not in result
    assert "recent question" in result
    assert "follow up?" in result


def test_retrieval_query_truncates_long_content(tutor):
    long = "x" * (QUERY_CONTEXT_CHARS + 100)
    history = [{"role": "user", "content": long}]
    result = tutor._retrieval_query("q", history)
    assert len(result) < len(long) + 2


@pytest.mark.parametrize(
    "history",
    [
        pytest.param([{"role": "user", "content": None}], id="content-is-none"),
        pytest.param([{"role": "user"}], id="no-content-key"),
        pytest.param([{"role": "user", "content": ""}], id="empty-content"),
    ],
)
def test_retrieval_query_skips_invalid_messages(tutor, history):
    result = tutor._retrieval_query("question", history)
    assert "question" in result


def test_build_messages_structure(tutor, chunks):
    messages = tutor.build_messages("What?", chunks, history=[])
    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "user"
    assert "What?" in messages[-1]["content"]


def test_build_messages_excerpts_included(tutor, chunks):
    messages = tutor.build_messages("What?", chunks, history=[])
    content = messages[-1]["content"]
    assert '<excerpt source="library/stdtypes.txt">' in content
    assert '<excerpt source="tutorial/datastructures.txt">' in content
    assert "list comprehension" in content


def test_build_messages_empty_chunks(tutor):
    messages = tutor.build_messages("What?", [], history=[])
    assert "Question: What?" in messages[-1]["content"]


@pytest.mark.parametrize(
    "version",
    [
        pytest.param(None, id="none"),
        pytest.param("", id="empty-string"),
        pytest.param("3.14", id="valid-version"),
    ],
)
def test_build_messages_docs_version(tutor, chunks, version):
    messages = tutor.build_messages("Q", chunks, history=[], docs_version=version)
    if version:
        assert f"Python {version}" in messages[0]["content"]
    else:
        assert "Available documentation" not in messages[0]["content"]


def test_build_messages_includes_history(tutor, chunks):
    history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
    ]
    messages = tutor.build_messages("Q", chunks, history=history)
    roles = [m["role"] for m in messages]
    assert "user" in roles[1:3]
    assert "assistant" in roles[1:3]


def test_build_messages_truncates_long_history(tutor, chunks):
    history = [
        {"role": "user", "content": f"Q{i}"}
        for i in range(MAX_HISTORY_TURNS * 2 + 4)
    ]
    messages = tutor.build_messages("Q", chunks, history=history)
    non_system = [m for m in messages if m["role"] != "system"]
    assert len(non_system) <= MAX_HISTORY_TURNS * 2 + 1


@pytest.mark.parametrize(
    "chunks, expected_count",
    [
        pytest.param([], 0, id="zero-chunks"),
        pytest.param(
            [{"source_path": "a.txt", "content": "a"}],
            1,
            id="one-chunk",
        ),
        pytest.param(
            [{"source_path": f"f{i}.txt", "content": f"c{i}"} for i in range(10)],
            10,
            id="ten-chunks",
        ),
    ],
)
def test_build_messages_chunk_count(tutor, chunks, expected_count):
    messages = tutor.build_messages("Q", chunks, history=[])
    content = messages[-1]["content"]
    assert content.count("<excerpt") == expected_count


def test_build_messages_system_prompt_present(tutor, chunks):
    messages = tutor.build_messages("Q", chunks, history=[])
    assert "PyTutor" in messages[0]["content"]
