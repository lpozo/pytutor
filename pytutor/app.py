from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Footer, Header, Input, Markdown, ProgressBar

from pytutor.tutor import Tutor

WELCOME = """# Welcome to PyTutor!

I'm your Python tutor, grounded in the **official Python documentation**.

Ask me anything about Python — language features, the standard library,
best practices, or how to fix an error.

- Type a question below and press **Enter**.
- Press **Ctrl+R** to start a new conversation.
- Press **Ctrl+C** to quit.
"""

CSS = """
Screen {
    layout: vertical;
}

#chat {
    width: 1fr;
    height: 1fr;
    padding: 0 2;
}

.user {
    width: 100%;
    background: $primary 15%;
    border: round $primary;
    padding: 0 1;
    margin: 0 0 1 0;
}

.assistant {
    width: 100%;
    border: round $surface-lighten-1;
    padding: 0 1;
    margin: 0 0 1 0;
}

#prompt {
    height: 3;
    margin: 0 2 1 2;
}
"""


class PyTutorApp(App):
    TITLE = "PyTutor"
    SUB_TITLE = "Grounded in the official Python docs"
    CSS = CSS
    BINDINGS = [
        ("ctrl+r", "reset", "New conversation"),
        ("ctrl+c", "quit", "Quit"),
    ]

    def __init__(self, settings: dict, build_retrieval, **kwargs):
        super().__init__(**kwargs)
        self.settings = settings
        self._build_retrieval = build_retrieval
        self.tutor: Tutor | None = None
        self.docs_version: str | None = None
        self.history: list[dict] = []
        self.busy = False
        self._quitting = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield VerticalScroll(id="chat")
        yield Input(placeholder="Ask me about Python...", id="prompt")
        yield Footer()

    def on_mount(self) -> None:
        self._add_bubble(WELCOME, "assistant")
        self.status_bubble = self._add_bubble("_Loading…_", "assistant")
        self.progress_bar = ProgressBar(total=1.0, show_eta=False)
        self._chat().mount(self.progress_bar)
        self._scroll_end()
        self._set_prompt_disabled(True)
        self.run_worker(
            self._init_backend, thread=True, exit_on_error=False, group="init"
        )

    def _init_backend(self) -> None:
        """Load or bootstrap the index in a worker thread with progress UI.

        Runs off the UI thread; every UI mutation is marshalled back through
        ``call_from_thread`` so the TUI stays responsive while the index loads
        or is built for the first time.
        """
        try:
            retrieval, meta = self._build_retrieval(
                status=self._set_status,
                download_progress=self._download_progress,
                embed_progress=self._embed_progress,
            )
        except Exception as e:
            self.call_from_thread(self._backend_failed, e)
        else:
            self.call_from_thread(self._backend_ready, retrieval, meta)

    def _backend_ready(self, retrieval, meta: dict) -> None:
        self.tutor = Tutor(self.settings, retrieval)
        self.docs_version = meta.get("docs_version")
        self._remove_progress_ui()
        self._set_prompt_disabled(False)
        self.query_one("#prompt", Input).focus()

    def _backend_failed(self, error: Exception) -> None:
        self.progress_bar.remove()
        self.status_bubble.update(
            f"**Index unavailable:** {error}\n\n"
            "Make sure Ollama is running and try again, or rebuild with `pytutor -u`."
        )

    def _remove_progress_ui(self) -> None:
        self.status_bubble.remove()
        self.progress_bar.remove()

    def _set_status(self, status: str) -> None:
        self.call_from_thread(self._set_progress, None, None, status)

    def _download_progress(self, done: int, total: int) -> None:
        self.call_from_thread(self._set_progress, done, total or None, None)

    def _embed_progress(self, done: int, total: int) -> None:
        self.call_from_thread(self._set_progress, done, total, None)

    def _set_progress(self, progress, total, status) -> None:
        """Update the status text and progress bar (must run on the UI thread)."""
        if self._quitting:
            return
        if status is not None:
            self.status_bubble.update(f"_{status}_")
        bar = self.progress_bar
        if total:
            bar.update(total=total, progress=progress)
        else:
            bar.update(total=None)
        self._scroll_end()

    def on_unmount(self) -> None:
        self._quitting = True

    def _set_prompt_disabled(self, disabled: bool) -> None:
        self.query_one("#prompt", Input).disabled = disabled

    def _chat(self) -> VerticalScroll:
        return self.query_one("#chat", VerticalScroll)

    def _add_bubble(self, content: str, kind: str) -> Markdown:
        bubble = Markdown(content, classes=kind)
        self._chat().mount(bubble)
        self._chat().scroll_end(animate=False)
        return bubble

    def _scroll_end(self) -> None:
        self._chat().scroll_end(animate=False)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        question = event.value.strip()
        event.input.clear()
        if not question or self.busy:
            return
        self._ask(question)

    def _ask(self, question: str) -> None:
        if self.tutor is None:
            return
        self.busy = True
        prompt = self.query_one("#prompt", Input)
        prompt.disabled = True
        self._add_bubble(question, "user")
        bubble = self._add_bubble("_Thinking..._", "assistant")
        self.run_worker(
            self._respond(question, bubble),
            exit_on_error=False,
            group="respond",
        )

    async def _respond(self, question: str, bubble: Markdown) -> None:
        if self.tutor is None:
            bubble.update("**Index still loading. Try again in a moment.**")
            return
        history = list(self.history)
        self.history.append({"role": "user", "content": question})
        prompt = self.query_one("#prompt", Input)
        parts: list[str] = []
        try:
            async for kind, payload in self.tutor.stream_answer(
                question, history, self.docs_version
            ):
                if kind == "token":
                    parts.append(payload)
                    bubble.update("".join(parts))
                    self._scroll_end()
        except Exception as e:
            bubble.update(f"**Something went wrong:**\n\n{e}")
        else:
            bubble.update("".join(parts))
            self.history.append({"role": "assistant", "content": "".join(parts)})
        finally:
            prompt.disabled = False
            prompt.focus()
            self.busy = False
            self._scroll_end()

    def action_reset(self) -> None:
        if self.busy:
            return
        self.history.clear()
        self._chat().remove_children()
        self._add_bubble(WELCOME, "assistant")
        self.query_one("#prompt", Input).focus()
