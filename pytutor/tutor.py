import asyncio
import json

import httpx

SYSTEM_PROMPT = """You are PyTutor, a friendly and patient Python tutor who helps \
learners understand Python using the official Python documentation.

Guidelines:
- Ground your answers in the documentation excerpts provided with each question. \
Prefer them over your own general knowledge.
- When the excerpts do not cover the question, say so honestly and point the \
learner to where in the docs they might look.
- Answer naturally, like a friendly tutor. Do not start every answer with \
phrases like "According to the documentation" or "Según las fuentes"; weave \
the information into a flowing explanation and only mention a specific doc \
file when it genuinely helps the learner find it.
- Be thorough and detailed: give a step-by-step explanation and complete, \
runnable code examples that illustrate every concept you mention. Show expected \
output where it helps.
- Cover the practical details a learner needs: common arguments, return values, \
side effects, and typical gotchas.
- Never use emojis or decorative symbols. Use plain Markdown formatting only.
- Keep explanations clear and beginner-friendly.
- Match the learner's level: if they seem like a beginner, keep it simple.
- End most answers with one short exercise or a follow-up question to encourage \
practice.
"""

MAX_HISTORY_TURNS = 8  # number of prior user/assistant turns to include
QUERY_CONTEXT_CHARS = 250  # previous-turn text blended into the retrieval query


class Tutor:
    def __init__(self, settings: dict, retrieval):
        self.settings = settings
        self.retrieval = retrieval
        self.top_k = int(settings.get("top_k", 5))
        self.model = settings["chat_model"]
        self.host = settings.get("ollama_host", "http://localhost:11434").rstrip("/")

    def build_context(self, question: str, history: list[dict] | None = None):
        """Retrieve relevant chunks for a question (blocking, run in a thread).

        The retrieval query blends the previous turn into the current question so
        referential follow-ups (e.g. "what are its methods?") are disambiguated by
        the conversation instead of pulling generic excerpts.
        """
        query = self._retrieval_query(question, history or [])
        chunks = self.retrieval.search(query, top_k=self.top_k)
        excerpts = "\n\n".join(
            f'<excerpt source="{c["source_path"]}">\n{c["content"]}\n</excerpt>'
            for c in chunks
        )
        return chunks, excerpts

    def _retrieval_query(self, question: str, history: list[dict]) -> str:
        """Build the search query from the current question and recent context."""
        if not history:
            return question
        context_parts = [
            message["content"][:QUERY_CONTEXT_CHARS]
            for message in history[-2:]
            if message.get("content")
        ]
        context_parts.append(question)
        return " ".join(context_parts)

    def build_messages(self, question, chunks, history, docs_version=None):
        """Build the Ollama message list.

        ``history`` must contain the prior user/assistant turns only (the current
        question is appended once, wrapped with the excerpts, as the last message).
        """
        system = SYSTEM_PROMPT
        if docs_version:
            system += f"\n\nAvailable documentation: Python {docs_version}."
        context = "\n\n".join(
            f'<excerpt source="{c["source_path"]}">\n{c["content"]}\n</excerpt>'
            for c in chunks
        )
        user = f"Documentation excerpts:\n\n{context}\n\nQuestion: {question}"
        messages = [{"role": "system", "content": system}]
        messages += list(history)[-MAX_HISTORY_TURNS * 2 :]
        messages.append({"role": "user", "content": user})
        return messages

    async def stream_answer(self, question, history, docs_version=None):
        """Yield ``("token", text)`` while generating and ``("sources", chunks)`` at the end.

        ``history`` must contain the prior turns only (excluding the current question).
        """
        chunks, _ = await asyncio.to_thread(self.build_context, question, history)
        messages = self.build_messages(question, chunks, history, docs_version)
        payload = {"model": self.model, "messages": messages, "stream": True}
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST", f"{self.host}/api/chat", json=payload
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if data.get("done"):
                        yield "sources", chunks
                        return
                    token = data["message"]["content"]
                    if token:
                        yield "token", token
