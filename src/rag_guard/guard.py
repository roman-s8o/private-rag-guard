"""The RAG-facing API: three checks around the generation step. Nothing here logs or stores text."""

from __future__ import annotations

import time
from dataclasses import dataclass

from .engines import Engine


@dataclass(frozen=True)
class Verdict:
    ok: bool  # True = let it through
    score: float  # engine score: P(injection) for inputs and documents, P(supported) for answers
    check: str  # "user_message" | "document" | "answer"
    engine: str
    latency_ms: float


class Guard:
    def __init__(self, engine: Engine, injection_threshold: float = 0.5, grounding_threshold: float = 0.5) -> None:
        self.engine = engine
        self.injection_threshold = injection_threshold
        self.grounding_threshold = grounding_threshold

    def check_user_message(self, text: str) -> Verdict:
        return self._injection(text, "user", "user_message")

    def check_document(self, text: str) -> Verdict:
        """Run on every retrieved chunk before it reaches the prompt (indirect injection)."""
        return self._injection(text, "document", "document")

    def check_answer(self, passage: str, question: str, answer: str) -> Verdict:
        t = time.perf_counter()
        s = self.engine.grounded(passage, question, answer)
        return Verdict(s >= self.grounding_threshold, s, "answer", self.engine.name, (time.perf_counter() - t) * 1000)

    def _injection(self, text: str, kind: str, check: str) -> Verdict:
        t = time.perf_counter()
        s = self.engine.injection(text, kind)
        return Verdict(s < self.injection_threshold, s, check, self.engine.name, (time.perf_counter() - t) * 1000)
