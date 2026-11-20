import json

from rag_guard.engines import Cascade
from rag_guard.guard import Guard
from rag_guard.make_sets import build_grounding, build_injection
from rag_guard.metrics import summarize, wilson


class Fake:
    def __init__(self, name, inj=0.0, gr=1.0):
        self.name, self.inj, self.gr = name, inj, gr
        self.calls = 0

    def injection(self, text, kind):
        self.calls += 1
        return self.inj

    def grounded(self, passage, question, answer):
        self.calls += 1
        return self.gr


def test_guard_blocks_and_passes():
    assert not Guard(Fake("f", inj=0.9)).check_user_message("x").ok
    assert Guard(Fake("f", inj=0.1)).check_document("x").ok
    assert not Guard(Fake("f", gr=0.2)).check_answer("p", "q", "a").ok


def test_cascade_only_calls_second_in_band():
    first, second = Fake("a", inj=0.5), Fake("b", inj=1.0)
    c = Cascade(first, second, 0.2, 0.9)
    assert c.injection("t", "user") == 1.0 and second.calls == 1
    first.inj = 0.05
    assert c.injection("t", "user") == 0.0 and second.calls == 1
    first.inj = 0.95
    assert c.injection("t", "user") == 1.0 and second.calls == 1
    assert (c.second_calls, c.total_calls) == (1, 3)


def test_wilson_bounds():
    lo, hi = wilson(5, 5)
    assert 0.5 < lo < 0.6 and hi == 1.0
    assert wilson(0, 0) == (0.0, 1.0)


def test_sets_split_by_template_and_balanced():
    for build in (build_injection, build_grounding):
        sets = build()
        assert not ({r["template"] for r in sets["dev"]} & {r["template"] for r in sets["test"]})
        for rows in sets.values():
            for lang in ("en", "fr"):
                labels = [r["label"] for r in rows if r["lang"] == lang]
                assert labels.count(sorted(set(labels))[0]) == labels.count(sorted(set(labels))[1])
    json.dumps(build_injection())  # serialisable


def test_summarize_counts():
    rows = [
        {"set": "injection", "label": "injection", "lang": "en", "level": 1, "pred": p} for p in (True, True, False)
    ]
    s = summarize(rows)[0]
    assert (s["k"], s["n"]) == (2, 3)
