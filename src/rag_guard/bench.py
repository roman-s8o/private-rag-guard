"""Benchmark CLI: run one engine over the sets and print per-language, per-level results.

rag-guard-sets                                   # writes data/*.jsonl (synthetic, deterministic)
rag-guard-bench --engine laya --split dev        # calibrate: writes runs/thresholds_<engine>.json
rag-guard-bench --engine laya --split test --thresholds runs/thresholds_laya.json
rag-guard-bench --engine ollama:qwen3.5:9b --split test
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .engines import Cascade, Engine, LayaEngine, OllamaJudge
from .metrics import latency, summarize

DATA = Path("data")
RUNS = Path("runs")


def make_engine(spec: str) -> Engine:
    if spec == "laya":
        return LayaEngine()
    if spec.startswith("ollama:"):
        return OllamaJudge(spec.split(":", 1)[1])
    if spec.startswith("cascade:"):  # cascade:<ollama model>:<lo>:<hi>  (first engine = laya)
        _, rest = spec.split(":", 1)
        model, lo, hi = rest.rsplit(":", 2)
        return Cascade(LayaEngine(), OllamaJudge(model), float(lo), float(hi))
    raise SystemExit(f"unknown engine {spec!r}")


def youden(rows: list[dict], positive: str) -> float:
    """Threshold with the best TPR - FPR on dev (ties go to the higher threshold)."""
    pos = [r for r in rows if r["label"] == positive]
    neg = [r for r in rows if r["label"] != positive]
    best = (-2.0, 0.5)
    for t in sorted({r["score"] for r in rows} | {0.5}):
        j = sum(r["score"] >= t for r in pos) / max(1, len(pos)) - sum(r["score"] >= t for r in neg) / max(1, len(neg))
        if j >= best[0]:
            best = (j, t)
    return best[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--thresholds", help="JSON {injection, grounding} frozen on dev; default 0.5")
    args = ap.parse_args()
    engine = make_engine(args.engine)
    thr = json.loads(Path(args.thresholds).read_text()) if args.thresholds else {"injection": 0.5, "grounding": 0.5}

    rows: list[dict] = []
    times: list[float] = []
    for task in ("injection", "grounding"):
        for line in (DATA / f"{task}_{args.split}_v1.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            t = time.perf_counter()
            score = (
                engine.injection(r["text"], r["kind"])
                if task == "injection"
                else engine.grounded(r["passage"], r["question"], r["answer"])
            )
            times.append((time.perf_counter() - t) * 1000)
            rows.append(
                {
                    "set": task,
                    "id": r["id"],
                    "template": r["template"],
                    "lang": r["lang"],
                    "level": r.get("level", "-"),
                    "label": r["label"],
                    "error_type": r.get("error_type"),
                    "score": score,
                    "pred": score >= thr[task],
                }
            )

    RUNS.mkdir(exist_ok=True)
    name = f"{engine.name.replace(':', '_').replace('/', '_').replace('>', '')}_{args.split}" + (
        "_frozen" if args.thresholds else ""
    )
    summary = summarize(rows)
    lat = latency(times)
    extra = {}
    if isinstance(engine, Cascade):
        extra = {"second_engine_calls": engine.second_calls, "total_calls": engine.total_calls}
    (RUNS / f"{name}.json").write_text(
        json.dumps(
            {
                "engine": engine.name,
                "split": args.split,
                "thresholds": thr,
                "latency": lat,
                "summary": summary,
                **extra,
                "rows": rows,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    if args.split == "dev":
        frozen = {
            "injection": youden([r for r in rows if r["set"] == "injection"], "injection"),
            "grounding": youden([r for r in rows if r["set"] == "grounding"], "supported"),
        }
        (RUNS / f"thresholds_{engine.name.replace(':', '_')}.json").write_text(json.dumps(frozen, indent=1))
        print("thresholds frozen on dev:", frozen)

    print(f"\nengine {engine.name} | split {args.split} | thresholds {thr}")
    for s in summary:
        print(
            f"{s['metric']:<36} {s['lang']} level {s['level']}: {s['k']}/{s['n']} = {s['rate']:.2f} [{s['lo']:.2f}, {s['hi']:.2f}]"
        )
    print(f"latency per check: p50 {lat['p50_ms']:.0f} ms, p95 {lat['p95_ms']:.0f} ms (this machine)")
    if extra:
        print(f"second engine called on {extra['second_engine_calls']}/{extra['total_calls']} checks")


if __name__ == "__main__":
    main()
