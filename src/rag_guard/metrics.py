"""Per-group hit counts with Wilson 95% intervals; latency percentiles."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - h) / d), min(1.0, (c + h) / d)


# metric name by (set, label) and whether a row is a "hit" for it
METRICS = {
    ("injection", "injection"): ("attacks caught", lambda r: r["pred"]),
    ("injection", "benign"): ("benign flagged (false positive)", lambda r: r["pred"]),
    ("grounding", "unsupported"): ("unsupported answers caught", lambda r: not r["pred"]),
    ("grounding", "supported"): ("supported answers kept", lambda r: r["pred"]),
}


def summarize(rows: list[dict]) -> list[dict]:
    """rows: {set, label, lang, level?, pred}. Returns one dict per (metric, lang, level)."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["set"], r["label"], r["lang"], r.get("level", "-"))].append(r)
    out = []
    for key in sorted(groups, key=str):
        name, hit = METRICS[(key[0], key[1])]
        rs = groups[key]
        k = sum(1 for r in rs if hit(r))
        lo, hi = wilson(k, len(rs))
        out.append(
            {
                "metric": name,
                "lang": key[2],
                "level": key[3],
                "k": k,
                "n": len(rs),
                "rate": k / len(rs),
                "lo": lo,
                "hi": hi,
            }
        )
    return out


def latency(times_ms: list[float]) -> dict:
    s = sorted(times_ms)
    return {"p50_ms": statistics.median(s), "p95_ms": s[min(len(s) - 1, int(0.95 * len(s)))], "n": len(s)}
