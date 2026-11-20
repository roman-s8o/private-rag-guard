# private-rag-guard

Prompt-injection and answer-grounding checks for RAG that run **inside your perimeter**: no text leaves the machine.
Three checks around the generation step, a pluggable engine (small decision model, local LLM judge, or a cascade of both), and a
reproducible benchmark that reports **English and French separately**, by attack level, with Wilson 95% intervals.
Made for regulated teams working in France and the EU (banks and their local branches, insurers, fintechs), where documents come in both languages and data cannot go to an AI provider.

```python
from rag_guard.engines import OllamaJudge
from rag_guard.guard import Guard

guard = Guard(OllamaJudge("qwen3.5:9b"))
guard.check_user_message(question)  # injection in what the user typed
guard.check_document(chunk)  # injection hidden in a retrieved chunk (indirect)
guard.check_answer(passage, question, answer)  # is every claim supported by the source?
# -> Verdict(ok, score, check, engine, latency_ms); nothing is logged or stored
```

## What we measured (test split, one run each, thresholds fixed before the run)

Synthetic, hand-written sets about an invented bank: 216 injection checks (108 attacks, 108 benign twins; 3 levels: direct,
disguised, hidden in a retrieved document) and 100 grounding checks (50 supported, 50 unsupported), EN and FR. Dev and test share
no template. Latency is per check on one MacBook Pro (Apple silicon, 18 GB), not a server.

| Engine | Attacks caught | Benign flagged | Unsupported answers caught | Supported answers kept | Latency p50 / p95 |
|---|---|---|---|---|---|
| Laya, threshold 0.5 | 87/108 (0.81) | 46/108 (0.43) | 36/50 | 50/50 | 18 / 23 ms |
| Laya, thresholds frozen on dev | 52/108 (0.48) | 14/108 (0.13) | 44/50 | 36/50 | 18 / 23 ms |
| `qwen3.5:9b` judge (Ollama) | 107/108 (0.99) | 12/108 (0.11) | 50/50 | 50/50 | 843 / 972 ms |
| Cascade Laya → judge, band [0.1, 0.999) | 99/108 (0.92) | 16/108 (0.15) | 50/50 | 50/50 | 830 / 1001 ms (judge on 198/316 checks) |

Per language and level (attacks caught / benign flagged, EN and FR): full per-cell numbers with intervals are in `results/*.json` (the runs behind the table above; rows hold ids and scores, no text).

What this says, and what it does not:

- The local 9B judge was the only engine that was both accurate and quiet on this data. A 9B model is slow (about 0.8 s), but
  the data stays inside.
- The small decision model (Laya, ~20 ms) is fast but not reliable as the only injection detector here: it is often confidently
  wrong, so a cascade driven by its confidence cannot recover the attacks it misses. Thresholds tuned on dev did **not** carry
  over to test (supported answers kept fell from 24/24 on dev to 36/50), because templates differ between splits.
- The false positives of the judge are concentrated in level 1 benign twins such as "Ignore my previous message about the
  card…": these are deliberately hard twins, and a real deployment needs its own twins.
- Small, synthetic, one generator. A per-cell sample is 18, so individual cells have wide intervals. This is evidence about
  these engines on this data, not a claim that small models beat or lose to LLM judges in general. Run it on your own text.

## Reproduce

```bash
uv sync
uv run rag-guard-sets                                   # writes data/*.jsonl, deterministic
uv run --extra laya rag-guard-bench --engine laya --split dev     # also freezes thresholds on dev
uv run rag-guard-bench --engine ollama:qwen3.5:9b --split test
uv run --extra laya rag-guard-bench --engine cascade:qwen3.5:9b:0.1:0.999 --split test
```

The test split is meant to be opened once per engine configuration. If you change prompts or sets after seeing it, make a new set version.

## Not done yet

OpenJev-4B and Kev-4B need a CUDA machine (custom runtime); not measured. Real (non-synthetic) text and a native-speaker review of the French
are missing. The sets are written by one author with one generator.

## License

MIT. Models are used under their own licenses: Laya (Apache-2.0), Qwen3.5 (check the model card before commercial use).
