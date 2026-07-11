"""LLM token usage tracker + structured-invoke helper.

Records one JSONL row per LLM invoke to ``outputs/usage/<run_id>/<lang>.jsonl``
so token cost (input / output / thinking) can be aggregated per agent, language,
and retry for thesis reporting.

Why a separate log (not per-sentence): token usage is per-CALL, not per-sentence
(Contextualizer processes 5 seeds in 1 call; SV/LV process a seed-group per call).
Retries (parse-retry / validation-retry) each burn tokens and are recorded as
separate rows, so "tokens wasted on retry" stays measurable.

Thread-safety: SV and LV invoke in parallel (ThreadPoolExecutor). File append is
guarded by a module-level Lock (append is cheap vs LLM latency, so no bottleneck).

Token semantics (Gemini 3, via langchain-google-genai usage_metadata):
  - output_tokens ALREADY includes thinking (candidates + thoughts) — not double-counted.
  - reasoning_tokens is the thinking breakout (output_token_details.reasoning).
See agent_doc/vertex_ai_setup_and_token_tracking.md for full context.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

# ── Module state ────────────────────────────────────────────────────────────
_lock = threading.Lock()
_run_dir: Path | None = None
_enabled = False
_run_id: str | None = None


def init_usage_tracker(base_dir: str | Path) -> None:
    """Enable usage logging under ``base_dir/<run_id>/<lang>.jsonl``.

    Call once at pipeline start (from run_pipeline). Each run writes to its OWN
    folder (``run_id`` = start timestamp), so re-runs are physically separated —
    no mixing, no need to clear files, and one run's tokens = one folder.

    ``run_id`` uses a filesystem-safe timestamp (``_`` and ``-`` only, no colons)
    so it is valid as a Windows folder name. Idempotent.
    """
    global _run_dir, _enabled, _run_id
    _run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    _run_dir = Path(base_dir) / _run_id
    _enabled = True
    print(f"Usage tracker: aktif (run={_run_id}) -> {_run_dir}/<lang>.jsonl")


def get_run_id() -> str | None:
    """Return the run_id (FS-safe start timestamp) captured at init, or None if not
    yet initialized. Shared so pipeline output can use the SAME per-run folder name.
    """
    return _run_id


def record_usage(
    *,
    lang: str,
    label: str,
    agent: str,
    seed_id: int | None,
    attempt: int,
    ok: bool,
    usage: dict | None,
    model: str,
) -> None:
    """Append one usage row. No-op if tracker not initialized. Thread-safe.

    ``usage`` is a LangChain ``AIMessage.usage_metadata`` dict (or None).
    """
    if not _enabled or _run_dir is None:
        return

    um = usage or {}
    otd = um.get("output_token_details") or {}
    row = {
        "lang": lang,
        "label": label,
        "agent": agent,
        "seed_id": seed_id,
        "attempt": attempt,
        "ok": ok,
        "input_tokens": um.get("input_tokens", 0),
        "output_tokens": um.get("output_tokens", 0),  # includes reasoning
        "reasoning_tokens": otd.get("reasoning", 0),
        "total_tokens": um.get("total_tokens", 0),
        "model": model,
    }
    path = _run_dir / f"{lang}.jsonl"
    line = json.dumps(row, ensure_ascii=False)
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def invoke_structured_tracked(
    structured_llm: Any,
    messages: Any,
    *,
    agent_display: str,
    fail_consequence: str,
    lang: str,
    label: str,
    model: str,
    parse_max_retry: int,
    seed_id: int | None = None,
) -> Any | None:
    """Invoke a structured LLM (built with ``include_raw=True``), record token usage
    per attempt, and return the parsed object (or None after exhausting retries).

    Behaviour mirrors the old ``invoke_with_parse_retry`` closures, but with
    ``include_raw=True`` two failure modes are now distinguished:
      - API/network error (429, timeout, ...) -> ``.invoke()`` raises -> caught here.
      - Schema parse failure -> no exception; surfaced as ``out["parsing_error"]``.
    Usage is recorded for every attempt that returned a response (even parse
    failures), because those tokens were still billed.

    ``agent_display`` is used in console messages ("SV", "Contextualizer", ...) and
    lowercased for the ``agent`` field in the usage log ("sv", "contextualizer").
    ``fail_consequence`` is the tail printed on final failure (e.g. "Batch dibuang.").
    """
    seed_tag = f" seed_id={seed_id}" if seed_id is not None else ""
    agent_key = agent_display.lower()
    total_attempts = parse_max_retry + 1

    for attempt in range(total_attempts):
        try:
            out = structured_llm.invoke(messages)
        except Exception as e:  # API/network error (still raises with include_raw)
            if attempt < parse_max_retry:
                print(f"{agent_display} WARNING{seed_tag}: API error (attempt {attempt + 1}/{total_attempts}: {type(e).__name__}). Coba lagi.")
                continue
            print(f"{agent_display} FAILED{seed_tag}: API error setelah {total_attempts} attempts ({type(e).__name__}). {fail_consequence}")
            return None

        raw = out.get("raw")
        parsed = out.get("parsed")
        parse_err = out.get("parsing_error")
        ok = parse_err is None and parsed is not None

        record_usage(
            lang=lang,
            label=label,
            agent=agent_key,
            seed_id=seed_id,
            attempt=attempt,
            ok=ok,
            usage=getattr(raw, "usage_metadata", None),
            model=model,
        )

        if ok:
            return parsed

        # Parse failure (no exception raised when include_raw=True)
        err_name = type(parse_err).__name__ if parse_err is not None else "EmptyParse"
        if attempt < parse_max_retry:
            print(f"{agent_display} WARNING{seed_tag}: parse failed (attempt {attempt + 1}/{total_attempts}: {err_name}). Coba lagi.")
        else:
            print(f"{agent_display} FAILED{seed_tag}: parse failed setelah {total_attempts} attempts ({err_name}). {fail_consequence}")

    return None
