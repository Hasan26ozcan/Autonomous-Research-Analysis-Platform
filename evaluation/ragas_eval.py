"""
ARAP RAGAS and Evaluation Suite
===============================

This module provides a comprehensive evaluation workflow for the ARAP system.
It is intended for both development-time validation and production monitoring
of the adaptive RAG pipeline.

The suite covers:
- loading realistic Q/A examples from PostgreSQL query history,
- running them through the live orchestrator,
- collecting the generated answer, retrieved sources, and faithfulness signal,
- executing standard RAGAS metrics when the optional dependencies are present,
- saving a structured JSON report for later analysis or CI/CD use.

Example:
    python -m evaluation.ragas_eval --limit 20 --save reports/ragas_report.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SEED_QA: list[dict[str, str]] = [
    {
        "question": "What is the main contribution of this document?",
        "ground_truth": "The document presents its main contribution in the "
        "introduction or abstract.",
    },
    {
        "question": "What methodology is described in this document?",
        "ground_truth": "The methodology section describes the research or "
        "engineering approach used.",
    },
    {
        "question": "What are the key results or findings?",
        "ground_truth": "The results section presents the quantitative or qualitative findings.",
    },
    {
        "question": "What are the limitations mentioned?",
        "ground_truth": "The limitations section discusses the boundaries of the work.",
    },
    {
        "question": "Who are the target users or audience of this document?",
        "ground_truth": "The document targets practitioners or researchers in its subject domain.",
    },
    {
        "question": "What datasets or benchmarks are referenced?",
        "ground_truth": "The document references one or more datasets or "
        "benchmark tasks used for validation.",
    },
    {
        "question": "How does the proposed method compare to baselines?",
        "ground_truth": "The document explains how the proposed method "
        "improves over the baseline methods.",
    },
]


async def _load_test_set(limit: int = 20, include_seed: bool = True) -> list[dict[str, str]]:
    """Load real Q/A pairs from PostgreSQL query_history when possible."""
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor

        from app.core.config import settings

        conn = psycopg2.connect(settings.postgres_url, connect_timeout=5)
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT question, answer
                    FROM query_history
                    WHERE answer IS NOT NULL AND LENGTH(answer) > 20
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = cur.fetchall()
        finally:
            conn.close()
        if rows:
            return [
                {"question": str(r["question"]), "ground_truth": str(r["answer"])}
                for r in rows
            ]
    except Exception as exc:
        logger.warning(
            "Could not load evaluation set from PostgreSQL; using seed questions. Error: %s",
            exc,
        )

    if include_seed:
        return SEED_QA[:limit]
    return []


async def _run_single_query(orchestrator: Any, item: dict[str, str]) -> dict[str, Any]:
    """Run one evaluation item through the live orchestrator."""
    try:
        result = await orchestrator.query(
            question=item["question"],
            session_id="ragas-eval",
            user_id="ragas",
        )
    except Exception as exc:
        logger.warning("Evaluation query failed for '%s': %s", item["question"][:50], exc)
        return {
            "question": item["question"],
            "answer": "",
            "contexts": [],
            "sources": [],
            "ground_truth": item.get("ground_truth", ""),
            "query_type": None,
            "faithfulness_score": None,
            "latency_ms": {},
            "token_usage": {},
            "error": str(exc),
        }

    sources = result.get("sources") or []
    contexts: list[str] = []
    for source in sources:
        text = source.get("text") if isinstance(source, dict) else None
        if text:
            contexts.append(str(text))

    return {
        "question": item["question"],
        "answer": result.get("answer") or "",
        "contexts": contexts,
        "sources": sources,  # full source dicts (doc_id, chunk_index, ...) for Phase 9
        "ground_truth": item.get("ground_truth", ""),
        "query_type": result.get("query_type"),
        "faithfulness_score": result.get("faithfulness_score"),
        "latency_ms": result.get("latency_ms") or {},
        "token_usage": result.get("token_usage") or {},
        "error": None,
    }


async def run_ragas_evaluation(
    orchestrator: Any,
    limit: int = 20,
    include_seed: bool = True,
    save_path: str | None = None,
) -> dict[str, Any]:
    """
    Run the full evaluation workflow.

    The function returns a structured report with:
    - standard RAGAS metrics when the optional dependencies are installed,
    - a fallback heuristic summary when they are not,
    - per-question execution details,
    - overall success statistics.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as exc:
        logger.warning("RAGAS dependencies are not available: %s", exc)
        return _build_fallback_report(
            orchestrator=orchestrator,
            limit=limit,
            include_seed=include_seed,
            save_path=save_path,
            reason=str(exc),
        )

    test_set = await _load_test_set(limit=limit, include_seed=include_seed)
    logger.info("Starting RAGAS evaluation over %d questions", len(test_set))

    processed: list[dict[str, Any]] = []
    for item in test_set:
        processed.append(await _run_single_query(orchestrator, item))

    successful = [
        record for record in processed if not record.get("error") and bool(record.get("answer"))
    ]
    if not successful:
        report = {
            "status": "failed",
            "error": "No evaluation questions could be processed successfully.",
            "metrics": {
                "faithfulness": None,
                "answer_relevancy": None,
                "context_precision": None,
                "context_recall": None,
            },
            "num_questions": 0,
            "processed": processed,
        }
        if save_path:
            _write_report(report, save_path)
        await _persist_eval_run(report, processed)
        return report

    dataset = Dataset.from_dict(
        {
            "question": [record["question"] for record in successful],
            "answer": [record["answer"] for record in successful],
            "contexts": [record["contexts"] for record in successful],
            "ground_truth": [record["ground_truth"] for record in successful],
        }
    )

    try:
        ragas_result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
    except Exception as exc:
        logger.exception("RAGAS execution failed: %s", exc)
        report = {
            "status": "failed",
            "error": f"RAGAS execution failed: {exc}",
            "metrics": {
                "faithfulness": None,
                "answer_relevancy": None,
                "context_precision": None,
                "context_recall": None,
            },
            "num_questions": len(successful),
            "processed": processed,
        }
        if save_path:
            _write_report(report, save_path)
        await _persist_eval_run(report, processed)
        return report

    report = {
        "status": "ok",
        "metrics": {
            "faithfulness": _round_metric(ragas_result.get("faithfulness")),
            "answer_relevancy": _round_metric(ragas_result.get("answer_relevancy")),
            "context_precision": _round_metric(ragas_result.get("context_precision")),
            "context_recall": _round_metric(ragas_result.get("context_recall")),
        },
        "num_questions": len(successful),
        "processed": processed,
        "summary": {
            "questions_attempted": len(test_set),
            "questions_succeeded": len(successful),
            "questions_failed": len(processed) - len(successful),
            "average_faithfulness": _average(
                [
                    record.get("faithfulness_score")
                    for record in successful
                    if record.get("faithfulness_score") is not None
                ]
            ),
        },
    }

    if save_path:
        _write_report(report, save_path)

    await _persist_eval_run(report, processed)
    logger.info("RAGAS report: %s", report)
    return report


async def _build_fallback_report(
    orchestrator: Any,
    limit: int,
    include_seed: bool,
    save_path: str | None,
    reason: str,
) -> dict[str, Any]:
    """Generate a usable report even when RAGAS itself is unavailable."""
    test_set = await _load_test_set(limit=limit, include_seed=include_seed)
    processed: list[dict[str, Any]] = []
    for item in test_set:
        processed.append(await _run_single_query(orchestrator, item))

    successful = [
        record
        for record in processed
        if not record.get("error") and bool(record.get("answer"))
    ]
    report = {
        "status": "partial",
        "error": reason,
        "metrics": {
            "faithfulness": None,
            "answer_relevancy": None,
            "context_precision": None,
            "context_recall": None,
        },
        "num_questions": len(successful),
        "processed": processed,
        "summary": {
            "questions_attempted": len(test_set),
            "questions_succeeded": len(successful),
            "questions_failed": len(processed) - len(successful),
            "average_faithfulness": _average(
                [
                    record.get("faithfulness_score")
                    for record in successful
                    if record.get("faithfulness_score") is not None
                ]
            ),
        },
    }
    if save_path:
        _write_report(report, save_path)
    return report


def _round_metric(value: Any) -> float | None:
    """Normalize a metric value to a float or None."""
    if value is None:
        return None
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def _average(values: list[float | None]) -> float | None:
    """Compute the average of numeric values."""
    numeric_values = [float(v) for v in values if v is not None]
    if not numeric_values:
        return None
    return round(sum(numeric_values) / len(numeric_values), 4)


def _write_report(report: dict[str, Any], save_path: str | None) -> None:
    """Persist the evaluation report to disk as JSON."""
    if not save_path:
        return
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


async def _persist_eval_run(report: dict[str, Any], processed: list[dict[str, Any]]) -> None:
    """
    Phase 9 — persist an evaluation report (and its retrieval results) to Postgres.

    Best-effort: any Postgres failure is logged and swallowed so evaluation
    still returns its in-memory report. No-op if Postgres was unavailable.
    """
    try:
        from app.services.eval_store import (
            finish_run,
            record_retrieval_results,
            start_run,
        )

        run_id = start_run(len(processed))
        if run_id is None:
            return

        # Aggregate token usage across all processed questions.
        tot = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for rec in processed:
            tu = rec.get("token_usage") or {}
            tot["prompt_tokens"] += int(tu.get("prompt_tokens", 0) or 0)
            tot["completion_tokens"] += int(tu.get("completion_tokens", 0) or 0)
            tot["total_tokens"] += int(tu.get("total_tokens", 0) or 0)

        finish_run(
            run_id,
            status=report.get("status", "unknown"),
            metrics=report.get("metrics", {}),
            token_usage=tot,
            average_faithfulness=(report.get("summary") or {}).get("average_faithfulness"),
            notes=report.get("error"),
        )

        # One retrieval_results row per retrieved chunk observed this run.
        items: list[dict[str, Any]] = []
        for rec in processed:
            for s in (rec.get("sources") or []):
                if not isinstance(s, dict):
                    continue
                items.append({
                    "question": rec.get("question", ""),
                    "doc_id": s.get("doc_id", ""),
                    "chunk_index": s.get("chunk_index", 0),
                    "score": s.get("rerank_score") or 0.0,
                    "source": s.get("source", ""),
                })
        record_retrieval_results(run_id, items)
    except Exception as exc:  # pragma: no cover - depends on Postgres
        logger.warning("Evaluation persistence failed (non-fatal): %s", exc)


def build_argument_parser() -> argparse.ArgumentParser:
    """Create a CLI parser for running evaluation from the shell."""
    parser = argparse.ArgumentParser(description="Run ARAP RAGAS and evaluation suite")
    parser.add_argument("--limit", type=int, default=20, help="How many questions to evaluate")
    parser.add_argument(
        "--save", type=str, default=None, help="Optional JSON path to store the report"
    )
    parser.add_argument(
        "--no-seed", action="store_true", help="Do not fall back to seeded questions"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser


async def main() -> int:
    """CLI entrypoint used by python -m evaluation.ragas_eval."""
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARNING,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from app.core.orchestrator import orchestrator

    report = await run_ragas_evaluation(
        orchestrator=orchestrator,
        limit=args.limit,
        include_seed=not args.no_seed,
        save_path=args.save,
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report.get("status") in {"ok", "partial"} else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
