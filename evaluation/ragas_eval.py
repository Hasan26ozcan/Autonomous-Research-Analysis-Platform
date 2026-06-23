"""
RAGAS Evaluation Suite
=======================
Measures four standard RAG quality dimensions (2026 standard):

  faithfulness      — Are answer claims supported by the retrieved context?
                      Computed via NLI entailment between answer sentences and chunks.

  answer_relevancy  — Is the answer responsive to the question?
                      Computed by embedding the answer, reverse-generating questions,
                      and measuring cosine similarity to the original.

  context_precision — Are the retrieved chunks actually useful?
                      High precision = fewer irrelevant chunks retrieved.

  context_recall    — Did retrieval capture all the information needed?
                      Measured against ground-truth answer coverage.

Test set:
  Loaded from PostgreSQL query_history table (populated during normal use).
  Falls back to built-in seed questions if the table is empty.

Reference:
  RAGAS: "RAGAS: Automated Evaluation of Retrieval Augmented Generation"
  (Es et al., 2023) — arXiv:2309.15217
"""
from __future__ import annotations
import asyncio
import logging

logger = logging.getLogger(__name__)

SEED_QA = [
    {
        "question": "What is the main contribution of this document?",
        "ground_truth": "The document presents its main contribution in the introduction or abstract.",
    },
    {
        "question": "What methodology is described in this document?",
        "ground_truth": "The methodology section describes the research or engineering approach used.",
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
]


async def _load_test_set(limit: int = 20) -> list[dict]:
    """Load real user Q&A pairs from PostgreSQL query_history."""
    try:
        import asyncpg
        from app.core.config import settings
        conn = await asyncpg.connect(settings.postgres_url)
        rows = await conn.fetch(
            """
            SELECT question, answer FROM query_history
            WHERE answer IS NOT NULL AND LENGTH(answer) > 20
            ORDER BY created_at DESC
            LIMIT $1
            """,
            limit,
        )
        await conn.close()
        if rows:
            return [{"question": r["question"], "ground_truth": r["answer"]} for r in rows]
    except Exception as e:
        logger.warning("Could not load test set from DB (using seed): %s", e)
    return SEED_QA


async def run_ragas(orchestrator) -> dict:
    """
    Execute RAGAS evaluation. Returns a dict of metric scores.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from datasets import Dataset
    except ImportError:
        return {
            "error": "Install RAGAS: pip install ragas datasets",
            "faithfulness": None,
            "answer_relevancy": None,
            "context_precision": None,
            "context_recall": None,
        }

    test_set = await _load_test_set()
    logger.info("RAGAS: running evaluation on %d questions", len(test_set))

    questions, answers, contexts, ground_truths = [], [], [], []

    for item in test_set:
        try:
            result = await orchestrator.query(
                question=item["question"],
                session_id="ragas-eval",
                user_id="ragas",
            )
            questions.append(item["question"])
            answers.append(result["answer"])
            contexts.append([s["text"] for s in result["sources"]])
            ground_truths.append(item["ground_truth"])
        except Exception as e:
            logger.warning("RAGAS: query failed for '%s': %s", item["question"][:50], e)

    if not questions:
        return {"error": "No evaluation questions could be processed."}

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    scores = {
        "faithfulness": round(float(result["faithfulness"]), 4),
        "answer_relevancy": round(float(result["answer_relevancy"]), 4),
        "context_precision": round(float(result["context_precision"]), 4),
        "context_recall": round(float(result["context_recall"]), 4),
        "num_questions": len(questions),
    }
    logger.info("RAGAS scores: %s", scores)
    return scores


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from app.core.orchestrator import orchestrator as orc
    scores = asyncio.run(run_ragas(orc))
    print(scores)
