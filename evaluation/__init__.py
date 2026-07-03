"""Evaluation package for ARAP.

This package exposes the RAGAS-based evaluation workflow in a compact,
clean API so the rest of the project can import evaluation helpers without
reaching into the implementation module directly.
"""

from __future__ import annotations

from .ragas_eval import (
    SEED_QA,
    build_argument_parser,
    main,
    run_ragas_evaluation,
)

__all__ = [
    "SEED_QA",
    "build_argument_parser",
    "main",
    "run_ragas_evaluation",
]
