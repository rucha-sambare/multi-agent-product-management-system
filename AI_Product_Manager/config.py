"""Central configuration for the AI Product Manager application."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    package_root: Path = PACKAGE_ROOT
    model_dir: Path = PACKAGE_ROOT / "models"
    data_dir: Path = PACKAGE_ROOT / "data"
    raw_data_dir: Path = PACKAGE_ROOT / "data" / "raw_reviews"
    run_data_dir: Path = PACKAGE_ROOT / "data" / "runs"
    report_dir: Path = PROJECT_ROOT / "reports"
    knowledge_dir: Path = PACKAGE_ROOT / "rag" / "knowledge"
    chroma_dir: Path = PACKAGE_ROOT / "data" / "chroma"
    state_db: Path = PACKAGE_ROOT / "data" / "pipeline_state.sqlite3"
    review_queue_dir: Path = PACKAGE_ROOT / "data" / "human_review"
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    llm_model: str = os.getenv("APM_LLM_MODEL", "gemini-2.5-flash")
    review_count: int = _int("APM_REVIEW_COUNT", 1000)
    competitor_review_count: int = _int("APM_COMPETITOR_REVIEW_COUNT", 300)
    max_competitors: int = _int("APM_MAX_COMPETITORS", 3)
    classification_threshold: float = _float(
        # Selected from held-out model coverage/accuracy diagnostics. 0.95
        # incorrectly routes ordinary, high-quality predictions to review.
        "APM_CLASSIFICATION_THRESHOLD", 0.60
    )
    dedupe_threshold: float = _float("APM_DEDUPE_THRESHOLD", 0.90)
    country: str = os.getenv("APM_COUNTRY", "us")
    language: str = os.getenv("APM_LANGUAGE", "en")

    def ensure_directories(self) -> None:
        for path in (
            self.raw_data_dir,
            self.run_data_dir,
            self.report_dir,
            self.knowledge_dir,
            self.chroma_dir,
            self.review_queue_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
