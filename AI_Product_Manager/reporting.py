"""Evidence-based Markdown report generation with an offline fallback."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import re
import pandas as pd

try:
    from AI_Product_Manager.config import settings
except ImportError:
    from config import settings


def summarize_feedback(df: pd.DataFrame) -> dict:
    confidence = pd.to_numeric(df["prediction_confidence"], errors="coerce").dropna()
    bins = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.00001]
    labels = ["0.00-0.49", "0.50-0.59", "0.60-0.69", "0.70-0.79", "0.80-0.89", "0.90-1.00"]
    return {
        "total_reviews": int(len(df)),
        "average_rating": round(float(pd.to_numeric(df["rating"], errors="coerce").mean()), 2),
        "category_distribution": df["category"].value_counts().to_dict(),
        "sentiment_distribution": df["sentiment"].value_counts().to_dict(),
        "priority_distribution": df["priority"].value_counts().to_dict(),
        "low_confidence_reviews": int(df["needs_human_review"].sum()),
        "average_confidence": round(float(df["prediction_confidence"].mean()), 3),
        "confidence_diagnostics": {
            "minimum": round(float(confidence.min()), 3) if not confidence.empty else None,
            "maximum": round(float(confidence.max()), 3) if not confidence.empty else None,
            "mean": round(float(confidence.mean()), 3) if not confidence.empty else None,
            "histogram": pd.cut(confidence, bins=bins, labels=labels, include_lowest=True)
            .value_counts().sort_index().to_dict(),
            "llm_ambiguity_reviews": int(df.get("requires_llm_review", pd.Series(False, index=df.index)).sum()),
            "translation_or_human_reviews": int((df.get("review_route", pd.Series("", index=df.index)) == "translation_or_human").sum()),
        },
    }


def offline_report(app_name: str, context: dict) -> str:
    summary = context["feedback_summary"]
    categories = "\n".join(
        f"- {name}: {count}" for name, count in summary["category_distribution"].items()
    )
    sentiments = "\n".join(
        f"- {name}: {count}" for name, count in summary["sentiment_distribution"].items()
    )
    features = context.get("feature_ranking", [])
    feature_lines = "\n".join(
        f"{i}. {item['request']} — score {item['rice_score']}, reach {item['reach']}"
        for i, item in enumerate(features, 1)
    ) or "No feature requests met the ranking criteria."
    competitor_records = context["market_data"].get("competitors", [])
    competitors = ", ".join(
        item.get("name", str(item)) if isinstance(item, dict) else str(item)
        for item in competitor_records
    ) or "Not verified"
    competitor_section = _competitor_section(context.get("competitor_analysis", {}))
    diagnostics = summary.get("confidence_diagnostics", {})
    confidence_line = (
        f"min {diagnostics.get('minimum')}, max {diagnostics.get('maximum')}, "
        f"mean {diagnostics.get('mean')}; histogram {diagnostics.get('histogram', {})}."
    )
    return f"""# AI Product Manager Report: {app_name}

Generated: {datetime.now(timezone.utc).isoformat()}

## 1. Executive Summary

Analyzed **{summary['total_reviews']}** reviews with an average rating of
**{summary['average_rating']}**. The average classification confidence was
**{summary['average_confidence']}**.

## 2. Product and Market Position

App metadata:

```json
{json.dumps(context['market_data'].get('application_information', {}), indent=2, default=str)}
```

Known competitor candidates: {competitors}.

## 3. Customer Feedback Overview

### Categories

{categories}

### Sentiment

{sentiments}

## 4. Strengths

Praise reviews provide the strongest direct evidence of current product strengths.
Review the classified CSV for representative examples before making roadmap claims.

## 5. Weaknesses and Root-Cause Hypotheses

Prioritize repeated high-scoring Bug, Performance Issue, Complaint, and UI Issue
groups. Root causes require telemetry or engineering investigation and cannot be
proven from reviews alone.

## 6. Competitor Analysis

{competitor_section}

## 7. Prioritized Feature Opportunities

{feature_lines}

## 8. Risks and Human-Review Items

- {summary['low_confidence_reviews']} reviews are below the classification threshold.
- Confidence distribution: {confidence_line}
- {diagnostics.get('llm_ambiguity_reviews', 0)} reviews are genuine in-vocabulary
  ambiguities eligible for LLM review; {diagnostics.get('translation_or_human_reviews', 0)}
  are out-of-vocabulary and should be translated or manually triaged first.
- Low-confidence reviews remain in rating and sentiment totals. Category shares are
  confidence-weighted until a human approves a label; uninterpretable text is counted
  separately rather than silently discarded.
- App identity and competitor identities must be confirmed when not exact.
- Sentiment is lexicon-based and should be treated as directional.

## 9. Recommended 30/60/90-Day Plan

- **30 days:** validate high-priority bug groups and label low-confidence reviews.
- **60 days:** test the leading feature opportunities and verify competitor evidence.
- **90 days:** measure shipped improvements against rating, complaint, and retention trends.

## 10. Conclusion

Use the deterministic scores for triage, then apply PM and engineering judgment
before committing roadmap resources.
"""


def _themes(items: list[dict]) -> str:
    return "; ".join(
        f"{item['theme']} — {item['reviews']} reviews ({item['percentage']}%)"
        for item in items
    ) if items else "no recurring product theme met the frequency threshold"


def _percentage(value: float) -> float:
    return round(float(value) * 100, 1)


def _competitor_section(analysis: dict) -> str:
    """Render a useful report even when scraping some competitors fails."""
    if not analysis:
        return "No matched competitor-review sample was available for this run."
    own = analysis["our_product"]
    position = analysis["market_position"]
    lines = [f"Method: {analysis['method']}."]
    if position["rank"]:
        lines.extend([
            f"Market position in this sample: **#{position['rank']} of {position['out_of']}** "
            f"({position['basis']}).",
            "",
        ])
    else:
        lines.extend([
            "No competitor review sample was available, so no market position or comparative claim is reported.",
            "",
        ])
    lines.extend([
        "| Product | Reviews | Sample rating | Store rating | Store ratings | Downloads | Category | Developer | Updated | Size | Positive | Praise* | Issues* |",
        "|---|---:|---:|---:|---:|---|---|---|---|---|---:|---:|---:|",
        _competitor_row(own, ours=True),
    ])
    for item in analysis["competitors"]:
        lines.append(_competitor_row(item))
    lines.extend(["", "*Praise and issue shares are confidence-weighted category estimates, not ratings."])
    for item in analysis["competitors"]:
        if item.get("metadata", {}).get("Resolution"):
            lines.append(f"- **{item['name']} resolution:** {item['metadata']['Resolution']}")
    for comparison in analysis["comparisons"]:
        strengths = ", ".join(comparison["our_advantages"]) or "no material advantage in this sample"
        gaps = ", ".join(comparison["our_gaps"]) or "no material gap in this sample"
        lines.append(f"- Versus {comparison['competitor']}: our advantages — {strengths}; gaps to investigate — {gaps}.")
    lines.append("\n### Review themes and feature gaps")
    for item in [own, *analysis["competitors"]]:
        rates = item["category_rates"]
        lines.append(
            f"- **{item['name']}:** praised for {_themes(item['praise_themes'])}; "
            f"complaints/bugs mention {_themes(item['issue_themes'])}. "
            f"Feature requests {rates['Feature Request']}%, bugs {rates['Bug']}%, "
            f"performance {rates['Performance Issue']}%, UI {rates['UI Issue']}%, complaints {rates['Complaint']}%."
        )
    if own["queued_reviews"] or own["uninterpretable_reviews"]:
        total = own["sample_size"] or 1
        lines.extend([
            "\n### Data quality and review routing",
            f"- **Queued labels:** {own['queued_reviews']} ({_percentage(own['queued_reviews'] / total)}%). These are XGBoost predictions below the confidence threshold; in-vocabulary rows are sent to LLM categorisation when configured. They remain in sentiment totals, while their category labels are provisional until resolved.",
            f"- **Uninterpretable reviews:** {own['uninterpretable_reviews']} ({_percentage(own['uninterpretable_reviews'] / total)}%). This includes empty or emoji-only text, unsupported-language text, very short text, and text with insufficient recognised words after preprocessing. They remain in the sample count but are excluded from category/theme claims.",
        ])
    if analysis["skipped_competitors"]:
        lines.append("- Not compared this run: " + "; ".join(f"{x['name']} ({x['reason']})" for x in analysis["skipped_competitors"]) + ".")
    return "\n".join(lines)


def _competitor_row(item: dict, ours: bool = False) -> str:
    metadata = item.get("metadata", {})
    name = f"{item['name']} (ours)" if ours else item["name"]
    def value(key: str) -> str:
        return str(metadata.get(key, "—")).replace("|", "/")
    return (
        f"| {name} | {item['sample_size']} | {item['average_rating']:.2f} | "
        f"{value('Rating')} | {value('Total Ratings')} | {value('Downloads')} | {value('Category')} | "
        f"{value('Developer')} | {value('Last Updated')} | {value('Size')} | "
        f"{item['positive_rate']}% | {item['praise_rate']}% | {item['issue_rate']}% |"
    )


def save_report(app_name: str, text: str):
    settings.ensure_directories()
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", app_name).strip("_").lower()
    path = settings.report_dir / f"{safe}_{datetime.now():%Y%m%d_%H%M%S}.md"
    path.write_text(text, encoding="utf-8")
    return path


def save_pdf(app_name: str, markdown_text: str, output_dir=None):
    """Write a readable PDF without requiring an external browser."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    except ImportError:
        return None
    settings.ensure_directories()
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", app_name).strip("_").lower()
    path = (output_dir or settings.report_dir) / (
        f"{safe}_{datetime.now():%Y%m%d_%H%M%S}.pdf"
    )
    styles = getSampleStyleSheet()
    story = []
    for raw in markdown_text.splitlines():
        line = raw.strip()
        if not line:
            story.append(Spacer(1, 2 * mm))
            continue
        if line.startswith("# "):
            style, line = styles["Title"], line[2:]
        elif line.startswith("## "):
            style, line = styles["Heading2"], line[3:]
        elif line.startswith("### "):
            style, line = styles["Heading3"], line[4:]
        else:
            style = styles["BodyText"]
        clean = (
            line.replace("**", "")
            .replace("`", "")
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        story.append(Paragraph(clean, style))
    SimpleDocTemplate(
        str(path), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm
    ).build(story)
    return path
