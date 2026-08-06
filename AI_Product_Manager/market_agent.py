"""Market research and optional Gemini-backed report synthesis."""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

try:
    from AI_Product_Manager.config import settings
    from AI_Product_Manager.tools.app_info_tool import get_app_information
    from AI_Product_Manager.tools.competitor_tool import get_competitor_records
except ImportError:
    from config import settings
    from tools.app_info_tool import get_app_information
    from tools.competitor_tool import get_competitor_records


class MarketAgent:
    def __init__(self, api_key: str | None = None, model_name: str | None = None):
        self.api_key = api_key or settings.google_api_key
        self.model_name = model_name or settings.llm_model

    def collect_market_data(self, app_name: str, app_id: str) -> dict:
        return {
            "application_information": get_app_information(app_id),
            "competitors": get_competitor_records(app_name),
        }

    def resolve_ambiguous_reviews(self, reviews: pd.DataFrame) -> pd.DataFrame:
        """Use the LLM only for in-vocabulary, low-confidence classifications.

        Out-of-vocabulary rows are intentionally left for translation/human review;
        asking the LLM to guess category labels for empty or foreign-language text
        would hide a data-quality issue.
        """
        result = reviews.copy()
        selected = result.index[result.get("requires_llm_review", pd.Series(False, index=result.index))]
        if not self.api_key or not len(selected):
            return result
        categories = ["Bug", "Complaint", "Feature Request", "Performance Issue", "Praise", "UI Issue"]
        try:
            from google import genai
            client = genai.Client(api_key=self.api_key)
            for start in range(0, len(selected), 100):
                indexes = selected[start : start + 100]
                payload = [{"row": int(i), "review": str(result.at[i, "review_text"])} for i in indexes]
                prompt = (
                    "Classify each app review into exactly one category: " + ", ".join(categories)
                    + ". Return JSON only as [{\"row\": integer, \"category\": string}].\n"
                    + json.dumps(payload, ensure_ascii=False)
                )
                response = client.models.generate_content(model=self.model_name, contents=prompt)
                parsed = json.loads(response.text.strip().removeprefix("```json").removesuffix("```"))
                for item in parsed:
                    row, category = item.get("row"), item.get("category")
                    if row in indexes and category in categories:
                        result.at[row, "category"] = category
                        result.at[row, "prediction_confidence"] = 1.0
                        result.at[row, "needs_human_review"] = False
                        result.at[row, "requires_llm_review"] = False
                        result.at[row, "review_route"] = "llm_resolved"
                        result.at[row, "classification_source"] = "llm_review"
        except Exception:
            # The original XGBoost prediction and queue remain intact on any API failure.
            return reviews
        return result

    def generate_llm_report(self, context: dict) -> str | None:
        if not self.api_key:
            return None
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(
                "Gemini support requires the google-genai package."
            ) from exc

        prompt_path = Path(__file__).parent / "prompts" / "report_prompt.txt"
        template = prompt_path.read_text(encoding="utf-8")
        prompt = template.replace(
            "{{CONTEXT}}", json.dumps(context, indent=2, default=str)
        )
        client = genai.Client(api_key=self.api_key)
        response = client.models.generate_content(model=self.model_name, contents=prompt)
        return response.text

    def add_feature_rationales(self, features: list[dict]) -> list[dict]:
        """Add concise rationale; use Gemini when configured, otherwise deterministic evidence."""
        enriched = []
        for feature in features:
            item = dict(feature)
            item["rationale"] = (
                f"Ranked from {int(item['reach'])} grouped request(s), average "
                f"priority {item['average_priority']:.2f}, and classification "
                f"confidence {item['confidence']:.2f}. Validate user reach and effort "
                "with analytics and engineering before roadmap commitment."
            )
            enriched.append(item)
        if not self.api_key or not enriched:
            return enriched
        try:
            from google import genai
            client = genai.Client(api_key=self.api_key)
            prompt = (
                "Write one evidence-bounded product rationale for every feature. "
                "Return JSON only and preserve all original fields. Do not invent metrics.\n"
                + json.dumps(enriched, default=str)
            )
            response = client.models.generate_content(model=self.model_name, contents=prompt)
            parsed = json.loads(response.text.strip().removeprefix("```json").removesuffix("```"))
            return parsed if isinstance(parsed, list) else enriched
        except Exception:
            return enriched
