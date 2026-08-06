"""End-to-end deterministic pipeline controller."""

from __future__ import annotations

from datetime import datetime
import pandas as pd

try:
    from AI_Product_Manager.config import settings
    from AI_Product_Manager.feedback_collection_agent.scrapper import FeedbackCollectionAgent
    from AI_Product_Manager.feedback_analysis_mgr.feedback_classifier import FeedbackClassifierAgent
    from AI_Product_Manager.feedback_analysis_mgr.insights import (
        add_duplicate_groups, add_priority, add_sentiment, build_feature_ranking,
    )
    from AI_Product_Manager.market_agent import MarketAgent
    from AI_Product_Manager.competitive_analysis import build_competitive_analysis
    from AI_Product_Manager.tools.app_info_tool import get_app_information
    from AI_Product_Manager.rag.retriever import LocalKnowledgeRetriever
    from AI_Product_Manager.reporting import offline_report, save_pdf, save_report, summarize_feedback
    from AI_Product_Manager.workflow_state import WorkflowStore
    from AI_Product_Manager.workflow_graph import analyze_with_graph
except ImportError:
    from config import settings
    from feedback_collection_agent.scrapper import FeedbackCollectionAgent
    from feedback_analysis_mgr.feedback_classifier import FeedbackClassifierAgent
    from feedback_analysis_mgr.insights import (
        add_duplicate_groups, add_priority, add_sentiment, build_feature_ranking,
    )
    from market_agent import MarketAgent
    from competitive_analysis import build_competitive_analysis
    from tools.app_info_tool import get_app_information
    from rag.retriever import LocalKnowledgeRetriever
    from reporting import offline_report, save_pdf, save_report, summarize_feedback
    from workflow_state import WorkflowStore
    from workflow_graph import analyze_with_graph


class FeedbackOrchestrator:
    def __init__(
        self, collector=None, classifier=None, market_agent=None, retriever=None, store=None
    ):
        self.collector = collector or FeedbackCollectionAgent()
        self.classifier = classifier
        self.market_agent = market_agent or MarketAgent()
        self.retriever = retriever or LocalKnowledgeRetriever()
        self.store = store or WorkflowStore()

    def analyze_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        classifier = self.classifier or FeedbackClassifierAgent()
        analyzed = analyze_with_graph(df, classifier)
        # This is a narrow escalation: only real model ambiguities reach Gemini.
        # Re-score category-dependent priority after any resolved label changes.
        return add_priority(self.market_agent.resolve_ambiguous_reviews(analyzed))

    def run(
        self,
        app_name: str,
        count: int | None = None,
        app_id: str | None = None,
        run_id: str | None = None,
    ) -> dict:
        settings.ensure_directories()
        run_id = run_id or self.store.create(app_name, app_id)
        try:
            self.store.update(run_id, step="collecting")
            raw = self.collector.collect(
                app_name, count=count or settings.review_count, app_id=app_id
            )
            if raw.empty:
                raise RuntimeError("No reviews were collected.")
            app_id = str(raw["app_id"].iloc[0])
            self.store.update(run_id, step="analyzing", payload={"app_id": app_id})
            analyzed = self.analyze_dataframe(raw)
            run_name = (
                f"{app_name.lower().replace(' ', '_')}_{datetime.now():%Y%m%d_%H%M%S}.csv"
            )
            data_path = settings.run_data_dir / run_name
            analyzed.to_csv(data_path, index=False)
            review_queue = analyzed[analyzed["needs_human_review"]].copy()
            queue_path = settings.review_queue_dir / f"{run_id}.csv"
            review_queue.to_csv(queue_path, index=False)

            self.store.update(run_id, step="market_research")
            market_data = self.market_agent.collect_market_data(app_name, app_id)
            # RAG remains supporting evidence for documented capabilities. Market
            # position itself comes from matched customer-review samples.
            competitor_feedback, competitor_metadata, skipped_competitors = {}, {}, []
            comparison_count = count or settings.review_count
            for competitor in market_data["competitors"][: settings.max_competitors]:
                name = competitor["name"]
                try:
                    resolved_id = competitor.get("app_id")
                    resolution_note = competitor.get("resolution_note")
                    if not resolved_id:
                        resolved = self.collector.resolve_competitor(name)
                        resolved_id = resolved["app_id"]
                        resolution_note = (
                            f"Resolved automatically to {resolved['title']} ({resolved_id}); "
                            "selected by exact title or highest Play Store installs."
                        )
                    raw_competitor = self.collector.collect(
                        name,
                        # Matched sample sizes are required for fair comparison.
                        count=comparison_count,
                        app_id=resolved_id,
                    )
                    if raw_competitor.empty:
                        raise RuntimeError("No reviews returned")
                    competitor_feedback[name] = self.analyze_dataframe(raw_competitor)
                    competitor_metadata[name] = get_app_information(
                        str(raw_competitor["app_id"].iloc[0])
                    )
                    if resolution_note:
                        competitor_metadata[name]["Resolution"] = resolution_note
                except Exception as exc:
                    # One unavailable competitor must not fail the product report.
                    skipped_competitors.append({"name": name, "reason": str(exc)})
            competitor_analysis = build_competitive_analysis(
                app_name, analyzed, competitor_feedback, skipped_competitors,
                metadata_by_app={
                    app_name: market_data["application_information"],
                    **competitor_metadata,
                },
            )
            feature_ranking = self.market_agent.add_feature_rationales(
                build_feature_ranking(analyzed)
            )
            context = {
                "app_name": app_name,
                "app_id": app_id,
                "feedback_summary": summarize_feedback(analyzed),
                "feature_ranking": feature_ranking,
                "market_data": market_data,
                "competitor_analysis": competitor_analysis,
                "retrieved_evidence": self.retriever.retrieve(
                    f"{app_name} competitors product features release notes"
                ),
            }
            self.store.update(run_id, step="reporting")
            report = self.market_agent.generate_llm_report(context)
            report_mode = "llm"
            if not report:
                report = offline_report(app_name, context)
                report_mode = "offline"
            report_path = save_report(app_name, report)
            pdf_path = save_pdf(app_name, report)
            result = {
                "run_id": run_id,
                "app_name": app_name,
                "app_id": app_id,
                "data_path": str(data_path),
                "review_queue_path": str(queue_path),
                "report_path": str(report_path),
                "pdf_path": str(pdf_path) if pdf_path else None,
                "report_mode": report_mode,
                "summary": context["feedback_summary"],
            }
            status = "needs_review" if len(review_queue) else "completed"
            self.store.update(run_id, status=status, step="finished", payload=result)
            return result
        except Exception as exc:
            self.store.update(
                run_id, status="failed", step="failed", error=f"{type(exc).__name__}: {exc}"
            )
            raise
