"""LangGraph implementation of the feedback-analysis state machine."""

from __future__ import annotations

from typing import Any, TypedDict
import pandas as pd

from AI_Product_Manager.config import settings
from AI_Product_Manager.feedback_analysis_mgr.insights import (
    add_duplicate_groups,
    add_priority,
    add_sentiment,
)


class AnalysisState(TypedDict, total=False):
    reviews: Any
    classifier: Any
    requires_human_review: bool


def build_analysis_graph():
    try:
        from langgraph.graph import END, StateGraph
    except ImportError:
        return None

    def classify(state):
        return {"reviews": state["classifier"].classify(state["reviews"])}

    def sentiment(state):
        return {"reviews": add_sentiment(state["reviews"])}

    def dedupe(state):
        return {
            "reviews": add_duplicate_groups(
                state["reviews"], settings.dedupe_threshold
            )
        }

    def confidence_gate(state):
        return {
            "requires_human_review": bool(
                state["reviews"]["needs_human_review"].any()
            )
        }

    def priority(state):
        return {"reviews": add_priority(state["reviews"])}

    graph = StateGraph(AnalysisState)
    graph.add_node("classify", classify)
    graph.add_node("sentiment", sentiment)
    graph.add_node("deduplicate", dedupe)
    graph.add_node("confidence_gate", confidence_gate)
    graph.add_node("priority", priority)
    graph.set_entry_point("classify")
    graph.add_edge("classify", "sentiment")
    graph.add_edge("sentiment", "deduplicate")
    graph.add_edge("deduplicate", "confidence_gate")
    graph.add_edge("confidence_gate", "priority")
    graph.add_edge("priority", END)
    return graph.compile()


def analyze_with_graph(df: pd.DataFrame, classifier) -> pd.DataFrame:
    graph = build_analysis_graph()
    if graph is None:
        result = classifier.classify(df)
        result = add_sentiment(result)
        result = add_duplicate_groups(result, settings.dedupe_threshold)
        return add_priority(result)
    state = graph.invoke({"reviews": df, "classifier": classifier})
    return state["reviews"]

