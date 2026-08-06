"""Command-line entry point."""

from __future__ import annotations

import argparse
import json

from AI_Product_Manager.orchestar_agent.orchestrator import FeedbackOrchestrator
from AI_Product_Manager.feedback_collection_agent.scrapper import AmbiguousAppError


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Google Play customer feedback.")
    parser.add_argument("app_name", nargs="?", help="Exact Google Play app title")
    parser.add_argument("--count", type=int, default=None, help="Number of recent reviews")
    parser.add_argument("--app-id", default=None, help="Verified package ID override")
    args = parser.parse_args()
    app_name = args.app_name or input("Enter exact app name: ").strip()
    try:
        result = FeedbackOrchestrator().run(app_name, args.count, args.app_id)
    except AmbiguousAppError as exc:
        print(str(exc))
        return 2
    except Exception as exc:
        print(f"Analysis failed: {exc}")
        return 1
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
