"""Collect source-attributed competitor descriptions for the RAG index."""

from __future__ import annotations

from datetime import datetime, timezone
import re
from google_play_scraper import app, search

from AI_Product_Manager.config import settings
from AI_Product_Manager.tools.competitor_tool import get_competitors

VERIFIED_COMPETITOR_IDS = {
    "TikTok": "com.zhiliaoapp.musically",
    "Snapchat": "com.snapchat.android",
    "Facebook": "com.facebook.katana",
    "YouTube Music": "com.google.android.apps.youtube.music",
    "Amazon Music": "com.amazon.mp3",
    "JioSaavn": "com.jio.media.jiobeats",
    "Slack": "com.Slack",
    "Zoom": "us.zoom.videomeetings",
    "Google Meet": "com.google.android.apps.tachyon",
    "Lyft": "me.lyft.android",
    "PhonePe": "com.phonepe.app",
    "Paytm": "net.one97.paytm",
    "Flipkart": "com.flipkart.android",
    "Myntra": "com.myntra.android",
    "Zomato": "com.application.zomato",
    "Swiggy": "in.swiggy.android",
    "LinkedIn": "com.linkedin.android",
    "Indeed": "com.indeed.android.jobsearch",
    "Evernote": "com.evernote",
    "Microsoft OneNote": "com.microsoft.office.onenote",
    "Duolingo": "com.duolingo",
    "Canva": "com.canva.editor",
    "Adobe Express": "com.adobe.spark.post",
    "Figma": "com.figma.mirror",
}


def collect_competitor_knowledge(app_name: str) -> list[dict]:
    settings.ensure_directories()
    results = []
    for competitor in get_competitors(app_name):
        verified_id = VERIFIED_COMPETITOR_IDS.get(competitor)
        candidates = [] if verified_id else [
            item for item in search(
                competitor, lang=settings.language, country=settings.country
            ) if item.get("appId")
            and item.get("title", "").casefold() == competitor.casefold()
        ]
        if not verified_id and len(candidates) != 1:
            results.append(
                {
                    "name": competitor,
                    "verified": False,
                    "reason": f"Expected one exact match; found {len(candidates)}",
                }
            )
            continue
        app_id = verified_id or candidates[0]["appId"]
        details = app(app_id, lang=settings.language, country=settings.country)
        if not details.get("title", "").casefold().startswith(competitor.casefold()):
            results.append(
                {
                    "name": competitor,
                    "verified": False,
                    "reason": f"Package title mismatch: {details.get('title')}",
                }
            )
            continue
        source_url = f"https://play.google.com/store/apps/details?id={app_id}"
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", competitor).strip("_").lower()
        path = settings.knowledge_dir / f"{safe}.md"
        path.write_text(
            f"""# {details.get('title', competitor)}

Source: {source_url}
Retrieved: {datetime.now(timezone.utc).isoformat()}
Package ID: {app_id}
Developer: {details.get('developer')}

## Description

{details.get('description', '')}

## Recent changes

{details.get('recentChanges', '')}
""",
            encoding="utf-8",
        )
        results.append(
            {
                "name": competitor,
                "app_id": app_id,
                "verified": True,
                "source": source_url,
                "document": str(path),
            }
        )
    try:
        from AI_Product_Manager.rag.vector_store import ChromaKnowledgeStore
        ChromaKnowledgeStore().ingest_directory()
    except (ImportError, RuntimeError):
        pass
    return results
