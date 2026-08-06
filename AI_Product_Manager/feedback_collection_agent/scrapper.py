import re
from datetime import datetime
import pandas as pd
try:
    from google_play_scraper import search, reviews, Sort
except ImportError:
    search = reviews = Sort = None
try:
    from AI_Product_Manager.config import settings
except ImportError:
    from config import settings


class AmbiguousAppError(ValueError):
    def __init__(self, app_name, candidates):
        self.app_name = app_name
        self.candidates = candidates
        names = ", ".join(f"{x['title']} ({x['app_id']})" for x in candidates)
        super().__init__(f"Could not identify '{app_name}' exactly. Candidates: {names}")


VERIFIED_APP_IDS = {
    "facebook": "com.facebook.katana",
    "instagram": "com.instagram.android",
    "spotify": "com.spotify.music",
    "google pay": "com.google.android.apps.nbu.paisa.user",
    "gpay": "com.google.android.apps.nbu.paisa.user",
    "swiggy": "in.swiggy.android",
    "amazon": "in.amazon.mShop.android.shopping",
    "flipkart": "com.flipkart.android",
    "youtube": "com.google.android.youtube",
    "zomato": "com.application.zomato",
    "microsoft teams": "com.microsoft.teams",
    "linkedin": "com.linkedin.android",
    "canva": "com.canva.editor",
    "uber": "com.ubercab",
    "duolingo": "com.duolingo",
    "paytm": "net.one97.paytm",
    "notion": "notion.id",
    "phonepe": "com.phonepe.app",
}


class FeedbackCollectionAgent:

    @staticmethod
    def _require_scraper():
        if search is None:
            raise RuntimeError(
                "Review collection requires google-play-scraper. "
                "Run: pip install -r requirements.txt"
            )

    def resolve_app(self, app_name):
        self._require_scraper()
        known_id = VERIFIED_APP_IDS.get(app_name.lower().strip())
        if known_id:
            return {"app_id": known_id, "title": app_name, "confidence": 1.0}

        results = search(
            app_name,
            lang=settings.language,
            country=settings.country
        )

        valid_apps = [
            app for app in results
            if app.get("appId")
        ]

        if not valid_apps:
            raise Exception("No app found")


        for app in valid_apps:

            if app["title"].lower() == app_name.lower():

                return {"app_id": app["appId"], "title": app["title"], "confidence": 1.0}

        candidates = [
            {"app_id": app["appId"], "title": app.get("title", app["appId"])}
            for app in valid_apps[:5]
        ]
        raise AmbiguousAppError(app_name, candidates)

    def get_app_id(self, app_name):
        return self.resolve_app(app_name)["app_id"]

    def resolve_competitor(self, app_name):
        """Best-effort competitor resolution; primary-app resolution stays strict."""
        self._require_scraper()
        results = [item for item in search(app_name, lang=settings.language, country=settings.country) if item.get("appId")]
        if not results:
            raise ValueError(f"No Play Store result for competitor '{app_name}'.")
        def installs(item):
            digits = re.sub(r"\D", "", str(item.get("installs", "0")))
            return int(digits or 0)
        # Prefer exact title, then use highest installs as an explicit fallback.
        exact = [item for item in results if item.get("title", "").casefold() == app_name.casefold()]
        selected = max(exact or results, key=installs)
        return {"app_id": selected["appId"], "title": selected.get("title"), "confidence": 0.8 if exact else 0.5}



    def collect(self, app_name, count=5000, app_id=None):
        self._require_scraper()
        app_id = app_id or self.get_app_id(app_name)


        reviews_data = []

        continuation_token = None


        while len(reviews_data) < count:


            batch, continuation_token = reviews(

                app_id,

                lang=settings.language,

                country=settings.country,

                sort=Sort.NEWEST,

                count=200,

                continuation_token=continuation_token
            )


            if not batch:
                break


            reviews_data.extend(batch)


            if continuation_token is None:
                break



        review_list = reviews_data[:count]


        data = []


        for r in review_list:

            data.append({

                "app_id": app_id,

                "app_name": app_name,

                "review_text": r["content"],

                "rating": r["score"],

                "thumbs_up": r["thumbsUpCount"],

                "review_date": r["at"],

                "reply_text": r["replyContent"],

                "reply_date": r["repliedAt"],

                "source": "play_store"

            })


        df = pd.DataFrame(data)


        settings.ensure_directories()
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", app_name).strip("_")
        file_path = settings.raw_data_dir / (
            f"{safe_name}_{datetime.now():%Y%m%d_%H%M%S}_raw_reviews.csv"
        )


        df.to_csv(
            file_path,
            index=False
        )


        print("Total reviews collected:", len(df))

        print("Saved:", file_path)


        return df



if __name__=="__main__":

    agent = FeedbackCollectionAgent()

    df = agent.collect(
        "Instagram",
        1000
    )

    print(df.head())
