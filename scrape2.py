from google_play_scraper import reviews, app, Sort
import pandas as pd
import time

# =========================
# STEP 1: DEFINE APPS
# =========================

apps = {
    "Microsoft Teams": "com.microsoft.teams",
    "LinkedIn": "com.linkedin.android",
    "Canva": "com.canva.editor",
    "Uber": "com.ubercab",
    "Duolingo": "com.duolingo",
    "Paytm": "net.one97.paytm",
    "Notion": "notion.id",
    "PhonePe": "com.phonepe.app"
}

# =========================
# CONFIGURATION (IMPORTANT)
# =========================

BATCH_SIZE = 400       # how many reviews per request
MAX_REVIEWS = 7000     # total reviews per app

all_reviews = []
app_metadata = []

# =========================
# STEP 2: SCRAPE REVIEWS (BATCH MODE)
# =========================

for app_name, app_id in apps.items():
    print(f"\nScraping {app_name}...")

    collected = 0

    while collected < MAX_REVIEWS:

        remaining = MAX_REVIEWS - collected
        current_batch = min(BATCH_SIZE, remaining)

        result, _ = reviews(
            app_id,
            lang="en",
            country="us",
            sort=Sort.NEWEST,
            count=current_batch
        )

        if not result:
            break  # stop if no more data

        for r in result:
            all_reviews.append({
                "app_id": app_id,
                "app_name": app_name,
                "review_id": r.get("reviewId"),
                "user_name": r.get("userName"),

                "review_text": r.get("content"),
                "rating": r.get("score"),
                "thumbs_up": r.get("thumbsUpCount"),

                "review_date": r.get("at"),

                "reply_text": r.get("replyContent"),
                "reply_date": r.get("repliedAt"),
                

                # ML placeholders
                "sentiment": None,
                "category": None,
                "priority": None,
                "clean_text": None
            })

        collected += len(result)

        print(f"{app_name}: collected {collected}/{MAX_REVIEWS}")

        time.sleep(1)  # prevents blocking / rate limit

# =========================
# STEP 3: SCRAPE APP METADATA
# =========================

for app_name, app_id in apps.items():
    info = app(app_id)

    app_metadata.append({
        "app_id": app_id,
        "app_name": app_name,
        "title": info.get("title"),
        "description": info.get("description"),
        "summary": info.get("summary"),
        "installs": info.get("installs"),
        "ratings": info.get("ratings"),
        "score": info.get("score"),
        "reviews_count": info.get("reviews"),
        "free": info.get("free"),
        "genre": info.get("genre"),
        "developer": info.get("developer")
    })

# =========================
# STEP 4: SAVE DATA
# =========================

df_reviews = pd.DataFrame(all_reviews)
df_apps = pd.DataFrame(app_metadata)
df_reviews["review_length"] = df_reviews["review_text"].str.len()

df_reviews["word_count"] = df_reviews["review_text"].str.split().str.len()

df_reviews["has_reply"] = df_reviews["reply_text"].notna().astype(int)
df_reviews["review_month"] = pd.to_datetime(
    df_reviews["review_date"]
).dt.to_period("M")

df_reviews.to_csv("feedback_data.csv", index=False)
df_apps.to_csv("app_metadata.csv", index=False)

print("\nScraping completed successfully!")
print("Files saved:")
print("- feedback_data.csv")
print("- app_metadata.csv")
