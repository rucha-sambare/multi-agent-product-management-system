"""
Competitor Tool

Returns major competitors of an application.
"""

competitor_database = {

    "spotify": [
        "YouTube Music",
        "Apple Music",
        "Amazon Music",
        "JioSaavn",
        "Gaana"
    ],

    "microsoft teams": [
        "Slack",
        "Zoom",
        "Google Meet",
        "Cisco Webex"
    ],

    "instagram": [
        "TikTok",
        "Snapchat",
        "Facebook"
    ],

    "facebook": [
        "Instagram",
        "TikTok",
        "Snapchat",
    ],

    "youtube": [
        "Netflix",
        "Disney+",
        "Instagram Reels"
    ],

    "uber": [
        "Ola",
        "Lyft",
        "Rapido"
    ],

    "gpay": [
        "PhonePe",
        "Paytm",
        "Amazon Pay"
    ],

    "phonepe": [
        "Google Pay",
        "Paytm",
        "Amazon Pay"
    ],

    "paytm": [
        "PhonePe",
        "Google Pay",
        "Amazon Pay"
    ],

    "flipkart": [
        "Amazon",
        "Myntra",
        "Meesho"
    ],

    "amazon": [
        "Flipkart",
        "eBay",
        "Walmart"
    ],

    "swiggy": [
        "Zomato",
        "Blinkit",
        "Zepto"
    ],

    "zomato": [
        "Swiggy",
        "Blinkit",
        "Zepto"
    ],

    "linkedin": [
        "Indeed",
        "Naukri",
        "Glassdoor"
    ],

    "notion": [
        "Evernote",
        "Microsoft OneNote",
        "ClickUp"
    ],

    "duolingo": [
        "Babbel",
        "Busuu",
        "Memrise"
    ],

    "canva": [
        "Adobe Express",
        "Figma",
        "VistaCreate"
    ]

}

# Exact package IDs avoid silently comparing against a similarly named app.
VERIFIED_COMPETITOR_IDS = {
    "tiktok": "com.zhiliaoapp.musically", "snapchat": "com.snapchat.android",
    "facebook": "com.facebook.katana", "youtube music": "com.google.android.apps.youtube.music",
    "amazon music": "com.amazon.mp3", "jiosaavn": "com.jio.media.jiobeats",
    "slack": "com.Slack", "zoom": "us.zoom.videomeetings",
    "google meet": "com.google.android.apps.tachyon", "lyft": "me.lyft.android",
    "phonepe": "com.phonepe.app", "paytm": "net.one97.paytm",
    "flipkart": "com.flipkart.android", "myntra": "com.myntra.android",
    "zomato": "com.application.zomato", "swiggy": "in.swiggy.android",
    "linkedin": "com.linkedin.android", "indeed": "com.indeed.android.jobsearch",
    "evernote": "com.evernote", "microsoft onenote": "com.microsoft.office.onenote",
    "duolingo": "com.duolingo", "canva": "com.canva.editor",
    "adobe express": "com.adobe.spark.post", "figma": "com.figma.mirror",
    "disney+": "com.disney.disneyplus",
    # Reels is an Instagram feature, not an independently installable app.
    "instagram reels": "com.instagram.android",
}


def get_competitors(app_name):
    """
    Returns competitor list for a given app.
    """

    app_name = app_name.lower().strip()

    return competitor_database.get(app_name, [])


def get_competitor_records(app_name):
    """Return explicit provenance so seed suggestions are never presented as verified."""
    return [
        {
            "name": name,
            "source": "offline_seed_dictionary",
            "app_id": VERIFIED_COMPETITOR_IDS.get(name.casefold()),
            "verified": name.casefold() in VERIFIED_COMPETITOR_IDS,
            "verification_required": name.casefold() not in VERIFIED_COMPETITOR_IDS,
            "resolution_note": (
                "Reels is a feature within Instagram; comparison uses Instagram reviews."
                if name.casefold() == "instagram reels" else None
            ),
        }
        for name in get_competitors(app_name)
    ]
