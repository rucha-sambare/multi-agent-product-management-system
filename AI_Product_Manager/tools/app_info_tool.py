try:
    from google_play_scraper import app
except ImportError:
    app = None


def get_app_information(app_id):
    """
    Fetch application information from Google Play Store.

    Parameters
    ----------
    app_id : str
        Example:
        com.spotify.music
        com.microsoft.teams

    Returns
    -------
    dict
    """

    try:
        if app is None:
            raise RuntimeError(
                "App metadata requires google-play-scraper. "
                "Run: pip install -r requirements.txt"
            )

        data = app(
            app_id,
            lang="en",
            country="us"
        )

        return {

            "App Name": data.get("title"),

            "Developer": data.get("developer"),

            "Category": data.get("genre"),

            "Rating": data.get("score"),

            "Total Ratings": data.get("ratings"),

            "Downloads": data.get("installs"),

            "Released": data.get("released"),

            "Last Updated": data.get("updated"),

            "Version": data.get("version"),

            "Size": data.get("size"),

            "Description": data.get("description")

        }

    except Exception as e:

        return {

            "Error": str(e)

        }
