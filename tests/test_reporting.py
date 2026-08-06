from AI_Product_Manager.reporting import offline_report, save_pdf


def test_offline_report_and_pdf(tmp_path):
    context = {
        "feedback_summary": {
            "total_reviews": 2,
            "average_rating": 3.0,
            "category_distribution": {"Bug": 1, "Praise": 1},
            "sentiment_distribution": {"Negative": 1, "Positive": 1},
            "priority_distribution": {"High": 1, "Low": 1},
            "low_confidence_reviews": 1,
            "average_confidence": 0.8,
        },
        "feature_ranking": [],
        "market_data": {
            "application_information": {"App Name": "Example"},
            "competitors": [{"name": "Competitor", "verified": False}],
        },
        "competitor_analysis": {
            "method": "matched samples",
            "our_product": {"name": "Example", "sample_size": 2, "average_rating": 3.0, "positive_rate": 50, "praise_rate": 50, "issue_rate": 50, "praise_themes": [], "queued_reviews": 1, "uninterpretable_reviews": 0, "metadata": {}, "category_rates": {"Feature Request": 0, "Bug": 50, "Performance Issue": 0, "UI Issue": 0, "Complaint": 0}, "issue_themes": []},
            "competitors": [], "comparisons": [], "skipped_competitors": [],
            "market_position": {"rank": 1, "out_of": 1, "basis": "sample only"},
        },
    }
    report = offline_report("Example", context)
    assert "AI Product Manager Report" in report
    assert "Market position in this sample" in report
    pdf = save_pdf("Example", report, tmp_path)
    assert pdf and pdf.exists() and pdf.stat().st_size > 0
