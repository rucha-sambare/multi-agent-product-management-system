"""Clean Streamlit interface for the AI Product Manager pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from AI_Product_Manager.feedback_collection_agent.scrapper import AmbiguousAppError
from AI_Product_Manager.orchestar_agent.orchestrator import FeedbackOrchestrator


st.set_page_config(
    page_title="ProductLens",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stApp {
        background: #f7f8fa;
        color: #182230;
    }
    .block-container {Share request for ‘project.zip’
Inbox
Search for all messages with label Inbox
Remove label Inbox from this conversation
￼
Sanskruti Pawarpatil (via Google Drive)
Wed 29 Jul, 21:44 (11 hours ago)
Share an item? Sanskruti Pawarpatil (sanskrutipawarpatil@gmail.com) is requesting access to the following item:project.zip Manage sharingGoogle LLC, 1600 Amphit
￼
Rucha Sambare
￼
Wed 29 Jul, 22:27 (10 hours ago)
On Wed, 29 Jul 2026 at 21:44, Sanskruti Pawarpatil (via Google Drive) <drive-shares-dm-noreply@google.com> wrote: Share an item? Sanskruti Pawarpatil (sanskruti
￼
Rucha Sambare <ruchasambare1@gmail.com>
￼
Wed 29 Jul, 22:30 (10 hours ago)
￼
Add reaction
￼
Reply
￼
More
to Sanskruti
￼
￼

        max-width: 1080px;
        padding-top: 3rem;
        padding-bottom: 4rem;
    }
    #MainMenu, footer, header {visibility: hidden;}
    .brand {
        color: #2563eb;
        font-size: .78rem;
        font-weight: 750;
        letter-spacing: .12em;
        margin-bottom: .65rem;
        text-transform: uppercase;
    }
    .hero-title {
        color: #101828;
        font-size: clamp(2rem, 5vw, 3.15rem);
        font-weight: 760;
        letter-spacing: -.045em;
        line-height: 1.05;
        margin: 0;
    }
    .hero-copy {
        color: #667085;
        font-size: 1.05rem;
        line-height: 1.65;
        margin: .9rem 0 2rem;
        max-width: 650px;
    }
    div[data-testid="stForm"] {
        background: #ffffff;
        border: 1px solid #e4e7ec;
        border-radius: 16px;
        box-shadow: 0 8px 28px rgba(16, 24, 40, .06);
        padding: 1.3rem 1.4rem .45rem;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e4e7ec;
        border-radius: 14px;
        padding: 1rem 1.1rem;
    }
    div[data-testid="stMetricLabel"] {color: #667085;}
    div[data-testid="stMetricValue"] {
        color: #101828;
        font-size: 1.7rem;
    }
    .section-title {
        color: #101828;
        font-size: 1.15rem;
        font-weight: 700;
        margin: 2.2rem 0 .2rem;
    }
    .section-copy {
        color: #667085;
        font-size: .9rem;
        margin-bottom: 1rem;
    }
    .status {
        background: #ecfdf3;
        border: 1px solid #abefc6;
        border-radius: 12px;
        color: #067647;
        margin: 1.4rem 0;
        padding: .8rem 1rem;
    }
    .stButton > button, .stDownloadButton > button {
        border-radius: 9px;
        font-weight: 650;
        min-height: 2.8rem;
    }
    .stButton > button[kind="primary"] {
        background: #2563eb;
        border-color: #2563eb;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def show_results(result: dict) -> None:
    summary = result["summary"]
    st.markdown(
        f'<div class="status">Analysis complete for <strong>{result["app_name"]}</strong></div>',
        unsafe_allow_html=True,
    )

    one, two, three, four = st.columns(4)
    one.metric("Reviews", f"{summary['total_reviews']:,}")
    two.metric("Average rating", f"{summary['average_rating']:.1f} / 5")
    three.metric("High priority", summary["priority_distribution"].get("High", 0))
    four.metric("Needs review", summary["low_confidence_reviews"])

    st.markdown('<div class="section-title">What customers are saying</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">A practical breakdown of the most common feedback types.</div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.35, 1], gap="large")
    with left:
        categories = pd.Series(summary["category_distribution"], name="Reviews")
        st.bar_chart(categories, color="#2563eb", horizontal=True)
    with right:
        st.dataframe(
            pd.DataFrame(
                {
                    "Sentiment": summary["sentiment_distribution"].keys(),
                    "Reviews": summary["sentiment_distribution"].values(),
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

    st.markdown('<div class="section-title">Your report</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Download the evidence, recommendations, and prioritized next steps.</div>',
        unsafe_allow_html=True,
    )
    markdown_path = Path(result["report_path"])
    pdf_path = Path(result["pdf_path"]) if result.get("pdf_path") else None
    first, second, _ = st.columns([1, 1, 2])
    with first:
        st.download_button(
            "Download report",
            markdown_path.read_text(encoding="utf-8"),
            file_name=markdown_path.name,
            mime="text/markdown",
            use_container_width=True,
        )
    with second:
        if pdf_path and pdf_path.exists():
            st.download_button(
                "Download PDF",
                pdf_path.read_bytes(),
                file_name=pdf_path.name,
                mime="application/pdf",
                use_container_width=True,
            )
    with st.expander("Read report here"):
        st.markdown(markdown_path.read_text(encoding="utf-8"))


st.markdown('<div class="brand">ProductLens</div>', unsafe_allow_html=True)
st.markdown(
    '<h1 class="hero-title">Turn app reviews into<br>clear product decisions.</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero-copy">Analyze recent Google Play feedback, find the issues that matter, '
    "and leave with a prioritized product report.</div>",
    unsafe_allow_html=True,
)

with st.form("analysis_form"):
    app_col, count_col = st.columns([3, 1])
    with app_col:
        app_name = st.text_input(
            "Google Play app",
            placeholder="e.g. Instagram",
            help="Use the exact public app name.",
        )
    with count_col:
        review_count = st.selectbox("Reviews", [ 500, 1000 ,2000,3000,4000,5000], index=1)
    with st.expander("Advanced"):
        app_id = st.text_input(
            "Package ID override",
            placeholder="com.example.app",
            help="Only needed when the app name is ambiguous.",
        )
    submitted = st.form_submit_button(
        "Analyze customer feedback",
        type="primary",
        use_container_width=True,
    )

if submitted:
    if not app_name.strip():
        st.warning("Enter an app name to continue.")
    else:
        try:
            with st.spinner("Collecting reviews and building your report…"):
                st.session_state["analysis_result"] = FeedbackOrchestrator().run(
                    app_name.strip(),
                    review_count,
                    app_id.strip() or None,
                )
        except AmbiguousAppError as exc:
            st.warning("We found more than one possible app. Add its package ID under Advanced.")
            st.dataframe(exc.candidates, hide_index=True, use_container_width=True)
        except Exception as exc:
            st.error(f"Analysis could not be completed: {exc}")

if result := st.session_state.get("analysis_result"):
    show_results(result)

