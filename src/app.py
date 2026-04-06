"""
src/app.py

Streamlit web application for the semantic search engine.
This is what users see and interact with in the browser.

Run: streamlit run src/app.py
Opens automatically at: http://localhost:8501

Requires: indexer.py must have been run first.
"""

import streamlit as st
import sys
import os

# Add parent directory so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.search import semantic_search
from src.database import get_document_count, collection_exists

# ── PAGE CONFIGURATION ────────────────────────────────────────────────────────
# Must be the very first Streamlit command in the file
# Sets the browser tab title, icon, and layout width
st.set_page_config(
    page_title="Semantic Search Engine",
    page_icon="🔍",
    layout="wide",           # use full browser width
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
# Small style tweaks to make the app look more professional
st.markdown("""
    <style>
    .result-card {
        background-color: #f8f9fa;
        border-left: 4px solid #2E75B6;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .similarity-high   { color: #1E6B4A; font-weight: bold; }
    .similarity-medium { color: #C55A11; font-weight: bold; }
    .similarity-low    { color: #C00000; font-weight: bold; }
    .model-badge {
        background-color: #1F4E79;
        color: white;
        padding: 3px 10px;
        border-radius: 10px;
        font-size: 12px;
    }
    </style>
""", unsafe_allow_html=True)


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Settings")

    # Number of results slider
    # min=1, max=10, default=5
    n_results = st.slider(
        "Number of results",
        min_value=1,
        max_value=10,
        value=5,
        help="How many articles to show per search"
    )

    st.markdown("---")

    # Show database status
    st.subheader("Database Status")

    if collection_exists():
        count = get_document_count()
        st.success(f"{count} articles indexed")
    else:
        st.error("No articles indexed.")
        st.warning("Run: python src/indexer.py")

    st.markdown("---")

    # Model information
    st.subheader("Model Info")
    st.markdown('<span class="model-badge">BGE-M3</span>',
                unsafe_allow_html=True)
    st.caption("Dimensions: 1024")
    st.caption("Similarity: Cosine")
    st.caption("Languages: 100+")
    st.caption("GDPR: Fully local")

    st.markdown("---")

    # About section
    st.subheader("About")
    st.caption(
        "Semantic search finds articles by MEANING "
        "not by keyword matching. "
        "Search in any language — BGE-M3 understands "
        "English, German, French, Dutch and 100+ more."
    )


# ── MAIN PAGE ─────────────────────────────────────────────────────────────────
st.title("Semantic Search Engine")
st.caption(
    "Powered by BGE-M3 embeddings and ChromaDB  |  "
    "Finds articles by meaning, not just keywords"
)

st.markdown("---")

# ── SEARCH INPUT ──────────────────────────────────────────────────────────────
# Two columns: wide text input + narrow search button
col_input, col_button = st.columns([5, 1])

with col_input:
    query = st.text_input(
        label="Search",
        placeholder="e.g. renewable energy, machine learning, ancient Rome...",
        label_visibility="collapsed"  # hide label, placeholder is enough
    )

with col_button:
    search_clicked = st.button(
        "Search",
        type="primary",
        use_container_width=True
    )

# ── EXAMPLE QUERIES ───────────────────────────────────────────────────────────
# Clickable example buttons so users can try the app immediately
st.caption("Try an example:")

# 4 example buttons in a row
ex1, ex2, ex3, ex4 = st.columns(4)

with ex1:
    if st.button("renewable energy"):
        query = "renewable energy"
        search_clicked = True

with ex2:
    if st.button("machine learning"):
        query = "machine learning"
        search_clicked = True

with ex3:
    if st.button("ancient civilizations"):
        query = "ancient civilizations"
        search_clicked = True

with ex4:
    if st.button("space exploration"):
        query = "space exploration"
        search_clicked = True

# ── SEARCH RESULTS ────────────────────────────────────────────────────────────
if query and query.strip():

    # Show spinner while searching
    with st.spinner(f"Searching for: {query}"):
        results = semantic_search(query.strip(), n_results=n_results)

    if results:
        st.success(f"Found {len(results)} results for: **{query}**")
        st.markdown("---")

        # Display each result as a card
        for result in results:

            score = result["similarity"]

            # Choose colour based on similarity score
            # Green = high confidence, Orange = medium, Red = low
            if score >= 0.75:
                score_color = "similarity-high"
                score_label = "High match"
            elif score >= 0.55:
                score_color = "similarity-medium"
                score_label = "Medium match"
            else:
                score_color = "similarity-low"
                score_label = "Low match"

            # Result card layout: rank + content + score
            col_rank, col_content, col_score = st.columns([0.5, 7, 2.5])

            with col_rank:
                # Show rank number large
                st.markdown(f"### {result['rank']}")

            with col_content:
                # Title as bold text
                st.markdown(f"**{result['title']}**")

                # Preview text — first 500 chars
                st.write(result["text"])

                # Show URL if available
                if result.get("url"):
                    st.caption(f"Source: {result['url']}")

                # Expandable section for full article text
                with st.expander("Show full text"):
                    st.write(result["full_text"])

            with col_score:
                # Similarity score displayed as metric
                st.metric(
                    label="Similarity",
                    value=f"{score:.3f}",
                )
                # Colour-coded match label
                st.markdown(
                    f'<span class="{score_color}">{score_label}</span>',
                    unsafe_allow_html=True
                )

            # Divider between results
            st.markdown("---")

    else:
        st.warning(
            "No results found. "
            "Try a different query or check that indexer.py has been run."
        )

elif not collection_exists():
    # Show instructions if database is empty
    st.info(
        "Database is empty. "
        "Run the indexer first: **python src/indexer.py**"
    )

else:
    # Show instructions when app first loads
    st.info(
        "Type a search query above and press Search. "
        "Try: 'renewable energy', 'machine learning', 'ancient Rome'"
    )

    # Show some stats about what is indexed
    if collection_exists():
        count = get_document_count()
        st.markdown(f"""
        **What is indexed:**
        - {count} Wikipedia articles
        - Embedded with BGE-M3 (1024 dimensions each)
        - Searchable in any language

        **How it works:**
        - Your query is converted to a 1024-dimensional vector
        - ChromaDB finds the 5 most similar article vectors
        - Results ranked by cosine similarity score
        """)