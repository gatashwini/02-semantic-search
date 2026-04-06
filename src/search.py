"""
src/search.py

The search pipeline — combines embeddings.py and database.py
into one clean function that the Streamlit app calls.

This is the file that runs every time a user searches.
It does 3 things:
1. Convert the user query to a vector using BGE-M3
2. Find the most similar vectors in ChromaDB
3. Format and return the results cleanly

Run this file directly to test search works:
    python src/search.py
"""

import sys
import os
import numpy as np

# Add parent directory so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embeddings import get_embedding
from src.database import search as db_search


def semantic_search(query: str, n_results: int = 5) -> list:
    """
    Main search function. Takes a text query, finds similar articles.

    Pipeline:
        1. Validate query is not empty
        2. Embed query using BGE-M3 in query mode
        3. Search ChromaDB using cosine similarity
        4. Convert distances to similarity scores
        5. Return formatted list of results

    Args:
        query     (str): text typed by the user in the search box
        n_results (int): how many results to return. Default 5.

    Returns:
        list of dicts, each containing:
            rank       (int)  : position in results, 1 = most similar
            title      (str)  : article title
            text       (str)  : first 500 characters as preview
            full_text  (str)  : complete article text
            similarity (float): cosine similarity score 0.0 to 1.0
            url        (str)  : Wikipedia URL if available

    Example:
        results = semantic_search("renewable energy", n_results=3)
        for r in results:
            print(r["rank"], r["title"], r["similarity"])
    """

    # Guard against empty queries
    # strip() removes whitespace — catches queries like "   "
    if not query or not query.strip():
        return []

    # ── STEP 1: EMBED THE QUERY ───────────────────────────────────────────────
    # is_query=True uses BGE-M3 query encoding mode
    # This is different from document encoding mode used during indexing
    # Using the correct mode is important for retrieval quality
    query_embedding = get_embedding(query.strip(), is_query=True)

    # ── STEP 2: SEARCH CHROMADB ───────────────────────────────────────────────
    # Returns raw ChromaDB results dict with distances not similarities
    raw_results = db_search(query_embedding, n_results=n_results)

    # ── STEP 3: FORMAT RESULTS ────────────────────────────────────────────────
    # ChromaDB returns nested lists because it supports batch queries
    # raw_results["documents"][0] = documents for first (only) query
    # raw_results["distances"][0] = distances for first query
    # raw_results["metadatas"][0] = metadatas for first query
    documents = raw_results["documents"][0]
    distances = raw_results["distances"][0]
    metadatas = raw_results["metadatas"][0]

    formatted_results = []

    for rank, (doc, dist, meta) in enumerate(
        zip(documents, distances, metadatas), start=1
    ):
        # Convert ChromaDB cosine DISTANCE to cosine SIMILARITY
        # ChromaDB cosine distance: 0 = identical, 2 = completely opposite
        # Cosine similarity:        1 = identical, -1 = completely opposite
        # Formula: similarity = 1 - distance
        similarity = round(1 - dist, 4)

        formatted_results.append({
            "rank"      : rank,
            "title"     : meta.get("title", "Unknown Title"),
            "text"      : doc[:500],     # preview: first 500 characters
            "full_text" : doc,           # full text for expanded view
            "similarity": similarity,
            "url"       : meta.get("url", ""),
        })

    return formatted_results


def format_results_for_terminal(results: list) -> None:
    """
    Print search results in a readable format for terminal testing.

    Used when running search.py directly to verify search works
    before building the Streamlit UI.

    Args:
        results (list): output from semantic_search()
    """
    if not results:
        print("No results found.")
        return

    for r in results:
        print(f"\n  Rank {r['rank']}: {r['title']}")
        print(f"  Similarity : {r['similarity']:.4f}")
        print(f"  Preview    : {r['text'][:120]}...")
        if r["url"]:
            print(f"  URL        : {r['url']}")


# ── QUICK TEST ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Test the complete search pipeline with real queries.
    Run: python src/search.py

    Requires: indexer.py must have been run first.
    If you see 'No results' check that chroma_db folder exists.
    """

    print("="*55)
    print("TESTING search.py")
    print("="*55)

    # Check database has data
    from src.database import get_document_count
    count = get_document_count()

    if count == 0:
        print("\nERROR: No documents in database!")
        print("Run indexer.py first: python src/indexer.py")
        sys.exit(1)

    print(f"\nDatabase has {count} articles indexed. Good.")

    # Test queries — chosen to demonstrate semantic search
    # These work even if exact words are not in article titles
    test_queries = [
        "renewable energy and solar power",
        "machine learning and artificial intelligence",
        "ancient history and civilizations",
        "space exploration and planets",
        "world war and military history",
    ]

    for query in test_queries:
        print(f"\n{'='*55}")
        print(f"Query: '{query}'")
        print(f"{'='*55}")

        results = semantic_search(query, n_results=3)
        format_results_for_terminal(results)

    print(f"\n{'='*55}")
    print("search.py test complete.")
    print("If results look relevant, search is working correctly.")
    print("Next step: streamlit run src/app.py")
    print("="*55)