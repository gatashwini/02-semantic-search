"""
src/indexer.py

Loads Wikipedia articles from HuggingFace and indexes them
into ChromaDB as embedding vectors.

RUN THIS ONCE before starting the search app.
After running, all vectors are saved to ./chroma_db/ on disk.
They persist between sessions — no need to re-index every time.

To re-index from scratch:
    Delete the ./chroma_db/ folder
    Run this file again

Run: python src/indexer.py
"""

from datasets import load_dataset
import sys
import os

# Add parent directory to path so we can import our own modules
# sys.path.insert(0, '..') tells Python to look in the parent folder
# This is needed because indexer.py is inside src/ folder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embeddings import get_embeddings_batch
from src.database import add_documents, collection_exists, get_document_count, delete_collection

# ── SETTINGS ──────────────────────────────────────────────────────────────────
# Number of Wikipedia articles to index.
# 500 is a good balance:
# - Large enough to make search interesting and demonstrate real results
# - Small enough to embed in 10-15 minutes on CPU without GPU
# - Increase to 2000+ if you want a larger demo (takes longer)
NUM_ARTICLES = 500

# How many articles to add to ChromaDB in one batch.
# Larger batch = faster but uses more memory.
# 100 is safe for most laptops.
BATCH_SIZE = 100


# ── LOAD DATASET ──────────────────────────────────────────────────────────────
def load_wikipedia_articles(num_articles: int) -> list:
    """
    Load Wikipedia articles from HuggingFace datasets library.

    The dataset downloads automatically on first run (~3GB).
    It is cached locally after first download so repeat runs are instant.
    Cache location: C:/Users/gatas/.cache/huggingface/datasets/

    Args:
        num_articles (int): how many articles to load

    Returns:
        list: list of dicts, each with keys: title, text, url

    Example output:
        [
            {
                "title": "Anarchism",
                "text" : "Anarchism is a political philosophy...",
                "url"  : "https://en.wikipedia.org/wiki/Anarchism"
            },
            ...
        ]
    """
    print(f"Loading {num_articles} Wikipedia articles from HuggingFace...")
    print("First run downloads ~3GB dataset. This takes a few minutes.")
    print("Subsequent runs load from cache instantly.")

    # load_dataset downloads and caches the Wikipedia dataset
    # '20220301.en' = English Wikipedia snapshot from March 2022
    # split='train[:500]' = first 500 articles from training split
    # trust_remote_code=True = allow dataset's own loading script to run
    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split=f"train[:{num_articles}]",
    )

    articles = []
    for item in dataset:
        articles.append({
            "title": item["title"],

            # Limit text to first 1000 characters.
            # Full Wikipedia articles can be 50,000+ characters.
            # Embedding the full article would:
            # 1. Exceed BGE-M3's 8192 token limit
            # 2. Make the embedding too general (averages all topics)
            # The first 1000 chars is usually the intro paragraph
            # which is the best summary of the article's main topic.
            # In Project 3 (RAG) we will chunk properly instead.
            "text": item["text"][:1000],

            "url":  item["url"],
        })

    print(f"Loaded {len(articles)} articles successfully.")
    return articles


# ── INDEX ARTICLES ────────────────────────────────────────────────────────────
def index_articles(articles: list) -> None:
    """
    Embed all articles with BGE-M3 and store vectors in ChromaDB.

    Process:
    1. Prepare text for each article (title + first 1000 chars)
    2. Batch encode all texts with BGE-M3 (the slow step)
    3. Store vectors + original text + metadata in ChromaDB

    Why we combine title and text:
    The title alone is too short — 'Machine learning' is only 2 words.
    Combining title + text gives richer embeddings that capture
    both the topic label and the actual content.

    Args:
        articles (list): list of article dicts from load_wikipedia_articles()
    """
    total = len(articles)
    print(f"\nIndexing {total} articles...")
    print(f"This is the slow step — BGE-M3 embeds each article on CPU.")
    print(f"Expected time: 5-15 minutes depending on your laptop.")
    print(f"Do not close the terminal.\n")

    # Process in batches to avoid memory issues on laptops
    # batch_size=100 means we embed 100 articles at a time
    for batch_start in range(0, total, BATCH_SIZE):
        # Get current batch slice
        batch_end      = min(batch_start + BATCH_SIZE, total)
        batch_articles = articles[batch_start:batch_end]
        batch_num      = (batch_start // BATCH_SIZE) + 1
        total_batches  = (total + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"Batch {batch_num}/{total_batches} "
              f"(articles {batch_start+1} to {batch_end})")

        # Prepare text for embedding
        # Format: "Title. First 1000 chars of article text"
        # The period after title helps the model treat them as one sentence
        texts = [
            f"{a['title']}. {a['text']}"
            for a in batch_articles
        ]

        # Unique ID for each article in ChromaDB
        # Format: article_0001, article_0002, etc.
        # Zero-padded to 4 digits so sorting works correctly
        ids = [
            f"article_{batch_start + i:04d}"
            for i in range(len(batch_articles))
        ]

        # Metadata stored alongside each vector
        # Returned in search results for display in the UI
        metadatas = [
            {
                "title": a["title"],
                "url":   a["url"],
            }
            for a in batch_articles
        ]

        # ── THE KEY STEP: EMBED WITH BGE-M3 ──────────────────────────
        # get_embeddings_batch() sends all texts to BGE-M3 at once
        # is_query=False means we use document encoding mode
        # This is the correct mode for indexing documents
        # (query mode is used only when a user searches)
        embeddings = get_embeddings_batch(texts, is_query=False)

        # Store everything in ChromaDB
        add_documents(
            documents=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )

    print(f"\nIndexing complete!")
    print(f"Total articles indexed: {get_document_count()}")
    print(f"Vectors saved to: ./chroma_db/")
    print(f"You can now run the search app: streamlit run src/app.py")


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    """
    Main entry point for the indexer.

    Checks if articles are already indexed to avoid re-indexing.
    Delete ./chroma_db/ folder to force re-indexing.
    """
    print("="*55)
    print("WIKIPEDIA SEMANTIC SEARCH INDEXER")
    print("="*55)

    # Check if already indexed — skip if data exists
    if collection_exists():
        count = get_document_count()
        print(f"\nDatabase already has {count} articles indexed.")
        print("Skipping indexing to save time.")
        print("\nTo re-index from scratch:")
        print("  1. Delete the ./chroma_db/ folder")
        print("  2. Run: python src/indexer.py")
        return

    # Load articles from HuggingFace
    articles = load_wikipedia_articles(NUM_ARTICLES)

    # Embed and store in ChromaDB
    index_articles(articles)


if __name__ == "__main__":
    main()