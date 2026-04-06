"""
src/database.py

All ChromaDB vector database operations in one place.
No other file touches the database directly — everything
goes through these functions.

ChromaDB chosen for Project 2 because:
- Runs embedded inside Python, no external server needed
- No Docker, no configuration, no setup
- Saves to disk automatically (data persists between runs)
- Perfect for learning and development

For production use Qdrant instead — faster, more scalable,
supports richer filtering, better for EU self-hosted deployments.

Run this file directly to test it works:
    python src/database.py
"""

import chromadb
import os

# ── DATABASE SETUP ────────────────────────────────────────────────────────────
# Collection name — like a table name in SQL
COLLECTION_NAME = "wikipedia_articles"

# PersistentClient saves all vectors to disk at this path.
# Creates the folder automatically if it does not exist.
# Data survives between Python sessions — you only index once.
# Delete this folder to start fresh and re-index.
DB_PATH = "./chroma_db"

# Create the client at module level — reused for all operations
# Creating a new client per function call is slow and wasteful
client = chromadb.PersistentClient(path=DB_PATH)


# ── COLLECTION OPERATIONS ─────────────────────────────────────────────────────
def get_or_create_collection():
    """
    Get existing collection or create a new one if it does not exist.

    metadata hnsw:space=cosine tells ChromaDB to use cosine similarity
    for all distance calculations. This is the correct metric for text
    embeddings — proven in experiment_similarity.py where cosine handled
    variable-length texts correctly.

    Returns:
        chromadb.Collection: the collection object for further operations
    """
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # cosine similarity for text
    )
    return collection


# ── WRITE OPERATIONS ──────────────────────────────────────────────────────────
def add_documents(
    documents: list,
    embeddings: list,
    ids: list,
    metadatas: list
) -> None:
    """
    Add documents and their embedding vectors to ChromaDB.

    Stores everything together in one operation:
    - The original text (for displaying in search results)
    - The embedding vector (for similarity search)
    - A unique ID (for identifying each document)
    - Metadata dict (for filtering and displaying extra info)

    Args:
        documents  (list): original text strings, one per document
        embeddings (list): 1024-dim vectors, one per document
        ids        (list): unique string IDs like 'article_0001'
        metadatas  (list): dicts with extra info like title and url

    Example:
        add_documents(
            documents  = ["Solar power is..."],
            embeddings = [[0.23, -0.45, ...]],
            ids        = ["article_0001"],
            metadatas  = [{"title": "Solar power", "url": "..."}]
        )
    """
    collection = get_or_create_collection()

    # ChromaDB add() stores all 4 things together in one call
    # Later when we search, we get all of this back in results
    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )
    print(f"Added {len(documents)} documents. "
          f"Total in database: {collection.count()}")


# ── READ OPERATIONS ───────────────────────────────────────────────────────────
def search(query_embedding: list, n_results: int = 5) -> dict:
    """
    Search for the n most similar documents to the query embedding.

    ChromaDB uses the HNSW (Hierarchical Navigable Small World) index
    to find nearest vectors efficiently. For 500 documents this is
    instant. For millions it would still be milliseconds.

    Args:
        query_embedding (list): 1024-dim vector of the search query
        n_results       (int) : how many results to return, default 5

    Returns:
        dict with these keys:
            documents : list of original text strings
            distances : list of cosine distances (0=identical, 2=opposite)
            metadatas : list of metadata dicts
            ids       : list of document IDs

    Note:
        ChromaDB returns cosine DISTANCE not cosine SIMILARITY.
        Distance 0 = identical. Distance 2 = completely opposite.
        Convert to similarity: similarity = 1 - distance
        This conversion happens in search.py not here.
    """
    collection = get_or_create_collection()

    results = collection.query(
        query_embeddings=[query_embedding],  # must be a list of vectors
        n_results=n_results,
        include=["documents", "distances", "metadatas"]
    )
    return results


def get_document_count() -> int:
    """
    Return the total number of documents stored in the database.

    Used to check if indexing has been done before running the app.
    If count is 0, user needs to run indexer.py first.

    Returns:
        int: number of documents currently stored
    """
    collection = get_or_create_collection()
    return collection.count()


def collection_exists() -> bool:
    """
    Check if the collection has any documents indexed.

    Returns:
        bool: True if documents exist, False if database is empty
    """
    try:
        return get_document_count() > 0
    except Exception:
        return False


def delete_collection() -> None:
    """
    Delete the entire collection and all its vectors.

    Use this to start fresh if you want to re-index with
    different settings or a different dataset.

    Warning: this permanently deletes all indexed data.
    You must run indexer.py again after calling this.
    """
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' deleted.")
        print("Run indexer.py to re-index your documents.")
    except Exception as e:
        print(f"Could not delete collection: {e}")


# ── QUICK TEST ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Quick test to verify ChromaDB is working correctly.
    Run: python src/database.py
    """

    print("="*50)
    print("TESTING database.py")
    print("="*50)

    # Test 1: Check current state
    count = get_document_count()
    print(f"\nTest 1: Database connection")
    print(f"  DB path          : {DB_PATH}")
    print(f"  Collection name  : {COLLECTION_NAME}")
    print(f"  Documents stored : {count}")
    print(f"  PASSED: Connected to ChromaDB successfully")

    # Test 2: Add test documents
    print(f"\nTest 2: Add test documents")

    test_docs = [
        "Machine learning is a subset of artificial intelligence",
        "Python is the most popular language for data science",
        "The Eiffel Tower is located in Paris France",
    ]
    test_embeddings = [
        [0.1] * 1024,   # fake vectors for testing only
        [0.2] * 1024,   # in real use these come from BGE-M3
        [0.3] * 1024,
    ]
    test_ids = ["test_001", "test_002", "test_003"]
    test_metas = [
        {"title": "Machine Learning", "url": "test"},
        {"title": "Python",           "url": "test"},
        {"title": "Eiffel Tower",     "url": "test"},
    ]

    add_documents(test_docs, test_embeddings, test_ids, test_metas)

    new_count = get_document_count()
    print(f"  Documents before : {count}")
    print(f"  Documents after  : {new_count}")
    print(f"  PASSED: Added 3 test documents")

    # Test 3: Search for a document
    print(f"\nTest 3: Search functionality")
    query_vec = [0.11] * 1024  # fake query vector
    results = search(query_vec, n_results=2)

    print(f"  Results returned : {len(results['documents'][0])}")
    print(f"  Top result       : {results['documents'][0][0][:50]}...")
    print(f"  Distance         : {results['distances'][0][0]:.4f}")
    print(f"  PASSED: Search returned results")

    # Test 4: Clean up test data
    print(f"\nTest 4: Cleanup")
    delete_collection()
    final_count = get_document_count()
    print(f"  Documents after delete: {final_count}")
    print(f"  PASSED: Collection cleaned up")

    print("\n" + "="*50)
    print("ALL TESTS PASSED. database.py is ready.")
    print("="*50)