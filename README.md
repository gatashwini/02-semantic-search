# Semantic Search Engine

Finds Wikipedia articles by MEANING not by keyword matching.
Built with BGE-M3 embeddings and ChromaDB.

## The Key Difference

Search: renewable energy
Keyword search: only finds articles with those exact words
This project:   finds Solar power, Wind turbine, Hydroelectricity
                because they share the same meaning

## Tech Stack

- Embedding Model : BAAI/bge-m3 — ranked 1 on MTEB (Massive Text Embedding Benchmark)  2025
- Vector Database : ChromaDB — local, persistent, zero setup
- Dataset         : HuggingFace Wikipedia — 500 articles indexed
- UI              : Streamlit web application
- Privacy         : Fully local — no data leaves your machine (GDPR)

## Architecture

INDEXING (run once):
Wikipedia Dataset -> BGE-M3 Embeddings -> ChromaDB Vector Store

SEARCH (every query):
User Query -> BGE-M3 Embedding -> Cosine Similarity Search -> Results

## Quick Start

1. Clone this repository
   git clone https://github.com/gatashwini/02-semantic-search

2. Create environment
   conda create -n genai python=3.11 -y
   conda activate genai

3. Install dependencies
   pip install -r requirements.txt

4. Index Wikipedia articles (run once, takes 10-15 minutes)
   python src/indexer.py

5. Launch the search app
   streamlit run src/app.py

6. Open browser at http://localhost:8501

## Project Structure

src/embeddings.py  BGE-M3 model loading and vector creation
src/database.py    ChromaDB operations
src/indexer.py     Dataset loading and one-time indexing
src/search.py      Search pipeline
src/app.py         Streamlit web interface

## Key Technical Decisions

BGE-M3 chosen because it is ranked 1 on MTEB and supports
100+ languages. Tested: 274% better than all-MiniLM on
cross-language EU search tasks (English vs German, French, Dutch).

ChromaDB for development. Production version would use
self-hosted Qdrant on EU infrastructure for GDPR compliance.

## License

MIT
