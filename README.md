# MTG Card Search

A semantic search system for Magic: The Gathering cards using sentence transformers and FAISS.

## Features

- Downloads and processes the complete MTG card database from MTGJSON
- Creates semantic embeddings for card text using sentence-transformers
- Builds a FAISS index for fast similarity search
- Supports filtering cards by Scryfall IDs via CSV
- Returns detailed card information with similarity scores

## Installation

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Search

Run the example search:
```bash
python mtg_search.py
```

This will:
1. Download the AllPrintings.json.xz file (if not cached)
2. Process the card data
3. Create embeddings
4. Build the FAISS index
5. Run an example search for dragon cards

### Using the API

```python
from mtg_search import MTGSearch

# Initialize the search system
searcher = MTGSearch()

# Download and process data (if needed)
searcher.download_all_printings(force=False)
searcher.load_cards()
searcher.create_embeddings(force=False)
searcher.build_index()

# Search for cards
results = searcher.search("Find me some dragon cards", k=5)

# Print results
for result in results:
    print(f"Name: {result['name']}")
    print(f"Type: {result['type_line']}")
    print(f"Mana Cost: {result['mana_cost']}")
    print(f"Oracle Text: {result['oracle_text']}")
    print(f"Similarity Score: {result['similarity_score']:.4f}")
    print("-" * 80)
```

### Filtering Cards

To filter cards using a CSV file containing Scryfall IDs:

```python
searcher.filter_by_csv("path/to/filter.csv")
```

The CSV file should have a column named "Scryfall Id", "scryfall_id", or "Scryfall id".

## Configuration

The system is configured via the `.cursorai` file, which includes:

- Data source configuration
- Schema definitions
- Embedding model settings
- Vector store configuration
- Pipeline definitions

## Caching

The system caches:
- The AllPrintings.json.xz file
- Card embeddings (embeddings.npy)
- FAISS index (mtg_index.faiss)

Use the `force` parameter to force re-download or re-embedding:
```python
searcher.download_all_printings(force=True)
searcher.create_embeddings(force=True)
``` 