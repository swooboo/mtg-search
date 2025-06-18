import json
import lzma
import os
from typing import List, Dict, Any, Optional
import requests
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import yaml
import csv
import pickle
from tqdm import tqdm

class MTGSearch:
    def __init__(self, config_path: str = ".cursorai"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.index = None
        self.cards = []
        self.card_map = {}
        
        # Find and load CSV filters
        csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
        if csv_files:
            print(f"Found {len(csv_files)} CSV filter(s): {', '.join(csv_files)}")
            self.load_csv_filters(csv_files)
            
    def load_csv_filters(self, csv_files: List[str]) -> None:
        """Load filters from multiple CSV files."""
        self.allowed_ids = set()
        
        for csv_file in csv_files:
            print(f"Loading filter from {csv_file}...")
            with open(csv_file, newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Try all possible column names for Scryfall ID
                    key = None
                    for col in row:
                        if col.lower().replace(' ', '') in ['scryfallid', 'scryfall_id']:
                            key = row[col].strip('"')  # Remove quotes if present
                            break
                    if key:
                        self.allowed_ids.add(key)
                        
        if self.allowed_ids:
            print(f"Loaded {len(self.allowed_ids)} unique card IDs for filtering")
        else:
            print("No valid Scryfall IDs found in CSV files")
            self.allowed_ids = None
            
    def verify_filter_cards(self) -> None:
        """Verify that all filter cards exist in the database."""
        if not hasattr(self, 'allowed_ids') or not self.allowed_ids:
            return
            
        missing_cards = set()
        for card_id in self.allowed_ids:
            if not any(card.get("identifiers", {}).get("scryfallId") == card_id for card in self.cards):
                missing_cards.add(card_id)
            
        if missing_cards:
            print(f"\nWarning: {len(missing_cards)} cards from filter not found in database:")
            for card_id in sorted(missing_cards):
                print(f"  - {card_id}")
        else:
            print("All filter cards found in database")
            
    def download_all_printings(self) -> None:
        """Download the AllPrintings.json.xz file."""
        output_path = "raw_allprintings.json.xz"
        pickle_path = "cards.pkl"
        
        # Skip if we already have the raw file or pickled cards
        if os.path.exists(output_path):
            print("Using existing raw card data file")
            return
            
        if os.path.exists(pickle_path):
            print("Using existing card data from pickle file")
            return
            
        print("Downloading AllPrintings.json.xz...")
        url = "https://mtgjson.com/api/v5/AllPrintings.json.xz"
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        
        with open(output_path, 'wb') as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(block_size):
                size = f.write(data)
                pbar.update(size)
            
    def load_cards(self) -> None:
        """Load and process cards from AllPrintings.json.xz."""
        pickle_path = "cards.pkl"
        
        if os.path.exists(pickle_path):
            print("Loading cards from pickle file...")
            with open(pickle_path, 'rb') as f:
                self.cards = pickle.load(f)
                self.card_map = {card.get("identifiers", {}).get("scryfallId"): card for card in self.cards if card.get("identifiers", {}).get("scryfallId")}
            print(f"Loaded {len(self.cards)} cards")
            return
            
        print("Loading cards from AllPrintings.json.xz...")
        with lzma.open("raw_allprintings.json.xz", 'rt', encoding='utf-8') as f:
            data = json.load(f)
            
        # Flatten the nested data structure
        all_cards = []
        total_cards = sum(len(set_data.get("cards", [])) for set_data in data.get("data", {}).values())
        print(f"Found {total_cards} total cards in JSON")
        
        with tqdm(total=total_cards, desc="Loading cards") as pbar:
            for set_code, set_data in data.get("data", {}).items():
                for card in set_data.get("cards", []):
                    # Add set code to card data
                    card["setCode"] = set_code
                    all_cards.append(card)
                    pbar.update(1)
            
        print(f"Processed {len(all_cards)} cards")
        
        # Filter cards if we have a filter
        if hasattr(self, 'allowed_ids') and self.allowed_ids:
            print(f"Filtering cards to match {len(self.allowed_ids)} allowed IDs...")
            self.cards = [card for card in all_cards if card.get("identifiers", {}).get("scryfallId") in self.allowed_ids]
            print(f"Filtered to {len(self.cards)} cards")
        else:
            self.cards = all_cards
            
        # Create card map for quick lookup
        self.card_map = {card.get("identifiers", {}).get("scryfallId"): card for card in self.cards if card.get("identifiers", {}).get("scryfallId")}
        print(f"Created card map with {len(self.card_map)} entries")
        
        # Save to pickle file
        print("Saving cards to pickle file...")
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.cards, f)
        
        print(f"Saved {len(self.cards)} cards to pickle file")
        
    def create_embeddings(self) -> None:
        """Create embeddings for all cards."""
        embeddings_path = "embeddings.npy"
        
        if os.path.exists(embeddings_path):
            print("Using existing embeddings")
            return
            
        print("Creating embeddings...")
        texts = []
        for card in self.cards:
            text = f"{card.get('name', '')} || {card.get('type', '')} || {card.get('text', '')} || {card.get('flavor', '')}"
            texts.append(text)
            
        print("Encoding texts to embeddings...")
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        
        # Save embeddings
        np.save(embeddings_path, embeddings)
        
    def build_index(self) -> None:
        """Build FAISS index from embeddings."""
        index_path = "mtg_index.faiss"
        
        if os.path.exists(index_path):
            print("Loading existing FAISS index...")
            self.index = faiss.read_index(index_path)
            return
            
        print("Building FAISS index...")
        embeddings = np.load("embeddings.npy")
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(x=embeddings.astype('float32'))
        
        # Save index
        faiss.write_index(self.index, index_path)
        
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for cards using semantic similarity."""
        if self.index is None:
            raise RuntimeError("Index not built. Run build_index() first.")
            
        # Encode query
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        
        # Search
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Get results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.cards):  # Ensure index is valid
                card = self.cards[idx]
                results.append({
                    "name": card.get("name"),
                    "type_line": card.get("type"),
                    "mana_cost": card.get("manaCost"),
                    "oracle_text": card.get("text"),
                    "scryfall_id": card.get("identifiers", {}).get("scryfallId"),
                    "set_code": card.get("setCode"),
                    "similarity_score": float(1 / (1 + distances[0][i]))  # Convert distance to similarity
                })
                
        return results
        
    def filter_by_csv(self, csv_path: str) -> None:
        """Filter cards based on a CSV file containing Scryfall IDs."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
        allowed_ids = set()
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = row.get("Scryfall Id") or row.get("scryfall_id") or row.get("Scryfall id")
                if key:
                    allowed_ids.add(key.strip())
                    
        # Filter cards
        self.cards = [card for card in self.cards if card.get("identifiers", {}).get("scryfallId") in allowed_ids]
        self.card_map = {card.get("identifiers", {}).get("scryfallId"): card for card in self.cards}
        
def main():
    # Initialize search
    searcher = MTGSearch()
    
    # Download and process data
    print("Initializing MTG Search...")
    searcher.download_all_printings()
    searcher.load_cards()
    searcher.create_embeddings()
    searcher.build_index()
    
    print("\nMTG Card Search is ready! Type 'quit' to exit.")
    print("Enter your search query to find cards with similar flavor or theme.")
    
    while True:
        query = input("\nEnter your search query: ").strip()
        
        if query.lower() == 'quit':
            print("Goodbye!")
            break
            
        if not query:
            print("Please enter a valid search query.")
            continue
            
        try:
            results = searcher.search(query, k=5)
            
            print(f"\nResults for query: '{query}'\n")
            for i, result in enumerate(results, 1):
                print(f"Result {i}:")
                print(f"Name: {result['name']}")
                print(f"Type: {result['type_line']}")
                print(f"Mana Cost: {result['mana_cost']}")
                print(f"Oracle Text: {result['oracle_text']}")
                print(f"Scryfall ID: {result['scryfall_id']}")
                print(f"Set: {result['set_code']}")
                print(f"Similarity Score: {result['similarity_score']:.4f}")
                print("-" * 80)
                
        except Exception as e:
            print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 