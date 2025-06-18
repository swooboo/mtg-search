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

class MTGSearch:
    def __init__(self, config_path: str = ".cursorai"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.index = None
        self.cards = []
        self.card_map = {}
        
    def download_all_printings(self, force: bool = False) -> None:
        """Download and cache the AllPrintings.json.xz file."""
        output_path = "raw_allprintings.json.xz"
        
        if not force and os.path.exists(output_path):
            print("Using cached AllPrintings.json.xz")
            return
            
        print("Downloading AllPrintings.json.xz...")
        url = "https://mtgjson.com/api/v5/AllPrintings.json.xz"
        response = requests.get(url)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
            
    def load_cards(self) -> None:
        """Load and process cards from AllPrintings.json.xz."""
        print("Loading cards from AllPrintings.json.xz...")
        with lzma.open("raw_allprintings.json.xz", 'rt', encoding='utf-8') as f:
            data = json.load(f)
            
        # Flatten the nested data structure
        self.cards = []
        for set_code, set_data in data.get("data", {}).items():
            for card in set_data.get("cards", []):
                # Add set code to card data
                card["setCode"] = set_code
                self.cards.append(card)
            
        # Create card map for quick lookup
        self.card_map = {card.get("scryfallId"): card for card in self.cards if card.get("scryfallId")}
        
        print(f"Loaded {len(self.cards)} cards")
        
    def create_embeddings(self, force: bool = False) -> None:
        """Create embeddings for all cards."""
        if not force and os.path.exists("embeddings.pkl"):
            print("Using cached embeddings")
            return
            
        print("Creating embeddings...")
        texts = []
        for card in self.cards:
            text = f"{card.get('name', '')} || {card.get('type', '')} || {card.get('text', '')} || {card.get('flavor', '')}"
            texts.append(text)
            
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        
        # Save embeddings
        np.save("embeddings.npy", embeddings)
        
    def build_index(self) -> None:
        """Build FAISS index from embeddings."""
        print("Building FAISS index...")
        embeddings = np.load("embeddings.npy")
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        
        # Save index
        faiss.write_index(self.index, "mtg_index.faiss")
        
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
        self.cards = [card for card in self.cards if card.get("scryfallId") in allowed_ids]
        self.card_map = {card.get("scryfallId"): card for card in self.cards}
        
def main():
    # Initialize search
    searcher = MTGSearch()
    
    # Download and process data
    searcher.download_all_printings(force=False)
    searcher.load_cards()
    searcher.create_embeddings(force=False)
    searcher.build_index()
    
    # Example search
    query = "Find me some dragon cards"
    results = searcher.search(query)
    
    print(f"\nResults for query: '{query}'\n")
    for result in results:
        print(f"Name: {result['name']}")
        print(f"Type: {result['type_line']}")
        print(f"Mana Cost: {result['mana_cost']}")
        print(f"Oracle Text: {result['oracle_text']}")
        print(f"Similarity Score: {result['similarity_score']:.4f}")
        print("-" * 80)

if __name__ == "__main__":
    main() 