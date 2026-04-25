from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
from .search_utils import load_movies
import re
import json
from collections import defaultdict

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}
        self.embeddings_path = Path('cache/movie_embeddings.npy')
        
    def build_embeddings(self, documents: list[str]):
        self.documents = documents
        self.document_map = {}
        movie_strings = []
        for doc in self.documents:
            self.document_map[doc['id']] = doc
            movie_strings.append(f"{doc['title']} {doc['description']}")
        self.embeddings = self.model.encode(movie_strings, show_progress_bar=True)
        np.save(self.embeddings_path, self.embeddings)
        return self.embeddings
        
    def load_or_create_embeddings(self, documents: list[str]):
        self.documents = documents
        self.document_map = {}
        if self.embeddings_path.exists():
            self.embeddings = np.load(self.embeddings_path)
            if len(self.documents) == len(self.embeddings):
                return self.embeddings
        return self.build_embeddings(documents)
        
    def generate_embedding(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty or whitespace.")
        return self.model.encode([text])[0]
    
    def search(self, query: str, limit: int) -> list[dict]:
        if self.embeddings is None or self.documents is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        qry_emb = self.generate_embedding(query)
        
        similarities = []
        for doc_emb, doc in zip(self.embeddings, self.documents):
            _similarity = cosine_similarity(qry_emb, doc_emb)
            similarities.append((_similarity, doc))
        
        similarities.sort(key=lambda x: x[0], reverse=True)
        res = []
        for sc, doc in similarities[:limit]:
            res.append(
                {
                    'score': sc,
                    'title': doc['title'],
                    'description': doc['description']
                }
            )
        return res


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self) -> None:
        super().__init__()
        self.chunk_embeddings = None
        self.chunk_embeddings_path = Path('cache/chunk_embeddings.npy')
        self.chunk_metadata = None
        self.chunk_metadata_path = Path('cache/chunk_metadata.json')
        
    def build_chunk_embeddings(self, documents: list[str]):
        self.documents = documents
        self.document_map = {doc['id']: doc for doc in self.documents}

        all_chunks = []
        chunk_metadata = []
        
        for midx, doc in enumerate(self.documents):
            if doc['description'].strip() == "":
                continue
            _chunks = semantic_chunking(doc['description'], overlap=1, max_chunk_size=4)
            all_chunks.extend(_chunks)
            for cidx in range(len(_chunks)):
                chunk_metadata.append({
                    'movie_idx': midx,
                    'chunk_idx': cidx,
                    'total_chunks': len(_chunks)
                })

        self.chunk_embeddings = self.model.encode(all_chunks)
        self.chunk_metadata = {"chunks": chunk_metadata, "total_chunks": len(all_chunks)}
        
        self.chunk_embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.chunk_embeddings_path, self.chunk_embeddings)
        
        with open(self.chunk_metadata_path, 'w') as f:
            json.dump(self.chunk_metadata, f, indent=2)
            
        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(self, documents: list[str]):
        self.documents = documents
        self.document_map = {doc['id']: doc for doc in self.documents}
        
        if self.chunk_embeddings_path.exists() and self.chunk_metadata_path.exists():
            self.chunk_embeddings = np.load(self.chunk_embeddings_path)
            with open(self.chunk_metadata_path, 'r') as f:
                self.chunk_metadata = json.load(f)
            return self.chunk_embeddings
        return self.build_chunk_embeddings(documents)
    
    def search_chunks(self, query: str, limit: int) -> list[dict]:
        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise ValueError("No chunk embeddings loaded. Call `load_or_create_chunk_embeddings` first.")
        query_emb = self.generate_embedding(query)
        chunk_scores = []
        movie_scores = defaultdict(lambda: 0)
        
        for idx in range(len(self.chunk_embeddings)):
            chunk_emb = self.chunk_embeddings[idx]
            metadata = self.chunk_metadata['chunks'][idx]
            score = cosine_similarity(query_emb, chunk_emb)
            midx, cidx = metadata['movie_idx'], metadata['chunk_idx']
            sim = cosine_similarity(query_emb, chunk_emb)
            chunk_scores.append({
                'movie_idx': midx,
                'chunk_idx': cidx,
                'score': sim
            })
            movie_scores[midx] = max(movie_scores[midx], sim)
        movie_scores_sorted = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
        
        res = []
        for midx, score in movie_scores_sorted[:limit]:
            doc = self.documents[midx]
            res.append({
                'id': doc['id'],
                'title': doc['title'],
                'description': doc['description'][:100],
                'score': round(score, 4),
                'metadata': {},
            })
            
        return res

def search_chunked(query: str, limit: int = 5) -> list[dict]:
    css = ChunkedSemanticSearch()
    movies = load_movies()
    _ =  css.load_or_create_chunk_embeddings(movies)
    res = css.search_chunks(query, limit)
    
    for i, result in enumerate(res, 1):
        print(f"{i}. {result['title']} (Score: {result['score']:.4f})")
        print(f"   Description: {result['description']}...\n")


def embed_chunks() -> None:
    movies = load_movies()
    css = ChunkedSemanticSearch()
    embeddings = css.load_or_create_chunk_embeddings(movies)
    print(f"Generated {len(embeddings)} chunked embeddings.")

def chunk_text_semantic(text: str, overlap: int = 0, max_chunk_size: int = 4) -> list[str]:
    chunks = semantic_chunking(text, overlap, max_chunk_size)
    print(f"Semantically chunking {len(text)} characters.")
    for idx, chunk in enumerate(chunks, 1):
        print(f"{idx}. {chunk}")

def semantic_chunking(text: str, overlap: int = 0, max_chunk_size: int = 4) -> list[str]:
    text = text.strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) == 1 and sentences[0].endswith(('!', '.', '?')):
        pass
        
    chunks = []
    step_size = max_chunk_size - overlap
    sentences = [s.strip() for s in sentences if s.strip()]
    for i in range(0, len(sentences), step_size):
        chunk_sentences = sentences[i:i + max_chunk_size]
        if len(chunk_sentences) <= overlap:
            break
        chunk = ' '.join(chunk_sentences)
        chunks.append(chunk) 
    return chunks

def fixed_sized_chunking(text: str, overlap: int, chunk_size: int = 200) -> list[str]:
    words = text.split()
    chunks = []
    step_size = chunk_size - overlap
    for i in range(0, len(words), step_size):
        chunk_words = words[i:i + chunk_size]
        if len(chunk_words) <= overlap:
            break
        chunk = ' '.join(chunk_words)
        chunks.append(chunk)
    return chunks

def chunk_text(text: str, overlap: int = 50, chunk_size: int = 200) -> list[str]:
    chunks = fixed_sized_chunking(text, overlap, chunk_size)
    print(f"Chunking {len(text)} characters.")
    for idx, chunk in enumerate(chunks, 1):
        print(f"{idx}. {chunk}")
        
def search(query: str, limit: int = 5) -> list[dict]:
    ss = SemanticSearch()
    movies = load_movies()
    ss.load_or_create_embeddings(movies)
    search_results = ss.search(query, limit)
    for idx, result in enumerate(search_results, 1):
        print(f"{idx}. {result['title']} (Score: {result['score']:.4f})")
        print(f"   Description: {result['description'][:100]}\n")
        
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
  
def embed_query_text(query: str) -> list[float]:
    ss = SemanticSearch()
    embedding = ss.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")
  
def verify_embeddings():
    ss = SemanticSearch()
    documents = load_movies()
    embeddings = ss.load_or_create_embeddings(documents)
    print(f"Number of documents: {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")
  
def embed_text(text: str) -> list[float]:
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_model():
    ss = SemanticSearch()
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")