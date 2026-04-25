from lib.search_utils import (
    load_movies, 
    load_stopwords, 
    CACHE_PATH,
    BM25_K1,
    BM25_B
)
import string
import os
from collections import defaultdict, Counter
from nltk.stem import PorterStemmer
import pickle
import math

stemmer = PorterStemmer()

class InvertedIndex():
    def __init__(self):
        self.index = defaultdict(set)   # token: set(document IDs)
        self.docmap = {}    # Maps document ID: document content
        self.index_path = CACHE_PATH / 'index.pkl'
        self.docmap_path = CACHE_PATH / 'docmap.pkl'
        self.term_frequencies = defaultdict(Counter)  # token: frequency across all documents
        self.term_frequencies_path = CACHE_PATH / 'term_frequencies.pkl'
        self.doc_lengths = {}  # doc_id: length of document in tokens
        self.doc_lengths_path = CACHE_PATH / 'doc_lengths.pkl'
        
    def __add_document(self, doc_id: int, text: str):
        tokens = tokenize_text(text)
        for tok in set(tokens):
            self.index[tok].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)
        self.doc_lengths[doc_id] = len(tokens)
        
    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)
            
    def get_documents(self, term: str) -> set[int]:
        return sorted(list(self.index[term]))

    def bm25_search(self, query: str, limit: int) -> list[dict]:
        query_tokens = tokenize_text(query)
        scores = {}
        for doc_id in self.docmap:
            score = 0.0
            for token in query_tokens:
                score += self.get_bm25(doc_id, token)
            scores[doc_id] = score
        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = ranked_docs[:limit]
        formatted_results = []
        for doc_id, score in results:
            title = self.docmap[doc_id]['title']
            description = self.docmap[doc_id]['description']
            formatted_results.append(
                {'id': doc_id, 
                 'title': title, 
                 'score': score,
                 'description': description
                }
            )
        return formatted_results
    
    def get_bm25(self, doc_id: int, term: str) -> float:
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_tf * bm25_idf
    
    def get_bm25_tf(self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()
        if avg_doc_length > 0:
            length_norm = 1 - b + b * (doc_length / avg_doc_length)
        else:
            length_norm = 1.0
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * length_norm
        if denominator == 0:
            return 0.0
        return numerator / denominator

    def get_bm25_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("get_bm25_idf expects a single term")
        token = tokens[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log((doc_count - term_doc_count + 0.5) / (term_doc_count + 0.5) + 1)

    def get_tfidf(self, doc_id: int, term: str) -> float:
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf
    
    def get_tf(self, doc_id: int, term: str) -> int:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("get_tf expects a single term")
        token = tokens[0]
        return self.term_frequencies[doc_id][token]
    
    def get_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("get_idf expects a single term")
        token = tokens[0]
        doc_count = len(self.docmap)
        if doc_count == 0:
            return 0.0
        term_doc_count = len(self.index.get(token, []))
        return math.log((doc_count + 1) / (term_doc_count + 1))
    
    def build(self):
        movies = load_movies()
        for movie in movies:
            doc_id = movie['id']
            text = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id, text)
            self.docmap[doc_id] = movie
            
    def save(self):
        os.makedirs(CACHE_PATH, exist_ok=True)
        with open(self.index_path, 'wb') as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, 'wb') as f:
            pickle.dump(self.docmap, f)
        with open(self.term_frequencies_path, 'wb') as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.doc_lengths_path, 'wb') as f:
            pickle.dump(self.doc_lengths, f)
            
    def load(self):
        with open(self.index_path, 'rb') as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, 'rb') as f:
            self.docmap = pickle.load(f)
        if os.path.exists(self.term_frequencies_path):
            with open(self.term_frequencies_path, 'rb') as f:
                self.term_frequencies = pickle.load(f)
        else:
            self.term_frequencies = defaultdict(Counter)
        if os.path.exists(self.doc_lengths_path):
            with open(self.doc_lengths_path, 'rb') as f:
                self.doc_lengths = pickle.load(f)
        else:
            self.doc_lengths = defaultdict(int)
            for doc_id, movie in self.docmap.items():
                text = f"{movie['title']} {movie['description']}"
                self.doc_lengths[doc_id] = len(tokenize_text(text))
            os.makedirs(CACHE_PATH, exist_ok=True)
            with open(self.doc_lengths_path, 'wb') as f:
                pickle.dump(self.doc_lengths, f)

def bm25_search(query: str, limit: int = 5) -> list[dict]:
    idx = InvertedIndex()
    idx.load()
    results = idx.bm25_search(query, limit)
    return results

def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B):
    idx = InvertedIndex()
    idx.load()
    bm25_tf = idx.get_bm25_tf(doc_id, term, k1, b)
    return bm25_tf

def bm25_idf_command(term: str):
    idx = InvertedIndex()
    idx.load()
    idf = idx.get_bm25_idf(term)
    return idf

def tfidf_command(doc_id: int, term: str):
    idx = InvertedIndex()
    idx.load()
    tf_idf = idx.get_tfidf(doc_id, term)
    print(f"TF-IDF for term of '{term}' in document {doc_id}: {tf_idf:.2f}")

def idf_command(term: str):
    idx = InvertedIndex()
    idx.load()
    idf = idx.get_idf(term)
    if idf is None:
        idf = 0.0
    print(f"IDF for term of '{term}': {idf:.2f}")

def tf_command(doc_id: int, term: str):
    idx = InvertedIndex()
    idx.load()
    print(idx.get_tf(doc_id, term))

def build_command():
    idx = InvertedIndex()
    idx.build()
    idx.save()

def clean_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def tokenize_text(text: str) -> list[str]:
    text = clean_text(text)
    stopwords = load_stopwords()
    tokens = [tok for tok in text.split() if tok and tok not in stopwords]
    return [stemmer.stem(tok) for tok in tokens]

def has_matching_token(query_tokens, movie_tokens):
    for query_tok in query_tokens:
        for movie_tok in movie_tokens:
            if query_tok in movie_tok:
                return True
    return False       

def search_command(query: str, n_results: int):
    movies = load_movies()
    idx = InvertedIndex()
    idx.load()
    seen, res =  set(), []
    query_tokens = tokenize_text(query)
    for query_tok in query_tokens:
        matching_doc_ids = idx.get_documents(query_tok)
        for matching_doc_id in matching_doc_ids:
            if matching_doc_id in seen:
                continue
            seen.add(matching_doc_id)
            matching_doc = idx.docmap[matching_doc_id]
            res.append(matching_doc)
            if len(res) >= n_results:
                return res
    return res
