import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import load_movies
from .llm import augment_prompt, llm_judge
from .prompts.rerank import individual_rank, batch_rank, cross_encoder_rerank


class HybridSearch:
    def __init__(self, documents: list[dict]):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5):
        bm25_results = self._bm25_search(query, limit*500)
        sem_results = self.semantic_search.search_chunks(query, limit*500)
        combined_results = combine_search_results(bm25_results, sem_results, alpha)
        return combined_results[:limit]

    def rrf_search(self, query: str, k: int, limit: int = 10, debug: str = None):
        bm25_results = self._bm25_search(query, limit*500)
        sem_results = self.semantic_search.search_chunks(query, limit*500)
        combined_results = rrf_combine_search_results(bm25_results, sem_results, k)
        # print(f"RRF DEBUG={debug}")
        if debug:
            for r in combined_results:
                if debug.lower().strip() in r['title'].lower().strip():
                    print(f"Phase 1 Search for {r['title']:}")
                    print(f"{r['bm25_rank']=} | {r['sem_rank']=}")
        return combined_results[:limit]

def rrf_search(query: str, k: int = 60, limit: int = 5, enhance: str = None, rerank_method: str = None, debug: str = None, evaluate: str = None) -> list[dict]:
    hs = HybridSearch(load_movies())
    if enhance:
        try:
            new_query = augment_prompt(query, enhance)
            print(f"Enhanced query {enhance}: '{query}' -> '{new_query}'\n")
            query = new_query
        except RuntimeError as e:
            print(f"Query enhancement skipped: {e}")
    
    rrf_limit = limit * 5 if rerank_method else 5
    results = hs.rrf_search(query, k, rrf_limit, debug)
    if debug:
        found = False
        for idx, r in enumerate(results):
            if debug.lower().strip() in r['title'].lower().strip():
                found = True
                break
        print(f"DEBUG: Hybrid Search position for {debug} is {idx if found else 'not found'}")
    
    match rerank_method:
        case "individual":
            try:
                results = individual_rank(query, results)
                print(f"Reranking top {limit} results using individual rank method...")
            except RuntimeError as e:
                print(f"Reranking skipped: {e}")
        case "batch":
            try: 
                results = batch_rank(query, results)
                print(f"Reranking top {limit} results using batch rank method...")
            except RuntimeError as e:
                print(f"Reranking skipped: {e}")
        case "cross_encoder":
            try:
                results = cross_encoder_rerank(query, results)
                print(f"Reranking top {limit} results using cross-encoder method...")
            except RuntimeError as e:
                print(f"Reranking skipped: {e}")
        case _:
            pass
        
    if debug:
        found = False
        for idx, r in enumerate(results):
            if debug.lower().strip() in r['title'].lower().strip():
                found = True
                break
        print(f"DEBUG: Reranking position for {debug} is {idx if found else 'not found'}")
    
    if evaluate: formatted_results = []
    for idx, r in enumerate(results[:limit], 1):
        print(f"{idx} {r['title']}")
        if rerank_method:
            print(f"Cross-Encoder Score: {r.get('cross_encoder_score', 0.0):.4f}")
        print(f"RRF Score: {r['rrf_score']:.4f}")
        print(f"BM25 Rank: {r['bm25_rank']}, Semantic Rank: {r['sem_rank']}")
        print(f"Description: {r['description'][:100]}\n")
        
        if evaluate: 
            formatted_results.append(f"<result id={idx}>{r['title']}: {r['description']}</result>")
        
    if evaluate:
        llm_results = llm_judge(query, "\n".join(formatted_results))
        for idx, r in enumerate(results[:limit], 1):
            print(f"{idx} {r['title']}: {llm_results[idx-1]}/3")
        
    return results[:limit]

def weighted_search(query: str, alpha: float, limit: int = 5) -> list[dict]:
    hs = HybridSearch(load_movies())
    results = hs.weighted_search(query, alpha, limit)
    for idx, r in enumerate(results, 1):
        print(f"{idx} {r['title']}")
        print(f"Hybrid Score: {r['hybrid_score']:.4f}")
        print(f"BM25 Score: {r['bm25_score']:.4f}, Semantic Score: {r['sem_score']:.4f}")
        print(f"Description: {r['description'][:100]}\n")
    return results[:limit]
    
def hybrid_score(bm25_score: float, sem_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * sem_score

def normalize_search_results(results: list[dict]) -> list[dict]:
    scores = [r['score'] for r in results]
    norm_scores = normalize_scores(scores)
    for idx, norm_score in enumerate(norm_scores):
        results[idx]['normalized_score'] = norm_score
    return results

def rrf_score(rank: int, k: int) -> float:
    return 1 / (k + rank)

def rrr_final_score(r1: int, r2: int, k: int) -> float:
    if r1 and r2:
        return rrf_score(r1, k) + rrf_score(r2, k)
    return 0.0

def rrf_combine_search_results(bm25_results: list[dict], sem_results: list[dict], k: int = 60) -> list[dict]:
    scores = {}
    for rank, res in enumerate(bm25_results, 1):
        doc_id = res['id']
        scores[doc_id] = {
            'doc_id': doc_id,
            'bm25_rank': rank,
            'bm25_score': rrf_score(rank, k),
            'sem_rank': None,
            'sem_score': 0.0,
            'title': res['title'],
            'description': res['description'] 
        }
    
    for rank, res in enumerate(sem_results, 1):
        doc_id = res['id']
        if doc_id not in scores:
            scores[doc_id] = {
                'doc_id': doc_id,
                'bm25_rank': None,
                'bm25_score': 0.0,
                'sem_rank': rank,
                'sem_score': rrf_score(rank, k),
                'title': res['title'],
                'description': res['description'] 
            }
        scores[doc_id]['sem_rank'] = rank
        scores[doc_id]['sem_score'] = rrf_score(rank, k)
        
    for doc_id in scores.keys():
        bm25_rank = scores[doc_id]['bm25_rank']
        sem_rank = scores[doc_id]['sem_rank']
        scores[doc_id]['rrf_score'] = rrr_final_score(bm25_rank, sem_rank, k)
        
    results = sorted(scores.values(), key=lambda x: x['rrf_score'], reverse=True)
    return results
    
def combine_search_results(bm25_results: list[dict], sem_results: list[dict], alpha: float = 0.5) -> list[dict]:
    bm25_norm = normalize_search_results(bm25_results)
    sem_norm = normalize_search_results(sem_results)
    combined_norm = {}
    for norm in bm25_norm:
        doc_id = norm['id']
        combined_norm[doc_id] = {
            'doc_id': doc_id,
            'bm25_score': norm['normalized_score'], 
            'sem_score': 0.0,   
            'title': norm['title'],
            'description': norm['description']
        }

    for norm in sem_norm:
        doc_id = norm['id']
        if doc_id not in combined_norm:
            combined_norm[doc_id] = {
                'doc_id': doc_id,
                'bm25_score': 0.0,   
                'sem_score': 0.0, 
                'title': norm['title'],
                'description': norm['description']
            }
        combined_norm[doc_id]['sem_score'] = norm['normalized_score']
        
    for k, v in combined_norm.items():
        combined_norm[k]['hybrid_score'] = hybrid_score(v['bm25_score'], v['sem_score'], alpha) 
        
    results = sorted(combined_norm.values(), key=lambda x: x['hybrid_score'], reverse=True)
    return results

def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [1.0] * len(scores)
    return [(s - min_score) / (max_score - min_score) for s in scores]