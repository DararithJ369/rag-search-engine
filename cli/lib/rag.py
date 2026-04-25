from .llm import answer_question, summarize_documents, citations_documents, detailed_question_answering
from .search_utils import load_movies
from .hybrid_search import HybridSearch


def query_answering(query: str) -> str:
    hs = HybridSearch(load_movies())
    rrf_results = hs.rrf_search(query, k=60, limit=5)
    print("Search Results:\n")
    for res in rrf_results:
        print(f"- {res['title']}")
    
    rag_results = answer_question(query, rrf_results)
    print("\nRAG Response:\n")
    print(rag_results)
    
    return rag_results

def doc_summarization(query: str, limit: int = 5) -> str:
    hs = HybridSearch(load_movies())
    rrf_results = hs.rrf_search(query, k=60, limit=limit)
    print("Search Results:\n")
    for res in rrf_results:
        print(f"- {res['title']}")
    
    rag_results = summarize_documents(query, rrf_results)
    print("\nLLM Summary:\n")
    print(rag_results)
    
    return rag_results

def doc_citations(query: str, limit: int = 5) -> str:
    hs = HybridSearch(load_movies())
    rrf_results = hs.rrf_search(query, k=60, limit=limit)
    print("Search Results:\n")
    for res in rrf_results:
        print(f"- {res['title']}")
    
    rag_results = citations_documents(query, rrf_results)
    print("\nLLM Answer:\n")
    print(rag_results)
    
    return rag_results

def answer_detailed_question(query: str, limit: int = 5) -> str:
    hs = HybridSearch(load_movies())
    rrf_results = hs.rrf_search(query, k=60, limit=limit)
    print("Search Results:\n")
    for res in rrf_results:
        print(f"- {res['title']}")
    
    rag_results = detailed_question_answering(query, rrf_results)
    print("\nAnswer:\n")
    print(rag_results)
    
    return rag_results
