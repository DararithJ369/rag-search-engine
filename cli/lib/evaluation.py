import json

from lib.search_utils import PROJECT_ROOT, load_movies
from lib.hybrid_search import HybridSearch

def load_test_cases():
    with open(PROJECT_ROOT / 'data/golden_dataset.json', 'r') as f:
        test_cases = json.load(f)['test_cases']
        return test_cases
    
def evaluate(limit: int = 5):
    print(f"k={limit}")
    test_cases = load_test_cases()
    hs = HybridSearch(load_movies())
    
    for test_case in test_cases:
        qry = test_case['query']
        exp = test_case['relevant_docs']
        rrf_results = hs.rrf_search(qry, k=60, limit=limit)
        relevant_cnt = 0
        for rrf_res in rrf_results:
            relevant_cnt += rrf_res['title'] in exp
    
        precision = relevant_cnt / limit if limit > 0 else 0.0 
        recall = relevant_cnt / len(exp) if exp else 0.0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"Query: {qry}")
        print(f"Precision@{limit}: {precision:.4f} ({relevant_cnt}/{limit} relevant in top {limit})")
        print(f"Recall@{limit}: {recall:.4f} ({relevant_cnt}/{len(exp)} relevant retrieved)")
        print(f"F1 Score@{limit}: {f1_score:.4f}")
        print(f"Retrieved: {[r['title'] for r in rrf_results]}")
        print(f"Expected: {', '.join(exp)}\n") 