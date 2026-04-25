import argparse
from lib.hybrid_search import (
    normalize_scores,
    weighted_search,
    rrf_search
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")
    
    normalize_parser = subparser.add_parser("normalize", help="Normalize a list of scores")
    normalize_parser.add_argument("scores", nargs="+", type=float, help="Scores to normalize")

    ws_parser = subparser.add_parser("weighted_search", help="Perform a weighted hybrid search")
    ws_parser.add_argument("query", type=str, help="Search query")
    ws_parser.add_argument("--alpha", type=float, help="Weighting factor for BM25 vs Semantic Search (0 to 1)")
    ws_parser.add_argument("--limit", type=int, default=5, help="Number of search results to return")

    rrf_parser = subparser.add_parser("rrf_search", help="Perform a hybrid search using Reciprocal Rank Fusion")
    rrf_parser.add_argument("query", type=str, help="Search query")
    rrf_parser.add_argument("--limit", type=int, default=5, help="Number of search results to return")
    rrf_parser.add_argument("--k", type=int, default=60, help="Parameter k for Reciprocal Rank Fusion")
    rrf_parser.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enhancement method")
    rrf_parser.add_argument("--rerank_method", type=str, choices=["individual", "batch", "cross_encoder"], help="Reranking method")
    rrf_parser.add_argument("--debug", type=str, help="Enable debug mode")
    rrf_parser.add_argument("--evaluate", action="store_true", help="Run LLM as a judge on results")

    args = parser.parse_args()

    match args.command:
        case "rrf_search":
            rrf_search(args.query, args.k, args.limit, args.enhance, args.rerank_method, args.debug, args.evaluate)
        case "weighted_search":
            weighted_search(args.query, args.alpha, args.limit)
        case "normalize":
            norm_scores = normalize_scores(args.scores)
            for norm_score in norm_scores:
                print(f"* {norm_score:.4f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()