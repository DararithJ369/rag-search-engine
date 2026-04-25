import argparse
from lib.rag import query_answering, doc_summarization, doc_citations, answer_detailed_question


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    qa_parser = subparsers.add_parser("rag", help="Perform RAG (search + generate answer)")
    qa_parser.add_argument("query", type=str, help="Search query for RAG")
    
    sum_parser = subparsers.add_parser("summarize", help="Perform RAG (search + generate summary)")
    sum_parser.add_argument("query", type=str, help="Search query for summarization")
    sum_parser.add_argument("--limit", type=int, default=5, help="Limit the number of search results for summarization")

    ci_parser = subparsers.add_parser("citations", help="Retrieve documents with citations for a query")
    ci_parser.add_argument("query", type=str, help="Search query for retrieving documents with citations")
    ci_parser.add_argument("--limit", type=int, default=5, help="Limit the number of search results for citations")

    detailed_qa_parser = subparsers.add_parser("question", help="Perform detailed question answering with RAG")
    detailed_qa_parser.add_argument("query", type=str, help="Search query for detailed question answering")
    detailed_qa_parser.add_argument("--limit", type=int, default=5, help="Limit the number of search results for detailed question answering")

    args = parser.parse_args()


    match args.command:
        case "question":
            answer_detailed_question(args.query, args.limit)
        case "citations":
            doc_citations(args.query, args.limit)
        case "summarize":
            doc_summarization(args.query, args.limit)
        case "rag":
            query_answering(args.query)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()