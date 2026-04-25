#!/usr/bin/env python3

import argparse
from lib.semantic_search import (
    verify_embeddings, 
    verify_model, 
    embed_text,
    embed_query_text,
    search,
    chunk_text,
    chunk_text_semantic,
    embed_chunks,
    search_chunked,
)
def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="Verify the semantic search model")
    
    embed_parser = subparsers.add_parser("embed_text", help="Generate embeddings for a given text")
    embed_parser.add_argument("text", type=str, help="Text to generate embeddings for")

    subparsers.add_parser("verify_embeddings", help="Verify that embeddings are generated and cached correctly")

    embedquery_parser = subparsers.add_parser("embedquery", help="Generate embeddings for a given query text")
    embedquery_parser.add_argument("query", type=str, help="Query text to generate embeddings for")

    search_parser = subparsers.add_parser("search", help="Search movies using semantic search")         
    search_parser.add_argument("query", type=str, help="Query text to search for")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of search results to return")

    chunk_parser = subparsers.add_parser("chunk", help="Chunk a long text into smaller pieces")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk_size", type=int, default=200, help="Chunk size in words")
    chunk_parser.add_argument("--overlap", type=int, default=50, help="Number of overlapping words between chunks")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Chunk a long text into smaller pieces using semantic boundaries")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument("--max_chunk_size", type=int, default=4, help="Maximum number of sentences per chunk")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0, help="Number of overlapping sentences between chunks")
    
    embed_chunks_parser = subparsers.add_parser("embed_chunks", help="Generate embeddings for all movie description chunks")
    
    search_chunked_parser = subparsers.add_parser("search_chunked", help="Search movies using chunked semantic search")
    search_chunked_parser.add_argument("query", type=str, help="Query text to search for")
    search_chunked_parser.add_argument("--limit", type=int, default=5, help="Number of search results to return")

    args = parser.parse_args()

    match args.command:
        case "search_chunked":
            search_chunked(args.query, args.limit)
        case "embed_chunks":
            embed_chunks()
        case "semantic_chunk":
            chunk_text_semantic(args.text, args.overlap, args.max_chunk_size)
        case "chunk":
            chunk_text(args.text, args.overlap, args.chunk_size)
        case "search":
            search(args.query, args.limit)
        case "embedquery":
            embed_query_text(args.query)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_text":
            embed_text(args.text)
        case "verify":
            verify_model()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()