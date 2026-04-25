import argparse
from lib.multimodal_search import verify_image_embedding, image_search_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify_image_embedding", help="Verify image embedding generation")
    verify_parser.add_argument("image_fpath", type=str, help="Path to the image file to verify embedding generation")
    
    ims = subparsers.add_parser("image_search", help="Search movies using an image query")
    ims.add_argument("image_fpath", type=str, help="Path to the image file to use as a search query")
    ims.add_argument("--limit", type=int, default=5, help="Number of search results to return")

    args = parser.parse_args()

    match args.command:
        case "image_search":
            image_search_command(args.image_fpath, args.limit)
        case "verify_image_embedding":
            verify_image_embedding(args.image_fpath)
        case _:
            parser.print_help()
            
if __name__ == "__main__":
    main()