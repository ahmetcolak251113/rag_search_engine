import argparse
from lib.multimodal_search import verify_image_embedding, image_search_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify_image_embedding",                                  help="Loads an image, generate an embedding, and prints its shape.")
    verify_parser.add_argument("image_path", type=str, help="Path to the image file.")

    image_search_parser = subparsers.add_parser("image_search", help="Image Search CLI")
    image_search_parser.add_argument("image_path", type=str, help="Path to the image file.")
    image_search_parser.add_argument("--limit", type=int, default=5, help="Number of top results to return.")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image_path)

        case "image_search":
            results = image_search_command(args.image_path, args.limit)

            for i, res in enumerate(results, 1):
                score = res.get("score", 0.0)
                print(f"{i}. {res['title']} (similarity: {score:.3f})")
                print(f"   {res['document'][:100]}...")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()