import argparse
import os
import mimetypes
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

def rewrite_query_with_image(image_path: str, query: str):

    if not API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    client = genai.Client(api_key=API_KEY)
    if not Path(image_path).is_file():
        raise FileNotFoundError(f"Image file not found at: {image_path}")

    mime, _ = mimetypes.guess_type(image_path)
    mime = mime or "image/jpeg"

    with open(image_path, "rb") as img_file:
        img_bytes = img_file.read()


    system_prompt = (
        "Given the included image and text query, rewrite the text query to improve search results from a movie database. "
        "Make sure to:\n"
        "- Synthesize visual and textual information\n"
        "- Focus on movie-specific details (actors, scenes, style, etc.)\n"
        "- Return only the rewritten query, without any additional commentary"
    )

    parts = [system_prompt, types.Part.from_bytes(data=img_bytes, mime_type=mime), query.strip()]

    try:
        response = client.models.generate_content(model="gemini-2.5-flash",contents=parts,)
    except Exception as e:
        print(f"Error during Gemini API call: {e}")
        return

    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")

def main():
    parser = argparse.ArgumentParser(description="Rewrite a search query using image context.")

    parser.add_argument("--image",type=str,required=True,help="Path to the image file (e.g., data/paddington.jpeg)")
    parser.add_argument("--query",type=str,required=True,help="Text query to be rewritten based on the image.")

    args = parser.parse_args()

    try:
        rewrite_query_with_image(args.image, args.query)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()