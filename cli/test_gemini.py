
import os
from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    print("ERROR: GEMINI_API_KEY environment variable could not be loaded. Please check your .env file.")
else:
    print(f"Using key {api_key[:6]}...")

    try:
        client = genai.Client(api_key=api_key)

        model_name = "gemini-2.0-flash-001"
        prompt = "Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum."

        print(f"\nModel: {model_name}")
        print(f"Prompt: \"{prompt}\"")
        print("---")

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )

        print("Model Response:")
        print(response.text)
        print("---")

        usage = response.usage_metadata
        print("Token Consumption Metrics:")
        print(f"Prompt Tokens: {usage.prompt_token_count}")
        print(f"Response Tokens: {usage.candidates_token_count}")

    except APIError as e:
        print("\nAPI Error Occurred: Could not get a response from Gemini.")
        print(f"Error Details: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
