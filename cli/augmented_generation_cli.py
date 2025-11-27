import argparse
import os
import json
from dotenv import load_dotenv
from google import genai
from lib.search_utils import load_movies
from lib.hybrid_search import rrf_search_command

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")
CLIENT = genai.Client(api_key=API_KEY)


def _format_docs_for_llm(search_results: list[dict]) -> str:
    docs_str = ""
    for i, movie in enumerate(search_results):
        docs_str += f"--- Document {i + 1} ---\n"
        docs_str += f"Title: {movie.get('title', "Unknown Title")}\n"
        docs_str += f"Snippet: {movie.get('document', 'No description available.')}\n\n"
    return docs_str.strip()


def get_rag(query: str, limit: int = 5) -> tuple[list[str], str]:
    rrf_result = rrf_search_command(query, k=60, limit=limit)
    search_results = rrf_result["results"]
    docs_str = _format_docs_for_llm(search_results)
    search_titles = [movie.get("title", "Unknown Title") for movie in search_results]

    prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.
Query: {query}

Documents:
{docs_str}

Provide a comprehensive answer that addresses the query:"""

    try:
        response = CLIENT.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        rag_response = response.text.strip()
    except Exception as e:
        rag_response = f"LLM Call Error: {e}"

    return search_titles, rag_response


def summarize(query: str, limit: int = 5) -> tuple[list[str], str]:
    rrf_result = rrf_search_command(query, k=60, limit=limit)
    search_results = rrf_result['results']

    docs_str = _format_docs_for_llm(search_results)
    search_titles = [movie.get('title', 'Unknown Title') for movie in search_results]

    prompt = f"""
Provide information useful to this query by synthesizing information from multiple search results in detail.
The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
This should be tailored to Hoopla users. Hoopla is a movie streaming service.
Query: {query}
Search Results:
{docs_str}
Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
"""

    try:
        response = CLIENT.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        summary_response = response.text.strip()
    except Exception as e:
        summary_response = f"LLM Call Error: {e}"

    return search_titles, summary_response


def citations(query: str, limit: int = 5) -> tuple[list[str], str]:
    rrf_result = rrf_search_command(query, k=60, limit=limit)
    search_results = rrf_result['results']

    docs_str = _format_docs_for_llm(search_results)
    search_titles = [movie.get('title', 'Unknown Title') for movie in search_results]

    prompt = f"""Answer the question or provide information based on the provided documents.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

Query: {query}

Documents:
{docs_str}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources using [1], [2], etc. format when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the documents, say "I don't have enough information"
- Be direct and informative

Answer:"""

    try:
        response = CLIENT.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        citations_response = response.text.strip()
    except Exception as e:
        citations_response = f"LLM Call Error: {e}"
    return search_titles, citations_response


def question(question: str, limit: int = 5) -> tuple[list[str], str]:
    rrf_result = rrf_search_command(question, k=60, limit=limit)
    search_results = rrf_result['results']

    docs_str = _format_docs_for_llm(search_results)
    search_titles = [movie.get('title', 'Unknown Title') for movie in search_results]

    # GÖREVİN İSTEDİĞİ CONVERSATIONAL PROMPT KULLANILDI
    prompt = f"""Answer the user's question based on the provided movies that are available on Hoopla.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Question: {question}

Documents:
{docs_str}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Answer:"""

    try:
        response = CLIENT.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        question_response = response.text.strip()
    except Exception as e:
        question_response = f"LLM Call Error: {e}"

    return search_titles, question_response


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser("rag", help="Perform RAG (search + generate answer)")
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser("summarize", help="Perform RAG to summarize search results")
    summarize_parser.add_argument("query", type=str, help="Search query for summarization")
    summarize_parser.add_argument("--limit", type=int, default=5,
                                  help="Number of search results to retrieve (default: 5)")

    citations_parser = subparsers.add_parser("citations", help="Perform RAG with citations")
    citations_parser.add_argument("query", type=str, help="Search query for citations")
    citations_parser.add_argument("--limit", type=int, default=5,
                                  help="Number of search results to retrieve (default: 5)")

    # KOMUT ADI VE ARGÜMAN ADI DÜZELTİLDİ: "questions" -> "question"
    question_parser = subparsers.add_parser("question", help="Answer user questions using RAG")
    question_parser.add_argument("question", type=str, help="Question to answer using RAG")  # Argüman adı düzeltildi
    question_parser.add_argument("--limit", type=int, default=5,
                                 help="Number of search results to retrieve (default: 5)")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            limit = 5
            search_titles, rag_response = get_rag(query, limit)

            print("Search Results:")
            for title in search_titles:
                print(f"  - {title}")

            print("\nRAG Response:")
            print(rag_response)

        case "summarize":
            query = args.query
            limit = args.limit
            search_titles, summary_response = summarize(query, limit)

            print("Search Results:")
            for title in search_titles:
                print(f"  - {title}")

            print("\nLLM Summary:")
            print(summary_response)

        case "citations":
            query = args.query
            limit = args.limit
            search_titles, citations_response = citations(query, limit)
            print("Search Results:")
            for title in search_titles:
                print(f"  - {title}")
            print("\nLLM Answer:")
            print(citations_response)

        case "question":
            question_text = args.question
            limit = args.limit
            search_titles, question_response = question(question_text, limit)
            print("Search Results:")
            for title in search_titles:
                print(f"  - {title}")
            print("\nAnswer:")
            print(question_response)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
