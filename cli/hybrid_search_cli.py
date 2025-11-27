import argparse
from lib.hybrid_search import normalize_scores, weighted_search_command, rrf_search_command
from google import genai
import os
from dotenv import load_dotenv
import time
import json
from sentence_transformers import CrossEncoder

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# API Anahtarının yüklenip yüklenmediğini kontrol edin
if not API_KEY:
    raise ValueError("GEMINI_API_KEY could not be loaded from .env file.")

# CLIENT oluşturulur
CLIENT = genai.Client(api_key=API_KEY)


def enhance_query_spell_correction(query: str) -> str:
    """Gemini kullanarak sorgudaki yazım hatalarını düzeltir."""
    system_prompt = f"""Fix any spelling errors in this movie search query.

Only correct obvious typos. Don't change correctly spelled words.

Query: "{query}"

If no errors, return the original query.
Corrected:"""

    response = CLIENT.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=system_prompt,
    )

    return response.text.strip()


def enhance_query_rewrite(query: str) -> str:
    """Gemini kullanarak belirsiz kullanıcı sorgusunu arama için daha uygun olacak şekilde yeniden yazar."""
    system_prompt = f"""Rewrite this movie search query to be more specific and searchable.

Original: "{query}"

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep it concise (under 10 words)
- It should be a google style search query that's very specific
- Don't use boolean logic

Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

Rewritten query:"""

    response = CLIENT.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=system_prompt,
    )
    return response.text.strip()


def enhance_query_expand(query: str) -> str:
    """Gemini kullanarak sorguyu ilgili terimlerle genişletir."""
    system_prompt = f"""Expand this movie search query with related terms, including synonyms, genres, and possible famous people if relevant.

Add concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
This will be appended to the original query.

IMPORTANT: The output MUST contain only English words, separated by spaces. Do not use punctuation, quotes, or numbers.

Examples:

- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

Query: "{query}"
"""

    response = CLIENT.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=system_prompt,
    )
    return response.text.strip().replace('"', '').replace('\n', ' ').strip()


def get_llm_rerank_score(query: str, doc: dict) -> float:
    """Tek bir belge için LLM'den yeniden sıralama puanı (0-10) alır."""
    doc_title = doc.get("title", "")
    doc_summary = doc.get("document", "")[:500]

    system_prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc_title} - {doc_summary}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Give me ONLY the number in your response, no other text or explanation.

Score:"""

    try:
        response = CLIENT.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=system_prompt,
        )
        score_text = response.text.strip()
        if "/" in score_text:
            score_text = score_text.split('/')[0].strip()

        return float(score_text)
    except Exception as e:
        print(f"Error getting rerank score for document: {doc_title}. Error: {e}")
        return 0.0


def get_llm_batch_ranks(query: str, docs: list[dict]) -> list[int]:
    """Tüm belgeleri tek bir LLM isteğiyle sıralar ve ID'lerin JSON listesini döndürür."""
    doc_list_str = ""
    for i, doc in enumerate(docs, 1):
        doc_title = doc.get("title", "")
        doc_summary = doc.get("document", "")[:200]
        doc_list_str += f"\n  ID: {doc.get('id', 0)}, Title: {doc_title}, Snippet: {doc_summary}..."

    system_prompt = \
        f"""
    Rank these movies by relevance to the search query. Pay special attention to "family" and "cartoon" themes.

    Query: "{query}"

    Movies:
    {doc_list_str}

    Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:
    [75, 12, 34, 2, 1]
    """

    try:
        # LLM isteğini tek bir kez yap
        response = CLIENT.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=system_prompt,
        )

        json_str = response.text.strip().replace("`", "").replace("json", "").strip()

        if json_str.startswith('[') and json_str.endswith(']'):
            pass
        else:
            start = json_str.find('[')
            end = json_str.rfind(']')
            if start != -1 and end != -1:
                json_str = json_str[start:end + 1]

        return json.loads(json_str)
    except Exception as e:
        print(f"\nError processing batch rerank JSON output. Prompt: {query}. Error: {e}")
        return []


# YENİ ÖZELLİK: LLM Değerlendirme Fonksiyonu
def get_llm_evaluation_scores(query: str, results: list[dict]) -> list[int]:
    """Final arama sonuçlarını LLM'e gönderir ve 0-3 ölçeğinde JSON puanları alır."""

    formatted_results = []
    for i, doc in enumerate(results, 1):
        # LLM'e gönderilecek her belgenin formatı
        formatted_results.append(f"{i}. Title: {doc.get('title', '')} - Snippet: {doc.get('document', '')[:100]}...")

    # Sonuç listesini '\n' ile birleştir
    formatted_results_str = '\n'.join(formatted_results)

    system_prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{formatted_results_str}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers out than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""

    try:
        response = CLIENT.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=system_prompt,
        )

        # JSON'ı temizle ve parse et
        json_str = response.text.strip().replace("`", "").replace("json", "").strip()

        # Puanların integer listesi olduğundan emin ol
        scores = json.loads(json_str)
        return [int(s) for s in scores]

    except Exception as e:
        print(f"\n[EVALUATION ERROR] Could not parse LLM score response: {e}")
        return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize a list of scores")
    normalize_parser.add_argument("scores", nargs="+", type=float, help="List of scores to normalize")

    weighted_parser = subparsers.add_parser("weighted-search", help="Perform weighted hybrid search")
    weighted_parser.add_argument("query", type=str, help="Search query")
    weighted_parser.add_argument("--alpha", type=float, default=0.5,
                                 help="Weight for BM25 vs semantic (0=all semantic, 1=all BM25, default=0.5)", )
    weighted_parser.add_argument("--limit", type=int, default=5, help="Number of results to return (default=5)")

    rrf_parser = subparsers.add_parser("rrf-search", help="Perform hybrid search using Reciprocal Rank Fusion")
    rrf_parser.add_argument("query", type=str, help="Search query")
    rrf_parser.add_argument("--k", type=int, default=60, help="RRF constant k (default=60)")
    rrf_parser.add_argument("--limit", type=int, default=5, help="Number of results to return (default=5)")
    rrf_parser.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"],
                            help="Query enhancement method (e.g., 'spell', 'rewrite', or 'expand')", )
    rrf_parser.add_argument("--rerank-method", type=str, choices=["individual", "batch", "cross_encoder"], default=None,
                            help="Reranking method to apply after RRF (e.g., 'individual' for LLM scoring, 'batch' for LLM ranking)", )
    # YENİ ÖZELLİK: --evaluate boolean flag'i eklendi
    rrf_parser.add_argument("--evaluate", action="store_true",
                            help="Evaluate final search results using an LLM on a 0-3 scale.", )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized = normalize_scores(args.scores)
            for score in normalized:
                print(f"* {score:.4f}")
        case "weighted-search":
            result = weighted_search_command(args.query, args.alpha, args.limit)

            print(
                f"Weighted Hybrid Search Results for '{result['query']}' (alpha={result['alpha']}):"
            )
            print(
                f"  Alpha {result['alpha']}: {int(result['alpha'] * 100)}% Keyword, {int((1 - result['alpha']) * 100)}% Semantic"
            )
            for i, res in enumerate(result["results"], 1):
                print(f"{i}. {res['title']}")
                print(f"   Hybrid Score: {res.get('score', 0):.3f}")
                metadata = res.get("metadata", {})
                if "bm25_score" in metadata and "semantic_score" in metadata:
                    print(f"   BM25: {metadata['bm25_score']:.3f}, Semantic: {metadata['semantic_score']:.3f}")
                print(f"   {res['document'][:100]}...")
                print()

        case "rrf-search":
            original_query = args.query
            enhanced_query = original_query
            method = None
            rerank_method = args.rerank_method

            # DEBUG LOGS START
            print(f"\n--- DEBUG: Pipeline Start ---")
            print(f"Original Query: '{original_query}'")
            print(f"Rerank Method: {rerank_method}")
            print(f"Initial Limit: {args.limit}")

            # 1. Query Enhancement
            if args.enhance == "spell":
                enhanced_query = enhance_query_spell_correction(original_query)
                method = "spell"
            elif args.enhance == "rewrite":
                enhanced_query = enhance_query_rewrite(original_query)
                method = "rewrite"
            elif args.enhance == "expand":
                expansion = enhance_query_expand(original_query)
                enhanced_query = f"{original_query} {expansion}"
                method = "expand"

            if method:
                print(f"Enhanced query ({method}): '{original_query}' -> '{enhanced_query}'\n")
            if enhanced_query != original_query:
                print(f"DEBUG: Enhanced Query Used: '{enhanced_query}'")

            # 2. RRF Arama Limitini Ayarlama
            initial_limit = args.limit
            search_limit = initial_limit

            if rerank_method is not None:
                search_limit = initial_limit * 5
                if rerank_method == "individual":
                    print(f"Reranking top {initial_limit} results using individual method...")
                elif rerank_method == "batch":
                    print(f"Reranking top {initial_limit} results using batch method...")
                elif rerank_method == "cross_encoder":
                    print(f"Reranking top {initial_limit} results using cross_encoder method...")

            # 3. RRF Aramasını Çalıştır (Veriyi Al)
            result = rrf_search_command(enhanced_query, args.k, search_limit)
            final_results = result["results"]

            print(f"\nDEBUG: Initial RRF Results (Top {min(len(final_results), 5)} of {len(final_results)} gathered):")
            for i, res in enumerate(final_results[:5]):
                print(
                    f"  {i + 1}. {res['title']} (RRF: {res['score']:.4f}, BM25 Rank: {res['metadata'].get('bm25_rank')})")

            # 4. Reranking Mantığı
            if rerank_method == "individual":
                reranked_docs = []
                for i, res in enumerate(final_results):
                    if i > 0:
                        time.sleep(3)
                    llm_score = get_llm_rerank_score(enhanced_query, res)
                    res["llm_rerank_score"] = llm_score
                    reranked_docs.append(res)
                final_results = sorted(
                    reranked_docs,
                    key=lambda x: x["llm_rerank_score"],
                    reverse=True
                )

            elif rerank_method == "batch":
                reranked_ids = get_llm_batch_ranks(enhanced_query, final_results)
                results_map = {doc['id']: doc for doc in final_results}
                new_ranked_results = []
                for rank, doc_id in enumerate(reranked_ids, 1):
                    if doc_id in results_map:
                        doc = results_map[doc_id]
                        doc["llm_rerank_rank"] = rank
                        new_ranked_results.append(doc)
                final_results = new_ranked_results

            elif rerank_method == "cross_encoder":
                pairs = []
                for doc in final_results:
                    pair_text = f"{doc.get('title', '')} - {doc.get('document', '')}"
                    pairs.append([enhanced_query, pair_text])

                cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
                scores = cross_encoder.predict(pairs)

                for i, doc in enumerate(final_results):
                    doc["cross_encoder_score"] = float(scores[i])

                final_results = sorted(
                    final_results,
                    key=lambda x: x["cross_encoder_score"],
                    reverse=True
                )

            # 5. Sonuçları Kırp
            final_results = final_results[:initial_limit]

            # DEBUG LOGS END
            print(f"\nDEBUG: Final {initial_limit} Results (After Reranking):")
            for i, res in enumerate(final_results):
                score_label = "CE Score" if rerank_method == "cross_encoder" else "LLM Rank"
                score = res.get("cross_encoder_score") or res.get("llm_rerank_score") or res.get("llm_rerank_rank")
                print(f"  {i + 1}. {res['title']} ({score_label}: {score})")
            print("--- DEBUG: End Pipeline ---")

            # YENİ ÖZELLİK: LLM Değerlendirme Raporu
            if args.evaluate:
                llm_scores = get_llm_evaluation_scores(enhanced_query, final_results)

                print("\n--- LLM Evaluation Report (0-3 Scale) ---")
                for i, doc in enumerate(final_results):
                    score = llm_scores[i] if i < len(llm_scores) else 0  # Puan yoksa 0 kullan
                    print(f"{i + 1}. {doc.get('title', 'N/A')}: {score}/3")
                print("------------------------------------------")

            # 6. Normal Çıktı Yazdırma
            if rerank_method == "batch":
                print()

            print(f"Reciprocal Rank Fusion Results for '{enhanced_query}' (k={args.k}):\n")

            for i, res in enumerate(final_results, 1):
                print(f"{i}. {res['title']}")

                if rerank_method == "individual":
                    llm_score = res.get("llm_rerank_score", 0.0)
                    print(f"   Rerank Score: {llm_score:.3f}/10")

                if rerank_method == "batch":
                    rerank_rank = res.get("llm_rerank_rank", "-")
                    print(f"   Rerank Rank: {rerank_rank}")

                if rerank_method == "cross_encoder":
                    ce_score = res.get("cross_encoder_score", 0.0)
                    print(f"   Cross Encoder Score: {ce_score:.3f}")

                # RRF Score her zaman yazdırılır
                print(f"   RRF Score: {res.get('score', 0):.3f}")

                metadata = res.get("metadata", {})
                bm25_rank = metadata.get("bm25_rank")
                semantic_rank = metadata.get("semantic_rank")

                bm25_str = str(bm25_rank) if bm25_rank is not None else "-"
                semantic_str = str(semantic_rank) if semantic_rank is not None else "-"

                print(f"   BM25 Rank: {bm25_str}, Semantic Rank: {semantic_str}")
                print(f"   {res['document'][:100]}...")
                print()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()