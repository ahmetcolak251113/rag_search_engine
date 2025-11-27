import argparse
import json
import os
from lib.hybrid_search import rrf_search_command


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument("--limit", type=int, default=5,
                        help="Number of results to evaluate (k for precision@k, recall@k)", )

    args = parser.parse_args()
    limit = args.limit

    # 1. Dosya Yolu Kontrolü
    file_path = "golden_dataset.json"
    if not os.path.exists(file_path):
        file_path = "data/golden_dataset.json"

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: golden_dataset.json file not found in root or data/ folder.")
        return

    test_cases = []

    if isinstance(raw_data, dict) and "test_cases" in raw_data:
        test_cases = raw_data["test_cases"]
    elif isinstance(raw_data, list):
        test_cases = raw_data

    if not test_cases:
        print("Error: Could not find 'test_cases' in the JSON file.")
        return

    print(f"k={limit}\n")

    for case in test_cases:
        query = case.get("query")

        relevant_docs = case.get("relevant_docs", [])
        if relevant_docs is None:
            relevant_docs = []

        relevant_titles = [str(doc) for doc in relevant_docs]

        # RRF araması (k=10 olarak kalmış, dilerseniz görevde k=60 istenip istenmediğini kontrol edin)
        search_result = rrf_search_command(query, k=60, limit=limit)  # Genelde 60 istenir, kodunuzda 10'du.
        results = search_result.get("results", [])

        retrieved_titles = [res["title"] for res in results]

        matches = 0
        for title in retrieved_titles:
            if title in relevant_titles:
                matches += 1

        # Precision
        precision = matches / limit if limit > 0 else 0.0

        # Recall
        total_relevant = len(relevant_titles)
        recall = matches / total_relevant if total_relevant > 0 else 0.0

        # F1 Score (Sıfıra bölünme hatasını önlemek için kontrol)
        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        # DÜZELTME 1: "F1 Score:" formatı
        print(f"  - F1 Score: {f1:.4f}")
        print(f"  - Retrieved: {', '.join(retrieved_titles)}")
        # DÜZELTME 2: Relevant satırı eklendi
        print(f"  - Relevant: {', '.join(relevant_titles)}")
        print()


if __name__ == "__main__":
    main()