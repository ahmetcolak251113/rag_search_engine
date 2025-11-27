from PIL import Image
from sentence_transformers import SentenceTransformer
import numpy as np
from .search_utils import load_movies, format_search_result


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return dot_product / (norm_vec1 * norm_vec2)


class MultimodalSearch:
    def __init__(self, documents: list[dict], model_name="clip-ViT-B-32") -> None:
        self.model = SentenceTransformer(model_name)
        self.documents = documents

        self.texts = []
        for doc in self.documents:
            self.texts.append(f"{doc['title']}: {doc['description']}")

        print("Generating text embeddings (this may take a moment)...")
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)
        print("Text embeddings generation complete.")

    def embed_image(self, image_path: str) -> np.ndarray:
        img = Image.open(image_path)
        embedding_list = self.model.encode([img])
        return embedding_list[0]

    def search_with_image(self, image_path: str, limit: int = 5) -> list[dict]:
        image_embedding = self.embed_image(image_path)
        similarities = []

        for i, text_embedding in enumerate(self.text_embeddings):
            score = cosine_similarity(image_embedding, text_embedding)
            doc = self.documents[i]

            similarities.append((score, doc))

        similarities.sort(key=lambda x: x[0], reverse=True)

        results = []

        for score, doc in similarities[:limit]:
            results.append(
                format_search_result(
                    doc_id=doc["id"],
                    title=doc["title"],
                    document=doc["description"],
                    score=score,
                )
            )
        return results


def verify_image_embedding(image_path: str) -> None:
    searcher = MultimodalSearch(documents=[{"title": "Test", "description": "Test", "id": 0}])
    embedding = searcher.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search_command(image_path: str, limit: int = 5) -> list[dict]:
    movies = load_movies()
    searcher = MultimodalSearch(movies)

    results = searcher.search_with_image(image_path, limit)
    return results