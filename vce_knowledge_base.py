import pickle
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer


class VCEDatabase:
    def __init__(self, embeddings_path="vcaa_embeddings.pkl"):
        self.embeddings_path = embeddings_path
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.tfidf_matrix = None
        self.chunks = []
        self.metas = []
        self._load_database()

    def _load_database(self):
        try:
            path = self.embeddings_path if os.path.exists(self.embeddings_path) else "vcaa_simple_embeddings.pkl"
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.chunks = data.get("chunks", [])
            self.metas = data.get("metas", [])
            if "tfidf_matrix" in data and data["tfidf_matrix"] is not None:
                self.tfidf_matrix = data["tfidf_matrix"]
            else:
                self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks)
        except Exception:
            self.chunks = []
            self.metas = []
            self.tfidf_matrix = None

    def search(self, query, k=3):
        if not self.chunks:
            return []
        if self.tfidf_matrix is None:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks)
        query_vec = self.vectorizer.transform([query])
        similarities = (self.tfidf_matrix * query_vec.T).toarray().flatten()
        top_idx = np.argsort(similarities)[-k:][::-1]
        results = []
        for i in top_idx:
            score = float(similarities[i])
            if score > 0.01:
                results.append((self.chunks[i], self.metas[i], score))
        return results
