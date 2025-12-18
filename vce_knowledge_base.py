import pickle, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class VCEDatabase:
    def __init__(self, path="vcaa_simple_embeddings.pkl"):
        self.vectorizer = TfidfVectorizer(max_features=800)
        self.tfidf_matrix = None
        self.chunks = []
        self.metas = []
        self._load_db(path)
    
    def _load_db(self, path):
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            self.chunks = data['chunks']
            self.metas = data['metas']
            self.tfidf_matrix = data['tfidf_matrix']
        except: pass
    
    def search(self, query, k=3):
        if not self.chunks: return []
        query_vec = self.vectorizer.transform([query])
        similarities = (self.tfidf_matrix * query_vec.T).toarray().flatten()
        top_idx = np.argsort(similarities)[-k:][::-1]
        return [(self.chunks[i], self.metas[i], float(similarities[i]))
                for i in top_idx if similarities[i] > 0.05][:k]
