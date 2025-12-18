import pickle, numpy as np, os
from sklearn.feature_extraction.text import TfidfVectorizer

class VCEDatabase:
    def __init__(self, path="vcaa_simple_embeddings.pkl"):
        self.vectorizer = TfidfVectorizer(max_features=800)
        self.tfidf_matrix = None
        self.chunks = []
        self.metas = []
        self._load_db(path)
    
    def _load_db(self, path):
        """Load the TF-IDF database. Prints explicit errors to logs."""
        try:
            print(f"VCAA DB: Attempting to load file from: {path}")
            print(f"VCAA DB: Current working dir is: {os.getcwd()}")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Database file not found at: {path}")
            with open(path, 'rb') as f:
                data = pickle.load(f)
            self.chunks = data.get('chunks', [])
            self.metas = data.get('metas', [])
            # Fit vectorizer on loaded chunks to ensure vocabulary alignment
            if self.chunks:
                self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks)
            else:
                self.tfidf_matrix = None
            print(f"VCAA DB: Successfully loaded {len(self.chunks)} chunks.")
            if len(self.chunks) == 0:
                print("VCAA DB WARNING: Database is empty.")
        except Exception as e:
            print(f"VCAA DB CRITICAL ERROR: {type(e).__name__}: {e}")
            self.chunks = []
            self.metas = []
            self.tfidf_matrix = None
    
    def search(self, query, k=3, subject_filter=None):
        if not self.chunks: return []
        query_vec = self.vectorizer.transform([query])
        similarities = (self.tfidf_matrix * query_vec.T).toarray().flatten()
        
        # Apply subject filtering if provided
        if subject_filter:
            # Normalize allowed subjects to lowercase
            allowed = set(s.lower() for s in subject_filter)
            # Mask out irrelevant subjects
            for i in range(len(similarities)):
                if similarities[i] <= 0: continue
                
                # Get chunk subject
                subj_meta = str(self.metas[i].get('subject', '')).lower()
                # If the chunk's subject doesn't match any of our allowed subjects, zero the score
                # We check if any allowed subject string appears in the meta subject string
                if not any(a in subj_meta for a in allowed):
                    similarities[i] = 0.0

        top_idx = np.argsort(similarities)[-k:][::-1]
        return [(self.chunks[i], self.metas[i], float(similarities[i]))
                for i in top_idx if similarities[i] > 0.05][:k]
