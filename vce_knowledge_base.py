import json, pickle, numpy as np
from sentence_transformers import SentenceTransformer
class VCEDatabase:
    def __init__(self): self.model = SentenceTransformer('all-MiniLM-L6-v2')
    def build(self):
        with open("vcaa_index.json", 'r') as f: docs = json.load(f)
        chunks, metas = [], []
        for doc in docs:
            try:
                with open(doc['path'], 'r', encoding='utf-8') as f: text = f.read()
                parts = text.split('. ')
                for i in range(0, len(parts), 5):
                    chunk = '. '.join(parts[i:i+5])
                    if len(chunk) > 100:
                        chunks.append(f"[{doc['subject']} {doc['year']}] {chunk}")
                        metas.append(doc)
            except: continue
        print(f"Creating embeddings for {len(chunks)} chunks...")
        embeddings = self.model.encode(chunks)
        with open("vcaa_embeddings.pkl", 'wb') as f:
            pickle.dump({"chunks":chunks,"embeddings":embeddings,"metas":metas},f)
        print(f"✅ Saved {len(chunks)} chunks")
    def search(self, query, k=3):
        with open("vcaa_embeddings.pkl", 'rb') as f: data = pickle.load(f)
        q_emb = self.model.encode([query])
        scores = np.dot(data["embeddings"], q_emb.T).flatten()
        top = np.argsort(scores)[-k:][::-1]
        return [(data["chunks"][i], data["metas"][i], scores[i]) for i in top]
if __name__ == "__main__": VCEDatabase().build()
