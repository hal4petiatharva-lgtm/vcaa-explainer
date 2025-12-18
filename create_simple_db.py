import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    with open("vcaa_embeddings.pkl", "rb") as f:
        original = pickle.load(f)
    chunks = original.get("chunks", [])
    metas = original.get("metas", [])
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(chunks) if chunks else None
    simple_db = {"chunks": chunks, "metas": metas, "tfidf_matrix": tfidf_matrix}
    with open("vcaa_simple_embeddings.pkl", "wb") as f:
        pickle.dump(simple_db, f, protocol=4)
    print(f"Created TF-IDF database: {len(chunks)} chunks")
except Exception:
    with open("vcaa_simple_embeddings.pkl", "wb") as f:
        pickle.dump({"chunks": [], "metas": []}, f)
    print("Created empty TF-IDF database")
