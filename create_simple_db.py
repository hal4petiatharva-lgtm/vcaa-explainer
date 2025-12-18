import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    with open("vcaa_embeddings.pkl", 'rb') as f:
        data = pickle.load(f)

    vectorizer = TfidfVectorizer(max_features=800)
    tfidf_matrix = vectorizer.fit_transform(data['chunks'])

    simple_db = {
        'chunks': data['chunks'],
        'metas': data['metas'],
        'tfidf_matrix': tfidf_matrix,
    }

    with open("vcaa_simple_embeddings.pkl", 'wb') as f:
        pickle.dump(simple_db, f, protocol=4)

    print(f"✅ Created TF-IDF DB: {len(data['chunks'])} chunks")

except FileNotFoundError:
    print("❌ Error: vcaa_embeddings.pkl not found. Please ensure the source embeddings file is present.")
except Exception as e:
    print(f"❌ Error: {str(e)}")
