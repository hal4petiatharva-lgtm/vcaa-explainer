import os
parts = ["vcaa_db.aa", "vcaa_db.ab", "vcaa_db.ac", "vcaa_db.ad", "vcaa_db.ae"]
with open("vcaa_simple_embeddings.pkl", "wb") as f:
    for p in parts:
        if os.path.exists(p):
            f.write(open(p, "rb").read())
            print(f"Added {p}")
print("Database joined.")
