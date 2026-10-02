"""Step 2b: compute features and build the retrieval index.

Usage:
    python -m kitaabcopy.build_index                 # TF-IDF only
    python -m kitaabcopy.build_index --embeddings    # + sentence-transformers dense vectors (optional)
"""
from __future__ import annotations

import argparse
import time

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from . import config
from .features import add_features
from .text import build_doc, parse_shelves


def build(books_parquet=config.BOOKS_PARQUET, out_dir=config.ARTIFACTS, embeddings: bool = False) -> None:
    out_dir = config.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet(books_parquet).reset_index(drop=True)
    print(f"loaded {len(df):,} books")

    df = add_features(df)
    print(f"features done ({time.time() - t0:.0f}s)")

    docs = [
        build_doc(t, d, parse_shelves(s))
        for t, d, s in zip(df["title"].fillna(""), df["description"].fillna(""), df["shelves_json"])
    ]
    vec = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b", stop_words="english", ngram_range=(1, 2), min_df=2,
        max_df=0.5, sublinear_tf=True, max_features=400_000, dtype=np.float32,
    )
    X = vec.fit_transform(docs)  # rows are L2-normalized -> dot product == cosine
    print(f"tf-idf {X.shape} ({time.time() - t0:.0f}s)")

    df.to_parquet(out_dir / "books_features.parquet", index=False)
    sp.save_npz(out_dir / "tfidf_matrix.npz", X)
    joblib.dump(vec, out_dir / "tfidf_vectorizer.joblib")

    if embeddings:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        texts = [d[:1200] for d in docs]
        emb = model.encode(texts, batch_size=128, show_progress_bar=True, normalize_embeddings=True)
        np.save(out_dir / "embeddings.npy", emb.astype(np.float16))
        print("dense embeddings saved")
    print(f"done in {time.time() - t0:.0f}s -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", default=str(config.BOOKS_PARQUET))
    ap.add_argument("--out", default=str(config.ARTIFACTS))
    ap.add_argument("--embeddings", action="store_true")
    a = ap.parse_args()
    build(a.books, a.out, a.embeddings)


if __name__ == "__main__":
    main()
