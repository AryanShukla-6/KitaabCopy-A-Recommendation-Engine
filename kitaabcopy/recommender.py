"""Step 3: the recommendation engine.

    query (subject, level, genre)
      -> hybrid retrieval (TF-IDF [+ dense]) of N candidates
      -> multi-signal scoring: relevance, shelf affinity, level match, genre match, quality
      -> MMR re-ranking with author / near-duplicate constraints for diversity
      -> template explanations
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import config
from .explain import explain_pick
from .text import is_generic_shelf, norm_text, parse_shelves


@dataclass
class Query:
    subject: str
    level: str = "Intermediate"
    genre: str = "Practical"

    def key(self) -> str:
        return f"{self.subject}|{self.level}|{self.genre}"


@dataclass
class Settings:
    """Everything an ablation / tuning run may change."""
    weights: dict[str, float] = field(default_factory=lambda: dict(config.DEFAULT_WEIGHTS))
    diversity: bool = True
    mmr_lambda: float = config.MMR_LAMBDA
    n_candidates: int = config.N_CANDIDATES
    gate: float = 0.30            # candidates with relevance < gate * best are down-weighted
    dedupe_sim: float = 0.85      # near-duplicate threshold for diversity filtering


def expand_subject(subject: str) -> tuple[str, list[str]]:
    """Return (query text for TF-IDF, normalized phrases used for shelf matching)."""
    s = norm_text(subject)
    aliases = [norm_text(a) for a in config.SUBJECT_ALIASES.get(s, [])]
    text = " ".join([s] * 3 + aliases)
    phrases = [s] + aliases
    return text, phrases


class KitaabRecommender:
    def __init__(self, artifacts_dir: str | Path = config.ARTIFACTS, use_dense: bool = True):
        d = Path(artifacts_dir)
        self.df = pd.read_parquet(d / "books_features.parquet").reset_index(drop=True)
        self.X: sp.csr_matrix = sp.load_npz(d / "tfidf_matrix.npz").tocsr()
        self.vec = joblib.load(d / "tfidf_vectorizer.joblib")
        self.shelves = [parse_shelves(s) for s in self.df["shelves_json"]]
        self.emb = None
        self._dense_model = None
        emb_path = d / "embeddings.npy"
        if use_dense and emb_path.exists():
            self.emb = np.load(emb_path).astype(np.float32)

    # ------------------------------------------------------------------ retrieval
    def _relevance(self, query_text: str) -> tuple[np.ndarray, np.ndarray]:
        q = self.vec.transform([query_text])
        tf = (self.X @ q.T).toarray().ravel()
        if self.emb is None:
            return tf, tf
        if self._dense_model is None:
            from sentence_transformers import SentenceTransformer

            self._dense_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        qe = self._dense_model.encode([query_text], normalize_embeddings=True)[0].astype(np.float32)
        dn = self.emb @ qe
        hybrid = 0.5 * tf / (tf.max() + 1e-9) + 0.5 * np.clip(dn, 0, None) / (dn.max() + 1e-9)
        return hybrid, tf

    def _shelf_affinity(self, idx: int, phrases: list[str]) -> tuple[float, list[tuple[str, int]]]:
        shelves = [(n, c) for n, c in self.shelves[idx][:30] if not is_generic_shelf(n)]
        total = sum(c for _, c in shelves)
        if total <= 0:
            return 0.0, []
        matched, hits = 0, []
        for name, cnt in shelves:
            padded = f" {norm_text(name)} "
            if any(f" {p} " in padded for p in phrases if p):
                matched += cnt
                hits.append((name, cnt))
        return min(1.0, 4.0 * matched / total), hits

    # ------------------------------------------------------------------ main entry
    def recommend(
        self, query: Query, k: int = 3, settings: Settings | None = None, explain: bool = True
    ) -> list[dict]:
        st = settings or Settings()
        w = st.weights
        qtext, phrases = expand_subject(query.subject)
        rel_all, tf_all = self._relevance(qtext)

        n = min(st.n_candidates, len(rel_all))
        cand = np.argpartition(-rel_all, n - 1)[:n]
        cand = cand[tf_all[cand] >= config.MIN_RELEVANCE]
        if len(cand) == 0:
            return []
        rel = rel_all[cand]
        rel_n = rel / (rel.max() + 1e-9)

        aff, hits = np.zeros(len(cand)), []
        for j, i in enumerate(cand):
            a, h = self._shelf_affinity(int(i), phrases)
            aff[j] = a
            hits.append(h)

        sub = self.df.iloc[cand]
        p_level = sub[f"p_{query.level.lower()}"].to_numpy()
        p_genre = sub[f"p_{query.genre.lower()}"].to_numpy()
        quality = sub["quality"].to_numpy()

        score = (
            w.get("relevance", 0) * rel_n + w.get("shelf", 0) * aff + w.get("level", 0) * p_level
            + w.get("genre", 0) * p_genre + w.get("quality", 0) * quality
        )
        if st.gate > 0:
            score = score * np.clip(rel_n / st.gate, 0, 1)

        order = self._select(cand, score, k, st)
        picks: list[dict] = []
        Xc = self.X[cand]
        for rank, j in enumerate(order):
            row = sub.iloc[j]
            sim_prev = None
            if rank > 0:
                sim_prev = float(max((Xc[j] @ Xc[o].T).toarray()[0, 0] for o in order[:rank]))
            rec = {
                "rank": rank + 1,
                "book_id": row["book_id"],
                "title": row["title"],
                "authors": row.get("author_names") or "Unknown",
                "author_ids": row.get("author_ids") or "",
                "rating": float(row["average_rating"]),
                "ratings_count": int(row["ratings_count"]),
                "year": None if pd.isna(row.get("publication_year")) else int(row["publication_year"]),
                "pages": None if pd.isna(row.get("num_pages")) else int(row["num_pages"]),
                "image_url": row.get("image_url"),
                "url": row.get("url") or row.get("link") or f"https://www.goodreads.com/book/show/{row['book_id']}",
                "topics": row.get("topics", ""),
                "scores": {
                    "relevance": float(rel_n[j]), "shelf": float(aff[j]), "level": float(p_level[j]),
                    "genre": float(p_genre[j]), "quality": float(quality[j]), "final": float(score[j]),
                },
                "shelf_hits": hits[j],
                "max_sim_to_previous": sim_prev,
            }
            if explain:
                rec["explanation"] = explain_pick(rec, row, query)
            picks.append(rec)
        return picks

    # ------------------------------------------------------------------ MMR
    def _select(self, cand: np.ndarray, score: np.ndarray, k: int, st: Settings) -> list[int]:
        k = min(k, len(cand))
        if not st.diversity:
            return list(np.argsort(-score)[:k])

        Xc = self.X[cand]
        sim = (Xc @ Xc.T).toarray()
        s_norm = (score - score.min()) / (score.max() - score.min() + 1e-9)
        author_sets = [set(str(a).split("|")) - {""} for a in self.df.iloc[cand]["author_ids"].fillna("")]
        order = list(np.argsort(-score))
        selected = [order[0]]
        while len(selected) < k:
            best_j, best_val = None, -math.inf
            for j in order:
                if j in selected:
                    continue
                if any(author_sets[j] & author_sets[o] for o in selected):
                    continue                                   # same author as an earlier pick
                max_sim = max(sim[j, o] for o in selected)
                if max_sim >= st.dedupe_sim:
                    continue                                   # near-duplicate (other edition / reprint)
                val = st.mmr_lambda * s_norm[j] - (1 - st.mmr_lambda) * max_sim
                if val > best_val:
                    best_j, best_val = j, val
            if best_j is None:                                 # constraints exhausted: relax and fill by score
                rest = [j for j in order if j not in selected]
                selected.extend(rest[: k - len(selected)])
                break
            selected.append(best_j)
        return selected
