"""Step 4: measurable evaluation (Precision@3, Recall@K, NDCG@3) on a human-labeled pool.

Workflow
--------
1. python -m kitaabcopy.evaluate pool  --queries eval/queries.csv --out eval/annotation_sheet.csv
   Runs every system (full model + baselines + ablations), pools their top-10 per query, shuffles,
   and writes a sheet with an empty `label` column.
2. Fill `label` (0-3). Optionally have a second person fill `label_2` for a subset -> agreement.
     3 = excellent: right subject AND right level AND right style, and a good book
     2 = good: right subject, one of level/style is off, or a slightly weaker book
     1 = weak: right subject but wrong level and style, or marginal relevance
     0 = irrelevant / not a technical book on the subject
3. python -m kitaabcopy.evaluate score --labels eval/annotation_sheet.csv
4. python -m kitaabcopy.evaluate tune  --labels eval/annotation_sheet.csv   (weight search, held-out queries)

Caveat: pooled labels measure *relative* quality of the compared systems. Recall@K is relative to
the labeled-relevant books in the pool, not to every relevant book in Goodreads.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .recommender import KitaabRecommender, Query, Settings

REL_THRESHOLD = 2  # label >= 2 counts as "relevant" for P@k / Recall@k


# ---------------------------------------------------------------------------- metrics
def precision_at_k(labels: list[int], k: int, thr: int = REL_THRESHOLD) -> float:
    top = labels[:k]
    return sum(1 for x in top if x >= thr) / k if k else 0.0


def recall_at_k(labels: list[int], n_relevant: int, k: int, thr: int = REL_THRESHOLD) -> float:
    if n_relevant == 0:
        return float("nan")
    return sum(1 for x in labels[:k] if x >= thr) / n_relevant


def dcg_at_k(labels: list[int], k: int) -> float:
    return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(labels[:k]))


def ndcg_at_k(labels: list[int], all_labels: list[int], k: int) -> float:
    ideal = dcg_at_k(sorted(all_labels, reverse=True), k)
    return dcg_at_k(labels, k) / ideal if ideal > 0 else float("nan")


def list_diversity(recs: list[dict]) -> float:
    """Mean pairwise (1 - similarity-to-previous); 1.0 = fully distinct. Uses engine's stored similarity."""
    sims = [r["max_sim_to_previous"] for r in recs if r.get("max_sim_to_previous") is not None]
    return float(1 - np.mean(sims)) if sims else float("nan")


# ---------------------------------------------------------------------------- systems
def make_systems() -> dict[str, Settings]:
    full = Settings()

    def with_w(**over) -> Settings:
        s = Settings()
        s.weights.update(over)
        return s

    return {
        "full": full,
        "no_diversity": Settings(diversity=False),
        "no_level": with_w(level=0.0),
        "no_genre": with_w(genre=0.0),
        "no_level_genre": with_w(level=0.0, genre=0.0),
        "no_quality": with_w(quality=0.0),
        "no_shelf": with_w(shelf=0.0),
        "baseline_tfidf": Settings(
            weights={"relevance": 1.0, "shelf": 0, "level": 0, "genre": 0, "quality": 0}, diversity=False, gate=0
        ),
        "baseline_rating": Settings(
            weights={"relevance": 0, "shelf": 0, "level": 0, "genre": 0, "quality": 1.0},
            diversity=False, gate=0, n_candidates=50,
        ),
    }


def load_queries(path: str | Path) -> pd.DataFrame:
    q = pd.read_csv(path)
    q = q.reset_index(drop=True)
    if "query_id" not in q.columns:
        q.insert(0, "query_id", [f"q{i:03d}" for i in range(len(q))])
    return q


# ---------------------------------------------------------------------------- pooling
def build_pool(rec: KitaabRecommender, queries: pd.DataFrame, depth: int = 10, seed: int = 13) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    systems = make_systems()
    for q in queries.itertuples(index=False):
        query = Query(q.subject, q.level, q.genre)
        pooled: dict[str, dict] = {}
        for name, st in systems.items():
            for r in rec.recommend(query, k=depth, settings=st, explain=False):
                pooled.setdefault(r["book_id"], r)
        books = list(pooled.values())
        rng.shuffle(books)
        for r in books:
            b = rec.df[rec.df["book_id"] == r["book_id"]].iloc[0]
            shelves = ", ".join(n for n, _ in json.loads(b["shelves_json"])[:8])
            rows.append({
                "query_id": q.query_id, "subject": q.subject, "level": q.level, "genre": q.genre,
                "book_id": r["book_id"], "title": r["title"], "authors": r["authors"],
                "year": r["year"], "pages": r["pages"], "rating": r["rating"], "ratings_count": r["ratings_count"],
                "top_shelves": shelves, "description_snippet": str(b["description"] or "")[:300],
                "url": r["url"], "label": "", "label_2": "",
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- scoring
def load_labels(path: str | Path) -> dict[str, dict[str, int]]:
    df = pd.read_csv(path, dtype={"book_id": str})
    df = df[df["label"].notna() & (df["label"].astype(str).str.strip() != "")]
    out: dict[str, dict[str, int]] = {}
    for r in df.itertuples(index=False):
        out.setdefault(r.query_id, {})[str(r.book_id)] = int(r.label)
    return out


def evaluate_system(
    rec: KitaabRecommender, queries: pd.DataFrame, labels: dict[str, dict[str, int]], settings: Settings,
    ks: tuple[int, ...] = (3, 5, 10),
) -> tuple[pd.DataFrame, dict[str, float]]:
    rows = []
    for q in queries.itertuples(index=False):
        if q.query_id not in labels:
            continue
        lab = labels[q.query_id]
        recs = rec.recommend(Query(q.subject, q.level, q.genre), k=max(ks), settings=settings, explain=False)
        ranked = [lab.get(r["book_id"], 0) for r in recs]  # unlabeled (outside the pool) counts as 0
        all_labels = list(lab.values())
        n_rel = sum(1 for x in all_labels if x >= REL_THRESHOLD)
        row = {"query_id": q.query_id, "P@3": precision_at_k(ranked, 3), "NDCG@3": ndcg_at_k(ranked, all_labels, 3),
               "diversity@3": list_diversity(recs[:3]),
               "mean_rating@3": float(np.mean([r["rating"] for r in recs[:3]])) if recs else float("nan")}
        for k in ks:
            row[f"Recall@{k}"] = recall_at_k(ranked, n_rel, k)
        rows.append(row)
    per_q = pd.DataFrame(rows)
    summary = per_q.drop(columns=["query_id"]).mean(numeric_only=True).to_dict() if len(per_q) else {}
    return per_q, summary


def score_all(rec: KitaabRecommender, queries: pd.DataFrame, labels: dict, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    table = {}
    for name, st in make_systems().items():
        per_q, summary = evaluate_system(rec, queries, labels, st)
        table[name] = summary
        per_q.to_csv(out_dir / f"per_query_{name}.csv", index=False)
    res = pd.DataFrame(table).T.round(3)
    res.to_csv(out_dir / "results.csv")
    return res


def cohen_kappa(a: list[int], b: list[int]) -> float:
    """Quadratic-weighted kappa between two annotators (labels 0-3)."""
    from sklearn.metrics import cohen_kappa_score

    return float(cohen_kappa_score(a, b, weights="quadratic", labels=[0, 1, 2, 3]))


def agreement(path: str | Path) -> float | None:
    df = pd.read_csv(path)
    d = df[df["label"].notna() & df["label_2"].notna() & (df["label"].astype(str) != "") & (df["label_2"].astype(str) != "")]
    if len(d) < 10:
        return None
    return cohen_kappa(d["label"].astype(int).tolist(), d["label_2"].astype(int).tolist())


# ---------------------------------------------------------------------------- tuning
def tune_weights(rec, queries, labels, n_iter: int = 150, seed: int = 7, test_frac: float = 0.3):
    """Random search over the 5 ranking weights; select on train queries, report on held-out queries."""
    rng = np.random.default_rng(seed)
    ids = [q for q in queries["query_id"] if q in labels]
    rng.shuffle(ids)
    n_test = max(1, int(len(ids) * test_frac))
    test_ids, train_ids = set(ids[:n_test]), set(ids[n_test:])
    qtrain = queries[queries["query_id"].isin(train_ids)]
    qtest = queries[queries["query_id"].isin(test_ids)]

    best_w, best = dict(config.DEFAULT_WEIGHTS), -1.0
    names = list(config.DEFAULT_WEIGHTS)
    for _ in range(n_iter):
        w = {n: float(v) for n, v in zip(names, rng.dirichlet(np.ones(len(names))))}
        _, s = evaluate_system(rec, qtrain, labels, Settings(weights=w))
        if s and s["NDCG@3"] > best:
            best, best_w = s["NDCG@3"], w
    _, s_default = evaluate_system(rec, qtest, labels, Settings())
    _, s_tuned = evaluate_system(rec, qtest, labels, Settings(weights=best_w))
    return best_w, {"train_ndcg@3": best, "heldout_default": s_default, "heldout_tuned": s_tuned}


# ---------------------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("pool", "score", "tune"):
        p = sub.add_parser(name)
        p.add_argument("--artifacts", default=str(config.ARTIFACTS))
        p.add_argument("--queries", default=str(config.ROOT / "eval" / "queries.csv"))
        if name == "pool":
            p.add_argument("--out", default=str(config.ROOT / "eval" / "annotation_sheet.csv"))
            p.add_argument("--depth", type=int, default=10)
        else:
            p.add_argument("--labels", default=str(config.ROOT / "eval" / "annotation_sheet.csv"))
            p.add_argument("--out-dir", default=str(config.ROOT / "eval" / "results"))
    a = ap.parse_args()

    rec = KitaabRecommender(a.artifacts, use_dense=True)
    queries = load_queries(a.queries)
    if a.cmd == "pool":
        pool = build_pool(rec, queries, a.depth)
        pool.to_csv(a.out, index=False)
        print(f"wrote {len(pool)} candidate rows for {queries.shape[0]} queries -> {a.out}\nFill the 'label' column (0-3).")
        return
    labels = load_labels(a.labels)
    if a.cmd == "score":
        res = score_all(rec, queries, labels, Path(a.out_dir))
        print(res.to_string())
        kappa = agreement(a.labels)
        if kappa is not None:
            print(f"\nInter-annotator quadratic-weighted kappa: {kappa:.2f}")
    elif a.cmd == "tune":
        w, info = tune_weights(rec, queries, labels)
        print("best weights:", json.dumps({k: round(v, 3) for k, v in w.items()}, indent=2))
        print(json.dumps(info, indent=2, default=float))


if __name__ == "__main__":
    main()
