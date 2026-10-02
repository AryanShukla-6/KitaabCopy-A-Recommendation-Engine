"""Explanations are assembled from the same numbers the ranker used, so they are faithful by
construction (no free-form generation that could drift from the real reasons)."""
from __future__ import annotations

import pandas as pd

from .features import explain_cues


def _level_phrase(p: float) -> str:
    if p >= 0.55:
        return "strong fit"
    if p >= 0.38:
        return "reasonable fit"
    return "closest available fit"


def explain_pick(rec: dict, row: pd.Series, query) -> dict:
    """Return {'summary': str, 'bullets': [str, ...]}."""
    s = rec["scores"]
    bullets: list[str] = []

    # 1. relevance
    if rec["shelf_hits"]:
        tags = ", ".join(f"'{n}' ({c})" for n, c in rec["shelf_hits"][:3])
        bullets.append(
            f"Topic: readers shelve it under {tags}; text similarity to '{query.subject}' is {s['relevance']:.0%} of the best match."
        )
    else:
        bullets.append(
            f"Topic: title/description closely match '{query.subject}' ({s['relevance']:.0%} of the best match)."
        )

    # 2. level
    lvl_ev = explain_cues(row, "level", query.level)
    ev = f" – cues: {', '.join(lvl_ev)}" if lvl_ev else ""
    bullets.append(f"Level: {_level_phrase(s['level'])} for {query.level} (P={s['level']:.2f}){ev}.")

    # 3. genre
    gen_ev = explain_cues(row, "genre", query.genre)
    ev = f" – cues: {', '.join(gen_ev)}" if gen_ev else ""
    bullets.append(f"Style: {_level_phrase(s['genre'])} for {query.genre} (P={s['genre']:.2f}){ev}.")

    # 4. quality
    top_pct = max(1, round((1 - s["quality"]) * 100))
    bullets.append(
        f"Quality: {rec['rating']:.2f}★ from {rec['ratings_count']:,} ratings (top {top_pct}% of technical books by quality score)."
    )

    # 5. diversity
    if rec["rank"] > 1 and rec.get("max_sim_to_previous") is not None:
        bullets.append(
            f"Diversity: different author and only {rec['max_sim_to_previous']:.0%} text overlap with the books ranked above it."
        )

    summary = (
        f"A {_level_phrase(s['level'])} {query.level.lower()}-level, {query.genre.lower()} pick on {query.subject}, "
        f"rated {rec['rating']:.2f}★ by {rec['ratings_count']:,} readers."
    )
    return {"summary": summary, "bullets": bullets}
