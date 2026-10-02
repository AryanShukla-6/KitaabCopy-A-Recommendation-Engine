"""Step 2a: derive difficulty / genre / quality signals.

Goodreads has no difficulty or genre labels, so both are *inferred* from weak cues
(title wording, description wording, reader shelves, page count). Every cue is a (pattern, weight)
pair so the same lists drive the bulk scoring and the per-book explanation text.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .text import parse_shelves

# ---------------------------------------------------------------------------------------
# Cue tables: (regex, weight). "title" cues are matched on the title, "desc" on the description,
# "shelf" cues are exact shelf names (weight scaled by how much of the shelf mass they hold).
# ---------------------------------------------------------------------------------------
LEVEL_CUES: dict[str, dict[str, list[tuple[str, float]]]] = {
    "Beginner": {
        "title": [
            (r"\bintro(duction|ductory)?\b", 1.0), (r"\bbeginner'?s?\b", 1.6), (r"\bfor dummies\b", 1.8),
            (r"\bbasics?\b", 1.0), (r"\bgetting started\b", 1.4), (r"\bhead first\b", 1.4),
            (r"\bcrash course\b", 1.4), (r"\bprimer\b", 1.2), (r"\bfirst (steps|course|book)\b", 1.2),
            (r"\bin (24|21|10|7) (hours|days|minutes)\b", 1.6), (r"\bstep[- ]by[- ]step\b", 1.0),
            (r"\bmade (easy|simple)\b", 1.2), (r"\bessentials\b", 0.8), (r"\blearn(ing)?\b", 0.6),
            (r"\bgrokking\b", 1.2), (r"\bfrom scratch\b", 0.8), (r"\bautomate the boring\b", 1.4),
            (r"\bfor (kids|everyone|absolute)\b", 1.0), (r"\bfundamentals?\b", 0.5),
        ],
        "desc": [
            (r"\bno (prior|previous) (experience|knowledge|programming)\b", 1.6), (r"\bbeginners?\b", 1.0),
            (r"\bnewcomers?\b", 0.8), (r"\bno background\b", 1.2), (r"\bfrom scratch\b", 0.6),
            (r"\bassumes no\b", 1.4), (r"\bgentle introduction\b", 1.4), (r"\bfirst course\b", 0.8),
            (r"\bnovices?\b", 0.8), (r"\bsimple (and|,) (clear|intuitive)\b", 0.6),
        ],
        "shelf": [
            ("beginner", 2.0), ("beginners", 2.0), ("for-beginners", 2.0), ("introduction", 1.2), ("intro", 1.2),
            ("basics", 1.2), ("starter", 1.0), ("tutorial", 0.8), ("tutorials", 0.8), ("easy", 0.6),
            ("entry-level", 1.2), ("learning", 0.4), ("beginner-friendly", 2.0),
        ],
    },
    "Intermediate": {
        "title": [
            (r"\bpractical\b", 0.5), (r"\bapplied\b", 0.7), (r"\beffective\b", 1.0), (r"\bpatterns?\b", 0.8),
            (r"\bcookbook\b", 0.9), (r"\bin action\b", 0.9), (r"\bhands[- ]on\b", 0.6), (r"\bpragmatic\b", 1.0),
            (r"\bin practice\b", 0.8), (r"\bguide\b", 0.4), (r"\bclean\b", 0.6), (r"\brefactoring\b", 1.0),
            (r"\bdesigning\b", 0.8), (r"\bengineering\b", 0.6), (r"\bconcepts\b", 0.6), (r"\bprinciples\b", 0.4),
            (r"\bdata[- ]intensive\b", 1.0), (r"\bmodern\b", 0.4), (r"\bprogramming\b", 0.2),
        ],
        "desc": [
            (r"\bintermediate\b", 1.6), (r"\bworking knowledge\b", 1.0), (r"\bsome (experience|familiarity)\b", 1.2),
            (r"\bpractising\b|\bpracticing\b", 0.6), (r"\bbest practices\b", 0.8), (r"\bwhether you are a\b", 0.3),
            (r"\bprofessional (developers?|programmers?|engineers?)\b", 1.0), (r"\breal[- ]world\b", 0.4),
            (r"\bundergraduate\b", 0.8), (r"\bcomprehensive\b", 0.4),
        ],
        "shelf": [("intermediate", 2.0), ("best-practices", 0.8), ("software-craftsmanship", 0.6), ("classics", 0.3)],
    },
    "Expert": {
        "title": [
            (r"\badvanced\b", 1.8), (r"\bhandbook\b", 1.4), (r"\bcompanion\b", 0.8), (r"\bmastering\b", 1.0),
            (r"\bproofs?\b", 1.2), (r"\btheory\b", 0.9), (r"\bmathematical\b", 1.0), (r"\bprobabilistic\b", 1.4),
            (r"\bthe art of\b", 1.2), (r"\banalysis\b", 0.7), (r"\bfoundations?\b", 0.8), (r"\bstatistical\b", 1.0),
            (r"\bcomplexity\b", 1.0), (r"\bformal\b", 1.0), (r"\bresearch\b", 1.0), (r"\bquantitative\b", 1.2),
            (r"\binternals\b", 1.2), (r"\bperspective\b", 0.8), (r"\bmodern approach\b", 0.5),
            (r"\b(volume|vol\.?) [2-9]\b", 1.0), (r"\bprinciples and practice\b", 0.8), (r"\bcomputational\b", 0.6),
        ],
        "desc": [
            (r"\bgraduate\b", 1.2), (r"\bphd\b", 1.4), (r"\bresearchers?\b", 1.0), (r"\badvanced\b", 1.4),
            (r"\brigorous(ly)?\b", 1.4), (r"\bmathematical(ly)?\b", 0.8), (r"\bstate[- ]of[- ]the[- ]art\b", 1.0),
            (r"\bin[- ]depth\b", 0.8), (r"\bcomprehensive reference\b", 1.0), (r"\bproofs?\b", 1.0),
            (r"\bprerequisites?\b", 0.6), (r"\bmathematical maturity\b", 1.6), (r"\bcutting[- ]edge\b", 0.6),
        ],
        "shelf": [
            ("advanced", 2.0), ("graduate", 1.8), ("phd", 1.8), ("research", 1.2), ("academic", 1.0),
            ("textbook", 0.5), ("expert", 2.0), ("reference", 0.4), ("theory", 0.6), ("papers", 1.0),
        ],
    },
}

GENRE_CUES: dict[str, dict[str, list[tuple[str, float]]]] = {
    "Practical": {
        "title": [
            (r"\bhands[- ]on\b", 1.6), (r"\bpractical\b", 1.4), (r"\bcookbook\b", 1.6), (r"\bin action\b", 1.4),
            (r"\bin practice\b", 1.2), (r"\bprojects?\b", 1.0), (r"\bwith (python|r|java|c\+\+|scikit|tensorflow|pytorch|keras|matlab)\b", 1.2),
            (r"\bguide\b", 0.8), (r"\bcode\b", 0.8), (r"\brecipes\b", 1.4), (r"\bexamples\b", 0.8),
            (r"\bpragmatic\b", 1.4), (r"\beffective\b", 1.2), (r"\bhead first\b", 1.0), (r"\bbuilding\b", 0.8),
            (r"\bdevelopers?\b", 0.8), (r"\bengineers?\b", 0.5), (r"\bprogramming\b", 0.5), (r"\bpatterns\b", 0.8),
            (r"\bclean\b", 0.8), (r"\bexercises\b", 0.6), (r"\bpython\b", 0.4), (r"\bapplied\b", 0.8),
            (r"\bfor (programmers|developers|engineers)\b", 0.8), (r"\bautomate\b", 1.2), (r"\bdesigning\b", 0.6),
            (r"\bcracking\b", 1.0), (r"\binterview\b", 1.0), (r"\bimplementation\b", 0.8),
        ],
        "desc": [
            (r"\bhands[- ]on\b", 1.4), (r"\bpractical\b", 1.0), (r"\bcode (examples?|samples?)\b", 1.4),
            (r"\bexamples?\b", 0.4), (r"\bprojects?\b", 0.6), (r"\bstep[- ]by[- ]step\b", 1.0),
            (r"\breal[- ]world\b", 0.8), (r"\bimplement(ing|ation)?\b", 0.6), (r"\bbuild(ing)?\b", 0.4),
            (r"\bexercises\b", 0.5), (r"\bhow to\b", 0.4), (r"\bpython code\b", 1.0), (r"\bgithub\b", 0.8),
            (r"\bbest practices\b", 0.8), (r"\btutorial\b", 0.8),
        ],
        "shelf": [
            ("practical", 1.8), ("hands-on", 1.8), ("cookbook", 1.8), ("projects", 1.0), ("tutorial", 1.0),
            ("tutorials", 1.0), ("how-to", 1.2), ("best-practices", 1.0), ("practical-programming", 1.8),
            ("coding-interview", 1.2), ("programming-interviews", 1.2), ("clean-code", 1.0),
        ],
    },
    "Theoretical": {
        "title": [
            (r"\btheory\b", 1.8), (r"\btheoretical\b", 1.8), (r"\bfoundations?\b", 1.4), (r"\bprinciples\b", 1.0),
            (r"\bmathematical\b|\bmathematics\b", 1.4), (r"\bproofs?\b", 1.4), (r"\bprobabilistic\b", 1.2),
            (r"\bcomplexity\b", 1.2), (r"\banalysis\b", 1.0), (r"\bformal\b", 1.2), (r"\bcomputation\b", 0.8),
            (r"\bautomata\b|\bcomputability\b", 1.6), (r"\blogic\b", 1.0), (r"\bstatistical\b", 1.0),
            (r"\balgorithms\b", 0.6), (r"\bdiscrete\b", 1.2), (r"\bconcepts\b", 0.8), (r"\bconcrete mathematics\b", 1.0),
            (r"\bart of computer programming\b", 1.2), (r"\bmodern approach\b", 0.6), (r"\bperspective\b", 0.8),
            (r"\bintroduction to the (theory|design)\b", 1.0), (r"\bfundamentals\b", 0.6), (r"\bcomputational\b", 0.8),
            (r"\barchitecture\b", 0.3), (r"\bsystems\b", 0.2), (r"\bgraphical models\b", 1.2),
        ],
        "desc": [
            (r"\btheor(y|etical|ems?)\b", 1.2), (r"\bproofs?\b", 1.2), (r"\bmathematical(ly)?\b", 1.0),
            (r"\brigorous(ly)?\b", 1.2), (r"\bfoundations?\b", 0.8), (r"\bformal\b", 0.8), (r"\bconcepts\b", 0.4),
            (r"\bprinciples\b", 0.6), (r"\bgraduate\b", 0.6), (r"\bresearch\b", 0.4), (r"\bacademic\b", 0.6),
            (r"\bundergraduate (course|textbook)\b", 0.8), (r"\btextbook\b", 0.6), (r"\bfundamental\b", 0.4),
        ],
        "shelf": [
            ("theory", 1.8), ("theoretical", 1.8), ("mathematics", 1.2), ("math", 1.2), ("academic", 1.0),
            ("textbook", 0.8), ("theory-of-computation", 1.8), ("computational-complexity", 1.6),
            ("discrete-mathematics", 1.4), ("formal-methods", 1.4), ("computer-science-theory", 1.8),
            ("algorithms", 0.5), ("statistics", 0.6), ("logic", 1.0), ("cs-theory", 1.8),
        ],
    },
}

LEVEL_NAMES = list(LEVEL_CUES)
GENRE_NAMES = list(GENRE_CUES)
_LEVEL_PRIOR = {"Beginner": 0.0, "Intermediate": 0.35, "Expert": 0.0}  # most technical books are intermediate
_TEMPERATURE = 1.2


def _compiled(cues: dict[str, dict[str, list[tuple[str, float]]]]):
    return {
        cls: {
            "title": [(re.compile(p, re.I), w) for p, w in c["title"]],
            "desc": [(re.compile(p, re.I), w) for p, w in c["desc"]],
            "shelf": dict(c["shelf"]),
        }
        for cls, c in cues.items()
    }


_LEVEL_RE = _compiled(LEVEL_CUES)
_GENRE_RE = _compiled(GENRE_CUES)


def _shelf_score(shelves: list[tuple[str, int]], shelf_cues: dict[str, float]) -> tuple[float, list[str]]:
    """Weight a cue by the (log) share of reader votes it holds among informative shelves."""
    if not shelves:
        return 0.0, []
    total = sum(c for _, c in shelves[:30]) or 1
    score, hits = 0.0, []
    for name, cnt in shelves[:30]:
        w = shelf_cues.get(name)
        if w:
            share = min(1.0, 8.0 * cnt / total)  # 12.5% of votes = full weight
            score += w * share
            hits.append(name)
    return score, hits


def _score_book(
    title: str, desc: str, shelves: list[tuple[str, int]], compiled_cls: dict, with_hits: bool = False
):
    score, hits = 0.0, []
    for rx, w in compiled_cls["title"]:
        m = rx.search(title)
        if m:
            score += w
            if with_hits:
                hits.append(f"title '{m.group(0)}'")
    d = desc[:2000]
    desc_score = 0.0
    for rx, w in compiled_cls["desc"]:
        m = rx.search(d)
        if m:
            desc_score += w
            if with_hits:
                hits.append(f"description '{m.group(0)}'")
    score += min(desc_score, 2.5)  # cap so long blurbs can't dominate
    s_score, s_hits = _shelf_score(shelves, compiled_cls["shelf"])
    score += min(s_score, 3.0)
    if with_hits:
        hits.extend(f"shelf '{h}'" for h in s_hits)
    return score, hits


def _pages_level_bonus(pages: float | None) -> dict[str, float]:
    if pages is None or pd.isna(pages) or pages <= 0:
        return {"Beginner": 0.0, "Intermediate": 0.0, "Expert": 0.0}
    if pages < 180:
        return {"Beginner": 0.6, "Intermediate": 0.0, "Expert": -0.4}
    if pages < 320:
        return {"Beginner": 0.25, "Intermediate": 0.15, "Expert": -0.2}
    if pages < 600:
        return {"Beginner": -0.2, "Intermediate": 0.2, "Expert": 0.1}
    if pages < 900:
        return {"Beginner": -0.5, "Intermediate": 0.1, "Expert": 0.5}
    return {"Beginner": -0.7, "Intermediate": 0.0, "Expert": 0.8}


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def add_features(df: pd.DataFrame, prior_m: float = 200.0) -> pd.DataFrame:
    """Add p_beginner/p_intermediate/p_expert, p_practical/p_theoretical, bayes_rating, quality."""
    df = df.copy()
    n = len(df)
    titles = df["title"].fillna("").astype(str).tolist()
    descs = df["description"].fillna("").astype(str).tolist()
    shelves = [parse_shelves(s) for s in df["shelves_json"].fillna("[]")]
    pages = df["num_pages"].astype("float64").tolist()

    lvl = np.zeros((n, 3))
    gen = np.zeros((n, 2))
    for i in range(n):
        bonus = _pages_level_bonus(pages[i])
        for j, cls in enumerate(LEVEL_NAMES):
            s, _ = _score_book(titles[i], descs[i], shelves[i], _LEVEL_RE[cls])
            lvl[i, j] = s + bonus[cls] + _LEVEL_PRIOR[cls]
        for j, cls in enumerate(GENRE_NAMES):
            s, _ = _score_book(titles[i], descs[i], shelves[i], _GENRE_RE[cls])
            gen[i, j] = s
    p_lvl = _softmax(lvl / _TEMPERATURE)
    p_gen = _softmax(gen / _TEMPERATURE)
    for j, cls in enumerate(LEVEL_NAMES):
        df[f"p_{cls.lower()}"] = p_lvl[:, j]
    for j, cls in enumerate(GENRE_NAMES):
        df[f"p_{cls.lower()}"] = p_gen[:, j]

    # Quality: Bayesian average (shrinks few-vote ratings to the global mean) blended with popularity.
    r = df["average_rating"].astype("float64").fillna(0)
    v = df["ratings_count"].astype("float64").fillna(0)
    c = float(r[v > 0].mean()) if (v > 0).any() else 3.9
    df["bayes_rating"] = (v / (v + prior_m)) * r + (prior_m / (v + prior_m)) * c
    pct_bayes = df["bayes_rating"].rank(pct=True)
    pct_pop = np.log1p(v).rank(pct=True)
    df["quality"] = 0.75 * pct_bayes + 0.25 * pct_pop
    return df


def explain_cues(row: pd.Series, dimension: str, label: str, max_hits: int = 4) -> list[str]:
    """Human-readable evidence for a book's level ('level') or genre ('genre') classification."""
    cues = _LEVEL_RE if dimension == "level" else _GENRE_RE
    score, hits = _score_book(
        str(row.get("title") or ""), str(row.get("description") or ""),
        parse_shelves(row.get("shelves_json")), cues[label], with_hits=True,
    )
    out = hits[:max_hits]
    if dimension == "level":
        pages = row.get("num_pages")
        if pages is not None and not pd.isna(pages) and pages > 0:
            out.append(f"{int(pages)} pages")
    return out
