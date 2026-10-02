"""End-to-end smoke test on synthetic Goodreads-format data (no real dataset needed)."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("kc")
    run = lambda *a: subprocess.run([sys.executable, *a], cwd=ROOT, check=True, capture_output=True)
    run("tests/make_synthetic.py", "--out", str(tmp / "raw"), "--n-tech", "600", "--n-fiction", "300")
    run("-m", "kitaabcopy.prepare_data", "--books", str(tmp / "raw/goodreads_books.json.gz"),
        "--authors", str(tmp / "raw/goodreads_book_authors.json.gz"), "--out", str(tmp / "books.parquet"))
    run("-m", "kitaabcopy.build_index", "--books", str(tmp / "books.parquet"), "--out", str(tmp / "art"))
    from kitaabcopy.recommender import KitaabRecommender
    return KitaabRecommender(tmp / "art", use_dense=False)


def test_fiction_filtered_out(engine):
    assert not engine.df["title"].str.contains("Dragon|Winter|Ember").any()


def test_returns_three_diverse_books(engine):
    from kitaabcopy.recommender import Query
    picks = engine.recommend(Query("Machine Learning", "Beginner", "Practical"))
    assert len(picks) == 3
    assert len({p["book_id"] for p in picks}) == 3
    authors = [set(p["author_ids"].split("|")) for p in picks]
    assert not (authors[0] & authors[1]) and not (authors[0] & authors[2]) and not (authors[1] & authors[2])
    for p in picks:
        assert p["url"].startswith("http") and p["explanation"]["bullets"]


def test_level_changes_ranking(engine):
    from kitaabcopy.recommender import Query
    beg = engine.recommend(Query("Algorithms", "Beginner", "Practical"))
    exp = engine.recommend(Query("Algorithms", "Expert", "Theoretical"))
    assert [p["book_id"] for p in beg] != [p["book_id"] for p in exp]
    assert sum(p["scores"]["level"] for p in beg) / 3 > 0.5


def test_cpp_and_c_are_distinguished(engine):
    from kitaabcopy.recommender import Query
    cpp = engine.recommend(Query("C++", "Intermediate", "Practical"))
    assert all("c++" in p["title"].lower() or "cpp" in p["title"].lower() for p in cpp)
