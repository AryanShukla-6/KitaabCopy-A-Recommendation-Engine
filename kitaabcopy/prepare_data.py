"""Step 1: stream the 2 GB goodreads_books.json.gz with DuckDB and keep technical books.

Usage:
    python -m kitaabcopy.prepare_data \
        --books data/raw/goodreads_books.json.gz \
        --authors data/raw/goodreads_book_authors.json.gz \
        --out data/processed/books.parquet

A book is kept when a meaningful share of its (non-generic) reader shelves are technical
(see config.TOPIC_SHELVES), or when its title contains a technical keyword and it has at least a
small technical shelf share. Fiction-dominated books are dropped. Editions are collapsed by work_id.
The thresholds are deliberately recall-oriented: wrong inclusions are filtered later by subject
relevance, but wrong exclusions can never be recovered.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd

from . import config

BOOK_COLUMNS = {
    "book_id": "VARCHAR", "work_id": "VARCHAR", "title": "VARCHAR", "title_without_series": "VARCHAR",
    "description": "VARCHAR", "average_rating": "VARCHAR", "ratings_count": "VARCHAR",
    "text_reviews_count": "VARCHAR", "num_pages": "VARCHAR", "publication_year": "VARCHAR",
    "publisher": "VARCHAR", "format": "VARCHAR", "is_ebook": "VARCHAR", "language_code": "VARCHAR",
    "url": "VARCHAR", "link": "VARCHAR", "image_url": "VARCHAR", "isbn13": "VARCHAR",
    "popular_shelves": "STRUCT(count VARCHAR, name VARCHAR)[]",
    "authors": "STRUCT(author_id VARCHAR, role VARCHAR)[]",
    "similar_books": "VARCHAR[]",
}
ENGLISH = "('', 'eng', 'en', 'en-US', 'en-GB', 'en-CA', 'en-AU', 'enm')"


def _vocab_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    seen: dict[str, str] = {}
    for topic, names in config.TOPIC_SHELVES.items():
        for n in names:
            seen.setdefault(n, topic)
    tech = pd.DataFrame({"name": list(seen), "topic": list(seen.values())})
    nontech = pd.DataFrame({"name": sorted(set(config.NONTECH_SHELVES))})
    return tech, nontech


def prepare(
    books_path: str | Path,
    authors_path: str | Path | None,
    out_path: str | Path,
    min_ratings: int = 25,
    min_share: float = 0.12,
    min_tech_cnt: int = 6,
    title_rescue_share: float = 0.04,
    max_fiction_share: float = 0.5,
    top_shelves: int = 40,
    threads: int | None = None,
) -> pd.DataFrame:
    con = duckdb.connect()
    if threads:
        con.execute(f"PRAGMA threads={int(threads)}")
    tech_vocab, nontech_vocab = _vocab_frames()
    con.register("tech_vocab", tech_vocab)
    con.register("nontech_vocab", nontech_vocab)

    cols = ", ".join(f"'{k}': '{v}'" for k, v in BOOK_COLUMNS.items())
    generic = config.GENERIC_REGEX.replace("'", "''")
    rescue = config.TITLE_RESCUE_REGEX.replace("'", "''")

    print("[1/5] reading + pre-filtering raw JSON ...")
    con.execute(f"""
        CREATE TEMP TABLE base AS
        SELECT book_id, work_id, title, title_without_series, description,
               TRY_CAST(average_rating AS DOUBLE)        AS average_rating,
               TRY_CAST(ratings_count AS BIGINT)         AS ratings_count,
               TRY_CAST(text_reviews_count AS BIGINT)    AS text_reviews_count,
               TRY_CAST(num_pages AS INTEGER)            AS num_pages,
               TRY_CAST(publication_year AS INTEGER)     AS publication_year,
               publisher, format, is_ebook, language_code, url, link, image_url, isbn13,
               popular_shelves, authors, similar_books
        FROM read_json('{Path(books_path).as_posix()}', format='newline_delimited',
                       columns={{{cols}}}, maximum_object_size=33554432, ignore_errors=true)
        WHERE TRY_CAST(ratings_count AS BIGINT) >= {int(min_ratings)}
          AND COALESCE(language_code, '') IN {ENGLISH}
          AND title IS NOT NULL AND length(title) > 1
    """)
    n_base = con.execute("SELECT count(*) FROM base").fetchone()[0]
    print(f"      {n_base:,} English books with >= {min_ratings} ratings")

    print("[2/5] aggregating shelves ...")
    con.execute(f"""
        CREATE TEMP TABLE shelf_rows AS
        SELECT book_id, lower(s.name) AS name, COALESCE(TRY_CAST(s.count AS INTEGER), 0) AS cnt
        FROM (SELECT book_id, unnest(list_slice(popular_shelves, 1, {int(top_shelves)})) AS s FROM base)
    """)
    con.execute(f"""
        CREATE TEMP TABLE agg AS
        SELECT r.book_id,
               SUM(CASE WHEN regexp_matches(r.name, '{generic}') THEN 0 ELSE r.cnt END)            AS total_cnt,
               SUM(CASE WHEN v.name IS NOT NULL THEN r.cnt ELSE 0 END)                             AS tech_cnt,
               SUM(CASE WHEN f.name IS NOT NULL THEN r.cnt ELSE 0 END)                             AS fic_cnt,
               list_distinct(list(v.topic) FILTER (WHERE v.topic IS NOT NULL))                     AS topics
        FROM shelf_rows r
        LEFT JOIN tech_vocab v    ON r.name = v.name
        LEFT JOIN nontech_vocab f ON r.name = f.name
        GROUP BY r.book_id
    """)

    print("[3/5] selecting technical books + collapsing editions ...")
    con.execute(f"""
        CREATE TEMP TABLE kept AS
        SELECT * EXCLUDE (rn) FROM (
            SELECT *, row_number() OVER (
                       PARTITION BY COALESCE(NULLIF(work_id, ''), book_id)
                       ORDER BY ratings_count DESC) AS rn
            FROM (
                SELECT b.*, a.total_cnt, a.tech_cnt, a.fic_cnt, a.topics,
                       a.tech_cnt * 1.0 / NULLIF(a.total_cnt, 0) AS tech_share,
                       a.fic_cnt  * 1.0 / NULLIF(a.total_cnt, 0) AS fic_share,
                       regexp_matches(lower(b.title), '{rescue}') AS title_hit
                FROM base b JOIN agg a USING (book_id)
            )
            WHERE COALESCE(fic_share, 0) < {max_fiction_share}
              AND ( (tech_share >= {min_share} AND tech_cnt >= {int(min_tech_cnt)})
                 OR (title_hit AND tech_share >= {title_rescue_share} AND tech_cnt >= 2) )
        ) WHERE rn = 1
    """)
    n_kept = con.execute("SELECT count(*) FROM kept").fetchone()[0]
    print(f"      kept {n_kept:,} technical works")

    print("[4/5] collecting shelves + authors ...")
    shelves = con.execute("""
        SELECT s.book_id, list(s.name ORDER BY s.cnt DESC) AS names, list(s.cnt ORDER BY s.cnt DESC) AS cnts
        FROM shelf_rows s JOIN kept k USING (book_id) GROUP BY s.book_id
    """).fetchdf()
    shelf_map = {
        r.book_id: json.dumps(list(zip(list(r.names), [int(c) for c in r.cnts])))
        for r in shelves.itertuples(index=False)
    }
    df = con.execute("SELECT * FROM kept").fetchdf()
    df["shelves_json"] = df["book_id"].map(shelf_map).fillna("[]")
    df["topics"] = df["topics"].apply(lambda t: "|".join(sorted(t)) if t is not None else "")

    def _author_ids(a):
        a = list(a) if a is not None else []
        primary = [x["author_id"] for x in a if (x.get("role") or "").strip().lower() in ("", "author")]
        return primary or [x["author_id"] for x in a]

    df["author_ids"] = df["authors"].apply(_author_ids)
    id_to_name: dict[str, str] = {}
    if authors_path and Path(authors_path).exists():
        names = con.execute(f"""
            SELECT author_id, name FROM read_json('{Path(authors_path).as_posix()}',
                   format='newline_delimited', columns={{'author_id':'VARCHAR','name':'VARCHAR'}},
                   ignore_errors=true)
        """).fetchdf()
        id_to_name = dict(zip(names.author_id, names.name))
    df["author_names"] = df["author_ids"].apply(lambda ids: "|".join(id_to_name.get(i, "") for i in ids[:3] if id_to_name.get(i)))
    df["author_ids"] = df["author_ids"].apply(lambda ids: "|".join(ids))
    df["similar_books"] = df["similar_books"].apply(lambda s: "|".join(s) if s is not None else "")
    df = df.drop(columns=["authors", "popular_shelves", "title_hit"], errors="ignore")

    print("[5/5] writing parquet ...")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"      wrote {len(df):,} rows -> {out_path}")
    topic_counts = df["topics"].str.split("|").explode().value_counts()
    print("      books per topic (a book can have several):")
    print(topic_counts.to_string())
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--books", default=str(config.DATA_RAW / "goodreads_books.json.gz"))
    ap.add_argument("--authors", default=str(config.DATA_RAW / "goodreads_book_authors.json.gz"))
    ap.add_argument("--out", default=str(config.BOOKS_PARQUET))
    ap.add_argument("--min-ratings", type=int, default=25)
    ap.add_argument("--min-share", type=float, default=0.12)
    ap.add_argument("--min-tech-cnt", type=int, default=6)
    ap.add_argument("--threads", type=int, default=None)
    a = ap.parse_args()
    prepare(a.books, a.authors, a.out, a.min_ratings, a.min_share, a.min_tech_cnt, threads=a.threads)


if __name__ == "__main__":
    main()
