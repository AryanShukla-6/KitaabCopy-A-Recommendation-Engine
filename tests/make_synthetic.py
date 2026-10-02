"""Generate a small Goodreads-format dataset (JSON Lines, gzipped) for offline smoke tests.

NOT real data: titles/descriptions come from templates. It exists so the whole pipeline
(prepare -> index -> recommend -> evaluate -> UI) can be tested without the 2 GB file.

    python tests/make_synthetic.py --out data/raw_synth --n-tech 1500 --n-fiction 800
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
from pathlib import Path

TOPICS = {
    "machine learning": ["machine-learning", "data-science", "artificial-intelligence", "statistics"],
    "deep learning": ["deep-learning", "neural-networks", "machine-learning", "tensorflow"],
    "artificial intelligence": ["artificial-intelligence", "ai", "machine-learning", "computer-science"],
    "reinforcement learning": ["reinforcement-learning", "machine-learning", "artificial-intelligence"],
    "natural language processing": ["nlp", "natural-language-processing", "computational-linguistics"],
    "computer vision": ["computer-vision", "image-processing", "opencv", "machine-learning"],
    "evolutionary computing": ["evolutionary-computation", "genetic-algorithms", "optimization", "artificial-intelligence"],
    "data science": ["data-science", "data-analysis", "python", "statistics"],
    "algorithms": ["algorithms", "data-structures", "computer-science", "programming"],
    "c++": ["c-plus-plus", "programming", "software-development", "computer-science"],
    "c": ["c-programming", "c", "programming", "computer-science"],
    "python": ["python", "programming", "software-development", "data-science"],
    "java": ["java", "programming", "software-engineering"],
    "operating systems": ["operating-systems", "linux", "computer-science", "systems-programming"],
    "computer networks": ["computer-networking", "networking", "tcp-ip", "computer-science"],
    "databases": ["databases", "sql", "database-design", "computer-science"],
    "compilers": ["compilers", "programming-language-theory", "computer-science", "parsing"],
    "cryptography": ["cryptography", "security", "computer-science", "mathematics"],
    "distributed systems": ["distributed-systems", "cloud-computing", "software-architecture"],
    "software engineering": ["software-engineering", "agile", "software-architecture", "programming"],
    "theory of computation": ["theory-of-computation", "computational-complexity", "automata", "computer-science"],
    "computer graphics": ["computer-graphics", "opengl", "game-development", "computer-science"],
}
TEMPLATES = [
    ("Introduction to {t}", "Beginner", "Practical"),
    ("{T} for Beginners", "Beginner", "Practical"),
    ("{T} in Action", "Intermediate", "Practical"),
    ("Hands-On {T}", "Intermediate", "Practical"),
    ("Practical {T} Cookbook", "Intermediate", "Practical"),
    ("Effective {T}", "Intermediate", "Practical"),
    ("{T}: Principles and Concepts", "Intermediate", "Theoretical"),
    ("Foundations of {T}", "Expert", "Theoretical"),
    ("Advanced {T}", "Expert", "Theoretical"),
    ("The Theory of {T}", "Expert", "Theoretical"),
    ("{T}: A Probabilistic Perspective", "Expert", "Theoretical"),
    ("The {T} Handbook", "Expert", "Practical"),
]
DESC = {
    "Beginner": "A gentle introduction for beginners with no prior experience. Step-by-step examples.",
    "Intermediate": "For practicing developers with some experience. Covers best practices and real-world projects.",
    "Expert": "A rigorous graduate-level treatment with proofs, for researchers. Requires mathematical maturity.",
}
FICTION_SHELVES = ["fiction", "fantasy", "romance", "novels", "young-adult", "mystery", "thriller"]
FICTION_TITLES = ["The Last {w}", "A Song of {w}", "Shadows of {w}", "The {w} Chronicles", "{w} and Roses"]
WORDS = ["Winter", "Dragon", "Ember", "Moon", "Crown", "River", "Storm", "Glass", "Iron", "Silver"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw_synth")
    ap.add_argument("--n-tech", type=int, default=1500)
    ap.add_argument("--n-fiction", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    authors = {str(i): f"Author {i}" for i in range(1, 600)}
    generic = ["to-read", "currently-reading", "favorites", "non-fiction", "owned", "kindle"]
    books = []
    for i in range(a.n_tech):
        topic = rng.choice(list(TOPICS))
        tmpl, level, genre = rng.choice(TEMPLATES)
        title = tmpl.format(t=topic, T=topic.title())
        if rng.random() < 0.3:
            title += f", {rng.randint(2, 5)}th Edition"
        pages = {"Beginner": rng.randint(120, 400), "Intermediate": rng.randint(250, 700),
                 "Expert": rng.randint(500, 1200)}[level]
        n_rat = int(rng.lognormvariate(5, 1.5)) + 30
        shelves = [{"count": str(rng.randint(30, 400) * (4 - j)), "name": s} for j, s in enumerate(TOPICS[topic])]
        shelves += [{"count": str(rng.randint(200, 3000)), "name": g} for g in rng.sample(generic, 3)]
        if level == "Beginner" and rng.random() < 0.6:
            shelves.append({"count": str(rng.randint(10, 80)), "name": "beginner"})
        if genre == "Practical" and rng.random() < 0.5:
            shelves.append({"count": str(rng.randint(10, 60)), "name": "practical"})
        shelves.sort(key=lambda s: -int(s["count"]))
        books.append({
            "book_id": str(100000 + i), "work_id": str(500000 + i), "title": title, "title_without_series": title,
            "description": f"{DESC[level]} Topic: {topic}. {'Includes code examples.' if genre == 'Practical' else 'Develops the theory.'}",
            "average_rating": f"{min(4.9, max(3.0, rng.gauss(4.05, 0.3))):.2f}", "ratings_count": str(n_rat),
            "text_reviews_count": str(n_rat // 10), "num_pages": str(pages), "publication_year": str(rng.randint(1995, 2023)),
            "publisher": "Synth Press", "format": "Paperback", "is_ebook": "false", "language_code": "eng",
            "url": f"https://www.goodreads.com/book/show/{100000 + i}", "link": f"https://www.goodreads.com/book/show/{100000 + i}",
            "image_url": "", "isbn13": "", "popular_shelves": shelves,
            "authors": [{"author_id": str(rng.randint(1, 599)), "role": ""}], "similar_books": [],
        })
    for i in range(a.n_fiction):
        title = rng.choice(FICTION_TITLES).format(w=rng.choice(WORDS))
        shelves = [{"count": str(rng.randint(500, 5000)), "name": rng.choice(FICTION_SHELVES)} for _ in range(3)]
        shelves += [{"count": str(rng.randint(200, 3000)), "name": g} for g in rng.sample(generic, 3)]
        n = a.n_tech + i
        books.append({
            "book_id": str(100000 + n), "work_id": str(500000 + n), "title": title, "title_without_series": title,
            "description": "A sweeping tale of love and loss in a dying kingdom.", "average_rating": f"{rng.uniform(3.2, 4.6):.2f}",
            "ratings_count": str(rng.randint(50, 50000)), "text_reviews_count": "10", "num_pages": str(rng.randint(200, 600)),
            "publication_year": "2010", "publisher": "Synth Press", "format": "Paperback", "is_ebook": "false",
            "language_code": "eng", "url": f"https://www.goodreads.com/book/show/{100000 + n}", "link": "", "image_url": "",
            "isbn13": "", "popular_shelves": shelves, "authors": [{"author_id": str(rng.randint(1, 599)), "role": ""}],
            "similar_books": [],
        })
    rng.shuffle(books)
    with gzip.open(out / "goodreads_books.json.gz", "wt") as f:
        for b in books:
            f.write(json.dumps(b) + "\n")
    with gzip.open(out / "goodreads_book_authors.json.gz", "wt") as f:
        for aid, name in authors.items():
            f.write(json.dumps({"author_id": aid, "name": name, "average_rating": "4.0", "text_reviews_count": "1", "ratings_count": "1"}) + "\n")
    print(f"wrote {len(books)} books -> {out}")


if __name__ == "__main__":
    main()
