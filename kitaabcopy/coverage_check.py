"""Sanity check for the requirement "the index must contain most core CS / AI technical books".

Looks up ~60 canonical technical titles in the processed parquet and reports which are missing,
plus the number of books per topic. If recall of this seed list is low, loosen the thresholds in
prepare_data (--min-share, --min-tech-cnt, --min-ratings) and re-run.

    python -m kitaabcopy.coverage_check
"""
from __future__ import annotations

import argparse
import re

import pandas as pd

from . import config

SEED_TITLES = [
    ("Introduction to Algorithms", r"^introduction to algorithms"),
    ("Structure and Interpretation of Computer Programs", r"structure and interpretation of computer programs"),
    ("The C Programming Language", r"^the c programming language"),
    ("Artificial Intelligence: A Modern Approach", r"artificial intelligence:? a modern approach"),
    ("Deep Learning (Goodfellow)", r"^deep learning( \(adaptive|$)"),
    ("Pattern Recognition and Machine Learning", r"pattern recognition and machine learning"),
    ("The Elements of Statistical Learning", r"elements of statistical learning"),
    ("Hands-On Machine Learning with Scikit-Learn", r"hands[- ]on machine learning"),
    ("Machine Learning: A Probabilistic Perspective", r"machine learning:? a probabilistic perspective"),
    ("Reinforcement Learning: An Introduction", r"reinforcement learning:? an introduction"),
    ("Computer Networking: A Top-Down Approach", r"computer networking:? a top[- ]down"),
    ("Operating System Concepts", r"^operating system concepts"),
    ("Modern Operating Systems", r"^modern operating systems"),
    ("Computer Organization and Design", r"computer organization and design"),
    ("Computer Architecture: A Quantitative Approach", r"computer architecture:? a quantitative approach"),
    ("Compilers: Principles, Techniques, and Tools", r"^compilers:? principles"),
    ("Introduction to the Theory of Computation", r"introduction to the theory of computation"),
    ("Database System Concepts", r"^database system concepts"),
    ("Design Patterns: Elements of Reusable OO Software", r"^design patterns:? elements"),
    ("Clean Code", r"^clean code"),
    ("The Pragmatic Programmer", r"^the pragmatic programmer"),
    ("Code Complete", r"^code complete"),
    ("The Art of Computer Programming", r"^the art of computer programming"),
    ("Effective C++", r"^effective c\+\+"),
    ("The C++ Programming Language", r"^the c\+\+ programming language"),
    ("A Tour of C++", r"^a tour of c\+\+"),
    ("Effective Java", r"^effective java"),
    ("Fluent Python", r"^fluent python"),
    ("Designing Data-Intensive Applications", r"designing data[- ]intensive applications"),
    ("Applied Cryptography", r"^applied cryptography"),
    ("Speech and Language Processing", r"^speech and language processing"),
    ("Computer Vision: Algorithms and Applications", r"computer vision:? algorithms and applications"),
    ("An Introduction to Genetic Algorithms", r"introduction to genetic algorithms"),
    ("Introduction to Evolutionary Computing", r"introduction to evolutionary computing"),
    ("Genetic Algorithms in Search, Optimization", r"genetic algorithms in search"),
    ("Concrete Mathematics", r"^concrete mathematics"),
    ("Discrete Mathematics and Its Applications", r"discrete mathematics and its applications"),
    ("Grokking Algorithms", r"^grokking algorithms"),
    ("Cracking the Coding Interview", r"^cracking the coding interview"),
    ("Programming Pearls", r"^programming pearls"),
    ("Refactoring", r"^refactoring"),
    ("Head First Design Patterns", r"head first design patterns"),
    ("Python Crash Course", r"^python crash course"),
    ("Automate the Boring Stuff with Python", r"automate the boring stuff"),
    ("Introduction to Information Retrieval", r"introduction to information retrieval"),
    ("Probabilistic Graphical Models", r"probabilistic graphical models"),
    ("Neural Networks and Deep Learning", r"neural networks and deep learning"),
    ("The Mythical Man-Month", r"the mythical man[- ]month"),
    ("Computer Graphics: Principles and Practice", r"computer graphics:? principles and practice"),
    ("Real-Time Rendering", r"^real[- ]time rendering"),
    ("Computer Systems: A Programmer's Perspective", r"computer systems:? a programmer"),
    ("Understanding Machine Learning", r"understanding machine learning"),
    ("Pattern Classification", r"^pattern classification"),
    ("Data Structures and Algorithms in Java/C++", r"^data structures and algorithms in (java|c\+\+)"),
    ("Algorithm Design (Kleinberg)", r"^algorithm design$"),
    ("Mining of Massive Datasets", r"mining of massive datasets"),
    ("Site Reliability Engineering", r"^site reliability engineering"),
    ("The Linux Programming Interface", r"the linux programming interface"),
    ("Network Security: Private Communication", r"cryptography and network security|network security"),
    ("Quantum Computation and Quantum Information", r"quantum computation and quantum information"),
    ("Probabilistic Robotics", r"^probabilistic robotics"),
    ("Metaheuristics / Swarm Intelligence", r"swarm intelligence|metaheuristics"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", default=str(config.BOOKS_PARQUET))
    a = ap.parse_args()
    df = pd.read_parquet(a.books, columns=["title", "topics"])
    titles = df["title"].fillna("").str.lower()
    found, missing = [], []
    for name, pat in SEED_TITLES:
        (found if titles.str.contains(re.compile(pat, re.I)).any() else missing).append(name)
    print(f"seed-list coverage: {len(found)}/{len(SEED_TITLES)} = {len(found) / len(SEED_TITLES):.0%}")
    if missing:
        print("missing:\n  " + "\n  ".join(missing))
    print("\nbooks per topic:")
    print(df["topics"].str.split("|").explode().value_counts().to_string())


if __name__ == "__main__":
    main()
