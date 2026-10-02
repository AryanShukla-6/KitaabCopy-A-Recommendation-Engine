"""Central configuration: paths, technical-shelf taxonomy, subject aliases, default weights."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"

BOOKS_PARQUET = DATA_PROCESSED / "books.parquet"

# --------------------------------------------------------------------------------------
# Technical taxonomy. Goodreads has no "genre" field, so "is this a technical book?" is
# decided from user shelves (popular_shelves). Keys are topics, values are shelf names.
# Each shelf name belongs to exactly one topic (first occurrence wins).
# --------------------------------------------------------------------------------------
TOPIC_SHELVES: dict[str, list[str]] = {
    "artificial-intelligence": [
        "artificial-intelligence", "ai", "ai-ml", "knowledge-representation", "expert-systems",
        "intelligent-agents", "agi", "ai-and-ml", "artificial-intelligence-ai", "cognitive-computing",
    ],
    "machine-learning": [
        "machine-learning", "machine-learning-ai", "ml", "statistical-learning", "pattern-recognition",
        "data-mining", "bayesian", "bayesian-statistics", "reinforcement-learning", "probabilistic-models",
        "scikit-learn", "predictive-analytics", "supervised-learning", "kaggle", "machine-learning-data-science",
        "graphical-models", "kernel-methods",
    ],
    "deep-learning": [
        "deep-learning", "neural-networks", "neural-network", "tensorflow", "pytorch", "keras",
        "convolutional-neural-networks", "generative-ai", "llm", "transformers", "deep-learning-ai",
    ],
    "nlp": [
        "nlp", "natural-language-processing", "computational-linguistics", "text-mining",
        "information-retrieval", "speech-recognition", "text-analytics",
    ],
    "computer-vision": ["computer-vision", "image-processing", "opencv", "machine-vision", "image-analysis"],
    "robotics": ["robotics", "robots", "autonomous-vehicles", "control-systems", "control-theory", "mechatronics"],
    "evolutionary-computation": [
        "evolutionary-computation", "evolutionary-algorithms", "genetic-algorithms", "genetic-programming",
        "swarm-intelligence", "metaheuristics", "optimization", "fuzzy-logic", "soft-computing",
        "computational-intelligence", "artificial-life", "evolutionary-computing", "heuristics",
        "complexity-science",
    ],
    "data-science": [
        "data-science", "data-analysis", "data-analytics", "big-data", "data-visualization", "statistics",
        "pandas", "numpy", "data-engineering", "business-intelligence", "analytics", "data", "r-programming",
        "data-mining-analytics", "bioinformatics", "computational-biology",
    ],
    "algorithms": [
        "algorithms", "algorithm", "data-structures", "algorithms-data-structures", "competitive-programming",
        "algorithm-design", "programming-interviews", "coding-interview", "interview-prep", "leetcode",
        "algorithms-and-data-structures",
    ],
    "theory": [
        "theory-of-computation", "computational-complexity", "automata", "computability", "formal-languages",
        "discrete-mathematics", "discrete-math", "logic", "mathematical-logic", "graph-theory", "combinatorics",
        "information-theory", "category-theory", "type-theory", "formal-methods", "lambda-calculus",
        "computation", "theoretical-computer-science", "cs-theory", "complexity-theory", "numerical-analysis",
        "numerical-methods", "scientific-computing", "computational-science", "cellular-automata",
    ],
    "programming": [
        "programming", "computer-programming", "coding", "software-development", "programming-languages",
        "programming-language", "code", "scripting", "programming-books", "development", "developer",
        "software-development-programming", "learn-to-code", "computer-programming-languages", "programmer",
    ],
    "c": ["c", "c-programming", "c-language", "ansi-c", "k-r", "c-programming-language"],
    "cpp": ["c-plus-plus", "cpp", "c++", "c-2", "modern-cpp", "stl", "c-and-c", "c-c", "cplusplus"],
    "java": ["java", "java-programming", "jvm", "spring", "scala", "kotlin", "j2ee", "java-ee", "android-java"],
    "python": ["python", "python-programming", "django", "flask", "python3", "python-3", "jupyter"],
    "web": [
        "javascript", "js", "nodejs", "node-js", "typescript", "react", "angular", "jquery", "ecmascript",
        "web-development", "html", "css", "frontend", "front-end", "web", "php", "ruby", "rails",
        "ruby-on-rails", "perl", "web-programming", "html5", "web-design", "vue", "web-dev", "internet",
    ],
    "dotnet": ["c-sharp", "csharp", "dotnet", "net", "asp-net", "microsoft", "visual-basic", "vb"],
    "other-languages": [
        "rust", "go", "golang", "haskell", "lisp", "scheme", "clojure", "erlang", "elixir", "ocaml", "fsharp",
        "functional-programming", "swift", "objective-c", "assembly", "assembly-language", "lua", "r", "matlab",
        "julia", "prolog", "smalltalk", "racket", "common-lisp", "fortran", "cobol", "pascal", "delphi",
    ],
    "software-engineering": [
        "software-engineering", "software-architecture", "software-design", "design-patterns", "agile", "scrum",
        "refactoring", "clean-code", "testing", "tdd", "test-driven-development", "object-oriented",
        "object-oriented-programming", "oop", "uml", "version-control", "git", "code-quality",
        "extreme-programming", "software-craftsmanship", "domain-driven-design", "api", "software",
        "software-project-management", "requirements", "qa", "software-testing", "software-engineering-programming",
        "systems-analysis", "systems-engineering", "architecture", "patterns",
    ],
    "databases": [
        "databases", "database", "sql", "mysql", "postgresql", "nosql", "mongodb", "database-design",
        "data-modeling", "data-warehousing", "oracle", "redis", "dbms", "relational-databases", "data-management",
    ],
    "networking": [
        "networking", "networks", "computer-networking", "tcp-ip", "network-programming", "internet-protocols",
        "wireless", "computer-networks", "network-engineering", "cisco", "telecommunications",
    ],
    "systems": [
        "operating-systems", "operating-system", "linux", "unix", "kernel", "linux-kernel", "windows",
        "system-administration", "sysadmin", "shell", "bash", "command-line", "embedded-systems", "embedded",
        "firmware", "real-time", "concurrency", "parallel-computing", "multithreading", "high-performance-computing",
        "hpc", "gpu", "cuda", "systems-programming", "os", "arduino", "raspberry-pi", "it", "sys-admin", "devices",
    ],
    "architecture": [
        "computer-architecture", "computer-organization", "digital-logic", "hardware", "microprocessors", "vlsi",
        "fpga", "cpu", "computer-hardware", "digital-design", "computer-engineering", "microcontrollers",
    ],
    "compilers": [
        "compilers", "compiler", "compiler-design", "interpreters", "language-design", "parsing",
        "programming-language-theory", "language-implementation", "programming-language-design",
    ],
    "security": [
        "security", "computer-security", "information-security", "infosec", "cybersecurity", "hacking",
        "cryptography", "network-security", "penetration-testing", "malware", "security-engineering", "hackers",
        "cyber-security", "crypto", "reverse-engineering", "forensics", "computer-forensics", "ethical-hacking",
    ],
    "distributed": [
        "distributed-systems", "cloud-computing", "cloud", "aws", "docker", "kubernetes", "devops", "sre",
        "system-design", "scalability", "microservices", "distributed-computing", "azure", "infrastructure",
        "site-reliability-engineering", "serverless", "performance", "web-services",
    ],
    "graphics-games": [
        "computer-graphics", "graphics", "game-development", "game-programming", "game-design", "opengl", "3d",
        "rendering", "unity", "game-engine", "ray-tracing", "directx", "shaders", "game-dev", "gamedev",
        "animation", "unreal-engine", "computer-animation", "3d-graphics",
    ],
    "hci": [
        "hci", "human-computer-interaction", "user-interface", "usability", "ux", "ui", "interaction-design",
        "user-experience", "ux-design", "ui-design", "information-architecture",
    ],
    "quantum": ["quantum-computing", "quantum-information", "quantum-computation", "quantum-cryptography"],
    "blockchain": ["blockchain", "bitcoin", "cryptocurrency", "smart-contracts", "ethereum", "cryptocurrencies"],
    "mobile": ["android", "ios", "mobile", "mobile-development", "iphone", "app-development", "ipad", "mobile-apps"],
    "computer-science": [
        "computer-science", "cs", "compsci", "comp-sci", "computing", "computers", "computer", "technology",
        "tech", "computing-history", "history-of-computing", "computer-history", "information-technology",
        "tech-books", "technical", "technical-books", "computer-books", "computer-science-books", "informatics",
        "information-systems", "internet-technology", "digital", "cs-books", "mis", "it-books", "eecs",
    ],
}

# Fiction/etc shelves used only as a negative signal ("this is a novel, not a textbook").
NONTECH_SHELVES = [
    "fiction", "novels", "novel", "fantasy", "romance", "young-adult", "ya", "mystery", "thriller",
    "science-fiction", "sci-fi", "scifi", "historical-fiction", "horror", "paranormal", "urban-fantasy",
    "contemporary", "chick-lit", "erotica", "manga", "graphic-novels", "comics", "childrens", "children",
    "picture-books", "poetry", "cozy-mystery", "paranormal-romance", "contemporary-romance", "dystopian",
    "classics", "adventure", "humor", "crime", "short-stories", "fairy-tales", "drama", "plays", "literature",
    "memoir", "biography", "cookbooks", "cooking", "christian", "spirituality", "self-help", "magic",
]

# Shelves that describe the reader rather than the book; ignored when computing shares.
GENERIC_SHELVES = [
    "to-read", "currently-reading", "favorites", "favourites", "favorite", "owned", "owned-books", "my-books",
    "books-i-own", "library", "kindle", "ebook", "ebooks", "e-books", "audiobook", "audiobooks", "wish-list",
    "wishlist", "default", "own", "read", "physical-books", "to-buy", "abandoned", "dnf", "tbr", "finished",
    "did-not-finish", "non-fiction", "nonfiction", "non-fic", "nf", "library-books", "kindle-books", "owned-print",
    "school", "all-time-favorites", "have", "books", "read-more-than-once", "reference", "textbook", "textbooks",
    "currently-reading-nonfiction", "to-read-nonfiction", "bookshelf", "unread", "i-own", "e-book", "on-hold",
    "borrowed", "purchased", "gift", "to-reread", "reread",
]
GENERIC_REGEX = (
    r"^(?:" + "|".join(re.escape(s) for s in GENERIC_SHELVES) + r")$|^read-in-|^(?:19|20)[0-9]{2}$|^to-read-|-to-read$"
)

# Title patterns that rescue technical books whose shelves are sparse.
TITLE_RESCUE_REGEX = (
    r"\b(algorithms?|programming|programmer|software|machine learning|deep learning|neural networks?|"
    r"artificial intelligence|data structures?|compilers?|operating systems?|databases?|computer science|"
    r"computing|cryptography|python|java|javascript|c\+\+|kubernetes|linux|unix|sql|devops|data science|"
    r"computer networks?|networking|genetic algorithms?|evolutionary|information retrieval|"
    r"natural language processing|computer vision|robotics|distributed systems|cloud|web development|"
    r"design patterns|object[- ]oriented|computer architecture|theory of computation|automata|"
    r"computer graphics|game development|cybersecurity|hacking|embedded|reinforcement learning)\b"
)

# --------------------------------------------------------------------------------------
# Subject aliases: normalized subject -> extra phrases appended to the query.
# Subjects are normalized with kitaabcopy.text.norm_text ("C++" -> "cpp").
# --------------------------------------------------------------------------------------
SUBJECT_ALIASES: dict[str, list[str]] = {
    "machine learning": ["ml", "statistical learning", "pattern recognition", "data mining", "predictive modeling"],
    "ml": ["machine learning", "statistical learning", "pattern recognition"],
    "deep learning": ["neural networks", "neural network", "tensorflow", "pytorch", "keras"],
    "neural networks": ["deep learning", "neural network", "connectionist"],
    "artificial intelligence": ["ai", "intelligent agents", "knowledge representation", "machine learning"],
    "ai": ["artificial intelligence", "intelligent agents", "machine learning"],
    "nlp": ["natural language processing", "computational linguistics", "text mining", "speech and language"],
    "natural language processing": ["nlp", "computational linguistics", "text mining", "speech and language"],
    "computer vision": ["image processing", "opencv", "machine vision", "image analysis"],
    "reinforcement learning": ["markov decision processes", "q learning", "policy gradient", "agents"],
    "evolutionary computing": [
        "evolutionary computation", "evolutionary algorithms", "genetic algorithms", "genetic programming",
        "swarm intelligence", "metaheuristics", "optimization",
    ],
    "evolutionary computation": ["evolutionary computing", "evolutionary algorithms", "genetic algorithms", "genetic programming", "metaheuristics"],
    "genetic algorithms": ["evolutionary computation", "evolutionary algorithms", "genetic programming", "optimization"],
    "cpp": ["c plus plus", "stl", "modern cpp", "c++ programming", "object oriented programming"],
    "c": ["c programming", "c language", "ansi c", "pointers", "systems programming"],
    "csharp": ["c sharp", "dotnet", "net framework", "asp net"],
    "python": ["python programming", "python 3", "scripting"],
    "java": ["java programming", "jvm", "object oriented programming"],
    "javascript": ["js", "web development", "node js", "ecmascript", "typescript"],
    "algorithms": ["algorithm design", "data structures", "algorithm analysis", "competitive programming"],
    "data structures": ["algorithms", "algorithm design", "data structures and algorithms"],
    "operating systems": ["os", "operating system concepts", "kernel", "unix", "concurrency"],
    "os": ["operating systems", "operating system concepts", "kernel"],
    "computer networks": ["networking", "tcp ip", "internet protocols", "network programming"],
    "networking": ["computer networks", "tcp ip", "internet protocols"],
    "databases": ["database systems", "sql", "dbms", "relational databases", "data modeling"],
    "sql": ["databases", "relational databases", "query"],
    "compilers": ["compiler design", "parsing", "interpreters", "programming language implementation"],
    "computer architecture": ["computer organization", "microprocessors", "digital logic", "hardware"],
    "cryptography": ["security", "encryption", "cryptographic protocols", "information security"],
    "security": ["cybersecurity", "information security", "hacking", "cryptography"],
    "distributed systems": ["cloud computing", "scalability", "system design", "consensus"],
    "software engineering": ["software design", "software architecture", "design patterns", "agile", "clean code"],
    "design patterns": ["software design", "object oriented", "software architecture", "refactoring"],
    "theory of computation": ["automata", "computability", "computational complexity", "formal languages"],
    "discrete mathematics": ["discrete math", "combinatorics", "graph theory", "logic"],
    "data science": ["data analysis", "statistics", "machine learning", "data visualization", "pandas"],
    "robotics": ["robots", "control systems", "autonomous", "mechatronics"],
    "computer graphics": ["rendering", "opengl", "ray tracing", "3d graphics"],
    "quantum computing": ["quantum information", "quantum computation", "quantum algorithms"],
    "information retrieval": ["search engines", "text mining", "ranking"],
    "linux": ["unix", "system administration", "command line", "kernel"],
    "game development": ["game programming", "game engine", "game design", "unity"],
}

POPULAR_SUBJECTS = [
    "Machine Learning", "Deep Learning", "Artificial Intelligence", "Reinforcement Learning", "NLP",
    "Computer Vision", "Evolutionary Computing", "Data Science", "Algorithms", "Data Structures",
    "C++", "C", "Python", "Java", "JavaScript", "Rust", "Operating Systems", "Computer Networks",
    "Databases", "Compilers", "Computer Architecture", "Cryptography", "Computer Security",
    "Distributed Systems", "Software Engineering", "Design Patterns", "Theory of Computation",
    "Discrete Mathematics", "Computer Graphics", "Robotics", "Quantum Computing", "Information Retrieval",
]

LEVELS = ["Beginner", "Intermediate", "Expert"]
GENRES = ["Practical", "Theoretical"]

# --------------------------------------------------------------------------------------
# Ranking weights (tuned against the labeled set with `python -m kitaabcopy.evaluate tune`).
# --------------------------------------------------------------------------------------
DEFAULT_WEIGHTS = {
    "relevance": 0.35,   # hybrid TF-IDF (+ dense) similarity between query and book text
    "shelf": 0.15,       # share of reader shelves that name the subject
    "level": 0.20,       # P(book is at requested difficulty)
    "genre": 0.10,       # P(book is practical / theoretical as requested)
    "quality": 0.20,     # Bayesian rating + popularity percentile
}
MMR_LAMBDA = 0.75
N_CANDIDATES = 400
MIN_RELEVANCE = 0.02    # raw similarity floor; candidates below are dropped
