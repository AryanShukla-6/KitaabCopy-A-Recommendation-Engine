# KitaabCopy – technical-book recommender

Input: **subject** (any text, e.g. "Evolutionary Computing", "C++"), **difficulty** (Beginner / Intermediate / Expert),
**style** (Practical / Theoretical). Output: **Top 3** books that are relevant, well-rated, level/style-matched and diverse,
with ratings, a score breakdown, a faithful explanation and a Goodreads link. Evaluated with P@3, Recall@K and NDCG@3
on a human-labeled pool.

## Architecture
```
goodreads_books.json.gz (2 GB)
  └─ prepare_data.py   DuckDB stream -> English, >=25 ratings -> technical filter (shelf taxonomy + title rescue)
                       -> collapse editions by work_id -> books.parquet (+ author names)
  └─ build_index.py    features (difficulty P, genre P, Bayesian quality) + TF-IDF (+ optional MiniLM embeddings)
  └─ recommender.py    candidates (hybrid retrieval) -> 5-signal score -> MMR diversity -> explanations
  └─ app.py            Streamlit UI
  └─ evaluate.py       pool -> label -> P@3 / Recall@K / NDCG@3, baselines, ablations, weight tuning
```
Score = 0.35·relevance + 0.15·shelf-affinity + 0.20·P(level) + 0.10·P(genre) + 0.20·quality (weights tunable),
multiplied by a relevance gate so off-topic books can't win on rating alone; Top-3 chosen by MMR with one-author-per-list
and near-duplicate filtering.

## Run on the real data
```bash
pip install -r requirements.txt
# put goodreads_books.json.gz and goodreads_book_authors.json.gz in data/raw/
python -m kitaabcopy.prepare_data            # ~ minutes; prints books-per-topic
python -m kitaabcopy.coverage_check          # are canonical CS/AI books in the index? loosen thresholds if not
python -m kitaabcopy.build_index             # add --embeddings (pip install sentence-transformers) for hybrid retrieval
streamlit run app.py
```
**Coverage ("include most core CS / AI books")**: `prepare_data` is deliberately recall-oriented (shelf share >= 12%,
or a technical title keyword with >= 4% share; fiction-dominated books removed). Run `coverage_check` — it looks up ~60
canonical titles (CLRS, SICP, AIMA, Goodfellow, K&R, Dragon Book, ...) and lists misses. If recall is low, lower
`--min-share` / `--min-tech-cnt` / `--min-ratings`, or add shelf names to `config.TOPIC_SHELVES`.

## Evaluation
```bash
python -m kitaabcopy.evaluate pool  --queries eval/queries.csv     # writes eval/annotation_sheet.csv (50 queries)
# fill the `label` column 0-3 (rubric in evaluate.py docstring); a 2nd person fills `label_2` for ~20% of rows
python -m kitaabcopy.evaluate score --labels eval/annotation_sheet.csv   # table: full vs baselines vs ablations + kappa
python -m kitaabcopy.evaluate tune  --labels eval/annotation_sheet.csv   # weight search, reported on held-out queries
```
Relevant = label >= 2. Systems compared: `full`, `no_diversity`, `no_level`, `no_genre`, `no_level_genre`, `no_quality`,
`no_shelf`, `baseline_tfidf`, `baseline_rating`. Also reports list diversity@3 and mean rating@3.

## Honest limitations (put these in your write-up)
* Difficulty and genre are **inferred** from title/description/shelf/page-count cues; Goodreads has no such labels.
* Recall@K is relative to the **pooled labeled** relevant books, not all relevant books in Goodreads.
* Ratings are noisy and popularity-biased; the Bayesian average + popularity blend mitigates but doesn't remove this.
* `tests/make_synthetic.py` data is for smoke tests only; numbers from it are not results.

## Tests
`python -m pytest -q`  (metrics unit tests + end-to-end pipeline on synthetic data)
