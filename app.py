"""KitaabCopy – Streamlit UI.   Run:  streamlit run app.py"""
from __future__ import annotations

import streamlit as st

from kitaabcopy import config
from kitaabcopy.recommender import KitaabRecommender, Query, Settings

st.set_page_config(page_title="KitaabCopy", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Loading book index ...")
def get_engine() -> KitaabRecommender:
    return KitaabRecommender(config.ARTIFACTS)


st.title("📚 KitaabCopy")
st.caption("Top-3 technical book recommendations: relevant to your subject, matched to your level and style, "
           "high quality, and diverse.")

try:
    engine = get_engine()
except FileNotFoundError:
    st.error("Index not found. Run `python -m kitaabcopy.prepare_data` and `python -m kitaabcopy.build_index` first.")
    st.stop()

with st.sidebar:
    st.header("Your request")
    preset = st.selectbox("Popular subjects", ["(type your own)"] + config.POPULAR_SUBJECTS)
    custom = st.text_input("Subject", value="" if preset == "(type your own)" else preset,
                           placeholder="e.g. Evolutionary Computing, C++, Compilers")
    level = st.radio("Difficulty", config.LEVELS, index=1, horizontal=True)
    genre = st.radio("Style", config.GENRES, horizontal=True)
    go = st.button("Recommend", type="primary", use_container_width=True)
    with st.expander("Advanced"):
        lam = st.slider("Diversity (lower = more diverse)", 0.4, 1.0, config.MMR_LAMBDA, 0.05)
        weights = {k: st.slider(f"weight: {k}", 0.0, 1.0, v, 0.05) for k, v in config.DEFAULT_WEIGHTS.items()}

if go:
    subject = custom.strip()
    if not subject:
        st.warning("Please enter a subject.")
        st.stop()
    settings = Settings(weights=weights, mmr_lambda=lam)
    with st.spinner("Ranking ..."):
        picks = engine.recommend(Query(subject, level, genre), k=3, settings=settings)
    if not picks:
        st.info("No technical books matched that subject. Try a broader or differently worded subject.")
        st.stop()

    st.subheader(f"Top {len(picks)} for **{subject}** · {level} · {genre}")
    for p in picks:
        with st.container(border=True):
            c_img, c_main, c_scores = st.columns([1, 4, 2])
            with c_img:
                if p["image_url"] and "nophoto" not in str(p["image_url"]):
                    st.image(p["image_url"], width=110)
                else:
                    st.markdown("### 📕")
            with c_main:
                st.markdown(f"### #{p['rank']} · {p['title']}")
                meta = [p["authors"]]
                if p["year"]:
                    meta.append(str(p["year"]))
                if p["pages"]:
                    meta.append(f"{p['pages']} pages")
                st.markdown(" · ".join(meta))
                st.markdown(f"⭐ **{p['rating']:.2f}** from {p['ratings_count']:,} ratings")
                st.write(p["explanation"]["summary"])
                with st.expander("Why this book?", expanded=True):
                    for b in p["explanation"]["bullets"]:
                        st.markdown(f"- {b}")
                st.link_button("Open on Goodreads", p["url"])
            with c_scores:
                st.markdown("**Score breakdown**")
                for name in ("relevance", "shelf", "level", "genre", "quality"):
                    v = p["scores"][name]
                    st.progress(min(1.0, max(0.0, v)), text=f"{name}: {v:.2f}")
                st.caption(f"final score {p['scores']['final']:.3f}")
else:
    st.info("Pick a subject, difficulty and style in the sidebar, then press **Recommend**.")
