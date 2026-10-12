import os

import streamlit as st

from recommender import HybridRecommender

st.set_page_config(page_title="Hybrid Movie Recommender", page_icon="🎬", layout="wide")


@st.cache_resource
def load_model():
    if not os.path.exists("model.pkl"):
        import build_model, sys
        sys.argv = ["build_model.py"]
        build_model.main()          # first run: download MovieLens and build
    return HybridRecommender("model.pkl")


rec = load_model()

st.title("🎬 Hybrid Movie Recommender")
st.caption("Content-based (genres + tags) blended with collaborative filtering (what similar viewers rated).")

with st.sidebar:
    st.header("Settings")
    alpha = st.slider(
        "Content ⟷ Collaborative", 0.0, 1.0, 0.5, 0.05,
        help="1.0 = pure content-based, 0.0 = pure collaborative filtering.",
    )
    top_n = st.slider("Number of recommendations", 5, 30, 10)
    min_ratings = st.slider("Minimum ratings a movie must have", 1, 100, 10)
    st.markdown(
        "**How it works**\n\n"
        "- *Content*: TF-IDF similarity on genres/tags\n"
        "- *Collaborative*: item-item similarity on mean-centred ratings\n"
        "- If your picks have few ratings, the app leans on content automatically (cold start)."
    )

liked = st.multiselect(
    "Pick one or more movies you like", rec.titles,
    placeholder="Start typing a title…",
)

if st.button("Recommend", type="primary", disabled=not liked):
    results, info = rec.recommend(liked, alpha=alpha, top_n=top_n, min_ratings=min_ratings)
    st.caption(
        f"Effective content weight: {info['w_content']:.0%} · "
        f"collaborative confidence for your picks: {info['collab_confidence']:.0%}"
    )
    st.dataframe(
        results,
        hide_index=True,
        use_container_width=True,
        column_config={
            "title": "Title",
            "genres": "Genres",
            "avg_rating": st.column_config.NumberColumn("Avg rating", format="%.2f ⭐"),
            "n_ratings": "# ratings",
            "hybrid": st.column_config.ProgressColumn("Hybrid score", min_value=0, max_value=1, format="%.2f"),
            "content_score": st.column_config.ProgressColumn("Content", min_value=0, max_value=1, format="%.2f"),
            "collab_score": st.column_config.ProgressColumn("Collaborative", min_value=0, max_value=1, format="%.2f"),
        },
    )
elif not liked:
    st.info("Choose at least one movie to get recommendations.")
