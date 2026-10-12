"""Build the recommender artifacts (model.pkl) from MovieLens data.

Usage:
    python build_model.py                      # downloads MovieLens ml-latest-small
    python build_model.py --data-dir some/dir  # use local movies.csv / ratings.csv / tags.csv

Two signals are prepared:
  * content   -> TF-IDF over genres + user tags (+ decade)
  * collab    -> item vectors from mean-centred ratings (item-item collaborative filtering)
"""
import argparse
import io
import pickle
import re
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


def load_data(data_dir):
    if data_dir:
        read = lambda name: pd.read_csv(f"{data_dir}/{name}")
    else:
        print("Downloading MovieLens ml-latest-small ...")
        raw = urllib.request.urlopen(URL, timeout=60).read()
        zf = zipfile.ZipFile(io.BytesIO(raw))
        read = lambda name: pd.read_csv(zf.open(f"ml-latest-small/{name}"))
    movies, ratings = read("movies.csv"), read("ratings.csv")
    try:
        tags = read("tags.csv")
    except Exception:
        tags = pd.DataFrame(columns=["movieId", "tag"])
    return movies, ratings, tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default="model.pkl")
    args = ap.parse_args()

    movies, ratings, tags = load_data(args.data_dir)

    # ---- movie table -------------------------------------------------------
    movies = movies.copy()
    year = movies["title"].str.extract(r"\((\d{4})\)\s*$")[0]
    movies["year"] = pd.to_numeric(year, errors="coerce")
    movies["clean_title"] = movies["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)
    movies = movies.reset_index(drop=True)
    mid_to_idx = {m: i for i, m in enumerate(movies["movieId"])}

    stats = ratings.groupby("movieId")["rating"].agg(n_ratings="count", avg_rating="mean")
    movies = movies.join(stats, on="movieId")
    movies["n_ratings"] = movies["n_ratings"].fillna(0).astype(int)
    movies["avg_rating"] = movies["avg_rating"].round(2)

    # ---- content-based: TF-IDF on genres + tags + decade -------------------
    tag_text = (
        tags.assign(tag=tags["tag"].astype(str).str.lower())
        .groupby("movieId")["tag"].apply(" ".join)
    )
    def doc(row):
        g = [re.sub(r"[^a-z0-9]", "", x.lower()) for x in str(row.genres).split("|")
             if x != "(no genres listed)"]
        parts = g * 3                                   # genres matter most
        if pd.notna(row.year):
            parts.append(f"decade{int(row.year) // 10 * 10}")
        t = tag_text.get(row.movieId, "")
        return " ".join(parts) + " " + t
    docs = [doc(r) for r in movies.itertuples()]
    tfidf = TfidfVectorizer(token_pattern=r"[^\s]+", sublinear_tf=True, min_df=1)
    content = normalize(tfidf.fit_transform(docs)).tocsr().astype(np.float32)

    # ---- collaborative: mean-centred item vectors --------------------------
    r = ratings[ratings["movieId"].isin(mid_to_idx)].copy()
    r["rc"] = r["rating"] - r.groupby("userId")["rating"].transform("mean")
    users = {u: i for i, u in enumerate(sorted(r["userId"].unique()))}
    rows = r["movieId"].map(mid_to_idx).to_numpy()
    cols = r["userId"].map(users).to_numpy()
    shape = (len(movies), len(users))
    centred = sp.csr_matrix((r["rc"].to_numpy(np.float32), (rows, cols)), shape=shape)
    binary = sp.csr_matrix((np.ones(len(r), np.float32), (rows, cols)), shape=shape)
    collab = normalize(centred).tocsr()

    with open(args.out, "wb") as f:
        pickle.dump(
            {"movies": movies.drop(columns=["movieId"]).assign(movie_id=movies["movieId"]),
             "content": content, "collab": collab, "binary": binary},
            f, protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"Saved {args.out}: {len(movies)} movies, {len(users)} users, {len(r)} ratings")


if __name__ == "__main__":
    main()
