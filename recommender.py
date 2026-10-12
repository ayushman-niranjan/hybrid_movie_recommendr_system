"""Hybrid (content + collaborative) movie recommender."""
import pickle

import numpy as np
import pandas as pd


class HybridRecommender:
    SHRINK = 10      # damp item-item similarity when few users co-rated both movies
    CONF_K = 20      # a movie needs ~20 ratings before collaborative signal is trusted ~50%

    def __init__(self, path="model.pkl"):
        with open(path, "rb") as f:
            m = pickle.load(f)
        self.movies = m["movies"]
        self.content = m["content"]
        self.collab = m["collab"]
        self.binary = m["binary"]
        self.titles = self.movies["title"].tolist()
        self._lookup = {t: i for i, t in enumerate(self.titles)}

    # ---- individual signals ----------------------------------------------
    def _content_scores(self, idx):
        return np.asarray((self.content @ self.content[idx].T).mean(axis=1)).ravel()

    def _collab_scores(self, idx):
        sim = (self.collab @ self.collab[idx].T).toarray()          # (n_movies, n_liked)
        co = (self.binary @ self.binary[idx].T).toarray()           # co-rating counts
        sim = sim * (co / (co + self.SHRINK))                       # significance shrinkage
        return sim.mean(axis=1)

    @staticmethod
    def _scale(x):
        x = np.clip(x, 0, None)
        m = x.max()
        return x / m if m > 0 else x

    # ---- public API ------------------------------------------------------
    def recommend(self, liked_titles, alpha=0.5, top_n=10, min_ratings=5):
        """alpha = weight on content-based; (1 - alpha) on collaborative.

        The collaborative weight is automatically reduced when the liked movies
        have few ratings (cold start), so unknown movies fall back to content.
        """
        idx = [self._lookup[t] for t in liked_titles]
        c, f = self._content_scores(idx), self._collab_scores(idx)
        c[idx], f[idx] = 0.0, 0.0       # a movie is always most similar to itself; keep it out of scaling
        c, f = self._scale(c), self._scale(f)

        n = self.movies["n_ratings"].to_numpy()[idx]
        confidence = float(np.mean(n / (n + self.CONF_K)))
        w_c, w_f = alpha, (1 - alpha) * confidence
        total = w_c + w_f
        score = (w_c * c + w_f * f) / total if total > 0 else c

        out = self.movies.assign(hybrid=score, content_score=c, collab_score=f)
        out = out.drop(index=idx)
        out = out[out["n_ratings"] >= min_ratings]
        out = out.sort_values(["hybrid", "avg_rating"], ascending=False).head(top_n)
        return out[["title", "genres", "avg_rating", "n_ratings",
                    "hybrid", "content_score", "collab_score"]].reset_index(drop=True), \
               {"collab_confidence": confidence, "w_content": w_c / total if total else 1.0}
