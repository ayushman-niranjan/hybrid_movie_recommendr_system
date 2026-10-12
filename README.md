# 🎬 Hybrid Movie Recommender (Streamlit + Heroku)

Pick movies you like → get recommendations that blend two signals:

| Signal | How |
|---|---|
| **Content-based** | TF-IDF on genres (+ user tags + decade), cosine similarity |
| **Collaborative** | Item-item cosine on mean-centred ratings, shrunk when few users co-rated |

`score = (α·content + (1-α)·confidence·collab) / (α + (1-α)·confidence)`

`α` is the sidebar slider. `confidence` drops when your picks have few ratings, so
unpopular / new movies automatically fall back to content-based (cold start).

## Run locally
```bash
pip install -r requirements.txt
python build_model.py          # downloads MovieLens (ml-latest-small) -> model.pkl
streamlit run app.py
```
(`app.py` also builds `model.pkl` itself on first run if it's missing.)

## Deploy to Heroku
```bash
# one-time: install Heroku CLI, then
heroku login
git init && git add . && git commit -m "hybrid recommender"
heroku create my-movie-recommender
git push heroku main        # use `master` if your branch is named master
heroku open
```
Commit `model.pkl` (it's small, a few MB) so the dyno starts instantly and doesn't need to re-download data.
`Procfile`, `requirements.txt` and `.python-version` are already set up.

## Files
- `build_model.py` – downloads data, builds `model.pkl`
- `recommender.py` – the hybrid engine
- `app.py` – Streamlit UI
- `dev/make_sample_data.py` – tiny fake dataset for offline testing only
