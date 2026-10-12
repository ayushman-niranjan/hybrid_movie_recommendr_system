# 🎬 Hybrid Movie Recommendation System

A movie recommender that blends **content-based filtering** and **collaborative filtering** into a single hybrid score, with an interactive [Streamlit](https://streamlit.io) front end that is ready to deploy on [Heroku](https://www.heroku.com).

**[Live Demo](https://REPLACE-WITH-YOUR-HEROKU-APP-URL.herokuapp.com)** · Built with Python, scikit-learn, pandas, SciPy and Streamlit

> Pick one or more movies you like and the app returns recommendations, showing for each one how much of its score came from the content signal and how much from the collaborative signal.

---

## Table of Contents
- [Features](#features)
- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Deploying to Heroku](#deploying-to-heroku)
- [Testing](#testing)
- [Design Decisions & Limitations](#design-decisions--limitations)
- [Roadmap](#roadmap)
- [Dataset & Acknowledgements](#dataset--acknowledgements)
- [Author](#author)

---

## Features
- **Hybrid recommendations**: combines two complementary signals instead of relying on one.
- **Multi-movie input**: select several liked movies; scores are averaged across all of them.
- **Adjustable blend**: a slider moves the weighting from pure content-based to pure collaborative.
- **Cold-start handling**: if the movies you pick have few ratings, the app automatically leans on content-based similarity.
- **Transparent scoring**: the results table shows the hybrid, content and collaborative score of every recommendation.
- **Popularity filter**: a minimum-ratings threshold keeps obscure, barely-rated titles out of the results.
- **Lightweight deployment**: similarities are computed on demand from sparse matrices, so there is no huge precomputed similarity matrix and the app fits comfortably within Heroku's slug size limit.

## How It Works

The recommender uses two signals and merges them.

### 1. Content-based filtering
Each movie is turned into a short text document made of its **genres** (weighted more heavily), user-generated **tags** and **release decade**. These documents are vectorised with **TF-IDF** and compared using **cosine similarity**. Movies that share genres and themes with your picks score high.

### 2. Collaborative filtering (item-item)
Each movie is represented as a vector of user ratings, **mean-centred per user** so that generous and harsh raters are comparable. Cosine similarity between these vectors captures *"people who liked X also liked Y"*. Similarities are damped by a **significance shrinkage** factor, `co / (co + 10)`, where `co` is the number of users who rated both movies, so pairs supported by only a handful of users are not over-trusted.

### 3. Blending
Both score vectors are scaled to `[0, 1]` (the liked movies themselves are excluded before scaling), then combined:

```
confidence = mean over liked movies of  n / (n + 20)        # n = number of ratings

w_content  = alpha
w_collab   = (1 - alpha) * confidence

hybrid     = (w_content * content + w_collab * collab) / (w_content + w_collab)
```

`alpha` is the sidebar slider. Because `confidence` shrinks when the selected movies have few ratings, the collaborative weight drops automatically and the system falls back to content-based recommendations, which is how it handles the **cold-start problem**.

## Tech Stack
| Area | Tools |
|---|---|
| Language | Python 3.12 |
| Data & ML | pandas, NumPy, SciPy (sparse matrices), scikit-learn (TF-IDF, normalisation) |
| Web app | Streamlit |
| Deployment | Heroku (`Procfile`, `requirements.txt`, `.python-version`) |
| Testing | pytest |

## Project Structure
```
.
├── app.py                  # Streamlit user interface
├── recommender.py          # HybridRecommender: content + collaborative + blending
├── build_model.py          # Downloads MovieLens and builds model.pkl
├── requirements.txt        # Python dependencies
├── Procfile                # Heroku process definition
├── .python-version         # Python version for Heroku
├── .streamlit/config.toml  # Streamlit server/browser settings
├── dev/
│   └── make_sample_data.py # Tiny synthetic dataset for offline testing
└── tests/
    └── test_recommender.py # Unit tests
```

## Getting Started

### Prerequisites
- Python 3.12 (or a recent 3.x)
- Internet access on the first run, to download the MovieLens dataset

### Installation
```bash
git clone https://github.com/ayushman-niranjan/hybrid-movie-recommender.git
cd hybrid-movie-recommender

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Build the model
```bash
python build_model.py
```
This downloads MovieLens `ml-latest-small`, builds the TF-IDF and rating matrices, and writes `model.pkl`.
(If `model.pkl` is missing, the app will build it automatically on first launch.)

### Run the app
```bash
streamlit run app.py
```
Open the URL Streamlit prints (usually http://localhost:8501).

## Deploying to Heroku

1. Build the model locally and commit it, so the dyno starts quickly without re-downloading data:
   ```bash
   python build_model.py
   git add model.pkl
   git commit -m "Add trained model artifacts"
   ```
2. Create the app and deploy:
   ```bash
   heroku login
   heroku create <your-app-name>
   git push heroku main
   heroku open
   ```
3. Heroku detects Python from `requirements.txt` and `.python-version`, and starts the app using the `Procfile`:
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
   ```

> Heroku no longer offers a free tier, so a paid dyno is required. Check current pricing before deploying.

## Testing
```bash
pip install pytest
python -m pytest tests
```
The tests generate a small synthetic dataset in MovieLens format (no download needed), build a model and check that recommendations are valid: the liked movies are never recommended, scores lie in `[0, 1]` and are sorted, all blend settings work, and content-only recommendations share genres with the selected movie.

## Design Decisions & Limitations
- **On-demand similarity instead of a full matrix.** A dense 9.7k × 9.7k similarity matrix would be hundreds of MB. Computing similarity only for the selected movies keeps memory and slug size small.
- **Item-item over matrix factorisation.** It is simple, explainable, and works well with the small dataset; the per-recommendation score breakdown comes for free.
- **Small dataset.** `ml-latest-small` has roughly 100k ratings from about 600 users, so collaborative signals are noisy for less-rated movies. That is exactly why the blend leans on content when ratings are scarce.
- **Not yet evaluated offline.** There are no ranking metrics such as precision@k or NDCG yet; the unit tests check correctness of the pipeline, not recommendation quality.
- **Title-based input only.** There are no user accounts or personalised rating history.

## Roadmap
- [ ] Offline evaluation (train/test split, precision@k, recall@k, NDCG) to tune `alpha`, shrinkage and confidence constants
- [ ] Movie posters and overviews via the TMDB API
- [ ] User-based mode using a user's own rating history
- [ ] Larger MovieLens dataset and matrix-factorisation (SVD) comparison
- [ ] Screenshots / demo GIF in this README

## Dataset & Acknowledgements
This project uses the [MovieLens](https://grouplens.org/datasets/movielens/) `ml-latest-small` dataset from GroupLens Research. The data is downloaded at build time and is **not** redistributed in this repository, in line with the dataset's usage terms.

> F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.* ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. https://doi.org/10.1145/2827872

Inspired by the CampusX tutorial on building and deploying a movie recommender.

## Author
**Ayushman Niranjan**, B.Tech ECE, BIT Mesra
[LinkedIn](https://www.linkedin.com/in/ayushman-niranjan) · [GitHub](https://github.com/ayushman-niranjan)
