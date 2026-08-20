# KV Money Market

NBA player prop markets with model-generated projections. Search a player, build a prop (category, line, opponent), and the model prices both sides of the contract in cents — the way a prediction market quotes YES/NO.

**Live:** [kv-money-market.vercel.app](https://kv-money-market.vercel.app)

---

## How it works

A gradient-boosted ensemble trains per request on two seasons of game logs, then projects a stat line against a specific opponent. Model confidence becomes an implied price: if the model likes the over at 77%, OVER quotes 77¢ and UNDER 23¢.

**Features the model uses:** recency-weighted stat history (30-day half-life), opponent defensive strength normalized to league average, head-to-head history, home/away, rest days, minutes load, usage rate, true shooting, and EMA trend.

**Base models:** RandomForest, GradientBoosting, XGBoost and LightGBM, combined through a Ridge stacking meta-model with a 95% confidence interval derived from residual spread.

---

## The snapshot architecture

The single most important thing to understand about this repo.

`stats.nba.com` **silently drops traffic from datacenter IP ranges** (AWS, GCP, Azure). Requests hang until timeout rather than returning an error. Every free host runs in one of those ranges, so a deployed backend cannot reach the NBA API — regardless of headers, retries, or user agent. `nba_api` already sends a fully current browser header set; the gate is the source IP, not the request.

The workaround is to stop needing the API at request time:

```
Your Mac                    GitHub                  Vercel
build_snapshot.py    →      backend/data/     →     server.py
(residential IP,            (~10 MB JSON,           (reads files,
 API reachable)              committed)              zero network)
```

`nba_source.py` is the single access point for all NBA data and has two modes:

| Mode | Trigger | Behavior |
| --- | --- | --- |
| **live** | default | Calls `stats.nba.com`. Used locally and to build snapshots. |
| **snapshot** | `NBA_SNAPSHOT=1` | Reads `backend/data/`. No network calls at all. Used in production. |

Both modes return identical DataFrames, so callers cannot tell them apart. Verified: a prediction with every socket severed returns byte-identical output to live mode.

---

## Project layout

```
backend/
  server.py              Flask API (Vercel entrypoint — exports `app`)
  nba_source.py          Data access layer: live vs snapshot
  build_snapshot.py      Builds backend/data/ from the NBA API
  predictive_model.py    Feature engineering + ensemble
  bet_calculations.py    Stats, caching, season constants
  data/                  Committed snapshot (~10 MB, 578 files)
  requirements.txt       Pinned; Python 3.13/3.14
frontend/
  src/
    App.js                  Markets page
    SavedPredictions.js     Positions portfolio (localStorage)
    PredictionScorecard.js  Two-sided market card
    api.js                  Backend origin (REACT_APP_API_URL)
```

---

## Running locally

**Backend** — needs Python 3.13 or 3.14 (pandas 2.x; **not** pandas 3.x):

```bash
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python server.py                    # live mode, port 5001
NBA_SNAPSHOT=1 python server.py     # snapshot mode, no network
```

**Frontend** — Node 20:

```bash
cd frontend
npm install
npm start                           # port 3000
```

Start the backend first — the frontend fetches players and teams on load with a 20-second timeout.

Confirm which mode you are in:

```bash
curl -s localhost:5001/ | python3 -m json.tool
# "data_source": "snapshot" | "live"
```

---

## Refreshing the data

Run on a machine with a residential connection. Takes roughly 30 minutes for 530 players (~2,300 API calls, deliberately rate-limited).

```bash
cd backend && source venv/bin/activate

python build_snapshot.py                # resumable — skips files already on disk
python build_snapshot.py --force        # rebuild everything
python build_snapshot.py --ids 201939   # one player, for testing
python build_snapshot.py --workers 1    # gentler on the API

git add data && git commit -m "Refresh snapshot" && git push
```

The builder backs off globally when the API starts rate-limiting, and writes atomically so an interrupted run never leaves a half-file. If it stalls, re-run it — it picks up where it stopped.

**Staleness matters.** `recency_weight` is computed against the current date while game dates are frozen, so an old snapshot drifts:

| Snapshot age | Predicted | Confidence |
| --- | --- | --- |
| 0–14 days | 27 | 77.3 |
| 30+ days | 26 | 58.8 |

Refresh at least every two weeks; weekly in-season.

---

## API

| Endpoint | Network needed | Notes |
| --- | --- | --- |
| `GET /` | no | Health + data mode |
| `GET /api/teams` | no | Bundled in `nba_api` |
| `GET /api/all-players` | no | Bundled in `nba_api` |
| `GET /api/player-details/<name>` | yes\* | Bio, team color |
| `GET /api/predict` | yes\* | Trains and projects |

\* unless `NBA_SNAPSHOT=1`

**`/api/predict` params:** `player_name`, `category`, `opponent_abbr`, `betting_line`, `season_type`

**Categories:** Points, Rebounds, Assists, Blocks, Steals, Points+Rebounds+Assists, Rebounds+Assists, Points+Rebounds, Points+Assists, Blocks+Steals

The offline endpoints work even when the NBA API is unreachable — so a misconfigured deployment looks healthy while predictions hang. Check `data_source` on `/` first when debugging.

---

## Deployment

Two Vercel projects from one repo.

**Backend** — Root Directory `backend`, framework Flask:

| Variable | Value | Why |
| --- | --- | --- |
| `NBA_SNAPSHOT` | `1` | Read committed data. Without it, predictions hang. |
| `BET_CACHE_DIR` | `/tmp/bet_cache` | Filesystem is read-only except `/tmp`; joblib writes at import. |
| `VERCEL_SUPPORT_LARGE_FUNCTIONS` | `1` | Bundle is ~638 MB vs the 500 MB Python default. |
| `ALLOWED_ORIGINS` | frontend URL | CORS allowlist. |

`VERCEL_SUPPORT_LARGE_FUNCTIONS` is rejected on the project-creation form (reserved `VERCEL_` prefix). Add it after the project exists, via Settings or `vercel env add`.

**Frontend** — Root Directory `frontend`, framework Create React App:

| Variable | Value |
| --- | --- |
| `REACT_APP_API_URL` | backend production URL |

CRA inlines `REACT_APP_*` at **build** time, so changing it requires a rebuild, not a restart.

**Use production domains, not generated URLs.** On Hobby, anything containing a hash, `-git-<branch>-`, or your account slug is protected and 302s to Vercel SSO — including requests from your frontend.

---

## Testing

```bash
cd frontend
CI=true npx react-scripts test --watchAll=false   # 7 tests
CI=true npx react-scripts build                   # must be warning-free
```

---

## Known limitations

- **Snapshot-only players.** A newly signed player returns 503 until the next refresh.
- **~1,000 predictions/month** on Vercel Hobby's 4 CPU-hour allowance — each prediction trains from scratch (~10s).
- **Cold starts** of roughly 5s while ~300 MB of scientific Python loads.
- **Positions are per-browser**, stored in `localStorage`.
- **`PP_AI.py`** is a stale CLI script with a broken import; nothing references it.
