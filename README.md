# NLP-Powered Auslan Sign Retrieval System

This system uses transformer models and NLP techniques to match natural language text to Auslan (Australian Sign Language) sign videos. It features a multi-strategy matching pipeline, sentiment/emotion analysis, and a modern web interface.

## Research Question

How can natural language processing be effectively applied to retrieve and display appropriate Auslan signs?

## Quick Start

### Backend

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# FastAPI backend (primary — used by the frontend)
uvicorn api:app --reload --host 0.0.0.0 --port 8000

# Or use the CLI directly
python main.py "I feel happy"
python main.py --test
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# Open http://localhost:3000
```

### Docker

```bash
docker-compose up
```

## Project Structure

```
main.py                # AuslanSignSystem orchestrator (framework-agnostic)
api.py                 # FastAPI REST backend (primary)
app.py                 # Flask web app (legacy)
evaluate.py            # Evaluation and benchmarking script
src/
  preprocessing.py     # TextPreprocessor — lowercasing, contractions, tokenization
  matcher.py           # SignMatcher — exact/synonym/semantic matching pipeline
  phrase_matcher.py    # IntelligentPhraseMatcher — spaCy phrase segmentation, grammar reordering
  nlp_features.py      # EnhancedNLPProcessor — sentiment, emotion, NER, intent analysis
scripts/
  scraper.py           # Auslan dictionary scraper
frontend/              # Next.js 14 + React 18 + Tailwind CSS
  app/                 # Pages (home, about)
  components/          # SearchBar, SignCard, SignResults, NLPAnalysis, Navbar, etc.
tests/                 # pytest suite (175 tests)
  test_api.py          # FastAPI endpoint tests
  test_matcher.py      # SignMatcher tests
  test_phrase_matcher.py
  test_preprocessing.py
data/
  gloss/auslan_dictionary.json   # 46-sign dictionary (gloss, video URL, synonyms, category)
  synonyms/synonym_mapping.json  # Synonym-to-primary mappings
  target_words.json              # Target words with synonym arrays
media/videos/          # 46 .mp4 sign videos (Git LFS)
templates/             # Jinja2 templates (Flask legacy)
.github/workflows/ci.yml  # CI pipeline
```

## Matching Pipeline

Signs are matched in priority order:

1. **Exact match** — direct dictionary key lookup (100% confidence)
2. **Synonym match** — dictionary synonyms + external mappings (90% confidence)
3. **Semantic match** — `all-MiniLM-L6-v2` cosine similarity against pre-computed embeddings (confidence = similarity score)

The `use_synonym` parameter (default `True`) controls whether synonym matching is enabled. The system gracefully degrades when ML dependencies are unavailable, falling back to lexicon-based approaches.

## AI/ML Models

| Model | Purpose |
|-------|---------|
| `all-MiniLM-L6-v2` | Semantic similarity (384-dim embeddings) |
| `distilbert-base-uncased-finetuned-sst-2-english` | Sentiment analysis |
| `j-hartmann/emotion-english-distilroberta-base` | 7-class emotion detection |
| spaCy `en_core_web_sm` | NER, dependency parsing, phrase extraction |

## API Endpoints (FastAPI)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/models/status` | Model availability |
| `POST` | `/api/process` | Process text, return sign matches + NLP analysis |
| `POST` | `/api/evaluate` | Batch evaluation |
| `POST` | `/api/analyze` | NLP analysis only |
| `POST` | `/api/suggestions` | Phrase autocomplete |
| `GET` | `/api/dictionary` | Dictionary info |
| `GET` | `/media/videos/{filename}` | Serve sign videos |

## Vocabulary (46 Signs)

| Category | Signs |
|----------|-------|
| Greetings & Social (7) | hello, goodbye, thank, please, good, friend, see |
| Fitness & Exercise (15) | exercise, strong, muscle, weight, lift, stretch, breathe, rest, warm, cool, run, bike, chest, arms, legs |
| Basic Needs (6) | eat, drink, sleep, help, food, water |
| Emotions (3) | happy, sad, angry |
| Actions (6) | go, come, sit, stand, walk, buy |
| Places & Objects (4) | house, toilet, big, speak |
| Temporal & Descriptive (5) | today, tomorrow, time, many, more |

## Expanding the Dictionary (Scraper)

The base dictionary ships with 46 signs. The scraper crawls [Auslan Signbank](https://auslan.org.au/) to expand it by discovering new signs and merging them into the dictionary. It downloads sign videos and extracts keywords, descriptions, and metadata.

```bash
# Full scrape — crawl all signs, download videos, merge into dictionary
python scripts/scraper.py

# Scrape first 100 signs only (good for testing)
python scripts/scraper.py --limit 100

# Preview what would be scraped without writing any files
python scripts/scraper.py --dry-run

# Scrape without downloading videos
python scripts/scraper.py --no-download

# Scrape without merging into the main dictionary
python scripts/scraper.py --no-merge

# Custom output path and request delay
python scripts/scraper.py --output data/gloss/signbank_scraped.json --delay 2.0
```

The scraper respects `robots.txt`, rate-limits requests (default 1.5s delay), and saves progress incrementally so it can resume interrupted runs. Existing dictionary entries are never overwritten during merge.

Videos are saved to `D:\nlp\auslan-videos` by default (override with `SCRAPED_VIDEO_DIR` env var).

> **Note**: Auslan Signbank content is licensed CC BY-NC-ND 4.0 — non-commercial research/education use only.

## CLI Flags

Applies to `main.py` and `evaluate.py`:

- `--no-stops` — remove stop words
- `--no-semantic` — disable semantic matching
- `--stem` — enable Porter stemming
- `--thresh=0.6` — semantic similarity threshold (0.0-1.0)

## Testing

```bash
pytest tests/ -v --tb=short -x
```

175 tests covering the API, matcher, phrase matcher, and preprocessing. Tests use mocks — no ML model downloads needed.

## CI Pipeline

GitHub Actions (`.github/workflows/ci.yml`) runs on push to `main` and PRs:

- **Backend lint** — `ruff check` + `ruff format --check`
- **Backend tests** — `pytest` (no ML models)
- **Frontend lint** — ESLint (`next/core-web-vitals`) + TypeScript type check

## Tech Stack

- **Backend**: Python 3.11+, FastAPI, spaCy, HuggingFace Transformers, SentenceTransformers
- **Frontend**: Next.js 14, React 18, Tailwind CSS, Framer Motion, Radix UI, shadcn/ui
- **Legacy**: Flask + Jinja2 (`app.py`)
- **Testing**: pytest, httpx
- **Linting**: ruff (Python), ESLint + TypeScript (frontend)
- **Infrastructure**: Docker, GitHub Actions CI
