# Auslan NLP Sign Retrieval

NLP-powered system that matches natural language text input to Auslan (Australian Sign Language) sign videos. Uses a pipeline of exact, synonym, and semantic matching strategies with graceful degradation when heavy ML dependencies are unavailable.

## Quick Start

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# CLI (interactive)
python main.py

# CLI (single input)
python main.py "I feel happy"

# CLI (batch test)
python main.py --test

# Web app (Flask, http://localhost:5000)
python app.py

# Evaluation
python evaluate.py
python evaluate.py --export results.json
```

## CLI Flags

Applies to both `main.py` and `evaluate.py`:
- `--no-stops` — remove stop words during preprocessing
- `--no-semantic` — disable transformer-based semantic matching
- `--stem` — enable Porter stemming
- `--thresh=0.6` — semantic similarity threshold (0.0–1.0)

## Project Structure

```
main.py              # CLI entry point, AuslanSignSystem orchestrator class
app.py               # Flask web app with REST API
evaluate.py          # Standalone evaluation/benchmarking script
src/
  preprocessing.py   # TextPreprocessor — lowercasing, contraction expansion, tokenization
  matcher.py         # SignMatcher — exact/synonym/semantic matching (3-tier pipeline)
  phrase_matcher.py  # IntelligentPhraseMatcher — spaCy-based phrase segmentation, ASL/Auslan grammar reordering
  nlp_features.py    # EnhancedNLPProcessor — sentiment, emotion, NER, intent, formality analysis
data/
  gloss/auslan_dictionary.json   # 46-sign dictionary (gloss, video URL, synonyms, category)
  synonyms/synonym_mapping.json  # External synonym mappings
  target_words.json              # Target words with synonym arrays
media/videos/        # 46 .mp4 sign videos (Git LFS)
templates/           # Jinja2 templates (base.html, index.html, about.html)
```

## Architecture

Matching pipeline in priority order:
1. **Exact match** — direct dictionary key lookup (100% confidence)
2. **Synonym match** — dictionary synonyms + external synonym_mapping.json (90% confidence)
3. **Semantic match** — `all-MiniLM-L6-v2` cosine similarity against pre-computed embeddings (confidence = similarity score, filtered by threshold)

The system gracefully degrades: if transformers/spaCy are not installed, it falls back to lexicon-based approaches (TextBlob for sentiment, regex for NER/intent).

## API Endpoints (Flask)

- `POST /api/process` — process text, return sign matches + NLP analysis
- `POST /api/evaluate` — batch evaluation
- `POST /api/analyze` — detailed NLP analysis only
- `POST /api/suggestions` — phrase autocomplete
- `GET /api/dictionary` — dictionary info
- `GET /api/models/status` — model availability
- `GET /media/videos/<filename>` — serve video files

## Testing

No formal test framework (no pytest/unittest). Run evaluations with:
- `python main.py --test` — batch test on sample phrases
- `python evaluate.py` — structured evaluation (12 fitness coaching phrases across 3 categories)
- `python src/preprocessing.py` / `python src/matcher.py` — module self-tests via `__main__` blocks

## Code Conventions

- Python 3.8+, type hints throughout, `dataclasses` for structured data
- Components initialize with graceful fallbacks (try/except around heavy imports)
- Dictionary data lives in JSON files under `data/`
- Video files tracked with Git LFS (`.gitattributes`)
- No linter/formatter configured; follow existing style
