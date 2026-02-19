# Web Interface Guide

## Quick Start

### 1. Start the Backend
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### 2. Start the Frontend
```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000

### Alternative: Docker
```bash
docker-compose up
```

### Legacy Flask App
```bash
python app.py
# Open http://localhost:5000
```

---

## Frontend Features

### AI Analysis Dashboard
- **Sentiment Analysis**: Real-time positive/negative/neutral detection
- **Emotion Classification**: 7-class emotion recognition (joy, sadness, anger, fear, surprise, disgust, neutral)
- **Intent Recognition**: Understands user goals (instruction, question, greeting, request)
- **Formality Detection**: Formal/informal/neutral language analysis

### Sign Matching
- **Exact Match**: Direct dictionary lookup (100% confidence)
- **Synonym Match**: Comprehensive synonym mapping (90% confidence)
- **Semantic Match**: Transformer-based AI similarity with configurable threshold

### Processing Options
- **Semantic Matching**: Enable transformer-based matching
- **Intelligent Matching**: Enable phrase-level context analysis
- **Semantic Threshold**: Adjust similarity sensitivity (0.3-0.9)
- **Stop Word Removal**: Filter common words
- **Stemming**: Word root matching

---

## Usage Examples

### Basic Sign Search
1. Enter text: "I feel delighted and thrilled"
2. Keep semantic matching enabled
3. Set threshold to 0.5 (recommended starting point)
4. Click "Find Signs"
5. View matches with confidence scores and NLP analysis

### Threshold Tuning
- **0.3-0.4**: More matches, lower precision
- **0.5-0.6**: Balanced (recommended)
- **0.7-0.8**: Fewer but more precise matches

---

## Troubleshooting

### No Semantic Matches?
- Ensure semantic matching is enabled
- Lower the threshold to 0.3-0.4
- Check that the backend is running on port 8000

### Slow First Load?
- First run downloads AI models (~500MB)
- Subsequent runs use cached models

### Backend Connection Error?
- Verify the FastAPI backend is running: `curl http://localhost:8000/api/health`
- Check CORS settings if frontend and backend are on different hosts
