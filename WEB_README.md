# Auslan Sign Retrieval Web Interface

## Quick Start

1. **Install Dependencies:**
   ```bash
   pip install flask numpy nltk
   ```

2. **Run the Web Application:**
   ```bash
   python app.py
   ```

3. **Open in Browser:**
   Navigate to: http://localhost:5000

## Features

### 🖥️ **Web Interface**
- Clean, responsive design with Bootstrap 5
- Real-time text processing
- Interactive options panel
- Video integration ready

### 🔍 **Sign Processing**
- Enter any text to find corresponding Auslan signs
- Multiple matching strategies (exact, synonym, semantic)
- Adjustable processing options
- Real-time statistics and coverage metrics

### 📊 **Evaluation Tools**
- Built-in system evaluation with sample texts
- Performance metrics and analysis
- Customizable test parameters

### 🎥 **Video Integration**
- Direct video playback for sign demonstrations
- Video files served from `/media/videos/` endpoint
- Supports MP4 format

## API Endpoints

### `POST /api/process`
Process text input and return sign matches.

**Request:**
```json
{
  "text": "Hello, I need help",
  "options": {
    "remove_stops": false,
    "use_semantic": true,
    "use_stemming": false,
    "semantic_threshold": 0.6
  }
}
```

**Response:**
```json
{
  "original_text": "Hello, I need help",
  "total_tokens": 4,
  "signs_found": 2,
  "successful_matches": [...],
  "coverage_stats": {...}
}
```

### `POST /api/evaluate`
Run system evaluation with test texts.

### `GET /api/dictionary`
Get dictionary statistics and information.

### `GET /media/videos/<filename>`
Serve video files for sign demonstrations.

## Usage Examples

### Basic Text Processing
1. Enter text: "Warm up before lifting weights"
2. Adjust options (semantic threshold, stemming, etc.)
3. Click "Find Signs"
4. View results with videos, confidence scores, and statistics

### System Evaluation
1. Click "Run Evaluation" button
2. System tests with default fitness coaching phrases
3. Returns overall coverage statistics

### Advanced Options
- **Remove Stop Words:** Filters out common words (the, and, is)
- **Enable Stemming:** Reduces words to root forms (running → run)  
- **Semantic Matching:** AI-powered contextual matching
- **Semantic Threshold:** Similarity threshold (0.3-0.9)

## File Structure

```
capstone/
├── app.py                 # Flask web application
├── templates/
│   ├── base.html         # Base template with navigation
│   ├── index.html        # Main interface
│   └── about.html        # System information
├── static/               # Static files (auto-created)
├── media/videos/         # Sign demonstration videos
├── data/
│   ├── gloss/
│   │   └── auslan_dictionary.json
│   ├── synonyms/
│   │   └── synonym_mapping.json
│   └── target_words.json
└── src/                  # Core processing modules
```

## Development Notes

### Adding New Signs
1. Add entry to `data/gloss/auslan_dictionary.json`
2. Add synonyms to `data/synonyms/synonym_mapping.json`  
3. Place video file in `media/videos/`
4. Update `data/target_words.json` if needed

### Extending the Web Interface
- Templates use Bootstrap 5 and jQuery
- All processing happens via AJAX API calls
- Video integration ready for MP4 files
- Responsive design works on mobile devices

### Performance Considerations
- Semantic model loads once on startup
- Dictionary and embeddings cached in memory
- Videos served directly via Flask (development only)
- Consider CDN or separate video server for production

## Browser Support
- Chrome/Edge (recommended)
- Firefox  
- Safari
- Requires JavaScript enabled for full functionality