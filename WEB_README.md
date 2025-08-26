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

### **Sign Processing**
- Enter any text to find corresponding Auslan signs
- Multiple matching strategies (exact, synonym, semantic)

### Basic Text Processing
1. Enter text: "Warm up before lifting weights"
2. Adjust options (semantic threshold, stemming, etc.)
3. Click "Find Signs"
4. View results with videos, confidence scores, and statistics

### System Evaluation
1. Click "Run Evaluation" button
2. System tests with default fitness coaching phrases
3. Returns overall coverage statistics
