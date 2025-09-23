# NLP-Powered Sign Retrieval System

## 🚀 Project Overview

This system leverages modern transformer models to provide intelligent Auslan sign retrieval with deep natural language understanding. Features include sentiment analysis, emotion detection, intelligent phrase matching, and context-aware sign selection.

### ✨ Key Features

- 🧠 **AI-Powered Analysis**: DistilBERT & RoBERTa transformer models
- 📝 **Intelligent Phrase Matching**: Context-aware sign selection
- 😊 **Emotion & Sentiment Detection**: Advanced psychological analysis
- 🎯 **Intent Recognition**: Understands user goals and needs
- 🌐 **Modern Web Interface**: Responsive design with dark mode
- 🎥 **46 High-Quality Sign Videos**: Comprehensive vocabulary coverage

## 🔬 Research Evolution

**Original Question**: How can natural language processing be effectively applied to retrieve and display appropriate Auslan signs?

## Project Structure

```
capstone/
├── data/
│   ├── gloss/
│   │   ├── auslan_dictionary.json    # Main Auslan dictionary with sign data
│   │   └── initial_gloss_dictionary.csv
│   ├── synonyms/
│   │   └── synonym_mapping.json      # Synonym to primary word mappings
│   └── target_words.json             # Target words with synonyms
├── media/
│   ├── videos/                       # Sign language video files
│   └── images/                       # Sign language image files
├── src/
│   ├── preprocessing.py              # Text preprocessing and cleaning
│   └── matcher.py                    # Sign matching algorithms
├── tests/                            # Test files
├── docs/                             # Documentation
├── main.py                           # Main application entry point
└── README.md
```

## 📚 Vocabulary Coverage (46 Signs)

Expanded from 16 to **46 comprehensive signs** across multiple domains:

### 👋 **Greetings & Social (7 signs)**
`hello`, `goodbye`, `thank`, `please`, `good`, `friend`, `see`

### 🏃‍♂️ **Fitness & Exercise (15 signs)**
`exercise`, `strong`, `muscle`, `weight`, `lift`, `stretch`, `breathe`, `rest`, `warm`, `cool`, `run`, `bike`, `chest`, `arms`, `legs`

### 🍎 **Basic Needs (6 signs)**
`eat`, `drink`, `sleep`, `help`, `food`, `water`

### 🎭 **Emotions (3 signs)**
`happy`, `sad`, `angry`

### 🏃 **Actions (6 signs)**
`go`, `come`, `sit`, `stand`, `walk`, `buy`

### 📍 **Places & Objects (4 signs)**
`house`, `toilet`, `big`, `speak`

### ⏰ **Temporal & Descriptive (5 signs)**
`today`, `tomorrow`, `time`, `many`, `more`

## 🛠️ Technical Architecture

### Core Technologies
- **Language**: Python 3.8+
- **Web Framework**: Flask with responsive Bootstrap 5
- **Platform**: GitHub with automated deployments

### 🤖 AI/ML Models Stack

#### Primary NLP Models
1. **Semantic Similarity**: `all-MiniLM-L6-v2` (SentenceTransformers)
   - Advanced semantic understanding
   - Context-aware matching
   - 384-dimensional embeddings

2. **Sentiment Analysis**: `distilbert-base-uncased-finetuned-sst-2-english`
   - Transformer-based sentiment detection
   - Fine-tuned on Stanford Sentiment Treebank
   - 99.7% accuracy on validation set

3. **Emotion Classification**: `j-hartmann/emotion-english-distilroberta-base`
   - 7-class emotion detection (joy, sadness, anger, fear, surprise, disgust, neutral)
   - RoBERTa-based architecture
   - Fine-tuned on emotion datasets

4. **Named Entity Recognition**: spaCy `en_core_web_sm`
   - Real-time entity extraction
   - Support for temporal, person, location entities
   - Grammar and dependency parsing

#### Matching Strategies
- **Exact Match**: Direct dictionary lookup
- **Synonym Matching**: Comprehensive synonym mapping
- **Semantic Matching**: Transformer-based similarity
- **Intelligent Phrase Matching**: Context and grammar-aware
- **Grammar-Optimized Ordering**: ASL/Auslan structure compliance
