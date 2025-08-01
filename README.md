# Applying Natural Language Processing for Sign Retrieval and Display

A capstone project developing a text-to-Auslan translation tool using NLP techniques.

## Project Overview

This project develops a practical text-to-sign language translation tool using Natural Language Processing (NLP) techniques, focusing on retrieving and displaying appropriate Auslan (Australian Sign Language) signs corresponding to text input.

## Research Question

How can natural language processing be effectively applied to retrieve and display appropriate Auslan signs corresponding to a word entered by a user?

## Project Structure

```
capstone/
├── data/
│   ├── gloss/                 # GLOSS dictionary files
│   ├── synonyms/              # Synonym mapping files
│   └── target_words.json      # Selected target words for development
├── media/
│   ├── videos/                # Sign language video files
│   └── images/                # Sign language image files
├── src/                       # Source code
├── tests/                     # Test files
├── docs/                      # Documentation
└── README.md
```

## Target Words (18 total)

The project focuses on 18 common words across semantic categories:
- **Greetings**: hello, goodbye, thank, please
- **Basic Needs**: eat, drink, sleep, help
- **Actions**: go, come, sit, stand, walk, run
- **Emotions**: happy, sad, angry
- **Family**: mother, father, family
- **Time**: today, tomorrow

## Technical Approach

- **Language**: Python
- **Platform**: GitHub
- **NLP Models**:
  - Exact match model
  - Rule-based synonym model
  - Semantic similarity model (DistilBERT embeddings)

## Week 1 Deliverables (Current)

- ✅ Repository initialization
- ✅ Directory structure setup
- ✅ Target word selection (18 words across 6 categories)
- ✅ Initial GLOSS dictionary structure

## Next Steps

1. Download and organize Auslan sign media files
2. Implement NLP preprocessing pipeline
3. Develop matching algorithms
4. Create evaluation framework

## Author

Oliver Dubois (14269359)  
Supervisor: Thuy Pham