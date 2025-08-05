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

## Target Words (16 total)

<<<<<<< HEAD
The project focuses on 16 common words with their synonyms:
- **happy** (glad, joyful, cheerful)
- **help** (assist, aid, support)
- **go** (leave, depart, proceed)
- **come** (arrive, approach, enter)
- **eat** (dine, consume, have a meal)
- **house** (home, residence, dwelling)
- **friend** (buddy, pal, companion)
- **see** (look, watch, view)
- **speak** (talk, say, converse)
- **walk** (stroll, hike, move)
- **big** (large, huge, enormous)
- **buy** (purchase, get, acquire)
- **good** (great, nice, excellent)
- **hello** (hi, hey, greetings)
- **toilet** (bathroom, restroom, washroom)
- **goodbye** (bye, farewell, see you)
=======
The project focuses on 16 common words across semantic categories:

Greetings & Social Interaction: Hello, Good, Friend, Goodbye

Actions/Verbs: Help, Go, Come, See, Speak, Buy, Eat, Walk

Places/Objects: House, Toilet

Descriptive/Adjectives: Happy, Big
>>>>>>> ad8e583bad0a1565806c6e3d01804a6201341922

## Technical Approach

- **Language**: Python
- **Platform**: GitHub
- **NLP Models**:
  - Exact match model
  - Rule-based synonym model
  - Semantic similarity model (DistilBERT embeddings)
