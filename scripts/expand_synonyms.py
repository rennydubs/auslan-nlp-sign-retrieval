"""
WordNet synonym expansion script for the Auslan Sign Retrieval System.

Generates data/synonyms/wordnet_synonyms.json by looking up synonyms for
every word in the Auslan dictionary via NLTK WordNet.

Only synonyms that cannot already be resolved to a dictionary entry via the
existing manual synonym_mapping.json are written to the output file — this
avoids duplicates and lets the manual mapping keep priority.

Usage:
    python scripts/expand_synonyms.py
    python scripts/expand_synonyms.py --dict data/gloss/auslan_dictionary.json
    python scripts/expand_synonyms.py --dry-run   # Print without writing
"""

import argparse
import json
import os
import sys

# Allow running from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

# NLTK downloads handled gracefully
try:
    import nltk
    from nltk.corpus import wordnet as wn
    WORDNET_AVAILABLE = True
except ImportError:
    WORDNET_AVAILABLE = False
    print("ERROR: nltk is not installed. Run: pip install nltk")
    sys.exit(1)


def ensure_wordnet():
    """Download WordNet data if not already present."""
    for resource in ['wordnet', 'omw-1.4']:
        try:
            nltk.data.find(f'corpora/{resource}')
        except LookupError:
            print(f"Downloading NLTK resource: {resource}")
            nltk.download(resource, quiet=True)


# Words that should never be generated as synonyms regardless of WordNet data.
# Blocks obscure/slang/unrelated senses that WordNet surfaces for common words.
_SYNONYM_BLOCKLIST = {
    # drug slang mapped to innocent words
    'adam', 'ecstasy', 'hug drug', 'xtc', 'cristal', 'disco biscuit',
    # medical/anatomical terms too obscure for sign retrieval
    'rhytidectomy', 'rhytidoplasty', 'seminal fluid', 'semen', 'cum', 'ejaculate',
    # highly unusual/archaic words
    'forsooth', 'prithee', 'verily',
    # generic words that would create false positives
    'total', 'amount', 'do', 'make', 'get', 'give', 'put', 'set',
    'fall', 'pass', 'turn', 'run low', 'run short',
    # wrong senses for our words
    'x', 'crack', 'steal', 'corrupt', 'bribe',  # wrong senses of "go" and "buy"
    'gutter', 'sewer', 'crapper',               # unsavoury toilet synonyms
    'seed',                                      # wrong sense of "come"
    'clip',                                      # slang sense of "time"
    'sopor',                                     # medical term for sleep
    'potable', 'drinkable',                      # technical terms for drink
    'utilisation', 'utilization',               # British/technical spellings
    'recitation',                               # wrong sense of exercise
    'musculus', 'muscleman',                    # anatomical/informal terms
    'blazon', 'blazonry', 'munition',           # heraldry/military terms
    'pectus', 'dorsum',                         # anatomical Latin terms
    'goodness', 'felicitous',                   # overly formal/uncommon
    'h2o',                                      # chemical formula not a word
    'aplomb', 'sang-froid', 'assuredness',      # French loanwords
    'weightiness',                              # abstract sense of weight
}


def get_wordnet_synonyms(word: str, max_synsets: int = 3) -> set:
    """
    Return synonym lemmas for a word from its most common synsets.

    Only uses the top `max_synsets` synsets (ranked by frequency) to avoid
    obscure/slang senses. Excludes the word itself, multi-word phrases, and
    entries in the blocklist.

    Args:
        word: The dictionary word to expand.
        max_synsets: Maximum number of synsets (senses) to consider.
    """
    synonyms = set()
    synsets = wn.synsets(word)[:max_synsets]
    for synset in synsets:
        for lemma in synset.lemmas():
            name = lemma.name().lower().replace('_', ' ')
            if (name != word
                    and len(name.split()) == 1  # single words only for precision
                    and name not in _SYNONYM_BLOCKLIST
                    and not name[0].isupper()):  # skip proper nouns
                synonyms.add(name)
    return synonyms


def get_wordnet_hypernyms(word: str) -> set:
    """
    Return one-level-up hypernyms (parent concepts) for a word.
    Only uses the first (most common) noun synset.
    """
    hypernyms = set()
    synsets = wn.synsets(word, pos=wn.NOUN)
    if not synsets:
        return hypernyms
    for hypernym in synsets[0].hypernyms():
        for lemma in hypernym.lemmas():
            name = lemma.name().lower().replace('_', ' ')
            if (name != word
                    and len(name.split()) == 1
                    and name not in _SYNONYM_BLOCKLIST):
                hypernyms.add(name)
    return hypernyms


def load_json(path: str) -> dict:
    """Load JSON file or return empty dict if not found."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def expand_synonyms(dict_path: str, manual_mapping_path: str,
                    include_hypernyms: bool = False,
                    dry_run: bool = False) -> dict:
    """
    Generate a {synonym -> primary_word} mapping using WordNet.

    Filters:
    - Synonym must NOT already be a primary dictionary key
    - Synonym must NOT already be covered by the manual synonym mapping
    - Synonym must resolve to a valid dictionary entry

    Args:
        dict_path: Path to auslan_dictionary.json
        manual_mapping_path: Path to synonym_mapping.json (manual overrides)
        include_hypernyms: Whether to include hypernyms (broader terms)
        dry_run: If True, print results without writing

    Returns:
        The generated synonym mapping dict
    """
    gloss_dict = load_json(dict_path)
    manual_mapping = load_json(manual_mapping_path)

    if not gloss_dict:
        print(f"ERROR: Could not load dictionary from {dict_path}")
        sys.exit(1)

    # Normalise manual mapping keys for de-dup check
    manual_keys = {k.lower() for k in manual_mapping}
    dict_keys = set(gloss_dict.keys())

    generated: dict = {}
    stats = {'words_processed': 0, 'synonyms_added': 0, 'duplicates_skipped': 0}

    print(f"Processing {len(gloss_dict)} dictionary entries...")

    for word in sorted(gloss_dict.keys()):
        stats['words_processed'] += 1

        candidates = get_wordnet_synonyms(word)
        if include_hypernyms:
            candidates |= get_wordnet_hypernyms(word)

        word_additions = []
        for candidate in sorted(candidates):
            # Skip if already a primary key (exact_match handles it)
            if candidate in dict_keys:
                stats['duplicates_skipped'] += 1
                continue
            # Skip if already covered by manual mapping
            if candidate in manual_keys:
                stats['duplicates_skipped'] += 1
                continue
            # Skip if we already generated this synonym pointing elsewhere
            if candidate in generated and generated[candidate] != word:
                stats['duplicates_skipped'] += 1
                continue

            generated[candidate] = word
            word_additions.append(candidate)
            stats['synonyms_added'] += 1

        if word_additions:
            print(f"  {word:15s} <- {', '.join(sorted(word_additions))}")

    return generated, stats


def main():
    parser = argparse.ArgumentParser(
        description='Generate WordNet synonym mapping for the Auslan dictionary',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--dict', default=config.GLOSS_DICT_PATH,
                        help='Path to auslan_dictionary.json')
    parser.add_argument('--manual', default=config.SYNONYM_MAPPING_PATH,
                        help='Path to manual synonym_mapping.json')
    parser.add_argument('--output', default=config.WORDNET_SYNONYMS_PATH,
                        help='Output path for generated wordnet_synonyms.json')
    parser.add_argument('--hypernyms', action='store_true',
                        help='Also include one-level hypernyms (broader concepts)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print results without writing to disk')
    args = parser.parse_args()

    ensure_wordnet()

    print("\n=== Auslan WordNet Synonym Expander ===\n")
    generated, stats = expand_synonyms(
        args.dict, args.manual,
        include_hypernyms=args.hypernyms,
        dry_run=args.dry_run
    )

    print(f"\n--- Summary ---")
    print(f"Words processed:   {stats['words_processed']}")
    print(f"Synonyms generated:{stats['synonyms_added']}")
    print(f"Duplicates skipped:{stats['duplicates_skipped']}")

    if args.dry_run:
        print("\n[DRY RUN] Would write:")
        print(json.dumps(generated, indent=2, ensure_ascii=False))
    else:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(generated, f, indent=2, ensure_ascii=False, sort_keys=True)
        print(f"\nWritten {stats['synonyms_added']} synonyms to: {args.output}")


if __name__ == '__main__':
    main()
