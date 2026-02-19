"""
Matching engine for Auslan sign retrieval system.
Implements exact matching, fuzzy matching, synonym-based matching,
semantic similarity matching, and LLM-assisted fallback.
"""

import hashlib
import json
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer

    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    logger.info("sentence-transformers not available. Semantic matching disabled.")

try:
    from rapidfuzz import fuzz
    from rapidfuzz import process as rfprocess

    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    logger.info("rapidfuzz not available. Fuzzy matching disabled.")


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between 2D arrays a (m x d) and b (n x d)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_norm[a_norm == 0] = 1e-12
    b_norm[b_norm == 0] = 1e-12
    return (a @ b.T) / (a_norm @ b_norm.T)


class SignMatcher:
    """Main matching engine that combines multiple matching strategies.

    Pipeline order: exact → fuzzy → synonym → semantic → LLM fallback
    """

    def __init__(
        self,
        gloss_dict_path: str,
        target_words_path: str = None,
        synonym_mapping_path: str = None,
        wordnet_synonyms_path: str = None,
        semantic_model_name: str = "intfloat/e5-base-v2",
        shared_semantic_model=None,
        embedding_cache_dir: str = None,
        fuzzy_threshold: int = 85,
        fuzzy_confidence: float = 0.85,
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
        llm_processor=None,
    ):
        """
        Initialize the matcher.

        Args:
            gloss_dict_path: Path to the GLOSS dictionary JSON.
            target_words_path: Optional path to target words JSON.
            synonym_mapping_path: Optional path to manual synonym mapping JSON.
            wordnet_synonyms_path: Optional path to WordNet-generated synonyms JSON.
            semantic_model_name: Name of the sentence-transformers model.
            shared_semantic_model: Pre-loaded SentenceTransformer to avoid duplicate loading.
            embedding_cache_dir: Directory for caching precomputed embeddings.
            fuzzy_threshold: Minimum RapidFuzz score (0-100) for fuzzy matches.
            fuzzy_confidence: Confidence score to assign to fuzzy matches.
            query_prefix: Prefix for query embeddings (e.g. "query: " for E5 models).
            passage_prefix: Prefix for passage embeddings (e.g. "passage: " for E5 models).
            llm_processor: Optional LLMProcessor instance for LLM-assisted fallback matching.
        """
        self.gloss_dict = self._load_json(gloss_dict_path, "GLOSS dictionary")
        self.target_words = (
            self._load_json(target_words_path, "target words")
            if target_words_path
            else None
        )
        self._synonym_mapping_path = synonym_mapping_path
        self._wordnet_synonyms_path = wordnet_synonyms_path
        self._fuzzy_threshold = fuzzy_threshold
        self._fuzzy_confidence = fuzzy_confidence
        self._llm_processor = llm_processor

        # Build synonym and phrase maps
        self.synonym_map = self._build_synonym_map()
        self.phrase_to_main: Dict[str, str] = {}
        for key, main in self.synonym_map.items():
            if " " in key:
                self.phrase_to_main[key] = main
        for word in self.gloss_dict.keys():
            if " " in word:
                self.phrase_to_main[word] = word
        self.max_phrase_len = max(
            (len(p.split()) for p in self.phrase_to_main), default=1
        )

        # Pre-compute fuzzy match choices (dictionary keys as a list)
        self._fuzzy_choices = list(self.gloss_dict.keys()) if FUZZY_AVAILABLE else []

        # Semantic model
        self.semantic_model = None
        self.gloss_embeddings = None
        self.embedding_terms = None
        self.term_to_word = None
        self._query_cache: Dict[str, np.ndarray] = {}
        self._embedding_cache_dir = embedding_cache_dir
        self._semantic_model_name = semantic_model_name
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix

        if SEMANTIC_AVAILABLE:
            self._initialize_semantic_model(shared_model=shared_semantic_model)

    @staticmethod
    def _load_json(path: str, label: str) -> Dict:
        """Load a JSON file with error handling."""
        if not path:
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error("Error loading %s from %s: %s", label, path, e)
            return {}

    def _load_external_synonyms(self) -> Dict[str, str]:
        """Load and merge external synonym mappings (synonym -> primary word)."""
        merged: Dict[str, str] = {}

        # Load manual synonym mapping + WordNet synonyms
        for path in [self._synonym_mapping_path, self._wordnet_synonyms_path]:
            if not path:
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for k, v in data.items():
                        merged[k.lower()] = v.lower()
            except FileNotFoundError:
                logger.debug("Synonym file not found: %s", path)
            except Exception as e:
                logger.debug("Error loading synonym file %s: %s", path, e)

        return merged

    def _build_synonym_map(self) -> Dict[str, str]:
        """Build a mapping from synonyms to main dictionary keys."""
        synonym_map: Dict[str, str] = {}

        # Map dictionary entries and their embedded synonyms
        for word, data in self.gloss_dict.items():
            synonym_map[word.lower()] = word
            if "synonyms" in data:
                for synonym in data["synonyms"]:
                    syn_lower = synonym.lower()
                    if syn_lower not in self.gloss_dict:
                        synonym_map[syn_lower] = word

        # Merge external synonyms (manual + WordNet)
        external = self._load_external_synonyms()
        for syn, main_word in external.items():
            if main_word in self.gloss_dict and syn not in self.gloss_dict:
                synonym_map[syn] = main_word

        return synonym_map

    def _initialize_semantic_model(self, shared_model=None):
        """Initialize the semantic similarity model and precompute embeddings."""
        try:
            if shared_model is not None:
                self.semantic_model = shared_model
                logger.info("Using shared semantic model")
            else:
                self.semantic_model = SentenceTransformer(self._semantic_model_name)
                logger.info("Loaded semantic model: %s", self._semantic_model_name)

            # Collect all terms to embed
            all_terms = []
            self.term_to_word = {}
            for word, data in self.gloss_dict.items():
                all_terms.append(word)
                self.term_to_word[word] = word
                if "synonyms" in data:
                    for synonym in data["synonyms"]:
                        all_terms.append(synonym)
                        self.term_to_word[synonym] = word

            self.embedding_terms = all_terms

            # Try loading from cache
            if self._try_load_embedding_cache(all_terms):
                return

            # Compute embeddings with passage prefix for indexed terms
            prefixed_terms = [self._passage_prefix + t for t in all_terms]
            self.gloss_embeddings = self.semantic_model.encode(prefixed_terms)
            logger.info("Computed embeddings for %d terms", len(all_terms))

            # Save to cache
            self._save_embedding_cache(all_terms)

        except Exception as e:
            logger.error("Error initializing semantic model: %s", e)
            self.semantic_model = None

    def _embedding_cache_path(self, all_terms: List[str]) -> Optional[str]:
        """Get the cache file path for the current dictionary state."""
        if not self._embedding_cache_dir:
            return None
        terms_hash = hashlib.md5(json.dumps(sorted(all_terms)).encode()).hexdigest()[
            :12
        ]
        model_slug = self._semantic_model_name.replace("/", "_")
        return os.path.join(
            self._embedding_cache_dir, f"embeddings_{model_slug}_{terms_hash}.npy"
        )

    def _try_load_embedding_cache(self, all_terms: List[str]) -> bool:
        """Try to load embeddings from disk cache."""
        cache_path = self._embedding_cache_path(all_terms)
        if cache_path and os.path.exists(cache_path):
            try:
                self.gloss_embeddings = np.load(cache_path)
                if self.gloss_embeddings.shape[0] == len(all_terms):
                    logger.info("Loaded embeddings from cache: %s", cache_path)
                    return True
                logger.warning("Cache size mismatch, recomputing.")
            except Exception as e:
                logger.warning("Failed to load embedding cache: %s", e)
        return False

    def _save_embedding_cache(self, all_terms: List[str]):
        """Save embeddings to disk cache."""
        cache_path = self._embedding_cache_path(all_terms)
        if cache_path and self.gloss_embeddings is not None:
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                np.save(cache_path, self.gloss_embeddings)
                logger.info("Saved embeddings cache: %s", cache_path)
            except Exception as e:
                logger.warning("Failed to save embedding cache: %s", e)

    # ------------------------------------------------------------------
    # Matching stages
    # ------------------------------------------------------------------

    def exact_match(self, token: str) -> Optional[Dict[str, Any]]:
        """Stage 1: Exact string matching against dictionary keys."""
        token_lower = token.lower()
        if token_lower in self.gloss_dict:
            return {
                "word": token_lower,
                "match_type": "exact",
                "confidence": 1.0,
                "sign_data": self.gloss_dict[token_lower],
            }
        return None

    def fuzzy_match(
        self, token: str, threshold: int = None
    ) -> Optional[Dict[str, Any]]:
        """Stage 2: Fuzzy string matching for typo tolerance using RapidFuzz."""
        if not FUZZY_AVAILABLE or not self._fuzzy_choices:
            return None

        if threshold is None:
            threshold = self._fuzzy_threshold

        token_lower = token.lower()

        # Use extractOne for the best match above threshold
        result = rfprocess.extractOne(
            token_lower, self._fuzzy_choices, scorer=fuzz.ratio, score_cutoff=threshold
        )

        if result is not None:
            matched_word, score, _ = result
            # Don't return fuzzy match if it's identical to the input (that's exact)
            if matched_word == token_lower:
                return None
            return {
                "word": matched_word,
                "match_type": "fuzzy",
                "matched_input": token_lower,
                "confidence": self._fuzzy_confidence,
                "fuzzy_score": score,
                "sign_data": self.gloss_dict[matched_word],
            }

        return None

    def synonym_match(self, token: str) -> Optional[Dict[str, Any]]:
        """Stage 3: Synonym-based matching using the synonym map."""
        token_lower = token.lower()
        if token_lower in self.synonym_map:
            main_word = self.synonym_map[token_lower]
            # Skip self-mappings (caught by exact_match)
            if main_word == token_lower:
                return None
            return {
                "word": main_word,
                "match_type": "synonym",
                "matched_synonym": token_lower,
                "confidence": 0.9,
                "sign_data": self.gloss_dict[main_word],
            }
        return None

    def semantic_match(
        self, token: str, threshold: float = 0.6
    ) -> Optional[Dict[str, Any]]:
        """Stage 4: Semantic similarity matching using embeddings."""
        if not self.semantic_model or self.gloss_embeddings is None:
            return None

        try:
            # Check query cache
            cache_key = token.lower()
            if cache_key in self._query_cache:
                token_embedding = self._query_cache[cache_key]
            else:
                # Apply query prefix for E5/BGE-style models
                prefixed_query = self._query_prefix + token
                token_embedding = self.semantic_model.encode([prefixed_query])
                self._query_cache[cache_key] = token_embedding

            similarities = _cosine_similarity(
                np.asarray(token_embedding), np.asarray(self.gloss_embeddings)
            )[0]

            best_idx = np.argmax(similarities)
            best_similarity = similarities[best_idx]

            if best_similarity >= threshold:
                matched_term = self.embedding_terms[best_idx]
                main_word = self.term_to_word[matched_term]
                return {
                    "word": main_word,
                    "match_type": "semantic",
                    "matched_term": matched_term,
                    "confidence": float(best_similarity),
                    "sign_data": self.gloss_dict[main_word],
                }

        except Exception as e:
            logger.error("Error in semantic matching: %s", e)

        return None

    def llm_match(self, token: str, context: str = "") -> Optional[Dict[str, Any]]:
        """Stage 5: LLM-assisted query expansion fallback.

        Asks the LLM to suggest up to 5 dictionary words related to *token*,
        then retries matching (exact + synonym only) against those candidates.
        Falls back to None if LLM is unavailable or no candidates match.

        Args:
            token:   The unmatched token to look up.
            context: Optional surrounding sentence for disambiguation.
        """
        if not self._llm_processor or not getattr(
            self._llm_processor, "available", False
        ):
            return None

        try:
            dict_words = list(self.gloss_dict.keys())
            candidates = self._llm_processor.expand_query(token, dict_words, n=5)

            for candidate in candidates:
                # Try exact match on each LLM candidate
                result = self.exact_match(candidate)
                if result:
                    return {
                        **result,
                        "match_type": "llm",
                        "confidence": 0.7,
                        "llm_original_token": token,
                        "llm_candidate": candidate,
                    }
                # Try synonym match
                result = self.synonym_match(candidate)
                if result:
                    return {
                        **result,
                        "match_type": "llm",
                        "confidence": 0.65,
                        "llm_original_token": token,
                        "llm_candidate": candidate,
                    }
        except Exception as e:
            logger.warning("LLM match failed for token '%s': %s", token, e)

        return None

    # ------------------------------------------------------------------
    # Combined matching
    # ------------------------------------------------------------------

    def match_token(
        self,
        token: str,
        use_semantic: bool = True,
        threshold: float = 0.6,
        use_llm: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Match a single token using all available strategies.

        Pipeline: exact → fuzzy → synonym → semantic → LLM
        """
        result = self.exact_match(token)
        if result:
            return result

        result = self.fuzzy_match(token)
        if result:
            return result

        result = self.synonym_match(token)
        if result:
            return result

        if use_semantic and SEMANTIC_AVAILABLE:
            result = self.semantic_match(token, threshold=threshold)
            if result:
                return result

        if use_llm:
            result = self.llm_match(token)
            if result:
                return result

        return None

    def match_tokens(
        self,
        tokens: List[str],
        use_semantic: bool = True,
        threshold: float = 0.6,
        use_llm: bool = False,
    ) -> List[Dict[str, Any]]:
        """Match multiple tokens with n-gram phrase detection."""
        results: List[Dict[str, Any]] = []
        i = 0
        n = len(tokens)

        while i < n:
            matched_phrase = False
            for window in range(min(self.max_phrase_len, n - i), 1, -1):
                phrase = " ".join(tokens[i : i + window]).lower()
                if phrase in self.phrase_to_main:
                    main_word = self.phrase_to_main[phrase]
                    results.append(
                        {
                            "word": main_word,
                            "match_type": "synonym" if main_word != phrase else "exact",
                            "matched_synonym": phrase if main_word != phrase else None,
                            "confidence": 0.95 if main_word != phrase else 1.0,
                            "sign_data": self.gloss_dict.get(main_word),
                        }
                    )
                    i += window
                    matched_phrase = True
                    break

            if matched_phrase:
                continue

            match_result = self.match_token(
                tokens[i], use_semantic, threshold, use_llm=use_llm
            )
            if match_result:
                results.append(match_result)
            else:
                results.append(
                    {
                        "word": tokens[i],
                        "match_type": "no_match",
                        "confidence": 0.0,
                        "sign_data": None,
                    }
                )
            i += 1

        return results

    def get_coverage_stats(
        self,
        tokens: List[str],
        use_semantic: bool = True,
        threshold: float = 0.6,
        use_llm: bool = False,
    ) -> Dict[str, Any]:
        """Calculate coverage statistics for a list of tokens."""
        results = self.match_tokens(
            tokens, use_semantic=use_semantic, threshold=threshold, use_llm=use_llm
        )
        total = len(tokens)
        matched = sum(1 for r in results if r["match_type"] != "no_match")
        exact = sum(1 for r in results if r["match_type"] == "exact")
        fuzzy = sum(1 for r in results if r["match_type"] == "fuzzy")
        synonym = sum(1 for r in results if r["match_type"] == "synonym")
        semantic = sum(1 for r in results if r["match_type"] == "semantic")
        llm = sum(1 for r in results if r["match_type"] == "llm")

        return {
            "total_tokens": total,
            "matched_tokens": matched,
            "unmatched_tokens": total - matched,
            "coverage_rate": matched / total if total > 0 else 0,
            "exact_matches": exact,
            "fuzzy_matches": fuzzy,
            "synonym_matches": synonym,
            "semantic_matches": semantic,
            "llm_matches": llm,
            "match_breakdown": {
                "exact": exact / total if total > 0 else 0,
                "fuzzy": fuzzy / total if total > 0 else 0,
                "synonym": synonym / total if total > 0 else 0,
                "semantic": semantic / total if total > 0 else 0,
                "llm": llm / total if total > 0 else 0,
            },
        }


if __name__ == "__main__":
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    logging.basicConfig(level=logging.INFO)

    matcher = SignMatcher(
        "data/gloss/auslan_dictionary.json",
        "data/target_words.json",
        synonym_mapping_path="data/synonyms/synonym_mapping.json",
        wordnet_synonyms_path="data/synonyms/wordnet_synonyms.json",
    )

    test_tokens = [
        "hello",
        "happy",
        "assist",
        "large",
        "home",
        "unknown_word",
        "halp",
        "exercize",
        "runing",
    ]  # typos for fuzzy testing
    print("Testing Sign Matcher:")
    print("=" * 50)

    for token in test_tokens:
        result = matcher.match_token(token)
        if result:
            print(f"Token: '{token}' -> {result['match_type']} match")
            print(f"  Matched word: {result['word']}")
            print(f"  Confidence: {result['confidence']:.2f}")
            if result.get("fuzzy_score"):
                print(f"  Fuzzy score: {result['fuzzy_score']:.0f}")
            if result["sign_data"]:
                print(f"  GLOSS: {result['sign_data'].get('gloss', 'N/A')}")
        else:
            print(f"Token: '{token}' -> No match found")
        print("-" * 30)

    stats = matcher.get_coverage_stats(test_tokens)
    print(f"\nCoverage: {stats['coverage_rate']:.2%}")
    print(
        f"  Exact: {stats['exact_matches']}, Fuzzy: {stats['fuzzy_matches']}, "
        f"Synonym: {stats['synonym_matches']}, Semantic: {stats['semantic_matches']}"
    )
