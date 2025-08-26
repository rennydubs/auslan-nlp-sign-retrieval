"""
Matching engine for Auslan sign retrieval system.
Implements exact matching, synonym-based matching, and semantic similarity matching.
"""

import json
import os
from typing import List, Dict, Any, Optional
from collections import defaultdict
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    print("Warning: sentence-transformers not available. Semantic matching disabled.")

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between 2D arrays a (m x d) and b (n x d).
    Returns (m x n) similarity matrix.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    # Avoid division by zero
    a_norm[a_norm == 0] = 1e-12
    b_norm[b_norm == 0] = 1e-12
    return (a @ b.T) / (a_norm @ b_norm.T)

class SignMatcher:
    """
    Main matching engine that combines multiple matching strategies.
    """
    
    def __init__(self, gloss_dict_path: str, target_words_path: str = None):
        """
        Initialize the matcher with dictionary and optional target words.
        
        Args:
            gloss_dict_path (str): Path to the GLOSS dictionary JSON
            target_words_path (str): Optional path to target words JSON
        """
        self.gloss_dict = self._load_gloss_dictionary(gloss_dict_path)
        self.target_words = self._load_target_words(target_words_path) if target_words_path else None
        
        # Initialize synonym mapping from the gloss dictionary and external mapping
        self.synonym_map = self._build_synonym_map()

        # Build phrase map (multi-word keys) for n-gram matching
        self.phrase_to_main: Dict[str, str] = {}
        for key, main in self.synonym_map.items():
            if ' ' in key:
                self.phrase_to_main[key] = main
        for word in self.gloss_dict.keys():
            if ' ' in word:
                self.phrase_to_main[word] = word
        # Maximum phrase length in words
        self.max_phrase_len = 1
        if self.phrase_to_main:
            self.max_phrase_len = max(len(p.split()) for p in self.phrase_to_main.keys())
        
        # Initialize semantic model if available
        self.semantic_model = None
        self.gloss_embeddings = None
        if SEMANTIC_AVAILABLE:
            self._initialize_semantic_model()
    
    def _load_gloss_dictionary(self, path: str) -> Dict[str, Any]:
        """Load the GLOSS dictionary from JSON file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading GLOSS dictionary: {e}")
            return {}
    
    def _load_target_words(self, path: str) -> Dict[str, Any]:
        """Load target words from JSON file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading target words: {e}")
            return None
    
    def _load_external_synonyms(self) -> Dict[str, List[str]]:
        """Load external synonym mapping from data/synonyms/synonym_mapping.json if present."""
        path = os.path.join('data', 'synonyms', 'synonym_mapping.json')
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Normalize keys and values to lowercase
                return {k.lower(): [s.lower() for s in v] for k, v in data.items()}
        except Exception:
            return {}

    def _build_synonym_map(self) -> Dict[str, str]:
        """
        Build a mapping from synonyms to main dictionary keys.
        
        Returns:
            Dict[str, str]: Mapping from synonym to main word
        """
        synonym_map: Dict[str, str] = {}
        
        for word, data in self.gloss_dict.items():
            # Map the word to itself
            synonym_map[word.lower()] = word
            
            # Map all synonyms to the main word
            if 'synonyms' in data:
                for synonym in data['synonyms']:
                    synonym_map[synonym.lower()] = word

        # Merge in external synonyms if available (only for words present in dictionary)
        external = self._load_external_synonyms()
        for main_word, syns in external.items():
            if main_word in self.gloss_dict:
                for syn in syns:
                    synonym_map[syn] = main_word
        
        return synonym_map
    
    def _initialize_semantic_model(self):
        """Initialize the semantic similarity model and precompute embeddings."""
        try:
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Precompute embeddings for all dictionary words and their synonyms
            all_terms = []
            self.term_to_word = {}
            
            for word, data in self.gloss_dict.items():
                all_terms.append(word)
                self.term_to_word[word] = word
                
                if 'synonyms' in data:
                    for synonym in data['synonyms']:
                        all_terms.append(synonym)
                        self.term_to_word[synonym] = word
            
            # Compute embeddings
            self.gloss_embeddings = self.semantic_model.encode(all_terms)
            self.embedding_terms = all_terms
            
        except Exception as e:
            print(f"Error initializing semantic model: {e}")
            self.semantic_model = None
    
    def exact_match(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Perform exact string matching against dictionary keys.
        
        Args:
            token (str): Input token to match
            
        Returns:
            Optional[Dict[str, Any]]: Matched sign data or None
        """
        token_lower = token.lower()
        
        # Direct match
        if token_lower in self.gloss_dict:
            return {
                'word': token_lower,
                'match_type': 'exact',
                'confidence': 1.0,
                'sign_data': self.gloss_dict[token_lower]
            }
        
        return None
    
    def synonym_match(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Perform synonym-based matching using the synonym map.
        
        Args:
            token (str): Input token to match
            
        Returns:
            Optional[Dict[str, Any]]: Matched sign data or None
        """
        token_lower = token.lower()
        
        if token_lower in self.synonym_map:
            main_word = self.synonym_map[token_lower]
            return {
                'word': main_word,
                'match_type': 'synonym',
                'matched_synonym': token_lower,
                'confidence': 0.9,
                'sign_data': self.gloss_dict[main_word]
            }
        
        return None
    
    def semantic_match(self, token: str, threshold: float = 0.6) -> Optional[Dict[str, Any]]:
        """
        Perform semantic similarity matching using embeddings.
        
        Args:
            token (str): Input token to match
            threshold (float): Minimum similarity threshold
            
        Returns:
            Optional[Dict[str, Any]]: Best matched sign data or None
        """
        if not self.semantic_model or self.gloss_embeddings is None:
            return None
        
        try:
            # Encode the input token
            token_embedding = self.semantic_model.encode([token])
            
            # Compute similarities using lightweight NumPy-based cosine
            similarities = _cosine_similarity(np.asarray(token_embedding), np.asarray(self.gloss_embeddings))[0]
            
            # Find best match above threshold
            best_idx = np.argmax(similarities)
            best_similarity = similarities[best_idx]
            
            if best_similarity >= threshold:
                matched_term = self.embedding_terms[best_idx]
                main_word = self.term_to_word[matched_term]
                
                return {
                    'word': main_word,
                    'match_type': 'semantic',
                    'matched_term': matched_term,
                    'confidence': float(best_similarity),
                    'sign_data': self.gloss_dict[main_word]
                }
        
        except Exception as e:
            print(f"Error in semantic matching: {e}")
        
        return None
    
    def match_token(self, token: str, use_semantic: bool = True, threshold: float = 0.6) -> Optional[Dict[str, Any]]:
        """
        Match a single token using all available matching strategies.
        
        Args:
            token (str): Input token to match
            use_semantic (bool): Whether to use semantic matching
            
        Returns:
            Optional[Dict[str, Any]]: Best matched sign data or None
        """
        # Try exact match first (highest confidence)
        result = self.exact_match(token)
        if result:
            return result
        
        # Try synonym match
        result = self.synonym_match(token)
        if result:
            return result
        
        # Try semantic match if available and enabled
        if use_semantic and SEMANTIC_AVAILABLE:
            result = self.semantic_match(token, threshold=threshold)
            if result:
                return result
        
        return None
    
    def match_tokens(self, tokens: List[str], use_semantic: bool = True, threshold: float = 0.6) -> List[Dict[str, Any]]:
        """
        Match multiple tokens and return results.
        
        Args:
            tokens (List[str]): List of tokens to match
            use_semantic (bool): Whether to use semantic matching
            
        Returns:
            List[Dict[str, Any]]: List of match results
        """
        results: List[Dict[str, Any]] = []
        
        i = 0
        n = len(tokens)
        while i < n:
            matched_any_phrase = False
            # Try longest-first phrase match up to max_phrase_len
            for window in range(min(self.max_phrase_len, n - i), 1, -1):
                phrase = ' '.join(tokens[i:i+window]).lower()
                if phrase in self.phrase_to_main:
                    main_word = self.phrase_to_main[phrase]
                    # Build a synthetic match result for the phrase
                    results.append({
                        'word': main_word,
                        'match_type': 'synonym' if main_word != phrase else 'exact',
                        'matched_synonym': phrase if main_word != phrase else None,
                        'confidence': 0.95 if main_word != phrase else 1.0,
                        'sign_data': self.gloss_dict.get(main_word)
                    })
                    i += window
                    matched_any_phrase = True
                    break
            if matched_any_phrase:
                continue
            
            token = tokens[i]
            match_result = self.match_token(token, use_semantic, threshold)
            if match_result:
                results.append(match_result)
            else:
                results.append({
                    'word': token,
                    'match_type': 'no_match',
                    'confidence': 0.0,
                    'sign_data': None
                })
            i += 1
        
        return results
    
    def get_coverage_stats(self, tokens: List[str], use_semantic: bool = True, threshold: float = 0.6) -> Dict[str, Any]:
        """
        Calculate coverage statistics for a list of tokens.
        
        Args:
            tokens (List[str]): List of tokens to analyze
            
        Returns:
            Dict[str, Any]: Coverage statistics
        """
        results = self.match_tokens(tokens, use_semantic=use_semantic, threshold=threshold)
        
        total_tokens = len(tokens)
        matched_tokens = sum(1 for r in results if r['match_type'] != 'no_match')
        exact_matches = sum(1 for r in results if r['match_type'] == 'exact')
        synonym_matches = sum(1 for r in results if r['match_type'] == 'synonym')
        semantic_matches = sum(1 for r in results if r['match_type'] == 'semantic')
        
        return {
            'total_tokens': total_tokens,
            'matched_tokens': matched_tokens,
            'unmatched_tokens': total_tokens - matched_tokens,
            'coverage_rate': matched_tokens / total_tokens if total_tokens > 0 else 0,
            'exact_matches': exact_matches,
            'synonym_matches': synonym_matches,
            'semantic_matches': semantic_matches,
            'match_breakdown': {
                'exact': exact_matches / total_tokens if total_tokens > 0 else 0,
                'synonym': synonym_matches / total_tokens if total_tokens > 0 else 0,
                'semantic': semantic_matches / total_tokens if total_tokens > 0 else 0
            }
        }

# Test the matcher
if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Initialize matcher
    gloss_path = "data/gloss/auslan_dictionary.json"
    target_words_path = "data/target_words.json"
    
    matcher = SignMatcher(gloss_path, target_words_path)
    
    # Test tokens
    test_tokens = ["hello", "happy", "assist", "large", "home", "unknown_word"]
    
    print("Testing Sign Matcher:")
    print("=" * 50)
    
    for token in test_tokens:
        result = matcher.match_token(token)
        if result:
            print(f"Token: '{token}' -> {result['match_type']} match")
            print(f"  Matched word: {result['word']}")
            print(f"  Confidence: {result['confidence']:.2f}")
            if result['sign_data']:
                print(f"  GLOSS: {result['sign_data'].get('gloss', 'N/A')}")
        else:
            print(f"Token: '{token}' -> No match found")
        print("-" * 30)
    
    # Coverage stats
    stats = matcher.get_coverage_stats(test_tokens)
    print(f"\nCoverage Statistics:")
    print(f"Total tokens: {stats['total_tokens']}")
    print(f"Matched: {stats['matched_tokens']}")
    print(f"Coverage rate: {stats['coverage_rate']:.2%}")