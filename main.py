# -*- coding: utf-8 -*-
"""
Main application for Auslan Sign Retrieval and Display System.
Integrates text preprocessing, matching, and sign display functionality.
"""

import os
import sys
import json
from typing import List, Dict, Any
import re
from src.preprocessing import TextPreprocessor
from src.matcher import SignMatcher
from src.phrase_matcher import IntelligentPhraseMatcher
from src.nlp_features import EnhancedNLPProcessor

class AuslanSignSystem:
    """
    Main system class that coordinates all components.
    """
    
    def __init__(self):
        """Initialize the system with all components."""
        self.preprocessor = TextPreprocessor()

        # Load data paths
        self.gloss_dict_path = "data/gloss/auslan_dictionary.json"
        self.target_words_path = "data/target_words.json"

        # Initialize matchers with error handling
        try:
            self.matcher = SignMatcher(self.gloss_dict_path, self.target_words_path)
            print("Basic matching engine loaded")
        except Exception as e:
            print(f"WARNING: Basic matcher failed to load: {e}")
            self.matcher = None

        try:
            self.phrase_matcher = IntelligentPhraseMatcher(self.gloss_dict_path, self.target_words_path)
            print("OK: Intelligent phrase matcher loaded")
        except Exception as e:
            print(f"WARNING:  Phrase matcher failed to load: {e}")
            self.phrase_matcher = None

        try:
            self.nlp_processor = EnhancedNLPProcessor()
            print("OK: Enhanced NLP processor loaded")
        except Exception as e:
            print(f"WARNING:  NLP processor failed to load: {e}")
            self.nlp_processor = None

        print(">> Auslan AI v2.0 initialized successfully!")

        if not hasattr(self.matcher, 'semantic_model') or self.matcher.semantic_model is None:
            print("WARNING:  Note: Semantic matching not available. Install sentence-transformers for full functionality.")
        else:
            print("OK: Semantic similarity (DistilBERT) available")
    
    def process_input(self, text: str, remove_stops: bool = False, use_semantic: bool = True,
                     semantic_threshold: float = 0.5, use_stemming: bool = False,
                     use_intelligent_matching: bool = True) -> Dict[str, Any]:
        """
        Process user input text with enhanced NLP analysis and intelligent phrase matching.

        Args:
            text (str): User input text
            remove_stops (bool): Whether to remove stop words
            use_semantic (bool): Whether to use semantic matching
            semantic_threshold (float): Semantic similarity threshold
            use_stemming (bool): Whether to use stemming
            use_intelligent_matching (bool): Whether to use intelligent phrase matching

        Returns:
            Dict[str, Any]: Enhanced processing results with NLP analysis
        """
        # Perform NLP analysis first (with fallback)
        try:
            nlp_analysis = self.nlp_processor.analyze_text(text) if self.nlp_processor else None
        except Exception as e:
            print(f"WARNING:  NLP analysis failed: {e}")
            nlp_analysis = None

        # Try intelligent matching first, fallback to basic matching
        if use_intelligent_matching and self.phrase_matcher:
            try:
                # Use intelligent phrase matching
                phrase_match = self.phrase_matcher.match_phrase_intelligently(
                    text, use_semantic=use_semantic, threshold=semantic_threshold
                )
                successful_matches = phrase_match.matched_signs
                failed_matches = []

                # Calculate coverage
                total_words = len(text.split())
                signs_found = len(successful_matches)
                coverage_rate = signs_found / total_words if total_words > 0 else 0

                # Create enhanced coverage stats
                coverage_stats = {
                'coverage_rate': coverage_rate,
                'exact_matches': sum(1 for m in successful_matches if m.get('match_type') == 'exact'),
                'synonym_matches': sum(1 for m in successful_matches if m.get('match_type') == 'synonym'),
                'semantic_matches': sum(1 for m in successful_matches if m.get('match_type') == 'semantic'),
                'unmatched_tokens': total_words - signs_found,
                'match_breakdown': {
                    'exact': (sum(1 for m in successful_matches if m.get('match_type') == 'exact') / max(total_words, 1)),
                    'synonym': (sum(1 for m in successful_matches if m.get('match_type') == 'synonym') / max(total_words, 1)),
                    'semantic': (sum(1 for m in successful_matches if m.get('match_type') == 'semantic') / max(total_words, 1))
                }
                }

                return {
                    'original_text': text,
                    'processed_tokens': text.split(),  # Simple tokenization for display
                    'total_tokens': total_words,
                    'successful_matches': successful_matches,
                    'failed_matches': failed_matches,
                    'coverage_stats': coverage_stats,
                    'signs_found': signs_found,
                    # Enhanced NLP results
                    'nlp_analysis': {
                        'sentiment': nlp_analysis.sentiment_label if nlp_analysis else 'neutral',
                        'sentiment_score': nlp_analysis.sentiment_score if nlp_analysis else 0.0,
                        'emotion': nlp_analysis.emotion if nlp_analysis else 'neutral',
                        'intent': nlp_analysis.intent if nlp_analysis else 'statement',
                        'entities': nlp_analysis.entities if nlp_analysis else [],
                        'key_phrases': nlp_analysis.key_phrases if nlp_analysis else [],
                        'formality': nlp_analysis.formality_level if nlp_analysis else 'neutral',
                        'complexity': nlp_analysis.complexity_score if nlp_analysis else 0.5,
                        'readability': nlp_analysis.readability_score if nlp_analysis else 0.5
                    },
                    'phrase_analysis': {
                        'phrase_type': phrase_match.phrase_type,
                        'overall_confidence': phrase_match.confidence,
                        'grammar_structure': phrase_match.grammar_structure
                    }
                }
            except Exception as e:
                print(f"WARNING:  Intelligent phrase matching failed: {e}")
                use_intelligent_matching = False

        # Fallback to basic matching if intelligent matching failed or disabled
        if not use_intelligent_matching or not self.phrase_matcher:
            # Use original token-based matching
            if self.matcher:
                tokens = self.preprocessor.preprocess(text, remove_stops=remove_stops, use_stemming=use_stemming)
                match_results = self.matcher.match_tokens(tokens, use_semantic=use_semantic, threshold=semantic_threshold)
                coverage_stats = self.matcher.get_coverage_stats(tokens, use_semantic=use_semantic, threshold=semantic_threshold)

                successful_matches = [result for result in match_results if result['match_type'] != 'no_match']
                failed_matches = [result for result in match_results if result['match_type'] == 'no_match']
            else:
                # Even basic matcher failed
                tokens = text.split()
                successful_matches = []
                failed_matches = []
                coverage_stats = {'coverage_rate': 0, 'exact_matches': 0, 'synonym_matches': 0, 'semantic_matches': 0, 'unmatched_tokens': len(tokens), 'match_breakdown': {'exact': 0, 'synonym': 0, 'semantic': 0}}

            return {
                'original_text': text,
                'processed_tokens': tokens,
                'total_tokens': len(tokens),
                'successful_matches': successful_matches,
                'failed_matches': failed_matches,
                'coverage_stats': coverage_stats,
                'signs_found': len(successful_matches),
                'nlp_analysis': {
                    'sentiment': nlp_analysis.sentiment_label if nlp_analysis else 'neutral',
                    'sentiment_score': nlp_analysis.sentiment_score if nlp_analysis else 0.0,
                    'emotion': nlp_analysis.emotion if nlp_analysis else 'neutral',
                    'intent': nlp_analysis.intent if nlp_analysis else 'statement',
                    'entities': nlp_analysis.entities if nlp_analysis else [],
                    'key_phrases': nlp_analysis.key_phrases if nlp_analysis else [],
                    'formality': nlp_analysis.formality_level if nlp_analysis else 'neutral',
                    'complexity': nlp_analysis.complexity_score if nlp_analysis else 0.5,
                    'readability': nlp_analysis.readability_score if nlp_analysis else 0.5
                }
            }
    
    def display_results(self, results: Dict[str, Any]) -> None:
        """
        Display processing results in a user-friendly format.
        
        Args:
            results (Dict[str, Any]): Results from process_input
        """
        print("\n" + "="*60)
        print(f"INPUT: \"{results['original_text']}\"")
        print("="*60)
        
        print(f"Processed tokens: {results['processed_tokens']}")
        print(f"Total tokens: {results['total_tokens']}")
        print(f"Signs found: {results['signs_found']}")
        print(f"Coverage: {results['coverage_stats']['coverage_rate']:.1%}")

        # Display NLP Analysis
        if 'nlp_analysis' in results:
            nlp = results['nlp_analysis']
            print(f"\nAI NLP ANALYSIS:")
            print(f"   Sentiment: {nlp['sentiment']} ({nlp['sentiment_score']:.2f})")
            print(f"   Emotion: {nlp['emotion']}")
            print(f"   Intent: {nlp['intent']}")
            print(f"   Formality: {nlp['formality']}")

            if nlp['entities']:
                entity_strings = [f"{e['text']} ({e['label']})" for e in nlp['entities']]
                print(f"   Entities: {', '.join(entity_strings)}")

            if nlp['key_phrases']:
                print(f"   Key phrases: {', '.join(nlp['key_phrases'][:3])}")

        # Display Phrase Analysis if available
        if 'phrase_analysis' in results:
            phrase = results['phrase_analysis']
            print(f"\nPHRASE PHRASE ANALYSIS:")
            print(f"   Type: {phrase['phrase_type']}")
            print(f"   Confidence: {phrase['overall_confidence']:.1%}")
            print(f"   Grammar: {phrase['grammar_structure']}")
        
        if results['successful_matches']:
            print("\nMATCHED SIGNS:")
            print("-" * 40)
            
            for i, match in enumerate(results['successful_matches'], 1):
                sign_data = match['sign_data']
                print(f"{i}. {match['word'].upper()}")
                print(f"   GLOSS: {sign_data.get('gloss', 'N/A')}")
                print(f"   Match type: {match['match_type']} ({match['confidence']:.1%} confidence)")
                print(f"   Description: {sign_data.get('description', 'No description available')}")
                
                # Display media information
                if 'video_url' in sign_data:
                    video_path = sign_data['video_url']
                    if video_path.startswith('media/'):
                        print(f"   Video: {video_path}")
                    else:
                        print(f"   Video: {video_path} (external link)")
                
                if 'image_url' in sign_data:
                    print(f"   Image: {sign_data['image_url']}")
                
                print(f"   Category: {sign_data.get('category', 'N/A')}")
                
                if 'synonyms' in sign_data:
                    print(f"   Synonyms: {', '.join(sign_data['synonyms'])}")
                
                print()
        
        if results['failed_matches']:
            print("UNMATCHED WORDS:")
            print("-" * 40)
            unmatched_words = [match['word'] for match in results['failed_matches']]
            print(f"   {', '.join(unmatched_words)}")
            print("   Consider adding these to the dictionary or using different words.")
        
        # Display detailed statistics
        stats = results['coverage_stats']
        print(f"\nSTATISTICS:")
        print("-" * 40)
        print(f"   Exact matches: {stats['exact_matches']} ({stats['match_breakdown']['exact']:.1%})")
        print(f"   Synonym matches: {stats['synonym_matches']} ({stats['match_breakdown']['synonym']:.1%})")
        print(f"   Semantic matches: {stats['semantic_matches']} ({stats['match_breakdown']['semantic']:.1%})")
        print(f"   Unmatched: {stats['unmatched_tokens']} ({(1-stats['coverage_rate']):.1%})")
    
    def interactive_mode(self):
        """Run the system in interactive mode for user testing."""
        print("\nWelcome to the Auslan Sign Retrieval System!")
        print("Enter text to find corresponding Auslan signs.")
        print("Commands: 'quit' to exit")
        print("Options: Add '--no-stops' to remove stop words, '--no-semantic' to disable semantic matching, '--stem' to enable stemming, '--thresh=0.6' to set semantic threshold")
        print("-" * 60)
        
        while True:
            try:
                user_input = input("\n> Enter text: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                # Parse options
                remove_stops = '--no-stops' in user_input
                use_semantic = '--no-semantic' not in user_input
                use_stemming = '--stem' in user_input
                semantic_threshold = 0.6
                m = re.search(r'--thresh=([0-9]*\.?[0-9]+)', user_input)
                if m:
                    try:
                        semantic_threshold = float(m.group(1))
                    except ValueError:
                        pass
                
                # Clean the input of options
                text = user_input.replace('--no-stops', '').replace('--no-semantic', '').replace('--stem', '')
                text = re.sub(r'--thresh=([0-9]*\.?[0-9]+)', '', text).strip()
                
                if not text:
                    continue
                
                # Process the input
                results = self.process_input(text, remove_stops=remove_stops, use_semantic=use_semantic, semantic_threshold=semantic_threshold, use_stemming=use_stemming)
                
                # Display results
                self.display_results(results)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def batch_evaluation(self, test_texts: List[str], remove_stops: bool = False, use_semantic: bool = True, semantic_threshold: float = 0.6, use_stemming: bool = False) -> Dict[str, Any]:
        """
        Evaluate the system on a batch of test texts.
        
        Args:
            test_texts (List[str]): List of test texts
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        all_results = []
        total_coverage = 0
        
        print("Running batch evaluation...")
        print("=" * 50)
        
        for i, text in enumerate(test_texts, 1):
            results = self.process_input(
                text,
                remove_stops=remove_stops,
                use_semantic=use_semantic,
                semantic_threshold=semantic_threshold,
                use_stemming=use_stemming,
            )
            all_results.append(results)
            total_coverage += results['coverage_stats']['coverage_rate']
            
            print(f"{i}. \"{text}\" -> {results['signs_found']}/{results['total_tokens']} signs "
                  f"({results['coverage_stats']['coverage_rate']:.1%} coverage)")
        
        average_coverage = total_coverage / len(test_texts) if test_texts else 0
        
        return {
            'test_texts': test_texts,
            'individual_results': all_results,
            'average_coverage': average_coverage,
            'total_tests': len(test_texts)
        }

def main():
    """Main function to run the Auslan Sign System."""
    # Initialize system
    system = AuslanSignSystem()
    
    # Check if we have command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--test':
            # Run test evaluation
            test_texts = [
                "Hello, how are you today?",
                "I need help finding the toilet",
                "Let's go buy some food and eat together",
                "My friend lives in a big house",
                "Can you speak more slowly please?",
                "I am happy to see you",
                "Goodbye and have a good day"
            ]
            
            # Parse optional flags after --test
            extra_args = ' '.join(sys.argv[2:]) if len(sys.argv) > 2 else ''
            remove_stops = '--no-stops' in extra_args
            use_semantic = '--no-semantic' not in extra_args
            use_stemming = '--stem' in extra_args
            semantic_threshold = 0.6
            m = re.search(r'--thresh=([0-9]*\.?[0-9]+)', extra_args)
            if m:
                try:
                    semantic_threshold = float(m.group(1))
                except ValueError:
                    pass

            print("Running test evaluation on sample texts...")
            evaluation = system.batch_evaluation(
                test_texts,
                remove_stops=remove_stops,
                use_semantic=use_semantic,
                semantic_threshold=semantic_threshold,
                use_stemming=use_stemming,
            )
            print(f"\nOverall average coverage: {evaluation['average_coverage']:.1%}")
            
        elif sys.argv[1] == '--interactive':
            # Run interactive mode
            system.interactive_mode()
        
        else:
            # Process command line input with optional flags
            raw_input = ' '.join(sys.argv[1:])
            remove_stops = '--no-stops' in raw_input
            use_semantic = '--no-semantic' not in raw_input
            use_stemming = '--stem' in raw_input
            semantic_threshold = 0.6
            m = re.search(r'--thresh=([0-9]*\.?[0-9]+)', raw_input)
            if m:
                try:
                    semantic_threshold = float(m.group(1))
                except ValueError:
                    pass
            text = (raw_input
                    .replace('--no-stops', '')
                    .replace('--no-semantic', '')
                    .replace('--stem', ''))
            text = re.sub(r'--thresh=([0-9]*\.?[0-9]+)', '', text).strip()

            results = system.process_input(
                text,
                remove_stops=remove_stops,
                use_semantic=use_semantic,
                semantic_threshold=semantic_threshold,
                use_stemming=use_stemming,
            )
            system.display_results(results)
    
    else:
        # Default to interactive mode
        system.interactive_mode()

if __name__ == "__main__":
    main()