# -*- coding: utf-8 -*-
"""
Main application for Auslan Sign Retrieval and Display System.
Integrates text preprocessing, matching, and sign display functionality.
"""

import logging
import re
import sys
from typing import Any, Dict, List

import config
from src.llm_processor import LLMProcessor
from src.matcher import SignMatcher
from src.nlp_features import EnhancedNLPProcessor
from src.phrase_matcher import IntelligentPhraseMatcher
from src.preprocessing import TextPreprocessor

logger = logging.getLogger(__name__)


def _serialize_nlp_analysis(nlp_analysis) -> Dict[str, Any]:
    """Serialize an NLPAnalysis object to a dict for API responses."""
    if nlp_analysis is None:
        return {
            "sentiment": "neutral",
            "sentiment_score": 0.0,
            "emotion": "neutral",
            "intent": "statement",
            "entities": [],
            "key_phrases": [],
            "formality": "neutral",
            "complexity": 0.5,
            "readability": 0.5,
        }
    return {
        "sentiment": nlp_analysis.sentiment_label,
        "sentiment_score": nlp_analysis.sentiment_score,
        "emotion": nlp_analysis.emotion,
        "intent": nlp_analysis.intent,
        "entities": nlp_analysis.entities,
        "key_phrases": nlp_analysis.key_phrases,
        "formality": nlp_analysis.formality_level,
        "complexity": nlp_analysis.complexity_score,
        "readability": nlp_analysis.readability_score,
    }


class AuslanSignSystem:
    """Main system class that coordinates all components."""

    def __init__(self):
        """Initialize the system with all components, sharing models where possible."""
        # Initialize LLM processor (connects to Ollama if available; graceful no-op otherwise)
        try:
            self.llm_processor = (
                LLMProcessor(
                    model=config.LLM_MODEL,
                    host=config.LLM_HOST,
                    timeout=config.LLM_TIMEOUT,
                )
                if config.LLM_ENABLED
                else None
            )
            if self.llm_processor and self.llm_processor.available:
                logger.info(
                    "LLM processor ready: %s @ %s", config.LLM_MODEL, config.LLM_HOST
                )
            else:
                logger.info(
                    "LLM processor unavailable (Ollama not running or LLM_ENABLED=False)."
                )
        except Exception as e:
            logger.warning("LLM processor failed to initialize: %s", e)
            self.llm_processor = None

        # Initialize matcher first (it owns the semantic model)
        try:
            self.matcher = SignMatcher(
                config.GLOSS_DICT_PATH,
                config.TARGET_WORDS_PATH,
                synonym_mapping_path=config.SYNONYM_MAPPING_PATH,
                wordnet_synonyms_path=config.WORDNET_SYNONYMS_PATH,
                semantic_model_name=config.SEMANTIC_MODEL_NAME,
                embedding_cache_dir=config.EMBEDDING_CACHE_DIR,
                fuzzy_threshold=config.FUZZY_MATCH_THRESHOLD,
                fuzzy_confidence=config.FUZZY_CONFIDENCE,
                query_prefix=config.SEMANTIC_QUERY_PREFIX,
                passage_prefix=config.SEMANTIC_PASSAGE_PREFIX,
                llm_processor=self.llm_processor,
            )
            logger.info("Matching engine loaded")
        except Exception as e:
            logger.warning("Basic matcher failed to load: %s", e)
            self.matcher = None

        # Initialize NLP processor next (owns the spaCy model)
        shared_semantic = (
            getattr(self.matcher, "semantic_model", None) if self.matcher else None
        )
        try:
            self.nlp_processor = EnhancedNLPProcessor(
                spacy_model_name=config.SPACY_MODEL_NAME,
                sentiment_model_name=config.SENTIMENT_MODEL_NAME,
                emotion_model_name=config.EMOTION_MODEL_NAME,
                shared_semantic_model=shared_semantic,
            )
            logger.info("Enhanced NLP processor loaded")
        except Exception as e:
            logger.warning("NLP processor failed to load: %s", e)
            self.nlp_processor = None

        # Share the spaCy model with the preprocessor (avoid loading it twice)
        shared_spacy = (
            getattr(self.nlp_processor, "nlp", None) if self.nlp_processor else None
        )
        self.preprocessor = TextPreprocessor(
            spacy_model_name=config.SPACY_MODEL_NAME,
            shared_spacy_model=shared_spacy,
        )

        # Initialize phrase matcher, sharing SignMatcher and spaCy model
        try:
            self.phrase_matcher = IntelligentPhraseMatcher(
                config.GLOSS_DICT_PATH,
                config.TARGET_WORDS_PATH,
                sign_matcher=self.matcher,
                spacy_model_name=config.SPACY_MODEL_NAME,
            )
            # Reuse the already-loaded spaCy model to avoid a third load
            if shared_spacy and self.phrase_matcher.nlp is None:
                self.phrase_matcher.nlp = shared_spacy
            logger.info("Intelligent phrase matcher loaded")
        except Exception as e:
            logger.warning("Phrase matcher failed to load: %s", e)
            self.phrase_matcher = None

        logger.info("Auslan AI v2.0 initialized")

        if not self.matcher or not getattr(self.matcher, "semantic_model", None):
            logger.warning(
                "Semantic matching not available. Install sentence-transformers for full functionality."
            )
        else:
            logger.info("Semantic similarity available")

    def process_input(
        self,
        text: str,
        remove_stops: bool = False,
        use_semantic: bool = True,
        semantic_threshold: float = None,
        use_stemming: bool = False,
        use_intelligent_matching: bool = True,
        use_llm: bool = False,
    ) -> Dict[str, Any]:
        """Process user input text with NLP analysis and sign matching."""
        if semantic_threshold is None:
            semantic_threshold = config.DEFAULT_SEMANTIC_THRESHOLD

        # NLP analysis
        try:
            nlp_analysis = (
                self.nlp_processor.analyze_text(text) if self.nlp_processor else None
            )
        except Exception as e:
            logger.warning("NLP analysis failed: %s", e)
            nlp_analysis = None

        # Try intelligent matching first
        if use_intelligent_matching and self.phrase_matcher:
            try:
                phrase_match = self.phrase_matcher.match_phrase_intelligently(
                    text, use_semantic=use_semantic, threshold=semantic_threshold
                )
                successful_matches = phrase_match.matched_signs
                failed_matches = []

                total_words = len(text.split())
                signs_found = len(successful_matches)
                coverage_rate = signs_found / total_words if total_words > 0 else 0

                n = max(total_words, 1)
                exact_c = sum(
                    1 for m in successful_matches if m.get("match_type") == "exact"
                )
                fuzzy_c = sum(
                    1 for m in successful_matches if m.get("match_type") == "fuzzy"
                )
                synonym_c = sum(
                    1 for m in successful_matches if m.get("match_type") == "synonym"
                )
                semantic_c = sum(
                    1 for m in successful_matches if m.get("match_type") == "semantic"
                )
                llm_c = sum(
                    1 for m in successful_matches if m.get("match_type") == "llm"
                )
                coverage_stats = {
                    "coverage_rate": coverage_rate,
                    "exact_matches": exact_c,
                    "fuzzy_matches": fuzzy_c,
                    "synonym_matches": synonym_c,
                    "semantic_matches": semantic_c,
                    "llm_matches": llm_c,
                    "unmatched_tokens": total_words - signs_found,
                    "match_breakdown": {
                        "exact": exact_c / n,
                        "fuzzy": fuzzy_c / n,
                        "synonym": synonym_c / n,
                        "semantic": semantic_c / n,
                        "llm": llm_c / n,
                    },
                }

                return {
                    "original_text": text,
                    "processed_tokens": text.split(),
                    "total_tokens": total_words,
                    "successful_matches": successful_matches,
                    "failed_matches": failed_matches,
                    "coverage_stats": coverage_stats,
                    "signs_found": signs_found,
                    "nlp_analysis": _serialize_nlp_analysis(nlp_analysis),
                    "phrase_analysis": {
                        "phrase_type": phrase_match.phrase_type,
                        "overall_confidence": phrase_match.confidence,
                        "grammar_structure": phrase_match.grammar_structure,
                    },
                }
            except Exception as e:
                logger.warning("Intelligent phrase matching failed: %s", e)
                use_intelligent_matching = False

        # Fallback to basic token matching
        if self.matcher:
            tokens = self.preprocessor.preprocess(
                text, remove_stops=remove_stops, use_stemming=use_stemming
            )
            match_results = self.matcher.match_tokens(
                tokens,
                use_semantic=use_semantic,
                threshold=semantic_threshold,
                use_llm=use_llm,
            )
            coverage_stats = self.matcher.get_coverage_stats(
                tokens,
                use_semantic=use_semantic,
                threshold=semantic_threshold,
                use_llm=use_llm,
            )
            successful_matches = [
                r for r in match_results if r["match_type"] != "no_match"
            ]
            failed_matches = [r for r in match_results if r["match_type"] == "no_match"]
        else:
            tokens = text.split()
            successful_matches = []
            failed_matches = []
            coverage_stats = {
                "coverage_rate": 0,
                "exact_matches": 0,
                "synonym_matches": 0,
                "semantic_matches": 0,
                "unmatched_tokens": len(tokens),
                "match_breakdown": {"exact": 0, "synonym": 0, "semantic": 0},
            }

        return {
            "original_text": text,
            "processed_tokens": tokens,
            "total_tokens": len(tokens),
            "successful_matches": successful_matches,
            "failed_matches": failed_matches,
            "coverage_stats": coverage_stats,
            "signs_found": len(successful_matches),
            "nlp_analysis": _serialize_nlp_analysis(nlp_analysis),
        }

    def display_results(self, results: Dict[str, Any]) -> None:
        """Display processing results in a user-friendly format."""
        print("\n" + "=" * 60)
        print(f'INPUT: "{results["original_text"]}"')
        print("=" * 60)

        print(f"Processed tokens: {results['processed_tokens']}")
        print(f"Total tokens: {results['total_tokens']}")
        print(f"Signs found: {results['signs_found']}")
        print(f"Coverage: {results['coverage_stats']['coverage_rate']:.1%}")

        if "nlp_analysis" in results:
            nlp = results["nlp_analysis"]
            print("\nAI NLP ANALYSIS:")
            print(f"   Sentiment: {nlp['sentiment']} ({nlp['sentiment_score']:.2f})")
            print(f"   Emotion: {nlp['emotion']}")
            print(f"   Intent: {nlp['intent']}")
            print(f"   Formality: {nlp['formality']}")
            if nlp["entities"]:
                entity_strings = [
                    f"{e['text']} ({e['label']})" for e in nlp["entities"]
                ]
                print(f"   Entities: {', '.join(entity_strings)}")
            if nlp["key_phrases"]:
                print(f"   Key phrases: {', '.join(nlp['key_phrases'][:3])}")

        if "phrase_analysis" in results:
            phrase = results["phrase_analysis"]
            print("\nPHRASE ANALYSIS:")
            print(f"   Type: {phrase['phrase_type']}")
            print(f"   Confidence: {phrase['overall_confidence']:.1%}")
            print(f"   Grammar: {phrase['grammar_structure']}")

        if results["successful_matches"]:
            print("\nMATCHED SIGNS:")
            print("-" * 40)
            for i, match in enumerate(results["successful_matches"], 1):
                sign_data = match.get("sign_data") or {}
                print(f"{i}. {match['word'].upper()}")
                print(f"   GLOSS: {sign_data.get('gloss', 'N/A')}")
                print(
                    f"   Match type: {match['match_type']} ({match['confidence']:.1%} confidence)"
                )
                print(
                    f"   Description: {sign_data.get('description', 'No description available')}"
                )
                if "video_url" in sign_data:
                    print(f"   Video: {sign_data['video_url']}")
                print(f"   Category: {sign_data.get('category', 'N/A')}")
                if "synonyms" in sign_data:
                    print(f"   Synonyms: {', '.join(sign_data['synonyms'])}")
                print()

        if results["failed_matches"]:
            print("UNMATCHED WORDS:")
            print("-" * 40)
            unmatched_words = [match["word"] for match in results["failed_matches"]]
            print(f"   {', '.join(unmatched_words)}")

        stats = results["coverage_stats"]
        breakdown = stats.get("match_breakdown", {})
        print("\nSTATISTICS:")
        print("-" * 40)
        print(
            f"   Exact matches:   {stats['exact_matches']} ({breakdown.get('exact', 0):.1%})"
        )
        if stats.get("fuzzy_matches", 0):
            print(
                f"   Fuzzy matches:   {stats['fuzzy_matches']} ({breakdown.get('fuzzy', 0):.1%})"
            )
        print(
            f"   Synonym matches: {stats['synonym_matches']} ({breakdown.get('synonym', 0):.1%})"
        )
        print(
            f"   Semantic matches:{stats['semantic_matches']} ({breakdown.get('semantic', 0):.1%})"
        )
        if stats.get("llm_matches", 0):
            print(
                f"   LLM matches:     {stats['llm_matches']} ({breakdown.get('llm', 0):.1%})"
            )
        print(
            f"   Unmatched:       {stats['unmatched_tokens']} ({(1 - stats['coverage_rate']):.1%})"
        )

    def interactive_mode(self):
        """Run the system in interactive mode."""
        print("\nWelcome to the Auslan Sign Retrieval System!")
        print("Enter text to find corresponding Auslan signs.")
        print("Commands: 'quit' to exit")
        print(
            f"Options: --no-stops, --no-semantic, --stem, --thresh={config.DEFAULT_SEMANTIC_THRESHOLD}"
        )
        print("-" * 60)

        while True:
            try:
                user_input = input("\n> Enter text: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("quit", "exit", "q"):
                    print("Goodbye!")
                    break

                remove_stops = "--no-stops" in user_input
                use_semantic = "--no-semantic" not in user_input
                use_stemming = "--stem" in user_input
                threshold = config.DEFAULT_SEMANTIC_THRESHOLD
                m = re.search(r"--thresh=([0-9]*\.?[0-9]+)", user_input)
                if m:
                    try:
                        threshold = float(m.group(1))
                    except ValueError:
                        pass

                text = (
                    user_input.replace("--no-stops", "")
                    .replace("--no-semantic", "")
                    .replace("--stem", "")
                )
                text = re.sub(r"--thresh=([0-9]*\.?[0-9]+)", "", text).strip()
                if not text:
                    continue

                results = self.process_input(
                    text,
                    remove_stops=remove_stops,
                    use_semantic=use_semantic,
                    semantic_threshold=threshold,
                    use_stemming=use_stemming,
                )
                self.display_results(results)

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

    def batch_evaluation(
        self,
        test_texts: List[str],
        remove_stops: bool = False,
        use_semantic: bool = True,
        semantic_threshold: float = None,
        use_stemming: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate the system on a batch of test texts."""
        if semantic_threshold is None:
            semantic_threshold = config.DEFAULT_SEMANTIC_THRESHOLD

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
            total_coverage += results["coverage_stats"]["coverage_rate"]
            print(
                f'{i}. "{text}" -> {results["signs_found"]}/{results["total_tokens"]} signs '
                f"({results['coverage_stats']['coverage_rate']:.1%} coverage)"
            )

        average_coverage = total_coverage / len(test_texts) if test_texts else 0

        return {
            "test_texts": test_texts,
            "individual_results": all_results,
            "average_coverage": average_coverage,
            "total_tests": len(test_texts),
        }


def main():
    """Main function to run the Auslan Sign System."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    system = AuslanSignSystem()

    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            test_texts = [
                "Hello, how are you today?",
                "I need help finding the toilet",
                "Let's go buy some food and eat together",
                "My friend lives in a big house",
                "Can you speak more slowly please?",
                "I am happy to see you",
                "Goodbye and have a good day",
            ]

            extra_args = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else ""
            remove_stops = "--no-stops" in extra_args
            use_semantic = "--no-semantic" not in extra_args
            use_stemming = "--stem" in extra_args
            threshold = config.DEFAULT_SEMANTIC_THRESHOLD
            m = re.search(r"--thresh=([0-9]*\.?[0-9]+)", extra_args)
            if m:
                try:
                    threshold = float(m.group(1))
                except ValueError:
                    pass

            print("Running test evaluation on sample texts...")
            evaluation = system.batch_evaluation(
                test_texts,
                remove_stops=remove_stops,
                use_semantic=use_semantic,
                semantic_threshold=threshold,
                use_stemming=use_stemming,
            )
            print(f"\nOverall average coverage: {evaluation['average_coverage']:.1%}")

        elif sys.argv[1] == "--interactive":
            system.interactive_mode()

        else:
            raw_input = " ".join(sys.argv[1:])
            remove_stops = "--no-stops" in raw_input
            use_semantic = "--no-semantic" not in raw_input
            use_stemming = "--stem" in raw_input
            threshold = config.DEFAULT_SEMANTIC_THRESHOLD
            m = re.search(r"--thresh=([0-9]*\.?[0-9]+)", raw_input)
            if m:
                try:
                    threshold = float(m.group(1))
                except ValueError:
                    pass
            text = (
                raw_input.replace("--no-stops", "")
                .replace("--no-semantic", "")
                .replace("--stem", "")
            )
            text = re.sub(r"--thresh=([0-9]*\.?[0-9]+)", "", text).strip()

            results = system.process_input(
                text,
                remove_stops=remove_stops,
                use_semantic=use_semantic,
                semantic_threshold=threshold,
                use_stemming=use_stemming,
            )
            system.display_results(results)

    else:
        system.interactive_mode()


if __name__ == "__main__":
    main()
