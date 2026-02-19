# -*- coding: utf-8 -*-
"""
Evaluation script for the Auslan Sign Retrieval System.
Tests coverage and performance across different matching strategies.
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime

import config
from main import AuslanSignSystem

logger = logging.getLogger(__name__)


def parse_flags(argv):
    """Parse command-line arguments with improved argparse."""
    parser = argparse.ArgumentParser(
        description="Evaluate Auslan Sign Retrieval System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--no-stops", action="store_true", help="Remove stop words during processing"
    )
    parser.add_argument(
        "--no-semantic", action="store_true", help="Disable semantic matching"
    )
    parser.add_argument("--stem", action="store_true", help="Enable word stemming")
    parser.add_argument(
        "--thresh",
        type=float,
        default=config.DEFAULT_SEMANTIC_THRESHOLD,
        help="Semantic matching threshold (0.0-1.0)",
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Export results to JSON file (e.g., results.json)",
    )
    parser.add_argument(
        "--no-synonym",
        action="store_true",
        help="Disable synonym and phrase-based matching",
    )

    args = parser.parse_args(argv)

    return (
        args.no_stops,
        not args.no_semantic,
        args.thresh,
        args.stem,
        args.export,
        not args.no_synonym,
    )


def run_evaluation(argv=None):
    """Run comprehensive evaluation of the system."""
    if argv is None:
        argv = sys.argv[1:]
    (
        remove_stops,
        use_semantic,
        semantic_threshold,
        use_stemming,
        export_path,
        use_synonym,
    ) = parse_flags(argv)

    start_time = time.time()

    system = AuslanSignSystem()

    # Fitness coaching test scenarios
    workout_instructions = [
        "Do three sets of ten squats",
        "Warm up before lifting weights",
        "Focus on your chest and arms today",
        "Remember to breathe during each exercise",
    ]

    fitness_goals = [
        "Build muscle and get strong",
        "Exercise regularly to stay fit",
        "Stretch after every workout session",
        "Drink water and eat protein daily",
    ]

    training_guidance = [
        "Keep your back straight during squats",
        "Rest between sets for good form",
        "Run on the treadmill for cardio",
        "Cool down with stretching exercises",
    ]

    all_test_sets = {
        "Workout Instructions": workout_instructions,
        "Fitness Goals": fitness_goals,
        "Training Guidance": training_guidance,
    }

    logger.info("Fitness Coach Auslan Translator - Comprehensive Evaluation")
    logger.info("=" * 60)

    overall_results = {}

    for category, test_texts in all_test_sets.items():
        logger.info("\n%s:", category.upper())
        logger.info("-" * 40)

        evaluation = system.batch_evaluation(
            test_texts,
            remove_stops=remove_stops,
            use_semantic=use_semantic,
            semantic_threshold=semantic_threshold,
            use_stemming=use_stemming,
        )
        overall_results[category] = evaluation

        logger.info("Average coverage: %.1f%%", evaluation["average_coverage"] * 100)

        # Detailed breakdown
        total_exact = sum(
            r["coverage_stats"]["exact_matches"]
            for r in evaluation["individual_results"]
        )
        total_synonym = sum(
            r["coverage_stats"]["synonym_matches"]
            for r in evaluation["individual_results"]
        )
        total_semantic = sum(
            r["coverage_stats"]["semantic_matches"]
            for r in evaluation["individual_results"]
        )
        total_tokens = sum(r["total_tokens"] for r in evaluation["individual_results"])

        logger.info("Total tokens processed: %d", total_tokens)
        if total_tokens > 0:
            logger.info(
                "Exact matches: %d (%.1f%%)",
                total_exact,
                total_exact / total_tokens * 100,
            )
            logger.info(
                "Synonym matches: %d (%.1f%%)",
                total_synonym,
                total_synonym / total_tokens * 100,
            )
            logger.info(
                "Semantic matches: %d (%.1f%%)",
                total_semantic,
                total_semantic / total_tokens * 100,
            )
        else:
            logger.info("No tokens to process in this category")

    # Overall statistics
    all_coverages = [result["average_coverage"] for result in overall_results.values()]
    overall_avg = sum(all_coverages) / len(all_coverages) if all_coverages else 0

    elapsed_time = time.time() - start_time

    logger.info("\nOVERALL PERFORMANCE")
    logger.info("=" * 60)
    logger.info("Average coverage across all categories: %.1f%%", overall_avg * 100)

    logger.info("\nPerformance by category:")
    for category, result in overall_results.items():
        logger.info("  %s: %.1f%%", category, result["average_coverage"] * 100)

    logger.info("\nEvaluation completed in %.2f seconds", elapsed_time)

    # Prepare export data
    export_data = {
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "remove_stops": remove_stops,
            "use_semantic": use_semantic,
            "semantic_threshold": semantic_threshold,
            "use_stemming": use_stemming,
        },
        "overall_performance": {
            "average_coverage": overall_avg,
            "elapsed_time_seconds": elapsed_time,
        },
        "category_results": overall_results,
    }

    if export_path:
        try:
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            logger.info("Results exported to: %s", export_path)
        except Exception as e:
            logger.error("Error exporting results: %s", e)

    return overall_results


def test_semantic_vs_exact(argv=None):
    """Compare semantic matching vs exact/synonym matching."""
    if argv is None:
        argv = sys.argv[1:]
    _, _, semantic_threshold, use_stemming, _, _ = parse_flags(argv)

    system = AuslanSignSystem()

    test_phrases = [
        "I require assistance",  # Should match "help" via semantic
        "Purchase some food",  # Should match "buy" via semantic
        "Large building",  # Should match "big" and "house" via semantic
        "Pleased to meet you",  # Should match "happy" via semantic
        "Communicate with me",  # Should match "speak" via semantic
    ]

    logger.info("\nSemantic Matching Comparison:")
    logger.info("=" * 40)

    for phrase in test_phrases:
        logger.info('\nPhrase: "%s"', phrase)

        results_no_semantic = system.process_input(
            phrase, use_semantic=False, use_stemming=use_stemming
        )
        coverage_no_semantic = results_no_semantic["coverage_stats"]["coverage_rate"]

        results_with_semantic = system.process_input(
            phrase,
            use_semantic=True,
            semantic_threshold=semantic_threshold,
            use_stemming=use_stemming,
        )
        coverage_with_semantic = results_with_semantic["coverage_stats"][
            "coverage_rate"
        ]

        logger.info("  Without semantic: %.1f%%", coverage_no_semantic * 100)
        logger.info("  With semantic: %.1f%%", coverage_with_semantic * 100)
        logger.info(
            "  Improvement: %.1f%%",
            (coverage_with_semantic - coverage_no_semantic) * 100,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    results = run_evaluation(sys.argv[1:])
    test_semantic_vs_exact(sys.argv[1:])

    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Complete!")
    logger.info("=" * 60)
