# -*- coding: utf-8 -*-
"""
Evaluation script for the Auslan Sign Retrieval System.
Tests coverage and performance across different matching strategies.
"""

import json
import re
import sys
from main import AuslanSignSystem

def parse_flags(argv):
    args = ' '.join(argv)
    remove_stops = '--no-stops' in args
    use_semantic = '--no-semantic' not in args
    use_stemming = '--stem' in args
    semantic_threshold = 0.6
    m = re.search(r'--thresh=([0-9]*\.?[0-9]+)', args)
    if m:
        try:
            semantic_threshold = float(m.group(1))
        except ValueError:
            pass
    return remove_stops, use_semantic, semantic_threshold, use_stemming


def run_evaluation(argv=None):
    """Run comprehensive evaluation of the system."""
    if argv is None:
        argv = sys.argv[1:]
    remove_stops, use_semantic, semantic_threshold, use_stemming = parse_flags(argv)
    
    # Initialize system
    system = AuslanSignSystem()
    
    # Fitness coaching test scenarios
    workout_instructions = [
        "Do three sets of ten squats",
        "Warm up before lifting weights",
        "Focus on your chest and arms today", 
        "Remember to breathe during each exercise"
    ]
    
    fitness_goals = [
        "Build muscle and get strong",
        "Exercise regularly to stay fit",
        "Stretch after every workout session",
        "Drink water and eat protein daily"
    ]
    
    training_guidance = [
        "Keep your back straight during squats",
        "Rest between sets for good form",
        "Run on the treadmill for cardio",
        "Cool down with stretching exercises"
    ]
    
    # Combine all test sets
    all_test_sets = {
        "Workout Instructions": workout_instructions,
        "Fitness Goals": fitness_goals, 
        "Training Guidance": training_guidance
    }
    
    print("Fitness Coach Auslan Translator - Comprehensive Evaluation")
    print("=" * 60)
    
    overall_results = {}
    
    for category, test_texts in all_test_sets.items():
        print(f"\n{category.upper()}:")
        print("-" * 40)
        
        evaluation = system.batch_evaluation(
            test_texts,
            remove_stops=remove_stops,
            use_semantic=use_semantic,
            semantic_threshold=semantic_threshold,
            use_stemming=use_stemming,
        )
        overall_results[category] = evaluation
        
        print(f"Average coverage: {evaluation['average_coverage']:.1%}")
        
        # Detailed breakdown
        total_exact = sum(r['coverage_stats']['exact_matches'] for r in evaluation['individual_results'])
        total_synonym = sum(r['coverage_stats']['synonym_matches'] for r in evaluation['individual_results'])
        total_semantic = sum(r['coverage_stats']['semantic_matches'] for r in evaluation['individual_results'])
        total_tokens = sum(r['total_tokens'] for r in evaluation['individual_results'])
        
        print(f"Total tokens processed: {total_tokens}")
        print(f"Exact matches: {total_exact} ({total_exact/total_tokens:.1%})")
        print(f"Synonym matches: {total_synonym} ({total_synonym/total_tokens:.1%})")
        print(f"Semantic matches: {total_semantic} ({total_semantic/total_tokens:.1%})")
    
    # Overall statistics
    all_coverages = [result['average_coverage'] for result in overall_results.values()]
    overall_avg = sum(all_coverages) / len(all_coverages)
    
    print(f"\n{'OVERALL PERFORMANCE'}")
    print("=" * 60)
    print(f"Average coverage across all categories: {overall_avg:.1%}")
    
    # Performance by category
    print(f"\nPerformance by category:")
    for category, result in overall_results.items():
        print(f"  {category}: {result['average_coverage']:.1%}")
    
    return overall_results

def test_semantic_vs_exact(argv=None):
    """Compare semantic matching vs exact/synonym matching."""
    if argv is None:
        argv = sys.argv[1:]
    _, _, semantic_threshold, use_stemming = parse_flags(argv)
    
    system = AuslanSignSystem()
    
    test_phrases = [
        "I require assistance",  # Should match "help" via semantic
        "Purchase some food",    # Should match "buy" via semantic  
        "Large building",        # Should match "big" and "house" via semantic
        "Pleased to meet you",   # Should match "happy" via semantic
        "Communicate with me"    # Should match "speak" via semantic
    ]
    
    print("\nSemantic Matching Comparison:")
    print("=" * 40)
    
    for phrase in test_phrases:
        print(f"\nPhrase: \"{phrase}\"")
        
        # Test without semantic
        results_no_semantic = system.process_input(phrase, use_semantic=False, use_stemming=use_stemming)
        coverage_no_semantic = results_no_semantic['coverage_stats']['coverage_rate']
        
        # Test with semantic
        results_with_semantic = system.process_input(phrase, use_semantic=True, semantic_threshold=semantic_threshold, use_stemming=use_stemming)
        coverage_with_semantic = results_with_semantic['coverage_stats']['coverage_rate']
        
        print(f"  Without semantic: {coverage_no_semantic:.1%}")
        print(f"  With semantic: {coverage_with_semantic:.1%}")
        print(f"  Improvement: {(coverage_with_semantic - coverage_no_semantic):.1%}")

if __name__ == "__main__":
    # Run main evaluation
    results = run_evaluation(sys.argv[1:])
    
    # Test semantic matching
    test_semantic_vs_exact(sys.argv[1:])
    
    print(f"\nEvaluation complete! System is ready for weeks 6-8 development.")