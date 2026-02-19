"""
Flask web application for Auslan Sign Retrieval System.
Provides a user-friendly web interface for testing sign matching.
"""

import logging
import os
from datetime import datetime

from flask import Flask, render_template, request, jsonify, send_from_directory

import config
from main import AuslanSignSystem

logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = config.SECRET_KEY

# Lazy singleton for the sign system
_sign_system = None


def get_sign_system():
    """Get or initialize the sign system."""
    global _sign_system
    if _sign_system is None:
        _sign_system = AuslanSignSystem()
    return _sign_system


@app.route('/')
def index():
    """Main page with input form and results display."""
    return render_template('index.html')


@app.route('/api/process', methods=['POST'])
def process_text():
    """API endpoint to process user input text."""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        options = data.get('options', {})
        remove_stops = options.get('remove_stops', False)
        use_semantic = options.get('use_semantic', True)
        use_stemming = options.get('use_stemming', False)
        semantic_threshold = options.get('semantic_threshold', config.DEFAULT_SEMANTIC_THRESHOLD)
        use_intelligent_matching = options.get('use_intelligent_matching', True)

        system = get_sign_system()
        results = system.process_input(
            text,
            remove_stops=remove_stops,
            use_semantic=use_semantic,
            semantic_threshold=semantic_threshold,
            use_stemming=use_stemming,
            use_intelligent_matching=use_intelligent_matching
        )

        results['processed_at'] = datetime.now().isoformat()
        return jsonify(results)

    except Exception as e:
        logger.exception("Error processing text")
        return jsonify({'error': str(e)}), 500


@app.route('/api/evaluate', methods=['POST'])
def evaluate_system():
    """API endpoint to run system evaluation."""
    try:
        data = request.get_json()
        test_texts = data.get('test_texts', [])

        if not test_texts:
            test_texts = [
                "Hello, how are you today?",
                "I need help finding the toilet",
                "Let's exercise and build muscle strength",
                "Warm up before lifting weights",
                "Cool down with stretching exercises"
            ]

        options = data.get('options', {})
        remove_stops = options.get('remove_stops', False)
        use_semantic = options.get('use_semantic', True)
        use_stemming = options.get('use_stemming', False)
        semantic_threshold = options.get('semantic_threshold', config.DEFAULT_SEMANTIC_THRESHOLD)

        system = get_sign_system()
        evaluation = system.batch_evaluation(
            test_texts,
            remove_stops=remove_stops,
            use_semantic=use_semantic,
            semantic_threshold=semantic_threshold,
            use_stemming=use_stemming
        )

        return jsonify(evaluation)

    except Exception as e:
        logger.exception("Error running evaluation")
        return jsonify({'error': str(e)}), 500


@app.route('/media/videos/<filename>')
def serve_video(filename):
    """Serve video files, checking both the repo and scraped video dirs."""
    for video_dir in config.VIDEO_DIRS:
        video_path = os.path.join(video_dir, filename)
        if os.path.isfile(video_path):
            return send_from_directory(video_dir, filename)
    return "Video not found", 404


@app.route('/api/dictionary')
def get_dictionary():
    """API endpoint to get dictionary information."""
    try:
        system = get_sign_system()
        if not system.matcher:
            return jsonify({'error': 'Matcher not available'}), 503

        dictionary_info = {
            'total_entries': len(system.matcher.gloss_dict),
            'categories': {},
            'sample_entries': {}
        }

        for word, data in system.matcher.gloss_dict.items():
            category = data.get('category', 'unknown')
            if category not in dictionary_info['categories']:
                dictionary_info['categories'][category] = 0
            dictionary_info['categories'][category] += 1

            description = data.get('description', '')
            dictionary_info['sample_entries'][word] = {
                'gloss': data.get('gloss'),
                'category': category,
                'synonyms': data.get('synonyms', [])[:3],
                'description': description[:100] + '...' if len(description) > 100 else description
            }

        return jsonify(dictionary_info)

    except Exception as e:
        logger.exception("Error getting dictionary")
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze', methods=['POST'])
def analyze_text():
    """API endpoint for detailed NLP analysis."""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        system = get_sign_system()
        if not system.nlp_processor:
            return jsonify({'error': 'NLP processor not available'}), 503

        analysis = system.nlp_processor.analyze_text(text)

        return jsonify({
            'text': text,
            'sentiment': {
                'label': analysis.sentiment_label,
                'score': analysis.sentiment_score,
                'confidence': analysis.confidence
            },
            'emotion': analysis.emotion,
            'intent': analysis.intent,
            'entities': analysis.entities,
            'key_phrases': analysis.key_phrases,
            'formality': analysis.formality_level,
            'complexity': analysis.complexity_score,
            'readability': analysis.readability_score,
            'analyzed_at': datetime.now().isoformat()
        })

    except Exception as e:
        logger.exception("Error analyzing text")
        return jsonify({'error': str(e)}), 500


@app.route('/api/suggestions', methods=['POST'])
def get_suggestions():
    """API endpoint for intelligent phrase suggestions."""
    try:
        data = request.get_json()
        partial_text = data.get('text', '').strip()

        if len(partial_text) < 2:
            return jsonify({'suggestions': []})

        system = get_sign_system()
        if not system.phrase_matcher:
            return jsonify({'suggestions': []})

        suggestions = system.phrase_matcher.get_phrase_suggestions(partial_text, limit=8)

        return jsonify({
            'partial_text': partial_text,
            'suggestions': suggestions
        })

    except Exception as e:
        logger.exception("Error getting suggestions")
        return jsonify({'error': str(e)}), 500


@app.route('/api/models/status')
def model_status():
    """API endpoint to check which AI models are available."""
    try:
        system = get_sign_system()

        status = {
            'spacy_available': system.nlp_processor is not None and system.nlp_processor.nlp is not None,
            'semantic_model_available': system.matcher is not None and getattr(system.matcher, 'semantic_model', None) is not None,
            'sentiment_model_available': system.nlp_processor is not None and system.nlp_processor.sentiment_model is not None,
            'emotion_model_available': system.nlp_processor is not None and system.nlp_processor.emotion_model is not None,
            'intelligent_matching_available': system.phrase_matcher is not None,
            'total_signs': len(system.matcher.gloss_dict) if system.matcher else 0,
            'system_version': '2.0'
        }

        return jsonify(status)

    except Exception as e:
        logger.exception("Error getting model status")
        return jsonify({'error': str(e)}), 500


@app.route('/about')
def about():
    """About page with system information."""
    return render_template('about.html')


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

    logger.info("Starting Auslan Sign Retrieval Web Application...")
    logger.info("Open your browser to: http://localhost:%d", config.SERVER_PORT)

    app.run(debug=config.DEBUG, host=config.SERVER_HOST, port=config.SERVER_PORT)
