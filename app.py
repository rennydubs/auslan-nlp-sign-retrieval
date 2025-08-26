"""
Flask web application for Auslan Sign Retrieval System.
Provides a user-friendly web interface for testing sign matching.
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import json
from main import AuslanSignSystem
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'auslan-sign-system-2024'

# Initialize the sign system
sign_system = None

def get_sign_system():
    """Get or initialize the sign system."""
    global sign_system
    if sign_system is None:
        sign_system = AuslanSignSystem()
    return sign_system

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
        
        # Get processing options
        options = data.get('options', {})
        remove_stops = options.get('remove_stops', False)
        use_semantic = options.get('use_semantic', True)
        use_stemming = options.get('use_stemming', False)
        semantic_threshold = options.get('semantic_threshold', 0.6)
        
        # Process the text
        system = get_sign_system()
        results = system.process_input(
            text,
            remove_stops=remove_stops,
            use_semantic=use_semantic,
            semantic_threshold=semantic_threshold,
            use_stemming=use_stemming
        )
        
        # Add processing timestamp
        results['processed_at'] = datetime.now().isoformat()
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/evaluate', methods=['POST'])
def evaluate_system():
    """API endpoint to run system evaluation."""
    try:
        data = request.get_json()
        test_texts = data.get('test_texts', [])
        
        if not test_texts:
            # Use default test texts
            test_texts = [
                "Hello, how are you today?",
                "I need help finding the toilet",
                "Let's exercise and build muscle strength",
                "Warm up before lifting weights",
                "Cool down with stretching exercises"
            ]
        
        # Get processing options
        options = data.get('options', {})
        remove_stops = options.get('remove_stops', False)
        use_semantic = options.get('use_semantic', True)
        use_stemming = options.get('use_stemming', False)
        semantic_threshold = options.get('semantic_threshold', 0.6)
        
        # Run evaluation
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
        return jsonify({'error': str(e)}), 500

@app.route('/media/videos/<filename>')
def serve_video(filename):
    """Serve video files from the media directory."""
    video_dir = os.path.join(os.getcwd(), 'media', 'videos')
    return send_from_directory(video_dir, filename)

@app.route('/api/dictionary')
def get_dictionary():
    """API endpoint to get dictionary information."""
    try:
        system = get_sign_system()
        dictionary_info = {
            'total_entries': len(system.matcher.gloss_dict),
            'categories': {},
            'sample_entries': {}
        }
        
        # Count by category and get samples
        for word, data in list(system.matcher.gloss_dict.items())[:10]:
            category = data.get('category', 'unknown')
            if category not in dictionary_info['categories']:
                dictionary_info['categories'][category] = 0
            dictionary_info['categories'][category] += 1
            
            dictionary_info['sample_entries'][word] = {
                'gloss': data.get('gloss'),
                'category': category,
                'synonyms': data.get('synonyms', [])[:3],  # First 3 synonyms
                'description': data.get('description', '')[:100] + '...' if len(data.get('description', '')) > 100 else data.get('description', '')
            }
        
        return jsonify(dictionary_info)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/about')
def about():
    """About page with system information."""
    return render_template('about.html')

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    templates_dir = os.path.join(os.getcwd(), 'templates')
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)
    
    # Create static directory if it doesn't exist
    static_dir = os.path.join(os.getcwd(), 'static')
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
    
    print("Starting Auslan Sign Retrieval Web Application...")
    print("Open your browser to: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)