from flask import Flask, render_template, request, jsonify, Response
from llm_model import LLMModel
import threading
import json

app = Flask(__name__)

# Initialize model instance
llm_model = LLMModel()
model_loaded = False

def load_model_background():
    """Load model in background thread"""
    global model_loaded
    try:
        print(f"Before loading: model_loaded = {model_loaded}")
        success = llm_model.load_model()
        print(f"load_model() returned: {success}")
        print(f"llm_model.is_loaded: {llm_model.is_loaded}")
        model_loaded = success
        print(f"After setting: model_loaded = {model_loaded}")
        print(f"Background model loading: {'Success' if success else 'Failed'}")
    except Exception as e:
        print(f"Background model loading failed: {e}")
        model_loaded = False

# Start loading model in background when app starts
print("Starting model loading in background...")
loading_thread = threading.Thread(target=load_model_background)
loading_thread.daemon = True
loading_thread.start()

print("http://localhost:5000")

@app.route('/')
def index():
    """Main chat interface"""
    return render_template('chat.html')

@app.route('/status')
def status():
    """Check if model is loaded"""
    # Use the model object's state directly instead of global variable
    is_loaded = llm_model.is_loaded
    status_data = {
        'loaded': is_loaded,
        'message': 'Model ready' if is_loaded else 'Model loading...'
    }
    
    if is_loaded:
        history_info = llm_model.get_history_info()
        status_data.update(history_info)
    
    return jsonify(status_data)

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    if not llm_model.is_loaded:
        return jsonify({
            'error': 'Model not loaded yet. Please wait.',
            'loaded': False
        }), 503
    
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
        
        # Generate response
        response = llm_model.generate_response(user_message)
        
        return jsonify({
            'response': response,
            'loaded': llm_model.is_loaded
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Error generating response: {str(e)}',
            'loaded': llm_model.is_loaded
        }), 500

@app.route('/chat/stream', methods=['POST'])
def chat_stream():
    """Handle streaming chat messages"""
    if not llm_model.is_loaded:
        return jsonify({
            'error': 'Model not loaded yet. Please wait.',
            'loaded': False
        }), 503
    
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
        
        def generate():
            try:
                for token in llm_model.generate_response_stream(user_message):
                    yield f"data: {json.dumps({'token': token})}\n\n"
                yield f"data: {json.dumps({'done': True})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return Response(
            generate(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*'
            }
        )
        
    except Exception as e:
        return jsonify({
            'error': f'Error generating response: {str(e)}',
            'loaded': llm_model.is_loaded
        }), 500

@app.route('/clear', methods=['POST'])
def clear_conversation():
    """Clear conversation history"""
    if not llm_model.is_loaded:
        return jsonify({
            'error': 'Model not loaded yet.',
            'loaded': False
        }), 503
    
    try:
        llm_model.clear_history()
        return jsonify({
            'message': 'Conversation cleared',
            'loaded': True
        })
    except Exception as e:
        return jsonify({
            'error': f'Error clearing conversation: {str(e)}',
            'loaded': llm_model.is_loaded
        }), 500

if __name__ == '__main__':
    print("Starting Flask app...")
    print("Model will load in background. Check /status endpoint to see when ready.")
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)