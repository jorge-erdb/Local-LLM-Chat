from flask import Flask, render_template, request, jsonify, Response, session
from llm_model import LLMModel
import threading
import json
import uuid
import os

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'

# Global model instance for shared model loading
global_llm_model = None
model_loaded = False

# Store session-specific models
session_models = {}

def load_model_background():
    """Load model in background thread"""
    global model_loaded, global_llm_model
    try:
        print(f"Before loading: model_loaded = {model_loaded}")
        global_llm_model = LLMModel()  # Create without session for model loading
        success = global_llm_model.load_model()
        print(f"load_model() returned: {success}")
        print(f"global_llm_model.is_loaded: {global_llm_model.is_loaded}")
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

def get_session_model():
    """Get or create a session-specific model instance"""
    global session_models, global_llm_model
    
    # Get or create session ID
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    session_id = session['session_id']
    
    # Check if we already have a model for this session
    if session_id not in session_models:
        if global_llm_model and global_llm_model.is_loaded:
            # Create a new session-specific model that shares the loaded LLM
            session_model = LLMModel(session_id)
            session_model.llm = global_llm_model.llm  # Share the loaded model
            session_model.is_loaded = True
            session_models[session_id] = session_model
        else:
            return None
    
    return session_models[session_id]

@app.route('/')
def index():
    """Main chat interface"""
    return render_template('chat.html')

@app.route('/status')
def status():
    """Check if model is loaded and return session info"""
    session_model = get_session_model()
    is_loaded = session_model is not None and session_model.is_loaded
    
    status_data = {
        'loaded': is_loaded,
        'message': 'Model ready' if is_loaded else 'Model loading...',
        'session_id': session.get('session_id', 'No session')
    }
    
    if is_loaded and session_model:
        history_info = session_model.get_history_info()
        session_info = session_model.get_session_info()
        status_data.update(history_info)
        status_data.update(session_info)
    
    return jsonify(status_data)

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    session_model = get_session_model()
    if not session_model or not session_model.is_loaded:
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
        response = session_model.generate_response(user_message)
        
        return jsonify({
            'response': response,
            'loaded': session_model.is_loaded,
            'session_id': session_model.session_id
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Error generating response: {str(e)}',
            'loaded': session_model.is_loaded if session_model else False
        }), 500

@app.route('/chat/stream', methods=['POST'])
def chat_stream():
    """Handle streaming chat messages"""
    session_model = get_session_model()
    if not session_model or not session_model.is_loaded:
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
                for token in session_model.generate_response_stream(user_message):
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
            'loaded': session_model.is_loaded if session_model else False
        }), 500

@app.route('/clear', methods=['POST'])
def clear_conversation():
    """Clear conversation history"""
    session_model = get_session_model()
    if not session_model or not session_model.is_loaded:
        return jsonify({
            'error': 'Model not loaded yet.',
            'loaded': False
        }), 503
    
    try:
        session_model.clear_history()
        return jsonify({
            'message': 'Conversation cleared',
            'loaded': True,
            'session_id': session_model.session_id
        })
    except Exception as e:
        return jsonify({
            'error': f'Error clearing conversation: {str(e)}',
            'loaded': session_model.is_loaded if session_model else False
        }), 500

@app.route('/conversations', methods=['GET'])
def list_conversations():
    """List all available conversations"""
    try:
        conversations = LLMModel.list_all_conversations()
        return jsonify({
            'conversations': conversations,
            'current_session': session.get('session_id')
        })
    except Exception as e:
        return jsonify({'error': f'Error listing conversations: {str(e)}'}), 500

@app.route('/conversations/new', methods=['POST'])
def new_conversation():
    """Start a new conversation"""
    try:
        # Generate new session ID
        new_session_id = str(uuid.uuid4())
        session['session_id'] = new_session_id
        
        # Remove old session model if it exists
        if new_session_id in session_models:
            del session_models[new_session_id]
        
        return jsonify({
            'message': 'New conversation started',
            'session_id': new_session_id
        })
    except Exception as e:
        return jsonify({'error': f'Error starting new conversation: {str(e)}'}), 500

@app.route('/conversations/<session_id>/switch', methods=['POST'])
def switch_conversation(session_id):
    """Switch to an existing conversation"""
    try:
        # Validate session exists
        conversations_dir = './conversations'
        conversation_file = os.path.join(conversations_dir, f'{session_id}.jsonl')
        
        if not os.path.exists(conversation_file):
            return jsonify({'error': 'Conversation not found'}), 404
        
        # Switch to the session
        session['session_id'] = session_id
        
        # Remove from session_models to force reload
        if session_id in session_models:
            del session_models[session_id]
        
        return jsonify({
            'message': 'Switched to conversation',
            'session_id': session_id
        })
    except Exception as e:
        return jsonify({'error': f'Error switching conversation: {str(e)}'}), 500

@app.route('/conversations/<session_id>/delete', methods=['DELETE'])
def delete_conversation(session_id):
    """Delete a conversation"""
    try:
        conversations_dir = './conversations'
        conversation_file = os.path.join(conversations_dir, f'{session_id}.jsonl')
        
        if not os.path.exists(conversation_file):
            return jsonify({'error': 'Conversation not found'}), 404
        
        # Don't delete current active conversation
        if session.get('session_id') == session_id:
            return jsonify({'error': 'Cannot delete active conversation'}), 400
        
        # Delete the file
        os.remove(conversation_file)
        
        # Remove from session_models if loaded
        if session_id in session_models:
            del session_models[session_id]
        
        return jsonify({'message': 'Conversation deleted'})
    except Exception as e:
        return jsonify({'error': f'Error deleting conversation: {str(e)}'}), 500

@app.route('/conversations/<session_id>/history', methods=['GET'])
def get_conversation_history(session_id):
    """Get the full conversation history for a session"""
    try:
        conversations_dir = './conversations'
        conversation_file = os.path.join(conversations_dir, f'{session_id}.jsonl')
        
        if not os.path.exists(conversation_file):
            return jsonify({'error': 'Conversation not found'}), 404
        
        messages = []
        title = None
        
        with open(conversation_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    
                    # Get title from first entry
                    if title is None and "title" in entry:
                        title = entry["title"]
                    
                    # Collect messages (skip title-only entries)
                    if "role" in entry and "content" in entry:
                        messages.append({
                            "role": entry["role"],
                            "content": entry["content"],
                            "timestamp": entry.get("timestamp", "")
                        })
        
        return jsonify({
            'session_id': session_id,
            'title': title,
            'messages': messages
        })
        
    except Exception as e:
        return jsonify({'error': f'Error fetching conversation history: {str(e)}'}), 500

if __name__ == '__main__':
    print("Starting Flask app...")
    print("Model will load in background. Check /status endpoint to see when ready.")
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)