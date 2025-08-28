from llama_cpp import Llama
import os
import json
import uuid
from datetime import datetime

class LLMModel:
    def __init__(self, session_id=None):
        # Model Parameters
        self.MODEL_PATH = './LLM/Llama-3.1-8B-Instruct-Q5_K_M.gguf'
        self.N_GPU_LAYERS = 16
        self.CONTEXT_SIZE = 8192
        self.TEMPERATURE = 0.8
        self.TOP_K = 40
        self.TOP_P = 0.95
        self.MIN_P = 0.050
        self.MAX_TOKENS = -1
        self.REPEAT_PENALTY = 1
        self.REPEAT_LAST_N = 64
        self.BATCH_SIZE = 2048
        self.M_LOCK = True
        self.FLASH_ATTENTION = True
        self.VERBOSE = False
        
        self.llm = None
        self.is_loaded = False
        
        # Session and conversation management
        self.session_id = session_id or str(uuid.uuid4())
        self.conversations_dir = './conversations'
        self.conversation_file = os.path.join(self.conversations_dir, f'{self.session_id}.jsonl')
        self.conversation_history = []
        self.conversation_title = None
        self.system_prompt = "You are Lumina, a helpful, harmless, and honest AI assistant."
        
        # Create conversations directory if it doesn't exist
        os.makedirs(self.conversations_dir, exist_ok=True)
        
        # Load existing conversation history
        self.load_conversation_history()
    
    def load_model(self):
        """Load the LLM model"""
        if self.is_loaded:
            print("Model already loaded!")
            return True
            
        # Check if the model file exists
        if not os.path.exists(self.MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {self.MODEL_PATH}")
        
        print("Loading model... This may take a few minutes.")
        
        try:
            self.llm = Llama(
                model_path=self.MODEL_PATH,
                n_gpu_layers=self.N_GPU_LAYERS,
                n_ctx=self.CONTEXT_SIZE,
                n_batch=self.BATCH_SIZE,
                use_mlock=self.M_LOCK,
                flash_attn=self.FLASH_ATTENTION,
                verbose=self.VERBOSE,
                repeat_last_n=self.REPEAT_LAST_N,
            )
            
            self.is_loaded = True
            print("Model loaded successfully! Ready for inference.")
            print(f"GPU layers: {self.N_GPU_LAYERS}")
            print(f"Context size: {self.CONTEXT_SIZE}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def load_conversation_history(self):
        """Load conversation history from JSONL file"""
        self.conversation_history = []
        self.conversation_title = None
        if os.path.exists(self.conversation_file):
            try:
                with open(self.conversation_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            entry = json.loads(line)
                            # Load title from first entry (if available)
                            if self.conversation_title is None and "title" in entry:
                                self.conversation_title = entry["title"]
                            
                            # Only load the role and content for conversation context
                            if "role" in entry and "content" in entry:
                                self.conversation_history.append({
                                    "role": entry["role"],
                                    "content": entry["content"]
                                })
            except Exception as e:
                print(f"Error loading conversation history: {e}")
                self.conversation_history = []
    
    def generate_title_from_message(self, message):
        """Generate a conversation title from the first user message"""
        # Take first 50 characters and clean it up
        title = message.strip()[:50]
        
        # Remove newlines and extra spaces
        title = ' '.join(title.split())
        
        # If it's too short, use a default
        if len(title) < 10:
            title = "New Conversation"
        
        # Add ellipsis if truncated
        if len(message.strip()) > 50:
            title += "..."
            
        return title
    
    def save_title_to_file(self):
        """Save or update the conversation title"""
        if not self.conversation_title:
            return
            
        try:
            # Read existing content
            lines = []
            if os.path.exists(self.conversation_file):
                with open(self.conversation_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            
            # Check if first line already has title
            title_exists = False
            if lines:
                try:
                    first_entry = json.loads(lines[0].strip())
                    if "title" in first_entry:
                        title_exists = True
                except:
                    pass
            
            if not title_exists:
                # Insert title as first line
                title_entry = {
                    "session_id": self.session_id,
                    "timestamp": datetime.now().isoformat(),
                    "title": self.conversation_title
                }
                
                with open(self.conversation_file, 'w', encoding='utf-8') as f:
                    f.write(json.dumps(title_entry, ensure_ascii=False) + '\n')
                    for line in lines:
                        f.write(line)
                        
        except Exception as e:
            print(f"Error saving title to file: {e}")

    def save_message_to_file(self, role, content):
        """Save a single message to the JSONL file"""
        try:
            entry = {
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "role": role,
                "content": content
            }
            
            with open(self.conversation_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"Error saving message to file: {e}")
    
    def add_to_history(self, role, content):
        """Add a message to conversation history and save to file"""
        self.conversation_history.append({"role": role, "content": content})
        self.save_message_to_file(role, content)
        
        # Generate title from first user message
        if role == "user" and not self.conversation_title and len(self.conversation_history) <= 2:
            self.conversation_title = self.generate_title_from_message(content)
            self.save_title_to_file()
    
    def clear_history(self):
        """Clear conversation history and delete the session file"""
        self.conversation_history = []
        self.conversation_title = None
        try:
            if os.path.exists(self.conversation_file):
                os.remove(self.conversation_file)
        except Exception as e:
            print(f"Error deleting conversation file: {e}")
    
    def get_session_info(self):
        """Get information about the current session"""
        return {
            "session_id": self.session_id,
            "conversation_file": self.conversation_file,
            "file_exists": os.path.exists(self.conversation_file),
            "title": self.conversation_title
        }
    
    @staticmethod
    def list_all_conversations(conversations_dir='./conversations'):
        """List all conversation files with their titles and metadata"""
        conversations = []
        
        if not os.path.exists(conversations_dir):
            return conversations
            
        try:
            for filename in os.listdir(conversations_dir):
                if filename.endswith('.jsonl'):
                    session_id = filename[:-6]  # Remove .jsonl extension
                    filepath = os.path.join(conversations_dir, filename)
                    
                    try:
                        # Read first few lines to get title and basic info
                        with open(filepath, 'r', encoding='utf-8') as f:
                            lines = list(f)
                            
                        if not lines:
                            continue
                            
                        # Parse first line to get title and timestamp
                        first_line = json.loads(lines[0].strip())
                        title = first_line.get('title', 'Untitled Conversation')
                        created_at = first_line.get('timestamp', '')
                        
                        # Count messages (exclude title line)
                        message_count = len([line for line in lines[1:] if line.strip()])
                        
                        # Get last message timestamp
                        last_timestamp = created_at
                        if len(lines) > 1:
                            try:
                                last_line = json.loads(lines[-1].strip())
                                last_timestamp = last_line.get('timestamp', created_at)
                            except:
                                pass
                        
                        conversations.append({
                            'session_id': session_id,
                            'title': title,
                            'created_at': created_at,
                            'last_activity': last_timestamp,
                            'message_count': message_count,
                            'file_path': filepath
                        })
                        
                    except Exception as e:
                        print(f"Error reading conversation {filename}: {e}")
                        continue
                        
        except Exception as e:
            print(f"Error listing conversations: {e}")
            
        # Sort by last activity (most recent first)
        conversations.sort(key=lambda x: x.get('last_activity', ''), reverse=True)
        return conversations
    
    def get_conversation_prompt(self, user_input):
        """Build full conversation prompt with history"""
        # Start with system prompt
        prompt = f"<|start_header_id|>system<|end_header_id|>\n{self.system_prompt}<|eot_id|>"
        
        # Add conversation history
        for msg in self.conversation_history:
            prompt += f"<|start_header_id|>{msg['role']}<|end_header_id|>\n{msg['content']}<|eot_id|>"
        
        # Add current user input
        prompt += f"<|start_header_id|>user<|end_header_id|>\n{user_input}<|eot_id|>"
        
        # Start assistant response
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n"
        
        return prompt
    
    def estimate_token_count(self, text):
        """Rough estimation of token count (1 token ≈ 4 chars for English)"""
        return len(text) // 4
    
    def trim_history_if_needed(self, new_prompt):
        """Trim history if prompt would exceed context window"""
        estimated_tokens = self.estimate_token_count(new_prompt)
        
        # Keep some buffer for response tokens
        max_prompt_tokens = self.CONTEXT_SIZE - 1024
        
        while estimated_tokens > max_prompt_tokens and len(self.conversation_history) > 0:
            # Remove oldest message pair (user + assistant)
            if len(self.conversation_history) >= 2:
                self.conversation_history = self.conversation_history[2:]
            else:
                self.conversation_history = []
            
            # Recalculate with trimmed history
            new_prompt = self.get_conversation_prompt("")
            estimated_tokens = self.estimate_token_count(new_prompt)
    
    def generate_response(self, user_input, max_tokens=-1):
        """Generate response with conversation context"""
        if not self.is_loaded:
            raise Exception("Model not loaded. Call load_model() first.")
        
        try:
            # Build conversation prompt
            prompt = self.get_conversation_prompt(user_input)
            
            # Trim history if needed
            self.trim_history_if_needed(prompt)
            prompt = self.get_conversation_prompt(user_input)
            
            response = self.llm(
                prompt,
                max_tokens=max_tokens if max_tokens > 0 else self.MAX_TOKENS,
                temperature=self.TEMPERATURE,
                top_k=self.TOP_K,
                top_p=self.TOP_P,
                min_p=self.MIN_P,
                repeat_penalty=self.REPEAT_PENALTY,
                stop=["<|eot_id|>"],
                stream=False
            )
            
            assistant_response = response['choices'][0]['text'].strip()
            
            # Add to conversation history
            self.add_to_history("user", user_input)
            self.add_to_history("assistant", assistant_response)
            
            return assistant_response
            
        except Exception as e:
            raise Exception(f"Error generating response: {e}")
    
    def generate_response_stream(self, user_input, max_tokens=-1):
        """Generate streaming response with conversation context"""
        if not self.is_loaded:
            raise Exception("Model not loaded. Call load_model() first.")
        
        try:
            # Build conversation prompt
            prompt = self.get_conversation_prompt(user_input)
            
            # Trim history if needed
            self.trim_history_if_needed(prompt)
            prompt = self.get_conversation_prompt(user_input)
            
            response = self.llm(
                prompt,
                max_tokens=max_tokens if max_tokens > 0 else self.MAX_TOKENS,
                temperature=self.TEMPERATURE,
                top_k=self.TOP_K,
                top_p=self.TOP_P,
                min_p=self.MIN_P,
                repeat_penalty=self.REPEAT_PENALTY,
                stop=["<|eot_id|>"],
                stream=True
            )
            
            # Collect full response for history
            full_response = ""
            
            for output in response:
                token = output['choices'][0]['text']
                full_response += token
                yield token
            
            # Add to conversation history after streaming is complete
            self.add_to_history("user", user_input)
            self.add_to_history("assistant", full_response.strip())
                
        except Exception as e:
            raise Exception(f"Error generating streaming response: {e}")
    
    def get_history_info(self):
        """Get information about current conversation"""
        total_messages = len(self.conversation_history)
        total_chars = sum(len(msg['content']) for msg in self.conversation_history)
        estimated_tokens = self.estimate_token_count(
            self.get_conversation_prompt("") + self.system_prompt
        )
        
        return {
            "message_count": total_messages,
            "total_characters": total_chars,
            "estimated_tokens": estimated_tokens,
            "context_usage_percent": round((estimated_tokens / self.CONTEXT_SIZE) * 100, 1)
        }