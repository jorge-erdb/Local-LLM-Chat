from llama_cpp import Llama
import os

class LLMModel:
    def __init__(self):
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
        
        # Conversation history management
        self.conversation_history = []
        self.system_prompt = "You are a helpful, harmless, and honest AI assistant."
    
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
    
    def add_to_history(self, role, content):
        """Add a message to conversation history"""
        self.conversation_history.append({"role": role, "content": content})
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
    
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
    
    def generate_response(self, user_input, max_tokens=512):
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
                max_tokens=max_tokens if max_tokens > 0 else (self.MAX_TOKENS if self.MAX_TOKENS > 0 else 512),
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
    
    def generate_response_stream(self, user_input, max_tokens=512):
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
                max_tokens=max_tokens if max_tokens > 0 else (self.MAX_TOKENS if self.MAX_TOKENS > 0 else 512),
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