# Local LLM Chat Interface

A Flask-based web application for running local Large Language Models with conversation history and memory features.

## Features

- **Web Interface**: Clean, responsive chat interface with real-time status updates
- **Conversation Memory**: Persistent conversation history with automatic context management
- **Memory Search**: Advanced search functionality across conversation history
- **Session Management**: Individual conversation sessions with unique memory contexts
- **GPU Optimization**: Configurable GPU layers for optimal performance
- **Context Management**: Automatic token limit handling with conversation trimming

## Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- Sufficient VRAM for your chosen model

## Installation

1. Clone the repository:
```bash
git clone https://github.com/jorge-erdb/Local-LLM-Chat.git
cd Local-LLM-Chat
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

**For CUDA GPU support (recommended):**
```bash
pip install --upgrade pip
pip install -r "requirements CUDA.txt"
```
*Uses precompiled CUDA wheel for faster installation*

**For CPU-only:**
```bash
pip install --upgrade pip
pip install -r "requirements no CUDA.txt"
```

**If the precompiled CUDA wheel doesn't work:** Install from source with CUDA support:
```bash
CMAKE_ARGS="-DGGML_CUDA=ON" pip install llama-cpp-python --no-cache --no-binary llama-cpp-python
```

4. Set up your model:
   - Create an `LLM/` directory
   - Download a compatible GGUF model file and place it there
   - Update the `MODEL_PATH` in `llm_model.py` to point to your model

## Usage

### Web Interface
```bash
python app.py
```
Access the web interface at [http://localhost:5000](http://localhost:5000)

## Configuration

### Model Settings (`llm_model.py`)
- `MODEL_PATH`: Path to your GGUF model file
- `n_gpu_layers`: Number of layers to offload to GPU
- `n_ctx`: Context window size (default: 8192 tokens)
- `temperature`: Response randomness (0.0-1.0)

### Recommended Llama Models

Compatible GGUF format models (place in `LLM/` directory):

**8B Parameter Models:**
- `Llama-3.1-8B-Instruct-Q5_K_M.gguf` - Balanced quality/performance
- `Llama-3.1-8B-Instruct-Q4_K_M.gguf` - Lower VRAM usage
- `Llama-3.1-8B-Instruct-Q6_K.gguf` - Higher quality

**70B Parameter Models (requires 32GB+ VRAM):**
- `Llama-3.1-70B-Instruct-Q4_K_M.gguf`
- `Llama-3.1-70B-Instruct-Q5_K_M.gguf`

Download models from [Hugging Face](https://huggingface.co/models?search=llama-3.1+gguf) or other compatible sources.

## API Endpoints

- `GET /` - Main chat interface
- `POST /chat` - Send message and receive response
- `GET /status` - Model status and context information
- `POST /clear` - Clear conversation history
- `POST /search` - Search conversation memory

## Memory System

The application includes a simple memory system (work in progress) that:
- Stores conversations with basic search functionality
- User command: `/recall query` (single word queries only)
- Model can invoke `<function_call>search_memory('one word query')</function_call>` to access past conversations (if specified in system prompt)
- Basic implementation that may have limitations and is being improved

## Performance Notes

- Model loading takes 2-3 minutes on first startup
- Monitor GPU memory usage with `nvidia-smi` (NVIDIA GPUs) or equivalent GPU monitoring tools
- Adjust `n_gpu_layers` based on available VRAM
- GPU acceleration requires CUDA-enabled llama-cpp-python installation

## Development

The project structure:
- `app.py` - Flask web server with background model loading
- `llm_model.py` - Core model management and conversation handling
- `memory_search.py` - Advanced memory search functionality
- `memory_analyzer.py` - Conversation analysis tools
- `templates/` - Web interface templates

## License

Licensed under the terms specified in the LICENSE file.