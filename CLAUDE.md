# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a local LLM environment for running Llama 3.1 8B with two interfaces:
- **Console interface** (`LLM.py`) - Direct terminal chat
- **Web interface** (`app.py`) - Flask web app with conversation history

## Core Architecture

### Model Management (`llm_model.py`)
- **LLMModel class** - Centralized model configuration and conversation management
- **Context window**: 8192 tokens with automatic history trimming
- **Conversation persistence** - Maintains full conversation history with system prompt
- **GPU optimization** - 16 GPU layers configured for RTX 3050 4GB

### Web Interface (`app.py`) 
- **Background model loading** - Model loads in separate thread to avoid blocking Flask startup
- **Threading considerations** - Run with `debug=False` to avoid global variable threading issues
- **API endpoints**: `/status`, `/chat`, `/clear` for model state and conversation management
- **Real-time updates** - Status endpoint provides context usage statistics

### Model Configuration
- **Model file**: `./LLM/Llama-3.1-8B-Instruct-Q5_K_M.gguf` (3.6GB VRAM usage)
- **Chat format**: Llama 3.1 instruction format with proper token handling
- **Context management**: Automatic trimming when approaching 8192 token limit

## Common Commands

### Running the Applications
```bash
# Activate virtual environment
source venv/bin/activate

# Console interface (simple chat)
python LLM.py

# Web interface (full-featured with history)
python app.py
# Access at http://localhost:5000
```

### Environment Setup
```bash
# Install dependencies
pip install llama-cpp-python flask

# Check GPU memory before running
nvidia-smi
```

### Development Notes
- **GPU memory conflicts**: Only one interface can run at a time (both use ~3.6GB VRAM)
- **Model loading time**: ~2-3 minutes on first startup
- **Context efficiency**: Web interface uses full 8192 token context with conversation history
- **Threading**: Flask must run with `debug=False` for proper model state management

### Key Files
- `LLM.py` - Standalone console chat (legacy, simpler interface)
- `app.py` - Flask web server with background model loading
- `llm_model.py` - Reusable model class with conversation management
- `templates/chat.html` - Web interface with real-time status and context tracking