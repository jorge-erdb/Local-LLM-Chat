Flask App for simple Local LLM Chat

python -m venv venv

source venv/bin/activate

pip install --upgrade pip

pip install -r requirements.txt

1) Create LLM/your_model.gguf
2) update it llm_model.py "self.MODEL_PATH = 'your/model/path/*.gguf'"
3) Modify settings in llm_model.py