# Setup Guide

## System Requirements

- Python 3.11 or higher
- CUDA-compatible GPU (optional)
- 4GB RAM minimum (8GB recommended)
- Windows/Linux/MacOS
- **Ollama (optional, for AI features)**

## Installation Steps

1. **Create Virtual Environment**
   ```bash
   uv venv -p 3.11 .venv
   ```

2. **Activate Virtual Environment**
   ```bash
   # Windows
   .venv\Scripts\activate
   
   # Unix/MacOS
   source .venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   uv pip install -r requirements.txt
   ```

## GPU Support Setup

1. Install CUDA Toolkit (if using NVIDIA GPU)
2. Verify GPU detection:
   ```python
   python -c "import torch; print(torch.cuda.is_available())"
   ```

## Ollama Setup (Optional)

For AI-powered annotation features:

### 1. Install Ollama

**Windows/MacOS:**
- Download from [ollama.ai](https://ollama.ai)
- Run the installer

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### 2. Install Models

**Vision Models (Recommended for image annotation):**
```bash

ollama pull gemma3:4b
ollama pull llava

```

**Text Models (Basic support):**
```bash
ollama pull llama2
ollama pull mistral
```

### 3. Verify Installation

```bash
# List installed models
ollama list

# Test a model
ollama run gemma3:4b "Describe what you see" --image path/to/image.jpg
```

### 4. Python Integration Test

```python
import asyncio
from ollama_tools import OllamaCommands

async def test_ollama():
    ollama = OllamaCommands()
    try:
        models = await ollama.list_models()
        print(f"Available models: {[m['name'] for m in models]}")
        return True
    except Exception as e:
        print(f"Ollama not available: {e}")
        return False

# Test connection
asyncio.run(test_ollama())
```

## Common Issues

### HEIC Support
If HEIC support is needed, ensure `pillow-heif` is installed:
```bash
uv pip install pillow-heif
```

### GPU Not Detected
- Verify CUDA installation
- Update GPU drivers
- Check PyTorch installation matches CUDA version

### Ollama Issues

**Service Not Running:**
```bash
# Start Ollama service (Linux/MacOS)
ollama serve

# Windows: Ollama runs as a service automatically
```

**Models Not Found:**
```bash
# Re-pull models
ollama pull llava
ollama list
```

**Connection Errors:**
- Ensure Ollama service is running
- Check if port 11434 is available
- Verify firewall settings

**Performance Issues:**
- Use smaller models (e.g., `llava:7b` instead of `llava:13b`)
- Close other applications to free RAM
- Consider using CPU-only models for lower-end hardware
- Consider finetuning your vision model with unsloth, this can be done by making a high quality image annotation dataset that aligns with your goals, finetuning the model, and loading it in oarc-image-annotation-ui (You can learn more in the [unsloth vision fine tuning docs](https://docs.unsloth.ai/basics/vision-fine-tuning))
- Try modifying the system prompt in the backend code, in the next update I will add a system prompt control box in the settings

### Python Package Issues

**Missing ollama package:**
```bash
pip install ollama
```

**Async/await errors:**
```python
# Use asyncio.run() for standalone scripts
import asyncio
result = asyncio.run(your_async_function())
```