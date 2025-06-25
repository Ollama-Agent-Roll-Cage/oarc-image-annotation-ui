# 🤖 oarc-image-annotation-ui 🤖  
<p align="center">  
  <img src="https://raw.githubusercontent.com/Ollama-Agent-Roll-Cage/oarc-image-annotation-ui/main/assets/wizardPicCrop.png" alt="OARC img anno wizard" width="450"/>  
</p>

A collection of Python tools for preparing image datasets with GPU acceleration, metadata management, and AI-powered annotation.

## Dataset Types

- Tools for producing high quality human annotated text to image datasets for stable diffusion, pix2pix, and other text to image generation tasks.
- Tools for producing high quality human annotated image to text datasets for finetuning or training Large Language and Vision Assistants such LLaVa, or Gemma3:4b
- image to text datasets can also be used for training OCR models.
- with some simple modifications this toolkit could be used for annotating tons of other dataset types as well, the skys the limit!
  
## Features

- GPU-accelerated image format conversion
- Metadata stripping and standardization
- Support for HEIC/HEIF formats
- Batch processing capabilities
- Image annotation management system
- Builds CSV metadata datasets in Hugging Face Format 
- **Ollama Vision model automatic image annotation**
- Scripting with the backend modules is supported

## Installation

### Prerequisites

- Python 3.11 or higher
- GPU support (optional, for acceleration)
- **Ollama (optional, for AI annotation features)**

### Setup with UV (Recommended)

```bash
# Create virtual environment
uv venv -p 3.11 .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Unix/MacOS:
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### Ollama Setup (Optional)
For AI annotation features, install Ollama:
1. Download and install Ollama from [ollama.ai](https://ollama.ai)
2. Pull a vision model (recommended): `ollama pull llava`
3. Or pull any text model for basic features: `ollama pull llama2`

## Modules

### 1. Image Format Master
GPU-accelerated image conversion tool with metadata stripping capabilities.

```bash
python image_format_master.py <input_dir> <output_dir> [format] [quality]
```

### 2. Image Text Annotation
Tool for managing image annotations and generating metadata CSV files.

```bash
python image_text_annotation_csv.py <input_dir> <output_dir>
```

### 3. Ollama Tools Integration
AI-powered annotation using local Ollama models for automatic image description.

Features:
- Model listing and selection
- Vision model support (llava, gemma3:4b, llama3.2-vision, etc)
- Streaming responses
- Batch annotation processing

### 4. Combined UI Application
Combines format conversion, annotation management, and AI tools into a unified PyQt6 interface.

```bash
python image_annotation_ui.py
```

**New AI Features in UI:**
- Automatic model detection
- Vision-based image annotation
- Batch AI annotation for entire datasets
- Customizable system prompts

<p align="center">  
  <img src="https://raw.githubusercontent.com/Ollama-Agent-Roll-Cage/oarc-image-annotation-ui/main/assets/imagePrepUiExample.png" alt="OARC img anno ui" width="900"/>  
</p>

## Documentation

Detailed documentation is available in the `docs` folder:
- [Setup Guide](docs/setup.md)
- [CLI Usage Guide](docs/cli_usage.md)
- [API Reference](docs/api_reference.md)

## License

MIT License - See LICENSE file for details.

## Author

@BorcherdingL
