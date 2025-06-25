# 🤖 Image Dataset Preparation Tools 🤖  
<p align="center">  
  <img src="https://raw.githubusercontent.com/Ollama-Agent-Roll-Cage/oarc-image-annotation-ui/main/assets/wizardPic.png" alt="OARC img anno wizard" width="450"/>  
</p>

A collection of Python tools for preparing image datasets with GPU acceleration, metadata management, and AI-powered annotation.

## Features

- GPU-accelerated image format conversion
- Metadata stripping and standardization
- Support for HEIC/HEIF formats
- Batch processing capabilities
- Image annotation management
- CSV metadata generation
- **AI-powered image annotation using Ollama models**
- **Vision model support for automatic image description**

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
- Vision model support (llava, bakllava, etc.)
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
