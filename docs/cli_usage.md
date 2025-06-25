# CLI Usage Guide

## Image Format Master

### Basic Usage

```bash
python image_format_master.py <input_dir> <output_dir> [format] [quality]
```

### Parameters

- `input_dir`: Directory containing source images
- `output_dir`: Directory for processed images
- `format`: Output format (jpg/png, default: jpg)
- `quality`: JPG quality (1-100, default: 95)

### Examples

```bash
# Convert to JPG with default settings
python image_format_master.py ./raw_images ./processed

# Convert to PNG
python image_format_master.py ./raw_images ./processed png

# Convert to JPG with custom quality
python image_format_master.py ./raw_images ./processed jpg 85
```

### Interactive Mode

Running without parameters enters interactive mode:
```bash
python image_format_master.py
```

## Image Text Annotation

### Command Line Usage

```bash
python image_text_annotation_csv.py <input_dir> <output_dir>
```

### Example Usage with Annotations

```bash
python image_text_annotation_csv.py ./images ./annotated
```

## UI Applications

### Main Annotation UI

```bash
python image_annotation_ui.py
```

**AI Features:**
- Select Ollama models from dropdown
- Use "🔍 Current" to annotate current image
- Use "🚀 All" to batch annotate entire dataset
- Vision models (llava, bakllava) work best for image description

### AI Model Requirements

For best results with AI annotation:
- **Vision models** (recommended): `llava`, `bakllava`, `llava-phi3`
- **Text models** (basic support): `llama2`, `mistral`, `codellama`

### Example Ollama Setup

```bash
# Install a vision model
ollama pull llava

# Or install a smaller vision model
ollama pull llava:7b

# List available models
ollama list
```

## Ollama Tools Module

The Ollama integration can also be used programmatically:

```python
from ollama_tools import OllamaCommands

# Initialize
ollama = OllamaCommands()

# List available models
models = await ollama.list_models()

# Vision annotation
annotation = await ollama.vision_chat(
    model="llava",
    prompt="Describe this image in detail",
    image_data=base64_image_data
)
```
