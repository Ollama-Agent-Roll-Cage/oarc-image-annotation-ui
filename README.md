# Image Dataset Preparation Tools

A collection of Python tools for preparing image datasets with GPU acceleration and metadata management.

## Features

- GPU-accelerated image format conversion
- Metadata stripping and standardization
- Support for HEIC/HEIF formats
- Batch processing capabilities
- Image annotation management
- CSV metadata generation

## Installation

### Prerequisites

- Python 3.11 or higher
- GPU support (optional, for acceleration)

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

## Modules

### 1. Image Format Master
GPU-accelerated image conversion tool with metadata stripping capabilities.

```bash
python image_format_master.py <input_dir> <output_dir> [format] [quality]
```

### 2. Image Text Annotation
Tool for managing image annotations and generating metadata CSV files.

```python
from img_text_annotation_csv import ImageAnnotationProcessor

processor = ImageAnnotationProcessor(
    input_dir="./input_images",
    output_dir="./output_images"
)
```

## Documentation

Detailed documentation is available in the `docs` folder:
- [Setup Guide](docs/setup.md)
- [CLI Usage Guide](docs/cli_usage.md)
- [API Reference](docs/api_reference.md)

## License

MIT License - See LICENSE file for details.

## Author

@BorcherdingL