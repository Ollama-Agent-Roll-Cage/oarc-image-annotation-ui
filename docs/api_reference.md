# API Reference

## ImageAnnotationProcessor

### Class Methods

#### `__init__(input_dir: str, output_dir: str, annotations: Optional[Dict[str, str]] = None)`
Initialize the processor with directories and optional annotations.

#### `add_annotations(annotations: Dict[str, str]) -> None`
Add or update image annotations.

#### `generate_metadata_csv() -> str`
Process files and generate metadata CSV.

### Factory Methods

#### `create_with_annotations(input_dir: str, output_dir: str, annotations: Dict[str, str])`
Create processor with pre-defined annotations.

#### `create_instance(input_dir: str, output_dir: str)`
Create processor without annotations.

### Example Usage

```python
from image_text_annotation_csv import ImageAnnotationProcessor

# With annotations
annotations = {
    "image1.jpg": "Mountain landscape",
    "image2.png": "City street at night"
}

processor = ImageAnnotationProcessor.create_with_annotations(
    input_dir="./input",
    output_dir="./output",
    annotations=annotations
)

csv_path = processor.generate_metadata_csv()
```

## GPU-Accelerated Image Processing

### Main Functions

#### `convert_image_gpu(input_path, output_dir, output_format='jpg', quality=95)`
Convert image using GPU acceleration when available.

#### `process_batch(image_batch, output_dir, output_format, quality)`
Process multiple images in parallel.

### Example Usage

```python
from image_format_master import convert_image_gpu

success, message = convert_image_gpu(
    input_path="input.heic",
    output_dir="./output",
    output_format="jpg",
    quality=95
)
```

## OllamaCommands

### Initialization

```python
from ollama_tools import OllamaCommands

ollama = OllamaCommands()
```

### Model Management

#### `list_models() -> List[Dict[str, Any]]`
Get list of available Ollama models.

#### `show_model_info(model: str) -> Dict[str, Any]`
Show detailed information about a specific model.

#### `get_loaded_models() -> List[Dict[str, Any]]`
Get list of currently loaded models.

### Text Generation

#### `chat(model: str, messages: List[Dict[str, str]], stream: bool = False)`
Send chat messages to a model.

```python
messages = [
    {"role": "user", "content": "Explain machine learning"}
]

response = await ollama.chat(
    model="llama2",
    messages=messages
)
```

#### `generate(model: str, prompt: str, stream: bool = False)`
Generate text from a simple prompt.

```python
response = await ollama.generate(
    model="llama2",
    prompt="Write a haiku about coding"
)
```

### Vision Capabilities

#### `vision_chat(model: str, prompt: str, image_data: str, stream: bool = False)`
Send image and text prompt to a vision model.

```python
import base64

# Load and encode image
with open("image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

response = await ollama.vision_chat(
    model="llava",
    prompt="Describe this image in detail",
    image_data=image_data
)
```

### Streaming Responses

All generation methods support streaming:

```python
async for chunk in ollama.chat(model="llama2", messages=messages, stream=True):
    print(chunk, end="", flush=True)
```

### Error Handling

```python
try:
    models = await ollama.list_models()
except Exception as e:
    print(f"Ollama not available: {e}")
```

## Integration Example

Complete workflow using all components:

```python
import asyncio
import base64
from pathlib import Path
from image_format_master import convert_image_gpu
from image_text_annotation_csv import ImageAnnotationProcessor
from ollama_tools import OllamaCommands

async def process_dataset_with_ai():
    # 1. Convert images
    input_dir = Path("raw_images")
    processed_dir = Path("processed")
    
    for img in input_dir.glob("*"):
        convert_image_gpu(img, processed_dir, "jpg", 95)
    
    # 2. Generate AI annotations
    ollama = OllamaCommands()
    annotations = {}
    
    for img_path in processed_dir.glob("*.jpg"):
        # Encode image
        with open(img_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()
        
        # Get AI annotation
        annotation = await ollama.vision_chat(
            model="llava",
            prompt="Describe this image for a dataset",
            image_data=image_data
        )
        
        annotations[img_path.name] = annotation
    
    # 3. Create metadata CSV
    processor = ImageAnnotationProcessor.create_with_annotations(
        input_dir=str(processed_dir),
        output_dir=str(processed_dir),
        annotations=annotations
    )
    
    csv_path = processor.generate_metadata_csv()
    print(f"Dataset ready: {csv_path}")

# Run the workflow
asyncio.run(process_dataset_with_ai())
```