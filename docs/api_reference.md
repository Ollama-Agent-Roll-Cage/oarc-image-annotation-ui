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
from img_text_annotation_csv import ImageAnnotationProcessor

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