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

### UI usage

```bash
python image_annotation_ui.py
```
