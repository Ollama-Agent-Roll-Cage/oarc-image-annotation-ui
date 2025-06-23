# Setup Guide

## System Requirements

- Python 3.11 or higher
- CUDA-compatible GPU (optional)
- 4GB RAM minimum (8GB recommended)
- Windows/Linux/MacOS

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

## Common Issues

### HEIC Support
If HEIC support is needed, ensure `pillow_heif` is installed:
```bash
uv pip install pillow-heif
```

### GPU Not Detected
- Verify CUDA installation
- Update GPU drivers
- Check PyTorch installation matches CUDA version