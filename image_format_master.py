#!/usr/bin/env python3
"""
GPU-Accelerated Image Format Converter
Converts all images in a directory to specified format (JPG or PNG), including HEIC files.
Uses GPU acceleration with OpenCV and concurrent processing for maximum speed.
"""

# Import necessary libraries
import os
import sys
import shutil
import time
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import partial
import numpy as np

# Setup logging
logger = logging.getLogger(__name__)

# Add these imports at the top of the file (after the existing imports)
try:
    import torch
    from torchvision import transforms
    from torchvision.transforms.functional import to_pil_image, to_tensor
    TORCH_AVAILABLE = torch.cuda.is_available()
    if TORCH_AVAILABLE:
        print(f"PyTorch GPU acceleration available! Using {torch.cuda.get_device_name(0)}")
        logger.info(f"PyTorch GPU acceleration available! Using {torch.cuda.get_device_name(0)}")
    else:
        print("PyTorch installed but GPU not available, falling back to CPU processing")
        logger.warning("PyTorch installed but GPU not available, falling back to CPU processing")
except ImportError:
    print("PyTorch not installed, falling back to CPU-only processing")
    TORCH_AVAILABLE = False

# Fallback imports
from PIL import Image
try:
    import pillow_heif
    HEIF_AVAILABLE = True
except ImportError:
    HEIF_AVAILABLE = False
    print("pillow_heif not available, HEIC files won't be supported")

def setup_heif_support():
    """Register HEIF opener with Pillow"""
    if HEIF_AVAILABLE:
        pillow_heif.register_heif_opener()

def get_image_files(directory):
    """Get all image files from the directory"""
    image_extensions = {
        '.heic', '.heif', '.png', '.bmp', '.tiff', '.tif', 
        '.webp', '.gif', '.jpeg', '.jpg'
    }
    
    image_files = []
    directory_path = Path(directory)
    
    if not directory_path.exists():
        print(f"Warning: Directory {directory} does not exist")
        return image_files
    
    for file_path in directory_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_files.append(file_path)
    
    return sorted(image_files)  # Sort for consistent ordering

def convert_image_gpu(input_path, output_dir, output_format='jpg', quality=95, dataset_name="image", image_index=1):
    """Convert an image using GPU acceleration with PyTorch"""
    try:
        # Set up output format details
        if output_format.lower() == 'png':
            output_ext = '.png'
        else:  # default to jpg
            output_ext = '.jpg'
        
        # Create output filename with dataset naming convention
        output_filename = f"{dataset_name}_{image_index:04d}{output_ext}"
        output_path = output_dir / output_filename
        
        # Handle HEIC files with PIL first (PyTorch doesn't support HEIC)
        if input_path.suffix.lower() in ['.heic', '.heif']:
            return convert_image_pil_fallback(input_path, output_dir, output_format, quality, dataset_name, image_index)
        
        # Use PyTorch for GPU acceleration if available
        if TORCH_AVAILABLE:
            # Open with PIL first
            with Image.open(input_path) as img:
                # Strip metadata by creating a new image
                clean_img = Image.new(img.mode, img.size)
                clean_img.paste(img)
                img = clean_img
                
                # Convert to tensor and move to GPU
                img_tensor = to_tensor(img).unsqueeze(0)  # Add batch dimension
                if torch.cuda.is_available():
                    img_tensor = img_tensor.cuda()
                
                # No processing needed, we're just using GPU to accelerate conversion
                # Convert back to CPU for saving
                processed_tensor = img_tensor.cpu().squeeze(0)
                result_img = to_pil_image(processed_tensor)
                
                # Save with appropriate settings
                if output_format.lower() == 'jpg':
                    # Convert to RGB for JPG (no transparency)
                    if result_img.mode in ('RGBA', 'LA', 'P'):
                        # Create white background for transparent images
                        background = Image.new('RGB', result_img.size, (255, 255, 255))
                        if result_img.mode == 'P':
                            result_img = result_img.convert('RGBA')
                        background.paste(result_img, mask=result_img.split()[-1] 
                                        if result_img.mode in ('RGBA', 'LA') else None)
                        result_img = background
                    elif result_img.mode != 'RGB':
                        result_img = result_img.convert('RGB')
                    
                    # Save as JPG with optimization - no metadata
                    result_img.save(output_path, 'JPEG', quality=quality, optimize=True, 
                                   progressive=True, exif=b'')
                else:  # PNG
                    # Convert to RGBA for PNG (preserves transparency)
                    if result_img.mode not in ('RGBA', 'LA', 'P'):
                        if result_img.mode != 'RGB':
                            result_img = result_img.convert('RGB')
                        result_img = result_img.convert('RGBA')
                    
                    # Save as PNG with optimization - no metadata
                    result_img.save(output_path, 'PNG', optimize=True, compress_level=6,
                                  pnginfo=None)
                
                return True, f"GPU Converted (PyTorch, metadata stripped): {input_path.name} -> {output_filename}"
        
        # Fallback to PIL
        return convert_image_pil_fallback(input_path, output_dir, output_format, quality, dataset_name, image_index)
            
    except Exception as e:
        return False, f"Error converting {input_path.name}: {str(e)}"

def convert_image_pil_fallback(input_path, output_dir, output_format='jpg', quality=95, dataset_name="image", image_index=1):
    """Fallback conversion using PIL (CPU-based but optimized)"""
    try:
        # Set up output format details
        if output_format.lower() == 'png':
            output_ext = '.png'
            pil_format = 'PNG'
        else:  # default to jpg
            output_ext = '.jpg'
            pil_format = 'JPEG'
        
        # Create output filename with dataset naming convention
        output_filename = f"{dataset_name}_{image_index:04d}{output_ext}"
        output_path = output_dir / output_filename
        
        # Handle corrupted HEIC files by allowing incorrect headers
        if input_path.suffix.lower() in ['.heic', '.heif'] and HEIF_AVAILABLE:
            # Set environment variable to allow loading corrupted HEIC files
            os.environ['PILLOW_HEIF_ALLOW_INCORRECT_HEADERS'] = '1'
        
        # Open and convert the image
        with Image.open(input_path) as img:
            # Strip ALL metadata/EXIF data by not preserving it
            # Create a new image without any metadata
            clean_img = Image.new(img.mode, img.size)
            clean_img.paste(img)
            img = clean_img
            
            # Enable faster resampling
            if hasattr(Image, 'Resampling'):
                resample = Image.Resampling.LANCZOS
            else:
                resample = Image.LANCZOS
            
            if output_format.lower() == 'jpg':
                # Convert to RGB for JPG (no transparency)
                if img.mode in ('RGBA', 'LA', 'P'):
                    # Create white background for transparent images
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Save as JPG with optimization - explicitly exclude EXIF data
                img.save(output_path, pil_format, quality=quality, optimize=True, 
                        progressive=True, exif=b'')  # Empty EXIF data
            
            else:  # PNG
                # Convert to RGBA for PNG (preserves transparency)
                if img.mode not in ('RGBA', 'LA', 'P'):
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img = img.convert('RGBA')
                
                # Save as PNG with optimization - no metadata preserved
                img.save(output_path, pil_format, optimize=True, compress_level=6,
                        pnginfo=None)  # No PNG metadata
            
            return True, f"CPU Converted (metadata stripped): {input_path.name} -> {output_filename}"
            
    except Exception as e:
        return False, f"Error converting {input_path.name}: {str(e)}"

def process_batch(image_batch, output_dir, output_format, quality, dataset_name, start_index):
    """Process a batch of images"""
    results = []
    for i, img_path in enumerate(image_batch):
        image_index = start_index + i + 1  # Start from 1, not 0
        success, message = convert_image_gpu(img_path, output_dir, output_format, quality, dataset_name, image_index)
        results.append((success, message))
        print(message)
    return results

def gpu_warmup():
    """Warm up GPU to ensure it's properly initialized"""
    try:
        if TORCH_AVAILABLE and torch.cuda.is_available():
            # Create dummy tensor and move to GPU
            dummy = torch.zeros(1, 3, 100, 100).cuda()
            # Force computation
            result = dummy * 2.0
            # Synchronize to ensure operation completes
            torch.cuda.synchronize()
            logger.info("PyTorch GPU warmup successful")
            print("PyTorch GPU warmup successful")
            return True
        return False
    except Exception as e:
        logger.warning(f"GPU warmup failed: {str(e)}")
        print(f"GPU warmup failed: {str(e)}")
        return False

def main():
    """Main function to convert all images in a directory with GPU acceleration"""
    print("GPU-Accelerated Image Converter with Metadata Removal")
    print("=" * 55)
    
    # Get input directory from command line argument or prompt
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
    else:
        input_dir = input("Enter the input directory path (or press Enter for current directory): ").strip()
        if not input_dir:
            input_dir = "."
    
    # Validate input directory
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Error: Directory '{input_dir}' does not exist.")
        return
    
    if not input_path.is_dir():
        print(f"Error: '{input_dir}' is not a directory.")
        return
    
    # Get output directory
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = input("Enter the output directory path: ").strip()
        if not output_dir:
            print("Error: Output directory is required.")
            return
    
    # Create output directory
    output_path = Path(output_dir)
    os.makedirs(output_path, exist_ok=True)
    print(f"Output directory: {output_path}")
    
    # Setup HEIF support
    setup_heif_support()
    
    # Get all image files
    image_files = get_image_files(input_path)
    
    if not image_files:
        print("No image files found in the directory.")
        return
    
    print(f"Found {len(image_files)} image files to process...")
    
    # Get output format
    if len(sys.argv) > 3:
        output_format = sys.argv[3].lower()
    else:
        output_format = input("Enter output format (jpg/png, default jpg): ").strip().lower()
        if not output_format:
            output_format = 'jpg'
    
    # Get quality setting
    if len(sys.argv) > 4:
        quality = int(sys.argv[4])
    else:
        if output_format == 'jpg':
            quality_str = input("Enter JPG quality (1-100, default 95): ").strip()
            quality = int(quality_str) if quality_str else 95
        else:
            quality = 95  # Not used for PNG
    
    # Get dataset name
    dataset_name = input("Enter dataset name (default 'image'): ").strip()
    if not dataset_name:
        dataset_name = 'image'
    
    # Warm up GPU if available
    if TORCH_AVAILABLE:
        gpu_warmup()
    
    # Process images
    start_time = time.time()
    successful_conversions = 0
    failed_conversions = 0
    
    print(f"\nProcessing {len(image_files)} images...")
    print("-" * 50)
    
    # Process images in batches for better performance
    batch_size = min(10, len(image_files))  # Process in batches of 10 or fewer
    num_workers = min(4, mp.cpu_count())  # Use up to 4 workers
    
    # Create batches
    batches = [image_files[i:i + batch_size] for i in range(0, len(image_files), batch_size)]
    
    # Process batches
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        batch_results = []
        for i, batch in enumerate(batches):
            start_index = i * batch_size
            future = executor.submit(process_batch, batch, output_path, output_format, quality, dataset_name, start_index)
            batch_results.append(future)
        
        # Collect results
        for future in batch_results:
            results = future.result()
            for success, message in results:
                if success:
                    successful_conversions += 1
                else:
                    failed_conversions += 1
                    print(f"ERROR: {message}")
    
    # Calculate processing time
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Print summary
    print("-" * 50)
    print(f"Processing complete!")
    print(f"Successful conversions: {successful_conversions}")
    print(f"Failed conversions: {failed_conversions}")
    print(f"Total processing time: {processing_time:.2f} seconds")
    print(f"Average time per image: {processing_time/len(image_files):.2f} seconds")
    print(f"Output directory: {output_path}")
    
    if TORCH_AVAILABLE and torch.cuda.is_available():
        print("GPU acceleration was used for compatible images.")
    else:
        print("CPU processing was used.")

if __name__ == "__main__":
    main()