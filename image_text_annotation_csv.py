"""img_text_annotation_csv.py

This script processes image files and their associated text annotations,
compiling them into a metadata CSV file. It can either use annotations provided directly as a dictionary
or process images from a directory and generate a CSV file.

Author: @BorcherdingL
Date: 6/22/2025
"""

import os
import csv
import shutil
from pathlib import Path
from typing import List, Dict, Optional

class ImageAnnotationProcessor:
    """
    A class for processing image files and their associated text annotations,
    compiling them into a metadata CSV file.
    """
    
    def __init__(self, input_dir: str, output_dir: str, annotations: Optional[Dict[str, str]] = None):
        """
        Initialize the processor with input and output directories and optional annotations.
        
        Args:
            input_dir: Directory containing images
            output_dir: Directory where processed files and metadata will be saved
            annotations: Optional dictionary with image filenames as keys and annotation text as values
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.image_extensions = ['.png', '.jpg', '.jpeg', '.heic']
        self.annotations = annotations or {}
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def add_annotations(self, annotations: Dict[str, str]) -> None:
        """
        Add or update annotations for images.
        
        Args:
            annotations: Dictionary with image filenames as keys and annotation text as values
        """
        self.annotations.update(annotations)
    
    def _get_image_files(self) -> List[Path]:
        """
        Find all image files in the input directory.
        
        Returns:
            List of paths to image files
        """
        all_files = list(self.input_dir.glob('*'))
        return [f for f in all_files if f.suffix.lower() in self.image_extensions]
    
    def process_files(self) -> Dict[str, str]:
        """
        Process all image files and copy them to output directory.
        If annotations are provided, they are used; otherwise, empty annotations are created.
        
        Returns:
            Dictionary with image filenames as keys and annotation text as values
        """
        metadata = {}
        image_files = self._get_image_files()
        
        for img_path in image_files:
            # Copy image to output directory
            output_path = self.output_dir / img_path.name
            if not output_path.exists():
                shutil.copy2(img_path, output_path)
            
            # Get annotation or use empty string
            annotation = self.annotations.get(img_path.name, "")
            metadata[img_path.name] = annotation
        
        # Also include annotations for files that might not be in the input directory
        # but are specified in the annotations dictionary
        for filename, annotation in self.annotations.items():
            if filename not in metadata:
                metadata[filename] = annotation
        
        return metadata
    
    def generate_metadata_csv(self) -> str:
        """
        Process files and generate metadata CSV file.
        
        Returns:
            Path to the generated CSV file
        """
        metadata = self.process_files()
        
        # Create metadata.csv file
        csv_path = self.output_dir / "metadata.csv"
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(['file_name', 'text'])
            
            # Write data
            for filename, annotation in metadata.items():
                writer.writerow([filename, annotation])
        
        return str(csv_path)

    @classmethod
    def create_with_annotations(cls, input_dir: str, output_dir: str, annotations: Dict[str, str]) -> 'ImageAnnotationProcessor':
        """
        Create an ImageAnnotationProcessor with pre-defined annotations.
        
        Args:
            input_dir: Directory containing images
            output_dir: Directory where processed files and metadata will be saved
            annotations: Dictionary with image filenames and their annotations
            
        Returns:
            ImageAnnotationProcessor instance
        """
        return cls(input_dir, output_dir, annotations)
    
    @classmethod
    def create_instance(cls, input_dir: str, output_dir: str) -> 'ImageAnnotationProcessor':
        """
        Create an ImageAnnotationProcessor without pre-defined annotations.
        
        Args:
            input_dir: Directory containing images
            output_dir: Directory where processed files and metadata will be saved
            
        Returns:
            ImageAnnotationProcessor instance
        """
        return cls(input_dir, output_dir)


# Example usage:
if __name__ == "__main__":
    # Example with direct annotations
    sample_annotations = {
        "image1.jpg": "A scenic mountain landscape",
        "image2.png": "A busy city street at night"
    }
    
    processor = ImageAnnotationProcessor.create_with_annotations(
        input_dir="./input_images", 
        output_dir="./processed_output",
        annotations=sample_annotations
    )
    
    csv_path = processor.generate_metadata_csv()
    print(f"Metadata CSV generated at: {csv_path}")
