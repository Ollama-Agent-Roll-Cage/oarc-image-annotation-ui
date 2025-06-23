"""image_annotation_ui.py

Modern Image Annotation UI
A sleek PyQt6 interface for image format conversion and text annotation management.

Author: @BorcherdingL
Date: 6/22/2025
"""

import sys
import os
import shutil  # Add this import
from pathlib import Path
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *

# Import your modules
from image_format_master import main as format_images, get_image_files
from img_text_annotation_csv import ImageAnnotationProcessor


class ModernImageAnnotationUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_dir = None
        self.output_dir = None
        self.image_files = []
        self.current_index = 0
        self.annotations = {}
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Image Annotation Studio")
        self.setGeometry(100, 100, 1200, 800)
        
        # Modern dark theme with green/purple accents
        self.setStyleSheet("""
            QMainWindow { background: #1a1a2e; }
            QWidget { background: #16213e; color: #ffffff; font-family: 'Segoe UI'; }
            QPushButton { 
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #16a085, stop:1 #8e44ad);
                border: none; padding: 12px 24px; border-radius: 8px; font-weight: bold;
                color: white; font-size: 14px;
            }
            QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #1abc9c, stop:1 #9b59b6); }
            QPushButton:pressed { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #138d75, stop:1 #7d3c98); }
            QTextEdit { 
                background: #0f1419; border: 2px solid #16a085; border-radius: 8px; 
                padding: 12px; font-size: 13px; color: #ecf0f1;
            }
            QLabel { color: #ecf0f1; font-size: 14px; }
            QFrame { background: #0f1419; border-radius: 12px; }
            QScrollArea { background: transparent; border: none; }
            QToolTip { 
                background-color: #2c3e50; 
                color: #ecf0f1; 
                border: 1px solid #16a085;
                padding: 5px;
                opacity: 230;
            }
        """)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)
        
        # Left panel - Controls
        left_panel = QFrame()
        left_panel.setFixedWidth(300)
        left_layout = QVBoxLayout(left_panel)
        
        # Title
        title = QLabel("🎨 Image Annotation Studio")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #8e44ad; margin: 10px;")
        left_layout.addWidget(title)
        
        # Directory Selection with improved labels and tooltips
        dir_group = QGroupBox("📁 Directories & Modes")
        dir_group.setStyleSheet("QGroupBox { font-weight: bold; color: #16a085; margin-top: 10px; padding-top: 10px; }")
        dir_layout = QVBoxLayout(dir_group)
        
        # Format raw images button with tooltip
        self.format_btn = QPushButton("🔄 Format Raw Images")
        self.format_btn.setToolTip("Select a directory with raw images to convert them to a consistent format.\n"
                                 "Images will be renamed, resized, and converted to JPG or PNG.")
        self.format_btn.clicked.connect(self.format_images)
        
        # Input directory selection with better label and tooltip
        self.select_btn = QPushButton("📂 Load Source Images")
        self.select_btn.setToolTip("Select a directory containing formatted images to annotate.\n"
                                  "These images will be displayed for annotation.")
        self.select_btn.clicked.connect(self.select_directory)
        
        # Output directory selection with better label and tooltip
        self.output_btn = QPushButton("📁 Set Annotation Output")
        self.output_btn.setToolTip("Select where to save your annotation files (TXT) and final CSV.\n"
                                  "This is where the processed dataset will be stored.")
        self.output_btn.clicked.connect(self.select_output_directory)
        
        # Load existing dataset with tooltip
        self.load_existing_btn = QPushButton("📥 Load Existing Dataset")
        self.load_existing_btn.setToolTip("Load a directory containing images and their annotations (TXT files).\n"
                                        "Use this to continue work on an existing annotation set.")
        self.load_existing_btn.clicked.connect(self.load_existing_dataset)
        
        # Help button
        self.help_btn = QPushButton("❓ How To Use")
        self.help_btn.setToolTip("View instructions on how to use this tool")
        self.help_btn.clicked.connect(self.show_help)
        
        dir_layout.addWidget(self.format_btn)
        dir_layout.addWidget(self.select_btn)
        dir_layout.addWidget(self.output_btn)
        dir_layout.addWidget(self.load_existing_btn)
        dir_layout.addWidget(self.help_btn)
        
        # Add status labels to show selected directories
        self.source_label = QLabel("Source: Not selected")
        self.source_label.setStyleSheet("font-size: 12px; color: #7f8c8d; margin-left: 5px;")
        self.output_label = QLabel("Output: Not selected")
        self.output_label.setStyleSheet("font-size: 12px; color: #7f8c8d; margin-left: 5px;")
        
        dir_layout.addWidget(self.source_label)
        dir_layout.addWidget(self.output_label)
        
        left_layout.addWidget(dir_group)
        
        # Navigation
        nav_group = QGroupBox("🧭 Navigation")
        nav_group.setStyleSheet("QGroupBox { font-weight: bold; color: #16a085; margin-top: 10px; padding-top: 10px; }")
        nav_layout = QVBoxLayout(nav_group)
        
        nav_buttons = QHBoxLayout()
        self.prev_btn = QPushButton("◀ Prev")
        self.next_btn = QPushButton("Next ▶")
        self.prev_btn.clicked.connect(self.prev_image)
        self.next_btn.clicked.connect(self.next_image)
        nav_buttons.addWidget(self.prev_btn)
        nav_buttons.addWidget(self.next_btn)
        
        self.image_counter = QLabel("No images loaded")
        self.image_counter.setStyleSheet("color: #8e44ad; font-weight: bold; text-align: center;")
        
        nav_layout.addLayout(nav_buttons)
        nav_layout.addWidget(self.image_counter)
        left_layout.addWidget(nav_group)
        
        # Annotation
        ann_group = QGroupBox("✏️ Annotation")
        ann_group.setStyleSheet("QGroupBox { font-weight: bold; color: #16a085; margin-top: 10px; padding-top: 10px; }")
        ann_layout = QVBoxLayout(ann_group)
        
        self.annotation_text = QTextEdit()
        self.annotation_text.setPlaceholderText("Enter image description here...")
        self.annotation_text.textChanged.connect(self.save_annotation)
        
        ann_layout.addWidget(self.annotation_text)
        
        # Add instruction label below the annotation text area
        hint_label = QLabel("💡 Describe the image as clearly as possible")
        hint_label.setStyleSheet("font-size: 12px; color: #7f8c8d; margin-left: 5px;")
        ann_layout.addWidget(hint_label)
        
        left_layout.addWidget(ann_group)
        
        # Export
        export_group = QGroupBox("💾 Export")
        export_group.setStyleSheet("QGroupBox { font-weight: bold; color: #16a085; margin-top: 10px; padding-top: 10px; }")
        export_layout = QVBoxLayout(export_group)
        
        self.save_txt_btn = QPushButton("💾 Save All TXT Files")
        self.save_txt_btn.setToolTip("Save all annotations as individual TXT files with same filenames as images")
        self.save_csv_btn = QPushButton("📊 Generate Dataset CSV")
        self.save_csv_btn.setToolTip("Create a metadata.csv file that links images to their annotations\n"
                                   "This is the final step to create a complete dataset")
        self.save_txt_btn.clicked.connect(self.save_txt_files)
        self.save_csv_btn.clicked.connect(self.generate_csv)
        
        export_layout.addWidget(self.save_txt_btn)
        export_layout.addWidget(self.save_csv_btn)
        left_layout.addWidget(export_group)
        
        left_layout.addStretch()
        layout.addWidget(left_panel)
        
        # Right panel - Image viewer
        right_panel = QFrame()
        right_layout = QVBoxLayout(right_panel)
        
        self.image_label = QLabel("Select a directory to begin")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel { 
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2c3e50, stop:1 #34495e);
                border: 3px dashed #16a085; border-radius: 12px; 
                color: #8e44ad; font-size: 18px; font-weight: bold;
                min-height: 400px;
            }
        """)
        
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.image_label)
        scroll_area.setWidgetResizable(True)
        
        right_layout.addWidget(scroll_area)
        layout.addWidget(right_panel)
        
        self.show()
    
    def format_images(self):
        """Open a directory of raw images to format and rename them"""
        # Step 1: Initial explanation dialog
        explain_msg = QMessageBox(self)
        explain_msg.setWindowTitle("Format Raw Images")
        explain_msg.setText("This process will help you prepare raw images for annotation.")
        explain_msg.setInformativeText("You'll need to select:\n"
                            "1. Input directory containing your raw images\n"
                            "2. Output directory for saving formatted images\n"
                            "3. Format settings like JPG/PNG and quality")
        explain_msg.setIcon(QMessageBox.Icon.Information)
        explain_msg.exec()

        # Step 2: Select input directory
        input_msg = QMessageBox(self)
        input_msg.setWindowTitle("Select Input Directory")
        input_msg.setText("Please select the folder containing your raw images that need formatting.")
        input_msg.setIcon(QMessageBox.Icon.Information)
        input_msg.exec()
        
        input_dir = QFileDialog.getExistingDirectory(self, "Select Input Directory with Raw Images")
        if not input_dir:
            return

        # Step 3: Select output directory
        output_msg = QMessageBox(self)
        output_msg.setWindowTitle("Select Output Directory")
        output_msg.setText("Please select where you want to save the formatted images.")
        output_msg.setInformativeText("This will also be set as your annotation output directory.")
        output_msg.setIcon(QMessageBox.Icon.Information)
        output_msg.exec()
        
        output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory for Formatted Images")
        if not output_dir:
            return
    
        # Step 4: Choose output format
        output_format, ok = QInputDialog.getItem(self, "Select Format", 
                        "Choose output format:", 
                        ["jpg", "png"], 0, False)
        if not ok:
            return

        # Step 5: Set quality if jpg
        quality = 95  # Default quality for jpg
        if output_format == 'jpg':
            quality, ok = QInputDialog.getInt(self, "JPEG Quality", 
                    "Quality (1-100):", 95, 1, 100, 1)
            if not ok:
                return

        # Step 6: Get dataset name
        dataset_name, ok = QInputDialog.getText(self, "Dataset Name", 
                    "Enter dataset name (used for file naming):", 
                    text="image")
        if not ok or not dataset_name:
            dataset_name = "image"

        # Show progress dialog
        progress = QProgressDialog("Formatting images...", "Cancel", 0, 0, self)
        progress.setWindowTitle("Processing")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()
        QApplication.processEvents()
        
        try:
            # ---- IMPORTANT: MODIFIED TO CALL A CUSTOM VERSION OF THE FUNCTION ----
            from image_format_master import get_image_files
            import time
            
            # Create a modified version of the format_master function that doesn't ask for inputs
            def run_formatter():
                # Setup the same logging and imports from image_format_master
                from image_format_master import setup_heif_support, get_image_files, gpu_warmup
                from image_format_master import convert_image_gpu, process_batch
                import multiprocessing as mp
                from concurrent.futures import ThreadPoolExecutor
                import time
                
                # Set up the directories and parameters
                input_path = Path(input_dir)
                output_path = Path(output_dir)
                os.makedirs(output_path, exist_ok=True)
                
                # Setup HEIF support
                setup_heif_support()
                
                # Get all image files
                image_files = get_image_files(input_path)
                
                if not image_files:
                    return 0, "No image files found in the directory."
                
                # Warm up GPU if available
                if 'TORCH_AVAILABLE' in globals() and TORCH_AVAILABLE:
                    gpu_warmup()
                
                # Process images
                start_time = time.time()
                successful_conversions = 0
                failed_conversions = 0
                
                # Process images in batches for better performance
                batch_size = min(10, len(image_files))
                num_workers = min(4, mp.cpu_count())
                
                # Create batches
                batches = [image_files[i:i + batch_size] for i in range(0, len(image_files), batch_size)]
                
                # Process batches with a timeout to prevent hanging
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    batch_results = []
                    
                    for i, batch in enumerate(batches):
                        start_index = i * batch_size
                        # Use a shorter timeout to prevent UI hanging
                        future = executor.submit(process_batch, batch, output_path, 
                                               output_format, quality, dataset_name, start_index)
                        batch_results.append(future)
                    
                    # Collect results with timeout to avoid hanging
                    for future in batch_results:
                        try:
                            results = future.result(timeout=30)  # 30 second timeout per batch
                            for success, message in results:
                                if success:
                                    successful_conversions += 1
                                else:
                                    failed_conversions += 1
                        except Exception as e:
                            failed_conversions += len(batch)
                            print(f"Batch processing error: {str(e)}")
                
                # Calculate processing time
                end_time = time.time()
                processing_time = end_time - start_time
                
                return successful_conversions, f"Processed {successful_conversions} images in {processing_time:.2f} seconds."
                
            # Run our custom formatter function and get results
            successful_conversions, message = run_formatter()
            
            # Close progress dialog
            progress.close()
            
            # Set both source and output directories automatically
            self.current_dir = Path(output_dir)
            self.output_dir = Path(output_dir)
            
            # Update the UI labels
            self.source_label.setText(f"Source: {self.current_dir}")
            self.output_label.setText(f"Output: {self.output_dir}")
            
            # Give file system time to finish writing
            time.sleep(1)
            
            # Manually check the output directory for new files
            formatted_files = []
            for ext in ['.jpg', '.jpeg', '.png']:
                formatted_files.extend(list(Path(output_dir).glob(f'*{ext}')))
            
            print(f"Found {len(formatted_files)} formatted files in output directory")
            
            # If we found formatted files, set them directly
            if formatted_files:
                self.image_files = sorted(formatted_files)
                self.current_index = 0
                self.annotations = {}
                self.image_counter.setText(f"Image 1 of {len(self.image_files)}")
                self.display_current_image()
                
                # Show completion message
                complete_msg = QMessageBox(self)
                complete_msg.setWindowTitle("Format Complete")
                complete_msg.setText(f"Successfully formatted {len(formatted_files)} images!")
                complete_msg.setInformativeText(
                    f"Images saved to: {output_dir}\n"
                    f"Format: {output_format.upper()}\n"
                    f"Quality: {quality}\n"
                    f"Dataset name: {dataset_name}\n\n"
                    f"This directory is now set as both your source and output location."
                )
                complete_msg.setIcon(QMessageBox.Icon.Information)
                complete_msg.exec()
            else:
                QMessageBox.warning(self, "Warning", "The formatting process completed but no images were found in the output directory.")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            progress.close()
            
            error_box = QMessageBox(self)
            error_box.setWindowTitle("Error")
            error_box.setText(f"Error formatting images: {str(e)}")
            error_box.setDetailedText(traceback.format_exc())
            error_box.setIcon(QMessageBox.Icon.Critical)
            error_box.exec()
    
    def select_directory(self):
        """Select a directory containing already-formatted images to annotate"""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory with Images to Annotate")
        if dir_path:
            self.current_dir = Path(dir_path)
            self.source_label.setText(f"Source: {self.current_dir}")
            self.load_images()
    
    def select_output_directory(self):
        """Select where to save annotation txt files and final CSV"""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Where to Save Annotations and CSV")
        if dir_path:
            self.output_dir = Path(dir_path)
            self.output_label.setText(f"Output: {self.output_dir}")
            QMessageBox.information(self, "Output Directory", f"Output directory set to:\n{dir_path}")
    
    def load_existing_dataset(self):
        """Load a directory that already has images and annotation TXT files"""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory with Existing Dataset")
        if not dir_path:
            return
            
        try:
            # Set both input and output directory to the same location
            self.current_dir = Path(dir_path)
            self.output_dir = Path(dir_path)
            
            # Update labels
            self.source_label.setText(f"Source: {self.current_dir}")
            self.output_label.setText(f"Output: {self.output_dir}")
            
            # Load the images and annotations
            self.load_images()
            
            QMessageBox.information(self, "Dataset Loaded", 
                                   f"Loaded existing dataset with {len(self.image_files)} images and {len(self.annotations)} annotations.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading dataset:\n{str(e)}")
    
    def show_help(self):
        """Show help dialog with workflow explanation"""
        help_text = """
<h2>📝 Image Annotation Tool Guide</h2>

<h3>Quick Start Guide:</h3>

<b>1. Format Raw Images:</b>
   • Click <b>Format Raw Images</b> button
   • Select your input folder with raw images
   • Select where to save the formatted images
   • Choose format (JPG/PNG) and quality settings
   • The tool will automatically set this as both your source and output directory

<b>2. Working with Formatted Images:</b>
   • Use <b>Load Source Images</b> if you already have formatted images
   • Use <b>Set Annotation Output</b> to choose where to save annotations
   • Navigate through images with Prev/Next buttons
   • Type descriptions in the text box - they're saved automatically

<b>3. Continuing Previous Work:</b>
   • Click <b>Load Existing Dataset</b> to continue work on a previous dataset
   • This loads both images and their existing text annotations

<h3>Creating the Final Dataset:</h3>
   • Click <b>Save All TXT Files</b> to save individual annotation text files
   • Click <b>Generate Dataset CSV</b> to create the final metadata.csv file
   
<h3>Pro Tips:</h3>
   • The formatted images, text files, and CSV will all be saved to your output directory
   • Write detailed, descriptive annotations for better AI training results
   • The metadata.csv file connects your images with their annotations
   • You don't need to click Save after each annotation - it happens automatically
"""
    
        help_dialog = QDialog(self)
        help_dialog.setWindowTitle("How to Use This Tool")
        help_dialog.setMinimumSize(600, 500)
        
        layout = QVBoxLayout(help_dialog)
        
        help_text_browser = QTextBrowser()
        help_text_browser.setHtml(help_text)
        help_text_browser.setOpenExternalLinks(True)
        
        close_button = QPushButton("Close")
        close_button.clicked.connect(help_dialog.accept)
        
        layout.addWidget(help_text_browser)
        layout.addWidget(close_button)
        
        help_dialog.setLayout(layout)
        help_dialog.exec()
    
    def load_images(self):
        if not self.current_dir:
            return
    
        # Ensure output directory is set
        if not self.output_dir:
            reply = QMessageBox.question(self, "No Output Directory", 
                                       "No output directory has been set. Would you like to set one now?",
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            
            if reply == QMessageBox.StandardButton.Yes:
                self.select_output_directory()
            else:
                QMessageBox.warning(self, "Operation Cancelled", "Please set an output directory before continuing.")
                return
    
        # If we still don't have an output directory, stop
        if not self.output_dir:
            return
            
        # Get all image files from the selected directory using our helper function
        try:
            # Debug message to verify what directory we're searching
            print(f"Searching for images in: {self.current_dir}")
            
            # Make sure we're using Path objects consistently
            self.current_dir = Path(self.current_dir)
            
            # Get image files using our helper function
            self.image_files = get_image_files(self.current_dir)
            
            print(f"Found {len(self.image_files)} images")
            
            if not self.image_files:
                # Check if directory exists
                if not os.path.exists(self.current_dir):
                    QMessageBox.critical(self, "Error", f"Directory doesn't exist: {self.current_dir}")
                    return
                    
                # List contents of directory to debug
                files = list(os.listdir(self.current_dir))
                print(f"Directory contents: {files[:10]}...")
                
                QMessageBox.information(self, "No Images", "No image files found in the selected directory.")
                return
                
            # Reset to the first image
            self.current_index = 0
            
            # Load existing annotations if available
            self.load_existing_annotations()
            
            # Update the image counter
            self.image_counter.setText(f"Image 1 of {len(self.image_files)}")
            
            # Display the first image
            self.display_current_image()
            
            QMessageBox.information(self, "Images Loaded", f"Successfully loaded {len(self.image_files)} images and {len(self.annotations)} annotations.")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Error loading images:\n{str(e)}")

    def load_existing_annotations(self):
        """Load existing annotations from txt files if available"""
        self.annotations = {}  # Reset annotations
        
        for img_path in self.image_files:
            # Check if there's a corresponding txt file in the output directory
            txt_path = self.output_dir / f"{img_path.stem}.txt"
            if txt_path.exists():
                try:
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        self.annotations[img_path.name] = f.read()
                except Exception as e:
                    print(f"Error loading annotation for {img_path.name}: {str(e)}")

    def display_current_image(self):
        """Display the current image and its annotation"""
        if not self.image_files or self.current_index >= len(self.image_files):
            return
            
        img_path = self.image_files[self.current_index]
        
        try:
            # Load and display the image
            pixmap = QPixmap(str(img_path))
            
            if pixmap.isNull():
                self.image_label.setText(f"Error loading image: {img_path.name}")
                return
            
            # Get the size of the scroll area viewport
            scroll_width = self.image_label.parentWidget().width() - 20  # subtract padding
            scroll_height = self.image_label.parentWidget().height() - 20  # subtract padding
            
            # Scale the image to fit the label while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                scroll_width, 
                scroll_height,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            
            self.image_label.setPixmap(scaled_pixmap)
            
            # Update the image counter
            self.image_counter.setText(f"Image {self.current_index + 1} of {len(self.image_files)}")
            
            # Load annotation if available
            current_annotation = self.annotations.get(img_path.name, "")
            
            # Block signals temporarily to avoid recursive signal triggering
            self.annotation_text.blockSignals(True)
            self.annotation_text.setPlainText(current_annotation)
            self.annotation_text.blockSignals(False)
            
        except Exception as e:
            self.image_label.setText(f"Error loading image:\n{str(e)}")

    def next_image(self):
        """Navigate to the next image"""
        if not self.image_files:
            return
            
        # Save current annotation before moving
        self.save_annotation()
        
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.display_current_image()
        else:
            # Optionally wrap around to the first image
            QMessageBox.information(self, "End of Images", "You've reached the last image.")

    def prev_image(self):
        """Navigate to the previous image"""
        if not self.image_files:
            return
            
        # Save current annotation before moving
        self.save_annotation()
        
        if self.current_index > 0:
            self.current_index -= 1
            self.display_current_image()
        else:
            # Optionally wrap around to the last image
            QMessageBox.information(self, "Start of Images", "You're at the first image.")

    def save_annotation(self):
        """Save the current annotation"""
        if not self.image_files or self.current_index >= len(self.image_files):
            return
            
        current_text = self.annotation_text.toPlainText()
        img_path = self.image_files[self.current_index]
        self.annotations[img_path.name] = current_text

    def save_txt_files(self):
        """Save all annotations as individual txt files"""
        if not self.output_dir or not self.annotations:
            QMessageBox.warning(self, "No Annotations", "No annotations to save.")
            return
            
        try:
            # Create output directory if it doesn't exist
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Save each annotation as a txt file
            saved_count = 0
            for filename, text in self.annotations.items():
                # Skip empty annotations
                if not text.strip():
                    continue
                    
                # Create txt file with the same base name
                txt_path = self.output_dir / f"{Path(filename).stem}.txt"
                
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                    saved_count += 1
                    
            QMessageBox.information(self, "Success", f"Saved {saved_count} annotation files to {self.output_dir}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saving annotation files:\n{str(e)}")

    def generate_csv(self):
        """Generate a CSV file with all annotations using ImageAnnotationProcessor"""
        if not self.current_dir or not self.output_dir:
            QMessageBox.warning(self, "Missing Directories", "Please select both input and output directories.")
            return
            
        try:
            # Save current annotation first to ensure it's up to date
            if self.image_files and self.current_index < len(self.image_files):
                self.save_annotation()
                
            # Use our ImageAnnotationProcessor to generate the CSV
            processor = ImageAnnotationProcessor.create_with_annotations(
                input_dir=str(self.current_dir),
                output_dir=str(self.output_dir),
                annotations=self.annotations
            )
            
            csv_path = processor.generate_metadata_csv()
            QMessageBox.information(self, "Success", f"Dataset CSV generated at:\n{csv_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error generating CSV:\n{str(e)}")

    def format_images_direct(self):
        """Modified version that directly processes images without relying on terminal input"""
        # Get all necessary parameters from GUI first
        
        # Step 1: Initial explanation dialog
        explain_msg = QMessageBox(self)
        explain_msg.setWindowTitle("Format Raw Images")
        explain_msg.setText("This process will help you prepare raw images for annotation.")
        explain_msg.setInformativeText("You'll need to select:\n"
                            "1. Input directory containing your raw images\n"
                            "2. Output directory for saving formatted images\n"
                            "3. Format settings like JPG/PNG and quality")
        explain_msg.setIcon(QMessageBox.Icon.Information)
        explain_msg.exec()

        # Step 2: Select input directory
        input_msg = QMessageBox(self)
        input_msg.setWindowTitle("Select Input Directory")
        input_msg.setText("Please select the folder containing your raw images that need formatting.")
        input_msg.setIcon(QMessageBox.Icon.Information)
        input_msg.exec()
        
        input_dir = QFileDialog.getExistingDirectory(self, "Select Input Directory with Raw Images")
        if not input_dir:
            return

        # Step 3: Select output directory
        output_msg = QMessageBox(self)
        output_msg.setWindowTitle("Select Output Directory")
        output_msg.setText("Please select where you want to save the formatted images.")
        output_msg.setInformativeText("This will also be set as your annotation output directory.")
        output_msg.setIcon(QMessageBox.Icon.Information)
        output_msg.exec()
        
        output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory for Formatted Images")
        if not output_dir:
            return
    
        # Step 4: Choose output format
        output_format, ok = QInputDialog.getItem(self, "Select Format", 
                        "Choose output format:", 
                        ["jpg", "png"], 0, False)
        if not ok:
            return

        # Step 5: Set quality if jpg
        quality = 95  # Default quality for jpg
        if output_format == 'jpg':
            quality, ok = QInputDialog.getInt(self, "JPEG Quality", 
                    "Quality (1-100):", 95, 1, 100, 1)
            if not ok:
                return

        # Step 6: Get dataset name
        dataset_name, ok = QInputDialog.getText(self, "Dataset Name", 
                    "Enter dataset name (used for file naming):", 
                    text="image")
        if not ok or not dataset_name:
            dataset_name = "image"

        # Show progress dialog
        progress = QProgressDialog("Formatting images...", "Cancel", 0, 0, self)
        progress.setWindowTitle("Processing")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()
        QApplication.processEvents()
        
        try:
            # Import directly from image_format_master
            from image_format_master import setup_heif_support, get_image_files, gpu_warmup
            from image_format_master import convert_image_gpu
            import os
            import time
            import concurrent.futures
            import multiprocessing as mp
            from pathlib import Path
            
            # Initialize directories
            input_path = Path(input_dir)
            output_path = Path(output_dir)
            os.makedirs(output_path, exist_ok=True)
            
            # Setup HEIF support
            setup_heif_support()
            
            # Get all image files
            image_files = get_image_files(input_path)
            
            if not image_files:
                progress.close()
                QMessageBox.warning(self, "Warning", "No image files found in the input directory.")
                return
                
            # Warm up GPU
            gpu_warmup()
            
            # Process images directly, one at a time
            successful_conversions = 0
            failed_conversions = 0
            
            # Update progress dialog
            progress.setMaximum(len(image_files))
            progress.setValue(0)
            
            for i, img_path in enumerate(image_files):
                # Check for cancel
                if progress.wasCanceled():
                    break
                    
                # Convert image
                image_index = i + 1  # Start from 1
                try:
                    success, message = convert_image_gpu(img_path, output_path, output_format, 
                                                       quality, dataset_name, image_index)
                    if success:
                        successful_conversions += 1
                    else:
                        failed_conversions += 1
                except Exception as e:
                    failed_conversions += 1
                    print(f"Error converting {img_path}: {str(e)}")
                    
                # Update progress
                progress.setValue(i + 1)
                progress.setLabelText(f"Processing image {i+1} of {len(image_files)}...")
                QApplication.processEvents()
                
            # Close progress
            progress.close()
            
            # Set both source and output directories automatically
            self.current_dir = Path(output_dir)
            self.output_dir = Path(output_dir)
            
            # Update the UI labels
            self.source_label.setText(f"Source: {self.current_dir}")
            self.output_label.setText(f"Output: {self.output_dir}")
            
            # Give file system time to finish writing
            time.sleep(1)
            
            # Load the formatted images
            formatted_files = get_image_files(output_path)
            
            if formatted_files:
                self.image_files = formatted_files
                self.current_index = 0
                self.annotations = {}
                self.image_counter.setText(f"Image 1 of {len(self.image_files)}")
                self.display_current_image()
                
                # Show completion message
                QMessageBox.information(self, "Format Complete", 
                    f"Successfully formatted {successful_conversions} images!\n\n"
                    f"Failed: {failed_conversions}\n"
                    f"Format: {output_format.upper()}\n"
                    f"Quality: {quality}\n"
                    f"Dataset name: {dataset_name}")
            else:
                QMessageBox.warning(self, "Warning", "No images found in the output directory after formatting.")
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            progress.close()
            
            error_box = QMessageBox(self)
            error_box.setWindowTitle("Error")
            error_box.setText(f"Error formatting images: {str(e)}")
            error_box.setDetailedText(traceback.format_exc())
            error_box.setIcon(QMessageBox.Icon.Critical)
            error_box.exec()

    # Add this method to handle window resize events
    def resizeEvent(self, event):
        """Handle window resize events to update image display"""
        super().resizeEvent(event)
        # Re-display the current image to fit the new size
        self.display_current_image()

if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        window = ModernImageAnnotationUI()
        sys.exit(app.exec())
    except Exception as e:
        print(f"Critical error: {str(e)}")
        # If we have QApplication initialized, show error dialog
        if 'app' in locals():
            QMessageBox.critical(None, "Critical Error", f"Application crashed: {str(e)}")
            sys.exit(1)