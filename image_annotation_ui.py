"""image_annotation_ui.py

Modern Image Annotation UI - FINAL FIXED VERSION
A sleek PyQt6 interface for image format conversion and text annotation management.

Author: @BorcherdingL
Date: 6/22/2025
FIXED: All layout issues, animation problems, and Qt stylesheet errors
"""

import sys
import os
import time
import math
import random
import asyncio
import base64
import traceback
import multiprocessing as mp
from pathlib import Path
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *

# Add numpy import for noise generation
import numpy as np

# Import your modules
from image_format_master import main as format_images, get_image_files
from image_text_annotation_csv import ImageAnnotationProcessor
from ollama_tools import OllamaCommands
from concurrent.futures import ThreadPoolExecutor

class PerlinNoiseBackground(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.noise_texture = None
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.animate_noise)
        self.time_offset = 0.0
        
        # Animation speed controls
        self.animation_speed = 2.0
        self.timer_interval = 150
        
        # Flow strength controls
        self.flow_strength = 0.3
        self.vortex_strength = 0.5
        self.turbulence_scale = 1.0
        self.flow_speed_multiplier = 1.0
        
        # Create permutation table for proper Perlin noise
        self.permutation = list(range(256))
        random.shuffle(self.permutation)
        self.permutation *= 2
        
        # Pre-compute gradient vectors for better performance
        self.grad_table = self._precompute_gradients()
        
        # FIXED: Initialize with a default size first
        self.setMinimumSize(800, 600)
        
        # Initialize the texture - this will create a valid pixmap
        self.generate_noise_texture()
        
        # Start animation timer only if we have a valid texture
        if self.noise_texture and not self.noise_texture.isNull():
            self.animation_timer.start(self.timer_interval)

    def _precompute_gradients(self):
        """Pre-compute gradient vectors to avoid runtime calculations"""
        gradients = []
        for i in range(256):
            angle = (i / 256.0) * 2 * math.pi
            gradients.append((math.cos(angle), math.sin(angle)))
        return gradients
    
    def fade(self, t):
        """Optimized fade function"""
        return t * t * t * (t * (t * 6 - 15) + 10)
    
    def lerp(self, a, b, t):
        """Linear interpolation"""
        return a + t * (b - a)
    
    def grad(self, hash_val, x, y):
        """Optimized gradient function using pre-computed table"""
        grad_x, grad_y = self.grad_table[hash_val & 255]
        return grad_x * x + grad_y * y
    
    def perlin_noise_2d(self, x, y):
        """2D Perlin noise implementation"""
        X = int(math.floor(x)) & 255
        Y = int(math.floor(y)) & 255
        
        x = x - math.floor(x)
        y = y - math.floor(y)
        
        u = self.fade(x)
        v = self.fade(y)
        
        perm = self.permutation
        A = perm[X] + Y
        AA = perm[A & 255]
        AB = perm[(A + 1) & 255]
        B = perm[(X + 1) & 255] + Y
        BA = perm[B & 255]
        BB = perm[(B + 1) & 255]
        
        return self.lerp(
            self.lerp(
                self.grad(AA, x, y),
                self.grad(BA, x - 1, y),
                u
            ),
            self.lerp(
                self.grad(AB, x, y - 1),
                self.grad(BB, x - 1, y - 1),
                u
            ),
            v
        )
        
    def generate_noise_texture(self):
        """Generate noise texture with configurable turbulent flow patterns"""
        # FIXED: Get actual widget size, with fallback to minimum size
        widget_size = self.size()
        if widget_size.width() <= 0 or widget_size.height() <= 0:
            widget_size = self.minimumSize()
        
        # Scale down for performance (can adjust these values)
        width = max(100, widget_size.width() // 14)  # Scale factor of ~14
        height = max(80, widget_size.height() // 14)
        
        # Create numpy array for the texture
        try:
            noise_array = np.zeros((height, width, 3), dtype=np.uint8)
        except Exception as e:
            print(f"Error creating noise array: {e}")
            return
        
        # Multiple time factors for complex movement
        time_factor = self.time_offset * 0.02 * self.flow_speed_multiplier
        
        try:
            for y in range(height):
                for x in range(width):
                    # Normalized coordinates (0-1)
                    nx = x / width
                    ny = y / height
                    
                    # Configurable flow field using sine/cosine functions
                    flow_x = math.sin(nx * math.pi * 2 * self.turbulence_scale + time_factor) * math.cos(ny * math.pi * 3 * self.turbulence_scale)
                    flow_y = math.cos(nx * math.pi * 3 * self.turbulence_scale + time_factor) * math.sin(ny * math.pi * 2 * self.turbulence_scale)
                    
                    # Configurable vortex patterns
                    vortex1_x, vortex1_y = 0.3, 0.3
                    vortex2_x, vortex2_y = 0.7, 0.7
                    
                    # Distance and angle from vortex centers
                    d1 = math.sqrt((nx - vortex1_x)**2 + (ny - vortex1_y)**2)
                    a1 = math.atan2(ny - vortex1_y, nx - vortex1_x)
                    
                    d2 = math.sqrt((nx - vortex2_x)**2 + (ny - vortex2_y)**2)
                    a2 = math.atan2(ny - vortex2_y, nx - vortex2_x)
                    
                    # Configurable vortex influence
                    base_vortex_strength1 = 1.0 / (1.0 + d1 * 5)
                    base_vortex_strength2 = 1.0 / (1.0 + d2 * 5)
                    
                    vortex_strength1 = base_vortex_strength1 * self.vortex_strength
                    vortex_strength2 = base_vortex_strength2 * self.vortex_strength
                    
                    # Rotating vortex motion with configurable strength
                    vortex1_x_offset = math.cos(a1 + time_factor * 2) * vortex_strength1 * 0.5
                    vortex1_y_offset = math.sin(a1 + time_factor * 2) * vortex_strength1 * 0.5
                    
                    vortex2_x_offset = math.cos(a2 - time_factor * 1.5) * vortex_strength2 * 0.5
                    vortex2_y_offset = math.sin(a2 - time_factor * 1.5) * vortex_strength2 * 0.5
                    
                    # Combine all movement influences with configurable strength
                    final_x = x * 0.1 + (flow_x * self.flow_strength) + vortex1_x_offset + vortex2_x_offset
                    final_y = y * 0.1 + (flow_y * self.flow_strength) + vortex1_y_offset + vortex2_y_offset
                    
                    # Multi-octave noise with the complex movement
                    noise_val = (
                        self.perlin_noise_2d(final_x, final_y) * 0.6 +
                        self.perlin_noise_2d(final_x * 2, final_y * 2) * 0.3 +
                        self.perlin_noise_2d(final_x * 4, final_y * 4) * 0.1
                    )
                    
                    # Normalize to 0-1
                    noise_val = (noise_val + 1) * 0.5
                    noise_val = max(0.0, min(1.0, noise_val))
                    
                    # Color mapping with configurable flow influence
                    flow_color_influence = flow_x * 0.2 * self.flow_strength
                    color_cycle = (noise_val * 2 + time_factor * 0.5 + flow_color_influence) % 1.0
                    
                    if color_cycle < 0.33:
                        # Blue to purple
                        r = int(30 + noise_val * 60)
                        g = int(50 + noise_val * 80)
                        b = int(120 + noise_val * 100)
                    elif color_cycle < 0.66:
                        # Purple to teal
                        r = int(60 + noise_val * 40)
                        g = int(100 + noise_val * 120)
                        b = int(150 + noise_val * 80)
                    else:
                        # Teal to blue
                        r = int(20 + noise_val * 80)
                        g = int(80 + noise_val * 100)
                        b = int(180 + noise_val * 60)
                    
                    # Apply brightness variation
                    brightness = 0.7 + noise_val * 0.6
                    r = int(min(255, r * brightness))
                    g = int(min(255, g * brightness))
                    b = int(min(255, b * brightness))
                    
                    noise_array[y, x] = [r, g, b]
            
            # FIXED: Better error handling for QImage creation
            bytes_per_line = 3 * width
            q_image = QImage(noise_array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            
            # Check if QImage was created successfully
            if q_image.isNull():
                print("Error: Failed to create QImage")
                return
                
            # Convert to QPixmap with error checking
            pixmap = QPixmap.fromImage(q_image)
            if pixmap.isNull():
                print("Error: Failed to create QPixmap from QImage")
                return
                
            self.noise_texture = pixmap
            
        except Exception as e:
            print(f"Error generating noise texture: {e}")
            # Create a simple fallback texture
            self.create_fallback_texture(width, height)

    def create_fallback_texture(self, width, height):
        """Create a simple fallback texture if noise generation fails"""
        try:
            # Create a simple gradient as fallback
            fallback_array = np.zeros((height, width, 3), dtype=np.uint8)
            for y in range(height):
                for x in range(width):
                    # Simple blue gradient
                    intensity = int((x + y) / (width + height) * 255)
                    fallback_array[y, x] = [30, 50 + intensity // 4, 120 + intensity // 2]
            
            bytes_per_line = 3 * width
            q_image = QImage(fallback_array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            self.noise_texture = QPixmap.fromImage(q_image)
            
        except Exception as e:
            print(f"Error creating fallback texture: {e}")
            # Ultimate fallback - solid color
            self.noise_texture = QPixmap(width, height)
            self.noise_texture.fill(QColor(30, 50, 120))

    # NEW: Methods to control flow effects
    def set_flow_strength(self, strength):
        """Set overall flow field strength (0.0 to 1.0)"""
        self.flow_strength = max(0.0, min(1.0, strength))
    
    def set_vortex_strength(self, strength):
        """Set vortex influence strength (0.0 to 1.0)"""
        self.vortex_strength = max(0.0, min(1.0, strength))
    
    def set_turbulence_scale(self, scale):
        """Set turbulence pattern scale (0.1 to 3.0)"""
        self.turbulence_scale = max(0.1, min(3.0, scale))
    
    def set_flow_speed_multiplier(self, multiplier):
        """Set how fast flow patterns move (0.1 to 3.0)"""
        self.flow_speed_multiplier = max(0.1, min(3.0, multiplier))
    
    def set_flow_preset(self, preset_name):
        """Set predefined flow configurations"""
        presets = {
            "calm": {
                "flow_strength": 0.1,
                "vortex_strength": 0.2,
                "turbulence_scale": 0.8,
                "flow_speed_multiplier": 0.5
            },
            "moderate": {
                "flow_strength": 0.3,
                "vortex_strength": 0.5,
                "turbulence_scale": 1.0,
                "flow_speed_multiplier": 1.0
            },
            "turbulent": {
                "flow_strength": 0.7,
                "vortex_strength": 0.8,
                "turbulence_scale": 1.5,
                "flow_speed_multiplier": 1.5
            },
            "chaotic": {
                "flow_strength": 1.0,
                "vortex_strength": 1.0,
                "turbulence_scale": 2.0,
                "flow_speed_multiplier": 2.0
            }
        }
        
        if preset_name in presets:
            preset = presets[preset_name]
            self.flow_strength = preset["flow_strength"]
            self.vortex_strength = preset["vortex_strength"]
            self.turbulence_scale = preset["turbulence_scale"]
            self.flow_speed_multiplier = preset["flow_speed_multiplier"]

    def animate_noise(self):
        """MODIFIED: Animate with configurable speed"""
        self.time_offset += self.animation_speed  # Use configurable speed
        if self.time_offset > 1000:
            self.time_offset = 0
        
        self.generate_noise_texture()
        self.update()
    
    def set_animation_speed(self, speed):
        """Set animation speed (0.1 = very slow, 5.0 = very fast)"""
        self.animation_speed = max(0.1, min(10.0, speed))
    
    def set_timer_interval(self, interval_ms):
        """Set timer interval in milliseconds (lower = smoother)"""
        self.timer_interval = max(16, min(500, interval_ms))  # 16ms = ~60fps, 500ms = 2fps
        self.animation_timer.stop()
        self.animation_timer.start(self.timer_interval)
    
    def paintEvent(self, event):
        """FIXED: Robust paint event handling with better error checking"""
        # Early return if no texture or invalid texture
        if not self.noise_texture or self.noise_texture.isNull():
            return
            
        # Check if widget has valid size
        widget_size = self.size()
        if widget_size.width() <= 0 or widget_size.height() <= 0:
            return
        
        painter = QPainter()
        
        # FIXED: Better painter initialization with error checking
        if not painter.begin(self):
            print("Error: Failed to begin painter")
            return
            
        try:
            # Set high-quality rendering hints
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            
            # Scale texture to widget size with error checking
            scaled_texture = self.noise_texture.scaled(
                widget_size, 
                Qt.AspectRatioMode.IgnoreAspectRatio, 
                Qt.TransformationMode.FastTransformation
            )
            
            # Check if scaling was successful
            if not scaled_texture.isNull():
                painter.drawPixmap(0, 0, scaled_texture)
            else:
                # Fallback: draw the original texture
                painter.drawPixmap(0, 0, self.noise_texture)
                
        except Exception as e:
            print(f"Error in paintEvent: {e}")
        finally:
            # FIXED: Always end the painter properly
            if painter.isActive():
                painter.end()

    def resizeEvent(self, event):
        """Handle resize events to regenerate texture if needed"""
        super().resizeEvent(event)
        # Regenerate texture on significant size changes
        if event.size().width() > 0 and event.size().height() > 0:
            # Only regenerate if size changed significantly (more than 50 pixels in any dimension)
            old_size = event.oldSize()
            new_size = event.size()
            
            if (abs(new_size.width() - old_size.width()) > 50 or 
                abs(new_size.height() - old_size.height()) > 50):
                # Delay regeneration to avoid excessive updates during window resizing
                if hasattr(self, '_resize_timer'):
                    self._resize_timer.stop()
                else:
                    self._resize_timer = QTimer()
                    self._resize_timer.setSingleShot(True)
                    self._resize_timer.timeout.connect(self.generate_noise_texture)
                
                self._resize_timer.start(200)  # 200ms delay

class ModernImageAnnotationUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_dir = None
        self.output_dir = None
        self.image_files = []
        self.current_index = 0
        self.annotations = {}
        
        # Initialize Ollama integration
        self.ollama_commands = OllamaCommands()
        self.available_models = []
        self.selected_model = None
        self.ai_annotation_enabled = False
        
        # Add system prompt setting with default value
        self.system_prompt = "Describe this image in detail. Focus on the main subjects, objects, actions, colors, composition, and any notable features. Be descriptive but concise."
        
        # Threading for async operations
        self.thread_pool = ThreadPoolExecutor(max_workers=2)
        
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Image Annotation Studio")
        self.setGeometry(100, 100, 1400, 900)
        
        # FIXED: Simplified Qt-compatible stylesheet with faster hover transitions
        self.setStyleSheet("""
            QMainWindow { 
                background-color: #1a237e; 
            }
            QWidget { 
                background-color: rgba(10, 15, 25, 180); 
                color: #FFFFFF; 
                font-family: 'Segoe UI', Arial, sans-serif;
                font-weight: 500;
                border-radius: 8px;
                border: 1px solid rgba(255, 255, 255, 40);
            }
            QPushButton:not([objectName="ai_button"]) { 
                background-color: rgba(50, 80, 120, 180);
                border: 2px solid rgba(100, 149, 237, 150); 
                padding: 10px 15px; 
                color: #FFFFFF; 
                font-size: 12px;
                font-weight: 600;
                border-radius: 6px;
                min-height: 25px;
                min-width: 120px;
            }
            QPushButton:hover { 
                background-color: rgba(70, 110, 160, 220); 
                border-color: rgba(135, 206, 250, 220);
            }
            QPushButton:pressed {
                background-color: rgba(40, 70, 110, 240);
            }
            QPushButton:disabled {
                background-color: rgba(60, 60, 60, 180);
                color: #AAAAAA;
                border-color: rgba(100, 100, 100, 120);
            }
            QTextEdit { 
                background-color: rgba(255, 255, 255, 220); 
                border: 2px solid rgba(135, 206, 250, 180); 
                padding: 12px; 
                font-size: 12px; 
                color: #1a1a1a;
                border-radius: 6px;
                min-height: 120px;
            }
            QTextEdit:focus {
                border-color: rgba(100, 149, 237, 220);
                background-color: rgba(255, 255, 255, 240);
            }
            QLabel { 
                color: #FFFFFF; 
                font-size: 12px; 
                background-color: transparent;
                font-weight: 600;
                padding: 4px;
            }
            QGroupBox {
                font-weight: bold; 
                color: #FFFFFF; 
                margin-top: 15px; 
                padding-top: 20px;
                padding-left: 10px;
                padding-right: 10px;
                padding-bottom: 10px;
                border: 2px solid rgba(255, 255, 255, 60);
                border-radius: 8px;
                background-color: rgba(15, 25, 40, 200);
                font-size: 16px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 5px 10px;
                background-color: rgba(20, 30, 50, 220);
                border-radius: 4px;
                color: #FFFFFF;
                border: 1px solid rgba(255, 255, 255, 80);
            }
            QComboBox { 
                background-color: rgba(255, 255, 255, 220); 
                border: 2px solid rgba(135, 206, 250, 180); 
                padding: 8px 8px; 
                color: #1a1a1a; 
                font-size: 12px;
                border-radius: 4px;
                font-weight: 600;
                min-height: 20px;
                min-width: 150px;
            }
            QComboBox:hover {
                border-color: rgba(100, 149, 237, 240);
                background-color: rgba(255, 255, 255, 250);
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
                background-color: rgba(135, 206, 250, 200);
                border-radius: 2px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 4px solid #1a1a1a;
                margin-right: 5px;
            }
            QComboBox QAbstractItemView {
                background-color: rgba(255, 255, 255, 230);
                color: #1a1a1a;
                selection-background-color: rgba(135, 206, 250, 180);
                border: 2px solid rgba(135, 206, 250, 200);
                border-radius: 4px;
                font-size: 12px;
            }
            QScrollArea {
                background-color: transparent;
                border: none;
                border-radius: 8px;
            }
            QSplitter::handle {
                background-color: rgba(100, 149, 237, 150);
                border: 1px solid rgba(255, 255, 255, 60);
                border-radius: 3px;
            }
            QSplitter::handle:horizontal {
                width: 6px;
                margin: 2px 0px;
            }
            QSplitter::handle:horizontal:hover {
                background-color: rgba(135, 206, 250, 220);
            }
            QCheckBox {
                color: #FFFFFF;
                font-size: 12px;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 3px;
                border: 2px solid rgba(100, 149, 237, 150);
                background-color: rgba(255, 255, 255, 200);
            }
            QCheckBox::indicator:checked {
                background-color: rgba(100, 149, 237, 200);
                border-color: rgba(135, 206, 250, 200);
            }
            QCheckBox::indicator:hover {
                border-color: rgba(135, 206, 250, 220);
            }
            QSlider::groove:horizontal {
                border: 1px solid rgba(255, 255, 255, 60);
                height: 6px;
                background: rgba(50, 80, 120, 180);
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: rgba(135, 206, 250, 200);
                border: 2px solid rgba(100, 149, 237, 150);
                width: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: rgba(135, 206, 250, 250);
                border-color: rgba(135, 206, 250, 220);
            }
            QDialog {
                background-color: rgba(10, 15, 25, 200);
                border: 2px solid rgba(255, 255, 255, 60);
                border-radius: 12px;
            }
        """)
        
        central_widget = QWidget()
        central_widget.setStyleSheet("background-color: transparent;")
        self.setCentralWidget(central_widget)
        
        # MODIFIED: Use QSplitter for resizable layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Create horizontal splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)  # Prevent panels from collapsing completely
        
        # Left panel - FIXED: Explicit size and spacing
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # Right panel - Image viewer
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        # Set initial splitter sizes (left panel smaller, right panel larger)
        splitter.setSizes([400, 1000])  # Left: 400px, Right: 1000px
        splitter.setStretchFactor(0, 0)  # Left panel doesn't stretch
        splitter.setStretchFactor(1, 1)  # Right panel stretches
        
        main_layout.addWidget(splitter)
        
        # Create noise background AFTER the layout is set up
        self.noise_background = PerlinNoiseBackground(self)
        
        # MODIFIED: Enable background animation by default and ensure it's visible
        self.noise_enabled = True
        self.noise_background.show()
        self.noise_background.lower()  # Put behind other widgets
    
        # Set custom animation speed
        self.noise_background.set_animation_speed(1.5)  # Moderate speed
        self.noise_background.set_timer_interval(75)    # ~13 FPS for smooth but efficient animation
        
        # OPTIMIZATION 6: Reduce animation quality during heavy operations
        self.noise_background.set_animation_speed(1.2)  # Slightly slower
        self.noise_background.set_timer_interval(100)   # Reduce to ~10 FPS for better performance
    
        # OPTIMIZATION 7: Add animation priority management
        self.animation_paused = False
        
        # Show the window
        self.show()
        
        # MODIFIED: Set initial size after window is shown
        QTimer.singleShot(100, self.update_noise_size)  # Small delay to ensure window is fully rendered
        
        # Initialize Ollama models
        QTimer.singleShot(500, self.refresh_ollama_models)

    def resizeEvent(self, event):
        """Handle window resize to update noise background size"""
        super().resizeEvent(event)
        if hasattr(self, 'noise_background'):
            self.update_noise_size()

    def update_noise_size(self):
        """Update the noise background size to match the window"""
        if hasattr(self, 'noise_background'):
            # Get the actual window size
            window_size = self.size()
            self.noise_background.setGeometry(0, 0, window_size.width(), window_size.height())
            print(f"Updated noise background size to: {window_size.width()}x{window_size.height()}")

    def showEvent(self, event):
        """Handle show event to ensure proper noise background sizing"""
        super().showEvent(event)
        if hasattr(self, 'noise_background'):
            # Small delay to ensure window is fully rendered
            QTimer.singleShot(50, self.update_noise_size)

    def create_directory_section(self):
        """MODIFIED: Directory section with properly organized layout to prevent overlapping"""
        dir_group = QGroupBox("📁 Directories and Modes")
        dir_group.setStyleSheet("""
            QGroupBox {
                background-color: rgba(15, 25, 40, 200);
                border: 2px solid rgba(255, 255, 255, 60);
                border-radius: 10px;
                padding: 10px;
            }
        """)
        # FIXED: Remove the problematic stylesheet
        layout = QVBoxLayout(dir_group)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 15, 10, 10)
        
        # MODIFIED: Create 2x2 grid layout directly without container widget
        grid_layout = QGridLayout()
        grid_layout.setSpacing(8)
        grid_layout.setHorizontalSpacing(5)  # Space between columns
        grid_layout.setVerticalSpacing(30)    # Space between rows
        
        # Button configuration
        button_height = 35
        button_width = 140
        buttons = [
            ("🔄 Format Images", self.format_images, "Select a directory with raw images to convert them to a consistent format"),
            ("📂 Load Images", self.select_directory, "Select a directory containing formatted images to annotate"),
            ("📁 Set Output", self.select_output_directory, "Select where to save your annotation files and final CSV"),
            ("📥 Load Dataset", self.load_existing_dataset, "Load a directory containing images and their annotations"),
        ]
        
        # Create buttons with fixed sizes
        for i, (text, handler, tooltip) in enumerate(buttons):
            btn = QPushButton(text)
            btn.setToolTip(tooltip)
            btn.clicked.connect(handler)
            
            # Fixed size to prevent expansion/overlap
            btn.setFixedSize(button_width, button_height)
            btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            
            # Custom styling for better appearance
            btn.setStyleSheet("""
                QPushButton {
                    background-color: rgba(50, 80, 120, 230);
                    border: 2px solid rgba(100, 149, 237, 150);
                    padding: 8px 8px;
                    color: #FFFFFF;
                    font-size: 12px;
                    font-weight: bold; 
                    border-radius: 6px;
                    text-align: center;
                }
                QPushButton:hover {
                    background-color: rgba(70, 110, 160, 250);
                    border-color: rgba(135, 206, 250, 200);
                }
                QPushButton:pressed {
                    background-color: rgba(40, 70, 110, 255);
                }
            """)
            
            # Position in 2x2 grid
            row = i // 2
            col = i % 2
            grid_layout.addWidget(btn, row, col, Qt.AlignmentFlag.AlignCenter)

        # Add the grid layout to main layout
        layout.addLayout(grid_layout)
        
        # Add spacing before status labels
        layout.addSpacing(12)
        
        # Status labels with consistent styling and fixed heights
        label_style = """
            font-size: 12px; 
            color: #FFFFFF; 
            background-color: rgba(50, 80, 120, 220); 
            padding: 6px 6px; 
            border-radius: 8px; 
            border: 2px solid rgba(255, 255, 255, 80);
        """
        
        self.source_label = QLabel("Source: Not selected")
        self.source_label.setStyleSheet(label_style)
        self.source_label.setFixedHeight(24)
        # self.source_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        
        self.output_label = QLabel("Output: Not selected")
        self.output_label.setStyleSheet(label_style)
        self.output_label.setFixedHeight(24)
        # self.output_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        
        # Add status labels with small spacing between them
        layout.addWidget(self.source_label)
        layout.addSpacing(4)
        layout.addWidget(self.output_label)
        
        # Set the group box to have a fixed height to prevent expansion
        dir_group.setMaximumHeight(180)  # Reduced height
        dir_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        
        return dir_group

    def create_navigation_section(self):
        """MODIFIED: Wider navigation section with better button arrangement"""
        nav_group = QGroupBox("🧭 Navigation and Export")
        nav_group.setStyleSheet("""
            QGroupBox {
                background-color: rgba(15, 25, 40, 200);
                border: 2px solid rgba(255, 255, 255, 60);
                border-radius: 8px;
                padding: 8px;
            }
        """)
        layout = QVBoxLayout(nav_group)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 15, 10, 10)
        
        # Navigation buttons in a horizontal layout
        nav_layout = QHBoxLayout()
        nav_layout.setSpacing(8)  # Space between prev/next buttons
        
        button_height = 35
        self.prev_btn = QPushButton("◀ Prev")
        self.next_btn = QPushButton("Next ▶")
        
        # Configure buttons
        for btn in [self.prev_btn, self.next_btn]:
            btn.setFixedHeight(button_height)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.prev_btn.clicked.connect(self.prev_image)
        self.next_btn.clicked.connect(self.next_image)
        
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.next_btn)
        layout.addLayout(nav_layout)
        
        # Image counter
        self.image_counter = QLabel("No images loaded")
        self.image_counter.setStyleSheet("""
            color: #FFFFFF; 
            font-weight: bold; 
            font-size: 12px; 
            background-color: rgba(50, 80, 120, 230); 
            padding: 8px; 
            border-radius: 4px; 
            border: 2px solid rgba(255, 255, 255, 90);
        """)
        self.image_counter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_counter.setFixedHeight(30)
        layout.addWidget(self.image_counter)
        
        # Add spacing before export buttons
        layout.addSpacing(8)
        
        # Export buttons moved here from separate section
        export_layout = QHBoxLayout()
        export_layout.setSpacing(8)
        
        self.save_txt_btn = QPushButton("💾 Save TXT")
        self.save_txt_btn.setToolTip("Save all annotations as individual TXT files")
        self.save_txt_btn.setFixedHeight(button_height)
        self.save_txt_btn.clicked.connect(self.save_txt_files)
        self.save_txt_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        
        self.save_csv_btn = QPushButton("📊 CSV")
        self.save_csv_btn.setToolTip("Create a metadata.csv file that links images to their annotations")
        self.save_csv_btn.setFixedHeight(button_height)
        self.save_csv_btn.clicked.connect(self.generate_csv)
        self.save_csv_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        
        export_layout.addWidget(self.save_txt_btn)
        export_layout.addWidget(self.save_csv_btn)
        layout.addLayout(export_layout)
        
        return nav_group

    def create_annotation_section(self):
        """MODIFIED: Wider annotation section with better layout control"""
        ann_group = QGroupBox("✏️ Annotation and AI Auto-Annotation")
        ann_group.setStyleSheet("""
            QGroupBox {
                background-color: rgba(15, 25, 40, 200);
                border: 2px solid rgba(255, 255, 255, 60);
                border-radius: 8px;
                padding: 8px;
            }
        """)
        
        layout = QVBoxLayout(ann_group)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 15, 10, 10)
        
        # Text area with increased height
        self.annotation_text = QTextEdit()
        self.annotation_text.setPlaceholderText("Enter image description here...")
        self.annotation_text.setFixedHeight(140)  # Increased height
        self.annotation_text.textChanged.connect(self.save_annotation)
        layout.addWidget(self.annotation_text)
        
        # AI section with controlled spacing
        ai_layout = QVBoxLayout()
        ai_layout.setSpacing(6)
        
        # Model selection row
        model_layout = QHBoxLayout()
        model_layout.setSpacing(8)
        
        model_label = QLabel("🤖 AI Model:")
        model_label.setFixedWidth(80)
        model_label.setStyleSheet("font-size: 12px; font-weight: bold;")
        
        self.model_combo = QComboBox()
        self.model_combo.setToolTip("Select an Ollama model for AI annotations")
        self.model_combo.addItem("Loading models...")
        self.model_combo.setFixedHeight(28)
        
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo, 1)  # Stretch factor
        ai_layout.addLayout(model_layout)
        
        # AI buttons in a horizontal layout for better space usage
        ai_btn_layout = QHBoxLayout()
        ai_btn_layout.setSpacing(6)
        
        # MODIFIED: Make buttons almost square and customizable
        button_height = 40  # Height of buttons - INCREASE THIS FOR TALLER BUTTONS
        button_width = 60   # Width of buttons - INCREASE THIS FOR WIDER BUTTONS
        button_font_size = 10  # Font size for the buttons - INCREASE THIS FOR BIGGER TEXT
        
        # Custom button style with font size control and faster hover response
        button_style = f"""
            QPushButton {{
                background-color: rgba(50, 80, 120, 180);
                border: 2px solid rgba(100, 149, 237, 150);
                padding: 4px 8px;
                color: #FFFFFF;
                font-size: {button_font_size}px;
                font-weight: 600;
                border-radius: 6px;
                text-align: center;
            }}
            QPushButton:hover {{
                background-color: rgba(80, 120, 180, 220);
                border-color: rgba(135, 206, 250, 240);
            }}
            QPushButton:pressed {{
                background-color: rgba(40, 70, 110, 240);
                border-color: rgba(100, 149, 237, 200);
            }}
            QPushButton:disabled {{
                background-color: rgba(60, 60, 60, 180);
                color: #AAAAAA;
                border-color: rgba(100, 100, 100, 120);
            }}
        """
        
        self.ai_current_btn = QPushButton("🔍\nCurrent")
        self.ai_current_btn.setObjectName("ai_button")  # Add this line
        self.ai_current_btn.setToolTip("Use AI to annotate only the current image")
        self.ai_current_btn.clicked.connect(self.ai_annotate_current)
        self.ai_current_btn.setEnabled(False)
        self.ai_current_btn.setFixedSize(button_width, button_height)
        self.ai_current_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.ai_current_btn.setStyleSheet(button_style)

        self.ai_all_btn = QPushButton("🚀\nAll")
        self.ai_all_btn.setObjectName("ai_button")  # Add this line
        self.ai_all_btn.setToolTip("Use AI to annotate all images in the dataset")
        self.ai_all_btn.clicked.connect(self.ai_annotate_all)
        self.ai_all_btn.setEnabled(False)
        self.ai_all_btn.setFixedSize(button_width, button_height)
        self.ai_all_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.ai_all_btn.setStyleSheet(button_style)

        self.refresh_models_btn = QPushButton("🔄\nRefresh")
        self.refresh_models_btn.setObjectName("ai_button")  # Add this line
        self.refresh_models_btn.setToolTip("Refresh the list of available Ollama models")
        self.refresh_models_btn.clicked.connect(self.refresh_ollama_models)
        self.refresh_models_btn.setFixedSize(button_width, button_height)
        self.refresh_models_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.refresh_models_btn.setStyleSheet(button_style)

        # Arrange buttons horizontally for better space usage
        ai_btn_layout.addWidget(self.ai_current_btn)
        ai_btn_layout.addWidget(self.ai_all_btn)
        ai_btn_layout.addWidget(self.refresh_models_btn)

        ai_layout.addLayout(ai_btn_layout)
        layout.addLayout(ai_layout)
        
        # Hint label with fixed height
        hint_label = QLabel("💡 Type manually or use AI (vision models like llava work best)")
        hint_label.setStyleSheet("""
            font-size: 12px; 
            color: #FFFFFF; 
            font-style: italic; 
            background-color: rgba(50, 80, 120, 180); 
            padding: 6px; 
            border-radius: 4px; 
            border: 1px solid rgba(255, 255, 255, 60);
        """)
        hint_label.setWordWrap(True)
        hint_label.setFixedHeight(35)
        layout.addWidget(hint_label)
        
        return ann_group

    def create_bottom_controls(self):
        """Create bottom controls with help and settings buttons"""
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(8)
        
        # MODIFIED: Customizable button sizes
        control_button_size = 50  # Increased from 40 to 50 (you can adjust this)
        
        # Help button with larger question mark
        help_btn = QPushButton("❓")
        help_btn.setToolTip("View instructions on how to use this tool")
        help_btn.clicked.connect(self.show_help)
        help_btn.setFixedSize(control_button_size, control_button_size)  # Square button
        help_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: rgba(30, 60, 100, 230);
                border: 2px solid rgba(100, 149, 237, 150);
                padding: 4px;
                color: #FFFFFF;
                font-size: {int(control_button_size * 0.4)}px;  /* Scales with button size */
                font-weight: 600;
                border-radius: 6px;
            }}
            QPushButton:hover {{
                background-color: rgba(50, 80, 120, 250);
                border-color: rgba(135, 206, 250, 200);
            }}
        """)
        
        # Settings button
        settings_btn = QPushButton("⚙️")
        settings_btn.setToolTip("Open settings to configure background animation and other options")
        settings_btn.clicked.connect(self.show_settings)
        settings_btn.setFixedSize(control_button_size, control_button_size)  # Square button
        settings_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: rgba(30, 60, 100, 230);
                border: 2px solid rgba(100, 149, 237, 150);
                padding: 4px;
                color: #FFFFFF;
                font-size: {int(control_button_size * 0.35)}px;  /* Slightly smaller than help button */
                font-weight: 600;
                border-radius: 6px;
            }}
            QPushButton:hover {{
                background-color: rgba(50, 80, 120, 250);
                border-color: rgba(135, 206, 250, 200);
            }}
        """)
        
        # Add stretch to push buttons to the right
        controls_layout.addStretch()
        controls_layout.addWidget(settings_btn)
        controls_layout.addWidget(help_btn)
        
        return controls_layout

    def create_left_panel(self):
        """MODIFIED: Create left panel with transparent background to show noise"""
        left_panel = QWidget()
        left_panel.setMinimumWidth(380)
        left_panel.setMaximumWidth(500)
        left_panel.setStyleSheet("""
            QWidget {
                background-color: rgba(10, 15, 25, 180);
                border: 2px solid rgba(255, 255, 255, 60);
                border-radius: 12px;
            }
        """)
        
        # MODIFIED: Better layout control
        layout = QVBoxLayout(left_panel)
        layout.setSpacing(12)  # Consistent spacing between sections
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Title
        title = QLabel("🎨 oarc-image-annotation-ui 🎨")
        title.setStyleSheet("""
            font-size: 18px; 
            font-weight: bold; 
            color: #FFFFFF; 
            text-align: center; 
            background-color: rgba(20, 30, 50, 200); 
            padding: 12px; 
            border-radius: 8px; 
            border: 2px solid rgba(255, 255, 255, 80);
        """)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        layout.addWidget(title)
        
        # Directory section
        dir_group = self.create_directory_section()
        layout.addWidget(dir_group)
        
        # Navigation section (now includes export buttons)
        nav_group = self.create_navigation_section()
        layout.addWidget(nav_group)
        
        # Annotation section (wider and better organized)
        ann_group = self.create_annotation_section()
        layout.addWidget(ann_group)
        
        # Bottom controls (help and settings)
        bottom_controls = self.create_bottom_controls()
        layout.addLayout(bottom_controls)
        
        # MODIFIED: Add stretch to push everything up but allow controlled expansion
        layout.addStretch(1)
        return left_panel

    def create_right_panel(self):
        """Create the right panel containing the image viewer with transparent background"""
        right_panel = QWidget()
        right_panel.setStyleSheet("""
            QWidget {
                background-color: rgba(10, 15, 25, 180);
                border: 2px solid rgba(255, 255, 255, 60);
                border-radius: 12px;
            }
        """)

        layout = QVBoxLayout(right_panel)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)

        # Title for the image viewer
        viewer_title = QLabel("🖼️ Image Viewer")
        viewer_title.setStyleSheet("""
            font-size: 16px; 
            font-weight: bold; 
            color: #FFFFFF; 
            text-align: center; 
            background-color: rgba(20, 30, 50, 200); 
            padding: 10px; 
            border-radius: 6px; 
            border: 2px solid rgba(255, 255, 255, 80);
        """)
        viewer_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        viewer_title.setFixedHeight(40)
        layout.addWidget(viewer_title)

        # Create scroll area for the image - FIXED: Semi-transparent background
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setStyleSheet("""
            QScrollArea {
                background-color: rgba(0, 0, 0, 60);
                border: 2px solid rgba(135, 206, 250, 180);
                border-radius: 8px;
            }
            QScrollBar:vertical {
                background-color: rgba(200, 200, 200, 150);
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: rgba(135, 206, 250, 200);
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: rgba(100, 149, 237, 250);
            }
            QScrollBar:horizontal {
                background-color: rgba(200, 200, 200, 150);
                height: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal {
                background-color: rgba(135, 206, 250, 200);
                border-radius: 6px;
                min-width: 20px;
            }
            QScrollBar::handle:horizontal:hover {
                background-color: rgba(100, 149, 237, 250);
            }
        """)

        # FIXED: Make image label completely transparent but with solid image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: transparent;
                color: #FFFFFF;
                font-weight: bold; 
                font-size: 18px;
                padding: 20px;
                border: none;
            }
        """)
        self.image_label.setText("Select a directory to begin\nimage display")
        self.image_label.setWordWrap(True)
        self.image_label.setMinimumSize(400, 300)

        # Set the image label directly as the scroll area's widget
        scroll_area.setWidget(self.image_label)

        layout.addWidget(scroll_area)

        return right_panel

    def toggle_noise_background(self):
        """Toggle the noise background animation on/off"""
        if self.noise_enabled:
            # Disable animation
            self.noise_background.animation_timer.stop()
            self.noise_background.hide()
            self.noise_enabled = False
            print("Background animation disabled")
        else:
            # Enable animation
            self.noise_background.show()
            self.noise_background.animation_timer.start(self.noise_background.timer_interval)
            self.noise_enabled = True
            print("Background animation enabled")

    def show_settings(self):
        """Show settings dialog"""
        settings_dialog = QDialog(self)
        settings_dialog.setWindowTitle("Settings")
        settings_dialog.setMinimumSize(450, 500)
        
        layout = QVBoxLayout(settings_dialog)
        
        # Background Animation Settings
        bg_group = QGroupBox("🌊 Background Animation")
        bg_layout = QVBoxLayout(bg_group)
        bg_group.setStyleSheet("""
            QGroupBox {
                background-color: rgba(20, 30, 50, 200);
                color: #FFFFFF;
                border: 2px solid rgba(255, 255, 255, 80);
                border-radius: 8px;
                padding: 8px;
            }
        """)
        
        # Enable/Disable toggle
        self.bg_enabled_cb = QCheckBox("Enable Background Animation")
        self.bg_enabled_cb.setChecked(self.noise_enabled)
        self.bg_enabled_cb.toggled.connect(self.toggle_noise_background)
        bg_layout.addWidget(self.bg_enabled_cb)
        
        # Animation speed
        speed_layout = QHBoxLayout()
        speed_label = QLabel("Animation Speed:")
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(50)
        self.speed_slider.setValue(int(self.noise_background.animation_speed * 10))
        self.speed_slider.valueChanged.connect(self.on_speed_changed)
        
        self.speed_value_label = QLabel(f"{self.noise_background.animation_speed:.1f}")
        
        speed_layout.addWidget(speed_label)
        speed_layout.addWidget(self.speed_slider)
        speed_layout.addWidget(self.speed_value_label)
        bg_layout.addLayout(speed_layout)
        
        # Frame rate
        fps_layout = QHBoxLayout()
        fps_label = QLabel("Frame Rate (FPS):")
        self.fps_slider = QSlider(Qt.Orientation.Horizontal)
        self.fps_slider.setMinimum(5)
        self.fps_slider.setMaximum(60)
        current_fps = 1000 // self.noise_background.timer_interval
        self.fps_slider.setValue(current_fps)
        self.fps_slider.valueChanged.connect(self.on_fps_changed)
        
        self.fps_value_label = QLabel(f"{current_fps}")
        
        fps_layout.addWidget(fps_label)
        fps_layout.addWidget(self.fps_slider)
        fps_layout.addWidget(self.fps_value_label)
        bg_layout.addLayout(fps_layout)
        
        layout.addWidget(bg_group)
        
        # NEW: Flow Effects Group
        flow_group = QGroupBox("🌪️ Flow Effects")
        flow_group.setStyleSheet("""
            QGroupBox {
                background-color: rgba(20, 30, 50, 200);
                color: #FFFFFF;
                border: 2px solid rgba(255, 255, 255, 80);
                border-radius: 8px;
                padding: 8px;
            }
        """)
        
        flow_layout = QVBoxLayout(flow_group)
        
        # Preset buttons
        preset_layout = QHBoxLayout()
        preset_label = QLabel("Presets:")
        preset_layout.addWidget(preset_label)
        
        for preset in ["calm", "moderate", "turbulent", "chaotic"]:
            btn = QPushButton(preset.title())
            btn.clicked.connect(lambda checked, p=preset: self.apply_flow_preset(p))
            preset_layout.addWidget(btn)
        
        flow_layout.addLayout(preset_layout)
        
        # Flow strength slider
        flow_strength_layout = QHBoxLayout()
        flow_strength_label = QLabel("Flow Strength:")
        self.flow_strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.flow_strength_slider.setMinimum(0)
        self.flow_strength_slider.setMaximum(100)
        self.flow_strength_slider.setValue(int(self.noise_background.flow_strength * 100))
        self.flow_strength_slider.valueChanged.connect(self.on_flow_strength_changed)
        
        self.flow_strength_value_label = QLabel(f"{self.noise_background.flow_strength:.2f}")
        
        flow_strength_layout.addWidget(flow_strength_label)
        flow_strength_layout.addWidget(self.flow_strength_slider)
        flow_strength_layout.addWidget(self.flow_strength_value_label)
        flow_layout.addLayout(flow_strength_layout)
        
        # Vortex strength slider
        vortex_strength_layout = QHBoxLayout()
        vortex_strength_label = QLabel("Vortex Strength:")
        self.vortex_strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.vortex_strength_slider.setMinimum(0)
        self.vortex_strength_slider.setMaximum(100)
        self.vortex_strength_slider.setValue(int(self.noise_background.vortex_strength * 100))
        self.vortex_strength_slider.valueChanged.connect(self.on_vortex_strength_changed)
        
        self.vortex_strength_value_label = QLabel(f"{self.noise_background.vortex_strength:.2f}")
        
        vortex_strength_layout.addWidget(vortex_strength_label)
        vortex_strength_layout.addWidget(self.vortex_strength_slider)
        vortex_strength_layout.addWidget(self.vortex_strength_value_label)
        flow_layout.addLayout(vortex_strength_layout)
        
        # Turbulence scale slider
        turbulence_scale_layout = QHBoxLayout()
        turbulence_scale_label = QLabel("Turbulence Scale:")
        self.turbulence_scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.turbulence_scale_slider.setMinimum(10)
        self.turbulence_scale_slider.setMaximum(300)
        self.turbulence_scale_slider.setValue(int(self.noise_background.turbulence_scale * 100))
        self.turbulence_scale_slider.valueChanged.connect(self.on_turbulence_scale_changed)
        
        self.turbulence_scale_value_label = QLabel(f"{self.noise_background.turbulence_scale:.2f}")
        
        turbulence_scale_layout.addWidget(turbulence_scale_label)
        turbulence_scale_layout.addWidget(self.turbulence_scale_slider)
        turbulence_scale_layout.addWidget(self.turbulence_scale_value_label)
        flow_layout.addLayout(turbulence_scale_layout)
        
        layout.addWidget(flow_group)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(settings_dialog.accept)
        layout.addWidget(close_btn)
        
        settings_dialog.setLayout(layout)
        settings_dialog.exec()

    # NEW: Flow control methods
    def apply_flow_preset(self, preset_name):
        """Apply a flow preset"""
        self.noise_background.set_flow_preset(preset_name)
        
        # Update sliders to reflect the new values
        if hasattr(self, 'flow_strength_slider'):
            self.flow_strength_slider.setValue(int(self.noise_background.flow_strength * 100))
            self.flow_strength_value_label.setText(f"{self.noise_background.flow_strength:.2f}")
            
        if hasattr(self, 'vortex_strength_slider'):
            self.vortex_strength_slider.setValue(int(self.noise_background.vortex_strength * 100))
            self.vortex_strength_value_label.setText(f"{self.noise_background.vortex_strength:.2f}")
            
        if hasattr(self, 'turbulence_scale_slider'):
            self.turbulence_scale_slider.setValue(int(self.noise_background.turbulence_scale * 100))
            self.turbulence_scale_value_label.setText(f"{self.noise_background.turbulence_scale:.2f}")
    
    def on_speed_changed(self, value):
        """Handle animation speed change"""
        speed = value / 10.0
        self.noise_background.set_animation_speed(speed)
        self.speed_value_label.setText(f"{speed:.1f}")

    def on_fps_changed(self, value):
        """Handle FPS change"""
        interval = 1000 // value
        self.noise_background.set_timer_interval(interval)
        self.fps_value_label.setText(f"{value}")

    # NEW: Flow control methods
    def apply_flow_preset(self, preset_name):
        """Apply a flow preset"""
        self.noise_background.set_flow_preset(preset_name)
        
        # Update sliders to reflect the new values
        if hasattr(self, 'flow_strength_slider'):
            self.flow_strength_slider.setValue(int(self.noise_background.flow_strength * 100))
            self.flow_strength_value_label.setText(f"{self.noise_background.flow_strength:.2f}")
            
        if hasattr(self, 'vortex_strength_slider'):
            self.vortex_strength_slider.setValue(int(self.noise_background.vortex_strength * 100))
            self.vortex_strength_value_label.setText(f"{self.noise_background.vortex_strength:.2f}")
            
        if hasattr(self, 'turbulence_scale_slider'):
            self.turbulence_scale_slider.setValue(int(self.noise_background.turbulence_scale * 100))
            self.turbulence_scale_value_label.setText(f"{self.noise_background.turbulence_scale:.2f}")
    
    def on_flow_strength_changed(self, value):
        """Handle flow strength change"""
        strength = value / 100.0
        self.noise_background.set_flow_strength(strength)
        self.flow_strength_value_label.setText(f"{strength:.2f}")
    
    def on_vortex_strength_changed(self, value):
        """Handle vortex strength change"""
        strength = value / 100.0
        self.noise_background.set_vortex_strength(strength)
        self.vortex_strength_value_label.setText(f"{strength:.2f}")
    
    def on_turbulence_scale_changed(self, value):
        """Handle turbulence scale change"""
        scale = value / 100.0
        self.noise_background.set_turbulence_scale(scale)
        self.turbulence_scale_value_label.setText(f"{scale:.2f}")

    # [All the other methods remain the same - just copying them over]
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
                try:
                    from image_format_master import TORCH_AVAILABLE
                    if TORCH_AVAILABLE:
                        gpu_warmup()
                except:
                    pass
                
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
                
                # Enable AI buttons if we have models
                self.update_ai_buttons()
                
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

<h3>🤖 AI Annotation Features:</h3>
<b>Prerequisites:</b>
   • Ensure Ollama is installed and running on your system
   • Install a vision model like: <code>ollama pull llava</code>
   • Click <b>Refresh Models</b> to load available models

<b>AI Annotation Options:</b>
   • <b>🔍 AI Current:</b> Generate annotation for the currently displayed image
   • <b>🚀 AI All:</b> Batch process all images in the dataset
   • Select any model from the dropdown (vision models like llava, moondream work best)
<b>Mixed Workflow:</b>
   • Use AI for bulk annotation, then manually refine descriptions
   • AI annotations appear in the text box where you can edit them
   • Perfect for quickly creating base annotations then adding specific details

<h3>Creating the Final Dataset:</h3>
   • Click <b>Save All TXT Files</b> to save individual annotation text files
   • Click <b>Generate Dataset CSV</b> to create the final metadata.csv file
   
<h3>Pro Tips:</h3>
   • The formatted images, text files, and CSV will all be saved to your output directory
   • Write detailed, descriptive annotations for better AI training results
   • The metadata.csv file connects your images with their annotations
   • You don't need to click Save after each annotation - it happens automatically
   • For best AI results, use vision models like llava, moondream, or minicpm
   • AI annotations can be used as starting points and refined manually
   • Batch AI annotation is great for large datasets - just review and edit afterward
"""
    
        help_dialog = QDialog(self)
        help_dialog.setWindowTitle("How to Use This Tool")
        help_dialog.setMinimumSize(700, 600)
        
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
            
            # Enable AI buttons if we have models and images
            self.update_ai_buttons()
            
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
            QMessageBox.information(self, "End of Images", "You're at the last image.")

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
            
        current_image = self.image_files[self.current_index]
        annotation = self.annotation_text.toPlainText()
        
        # Save to memory
        self.annotations[current_image.name] = annotation

    def save_txt_files(self):
        """Save all annotations as individual TXT files"""
        if not self.output_dir:
            QMessageBox.warning(self, "No Output Directory", "Please set an output directory first.")
            return
        
        if not self.annotations:
            QMessageBox.information(self, "No Annotations", "No annotations to save.")
            return
        
        try:
            saved_count = 0
            for filename, annotation in self.annotations.items():
                if annotation.strip():  # Only save non-empty annotations
                    txt_filename = Path(filename).stem + ".txt"
                    txt_path = self.output_dir / txt_filename
                    
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(annotation)
                    saved_count += 1
            
            QMessageBox.information(self, "Annotations Saved", 
                               f"Saved {saved_count} annotation files to:\n{self.output_dir}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saving annotations:\n{str(e)}")

    def generate_csv(self):
        """Generate a CSV file linking images to their annotations"""
        if not self.output_dir:
            QMessageBox.warning(self, "No Output Directory", "Please set an output directory first.")
            return
        
        if not self.image_files:
            QMessageBox.warning(self, "No Images", "No images loaded.")
            return
        
        try:
            processor = ImageAnnotationProcessor(str(self.output_dir))
            csv_path = processor.create_metadata_csv()
            
            QMessageBox.information(self, "CSV Generated", 
                               f"Metadata CSV created:\n{csv_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error generating CSV:\n{str(e)}")

    def refresh_ollama_models(self):
        """Refresh the list of available Ollama models"""
        self.model_combo.clear()
        self.model_combo.addItem("Loading models...")
        
        def get_models():
            """Get available Ollama models"""
            try:
                # FIXED: Handle async function properly
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    models = loop.run_until_complete(self.ollama_commands.list_models())
                    return models if models else []
                finally:
                    loop.close()
            except Exception as e:
                print(f"Error getting models: {e}")
                return []
    
        # Run model fetching in background thread
        future = self.thread_pool.submit(get_models)
        
        def update_models():
            try:
                models = future.result(timeout=10)  # 10 second timeout
                self.available_models = models
                self.model_combo.clear()
                
                if models:
                    print(f"Found {len(models)} Ollama models")
                    
                    # Add all models with proper name extraction
                    for model in models:
                        model_name = None
                        
                        # Handle object with .model attribute (new format)
                        if hasattr(model, 'model'):
                            model_name = model.model
                        # Handle dictionary format (older format)
                        elif isinstance(model, dict):
                            for key in ['name', 'model', 'id', 'title']:
                                if key in model and model[key]:
                                    model_name = model[key]
                                    break
                        
                        # Fallback
                        if not model_name:
                            model_name = str(model)
                        
                        self.model_combo.addItem(model_name, model_name)
                    
                    # Enable AI buttons based on availability
                    self.update_ai_buttons()
                    
                    # Select first model by default
                    if models:
                        self.model_combo.setCurrentIndex(0)
                        self.selected_model = self.model_combo.currentData() or self.model_combo.currentText()
                        # FIXED: Connect the signal after populating
                        self.model_combo.currentTextChanged.connect(self.on_model_changed)
            
                else:
                    self.model_combo.addItem("No models found")
                    self.ai_current_btn.setEnabled(False)
                    self.ai_all_btn.setEnabled(False)
                    
            except Exception as e:
                print(f"Error updating models: {e}")
                self.model_combo.clear()
                self.model_combo.addItem("Error loading models")
                self.ai_current_btn.setEnabled(False)
                self.ai_all_btn.setEnabled(False)
        
        # Use QTimer to update UI in main thread
        QTimer.singleShot(100, update_models)

    def update_ai_buttons(self):
        """Update AI button states based on models and images availability"""
        has_models = isinstance(self.available_models, list) and len(self.available_models) > 0
        has_images = len(self.image_files) > 0
        
        self.ai_current_btn.setEnabled(has_models and has_images)
        self.ai_all_btn.setEnabled(has_models and has_images)
    
    def on_model_changed(self):
        """Handle model selection change"""
        current_text = self.model_combo.currentText()
        current_data = self.model_combo.currentData()
        
        # Use the data if available, otherwise use the text
        if current_data:
            self.selected_model = current_data
        else:
            self.selected_model = current_text
            
        print(f"Selected model: {self.selected_model}")
    
    def ai_annotate_current(self):
        """Use AI to annotate the current image"""
        if not self.selected_model:
            QMessageBox.warning(self, "No Model", "Please select an AI model first.")
            return
        
        if not self.image_files or self.current_index >= len(self.image_files):
            QMessageBox.warning(self, "No Image", "No image selected for annotation.")
            return
        
        current_image = self.image_files[self.current_index]
        
        # Show progress dialog
        progress = QProgressDialog("Generating AI annotation...", "Cancel", 0, 0, self)
        progress.setWindowTitle("AI Annotation")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()
        
        def generate_annotation():
            try:
                # Create a new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Convert image to base64
                image_b64 = self.image_to_base64(current_image)
                if not image_b64:
                    return None, "Failed to convert image to base64"
                
                # Generate annotation
                prompt = "Describe this image in detail. Focus on the main subjects, objects, actions, colors, composition, and any notable features. Be descriptive but concise."
                
                annotation = loop.run_until_complete(
                    self.ollama_commands.vision_chat(
                        model=self.selected_model,
                        prompt=prompt,
                        image_data=image_b64
                    )
                )
                
                return annotation, None
            except Exception as e:
                return None, str(e)
            finally:
                loop.close()
        
        # Run in background thread
        future = self.thread_pool.submit(generate_annotation)
        
        def handle_result():
            if progress.wasCanceled():
                return
                
            try:
                annotation, error = future.result(timeout=60)  # 60 second timeout
                progress.close()
                
                if error:
                    QMessageBox.critical(self, "AI Annotation Error", f"Failed to generate annotation:\n{error}")
                elif annotation:
                    # Update the annotation text area
                    self.annotation_text.setPlainText(annotation.strip())
                    QMessageBox.information(self, "AI Annotation Complete", 
                        f"Generated annotation for {current_image.name}")
                else:
                    QMessageBox.warning(self, "AI Annotation", "No annotation generated.")
                    
            except Exception as e:
                progress.close()
                QMessageBox.critical(self, "Error", f"Error generating annotation: {str(e)}")
        
        # Check result periodically
        def check_result():
            if future.done():
                handle_result()
            elif not progress.wasCanceled():
                QTimer.singleShot(100, check_result)
                
        QTimer.singleShot(100, check_result)
    
    def ai_annotate_all(self):
        """Use AI to annotate all images in the dataset"""
        if not self.selected_model:
            QMessageBox.warning(self, "No Model", "Please select an AI model first.")
            return
        
        if not self.image_files:
            QMessageBox.warning(self, "No Images", "No images loaded for annotation.")
            return
        
        # Confirm batch annotation
        reply = QMessageBox.question(self, "Batch AI Annotation", 
            f"This will generate AI annotations for all {len(self.image_files)} images using the model '{self.selected_model}'.\n\n"
            "This may take several minutes. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # Show progress dialog
        progress = QProgressDialog("Generating AI annotations...", "Cancel", 0, len(self.image_files), self)
        progress.setWindowTitle("Batch AI Annotation")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()
        
        successful_annotations = 0
        failed_annotations = 0
        
        def generate_all_annotations():
            nonlocal successful_annotations, failed_annotations
            
            try:
                # Create a new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                for i, image_path in enumerate(self.image_files):
                    if progress.wasCanceled():
                        break
                    
                    try:
                        # Update progress
                        progress.setValue(i)
                        progress.setLabelText(f"Processing {image_path.name}...")
                        
                        # Convert image to base64
                        image_b64 = self.image_to_base64(image_path)
                        if not image_b64:
                            failed_annotations += 1
                            continue
                        
                        # Generate annotation
                        prompt = "Describe this image in detail. Focus on the main subjects, objects, actions, colors, composition, and any notable features. Be descriptive but concise."
                        
                        annotation = loop.run_until_complete(
                            self.ollama_commands.vision_chat(
                                model=self.selected_model,
                                prompt=prompt,
                                image_data=image_b64
                            )
                        )
                        
                        if annotation:
                            # Save annotation
                            self.annotations[image_path.name] = annotation.strip()
                            successful_annotations += 1
                        else:
                            failed_annotations += 1
                            
                    except Exception as e:
                        print(f"Error annotating {image_path.name}: {e}")
                        failed_annotations += 1
                
                return True
                
            except Exception as e:
                print(f"Batch annotation error: {e}")
                return False
            finally:
                loop.close()
        
        # Run in background thread
        future = self.thread_pool.submit(generate_all_annotations)
        
        def handle_batch_result():
            if progress.wasCanceled():
                return
                
            try:
                success = future.result(timeout=600)  # 10 minute timeout for batch
                progress.close()
                
                # Update current image annotation if we're viewing an image
                if self.image_files and self.current_index < len(self.image_files):
                    current_image = self.image_files[self.current_index]
                    if current_image.name in self.annotations:
                        self.annotation_text.setPlainText(self.annotations[current_image.name])
                
                # Show results
                QMessageBox.information(self, "Batch AI Annotation Complete", 
                    f"Annotation Results:\n"
                    f"✅ Successful: {successful_annotations}\n"
                    f"❌ Failed: {failed_annotations}\n"
                    f"📊 Total: {len(self.image_files)}\n\n"
                    f"Annotations are saved in memory. Use 'Save All TXT Files' to save to disk.")
                
            except Exception as e:
                progress.close()
                QMessageBox.critical(self, "Error", f"Batch annotation error: {str(e)}")
        
        # Check result periodically
        def check_batch_result():
            if future.done():
                handle_batch_result()
            elif not progress.wasCanceled():
                QTimer.singleShot(1000, check_batch_result)  # Check every second for batch
                
        QTimer.singleShot(1000, check_batch_result)
    
    def image_to_base64(self, image_path):
        """Convert image file to base64 string for Ollama vision"""
        try:
            with open(image_path, 'rb') as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Error converting image to base64: {e}")
            return None

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Image Annotation Studio")
    app.setApplicationVersion("1.0")
    
    # Set application icon if available
    try:
        app.setWindowIcon(QIcon("icon.png"))
    except:
        pass
    
    window = ModernImageAnnotationUI()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
