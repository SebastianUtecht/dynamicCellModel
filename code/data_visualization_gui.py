### Data Visualization GUI for 3D Cell Simulation Analysis ###

#General imports
import sys
import numpy as np
import os
import pickle
import json
import datetime
import shutil
from scipy.spatial import ConvexHull, cKDTree
import hashlib
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Neighbor search will use CPU only.")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# VisPy imports
import vispy
from vispy import scene
from vispy.scene import visuals
from vispy import app
import vispy.io

# PyQt6 imports
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, 
                            QSlider, QFileDialog, QMessageBox, QCheckBox,   
                            QScrollArea, QDialog, QVBoxLayout, QHBoxLayout, 
                            QTextEdit, QTextBrowser, QLabel, QComboBox, QSpinBox,
                            QInputDialog, QListWidget, QListWidgetItem, QFrame,
                            QSplitter, QGroupBox, QGridLayout, QProgressBar,
                            QProgressDialog)
import PyQt6.QtWidgets as QtWidgets
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QSize
from PyQt6.QtGui import QFont, QPalette, QColor

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import time

class CollapsibleSection(QWidget):
    """Collapsible section widget matching the main GUI style"""
    
    def __init__(self, title, expanded=True):
        super().__init__()
        self.content_widget = None
        self.title = title
        self.setup_ui(title, expanded)
        
    def setup_ui(self, title, expanded):
        """Setup the collapsible section UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Create header button
        self.header_button = QPushButton()
        self.header_button.setCheckable(True)
        self.header_button.setChecked(expanded)
        self.header_button.setStyleSheet("""
            QPushButton {
                color: #2c3e50;
                background-color: #ecf0f1;
                font-size: 14px;
                font-weight: bold;
                padding: 8px 12px;
                margin: 2px 0px;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                text-align: left;
            }
            QPushButton:hover {
                background-color: #d5dbdb;
            }
            QPushButton:checked {
                background-color: #2c4a6b;
                color: white;
                border-color: #1e3045;
            }
        """)
        
        # Create content widget
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(8, 4, 8, 8)
        self.content_layout.setSpacing(4)
        
        layout.addWidget(self.header_button)
        layout.addWidget(self.content_widget)
        
        # Connect toggle functionality and update text
        self.header_button.toggled.connect(self.toggle_content)
        self.content_widget.setVisible(expanded)
        self.update_button_text()
        
    def update_button_text(self):
        """Update button text with expand/collapse indicator"""
        indicator = "▼" if self.header_button.isChecked() else "▶"
        self.header_button.setText(f"{indicator} {self.title}")
        
    def toggle_content(self, checked):
        """Toggle content visibility"""
        self.content_widget.setVisible(checked)
        self.update_button_text()
        
    def add_widget(self, widget):
        """Add widget to content area"""
        self.content_layout.addWidget(widget)
        
    def add_layout(self, layout):
        """Add layout to content area"""
        self.content_layout.addLayout(layout)

class ColorPickerDialog(QDialog):
    """Custom color picker dialog with organized color categories and shades"""
    
    def __init__(self, current_color=None, parent=None):
        super().__init__(parent)
        self.selected_color = current_color if current_color is not None else (0.5, 0.5, 0.5)
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the color picker UI"""
        self.setWindowTitle("Choose Color")
        self.setModal(True)
        self.resize(600, 400)
        
        # Set dialog background for better contrast
        self.setStyleSheet("""
            QDialog {
                background-color: #2c3e50;
            }
        """)
        
        layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel("Select a color for this cell type:")
        title_label.setStyleSheet("""
            font-size: 18px; 
            font-weight: bold; 
            margin: 10px;
            color: white;
            background-color: #2c3e50;
            padding: 10px;
            border-radius: 5px;
        """)
        layout.addWidget(title_label)
        
        # Color categories
        self.create_color_grid(layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #95a5a6;
                color: white;
                border: none;
                padding: 10px 25px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #7f8c8d;
            }
        """)
        button_layout.addWidget(cancel_button)
        
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        ok_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px 25px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        button_layout.addWidget(ok_button)
        
        layout.addLayout(button_layout)
        
    def create_color_grid(self, layout):
        """Create the grid of color categories and shades"""
        scroll_area = QScrollArea()
        scroll_area.setStyleSheet("""
            QScrollArea {
                background-color: #2c3e50;
                border: none;
            }
        """)
        scroll_widget = QWidget()
        scroll_widget.setStyleSheet("""
            QWidget {
                background-color: #2c3e50;
            }
        """)
        grid_layout = QGridLayout(scroll_widget)
        
        # Define color categories with base colors
        color_categories = {
            'Red': (1.0, 0.0, 0.0),
            'Orange': (1.0, 0.5, 0.0),
            'Yellow': (1.0, 1.0, 0.0),
            'Green': (0.0, 1.0, 0.0),
            'Cyan': (0.0, 1.0, 1.0),
            'Blue': (0.0, 0.0, 1.0),
            'Purple': (0.5, 0.0, 1.0),
            'Magenta': (1.0, 0.0, 1.0),
            'Pink': (1.0, 0.7, 0.8),
            'Gray': (0.5, 0.5, 0.5)
        }
        
        row = 0
        for category_name, base_color in color_categories.items():
            # Category header
            header_label = QLabel(category_name)
            header_label.setStyleSheet("""
                font-weight: bold; 
                font-size: 16px; 
                color: white; 
                background-color: #2c3e50;
                padding: 8px;
                border-radius: 3px;
                margin: 2px;
            """)
            grid_layout.addWidget(header_label, row, 0, 1, 10)
            row += 1
            
            # Create 10 shades from light to dark
            for i in range(10):
                # Create shade: lighter colors (0-4) and darker colors (5-9)
                if i <= 4:
                    # Lighter shades: mix with white
                    mix_factor = (4 - i) * 0.2  # 0.8, 0.6, 0.4, 0.2, 0.0
                    r = base_color[0] + (1 - base_color[0]) * mix_factor
                    g = base_color[1] + (1 - base_color[1]) * mix_factor
                    b = base_color[2] + (1 - base_color[2]) * mix_factor
                else:
                    # Darker shades: multiply by factor
                    mix_factor = 1.0 - (i - 4) * 0.15  # 0.85, 0.70, 0.55, 0.40, 0.25
                    r = base_color[0] * mix_factor
                    g = base_color[1] * mix_factor
                    b = base_color[2] * mix_factor
                
                # Clamp values
                r = max(0.0, min(1.0, r))
                g = max(0.0, min(1.0, g))
                b = max(0.0, min(1.0, b))
                
                color_button = QPushButton()
                color_button.setFixedSize(40, 30)
                color_button.setStyleSheet(f"""
                    QPushButton {{
                        background-color: rgb({int(r*255)}, {int(g*255)}, {int(b*255)});
                        border: 2px solid #34495e;
                        border-radius: 4px;
                    }}
                    QPushButton:hover {{
                        border: 3px solid #2c3e50;
                    }}
                """)
                
                # Store color values in the button
                color_button.color_value = (r, g, b)
                color_button.clicked.connect(lambda checked, color=(r, g, b): self.select_color(color))
                
                grid_layout.addWidget(color_button, row, i)
            
            row += 1
        
        scroll_area.setWidget(scroll_widget)
        scroll_area.setMaximumHeight(300)
        layout.addWidget(scroll_area)
        
    def select_color(self, color):
        """Handle color selection"""
        self.selected_color = color
        self.accept()
        
    def get_selected_color(self):
        """Return the selected color"""
        return self.selected_color

class VisPy3DVisualizationWidget(QWidget):
    """
    VisPy-based 3D visualization widget for cell data
    """
    
    def __init__(self, parent_gui):
        super().__init__()
        self.parent_gui = parent_gui
        self.camera_bounds_set = False  # Track if camera bounds have been set
        self.init_vispy()
        
    def init_vispy(self):
        """Initialize VisPy 3D canvas"""
        # Create VisPy canvas embedded in Qt widget with compatible settings
        self.canvas = scene.SceneCanvas(keys='interactive', show=False, 
                                       bgcolor='black', size=(800, 600),
                                       resizable=True)
        
        # Embed the canvas in this Qt widget
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)  # Remove margins for full canvas
        layout.addWidget(self.canvas.native)
        self.setLayout(layout)
        
        # Add view
        self.view = self.canvas.central_widget.add_view()
        
        # Set up camera - use arcball camera to prevent auto-leveling/rolling behavior
        self.view.camera = 'fly'  # 'arcball' camera doesn't auto-level like 'fly' camera
        self.view.camera.fov = 60
        self.view.camera.auto_roll = False  # Disable auto-roll to prevent unwanted rotation
        self.view.camera.auto_roll = False
        
        # Configure better rendering settings (if supported)
        try:
            self.view.camera.depth_value = 1000.0  # Increase depth range
        except AttributeError:
            pass  # Not supported in this VisPy version
        
        # Create scatter plots for different cell types
        self.cell_scatters = {}  # Will hold scatter objects for each cell type
        self.polarity_scatters = {}  # Will hold polarity scatter objects
        
        # Initialize highlighting variables
        self.highlighted_visuals = {}  # Store highlight scatter objects
        self.pre_highlight_state = {}  # Store original visuals state
        
        # Ghost sphere for cell selection (similar to grab mode in 3D GUI)
        self.selection_sphere_distance = 10.0  # Distance from camera
        self.selection_sphere_radius = 3.0  # Radius of selection sphere
        self.ghost_sphere_visual = None
        self.ghost_sphere_position = None
        
        # Mouse tracking for click vs drag detection
        self.mouse_press_pos = None
        self.mouse_dragging = False
        
        # Initialize camera position for better default view
        self.view.camera.set_range()
        
        # Create ghost sphere for cell selection
        self.ghost_sphere_visual = visuals.Markers(scaling=True, alpha=0.5, spherical=True)
        self.view.add(self.ghost_sphere_visual)
        self.ghost_sphere_visual.visible = False
        
        # Create coordinate axes (X=green, Y=red, Z=blue)
        # Each axis extends 100 units in both directions from origin
        self.axis_lines = []
        axis_colors = [(0, 1, 0, 1), (1, 0, 0, 1), (0, 0, 1, 1)]  # Green, Red, Blue
        axis_endpoints = [
            [[-100, 0, 0], [100, 0, 0]],  # X-axis
            [[0, -100, 0], [0, 100, 0]],  # Y-axis
            [[0, 0, -100], [0, 0, 100]]   # Z-axis
        ]
        for i, (color, endpoints) in enumerate(zip(axis_colors, axis_endpoints)):
            axis_line = visuals.Line(pos=np.array(endpoints), color=color, width=3)
            self.view.add(axis_line)
            axis_line.visible = False  # Hidden by default
            self.axis_lines.append(axis_line)
        
        # Setup timer for continuous ghost sphere updates
        from vispy import app as vispy_app
        self._ghost_timer = vispy_app.Timer(interval=0.05, connect=self._update_ghost_sphere, start=True)
        
        # Connect mouse events for cell selection (right-click only)
        @self.canvas.events.mouse_press.connect
        def on_mouse_press(event):
            if event.button == 2:  # Right-click
                # Check if neighbor search is enabled
                if self.parent_gui and self.parent_gui.neighbor_search_enabled:
                    self.on_canvas_click(event)
                    # Prevent event from reaching VisPy camera
                    event.handled = True
        
        # Connect mouse wheel for adjusting ghost sphere distance
        @self.canvas.events.mouse_wheel.connect
        def on_mouse_wheel(event):
            # Check if Shift is pressed
            shift_pressed = False
            if hasattr(event, 'modifiers') and event.modifiers:
                modifiers_str = str(event.modifiers).lower()
                shift_pressed = ('shift' in modifiers_str)
            
            # Also check Qt's native key modifiers if available
            if hasattr(event, 'native') and hasattr(event.native, 'modifiers'):
                from PyQt6.QtCore import Qt
                qt_modifiers = event.native.modifiers()
                if not shift_pressed:
                    shift_pressed = bool(qt_modifiers & Qt.KeyboardModifier.ShiftModifier)
            
            # Get delta value
            delta = event.delta[1] if hasattr(event, 'delta') and len(event.delta) > 1 else 0
            
            if shift_pressed and self.parent_gui and self.parent_gui.neighbor_search_enabled:
                # Adjust selection sphere distance with Shift+mouse wheel
                if delta != 0:
                    self.selection_sphere_distance += delta * 1.0
                    self.selection_sphere_distance = max(3.0, min(50.0, self.selection_sphere_distance))
                
                # Block the event from reaching VisPy
                event.handled = True
                if hasattr(event, 'accept'):
                    event.accept()
        
        # Connect keyboard events
        @self.canvas.connect
        def on_key_press(event):
            if event.text == ' ':
                # Space to toggle play/pause
                self.parent_gui.toggle_play()
            elif event.text == ',':
                # Comma to go back 1 frame
                self.parent_gui.prev_frame()
            elif event.text == '.':
                # Period to go forward 1 frame
                self.parent_gui.next_frame()
            elif event.text == 'r':
                # R to go back 50 frames
                for _ in range(50):
                    self.parent_gui.prev_frame()
            elif event.text == 't':
                # T to go forward 50 frames
                for _ in range(50):
                    self.parent_gui.next_frame()
                
    def _update_ghost_sphere(self, event):
        """Update ghost sphere position continuously when neighbor search is enabled"""
        if self.parent_gui and self.parent_gui.neighbor_search_enabled:
            # Show ghost sphere
            ghost_pos = self.get_camera_centered_position(distance=self.selection_sphere_distance)
            self.ghost_sphere_position = ghost_pos
            
            # Update visual - render like other cells but semi-transparent
            if self.ghost_sphere_visual is not None:
                self.ghost_sphere_visual.set_data(
                    np.array([ghost_pos]), 
                    edge_width=0,
                    face_color='cyan',
                    size=self.selection_sphere_radius * 2
                )
                self.ghost_sphere_visual.visible = True
        else:
            # Hide ghost sphere
            if self.ghost_sphere_visual is not None:
                self.ghost_sphere_visual.visible = False
                
    def get_camera_position(self):
        """Get current camera position in world coordinates"""
        try:
            camera = self.view.camera
            
            # Method 1: Try center property first (most common)
            if hasattr(camera, 'center') and camera.center is not None:
                pos = np.array(camera.center)
                return pos
                
            # Method 2: Try _center (internal property)
            elif hasattr(camera, '_center') and camera._center is not None:
                pos = np.array(camera._center)
                return pos
                
            # Method 3: Try transform matrix approach
            elif hasattr(camera, 'transform') and hasattr(camera.transform, 'matrix'):
                transform_matrix = camera.transform.matrix
                if transform_matrix is not None and transform_matrix.size >= 16:
                    pos = np.array(transform_matrix[:3, 3], dtype=float)
                    return pos
            
            # Fallback position
            return np.array([0, 0, 0], dtype=float)
            
        except Exception as e:
            return np.array([0, 0, 0], dtype=float)
            
    def get_camera_view_direction(self):
        """Get current camera view direction vector"""
        try:
            camera = self.view.camera
            if hasattr(camera, 'transform') and camera.transform is not None:
                transform = camera.transform
                if hasattr(transform, 'map'):
                    # Forward vector in camera space is (0, 0, -1)
                    # Transform to world space to get view direction
                    view_direction = transform.map([0, 0, -1, 0])[:3]
                    
                    # Normalize the view direction
                    if np.linalg.norm(view_direction) > 1e-6:
                        view_direction = view_direction / np.linalg.norm(view_direction)
                    else:
                        view_direction = np.array([0, 0, -1])  # Fallback
                    return view_direction
            return np.array([0, 0, -1])
        except Exception as e:
            return np.array([0, 0, -1])
            
    def get_camera_up_direction(self):
        """Get current camera up direction vector"""
        try:
            camera = self.view.camera
            if hasattr(camera, 'transform') and camera.transform is not None:
                transform = camera.transform
                if hasattr(transform, 'map'):
                    # Up vector in camera space is (0, 1, 0)
                    # Transform to world space to get up direction
                    up_direction = transform.map([0, 1, 0, 0])[:3]
                    
                    # Normalize the up direction
                    if np.linalg.norm(up_direction) > 1e-6:
                        up_direction = up_direction / np.linalg.norm(up_direction)
                    else:
                        up_direction = np.array([0, 1, 0])  # Fallback
                    return up_direction
            return np.array([0, 1, 0])
        except Exception as e:
            return np.array([0, 1, 0])
            
    def get_camera_euler_angles(self):
        """Get camera rotation as Euler angles (in degrees)"""
        try:
            view_dir = self.get_camera_view_direction()
            up_dir = self.get_camera_up_direction()
            
            # Calculate yaw (rotation around Y axis)
            yaw = np.degrees(np.arctan2(-view_dir[0], -view_dir[2]))
            
            # Calculate pitch (rotation around X axis)  
            pitch = np.degrees(np.arcsin(np.clip(view_dir[1], -1, 1)))
            
            # Calculate roll (rotation around Z axis)
            # Create right vector from cross product
            right_dir = np.cross(view_dir, up_dir)
            right_norm = np.linalg.norm(right_dir)
            if right_norm > 1e-6:
                right_dir = right_dir / right_norm
                # Project up vector onto plane perpendicular to view direction
                up_projected = up_dir - np.dot(up_dir, view_dir) * view_dir
                up_projected_norm = np.linalg.norm(up_projected)
                if up_projected_norm > 1e-6:
                    up_projected = up_projected / up_projected_norm
                    world_up = np.array([0, 1, 0])
                    roll = np.degrees(np.arctan2(np.dot(right_dir, up_projected), 
                                                np.dot(world_up, up_projected)))
                else:
                    roll = 0.0
            else:
                roll = 0.0
                
            return np.array([pitch, yaw, roll])
        except Exception as e:
            return np.array([0, 0, 0])
    
    def get_camera_rotation_angle(self):
        """Get camera roll angle for Q/E movement display"""
        try:
            view_direction = self.get_camera_view_direction()
            up_direction = self.get_camera_up_direction()
            
            # Calculate roll (rotation around the viewing axis)
            # Project up vector onto plane perpendicular to view direction
            up_projected = up_direction - np.dot(up_direction, view_direction) * view_direction
            up_projected_norm = np.linalg.norm(up_projected)
            
            if up_projected_norm > 1e-6:
                up_projected = up_projected / up_projected_norm
                
                # World up vector
                world_up = np.array([0, 1, 0])
                
                # Calculate roll angle relative to world up
                # Use the right vector to determine sign
                right_dir = np.cross(view_direction, world_up)
                right_norm = np.linalg.norm(right_dir)
                
                if right_norm > 1e-6:
                    right_dir = right_dir / right_norm
                    
                    # Calculate roll using atan2 for proper quadrant handling
                    roll_cos = np.dot(world_up, up_projected)
                    roll_sin = np.dot(right_dir, up_projected)
                    roll = np.degrees(np.arctan2(roll_sin, roll_cos))
                    
                    return roll
            
            return 0.0
        except Exception as e:
            return 0.0
            
    def set_camera_position(self, position):
        """Set camera position using VisPy's proper camera interface"""
        try:
            camera = self.view.camera
            position = np.array(position, dtype=float)
            
            # For VisPy fly camera, the most reliable approach is usually direct assignment
            success = False
            
            # Method 1: Direct center assignment (most common for fly camera)
            if hasattr(camera, 'center'):
                try:
                    camera.center = tuple(position)  # VisPy often expects tuples
                    success = True
                except Exception as e:
                    pass
                    
            # Method 2: Try using set_state if center didn't work
            if not success and hasattr(camera, 'set_state') and hasattr(camera, 'get_state'):
                try:
                    state = camera.get_state()
                    state['center'] = tuple(position)
                    camera.set_state(state)
                    success = True
                except Exception as e:
                    pass
                    
            # Method 3: Transform matrix manipulation (last resort)
            if not success and hasattr(camera, 'transform') and hasattr(camera.transform, 'matrix'):
                try:
                    transform_matrix = camera.transform.matrix.copy()
                    transform_matrix[:3, 3] = position
                    camera.transform.matrix = transform_matrix
                    success = True
                except Exception as e:
                    pass
                
            # Force updates
            try:
                if hasattr(camera, 'view_changed'):
                    camera.view_changed()
                if hasattr(self.canvas, 'update'):
                    self.canvas.update()
            except:
                pass
                
        except Exception as e:
            pass
            
    def jump_to_camera_position(self, position, view_direction=None, up_direction=None):
        """Jump camera to specified position using simplified reliable approach"""
        try:
            position = np.array(position, dtype=float)
            
            # Use the simplified set_camera_position method
            self.set_camera_position(position)
            
            # Wait a moment for the change to take effect
            import time
            time.sleep(0.1)
            
        except Exception as e:
            pass
                
    def update_visualization(self):
        """Update the 3D visualization based on current settings"""
        if not hasattr(self.parent_gui, 'x_lst') or not self.parent_gui.x_lst:
            return

        current_timestep = self.parent_gui.current_timestep

        # Get current data
        x = self.parent_gui.x_lst[current_timestep]
        p_mask = self.parent_gui.p_mask_lst[current_timestep]

        # Get scalar data based on mode
        scalar_data = None
        if self.parent_gui.color_by_phi:
            if self.parent_gui.scalar_visualization_mode == "phi" and self.parent_gui.phi_lst is not None:
                scalar_data = self.parent_gui.phi_lst[current_timestep]
            elif self.parent_gui.scalar_visualization_mode == "theta" and self.parent_gui.theta_lst is not None:
                scalar_data = self.parent_gui.theta_lst[current_timestep]

        # Clear existing visuals
        for scatter_item in self.cell_scatters.values():
            if isinstance(scatter_item, list):
                # Handle list of scatter objects (like Voronoi meshes)
                for scatter in scatter_item:
                    try:
                        scatter.parent = None  # Remove from scene
                    except:
                        pass
            else:
                # Handle single scatter object
                try:
                    scatter_item.parent = None  # Remove from scene
                except:
                    pass
        for scatter in self.polarity_scatters.values():
            try:
                scatter.parent = None  # Remove from scene
            except:
                pass
        self.cell_scatters.clear()
        self.polarity_scatters.clear()
        
        if x is None or len(x) == 0:
            return
            
        # Apply filtering based on enabled modes
        if self.parent_gui.bisection_enabled:
            x, p_mask, filter_mask = self.apply_bisection_filter(x, p_mask)
            scalar_data = scalar_data[filter_mask] if scalar_data is not None else None
            if len(x) == 0:
                return
        elif self.parent_gui.cross_section_mode:
            # Apply cross-section filtering when bisection is disabled
            x, p_mask, filter_mask = self.apply_cross_section_filter(x, p_mask)
            scalar_data = scalar_data[filter_mask] if scalar_data is not None else None
            if len(x) == 0:
                return
        else:
            filter_mask = np.ones(len(x), dtype=bool)

            # Apply cell type filtering when no spatial filtering is active
            if p_mask is not None and len(self.parent_gui.visible_types) > 0:
                type_filter = np.isin(p_mask, list(self.parent_gui.visible_types))
                filter_mask = filter_mask & type_filter
                x = x[filter_mask]
                p_mask = p_mask[filter_mask] if p_mask is not None else None
                scalar_data = scalar_data[filter_mask] if scalar_data is not None else None

        # Color cells
        if self.parent_gui.color_by_phi and scalar_data is not None:
            self.visualize_by_scalar(x, scalar_data)
        elif self.parent_gui.color_by_depth:
            self.visualize_by_depth(x, current_timestep, filter_mask)
        elif self.parent_gui.color_by_type and p_mask is not None:
            self.visualize_by_cell_type(x, p_mask, current_timestep, filter_mask)
        else:
            self.visualize_single_type(x, current_timestep, filter_mask)

        # Update polarity if enabled
        if (self.parent_gui.show_polarity and not self.parent_gui.color_by_type and
            not self.parent_gui.color_by_depth and not self.parent_gui.color_by_phi):
            self.visualize_polarity(x, current_timestep, filter_mask)
            
        # Update camera bounds only once or when explicitly needed
        if len(x) > 0 and not self.camera_bounds_set:
            self.update_camera_bounds(x)
            self.camera_bounds_set = True
            
    def apply_bisection_filter(self, x, p_mask):
        """Apply bisection filtering to the data"""
        if len(x) == 0:
            return x, p_mask, np.array([], dtype=bool)
            
        # Get bounds
        bounds = np.array([x.min(axis=0), x.max(axis=0)])
        center = bounds.mean(axis=0)
        extent = bounds[1] - bounds[0]
        
        if self.parent_gui.bisection_plane == "Camera Orthogonal":
            # Get camera viewing direction using the same method as model_3d_gui_vispy.py
            try:
                camera = self.view.camera
                transform = camera.transform
                
                if transform is not None and hasattr(transform, 'map'):
                    # Forward vector in camera space is (0, 0, -1)
                    # Transform to world space to get view direction
                    view_direction = transform.map([0, 0, -1, 0])[:3]
                    
                    # Normalize the view direction
                    if np.linalg.norm(view_direction) > 0:
                        view_direction = view_direction / np.linalg.norm(view_direction)
                    else:
                        view_direction = np.array([0, 0, -1])  # Fallback
                else:
                    # Fallback if transform not available
                    view_direction = np.array([0, 0, -1])
                    
            except Exception as e:
                # Fallback to default view direction if camera state fails
                print(f"Camera transform error: {e}, using default direction")
                view_direction = np.array([0, 0, -1])
            
            # Calculate distance from each point to the plane
            # Plane passes through data center, perpendicular to view direction
            plane_point = center
            
            # Distance from each point to the plane
            point_to_plane = x - plane_point
            distances_to_plane = np.dot(point_to_plane, view_direction)
            
            # Use same scaling approach as standard planes for consistency
            max_extent = np.max(extent)  # Use maximum extent for consistent scaling
            plane_offset = self.parent_gui.bisection_position * max_extent * 0.5
            
            if self.parent_gui.cross_section_mode:
                # Cross-section: only show cells within width from plane
                # Use the same logic as standard planes: distance from points to the offset plane
                distances_from_offset_plane = np.abs(distances_to_plane - plane_offset)
                filter_mask = distances_from_offset_plane <= self.parent_gui.cross_section_width * 0.5
            else:
                # Half-space: only show cells on one side of plane
                filter_mask = distances_to_plane <= plane_offset
                
        else:
            # Original XY, XZ, YZ plane logic
            plane_idx = {'XY': 2, 'XZ': 1, 'YZ': 0}[self.parent_gui.bisection_plane]
            plane_pos = center[plane_idx] + self.parent_gui.bisection_position * extent[plane_idx] * 0.5
            
            if self.parent_gui.cross_section_mode:
                # Cross-section: only show cells within width from plane
                distances = np.abs(x[:, plane_idx] - plane_pos)
                filter_mask = distances <= self.parent_gui.cross_section_width * 0.5
            else:
                # Half-space: only show cells on one side of plane
                filter_mask = x[:, plane_idx] <= plane_pos
            
        # Apply visibility filter for cell types (always when p_mask available)
        if p_mask is not None and len(self.parent_gui.visible_types) > 0:
            type_filter = np.isin(p_mask, list(self.parent_gui.visible_types))
            filter_mask = filter_mask & type_filter
        
        if filter_mask.any():
            filtered_x = x[filter_mask]
            filtered_p_mask = p_mask[filter_mask] if p_mask is not None else None
        else:
            filtered_x = np.empty((0, 3))
            filtered_p_mask = np.array([]) if p_mask is not None else None
            
        return filtered_x, filtered_p_mask, filter_mask
        
    def apply_cross_section_filter(self, x, p_mask):
        """Apply cross-section filtering when bisection is disabled"""
        if len(x) == 0:
            return x, p_mask, np.array([], dtype=bool)
            
        # Get bounds for data centering
        bounds = np.array([x.min(axis=0), x.max(axis=0)])
        center = bounds.mean(axis=0)
        
        # Cross-section mode creates a slice around the data center
        # Default to XY plane (Z=2) when used independently
        plane_idx = 2  # Z-axis for XY plane
        center_pos = center[plane_idx]
        
        # Show cells within width from the center plane
        distances = np.abs(x[:, plane_idx] - center_pos)
        filter_mask = distances <= self.parent_gui.cross_section_width * 0.5
        
        # Apply visibility filter for cell types (always when p_mask available)
        if p_mask is not None and len(self.parent_gui.visible_types) > 0:
            type_filter = np.isin(p_mask, list(self.parent_gui.visible_types))
            filter_mask = filter_mask & type_filter
        
        if filter_mask.any():
            filtered_x = x[filter_mask]
            filtered_p_mask = p_mask[filter_mask] if p_mask is not None else None
        else:
            filtered_x = np.empty((0, 3))
            filtered_p_mask = np.array([]) if p_mask is not None else None
            
        return filtered_x, filtered_p_mask, filter_mask
        
    def visualize_by_cell_type(self, x, p_mask, current_timestep, filter_mask):
        """Visualize cells colored by type"""
        for cell_type in self.parent_gui.unique_types:
            if cell_type not in self.parent_gui.visible_types:
                continue
                
            # Get cells of this type
            type_mask = p_mask == cell_type
            if not type_mask.any():
                continue
                
            type_positions = x[type_mask]
            color = self.parent_gui.cell_type_colors[cell_type]
            
            # Create scatter for this type
            scatter = visuals.Markers(scaling=True, alpha=1.0, spherical=True)
            scatter.set_data(type_positions, 
                           edge_width=0, 
                           face_color=color, 
                           size=self.parent_gui.cell_size)
            
            self.view.add(scatter)
            self.cell_scatters[cell_type] = scatter
            
    def visualize_single_type(self, x, current_timestep, filter_mask):
        """Visualize cells as single type"""
        # Use green color for default single type visualization
        color = 'green'
        
        scatter = visuals.Markers(scaling=True, alpha=1.0, spherical=True)
        scatter.set_data(x, edge_width=0, face_color=color, size=self.parent_gui.cell_size)
        
        self.view.add(scatter)
        self.cell_scatters['default'] = scatter
        
    def visualize_by_depth(self, x, current_timestep, filter_mask):
        """Visualize cells colored by their distance from camera (depth)"""
        import numpy as np
        from vispy.color import ColorArray
        
        # Get camera position using the same method as Camera Orthogonal mode
        try:
            camera = self.view.camera
            transform = camera.transform
            
            if transform is not None and hasattr(transform, 'map'):
                # Get camera position (eye point)
                camera_pos = transform.map([0, 0, 0, 1])[:3]
            else:
                # Fallback camera position
                camera_pos = np.array([0, 0, 10])
                
        except Exception as e:
            print(f"Camera position error: {e}, using default position")
            camera_pos = np.array([0, 0, 10])
        
        # Calculate distances from camera to each cell
        distances = np.linalg.norm(x - camera_pos, axis=1)
        
        # Normalize distances to 0-1 range for coloring
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist > min_dist:
            normalized_distances = (distances - min_dist) / (max_dist - min_dist)
        else:
            normalized_distances = np.zeros_like(distances)
        
        # Create depth-based colors using a blue (close) to red (far) colormap
        # Blue represents close objects, red represents far objects
        colors = np.zeros((len(x), 4))  # RGBA
        colors[:, 0] = normalized_distances  # Red channel increases with distance
        colors[:, 1] = 0.2  # Small amount of green for better color gradation
        colors[:, 2] = 1.0 - normalized_distances  # Blue channel decreases with distance
        colors[:, 3] = 1.0  # Alpha channel (full opacity)
        
        # Create scatter plot with depth-based colors
        scatter = visuals.Markers(scaling=True, alpha=1.0, spherical=True)
        scatter.set_data(x, edge_width=0, face_color=colors, size=self.parent_gui.cell_size)
        
        self.view.add(scatter)
        self.cell_scatters['depth'] = scatter

    def visualize_by_scalar(self, x, scalar_values):
        """Visualize cells colored by their scalar value (phi or theta) with magma-like gradient

        Range: -1 to 1
        Colors: -1 = dark purple (0, 0, 0.3), 0 = red/orange (1, 0.2, 0), 1 = light yellow (1, 1, 0.7)

        Args:
            x: Cell positions (Nx3 array)
            scalar_values: Scalar values for each cell (N array), already filtered
        """
        import numpy as np
        from vispy.color import ColorArray

        # Determine which scalar we're visualizing and get appropriate range
        if self.parent_gui.scalar_visualization_mode == "phi":
            scalar_min = self.parent_gui.phi_min
            scalar_max = self.parent_gui.phi_max
            scalar_range = self.parent_gui.phi_range
        else:  # theta
            scalar_min = self.parent_gui.theta_min
            scalar_max = self.parent_gui.theta_max
            scalar_range = self.parent_gui.theta_range

        # Normalize scalar values to [0, 1] range
        if scalar_range > 0:
            scalar_normalized = (scalar_values - scalar_min) / scalar_range
        else:
            scalar_normalized = np.zeros_like(scalar_values)

        # Clamp to [0, 1] for safety
        scalar_normalized = np.clip(scalar_normalized, 0.0, 1.0)

        # Create magma-like colors: dark purple (-1) -> red/orange (0) -> light yellow (1)
        # Define key colors for the magma colormap
        # -1 (val=0): Dark purple/black
        color_start = np.array([0.0, 0.0, 0.3])
        # 0 (val=0.5): Red/orange
        color_mid = np.array([1.0, 0.2, 0.0])
        # 1 (val=1): Light yellow
        color_end = np.array([1.0, 1.0, 0.7])

        colors = np.zeros((len(x), 4))  # RGBA

        for i, val in enumerate(scalar_normalized):
            if val < 0.5:
                # Lower half: dark purple -> red/orange
                t = val * 2  # Map [0, 0.5] to [0, 1]
                colors[i, :3] = (1 - t) * color_start + t * color_mid
            else:
                # Upper half: red/orange -> light yellow
                t = (val - 0.5) * 2  # Map [0.5, 1] to [0, 1]
                colors[i, :3] = (1 - t) * color_mid + t * color_end

        colors[:, 3] = 1.0  # Alpha channel (full opacity)

        # Create scatter plot with scalar-based colors
        scatter = visuals.Markers(scaling=True, alpha=1.0, spherical=True)
        scatter.set_data(x, edge_width=0, face_color=colors, size=self.parent_gui.cell_size)

        self.view.add(scatter)
        self.cell_scatters['scalar'] = scatter

    def _safe_add_wireframe(self, cell_vertices, faces, color, fallback_index):
        """Create wireframe visualization using Line visuals - much more stable than Mesh"""
        from vispy.scene import visuals
        
        # Ensure arrays are numpy
        verts = np.asarray(cell_vertices)
        faces = np.asarray(faces)

        # Basic shape checks
        if verts.ndim != 2 or verts.shape[1] != 3:
            print(f"[Voronoi] Skipping cell {fallback_index}: vertices shape {verts.shape} invalid")
            return False
        if faces.ndim != 2 or faces.shape[1] != 3:
            print(f"[Voronoi] Skipping cell {fallback_index}: faces shape {faces.shape} invalid")
            return False

        # Convert to correct dtype
        verts = verts.astype(np.float32)
        faces = faces.astype(np.int32)

        # Check for NaN/Inf in verts and validate face indices
        if not np.isfinite(verts).all():
            print(f"[Voronoi] Skipping cell {fallback_index}: vertex coordinates contain NaN/Inf")
            return False
        if faces.min() < 0:
            print(f"[Voronoi] Skipping cell {fallback_index}: negative face index {faces.min()}")
            return False
        if faces.max() >= len(verts):
            print(f"[Voronoi] Skipping cell {fallback_index}: face index {faces.max()} >= vertex count {len(verts)}")
            return False

        try:
            # Create wireframe by extracting unique edges from triangular faces
            edges = set()
            for face in faces:
                edges.add(tuple(sorted((int(face[0]), int(face[1])))))
                edges.add(tuple(sorted((int(face[1]), int(face[2])))))
                edges.add(tuple(sorted((int(face[2]), int(face[0])))))

            # Convert edges to pairwise line segments
            if not edges:
                print(f"[Voronoi] No valid edges for cell {fallback_index}")
                return False

            line_points = np.empty((len(edges) * 2, 3), dtype=np.float32)
            idx = 0
            for a, b in edges:
                line_points[idx] = verts[a]
                line_points[idx + 1] = verts[b]
                idx += 2

            # Ensure fully opaque lines so they are visible on dark backgrounds
            if isinstance(color, (list, tuple)) and len(color) == 4:
                line_color = (float(color[0]), float(color[1]), float(color[2]), 1.0)
            else:
                line_color = color

            # Use GL renderer and explicit connect indices for independent segments
            n_pts = line_points.shape[0]
            n_seg = n_pts // 2
            connect_idx = np.column_stack((2*np.arange(n_seg, dtype=np.int32), 2*np.arange(n_seg, dtype=np.int32) + 1))

            line_visual = visuals.Line(pos=line_points, connect=connect_idx, color=line_color, method='gl', width=1.0)
            self.view.add(line_visual)
            return line_visual

        except Exception as e:
            print(f"[Voronoi] Wireframe creation exception for cell {fallback_index}: {e}")
            return False

    def _safe_add_mesh(self, cell_vertices, faces, color, fallback_index):
        """Safely create and add a mesh with comprehensive validation"""
        from vispy.geometry import MeshData
        from vispy.scene import visuals
        
        # Ensure arrays are numpy
        verts = np.asarray(cell_vertices)
        faces = np.asarray(faces)

        # Basic shape checks
        if verts.ndim != 2 or verts.shape[1] != 3:
            print(f"[Voronoi] Skipping cell {fallback_index}: vertices shape {verts.shape} invalid")
            return False
        if faces.ndim != 2 or faces.shape[1] != 3:
            print(f"[Voronoi] Skipping cell {fallback_index}: faces shape {faces.shape} invalid")
            return False

        # Convert to correct dtype and C-contiguous
        verts = np.ascontiguousarray(verts.astype(np.float32))
        # VisPy is happiest with signed ints for indices
        faces = np.ascontiguousarray(faces.astype(np.int32))

        # Check for NaN/Inf in verts and negative indices in faces
        if not np.isfinite(verts).all():
            print(f"[Voronoi] Skipping cell {fallback_index}: vertex coordinates contain NaN/Inf")
            return False
        if faces.min() < 0:
            print(f"[Voronoi] Skipping cell {fallback_index}: negative face index {faces.min()}")
            return False
        if faces.max() >= len(verts):
            print(f"[Voronoi] Skipping cell {fallback_index}: face index {faces.max()} >= vertex count {len(verts)}")
            return False

        # Create MeshData and add — MeshData is slightly more robust
        try:
            md = MeshData(vertices=verts, faces=faces)
            mesh = visuals.Mesh(meshdata=md, color=color, shading='flat')
            # Do not manually override transform; scene graph handles this when adding to the view
            self.view.add(mesh)
            return mesh
        except Exception as e:
            print(f"[Voronoi] Mesh creation exception for cell {fallback_index}: {e}")
            return False

    def _safe_add_polygon_filled(self, poly3d, color, fallback_index):
        """Create a filled mesh for a 2D polygon embedded in 3D using triangle fan."""
        from vispy.scene import visuals
        from vispy.geometry import MeshData

        try:
            verts = np.asarray(poly3d, dtype=np.float32)
            if verts.ndim != 2 or verts.shape[1] != 3 or len(verts) < 3:
                return False
            n = len(verts)
            faces = np.column_stack((np.zeros(n-2, dtype=np.int32),
                                     np.arange(1, n-1, dtype=np.int32),
                                     np.arange(2, n, dtype=np.int32)))
            md = MeshData(vertices=verts, faces=faces)
            rgba = (float(color[0]), float(color[1]), float(color[2]), 0.8) if isinstance(color, (list, tuple)) and len(color) == 4 else (0.0, 0.8, 0.0, 0.8)
            mesh = visuals.Mesh(meshdata=md, color=rgba, shading='flat')
            self.view.add(mesh)
            return mesh
        except Exception as e:
            print(f"[Voronoi-2D] Polygon filled mesh creation failed for cell {fallback_index}: {e}")
            return False

    def visualize_voronoi(self, x, current_timestep, filter_mask):
        """Visualize cells as Voronoi tessellation in 3D or as a 2D sheet."""
        if current_timestep not in self.parent_gui.voronoi_cache:
            return

        mode = getattr(self.parent_gui, 'voronoi_mode', '3D Cells (bounded)')
        voronoi_data = self.parent_gui.voronoi_cache[current_timestep]
        if voronoi_data is None:
            return

        from vispy.scene import visuals

        # Determine colors based on current coloring mode
        if self.parent_gui.color_by_depth:
            colors = self.get_depth_colors(x)
        elif self.parent_gui.color_by_type and hasattr(self.parent_gui, 'p_mask_lst') and self.parent_gui.p_mask_lst[current_timestep] is not None:
            colors = self.get_type_colors(self.parent_gui.p_mask_lst[current_timestep])
        else:
            # Default green color
            colors = [(0, 0.8, 0, 1.0)] * len(x)  # Opaque for lines

        # Limit the number of cells to render to prevent memory issues
        MAX_VORONOI_CELLS = 500

        if mode.startswith('2D'):
            # voronoi_data is a list of polygons (list of Nx3 vertices) or None
            rendered = 0
            for i, (poly3d, color) in enumerate(zip(voronoi_data, colors)):
                if not filter_mask[i]:
                    continue
                if poly3d is None or len(poly3d) < 3:
                    continue
                if rendered >= MAX_VORONOI_CELLS:
                    break

                # Prefer filled polygon mesh; fallback to outline
                mesh_obj = self._safe_add_polygon_filled(poly3d, color, i)
                if mesh_obj:
                    if 'voronoi' not in self.cell_scatters:
                        self.cell_scatters['voronoi'] = []
                    self.cell_scatters['voronoi'].append(mesh_obj)
                    rendered += 1
                else:
                    try:
                        poly = np.asarray(poly3d, dtype=np.float32)
                        closed = np.vstack([poly, poly[:1]])
                        rgba = (float(color[0]), float(color[1]), float(color[2]), 1.0) if isinstance(color, (list, tuple)) and len(color) == 4 else color
                        line = visuals.Line(pos=closed, color=rgba, method='agg', width=1.2)
                        self.view.add(line)
                        if 'voronoi' not in self.cell_scatters:
                            self.cell_scatters['voronoi'] = []
                        self.cell_scatters['voronoi'].append(line)
                        rendered += 1
                    except Exception as e:
                        print(f"[Voronoi-2D] Failed to render polygon for cell {i}: {e}")
                        continue
            if rendered == 0:
                print("[Voronoi-2D] Rendered 0 polygons. Check plane fit and data density.")
            else:
                print(f"Rendered {rendered} 2D Voronoi polygons successfully")
            return

        # 3D path: voronoi_data is list of (vertices, faces)
        rendered_count = 0
        for i, ((cell_vertices, faces), color) in enumerate(zip(voronoi_data, colors)):
            if not filter_mask[i]:
                continue
            if rendered_count >= MAX_VORONOI_CELLS:
                print(f"Voronoi rendering limited to {MAX_VORONOI_CELLS} cells for performance")
                break
            if cell_vertices is None or faces is None:
                continue
            if len(cell_vertices) < 4:
                continue

            # Prefer filled mesh, fallback to wireframe then markers
            mesh_obj = self._safe_add_mesh(cell_vertices, faces, color, i)
            if mesh_obj:
                if 'voronoi' not in self.cell_scatters:
                    self.cell_scatters['voronoi'] = []
                self.cell_scatters['voronoi'].append(mesh_obj)
                rendered_count += 1
            else:
                wireframe_obj = self._safe_add_wireframe(cell_vertices, faces, color, i)
                if wireframe_obj:
                    if 'voronoi' not in self.cell_scatters:
                        self.cell_scatters['voronoi'] = []
                    self.cell_scatters['voronoi'].append(wireframe_obj)
                    rendered_count += 1
                else:
                    try:
                        scatter = visuals.Markers(scaling=True, alpha=1.0, spherical=True)
                        scatter.set_data([x[i]], edge_width=0, face_color=color, size=self.parent_gui.cell_size)
                        self.view.add(scatter)
                        if 'voronoi_fallback' not in self.cell_scatters:
                            self.cell_scatters['voronoi_fallback'] = []
                        self.cell_scatters['voronoi_fallback'].append(scatter)
                        rendered_count += 1
                    except Exception as e2:
                        print(f"Fallback sphere creation failed for cell {i}: {e2}")
                        continue

        if rendered_count == 0:
            print("[Voronoi] Rendered 0 cells. Possible reasons: filter hides all cells, cache missing/invalid, or distance cap too small.")
        else:
            print(f"Rendered {rendered_count} Voronoi cells successfully")
    
    def get_depth_colors(self, x):
        """Get depth-based colors for Voronoi cells"""
        try:
            camera = self.view.camera
            transform = camera.transform
            
            if transform is not None and hasattr(transform, 'map'):
                camera_pos = transform.map([0, 0, 0, 1])[:3]
            else:
                camera_pos = np.array([0, 0, 10])
        except:
            camera_pos = np.array([0, 0, 10])
        
        distances = np.linalg.norm(x - camera_pos, axis=1)
        min_dist, max_dist = np.min(distances), np.max(distances)
        
        if max_dist > min_dist:
            normalized = (distances - min_dist) / (max_dist - min_dist)
        else:
            normalized = np.zeros_like(distances)
        
        # Blue to red gradient with transparency
        colors = []
        for norm_dist in normalized:
            r = norm_dist
            g = 0.2
            b = 1.0 - norm_dist
            colors.append((r, g, b, 0.7))  # Semi-transparent
        
        return colors
    
    def get_type_colors(self, p_mask):
        """Get type-based colors for Voronoi cells"""
        colors = []
        type_colors = self.parent_gui.cell_type_colors
        
        for cell_type in p_mask:
            if cell_type in type_colors:
                color = type_colors[cell_type]
                # Convert to RGBA with transparency
                if isinstance(color, str):
                    # Simple color name to RGB conversion
                    color_map = {'red': (1,0,0), 'green': (0,1,0), 'blue': (0,0,1), 
                               'yellow': (1,1,0), 'cyan': (0,1,1), 'magenta': (1,0,1)}
                    rgb = color_map.get(color, (0.5, 0.5, 0.5))
                    colors.append((rgb[0], rgb[1], rgb[2], 0.7))
                else:
                    colors.append((*color[:3], 0.7))
            else:
                colors.append((0.5, 0.5, 0.5, 0.7))  # Gray default
        
        return colors
        
    def visualize_polarity(self, x, current_timestep, filter_mask):
        """Visualize polarity vectors"""
        if self.parent_gui.polarity_type == 'p':
            polarity_data = self.parent_gui.p_lst[current_timestep]
        else:
            polarity_data = self.parent_gui.q_lst[current_timestep]
            
        if polarity_data is None or len(polarity_data) == 0:
            return
            
        # Get the original data before any filtering to apply the same filter_mask
        original_x = self.parent_gui.x_lst[current_timestep]
        
        # The filter_mask passed to this function represents which cells are visible
        # We need to apply the same mask to the polarity data to ensure consistency
        if len(filter_mask) == len(original_x) and len(original_x) == len(polarity_data):
            # Apply the same filter mask to polarity data that was applied to cell positions
            filtered_polarity = polarity_data[filter_mask]
            filtered_x = x  # x is already filtered
        else:
            # Fallback: try to match dimensions
            min_len = min(len(x), len(polarity_data))
            if min_len == 0:
                return
            filtered_x = x[:min_len]
            filtered_polarity = polarity_data[:min_len]
        
        # Normalize polarity vectors
        norms = np.linalg.norm(filtered_polarity, axis=1)
        valid_polarity = norms > 0
        
        if not valid_polarity.any():
            return
            
        final_polarity = filtered_polarity[valid_polarity]
        final_x = filtered_x[valid_polarity]
        final_norms = norms[valid_polarity]
        
        normalized_polarity = final_polarity / final_norms[:, np.newaxis]
        
        # Calculate polarity vector endpoints using increased offset for better separation
        polarity_scale = 0.35  # Increased from 0.2 to create more separation and reduce Z-fighting
        polarity_positions = final_x + polarity_scale * normalized_polarity
        
        # Create polarity scatter with proper colors and same size as cells for visibility
        if self.parent_gui.polarity_type == 'p':
            polarity_color = 'red'  # Apicobasal polarity in red
        else:
            polarity_color = 'blue'  # Planar polarity in blue
        
        polarity_scatter = visuals.Markers(scaling=True, alpha=1.0, spherical=True)
        polarity_scatter.set_data(polarity_positions, 
                                edge_width=0, 
                                face_color=polarity_color, 
                                size=self.parent_gui.cell_size)  # Same size as cells for better visibility
        
        self.view.add(polarity_scatter)
        self.polarity_scatters['polarity'] = polarity_scatter
        
    def update_camera_bounds(self, x):
        """Update camera to fit all visible cells"""
        if len(x) == 0:
            return
            
        # Calculate bounds
        bounds = np.array([x.min(axis=0), x.max(axis=0)])
        center = bounds.mean(axis=0)
        extent = bounds[1] - bounds[0]
        
        # Add some padding
        padding = 0.1
        extent = extent * (1 + padding)
        
        # Set camera to encompass all cells
        self.view.camera.set_range(x=[center[0] - extent[0]/2, center[0] + extent[0]/2],
                                 y=[center[1] - extent[1]/2, center[1] + extent[1]/2],
                                 z=[center[2] - extent[2]/2, center[2] + extent[2]/2])
                                 
    def set_background_color(self, color_name):
        """Set the background color of the visualization"""
        color_map = {
            'black': 'black',
            'white': 'white', 
            'gray': 'gray'
        }
        self.canvas.bgcolor = color_map.get(color_name.lower(), 'black')
    
    def on_canvas_click(self, event):
        """Handle right-click for cell selection using ghost sphere approach"""
        # Check if neighbor search is enabled
        if not self.parent_gui.neighbor_search_enabled:
            return
        
        # Check if neighbor data exists for current timestep
        if self.parent_gui.current_timestep not in self.parent_gui.neighbor_data:
            return
        
        # Get current cell positions
        if not self.parent_gui.x_lst or len(self.parent_gui.x_lst) == 0:
            return
        
        x = self.parent_gui.x_lst[self.parent_gui.current_timestep]
        
        # Use the current ghost sphere position
        if self.ghost_sphere_position is None:
            return
        
        sphere_center = self.ghost_sphere_position
        sphere_radius = self.selection_sphere_radius
        
        # Find all cells within the ghost sphere
        cells_in_sphere = []
        for i, cell_pos in enumerate(x):
            pos_3d = cell_pos[:3] if len(cell_pos) > 3 else cell_pos
            center_3d = sphere_center[:3] if len(sphere_center) > 3 else sphere_center
            
            distance = np.linalg.norm(pos_3d - center_3d)
            
            if distance <= sphere_radius:
                cells_in_sphere.append((distance, i))
        
        if cells_in_sphere:
            # Sort by distance and select the closest one
            cells_in_sphere.sort()
            closest_dist, closest_idx = cells_in_sphere[0]
            self.parent_gui.on_cell_selected(closest_idx)
    
    def highlight_cells(self, selected_idx, neighbor_indices, show_only=False):
        """Highlight selected cell and its neighbors by drawing on top"""
        # Clear any existing highlights first
        self.clear_highlights()
        
        # Get current data
        x = self.parent_gui.x_lst[self.parent_gui.current_timestep]
        
        # Don't touch existing cells - just draw highlights on top that are slightly larger
        # Use same rendering parameters as regular cells for consistent appearance
        
        # Create highlight for selected cell (white) - slightly larger to cover original
        selected_pos = x[selected_idx:selected_idx+1]
        selected_scatter = visuals.Markers(scaling=True, alpha=1.0, spherical=True)
        selected_scatter.set_data(selected_pos,
                                 edge_width=0,
                                 face_color='white',
                                 size=self.parent_gui.cell_size * 1.15)  # 15% larger to cover original
        self.view.add(selected_scatter)
        self.highlighted_visuals['selected'] = selected_scatter
        
        # Create highlight for neighbors (bright yellow) - slightly larger to cover originals
        if len(neighbor_indices) > 0:
            neighbor_pos = x[neighbor_indices]
            neighbor_scatter = visuals.Markers(scaling=True, alpha=1.0, spherical=True)
            neighbor_scatter.set_data(neighbor_pos,
                                     edge_width=0,
                                     face_color='yellow',
                                     size=self.parent_gui.cell_size * 1.15)  # 15% larger to cover originals
            self.view.add(neighbor_scatter)
            self.highlighted_visuals['neighbors'] = neighbor_scatter
        
        # Update canvas
        self.canvas.update()
    def get_camera_centered_position(self, distance=5.0):
        """Get 3D position at specified distance in front of camera center
        
        Args:
            distance: Distance from camera (default 5.0 units)
            
        Returns:
            np.array: 3D world position
        """
        try:
            # Get camera position and direction
            camera = self.view.camera
            
            # Get camera transform
            cam_transform = camera.transform
            
            # Camera is looking down -Z axis in its local space
            # Transform the forward direction to world space
            forward = np.array([0, 0, -1, 0], dtype=np.float64)  # Forward in camera space
            forward_world = cam_transform.map(forward)[:3]
            
            # Get camera position
            camera_pos = np.array([0, 0, 0, 1], dtype=np.float64)
            camera_pos_world = cam_transform.map(camera_pos)[:3]
            
            # Calculate position at distance along forward direction
            position = camera_pos_world + forward_world * distance
            
            return position.astype(np.float64)
            
        except Exception as e:
            print(f"Error getting camera centered position: {e}")
            return np.array([0.0, 0.0, 0.0], dtype=np.float64)
    
    def clear_highlights(self):
        """Clear all highlighting and restore original visualization"""
        # Remove highlight visuals
        for visual in self.highlighted_visuals.values():
            visual.parent = None
        self.highlighted_visuals.clear()
        
        # Clear state and trigger full re-render
        # This is simpler and more reliable than trying to restore individual scatter properties
        self.pre_highlight_state = {}
        
        # Trigger full visualization update to restore original state
        if self.parent_gui:
            self.parent_gui.update_visualization()
        
        self.canvas.update()
        
    def export_image(self, filename):
        """Export current view as image"""
        try:
            # Render the scene to an image
            img = self.canvas.render()
            # Save using vispy's built-in functionality
            vispy.io.write_png(filename, img)
            return True
        except Exception as e:
            print(f"Error exporting image: {e}")
            return False

class DataVisualizationGUI(QMainWindow):
    """
    GUI for visualizing and analyzing 3D cell simulation data exported from the main simulation
    """
    
    def __init__(self):
        super().__init__()
        
        # Data storage
        self.data = None
        self.sim_dict = None
        self.data_folder = None  # Store the folder path for cache files
        self.current_timestep = 0
        self.playing = False
        self.play_speed = 20  # milliseconds between frames
        
        # Visualization settings
        self.show_polarity = False
        self.polarity_type = "p"  # "p" or "q"
        self.phi_lst = None  # Initialize phi list (new 5th data element)
        self.theta_lst = None  # Initialize theta list (new 6th data element)
        self.color_by_type = True
        self.color_by_depth = False
        self.color_by_phi = False  # Color by phi scalar gradient
        self.scalar_visualization_mode = "phi"  # "phi" or "theta"
        self.show_axes = False  # Show coordinate axes
        self.phi_min = -1.0
        self.phi_max = 1.0
        self.phi_range = 2.0
        self.theta_min = -1.0
        self.theta_max = 1.0
        self.theta_range = 2.0
        self.visible_types = set()
        self.cell_type_colors = {}
        
        # Bisection settings
        self.bisection_enabled = False
        self.bisection_plane = "XY"
        self.bisection_position = 0.0
        self.cross_section_mode = False
        self.cross_section_width = 2.0
        
        # Camera and rendering
        self.cell_size = 2.0
        self.background_color = "black"
        
        # Camera control state
        self.last_camera_position = None
        self.last_camera_view_direction = None
        self.camera_moving = False
        self.user_editing_camera = False  # Track if user is currently editing camera fields
        self.camera_stillness_counter = 0  # Counter for detecting when camera stops moving
        
        # Neighbor search state
        self.neighbor_search_enabled = False
        self.neighbor_data = {}  # Dict[timestep] -> {'true_neighbors': list, 'potential_neighbors': (d, idx)}
        self.selected_cell_idx = None
        self.neighbor_search_button = None
        self.neighbor_stats_browser = None
        self.neighbor_clear_button = None
        self.neighbor_show_only_checkbox = None
        self.seethru_spinbox = None
        self.max_neighbors_spinbox = None
        
        self.setWindowTitle("3D Cell Data Visualization & Analysis")
        self.setGeometry(100, 100, 1400, 900)
        
        # Initialize data loading
        # self.show_intro_dialog()  # Disabled - startup dialog
        self.init_data_loading()
        
    def show_intro_dialog(self):
        """Show introduction dialog explaining the GUI"""
        intro_dialog = QDialog(self)
        intro_dialog.setWindowTitle("3D Cell Data Visualization & Analysis")
        intro_dialog.setMinimumSize(700, 500)
        
        layout = QVBoxLayout(intro_dialog)
        
        # Create text browser for scrollable content
        text_browser = QTextBrowser()
        text_browser.setHtml("""
        <h2>3D Cell Data Visualization & Analysis Tool</h2>
        
        <p>This application provides advanced visualization and analysis capabilities for 3D cell simulation data 
        exported from the main simulation GUI.</p>
        
        <h3>📁 Data Loading</h3>
        <ul>
        <li>Load simulation folders containing data.npy and sim_dict.json files</li>
        <li>Switch between different datasets during analysis</li>
        <li>View comprehensive simulation parameters and metadata</li>
        </ul>
        
        <h3>🎬 Playback Controls</h3>
        <ul>
        <li><b>Timestep Slider:</b> Navigate through simulation timesteps</li>
        <li><b>Play/Pause:</b> Animate the simulation at adjustable speed</li>
        <li><b>Frame Navigation:</b> Step forward/backward through frames</li>
        <li><b>Speed Control:</b> Adjust animation playback speed</li>
        </ul>
        
        <h3>🎨 Visualization Features</h3>
        <ul>
        <li><b>Cell Size:</b> Adjust visual size of cells</li>
        <li><b>Polarity Vectors:</b> Display apicobasal (p) or planar (q) cell polarity</li>
        <li><b>Background:</b> Choose black, white, or gray background</li>
        <li><b>3D Navigation:</b> Use mouse to fly around the 3D scene</li>
        </ul>
        
        <h3>🔵 Cell Type Analysis</h3>
        <ul>
        <li><b>Color by Type:</b> Visualize different cell types with distinct colors</li>
        <li><b>Type Visibility:</b> Toggle visibility of individual cell types</li>
        <li><b>Type Information:</b> View color coding and type numbers</li>
        </ul>
        
        <h3>✂️ Data Sectioning</h3>
        <ul>
        <li><b>Plane Selection:</b> Choose XY, XZ, or YZ sectioning planes</li>
        <li><b>Position Control:</b> Adjust plane position to reveal internal structure</li>
        <li><b>Cross-Section Mode:</b> View thin slices through the data</li>
        <li><b>Section Width:</b> Control thickness of cross-sections</li>
        </ul>
        
        <h3>💾 Export Capabilities</h3>
        <ul>
        <li><b>High-Quality Images:</b> Export publication-ready PNG/JPEG images</li>
        <li><b>Video Creation:</b> Generate MP4/AVI videos of simulations</li>
        <li><b>Custom Frame Ranges:</b> Export specific timestep ranges</li>
        <li><b>Adjustable Quality:</b> Control frame rate and video quality</li>
        </ul>
        
        <h3>🖱️ 3D Navigation Controls</h3>
        <ul>
        <li><b>Left Mouse:</b> Rotate view around the scene</li>
        <li><b>Right Mouse:</b> Pan/translate the view</li>
        <li><b>Mouse Wheel:</b> Zoom in/out</li>
        <li><b>Spacebar:</b> Toggle animation play/pause</li>
        </ul>
        
        <p><b>Note:</b> This tool requires simulation data exported from the main 3D cell simulation GUI. 
        Select a folder containing 'data.npy' and 'sim_dict.json' files to begin analysis.</p>
        """)
        
        layout.addWidget(text_browser)
        
        # Add OK button
        ok_button = QPushButton("Start Analysis")
        ok_button.clicked.connect(intro_dialog.accept)
        layout.addWidget(ok_button)
        
        intro_dialog.exec()
        
    def init_data_loading(self):
        """Initialize by prompting user to select data folder"""
        while True:  # Keep asking until valid data is loaded or user cancels
            folder_path = QFileDialog.getExistingDirectory(
                self, "Select Simulation Data Folder",
                "",  # Start in current directory
                QFileDialog.Option.ShowDirsOnly
            )
            
            if not folder_path:
                # Ask if user wants to create test data
                reply = QMessageBox.question(
                    self, "No Data Selected",
                    "No data folder selected. Would you like to:\n\n"
                    "• Create sample test data for demonstration?\n"
                    "• Exit the application?\n\n"
                    "Click Yes to create test data, No to exit.",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes
                )
                
                if reply == QMessageBox.StandardButton.Yes:
                    if self.create_test_data():
                        self.init_ui()
                        return
                    else:
                        QMessageBox.critical(self, "Error", "Failed to create test data. Exiting.")
                        self.close()
                        return
                else:
                    self.close()
                    return
                
            elif self.load_data_folder(folder_path):
                self.init_ui()
                return
            else:
                # Data loading failed, ask if user wants to try again
                reply = QMessageBox.question(
                    self, "Load Failed",
                    "Failed to load data from selected folder.\n\n"
                    "Would you like to select a different folder?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes
                )
                
                if reply == QMessageBox.StandardButton.No:
                    self.close()
                    return
                # Continue loop to ask for another folder
                
    def create_test_data(self):
        """Create test data for demonstration purposes"""
        try:
            # Ask user where to save test data
            folder_path = QFileDialog.getExistingDirectory(
                self, "Select Folder to Create Test Data",
                "",
                QFileDialog.Option.ShowDirsOnly
            )
            
            if not folder_path:
                return False
                
            test_folder = os.path.join(folder_path, "test_simulation_data")
            os.makedirs(test_folder, exist_ok=True)
            
            # Generate test simulation data
            np.random.seed(42)  # For reproducible test data
            n_timesteps = 50
            n_cells = 200
            
            # Initialize data lists
            p_mask_lst = []
            x_lst = []
            p_lst = []
            q_lst = []
            phi_lst = []
            
            # Generate expanding cluster simulation
            for t in range(n_timesteps):
                # Cell types (0 and 1)
                p_mask = np.random.choice([0, 1], size=n_cells, p=[0.7, 0.3])
                
                # Positions - expanding cluster over time
                expansion_factor = 1.0 + 0.02 * t  # Gradual expansion
                center = np.array([0.0, 0.0, 0.0])
                
                # Create clusters for each cell type
                positions = np.zeros((n_cells, 3))
                
                # Type 0 cells - central cluster
                type0_mask = p_mask == 0
                n_type0 = np.sum(type0_mask)
                if n_type0 > 0:
                    # Spherical distribution
                    phi = np.random.uniform(0, 2*np.pi, n_type0)
                    costheta = np.random.uniform(-1, 1, n_type0)
                    theta = np.arccos(costheta)
                    radii = np.random.gamma(2, expansion_factor * 3, n_type0)
                    
                    positions[type0_mask, 0] = radii * np.sin(theta) * np.cos(phi)
                    positions[type0_mask, 1] = radii * np.sin(theta) * np.sin(phi)
                    positions[type0_mask, 2] = radii * np.cos(theta)
                
                # Type 1 cells - outer ring
                type1_mask = p_mask == 1
                n_type1 = np.sum(type1_mask)
                if n_type1 > 0:
                    phi = np.random.uniform(0, 2*np.pi, n_type1)
                    costheta = np.random.uniform(-1, 1, n_type1)
                    theta = np.arccos(costheta)
                    radii = np.random.gamma(3, expansion_factor * 5, n_type1) + expansion_factor * 8
                    
                    positions[type1_mask, 0] = radii * np.sin(theta) * np.cos(phi)
                    positions[type1_mask, 1] = radii * np.sin(theta) * np.sin(phi)
                    positions[type1_mask, 2] = radii * np.cos(theta)
                
                # Add some noise and movement
                positions += np.random.normal(0, 0.5, positions.shape)
                
                # Polarity vectors
                # Apicobasal polarity - pointing outward from center
                p_vectors = positions - center
                p_norms = np.linalg.norm(p_vectors, axis=1)
                p_norms[p_norms == 0] = 1
                p_vectors = p_vectors / p_norms[:, np.newaxis]
                
                # Add some randomness to polarity
                p_vectors += np.random.normal(0, 0.2, p_vectors.shape)
                p_norms = np.linalg.norm(p_vectors, axis=1)
                p_norms[p_norms == 0] = 1
                p_vectors = p_vectors / p_norms[:, np.newaxis]
                
                # Planar polarity - random in xy plane
                q_vectors = np.random.normal(0, 1, (n_cells, 3))
                q_vectors[:, 2] = 0  # Keep in xy plane
                q_norms = np.linalg.norm(q_vectors, axis=1)
                q_norms[q_norms == 0] = 1
                q_vectors = q_vectors / q_norms[:, np.newaxis]

                # Phi scalars - random values in [-1, 1]
                phi_scalars = np.random.uniform(-1, 1, n_cells)

                # Theta scalars - random values in [-1, 1] with different distribution
                theta_scalars = np.random.uniform(-1, 1, n_cells)

                # Store data
                p_mask_lst.append(p_mask)
                x_lst.append(positions)
                p_lst.append(p_vectors)
                q_lst.append(q_vectors)
                phi_lst.append(phi_scalars)
                theta_lst.append(theta_scalars)
            
            # Save data
            data = (p_mask_lst, x_lst, p_lst, q_lst, phi_lst, theta_lst)
            with open(os.path.join(test_folder, "data.npy"), 'wb') as f:
                pickle.dump(data, f)
            
            # Create simulation dictionary
            sim_dict = {
                "simulation_name": "Test Simulation Data",
                "n_timesteps": n_timesteps,
                "n_cells_initial": n_cells,
                "cell_types": [0, 1],
                "simulation_type": "expansion_demo",
                "created_by": "Data Visualization GUI Test Generator",
                "notes": "Automatically generated test data for GUI demonstration"
            }
            
            with open(os.path.join(test_folder, "sim_dict.json"), 'w') as f:
                json.dump(sim_dict, f, indent=2)
            
            # Load the test data
            return self.load_data_folder(test_folder)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create test data:\n{str(e)}")
            return False
            
    def load_data_folder(self, folder_path):
        """Load data.npy and sim_dict.json from folder"""
        try:
            # Store the data folder path for cache files
            self.data_folder = folder_path
            
            # Load data.npy
            data_path = os.path.join(folder_path, "data.npy")
            if not os.path.exists(data_path):
                QMessageBox.critical(self, "Error", "data.npy not found in selected folder")
                return False
                
            with open(data_path, 'rb') as f:
                self.data = pickle.load(f)
                
            # Unpack data: (p_mask_lst, x_lst, p_lst, q_lst) or (p_mask_lst, x_lst, p_lst, q_lst, phi_lst) or (p_mask_lst, x_lst, p_lst, q_lst, phi_lst, theta_lst)
            if len(self.data) == 4:
                # Old format (backwards compatible)
                self.p_mask_lst, self.x_lst, self.p_lst, self.q_lst = self.data
                self.phi_lst = None
                self.theta_lst = None
            elif len(self.data) == 5:
                # Format with phi data
                self.p_mask_lst, self.x_lst, self.p_lst, self.q_lst, self.phi_lst = self.data
                self.theta_lst = None
            elif len(self.data) == 6:
                # Format with phi and theta data
                self.p_mask_lst, self.x_lst, self.p_lst, self.q_lst, self.phi_lst, self.theta_lst = self.data
            else:
                raise ValueError(f"Unexpected data format with {len(self.data)} elements. Expected 4, 5, or 6.")
            
            # Load sim_dict.json if it exists
            sim_dict_path = os.path.join(folder_path, "sim_dict.json")
            if os.path.exists(sim_dict_path):
                with open(sim_dict_path, 'r') as f:
                    self.sim_dict = json.load(f)
            else:
                self.sim_dict = {"notes": "No simulation parameters found"}
                
            # Initialize visualization parameters
            self.max_timesteps = len(self.x_lst)
            self.current_timestep = 0
            
            # Analyze cell types
            self.analyze_cell_types()

            # Compute global phi range for normalization (if phi data exists)
            if self.phi_lst is not None:
                all_phi = np.concatenate(self.phi_lst)
                self.phi_min = np.min(all_phi)
                self.phi_max = np.max(all_phi)
                self.phi_range = self.phi_max - self.phi_min
                if self.phi_range == 0:
                    self.phi_range = 1.0  # Avoid division by zero

            # Compute global theta range for normalization (if theta data exists)
            if self.theta_lst is not None:
                all_theta = np.concatenate(self.theta_lst)
                self.theta_min = np.min(all_theta)
                self.theta_max = np.max(all_theta)
                self.theta_range = self.theta_max - self.theta_min
                if self.theta_range == 0:
                    self.theta_range = 1.0  # Avoid division by zero

            return True
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load data:\n{str(e)}")
            return False
            
    def analyze_cell_types(self):
        """Analyze available cell types from p_mask data"""
        self.unique_types = set()
        
        for p_mask in self.p_mask_lst:
            if p_mask is not None:
                self.unique_types.update(np.unique(p_mask))
        
        # If no types found, assume single type
        if not self.unique_types:
            self.unique_types = {0}
            
        # Initialize type visibility and colors
        self.visible_types = set(self.unique_types)
        
        # Create distinct colors for each type
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.unique_types)))
        for i, cell_type in enumerate(sorted(self.unique_types)):
            self.cell_type_colors[cell_type] = colors[i][:3]  # RGB only
            
    def init_ui(self):
        """Initialize the user interface"""
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Create splitter for resizable panels (1/4 control panel, 3/4 visualization)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout = QHBoxLayout(main_widget)
        main_layout.addWidget(splitter)
        
        # Create control panel (left side - 1/4 width)
        self.control_panel = self.create_control_panel()
        splitter.addWidget(self.control_panel)
        
        # Create visualization panel (right side - 3/4 width)
        self.vis_panel = self.create_visualization_panel()
        splitter.addWidget(self.vis_panel)
        
        # Set splitter proportions (1:3 ratio)
        splitter.setSizes([350, 1050])  # Approximate 1/4 vs 3/4
        
        # Apply main window styling
        self.apply_main_styling()
        
        # Setup timer for animation
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        
        # Setup timer for camera updates
        self.camera_update_timer = QTimer()
        self.camera_update_timer.timeout.connect(self.update_camera_controls)
        self.camera_update_timer.start(100)  # Update every 100ms for smooth live updates
        
        # Initialize visualization
        self.update_visualization()
        
    def apply_main_styling(self):
        """Apply main window styling to match the main GUI"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                background-color: #1e1e1e;
            }
            QSlider::groove:horizontal {
                border: 1px solid #bdc3c7;
                height: 6px;
                background: #ecf0f1;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #2c4a6b;
                border: 1px solid #1e3045;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #1e3045;
            }
            QPushButton {
                background-color: #ecf0f1;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 12px;
                color: #2c3e50;
            }
            QPushButton:hover {
                background-color: #d5dbdb;
            }
            QPushButton:pressed {
                background-color: #bdc3c7;
            }
            QCheckBox {
                font-size: 12px;
                color: white;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #bdc3c7;
                border-radius: 3px;
                background-color: white;
            }
            QCheckBox::indicator:checked {
                background-color: #3498db;
                border-color: #2980b9;
            }
            QComboBox {
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                padding: 4px 8px;
                background-color: white;
                color: #2c3e50;
            }
            QComboBox:hover {
                border-color: #2980b9;
            }
            QTextBrowser {
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                background-color: white;
                padding: 4px;
                color: #2c3e50;
            }
            QListWidget {
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                background-color: white;
                color: #2c3e50;
            }
            QLabel {
                color: white;
            }
        """)
        
    def create_control_panel(self):
        """Create the left control panel with collapsible sections"""
        # Main control widget
        control_widget = QWidget()
        control_widget.setStyleSheet("""
            QWidget {
                background-color: #b8b8b8;
                border: none;
                border-radius: 8px;
                margin: 2px;
            }
        """)
        control_layout = QVBoxLayout(control_widget)
        control_layout.setContentsMargins(12, 12, 12, 12)
        control_layout.setSpacing(8)
        control_widget.setMaximumWidth(400)
        control_widget.setMinimumWidth(300)
        
        # Add title
        title_label = QLabel("Data Analysis Controls")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(14)
        title_label.setFont(title_font)
        title_label.setStyleSheet("""
            QLabel {
                color: #2c3e50;
                padding: 10px;
                background-color: #ecf0f1;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                margin: 5px 0px;
            }
        """)
        control_layout.addWidget(title_label)
        
        # Create scroll area for controls
        scroll_area = QScrollArea()
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollArea > QWidget > QWidget {
                background-color: transparent;
            }
        """)
        scroll_widget = QWidget()
        scroll_widget.setStyleSheet("background-color: transparent;")
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(5, 5, 5, 5)
        scroll_layout.setSpacing(8)
        
        # Add control sections
        scroll_layout.addWidget(self.create_data_info_section())
        scroll_layout.addWidget(self.create_playback_controls())
        scroll_layout.addWidget(self.create_camera_controls())
        scroll_layout.addWidget(self.create_visualization_controls())
        scroll_layout.addWidget(self.create_cell_types_section())
        scroll_layout.addWidget(self.create_data_bisection_section())
        scroll_layout.addWidget(self.create_neighbor_search_section())
        scroll_layout.addWidget(self.create_export_section())
        
        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        control_layout.addWidget(scroll_area)
        
        return control_widget
        
    def create_data_info_section(self):
        """Create collapsible data information section"""
        section = CollapsibleSection("DATA INFO", False)
        
        # Create info text (use QTextBrowser with scroll position management)
        self.info_text = QTextBrowser()
        self.info_text.setMinimumHeight(200)
        self.info_text.setMaximumHeight(300)
        self.info_text.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.info_text.setStyleSheet("""
            QTextBrowser {
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                background-color: white;
                padding: 8px;
                color: #2c3e50;
                font-family: monospace;
                font-size: 11px;
            }
        """)
        self.update_data_info()
        section.add_widget(self.info_text)
        
        # Add load new data button
        load_button = QPushButton("📁 Load New Dataset")
        load_button.clicked.connect(self.load_new_dataset)
        # Style button with fall-inspired warm orange colors
        load_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #d17a47, stop: 1 #b85f2f);
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 12px;
                color: white;
                min-width: 120px;
            }
            QPushButton:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #e08a56, stop: 1 #d17a47);
            }
            QPushButton:pressed {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #b85f2f, stop: 1 #9f4d1e);
            }
        """)
        section.add_widget(load_button)
        
        return section
        
    def create_camera_controls(self):
        """Create camera controls section"""
        section = CollapsibleSection("CAMERA CONTROLS", False)  # Start minimized
        
        # Create a grid layout for organized display
        grid_widget = QWidget()
        grid_layout = QGridLayout(grid_widget)
        grid_layout.setSpacing(4)
        grid_layout.setContentsMargins(4, 4, 4, 4)
        
        # Camera position labels and inputs
        pos_label = QLabel("Position (X, Y, Z):")
        pos_label.setStyleSheet("font-weight: bold; color: #2c3e50;")
        grid_layout.addWidget(pos_label, 0, 0, 1, 3)
        
        # Position input fields
        self.pos_x_input = QtWidgets.QDoubleSpinBox()
        self.pos_y_input = QtWidgets.QDoubleSpinBox()  
        self.pos_z_input = QtWidgets.QDoubleSpinBox()
        
        for spinbox in [self.pos_x_input, self.pos_y_input, self.pos_z_input]:
            spinbox.setRange(-9999.0, 9999.0)
            spinbox.setDecimals(2)
            spinbox.setSingleStep(0.1)
            spinbox.setStyleSheet("""
                QDoubleSpinBox {
                    background-color: white;
                    border: 1px solid #bdc3c7;
                    border-radius: 3px;
                    padding: 2px;
                    font-family: monospace;
                    font-size: 11px;
                }
            """)
            # Connect focus events to track user editing
            spinbox.focusInEvent = lambda event, sb=spinbox: self.on_camera_input_focus_in(event, sb)
            spinbox.focusOutEvent = lambda event, sb=spinbox: self.on_camera_input_focus_out(event, sb)
            # Connect value change events to track user modifications
            spinbox.valueChanged.connect(lambda value, sb=spinbox: self.on_camera_input_value_changed(value, sb))
            
        # Track user-modified values to prevent overwriting during live updates
        self.user_modified_position = False
        self.last_user_position = None
            
        grid_layout.addWidget(QLabel("X:"), 1, 0)
        grid_layout.addWidget(self.pos_x_input, 1, 1)
        grid_layout.addWidget(QLabel("Y:"), 2, 0)
        grid_layout.addWidget(self.pos_y_input, 2, 1)
        grid_layout.addWidget(QLabel("Z:"), 3, 0)
        grid_layout.addWidget(self.pos_z_input, 3, 1)
        
        # View direction labels and inputs
        view_label = QLabel("View Direction:")
        view_label.setStyleSheet("font-weight: bold; color: #2c3e50; margin-top: 8px;")
        grid_layout.addWidget(view_label, 4, 0, 1, 3)
        
        self.view_x_input = QtWidgets.QDoubleSpinBox()
        self.view_y_input = QtWidgets.QDoubleSpinBox()
        self.view_z_input = QtWidgets.QDoubleSpinBox()
        
        for spinbox in [self.view_x_input, self.view_y_input, self.view_z_input]:
            spinbox.setRange(-1.0, 1.0)
            spinbox.setDecimals(3)
            spinbox.setSingleStep(0.01)
            spinbox.setReadOnly(True)  # Make read-only since we can't reliably set view direction
            spinbox.setStyleSheet("""
                QDoubleSpinBox {
                    background-color: #f8f9fa;
                    border: 1px solid #bdc3c7;
                    border-radius: 3px;
                    padding: 2px;
                    font-family: monospace;
                    font-size: 11px;
                    color: #6c757d;
                }
            """)
            
        grid_layout.addWidget(QLabel("X:"), 5, 0)
        grid_layout.addWidget(self.view_x_input, 5, 1)
        grid_layout.addWidget(QLabel("Y:"), 6, 0)
        grid_layout.addWidget(self.view_y_input, 6, 1)
        grid_layout.addWidget(QLabel("Z:"), 7, 0)
        grid_layout.addWidget(self.view_z_input, 7, 1)
        
        # Rotation display (Q/E movement)
        rotation_label = QLabel("Roll (Q/E):")
        rotation_label.setStyleSheet("font-weight: bold; color: #2c3e50; margin-top: 8px;")
        grid_layout.addWidget(rotation_label, 8, 0, 1, 3)
        
        self.rotation_input = QtWidgets.QDoubleSpinBox()
        self.rotation_input.setRange(-360.0, 360.0)
        self.rotation_input.setDecimals(1)
        self.rotation_input.setSingleStep(1.0)
        self.rotation_input.setReadOnly(True)
        self.rotation_input.setStyleSheet("""
            QDoubleSpinBox {
                background-color: #f8f9fa;
                border: 1px solid #bdc3c7;
                border-radius: 3px;
                padding: 2px;
                font-family: monospace;
                font-size: 11px;
                color: #6c757d;
            }
        """)
        
        grid_layout.addWidget(QLabel("Angle (°):"), 9, 0)
        grid_layout.addWidget(self.rotation_input, 9, 1)
        
        section.add_widget(grid_widget)
        
        # Status indicator
        self.camera_status_label = QLabel("🔄 Live Updates Active")
        self.camera_status_label.setStyleSheet("""
            QLabel {
                color: #27ae60;
                font-size: 10px;
                font-style: italic;
                padding: 2px;
                border-radius: 3px;
                background-color: rgba(39, 174, 96, 0.1);
            }
        """)
        self.camera_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        section.add_widget(self.camera_status_label)
        
        # Control buttons
        button_layout = QVBoxLayout()
        
        # Jump to position button (full width, reset camera colors)
        jump_button = QPushButton("🎯 Jump to Position")
        jump_button.clicked.connect(self.jump_to_specified_camera_position)
        jump_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #d17a47, stop: 1 #b85f2f);
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
                font-size: 11px;
                color: white;
                min-width: 100px;
            }
            QPushButton:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #e08a56, stop: 1 #d17a47);
            }
            QPushButton:pressed {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #b85f2f, stop: 1 #9d4f28);
            }
        """)
        button_layout.addWidget(jump_button)
        
        # Reset camera view button (export image colors)
        reset_button = QPushButton("🎯 Reset Camera View")
        reset_button.clicked.connect(self.reset_camera_view)
        reset_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #6b8e5a, stop: 1 #4d6842);
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
                font-size: 11px;
                color: white;
                min-width: 100px;
            }
            QPushButton:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #7da46a, stop: 1 #6b8e5a);
            }
            QPushButton:pressed {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #4d6842, stop: 1 #3d5234);
            }
        """)
        button_layout.addWidget(reset_button)
        
        section.add_layout(button_layout)
        
        return section
        
    def get_camera_view_direction(self):
        """Get the current camera view direction for Camera Orthogonal mode"""
        try:
            camera = self.canvas_widget.view.camera
            transform = camera.transform
            
            if transform is not None and hasattr(transform, 'map'):
                # Forward vector in camera space is (0, 0, -1)
                # Transform to world space to get view direction
                view_direction = transform.map([0, 0, -1, 0])[:3]
                
                # Normalize the view direction
                if np.linalg.norm(view_direction) > 0:
                    view_direction = view_direction / np.linalg.norm(view_direction)
                else:
                    view_direction = np.array([0, 0, -1])  # Fallback
            else:
                # Fallback if transform not available
                view_direction = np.array([0, 0, -1])
                
        except Exception as e:
            # Fallback to default view direction if camera state fails
            print(f"Camera transform error: {e}, using default direction")
            view_direction = np.array([0, 0, -1])
            
        return view_direction
        
    def update_data_info(self):
        """Update the data info display"""
        if not hasattr(self, 'info_text'):
            return
            
        # Save current scroll position
        scrollbar = self.info_text.verticalScrollBar()
        scroll_position = scrollbar.value()
            
        # Format simulation info with HTML formatting
        info_str = f"<b>Simulation Data</b><br>"
        info_str += f"Timesteps: {self.max_timesteps}<br>"
        info_str += f"Current: {self.current_timestep + 1}/{self.max_timesteps}<br>"
        
        # Add cell count information
        if hasattr(self, 'x_lst') and self.x_lst:
            current_cells = len(self.x_lst[self.current_timestep]) if self.x_lst[self.current_timestep] is not None else 0
            info_str += f"Current Cells: {current_cells}<br>"
            
        # Add cell type information
        if hasattr(self, 'unique_types') and self.unique_types:
            info_str += f"Cell Types: {sorted(list(self.unique_types))}<br>"
            
        if self.sim_dict:
            info_str += f"<br><b>Parameters</b><br>"
            for key, value in self.sim_dict.items():
                if key not in ['notes', 'output_folder']:
                    # Truncate long values
                    str_value = str(value)
                    if len(str_value) > 50:
                        str_value = str_value[:47] + "..."
                    info_str += f"{key}: {str_value}<br>"
                    
        self.info_text.setHtml(info_str)
        
        # Restore scroll position after a short delay to ensure content is rendered
        QTimer.singleShot(10, lambda: scrollbar.setValue(scroll_position))
        
    def create_playback_controls(self):
        """Create playback control section"""
        section = CollapsibleSection("PLAYBACK", False)
        
        # Timestep slider
        timestep_layout = QVBoxLayout()
        timestep_control_layout = QHBoxLayout()
        
        # Slider and value input
        self.timestep_slider = QSlider(Qt.Orientation.Horizontal)
        self.timestep_slider.setMinimum(0)
        self.timestep_slider.setMaximum(self.max_timesteps - 1)
        self.timestep_slider.setValue(0)
        self.timestep_slider.valueChanged.connect(self.on_timestep_changed)
        
        self.timestep_spinbox = QSpinBox()
        self.timestep_spinbox.setMinimum(0)
        self.timestep_spinbox.setMaximum(self.max_timesteps - 1)
        self.timestep_spinbox.setValue(0)
        self.timestep_spinbox.valueChanged.connect(self.on_timestep_spinbox_changed)
        self.timestep_spinbox.setFixedWidth(80)
        
        timestep_control_layout.addWidget(QLabel("Timestep:"))
        timestep_control_layout.addWidget(self.timestep_slider)
        timestep_control_layout.addWidget(self.timestep_spinbox)
        
        # Unit explanation
        unit_label = QLabel("Unit: timestep")
        unit_label.setStyleSheet("color: gray; font-size: 10px;")
        
        timestep_layout.addLayout(timestep_control_layout)
        timestep_layout.addWidget(unit_label)
        section.add_layout(timestep_layout)
        
        # Play controls
        controls_layout = QHBoxLayout()
        
        self.play_button = QPushButton("▶️ Play")
        self.play_button.clicked.connect(self.toggle_play)
        controls_layout.addWidget(self.play_button)
        
        prev_button = QPushButton("⏮️")
        prev_button.clicked.connect(self.prev_frame)
        controls_layout.addWidget(prev_button)
        
        next_button = QPushButton("⏭️")
        next_button.clicked.connect(self.next_frame)
        controls_layout.addWidget(next_button)
        
        section.add_layout(controls_layout)
        
        # Speed control
        speed_layout = QVBoxLayout()
        speed_control_layout = QHBoxLayout()
        
        # Slider and value input
        speed_control_layout.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(100)
        self.speed_slider.setValue(20)
        self.speed_slider.valueChanged.connect(self.on_speed_changed)
        
        self.speed_spinbox = QSpinBox()
        self.speed_spinbox.setMinimum(1)
        self.speed_spinbox.setMaximum(100)
        # Set initial inverted value: if slider is 20, spinbox shows 81 (101-20)
        self.speed_spinbox.setValue(101 - 20)
        self.speed_spinbox.valueChanged.connect(self.on_speed_spinbox_changed)
        self.speed_spinbox.setFixedWidth(80)
        
        speed_control_layout.addWidget(self.speed_slider)
        speed_control_layout.addWidget(self.speed_spinbox)
        
        # Unit explanation
        speed_unit_label = QLabel("Unit: milliseconds between frames")
        speed_unit_label.setStyleSheet("color: gray; font-size: 10px;")
        
        speed_layout.addLayout(speed_control_layout)
        speed_layout.addWidget(speed_unit_label)
        section.add_layout(speed_layout)
        
        return section
        
    def on_speed_changed(self, value):
        """Handle speed slider change (invert so higher value = faster)"""
        # Invert the speed: higher slider value = lower delay = faster playback
        self.play_speed = 101 - value  # Maps 1-100 to 100-1
        if self.playing:
            self.timer.start(self.play_speed)
        # Update spinbox to show the inverted value (avoid circular updates)
        if hasattr(self, 'speed_spinbox'):
            inverted_value = 101 - value
            if self.speed_spinbox.value() != inverted_value:
                self.speed_spinbox.blockSignals(True)
                self.speed_spinbox.setValue(inverted_value)
                self.speed_spinbox.blockSignals(False)
    
    def on_speed_spinbox_changed(self, value):
        """Handle speed spinbox change"""
        # Convert spinbox value back to slider value (invert it)
        slider_value = 1010 - value
        # Update slider to match spinbox (avoid circular updates)
        if hasattr(self, 'speed_slider') and self.speed_slider.value() != slider_value:
            self.speed_slider.blockSignals(True)
            self.speed_slider.setValue(slider_value)
            self.speed_slider.blockSignals(False)
        # Apply the speed change using the slider value
        self.play_speed = value  # Use the spinbox value directly as it's already the delay
        if self.playing:
            self.timer.start(self.play_speed)
        
    def create_visualization_controls(self):
        """Create visualization control section"""
        section = CollapsibleSection("VISUALIZATION", False)
        
        # Cell size control
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Cell Size:"))
        self.size_slider = QSlider(Qt.Orientation.Horizontal)
        self.size_slider.setMinimum(1)
        self.size_slider.setMaximum(4)  # Restrict to first 4 options
        self.size_slider.setValue(2)
        self.size_slider.valueChanged.connect(self.on_size_changed)
        size_layout.addWidget(self.size_slider)
        section.add_layout(size_layout)
        
        # Coordinate axes checkbox
        self.axes_checkbox = QCheckBox("Show Coordinate Axes (X=Green, Y=Red, Z=Blue)")
        self.axes_checkbox.stateChanged.connect(self.on_axes_changed)
        section.add_widget(self.axes_checkbox)
        
        # Polarity controls
        self.polarity_checkbox = QCheckBox("Show Polarity")
        self.polarity_checkbox.stateChanged.connect(self.on_polarity_changed)
        section.add_widget(self.polarity_checkbox)
        
        polarity_type_layout = QHBoxLayout()
        polarity_type_layout.addWidget(QLabel("Type:"))
        self.polarity_combo = QComboBox()
        self.polarity_combo.addItems(["Apicobasal (p)", "Planar (q)"])
        self.polarity_combo.currentTextChanged.connect(self.on_polarity_type_changed)
        # Style combo box to be clearly distinguishable
        self.polarity_combo.setStyleSheet("""
            QComboBox {
                background-color: white;
                border: 2px solid #bdc3c7;
                border-radius: 4px;
                padding: 4px 8px;
                color: #2c3e50;
                font-weight: bold;
                min-width: 120px;
            }
            QComboBox:hover {
                border-color: #3498db;
            }
            QComboBox::drop-down {
                border: none;
                background-color: #ecf0f1;
                width: 20px;
            }
            QComboBox::down-arrow {
                width: 10px;
                height: 10px;
            }
        """)
        polarity_type_layout.addWidget(self.polarity_combo)
        section.add_layout(polarity_type_layout)
        
        # Color by depth checkbox
        self.color_by_depth_checkbox = QCheckBox("Color by Depth (Distance to Camera)")
        self.color_by_depth_checkbox.stateChanged.connect(self.on_color_by_depth_changed)
        section.add_widget(self.color_by_depth_checkbox)

        # Color by phi checkbox
        self.color_by_phi_checkbox = QCheckBox("Visualize Scalar Values")
        self.color_by_phi_checkbox.stateChanged.connect(self.on_color_by_phi_changed)
        # Enable if either phi or theta data exists
        has_scalar_data = (self.phi_lst is not None) or (self.theta_lst is not None)
        self.color_by_phi_checkbox.setEnabled(has_scalar_data)
        section.add_widget(self.color_by_phi_checkbox)

        # Scalar mode selection (phi/theta toggle)
        scalar_mode_layout = QHBoxLayout()
        scalar_mode_layout.addWidget(QLabel("Scalar Type:"))
        self.scalar_mode_combo = QComboBox()

        # Add available options based on data
        if self.phi_lst is not None:
            self.scalar_mode_combo.addItem("Phi")
        if self.theta_lst is not None:
            self.scalar_mode_combo.addItem("Theta")

        # Only enable if we have scalar data
        self.scalar_mode_combo.setEnabled(has_scalar_data)
        self.scalar_mode_combo.currentTextChanged.connect(self.on_scalar_mode_changed)

        # Style combo box
        self.scalar_mode_combo.setStyleSheet("""
            QComboBox {
                background-color: white;
                border: 2px solid #bdc3c7;
                border-radius: 4px;
                padding: 4px 8px;
                color: #2c3e50;
                font-weight: bold;
                min-width: 120px;
            }
            QComboBox:hover {
                border-color: #3498db;
            }
            QComboBox::drop-down {
                border: none;
                background-color: #ecf0f1;
                width: 20px;
            }
            QComboBox::down-arrow {
                width: 10px;
                height: 10px;
            }
        """)
        scalar_mode_layout.addWidget(self.scalar_mode_combo)
        section.add_layout(scalar_mode_layout)

        # Color legend label
        legend_label = QLabel("Color range (Magma): Dark Purple (-1) → Red/Orange (0) → Light Yellow (1)")
        legend_label.setWordWrap(True)
        legend_label.setStyleSheet("""
            QLabel {
                color: #7f8c8d;
                font-size: 10px;
                font-style: italic;
                padding: 3px;
                background-color: rgba(236, 240, 241, 0.8);
                border-radius: 3px;
                margin-top: 2px;
            }
        """)
        section.add_widget(legend_label)

        # Background color
        bg_layout = QHBoxLayout()
        bg_layout.addWidget(QLabel("Background:"))
        self.bg_combo = QComboBox()
        self.bg_combo.addItems(["Black", "White", "Gray"])
        self.bg_combo.currentTextChanged.connect(self.on_background_changed)
        # Style combo box to be clearly distinguishable
        self.bg_combo.setStyleSheet("""
            QComboBox {
                background-color: white;
                border: 2px solid #bdc3c7;
                border-radius: 4px;
                padding: 4px 8px;
                color: #2c3e50;
                font-weight: bold;
                min-width: 120px;
            }
            QComboBox:hover {
                border-color: #3498db;
            }
            QComboBox::drop-down {
                border: none;
                background-color: #ecf0f1;
                width: 20px;
            }
            QComboBox::down-arrow {
                width: 10px;
                height: 10px;
            }
        """)
        bg_layout.addWidget(self.bg_combo)
        section.add_layout(bg_layout)
        
        return section
        
    def create_cell_types_section(self):
        """Create cell types control section"""
        section = CollapsibleSection("CELL TYPES", False)
        
        # Color by type checkbox
        self.color_by_type_checkbox = QCheckBox("Color by Cell Type")
        self.color_by_type_checkbox.setChecked(True)
        self.color_by_type_checkbox.stateChanged.connect(self.on_color_by_type_changed)
        section.add_widget(self.color_by_type_checkbox)
        
        # Cell type list
        type_label = QLabel("Cell Types:")
        section.add_widget(type_label)
        
        self.type_list_widget = QListWidget()
        self.type_list_widget.setMaximumHeight(160)  # Increased height for better spacing
        self.type_list_widget.setStyleSheet("""
            QListWidget {
                background-color: white;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                padding: 2px;
            }
            QListWidget::item {
                border: none;
                padding: 0px;
                margin: 1px;
                min-height: 35px;
                max-height: 35px;
            }
            QListWidget::item:selected {
                background-color: rgba(255, 165, 0, 0.3);
                border: 1px solid rgba(255, 165, 0, 0.6);
                border-radius: 3px;
            }
            QListWidget::item:hover {
                background-color: rgba(255, 165, 0, 0.15);
                border-radius: 3px;
            }
        """)
        self.type_list_widget.itemDoubleClicked.connect(self.on_cell_type_double_click)
        self.type_list_widget.setToolTip("Double-click on any cell type to change its color")
        self.populate_type_list()
        section.add_widget(self.type_list_widget)
        
        # Add instructional text
        instruction_label = QLabel("💡 Double-click on cell type to change color")
        instruction_label.setStyleSheet("""
            QLabel {
                color: #7f8c8d;
                font-size: 11px;
                font-style: italic;
                padding: 2px 5px;
                background-color: rgba(236, 240, 241, 0.8);
                border-radius: 3px;
                margin-top: 3px;
            }
        """)
        section.add_widget(instruction_label)
        
        return section
        
    def create_data_bisection_section(self):
        """Create data sectioning control section"""
        section = CollapsibleSection("DATA SECTIONING", False)
        
        # Enable bisection
        self.bisection_checkbox = QCheckBox("Enable Bisection")
        self.bisection_checkbox.stateChanged.connect(self.on_bisection_changed)
        section.add_widget(self.bisection_checkbox)
        
        # Plane selection
        plane_layout = QHBoxLayout()
        plane_layout.addWidget(QLabel("Plane:"))
        self.plane_combo = QComboBox()
        self.plane_combo.addItems(["XY", "XZ", "YZ", "Camera Orthogonal"])
        self.plane_combo.currentTextChanged.connect(self.on_plane_changed)
        # Style combo box to be clearly distinguishable
        self.plane_combo.setStyleSheet("""
            QComboBox {
                background-color: white;
                border: 2px solid #bdc3c7;
                border-radius: 4px;
                padding: 4px 8px;
                color: #2c3e50;
                font-weight: bold;
                min-width: 80px;
            }
            QComboBox:hover {
                border-color: #3498db;
            }
            QComboBox::drop-down {
                border: none;
                background-color: #ecf0f1;
                width: 20px;
            }
            QComboBox::down-arrow {
                width: 10px;
                height: 10px;
            }
        """)
        plane_layout.addWidget(self.plane_combo)
        section.add_layout(plane_layout)
        
        # Position slider
        position_layout = QVBoxLayout()
        position_control_layout = QHBoxLayout()
        
        # Slider and value input
        position_control_layout.addWidget(QLabel("Position:"))
        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setMinimum(-100)
        self.position_slider.setMaximum(100)
        self.position_slider.setValue(0)
        self.position_slider.valueChanged.connect(self.on_position_changed)
        
        self.position_spinbox = QSpinBox()
        self.position_spinbox.setMinimum(-100)
        self.position_spinbox.setMaximum(100)
        self.position_spinbox.setValue(0)
        self.position_spinbox.valueChanged.connect(self.on_position_spinbox_changed)
        self.position_spinbox.setFixedWidth(80)
        
        position_control_layout.addWidget(self.position_slider)
        position_control_layout.addWidget(self.position_spinbox)
        
        # Unit explanation
        position_unit_label = QLabel("Unit: Extent of dataset in selected axis")
        position_unit_label.setStyleSheet("color: gray; font-size: 10px;")
        
        position_layout.addLayout(position_control_layout)
        position_layout.addWidget(position_unit_label)
        section.add_layout(position_layout)
        
        # Cross-section mode
        self.cross_section_checkbox = QCheckBox("Cross-section Mode")
        self.cross_section_checkbox.stateChanged.connect(self.on_cross_section_changed)
        section.add_widget(self.cross_section_checkbox)
        
        # Width slider
        width_layout = QVBoxLayout()
        width_control_layout = QHBoxLayout()
        
        # Slider and value input
        width_control_layout.addWidget(QLabel("Section Width:"))
        self.width_slider = QSlider(Qt.Orientation.Horizontal)
        self.width_slider.setMinimum(1)
        self.width_slider.setMaximum(20)
        self.width_slider.setValue(2)
        self.width_slider.valueChanged.connect(self.on_width_changed)
        
        self.width_spinbox = QSpinBox()
        self.width_spinbox.setMinimum(1)
        self.width_spinbox.setMaximum(20)
        self.width_spinbox.setValue(2)
        self.width_spinbox.valueChanged.connect(self.on_width_spinbox_changed)
        self.width_spinbox.setFixedWidth(80)
        
        width_control_layout.addWidget(self.width_slider)
        width_control_layout.addWidget(self.width_spinbox)
        
        # Unit explanation
        width_unit_label = QLabel("Unit: Simulation units (cell radii)")
        width_unit_label.setStyleSheet("color: gray; font-size: 10px;")
        
        width_layout.addLayout(width_control_layout)
        width_layout.addWidget(width_unit_label)
        section.add_layout(width_layout)
        
        return section
    
    def create_neighbor_search_section(self):
        """Create neighbor search control section"""
        section = CollapsibleSection("NEIGHBOR SEARCH", False)
        
        # Connect to section toggle to clear selection when collapsed
        section.header_button.toggled.connect(self.on_neighbor_search_section_toggled)
        
        # Instructions label
        instruction_label = QLabel("💡 Search for neighbors, then click a cell to see its connections")
        instruction_label.setWordWrap(True)
        instruction_label.setStyleSheet("""
            QLabel {
                color: #7f8c8d;
                font-size: 11px;
                font-style: italic;
                padding: 5px;
                background-color: rgba(236, 240, 241, 0.8);
                border-radius: 3px;
                margin: 2px 0px;
            }
        """)
        section.add_widget(instruction_label)
        
        # Search parameters layout
        params_layout = QVBoxLayout()
        
        # See-through threshold
        seethru_layout = QHBoxLayout()
        seethru_label = QLabel("See-through threshold:")
        seethru_label.setStyleSheet("color: #2c3e50; font-size: 11px;")
        self.seethru_spinbox = QSpinBox()
        self.seethru_spinbox.setMinimum(0)
        self.seethru_spinbox.setMaximum(10)
        self.seethru_spinbox.setValue(0)
        self.seethru_spinbox.setFixedWidth(80)
        self.seethru_spinbox.setToolTip("Number of cells that can be 'between' two cells for them to still be neighbors")
        seethru_layout.addWidget(seethru_label)
        seethru_layout.addWidget(self.seethru_spinbox)
        seethru_layout.addStretch()
        params_layout.addLayout(seethru_layout)
        
        # Max neighbors (k)
        k_layout = QHBoxLayout()
        k_label = QLabel("Max neighbors (k):")
        k_label.setStyleSheet("color: #2c3e50; font-size: 11px;")
        self.max_neighbors_spinbox = QSpinBox()
        self.max_neighbors_spinbox.setMinimum(10)
        self.max_neighbors_spinbox.setMaximum(200)
        self.max_neighbors_spinbox.setValue(50)
        self.max_neighbors_spinbox.setFixedWidth(80)
        self.max_neighbors_spinbox.setToolTip("Maximum number of potential neighbors to consider")
        k_layout.addWidget(k_label)
        k_layout.addWidget(self.max_neighbors_spinbox)
        k_layout.addStretch()
        params_layout.addLayout(k_layout)
        
        section.add_layout(params_layout)
        
        # Search button
        self.neighbor_search_button = QPushButton("🔍 SEARCH FOR NEIGHBORS")
        self.neighbor_search_button.clicked.connect(self.on_search_neighbors_clicked)
        self.neighbor_search_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #e67e22, stop:1 #d35400);
                color: white;
                border: 2px solid #d35400;
                border-radius: 6px;
                padding: 10px;
                font-weight: bold;
                font-size: 12px;
                margin: 5px 0px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f39c12, stop:1 #e67e22);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #d35400, stop:1 #c0392b);
            }
            QPushButton:disabled {
                background: #95a5a6;
                border-color: #7f8c8d;
            }
        """)
        section.add_widget(self.neighbor_search_button)
        
        # Show only neighbors checkbox (hidden initially)
        self.neighbor_show_only_checkbox = QCheckBox("Show only selected cell and neighbors")
        self.neighbor_show_only_checkbox.setStyleSheet("color: #2c3e50; font-size: 11px;")
        self.neighbor_show_only_checkbox.stateChanged.connect(self.on_show_only_neighbors_changed)
        self.neighbor_show_only_checkbox.setVisible(False)
        section.add_widget(self.neighbor_show_only_checkbox)
        
        # Statistics display (hidden initially)
        self.neighbor_stats_browser = QTextBrowser()
        self.neighbor_stats_browser.setMaximumHeight(150)
        self.neighbor_stats_browser.setStyleSheet("""
            QTextBrowser {
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                background-color: white;
                padding: 8px;
                color: #2c3e50;
                font-size: 11px;
            }
        """)
        self.neighbor_stats_browser.setVisible(False)
        section.add_widget(self.neighbor_stats_browser)
        
        # Clear selection button (hidden initially)
        self.neighbor_clear_button = QPushButton("Clear Selection")
        self.neighbor_clear_button.clicked.connect(self.clear_neighbor_selection)
        self.neighbor_clear_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #95a5a6, stop:1 #7f8c8d);
                color: white;
                border: 2px solid #7f8c8d;
                border-radius: 4px;
                padding: 6px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #b4bcc2, stop:1 #95a5a6);
            }
        """)
        self.neighbor_clear_button.setVisible(False)
        section.add_widget(self.neighbor_clear_button)
        
        return section
        
    def create_export_section(self):
        """Create export control section"""
        section = CollapsibleSection("EXPORT", False)
        
        # Export image button
        image_button = QPushButton("📷 Export Image")
        image_button.clicked.connect(self.export_image)
        # Style button with fall-inspired muted green colors
        image_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #6b8e5a, stop: 1 #4d6842);
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 12px;
                color: white;
                min-width: 120px;
            }
            QPushButton:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #7da46a, stop: 1 #6b8e5a);
            }
            QPushButton:pressed {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #4d6842, stop: 1 #3a4f32);
            }
        """)
        section.add_widget(image_button)
        
        # Export video button
        video_button = QPushButton("🎬 Export Video")
        video_button.clicked.connect(self.export_video)
        # Style button with fall-inspired warm orange colors (same as Load Dataset)
        video_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #d17a47, stop: 1 #b85f2f);
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 12px;
                color: white;
                min-width: 120px;
            }
            QPushButton:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #e08a56, stop: 1 #d17a47);
            }
            QPushButton:pressed {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #b85f2f, stop: 1 #9f4d1e);
            }
        """)
        section.add_widget(video_button)
        
        # Add separator line
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setStyleSheet("color: #555555;")
        section.add_widget(separator)
        
        # Import metadata button
        import_button = QPushButton("📁 Import Metadata")
        import_button.clicked.connect(self.import_metadata_file)
        # Style button with blue colors to distinguish from export
        import_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #4a7ba7, stop: 1 #326a94);
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 12px;
                color: white;
                min-width: 120px;
            }
            QPushButton:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #5a8bb7, stop: 1 #4a7ba7);
            }
            QPushButton:pressed {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #326a94, stop: 1 #225681);
            }
        """)
        section.add_widget(import_button)
        
        return section
        
    def create_visualization_panel(self):
        """Create the main visualization panel"""
        vis_widget = VisPy3DVisualizationWidget(self)
        # Set black background for the VisPy container
        vis_widget.setStyleSheet("background-color: #1e1e1e;")
        return vis_widget
        
    def populate_type_list(self):
        """Populate the cell type list widget"""
        self.type_list_widget.clear()
        
        for cell_type in sorted(self.unique_types):
            item = QListWidgetItem()
            
            # Create widget for type item
            item_widget = QWidget()
            item_layout = QHBoxLayout(item_widget)
            item_layout.setContentsMargins(8, 6, 8, 6)  # Increased margins for better spacing
            item_layout.setSpacing(10)  # Add spacing between elements
            
            # Visibility checkbox
            checkbox = QCheckBox()
            checkbox.setChecked(cell_type in self.visible_types)
            checkbox.stateChanged.connect(lambda state, t=cell_type: self.on_type_visibility_changed(t, state))
            item_layout.addWidget(checkbox)
            
            # Color indicator - larger circle
            color_label = QLabel("●")
            color = self.cell_type_colors[cell_type]
            color_label.setStyleSheet(f"""
                color: rgb({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)}); 
                font-size: 28px; 
                font-weight: bold;
                margin: 2px;
            """)
            color_label.setFixedWidth(35)  # Reduced width to prevent overlap
            item_layout.addWidget(color_label)
            
            # Type label with better spacing
            type_label = QLabel(f"Type {cell_type}")
            type_label.setStyleSheet("""
                font-size: 12px;
                font-weight: bold;
                color: #2c3e50;
                margin-left: 5px;
            """)
            item_layout.addWidget(type_label)
            
            item_layout.addStretch()
            
            # Set exact height to match list item styling
            item_widget.setFixedHeight(35)
            
            # Add to list
            item = QListWidgetItem()
            item.setSizeHint(QSize(-1, 35))  # Ensure list item matches widget height
            item.setData(Qt.ItemDataRole.UserRole, cell_type)  # Store cell type for double-click handling
            self.type_list_widget.addItem(item)
            self.type_list_widget.setItemWidget(item, item_widget)
    
    # Neighbor search methods
    @staticmethod
    def find_potential_neighbours(x, k=50, distance_upper_bound=np.inf, workers=-1):
        """Find potential neighbors using KDTree spatial indexing"""
        tree = cKDTree(x)
        d, idx = tree.query(x, k + 1, distance_upper_bound=distance_upper_bound, workers=workers)
        return d[:, 1:], idx[:, 1:]  # Exclude self (first result)
    
    def find_true_neighbours(self, d, dx, seethru, progress_callback=None):
        """Find true neighbors using geometric occlusion test with adaptive batching"""
        # Determine device and dtype
        if TORCH_AVAILABLE and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        
        dtype = torch.float32
        
        # Convert inputs to tensors
        d_tensor = torch.tensor(d, device=device, dtype=dtype)
        dx_tensor = torch.tensor(dx, device=device, dtype=dtype)
        
        with torch.no_grad():
            total_cells = dx_tensor.shape[0]
            neighbor_count = dx_tensor.shape[1]
            result_tensor = torch.empty(
                (total_cells, neighbor_count),
                dtype=torch.bool,
                device=device
            )
            
            # Adaptive batch sizing
            batch_size = 2048
            min_batch_size = 64
            
            i0 = 0
            while i0 < total_cells:
                i1 = min(i0 + batch_size, total_cells)
                try:
                    # Compute pairwise occlusion distances
                    n_dis = torch.sum((dx_tensor[i0:i1, :, None, :] / 2 - dx_tensor[i0:i1, None, :, :]) ** 2, dim=3)
                    n_dis += 1000 * torch.eye(n_dis.shape[1], device=device, dtype=dtype)[None, :, :]
                    
                    # Check if neighbors are truly visible (not occluded)
                    result_tensor[i0:i1] = (torch.sum(n_dis < (d_tensor[i0:i1, :, None] ** 2 / 4), dim=2) <= seethru)
                    i0 = i1
                    
                    # Progress callback
                    if progress_callback:
                        progress_callback(i1, total_cells)
                
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        batch_size = max(min_batch_size, int(batch_size * 0.8))  # Reduce by 20%
                        print(f"OOM at cell {i0}. Reducing batch size to {batch_size}.")
                        continue
                    else:
                        raise e
        
        return result_tensor.cpu().numpy()
    
    def compute_neighbors_for_timestep(self, timestep, k, seethru, progress_callback=None):
        """Compute neighbors for a specific timestep with caching"""
        # Check cache first
        if self.data_folder:
            cache_dir = os.path.join(self.data_folder, "neighbor_cache")
            os.makedirs(cache_dir, exist_ok=True)
            cache_filename = os.path.join(cache_dir, f"neighbors_t{timestep}_k{k}_s{seethru}.pkl")
            
            if os.path.exists(cache_filename):
                try:
                    with open(cache_filename, 'rb') as f:
                        cached_data = pickle.load(f)
                    print(f"Loaded neighbor data from cache: {cache_filename}")
                    return cached_data
                except Exception as e:
                    print(f"Error loading cache: {e}, recomputing...")
        
        # Get positions for this timestep
        x = self.x_lst[timestep]
        
        # Find potential neighbors
        if progress_callback:
            progress_callback(0, 100, "Finding potential neighbors...")
        d, idx = self.find_potential_neighbours(x, k=k)
        
        # Get displacement vectors
        dx = x[idx] - x[:, None, :]
        
        # Find true neighbors
        if progress_callback:
            def true_neighbor_progress(current, total):
                progress_callback(current, total, f"Computing true neighbors: {current}/{total} cells")
            neighbor_bool_mask = self.find_true_neighbours(d, dx, seethru, true_neighbor_progress)
        else:
            neighbor_bool_mask = self.find_true_neighbours(d, dx, seethru)
        
        # Build neighbor list
        true_neighbors = [idx[i][neighbor_bool_mask[i]] for i in range(len(x))]
        
        result = {
            'true_neighbors': true_neighbors,
            'potential_neighbors': (d, idx),
            'k': k,
            'seethru': seethru
        }
        
        # Save to cache
        if self.data_folder:
            try:
                with open(cache_filename, 'wb') as f:
                    pickle.dump(result, f)
            except Exception as e:
                print(f"Error saving cache: {e}")
        
        return result
            
    # Event handlers
    def on_timestep_changed(self, value):
        """Handle timestep slider change"""
        old_timestep = self.current_timestep
        self.current_timestep = value
        # Update spinbox to match slider (avoid circular updates)
        if hasattr(self, 'timestep_spinbox') and self.timestep_spinbox.value() != value:
            self.timestep_spinbox.blockSignals(True)
            self.timestep_spinbox.setValue(value)
            self.timestep_spinbox.blockSignals(False)
        
        # Clear neighbor selection when changing timesteps
        if old_timestep != value:
            self.clear_neighbor_selection()
            # Remove old timestep data to free memory
            if old_timestep in self.neighbor_data:
                del self.neighbor_data[old_timestep]
        
        self.update_visualization()
        self.update_data_info()
    
    def on_timestep_spinbox_changed(self, value):
        """Handle timestep spinbox change"""
        old_timestep = self.current_timestep
        self.current_timestep = value
        # Update slider to match spinbox (avoid circular updates)
        if hasattr(self, 'timestep_slider') and self.timestep_slider.value() != value:
            self.timestep_slider.blockSignals(True)
            self.timestep_slider.setValue(value)
            self.timestep_slider.blockSignals(False)
        
        # Clear neighbor selection when changing timesteps
        if old_timestep != value:
            self.clear_neighbor_selection()
            # Remove old timestep data to free memory
            if old_timestep in self.neighbor_data:
                del self.neighbor_data[old_timestep]
        
        self.update_visualization()
        self.update_data_info()
        
    def toggle_play(self):
        """Toggle play/pause"""
        self.playing = not self.playing
        if self.playing:
            self.play_button.setText("⏸️ Pause")
            self.timer.start(self.play_speed)
        else:
            self.play_button.setText("▶️ Play")
            self.timer.stop()
            
    def next_frame(self):
        """Go to next frame"""
        if self.current_timestep < self.max_timesteps - 1:
            self.current_timestep += 1
            self.timestep_slider.setValue(self.current_timestep)
        else:
            # Loop back to beginning
            self.current_timestep = 0
            self.timestep_slider.setValue(self.current_timestep)
            
    def prev_frame(self):
        """Go to previous frame"""
        if self.current_timestep > 0:
            self.current_timestep -= 1
            self.timestep_slider.setValue(self.current_timestep)
            
    def on_size_changed(self, value):
        """Handle cell size change"""
        self.cell_size = value
        self.update_visualization()
        
    def on_polarity_changed(self, state):
        """Handle polarity display change"""
        self.show_polarity = state == Qt.CheckState.Checked.value
        if self.show_polarity and self.color_by_type:
            self.color_by_type_checkbox.setChecked(False)
        if self.show_polarity and self.color_by_depth:
            self.color_by_depth_checkbox.setChecked(False)
        if self.show_polarity and self.color_by_phi:
            self.color_by_phi_checkbox.setChecked(False)
        self.update_visualization()
        
    def on_polarity_type_changed(self, text):
        """Handle polarity type change"""
        self.polarity_type = "p" if "Apicobasal" in text else "q"
        self.update_visualization()
        
    def on_background_changed(self, text):
        """Handle background color change"""
        self.background_color = text.lower()
        self.update_visualization()
        
    def reset_camera_view(self):
        """Reset camera to show all data"""
        if hasattr(self.vis_panel, 'camera_bounds_set'):
            self.vis_panel.camera_bounds_set = False
        self.update_visualization()
    
    def on_axes_changed(self, state):
        """Handle coordinate axes checkbox"""
        self.show_axes = state == Qt.CheckState.Checked.value
        # Update axis visibility
        if hasattr(self, 'vis_panel') and hasattr(self.vis_panel, 'axis_lines'):
            for axis_line in self.vis_panel.axis_lines:
                axis_line.visible = self.show_axes
            self.vis_panel.canvas.update()
        
    def on_color_by_type_changed(self, state):
        """Handle color by type change"""
        self.color_by_type = state == Qt.CheckState.Checked.value
        if self.color_by_type and self.show_polarity:
            self.polarity_checkbox.setChecked(False)
        if self.color_by_type and self.color_by_depth:
            self.color_by_depth_checkbox.setChecked(False)
        if self.color_by_type and self.color_by_phi:
            self.color_by_phi_checkbox.setChecked(False)
        self.update_visualization()
        
    def on_color_by_depth_changed(self, state):
        """Handle color by depth change"""
        self.color_by_depth = state == Qt.CheckState.Checked.value
        if self.color_by_depth and self.color_by_type:
            self.color_by_type_checkbox.setChecked(False)
        if self.color_by_depth and self.show_polarity:
            self.polarity_checkbox.setChecked(False)
        if self.color_by_depth and self.color_by_phi:
            self.color_by_phi_checkbox.setChecked(False)
        self.update_visualization()

    def on_color_by_phi_changed(self, state):
        """Handle color by scalar (phi/theta) change"""
        self.color_by_phi = state == Qt.CheckState.Checked.value
        if self.color_by_phi and self.color_by_type:
            self.color_by_type_checkbox.setChecked(False)
        if self.color_by_phi and self.color_by_depth:
            self.color_by_depth_checkbox.setChecked(False)
        if self.color_by_phi and self.show_polarity:
            self.polarity_checkbox.setChecked(False)
        self.update_visualization()

    def on_scalar_mode_changed(self, text):
        """Handle scalar visualization mode change (phi/theta)"""
        self.scalar_visualization_mode = text.lower()  # "phi" or "theta"
        # Only update if scalar visualization is currently enabled
        if self.color_by_phi:
            self.update_visualization()

    def on_type_visibility_changed(self, cell_type, state):
        """Handle cell type visibility change"""
        if state == Qt.CheckState.Checked.value:
            self.visible_types.add(cell_type)
        else:
            self.visible_types.discard(cell_type)
        self.update_visualization()
        
    def on_cell_type_double_click(self, item):
        """Handle double-click on cell type item to open color picker"""
        # Get cell type from item data
        cell_type = item.data(Qt.ItemDataRole.UserRole)
        
        if cell_type is not None:
            current_color = self.cell_type_colors.get(cell_type, (0.5, 0.5, 0.5))
            
            # Open color picker dialog
            dialog = ColorPickerDialog(current_color, self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                new_color = dialog.get_selected_color()
                # Update the cell type color
                self.cell_type_colors[cell_type] = new_color
                # Refresh the type list display
                self.populate_type_list()
                # Update visualization if color by type is enabled
                if self.color_by_type:
                    self.update_visualization()
        
    def on_bisection_changed(self, state):
        """Handle bisection enable/disable"""
        self.bisection_enabled = state == Qt.CheckState.Checked.value
        
        # If bisection is enabled, disable cross-section mode
        if self.bisection_enabled and self.cross_section_mode:
            self.cross_section_mode = False
            self.cross_section_checkbox.setChecked(False)
            
        self.update_visualization()
        
    def on_plane_changed(self, text):
        """Handle bisection plane change"""
        self.bisection_plane = text
        self.update_visualization()
        
    def on_position_changed(self, value):
        """Handle bisection position change"""
        self.bisection_position = value / 100.0  # Normalize to -1 to 1
        # Update spinbox to match slider (avoid circular updates)
        if hasattr(self, 'position_spinbox') and self.position_spinbox.value() != value:
            self.position_spinbox.blockSignals(True)
            self.position_spinbox.setValue(value)
            self.position_spinbox.blockSignals(False)
        self.update_visualization()
    
    def on_position_spinbox_changed(self, value):
        """Handle position spinbox change"""
        # Update slider to match spinbox (avoid circular updates)
        if hasattr(self, 'position_slider') and self.position_slider.value() != value:
            self.position_slider.blockSignals(True)
            self.position_slider.setValue(value)
            self.position_slider.blockSignals(False)
        # Apply the position change
        self.on_position_changed(value)
        
    def on_cross_section_changed(self, state):
        """Handle cross-section mode change"""
        self.cross_section_mode = state == Qt.CheckState.Checked.value
        
        # If cross-section is enabled, automatically enable bisection too
        # This is needed for Camera Orthogonal mode to work properly
        if self.cross_section_mode and not self.bisection_enabled:
            self.bisection_enabled = True
            self.bisection_checkbox.setChecked(True)
            
        self.update_visualization()
        
    def on_width_changed(self, value):
        """Handle cross-section width change"""
        self.cross_section_width = value
        # Update spinbox to match slider (avoid circular updates)
        if hasattr(self, 'width_spinbox') and self.width_spinbox.value() != value:
            self.width_spinbox.blockSignals(True)
            self.width_spinbox.setValue(value)
            self.width_spinbox.blockSignals(False)
        self.update_visualization()
    
    def on_width_spinbox_changed(self, value):
        """Handle width spinbox change"""
        # Update slider to match spinbox (avoid circular updates)
        if hasattr(self, 'width_slider') and self.width_slider.value() != value:
            self.width_slider.blockSignals(True)
            self.width_slider.setValue(value)
            self.width_slider.blockSignals(False)
        # Apply the width change
        self.on_width_changed(value)
    
    def on_search_neighbors_clicked(self):
        """Handle search for neighbors button click"""
        # Check if playback is paused
        if self.playing:
            QMessageBox.warning(self, "Playback Active", 
                              "Please pause playback before searching for neighbors.")
            return
        
        # Check if data is loaded
        if not self.x_lst or len(self.x_lst) == 0:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return
        
        # Get current parameters
        timestep = self.current_timestep
        k = self.max_neighbors_spinbox.value()
        seethru = self.seethru_spinbox.value()
        
        # Create progress dialog
        progress = QProgressDialog("Computing neighbors...", "Cancel", 0, 100, self)
        progress.setWindowTitle("Neighbor Search")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        
        # Grey out canvas
        original_button_text = self.neighbor_search_button.text()
        self.neighbor_search_button.setText("⏳ Searching...")
        self.neighbor_search_button.setEnabled(False)
        
        # Track computation time
        start_time = time.time()
        
        # Progress callback
        canceled = [False]
        def progress_callback(current, total, message="Computing..."):
            if progress.wasCanceled():
                canceled[0] = True
                return
            percentage = int((current / total) * 100)
            progress.setLabelText(message)
            progress.setValue(percentage)
            QApplication.processEvents()
        
        try:
            # Compute neighbors
            result = self.compute_neighbors_for_timestep(timestep, k, seethru, progress_callback)
            
            if canceled[0]:
                QMessageBox.information(self, "Canceled", "Neighbor search was canceled.")
            else:
                # Store result
                self.neighbor_data[timestep] = result
                self.neighbor_search_enabled = True
                
                # Update UI
                elapsed = time.time() - start_time
                QMessageBox.information(self, "Success", 
                                      f"Neighbor search complete!\n"
                                      f"Time: {elapsed:.2f} seconds\n"
                                      f"Click on a cell to see its neighbors.")
                
                # Show clear button
                self.neighbor_clear_button.setVisible(True)
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error computing neighbors:\n{str(e)}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Restore UI
            progress.close()
            self.neighbor_search_button.setText(original_button_text)
            self.neighbor_search_button.setEnabled(True)
    
    def on_show_only_neighbors_changed(self, state):
        """Handle show only neighbors checkbox change"""
        # Re-render current selection if one exists
        if self.selected_cell_idx is not None and self.current_timestep in self.neighbor_data:
            self.on_cell_selected(self.selected_cell_idx)
    
    def on_cell_selected(self, cell_idx):
        """Handle cell selection from 3D canvas click"""
        if cell_idx is None or self.current_timestep not in self.neighbor_data:
            return
        
        # Clear previous selection
        if hasattr(self, 'vis_panel'):
            self.vis_panel.clear_highlights()
        
        # Get neighbor data
        neighbor_data = self.neighbor_data[self.current_timestep]
        true_neighbors = neighbor_data['true_neighbors']
        
        if cell_idx >= len(true_neighbors):
            return
        
        neighbor_indices = true_neighbors[cell_idx]
        self.selected_cell_idx = cell_idx
        
        # Get show only mode
        show_only = self.neighbor_show_only_checkbox.isChecked()
        
        # Highlight cells in visualization
        self.vis_panel.highlight_cells(cell_idx, neighbor_indices, show_only)
        
        # Calculate statistics
        p_mask = self.p_mask_lst[self.current_timestep]
        selected_type = int(p_mask[cell_idx])
        
        # Count neighbors by type
        type_counts = {}
        for neighbor_idx in neighbor_indices:
            neighbor_type = int(p_mask[neighbor_idx])
            type_counts[neighbor_type] = type_counts.get(neighbor_type, 0) + 1
        
        # Format statistics
        stats_html = f"""
        <b>Selected Cell:</b> Cell {cell_idx} (Type {selected_type})<br>
        <b>Total neighbors:</b> {len(neighbor_indices)}<br>
        <br>
        <b>Neighbors by type:</b><br>
        """
        
        for cell_type in sorted(type_counts.keys()):
            stats_html += f"&nbsp;&nbsp;Type {cell_type}: {type_counts[cell_type]}<br>"
        
        # Update statistics display
        self.neighbor_stats_browser.setHtml(stats_html)
        self.neighbor_stats_browser.setVisible(True)
        self.neighbor_show_only_checkbox.setVisible(True)
        self.neighbor_clear_button.setVisible(True)
    
    def clear_neighbor_selection(self):
        """Clear the current neighbor selection"""
        if hasattr(self, 'vis_panel'):
            self.vis_panel.clear_highlights()
        
        self.selected_cell_idx = None
        self.neighbor_stats_browser.setVisible(False)
        self.neighbor_show_only_checkbox.setVisible(False)
        self.neighbor_show_only_checkbox.setChecked(False)
        
        # Don't hide clear button if neighbor data exists for this timestep
        if self.current_timestep not in self.neighbor_data:
            self.neighbor_clear_button.setVisible(False)
    
    def on_neighbor_search_section_toggled(self, expanded):
        """Handle when neighbor search section is expanded/collapsed"""
        if not expanded:
            # Section collapsed - disable neighbor search and clear selections
            self.neighbor_search_enabled = False
            self.clear_neighbor_selection()
    
    def on_camera_input_focus_in(self, event, spinbox):
        """Handle when user starts editing a camera input field"""
        # Only apply editing mode to position inputs (not read-only view direction)
        if spinbox in [self.pos_x_input, self.pos_y_input, self.pos_z_input]:
            self.user_editing_camera = True
            # Change background to indicate editing mode
            spinbox.setStyleSheet("""
                QDoubleSpinBox {
                    background-color: #fff3cd;
                    border: 2px solid #ffc107;
                    border-radius: 3px;
                    padding: 2px;
                    font-family: monospace;
                    font-size: 11px;
                }
            """)
        # Call the original focusInEvent
        QtWidgets.QDoubleSpinBox.focusInEvent(spinbox, event)
        
    def on_camera_input_focus_out(self, event, spinbox):
        """Handle when user finishes editing a camera input field"""
        # Only apply to position inputs
        if spinbox in [self.pos_x_input, self.pos_y_input, self.pos_z_input]:
            self.user_editing_camera = False
            # Restore normal background
            spinbox.setStyleSheet("""
                QDoubleSpinBox {
                    background-color: white;
                    border: 1px solid #bdc3c7;
                    border-radius: 3px;
                    padding: 2px;
                    font-family: monospace;
                    font-size: 11px;
                }
            """)
        # Call the original focusOutEvent
        QtWidgets.QDoubleSpinBox.focusOutEvent(spinbox, event)
    
    def on_camera_input_value_changed(self, value, spinbox):
        """Handle when user changes a position input value"""
        if spinbox in [self.pos_x_input, self.pos_y_input, self.pos_z_input]:
            # Only mark as user-modified if this wasn't triggered by live updates
            if not getattr(spinbox, '_updating_from_live', False):
                self.user_modified_position = True
                # Store the current position values that user entered
                self.last_user_position = np.array([
                    self.pos_x_input.value(),
                    self.pos_y_input.value(), 
                    self.pos_z_input.value()
                ])
        
    def is_camera_moving(self):
        """Check if the camera is currently moving by comparing positions"""
        try:
            if not hasattr(self.vis_panel, 'get_camera_position'):
                return False
                
            current_position = self.vis_panel.get_camera_position()
            current_view_direction = self.vis_panel.get_camera_view_direction()
            
            # First time or no previous data
            if (self.last_camera_position is None or 
                self.last_camera_view_direction is None):
                self.last_camera_position = current_position.copy()
                self.last_camera_view_direction = current_view_direction.copy()
                return False
            
            # Check if camera has moved significantly (threshold for noise tolerance)
            position_threshold = 0.01
            
            position_moved = np.linalg.norm(current_position - self.last_camera_position) > position_threshold
            view_moved = np.linalg.norm(current_view_direction - self.last_camera_view_direction) > position_threshold
            
            camera_moved = position_moved or view_moved
            
            if camera_moved:
                self.camera_stillness_counter = 0
                self.camera_moving = True
            else:
                self.camera_stillness_counter += 1
                # Consider camera stopped after 5 consecutive checks without movement (0.5 seconds)
                if self.camera_stillness_counter >= 5:
                    self.camera_moving = False
            
            # Update last known positions
            self.last_camera_position = current_position.copy()
            self.last_camera_view_direction = current_view_direction.copy()
            
            return self.camera_moving
            
        except Exception as e:
            return False
            
    def jump_to_specified_camera_position(self):
        """Jump camera to the position specified in the input fields"""
        try:
            # Temporarily disable live updates during jump
            self.user_editing_camera = True
            
            # Get values from input fields
            pos_x = self.pos_x_input.value()
            pos_y = self.pos_y_input.value()
            pos_z = self.pos_z_input.value()
            position = np.array([pos_x, pos_y, pos_z])
            
            # Get current camera position to compare
            if hasattr(self.vis_panel, 'get_camera_position'):
                current_pos = self.vis_panel.get_camera_position()
                distance_to_target = np.linalg.norm(position - current_pos)
                
                if distance_to_target < 0.01:
                    self.user_editing_camera = False
                    return
            
            # Jump to position in the visualization
            if hasattr(self.vis_panel, 'jump_to_camera_position'):
                self.vis_panel.jump_to_camera_position(position)
                
                # Clear user modification flags after jump
                self.user_modified_position = False
                self.last_user_position = None
                
                # Wait a moment for the camera to settle
                import time
                time.sleep(0.2)
                
                # Force camera bounds reset after jump to ensure proper positioning
                if hasattr(self.vis_panel, 'camera_bounds_set'):
                    self.vis_panel.camera_bounds_set = False
                    
                # Re-enable live updates after a delay
                QTimer.singleShot(500, self.re_enable_camera_updates)
                    
        except Exception as e:
            print(f"Error in jump_to_specified_camera_position: {e}")
            QMessageBox.warning(self, "Camera Jump", 
                              f"Position jump failed.\n"
                              f"Technical details: {str(e)}")
            # Re-enable updates even if there was an error
            self.user_editing_camera = False
            
    def re_enable_camera_updates(self):
        """Re-enable live camera updates after jump"""
        self.user_editing_camera = False
            
    def get_current_camera_position(self):
        """Get current camera position and update the input fields"""
        try:
            if hasattr(self.vis_panel, 'get_camera_position'):
                # Get position
                position = self.vis_panel.get_camera_position()
                self.pos_x_input.setValue(float(position[0]))
                self.pos_y_input.setValue(float(position[1]))
                self.pos_z_input.setValue(float(position[2]))
                
                # Get view direction
                view_direction = self.vis_panel.get_camera_view_direction()
                self.view_x_input.setValue(float(view_direction[0]))
                self.view_y_input.setValue(float(view_direction[1]))
                self.view_z_input.setValue(float(view_direction[2]))
                
                print(f"Captured camera position: {position}")
                print(f"Captured view direction: {view_direction}")
                
        except Exception as e:
            print(f"Error getting current camera position: {e}")
            
    def update_camera_controls(self):
        """Update camera control fields with live camera data (only when not editing and camera stopped)"""
        try:
            # Update status indicator
            if hasattr(self, 'camera_status_label'):
                if self.user_editing_camera:
                    self.camera_status_label.setText("✏️ Editing Mode - Updates Paused")
                    self.camera_status_label.setStyleSheet("""
                        QLabel {
                            color: #f39c12;
                            font-size: 10px;
                            font-style: italic;
                            padding: 2px;
                            border-radius: 3px;
                            background-color: rgba(243, 156, 18, 0.1);
                        }
                    """)
                elif self.is_camera_moving():
                    self.camera_status_label.setText("🎥 Live Updates (Moving)")
                    self.camera_status_label.setStyleSheet("""
                        QLabel {
                            color: #3498db;
                            font-size: 10px;
                            font-style: italic;
                            padding: 2px;
                            border-radius: 3px;
                            background-color: rgba(52, 152, 219, 0.1);
                        }
                    """)
                else:
                    self.camera_status_label.setText("🔄 Live Updates Active")
                    self.camera_status_label.setStyleSheet("""
                        QLabel {
                            color: #27ae60;
                            font-size: 10px;
                            font-style: italic;
                            padding: 2px;
                            border-radius: 3px;
                            background-color: rgba(39, 174, 96, 0.1);
                        }
                    """)
            
            # Skip updates if user is actively editing the fields
            if self.user_editing_camera:
                return
                
            if hasattr(self.vis_panel, 'get_camera_position'):
                # Only update if the camera controls section is visible
                # This prevents unnecessary updates when section is collapsed
                if hasattr(self, 'pos_x_input') and self.pos_x_input.isVisible():
                    # Get current camera data
                    position = self.vis_panel.get_camera_position()
                    view_direction = self.vis_panel.get_camera_view_direction()
                    rotation_angle = self.vis_panel.get_camera_rotation_angle()
                    
                    # Check if user has modified position values
                    should_update_position = True
                    if (self.user_modified_position and 
                        self.last_user_position is not None):
                        
                        current_distance = np.linalg.norm(position - self.last_user_position)
                        # If current camera position is far from user's entered values,
                        # don't overwrite them (user probably wants to jump there)
                        if current_distance > 0.1:
                            should_update_position = False
                    
                    # Block signals to prevent triggering camera jumps during live updates
                    all_inputs = [
                        self.pos_x_input, self.pos_y_input, self.pos_z_input,
                        self.view_x_input, self.view_y_input, self.view_z_input,
                        self.rotation_input
                    ]
                    
                    for input_widget in all_inputs:
                        input_widget.blockSignals(True)
                        # Mark inputs as being updated by live system
                        input_widget._updating_from_live = True
                    
                    # Update position values only if not user-modified
                    if should_update_position:
                        self.pos_x_input.setValue(float(position[0]))
                        self.pos_y_input.setValue(float(position[1]))
                        self.pos_z_input.setValue(float(position[2]))
                        # Clear user modification flag since we're now in sync
                        self.user_modified_position = False
                        self.last_user_position = None
                    
                    # Always update view direction and rotation values (read-only)
                    self.view_x_input.setValue(float(view_direction[0]))
                    self.view_y_input.setValue(float(view_direction[1]))
                    self.view_z_input.setValue(float(view_direction[2]))
                    self.rotation_input.setValue(float(rotation_angle))
                    
                    # Re-enable signals
                    for input_widget in all_inputs:
                        input_widget.blockSignals(False)
                        # Clear the live update flag
                        input_widget._updating_from_live = False
                    
        except Exception as e:
            # Silently handle errors to avoid spam in the console
            pass
            
    def update_visualization(self):
        """Update the 3D visualization"""
        if hasattr(self, 'vis_panel') and hasattr(self.vis_panel, 'update_visualization'):
            self.vis_panel.update_visualization()
            
        # Update background color if needed
        if hasattr(self, 'vis_panel') and hasattr(self.vis_panel, 'set_background_color'):
            self.vis_panel.set_background_color(self.background_color)
              
    def load_new_dataset(self):
        """Load a new dataset"""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select New Simulation Data Folder"
        )
        
        if folder_path and self.load_data_folder(folder_path):
            # Reset UI elements
            self.current_timestep = 0
            self.timestep_slider.setMaximum(self.max_timesteps - 1)
            self.timestep_slider.setValue(0)
            self.populate_type_list()
            
            # Reset visualization settings
            self.show_polarity = False
            self.polarity_checkbox.setChecked(False)
            self.color_by_type = False
            self.color_by_type_checkbox.setChecked(False)
            
            # Reset camera bounds
            if hasattr(self.vis_panel, 'camera_bounds_set'):
                self.vis_panel.camera_bounds_set = False
            
            self.update_visualization()
            self.update_data_info()
            
            QMessageBox.information(self, "Success", "New dataset loaded successfully!")
        elif folder_path:
            QMessageBox.warning(self, "Error", "Failed to load the selected dataset.")
    
    def collect_metadata(self):
        """Collect all current GUI settings and parameters for export"""
        metadata = {
            "export_info": {
                "export_date": datetime.datetime.now().isoformat(),
                "gui_version": "3D Cell Visualization GUI v1.0",
                "export_type": None  # Will be set by calling function
            },
            "dataset_info": {
                "dataset_path": self.current_data_folder if hasattr(self, 'current_data_folder') else None,
                "dataset_name": os.path.basename(self.current_data_folder) if hasattr(self, 'current_data_folder') else "Unknown",
                "max_timesteps": self.max_timesteps,
                "current_timestep": self.current_timestep
            },
            "camera_settings": {
                "position": list(self.vis_panel.get_camera_position()) if hasattr(self.vis_panel, 'get_camera_position') else [0, 0, 0],
                "view_direction": list(self.vis_panel.get_camera_view_direction()) if hasattr(self.vis_panel, 'get_camera_view_direction') else [0, 0, -1],
                "up_direction": list(self.vis_panel.get_camera_up_direction()) if hasattr(self.vis_panel, 'get_camera_up_direction') else [0, 1, 0],
                # Also store turntable camera parameters as backup
                "turntable_distance": float(self.vis_panel.camera.distance) if hasattr(self.vis_panel, 'camera') else 100.0,
                "turntable_elevation": float(self.vis_panel.camera.elevation) if hasattr(self.vis_panel, 'camera') else 30.0,
                "turntable_azimuth": float(self.vis_panel.camera.azimuth) if hasattr(self.vis_panel, 'camera') else 45.0,
                "turntable_fov": float(self.vis_panel.camera.fov) if hasattr(self.vis_panel, 'camera') else 60.0,
                "turntable_roll": float(self.vis_panel.camera.roll) if hasattr(self.vis_panel, 'camera') else 0.0
            },
            "visualization_settings": {
                "cell_size": self.cell_size if hasattr(self, 'cell_size') else 1.0,
                "alpha": self.alpha if hasattr(self, 'alpha') else 1.0,
                "background_color": [0, 0, 0, 1],  # Black background
                "show_axes": True,
                # Polarity settings
                "show_polarity": self.show_polarity if hasattr(self, 'show_polarity') else False,
                "polarity_type": self.polarity_type if hasattr(self, 'polarity_type') else "p",
                # Color mode settings
                "color_by_type": self.color_by_type if hasattr(self, 'color_by_type') else False,
                "color_by_depth": self.color_by_depth if hasattr(self, 'color_by_depth') else False
            },
            "sectioning_settings": {
                "bisection_enabled": self.bisection_enabled if hasattr(self, 'bisection_enabled') else False,
                "bisection_plane": self.bisection_plane if hasattr(self, 'bisection_plane') else "XY",
                "bisection_position": self.bisection_position if hasattr(self, 'bisection_position') else 0.0,
                "cross_section_mode": self.cross_section_mode if hasattr(self, 'cross_section_mode') else False,
                "cross_section_width": self.cross_section_width if hasattr(self, 'cross_section_width') else 2
            },
            "cell_colors": {},
            "filtering_settings": {
                "filtered_types": list(self.filtered_types) if hasattr(self, 'filtered_types') else []
            },
            "playback_settings": {
                "play_speed": self.play_speed if hasattr(self, 'play_speed') else 100,
                "loop_playback": True
            }
        }
        
        # Collect cell type colors
        if hasattr(self, 'type_colors'):
            for cell_type, color in self.type_colors.items():
                if isinstance(color, (list, tuple, np.ndarray)):
                    metadata["cell_colors"][str(cell_type)] = list(color)
                else:
                    metadata["cell_colors"][str(cell_type)] = [0.5, 0.5, 0.5, 1.0]
        
        return metadata
    
    def convert_numpy_types(self, obj):
        """Convert numpy types to JSON-serializable Python types"""
        import numpy as np
        
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64, np.int8, np.int16)):
            return int(obj)
        elif isinstance(obj, (np.bool_, bool)):  # Handle both numpy and regular boolean
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: self.convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self.convert_numpy_types(item) for item in obj)
        else:
            return obj
    
    def save_metadata(self, folder_path, metadata, filename="metadata.json"):
        """Save metadata to a JSON file in the specified folder"""
        metadata_path = os.path.join(folder_path, filename)
        try:
            # Convert numpy types to JSON-serializable types
            serializable_metadata = self.convert_numpy_types(metadata)
            
            with open(metadata_path, 'w') as f:
                json.dump(serializable_metadata, f, indent=2)
            return metadata_path
        except Exception as e:
            print(f"Error saving metadata: {e}")
            return None
    
    def load_metadata(self, metadata_path):
        """Load metadata from a JSON file and apply settings to GUI"""
        try:
            # Check if file exists
            if not os.path.exists(metadata_path):
                print(f"Metadata file does not exist: {metadata_path}")
                return False
            
            # Try to load JSON
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            print(f"Successfully loaded metadata with keys: {list(metadata.keys())}")
            
            # Apply camera settings
            if "camera_settings" in metadata:
                cam_settings = metadata["camera_settings"]
                if hasattr(self, 'vis_panel') and hasattr(self.vis_panel, 'camera'):
                    try:
                        # Try to use the saved camera position if available
                        if "position" in cam_settings and hasattr(self.vis_panel, 'set_camera_position'):
                            self.vis_panel.set_camera_position(cam_settings["position"])
                            print("Camera position applied successfully")
                        
                        # For now, use turntable camera parameters as they are more reliable to restore
                        if "turntable_distance" in cam_settings:
                            self.vis_panel.camera.distance = cam_settings["turntable_distance"]
                        if "turntable_elevation" in cam_settings:
                            self.vis_panel.camera.elevation = cam_settings["turntable_elevation"]
                        if "turntable_azimuth" in cam_settings:
                            self.vis_panel.camera.azimuth = cam_settings["turntable_azimuth"]
                        if "turntable_fov" in cam_settings:
                            self.vis_panel.camera.fov = cam_settings["turntable_fov"]
                        if "turntable_roll" in cam_settings:
                            self.vis_panel.camera.roll = cam_settings["turntable_roll"]
                        
                        # Alternative: use old-style center if no position available
                        if "position" not in cam_settings and hasattr(self.vis_panel.camera, 'center'):
                            self.vis_panel.camera.center = cam_settings.get("position", [0, 0, 0])
                        
                        print("Camera settings applied successfully")
                    except Exception as e:
                        print(f"Error applying camera settings: {e}")
                
                # Update camera position display
                try:
                    if hasattr(self, 'update_camera_display'):
                        self.update_camera_display()
                except Exception as e:
                    print(f"Error updating camera display: {e}")
            
            # Apply visualization settings
            if "visualization_settings" in metadata:
                vis_settings = metadata["visualization_settings"]
                try:
                    if hasattr(self, 'cell_size_slider') and "cell_size" in vis_settings:
                        self.cell_size_slider.setValue(int(vis_settings["cell_size"] * 50))
                    if hasattr(self, 'alpha_slider') and "alpha" in vis_settings:
                        self.alpha_slider.setValue(int(vis_settings["alpha"] * 100))
                    
                    # Apply polarity settings
                    if hasattr(self, 'polarity_checkbox') and "show_polarity" in vis_settings:
                        self.polarity_checkbox.setChecked(vis_settings["show_polarity"])
                    if hasattr(self, 'polarity_combo') and "polarity_type" in vis_settings:
                        polarity_type = vis_settings["polarity_type"]
                        index = self.polarity_combo.findText(polarity_type)
                        if index >= 0:
                            self.polarity_combo.setCurrentIndex(index)
                    
                    # Apply color mode settings
                    if hasattr(self, 'color_by_type_checkbox') and "color_by_type" in vis_settings:
                        self.color_by_type_checkbox.setChecked(vis_settings["color_by_type"])
                    if hasattr(self, 'color_by_depth_checkbox') and "color_by_depth" in vis_settings:
                        self.color_by_depth_checkbox.setChecked(vis_settings["color_by_depth"])
                    
                    print("Visualization settings applied successfully")
                except Exception as e:
                    print(f"Error applying visualization settings: {e}")
            
            # Apply sectioning settings
            if "sectioning_settings" in metadata:
                sect_settings = metadata["sectioning_settings"]
                try:
                    if hasattr(self, 'bisection_checkbox'):
                        self.bisection_checkbox.setChecked(sect_settings.get("bisection_enabled", False))
                    if hasattr(self, 'plane_combo'):
                        plane = sect_settings.get("bisection_plane", "XY")
                        index = self.plane_combo.findText(plane)
                        if index >= 0:
                            self.plane_combo.setCurrentIndex(index)
                    if hasattr(self, 'position_slider'):
                        self.position_slider.setValue(int(sect_settings.get("bisection_position", 0.0) * 100))
                    if hasattr(self, 'cross_section_checkbox'):
                        self.cross_section_checkbox.setChecked(sect_settings.get("cross_section_mode", False))
                    if hasattr(self, 'width_slider'):
                        self.width_slider.setValue(sect_settings.get("cross_section_width", 2))
                    print("Sectioning settings applied successfully")
                except Exception as e:
                    print(f"Error applying sectioning settings: {e}")
            
            # Apply cell colors
            if "cell_colors" in metadata:
                try:
                    if hasattr(self, 'type_colors'):
                        for cell_type, color in metadata["cell_colors"].items():
                            if len(color) >= 3:
                                self.type_colors[int(cell_type)] = tuple(color)
                    print("Cell colors applied successfully")
                except Exception as e:
                    print(f"Error applying cell colors: {e}")
            
            # Apply filtering settings
            if "filtering_settings" in metadata:
                try:
                    if hasattr(self, 'filtered_types'):
                        self.filtered_types = set(metadata["filtering_settings"].get("filtered_types", []))
                    print("Filtering settings applied successfully")
                except Exception as e:
                    print(f"Error applying filtering settings: {e}")
            
            # Apply timestep
            if "dataset_info" in metadata and "current_timestep" in metadata["dataset_info"]:
                try:
                    timestep = metadata["dataset_info"]["current_timestep"]
                    if hasattr(self, 'timestep_slider'):
                        self.timestep_slider.setValue(timestep)
                    self.current_timestep = timestep
                    print("Timestep applied successfully")
                except Exception as e:
                    print(f"Error applying timestep: {e}")
            
            # Apply playback settings
            if "playback_settings" in metadata:
                try:
                    if hasattr(self, 'speed_slider'):
                        speed = metadata["playback_settings"].get("play_speed", 100)
                        # Convert back to slider value (invert the 1010 - speed formula)
                        slider_val = 1010 - speed
                        self.speed_slider.setValue(slider_val)
                    print("Playback settings applied successfully")
                except Exception as e:
                    print(f"Error applying playback settings: {e}")
            
            # Update visualizations
            try:
                if hasattr(self, 'populate_type_list'):
                    self.populate_type_list()
                if hasattr(self, 'update_visualization'):
                    self.update_visualization()
                if hasattr(self, 'update_data_info'):
                    self.update_data_info()
                print("Visualizations updated successfully")
            except Exception as e:
                print(f"Error updating visualizations: {e}")
            
            return True
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return False
        except FileNotFoundError as e:
            print(f"File not found error: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error loading metadata: {e}")
            return False
    
    def import_metadata_file(self):
        """Import metadata file and apply settings to GUI"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Import Metadata", 
            os.path.expanduser("~"),
            "JSON files (*.json);;All files (*.*)"
        )
        
        if not filename:
            return
        
        # Confirm import
        reply = QMessageBox.question(
            self, "Import Metadata", 
            f"This will apply all settings from the metadata file:\n\n"
            f"• Camera position and orientation\n"
            f"• Data sectioning settings\n"
            f"• Cell colors and filtering\n"
            f"• Timestep and playback settings\n\n"
            f"Current settings will be overwritten. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # Load and apply metadata
        success = self.load_metadata(filename)
        
        if success:
            QMessageBox.information(self, "Success", 
                f"Metadata imported successfully!\n\n"
                f"All visualization settings have been restored from:\n{filename}")
        else:
            QMessageBox.warning(self, "Error", 
                f"Failed to import metadata from:\n{filename}\n\n"
                f"Please check that the file is a valid metadata JSON file.")
            
    def export_image(self):
        """Export current view as high-quality image"""
        # Prompt user for metadata export
        reply = QMessageBox.question(
            self, "Export Options", 
            "Do you want to export with metadata?\n\n"
            "Yes: Creates a folder with the image and metadata file\n"
            "No: Saves only the image file",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes
        )
        
        if reply == QMessageBox.StandardButton.Cancel:
            return
        
        include_metadata = reply == QMessageBox.StandardButton.Yes
        
        if include_metadata:
            # Get folder for export
            folder_path = QFileDialog.getExistingDirectory(
                self, "Select Export Folder", 
                os.path.expanduser("~")
            )
            
            if not folder_path:
                return
            
            # Create timestamped subfolder
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            export_folder = os.path.join(folder_path, f"image_export_{timestamp}")
            
            try:
                os.makedirs(export_folder, exist_ok=True)
                
                # Export image to folder
                image_filename = os.path.join(export_folder, f"simulation_frame_{self.current_timestep:04d}.png")
                
                if hasattr(self.vis_panel, 'export_image'):
                    success = self.vis_panel.export_image(image_filename)
                    if success:
                        # Collect and save metadata
                        metadata = self.collect_metadata()
                        metadata["export_info"]["export_type"] = "image"
                        metadata["export_info"]["image_filename"] = os.path.basename(image_filename)
                        
                        metadata_path = self.save_metadata(export_folder, metadata)
                        
                        if metadata_path:
                            QMessageBox.information(self, "Success", 
                                f"Image and metadata exported to:\n{export_folder}")
                        else:
                            QMessageBox.warning(self, "Warning", 
                                f"Image exported but metadata save failed:\n{export_folder}")
                    else:
                        QMessageBox.warning(self, "Error", "Failed to export image")
                        # Clean up empty folder
                        try:
                            os.rmdir(export_folder)
                        except:
                            pass
                else:
                    QMessageBox.warning(self, "Error", "Visualization not ready for export")
                    
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to create export folder:\n{e}")
        
        else:
            # Standard single file export
            filename, _ = QFileDialog.getSaveFileName(
                self, "Export Image", 
                f"simulation_frame_{self.current_timestep:04d}.png",
                "PNG files (*.png);;JPEG files (*.jpg);;All files (*.*)"
            )
            
            if filename:
                if hasattr(self.vis_panel, 'export_image'):
                    success = self.vis_panel.export_image(filename)
                    if success:
                        QMessageBox.information(self, "Success", f"Image exported to:\n{filename}")
                    else:
                        QMessageBox.warning(self, "Error", "Failed to export image")
                else:
                    QMessageBox.warning(self, "Error", "Visualization not ready for export")
        
    def export_video(self):
        """Export animation as high-quality video"""
        # Prompt user for metadata export
        reply = QMessageBox.question(
            self, "Export Options", 
            "Do you want to export with metadata?\n\n"
            "Yes: Creates a folder with the video and metadata file\n"
            "No: Saves only the video file",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes
        )
        
        if reply == QMessageBox.StandardButton.Cancel:
            return
        
        include_metadata = reply == QMessageBox.StandardButton.Yes
        
        # Get frame range
        start_frame, ok1 = QInputDialog.getInt(
            self, "Video Export", "Start frame:", 0, 0, self.max_timesteps - 1
        )
        if not ok1:
            return
            
        end_frame, ok2 = QInputDialog.getInt(
            self, "Video Export", "End frame:", self.max_timesteps - 1, start_frame, self.max_timesteps - 1
        )
        if not ok2:
            return
            
        # Get frame rate
        frame_rate, ok3 = QInputDialog.getInt(
            self, "Video Export", "Frame rate (fps):", 30, 1, 120
        )
        if not ok3:
            return
            
        # Get frame step (render every N frames)
        frame_step, ok4 = QInputDialog.getInt(
            self, "Video Export", "Render every N frames (1 = every frame):", 1, 1, 100
        )
        if not ok4:
            return
        
        if include_metadata:
            # Get folder for export
            folder_path = QFileDialog.getExistingDirectory(
                self, "Select Export Folder", 
                os.path.expanduser("~")
            )
            
            if not folder_path:
                return
            
            # Create timestamped subfolder
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            export_folder = os.path.join(folder_path, f"video_export_{timestamp}")
            
            try:
                os.makedirs(export_folder, exist_ok=True)
                
                # Create video filename in folder
                video_filename = os.path.join(export_folder, "simulation_video.mp4")
                
                # Collect and save metadata before video export
                metadata = self.collect_metadata()
                metadata["export_info"]["export_type"] = "video"
                metadata["export_info"]["video_filename"] = os.path.basename(video_filename)
                metadata["export_info"]["frame_range"] = [start_frame, end_frame]
                metadata["export_info"]["frame_rate"] = frame_rate
                metadata["export_info"]["frame_step"] = frame_step
                
                metadata_path = self.save_metadata(export_folder, metadata)
                
                # Export video
                success = self.export_video_sequence(video_filename, start_frame, end_frame, frame_rate, frame_step)
                
                if success and metadata_path:
                    QMessageBox.information(self, "Success", 
                        f"Video and metadata exported to:\n{export_folder}")
                elif success:
                    QMessageBox.warning(self, "Warning", 
                        f"Video exported but metadata save failed:\n{export_folder}")
                else:
                    QMessageBox.warning(self, "Error", "Failed to export video")
                    # Clean up folder if video failed
                    try:
                        if os.path.exists(export_folder):
                            shutil.rmtree(export_folder)
                    except:
                        pass
                    
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to create export folder:\n{e}")
        
        else:
            # Standard single file export
            filename, _ = QFileDialog.getSaveFileName(
                self, "Export Video", 
                "simulation_video.mp4",
                "MP4 files (*.mp4);;AVI files (*.avi);;All files (*.*)"
            )
            
            if filename:
                success = self.export_video_sequence(filename, start_frame, end_frame, frame_rate, frame_step)
                if success:
                    QMessageBox.information(self, "Success", f"Video exported to:\n{filename}")
                else:
                    QMessageBox.warning(self, "Error", "Failed to export video")
        
    def export_video_sequence(self, filename, start_frame, end_frame, frame_rate, frame_step=1):
        """Export a sequence of frames as a video"""
        try:
            import imageio
        except ImportError:
            QMessageBox.critical(self, "Error", "imageio package required for video export.\nInstall with: pip install imageio[ffmpeg]")
            return False
            
        # Calculate total frames (accounting for frame step)
        total_frames = len(range(start_frame, end_frame + 1, frame_step))
        
        # Create progress dialog with more robust settings
        progress = QProgressDialog("Preparing video export...", "Cancel", 0, total_frames, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)  # Show immediately
        progress.setAutoClose(False)    # Don't auto-close
        progress.setAutoReset(False)    # Don't auto-reset
        progress.show()
        QApplication.processEvents()
        
        # Store current timestep to restore later
        original_timestep = self.current_timestep
        
        # Collect frames
        frames = []
        frames_captured = 0
        user_cancelled = False
        
        try:
            for i, frame_idx in enumerate(range(start_frame, end_frame + 1, frame_step)):
                # Check for cancellation at the start of each iteration
                QApplication.processEvents()
                if progress.wasCanceled():
                    user_cancelled = True
                    print(f"Export canceled by user at frame {frame_idx}")
                    break
                    
                # Update progress label
                progress.setLabelText(f"Capturing frame {i + 1} of {total_frames} (timestep {frame_idx}, step={frame_step})...")
                
                # Update visualization to this frame
                self.current_timestep = frame_idx
                self.timestep_slider.setValue(frame_idx)
                self.update_visualization()
                
                # Give VisPy time to render properly
                for _ in range(3):  # Multiple process events for better rendering
                    QApplication.processEvents()
                    QTimer.singleShot(10, lambda: None)
                    QApplication.processEvents()
                
                # Render frame with improved settings
                try:
                    if hasattr(self.vis_panel, 'canvas') and self.vis_panel.canvas:
                        # Temporarily set canvas to a size divisible by 16 for better video compatibility
                        original_size = self.vis_panel.canvas.size
                        
                        # Calculate size divisible by 16 (macro block size)
                        width = ((original_size[0] + 15) // 16) * 16
                        height = ((original_size[1] + 15) // 16) * 16
                        
                        # Set temporary size for rendering
                        self.vis_panel.canvas.size = (width, height)
                        
                        # Render with alpha channel for better quality
                        img = self.vis_panel.canvas.render(alpha=True)
                            
                        # Restore original size
                        self.vis_panel.canvas.size = original_size
                        
                        if img is not None and img.size > 0:
                            frames.append(img)
                            frames_captured += 1
                        else:
                            print(f"Warning: Empty frame rendered at timestep {frame_idx}")
                    else:
                        print("Error: No canvas found for rendering")
                        break
                except Exception as e:
                    print(f"Error rendering frame {frame_idx}: {str(e)}")
                    # Try fallback rendering
                    try:
                        if hasattr(self.vis_panel, 'canvas') and self.vis_panel.canvas:
                            img = self.vis_panel.canvas.render()
                            if img is not None and img.size > 0:
                                frames.append(img)
                                frames_captured += 1
                    except:
                        pass
                
                # Update progress
                progress.setValue(i + 1)
                
            # Close progress dialog
            progress.close()
            
            print(f"Export complete: Captured {frames_captured} frames, user cancelled: {user_cancelled}")
            
            if frames and not user_cancelled:
                # Save video
                save_progress = QProgressDialog("Saving video file...", None, 0, 0, self)
                save_progress.setWindowModality(Qt.WindowModality.WindowModal)
                save_progress.setMinimumDuration(0)
                save_progress.show()
                QApplication.processEvents()
                
                try:
                    # Use imageio to create video with better settings
                    # Set macro_block_size to 1 to prevent resizing warnings
                    # Use higher quality settings to avoid truncation
                    imageio.mimsave(filename, frames, fps=frame_rate, 
                                  quality=9,  # Higher quality (1-10 scale)
                                  macro_block_size=1,  # Prevent automatic resizing
                                  ffmpeg_params=['-crf', '18',  # High quality encoding
                                               '-preset', 'slow',  # Better compression
                                               '-pix_fmt', 'yuv420p'])  # Move pix_fmt to end
                    save_progress.close()
                    QMessageBox.information(self, "Success", f"Video exported successfully!\nFrames: {len(frames)}\nFile: {filename}")
                    return True
                except Exception as e:
                    save_progress.close()
                    QMessageBox.critical(self, "Error", f"Failed to save video file:\n{str(e)}")
                    return False
                    
            elif user_cancelled:
                QMessageBox.information(self, "Cancelled", "Video export was cancelled by user")
                return False
            else:
                QMessageBox.warning(self, "Error", f"No frames were captured.\nCaptured: {frames_captured} frames out of {total_frames} expected")
                return False
                
        except Exception as e:
            progress.close()
            print(f"Video export error: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to export video:\n{str(e)}")
            return False
        finally:
            # Restore original timestep
            self.current_timestep = original_timestep
            self.timestep_slider.setValue(original_timestep)
            self.update_visualization()

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show the main window
    window = DataVisualizationGUI()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()