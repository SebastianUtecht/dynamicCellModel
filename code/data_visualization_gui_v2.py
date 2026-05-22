"""Data Visualization GUI v2 — for dict-of-lists simulation data."""

import sys
import os
import pickle
import json
import datetime
import time

import numpy as np
from scipy.spatial import cKDTree

import vispy
from vispy import scene
from vispy.scene import visuals

# Import Voronoi components
try:
    from voronoi_animator import VoronoiAnimator
    VORONOI_AVAILABLE = True
except ImportError:
    VORONOI_AVAILABLE = False

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QSlider, QFileDialog,
    QMessageBox, QCheckBox, QScrollArea, QVBoxLayout, QHBoxLayout,
    QTextBrowser, QLabel, QComboBox, QSpinBox, QDoubleSpinBox, QListWidget,
    QListWidgetItem, QSplitter, QProgressDialog, QRadioButton, QButtonGroup,
    QGroupBox, QTabWidget, QFormLayout, QFrame, QDialog,
)
from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt6.QtGui import QFont, QColor, QPalette

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Dark colour palette
# ---------------------------------------------------------------------------
_BG       = "#1a1a2e"   # window / main background
_PANEL    = "#16213e"   # panel / groupbox background
_SURFACE  = "#0f3460"   # slightly raised surfaces (tabs, headers)
_ACCENT   = "#e94560"   # accent (active tab underline, checked radio)
_TEXT     = "#e0e0f0"   # primary text
_SUBTEXT  = "#9090b0"   # secondary / label text
_BORDER   = "#2a2a4a"   # borders
_BTN_PRI  = "#533483"   # primary button (load, search)
_BTN_SUC  = "#1a6b4a"   # success button (export image)
_BTN_VID  = "#6b3a1a"   # video export button
_INPUT_BG = "#1e1e3a"   # spinbox / combobox background

DARK_STYLESHEET = f"""
/* ── Window & base ─────────────────────────────────────────────────── */
QMainWindow, QWidget {{
    background-color: {_BG};
    color: {_TEXT};
    font-family: "Segoe UI", "Inter", Arial, sans-serif;
    font-size: 12px;
}}

/* ── GroupBox ───────────────────────────────────────────────────────── */
QGroupBox {{
    background-color: {_PANEL};
    border: 1px solid {_BORDER};
    border-radius: 8px;
    margin-top: 18px;
    padding: 6px 8px 8px 8px;
    font-weight: 600;
    color: {_TEXT};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 2px 8px;
    color: {_TEXT};
    background-color: {_SURFACE};
    border-radius: 4px;
}}

/* ── Tabs ───────────────────────────────────────────────────────────── */
QTabWidget::pane {{
    border: 1px solid {_BORDER};
    border-radius: 6px;
    background-color: {_PANEL};
    top: -1px;
}}
QTabBar::tab {{
    background-color: {_BG};
    color: {_SUBTEXT};
    padding: 7px 16px;
    border: 1px solid transparent;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    min-width: 70px;
}}
QTabBar::tab:selected {{
    background-color: {_PANEL};
    color: {_TEXT};
    border-color: {_BORDER};
    border-bottom: 2px solid {_ACCENT};
}}
QTabBar::tab:hover:!selected {{
    background-color: {_SURFACE};
    color: {_TEXT};
}}

/* ── Buttons ────────────────────────────────────────────────────────── */
QPushButton {{
    background-color: {_BTN_PRI};
    color: {_TEXT};
    border: none;
    border-radius: 6px;
    padding: 6px 14px;
    font-weight: 600;
}}
QPushButton:hover {{
    background-color: #6a43a0;
}}
QPushButton:pressed {{
    background-color: #3d2260;
}}
QPushButton:disabled {{
    background-color: #2a2a4a;
    color: {_SUBTEXT};
}}
QPushButton#success {{
    background-color: {_BTN_SUC};
}}
QPushButton#success:hover {{
    background-color: #238a5e;
}}
QPushButton#video {{
    background-color: {_BTN_VID};
}}
QPushButton#video:hover {{
    background-color: #8a4a20;
}}
QPushButton#load {{
    background-color: #c0392b;
    padding: 5px 12px;
}}
QPushButton#load:hover {{
    background-color: #e74c3c;
}}

/* ── Sliders ────────────────────────────────────────────────────────── */
QSlider::groove:horizontal {{
    height: 4px;
    background: {_BORDER};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {_ACCENT};
    border: none;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}
QSlider::sub-page:horizontal {{
    background: {_ACCENT};
    border-radius: 2px;
}}

/* ── SpinBox / DoubleSpinBox ────────────────────────────────────────── */
QSpinBox, QDoubleSpinBox {{
    background-color: {_INPUT_BG};
    color: {_TEXT};
    border: 1px solid {_BORDER};
    border-radius: 5px;
    padding: 3px 6px;
    selection-background-color: {_ACCENT};
}}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    background-color: {_SURFACE};
    border: none;
    border-radius: 3px;
    width: 16px;
}}
QSpinBox::up-button:hover, QSpinBox::down-button:hover,
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {{
    background-color: {_ACCENT};
}}

/* ── ComboBox ───────────────────────────────────────────────────────── */
QComboBox {{
    background-color: {_INPUT_BG};
    color: {_TEXT};
    border: 1px solid {_BORDER};
    border-radius: 5px;
    padding: 4px 8px;
    selection-background-color: {_ACCENT};
}}
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 20px;
    border-left: 1px solid {_BORDER};
    border-top-right-radius: 5px;
    border-bottom-right-radius: 5px;
    background-color: {_SURFACE};
}}
QComboBox QAbstractItemView {{
    background-color: {_PANEL};
    color: {_TEXT};
    selection-background-color: {_ACCENT};
    border: 1px solid {_BORDER};
    border-radius: 4px;
}}

/* ── CheckBox / RadioButton ─────────────────────────────────────────── */
QCheckBox, QRadioButton {{
    color: {_TEXT};
    spacing: 6px;
}}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 14px;
    height: 14px;
    border: 2px solid {_BORDER};
    border-radius: 3px;
    background-color: {_INPUT_BG};
}}
QRadioButton::indicator {{
    border-radius: 7px;
}}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background-color: {_ACCENT};
    border-color: {_ACCENT};
}}

/* ── ListWidget ─────────────────────────────────────────────────────── */
QListWidget {{
    background-color: {_INPUT_BG};
    border: 1px solid {_BORDER};
    border-radius: 6px;
    color: {_TEXT};
    outline: none;
}}
QListWidget::item {{
    border-radius: 4px;
    padding: 2px 4px;
}}
QListWidget::item:selected {{
    background-color: {_SURFACE};
}}

/* ── TextBrowser ────────────────────────────────────────────────────── */
QTextBrowser {{
    background-color: {_INPUT_BG};
    border: 1px solid {_BORDER};
    border-radius: 6px;
    color: {_TEXT};
    padding: 4px;
}}

/* ── ScrollArea ─────────────────────────────────────────────────────── */
QScrollArea {{
    border: none;
    background-color: transparent;
}}
QScrollBar:vertical {{
    background-color: {_BG};
    width: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background-color: {_SURFACE};
    border-radius: 4px;
    min-height: 20px;
}}
QScrollBar::handle:vertical:hover {{
    background-color: {_ACCENT};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

/* ── Splitter ───────────────────────────────────────────────────────── */
QSplitter::handle {{
    background-color: {_BORDER};
    width: 2px;
}}

/* ── Label ──────────────────────────────────────────────────────────── */
QLabel {{
    color: {_TEXT};
    background-color: transparent;
}}

/* ── MessageBox / ProgressDialog ───────────────────────────────────── */
QMessageBox, QProgressDialog {{
    background-color: {_PANEL};
    color: {_TEXT};
}}
QProgressBar {{
    background-color: {_INPUT_BG};
    border: 1px solid {_BORDER};
    border-radius: 4px;
    text-align: center;
    color: {_TEXT};
}}
QProgressBar::chunk {{
    background-color: {_ACCENT};
    border-radius: 3px;
}}
"""


# ---------------------------------------------------------------------------
# Scalar colormap (3-point magma-like)
# ---------------------------------------------------------------------------
_COLOR_START = np.array([0.0, 0.0, 0.3])   # dark purple
_COLOR_MID   = np.array([1.0, 0.2, 0.0])   # red/orange
_COLOR_END   = np.array([1.0, 1.0, 0.7])   # light yellow

def scalar_to_rgba(values: np.ndarray, vmin: float, vmax: float,
                   vcenter: float | None = None) -> np.ndarray:
    """Map 1-D scalar array to RGBA using 3-point magma-like colormap.

    If `vcenter` is provided, uses piecewise normalization so that
    `vcenter` maps exactly to the midpoint (0.5) of the colormap.
    """
    if vcenter is None:
        vrange = vmax - vmin
        if vrange > 0:
            norm = np.clip((values - vmin) / vrange, 0.0, 1.0)
        else:
            norm = np.zeros_like(values)
    else:
        norm = np.full_like(values, 0.5, dtype=np.float32)
        lo_mask = values <= vcenter
        hi_mask = ~lo_mask

        left = vcenter - vmin
        right = vmax - vcenter

        if left > 1e-12:
            norm[lo_mask] = 0.5 * (values[lo_mask] - vmin) / left
        else:
            norm[lo_mask] = 0.5

        if right > 1e-12:
            norm[hi_mask] = 0.5 + 0.5 * (values[hi_mask] - vcenter) / right
        else:
            norm[hi_mask] = 0.5

        norm = np.clip(norm, 0.0, 1.0)

    colors = np.ones((len(values), 4), dtype=np.float32)
    lo = norm < 0.5
    t_lo = norm[lo] * 2.0
    colors[lo, :3] = np.outer(1 - t_lo, _COLOR_START) + np.outer(t_lo, _COLOR_MID)
    hi = ~lo
    t_hi = (norm[hi] - 0.5) * 2.0
    colors[hi, :3] = np.outer(1 - t_hi, _COLOR_MID) + np.outer(t_hi, _COLOR_END)
    return colors


# ---------------------------------------------------------------------------
# Colour picker helper: click a label to open a dialog
# ---------------------------------------------------------------------------
class _ColorLabel(QLabel):
    """Clickable colour swatch label."""

    def __init__(self, color: tuple, parent=None):
        super().__init__(parent)
        self.color = color
        self._refresh()

    def _refresh(self):
        r, g, b = [int(c * 255) for c in self.color[:3]]
        self.setStyleSheet(
            f"background-color: rgb({r},{g},{b}); border: 1px solid #555; border-radius: 3px;"
        )
        self.setFixedSize(20, 20)

    def set_color(self, color: tuple):
        self.color = color
        self._refresh()


# ---------------------------------------------------------------------------
# VisPy 3-D Widget
# ---------------------------------------------------------------------------
class VisPy3DWidget(QWidget):
    """VisPy-backed 3D visualization canvas."""

    # Per-frame state: updated before render() returns
    _cell_scatters: dict
    _polarity_markers: object
    _vector_visuals: dict
    _range_plane_visuals: list
    _range_plane_token: int
    _highlights: dict

    def __init__(self, gui: "DataVizGUI"):
        super().__init__()
        self.gui = gui
        self._cell_scatters = {}
        self._polarity_markers = None
        self._vector_visuals = {}
        self._range_plane_visuals = []
        self._range_plane_token = 0
        self._highlights = {}
        self._camera_bounds_set = False
        self._ghost_visual = None
        self._ghost_pos = None

        # Voronoi visualization
        self.voronoi_animator = None
        self.voronoi_mesh_batch = None  # Single batched Mesh visual for all cells
        self.voronoi_edges_visual = None  # Edges visualization
        self.voronoi_enabled = False
        self.voronoi_show_edges = False  # Toggle for edge visualization
        self.voronoi_edge_color = np.array([1.0, 1.0, 1.0, 1.0])  # White by default
        self.voronoi_transparency = 0.5
        # Store for edge rendering
        self._voronoi_vertices_combined = None
        self._voronoi_faces_combined = None
        self._voronoi_boundary_edges = []  # Boundary edges for rendering
        # Store base colors and face normals for camera-relative lighting
        self._voronoi_base_vertex_colors = None
        self._voronoi_face_normals = None

        self._init_canvas()

    # ------------------------------------------------------------------
    # Canvas setup
    # ------------------------------------------------------------------
    def _init_canvas(self):
        self.canvas = scene.SceneCanvas(
            keys="interactive", show=False, bgcolor="black",
            size=(800, 600), resizable=True,
        )
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas.native)
        self.setLayout(layout)

        self.view = self.canvas.central_widget.add_view()
        self.view.camera = "fly"
        self.view.camera.fov = 60
        self.view.camera.auto_roll = False

        # Coordinate axes (X=green, Y=red, Z=blue)
        self.axis_lines = []
        for color, pts in [
            ((0, 1, 0, 1), [[-100, 0, 0], [100, 0, 0]]),
            ((1, 0, 0, 1), [[0, -100, 0], [0, 100, 0]]),
            ((0, 0, 1, 1), [[0, 0, -100], [0, 0, 100]]),
        ]:
            ln = visuals.Line(pos=np.array(pts), color=color, width=3)
            self.view.add(ln)
            ln.visible = False
            self.axis_lines.append(ln)

        # Ghost sphere for neighbour selection
        self._ghost_visual = visuals.Markers(scaling=True, alpha=0.5, spherical=True)
        self.view.add(self._ghost_visual)
        self._ghost_visual.visible = False
        self._ghost_sphere_dist = 10.0
        self._ghost_sphere_radius = 3.0

        # Timer to update ghost sphere
        from vispy import app as _vapp
        self._ghost_timer = _vapp.Timer(
            interval=0.05, connect=self._update_ghost_sphere, start=True
        )

        # Mouse events
        @self.canvas.events.mouse_press.connect
        def _on_mouse_press(event):
            if event.button == 2 and (self.gui.neighbor_search_enabled or getattr(self.gui, "cell_pick_enabled", False)):
                self._on_right_click()
                event.handled = True

        @self.canvas.events.mouse_wheel.connect
        def _on_wheel(event):
            shift = False
            if hasattr(event, "modifiers") and event.modifiers:
                shift = "shift" in str(event.modifiers).lower()
            if not shift and hasattr(event, "native") and hasattr(event.native, "modifiers"):
                shift = bool(event.native.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            if shift and (self.gui.neighbor_search_enabled or getattr(self.gui, "cell_pick_enabled", False)):
                delta = event.delta[1] if hasattr(event, "delta") and len(event.delta) > 1 else 0
                self._ghost_sphere_dist = np.clip(
                    self._ghost_sphere_dist + delta, 3.0, 50.0
                )
                event.handled = True

        @self.canvas.connect
        def on_key_press(event):
            if event.text == " ":
                self.gui.toggle_play()
            elif event.text == ",":
                self.gui.prev_frame()
            elif event.text == ".":
                self.gui.next_frame()

    # ------------------------------------------------------------------
    # Ghost sphere
    # ------------------------------------------------------------------
    def _update_ghost_sphere(self, _event=None):
        if self.gui.neighbor_search_enabled or getattr(self.gui, "cell_pick_enabled", False):
            self._ghost_pos = self._camera_forward_pos(self._ghost_sphere_dist)
            self._ghost_visual.set_data(
                np.array([self._ghost_pos]),
                edge_width=0, face_color="cyan",
                size=self._ghost_sphere_radius * 2,
            )
            self._ghost_visual.visible = True
        else:
            self._ghost_visual.visible = False

        # Update Voronoi mesh lighting based on camera position
        self._apply_camera_relative_lighting()

    # ------------------------------------------------------------------
    # Camera helpers
    # ------------------------------------------------------------------
    def _camera_forward_pos(self, distance: float) -> np.ndarray:
        try:
            t = self.view.camera.transform
            fwd = t.map([0, 0, -1, 0])[:3]
            pos = t.map([0, 0, 0, 1])[:3]
            return pos + fwd * distance
        except Exception:
            return np.zeros(3)

    def get_camera_view_direction(self) -> np.ndarray:
        try:
            t = self.view.camera.transform
            if t is not None and hasattr(t, "map"):
                d = t.map([0, 0, -1, 0])[:3]
                n = np.linalg.norm(d)
                return d / n if n > 1e-6 else np.array([0, 0, -1.0])
        except Exception:
            pass
        return np.array([0, 0, -1.0])

    def update_camera_bounds(self, x: np.ndarray):
        if len(x) == 0:
            return
        bounds = np.array([x.min(axis=0), x.max(axis=0)])
        center = bounds.mean(axis=0)
        extent = (bounds[1] - bounds[0]) * 1.1
        self.view.camera.set_range(
            x=[center[0] - extent[0] / 2, center[0] + extent[0] / 2],
            y=[center[1] - extent[1] / 2, center[1] + extent[1] / 2],
            z=[center[2] - extent[2] / 2, center[2] + extent[2] / 2],
        )

    def set_background(self, name: str):
        self.canvas.bgcolor = {"black": "black", "white": "white", "gray": "gray"}.get(
            name.lower(), "black"
        )

    # ------------------------------------------------------------------
    # Section filter
    # ------------------------------------------------------------------
    def _apply_section_filter(self, x: np.ndarray, p_mask: np.ndarray):
        """Return (x_f, p_mask_f, bool_mask)."""
        g = self.gui
        if not g.bisection_enabled:
            mask = np.ones(len(x), dtype=bool)
            if p_mask is not None and g.visible_types:
                mask &= np.isin(p_mask, list(g.visible_types))
            return x[mask], (p_mask[mask] if p_mask is not None else None), mask

        bounds = np.array([x.min(axis=0), x.max(axis=0)])
        center = bounds.mean(axis=0)
        extent = bounds[1] - bounds[0]

        if g.bisection_plane == "Camera Orthogonal":
            view_dir = self.get_camera_view_direction()
            dists = np.dot(x - center, view_dir)
            max_ext = np.max(extent)
            offset = g.bisection_position * max_ext * 0.5
            if g.cross_section_mode:
                mask = np.abs(dists - offset) <= g.cross_section_width * 0.5
            else:
                mask = dists <= offset
        else:
            ax = {"XY": 2, "XZ": 1, "YZ": 0}[g.bisection_plane]
            plane_pos = center[ax] + g.bisection_position * extent[ax] * 0.5
            if g.cross_section_mode:
                mask = np.abs(x[:, ax] - plane_pos) <= g.cross_section_width * 0.5
            else:
                mask = x[:, ax] <= plane_pos

        if p_mask is not None and g.visible_types:
            mask &= np.isin(p_mask, list(g.visible_types))

        return (
            x[mask] if mask.any() else np.empty((0, 3)),
            (p_mask[mask] if p_mask is not None and mask.any() else
             (np.array([]) if p_mask is not None else None)),
            mask,
        )

    # ------------------------------------------------------------------
    # Clear helpers
    # ------------------------------------------------------------------
    def _clear_cells(self):
        for v in self._cell_scatters.values():
            try:
                v.parent = None
            except Exception as e:
                print(f"[clear] {e}")
        self._cell_scatters.clear()
        if self._polarity_markers is not None:
            try:
                self._polarity_markers.parent = None
            except Exception as e:
                print(f"[clear polarity] {e}")
            self._polarity_markers = None

        for v in self._vector_visuals.values():
            try:
                v.parent = None
            except Exception as e:
                print(f"[clear vectors] {e}")
        self._vector_visuals.clear()

        for v in self._range_plane_visuals:
            try:
                v.parent = None
            except Exception as e:
                print(f"[clear range planes] {e}")
        self._range_plane_visuals.clear()

    def show_range_planes(self, axis: int, lo: float, hi: float,
                          duration_s: float = 3.0, alpha: float = 0.5):
        """Show two translucent planes at `lo` and `hi` along `axis` for a short time."""
        g = self.gui
        if g.data is None:
            return
        t = g.current_timestep
        if "x" not in g.data or t >= len(g.data["x"]):
            return
        x = g.data["x"][t]
        if x is None or len(x) == 0:
            return

        try:
            lo_f = float(lo)
            hi_f = float(hi)
        except Exception:
            return
        if hi_f < lo_f:
            lo_f, hi_f = hi_f, lo_f

        # Remove any existing range indicators
        for v in self._range_plane_visuals:
            try:
                v.parent = None
            except Exception:
                pass
        self._range_plane_visuals.clear()

        bounds = np.array([np.min(x, axis=0), np.max(x, axis=0)], dtype=np.float32)
        extent = bounds[1] - bounds[0]
        pad = np.where(extent > 0, extent * 0.05, 1.0)
        lo_b = bounds[0] - pad
        hi_b = bounds[1] + pad

        # Build a quad spanning the other two axes
        axes_other = [0, 1, 2]
        if axis not in axes_other:
            return
        axes_other.remove(axis)
        a1, a2 = axes_other

        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        color = (1.0, 1.0, 1.0, float(alpha))

        def _make_plane(pos_value: float):
            v = np.zeros((4, 3), dtype=np.float32)
            v[:, axis] = float(pos_value)
            v[0, a1], v[0, a2] = lo_b[a1], lo_b[a2]
            v[1, a1], v[1, a2] = hi_b[a1], lo_b[a2]
            v[2, a1], v[2, a2] = hi_b[a1], hi_b[a2]
            v[3, a1], v[3, a2] = lo_b[a1], hi_b[a2]

            m = visuals.Mesh(vertices=v, faces=faces, color=color, shading=None)
            try:
                m.set_gl_state('translucent', depth_test=True)
            except Exception:
                pass
            self.view.add(m)
            self._range_plane_visuals.append(m)

        _make_plane(lo_f)
        _make_plane(hi_f)
        self.canvas.update()

        self._range_plane_token += 1
        token = self._range_plane_token

        def _hide_if_still_current():
            if token != self._range_plane_token:
                return
            for v in self._range_plane_visuals:
                try:
                    v.parent = None
                except Exception:
                    pass
            self._range_plane_visuals.clear()
            self.canvas.update()

        QTimer.singleShot(int(duration_s * 1000), _hide_if_still_current)

    def clear_highlights(self):
        for v in self._highlights.values():
            try:
                v.parent = None
            except Exception as e:
                print(f"[clear highlight] {e}")
        self._highlights.clear()
        self.render()

    def _clear_voronoi_meshes(self):
        """Remove batched Voronoi mesh visual and edges."""
        if self.voronoi_mesh_batch is not None:
            try:
                self.voronoi_mesh_batch.parent = None
            except Exception as e:
                print(f"[clear voronoi batch] {e}")
            self.voronoi_mesh_batch = None

        if self.voronoi_edges_visual is not None:
            try:
                self.voronoi_edges_visual.parent = None
            except Exception as e:
                print(f"[clear voronoi edges] {e}")
            self.voronoi_edges_visual = None

        # Clear stored data
        self._voronoi_vertices_combined = None
        self._voronoi_faces_combined = None
        self._voronoi_boundary_edges = []
        self._voronoi_base_vertex_colors = None
        self._voronoi_face_normals = None

    def _add_batched_voronoi_meshes(self, meshes_list, colors=None):
        """
        Batch render all Voronoi meshes as a single Mesh visual.

        This combines all per-cell meshes into one visual to reduce draw calls
        from ~1000 to ~1, dramatically improving performance.

        Parameters
        ----------
        meshes_list : list of VoronoiMesh
            List of Voronoi mesh objects to render
        colors : list of ndarray, optional
            List of RGBA colors for each mesh. If None, uses mesh.color
        """
        if not meshes_list:
            self._clear_voronoi_meshes()
            return

        try:
            # Concatenate all vertices and adjust face indices
            all_vertices = []
            all_faces = []
            all_boundary_edges = []  # Store boundary edges for each mesh
            vertex_offset = 0

            for mesh in meshes_list:
                if mesh.vertices_3d.shape[0] == 0:
                    continue  # Skip empty meshes

                all_vertices.append(mesh.vertices_3d)

                # Adjust face indices by vertex offset
                adjusted_faces = mesh.faces + vertex_offset
                all_faces.append(adjusted_faces)

                # Extract all edges from this individual mesh
                edges_in_mesh = {}
                for face in mesh.faces:
                    edges = [
                        tuple(sorted([face[0], face[1]])),
                        tuple(sorted([face[1], face[2]])),
                        tuple(sorted([face[2], face[0]])),
                    ]
                    for edge in edges:
                        edges_in_mesh[edge] = edges_in_mesh.get(edge, 0) + 1

                # Collect all unique edges from this mesh and adjust indices
                # Dictionary keys are already unique, each edge appears only once per cell
                for edge in edges_in_mesh.keys():
                    adjusted_edge = (edge[0] + vertex_offset, edge[1] + vertex_offset)
                    all_boundary_edges.append(adjusted_edge)

                vertex_offset += len(mesh.vertices_3d)

            if not all_vertices:  # All meshes are empty
                self._clear_voronoi_meshes()
                return

            # Stack into single arrays
            vertices_combined = np.vstack(all_vertices)
            faces_combined = np.vstack(all_faces)

            # Create vertex colors array (per-vertex, repeated from per-mesh colors)
            if colors is not None:
                vertex_colors = []
                mesh_idx = 0
                for mesh in meshes_list:
                    if mesh.vertices_3d.shape[0] == 0:
                        continue
                    mesh_color = colors[mesh_idx] if mesh_idx < len(colors) else np.array([0.5, 0.5, 0.5, 1.0])
                    vertex_colors.extend([mesh_color] * len(mesh.vertices_3d))
                    mesh_idx += 1
                vertex_colors = np.array(vertex_colors, dtype=np.float32)
            else:
                vertex_colors = None

            # Clear old mesh if exists
            self._clear_voronoi_meshes()

            # Store vertices and faces for edge rendering
            self._voronoi_vertices_combined = vertices_combined
            self._voronoi_faces_combined = faces_combined
            self._voronoi_base_vertex_colors = vertex_colors.copy() if vertex_colors is not None else None
            self._voronoi_boundary_edges = all_boundary_edges  # Store all edges for visualization

            print(f"[batch voronoi] Extracted {len(all_boundary_edges)} edges from {len(meshes_list)} Voronoi cells")

            # Compute face normals for lighting calculations
            self._voronoi_face_normals = self._compute_face_normals(vertices_combined, faces_combined)

            # Create single batched Mesh visual with flat shading for clean domain appearance
            # Flat shading means each triangle is uniformly shaded based on its face normal,
            # avoiding visible triangulation structure within domains
            self.voronoi_mesh_batch = visuals.Mesh(
                vertices=vertices_combined,
                faces=faces_combined,
                vertex_colors=vertex_colors,
                shading='flat'  # Flat shading creates faceted appearance without visible triangulation
            )

            # Add to view
            self.view.add(self.voronoi_mesh_batch)

            # Edges will be rendered when user toggles them on
        except Exception as e:
            print(f"[batch voronoi] Failed to create batched mesh: {e}")
            import traceback
            traceback.print_exc()
            self._clear_voronoi_meshes()

    def _compute_face_normals(self, vertices, faces):
        """Compute normal vector for each face."""
        try:
            face_normals = np.zeros((len(faces), 3), dtype=np.float32)
            for i, face in enumerate(faces):
                v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
                edge1 = v1 - v0
                edge2 = v2 - v0
                normal = np.cross(edge1, edge2)
                norm = np.linalg.norm(normal)
                if norm > 1e-6:
                    face_normals[i] = normal / norm
                else:
                    face_normals[i] = np.array([0, 0, 1], dtype=np.float32)
            return face_normals
        except Exception as e:
            print(f"[compute face normals] Failed: {e}")
            return None

    def _compute_smooth_vertex_normals(self, vertices, faces):
        """Compute smooth vertex normals by averaging adjacent face normals."""
        try:
            # Compute face normals
            face_normals = np.zeros((len(faces), 3), dtype=np.float32)
            for i, face in enumerate(faces):
                v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
                edge1 = v1 - v0
                edge2 = v2 - v0
                normal = np.cross(edge1, edge2)
                norm = np.linalg.norm(normal)
                if norm > 1e-6:
                    face_normals[i] = normal / norm
                else:
                    face_normals[i] = np.array([0, 0, 1], dtype=np.float32)

            # Initialize vertex normals
            vertex_normals = np.zeros_like(vertices)

            # Accumulate face normals to each vertex
            for face_idx, face in enumerate(faces):
                for vertex_idx in face:
                    vertex_normals[vertex_idx] += face_normals[face_idx]

            # Normalize
            norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
            norms[norms < 1e-6] = 1.0  # Avoid division by zero
            vertex_normals = vertex_normals / norms

            return vertex_normals
        except Exception as e:
            print(f"[compute smooth vertex normals] Failed: {e}")
            return None

    def _apply_camera_relative_lighting(self):
        """Apply camera-relative lighting to mesh by adjusting vertex colors."""
        if (not self.voronoi_enabled or self.voronoi_mesh_batch is None or
            self._voronoi_base_vertex_colors is None or
            self._voronoi_vertices_combined is None or
            self._voronoi_faces_combined is None or
            self._voronoi_face_normals is None):
            return

        try:
            # Get camera position and forward direction
            cam_pos = self.view.camera.pos
            cam_forward = self.get_camera_view_direction()

            # Place light behind and above the camera (similar to test_vispy.py's approach)
            # Position: camera position + offset in view direction and up
            light_offset = cam_forward * 5 + np.array([2, 2, 2])
            light_pos = cam_pos + light_offset

            # Compute per-vertex brightness based on face normals and distance to light
            vertex_brightness = np.ones(len(self._voronoi_vertices_combined), dtype=np.float32)

            for face_idx, face in enumerate(self._voronoi_faces_combined):
                normal = self._voronoi_face_normals[face_idx]

                # Get face center
                v0, v1, v2 = (self._voronoi_vertices_combined[face[0]],
                             self._voronoi_vertices_combined[face[1]],
                             self._voronoi_vertices_combined[face[2]])
                face_center = (v0 + v1 + v2) / 3.0

                # Vector from face to light
                to_light = light_pos - face_center
                dist = np.linalg.norm(to_light) + 1e-6
                to_light = to_light / dist

                # Compute two-sided lighting: use absolute value of dot product
                # so back-facing triangles are also illuminated (represents indirect light)
                brightness = abs(np.dot(normal, to_light))
                # Higher ambient term (0.6) for better visibility, lower diffuse (0.4)
                brightness = 0.6 + 0.4 * brightness

                # Apply to all vertices of this face
                for vertex_idx in face:
                    vertex_brightness[vertex_idx] = max(vertex_brightness[vertex_idx], brightness)

            # Apply brightness to base colors
            lit_colors = self._voronoi_base_vertex_colors.copy()
            for i in range(len(lit_colors)):
                lit_colors[i, 0] *= vertex_brightness[i]  # R
                lit_colors[i, 1] *= vertex_brightness[i]  # G
                lit_colors[i, 2] *= vertex_brightness[i]  # B
                # Keep alpha unchanged

            # Update mesh colors
            if self.voronoi_mesh_batch is not None:
                self.voronoi_mesh_batch.mesh_data.vertex_colors = lit_colors
        except Exception as e:
            pass  # Silent fail for lighting updates

    def _render_voronoi_edges(self, vertices_combined):
        """
        Render the edges of the Voronoi cell domains.

        Uses pre-extracted edges from individual cell meshes.
        """
        if not self.voronoi_show_edges:
            if self.voronoi_edges_visual is not None:
                self.voronoi_edges_visual.parent = None
                self.voronoi_edges_visual = None
            return

        try:
            # Use pre-extracted edges
            if not hasattr(self, '_voronoi_boundary_edges') or not self._voronoi_boundary_edges:
                print(f"[render edges] No edges available")
                return

            edges = self._voronoi_boundary_edges

            # Convert to edge pairs for Line visual
            edge_positions = []
            for edge in edges:
                edge_positions.append([vertices_combined[edge[0]], vertices_combined[edge[1]]])

            if not edge_positions:
                print(f"[render edges] No edge positions to render")
                return

            edge_positions = np.array(edge_positions, dtype=np.float32)

            # Remove old edges if they exist
            if self.voronoi_edges_visual is not None:
                self.voronoi_edges_visual.parent = None

            # Create new edges visual with better visibility
            self.voronoi_edges_visual = visuals.Line(
                pos=edge_positions,
                color=self.voronoi_edge_color,
                width=2.5,  # Increased from 1.0 for better visibility
                connect='segments',
                antialias=True
            )
            # Disable depth test for edges so they appear on top
            self.voronoi_edges_visual.set_gl_state('translucent', depth_test=False)
            self.view.add(self.voronoi_edges_visual)
            print(f"[render edges] Rendered {len(edges)} Voronoi cell edges")
        except Exception as e:
            print(f"[render edges] Failed: {e}")
            import traceback
            traceback.print_exc()
            if self.voronoi_edges_visual is not None:
                self.voronoi_edges_visual.parent = None
                self.voronoi_edges_visual = None

    def toggle_voronoi_edges(self, show: bool = None):
        """
        Toggle Voronoi edge visualization.

        Parameters
        ----------
        show : bool, optional
            If None, toggles current state. Otherwise sets to specified state.
        """
        if show is None:
            self.voronoi_show_edges = not self.voronoi_show_edges
        else:
            self.voronoi_show_edges = show

        # Re-render edges if mesh data is available
        if self._voronoi_vertices_combined is not None:
            self._render_voronoi_edges(self._voronoi_vertices_combined)

    def set_voronoi_edge_color(self, color: str = "white"):
        """
        Set the color of Voronoi edges.

        Parameters
        ----------
        color : str
            Color name: 'white', 'black', 'red', 'green', 'blue', 'yellow', 'cyan', 'magenta'
        """
        color_map = {
            'white': np.array([1.0, 1.0, 1.0, 1.0]),
            'black': np.array([0.0, 0.0, 0.0, 1.0]),
            'red': np.array([1.0, 0.0, 0.0, 1.0]),
            'green': np.array([0.0, 1.0, 0.0, 1.0]),
            'blue': np.array([0.0, 0.0, 1.0, 1.0]),
            'yellow': np.array([1.0, 1.0, 0.0, 1.0]),
            'cyan': np.array([0.0, 1.0, 1.0, 1.0]),
            'magenta': np.array([1.0, 0.0, 1.0, 1.0]),
        }
        if color.lower() in color_map:
            self.voronoi_edge_color = color_map[color.lower()]
            # Update edges if they're being shown
            if self.voronoi_show_edges and self.voronoi_edges_visual is not None:
                self.voronoi_edges_visual.color = self.voronoi_edge_color



    def update_voronoi_colors(self):
        """Update Voronoi mesh colors based on current color mode."""
        if not self.voronoi_animator or self.gui.current_timestep not in self.voronoi_animator.voronoi_meshes:
            return

        g = self.gui
        t = g.current_timestep

        # Update animator colors
        if g.color_mode == "scalar" and g.scalar_key:
            key = g.scalar_key
            if key in g.data and t < len(g.data[key]):
                scalar_field = g.data[key][t]
                vmin, vmax = g.scalar_ranges[key]
                self.voronoi_animator.update_scalar_colors(
                    t, scalar_field, vmin, vmax,
                    lambda val, vmin, vmax, _g=g, _key=key: _g._scalar_single_color(_key, val, vmin, vmax)
                )
        elif g.color_mode == "type" and "p_mask" in g.data:
            if t < len(g.data["p_mask"]):
                p_mask = g.data["p_mask"][t]
                type_color_map = {
                    ctype: np.array(g.cell_type_colors.get(ctype, (0.5, 0.5, 0.5, 1.0)), dtype=np.float32)
                    for ctype in g.unique_types
                }
                self.voronoi_animator.update_type_colors(t, p_mask, type_color_map)

        # Update transparency
        self.voronoi_animator.update_transparency(self.voronoi_transparency)

        # Re-render with new colors
        self.render()


    def render(self):
        """Render the current timestep according to gui settings."""
        g = self.gui

        if g.data is None:
            self._clear_cells()  # Ensure cells are cleared even if no data
            return
        t = g.current_timestep
        x = g.data["x"][t]
        p_mask = g.data["p_mask"][t] if "p_mask" in g.data else None

        if x is None or len(x) == 0:
            self._clear_cells()  # Ensure cells are cleared even if no cells
            return

        self._clear_cells()

        # Apply section filter
        x_f, pm_f, bool_mask = self._apply_section_filter(x, p_mask)
        if len(x_f) == 0:
            return

        # Choose color mode
        mode = g.color_mode  # "type" | "depth" | "scalar"

        # Check if we should show cell markers (skip if Voronoi enabled and toggle is off,
        # or if vector-only mode is enabled)
        show_cell_markers = not bool(getattr(g, "vector_mode", False))
        if show_cell_markers and self.voronoi_enabled:
            # Only check toggle if Voronoi UI was created (VORONOI_AVAILABLE)
            if hasattr(self, 'voronoi_show_cells_check'):
                show_cell_markers = self.voronoi_show_cells_check.isChecked()
            else:
                # If no toggle control exists, hide cells when Voronoi is enabled
                show_cell_markers = False

        # If not showing markers, ensure they're cleared
        if not show_cell_markers:
            for v in self._cell_scatters.values():
                try:
                    v.parent = None
                except:
                    pass
            self._cell_scatters.clear()
        elif show_cell_markers:
            if mode == "scalar":
                key = g.scalar_key
                raw = g.data[key][t]
                scalar_f = raw[bool_mask] if raw is not None else None
                if scalar_f is not None:
                    colors = g._scalar_colors(key, scalar_f)
                    self._add_scatter("scalar", x_f, colors)
                else:
                    self._add_scatter("default", x_f, "green")
            elif mode == "depth":
                self._render_depth(x_f)
            else:  # type or fallback
                if pm_f is not None and len(g.unique_types) > 0:
                    self._render_by_type(x_f, pm_f)
                else:
                    self._add_scatter("default", x_f, "green")

        # Polarity vectors
        if bool(getattr(g, "vector_mode", False)):
            p_all = g.data["p"][t] if ("p" in g.data and t < len(g.data["p"])) else None
            q_all = g.data["q"][t] if ("q" in g.data and t < len(g.data["q"])) else None
            if (p_all is not None and q_all is not None and
                len(p_all) == len(x) and len(q_all) == len(x)):
                self._render_dual_polarity_vectors(x_f, p_all[bool_mask], q_all[bool_mask])
        elif g.show_polarity:
            pkey = "p" if g.polarity_type == "p" else "q"
            pol_all = g.data.get(pkey, [None])[t]
            if pol_all is not None and len(pol_all) > 0:
                pol_f = pol_all[bool_mask]
                self._render_polarity(x_f, pol_f)

        # Voronoi tessellation
        if self.voronoi_enabled and VORONOI_AVAILABLE and self.voronoi_animator:
            # Apply current color mode to Voronoi meshes for this frame
            if t in self.voronoi_animator.voronoi_meshes:
                if mode == "scalar" and g.scalar_key:
                    key = g.scalar_key
                    if key in g.data and t < len(g.data[key]):
                        scalar_field = g.data[key][t]
                        vmin, vmax = g.scalar_ranges[key]
                        self.voronoi_animator.update_scalar_colors(
                            t, scalar_field, vmin, vmax,
                            lambda val, vmin, vmax, _g=g, _key=key: _g._scalar_single_color(_key, val, vmin, vmax)
                        )
                elif mode == "type" and "p_mask" in g.data and t < len(g.data["p_mask"]):
                    p_mask = g.data["p_mask"][t]
                    type_color_map = {
                        ctype: np.array(g.cell_type_colors.get(ctype, (0.5, 0.5, 0.5, 1.0)), dtype=np.float32)
                        for ctype in g.unique_types
                    }
                    self.voronoi_animator.update_type_colors(t, p_mask, type_color_map)

                # Update transparency
                self.voronoi_animator.update_transparency(self.voronoi_transparency)

            meshes = self.voronoi_animator.get_visible_meshes(t, bool_mask)

            if meshes:
                # Collect colors for each mesh
                colors = []
                for mesh in meshes:
                    if mesh.color is not None:
                        colors.append(mesh.color)
                    else:
                        colors.append(np.array([0.5, 0.5, 0.5, 1.0], dtype=np.float32))

                self._add_batched_voronoi_meshes(meshes, colors)
                # Apply camera-relative lighting
                self._apply_camera_relative_lighting()
            else:
                self._clear_voronoi_meshes()
        else:
            self._clear_voronoi_meshes()

        # Auto-set camera once
        if not self._camera_bounds_set:
            self.update_camera_bounds(x_f)
            self._camera_bounds_set = True

    # ------------------------------------------------------------------
    # Render helpers
    # ------------------------------------------------------------------
    def _add_scatter(self, key, positions, colors):
        s = visuals.Markers(scaling=True, alpha=1.0, spherical=True)
        s.set_data(positions, edge_width=0, face_color=colors, size=self.gui.cell_size)
        self.view.add(s)
        self._cell_scatters[key] = s

    def _render_by_type(self, x_f, pm_f):
        g = self.gui
        for ct in g.unique_types:
            if ct not in g.visible_types:
                continue
            mask = pm_f == ct
            if not mask.any():
                continue
            self._add_scatter(ct, x_f[mask], g.cell_type_colors.get(ct, (0.5, 0.5, 0.5, 1.0)))

    def _render_depth(self, x_f):
        try:
            cam_pos = self.view.camera.transform.map([0, 0, 0, 1])[:3]
        except Exception:
            cam_pos = np.array([0, 0, 10.0])
        dists = np.linalg.norm(x_f - cam_pos, axis=1)
        lo, hi = dists.min(), dists.max()
        norm = (dists - lo) / (hi - lo) if hi > lo else np.zeros_like(dists)
        colors = np.ones((len(x_f), 4), dtype=np.float32)
        colors[:, 0] = norm
        colors[:, 1] = 0.2
        colors[:, 2] = 1.0 - norm
        self._add_scatter("depth", x_f, colors)

    def _render_polarity(self, x_f, pol_f):
        norms = np.linalg.norm(pol_f, axis=1)
        valid = norms > 0
        if not valid.any():
            return
        pv = pol_f[valid] / norms[valid, np.newaxis]
        tips = x_f[valid] + 0.35 * pv
        color = "red" if self.gui.polarity_type == "p" else "blue"
        m = visuals.Markers(scaling=True, alpha=1.0, spherical=True)
        m.set_data(tips, edge_width=0, face_color=color, size=self.gui.cell_size)
        self.view.add(m)
        self._polarity_markers = m

    def _render_dual_polarity_vectors(self, x_f: np.ndarray,
                                     p_f: np.ndarray,
                                     q_f: np.ndarray):
        """Render ABP (p) and PCP (q) as two sets of vectors from each point."""
        if x_f is None or len(x_f) == 0:
            return

        def _segments(origins: np.ndarray, vecs: np.ndarray, length: float) -> np.ndarray:
            vecs = np.asarray(vecs, dtype=np.float32)
            origins = np.asarray(origins, dtype=np.float32)
            norms = np.linalg.norm(vecs, axis=1)
            valid = norms > 0
            if not valid.any():
                return np.empty((0, 3), dtype=np.float32)

            vhat = vecs[valid] / norms[valid, np.newaxis]
            a = origins[valid]
            b = a + length * vhat

            seg = np.empty((a.shape[0] * 2, 3), dtype=np.float32)
            seg[0::2] = a
            seg[1::2] = b
            return seg

        # Scale vectors relative to cell size for visual consistency.
        length = float(getattr(self.gui, "cell_size", 2.0)) * 0.9

        p_segs = _segments(x_f, p_f, length)
        q_segs = _segments(x_f, q_f, length)

        if len(p_segs) > 0:
            ln_p = visuals.Line(
                pos=p_segs,
                connect="segments",
                color=(1.0, 0.0, 0.0, 1.0),
                width=10,
            )
            self.view.add(ln_p)
            self._vector_visuals["abp"] = ln_p

        if len(q_segs) > 0:
            ln_q = visuals.Line(
                pos=q_segs,
                connect="segments",
                color=(0.0, 0.0, 1.0, 1.0),
                width=10,
            )
            self.view.add(ln_q)
            self._vector_visuals["pcp"] = ln_q

    # ------------------------------------------------------------------
    # Neighbour highlighting
    # ------------------------------------------------------------------
    def highlight_cells(self, sel_idx: int, nbr_indices):
        self.clear_highlights()
        g = self.gui
        if bool(getattr(g, "vector_mode", False)):
            self.canvas.update()
            return
        x = g.data["x"][g.current_timestep]
        # Selected cell — white, slightly larger
        s1 = visuals.Markers(scaling=True, alpha=1.0, spherical=True)
        s1.set_data(x[sel_idx:sel_idx + 1], edge_width=0, face_color="white",
                    size=g.cell_size * 1.15)
        self.view.add(s1)
        self._highlights["selected"] = s1
        if len(nbr_indices) > 0:
            s2 = visuals.Markers(scaling=True, alpha=1.0, spherical=True)
            s2.set_data(x[nbr_indices], edge_width=0, face_color="yellow",
                        size=g.cell_size * 1.15)
            self.view.add(s2)
            self._highlights["neighbors"] = s2
        self.canvas.update()

    # ------------------------------------------------------------------
    # Right-click cell selection
    # ------------------------------------------------------------------
    def _on_right_click(self):
        if self._ghost_pos is None:
            return
        g = self.gui
        x = g.data["x"][g.current_timestep]
        dists = np.linalg.norm(x - self._ghost_pos, axis=1)
        idx = int(np.argmin(dists))
        if dists[idx] <= self._ghost_sphere_radius:
            if hasattr(g, "on_cell_picked"):
                g.on_cell_picked(idx)
            else:
                g.on_cell_selected(idx)

    # ------------------------------------------------------------------
    # Image / video export
    # ------------------------------------------------------------------
    def export_image(self, filename: str) -> bool:
        try:
            img = self.canvas.render()
            vispy.io.write_png(filename, img)
            return True
        except Exception as e:
            print(f"Image export error: {e}")
            return False


# ---------------------------------------------------------------------------
# Main GUI window
# ---------------------------------------------------------------------------
class DataVizGUI(QMainWindow):
    """Tabbed data visualization GUI for dict-of-lists simulation data."""

    # Public state read by VisPy3DWidget
    data: dict | None           # loaded data dict
    current_timestep: int
    color_mode: str             # "type" | "depth" | "scalar"
    scalar_key: str             # one of scalar_keys
    scalar_ranges: dict         # {key: (min, max)}
    cell_size: float
    show_polarity: bool
    polarity_type: str          # "p" or "q"
    vector_mode: bool
    bisection_enabled: bool
    bisection_plane: str
    bisection_position: float
    cross_section_mode: bool
    cross_section_width: float
    unique_types: list
    visible_types: set
    cell_type_colors: dict
    neighbor_search_enabled: bool
    cell_pick_enabled: bool
    scalar_keys: list           # dynamically detected scalar keys

    def __init__(self):
        super().__init__()
        self._init_state()
        self._build_ui()
        self.setWindowTitle("Data Visualization v2")
        self.setGeometry(100, 100, 1400, 900)
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)

    # ------------------------------------------------------------------
    # State initialisation
    # ------------------------------------------------------------------
    def _init_state(self):
        self.data = None
        self.data_folder = None
        self.sim_dict = None
        self.current_timestep = 0
        self.max_timesteps = 1
        self.playing = False
        self.play_speed = 100

        self.color_mode = "type"
        self.scalar_key = ""
        self.scalar_keys = []
        self.scalar_ranges = {}
        self.cell_size = 2.0
        self.show_polarity = False
        self.polarity_type = "p"
        self.vector_mode = False

        self.bisection_enabled = False
        self.bisection_plane = "XY"
        self.bisection_position = 0.0
        self.cross_section_mode = False
        self.cross_section_width = 2.0

        self.unique_types = []
        self.visible_types = set()
        self.cell_type_colors = {}

        self.neighbor_search_enabled = False
        self.neighbor_data = {}

        # Generic picking (right-click through ghost sphere)
        self.cell_pick_enabled = False

        # Voronoi visualization state
        self.voronoi_enabled = False
        self.voronoi_transparency = 0.5
        self.voronoi_force_recalc = False
        self.voronoi_progress = None
        self.selected_cell_idx = None
        self._scalar_log_eps = 1e-12
        self.show_scalar_histogram = False

    def _is_logspace_scalar(self, key: str) -> bool:
        """Keys containing 'gamma' are interpreted as log-space variables."""
        return isinstance(key, str) and ("gamma" in key.lower())

    def _transform_scalar_values(self, key: str, values: np.ndarray) -> np.ndarray:
        """Transform scalar values for color mapping."""
        arr = np.asarray(values, dtype=np.float32)
        if self._is_logspace_scalar(key):
            arr = np.log(np.clip(arr, self._scalar_log_eps, None))
        return arr

    def _scalar_vcenter(self, key: str) -> float | None:
        """Return color midpoint in transformed space."""
        if self._is_logspace_scalar(key):
            return 0.0  # log(1.0)
        return None

    def _scalar_colors(self, key: str, values: np.ndarray) -> np.ndarray:
        """Map scalar values to RGBA with key-specific transforms."""
        vals_t = self._transform_scalar_values(key, values)
        if key in self.scalar_ranges:
            vmin, vmax = self.scalar_ranges[key]
        else:
            vmin, vmax = float(np.min(vals_t)), float(np.max(vals_t))
        return scalar_to_rgba(vals_t, vmin, vmax, self._scalar_vcenter(key))

    def _scalar_single_color(self, key: str, value: float,
                             vmin: float, vmax: float) -> np.ndarray:
        """Single-value color mapping wrapper (used by Voronoi animator)."""
        val_t = self._transform_scalar_values(key, np.array([value], dtype=np.float32))
        return scalar_to_rgba(val_t, vmin, vmax, self._scalar_vcenter(key))[0]

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(6, 6, 6, 6)
        root_layout.setSpacing(0)

        # Splitter: left tabs | right canvas — fills the whole window
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: tabs widget
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)

        # "Load Dataset" sits in the tab bar corner — no separate toolbar
        load_btn = QPushButton("Load")
        load_btn.setObjectName("load")
        load_btn.setFixedHeight(28)
        load_btn.clicked.connect(self._load_dataset_dialog)
        self.tabs.setCornerWidget(load_btn, Qt.Corner.TopRightCorner)

        self._build_playback_tab()
        self._build_viz_tab()
        self._build_section_tab()
        self._build_curate_tab()
        self._build_tools_tab()

        splitter.addWidget(self.tabs)

        # Right: canvas
        self.canvas_widget = VisPy3DWidget(self)
        splitter.addWidget(self.canvas_widget)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([350, 1050])

        root_layout.addWidget(splitter)

        # Disable Voronoi controls until data is loaded
        if VORONOI_AVAILABLE and hasattr(self, 'voronoi_check'):
            self.voronoi_check.setEnabled(False)
            self.voronoi_transparency_slider.setEnabled(False)

    # ------------------------------------------------------------------
    # Tab 1: Playback
    # ------------------------------------------------------------------
    def _build_playback_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Timestep
        ts_group = QGroupBox("Timestep")
        ts_layout = QVBoxLayout(ts_group)

        row1 = QHBoxLayout()
        self.ts_slider = QSlider(Qt.Orientation.Horizontal)
        self.ts_slider.setMinimum(0)
        self.ts_slider.setMaximum(0)
        self.ts_spinbox = QSpinBox()
        self.ts_spinbox.setMinimum(0)
        self.ts_spinbox.setMaximum(0)
        row1.addWidget(self.ts_slider, stretch=3)
        row1.addWidget(self.ts_spinbox, stretch=1)
        ts_layout.addLayout(row1)

        self.ts_slider.valueChanged.connect(lambda v: self._go_to_timestep(v))
        self.ts_spinbox.valueChanged.connect(lambda v: self._go_to_timestep(v))

        layout.addWidget(ts_group)

        # Controls
        ctrl_group = QGroupBox("Controls")
        ctrl_layout = QVBoxLayout(ctrl_group)

        btn_row = QHBoxLayout()
        prev_btn = QPushButton("|<")
        prev_btn.setFixedWidth(36)
        prev_btn.clicked.connect(self.prev_frame)
        self.play_btn = QPushButton("Play")
        self.play_btn.setCheckable(True)
        self.play_btn.clicked.connect(self.toggle_play)
        next_btn = QPushButton(">|")
        next_btn.setFixedWidth(36)
        next_btn.clicked.connect(self.next_frame)
        btn_row.addWidget(prev_btn)
        btn_row.addWidget(self.play_btn, stretch=1)
        btn_row.addWidget(next_btn)
        ctrl_layout.addLayout(btn_row)

        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(20)
        self.speed_slider.setValue(5)
        self.speed_slider.valueChanged.connect(self._on_speed_changed)
        speed_row.addWidget(self.speed_slider)
        ctrl_layout.addLayout(speed_row)

        layout.addWidget(ctrl_group)
        layout.addStretch()
        self.tabs.addTab(w, "Playback")

    # ------------------------------------------------------------------
    # Tab 2: Visualization
    # ------------------------------------------------------------------
    def _build_viz_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Cell size
        size_group = QGroupBox("Cell Size")
        size_layout = QHBoxLayout(size_group)
        self.size_slider = QSlider(Qt.Orientation.Horizontal)
        self.size_slider.setMinimum(1)
        self.size_slider.setMaximum(8)
        self.size_slider.setValue(2)
        self.size_label = QLabel("2")
        self.size_slider.valueChanged.connect(self._on_size_changed)
        size_layout.addWidget(self.size_slider)
        size_layout.addWidget(self.size_label)
        layout.addWidget(size_group)

        # Color mode
        color_group = QGroupBox("Color Mode")
        color_layout = QVBoxLayout(color_group)
        self._color_btn_group = QButtonGroup(self)
        for text, val in [("Cell Type", "type"), ("Depth", "depth"), ("Scalar", "scalar")]:
            rb = QRadioButton(text)
            rb.setProperty("color_mode", val)
            self._color_btn_group.addButton(rb)
            color_layout.addWidget(rb)
            if val == "type":
                rb.setChecked(True)
        self._color_btn_group.buttonClicked.connect(self._on_color_mode_changed)

        # Scalar key dropdown (only relevant when scalar mode active)
        scalar_row = QHBoxLayout()
        scalar_row.addWidget(QLabel("  Scalar field:"))
        self.scalar_combo = QComboBox()
        self.scalar_combo.currentTextChanged.connect(self._on_scalar_key_changed)
        scalar_row.addWidget(self.scalar_combo)
        color_layout.addLayout(scalar_row)
        layout.addWidget(color_group)

        # Colorbar
        self.colorbar_group = QGroupBox("Color Scale")
        cb_vlay = QVBoxLayout(self.colorbar_group)
        cb_vlay.setContentsMargins(4, 4, 4, 4)
        self.colorbar_label = QLabel()
        self.colorbar_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.colorbar_label.setMinimumHeight(120)
        cb_vlay.addWidget(self.colorbar_label)

        hist_ctrl_row = QHBoxLayout()
        hist_ctrl_row.addWidget(QLabel("Histogram bins:"))
        self.hist_bins_spin = QSpinBox()
        self.hist_bins_spin.setRange(5, 300)
        self.hist_bins_spin.setValue(30)
        self.hist_bins_spin.valueChanged.connect(self._on_hist_bins_changed)
        hist_ctrl_row.addWidget(self.hist_bins_spin)
        self.hist_btn = QPushButton("Show Histogram")
        self.hist_btn.clicked.connect(self._on_show_histogram_clicked)
        hist_ctrl_row.addWidget(self.hist_btn)
        hist_ctrl_row.addStretch()
        cb_vlay.addLayout(hist_ctrl_row)

        self.histogram_label = QLabel()
        self.histogram_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.histogram_label.setMinimumHeight(240)
        self.histogram_label.setVisible(False)
        cb_vlay.addWidget(self.histogram_label)

        self.colorbar_group.setVisible(False)
        layout.addWidget(self.colorbar_group)

        # Polarity
        pol_group = QGroupBox("Polarity Vectors")
        pol_layout = QHBoxLayout(pol_group)
        self.pol_checkbox = QCheckBox("Show")
        self.pol_checkbox.stateChanged.connect(self._on_polarity_changed)
        self.pol_combo = QComboBox()
        self.pol_combo.addItems(["Apicobasal (p)", "Planar (q)"])
        self.pol_combo.currentTextChanged.connect(self._on_polarity_type_changed)
        self.vector_mode_check = QCheckBox("Vector mode (p+q)")
        self.vector_mode_check.setToolTip("Hide cell spheres and render ABP (p, red) + PCP (q, blue) vectors")
        self.vector_mode_check.stateChanged.connect(self._on_vector_mode_changed)
        pol_layout.addWidget(self.pol_checkbox)
        pol_layout.addWidget(self.pol_combo)
        pol_layout.addWidget(self.vector_mode_check)
        layout.addWidget(pol_group)

        # Display
        disp_group = QGroupBox("Display")
        disp_layout = QFormLayout(disp_group)
        self.axes_checkbox = QCheckBox()
        self.axes_checkbox.stateChanged.connect(self._on_axes_changed)
        disp_layout.addRow("Show Axes:", self.axes_checkbox)
        self.bg_combo = QComboBox()
        self.bg_combo.addItems(["Black", "White", "Gray"])
        self.bg_combo.currentTextChanged.connect(self._on_bg_changed)
        disp_layout.addRow("Background:", self.bg_combo)
        layout.addWidget(disp_group)

        # Voronoi visualization (if available)
        if VORONOI_AVAILABLE:
            vor_group = QGroupBox("Voronoi Tessellation")
            vor_layout = QVBoxLayout(vor_group)

            # Enable/disable toggle
            self.voronoi_check = QCheckBox("Enable Voronoi Visualization")
            self.voronoi_check.stateChanged.connect(self._on_voronoi_enabled)
            vor_layout.addWidget(self.voronoi_check)

            # Transparency slider
            trans_row = QHBoxLayout()
            trans_row.addWidget(QLabel("Transparency:"))
            self.voronoi_transparency_slider = QSlider(Qt.Orientation.Horizontal)
            self.voronoi_transparency_slider.setMinimum(0)
            self.voronoi_transparency_slider.setMaximum(100)
            self.voronoi_transparency_slider.setValue(50)
            self.voronoi_transparency_label = QLabel("0.50")
            self.voronoi_transparency_slider.valueChanged.connect(self._on_voronoi_transparency)
            trans_row.addWidget(self.voronoi_transparency_slider)
            trans_row.addWidget(self.voronoi_transparency_label)
            vor_layout.addLayout(trans_row)

            # Show cell markers toggle
            self.voronoi_show_cells_check = QCheckBox("Show Cell Markers")
            self.voronoi_show_cells_check.setChecked(True)
            self.voronoi_show_cells_check.stateChanged.connect(self._on_voronoi_show_cells)
            vor_layout.addWidget(self.voronoi_show_cells_check)

            # Show edges toggle
            self.voronoi_edges_check = QCheckBox("Show Voronoi Edges")
            self.voronoi_edges_check.setChecked(False)
            self.voronoi_edges_check.stateChanged.connect(self._on_voronoi_edges_toggled)
            vor_layout.addWidget(self.voronoi_edges_check)

            # Edge color selection
            edge_color_row = QHBoxLayout()
            edge_color_row.addWidget(QLabel("Edge Color:"))
            self.voronoi_edge_color_combo = QComboBox()
            self.voronoi_edge_color_combo.addItems(["White", "Black", "Red", "Green", "Blue", "Yellow", "Cyan", "Magenta"])
            self.voronoi_edge_color_combo.currentTextChanged.connect(self._on_voronoi_edge_color_changed)
            edge_color_row.addWidget(self.voronoi_edge_color_combo)
            edge_color_row.addStretch()
            vor_layout.addLayout(edge_color_row)

            # Force recalculation toggle
            self.voronoi_force_recalc_check = QCheckBox("Force Recalculation (next enable)")
            self.voronoi_force_recalc_check.setChecked(False)
            vor_layout.addWidget(self.voronoi_force_recalc_check)

            layout.addWidget(vor_group)

        # Cell types list
        self.type_group = QGroupBox("Cell Types")
        type_layout = QVBoxLayout(self.type_group)
        self.type_list = QListWidget()
        self.type_list.setMaximumHeight(160)
        type_layout.addWidget(self.type_list)
        layout.addWidget(self.type_group)

        layout.addStretch()
        scroll.setWidget(inner)
        self.tabs.addTab(scroll, "Visualization")


    # ------------------------------------------------------------------
    # Tab 3: Data Sectioning
    # ------------------------------------------------------------------
    def _build_section_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.bisect_check = QCheckBox("Enable Bisection")
        self.bisect_check.stateChanged.connect(self._on_bisection_changed)
        layout.addWidget(self.bisect_check)

        plane_group = QGroupBox("Plane")
        plane_layout = QHBoxLayout(plane_group)
        self._plane_btn_group = QButtonGroup(self)
        for plane in ["XY", "XZ", "YZ", "Camera Orthogonal"]:
            rb = QRadioButton(plane)
            rb.setProperty("plane", plane)
            self._plane_btn_group.addButton(rb)
            plane_layout.addWidget(rb)
            if plane == "XY":
                rb.setChecked(True)
        self._plane_btn_group.buttonClicked.connect(self._on_plane_changed)
        layout.addWidget(plane_group)

        # Position
        pos_group = QGroupBox("Position")
        pos_layout = QHBoxLayout(pos_group)
        self.bisect_slider = QSlider(Qt.Orientation.Horizontal)
        self.bisect_slider.setMinimum(-100)
        self.bisect_slider.setMaximum(100)
        self.bisect_slider.setValue(0)
        self.bisect_spinbox = QDoubleSpinBox()
        self.bisect_spinbox.setRange(-1.0, 1.0)
        self.bisect_spinbox.setSingleStep(0.01)
        self.bisect_spinbox.setDecimals(2)
        self.bisect_slider.valueChanged.connect(
            lambda v: self._sync_slider_spinbox(self.bisect_slider, self.bisect_spinbox, v / 100.0, "bisection_position")
        )
        self.bisect_spinbox.valueChanged.connect(
            lambda v: self._sync_slider_spinbox(self.bisect_slider, self.bisect_spinbox, v, "bisection_position")
        )
        pos_layout.addWidget(self.bisect_slider, stretch=3)
        pos_layout.addWidget(self.bisect_spinbox, stretch=1)
        layout.addWidget(pos_group)

        # Cross-section
        self.cs_check = QCheckBox("Cross-section (thin slice)")
        self.cs_check.stateChanged.connect(self._on_cross_section_changed)
        layout.addWidget(self.cs_check)

        width_group = QGroupBox("Slice Width")
        width_layout = QHBoxLayout(width_group)
        self.width_slider = QSlider(Qt.Orientation.Horizontal)
        self.width_slider.setMinimum(1)
        self.width_slider.setMaximum(200)
        self.width_slider.setValue(20)
        self.width_spinbox = QDoubleSpinBox()
        self.width_spinbox.setRange(0.1, 20.0)
        self.width_spinbox.setSingleStep(0.1)
        self.width_spinbox.setDecimals(1)
        self.width_spinbox.setValue(2.0)
        self.width_slider.valueChanged.connect(
            lambda v: self._sync_slider_spinbox(self.width_slider, self.width_spinbox, v / 10.0, "cross_section_width")
        )
        self.width_spinbox.valueChanged.connect(
            lambda v: self._sync_slider_spinbox(self.width_slider, self.width_spinbox, v, "cross_section_width")
        )
        width_layout.addWidget(self.width_slider, stretch=3)
        width_layout.addWidget(self.width_spinbox, stretch=1)
        layout.addWidget(width_group)

        layout.addStretch()
        self.tabs.addTab(w, "Sectioning")

    # ------------------------------------------------------------------
    # Tab 4: Tools (neighbour search + export + data info)
    # ------------------------------------------------------------------
    def _build_tools_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Neighbour search
        nbr_group = QGroupBox("Neighbour Search")
        nbr_layout = QVBoxLayout(nbr_group)

        param_row = QHBoxLayout()
        param_row.addWidget(QLabel("See-through:"))
        self.seethru_spin = QSpinBox()
        self.seethru_spin.setRange(0, 10)
        self.seethru_spin.setValue(2)
        param_row.addWidget(self.seethru_spin)
        param_row.addWidget(QLabel("Max k:"))
        self.max_k_spin = QSpinBox()
        self.max_k_spin.setRange(10, 200)
        self.max_k_spin.setValue(50)
        param_row.addWidget(self.max_k_spin)
        nbr_layout.addLayout(param_row)

        self.search_btn = QPushButton("Search Neighbours")
        self.search_btn.clicked.connect(self._on_search_clicked)
        nbr_layout.addWidget(self.search_btn)

        self.nbr_stats = QTextBrowser()
        self.nbr_stats.setMaximumHeight(120)
        self.nbr_stats.setVisible(False)
        nbr_layout.addWidget(self.nbr_stats)

        self.show_only_check = QCheckBox("Show only selected neighbours")
        self.show_only_check.setVisible(False)
        self.show_only_check.stateChanged.connect(self._on_show_only_changed)
        nbr_layout.addWidget(self.show_only_check)

        self.clear_nbr_btn = QPushButton("Clear Selection")
        self.clear_nbr_btn.setVisible(False)
        self.clear_nbr_btn.clicked.connect(self.clear_neighbor_selection)
        nbr_layout.addWidget(self.clear_nbr_btn)

        layout.addWidget(nbr_group)

        # Export
        exp_group = QGroupBox("Export")
        exp_layout = QVBoxLayout(exp_group)

        img_btn = QPushButton("Export Image (PNG)")
        img_btn.setObjectName("success")
        img_btn.clicked.connect(self._export_image)
        exp_layout.addWidget(img_btn)

        vid_btn = QPushButton("Export Video (MP4)")
        vid_btn.setObjectName("video")
        vid_btn.clicked.connect(self._export_video)
        exp_layout.addWidget(vid_btn)

        layout.addWidget(exp_group)

        # Data info
        info_group = QGroupBox("Dataset Info")
        info_layout = QVBoxLayout(info_group)
        self.info_browser = QTextBrowser()
        self.info_browser.setMaximumHeight(180)
        info_layout.addWidget(self.info_browser)
        layout.addWidget(info_group)

        layout.addStretch()
        scroll.setWidget(inner)
        self.tabs.addTab(scroll, "Tools")

    # ------------------------------------------------------------------
    # Tab 5: Data Curating (edit p_mask + export single frame)
    # ------------------------------------------------------------------
    def _build_curate_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        pick_group = QGroupBox("Picking")
        pick_layout = QVBoxLayout(pick_group)
        self.cur_pick_check = QCheckBox("Enable click-pick (right-click)")
        self.cur_pick_check.setToolTip(
            "Shows a cyan ghost sphere in front of the camera.\n"
            "Right-click to pick the closest cell inside it.\n"
            "Hold Shift + mouse wheel to move the sphere in/out."
        )
        self.cur_pick_check.stateChanged.connect(self._on_cur_pick_toggled)
        pick_layout.addWidget(self.cur_pick_check)

        self.cur_selected_info = QTextBrowser()
        self.cur_selected_info.setMaximumHeight(90)
        self.cur_selected_info.setPlainText("Selected cell: (none)")
        pick_layout.addWidget(self.cur_selected_info)

        layout.addWidget(pick_group)

        wipe_group = QGroupBox("Wipe / Reset p_mask (current timestep)")
        wipe_layout = QHBoxLayout(wipe_group)
        wipe_layout.addWidget(QLabel("Set all p_mask to:"))
        self.cur_wipe_value = QSpinBox()
        self.cur_wipe_value.setRange(-10_000_000, 10_000_000)
        self.cur_wipe_value.setValue(0)
        wipe_layout.addWidget(self.cur_wipe_value)
        self.cur_wipe_btn = QPushButton("Wipe")
        self.cur_wipe_btn.clicked.connect(self._cur_wipe_pmask)
        wipe_layout.addWidget(self.cur_wipe_btn)
        layout.addWidget(wipe_group)

        range_group = QGroupBox("Assign by coordinate range (current timestep)")
        range_layout = QFormLayout(range_group)

        self.cur_axis_combo = QComboBox()
        self.cur_axis_combo.addItems(["X", "Y", "Z"])
        range_layout.addRow("Axis:", self.cur_axis_combo)

        mm_row = QHBoxLayout()
        self.cur_min_spin = QDoubleSpinBox()
        self.cur_min_spin.setDecimals(4)
        self.cur_min_spin.setRange(-1e9, 1e9)
        self.cur_min_spin.setValue(-1.0)
        self.cur_max_spin = QDoubleSpinBox()
        self.cur_max_spin.setDecimals(4)
        self.cur_max_spin.setRange(-1e9, 1e9)
        self.cur_max_spin.setValue(1.0)
        mm_row.addWidget(QLabel("Min"))
        mm_row.addWidget(self.cur_min_spin)
        mm_row.addSpacing(8)
        mm_row.addWidget(QLabel("Max"))
        mm_row.addWidget(self.cur_max_spin)
        mm_row.addStretch()
        mm_wrap = QWidget()
        mm_wrap.setLayout(mm_row)
        range_layout.addRow("Range:", mm_wrap)

        self.cur_range_type = QSpinBox()
        self.cur_range_type.setRange(-10_000_000, 10_000_000)
        self.cur_range_type.setValue(1)
        range_layout.addRow("Set p_mask to:", self.cur_range_type)

        self.cur_apply_range_btn = QPushButton("Apply Range")
        self.cur_apply_range_btn.clicked.connect(self._cur_apply_range)
        range_layout.addRow(self.cur_apply_range_btn)
        layout.addWidget(range_group)

        sel_group = QGroupBox("Assign selected cell (current timestep)")
        sel_layout = QHBoxLayout(sel_group)
        sel_layout.addWidget(QLabel("Set selected p_mask to:"))
        self.cur_sel_type = QSpinBox()
        self.cur_sel_type.setRange(-10_000_000, 10_000_000)
        self.cur_sel_type.setValue(1)
        sel_layout.addWidget(self.cur_sel_type)
        self.cur_apply_sel_btn = QPushButton("Apply to Selected")
        self.cur_apply_sel_btn.clicked.connect(self._cur_apply_selected)
        sel_layout.addWidget(self.cur_apply_sel_btn)
        self.cur_clear_sel_btn = QPushButton("Clear Selection")
        self.cur_clear_sel_btn.clicked.connect(self._cur_clear_selection)
        sel_layout.addWidget(self.cur_clear_sel_btn)
        layout.addWidget(sel_group)

        exp_group = QGroupBox("Export")
        exp_layout = QVBoxLayout(exp_group)
        self.cur_export_btn = QPushButton("Export Curated Frame (pkl)")
        self.cur_export_btn.setObjectName("success")
        self.cur_export_btn.clicked.connect(self._cur_export_frame)
        exp_layout.addWidget(self.cur_export_btn)
        layout.addWidget(exp_group)

        layout.addStretch()
        scroll.setWidget(inner)
        self.tabs.addTab(scroll, "Data Curating")

    def _on_cur_pick_toggled(self, state: int):
        self.cell_pick_enabled = bool(state)

    def on_cell_picked(self, cell_idx: int):
        """Handle a right-click pick from the VisPy canvas."""
        if self.data is None:
            return
        try:
            self.selected_cell_idx = int(cell_idx)
        except Exception:
            return

        # Highlight at least the selected cell
        try:
            self.canvas_widget.highlight_cells(self.selected_cell_idx, [])
        except Exception:
            pass

        # If neighbour data is available, keep the existing neighbour-selection behaviour
        if self.current_timestep in self.neighbor_data:
            self.on_cell_selected(self.selected_cell_idx)

        self._update_curate_selected_info()

    def _update_curate_selected_info(self):
        if not hasattr(self, "cur_selected_info"):
            return
        if self.data is None:
            self.cur_selected_info.setPlainText("Selected cell: (none)")
            return

        t = self.current_timestep
        idx = self.selected_cell_idx
        if idx is None:
            self.cur_selected_info.setPlainText("Selected cell: (none)")
            return

        x = self.data.get("x", [None])[t]
        pm = self.data.get("p_mask", [None])[t] if "p_mask" in self.data and t < len(self.data["p_mask"]) else None
        if x is None or idx < 0 or idx >= len(x):
            self.cur_selected_info.setPlainText("Selected cell: (none)")
            return

        pos = x[idx]
        typ = int(pm[idx]) if pm is not None and len(pm) == len(x) else None
        msg = f"Selected cell: {idx}\n"
        msg += f"Position: [{pos[0]:.4g}, {pos[1]:.4g}, {pos[2]:.4g}]\n"
        msg += f"Current p_mask: {typ}" if typ is not None else "Current p_mask: (missing)"
        self.cur_selected_info.setPlainText(msg)

    def _ensure_pmask_for_current_timestep(self) -> np.ndarray | None:
        if self.data is None or "x" not in self.data:
            return None
        t = self.current_timestep
        x = self.data["x"][t]
        if x is None:
            return None

        if "p_mask" not in self.data:
            self.data["p_mask"] = [None] * len(self.data["x"])

        pm = self.data["p_mask"][t] if t < len(self.data["p_mask"]) else None
        if pm is None or len(pm) != len(x):
            pm = np.zeros(len(x), dtype=np.int32)
        else:
            pm = np.asarray(pm, dtype=np.int32).copy()

        self.data["p_mask"][t] = pm
        return pm

    def _after_pmask_edit(self):
        """Refresh derived type state + UI + render after p_mask edits."""
        if self.data is None:
            return
        t = self.current_timestep
        pm = self.data.get("p_mask", [None])[t]
        if pm is None:
            return

        try:
            unique = sorted(set(int(v) for v in np.unique(pm)))
        except Exception:
            unique = []
        self.unique_types = unique
        self.visible_types = set(unique)
        self._generate_type_colors()
        if hasattr(self, "type_list"):
            self._populate_type_list()

        self._update_curate_selected_info()
        self._refresh()

    def _cur_wipe_pmask(self):
        if self.data is None:
            QMessageBox.warning(self, "No Data", "Load data first.")
            return
        pm = self._ensure_pmask_for_current_timestep()
        if pm is None:
            QMessageBox.warning(self, "Missing Data", "No x/p_mask available for this timestep.")
            return

        val = int(self.cur_wipe_value.value())
        pm[:] = val
        self.data["p_mask"][self.current_timestep] = pm
        self._after_pmask_edit()

    def _cur_apply_range(self):
        if self.data is None:
            QMessageBox.warning(self, "No Data", "Load data first.")
            return
        t = self.current_timestep
        x = self.data.get("x", [None])[t]
        if x is None:
            QMessageBox.warning(self, "Missing Data", "No x available for this timestep.")
            return

        pm = self._ensure_pmask_for_current_timestep()
        if pm is None:
            QMessageBox.warning(self, "Missing Data", "No p_mask available for this timestep.")
            return

        axis = {"X": 0, "Y": 1, "Z": 2}[self.cur_axis_combo.currentText()]
        lo = float(self.cur_min_spin.value())
        hi = float(self.cur_max_spin.value())
        if hi < lo:
            lo, hi = hi, lo

        mask = (x[:, axis] >= lo) & (x[:, axis] <= hi)
        pm[mask] = int(self.cur_range_type.value())
        self.data["p_mask"][t] = pm
        try:
            self.canvas_widget.show_range_planes(axis, lo, hi, duration_s=3.0, alpha=0.5)
        except Exception:
            pass
        self._after_pmask_edit()

    def _cur_apply_selected(self):
        if self.data is None:
            QMessageBox.warning(self, "No Data", "Load data first.")
            return
        if self.selected_cell_idx is None:
            QMessageBox.warning(self, "No Selection", "Pick a cell first (enable picking + right-click).")
            return

        t = self.current_timestep
        x = self.data.get("x", [None])[t]
        if x is None:
            QMessageBox.warning(self, "Missing Data", "No x available for this timestep.")
            return

        pm = self._ensure_pmask_for_current_timestep()
        if pm is None:
            QMessageBox.warning(self, "Missing Data", "No p_mask available for this timestep.")
            return

        idx = int(self.selected_cell_idx)
        if idx < 0 or idx >= len(pm):
            QMessageBox.warning(self, "Out of Range", "Selected cell index out of range.")
            return
        pm[idx] = int(self.cur_sel_type.value())
        self.data["p_mask"][t] = pm
        self._after_pmask_edit()

    def _cur_clear_selection(self):
        self.selected_cell_idx = None
        try:
            self.canvas_widget.clear_highlights()
        except Exception:
            pass
        self._update_curate_selected_info()

    def _cur_export_frame(self):
        if self.data is None:
            QMessageBox.warning(self, "No Data", "Load data first.")
            return
        t = self.current_timestep

        missing = []
        for k in ("x", "p_mask", "p", "q"):
            if k not in self.data or t >= len(self.data[k]) or self.data[k][t] is None:
                missing.append(k)
        if missing:
            QMessageBox.warning(self, "Missing Data", f"Cannot export; missing at timestep {t}: {', '.join(missing)}")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Curated Frame",
            f"curated_frame_t{t}.pkl",
            "Pickle files (*.pkl)",
        )
        if not filename:
            return

        frame = {
            "x": np.asarray(self.data["x"][t]).copy(),
            "p_mask": np.asarray(self.data["p_mask"][t], dtype=np.int32).copy(),
            "p": np.asarray(self.data["p"][t]).copy(),
            "q": np.asarray(self.data["q"][t]).copy(),
        }
        try:
            with open(filename, "wb") as f:
                pickle.dump(frame, f, protocol=pickle.HIGHEST_PROTOCOL)
            QMessageBox.information(self, "Exported", f"Saved curated frame:\n{filename}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export:\n{e}")

    # ------------------------------------------------------------------
    # Slider / spinbox sync helper
    # ------------------------------------------------------------------
    def _sync_slider_spinbox(self, slider, spinbox, value, attr: str):
        """Sync slider (integer *100) and spinbox (float), update state attr, re-render."""
        # Update state
        setattr(self, attr, float(value))

        # Sync the OTHER widget without triggering recursion
        if isinstance(value, float):
            # Called from spinbox — update slider
            int_val = int(round(value * 100))
            if slider.value() != int_val:
                slider.blockSignals(True)
                slider.setValue(int_val)
                slider.blockSignals(False)
        else:
            # Called from slider — update spinbox
            float_val = value / 100.0
            if abs(spinbox.value() - float_val) > 1e-6:
                spinbox.blockSignals(True)
                spinbox.setValue(float_val)
                spinbox.blockSignals(False)

        self._refresh()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def _load_dataset_dialog(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Simulation Data Folder")
        if folder:
            self._load_data(folder)

    def _load_data(self, folder: str):
        # Accept data.pkl; also fall back to data.npy for backwards compat
        for fname in ("data.pkl", "data.npy"):
            data_path = os.path.join(folder, fname)
            if os.path.exists(data_path):
                break
        else:
            QMessageBox.warning(self, "Error", f"No data.pkl found in:\n{folder}")
            return

        try:
            with open(data_path, "rb") as f:
                raw = pickle.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load data:\n{e}")
            return

        if not isinstance(raw, dict):
            QMessageBox.critical(self, "Error",
                "Expected a dict-of-lists data format.\n"
                "Got: " + str(type(raw)))
            return

        required = ["x", "p_mask"]
        missing = [k for k in required if k not in raw]
        if missing:
            QMessageBox.warning(self, "Warning",
                f"Data is missing expected keys: {missing}\n"
                "Continuing with available data.")

        self.data = raw
        self.data_folder = folder
        self.max_timesteps = len(raw["x"])
        self.current_timestep = 0

        # Detect scalar keys and compute their ranges
        self.scalar_keys = self._detect_scalar_keys(raw)
        self.scalar_ranges = {}
        for key in self.scalar_keys:
            if key in raw and raw[key]:
                try:
                    vals_list = [v for v in raw[key] if v is not None]

                    # Special-case: the last `energy` frame is a dummy of all zeros.
                    # Exclude it from global range so it doesn't crush contrast.
                    if key == "energy" and len(vals_list) > 1:
                        last = np.asarray(vals_list[-1], dtype=np.float32)
                        if last.size > 0 and np.all(np.isfinite(last)) and np.allclose(last, 0.0):
                            vals_list = vals_list[:-1]

                    all_vals = np.concatenate(vals_list).astype(np.float32)
                    if self._is_logspace_scalar(key):
                        pos = all_vals[all_vals > 0]
                        if len(pos) == 0:
                            print(f"[scalar range] Skipping {key}: no positive values for log scale")
                            continue
                        log_vals = np.log(np.clip(pos, self._scalar_log_eps, None))
                        lo = float(np.min(log_vals))
                        hi = float(np.max(log_vals))
                        # Keep gamma=1 (log=0) at the midpoint by ensuring center is in range.
                        lo = min(lo, 0.0)
                        hi = max(hi, 0.0)
                        self.scalar_ranges[key] = (lo, hi)
                    else:
                        self.scalar_ranges[key] = (float(np.min(all_vals)), float(np.max(all_vals)))
                except Exception as e:
                    print(f"Could not compute range for {key}: {e}")

        # Cell types
        if "p_mask" in raw and raw["p_mask"]:
            all_types = np.concatenate([v for v in raw["p_mask"] if v is not None])
            self.unique_types = sorted(set(all_types.tolist()))
            self._generate_type_colors()
            self.visible_types = set(self.unique_types)

        # Reset neighbour search
        self.neighbor_data.clear()
        self.neighbor_search_enabled = False
        self.selected_cell_idx = None

        # Reset camera
        self.canvas_widget._camera_bounds_set = False

        # Update UI
        self.ts_slider.setMaximum(self.max_timesteps - 1)
        self.ts_spinbox.setMaximum(self.max_timesteps - 1)
        self.ts_slider.setValue(0)
        self.ts_spinbox.setValue(0)
        self._populate_type_list()
        self._populate_scalar_combo()
        self._update_info_browser()
        self._update_colorbar_widget()
        self._refresh()

        # Load optional metadata
        meta_path = os.path.join(folder, "sim_dict.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    self.sim_dict = json.load(f)
                self._update_info_browser()
            except Exception as e:
                print(f"Could not load sim_dict.json: {e}")

        # Try to load cached Voronoi if available
        if VORONOI_AVAILABLE:
            try:
                animator = VoronoiAnimator(self)
                if animator.load_cache(folder):
                    self.canvas_widget.voronoi_animator = animator
                    print(f"[Voronoi] Loaded cache from {folder}")
            except Exception as e:
                print(f"[Voronoi] Could not load cache: {e}")

            # Enable Voronoi checkbox now that data is loaded
            if hasattr(self, 'voronoi_check'):
                self.voronoi_check.setEnabled(True)
                self.voronoi_check.setChecked(False)
                self.voronoi_transparency_slider.setEnabled(True)

    def _detect_scalar_keys(self, data: dict) -> list:
        """Detect 1D scalar arrays in the data dict.
        Returns list of keys that have 1D arrays at each timestep (excluding known non-scalars).
        """
        excluded = {"x", "p_mask", "p", "q"}  # Known non-scalar keys
        scalar_keys = []
        
        for key in data:
            if key in excluded:
                continue
            values = data[key]
            if not values or len(values) == 0:
                continue
            # Check first non-None entry
            first_val = next((v for v in values if v is not None), None)
            if first_val is None:
                continue
            # Verify it's a 1D array-like with length matching cells
            try:
                arr = np.asarray(first_val)
                if arr.ndim == 1:
                    scalar_keys.append(key)
            except Exception:
                pass
        
        return sorted(scalar_keys)

    def _populate_scalar_combo(self):
        """Populate the scalar field dropdown based on detected scalar keys."""
        self.scalar_combo.blockSignals(True)
        self.scalar_combo.clear()
        for key in self.scalar_keys:
            self.scalar_combo.addItem(key)
        if self.scalar_keys:
            self.scalar_key = self.scalar_keys[0]
            self.scalar_combo.setCurrentText(self.scalar_key)
        self.scalar_combo.blockSignals(False)

    def _generate_type_colors(self):
        # Persistent, deterministic colors by p_mask value.
        # Example: 0=red, 1=blue, 2=green, then repeats.
        cycle = [
            (1.0, 0.0, 0.0, 1.0),  # red
            (0.0, 0.0, 1.0, 1.0),  # blue
            (0.0, 1.0, 0.0, 1.0),  # green
            (1.0, 1.0, 0.0, 1.0),  # yellow
            (1.0, 0.0, 1.0, 1.0),  # magenta
            (0.0, 1.0, 1.0, 1.0),  # cyan
            (1.0, 0.5, 0.0, 1.0),  # orange
            (0.6, 0.2, 1.0, 1.0),  # purple
            (0.7, 0.7, 0.7, 1.0),  # gray
        ]

        self.cell_type_colors = {}
        n = len(cycle)
        for ct in self.unique_types:
            try:
                idx = int(ct) % n
            except Exception:
                idx = 0
            self.cell_type_colors[ct] = cycle[idx]

    def _populate_type_list(self):
        self.type_list.clear()
        for ct in self.unique_types:
            item = QListWidgetItem()
            w = QWidget()
            row = QHBoxLayout(w)
            row.setContentsMargins(4, 2, 4, 2)

            chk = QCheckBox()
            chk.setChecked(ct in self.visible_types)
            chk.stateChanged.connect(lambda state, t=ct: self._on_type_visibility_changed(t, state))
            row.addWidget(chk)

            color = self.cell_type_colors.get(ct, (0.5, 0.5, 0.5, 1.0))
            swatch = _ColorLabel(color)
            swatch.mousePressEvent = lambda ev, t=ct, sw=swatch: self._pick_type_color(t, sw)
            row.addWidget(swatch)

            row.addWidget(QLabel(f"Type {ct}"))
            row.addStretch()

            item.setSizeHint(QSize(0, 32))
            self.type_list.addItem(item)
            self.type_list.setItemWidget(item, w)

    def _pick_type_color(self, cell_type, swatch: _ColorLabel):
        from PyQt6.QtWidgets import QColorDialog
        r, g, b = [int(c * 255) for c in swatch.color[:3]]
        qc = QColor(r, g, b)
        new_qc = QColorDialog.getColor(qc, self, f"Pick color for Type {cell_type}")
        if new_qc.isValid():
            new_color = (new_qc.redF(), new_qc.greenF(), new_qc.blueF(), 1.0)
            self.cell_type_colors[cell_type] = new_color
            swatch.set_color(new_color)
            self._refresh()

    def _update_info_browser(self):
        lines = []
        if self.data:
            lines.append(f"Timesteps: {self.max_timesteps}")
            if "x" in self.data and self.data["x"]:
                lines.append(f"Cells/step: {len(self.data['x'][0])}")
            if self.scalar_keys:
                lines.append(f"Scalars: {', '.join(self.scalar_keys)}")
                for k in self.scalar_keys:
                    if k in self.scalar_ranges:
                        lo, hi = self.scalar_ranges[k]
                        if self._is_logspace_scalar(k):
                            lines.append(f"  {k} (log): [{lo:.3f}, {hi:.3f}] (center at gamma=1)")
                        else:
                            lines.append(f"  {k}: [{lo:.3f}, {hi:.3f}]")
        if self.sim_dict:
            lines.append("---")
            for k, v in list(self.sim_dict.items())[:15]:
                lines.append(f"{k}: {v}")
        self.info_browser.setPlainText("\n".join(lines))

    # ------------------------------------------------------------------
    # Timestep control
    # ------------------------------------------------------------------
    def _go_to_timestep(self, t: int):
        if self.data is None:
            return
        t = int(np.clip(t, 0, self.max_timesteps - 1))
        if t == self.current_timestep and self.ts_slider.value() == t and self.ts_spinbox.value() == t:
            return
        self.current_timestep = t

        if self.ts_slider.value() != t:
            self.ts_slider.blockSignals(True)
            self.ts_slider.setValue(t)
            self.ts_slider.blockSignals(False)
        if self.ts_spinbox.value() != t:
            self.ts_spinbox.blockSignals(True)
            self.ts_spinbox.setValue(t)
            self.ts_spinbox.blockSignals(False)

        # Clear neighbour selection on timestep change
        self.selected_cell_idx = None

        if self.color_mode == "scalar":
            self._update_colorbar_widget()

        self._refresh()

    def _refresh(self):
        """Re-render the canvas."""
        self.canvas_widget.render()

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------
    def toggle_play(self):
        self.playing = not self.playing
        if self.playing:
            self.play_btn.setText("Pause")
            self.timer.start(self.play_speed)
        else:
            self.play_btn.setText("Play")
            self.timer.stop()
            self.play_btn.setChecked(False)

    def next_frame(self):
        nxt = (self.current_timestep + 1) % self.max_timesteps
        self._go_to_timestep(nxt)

    def prev_frame(self):
        prv = (self.current_timestep - 1) % self.max_timesteps
        self._go_to_timestep(prv)

    def _on_speed_changed(self, val: int):
        # slider 1 (slowest) -> 200 ms, slider 20 (fastest) -> 10 ms
        self.play_speed = max(10, 210 - val * 10)
        if self.playing:
            self.timer.setInterval(self.play_speed)

    # ------------------------------------------------------------------
    # Visualization event handlers
    # ------------------------------------------------------------------
    def _on_size_changed(self, val: int):
        self.cell_size = float(val)
        self.size_label.setText(str(val))
        self._refresh()

    def _on_color_mode_changed(self, btn):
        self.color_mode = btn.property("color_mode")
        self._update_colorbar_widget()
        # Update Voronoi colors when color mode changes
        if self.canvas_widget.voronoi_animator:
            self.canvas_widget.update_voronoi_colors()
        self._refresh()

    def _on_scalar_key_changed(self, text: str):
        self.scalar_key = text
        if self.color_mode == "scalar":
            self._update_colorbar_widget()
            # Update Voronoi colors when scalar field changes
            if self.canvas_widget.voronoi_animator:
                self.canvas_widget.update_voronoi_colors()
            self._refresh()

    # ------------------------------------------------------------------
    # Colorbar rendering
    # ------------------------------------------------------------------
    def _render_colorbar_pixmap(self, key: str, vmin: float, vmax: float):
        """Render a horizontal colorbar as a QPixmap using matplotlib."""
        from PyQt6.QtGui import QPixmap, QImage
        import matplotlib.colors as mcolors

        # Guard against degenerate / inverted ranges
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            vmin, vmax = -1.0, 1.0
        if vmax < vmin:
            vmin, vmax = vmax, vmin
        if abs(vmax - vmin) < 1e-12:
            pad = 1e-6 if abs(vmin) < 1e-6 else abs(vmin) * 1e-6
            vmin -= pad
            vmax += pad

        # Build a ListedColormap matching the 3-point scalar_to_rgba gradient
        n = 256
        t = np.linspace(0.0, 1.0, n)
        rgb = np.ones((n, 3), dtype=np.float32)
        lo = t < 0.5
        t_lo = t[lo] * 2.0
        rgb[lo] = np.outer(1 - t_lo, _COLOR_START) + np.outer(t_lo, _COLOR_MID)
        hi = ~lo
        t_hi = (t[hi] - 0.5) * 2.0
        rgb[hi] = np.outer(1 - t_hi, _COLOR_MID) + np.outer(t_hi, _COLOR_END)
        cmap = mcolors.ListedColormap(rgb)
        use_center = self._is_logspace_scalar(key)
        vcenter = self._scalar_vcenter(key)

        # TwoSlopeNorm requires strict ordering: vmin < vcenter < vmax
        can_use_two_slope = (
            use_center and vcenter is not None and (vmin < vcenter < vmax)
        )

        if can_use_two_slope:
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        else:
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        # Render with matplotlib
        dpi = 100
        fig, ax = plt.subplots(figsize=(3.2, 1.1), dpi=dpi)
        fig.patch.set_facecolor("#1e1e3a")
        cb = matplotlib.colorbar.ColorbarBase(
            ax, cmap=cmap, norm=norm, orientation="horizontal"
        )
        if use_center:
            cb.set_label(f"{key} (log scale)", color="#e0e0f0", fontsize=10)
            lo_tick = vmin
            hi_tick = vmax
            if can_use_two_slope:
                mid_tick = vcenter
                ticks = [lo_tick, mid_tick, hi_tick]
                labels = [f"{np.exp(lo_tick):.3g}", "1.0", f"{np.exp(hi_tick):.3g}"]
            else:
                ticks = [lo_tick, hi_tick]
                labels = [f"{np.exp(lo_tick):.3g}", f"{np.exp(hi_tick):.3g}"]
            cb.set_ticks(ticks)
            cb.set_ticklabels(labels)
        else:
            cb.set_label(key, color="#e0e0f0", fontsize=10)
        cb.ax.tick_params(colors="#e0e0f0", labelsize=9)
        ax.set_facecolor("#1e1e3a")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2a4a")
        fig.tight_layout(pad=0.3)

        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        w, h = fig.canvas.get_width_height()
        img = QImage(buf, w, h, QImage.Format.Format_RGBA8888)
        pixmap = QPixmap.fromImage(img)
        plt.close(fig)
        return pixmap

    def _render_histogram_pixmap(self, key: str, values: np.ndarray, bins: int):
        """Render scalar histogram for current timestep as a QPixmap."""
        from PyQt6.QtGui import QPixmap, QImage

        arr = np.asarray(values, dtype=np.float32)
        arr = arr[np.isfinite(arr)]

        dpi = 100
        fig, ax = plt.subplots(figsize=(4.8, 2.25), dpi=dpi)
        fig.patch.set_facecolor("#1e1e3a")
        ax.set_facecolor("#1e1e3a")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2a4a")

        if len(arr) == 0:
            ax.text(0.5, 0.5, "No finite values", color="#e0e0f0",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            if self._is_logspace_scalar(key):
                pos = arr[arr > 0]
                dropped = len(arr) - len(pos)
                if len(pos) == 0:
                    ax.text(0.5, 0.5, "No positive values for log histogram", color="#e0e0f0",
                            ha="center", va="center", transform=ax.transAxes)
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    base_edges = np.histogram_bin_edges(pos, bins=bins)
                    lo, hi = float(base_edges[0]), float(base_edges[-1])

                    if hi > lo and lo > 0:
                        logbins = np.logspace(np.log10(lo), np.log10(hi), len(base_edges))
                        ax.hist(pos, bins=logbins, color="#e94560", edgecolor="#e0e0f0", alpha=0.85)
                        ax.set_xscale("log")
                    else:
                        ax.hist(pos, bins=bins, color="#e94560", edgecolor="#e0e0f0", alpha=0.85)
                    if dropped > 0:
                        ax.set_title(f"{key} histogram (dropped <=0: {dropped})", color="#e0e0f0", fontsize=9)
                    else:
                        ax.set_title(f"{key} histogram", color="#e0e0f0", fontsize=9)
            else:
                ax.hist(arr, bins=bins, color="#e94560", edgecolor="#e0e0f0", alpha=0.85)
                ax.set_title(f"{key} histogram", color="#e0e0f0", fontsize=9)

            ax.set_ylabel("Count", color="#e0e0f0", fontsize=18, fontweight="bold")

            # Ensure both major and minor tick labels are consistently styled
            ax.tick_params(axis="both", which="both", colors="#e0e0f0", labelcolor="#e0e0f0", labelsize=16)
            for lbl in ax.get_xticklabels(minor=False) + ax.get_xticklabels(minor=True):
                lbl.set_color("#e0e0f0")
                lbl.set_fontsize(16)
                lbl.set_fontweight("bold")
            for lbl in ax.get_yticklabels(minor=False) + ax.get_yticklabels(minor=True):
                lbl.set_color("#e0e0f0")
                lbl.set_fontsize(16)
                lbl.set_fontweight("bold")

            # Style offset/scientific notation text as well
            ax.xaxis.get_offset_text().set_color("#e0e0f0")
            ax.xaxis.get_offset_text().set_fontsize(16)
            ax.xaxis.get_offset_text().set_fontweight("bold")
            ax.yaxis.get_offset_text().set_color("#e0e0f0")
            ax.yaxis.get_offset_text().set_fontsize(16)
            ax.yaxis.get_offset_text().set_fontweight("bold")

        fig.tight_layout(pad=0.4)
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        w, h = fig.canvas.get_width_height()
        img = QImage(buf, w, h, QImage.Format.Format_RGBA8888)
        pixmap = QPixmap.fromImage(img)
        plt.close(fig)
        return pixmap

    def _update_histogram_widget(self):
        """Refresh histogram for current scalar/timestep if enabled."""
        if (not self.show_scalar_histogram or self.data is None or
            self.color_mode != "scalar" or not self.scalar_key):
            self.histogram_label.setVisible(False)
            return

        key = self.scalar_key
        if key not in self.data:
            self.histogram_label.setVisible(False)
            return
        if self.current_timestep >= len(self.data[key]):
            self.histogram_label.setVisible(False)
            return

        vals = self.data[key][self.current_timestep]
        if vals is None:
            self.histogram_label.setVisible(False)
            return

        bins = int(self.hist_bins_spin.value())
        pixmap = self._render_histogram_pixmap(key, vals, bins)
        self.histogram_label.setPixmap(
            pixmap.scaled(
                self.histogram_label.width() or pixmap.width(),
                pixmap.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self.histogram_label.setVisible(True)

    def _on_show_histogram_clicked(self):
        self.show_scalar_histogram = True
        self._update_histogram_widget()

    def _on_hist_bins_changed(self, _val: int):
        if self.show_scalar_histogram and self.color_mode == "scalar":
            self._update_histogram_widget()

    def _update_colorbar_widget(self):
        """Show/hide and refresh the colorbar based on current color mode."""
        if self.color_mode != "scalar" or self.data is None:
            self.colorbar_group.setVisible(False)
            self.histogram_label.setVisible(False)
            return
        key = self.scalar_key
        vmin, vmax = self.scalar_ranges.get(key, (-1.0, 1.0))
        pixmap = self._render_colorbar_pixmap(key, vmin, vmax)
        self.colorbar_label.setPixmap(
            pixmap.scaled(
                self.colorbar_label.width() or pixmap.width(),
                pixmap.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self._update_histogram_widget()
        self.colorbar_group.setVisible(True)

    def _on_polarity_changed(self, state: int):
        if self.vector_mode:
            self.show_polarity = False
            return
        self.show_polarity = bool(state)
        self._refresh()

    def _on_polarity_type_changed(self, text: str):
        self.polarity_type = "p" if "Apicobasal" in text else "q"
        if self.show_polarity and (not self.vector_mode):
            self._refresh()

    def _on_vector_mode_changed(self, state: int):
        self.vector_mode = bool(state)
        if self.vector_mode:
            # Supersede the old polarity-tip mode
            self.show_polarity = False
            if hasattr(self, "pol_checkbox"):
                self.pol_checkbox.blockSignals(True)
                self.pol_checkbox.setChecked(False)
                self.pol_checkbox.blockSignals(False)
                self.pol_checkbox.setEnabled(False)
            if hasattr(self, "pol_combo"):
                self.pol_combo.setEnabled(False)
        else:
            if hasattr(self, "pol_checkbox"):
                self.pol_checkbox.setEnabled(True)
            if hasattr(self, "pol_combo"):
                self.pol_combo.setEnabled(True)
        self._refresh()

    def _on_axes_changed(self, state: int):
        visible = bool(state)
        for ln in self.canvas_widget.axis_lines:
            ln.visible = visible
        self.canvas_widget.canvas.update()

    def _on_bg_changed(self, text: str):
        self.canvas_widget.set_background(text)
        self.canvas_widget.canvas.update()

    def _on_voronoi_enabled(self, state: int):
        """Handle Voronoi visualization toggle."""
        if not state:
            # Disabling Voronoi
            self.voronoi_enabled = False
            self.canvas_widget.voronoi_enabled = False
            self._refresh()
            return

        # Enabling Voronoi - check if already computed
        if self.data is None:
            QMessageBox.warning(self, "Warning", "Please load data first.")
            self.voronoi_check.setChecked(False)
            return

        # Check if force recalculation is requested
        if self.voronoi_force_recalc_check.isChecked():
            # Clear existing animator to force recalculation
            self.canvas_widget.voronoi_animator = None
            self.voronoi_force_recalc_check.setChecked(False)
            print("[voronoi] Force recalculation enabled - clearing cache")

        # Check if Voronoi is already cached for this data
        if (self.canvas_widget.voronoi_animator and
            len(self.canvas_widget.voronoi_animator.voronoi_meshes) > 0):
            # Voronoi already computed, just enable visualization
            self.voronoi_enabled = True
            self.canvas_widget.voronoi_enabled = True
            self._refresh()
            return

        # Ask user for timestep range
        if self.max_timesteps <= 1:
            start_ts = 0
            end_ts = self.max_timesteps - 1
        else:
            # Create dialog for timestep selection
            dialog = QDialog(self)
            dialog.setWindowTitle("Select Timestep Range for Voronoi")
            layout = QVBoxLayout(dialog)

            # Instructions
            layout.addWidget(QLabel(f"Total timesteps: 0 to {self.max_timesteps - 1}"))
            layout.addWidget(QLabel("Select range to compute Voronoi for:"))

            # Start timestep
            start_row = QHBoxLayout()
            start_row.addWidget(QLabel("Start:"))
            start_spin = QSpinBox()
            start_spin.setMinimum(0)
            start_spin.setMaximum(self.max_timesteps - 1)
            start_spin.setValue(0)
            start_row.addWidget(start_spin)
            layout.addLayout(start_row)

            # End timestep
            end_row = QHBoxLayout()
            end_row.addWidget(QLabel("End:"))
            end_spin = QSpinBox()
            end_spin.setMinimum(0)
            end_spin.setMaximum(self.max_timesteps - 1)
            end_spin.setValue(self.max_timesteps - 1)
            end_row.addWidget(end_spin)
            layout.addLayout(end_row)

            # Buttons
            btn_layout = QHBoxLayout()
            ok_btn = QPushButton("OK")
            cancel_btn = QPushButton("Cancel")

            def on_ok():
                dialog.accept()

            def on_cancel():
                dialog.reject()

            ok_btn.clicked.connect(on_ok)
            cancel_btn.clicked.connect(on_cancel)
            btn_layout.addWidget(ok_btn)
            btn_layout.addWidget(cancel_btn)
            layout.addLayout(btn_layout)

            dialog.setLayout(layout)

            # Show dialog
            if dialog.exec() == QDialog.DialogCode.Accepted:
                start_ts = start_spin.value()
                end_ts = end_spin.value()
                if start_ts > end_ts:
                    start_ts, end_ts = end_ts, start_ts
            else:
                # User cancelled
                self.voronoi_check.setChecked(False)
                return

        # Start pre-computation for the selected range
        self.voronoi_enabled = True
        self.canvas_widget.voronoi_enabled = True
        self._start_voronoi_precompute(start_ts, end_ts)

    def _on_voronoi_transparency(self, val: int):
        """Handle Voronoi transparency slider."""
        alpha = val / 100.0
        self.voronoi_transparency = alpha
        self.voronoi_transparency_label.setText(f"{alpha:.2f}")
        self.canvas_widget.voronoi_transparency = alpha
        if self.canvas_widget.voronoi_animator:
            self.canvas_widget.voronoi_animator.update_transparency(alpha)
            self._refresh()

    def _on_voronoi_show_cells(self, state: int):
        """Handle show/hide cell markers toggle when Voronoi is active."""
        show_cells = state == Qt.CheckState.Checked.value
        # This control is in the GUI, refresh render
        self._refresh()

    def _on_voronoi_edges_toggled(self, state: int):
        """Handle Voronoi edges toggle."""
        show_edges = state == Qt.CheckState.Checked.value
        self.canvas_widget.toggle_voronoi_edges(show_edges)
        self.canvas_widget.render()

    def _on_voronoi_edge_color_changed(self, color_name: str):
        """Handle Voronoi edge color selection."""
        self.canvas_widget.set_voronoi_edge_color(color_name.lower())
        self.canvas_widget.render()

    def _start_voronoi_precompute(self, start_ts: int = 0, end_ts: int = None):
        """
        Start pre-computing Voronoi tessellations for a range of frames.

        Parameters
        ----------
        start_ts : int
            Starting timestep (inclusive)
        end_ts : int
            Ending timestep (inclusive)
        """
        if self.data is None or "x" not in self.data or "p" not in self.data:
            QMessageBox.warning(
                self, "Warning",
                "Cannot compute Voronoi: data missing 'x' or 'p' keys."
            )
            self.voronoi_check.setChecked(False)
            return

        if end_ts is None:
            end_ts = len(self.data["x"]) - 1

        # Validate range
        start_ts = max(0, start_ts)
        end_ts = min(len(self.data["x"]) - 1, end_ts)

        if start_ts > end_ts:
            start_ts, end_ts = end_ts, start_ts

        nframes = end_ts - start_ts + 1

        # Create animator if needed
        if self.canvas_widget.voronoi_animator is None:
            self.canvas_widget.voronoi_animator = VoronoiAnimator(self)

        animator = self.canvas_widget.voronoi_animator

        # Lazily create progress dialog only when computation is explicitly started
        if self.voronoi_progress is None:
            self.voronoi_progress = QProgressDialog("Pre-computing Voronoi...", None, 0, 100, self)
            self.voronoi_progress.setAutoClose(True)
            self.voronoi_progress.setAutoReset(True)
            self.voronoi_progress.setWindowModality(Qt.WindowModality.WindowModal)

        # Show progress dialog
        self.voronoi_progress.setMaximum(nframes)
        self.voronoi_progress.setValue(0)
        self.voronoi_progress.show()

        def on_progress(t, n):
            self.voronoi_progress.setValue(t - start_ts + 1)
            QApplication.processEvents()

        # Pre-compute
        try:
            success = animator.precompute_all_frames(
                self.data,
                progress_callback=on_progress,
                apply_occlusion=False,
                start_ts=start_ts,
                end_ts=end_ts,
            )
            if success:
                # Save cache
                animator.save_cache(self.data_folder)
                QMessageBox.information(
                    self, "Success",
                    f"Voronoi pre-computation completed for {nframes} frames ({start_ts}-{end_ts})."
                )
            else:
                QMessageBox.critical(
                    self, "Error",
                    "Voronoi pre-computation failed. Check console for details."
                )
                self.voronoi_check.setChecked(False)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Voronoi pre-computation error:\n{e}")
            self.voronoi_check.setChecked(False)
        finally:
            self.voronoi_progress.setVisible(False)

        self._refresh()

    def _on_type_visibility_changed(self, cell_type, state: int):
        if state:
            self.visible_types.add(cell_type)
        else:
            self.visible_types.discard(cell_type)
        self._refresh()

    # ------------------------------------------------------------------
    # Sectioning event handlers
    # ------------------------------------------------------------------
    def _on_bisection_changed(self, state: int):
        self.bisection_enabled = bool(state)
        self._refresh()

    def _on_plane_changed(self, btn):
        self.bisection_plane = btn.property("plane")
        if self.bisection_enabled:
            self._refresh()

    def _on_cross_section_changed(self, state: int):
        self.cross_section_mode = bool(state)
        if self.bisection_enabled:
            self._refresh()

    # ------------------------------------------------------------------
    # Neighbour search
    # ------------------------------------------------------------------
    @staticmethod
    def _find_potential_neighbours(x: np.ndarray, k: int):
        tree = cKDTree(x)
        d, idx = tree.query(x, k + 1, workers=-1)
        return d[:, 1:], idx[:, 1:]

    @staticmethod
    def _find_true_neighbours(d: np.ndarray, dx: np.ndarray, seethru: int) -> np.ndarray:
        """Numpy fallback for geometric occlusion test."""
        N, K = d.shape
        result = np.empty((N, K), dtype=bool)
        batch = 512
        i0 = 0
        while i0 < N:
            i1 = min(i0 + batch, N)
            # dx[i0:i1, :, None, :] shape: (B, K, 1, 3)
            # dx[i0:i1, None, :, :] shape: (B, 1, K, 3)
            diff = dx[i0:i1, :, None, :] / 2.0 - dx[i0:i1, None, :, :]
            n_dis = np.sum(diff ** 2, axis=3)
            eye_b = np.eye(K)[np.newaxis, :, :]  # (1, K, K)
            n_dis += 1000 * eye_b
            result[i0:i1] = np.sum(n_dis < (d[i0:i1, :, None] ** 2 / 4), axis=2) <= seethru
            i0 = i1
        return result

    def _on_search_clicked(self):
        if self.playing:
            QMessageBox.warning(self, "Playback Active", "Pause playback first.")
            return
        if self.data is None:
            QMessageBox.warning(self, "No Data", "Load data first.")
            return

        t = self.current_timestep
        k = self.max_k_spin.value()
        seethru = self.seethru_spin.value()

        # Check cache
        cached_result = self._load_neighbour_cache(t, k, seethru)
        if cached_result is not None:
            self.neighbor_data[t] = cached_result
            self.neighbor_search_enabled = True
            self.clear_nbr_btn.setVisible(True)
            QMessageBox.information(self, "Loaded", "Loaded neighbour data from cache.")
            return

        progress = QProgressDialog("Computing neighbours...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        self.search_btn.setEnabled(False)
        orig_text = self.search_btn.text()
        self.search_btn.setText("Searching...")

        x = self.data["x"][t]
        start = time.time()
        try:
            d, idx = self._find_potential_neighbours(x, k)
            progress.setValue(30)
            QApplication.processEvents()
            if progress.wasCanceled():
                return

            dx = x[idx] - x[:, np.newaxis, :]
            result_mask = self._find_true_neighbours(d, dx, seethru)
            progress.setValue(90)
            QApplication.processEvents()

            true_neighbors = [idx[i][result_mask[i]] for i in range(len(x))]
            result = {"true_neighbors": true_neighbors, "k": k, "seethru": seethru}
            self.neighbor_data[t] = result
            self.neighbor_search_enabled = True
            self._save_neighbour_cache(t, k, seethru, result)
            elapsed = time.time() - start
            QMessageBox.information(self, "Done",
                f"Neighbour search complete in {elapsed:.1f}s.\n"
                "Right-click a cell to select it.")
            self.clear_nbr_btn.setVisible(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Neighbour search failed:\n{e}")
            import traceback; traceback.print_exc()
        finally:
            progress.close()
            self.search_btn.setEnabled(True)
            self.search_btn.setText(orig_text)

    def _cache_path(self, t: int, k: int, seethru: int) -> str | None:
        if not self.data_folder:
            return None
        cache_dir = os.path.join(self.data_folder, "neighbor_cache")
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"nbr_t{t}_k{k}_s{seethru}.pkl")

    def _load_neighbour_cache(self, t, k, seethru):
        path = self._cache_path(t, k, seethru)
        if path and os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Cache load error: {e}")
        return None

    def _save_neighbour_cache(self, t, k, seethru, data):
        path = self._cache_path(t, k, seethru)
        if path:
            try:
                with open(path, "wb") as f:
                    pickle.dump(data, f)
            except Exception as e:
                print(f"Cache save error: {e}")

    def on_cell_selected(self, cell_idx: int):
        if cell_idx is None or self.current_timestep not in self.neighbor_data:
            return
        nd = self.neighbor_data[self.current_timestep]
        true_neighbors = nd["true_neighbors"]
        if cell_idx >= len(true_neighbors):
            return

        self.selected_cell_idx = cell_idx
        nbr_idx = true_neighbors[cell_idx]
        self.canvas_widget.highlight_cells(cell_idx, nbr_idx)

        # Stats
        p_mask = self.data.get("p_mask", [None])[self.current_timestep]
        if p_mask is not None:
            sel_type = int(p_mask[cell_idx])
            counts = {}
            for ni in nbr_idx:
                ct = int(p_mask[ni])
                counts[ct] = counts.get(ct, 0) + 1
            html = (f"<b>Cell {cell_idx}</b> (Type {sel_type})<br>"
                    f"Neighbours: {len(nbr_idx)}<br>")
            for ct in sorted(counts):
                html += f"&nbsp;Type {ct}: {counts[ct]}<br>"
        else:
            html = f"<b>Cell {cell_idx}</b><br>Neighbours: {len(nbr_idx)}"

        self.nbr_stats.setHtml(html)
        self.nbr_stats.setVisible(True)
        self.show_only_check.setVisible(True)
        self.clear_nbr_btn.setVisible(True)

    def _on_show_only_changed(self, state):
        if self.selected_cell_idx is not None:
            self.on_cell_selected(self.selected_cell_idx)

    def clear_neighbor_selection(self):
        self.selected_cell_idx = None
        self.canvas_widget.clear_highlights()
        self.nbr_stats.setVisible(False)
        self.show_only_check.setVisible(False)
        self.show_only_check.setChecked(False)
        if self.current_timestep not in self.neighbor_data:
            self.clear_nbr_btn.setVisible(False)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def _export_image(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Image", "frame.png", "PNG files (*.png)"
        )
        if filename:
            ok = self.canvas_widget.export_image(filename)
            if ok:
                QMessageBox.information(self, "Done", f"Image saved:\n{filename}")
            else:
                QMessageBox.warning(self, "Error", "Image export failed.")

    def _export_video(self):
        try:
            import imageio  # noqa
        except ImportError:
            QMessageBox.critical(self, "Missing Dependency",
                "imageio is required for video export.\n"
                "Install with: pip install imageio[ffmpeg]")
            return

        if self.data is None:
            QMessageBox.warning(self, "No Data", "Load data first.")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Video", "simulation.mp4",
            "MP4 files (*.mp4);;AVI files (*.avi)"
        )
        if not filename:
            return

        success = self._export_video_sequence(filename, 0, self.max_timesteps - 1, fps=15)
        if success:
            QMessageBox.information(self, "Done", f"Video saved:\n{filename}")

    def _export_video_sequence(self, filename: str, start: int, end: int, fps: int) -> bool:
        import imageio

        total = end - start + 1
        progress = QProgressDialog("Capturing frames...", "Cancel", 0, total, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()

        orig_t = self.current_timestep
        frames = []
        try:
            for i, t in enumerate(range(start, end + 1)):
                QApplication.processEvents()
                if progress.wasCanceled():
                    return False
                progress.setValue(i)
                progress.setLabelText(f"Frame {i + 1}/{total}")
                self._go_to_timestep(t)
                for _ in range(3):
                    QApplication.processEvents()
                canvas = self.canvas_widget.canvas
                w = ((canvas.size[0] + 15) // 16) * 16
                h = ((canvas.size[1] + 15) // 16) * 16
                canvas.size = (w, h)
                img = canvas.render(alpha=True)
                canvas.size = self.canvas_widget.canvas.size
                if img is not None and img.size > 0:
                    frames.append(img)
            progress.close()
            if not frames:
                return False

            save_prog = QProgressDialog("Saving video...", None, 0, 0, self)
            save_prog.setWindowModality(Qt.WindowModality.WindowModal)
            save_prog.setMinimumDuration(0)
            save_prog.show()
            QApplication.processEvents()
            try:
                imageio.mimsave(
                    filename, frames, fps=fps, quality=9,
                    macro_block_size=1,
                    ffmpeg_params=["-crf", "18", "-preset", "slow", "-pix_fmt", "yuv420p"],
                )
                return True
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save video:\n{e}")
                return False
            finally:
                save_prog.close()
        except Exception as e:
            progress.close()
            QMessageBox.critical(self, "Error", f"Video export failed:\n{e}")
            return False
        finally:
            self._go_to_timestep(orig_t)


# ---------------------------------------------------------------------------
# Test data generator
# ---------------------------------------------------------------------------
def _make_test_data(n_cells: int = 200, n_steps: int = 30) -> dict:
    """Generate synthetic dict-of-lists test data."""
    rng = np.random.default_rng(42)
    data = {k: [] for k in ("p_mask", "x", "p", "q",
                             "alpha_par", "alpha_perp", "gamma", "energy")}
    for t in range(n_steps):
        spread = 5.0 + t * 0.1
        x = rng.normal(0, spread, (n_cells, 3)).astype(np.float32)
        p_mask = rng.integers(0, 3, n_cells)
        p = rng.normal(0, 1, (n_cells, 3)).astype(np.float32)
        p /= (np.linalg.norm(p, axis=1, keepdims=True) + 1e-9)
        q = rng.normal(0, 1, (n_cells, 3)).astype(np.float32)
        q /= (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
        phase = t / n_steps
        data["p_mask"].append(p_mask)
        data["x"].append(x)
        data["p"].append(p)
        data["q"].append(q)
        data["alpha_par"].append((np.sin(x[:, 0] * 0.3 + phase)).astype(np.float32))
        data["alpha_perp"].append((np.cos(x[:, 1] * 0.3 + phase)).astype(np.float32))
        data["gamma"].append(rng.uniform(-1, 1, n_cells).astype(np.float32))
        data["energy"].append((-np.linalg.norm(x, axis=1) / (spread * 2)).astype(np.float32))
    return data


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_STYLESHEET)
    gui = DataVizGUI()

    # Load test data if no argument given
    if "--test" in sys.argv:
        import tempfile
        tmp = tempfile.mkdtemp()
        d = _make_test_data()
        with open(os.path.join(tmp, "data.pkl"), "wb") as f:
            pickle.dump(d, f)
        gui._load_data(tmp)

    gui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
