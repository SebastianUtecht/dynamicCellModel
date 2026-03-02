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

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QSlider, QFileDialog,
    QMessageBox, QCheckBox, QScrollArea, QVBoxLayout, QHBoxLayout,
    QTextBrowser, QLabel, QComboBox, QSpinBox, QDoubleSpinBox, QListWidget,
    QListWidgetItem, QSplitter, QProgressDialog, QRadioButton, QButtonGroup,
    QGroupBox, QTabWidget, QFormLayout, QFrame,
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

def scalar_to_rgba(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Map 1-D scalar array to RGBA using 3-point magma-like colormap."""
    vrange = vmax - vmin
    if vrange > 0:
        norm = np.clip((values - vmin) / vrange, 0.0, 1.0)
    else:
        norm = np.zeros_like(values)

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
    _highlights: dict

    def __init__(self, gui: "DataVizGUI"):
        super().__init__()
        self.gui = gui
        self._cell_scatters = {}
        self._polarity_markers = None
        self._highlights = {}
        self._camera_bounds_set = False
        self._ghost_visual = None
        self._ghost_pos = None

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
            if event.button == 2 and self.gui.neighbor_search_enabled:
                self._on_right_click()
                event.handled = True

        @self.canvas.events.mouse_wheel.connect
        def _on_wheel(event):
            shift = False
            if hasattr(event, "modifiers") and event.modifiers:
                shift = "shift" in str(event.modifiers).lower()
            if not shift and hasattr(event, "native") and hasattr(event.native, "modifiers"):
                shift = bool(event.native.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            if shift and self.gui.neighbor_search_enabled:
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
        if self.gui.neighbor_search_enabled:
            self._ghost_pos = self._camera_forward_pos(self._ghost_sphere_dist)
            self._ghost_visual.set_data(
                np.array([self._ghost_pos]),
                edge_width=0, face_color="cyan",
                size=self._ghost_sphere_radius * 2,
            )
            self._ghost_visual.visible = True
        else:
            self._ghost_visual.visible = False

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

    def clear_highlights(self):
        for v in self._highlights.values():
            try:
                v.parent = None
            except Exception as e:
                print(f"[clear highlight] {e}")
        self._highlights.clear()
        self.render()

    # ------------------------------------------------------------------
    # Main render entry point
    # ------------------------------------------------------------------
    def render(self):
        """Render the current timestep according to gui settings."""
        g = self.gui
        if g.data is None:
            return
        t = g.current_timestep
        x = g.data["x"][t]
        p_mask = g.data["p_mask"][t] if "p_mask" in g.data else None

        if x is None or len(x) == 0:
            return

        self._clear_cells()

        # Apply section filter
        x_f, pm_f, bool_mask = self._apply_section_filter(x, p_mask)
        if len(x_f) == 0:
            return

        # Choose color mode
        mode = g.color_mode  # "type" | "depth" | "scalar"
        if mode == "scalar":
            key = g.scalar_key
            raw = g.data[key][t]
            scalar_f = raw[bool_mask] if raw is not None else None
            if scalar_f is not None:
                colors = scalar_to_rgba(scalar_f, g.scalar_ranges[key][0], g.scalar_ranges[key][1])
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
        if g.show_polarity:
            pkey = "p" if g.polarity_type == "p" else "q"
            pol_all = g.data.get(pkey, [None])[t]
            if pol_all is not None and len(pol_all) > 0:
                pol_f = pol_all[bool_mask]
                self._render_polarity(x_f, pol_f)

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

    # ------------------------------------------------------------------
    # Neighbour highlighting
    # ------------------------------------------------------------------
    def highlight_cells(self, sel_idx: int, nbr_indices):
        self.clear_highlights()
        g = self.gui
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
    scalar_key: str             # one of SCALAR_KEYS
    scalar_ranges: dict         # {key: (min, max)}
    cell_size: float
    show_polarity: bool
    polarity_type: str          # "p" or "q"
    bisection_enabled: bool
    bisection_plane: str
    bisection_position: float
    cross_section_mode: bool
    cross_section_width: float
    unique_types: list
    visible_types: set
    cell_type_colors: dict
    neighbor_search_enabled: bool

    SCALAR_KEYS = ("alpha_par", "alpha_perp", "gamma", "energy")

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
        self.scalar_key = "alpha_par"
        self.scalar_ranges = {k: (-1.0, 1.0) for k in self.SCALAR_KEYS}
        self.cell_size = 2.0
        self.show_polarity = False
        self.polarity_type = "p"

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
        self.selected_cell_idx = None

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
        self._build_tools_tab()

        splitter.addWidget(self.tabs)

        # Right: canvas
        self.canvas_widget = VisPy3DWidget(self)
        splitter.addWidget(self.canvas_widget)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([350, 1050])

        root_layout.addWidget(splitter)

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
        for k in self.SCALAR_KEYS:
            self.scalar_combo.addItem(k)
        self.scalar_combo.currentTextChanged.connect(self._on_scalar_key_changed)
        scalar_row.addWidget(self.scalar_combo)
        color_layout.addLayout(scalar_row)
        layout.addWidget(color_group)

        # Polarity
        pol_group = QGroupBox("Polarity Vectors")
        pol_layout = QHBoxLayout(pol_group)
        self.pol_checkbox = QCheckBox("Show")
        self.pol_checkbox.stateChanged.connect(self._on_polarity_changed)
        self.pol_combo = QComboBox()
        self.pol_combo.addItems(["Apicobasal (p)", "Planar (q)"])
        self.pol_combo.currentTextChanged.connect(self._on_polarity_type_changed)
        pol_layout.addWidget(self.pol_checkbox)
        pol_layout.addWidget(self.pol_combo)
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

        # Compute scalar ranges
        for key in self.SCALAR_KEYS:
            if key in raw and raw[key]:
                try:
                    all_vals = np.concatenate([v for v in raw[key] if v is not None])
                    self.scalar_ranges[key] = (float(all_vals.min()), float(all_vals.max()))
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
        self._update_info_browser()
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

    def _generate_type_colors(self):
        cmap = plt.get_cmap("Set1")
        self.cell_type_colors = {}
        for i, ct in enumerate(self.unique_types):
            rgba = cmap(i % 9)
            self.cell_type_colors[ct] = rgba

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
            for k in self.SCALAR_KEYS:
                if k in self.data:
                    lo, hi = self.scalar_ranges[k]
                    lines.append(f"{k}: [{lo:.3f}, {hi:.3f}]")
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
        self._refresh()

    def _on_scalar_key_changed(self, text: str):
        self.scalar_key = text
        if self.color_mode == "scalar":
            self._refresh()

    def _on_polarity_changed(self, state: int):
        self.show_polarity = bool(state)
        self._refresh()

    def _on_polarity_type_changed(self, text: str):
        self.polarity_type = "p" if "Apicobasal" in text else "q"
        if self.show_polarity:
            self._refresh()

    def _on_axes_changed(self, state: int):
        visible = bool(state)
        for ln in self.canvas_widget.axis_lines:
            ln.visible = visible
        self.canvas_widget.canvas.update()

    def _on_bg_changed(self, text: str):
        self.canvas_widget.set_background(text)
        self.canvas_widget.canvas.update()

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
