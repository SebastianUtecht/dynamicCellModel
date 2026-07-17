"""
Reusable PyVista + matplotlib rendering for publication figure panels.

Supports sphere and augmented-Voronoi cluster visualization with shared
camera, distance coloring, and layout helpers for multi-panel figures.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from matplotlib.colors import ListedColormap, Normalize, TwoSlopeNorm
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch
from scipy.spatial import cKDTree

from voronoi_computation import AugmentedVoronoi, VoronoiMesh

RENDER_SETTINGS = {
    "window_size": (420, 420),
    "cmap": "viridis_r",
    "cmin": None,  # distance colormap min (None = per-panel minimum)
    "cmax": None,  # distance colormap max (None = per-panel maximum)
    "bright_near_camera": True,  # True = closest cells get the bright end of cmap
    "elevation": 35,
    "azimuth": 45,
    "sphere_radius": None,
    "sphere_resolution": 128,
    "ambient": 0.18,
    "diffuse": 0.82,
    "zoom_margin": 0.165,
    "voronoi_thickness": 2.0,
    "border_cap": 1.0,
    "neighbor_search_factor": 1.6,
    "neighbor_search_radius": None,
    "show_voronoi_edges": True,
    "voronoi_edge_mode": "intercell",  # "intercell" | "all_ridges"
    "voronoi_edge_color": "#111111",
    "voronoi_edge_width": 2.0,
    # backward-compatible aliases
    "edge_color": "#111111",
    "edge_width": 2.0,
    "color_mode": "distance",  # "distance" | "scalar"
    "scalar_key": "alpha_par",
    "scalar_values": None,
    "scalar_vmin": None,
    "scalar_vmax": None,
}

_SCALAR_LOG_EPS = 1e-12
_COLOR_START = np.array([0.0, 0.0, 0.3])
_COLOR_MID = np.array([1.0, 0.2, 0.0])
_COLOR_END = np.array([1.0, 1.0, 0.7])

FIG3_ROW_LABELS = [
    "Deformation type",
    "Initial Configuration",
    "During deformation",
    "Post deformation",
]


def _merge_render_settings(
    render_settings=None,
    voronoi_edge_color=None,
    voronoi_edge_width=None,
):
    """Merge render_settings with RENDER_SETTINGS defaults and optional edge overrides."""
    settings = {**RENDER_SETTINGS, **(render_settings or {})}
    if voronoi_edge_color is not None:
        settings["voronoi_edge_color"] = voronoi_edge_color
        settings["edge_color"] = voronoi_edge_color
    if voronoi_edge_width is not None:
        settings["voronoi_edge_width"] = voronoi_edge_width
        settings["edge_width"] = voronoi_edge_width
    return settings


def _voronoi_edge_style(settings):
    """Resolve Voronoi inter-cell edge color and line width from settings."""
    color = settings.get("voronoi_edge_color", settings.get("edge_color", "#111111"))
    width = settings.get("voronoi_edge_width", settings.get("edge_width", 2.0))
    return color, float(width)


def _rotation_matrix(axis: str, degrees: float) -> np.ndarray:
    rad = np.radians(degrees)
    c, s = np.cos(rad), np.sin(rad)
    if axis == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    if axis == "y":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    if axis == "z":
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis!r}")


def rotate_positions(x, axis="x", degrees=90.0, center=None):
    """Rotate (N, 3) positions by degrees around axis through center."""
    x = np.asarray(x, dtype=float)
    if center is None:
        center = x.mean(axis=0)
    rel = x - center
    rot = _rotation_matrix(axis, degrees)
    return rel @ rot.T + center


def rotate_vectors(v, axis="x", degrees=90.0):
    """Rotate (N, 3) direction vectors (no translation)."""
    v = np.asarray(v, dtype=float)
    rot = _rotation_matrix(axis, degrees)
    out = v @ rot.T
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms = np.where(norms > 1e-8, norms, 1.0)
    return out / norms


def _apply_rotation(x, p, rotation):
    if not rotation:
        return x, p
    center = x.mean(axis=0)
    x_rot = rotate_positions(x, center=center, **rotation)
    p_rot = rotate_vectors(p, **rotation) if p is not None else None
    return x_rot, p_rot


def _setup_camera(pl, center, settings, position=None):
    pl.camera.focal_point = center
    if position is None:
        pl.reset_camera()
        dist = np.linalg.norm(np.array(pl.camera.position) - center)
        elev = np.radians(settings["elevation"])
        azim = np.radians(settings["azimuth"])
        position = center + dist * np.array(
            [np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)]
        )
    pl.camera.position = position
    pl.camera.view_up = (0, 0, 1)
    pl.camera.zoom(1.0 / (1.0 + settings["zoom_margin"]))
    return np.array(pl.camera_position[0])


def _distance_colors(x, cam_pos, settings):
    """
    Map cell-center distances to camera into RGB using settings['cmap'].

    Optional settings['cmin'] and settings['cmax'] clip the distance range
    before normalizing (values outside are clamped to the colormap ends).

    settings['bright_near_camera'] (default True): closest cells map to the
    bright end of the colormap. Handles both standard and reversed (*_r) cmaps.
    """
    distances = np.linalg.norm(np.asarray(x, dtype=float) - cam_pos, axis=1)
    dmin = settings.get("cmin")
    dmax = settings.get("cmax")
    if dmin is None:
        dmin = float(distances.min())
    if dmax is None:
        dmax = float(distances.max())
    norm = (distances - dmin) / (dmax - dmin + 1e-9)
    norm = np.clip(norm, 0.0, 1.0)

    cmap_name = settings.get("cmap", "viridis_r")
    bright_near = settings.get("bright_near_camera", True)
    # Matplotlib *_r colormaps have bright colors at norm=0
    bright_at_low = str(cmap_name).endswith("_r")
    if bright_near != bright_at_low:
        norm = 1.0 - norm

    return plt.get_cmap(cmap_name)(norm)[:, :3]


def _is_logspace_scalar(key: str) -> bool:
    return isinstance(key, str) and "gamma" in key.lower()


def _transform_scalar_values(key: str, values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if _is_logspace_scalar(key):
        arr = np.log(np.clip(arr, _SCALAR_LOG_EPS, None))
    return arr


def _scalar_vcenter(key: str) -> Optional[float]:
    if _is_logspace_scalar(key):
        return 0.0
    return None


def _scalar_norm_bounds(key: str, vmin: float, vmax: float) -> Tuple[float, float]:
    """Convert user-facing scalar bounds to normalization space (log for gamma)."""
    if _is_logspace_scalar(key):
        lo = float(np.log(np.clip(vmin, _SCALAR_LOG_EPS, None)))
        hi = float(np.log(np.clip(vmax, _SCALAR_LOG_EPS, None)))
        lo = min(lo, 0.0)
        hi = max(hi, 0.0)
        return lo, hi
    return float(vmin), float(vmax)


def scalar_to_rgba(
    values: np.ndarray,
    vmin: float,
    vmax: float,
    vcenter: Optional[float] = None,
) -> np.ndarray:
    """Map 1-D scalar array to RGBA using the GUI 3-point colormap."""
    values = np.asarray(values, dtype=np.float32)
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


def scalar_colors(
    key: str, values: np.ndarray, vmin: float, vmax: float
) -> np.ndarray:
    """Map per-cell scalar values to RGB (matches data_visualization_gui_v2)."""
    vals_t = _transform_scalar_values(key, values)
    rgba = scalar_to_rgba(vals_t, vmin, vmax, _scalar_vcenter(key))
    return rgba[:, :3]


def compute_scalar_ranges(columns: List[dict]) -> Dict[str, Tuple[float, float]]:
    """Shared vmin/vmax per scalar key across all fig3 panels."""
    chunks: Dict[str, List[np.ndarray]] = {}
    for col in columns:
        scalar_key = col["scalar"]
        data = col["data"]
        for frame in col["frames"]:
            vals = np.asarray(data[scalar_key][frame], dtype=np.float32)
            chunks.setdefault(scalar_key, []).append(_transform_scalar_values(scalar_key, vals))

    ranges: Dict[str, Tuple[float, float]] = {}
    for key, parts in chunks.items():
        all_vals = np.concatenate(parts)
        if _is_logspace_scalar(key):
            lo = min(float(np.min(all_vals)), 0.0)
            hi = max(float(np.max(all_vals)), 0.0)
        else:
            lo, hi = float(np.min(all_vals)), float(np.max(all_vals))
        ranges[key] = (lo, hi)
    return ranges


def _cell_colors(x, cam_pos, settings):
    """Per-cell RGB for Voronoi rendering (distance or scalar mode)."""
    if settings.get("color_mode") == "scalar":
        scalar_values = settings.get("scalar_values")
        if scalar_values is None:
            raise ValueError("color_mode='scalar' requires scalar_values in settings")
        key = settings.get("scalar_key", "alpha_par")
        vmin = settings.get("scalar_vmin")
        vmax = settings.get("scalar_vmax")
        vals_t = _transform_scalar_values(key, scalar_values)
        if vmin is None:
            vmin = float(vals_t.min())
        if vmax is None:
            vmax = float(vals_t.max())
        return scalar_colors(key, scalar_values, vmin, vmax)
    return _distance_colors(x, cam_pos, settings)


def build_voronoi_meshes(x, p, settings=None) -> dict[int, VoronoiMesh]:
    settings = {**RENDER_SETTINGS, **(settings or {})}
    comp = AugmentedVoronoi(
        x,
        p,
        thickness=settings["voronoi_thickness"],
        border_cap=settings.get("border_cap"),
        enable_lateral_boundary=settings.get("enable_lateral_boundary", True),
        neighbor_search_factor=settings.get("neighbor_search_factor", 1.6),
        neighbor_search_radius=settings.get("neighbor_search_radius"),
    )
    return comp.compute_voronoi_meshes()


def merge_voronoi_to_polydata(meshes: dict[int, VoronoiMesh], cell_colors: np.ndarray):
    """Merge per-cell Voronoi meshes into one PyVista PolyData with per-face RGB."""
    all_vertices = []
    all_faces = []
    face_colors = []
    vertex_offset = 0

    for cell_idx in sorted(meshes.keys()):
        mesh = meshes[cell_idx]
        if mesh.vertices_3d.shape[0] == 0 or mesh.faces.shape[0] == 0:
            continue
        all_vertices.append(mesh.vertices_3d)
        adjusted = mesh.faces + vertex_offset
        all_faces.append(adjusted)
        rgb = cell_colors[cell_idx]
        face_colors.extend([rgb] * len(adjusted))
        vertex_offset += len(mesh.vertices_3d)

    if not all_vertices:
        return None, None

    vertices = np.vstack(all_vertices)
    faces_flat = []
    for tri in np.vstack(all_faces):
        faces_flat.extend([3, int(tri[0]), int(tri[1]), int(tri[2])])
    poly = pv.PolyData(vertices, np.asarray(faces_flat, dtype=np.int64))
    return poly, np.asarray(face_colors, dtype=float)


def extract_voronoi_ridge_segments(
    meshes: dict[int, VoronoiMesh], tol: float = 1e-4
) -> np.ndarray:
    """
    Unique polyhedron ridge segments (true cell-cell / tissue-boundary edges).

    Uses ConvexHull ridges stored on each mesh, deduplicated geometrically so
    shared edges between neighboring cells are drawn only once.
    """
    seen: set = set()
    segments: List[np.ndarray] = []

    for mesh in meshes.values():
        if mesh.vertices_3d.shape[0] == 0 or mesh.edges.shape[0] == 0:
            continue
        verts = mesh.vertices_3d
        for v0, v1 in mesh.edges:
            p0 = tuple(np.round(verts[v0] / tol).astype(np.int64))
            p1 = tuple(np.round(verts[v1] / tol).astype(np.int64))
            key = tuple(sorted((p0, p1)))
            if key in seen:
                continue
            seen.add(key)
            segments.append(np.stack([verts[v0], verts[v1]], axis=0))

    if not segments:
        return np.empty((0, 2, 3), dtype=np.float32)
    return np.asarray(segments, dtype=np.float32)


def extract_intercell_ridge_segments(
    meshes: dict[int, VoronoiMesh], tol: float = 1e-4
) -> np.ndarray:
    """
    Ridge segments shared by exactly two real cells (interior cell-cell borders).
    """
    edge_cells: dict = {}
    edge_geom: dict = {}

    for cell_idx, mesh in meshes.items():
        if mesh.vertices_3d.shape[0] == 0 or mesh.edges.shape[0] == 0:
            continue
        verts = mesh.vertices_3d
        for v0, v1 in mesh.edges:
            p0 = tuple(np.round(verts[v0] / tol).astype(np.int64))
            p1 = tuple(np.round(verts[v1] / tol).astype(np.int64))
            key = tuple(sorted((p0, p1)))
            edge_cells.setdefault(key, set()).add(cell_idx)
            edge_geom[key] = (verts[v0], verts[v1])

    segments = []
    for key, cells in edge_cells.items():
        if len(cells) == 2:
            p0, p1 = edge_geom[key]
            segments.append(np.stack([p0, p1], axis=0))

    if not segments:
        return np.empty((0, 2, 3), dtype=np.float32)
    return np.asarray(segments, dtype=np.float32)


def extract_voronoi_edges(meshes: dict[int, VoronoiMesh]) -> np.ndarray:
    """Return (E, 2) global vertex-index pairs for unique ridge segments."""
    segments = extract_voronoi_ridge_segments(meshes)
    if len(segments) == 0:
        return np.empty((0, 2), dtype=np.int64)
    # Flatten to line connectivity for PyVista
    return np.arange(len(segments) * 2, dtype=np.int64).reshape(-1, 2)


def draw_polarity_arrows(ax, origin, polarity_arrow_len=0.20):
    """ABP (red, up) and PCP (green, -30 deg) from a shared origin."""
    legend_origin = np.asarray(origin, dtype=float)
    abp_vec = np.array([0.0, 1.0])
    ax.add_patch(
        FancyArrowPatch(
            legend_origin,
            legend_origin + abp_vec * polarity_arrow_len,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.8,
            color="#d62728",
            zorder=10,
        )
    )
    ax.text(
        legend_origin[0] - 0.24,
        legend_origin[1] + 0.5 * polarity_arrow_len,
        "ABP",
        color="#d62728",
        fontsize=11,
        ha="left",
        va="center",
        zorder=10,
    )

    pcp_angle = np.radians(-30)
    pcp_dir = np.array([np.cos(pcp_angle), np.sin(pcp_angle)])
    ax.add_patch(
        FancyArrowPatch(
            legend_origin,
            legend_origin + pcp_dir * polarity_arrow_len,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.8,
            color="#2ca02c",
            zorder=10,
        )
    )
    ax.text(
        legend_origin[0] + pcp_dir[0] * polarity_arrow_len + 0.03,
        legend_origin[1] + pcp_dir[1] * polarity_arrow_len,
        "PCP",
        color="#2ca02c",
        fontsize=11,
        ha="left",
        va="center",
        zorder=10,
    )


def render_sphere_cluster(x, settings=None):
    """Render one (N, 3) position array to an RGBA image array."""
    settings = {**RENDER_SETTINGS, **(settings or {})}
    x = np.asarray(x, dtype=float)
    x, _ = _apply_rotation(x, None, settings.get("rotation"))
    center = x.mean(axis=0)

    sphere_radius = settings["sphere_radius"]
    if sphere_radius is None:
        dists, _ = cKDTree(x).query(x, k=2)
        sphere_radius = 0.6 * dists[:, 1].mean()

    window_size = settings["window_size"]
    pv.global_theme.transparent_background = True
    pl = pv.Plotter(window_size=window_size, off_screen=True)
    pl.set_background(None)
    pl.show_axes = False

    actors = []
    for pos in x:
        sphere = pv.Sphere(
            radius=sphere_radius,
            center=pos,
            theta_resolution=settings["sphere_resolution"],
            phi_resolution=settings["sphere_resolution"],
        )
        actors.append(
            pl.add_mesh(
                sphere,
                color="white",
                smooth_shading=True,
                specular=0.0,
                diffuse=1.0,
                ambient=settings["ambient"],
                lighting=True,
                show_edges=False,
            )
        )

    cam_pos = _setup_camera(pl, center, settings)
    base_colors = _distance_colors(x, cam_pos, settings)
    for actor, rgb in zip(actors, base_colors):
        actor.GetProperty().SetColor(*rgb)

    pl.render()
    img = pl.screenshot(transparent_background=True, return_img=True)
    pl.close()
    return img


def render_voronoi_cluster(x, p, settings=None):
    """Render Voronoi domains for (N,3) positions and polarities to RGBA image."""
    settings = {**RENDER_SETTINGS, **(settings or {})}
    x = np.asarray(x, dtype=float)
    p = np.asarray(p, dtype=float)
    x, p = _apply_rotation(x, p, settings.get("rotation"))
    center = x.mean(axis=0)

    meshes = build_voronoi_meshes(x, p, settings)
    if not meshes:
        raise RuntimeError("Voronoi mesh computation produced no cells")

    window_size = settings["window_size"]
    pv.global_theme.transparent_background = True
    pl = pv.Plotter(window_size=window_size, off_screen=True)
    pl.set_background(None)
    pl.show_axes = False

    pl.add_mesh(pv.PolyData(x), opacity=0.0, point_size=0.1)
    cam_pos = _setup_camera(pl, center, settings)
    pl.clear()

    base_colors = _cell_colors(x, cam_pos, settings)
    poly, face_colors = merge_voronoi_to_polydata(meshes, base_colors)
    if poly is None:
        raise RuntimeError("Failed to merge Voronoi meshes")

    pl.add_mesh(
        poly,
        scalars=face_colors,
        rgb=True,
        smooth_shading=True,
        specular=0.0,
        diffuse=1.0,
        ambient=settings["ambient"],
        lighting=True,
        show_edges=False,
    )

    if settings.get("show_voronoi_edges", True):
        edge_mode = settings.get("voronoi_edge_mode", "intercell")
        if edge_mode == "all_ridges":
            edge_segments = extract_voronoi_ridge_segments(meshes)
        else:
            edge_segments = extract_intercell_ridge_segments(meshes)
        if len(edge_segments) > 0:
            edge_color, edge_width = _voronoi_edge_style(settings)
            n_seg = len(edge_segments)
            points = edge_segments.reshape(-1, 3)
            lines = np.hstack([
                np.full((n_seg, 1), 2, dtype=np.int64),
                np.arange(0, n_seg * 2, 2)[:, None],
                np.arange(1, n_seg * 2, 2)[:, None],
            ]).ravel()
            edge_mesh = pv.PolyData(points, lines=lines)
            pl.add_mesh(
                edge_mesh,
                color=edge_color,
                line_width=edge_width,
                lighting=False,
                opacity=1.0,
            )

    _setup_camera(pl, center, settings, position=cam_pos)

    pl.render()
    img = pl.screenshot(transparent_background=True, return_img=True)
    pl.close()
    return img


def render_cluster_panel(x, p=None, mode="sphere", settings=None):
    """Unified entry: render one panel as sphere or voronoi cluster."""
    if mode == "sphere":
        return render_sphere_cluster(x, settings)
    if mode == "voronoi":
        if p is None:
            raise ValueError("Voronoi mode requires polarity vectors p")
        return render_voronoi_cluster(x, p, settings)
    raise ValueError(f"mode must be 'sphere' or 'voronoi', got {mode!r}")


def plot_radial_comparison(
    datasets,
    center_index=6,
    ring_radius=1.55,
    panel_zoom=0.42,
    arrow_inset=0.60,
    arrow_color="#333333",
    arrow_lw=1.4,
    polarity_offset=(0.14, 0.30),
    polarity_arrow_len=0.20,
    figsize=(11, 11),
    output_path=None,
    render_settings=None,
    render_mode="sphere",
    voronoi_edge_color=None,
    voronoi_edge_width=None,
):
    """
    Place 6 datasets on a ring and one in the center; arrows run center -> outer.

    Each dataset dict must have key "x". For render_mode="voronoi", also "p".

    voronoi_edge_color, voronoi_edge_width : optional overrides for inter-cell
        ridge lines (also settable via render_settings).
    """
    if len(datasets) != 7:
        raise ValueError(f"Expected 7 datasets, got {len(datasets)}")

    outer_indices = [i for i in range(7) if i != center_index]
    if len(outer_indices) != 6:
        raise ValueError("center_index must leave exactly 6 outer panels")

    base_render = _merge_render_settings(
        render_settings, voronoi_edge_color, voronoi_edge_width
    )
    rendered = []
    for item in datasets:
        panel_settings = {**base_render, **item.get("settings", {})}
        rendered.append(
            render_cluster_panel(
                item["x"],
                p=item.get("p"),
                mode=render_mode,
                settings=panel_settings,
            )
        )

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.set_aspect("equal")
    ax.axis("off")

    def panel_position(index):
        if index == center_index:
            return np.array([0.0, 0.0])
        ring_slot = outer_indices.index(index)
        angle = np.pi / 2 - ring_slot * 2 * np.pi / 6
        return ring_radius * np.array([np.cos(angle), np.sin(angle)])

    centers = {i: panel_position(i) for i in range(7)}

    for i, img in enumerate(rendered):
        imbox = OffsetImage(img, zoom=panel_zoom)
        ab = AnnotationBbox(
            imbox,
            centers[i],
            frameon=False,
            pad=0.0,
            box_alignment=(0.5, 0.5),
        )
        ax.add_artist(ab)

        label = datasets[i].get("label")
        if label:
            offset = 0.46 if i == center_index else 0.45
            ax.text(
                centers[i][0],
                centers[i][1] - offset,
                label,
                ha="center",
                va="top",
                fontsize=12,
            )

    center = centers[center_index]
    for i in outer_indices:
        target = centers[i]
        direction = target - center
        direction = direction / (np.linalg.norm(direction) + 1e-12)
        start = center + direction * arrow_inset
        end = target - direction * arrow_inset
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=arrow_lw,
                color=arrow_color,
                shrinkA=0,
                shrinkB=0,
                zorder=0,
            )
        )

    draw_polarity_arrows(
        ax, center + np.array(polarity_offset), polarity_arrow_len=polarity_arrow_len
    )

    pad = ring_radius + 1.0
    ax.set_xlim(-pad, pad)
    ax.set_ylim(-pad, pad)

    if output_path:
        fig.savefig(output_path, dpi=400, transparent=True, bbox_inches="tight", pad_inches=0.1)

    return fig


def plot_dual_basecase_comparison(
    groups,
    corner_radius=1.55,
    upper_base_y=0.42,
    lower_base_y=-0.30,
    lower_corner_shift=0.22,
    panel_zoom=0.42,
    base_panel_scale=0.80,
    arrow_inset=0.55,
    arrow_color="#333333",
    arrow_lw=1.4,
    polarity_offset_right=0.20,
    polarity_y_offset=-0.16,
    polarity_arrow_len=0.20,
    figsize=(11, 11),
    output_path=None,
    render_settings=None,
    render_mode="sphere",
    voronoi_edge_color=None,
    voronoi_edge_width=None,
):
    """
    Two basecases stacked in the centre; each has two arrows to corner transforms.

    For render_mode="voronoi", each base/transform dict needs "p" alongside "x".

    voronoi_edge_color, voronoi_edge_width : optional overrides for inter-cell
        ridge lines (also settable via render_settings).
    """
    if len(groups) != 2:
        raise ValueError(f"Expected 2 groups, got {len(groups)}")
    for gi, group in enumerate(groups):
        if len(group.get("transforms", [])) != 2:
            raise ValueError(f"Group {gi} must have exactly 2 transforms")

    base_render = _merge_render_settings(
        render_settings, voronoi_edge_color, voronoi_edge_width
    )

    corner_dirs = {
        "up_left": np.array([-1.0, 1.0]) / np.sqrt(2),
        "up_right": np.array([1.0, 1.0]) / np.sqrt(2),
        "down_left": np.array([-1.0, -1.0]) / np.sqrt(2),
        "down_right": np.array([1.0, -1.0]) / np.sqrt(2),
    }
    base_positions = [
        np.array([0.0, upper_base_y]),
        np.array([0.0, lower_base_y]),
    ]

    panel_specs = []
    for group_idx, group in enumerate(groups):
        base_pos = base_positions[group_idx]
        group_rotation = group.get("rotation")
        panel_specs.append(
            {
                "x": group["base"]["x"],
                "p": group["base"].get("p"),
                "label": group["base"].get("label"),
                "position": base_pos,
                "role": "base",
                "settings": {"rotation": group_rotation} if group_rotation else {},
            }
        )
        for tr in group["transforms"]:
            corner = tr["corner"]
            if corner not in corner_dirs:
                raise ValueError(f"Unknown corner {corner!r}")
            tr_settings = dict(tr.get("settings", {}))
            if group_rotation and "rotation" not in tr_settings:
                tr_settings["rotation"] = group_rotation
            corner_pos = corner_radius * corner_dirs[corner]
            if corner in ("down_left", "down_right"):
                corner_pos = corner_pos + np.array([0.0, lower_corner_shift])
            panel_specs.append(
                {
                    "x": tr["x"],
                    "p": tr.get("p"),
                    "label": tr.get("label"),
                    "position": corner_pos,
                    "role": "transform",
                    "base_position": base_pos,
                    "settings": tr_settings,
                }
            )

    rendered = []
    for spec in panel_specs:
        panel_settings = {**base_render, **spec.get("settings", {})}
        rendered.append(
            render_cluster_panel(
                spec["x"],
                p=spec.get("p"),
                mode=render_mode,
                settings=panel_settings,
            )
        )

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.set_aspect("equal")
    ax.axis("off")

    for spec, img in zip(panel_specs, rendered):
        zoom = panel_zoom * base_panel_scale if spec["role"] == "base" else panel_zoom
        imbox = OffsetImage(img, zoom=zoom)
        ax.add_artist(
            AnnotationBbox(
                imbox,
                spec["position"],
                frameon=False,
                pad=0.0,
                box_alignment=(0.5, 0.5),
            )
        )

        if spec.get("label"):
            y_off = 0.46 if spec["role"] == "base" else 0.45
            ax.text(
                spec["position"][0],
                spec["position"][1] - y_off,
                spec["label"],
                ha="center",
                va="top",
                fontsize=12,
            )

    upper_base = base_positions[0]
    upper_arrow_lengths = [
        np.linalg.norm(spec["position"] - spec["base_position"]) - 2 * arrow_inset
        for spec in panel_specs
        if spec["role"] == "transform"
        and np.allclose(spec["base_position"], upper_base)
    ]
    fixed_arrow_length = float(np.mean(upper_arrow_lengths))

    for spec in panel_specs:
        if spec["role"] != "transform":
            continue
        start = spec["base_position"]
        end = spec["position"]
        direction = end - start
        direction = direction / (np.linalg.norm(direction) + 1e-12)
        arrow_start = start + direction * arrow_inset
        arrow_end = arrow_start + direction * fixed_arrow_length
        ax.add_patch(
            FancyArrowPatch(
                arrow_start,
                arrow_end,
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=arrow_lw,
                color=arrow_color,
                shrinkA=0,
                shrinkB=0,
                zorder=0,
            )
        )

    draw_polarity_arrows(
        ax,
        np.array([
            polarity_offset_right,
            0.5 * (upper_base_y + lower_base_y) + polarity_y_offset,
        ]),
        polarity_arrow_len=polarity_arrow_len,
    )

    pad = corner_radius + 1.0
    ax.set_xlim(-pad, pad)
    ax.set_ylim(-pad, pad)

    if output_path:
        fig.savefig(output_path, dpi=400, transparent=True, bbox_inches="tight", pad_inches=0.1)

    return fig


# Backward-compatible aliases
plot_radial_sphere_comparison = plot_radial_comparison


def _schematic_arrow(ax, start, end, color="#333333", lw=1.6):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=lw,
            color=color,
            zorder=5,
        )
    )


def _style_schematic_ax(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")


def load_svg_as_array(svg_path, dpi: int = 200) -> np.ndarray:
    """Rasterize an SVG file to an RGBA numpy array for matplotlib imshow."""
    path = Path(svg_path)
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        import cairosvg
    except ImportError as exc:
        raise ImportError(
            "SVG schematics require cairosvg (pip install cairosvg)"
        ) from exc
    png_bytes = cairosvg.svg2png(url=str(path.resolve()), dpi=dpi)
    return plt.imread(BytesIO(png_bytes), format="png")


def _resolve_schematic_svgs(
    columns: List[dict], schematic_svgs: Optional[List[str]] = None
) -> List[Optional[str]]:
    """Per-column SVG paths from schematic_svgs list or column['schematic_svg']."""
    if schematic_svgs is not None:
        if len(schematic_svgs) != len(columns):
            raise ValueError(
                f"schematic_svgs length {len(schematic_svgs)} must match "
                f"columns length {len(columns)}"
            )
        return list(schematic_svgs)
    return [col.get("schematic_svg") for col in columns]


def _show_schematic_on_ax(ax, img: np.ndarray, scale: float = 1.0):
    """Display a schematic image scaled by `scale` (>1 grows it, overflowing the cell)."""
    scale = max(float(scale), 0.1)
    half = 0.5 * scale
    im = ax.imshow(
        img,
        extent=(0.5 - half, 0.5 + half, 0.5 - half, 0.5 + half),
        aspect="equal",
        origin="upper",
    )
    # Let the image grow past the axes bounds instead of being clipped by them.
    im.set_clip_on(False)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.margins(0)


def draw_deformation_schematic(ax, kind: str, scale: float = 1.0):
    """
    Draw a simple deformation schematic in ax.

    kind: "curve" | "cylinder_sq" | "sphere_sq" | "sphere_pu"
    """
    _style_schematic_ax(ax)
    outline = "#222222"
    fill = "#f0f0f0"

    if kind == "curve":
        y_top = 0.86
        x_left, x_right = 0.12, 0.88
        meet_y = 0.40
        cx = 0.50
        radius = 0.38
        theta = np.linspace(0.0, np.pi, 64)
        ax.plot(
            cx + radius * np.cos(theta),
            y_top - radius * np.sin(theta),
            color=outline,
            linewidth=1.8,
            zorder=2,
        )
        ax.plot([x_left, x_right], [y_top, y_top], color=outline, linewidth=2.0, zorder=3)
        meet = (cx, meet_y)
        ax.add_patch(
            FancyArrowPatch(
                (x_left, y_top),
                meet,
                connectionstyle="arc3,rad=-0.5",
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=1.5,
                color="#333333",
                zorder=5,
            )
        )
        ax.add_patch(
            FancyArrowPatch(
                (x_right, y_top),
                meet,
                connectionstyle="arc3,rad=0.5",
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=1.5,
                color="#333333",
                zorder=5,
            )
        )
        _schematic_arrow(ax, (0.46, meet_y - 0.04), (0.46, 0.12))
        _schematic_arrow(ax, (0.54, meet_y - 0.04), (0.54, 0.12))

    elif kind == "cylinder_sq":
        y_lo, y_hi = 0.16, 0.84
        ax.plot([0.10, 0.10], [y_lo, y_hi], color=outline, linewidth=2.0, zorder=2)
        ax.plot([0.90, 0.90], [y_lo, y_hi], color=outline, linewidth=2.0, zorder=2)
        xl, xr = 0.43, 0.57
        y_open = 0.30
        y_cap = 0.52
        ax.plot([xl, xl], [y_cap, y_hi], color=outline, linewidth=2.0, zorder=3)
        ax.plot([xr, xr], [y_cap, y_hi], color=outline, linewidth=2.0, zorder=3)
        ax.add_patch(
            Ellipse(
                (0.50, y_open),
                xr - xl + 0.04,
                0.14,
                fill=False,
                edgecolor=outline,
                linewidth=1.8,
                zorder=3,
            )
        )
        _schematic_arrow(ax, (0.12, 0.50), (0.38, 0.50))
        _schematic_arrow(ax, (0.88, 0.50), (0.62, 0.50))

    elif kind == "sphere_sq":
        ax.add_patch(
            Circle(
                (0.50, 0.50),
                0.26,
                facecolor=fill,
                edgecolor=outline,
                linewidth=1.8,
                zorder=1,
            )
        )
        ax.plot([0.08, 0.08], [0.16, 0.84], color=outline, linewidth=2.0, zorder=2)
        ax.plot([0.92, 0.92], [0.16, 0.84], color=outline, linewidth=2.0, zorder=2)
        _schematic_arrow(ax, (0.10, 0.50), (0.22, 0.50))
        _schematic_arrow(ax, (0.90, 0.50), (0.78, 0.50))

    elif kind == "sphere_pu":
        ax.add_patch(
            Circle(
                (0.50, 0.50),
                0.26,
                facecolor=fill,
                edgecolor=outline,
                linewidth=1.8,
                zorder=1,
            )
        )
        _schematic_arrow(ax, (0.50, 0.76), (0.50, 0.92))
        _schematic_arrow(ax, (0.50, 0.24), (0.50, 0.08))

    else:
        raise ValueError(
            f"kind must be 'curve', 'cylinder_sq', 'sphere_sq', or 'sphere_pu', got {kind!r}"
        )
    if scale != 1.0:
        # Grow the drawing past the axes bounds (no clipping) instead of zooming
        # the view window, which would crop the illustration at the cell edges.
        for artist in list(ax.patches) + list(ax.lines):
            artist.set_clip_on(False)
        half = 0.5 / max(float(scale), 0.1)
        ax.set_xlim(0.5 - half, 0.5 + half)
        ax.set_ylim(0.5 - half, 0.5 + half)


def _scalar_listed_colormap() -> ListedColormap:
    n = 256
    vals = np.linspace(0.0, 1.0, n)
    rgba = scalar_to_rgba(vals, 0.0, 1.0, None)
    return ListedColormap(rgba[:, :3])


def _draw_scalar_colorbar(
    fig,
    gs_slot,
    vmin: float,
    vmax: float,
    scalar_key: str = "alpha_par",
    scalar_vmin_linear: Optional[float] = None,
    scalar_vmax_linear: Optional[float] = None,
):
    """Draw colorbar in normalization space; gamma uses log ticks at linear values."""
    ax = fig.add_subplot(gs_slot)
    cmap = _scalar_listed_colormap()
    if _is_logspace_scalar(scalar_key):
        vcenter = _scalar_vcenter(scalar_key)
        if vcenter is not None and vmin < vcenter < vmax:
            norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
            ticks = [vmin, vcenter, vmax]
            if scalar_vmin_linear is not None and scalar_vmax_linear is not None:
                labels = [
                    f"{scalar_vmin_linear:g}",
                    "1.0",
                    f"{scalar_vmax_linear:g}",
                ]
            else:
                labels = [f"{np.exp(t):.3g}" for t in ticks]
                labels[1] = "1.0"
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)
            ticks = [vmin, vmax]
            labels = [f"{np.exp(t):.3g}" for t in ticks]
        cb = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=ax,
            orientation="vertical",
        )
        cb.set_label(r"$\gamma$", fontsize=10)
        cb.set_ticks(ticks)
        cb.set_ticklabels(labels)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)
        cb = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=ax,
            orientation="vertical",
        )
        cb.set_label(r"$\alpha$ (°)", fontsize=10)
    cb.ax.tick_params(labelsize=8)


def plot_deformation_comparison(
    columns,
    render_settings=None,
    output_path=None,
    figsize=(16, 10),
    voronoi_edge_color=None,
    voronoi_edge_width=None,
    scalar_vmin: float = -60.0,
    scalar_vmax: float = 60.0,
    panel_window_size: Tuple[int, int] = (720, 720),
    schematic_svgs: Optional[List[str]] = None,
    schematic_dpi: int = 200,
    schematic_scale: float = 1.0,
):
    """
    Deformation grid: schematics + scalar-colored Voronoi panels (fig3/fig4).

    Each entry in `columns` must provide:
        scalar     : e.g. "alpha_par", "alpha_perp", or "gamma"
        frames     : [initial_idx, during_idx, post_idx]
        data       : loaded pkl dict
        key        : optional deformation id for matplotlib fallback schematics
        rotation   : optional {"axis": "x", "degrees": 90}
        schematic_svg / schematic_scale : optional per-column overrides

    schematic_svgs : optional list of SVG paths (one per column) for row 0.
    scalar_vmin/vmax : linear scale (alpha degrees, or gamma e.g. 0.5–1.5).
    """
    if len(columns) < 1:
        raise ValueError(f"Expected at least 1 column, got {len(columns)}")

    n_cols = len(columns)
    scalar_key = columns[0]["scalar"]

    base_render = _merge_render_settings(
        render_settings, voronoi_edge_color, voronoi_edge_width
    )
    base_render["window_size"] = panel_window_size
    if "zoom_margin" not in (render_settings or {}):
        base_render["zoom_margin"] = 0.06
    vmin_linear, vmax_linear = float(scalar_vmin), float(scalar_vmax)
    vmin, vmax = _scalar_norm_bounds(scalar_key, vmin_linear, vmax_linear)
    svg_paths = _resolve_schematic_svgs(columns, schematic_svgs)
    schematic_images = []
    for col_idx, path in enumerate(svg_paths):
        if not path:
            schematic_images.append(None)
            continue
        col_scale = columns[col_idx].get("schematic_scale", schematic_scale)
        load_dpi = int(schematic_dpi * max(1.0, float(col_scale)))
        schematic_images.append(load_svg_as_array(path, dpi=load_dpi))

    rendered = [[None] * n_cols for _ in range(4)]
    for col_idx, col in enumerate(columns):
        col_scalar = col["scalar"]
        col_label = col.get("key", f"col_{col_idx}")
        rotation = col.get("rotation")
        data = col["data"]
        frames = col["frames"]
        if len(frames) != 3:
            raise ValueError(f"Column {col_label!r} must have exactly 3 frame indices")

        for row_idx, frame in enumerate(frames, start=1):
            panel_settings = {
                **base_render,
                "color_mode": "scalar",
                "scalar_key": col_scalar,
                "scalar_values": np.asarray(data[col_scalar][frame], dtype=np.float32),
                "scalar_vmin": vmin,
                "scalar_vmax": vmax,
            }
            if rotation:
                panel_settings["rotation"] = rotation
            rendered[row_idx][col_idx] = render_voronoi_cluster(
                data["x"][frame],
                data["p"][frame],
                settings=panel_settings,
            )

    width_ratios = [0.10, 0.78] + [1.28] * n_cols + [0.12]
    height_ratios = [0.62, 1.12, 1.12, 1.12]
    n_gs_cols = 2 + n_cols + 1
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        4,
        n_gs_cols,
        width_ratios=width_ratios,
        height_ratios=height_ratios,
        wspace=0.015,
        hspace=0.015,
    )
    fig.patch.set_facecolor("white")

    ax_time = fig.add_subplot(gs[1:4, 0])
    ax_time.set_xlim(0, 1)
    ax_time.set_ylim(0, 1)
    ax_time.axis("off")
    ax_time.add_patch(
        FancyArrowPatch(
            (0.55, 0.92),
            (0.55, 0.08),
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2.0,
            color="#333333",
            zorder=2,
        )
    )
    ax_time.text(
        0.08,
        0.50,
        "Time",
        rotation=90,
        va="center",
        ha="center",
        fontsize=12,
        color="#333333",
    )

    for row_idx, label in enumerate(FIG3_ROW_LABELS):
        ax_lbl = fig.add_subplot(gs[row_idx, 1])
        ax_lbl.set_xlim(0, 1)
        ax_lbl.set_ylim(0, 1)
        ax_lbl.axis("off")
        ax_lbl.text(
            0.0,
            0.50,
            label,
            ha="left",
            va="center",
            fontsize=9.5,
        )

        for col_idx, col in enumerate(columns):
            ax = fig.add_subplot(gs[row_idx, col_idx + 2])
            ax.set_aspect("equal")
            ax.axis("off")
            ax.margins(0)
            if row_idx == 0:
                col_scale = col.get("schematic_scale", schematic_scale)
                schematic_img = schematic_images[col_idx]
                if schematic_img is not None:
                    _show_schematic_on_ax(ax, schematic_img, scale=col_scale)
                elif col.get("key"):
                    draw_deformation_schematic(ax, col["key"], scale=col_scale)
            else:
                ax.imshow(rendered[row_idx][col_idx], aspect="equal")

    cb_col = 2 + n_cols
    _draw_scalar_colorbar(
        fig,
        gs[1:4, cb_col],
        vmin,
        vmax,
        scalar_key=scalar_key,
        scalar_vmin_linear=vmin_linear if _is_logspace_scalar(scalar_key) else None,
        scalar_vmax_linear=vmax_linear if _is_logspace_scalar(scalar_key) else None,
    )

    if output_path:
        fig.savefig(output_path, dpi=400, bbox_inches="tight", pad_inches=0.03)

    return fig
