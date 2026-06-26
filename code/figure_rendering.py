"""
Reusable PyVista + matplotlib rendering for publication figure panels.

Supports sphere and augmented-Voronoi cluster visualization with shared
camera, distance coloring, and layout helpers for multi-panel figures.
"""

from __future__ import annotations

from typing import List

import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyArrowPatch
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
}


def _merge_render_settings(
    render_settings=None,
    voronoi_edge_color=None,
    voronoi_edge_width=None,
):
    """Merge optional Voronoi edge overrides into render_settings."""
    settings = dict(render_settings or {})
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

    base_colors = _distance_colors(x, cam_pos, settings)
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
