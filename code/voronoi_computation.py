"""
Augmented 3D Voronoi Computation and Mesh Generation

Computes 3D Voronoi diagrams with augmented point sets to create bounded,
tiling tessellations. For each cell, virtual cells are placed above and below
(along the polarity direction) to naturally bound the Voronoi regions.
Optional lateral boundary sites cap tissue-border extensions with planar faces.
"""

import numpy as np
from scipy.spatial import Voronoi as ScipyVoronoi, ConvexHull, cKDTree
from typing import Dict, List, Optional, Tuple


class VoronoiMesh:
    """
    Container for a single cell's 3D Voronoi mesh.

    Stores triangle vertices, polyhedron ridge edges, and rendering properties.
    """

    def __init__(
        self,
        cell_idx: int,
        vertices_3d: np.ndarray,
        faces: np.ndarray,
        edges: np.ndarray = None,
        color: np.ndarray = None,
    ):
        """
        Parameters
        ----------
        cell_idx : int
            Index of the cell this mesh represents
        vertices_3d : (V, 3) ndarray
            Polyhedron vertex positions in 3D
        faces : (F, 3) ndarray
            Triangle indices for rendering (CCW winding)
        edges : (E, 2) ndarray, optional
            Polyhedron ridge edges (true Voronoi edges, not triangulation diagonals)
        color : (4,) ndarray, optional
            RGBA color for mesh rendering
        """
        self.cell_idx = cell_idx
        self.vertices_3d = np.asarray(vertices_3d, dtype=np.float32)
        self.faces = np.asarray(faces, dtype=np.uint32)
        self.edges = (
            np.asarray(edges, dtype=np.uint32)
            if edges is not None
            else np.empty((0, 2), dtype=np.uint32)
        )
        self.color = np.asarray(color, dtype=np.float32) if color is not None else None
        self.is_visible = True

    def to_tuple(self) -> tuple:
        """Return (vertices, faces) for vispy rendering."""
        return self.vertices_3d, self.faces


class AugmentedVoronoi:
    """
    Compute 3D Voronoi domains using augmented cell positions.

    For each cell, creates virtual cells above and below (along the polarity
    direction), optionally adds lateral hull boundary sites, computes 3D
    Voronoi, then extracts only the original cell domains.
    """

    def __init__(
        self,
        x: np.ndarray,
        p: np.ndarray,
        thickness: float = 2.0,
        border_cap: Optional[float] = None,
        enable_lateral_boundary: bool = True,
        neighbor_search_factor: float = 1.6,
        neighbor_search_radius: Optional[float] = None,
    ):
        """
        Parameters
        ----------
        x : (N, 3) ndarray
            Cell center positions
        p : (N, 3) ndarray
            Polarity vectors (normalized internally)
        thickness : float
            Distance for apicobasal virtual cells along polarity
        border_cap : float, optional
            Distance for lateral boundary virtual sites (planar tissue edge)
        enable_lateral_boundary : bool
            Add hull-facet virtual neighbors for lateral bounding
        neighbor_search_factor : float
            Lateral neighbor search radius = factor × mean nearest-neighbor spacing
        neighbor_search_radius : float, optional
            Absolute neighbor search radius; overrides neighbor_search_factor if set
        """
        self.x = np.asarray(x, dtype=np.float32)
        self.p = np.asarray(p, dtype=np.float32)
        self.n_cells = len(self.x)
        self.thickness = thickness
        self.border_cap = border_cap
        self.enable_lateral_boundary = enable_lateral_boundary
        self.neighbor_search_factor = neighbor_search_factor
        self.neighbor_search_radius = neighbor_search_radius

        if self.x.shape[0] != self.p.shape[0]:
            raise ValueError("x and p must have same number of cells")

        p_norm = np.linalg.norm(self.p, axis=1, keepdims=True)
        p_norm = np.where(p_norm > 1e-8, p_norm, 1.0)
        self.p = self.p / p_norm

    def _mean_neighbor_spacing(self) -> float:
        if self.n_cells < 2:
            return 1.0
        dists, _ = cKDTree(self.x).query(self.x, k=2)
        return float(np.mean(dists[:, 1]))

    def _local_tangent_basis(self, cell_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Orthonormal basis (u, v) spanning the plane perpendicular to ABP."""
        p_dir = self.p[cell_idx]
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if abs(float(np.dot(p_dir, ref))) > 0.9:
            ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        u = np.cross(p_dir, ref)
        u = u / (np.linalg.norm(u) + 1e-12)
        v = np.cross(p_dir, u)
        v = v / (np.linalg.norm(v) + 1e-12)
        return u, v

    def _neighbor_search_radius(self) -> float:
        if self.neighbor_search_radius is not None:
            return float(self.neighbor_search_radius)
        return self.neighbor_search_factor * self._mean_neighbor_spacing()

    def _exposed_lateral_directions(
        self, cell_idx: int, n_dirs: int = 16, cone_cos: float = 0.5
    ) -> List[np.ndarray]:
        """
        Directions in the plane perpendicular to ABP where the cell has no neighbor
        within the configured search radius.
        """
        center = self.x[cell_idx]
        search_r = self._neighbor_search_radius()
        u, v = self._local_tangent_basis(cell_idx)
        exposed: List[np.ndarray] = []

        for k in range(n_dirs):
            angle = 2.0 * np.pi * k / n_dirs
            direction = np.cos(angle) * u + np.sin(angle) * v
            has_neighbor = False
            for j in range(self.n_cells):
                if j == cell_idx:
                    continue
                rel = self.x[j] - center
                dist = float(np.linalg.norm(rel))
                if dist < 1e-8 or dist > search_r:
                    continue
                if float(np.dot(rel / dist, direction)) > cone_cos:
                    has_neighbor = True
                    break
            if not has_neighbor:
                exposed.append(direction)
        return exposed

    def _exposed_lateral_directions_2d(
        self, cell_idx: int, n_dirs: int = 16
    ) -> List[np.ndarray]:
        """Backward-compatible wrapper returning xy components where possible."""
        dirs = self._exposed_lateral_directions(cell_idx, n_dirs=n_dirs)
        return [d[:2] / (np.linalg.norm(d[:2]) + 1e-12) for d in dirs if np.linalg.norm(d[:2]) > 1e-8]

    def _lateral_boundary_sites_2d(self) -> np.ndarray:
        """Per-border-cell lateral virtual sites in the epithelial plane."""
        sites: List[np.ndarray] = []
        seen: set = set()

        for i in range(self.n_cells):
            for direction in self._exposed_lateral_directions(i):
                site = self.x[i] + self.border_cap * direction
                key = tuple(np.round(site, 4))
                if key in seen:
                    continue
                seen.add(key)
                sites.append(site)

        if not sites:
            return np.empty((0, 3), dtype=np.float32)
        return np.asarray(sites, dtype=np.float32)

    def _lateral_boundary_sites_3d(self) -> np.ndarray:
        """Per-border-cell lateral sites in the plane perpendicular to ABP."""
        return self._lateral_boundary_sites_2d()

    def _lateral_boundary_sites(self) -> np.ndarray:
        """
        Virtual sites that create planar lateral tissue boundaries.

        Uses a 2D hull for coplanar epithelial sheets and a 3D hull otherwise.
        """
        if not self.enable_lateral_boundary or self.border_cap is None:
            return np.empty((0, 3), dtype=np.float32)

        z_range = float(np.ptp(self.x[:, 2])) if self.x.shape[1] > 2 else 0.0
        xy_span = float(max(np.ptp(self.x[:, 0]), np.ptp(self.x[:, 1])))
        if xy_span > 1e-8 and z_range < 0.05 * xy_span:
            return self._lateral_boundary_sites_2d()
        return self._lateral_boundary_sites_3d()

    @staticmethod
    def _ridge_direction(
        voronoi: ScipyVoronoi, ridge_points: np.ndarray, ridge_vertices: List[int]
    ) -> Optional[np.ndarray]:
        """Unit direction along an infinite Voronoi ridge (away from diagram interior)."""
        i, j = int(ridge_points[0]), int(ridge_points[1])
        t = voronoi.points[j] - voronoi.points[i]
        t_norm = np.linalg.norm(t)
        if t_norm < 1e-8:
            return None
        t = t / t_norm

        for axis in ([0, 0, 1], [0, 1, 0], [1, 0, 0]):
            n = np.cross(t, np.asarray(axis, dtype=float))
            n_norm = np.linalg.norm(n)
            if n_norm > 1e-8:
                n = n / n_norm
                midpoint = 0.5 * (voronoi.points[i] + voronoi.points[j])
                if np.dot(midpoint - voronoi.points[i], n) < 0:
                    n = -n
                return n
        return None

    def _cap_infinite_ridge_vertex(
        self,
        voronoi: ScipyVoronoi,
        ridge_points: np.ndarray,
        ridge_vertices: List[int],
        cap_distance: float,
        cell_idx: int,
    ) -> Optional[np.ndarray]:
        """
        Cap an infinite ridge close to the owning cell in the lateral plane.

        Extends from the cell center along the ridge direction (projected to be
        perpendicular to ABP) so border cells do not shoot outward indefinitely.
        """
        direction = self._ridge_direction(voronoi, ridge_points, ridge_vertices)
        if direction is None:
            return None

        p_dir = self.p[cell_idx]
        lat = direction - np.dot(direction, p_dir) * p_dir
        lat_norm = float(np.linalg.norm(lat))
        if lat_norm > 1e-8:
            direction = lat / lat_norm
        else:
            direction = direction / (np.linalg.norm(direction) + 1e-12)

        return self.x[cell_idx] + cap_distance * direction

    def _clip_lateral_vertices(
        self, vertices: np.ndarray, cell_idx: int
    ) -> np.ndarray:
        """Clamp vertices that protrude too far laterally from a border cell."""
        if self.border_cap is None:
            return vertices

        exposed = self._exposed_lateral_directions(cell_idx)
        if not exposed:
            return vertices

        center = self.x[cell_idx]
        p_dir = self.p[cell_idx]
        clipped = vertices.copy()
        for direction in exposed:
            lat = direction - np.dot(direction, p_dir) * p_dir
            norm = float(np.linalg.norm(lat))
            if norm < 1e-8:
                continue
            lat = lat / norm
            for k, v in enumerate(clipped):
                outward = float(np.dot(v - center, lat))
                if outward > self.border_cap:
                    abp = float(np.dot(v - center, p_dir))
                    clipped[k] = center + self.border_cap * lat + abp * p_dir
        return clipped

    def _extract_region_vertices(
        self, voronoi: ScipyVoronoi, cell_idx: int
    ) -> Optional[np.ndarray]:
        """Extract Voronoi vertices for one cell, including capped infinite ridges."""
        region_idx = voronoi.point_region[cell_idx]
        region = voronoi.regions[region_idx]
        cap = self.border_cap if self.border_cap is not None else self.thickness

        if -1 not in region:
            vertices = voronoi.vertices[region].copy()
        else:
            verts_list = [voronoi.vertices[v] for v in region if v != -1]
            for ridge_p, ridge_v in zip(voronoi.ridge_points, voronoi.ridge_vertices):
                if cell_idx not in ridge_p or -1 not in ridge_v:
                    continue
                capped = self._cap_infinite_ridge_vertex(
                    voronoi, ridge_p, ridge_v, cap, cell_idx
                )
                if capped is not None:
                    verts_list.append(capped)

            if len(verts_list) < 4:
                return None
            vertices = np.asarray(verts_list, dtype=np.float32)

        vertices = self._clip_lateral_vertices(vertices, cell_idx)

        if len(vertices) < 4:
            return None
        return vertices

    def compute_voronoi_meshes(self) -> Dict[int, VoronoiMesh]:
        """Compute Voronoi meshes for all original cells."""
        x_below = self.x - self.thickness * self.p
        x_above = self.x + self.thickness * self.p
        lateral = self._lateral_boundary_sites()
        parts = [self.x, x_below, x_above]
        if len(lateral) > 0:
            parts.append(lateral)
        x_augmented = np.vstack(parts)

        try:
            voronoi = ScipyVoronoi(x_augmented)
        except Exception as e:
            print(f"[AugmentedVoronoi] Failed to compute Voronoi: {e}")
            return {}

        meshes = {}
        for cell_idx in range(self.n_cells):
            try:
                vertices = self._extract_region_vertices(voronoi, cell_idx)
                if vertices is None:
                    print(
                        f"[AugmentedVoronoi] Cell {cell_idx} could not be meshed (skipping)"
                    )
                    continue

                faces, edges = self._triangulate_polyhedron(vertices)
                if len(faces) > 0:
                    meshes[cell_idx] = VoronoiMesh(cell_idx, vertices, faces, edges)
            except Exception as e:
                print(f"[AugmentedVoronoi] Failed for cell {cell_idx}: {e}")
                continue

        return meshes

    @staticmethod
    def _polyhedron_ridges(hull: ConvexHull) -> np.ndarray:
        """True polyhedron edges from a 3D ConvexHull (excludes face triangulation diagonals)."""
        ridges = set()
        for i, neigh in enumerate(hull.neighbors):
            for j in neigh:
                if j == -1 or j <= i:
                    continue
                shared = set(hull.simplices[i]) & set(hull.simplices[j])
                if len(shared) == 2:
                    ridges.add(tuple(sorted(shared)))
        if not ridges:
            return np.empty((0, 2), dtype=np.uint32)
        return np.array(list(ridges), dtype=np.uint32)

    @staticmethod
    def _triangulate_polyhedron(vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Triangulate a convex polyhedron; return triangle faces and ridge edges.
        """
        try:
            hull = ConvexHull(vertices)
            faces = hull.simplices.astype(np.uint32)
            edges = AugmentedVoronoi._polyhedron_ridges(hull)
            return faces, edges
        except Exception as e:
            print(f"[AugmentedVoronoi] ConvexHull failed: {e}")
            return np.empty((0, 3), dtype=np.uint32), np.empty((0, 2), dtype=np.uint32)
