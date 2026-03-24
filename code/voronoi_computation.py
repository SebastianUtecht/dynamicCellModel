"""
Augmented 3D Voronoi Computation and Mesh Generation

Computes 3D Voronoi diagrams with augmented point sets to create bounded,
tiling tessellations. For each cell, virtual cells are placed above and below
(along the polarity direction) to naturally bound the Voronoi regions.
"""

import numpy as np
from scipy.spatial import Voronoi as ScipyVoronoi, ConvexHull
from typing import Dict, Optional


class VoronoiMesh:
    """
    Container for a single cell's 3D Voronoi mesh.

    Stores triangle vertices and indices, along with rendering properties.
    """

    def __init__(
        self,
        cell_idx: int,
        vertices_3d: np.ndarray,
        faces: np.ndarray,
        color: np.ndarray = None,
    ):
        """
        Parameters
        ----------
        cell_idx : int
            Index of the cell this mesh represents
        vertices_3d : (V, 3) ndarray
            Triangle vertex positions in 3D
        faces : (F, 3) ndarray
            Triangle indices (CCW winding)
        color : (4,) ndarray, optional
            RGBA color for mesh rendering
        """
        self.cell_idx = cell_idx
        self.vertices_3d = np.asarray(vertices_3d, dtype=np.float32)
        self.faces = np.asarray(faces, dtype=np.uint32)
        self.color = np.asarray(color, dtype=np.float32) if color is not None else None
        self.is_visible = True

    def to_tuple(self) -> tuple:
        """Return (vertices, faces) for vispy rendering."""
        return self.vertices_3d, self.faces


class AugmentedVoronoi:
    """
    Compute 3D Voronoi domains using augmented cell positions.

    For each cell, creates virtual cells above and below (along the polarity
    direction), computes 3D Voronoi, then extracts only the original cell domains.

    This approach naturally creates bounded, tiling tessellations without
    infinite regions or complex edge handling.
    """

    def __init__(
        self,
        x: np.ndarray,
        p: np.ndarray,
        thickness: float = 2.0,
    ):
        """
        Initialize augmented Voronoi computer.

        Parameters
        ----------
        x : (N, 3) ndarray
            Cell center positions
        p : (N, 3) ndarray
            Polarity vectors (should be normalized, but will be normalized here)
        thickness : float
            Fixed distance to place virtual cells above/below along polarity
            (default: 2.0 units)
        """
        self.x = np.asarray(x, dtype=np.float32)
        self.p = np.asarray(p, dtype=np.float32)
        self.n_cells = len(self.x)
        self.thickness = thickness

        # Validate inputs
        if self.x.shape[0] != self.p.shape[0]:
            raise ValueError("x and p must have same number of cells")

        # Normalize polarity vectors
        p_norm = np.linalg.norm(self.p, axis=1, keepdims=True)
        p_norm = np.where(p_norm > 1e-8, p_norm, 1.0)
        self.p = self.p / p_norm

    def compute_voronoi_meshes(self) -> Dict[int, VoronoiMesh]:
        """
        Compute Voronoi meshes for all original cells.

        Returns
        -------
        meshes : dict {cell_idx: VoronoiMesh}
            Voronoi mesh for each original cell
        """
        # Create augmented point set: original cells, cells below, cells above
        x_below = self.x - self.thickness * self.p
        x_above = self.x + self.thickness * self.p
        x_augmented = np.vstack([self.x, x_below, x_above])

        # Compute 3D Voronoi on augmented set
        try:
            voronoi = ScipyVoronoi(x_augmented)
        except Exception as e:
            print(f"[AugmentedVoronoi] Failed to compute Voronoi: {e}")
            return {}

        meshes = {}

        # Extract regions for original cells only (indices 0..n_cells-1)
        # Skip virtual cells (indices n_cells..3*n_cells-1)
        for cell_idx in range(self.n_cells):
            try:
                region_idx = voronoi.point_region[cell_idx]
                region = voronoi.regions[region_idx]

                # Check for infinite region (shouldn't happen with augmented cells,
                # but handle gracefully)
                if -1 in region:
                    print(
                        f"[AugmentedVoronoi] Cell {cell_idx} has infinite region (skipping)"
                    )
                    continue

                # Get vertices of this Voronoi cell
                vertices = voronoi.vertices[region]

                if len(vertices) < 4:
                    # Too few vertices to form a 3D polyhedron
                    continue

                # Triangulate the Voronoi cell (convex polyhedron)
                faces = self._triangulate_polyhedron(vertices)

                if len(faces) > 0:
                    mesh = VoronoiMesh(cell_idx, vertices, faces)
                    meshes[cell_idx] = mesh
            except Exception as e:
                print(f"[AugmentedVoronoi] Failed for cell {cell_idx}: {e}")
                continue

        return meshes

    @staticmethod
    def _triangulate_polyhedron(vertices: np.ndarray) -> np.ndarray:
        """
        Triangulate a convex polyhedron's surface.

        Given the vertices of a convex polyhedron, returns the triangulated
        surface mesh using ConvexHull. This is exactly what we need for
        rendering the Voronoi cell boundaries.

        Parameters
        ----------
        vertices : (V, 3) ndarray
            Vertices of the convex polyhedron

        Returns
        -------
        faces : (F, 3) ndarray
            Triangle face indices (F faces, 3 vertices each)
        """
        try:
            hull = ConvexHull(vertices)
            return hull.simplices.astype(np.uint32)
        except Exception as e:
            print(f"[AugmentedVoronoi] ConvexHull failed: {e}")
            return np.empty((0, 3), dtype=np.uint32)
