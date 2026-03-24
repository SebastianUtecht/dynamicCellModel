"""
Manifold Surface Detection and Tangent Plane Generation

Detects tissue surface topology from cell point clouds and polarity vectors,
generating local 2D coordinate systems (tangent planes) for Voronoi computation
on manifolds embedded in 3D space.
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.linalg import svd


class ManifoldSurface:
    """
    Detects a 2D tissue surface embedded in 3D space and generates local
    tangent plane coordinate systems for each cell.

    The surface is detected by combining local PCA with apicobasal polarity
    vectors, providing a biologically-informed surface normal estimate.
    """

    def __init__(self, x: np.ndarray, p_normals: np.ndarray, neighbor_k: int = 20):
        """
        Initialize surface detector.

        Parameters
        ----------
        x : (N, 3) ndarray
            Cell center positions
        p_normals : (N, 3) ndarray
            Apicobasal polarity vectors (should be normalized)
        neighbor_k : int
            Number of nearest neighbors to use for local surface estimation
        """
        self.x = np.asarray(x, dtype=np.float32)
        self.p_normals = np.asarray(p_normals, dtype=np.float32)
        self.neighbor_k = neighbor_k
        self.n_cells = len(self.x)

        # Validate inputs
        if self.x.shape[0] != self.p_normals.shape[0]:
            raise ValueError("x and p_normals must have same number of cells")

        # Normalize polarity vectors
        p_norm = np.linalg.norm(self.p_normals, axis=1, keepdims=True)
        p_norm = np.where(p_norm > 1e-8, p_norm, 1.0)
        self.p_normals = self.p_normals / p_norm

        self.surface_normals = None
        self.tangent_bases = None

    def detect_surface(self) -> tuple:
        """
        Detect tissue surface normals using local PCA combined with polarity.

        Algorithm:
        1. Build kD-tree and find k-nearest neighbors for each cell
        2. Perform local PCA on neighborhoods to estimate local surface normal
        3. Combine with polarity vector: n_final = α*n_pca + (1-α)*n_polarity
        4. Normalize resulting normals

        Returns
        -------
        surface_normals : (N, 3) ndarray
            Surface normal vectors (pointing outward from tissue)
        """
        # Build spatial index
        tree = cKDTree(self.x)

        # Query neighbors (including self)
        _, neighbor_indices = tree.query(self.x, k=self.neighbor_k + 1, workers=-1)

        # Remove self from neighbor list
        neighbor_indices = neighbor_indices[:, 1:]

        # Compute local surface normals via PCA
        surface_normals = np.zeros((self.n_cells, 3), dtype=np.float32)

        for i in range(self.n_cells):
            # Get neighborhood
            neighbors = self.x[neighbor_indices[i]]

            # Center neighborhood
            center = neighbors.mean(axis=0)
            X_centered = neighbors - center

            # SVD for PCA
            U, s, Vt = svd(X_centered, full_matrices=False)

            # Surface normal is eigenvector of smallest singular value
            # (perpendicular to the 2D manifold)
            n_pca = Vt[-1, :]

            # Combine with polarity: weight polarity more heavily (more constrained)
            n_combined = 0.3 * n_pca + 0.7 * self.p_normals[i]

            # Normalize
            n_combined = n_combined / (np.linalg.norm(n_combined) + 1e-8)
            surface_normals[i] = n_combined

        self.surface_normals = surface_normals
        return surface_normals

    def get_tangent_planes(self) -> tuple:
        """
        Generate orthonormal tangent plane basis vectors for each cell.

        For each cell, computes two orthonormal vectors (u, v) that span
        the tangent plane perpendicular to the surface normal.

        Returns
        -------
        tangent_bases : (N, 3, 2) ndarray
            For each cell, two orthonormal basis vectors [u_i, v_i]
            spanning the tangent plane
        surface_normals : (N, 3) ndarray
            Surface normal vectors
        """
        if self.surface_normals is None:
            self.detect_surface()

        tangent_bases = np.zeros((self.n_cells, 3, 2), dtype=np.float32)

        for i in range(self.n_cells):
            # Get surface normal
            n = self.surface_normals[i]

            # Find arbitrary perpendicular vector
            # Choose the axis least aligned with n
            abs_n = np.abs(n)
            min_idx = np.argmin(abs_n)
            perp = np.zeros(3)
            perp[min_idx] = 1.0

            # First tangent vector: v1 = normalize(perp - (perp·n)n)
            u = perp - np.dot(perp, n) * n
            u = u / (np.linalg.norm(u) + 1e-8)

            # Second tangent vector: v2 = n × v1
            v = np.cross(n, u)
            v = v / (np.linalg.norm(v) + 1e-8)

            tangent_bases[i, :, 0] = u
            tangent_bases[i, :, 1] = v

        self.tangent_bases = tangent_bases
        return tangent_bases, self.surface_normals

    def project_to_tangent_plane(self, cell_idx: int, max_distance: float = None) -> tuple:
        """
        Project all cells to the tangent plane of a given cell.

        Parameters
        ----------
        cell_idx : int
            Index of reference cell for tangent plane
        max_distance : float, optional
            Maximum distance to include cell in projection.
            If None, all cells are included.

        Returns
        -------
        positions_2d : (M, 2) ndarray
            2D coordinates in tangent plane
        valid_indices : (M,) ndarray
            Indices of cells included in projection
        """
        if self.tangent_bases is None:
            self.get_tangent_planes()

        # Get tangent basis for reference cell
        u = self.tangent_bases[cell_idx, :, 0]
        v = self.tangent_bases[cell_idx, :, 1]
        ref_pos = self.x[cell_idx]

        # Relative positions
        rel_pos = self.x - ref_pos  # (N, 3)

        # Project onto tangent plane
        coord_u = np.dot(rel_pos, u)  # (N,)
        coord_v = np.dot(rel_pos, v)  # (N,)

        positions_2d = np.stack([coord_u, coord_v], axis=1)  # (N, 2)

        # Filter by distance if specified
        if max_distance is not None:
            distances = np.linalg.norm(positions_2d, axis=1)
            valid = distances <= max_distance
        else:
            valid = np.ones(self.n_cells, dtype=bool)

        valid_indices = np.where(valid)[0]

        return positions_2d[valid], valid_indices
