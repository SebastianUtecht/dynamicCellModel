"""
Voronoi Animation Controller and Cache Management

Pre-computes Voronoi tessellations for all timesteps and manages frame-specific
rendering with caching for efficient reuse.
"""

import os
import pickle
import numpy as np
from typing import Dict, Callable, Optional, List
import warnings

from voronoi_computation import AugmentedVoronoi, VoronoiMesh


class VoronoiAnimator:
    """
    Manages Voronoi visualization across multiple timesteps.

    Pre-computes Voronoi meshes for all frames, caches them to disk,
    and provides per-frame mesh access for rendering.
    """

    def __init__(self, gui_instance=None):
        """
        Initialize Voronoi animator.

        Parameters
        ----------
        gui_instance : optional
            Reference to main GUI (for progress callbacks and settings)
        """
        self.gui = gui_instance
        self.voronoi_meshes = {}  # {timestep: {cell_idx: VoronoiMesh}}
        self.is_computing = False
        self.cache_dir = None

    def precompute_all_frames(
        self,
        data: Dict,
        progress_callback: Optional[Callable] = None,
        apply_occlusion: bool = False,
        start_ts: int = 0,
        end_ts: Optional[int] = None,
    ) -> bool:
        """
        Pre-compute Voronoi tessellations for a range of timesteps.

        Parameters
        ----------
        data : dict
            Dictionary with keys "x", "p_mask", "p", etc.
            Format: "x" -> list of (N, 3) arrays per timestep
        progress_callback : callable, optional
            Callback function called with (timestep, n_timesteps) during computation
        apply_occlusion : bool
            If True, apply occlusion test when computing Voronoi
        start_ts : int
            Starting timestep (inclusive)
        end_ts : int, optional
            Ending timestep (inclusive). If None, compute to the last timestep.

        Returns
        -------
        success : bool
            True if computation completed successfully
        """
        if "x" not in data or "p" not in data:
            warnings.warn("Data missing 'x' or 'p' keys. Cannot compute Voronoi.")
            return False

        x_list = data["x"]
        p_list = data["p"]

        if len(x_list) != len(p_list):
            warnings.warn("x and p have different number of timesteps.")
            return False

        n_timesteps = len(x_list)

        # Set end timestep if not provided
        if end_ts is None:
            end_ts = n_timesteps - 1

        # Validate range
        start_ts = max(0, start_ts)
        end_ts = min(n_timesteps - 1, end_ts)

        if start_ts > end_ts:
            start_ts, end_ts = end_ts, start_ts

        self.is_computing = True

        try:
            for t in range(start_ts, end_ts + 1):
                x = x_list[t]
                p = p_list[t]

                if x is None or len(x) == 0:
                    continue

                # Compute Voronoi using augmented 3D approach
                voronoi_comp = AugmentedVoronoi(x, p, thickness=2.0)
                meshes_dict = voronoi_comp.compute_voronoi_meshes()

                self.voronoi_meshes[t] = meshes_dict

                # Call progress callback
                if progress_callback:
                    progress_callback(t, n_timesteps)

            self.is_computing = False
            return True

        except Exception as e:
            self.is_computing = False
            warnings.warn(f"Voronoi pre-computation failed: {e}")
            return False

    def get_visible_meshes(
        self,
        timestep: int,
        section_mask: Optional[np.ndarray] = None,
    ) -> List[VoronoiMesh]:
        """
        Get list of visible Voronoi meshes for a timestep.

        Parameters
        ----------
        timestep : int
            Timestep index
        section_mask : (N,) ndarray bool, optional
            Boolean mask for visible cells. If None, all meshes are returned.

        Returns
        -------
        meshes : list of VoronoiMesh
            List of visible mesh objects
        """
        if timestep not in self.voronoi_meshes:
            return []

        meshes = []
        meshes_dict = self.voronoi_meshes[timestep]

        if section_mask is None:
            # Return all meshes
            for cell_idx, mesh in meshes_dict.items():
                if mesh.vertices_3d.shape[0] > 0:  # Non-empty mesh
                    mesh.is_visible = True
                    meshes.append(mesh)
        else:
            # Filter by section mask
            for cell_idx, mesh in meshes_dict.items():
                if cell_idx < len(section_mask) and section_mask[cell_idx]:
                    if mesh.vertices_3d.shape[0] > 0:
                        mesh.is_visible = True
                        meshes.append(mesh)
                else:
                    mesh.is_visible = False

        return meshes

    def update_scalar_colors(
        self,
        timestep: int,
        scalar_field: np.ndarray,
        vmin: float,
        vmax: float,
        colormap_fn: Callable,
    ) -> None:
        """
        Update colors of all Voronoi meshes based on scalar field.

        Parameters
        ----------
        timestep : int
            Timestep index
        scalar_field : (N,) ndarray
            Scalar values for each cell
        vmin, vmax : float
            Value range for color normalization
        colormap_fn : callable
            Function mapping scalar value to RGBA color
        """
        if timestep not in self.voronoi_meshes:
            return

        meshes_dict = self.voronoi_meshes[timestep]

        for cell_idx, mesh in meshes_dict.items():
            if cell_idx < len(scalar_field):
                scalar_val = scalar_field[cell_idx]
                try:
                    color = colormap_fn(scalar_val, vmin, vmax)
                    if isinstance(color, np.ndarray) and len(color) >= 4:
                        mesh.color = color
                except Exception as e:
                    warnings.warn(f"Color mapping failed for cell {cell_idx}: {e}")

    def update_type_colors(
        self,
        timestep: int,
        p_mask: np.ndarray,
        type_color_map: Dict[int, tuple],
    ) -> None:
        """
        Update colors of Voronoi meshes by cell type.

        Parameters
        ----------
        timestep : int
            Timestep index
        p_mask : (N,) ndarray
            Cell type indices
        type_color_map : dict
            Mapping from cell type to RGBA color
        """
        if timestep not in self.voronoi_meshes:
            return

        meshes_dict = self.voronoi_meshes[timestep]

        for cell_idx, mesh in meshes_dict.items():
            if cell_idx < len(p_mask):
                cell_type = p_mask[cell_idx]
                if cell_type in type_color_map:
                    color = type_color_map[cell_type]
                    mesh.color = np.array(color, dtype=np.float32)

    def update_transparency(self, alpha: float) -> None:
        """
        Update alpha channel for all cached meshes.

        Parameters
        ----------
        alpha : float
            Alpha value in [0.0, 1.0]
        """
        alpha = np.clip(alpha, 0.0, 1.0)

        for meshes_dict in self.voronoi_meshes.values():
            for mesh in meshes_dict.values():
                if mesh.color is not None:
                    mesh.color[3] = alpha

    def save_cache(self, cache_dir: str) -> bool:
        """
        Save pre-computed Voronoi meshes to disk cache.

        Parameters
        ----------
        cache_dir : str
            Directory to save cache files

        Returns
        -------
        success : bool
        """
        try:
            os.makedirs(cache_dir, exist_ok=True)

            cache_file = os.path.join(cache_dir, "voronoi_meshes.pkl")

            # Extract mesh data (without colors, which are render-time computed)
            cache_data = {}
            for t, meshes_dict in self.voronoi_meshes.items():
                cache_data[t] = {}
                for cell_idx, mesh in meshes_dict.items():
                    cache_data[t][cell_idx] = {
                        "vertices_3d": mesh.vertices_3d,
                        "faces": mesh.faces,
                    }

            with open(cache_file, "wb") as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            return True

        except Exception as e:
            warnings.warn(f"Failed to save Voronoi cache: {e}")
            return False

    def load_cache(self, cache_dir: str) -> bool:
        """
        Load pre-computed Voronoi meshes from disk cache.

        Parameters
        ----------
        cache_dir : str
            Directory containing cache files

        Returns
        -------
        success : bool
        """
        try:
            cache_file = os.path.join(cache_dir, "voronoi_meshes.pkl")

            if not os.path.exists(cache_file):
                return False

            with open(cache_file, "rb") as f:
                cache_data = pickle.load(f)

            # Reconstruct VoronoiMesh objects
            for t, meshes_dict in cache_data.items():
                self.voronoi_meshes[t] = {}
                for cell_idx, mesh_data in meshes_dict.items():
                    mesh = VoronoiMesh(
                        cell_idx,
                        mesh_data["vertices_3d"],
                        mesh_data["faces"],
                    )
                    self.voronoi_meshes[t][cell_idx] = mesh

            return True

        except Exception as e:
            warnings.warn(f"Failed to load Voronoi cache: {e}")
            return False

    def clear_cache(self) -> None:
        """Clear all cached meshes from memory."""
        self.voronoi_meshes.clear()

    def get_timesteps(self) -> List[int]:
        """Get list of timesteps with cached Voronoi meshes."""
        return sorted(self.voronoi_meshes.keys())

    def has_meshes_for_timestep(self, timestep: int) -> bool:
        """Check if Voronoi meshes are available for timestep."""
        return timestep in self.voronoi_meshes and len(self.voronoi_meshes[timestep]) > 0

    def get_memory_usage(self) -> int:
        """
        Estimate memory usage of cached meshes in bytes.

        Returns
        -------
        bytes : int
        """
        total_bytes = 0
        for meshes_dict in self.voronoi_meshes.values():
            for mesh in meshes_dict.values():
                total_bytes += mesh.vertices_3d.nbytes + mesh.faces.nbytes
        return total_bytes
