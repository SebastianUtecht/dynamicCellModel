### Imports ###
import numpy as np
import torch
from scipy.spatial import cKDTree
import os
import itertools
import pickle
from time import time
import json

### Simulation ###
class Simulation:
    """
    Simulation class for running a cell-based model with polarity-based interactions

    This class handles the initialization and execution of the simulation, including the management
    of cell states, interactions, and the application of forces based on cell polarity.
    """

    def __init__(self, sim_dict):
        """
        Initializes the simulation with the given parameters.

        Parameters:
            sim_dict (dict): A dictionary containing simulation parameters.

        Returns:
            None
        """

        # Metasimulational parameters
        self.device         = sim_dict['device']            # Device for tensor operations. 'cuda' for GPU acceleration otherwise 'cpu'
        self.dtype          = sim_dict['dtype']             # Data type for tensors. float32 or float64
        self.random_seed    = sim_dict['random_seed']       # Random seed for reproducibility
        self.yield_every    = sim_dict['yield_every']       # How many timesteps between data yields
        self.min_batch_size = 1024                          # Minimum batch size for true neighbor search. The simulation will automatically find the optimal batch size based on available memory, but this is the lower limit for that process. Setting it higher can speed up the simulation, but also risks out of memory errors if set too high.

        # Model parameters
        self.k              = sim_dict['k']                # Number of nearest neighbors to consider
        self.true_neighbour_max     = 50                    # Maximum number of true neighbors in former timestep
        self.dt             = sim_dict['dt']                # Size of timestep for simulation
        self.sqrt_dt        = np.sqrt(self.dt)              # Square root of time step. We calculate it here instead of in the update loop
        self.max_cells      = sim_dict['max_cells']         # Maximum number of cells in the simulation. Once this is reached, the simulation terminates

        # Main model parameters
        self.lambdas        = torch.tensor(sim_dict['lambdas'], device=self.device, dtype=self.dtype)   # Lambda values determining cell interactions
        etas                = sim_dict['etas']              # Noise strength
        if isinstance(etas, list):
            self.eta0, self.eta1 = etas
            self.eta_lst = [self.eta0, self.eta1]
        elif isinstance(etas, (float, int)):
            self.eta0 = self.eta1 = etas
            self.eta_lst = [self.eta0]
        else:
            raise ValueError("etas should be either a list of two values or a single value")
        polar_etas = sim_dict['polar_etas']  # Noise strength for polarity updates
        if isinstance(polar_etas, list):
            self.polar_eta0, self.polar_eta1 = polar_etas
            self.polar_eta_lst = [self.polar_eta0, self.polar_eta1]
        elif isinstance(polar_etas, (float, int)):
            self.polar_eta0 = self.polar_eta1 = polar_etas
            self.polar_eta_lst = [self.polar_eta0]
        else:
            raise ValueError("polar_etas should be either a list of two values or a single value")
        self.alpha_params   = sim_dict['alpha_params']      # Determines how cells wedge
        self.gamma_params   = sim_dict['gamma_params']      # Determines how cells elongate
        self.prolif_rate    = sim_dict['prolif_rate']       # Probability of cell proliferation for each cell
        self.prolif_delay   = sim_dict['prolif_delay']      # Delay before cell proliferation
        self.nematic_pcp    = sim_dict['nematic_pcp']     # Whether planar cell polarity is nematic (True) or vectorial (False)
        self.update_x_bools = sim_dict['update_x_bools']   # List of booleans determining whether to update the parameters for each cell type. Order is [type0_alpha_par, type0_alpha_perp, type0_gamma, type1_alpha_par, type1_alpha_perp, type1_gamma]
        self.screen_out_defects = sim_dict['screen_out_defects']   # Whether to screen out defects in the neighbor calculations. Only relevant if neighbour_type is 'voronoi'
        self.wedge_pcp = sim_dict['wedge_pcp']                     # Whether to apply wedging rotations to the PCP as well as the ABP. Only relevant if alpha parameters are non-zero
        
        ### Parameters for this paper ###
        self.equi_time          = sim_dict['equi_time']             # Time to equilibrate before starting the simulation. Only relevant if stretch_factor is not 0

        # Boundary parameters
        self.bound_type         = sim_dict['bound_type']
        assert self.bound_type == 'planes' or self.bound_type == 'cylinder' or self.bound_type == None, 'Boundtype expected to be in ["planes", "cylinder", None]'
        self.bound_extents      = [sim_dict['bound_end']]      # Extents for boundary conditions. For planes, this is the x and y extent. For cylinder, this is the radius and height
        self.bound_move_times   = [self.equi_time, sim_dict['bound_move_times']]   # If using planes, this is the times at which the planes move. For cylinder, this is the time at which the cylinder starts moving
        self.bound_continuity   = sim_dict['bound_continuity']   # Continuity for boundary conditions
        self.bound_cur_ext      = self.bound_extents[0] if self.bound_move_times is not None else None # Current extent of the boundary. Only relevant if bound_move_times is not None

        # Stretching parameters
        self.stretch_factor     = sim_dict['stretch_factor']        # Strength of the stretching. 0 for no stretch, higher for stronger stretch. The stretch is applied to the cells in the stretch_frac fraction of the radius from the center
        self.stretch_time_stop  = sim_dict['stretch_time_stop']     # The time at which the stretch stops. Only relevant if stretch_factor is not 0
        self.stretch_bound_axis = sim_dict['stretch_bound_axis']    # The axis along which the stretch is applied. 0 for x, 1 for y, 2 for z. Only relevant if stretch_factor is not 0
        self.stretch_type       = sim_dict['stretch_type']          # The type of stretching to apply. Only relevant if stretch_factor is not 0


        # Relaxation length parameters 
        self.r0             = 5*np.log(5)/(5-1)
        self.r0_val         = np.exp(-self.r0)-np.exp(-self.r0/5)
        self.seethru        = 0
        self.cell_wall_interaction = sim_dict['cell_wall_interaction']  # 0 if only repulsion, otherwise up to 1 for attraction

        # stuff we need to initialize
        self.idx    = None                                      # Indices of nearest neighbors
        self.beta   = None                                      # Tensor used for cell division
        self.division = False                                   # Tensor used for cell division

        # Locking or freeing the parameters for autograd

        # Multiple cell types
        if isinstance(self.alpha_params[0], list):

            # Cell type 0
            self.type0_alpha_par_free = self.alpha_params[0][0][1] == 'free'
            self.type0_alpha_perp_free = self.alpha_params[0][1][1] == 'free'
            self.type0_gamma_free = self.gamma_params[0][1] == 'free'
            
            # Cell type 1
            self.type1_alpha_par_free = self.alpha_params[1][0][1] == 'free'
            self.type1_alpha_perp_free = self.alpha_params[1][1][1] == 'free'
            self.type1_gamma_free = self.gamma_params[1][1] == 'free'
        
        else:
            self.type0_alpha_par_free = self.alpha_params[0][1] == 'free'
            self.type0_alpha_perp_free = self.alpha_params[1][1] == 'free'
            self.type0_gamma_free = self.gamma_params[1] == 'free'

            self.type1_alpha_par_free = False
            self.type1_alpha_perp_free = False
            self.type1_gamma_free = False
        
        # lists
        self.alpha_par_bool_lst     = [self.type0_alpha_par_free, self.type1_alpha_par_free]
        self.alpha_perp_bool_lst    = [self.type0_alpha_perp_free, self.type1_alpha_perp_free]
        self.gamma_bool_lst         = [self.type0_gamma_free, self.type1_gamma_free]

        # Ranges
        self.alpha_range = torch.tensor([-sim_dict['alpha_range'] * np.pi/180.0, sim_dict['alpha_range'] * np.pi/180.0 ], device=self.device, dtype=self.dtype)
        self.gamma_range = torch.tensor([-np.log(sim_dict['gamma_range']), np.log(sim_dict['gamma_range']) ], device=self.device, dtype=self.dtype)

        self.get_neighbors = self.get_neighbors_vor        
        self.use_trans_neighbors = True
        self.gamma_diff_penalty = sim_dict['gamma_diff_penalty']

        # Set random seed
        torch.manual_seed(self.random_seed)                 # For reproducibility
        self.tstep = 0

    #Checked
    def elong_func_cos(self, q_mean, dx):
        return 2 * ((q_mean * dx).sum(dim=2))**2 - 1

    def safe_normalize(self, v, dim=-1, eps: float = 1e-12):
        """Normalise vectors safely (avoids NaNs when norm is ~0)."""
        n = torch.linalg.norm(v, dim=dim, keepdim=True)
        return v / torch.clamp(n, min=eps)

    def rotation_matrices_axis_angle(self, axis, angle, eps: float = 1e-12):
        """Create rotation matrices from axis-angle (Rodrigues)

        Parameters:
            axis: (..., 3) rotation axes
            angle: (...) rotation angles (radians)
        Returns:
            R: (..., 3, 3)
        """

        axis_norm = torch.linalg.norm(axis, dim=-1)
        valid = axis_norm > eps

        # Normalised axis; value doesn't matter where invalid because angle will be zeroed.
        u = axis / torch.clamp(axis_norm, min=eps)[..., None]
        angle = angle * valid.to(angle.dtype)

        ux, uy, uz = u.unbind(dim=-1)
        zero = torch.zeros_like(ux)

        K = torch.stack(
            (
                torch.stack((zero, -uz, uy), dim=-1),
                torch.stack((uz, zero, -ux), dim=-1),
                torch.stack((-uy, ux, zero), dim=-1),
            ),
            dim=-2,
        )

        I = torch.eye(3, device=axis.device, dtype=axis.dtype).expand(axis.shape[:-1] + (3, 3))

        theta = angle[..., None, None]
        c = torch.cos(theta)
        s = torch.sin(theta)
        one_minus_c = 1.0 - c

        uuT = u[..., :, None] * u[..., None, :]
        R = c * I + one_minus_c * uuT + s * K
        return R
    
    @staticmethod
    def find_potential_neighbours(x, k):

        with torch.no_grad():
            # cKD-tree method
            x_cpu = x.cpu().numpy()
            tree = cKDTree(x_cpu)
            n_cells = x_cpu.shape[0]
            k_query = min(k + 1, n_cells)
            d, idx = tree.query(x_cpu, k=k_query)  # Get the k+1 nearest neighbors (including self)
            d = torch.from_numpy(d).to(x.device).to(x.dtype)
            idx = torch.from_numpy(idx).to(x.device).long()

            # Handle 1D case (when k_query == 1, only self is returned)
            if k_query == 1:
                # For single-cell query, expand dimensions
                d = d.unsqueeze(1)
                idx = idx.unsqueeze(1)
            elif d.ndim == 1:
                # Convert 1D arrays to 2D if needed
                d = d.unsqueeze(1)
                idx = idx.unsqueeze(1)

        return d[:, 1:], idx[:, 1:]                             # Return distances and indices of neighbors. Note that we do not return the self-neighbor

    def find_max_safe_batch(self, total_cells):
        # Helper to test a batch size for memory safety
        def try_batch(batch_size, test_dx, test_d):
            with torch.no_grad():
                try:
                    i0, i1 = 0, min(batch_size, test_dx.shape[0])
                    n_dis = torch.sum((test_dx[i0:i1, :, None, :] / 2 - test_dx[i0:i1, None, :, :]) ** 2, dim=3)
                    n_dis += 1000 * torch.eye(n_dis.shape[1], device=self.device, dtype=self.dtype)[None, :, :]
                    _ = torch.sum(n_dis < (test_d[i0:i1, :, None] ** 2 / 4), dim=2)
                    del n_dis, _
                    torch.cuda.synchronize()
                    return True
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        if self.device == "cuda":
                            torch.cuda.empty_cache()
                        return False
                    raise e


        # Create dummy dataset: 5× larger, 40 features
        dummy_n = min(5 * total_cells, 20_000)  # Optional: cap for safety
        dummy_dx = torch.randn((dummy_n, 40, 3), device=self.device, dtype=self.dtype)
        dummy_d = torch.linalg.norm(dummy_dx, dim=2)

        batch_size = self.min_batch_size
        max_batch = self.min_batch_size

        while True:
            ok = try_batch(batch_size, dummy_dx, dummy_d)
            if ok:
                max_batch = batch_size
                batch_size *= 2
                if batch_size > dummy_dx.shape[0]:
                    break
            else:
                break

        self.max_safe_batch = max_batch
        print(f"Found max safe batch size: {self.max_safe_batch}")
        del dummy_dx, dummy_d
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def get_gamma_exponent(self, dx, pi, pj, q_mean, idx, gamma):

        wall_mask = (torch.sum(pi * pj , dim = 2) <= 0.0)

        gamma_i = gamma[:, None].expand(gamma.shape[0], idx.shape[1])
        gamma_j = gamma[idx]
        gamma_mean = torch.log((torch.exp(gamma_i) + torch.exp(gamma_j))/2)

        elong = self.elong_func_cos(q_mean, dx)

        exponent = gamma_mean * elong
        exponent[wall_mask] = 0.0

        return exponent, gamma_i, gamma_j
    
    def get_neighbors_vor(self, x, p, q, gamma, k):
        self.refresh_potential_neighbours_once(x, k)
        return self.true_neighbours_from_idx(x, p, q, gamma, self.idx)

    def refresh_potential_neighbours_once(self, x, k):
        """Refreshes the cached potential-neighbour list at most once per timestep."""
        if self.update_neighbors_bool():
            self.d, self.idx = self.find_potential_neighbours(x, k)

    def true_neighbours_from_idx(self, x, p, q, gamma, idx):
        """Builds true neighbours (Voronoi filter + trimming) from a provided potential-neighbour idx list.

        This is intentionally side-effect free with respect to neighbour caching, so it can be used
        multiple times within a single timestep (e.g. predictor/corrector) without triggering KDTree.
        """

        # Find neighbours from provided idx
        full_n_list = x[idx]
        dx = x[:, None, :] - full_n_list
        d = torch.sqrt(torch.sum(dx**2, dim=2))
        z_mask = self.find_true_neighbours(d, dx, idx, p, q, gamma)

        # Minimize size of z_mask and reorder idx and dx
        sort_idx = torch.argsort(z_mask.int(), dim=1, descending=True)
        z_mask = torch.gather(z_mask, 1, sort_idx)
        dx = torch.gather(dx, 1, sort_idx[:, :, None].expand(-1, -1, 3))
        idx = torch.gather(idx, 1, sort_idx)
        m = torch.max(torch.sum(z_mask, dim=1)) + 1
        z_mask = z_mask[:, :m]
        dx = dx[:, :m]
        idx = idx[:, :m]

        # Normalise dx
        d = torch.sqrt(torch.sum(dx**2, dim=2))
        dx = dx / d[:, :, None]

        return d, dx, idx, z_mask

    def find_true_neighbours(self, d, dx, idx, p, q, gamma, test_batches=True):
        with torch.no_grad():
            # Expanding ABP and PCP
            qi = q[:, None, :].expand(q.shape[0], idx.shape[1], 3)
            qj = q[idx]
            q_mean = (qi + qj)
            q_mean = self.safe_normalize(q_mean, dim=2)
            pi = p[:, None, :].expand(p.shape[0], idx.shape[1], 3)
            pj = p[idx]
            
            exponent, _, _ = self.get_gamma_exponent(dx / d[:,:,None], pi, pj, q_mean, idx, gamma)

            dx_tilde = dx * torch.exp(exponent)[:,:,None]
            d_tilde  = d * torch.exp(exponent)

            if self.screen_out_defects:
                defect_mask = (torch.sum(qi * qj, dim=2) < 0.7)
                dx_tilde[defect_mask] = dx[defect_mask]
                d_tilde[defect_mask] = d[defect_mask]


            total_cells = dx.shape[0]
            neighbor_count = dx.shape[1]
            result_tensor = torch.empty(
                (total_cells, neighbor_count),
                dtype=torch.bool,
                device=self.device
            )

            
            if self.tstep == 0:
                if test_batches:
                    self.find_max_safe_batch(total_cells=total_cells)
                else:
                    self.max_safe_batch = 1024

                
            batch_size = self.max_safe_batch
            if self.tstep % 50_000 == 0:
                print(f"Using batch size: {batch_size} for true neighbor search.")
            i0 = 0
            while i0 < total_cells:
                i1 = min(i0 + batch_size, total_cells)
                try:
                    n_dis = torch.sum((dx_tilde[i0:i1, :, None, :] / 2 - dx_tilde[i0:i1, None, :, :]) ** 2, dim=3)
                    n_dis += 1000 * torch.eye(n_dis.shape[1], device=self.device, dtype=self.dtype)[None, :, :]
                    result_tensor[i0:i1] = (torch.sum(n_dis < (d_tilde[i0:i1, :, None] ** 2 / 4), dim=2) <= self.seethru)
                    i0 = i1

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        if self.device == "cuda":
                            torch.cuda.empty_cache()
                        batch_size = max(self.min_batch_size, int(batch_size // 1.2))
                        print(f"OOM at cell {i0}. Reducing batch size to {batch_size}.")
                        continue
                    else:
                        raise e

        return result_tensor

    def advance_boundary_state(self):
        """Advance boundary state once per timestep (beginning-of-step)."""
        if self.bound_type is None or self.bound_move_times is None:
            return
        if self.bound_cur_ext is None:
            return
        if self.tstep > self.bound_move_times[0] and self.tstep < self.bound_move_times[1]:
            self.bound_cur_ext += self.bound_speed

    def bound(self, pos):
        if self.bound_type is None:
            return 0.0
        elif self.tstep > (self.bound_move_times[1] + self.bound_continuity):
            return 0.0
        elif self.bound_type == 'planes':
            return self.planes_bound(pos)
        elif self.bound_type == 'cylinder':
            return self.cylinder_bound(pos)
        else:
            return 0.0
        
    def planes_bound(self, pos):
        bound_dists = torch.abs(pos[:, self.stretch_bound_axis]) - self.bound_cur_ext/2
        v_add = torch.where(bound_dists > 0, 2 * bound_dists**2, 0.0)
        if torch.isnan(v_add).any() or torch.isinf(v_add).any():                        #check for nan or inf. This is mainly for debugging, but i've kept it as it sometimes does.... stuff.
            print("Warning: NaN or Inf detected in potential energy")
        V_add_sum = v_add.sum()                             
        return V_add_sum
    
    def cylinder_bound(self, pos):
        # cylinder along self.stretch_bound_axis, so we calculate the distance in the plane perpendicular to that axis
        if self.stretch_bound_axis == 0:
            bound_dists = torch.sqrt(torch.sum(pos[:, 1:3]**2, dim=1)) - self.bound_cur_ext/2
        elif self.stretch_bound_axis == 1:
            bound_dists = torch.sqrt(torch.sum(pos[:, ::2]**2, dim=1)) - self.bound_cur_ext/2
        else:
            bound_dists = torch.sqrt(torch.sum(pos[:, :2]**2, dim=1)) - self.bound_cur_ext/2
        v_add = torch.where(bound_dists > 0, 2 * bound_dists**2, 0.0)
        if torch.isnan(v_add).any() or torch.isinf(v_add).any():                        #check for nan or inf. This is mainly for debugging, but i've kept it as it sometimes does.... stuff.
            print("Warning: NaN or Inf detected in potential energy")
        V_add_sum = v_add.sum()                                                         # Add it all up as we need it in scalar form for gradient computation.
        return V_add_sum


    def sphere_bound(self, pos):
        """
        Calculating additions to the potential due to spherical boundary conditions.
        Only does anything if self.bound_radius is a float.

        Parameters:
            pos (torch.Tensor): Input tensor of shape (N, 3) where N is the number of points.

        Returns:
            V_add_sum (float): Addition to the potential energy due to spherical boundary conditions.

        """

        bound_dists = torch.sqrt(torch.sum((pos)**2, dim=1))                            # Calculate distances from the center
        v_add       = torch.where(bound_dists > self.bound_radius, bound_dists**2, 0.0) # Calculate additional potential energy
        if torch.isnan(v_add).any() or torch.isinf(v_add).any():                        #check for nan or inf. This is mainly for debugging, but i've kept it as it sometimes does.... stuff. 
            print("Warning: NaN or Inf detected in potential energy")
        V_add_sum = v_add.sum()                                                         # Add it all up as we need it in scalar form for gradient computation.
        return V_add_sum
    
    def rescale_s(self, S):
        S_rescaled = (S + 1.0) / 2.0
        return S_rescaled

    def potential(self, x, p, q, p_mask, alpha_par, alpha_perp, gamma, d, dx, idx, z_mask):
        """
        Calculate the potential energy between particles.

        Parameters:
            x (torch.Tensor): Positions of the particles.
            p (torch.Tensor): Apicobasal polarities of the cells
            q (torch.Tensor): Planar cell polarity of the cells
            p_mask (torch.Tensor): Mask denoting different cell types. Is bool in this implementation (only 2 cell types)
            alpha_par (torch.Tensor): The parallel alpha parameter.
            alpha_perp (torch.Tensor): The perpendicular alpha parameter.
            gamma (torch.Tensor): The gamma (elongation) parameter.
            d (torch.Tensor): The distances to the neighbors.
            dx (torch.Tensor): The normalized displacement vectors to the neighbors.
            idx (torch.Tensor): Indices of neighboring particles.
            z_mask (torch.Tensor): A boolean mask indicating which neighbors are true neighbors based on a distance cutoff
        Returns:
            V_sum (float): The total potential energy.
            Vij_normed (torch.Tensor): The potential energy between particles normalized by the number of interactions.
            particle_defect_mask (torch.Tensor): Boolean mask of shape (N,) indicating cells involved in defect interactions.
        """

        if torch.unique(p_mask).shape[0] > 1:        # Check if p_mask is a tensor indicating 2 cell types. Otherwise it is None
            # Making interaction mask
            assert torch.numel(self.lambdas) == 12, "Expected 3*4 lambda values"                    #Only 2 cell types means 12 total lambda values
            interaction_mask = p_mask[:,None].expand(p_mask.shape[0], idx.shape[1]) + p_mask[idx]   # Making the interaction mask. A tool for constructing a lambda tensor

            # Filling a lambda array with the right interactions
            l = torch.zeros(size=(interaction_mask.shape[0], interaction_mask.shape[1], 4),
                        device=self.device, dtype=self.dtype)   # Empty lambda tensor
            repulsion_mask_lst = []

            for i in range(3):                                          # We loop through the 3 interaction types (type0-type0, type0-type1, type1-type1)
                if torch.any(self.lambdas[i]):                          # If we have non-zero lambda values put em in
                    l[interaction_mask == i] = self.lambdas[i]          # Filling the lambda tensor according to the interaction mask
                else:
                    repulsion_mask_lst.append(interaction_mask == i)       # If only non-zero lambda values the interaction is purely repulsive for d<r0
        else:
            # If multiple lambda array exist this is an error
            assert self.lambdas.ndim == 1, "Multiple lambda arrays found"
            l = self.lambdas[None, None, :].expand(x.shape[0], idx.shape[1], 4)  # Using the same lambda array for all particles

        # Expanding ABP and PCP
        pi = p[:, None, :].expand(p.shape[0], idx.shape[1], 3)
        pj = p[idx]
        qi = q[:, None, :].expand(q.shape[0], idx.shape[1], 3)
        qj = q[idx]
        
        q_mean = (qi + qj)
        q_mean = self.safe_normalize(q_mean, dim=2)

        p_mean = (pi + pj)
        p_mean = self.safe_normalize(p_mean, dim=2)
        
        # Expanding alpha_par, alpha_perp
        alpha_par_i = alpha_par[:, None].expand(alpha_par.shape[0], idx.shape[1])
        alpha_par_j = alpha_par[idx]
        alpha_par_mean = (alpha_par_i + alpha_par_j) / 2.0

        alpha_perp_i = alpha_perp[:, None].expand(alpha_perp.shape[0], idx.shape[1])
        alpha_perp_j = alpha_perp[idx]
        alpha_perp_mean = (alpha_perp_i + alpha_perp_j) / 2.0

        q_axis = self.safe_normalize(q_mean, dim=2)
        p_axis = self.safe_normalize(p_mean, dim=2)
        perp_axis = torch.cross(q_axis, p_axis, dim=2)
        perp_axis = self.safe_normalize(perp_axis, dim=2)

        c_par = torch.sum(q_axis * dx, dim=2)
        c_perp = torch.sum(perp_axis * dx, dim=2)

        par_angles = alpha_par_mean * c_par
        perp_angles = alpha_perp_mean * c_perp
        tan_par = torch.tan(par_angles/2)
        tan_perp = torch.tan(perp_angles/2)
        arrival_vec = tan_par[:,:,None] * q_axis + tan_perp[:,:,None] * perp_axis + p_mean
        arrival_vec = self.safe_normalize(arrival_vec, dim=2)
        
        axis    = torch.cross(p_mean, arrival_vec, dim=2)
        angle_i = torch.atan2(torch.linalg.norm(axis, dim=2), torch.sum(p_mean * arrival_vec, dim=2))
        angle_j = -angle_i.clone()
        axis_i  = self.safe_normalize(axis, dim=2)
        axis_j  = axis_i.clone()

        # Using the same rotation for i and j to preserve symmetry

        rot_mat_i = self.rotation_matrices_axis_angle(axis_i, angle_i)
        rot_mat_j = self.rotation_matrices_axis_angle(axis_j, angle_j)

        pi_tilde = torch.einsum('...ij,...j->...i', rot_mat_i, pi)
        pj_tilde = torch.einsum('...ij,...j->...i', rot_mat_j, pj)
        pi_tilde = self.safe_normalize(pi_tilde, dim=2)
        pj_tilde = self.safe_normalize(pj_tilde, dim=2)

        qi_tilde = torch.einsum('...ij,...j->...i', rot_mat_i, qi)
        qj_tilde = torch.einsum('...ij,...j->...i', rot_mat_j, qj)
        qi_tilde = self.safe_normalize(qi_tilde, dim=2)
        qj_tilde = self.safe_normalize(qj_tilde, dim=2)

        with torch.no_grad():
            wall_mask = (torch.sum(pi * pj , dim = 2) <= 0.0)           #* (torch.sum(-dx * pj , dim = 2) < 0.0) #maybe comment in later

        exponent, gamma_i, gamma_j = self.get_gamma_exponent(dx, pi, pj, q_mean, idx, gamma)
        d_tilde = d * torch.exp(exponent)

        # All the S-terms are calculated
        S1 = torch.sum(torch.cross(pj_tilde, dx, dim=2) * torch.cross(pi_tilde, dx, dim=2), dim=2) * torch.abs(torch.sum(pi_tilde * pj_tilde, dim=2))                      # Calculating S1 (The ABP-position part of S). Scalar for each particle-interaction. Meaning we get array of size (n, m) , m being the max number of nearest neighbors for a particle 
        S2 = torch.sum(torch.cross(pi_tilde, qi_tilde, dim=2) * torch.cross(pj_tilde, qj_tilde, dim=2), dim=2) * torch.abs(torch.sum(qi_tilde * qj_tilde, dim=2))           # Calculating S2 (The ABP-PCP part of S).
        S3 = torch.sum(torch.cross(qi_tilde, dx, dim=2) * torch.cross(qj_tilde, dx, dim=2), dim=2)                                                              # Calculating S3 (The PCP-position part of S)

        S1 = self.rescale_s(S1)
        S2 = self.rescale_s(S2)
        S3 = self.rescale_s(S3)

        if self.cell_wall_interaction != 0.0:
            with torch.no_grad():
                l[wall_mask] = torch.tensor([self.cell_wall_interaction, 0.0, 0.0, 0.0], device=self.device, dtype=self.dtype)

        # Calculating S
        S = l[:,:,0] + l[:,:,1] * S1 + l[:,:,2] * S2 + l[:,:,3] * S3

        Vij = z_mask.float() * S * (torch.exp(-d_tilde) - torch.exp(-d_tilde/5))        # Calculating the potential energy between particles masking out false interactions via voronoi_mask
        
        particle_defect_mask = torch.zeros(x.shape[0], dtype=torch.bool, device=self.device)

        if self.screen_out_defects:

            with torch.no_grad():
                seed_defect_mask = (torch.sum(qi_tilde * qj_tilde, dim=2) < 0.7) * z_mask  # Seed: low q-alignment interactions among true neighbors

                if torch.any(seed_defect_mask):
                    particle_defect_mask |= torch.any(seed_defect_mask, dim=1)
                    particle_defect_mask[idx[seed_defect_mask]] = True

                    # Include all interactions of seed-defect particles, not only low q-alignment pairs
                    defect_mask = z_mask & (particle_defect_mask[:, None] | particle_defect_mask[idx])

                    # Recompute per-particle mask from the expanded interaction mask
                    particle_defect_mask = torch.zeros(x.shape[0], dtype=torch.bool, device=self.device)
                    particle_defect_mask |= torch.any(defect_mask, dim=1)
                    particle_defect_mask[idx[defect_mask]] = True
                else:
                    defect_mask = torch.zeros_like(z_mask)

            if torch.any(defect_mask):
                # Calculate S1 for defects using non-rotated vectors
                S1_def = torch.sum(pi * pj, dim=2) * torch.sum(torch.cross(pj, dx, dim=2) * torch.cross(pi, dx, dim=2), dim=2)
                S1_def = self.rescale_s(S1_def)

                # Calculate the new S term for defects
                # S_def = l[:,:,0] + (l[:,:,1] + l[:,:,2] + l[:,:,3]) * S1_def
                S_def = 0.5 + 0.5 * S1_def

                # Calculate the potential for defects using non-elongated distance 'd'
                Vij_def = z_mask.float() * S_def * (torch.exp(-d) - torch.exp(-d/5))

                # Update the Vij tensor only for the defect interactions
                Vij[defect_mask] = Vij_def[defect_mask]      # Recalculating the potential energy for the defect interactions masking out false interactions via voronoi_mask

        if torch.unique(p_mask).shape[0] > 1:        # If we have multiple cell types we need to add the repulsion for the purely repulsive interactions
            for repulsion_mask in repulsion_mask_lst:
                # find the masked interactions for which dists < eq_dist
                dist_mask = d < self.r0
                too_close_mask = (repulsion_mask * dist_mask) * z_mask
                not_close_mask = (repulsion_mask * (~dist_mask)) * z_mask
                Vij[too_close_mask] = (torch.exp(-d[too_close_mask]) - torch.exp(-d[too_close_mask]/5))     # - self.r0_val
                Vij[not_close_mask] = self.r0_val 

        if self.cell_wall_interaction == 0.0:
            with torch.no_grad():
                dist_mask = d < self.r0
                too_close_mask = (wall_mask * dist_mask) * z_mask
                not_close_mask = (wall_mask * (~dist_mask)) * z_mask
            Vij[too_close_mask] = (torch.exp(-d[too_close_mask]) - torch.exp(-d[too_close_mask]/5))         # - self.r0_val
            Vij[not_close_mask] = self.r0_val #0.0
        
        Vij_sum = torch.sum(Vij)

        if self.gamma_diff_penalty:
            # If p_mask == 1 cells are not updated we don't count them for the 
            # gamma penalty

            if not(self.update_x_bools[1]):
                masked_out_cells = torch.logical_or((interaction_mask == 2) , (interaction_mask == 1)) * (~z_mask)
            else:
                masked_out_cells = ~z_mask
            gamma_diff = (gamma_i - gamma_j)**2
            gamma_diff[masked_out_cells] = 0.0
            gamma_diff_sum = torch.sum(gamma_diff)
            Vij_sum += gamma_diff_sum * self.gamma_diff_penalty

        if self.tstep > self.equi_time:
            # Boundary conditions
            bc = self.bound(x)
        else:
            bc = 0.0

        V = Vij_sum + bc

        num_neighbors = torch.sum(z_mask, dim=1)           
        Vij_normed = Vij / num_neighbors[:, None]       
        Vij_normed[~z_mask] = 0.0
        Vi = torch.sum(Vij_normed, dim=1)

        return V , Vi, particle_defect_mask

    def init_simulation(self, x, p, q, p_mask, alpha_par, alpha_perp, gamma):
        """
        Initiating simulation parameters by transforming ndarrays to tensors and the like.

        Parameters:
            x (np.ndarray): Cell positions.
            p (np.ndarray): Apicobasal polarities.
            q (np.ndarray): Planar cell polarities
            p_mask (np.ndarray) or None: Mask denoting different cell types. If None, all cells are considered the same type.
            alpha_par (np.ndarray): The parallel alpha parameter.
            alpha_perp (np.ndarray): The perpendicular alpha parameter.
            gamma (np.ndarray): The gamma (elongation) parameter.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The initialized tensors for cell positions, apicobasal polarities, planar cell polarities, the particle mask, the parallel alpha parameter, the perpendicular alpha parameter, and the gamma parameter.
            OR
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None]: If p_mask is None, the last element is None.

        """

        # Check input lengths
        assert len(x) == len(p)
        assert len(q) == len(x)

        # Putting the data on the right device (GPU or CPU)
        x = torch.tensor(x, requires_grad=True, dtype=self.dtype, device=self.device)
        p = torch.tensor(p, requires_grad=True, dtype=self.dtype, device=self.device)
        q = torch.tensor(q, requires_grad=True, dtype=self.dtype, device=self.device)
        alpha_par = torch.tensor(alpha_par, requires_grad=True, dtype=self.dtype, device=self.device)
        alpha_perp = torch.tensor(alpha_perp, requires_grad=True, dtype=self.dtype, device=self.device)
        gamma = torch.tensor(gamma, requires_grad=True, dtype=self.dtype, device=self.device)

        p_mask = torch.tensor(p_mask, dtype=torch.int, device=self.device)
        self.beta   = torch.zeros(x.shape[0], dtype=self.dtype, device=self.device) # Initialization of beta tensor. Used for cell division.

        # we set bound_ext[0] to the largest distance between two cells alng self.stretch_bound_axis. This is used for stretching the tissue in the right direction.
        max_dist   = torch.max(x[:,self.stretch_bound_axis]) - torch.min(x[:,self.stretch_bound_axis]) + 4.0
        self.bound_extents =  [max_dist] + self.bound_extents
        self.bound_speed        = (self.bound_extents[1] - self.bound_extents[0]) / ( (self.bound_move_times[1] - self.bound_move_times[0]) )  if self.bound_move_times is not None else 0.0 # Speed of boundary movement. Only relevant if bound_move_times is not None


        # We calculate the initial x_mass_midpoint for stretching purposes.
        with torch.no_grad():
            self.mass_midpoint = torch.mean(x, dim=0)  # Calculate the mean position of all cells
            self.x_mass_midpoint = self.mass_midpoint[0]  # Store the initial mean position for later use

        return x, p, q, p_mask, alpha_par, alpha_perp, gamma # Returning the goods.
    
    def update_neighbors_bool(self):
        """
        Returns whether to update potential neighbors or not

        Returns:
            bool: Whether to update potential neighbors or not.

        """
        if self.idx is None:                                  # If we have not found any neighbors yet
            return True
        elif self.division or self.tstep < 1_000:        # If cell division has occurred or we are in the early stages
            return True
        return (self.tstep % 20 == 0)                           # Otherwise we update every 20th step

    def apply_constraints(self, p, q, p_mask, alpha_par, alpha_perp, gamma):
        """Apply unit-norm and parameter bounds in-place under no_grad."""
        with torch.no_grad():
            eps = 1e-12
            p_norm = torch.sqrt(torch.sum(p ** 2, dim=1))
            q_norm = torch.sqrt(torch.sum(q ** 2, dim=1))
            p.div_(torch.clamp(p_norm, min=eps)[:, None])
            q.div_(torch.clamp(q_norm, min=eps)[:, None])

            # Clamp only the parameters that are free for the given type
            for i, _ in enumerate(self.eta_lst):
                mask = p_mask == i
                if self.alpha_par_bool_lst[i]:
                    alpha_par[mask] = torch.clamp(alpha_par[mask], self.alpha_range[0], self.alpha_range[1])
                if self.alpha_perp_bool_lst[i]:
                    alpha_perp[mask] = torch.clamp(alpha_perp[mask], self.alpha_range[0], self.alpha_range[1])
                if self.gamma_bool_lst[i]:
                    gamma[mask] = torch.clamp(gamma[mask], self.gamma_range[0], self.gamma_range[1])

        return p, q, alpha_par, alpha_perp, gamma

    def time_step(self, x, p, q, p_mask, alpha_par, alpha_perp, gamma, tstep):
        """
        Progresses the simulation by one time step.

        Parameters:
            x (torch.Tensor): The cell positions.
            p (torch.Tensor): The apicobasal polarities.
            q (torch.Tensor): The planar cell polarities.
            p_mask (torch.Tensor or None): The particle mask.
            tstep (int): The current time step.

        Returns:
            x (torch.Tensor): The updated cell positions.
            p (torch.Tensor): The updated apicobasal polarities.
            q (torch.Tensor): The updated planar cell polarities.
            p_mask (torch.Tensor or None): The updated particle mask.
        """

        # Update proliferation rate from when we want cell proliferation to occur
        if tstep == (self.prolif_delay + 1) and self.prolif_rate is not None:
            if torch.unique(p_mask).shape[0] > 1:        # If we have multiple cell types
                self.beta[p_mask == 0] = self.prolif_rate[0]
                self.beta[p_mask == 1] = self.prolif_rate[1]
            else:
                self.beta[:] = self.prolif_rate

        # Start with cell division
        self.division, x, p, q, p_mask, self.beta, alpha_par, alpha_perp, gamma = self.cell_division(x, p, q, p_mask, alpha_par, alpha_perp, gamma)

        # Advance any stateful boundary/stretch logic once per timestep (beginning-of-step).
        if self.tstep > self.equi_time:
            with torch.no_grad():
                self.advance_boundary_state()

        # Refresh potential neighbours at most once per timestep (KDTree), then reuse idx for both stages.
        self.refresh_potential_neighbours_once(x, self.k)
        idx_base = self.idx

        d1, dx1, idx1, z_mask1 = self.true_neighbours_from_idx(x, p, q, gamma, idx_base)
        V1, Vi, defect_mask = self.potential(x, p, q, p_mask,
                                             alpha_par, alpha_perp, gamma,
                                             d1, dx1, idx1, z_mask1)

        g1_x, g1_p, g1_q, g1_alpha_par, g1_alpha_perp, g1_gamma = torch.autograd.grad(
            V1, (x, p, q, alpha_par, alpha_perp, gamma), create_graph=False, retain_graph=False
        )

        # Shared noise for additive-noise predictor/corrector
        with torch.no_grad():
            xi_x = torch.empty_like(x).normal_()
            xi_p = torch.empty_like(p).normal_()
            xi_q = torch.empty_like(q).normal_()
            xi_alpha_par = torch.empty_like(alpha_par).normal_()
            xi_alpha_perp = torch.empty_like(alpha_perp).normal_()
            xi_gamma = torch.empty_like(gamma).normal_()

            # Predictor state
            x_tilde = x.clone()
            p_tilde = p.clone()
            q_tilde = q.clone()
            alpha_par_tilde = alpha_par.clone()
            alpha_perp_tilde = alpha_perp.clone()
            gamma_tilde = gamma.clone()

            for i, eta in enumerate(self.eta_lst):
                mask = (p_mask == i)

                polar_eta = self.polar_eta_lst[i]

                if self.update_x_bools[i]:
                    x_tilde[mask] += (-g1_x[mask] * self.dt) + (eta * xi_x[mask] * self.sqrt_dt)
                p_tilde[mask] += (-g1_p[mask] * self.dt) + (polar_eta * xi_p[mask] * self.sqrt_dt)  # We add the rotational noise as an Euler step for the predictor, then apply the actual rotation after calculating the angle. This is to ensure the noise is properly scaled by eta and sqrt_dt, and to keep the code simpler.
                q_tilde[mask] += (-g1_q[mask] * self.dt) + (polar_eta * xi_q[mask] * self.sqrt_dt)

                if self.tstep > self.equi_time:
                    if self.alpha_par_bool_lst[i]:
                        alpha_par_tilde[mask] += (-g1_alpha_par[mask] * self.dt) + (eta * xi_alpha_par[mask] * self.sqrt_dt)

                    if self.alpha_perp_bool_lst[i]:
                        alpha_perp_tilde[mask] += (-g1_alpha_perp[mask] * self.dt) + (eta * xi_alpha_perp[mask] * self.sqrt_dt)

                    if self.gamma_bool_lst[i]:
                        gamma_tilde[mask] += (-g1_gamma[mask] * self.dt) + (eta * xi_gamma[mask] * self.sqrt_dt)

            # Apply constraints before stage 2 drift evaluation
            p_tilde, q_tilde, alpha_par_tilde, alpha_perp_tilde, gamma_tilde = self.apply_constraints(
                p_tilde, q_tilde, p_mask, alpha_par_tilde, alpha_perp_tilde, gamma_tilde
            )

        # Make predictor state leaf tensors for stage-2 grad evaluation
        x_tilde.requires_grad_(True)
        p_tilde.requires_grad_(True)
        q_tilde.requires_grad_(True)
        alpha_par_tilde.requires_grad_(True)
        alpha_perp_tilde.requires_grad_(True)
        gamma_tilde.requires_grad_(True)

        # ------------------------
        # Stage 2 (drift at X~)
        # ------------------------
        d2, dx2, idx2, z_mask2 = self.true_neighbours_from_idx(x_tilde, p_tilde, q_tilde, gamma_tilde, idx_base)
        V2, _, _ = self.potential(x_tilde, p_tilde, q_tilde, p_mask,
                                  alpha_par_tilde, alpha_perp_tilde, gamma_tilde,
                                  d2, dx2, idx2, z_mask2)

        g2_x, g2_p, g2_q, g2_alpha_par, g2_alpha_perp, g2_gamma = torch.autograd.grad(
            V2, (x_tilde, p_tilde, q_tilde, alpha_par_tilde, alpha_perp_tilde, gamma_tilde),
            create_graph=False, retain_graph=False
        )

        # ------------------------
        # Corrector (Heun)
        # ------------------------
        with torch.no_grad():
            for i, eta in enumerate(self.eta_lst):
                mask = p_mask == i

                if self.update_x_bools[i]:
                    x[mask] += (-0.5 * (g1_x[mask] + g2_x[mask]) * self.dt) + (eta * xi_x[mask] * self.sqrt_dt)
                p[mask] += (-0.5 * (g1_p[mask] + g2_p[mask]) * self.dt) + (eta * xi_p[mask] * self.sqrt_dt)  # We add the rotational noise as an Euler step for the corrector, then apply the actual rotation after calculating the angle. This is to ensure the noise is properly scaled by eta and sqrt_dt, and to keep the code simpler.
                q[mask] += (-0.5 * (g1_q[mask] + g2_q[mask]) * self.dt) + (eta * xi_q[mask] * self.sqrt_dt)

                if self.tstep > self.equi_time:
                    if self.alpha_par_bool_lst[i]:
                        alpha_par[mask] += (-0.5 * (g1_alpha_par[mask] + g2_alpha_par[mask]) * self.dt) + (eta * xi_alpha_par[mask] * self.sqrt_dt)

                    if self.alpha_perp_bool_lst[i]:
                        alpha_perp[mask] += (-0.5 * (g1_alpha_perp[mask] + g2_alpha_perp[mask]) * self.dt) + (eta * xi_alpha_perp[mask] * self.sqrt_dt)

                    if self.gamma_bool_lst[i]:
                        gamma[mask] += (-0.5 * (g1_gamma[mask] + g2_gamma[mask]) * self.dt) + (eta * xi_gamma[mask] * self.sqrt_dt)

            # Final constraints
            p, q, alpha_par, alpha_perp, gamma = self.apply_constraints(p, q, p_mask, alpha_par, alpha_perp, gamma)

        with torch.no_grad():
            if self.stretch_factor != 0.0 and self.tstep > self.equi_time:
                if self.stretch_type == 'straight':
                    x[:,self.stretch_bound_axis][p_mask == 1] += self.stretch_factor * self.dt * torch.sign(x[p_mask == 1][:,self.stretch_bound_axis] - self.x_mass_midpoint)
                elif self.stretch_type == 'curved':
                    if self.tstep == 0:
                        self.left_ones = (p_mask == 1) * (x[:,0] < self.x_mass_midpoint)
                        self.right_ones = (p_mask == 1) * (x[:,0] >= self.x_mass_midpoint)
                        self.down_stretch = False
                    
                    # We want the vectors pointing from each p_mask == 1 cells to the line constructed by the mass midpoint
                    # and the stretch_bound_axis.
                    if not(self.down_stretch):
                        radii_left = x[self.left_ones] - self.mass_midpoint
                        radii_left[:,self.stretch_bound_axis] = 0.0
                        axis = torch.zeros_like(radii_left)
                        axis[:,self.stretch_bound_axis] = 1.0
                        direction_left = self.safe_normalize(torch.cross(radii_left, axis, dim=1), dim=1)
                        radii_right = x[self.right_ones] - self.mass_midpoint
                        radii_right[:,self.stretch_bound_axis] = 0.0
                        axis = torch.zeros_like(radii_right)
                        axis[:,self.stretch_bound_axis] = 1.0
                        direction_right = self.safe_normalize(torch.cross(axis, radii_right, dim=1), dim=1)
                        x[self.left_ones] += self.stretch_factor * self.dt * direction_left
                        x[self.right_ones] += self.stretch_factor * self.dt * direction_right

                        min_dist_left_right = x[self.left_ones][:,None] - x[self.right_ones][None,:]
                        min_dist_left_right = torch.norm(min_dist_left_right, dim=2)
                        min_dist = torch.min(min_dist_left_right)
                        if min_dist < 2.1:
                            self.down_stretch = True
                    else:
                        x[:,2][p_mask == 1] += - self.stretch_factor * self.dt #* torch.sign(x[p_mask == 1][:,self.stretch_bound_axis] - self.x_mass_midpoint)
                else:
                    raise ValueError(f"Unknown stretch_type: {self.stretch_type}")

                if self.tstep >= self.stretch_time_stop:
                    self.stretch_factor = 0.0

                    # Let pulled cells go
                    p_mask[p_mask == 1] = 0
                    self.eta_lst = [self.eta0]
                    self.polar_eta_lst = [self.polar_eta0]
                    self.alpha_par_bool_lst = [self.type0_alpha_par_free, False]
                    self.alpha_perp_bool_lst = [self.type0_alpha_perp_free, False]
                    self.gamma_bool_lst = [self.type0_gamma_free, False]
                    self.lambdas = self.lambdas[0]

        return x, p, q, p_mask, alpha_par, alpha_perp, gamma, Vi, defect_mask  #Returning the goods.

    def simulation(self, x, p, q, p_mask, alpha_par, alpha_perp, gamma):
        """
        Runs the simulation.

        Parameters:
            x (torch.Tensor): The cell positions.
            p (torch.Tensor): The apicobasal polarities.
            q (torch.Tensor): The planar cell polarities.
            p_mask (torch.Tensor or None): The particle mask.
            alpha_par (torch.Tensor): The parallel alpha parameter.
            alpha_perp (torch.Tensor): The perpendicular alpha parameter.
            gamma (torch.Tensor): The gamma (elongation) parameter.
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor or None]: The updated cell positions, apicobasal polarities, planar cell polarities, and particle mask.
        """
        
        #Initializing simulation
        x, p, q, p_mask, alpha_par, alpha_perp, gamma = self.init_simulation(x, p, q, p_mask, alpha_par, alpha_perp, gamma) 

        tstep = 0
        while True:
            x, p, q, p_mask, alpha_par, alpha_perp, gamma, energy, defect_mask = self.time_step(x, p, q, p_mask, alpha_par, alpha_perp, gamma, tstep)        #Advancing the simulation one timestep
            
            tstep += 1
            self.tstep = tstep

            if tstep % self.yield_every == 0 or len(x) > self.max_cells:    #Yield data if we are at a 'yield step' or if we have too many cells and the simulation is aborted
                
                xx = x.detach().to("cpu").numpy().copy()                    #Copying data to CPU
                pp = p.detach().to("cpu").numpy().copy()
                qq = q.detach().to("cpu").numpy().copy()

                alpha_parpar = alpha_par.detach().clone() * 180.0/np.pi
                alpha_parpar = alpha_parpar.to("cpu").numpy()

                alpha_perpperp = alpha_perp.detach().clone() * 180.0/np.pi
                alpha_perpperp = alpha_perpperp.to("cpu").numpy()

                gammagamma = torch.exp(gamma.detach().clone())
                gammagamma = gammagamma.to("cpu").numpy()

                energy = energy.detach().to("cpu").numpy().copy()
                pp_mask = p_mask.detach().to("cpu").numpy().copy()
                defect_mask_out = defect_mask.detach().to("cpu").numpy().copy()
                
                yield xx, pp, qq, pp_mask, alpha_parpar, alpha_perpperp, gammagamma, energy, defect_mask_out                                  # Yielding the data baybeeee
    
    def cell_division(self, x, p, q, p_mask, alpha_par, alpha_perp, gamma):
        """
        Handles cell division events.

        Parameters:
            x (torch.Tensor): The cell positions.
            p (torch.Tensor): The apicobasal polarities.
            q (torch.Tensor): The planar cell polarities.
            p_mask (torch.Tensor or None): The particle mask.
            alpha_par (torch.Tensor): The parallel alpha parameter.
            alpha_perp (torch.Tensor): The perpendicular alpha parameter.
            gamma (torch.Tensor): The gamma (elongation) parameter.

        Returns:
            Tuple[bool, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor or None, torch.Tensor]: A tuple containing:
                - division (bool): Whether division occurred.
                - x (torch.Tensor): Cell positions with new cells
                - p (torch.Tensor): Apicobasal polarities with new cells
                - q (torch.Tensor): Planar cell polarities with new cells
                - p_mask (torch.Tensor or None): The updated particle mask.
                - beta (torch.Tensor): The division probabilities.
                - alpha_par (torch.Tensor): The parallel alpha parameter with new cells.
                - alpha_perp (torch.Tensor): The perpendicular alpha parameter with new cells.
                - gamma (torch.Tensor): The gamma (elongation) parameter with new cells.
        """

        beta = self.beta            

        if torch.sum(beta) < 1e-8:              # If division probabilities are negligible no division occurs
            return False, x, p, q, p_mask, beta, alpha_par, alpha_perp, gamma

        # set probability according to beta and dt
        d_prob = beta
        # flip coins
        draw = torch.empty_like(beta).uniform_()
        # find successes
        events = draw < d_prob
        division = False

        if torch.sum(events) > 0:
            with torch.no_grad():
                division = True
                # find cells that will divide
                idx = torch.nonzero(events)[:, 0]

                x0      = x[idx, :]
                p0      = p[idx, :]
                q0      = q[idx, :]
                p_mask0 = p_mask[idx]
                b0      = beta[idx]
                alpha_par0 = alpha_par[idx]
                alpha_perp0 = alpha_perp[idx]
                gamma0 = gamma[idx]

                # make a random vector
                move = torch.empty_like(x0).normal_()

                # place new cells
                x0 = x0 + move

                # append new cell data to the system state
                x = torch.cat((x, x0))
                p = torch.cat((p, p0))
                q = torch.cat((q, q0))
                p_mask = torch.cat((p_mask, p_mask0))
                beta = torch.cat((beta, b0))
                alpha_par = torch.cat((alpha_par, alpha_par0))
                alpha_perp = torch.cat((alpha_perp, alpha_perp0))
                gamma = torch.cat((gamma, gamma0))

        x.requires_grad = True
        p.requires_grad = True
        q.requires_grad = True
        alpha_par.requires_grad = True
        alpha_perp.requires_grad = True
        gamma.requires_grad = True

        return division, x, p, q, p_mask, beta, alpha_par, alpha_perp, gamma      #Returning the goods.

def save(data_tuple, name, output_folder):
    """
    Saves the simulation data to a pickle file with dict structure.

    Parameters:
        data_tuple (Tuple): (p_mask_lst, x_lst, p_lst, q_lst, alpha_par_lst, alpha_perp_lst, gamma_lst, energy_lst, defect_mask_lst)
        name (str): The name of the file (without extension).
        output_folder (str): The folder to save the file in.

    Returns:
        None, but saves the data
    """
    p_mask_lst, x_lst, p_lst, q_lst, alpha_par_lst, alpha_perp_lst, gamma_lst, energy_lst, defect_mask_lst = data_tuple

    energy_lst_copy = energy_lst.copy()
    last_energy = np.zeros_like(p_mask_lst[-1])
    energy_lst_copy.append(last_energy)

    defect_mask_lst_copy = defect_mask_lst.copy()
    last_defect_mask = np.zeros(p_mask_lst[-1].shape[0], dtype=bool)
    defect_mask_lst_copy.append(last_defect_mask)

    # Structure: query by variable name to get list across all timeframes
    data_dict = {
        'x': x_lst,           
        'p': p_lst,           
        'q': q_lst,           
        'p_mask': p_mask_lst, 
        'alpha_par': alpha_par_lst,
        'alpha_perp': alpha_perp_lst,
        'gamma': gamma_lst,
        'energy': energy_lst_copy,
        'defect_mask': defect_mask_lst_copy
    }
    
    with open(f'{output_folder}/{name}.pkl', 'wb') as f:
        pickle.dump(data_dict, f)


def run_simulation(sim_dict):
    """
    External simulation runner.

    Parameters:
        sim_dict (dict): The simulation parameters and data.

    Returns:
        None
    """

    # Make the simulation runner object:
    data        = sim_dict.pop('data')           # We don't want to save the data in a .json file so we pop it
    verbose     = sim_dict.pop('verbose')        # This is not really important info other, so we pop it too
    yield_steps, yield_every = sim_dict['yield_steps'], sim_dict['yield_every'] # How often to yield data

    np.random.seed(sim_dict['random_seed'])     # Setting the random seed

    if isinstance(data, dict):
        print('Using input data from dictionary')
        p_mask = data['p_mask']
        x = data['x']
        p = data['p']
        q = data['q']

        alpha_params = sim_dict['alpha_params']
        gamma_params = sim_dict['gamma_params']

        alpha_par = np.zeros_like(p_mask, dtype=np.float32)
        alpha_perp = np.zeros_like(p_mask, dtype=np.float32)
        gamma = np.zeros_like(p_mask, dtype=np.float32)

        if np.unique(p_mask).size > 1:
            assert isinstance(alpha_params[0], list),   "Expected alpha_params to be a list of lists for multiple cell types"
            assert isinstance(gamma_params, list),      "Expected gamma_params to be a list for multiple cell types"

            # Setting initial alpha values
            alpha_par[p_mask == 0]    = alpha_params[0][0][0] * np.pi/180.0
            alpha_perp[p_mask == 0]   = alpha_params[0][1][0] * np.pi/180.0
            alpha_par[p_mask == 1]    = alpha_params[1][0][0] * np.pi/180.0
            alpha_perp[p_mask == 1]   = alpha_params[1][1][0] * np.pi/180.0

            # Setting initial gamma values
            gamma[p_mask == 0]        = np.log(gamma_params[0][0])
            gamma[p_mask == 1]        = np.log(gamma_params[1][0])
        else:
            alpha_par[:]    = alpha_params[0][0] * np.pi/180.0
            alpha_perp[:]   = alpha_params[1][0] * np.pi/180.0
            gamma[:]        = np.log(gamma_params[0])

    elif isinstance(data, tuple):
        # Data generation tuple construction: (data_gen, data_gen_args)
        print('Using data generation function')
        data_gen_function = data[0]
        data_gen_parameters = data[1]                    
        alpha_params = sim_dict['alpha_params']
        gamma_params = sim_dict['gamma_params']
        p_mask, x, p, q, alpha_par, alpha_perp, gamma = data_gen_function(*data_gen_parameters,
                                                                          alpha_params=alpha_params,
                                                                          gamma_params=gamma_params) 
    else:
        raise ValueError('Data should be either a dictionary or a tuple (data_gen_function, data_gen_parameters)')

        
    sim = Simulation(sim_dict)                  # Initializing an instance of the Simulation class
    runner = sim.simulation(x, p, q, p_mask, alpha_par, alpha_perp, gamma)    # Making a runner

    output_folder = sim_dict['output_folder']   # The folder to save the output data

    # Create the output folder if it doesn't exist
    try: 
        os.mkdir(output_folder)
    except:
        pass

    # Initialize lists to store simulation data
    p_mask_lst = [p_mask]
    x_lst = [x]
    p_lst = [p]
    q_lst = [q]
    alpha_par_lst = [alpha_par * 180.0/np.pi]
    alpha_perp_lst = [alpha_perp * 180.0/np.pi]
    gamma_lst = [np.exp(gamma)]
    energy_lst = []
    defect_mask_lst = []

    # Save the simulation dictionary
    with open(output_folder + '/sim_dict.json', 'w') as f:
        sim_dict['dtype'] = str(sim_dict['dtype'])
        json.dump(sim_dict, f, indent = 2)

    # Save the initial simulation data
    save((p_mask_lst, x_lst, p_lst,  q_lst, alpha_par_lst, alpha_perp_lst, gamma_lst, energy_lst, defect_mask_lst), name='data', output_folder=output_folder)

    # Print the notes if verbose
    if verbose:
        notes = sim_dict['notes']
        print('Starting simulation with notes:')
        print(notes)

    i = 0       # Initialize the iteration counter
    t1 = time() # We timing stuff

    # Saving data at intervals specified by yield_every:
    for xx, pp, qq, pp_mask, alpha_parpar, alpha_perpperp, gammagamma, energyenergy, defect_maskenergy in itertools.islice(runner, yield_steps):
        i += 1
        if verbose:
            print(f'Running {i} of {yield_steps}   ({yield_every * i} of {yield_every * yield_steps})   ({len(xx)} cells)', end='\r')

        x_lst.append(xx)
        p_lst.append(pp)
        q_lst.append(qq)
        p_mask_lst.append(pp_mask)
        alpha_par_lst.append(alpha_parpar)
        alpha_perp_lst.append(alpha_perpperp)
        gamma_lst.append(gammagamma)
        energy_lst.append(energyenergy)
        defect_mask_lst.append(defect_maskenergy)
        
        if len(pp_mask) > sim_dict['max_cells']:
            break
        
        # Every 50 yield steps we dump the data
        if i % 50 == 0:
            save((p_mask_lst, x_lst, p_lst,  q_lst, alpha_par_lst, alpha_perp_lst, gamma_lst, energy_lst, defect_mask_lst), name='data', output_folder=output_folder)
    
    if verbose:
        print(f'Simulation done, saved {i} datapoints')
        print('Took', time() - t1, 'seconds')

    save((p_mask_lst, x_lst, p_lst,  q_lst, alpha_par_lst, alpha_perp_lst, gamma_lst, energy_lst, defect_mask_lst), name='data', output_folder=output_folder)  # Last iteration is saved

def make_sphere(N, type0_frac, alpha_params=None, gamma_params=None):
    """
    Generates cells uniformly distributed within a sphere with randomly
    initialized polarities

    Parameters
        N (int): The number of cells to generate.
        type0_frac (float): The fraction of cells of type 0.
        radius (float): The radius of the sphere.

    Returns
        tuple: A tuple containing the following elements:
            - mask (np.ndarray): The mask indicating the type of each cell.
            - x (np.ndarray): The positions of the cells.
            - p (np.ndarray): The apicobasal polarities of the cells.
            - q (np.ndarray): The planar cell polarities of the cells
    """
    cell_area = np.sqrt(3) / 2 * 2.1**2
    radius = np.sqrt(N * cell_area / (4 * np.pi))
    range = np.arange(N)
    angle = np.pi * (3.0 - np.sqrt(5.0))
    z = 1 - (2*range + 1)/N
    r = np.sqrt(1 - z*z)
    theta = angle * range
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    x = radius * np.column_stack((x, y, z))

    p = x / np.linalg.norm(x, axis=1)[:, None]
    q = np.cross(p, np.array([1, 0, 0])) / np.linalg.norm(np.cross(p, np.array([1, 0, 0])), axis=1)[:, None]


    # # Generate random cell types with specified fractions
    mask = np.random.choice([0,1], p=[type0_frac, 1-type0_frac], size=N)                #Mask detailing which cells are which type

    alpha_par = np.zeros(N)
    alpha_perp = np.zeros(N)
    gamma = np.zeros(N)

    # check for unique values in mask. If only one unique value, we can set mask to None and save some time later on.
    if np.unique(mask).size > 1:
        
        assert isinstance(alpha_params[0], list),   "Expected alpha_params to be a list of lists for multiple cell types"
        assert isinstance(gamma_params, list),      "Expected gamma_params to be a list for multiple cell types"

        # Setting initial alpha values
        alpha_par[mask == 0]    = alpha_params[0][0][0] * np.pi/180.0
        alpha_perp[mask == 0]   = alpha_params[0][1][0] * np.pi/180.0
        alpha_par[mask == 1]    = alpha_params[1][0][0] * np.pi/180.0
        alpha_perp[mask == 1]   = alpha_params[1][1][0] * np.pi/180.0

        # Setting initial gamma values
        gamma[mask == 0]        = np.log(gamma_params[0][0])
        gamma[mask == 1]        = np.log(gamma_params[1][0])
    else:
        alpha_par[:]    = alpha_params[0][0] * np.pi/180.0
        alpha_perp[:]   = alpha_params[1][0] * np.pi/180.0
        gamma[:]        = np.log(gamma_params[0])

    sphere_data = (mask, x, p, q, alpha_par, alpha_perp, gamma)

    return sphere_data

def make_sphere_surface_stretch(N, stretch_frac, mirrored=True, 
                               radius=30, alpha_params=None, gamma_params=None):
    """
    Generates cells uniformly distributed on a sphere 
    with abp polarities pointing radially outward and randomly initialized pcp polarities.

    Parameters
        N (int): The number of cells to generate.
        stretch_frac (float): The fraction of cells that will be stretched.
        mirrored (bool): Whether to stretch to both sides or only 1.
        radius (float): The radius of the sphere.

    Returns
        tuple: A tuple containing the following elements:
            - mask (np.ndarray): The mask indicating the type of each cell.
            - x (np.ndarray): The positions of the cells.
            - p (np.ndarray): The apicobasal polarities of the cells.
            - q (np.ndarray): The planar cell polarities of the cells
    """

    # Generate random positions on a sphere
    x = np.random.randn(N, 3)
    x /= np.sqrt(np.sum(x**2, axis=1))[:, None]
    x *= radius

    # Generate apicobasal polarities pointing radially outward
    p = x / np.linalg.norm(x, axis=1)[:, None]

    # Generate random planar cell polarities
    q = np.random.randn(N, 3)
    q /= np.sqrt(np.sum(q**2, axis=1))[:,None]

    # Generate cell types based on distance from the center
    mask = np.zeros(N, dtype=int)
    # All cells in the stretch_frace fraction of the radius from the center are type 1
    # Sorting cells by their distance from the center
    sorted_indices = np.argsort(x[:,0])  # Sort by x-coordinate
    if mirrored:
        mask[sorted_indices[:int(N*stretch_frac/2)]] = 1
        mask[sorted_indices[int(N*(1-stretch_frac/2)):]] = 1
    else:
        mask[sorted_indices[:int(N*stretch_frac)]] = 1

    alpha_par = np.zeros(N)
    alpha_perp = np.zeros(N)
    gamma = np.zeros(N)

    # check for unique values in mask. If only one unique value, we can set mask to None and save some time later on.
    if np.unique(mask).size > 1:
        
        assert isinstance(alpha_params[0], list),   "Expected alpha_params to be a list of lists for multiple cell types"
        assert isinstance(gamma_params, list),      "Expected gamma_params to be a list for multiple cell types"

        # Setting initial alpha values
        alpha_par[mask == 0]    = alpha_params[0][0][0] * np.pi/180.0
        alpha_perp[mask == 0]   = alpha_params[0][1][0] * np.pi/180.0
        alpha_par[mask == 1]    = alpha_params[1][0][0] * np.pi/180.0
        alpha_perp[mask == 1]   = alpha_params[1][1][0] * np.pi/180.0

        # Setting initial gamma values
        gamma[mask == 0]        = np.log(gamma_params[0][0])
        gamma[mask == 1]        = np.log(gamma_params[1][0])
    else:
        alpha_par[:]    = alpha_params[0][0] * np.pi/180.0
        alpha_perp[:]   = alpha_params[1][0] * np.pi/180.0
        gamma[:]        = np.log(gamma_params[0])


    sphere_data = (mask, x, p, q, alpha_par, alpha_perp, gamma)
    return sphere_data

def make_stretch_plain(N, stretch_frac, alpha_params=None, gamma_params=None):
    """
    Generates cells in the xy-plane with abp polarities pointing in the z-direction and randomly initialized pcp polarities.
    N is a side so the total number of cells is N^2.
    Cells are initially placed in a grid 2 units apart.
    The stretch_frac works as in make_sphere_surface_stretch,
        stretch_frac (float): The fraction of cells that will be stretched. Same as make_sphere_surface_stretch
        size (float): The size of the plain.

    Returns
        tuple: A tuple containing the following elements:
            - mask (np.ndarray): The mask indicating the type of each cell.
            - x (np.ndarray): The positions of the cells.
            - p (np.ndarray): The apicobasal polarities of the cells.
            - q (np.ndarray): The planar cell polarities of the cells
            - alpha_par (np.ndarray): The parallel alpha parameter for each cell.
            - alpha_perp (np.ndarray): The perpendicular alpha parameter for each cell.
            - gamma_par (np.ndarray): The parallel gamma (elongation) parameter for each cell.
            - gamma_perp (np.ndarray): The perpendicular gamma (elongation) parameter for each cell.
    """

    # Generate grid positions in the xy-plane
    x = np.array([[i*2, j*2, 0] for i in range(N) for j in range(N)], dtype=float)

    # Generate apicobasal polarities pointing in the z-direction
    p = np.array([[0, 0, 1] for _ in range(N**2)], dtype=float)

    # Generate random planar cell polarities
    q = np.array([[0, 1, 0] for _ in range(N**2)], dtype=float)
    # q = np.random.randn(N**2, 3)
    # q /= np.sqrt(np.sum(q**2, axis=1))[:,None]

    # Generate cell types based on distance from the center
    mask = np.zeros(N**2, dtype=int)
    # All cells in the stretch_frace fraction of the radius from the center are type 1
    # Sorting cells by their distance from the center
    sorted_indices = np.argsort(x[:,0])  # Sort by x-coordinate

    mask[sorted_indices[:int(N**2 * stretch_frac/2)]] = 1
    mask[sorted_indices[int(N**2 * (1-stretch_frac/2)):]] = 1

    alpha_par = np.zeros(N**2)
    alpha_perp = np.zeros(N**2)
    gamma = np.zeros(N**2)

    # check for unique values in mask. If only one unique value, we can set mask to None and save some time later on.
    if np.unique(mask).size > 1:
        
        assert isinstance(alpha_params[0], list),   "Expected alpha_params to be a list of lists for multiple cell types"
        assert isinstance(gamma_params, list),      "Expected gamma_params to be a list for multiple cell types"

        # Setting initial alpha values
        alpha_par[mask == 0]    = alpha_params[0][0][0] * np.pi/180.0
        alpha_perp[mask == 0]   = alpha_params[0][1][0] * np.pi/180.0
        alpha_par[mask == 1]    = alpha_params[1][0][0] * np.pi/180.0
        alpha_perp[mask == 1]   = alpha_params[1][1][0] * np.pi/180.0

        # Setting initial gamma values
        gamma[mask == 0]        = np.log(gamma_params[0][0])
        gamma[mask == 1]       = np.log(gamma_params[1][0])
    else:
        alpha_par[:]    = alpha_params[0][0] * np.pi/180.0
        alpha_perp[:]   = alpha_params[1][0] * np.pi/180.0
        gamma[:]        = np.log(gamma_params[0])

    plain_data = (mask, x, p, q, alpha_par, alpha_perp, gamma)

    return plain_data

def make_stretch_rectangle(length, width, stretch_frac, alpha_params=None, gamma_params=None):
    """
    Generates cells in the xy-plane with abp polarities pointing in the z-direction and randomly initialized pcp polarities.
    N is a side so the total number of cells is N^2.
    Cells are initially placed in a grid 2 units apart.
    The stretch_frac works as in make_sphere_surface_stretch,
        stretch_frac (float): The fraction of cells that will be stretched. Same as make_sphere_surface_stretch
        size (float): The size of the plain.

    Returns
        tuple: A tuple containing the following elements:
            - mask (np.ndarray): The mask indicating the type of each cell.
            - x (np.ndarray): The positions of the cells.
            - p (np.ndarray): The apicobasal polarities of the cells.
            - q (np.ndarray): The planar cell polarities of the cells
            - alpha_par (np.ndarray): The parallel alpha parameter for each cell.
            - alpha_perp (np.ndarray): The perpendicular alpha parameter for each cell.
            - gamma_par (np.ndarray): The parallel gamma (elongation) parameter for each cell.
            - gamma_perp (np.ndarray): The perpendicular gamma (elongation) parameter for each cell.
    """

    # Generate grid positions in the xy-plane
    x = np.array([[i*2, j*2, 0] for i in range(length) for j in range(width)], dtype=float)

    # Generate apicobasal polarities pointing in the z-direction
    p = np.array([[0, 0, 1] for _ in range(length * width)], dtype=float)

    # Generate random planar cell polarities
    q = np.array([[0, 1, 0] for _ in range(length * width)], dtype=float)
    # q = np.random.randn(length * width, 3)
    # q /= np.sqrt(np.sum(q**2, axis=1))[:,None]

    # Generate cell types based on distance from the center
    mask = np.zeros(length * width, dtype=int)
    # All cells in the stretch_frace fraction of the radius from the center are type 1
    # Sorting cells by their distance from the center
    sorted_indices = np.argsort(x[:,0])  # Sort by x-coordinate

    mask[sorted_indices[:int(length * width * stretch_frac/2)]] = 1
    mask[sorted_indices[int(length * width * (1-stretch_frac/2)):]] = 1

    alpha_par = np.zeros(length * width)
    alpha_perp = np.zeros(length * width)
    gamma = np.zeros(length * width)

    # check for unique values in mask. If only one unique value, we can set mask to None and save some time later on.
    if np.unique(mask).size > 1:
        
        assert isinstance(alpha_params[0], list),   "Expected alpha_params to be a list of lists for multiple cell types"
        assert isinstance(gamma_params, list),      "Expected gamma_params to be a list for multiple cell types"

        # Setting initial alpha values
        alpha_par[mask == 0]    = alpha_params[0][0][0] * np.pi/180.0
        alpha_perp[mask == 0]   = alpha_params[0][1][0] * np.pi/180.0
        alpha_par[mask == 1]    = alpha_params[1][0][0] * np.pi/180.0
        alpha_perp[mask == 1]   = alpha_params[1][1][0] * np.pi/180.0

        # Setting initial gamma values
        gamma[mask == 0]        = np.log(gamma_params[0][0])
        gamma[mask == 1]       = np.log(gamma_params[1][0])
    else:
        alpha_par[:]    = alpha_params[0][0] * np.pi/180.0
        alpha_perp[:]   = alpha_params[1][0] * np.pi/180.0
        gamma[:]        = np.log(gamma_params[0])

    plain_data = (mask, x, p, q, alpha_par, alpha_perp, gamma)

    return plain_data

def make_stretch_cylinder(circumference, length, stretch_frac, polarity='par', mirrored=True, alpha_params=None, gamma_params=None):
    """
    Generates cells on the surface of a cylinder with apicobasal polarities pointing radially outward and randomly initialized pcp polarities.

    Parameters
        circumference (int): The circumference of the cylinder in number of cells.
        length (int): The length of the cylinder in number of cells.
        stretch_frac (float): The fraction of cells that will be stretched. Same as make_sphere_surface_stretch
        mirrored (bool): Whether to stretch to both sides or only 1.

    Returns
        tuple: A tuple containing the following elements:
            - mask (np.ndarray): The mask indicating the type of each cell.
            - x (np.ndarray): The positions of the cells.
            - p (np.ndarray): The apicobasal polarities of the cells.
            - q (np.ndarray): The planar cell polarities of the cells
            - alpha_par (np.ndarray): The parallel alpha parameter for each cell.
            - alpha_perp (np.ndarray): The perpendicular alpha parameter for each cell.
            - gamma_par (np.ndarray): The parallel gamma (elongation) parameter for each cell.
            - gamma_perp (np.ndarray): The perpendicular gamma (elongation) parameter for each cell.
    """

    # Generate positions on the surface of a cylinder
    radius = (2.2 * circumference) / (2 * np.pi)
    theta = np.linspace(0, 2 * np.pi, circumference, endpoint=False)
    z = np.linspace(0, length*2, length, endpoint=False)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x = np.zeros((length * circumference, 3))
    x[:, 0] = radius * np.cos(theta_grid.flatten())
    x[:, 1] = radius * np.sin(theta_grid.flatten())
    x[:, 2] = z_grid.flatten()

    # Generate apicobasal polarities pointing radially outward
    p = np.zeros_like(x)
    p[:, 0] = np.cos(theta_grid.flatten())
    p[:, 1] = np.sin(theta_grid.flatten())

    # Generate planar cell polarities pointing along the cylinder
    if polarity == 'par':
        q = np.zeros_like(x)
        q[:, 2] = 1
    elif polarity == 'perp':
        q = np.zeros_like(x)
        q[:, 0] = -np.sin(theta_grid.flatten())
        q[:, 1] = np.cos(theta_grid.flatten())
        q /= np.linalg.norm(q, axis=1)[:, None]
    else:
        raise ValueError("Polarity must be either 'par' or 'perp'.")

    N = (length * circumference)
    # Generate cell types based on distance from the center
    mask = np.zeros(N, dtype=int)
    # All cells in the stretch_frace fraction of the radius from the center are type 1
    # Sorting cells by their distance from the center
    sorted_indices = np.argsort(x[:,0])  # Sort by x-coordinate
    if mirrored:
        mask[sorted_indices[:int(N*stretch_frac/2)]] = 1
        mask[sorted_indices[int(N*(1-stretch_frac/2)):]] = 1
    else:
        mask[sorted_indices[:int(N*stretch_frac)]] = 1
    alpha_par = np.zeros(N)
    alpha_perp = np.zeros(N)
    gamma = np.zeros(N)
    # check for unique values in mask. If only one unique value, we can set mask to None and save some time later on.
    if np.unique(mask).size > 1:
        
        assert isinstance(alpha_params[0], list),   "Expected alpha_params to be a list of lists for multiple cell types"
        assert isinstance(gamma_params, list),      "Expected gamma_params to be a list for multiple cell types"

        # Setting initial alpha values
        alpha_par[mask == 0]    = alpha_params[0][0][0] * np.pi/180.0
        alpha_perp[mask == 0]   = alpha_params[0][1][0] * np.pi/180.0
        alpha_par[mask == 1]    = alpha_params[1][0][0] * np.pi/180.0
        alpha_perp[mask == 1]   = alpha_params[1][1][0] * np.pi/180.0

        # Setting initial gamma values
        gamma[mask == 0]        = np.log(gamma_params[0][0])
        gamma[mask == 1]       = np.log(gamma_params[1][0])
    else:
        alpha_par[:]    = alpha_params[0][0] * np.pi/180.0
        alpha_perp[:]   = alpha_params[1][0] * np.pi/180.0
        gamma[:]        = np.log(gamma_params[0])

    cylinder_data = (mask, x, p, q, alpha_par, alpha_perp, gamma)
    return cylinder_data

def make_4cells_on_string(mask=None, alpha_params=None, gamma_params=None):
    x = np.array([[-3,0,0], [-1,0,0], [1,0,0], [3,0,0]])
    p = np.array([[0,1,0], [0,1,0], [0,1,0], [0,1,0]])
    q = np.array([[0,0,1], [0,0,1], [0,0,1], [0,0,1]])
    if mask is None:
        print('We doing mixed types')
        mask = np.array([1,0,1,0])                

    alpha_par = np.zeros(4)
    alpha_perp = np.zeros(4)
    gamma = np.zeros(4)

    # check for unique values in mask. If only one unique value, we can set mask to None and save some time later on.
    if np.unique(mask).size > 1:
        
        assert isinstance(alpha_params[0], list),   "Expected alpha_params to be a list of lists for multiple cell types"
        assert isinstance(gamma_params, list),      "Expected gamma_params to be a list for multiple cell types"

        # Setting initial alpha values
        alpha_par[mask == 0]    = alpha_params[0][0][0] * np.pi/180.0
        alpha_perp[mask == 0]   = alpha_params[0][1][0] * np.pi/180.0
        alpha_par[mask == 1]    = alpha_params[1][0][0] * np.pi/180.0
        alpha_perp[mask == 1]   = alpha_params[1][1][0] * np.pi/180.0

        # Setting initial gamma values
        gamma[mask == 0]        = np.log(gamma_params[0][0])
        gamma[mask == 1]        = np.log(gamma_params[1][0])
    else:
        alpha_par[:]    = alpha_params[0][0] * np.pi/180.0
        alpha_perp[:]   = alpha_params[1][0] * np.pi/180.0
        gamma[:]        = np.log(gamma_params[0])


    string_data = (mask, x, p, q, alpha_par, alpha_perp, gamma)
    return string_data

def make_two_particles_on_string(q_dir,p_mask=None,alpha_params=None, gamma_params=None):
    """Create a simulation with exactly 2 cells for testing."""
    # Create 2 particles on a 1D string
    x = np.array([[-1, 0, 0], [1, 0, 0]])
    p = np.array([[0, 1, 0], [0, 1, 0]])
    q = np.zeros_like(p)
    q[:, q_dir] = 1
    print(q)
    if p_mask is None:    
        mask = np.array([0, 0])  # Two different cell types for testing
    else:
        mask = p_mask

    alpha_par = np.zeros(2)
    alpha_perp = np.zeros(2)
    gamma = np.zeros(2)

    # Check for unique values in mask. If only one unique value, we can set mask to None and save some time later on.
    if np.unique(mask).size > 1:
        assert isinstance(alpha_params[0], list), "Expected alpha_params to be a list of lists for multiple cell types"
        assert isinstance(gamma_params, list), "Expected gamma_params to be a list for multiple cell types"

        # Setting initial alpha values
        alpha_par[mask == 0] = alpha_params[0][0][0] * np.pi / 180.0
        alpha_perp[mask == 0] = alpha_params[0][1][0] * np.pi / 180.0
        alpha_par[mask == 1] = alpha_params[1][0][0] * np.pi / 180.0
        alpha_perp[mask == 1] = alpha_params[1][1][0] * np.pi / 180.0

        # Setting initial gamma values
        gamma[mask == 0] = np.log(gamma_params[0][0])
        gamma[mask == 1] = np.log(gamma_params[1][0])
    else:
        alpha_par[:] = alpha_params[0][0] * np.pi / 180.0
        alpha_perp[:] = alpha_params[1][0] * np.pi / 180.0
        gamma[:] = np.log(gamma_params[0])

    two_cell_data = (mask, x, p, q, alpha_par, alpha_perp, gamma)
    return two_cell_data

def make_three_particles_on_string(q_dir,p_mask=None,alpha_params=None, gamma_params=None):
    """Create a simulation with exactly 3 cells for testing."""
    # Create 3 particles on a 1D string
    x = np.array([[-2, 0, 0], [0, 0, 0], [2, 0, 0]])
    p = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
    q = np.zeros_like(p)
    q[:, q_dir] = 1
    print(q)
    if p_mask is None:    
        mask = np.array([0, 0, 0])  # Two different cell types for testing
    else:
        mask = p_mask

    alpha_par = np.zeros(3)
    alpha_perp = np.zeros(3)
    gamma = np.zeros(3)

    # Check for unique values in mask. If only one unique value, we can set mask to None and save some time later on.
    if np.unique(mask).size > 1:
        assert isinstance(alpha_params[0], list), "Expected alpha_params to be a list of lists for multiple cell types"
        assert isinstance(gamma_params, list), "Expected gamma_params to be a list for multiple cell types"

        # Setting initial alpha values
        alpha_par[mask == 0] = alpha_params[0][0][0] * np.pi / 180.0
        alpha_perp[mask == 0] = alpha_params[0][1][0] * np.pi / 180.0
        alpha_par[mask == 1] = alpha_params[1][0][0] * np.pi / 180.0
        alpha_perp[mask == 1] = alpha_params[1][1][0] * np.pi / 180.0

        # Setting initial gamma values
        gamma[mask == 0] = np.log(gamma_params[0][0])
        gamma[mask == 1] = np.log(gamma_params[1][0])
    else:
        alpha_par[:] = alpha_params[0][0] * np.pi / 180.0
        alpha_perp[:] = alpha_params[1][0] * np.pi / 180.0
        gamma[:] = np.log(gamma_params[0])
    
    three_cell_data = (mask, x, p, q, alpha_par, alpha_perp, gamma)
    return three_cell_data