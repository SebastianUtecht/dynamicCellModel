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
        etas                = sim_dict['etas']                                                          # Noise strength
        
        if isinstance(etas, list):
            self.eta0, self.eta1 = etas
            self.eta_lst = [self.eta0, self.eta1]
        elif isinstance(etas, (float, int)):
            self.eta0 = self.eta1 = etas
            self.eta_lst = [self.eta0]
        else:
            raise ValueError("etas should be either a list of two values or a single value")
        
        self.alpha_params   = sim_dict['alpha_params']      # Determines how cells wedge
        self.gamma_params   = sim_dict['gamma_params']      # Determines how cells elongate
        self.prolif_rate    = sim_dict['prolif_rate']       # Probability of cell proliferation for each cell
        self.prolif_delay   = sim_dict['prolif_delay']      # Delay before cell proliferation
        self.nematic_pcp    = sim_dict['nematic_pcp']     # Whether planar cell polarity is nematic (True) or vectorial (False)
        self.update_cells_bools = sim_dict['update_cells_bools']   # List of booleans determining whether to update the parameters for each cell type. Order is [type0_alpha_par, type0_alpha_perp, type0_gamma, type1_alpha_par, type1_alpha_perp, type1_gamma]
        self.screen_out_defects = sim_dict['screen_out_defects']   # Whether to screen out defects in the neighbor calculations. Only relevant if neighbour_type is 'voronoi'
        
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
        self.use_q_mean = sim_dict['use_q_mean']
        self.elong_func_type = sim_dict['elong_func_type']
        if self.elong_func_type == 'linear':
            self.elong_func = self.elong_func_linear
        elif self.elong_func_type == 'cos':
            self.elong_func = self.elong_func_cos
        else:
            raise ValueError("elong_func_type should be either 'linear' or 'cos'")
        
        self.use_trans_neighbors = True
        self.use_gamma_mean = sim_dict['use_gamma_mean']
        self.gamma_diff_penalty = sim_dict['gamma_diff_penalty']
        self.gamma_update_speed = sim_dict['gamma_update_speed']    

        # Set random seed
        torch.manual_seed(self.random_seed)                 # For reproducibility
        self.tstep = 0
    
    def elong_func_linear(self, q_mean, dx):
        dot = (q_mean * dx).sum(dim=2)
        dot = torch.abs(dot)
        elong =  1 - 4/np.pi * torch.arccos(dot)
        return elong
    
    def elong_func_cos(self, q_mean, dx):
        return 2 * ((q_mean * dx).sum(dim=2))**2 - 1

    def safe_normalize(self, v, dim=-1, eps: float = 1e-12):
        """Normalise vectors safely (avoids NaNs when norm is ~0)."""
        n = torch.linalg.norm(v, dim=dim, keepdim=True)
        return v / torch.clamp(n, min=eps)

    @staticmethod
    def find_potential_neighbours(x, k):

        with torch.no_grad():
            # cKD-tree method
            x_cpu = x.cpu().numpy()
            tree = cKDTree(x_cpu)
            d, idx = tree.query(x_cpu, k=k+1)  # Get the k+1 nearest neighbors (including self)
            d = torch.from_numpy(d).to(x.device).to(x.dtype)
            idx = torch.from_numpy(idx).to(x.device).long()

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
        if self.use_gamma_mean:
            gamma_mean = torch.log((torch.exp(gamma_i) + torch.exp(gamma_j))/2)
        else:
            gamma_mean = gamma_i

        elong = self.elong_func(q_mean, dx)

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
            if self.use_q_mean:
                qj = q[idx]
                q_mean = (qi + qj)
                q_mean = self.safe_normalize(q_mean, dim=2)
            else:
                q_mean = qi
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
        if self.use_q_mean:
            q_mean = (qi + qj)
            q_mean = self.safe_normalize(q_mean, dim=2)

            p_mean = (pi + pj)
            p_mean = self.safe_normalize(p_mean, dim=2)
        else:
            q_mean = qi
            p_mean = pi
        
        # Expanding alpha_par, alpha_perp
        alpha_par_i = alpha_par[:, None].expand(alpha_par.shape[0], idx.shape[1])
        alpha_par_j = alpha_par[idx]
        alpha_par_mean = (alpha_par_i + alpha_par_j) / 2.0
        alpha_par_mean = torch.tan(alpha_par_mean/2)
        alpha_perp_i = alpha_perp[:, None].expand(alpha_perp.shape[0], idx.shape[1])
        alpha_perp_j = alpha_perp[idx]
        alpha_perp_mean = (alpha_perp_i + alpha_perp_j) / 2.0
        alpha_perp_mean = torch.tan(alpha_perp_mean/2)

        # Implementing cell wedging
        perp_dir = torch.cross(q_mean, pi, dim=2)
        perp_dir = perp_dir / torch.sqrt(torch.sum(perp_dir**2, dim=2))[:,:,None]

        Z_par = alpha_par_mean[:,:,None] * (q_mean * dx).sum(dim=2)[:,:,None] * q_mean                                    
        Z_perp = alpha_perp_mean[:,:,None] * (perp_dir * dx).sum(dim=2)[:,:,None] * perp_dir                                
        Z = Z_par + Z_perp

        pi_tilde = pi - Z
        pj_tilde = pj + Z

        # Normalizing the ABPs
        # wedged_interactions = torch.any((alpha_par_mean > 1e-5) | (alpha_perp_mean > 1e-5), dim=2)     # We only normalize the ABPs for wedged interactions, otherwise we mess with the non-wedged interactions for no reason
        wedged_interactions = torch.logical_or(alpha_par_mean > 1e-5, alpha_perp_mean > 1e-5)     # We only normalize the ABPs for wedged interactions, otherwise we mess with the non-wedged interactions for no reason
        pi_tilde[wedged_interactions] = pi_tilde[wedged_interactions] / torch.sqrt(torch.sum(pi_tilde[wedged_interactions] ** 2, dim=1))[:, None]
        pj_tilde[wedged_interactions] = pj_tilde[wedged_interactions] / torch.sqrt(torch.sum(pj_tilde[wedged_interactions] ** 2, dim=1))[:, None]

        with torch.no_grad():
            wall_mask = (torch.sum(pi * pj , dim = 2) <= 0.0)           #* (torch.sum(-dx * pj , dim = 2) < 0.0) #maybe comment in later

        exponent, gamma_i, gamma_j = self.get_gamma_exponent(dx, pi, pj, q_mean, idx, gamma)
        d_tilde = d * torch.exp(exponent)

        # All the S-terms are calculated
        S1 = torch.sum(torch.cross(pj_tilde, dx, dim=2) * torch.cross(pi_tilde, dx, dim=2), dim=2)      # Calculating S1 (The ABP-position part of S). Scalar for each particle-interaction. Meaning we get array of size (n, m) , m being the max number of nearest neighbors for a particle
        S2 = torch.sum(torch.cross(pi_tilde, qi, dim=2) * torch.cross(pj_tilde, qj, dim=2), dim=2)      # Calculating S2 (The ABP-PCP part of S).
        S3 = torch.sum(torch.cross(qi, dx, dim=2) * torch.cross(qj, dx, dim=2), dim=2)                  # Calculating S3 (The PCP-position part of S)

        S1 = self.rescale_s(S1)
        if self.nematic_pcp:
            S2 = torch.abs(S2)              # We take the absolute value of S2 as we only care about the strength of the interaction, not the direction. This is because we have already taken care of the directionality in the way we construct pi_tilde and pj_tilde
        S2 = self.rescale_s(S2)
        if self.nematic_pcp:
            S3 = torch.abs(S3)
        S3 = self.rescale_s(S3)

        if self.cell_wall_interaction != 0.0:
            with torch.no_grad():
                l[wall_mask] = torch.tensor([self.cell_wall_interaction, 0.0, 0.0, 0.0], device=self.device, dtype=self.dtype)

        # Calculating S
        S = l[:,:,0] + l[:,:,1] * S1 + l[:,:,2] * S2 + l[:,:,3] * S3

        Vij = z_mask.float() * S * (torch.exp(-d_tilde) - torch.exp(-d_tilde/5))        # Calculating the potential energy between particles masking out false interactions via voronoi_mask
        
        if self.screen_out_defects:
            # When  we have pcp defects we only want the cells to interact via the S0 and S1 term
            # We want to sum up the lambda contributions l2 and l3 and add them to l1 to keep the overall strength the same.
            assert not(self.nematic_pcp), "Defect screening only implemented for vectorial PCP"

            with torch.no_grad():
                defect_mask = (torch.sum(qi * qj, dim=2) < 0.7)

            if torch.any(defect_mask):
                # Calculate S1 for defects using non-rotated vectors
                S1_def = torch.sum(torch.cross(pj, dx, dim=2) * torch.cross(pi, dx, dim=2), dim=2)
                S1_def = self.rescale_s(S1_def)

                # Calculate the new S term for defects
                S_def = l[:,:,0] + (l[:,:,1] + l[:,:,2] + l[:,:,3]) * S1_def

                # Calculate the potential for defects using non-elongated distance 'd'
                Vij_def = z_mask.float() * S_def * (torch.exp(-d) - torch.exp(-d/5))

                # Update the Vij tensor only for the defect interactions
                Vij[defect_mask] = Vij_def[defect_mask]      # Recalculating the potential energy for the defect interactions masking out false interactions via voronoi_mask

        if torch.unique(p_mask).shape[0] > 1:        # If we have multiple cell types we need to add the repulsion for the purely repulsive interactions
            for repulsion_mask in repulsion_mask_lst:
                # find the masked interactions for which dists < eq_dist
                dist_mask = d < self.r0
                too_close_mask = repulsion_mask * dist_mask
                Vij[too_close_mask] = (torch.exp(-d[too_close_mask]) - torch.exp(-d[too_close_mask]/5)) - self.r0_val 

        if self.cell_wall_interaction == 0.0:
            with torch.no_grad():
                dist_mask = d < self.r0
                too_close_mask = (wall_mask * dist_mask) * z_mask
                not_close_mask = (wall_mask * (~dist_mask)) * z_mask
            Vij[too_close_mask] = (torch.exp(-d[too_close_mask]) - torch.exp(-d[too_close_mask]/5))# - self.r0_val
            Vij[not_close_mask] = self.r0_val #0.0
        
        Vij_sum = torch.sum(Vij)

        if self.gamma_diff_penalty:
            assert self.use_gamma_mean, "Gamma difference penalty only makes sense if we use the gamma mean for interactions"
            gamma_diff = (gamma_i - gamma_j)**2
            gamma_diff[~z_mask] = 0.0
            gamma_diff_sum = torch.sum(gamma_diff)
            Vij_sum += gamma_diff_sum * self.gamma_diff_penalty

        V = Vij_sum

        num_neighbors = torch.sum(z_mask, dim=1)           
        Vij_normed = Vij / num_neighbors[:, None]       
        Vij_normed[~z_mask] = 0.0
        Vi = torch.sum(Vij_normed, dim=1)

        return V , Vi

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

        # We calculate the initial x_mass_midpoint for stretching purposes.
        with torch.no_grad():
            self.x_mass_midpoint = torch.sum(x[:,0]) / x.shape[0]

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


        # Refresh potential neighbours at most once per timestep (KDTree), then reuse idx for both stages.
        self.refresh_potential_neighbours_once(x, self.k)
        idx_base = self.idx

        # 
        d1, dx1, idx1, z_mask1 = self.true_neighbours_from_idx(x, p, q, gamma, idx_base)
        V1, Vi = self.potential(x, p, q, p_mask,
                                alpha_par, alpha_perp, gamma,
                                d1, dx1, idx1, z_mask1)

        g1_x, g1_p, g1_q, g1_alpha_par, g1_alpha_perp, g1_gamma = torch.autograd.grad(
            V1, (x, p, q, alpha_par, alpha_perp, gamma), create_graph=False, retain_graph=False
        )

        # Shared noise for additive-noise predictor/corrector
        with torch.no_grad():
            xi_x = torch.empty_like(x).normal_()
            # Rotational noise for p/q: random axis + small angle (eta is angular std dev in radians)
            # we make axis_p and axis_q perpendicular to p and q respectively by taking a random vector and projecting out the parallel component, then normalizing. This ensures the noise is purely rotational and does not change the magnitude of p and q. 
            
            xi_p = torch.empty_like(p).normal_()
            xi_p = xi_p - torch.sum(xi_p * p, dim=1, keepdim=True) * p

            xi_q = torch.empty_like(q).normal_()
            xi_q = xi_q - torch.sum(xi_q * q, dim=1, keepdim=True) * q

            xi_alpha_par = torch.empty_like(alpha_par).normal_()
            xi_alpha_perp = torch.empty_like(alpha_perp).normal_()

            # Predictor state
            x_tilde = x.clone()
            p_tilde = p.clone()
            q_tilde = q.clone()
            alpha_par_tilde = alpha_par.clone()
            alpha_perp_tilde = alpha_perp.clone()
            gamma_tilde = gamma.clone()

            for i, eta in enumerate(self.eta_lst):
                if not self.update_cells_bools[i]:
                    continue
                mask = p_mask == i

                x_tilde[mask] += (-g1_x[mask] * self.dt) + (eta * xi_x[mask] * self.sqrt_dt)
                p_tilde[mask] += (-g1_p[mask] * self.dt) + (eta * xi_p[mask] * self.sqrt_dt)  # We add the rotational noise as an Euler step for the predictor, then apply the actual rotation after calculating the angle. This is to ensure the noise is properly scaled by eta and sqrt_dt, and to keep the code simpler.
                q_tilde[mask] += (-g1_q[mask] * self.dt) + (eta * xi_q[mask] * self.sqrt_dt)

                if self.alpha_par_bool_lst[i]:
                    alpha_par_tilde[mask] += (-g1_alpha_par[mask] * self.dt)    #+ (eta * xi_alpha_par[mask] * self.sqrt_dt)

                if self.alpha_perp_bool_lst[i]:
                    alpha_perp_tilde[mask] += (-g1_alpha_perp[mask] * self.dt)  #+ (eta * xi_alpha_perp[mask] * self.sqrt_dt)

                if self.gamma_bool_lst[i]:
                    gamma_tilde[mask] += self.gamma_update_speed * (-g1_gamma[mask] * self.dt)

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
        V2, _ = self.potential(x_tilde, p_tilde, q_tilde, p_mask,
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
                if not self.update_cells_bools[i]:
                    continue
                mask = p_mask == i

                x[mask] += (-0.5 * (g1_x[mask] + g2_x[mask]) * self.dt) + (eta * xi_x[mask] * self.sqrt_dt)
                p[mask] += (-0.5 * (g1_p[mask] + g2_p[mask]) * self.dt) + (eta * xi_p[mask] * self.sqrt_dt)  # We add the rotational noise as an Euler step for the corrector, then apply the actual rotation after calculating the angle. This is to ensure the noise is properly scaled by eta and sqrt_dt, and to keep the code simpler.
                q[mask] += (-0.5 * (g1_q[mask] + g2_q[mask]) * self.dt) + (eta * xi_q[mask] * self.sqrt_dt)

                if self.alpha_par_bool_lst[i]:
                    alpha_par[mask] += (-0.5 * (g1_alpha_par[mask] + g2_alpha_par[mask]) * self.dt) #+ (eta * xi_alpha_par[mask] * self.sqrt_dt)

                if self.alpha_perp_bool_lst[i]:
                    alpha_perp[mask] += (-0.5 * (g1_alpha_perp[mask] + g2_alpha_perp[mask]) * self.dt) #+ (eta * xi_alpha_perp[mask] * self.sqrt_dt)

                if self.gamma_bool_lst[i]:
                    gamma[mask] += self.gamma_update_speed * (-0.5 * (g1_gamma[mask] + g2_gamma[mask]) * self.dt) #+ (eta * xi_gamma[mask] * self.sqrt_dt)

            # Final constraints
            p, q, alpha_par, alpha_perp, gamma = self.apply_constraints(p, q, p_mask, alpha_par, alpha_perp, gamma)            
            
        return x, p, q, p_mask, alpha_par, alpha_perp, gamma, Vi  #Returning the goods.

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
            x, p, q, p_mask, alpha_par, alpha_perp, gamma, energy = self.time_step(x, p, q, p_mask, alpha_par, alpha_perp, gamma, tstep)        #Advancing the simulation one timestep
            
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
                
                yield xx, pp, qq, pp_mask, alpha_parpar, alpha_perpperp, gammagamma, energy                                  # Yielding the data baybeeee
    
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
                # gamma0 = gamma[idx]
                gamma0 = torch.ones_like(b0, device=self.device, dtype=self.dtype) #for now let's just do new cells have no elongation.

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
        data_tuple (Tuple): (p_mask_lst, x_lst, p_lst, q_lst, alpha_par_lst, alpha_perp_lst, gamma_lst, energy_lst)
        name (str): The name of the file (without extension).
        output_folder (str): The folder to save the file in.

    Returns:
        None, but saves the data
    """
    p_mask_lst, x_lst, p_lst, q_lst, alpha_par_lst, alpha_perp_lst, gamma_lst, energy_lst = data_tuple

    energy_lst_copy = energy_lst.copy()
    last_energy = np.zeros_like(p_mask_lst[-1])
    energy_lst_copy.append(last_energy)

    # Structure: query by variable name to get list across all timeframes
    data_dict = {
        'x': x_lst,           
        'p': p_lst,           
        'q': q_lst,           
        'p_mask': p_mask_lst, 
        'alpha_par': alpha_par_lst,
        'alpha_perp': alpha_perp_lst,
        'gamma': gamma_lst,
        'energy': energy_lst_copy
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

    # we make an initial energy

    # Save the simulation dictionary
    with open(output_folder + '/sim_dict.json', 'w') as f:
        sim_dict['dtype'] = str(sim_dict['dtype'])
        json.dump(sim_dict, f, indent = 2)

    # Save the initial simulation data
    save((p_mask_lst, x_lst, p_lst,  q_lst, alpha_par_lst, alpha_perp_lst, gamma_lst, energy_lst), name='data', output_folder=output_folder)

    # Print the notes if verbose
    if verbose:
        notes = sim_dict['notes']
        print('Starting simulation with notes:')
        print(notes)

    i = 0       # Initialize the iteration counter
    t1 = time() # We timing stuff

    # Saving data at intervals specified by yield_every:
    for xx, pp, qq, pp_mask, alpha_parpar, alpha_perpperp, gammagamma, energyenergy in itertools.islice(runner, yield_steps):
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
        
        if len(pp_mask) > sim_dict['max_cells']:
            break
        
        # Every 50 yield steps we dump the data
        if i % 50 == 0:
            save((p_mask_lst, x_lst, p_lst,  q_lst, alpha_par_lst, alpha_perp_lst, gamma_lst, energy_lst), name='data', output_folder=output_folder)
    
    if verbose:
        print(f'Simulation done, saved {i} datapoints')
        print('Took', time() - t1, 'seconds')

    save((p_mask_lst, x_lst, p_lst,  q_lst, alpha_par_lst, alpha_perp_lst, gamma_lst, energy_lst), name='data', output_folder=output_folder)  # Last iteration is saved