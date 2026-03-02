### Imports ###
from webbrowser import get

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

        # Model parameters
        self.k              = 12                            # Number of nearest neighbors to consider
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
        self.alpha_params   = sim_dict['alpha_params']      # Determines how cells wedge
        self.gamma_params   = sim_dict['gamma_params']      # Determines how cells elongate
        self.prolif_rate    = sim_dict['prolif_rate']       # Probability of cell proliferation for each cell
        self.prolif_delay   = sim_dict['prolif_delay']      # Delay before cell proliferation
        # self.nematic_pcp    = sim_dict['nematic_pcp']     # Whether planar cell polarity is nematic (True) or vectorial (False)

        # Boundary parameters
        self.bound_radius   = sim_dict['bound_radius']      # Radius for sphere boundary condition. Can be None or a float

        # Relaxation length parameters 
        self.r0             = 5*np.log(5)/(5-1)
        self.r0_val         = np.exp(-self.r0)-np.exp(-self.r0/5)
        # self.offsets        = torch.tensor(sim_dict['offsets'], device=self.device, dtype=self.dtype)
        self.interaction_dist = sim_dict['interaction_dist']      # Maximum distance for interactions. Should be larger than r0, but not too large to avoid memory issues
        self.cell_wall_interaction = sim_dict['cell_wall_interaction']  # 0 if only repulsion, otherwise up to 1 for attraction

        # stuff we need to initialize
        self.idx    = None                                  # Indices of nearest neighbors
        self.beta   = None                                  # Tensor used for cell division

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


        # Set random seed
        torch.manual_seed(self.random_seed)                 # For reproducibility


    def get_neighbors(self, x, k=100):
        """
        Finds the k nearest neighbors for each cell and applies a voronoi mask to exclude false neighbors.
        Parameters:
            x (torch.Tensor): The positions of the cells.
            k (int): The number of nearest neighbors to consider.
        Returns:
            d (torch.Tensor): The distances to the neighbors.
            dx (torch.Tensor): The normalized displacement vectors to the neighbors.
            idx (torch.Tensor): The indices of the neighbors.
            z_mask (torch.Tensor): A boolean mask indicating which neighbors are true neighbors based on a distance cutoff
        """

        # finding all potential neighbors via knn 
        all_dists = x[:, None] - x[None, :]
        d = torch.linalg.norm(all_dists, dim=2)
        d, idx = d.topk(k+1, dim=1, largest=False, sorted=True)
        d, idx = d[:, 1:], idx[:, 1:]
        
        full_neighbor_list = x[idx]                                                     # Get the full neighbor list
        dx = x[:, None, :] - full_neighbor_list                                         # Calculate pairwise distances
        
        # exclude cells too far away
        z_mask = d < self.interaction_dist

        # Shorten tensors to avoid unnecessary computations and memory issues
        sort_idx = torch.argsort(z_mask.int(), dim=1, descending=True)              # We sort the boolean voronoi mask in descending order, i.e 1,1,1,...,0,0
        z_mask = torch.gather(z_mask, 1, sort_idx)                                  # Reorder z_mask
        dx = torch.gather(dx, 1, sort_idx[:, :, None].expand(-1, -1, 3))            # Reorder dx
        idx = torch.gather(idx, 1, sort_idx)                                        # Reorder idx
        m = torch.max(torch.sum(z_mask, dim=1)) + 1                                 # Finding new maximum number of true neighbors
        self.true_neighbour_max = m                                                 # Saving it so we can use it again later
        z_mask = z_mask[:, :m]                                                      # Shorten z_mask
        dx = dx[:, :m]                                                              # Shorten dx
        idx = idx[:, :m]                                                            # Shorten idx

        d = torch.sqrt(torch.sum(dx**2, dim=2))                                     # Calculate w. new ordering
        dx = dx / d[:, :, None]                                                     # Normalize dx (also new ordering)

        return d, dx, idx, z_mask

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
        """
        #TODO: You are here and should continue from here.
        #Get the true neighbor distances, dx and more reordered and reduced

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

        # Expanding alpha_par, alpha_perp
        alpha_par_i = alpha_par[:, None].expand(alpha_par.shape[0], idx.shape[1])
        alpha_par_j = alpha_par[idx]
        alpha_par_mean = (alpha_par_i + alpha_par_j)/2                  # Minimum wedging determines interaction
        alpha_par_mean = torch.tan(alpha_par_mean/2)

        alpha_perp_i = alpha_perp[:, None].expand(alpha_perp.shape[0], idx.shape[1])
        alpha_perp_j = alpha_perp[idx]
        alpha_perp_mean = (alpha_perp_i + alpha_perp_j)/2               # Minimum wedging determines interaction
        alpha_perp_mean = torch.tan(alpha_perp_mean/2)

        # Implementing cell wedging
        # with torch.no_grad():
        perp_dir = torch.cross(qi, pi, dim=2)

        Z_par = alpha_par_mean[:,:,None] * (qi * dx).sum(dim=2)[:,:,None] * qi                                          #* dx
        Z_perp = alpha_perp_mean[:,:,None] * (perp_dir * dx).sum(dim=2)[:,:,None] * perp_dir                            #* dx
        Z = Z_par + Z_perp

        pi_tilde = pi - Z
        pj_tilde = pj + Z

        # Normalizing the ABPs
        # wedged_interactions = torch.any((alpha_par_mean > 1e-5) | (alpha_perp_mean > 1e-5), dim=2)     # We only normalize the ABPs for wedged interactions, otherwise we mess with the non-wedged interactions for no reason
        wedged_interactions = torch.logical_or(alpha_par_mean > 1e-5, alpha_perp_mean > 1e-5)     # We only normalize the ABPs for wedged interactions, otherwise we mess with the non-wedged interactions for no reason
        pi_tilde[wedged_interactions] = pi_tilde[wedged_interactions] / torch.sqrt(torch.sum(pi_tilde[wedged_interactions] ** 2, dim=1))[:, None]
        pj_tilde[wedged_interactions] = pj_tilde[wedged_interactions] / torch.sqrt(torch.sum(pj_tilde[wedged_interactions] ** 2, dim=1))[:, None]


        # Expanding gamma
        gamma_i = gamma[:, None].expand(gamma.shape[0], idx.shape[1])
        gamma_j = gamma[idx]
        log_gamma_mean = (gamma_i + gamma_j)/2
        cos2theta = 2 * ((qi * dx).sum(dim=2))**2 - 1
        d_tilde = d * torch.exp(log_gamma_mean * cos2theta)

        # All the S-terms are calculated
        S1 = torch.sum(torch.cross(pj_tilde, dx, dim=2) * torch.cross(pi_tilde, dx, dim=2), dim=2)      # Calculating S1 (The ABP-position part of S). Scalar for each particle-interaction. Meaning we get array of size (n, m) , m being the max number of nearest neighbors for a particle
        S2 = torch.sum(torch.cross(pi_tilde, qi, dim=2) * torch.cross(pj_tilde, qj, dim=2), dim=2)      # Calculating S2 (The ABP-PCP part of S).
        S3 = torch.sum(torch.cross(qi, dx, dim=2) * torch.cross(qj, dx, dim=2), dim=2)                  # Calculating S3 (The PCP-position part of S)

        S1 = self.rescale_s(S1)
        S2 = self.rescale_s(S2)
        S3 = self.rescale_s(S3)

        if self.cell_wall_interaction != 0.0:
            with torch.no_grad():
                wall_mask = (torch.sum(pi * pj , dim = 2) <= 0.0)        #* (torch.sum(-dx * pj , dim = 2) < 0.0) #maybe comment in later
                l[wall_mask] = torch.tensor([self.cell_wall_interaction, 0.0, 0.0, 0.0], device=self.device, dtype=self.dtype)

        # Calculating S
        S = l[:,:,0] + l[:,:,1] * S1 + l[:,:,2] * S2 + l[:,:,3] * S3

        Vij = z_mask.float() * S * (torch.exp(-d_tilde) - torch.exp(-d_tilde/5))        # Calculating the potential energy between particles masking out false interactions via voronoi_mask
        
        if torch.unique(p_mask).shape[0] > 1:        # If we have multiple cell types we need to add the repulsion for the purely repulsive interactions
            for repulsion_mask in repulsion_mask_lst:
                # find the masked interactions for which dists < eq_dist
                dist_mask = d < self.r0
                too_close_mask = repulsion_mask * dist_mask
                Vij[too_close_mask] = (torch.exp(-d[too_close_mask]) - torch.exp(-d[too_close_mask]/5)) - self.r0_val 

        if self.cell_wall_interaction == 0.0:
            wall_mask = (torch.sum(pi * pj , dim = 2) <= 0.0)        #* (torch.sum(-dx * pj , dim = 2) < 0.0) #maybe comment in later
            dist_mask = d < self.r0
            too_close_mask = wall_mask * dist_mask
            Vij[too_close_mask] = (torch.exp(-d[too_close_mask]) - torch.exp(-d[too_close_mask]/5)) - self.r0_val
        
        Vij_sum = torch.sum(Vij)                                                        # Summing the potential energy contributions

        # Utilize spherical boundary conditions?
        if self.bound_radius:               # If self.bound_radius is set, we apply spherical boundary conditions
            bc = self.sphere_bound(x)
        else:
            bc = 0.

        V = Vij_sum + bc

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
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The initialized tensors for cell positions, apicobasal polarities, planar cell polarities, and the particle mask.
            OR
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]: If p_mask is None, the last element is None.

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

        return x, p, q, p_mask, alpha_par, alpha_perp, gamma # Returning the goods.
    
    def update_k(self, true_neighbour_max):
        """
        Updates the number of k nearest neighbors found in find_potential_neighbours()
        based on the maximum number of true neighbors found in the last iteration.
        Parameters:

            true_neighbour_max (int): The maximum number of true neighbors found.

        Returns:
            k (int): The updated number of k nearest neighbors.

        """

        k = self.k
        fraction = true_neighbour_max / k       # Fraction between the maximimal number of nearest neighbors and the initial nunber of nearest neighbors we look for.
        if fraction < 0.25:                     # If fraction is small our k is too large and we make k smaller
            k = int(0.75 * k)
        elif fraction > 0.75:                   # Vice versa
            k = int(1.5 * k)
        self.k = k                              # We update k
        return k
    
    def update_neighbors_bool(self, tstep, division):
        """
        Returns whether to update potential neighbors or not

        Parameters:
            tstep (int): The current time step.
            division (bool): Whether cell division has occured in this timestep

        Returns:
            bool: Whether to update potential neighbors or not.

        """

        if division == True or tstep < 5_000:       # If cell division has occurred or we are in the early stages
            return True
        elif self.idx is None:                      # If we have not found any neighbors yet
            return True
        return (tstep % 20 == 0)                    # Otherwise we update every 20th step

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
        division, x, p, q, p_mask, self.beta, alpha_par, alpha_perp, gamma = self.cell_division(x, p, q, p_mask, alpha_par, alpha_perp, gamma)

        # k = self.update_k(self.true_neighbour_max)      # Update k based on last iteration
        # k = min(k, len(x) - 1)                          # No reason letting k be larger than number of cells
        d, dx, idx, z_mask = self.get_neighbors(x, k=self.k)
  
        # Calculate potential
        V, Vi = self.potential(x, p, q, p_mask,
                            alpha_par, alpha_perp, gamma,
                            d, dx, idx, z_mask)

        # Backpropagation
        V.backward()

        with torch.no_grad():
            for eta in self.eta_lst:
                # Cell positions and polarities are updated according to overdamped langevin dynamics. 
                x += -x.grad * self.dt + eta * torch.empty(*x.shape, dtype=self.dtype, device=self.device).normal_() * self.sqrt_dt
                p += -p.grad * self.dt + eta * torch.empty(*p.shape, dtype=self.dtype, device=self.device).normal_() * self.sqrt_dt
                q += -q.grad * self.dt + eta * torch.empty(*q.shape, dtype=self.dtype, device=self.device).normal_() * self.sqrt_dt

            # Updating alpha and gamma if they are free
            for i,val in enumerate(self.alpha_par_bool_lst):
                if val:
                    alpha_par[p_mask == i] += -alpha_par[p_mask == i].grad * self.dt + self.eta * torch.empty(*alpha_par[p_mask == i].shape, dtype=self.dtype, device=self.device).normal_() * self.sqrt_dt
                    alpha_par[p_mask == i] = torch.clamp(alpha_par[p_mask == i], self.alpha_range[0], self.alpha_range[1])

            for i,val in enumerate(self.alpha_perp_bool_lst):
                if val:
                    alpha_perp[p_mask == i] += -alpha_perp[p_mask == i].grad * self.dt + self.eta * torch.empty(*alpha_perp[p_mask == i].shape, dtype=self.dtype, device=self.device).normal_() * self.sqrt_dt
                    alpha_perp[p_mask == i] = torch.clamp(alpha_perp[p_mask == i], self.alpha_range[0], self.alpha_range[1])

            for i,val in enumerate(self.gamma_bool_lst):
                if val:
                    gamma[p_mask == i] += -gamma[p_mask == i].grad * self.dt + self.eta * torch.empty(*gamma[p_mask == i].shape, dtype=self.dtype, device=self.device).normal_() * self.sqrt_dt
                    gamma[p_mask == i] = torch.clamp(gamma[p_mask == i], self.gamma_range[0], self.gamma_range[1])


        # We zero out the gradients for next time step
        if any(self.alpha_par_bool_lst):
            alpha_par.grad.zero_()
        if any(self.alpha_perp_bool_lst):
            alpha_perp.grad.zero_()
        if any(self.gamma_bool_lst):
            gamma.grad.zero_()

        p.grad.zero_()
        q.grad.zero_()
        x.grad.zero_()

        #normalize p and q after altering them in the update timestep
        with torch.no_grad():
            p /= torch.sqrt(torch.sum(p ** 2, dim=1))[:, None]          # Normalizing p. Only the non-zero polarities are considered.
            q /= torch.sqrt(torch.sum(q ** 2, dim=1))[:, None]          # Normalizing q. Only the non-zero polarities are considered.

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
            tstep += 1
            x, p, q, p_mask, alpha_par, alpha_perp, gamma, energy = self.time_step(x, p, q, p_mask, alpha_par, alpha_perp, gamma, tstep)        #Advancing the simulation one timestep

            if tstep % self.yield_every == 0 or len(x) > self.max_cells:    #Yield data if we are at a 'yield step' or if we have too many cells and the simulation is aborted
                
                xx = x.detach().to("cpu").numpy().copy()                    #Copying data to CPU
                pp = p.detach().to("cpu").numpy().copy()
                qq = q.detach().to("cpu").numpy().copy()
                alpha_parpar = alpha_par.detach().clone() * 180.0/np.pi
                alpha_parpar = alpha_parpar.to("cpu").numpy()
                alpha_perpperp = alpha_perp.detach().clone() * 180.0/np.pi
                alpha_perpperp = alpha_perpperp.to("cpu").numpy()
                gammagamma = gamma.detach().to("cpu").numpy().copy()
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
    

# def save(data_tuple, name, output_folder):
#     """
#     Saves the simulation data to an .npy file.

#     Parameters:
#         data_tuple (Tuple): (p_mask, x, p, q)
#         name (str): The name of the file (without extension).
#         output_folder (str): The folder to save the file in.

#     Returns:
#         None, but saves the data
#     """

#     with open(f'{output_folder}/{name}.npy', 'wb') as f:
#         pickle.dump(data_tuple, f)

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
        x = data['x'][0]
        p = data['p'][0]
        q = data['q'][0]
        alpha_par = data['alpha_par']
        alpha_perp = data['alpha_perp']
        gamma = data['gamma']

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
    alpha_par_lst = [alpha_par]
    alpha_perp_lst = [alpha_perp]
    gamma_lst = [gamma]
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

def make_random_sphere(N, type0_frac , radius=30, alpha_params=None, gamma_params=None):
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

    # Generate random positions within a sphere
    x = np.random.randn(N, 3)
    r = radius * np.random.rand(N)**(1/3.)
    x /= np.sqrt(np.sum(x**2, axis=1))[:, None]
    x *= r[:, None]

    # Generate random polarities
    p = np.random.randn(N, 3)
    p /= np.sqrt(np.sum(p**2, axis=1))[:,None]
    q = np.random.randn(N, 3)
    q /= np.sqrt(np.sum(q**2, axis=1))[:,None]

    # Generate random cell types with specified fractions
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