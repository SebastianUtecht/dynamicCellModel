### Imports ###
import numpy as np
import torch
import os
import itertools
import pickle
from time import time
import json

# We might need these again later
# from scipy.spatial import cKDTree
# import gc
# import networkx as nx

### lambdas structure ###
# 0: stroma - stroma
# 1: stroma - allNPC 
# 2: preNPC - preNPC
# 3: preNPC - restNPC
# 4: isoNPC - isoNPC
# 5: anoNPC - anoNPC
# 6: anoNPC - isoNPC
# 7: vesNPC - isoNPC+anoNPC   (not used currently)
########################

class Simulation:
    def __init__(self, sim_dict):
        # General stuff
        self.device         = sim_dict['device']
        self.dtype          = sim_dict['dtype']
        self.random_seed    = sim_dict['random_seed']
        self.yield_every    = sim_dict['yield_every']
        self.min_batch_size = 128                       # Absolute minimum batch size for true neighbor search

        # Init parameters
        self.k              = sim_dict['init_k'] 
        self.true_neighbour_max     = sim_dict['init_k']//2
        self.dt             = sim_dict['dt']
        self.sqrt_dt        = np.sqrt(self.dt)
        self.max_cells      = sim_dict['max_cells']
    
        # Main model parameters
        self.nonpolar_NPC_eta = sim_dict['nonpolar_NPC_eta']
        self.NPC_eta        = sim_dict['NPC_eta']    
        self.stroma_eta     = sim_dict['stroma_eta']
        self.phi_eta          = sim_dict['phi_eta']
        self.lambdas        = torch.tensor(sim_dict['lambdas'], device=self.device, dtype=self.dtype)
        self.pre_lambdas    = sim_dict['pre_lambdas']
        self.offsets        = torch.tensor(sim_dict['offsets'], device=self.device, dtype=self.dtype)
        self.abs_s2s3       = sim_dict['abs_s2s3']

        # Wedging parameters
        self.vesicle_alpha  = sim_dict['vesicle_alpha']
        self.tube_alpha     = sim_dict['tube_alpha']

        # Proliferation parameters
        self.prolif_rates   = torch.tensor(sim_dict['prolif_rates'], device=self.device, dtype=self.dtype) # entries: [stroma, preNPC, vesNPC, isoNPC, anoNPC]
        self.prolif_start   = sim_dict['prolif_start']  # when to start proliferation: 'vesicle' or 'tube'
        assert self.prolif_start in ['vesicle', 'tube', 'none'], "prolif_start must be 'vesicle', 'tube' or 'none'"

        # Phase parameters
        self.pre_polar_dur  = sim_dict['pre_polar_dur']
        self.min_ves_time   = sim_dict['min_ves_time']
        self.gamma          = sim_dict['gamma']
        self.polar_initialization = sim_dict['polar_initialization']
        self.seethru        = sim_dict['seethru']

        # Boundary parameters
        self.bound_str      = sim_dict['bound_str']
        self.grav_str       = sim_dict['grav_str']
        self.bound_height   = sim_dict['bound_height']
        self.bound_radius   = sim_dict.get('bound_radius', 1e6)  # Default to a very large radius if not specified

        # cell fate parameters
        self.wnt_ago_center = torch.tensor(sim_dict['wnt_ago_center'], device=self.device, dtype=self.dtype)
        self.wnt_ago_ring   = sim_dict['wnt_ago_ring']

        # Flip parameters
        self.flip_time      = sim_dict['flip_time']
        rest = self.pre_polar_dur % self.flip_time
        if rest != 0:
            self.pre_polar_dur -= rest
            # Explanation: At the moment the simulation crashes if self.pre_polar_dur % self.flip_time != 0
            # This is a hacky fix, we should implement a better solution later.
        self.flip_radius    = sim_dict['flip_radius']

        # Basal lamina parameters
        self.tube_wall_str  = sim_dict['tube_wall_str']
        self.bouncy_wall  = sim_dict['bouncy_wall']
        self.avg_q          = False

        # WNT-PCP parameters
        self.WNT_str        = sim_dict['WNT_str']
        self.WNT_c          = sim_dict['WNT_c']
        self.wnt_ago_diff   = sim_dict['wnt_ago_diff']

        # State variables
        self.r0             = 5*np.log(5)/(5-1)
        self.r0_val         = np.exp(-self.r0)-np.exp(-self.r0/5)

        self.warming_up = False
        self.pre_polar  = False
        self.vesicle_formation  = False
        self.tube_formation     = False
        self.proliffing_cells   = False
        self.vesicle_fin        = False
        self.tube_fin           = False
        self.d = None
        self.idx = None
        self.beta = None

        # self.ves_idx    = []
        # self.ves
        self.ves_timing = {}

        torch.manual_seed(self.random_seed)

    
    @staticmethod
    def find_potential_neighbours(x, k=100):
        dx = x[:, None] - x[None, :]                            # Pairwise differences
        d = torch.linalg.norm(dx, dim=2)                        # Pairwise distances
        d, idx = d.topk(k+1, dim=1, largest=False, sorted=True)   # Get the k+1 smallest distances (including self-distance)
        return d[:, 1:], idx[:, 1:]                             # Return distances and indices of neighbors. Note that we do not return the self-neighbor
    
### OLD METHODS WE MAY HAVE TO REINTRODUCE LATER ###

    # @staticmethod
    # def find_potential_neighbours(x, k=100, distance_upper_bound=np.inf, workers=-1):

    #     tree = cKDTree(x)
    #     d, idx = tree.query(x, k + 1, distance_upper_bound=distance_upper_bound, workers=workers)
    #     return d[:, 1:], idx[:, 1:]

### END OF OLD METHODS ###

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

    def find_true_neighbours(self, d, dx, test_batches=True):
        with torch.no_grad():
            total_cells = dx.shape[0]
            neighbor_count = dx.shape[1]
            result_tensor = torch.empty(
                (total_cells, neighbor_count),
                dtype=torch.bool,
                device=self.device
            )

            
            if self.tstep == 0:
                if test_batches and not hasattr(self, 'max_safe_batch'):
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
                    n_dis = torch.sum((dx[i0:i1, :, None, :] / 2 - dx[i0:i1, None, :, :]) ** 2, dim=3)
                    n_dis += 1000 * torch.eye(n_dis.shape[1], device=self.device, dtype=self.dtype)[None, :, :]
                    result_tensor[i0:i1] = (torch.sum(n_dis < (d[i0:i1, :, None] ** 2 / 4), dim=2) <= self.seethru)
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
    
    @staticmethod
    def gauss_grad(d, dx, interaction_mask):
        with torch.no_grad():
            gauss   = torch.exp(-(d ** 2) * 0.04)
            gauss  *= (interaction_mask == 2).type(torch.int)
            grad    = torch.sum((gauss[:, :, None] * dx * d[:,:,None]), dim=1)
        return grad
    
    def rescale_s(self, S):
        S_rescaled = (S + 1.0) / 2.0
        return S_rescaled
    
    def dish_bound(self, pos):
        z = pos[:,2]
        v_add = torch.where(z < 0, self.bound_str * z**2, 0.)
        return v_add.sum()
    
    def cylinder_bound(self, pos):
        x = pos[:,0]
        y = pos[:,1]
        z = pos[:,2]
        r = torch.sqrt(x**2 + y**2)
        v_add = torch.where(r > self.bound_radius, self.bound_str * (r - self.bound_radius)**2, 0.)
        v_add += torch.where(torch.abs(z) > self.bound_height/2, self.bound_str * (torch.abs(z) - self.bound_height/2)**2, 0.)
        return v_add.sum()
    
    def gravity(self, pos):
        z = pos[:,2]
        v_add = self.grav_str * z
        return v_add.sum()
    
    def wnt_ago_conc(self, pos=None, distances=None):
        #Assert that only pos or distances is given
        assert (distances is None) != (pos is None), "Either pos or distances must be provided, not both."

        if distances is not None:
            conc = torch.exp(- (distances ** 2) / (2 * self.wnt_ago_diff ** 2))
            return conc
        
        center      = torch.tensor(self.wnt_ago_center, device=self.device, dtype=self.dtype)
        distances   = torch.sqrt(torch.sum((pos - center) ** 2, dim=1))
        diff        = self.wnt_ago_diff
        conc        = torch.exp(- (distances ** 2) / (2 * diff ** 2))
        return conc
    
    # Potential that aligns the planar cell polarity of cells near the source of the WNT perpendicular to the WNT gradient.
    def WNT_grad(self, x, dx, idx, z_mask, tube_idx):
        with torch.no_grad():
            tube_x, tube_dx, tube_idx_idx = x[tube_idx], dx[tube_idx], idx[tube_idx]
            dx = x[tube_idx], dx[tube_idx], idx[tube_idx]
            tube_neigh_pos  = x[tube_idx_idx]
            tube_z          = z_mask[tube_idx]

            WNT_x_dists     = torch.sqrt(torch.sum((self.wnt_ago_center - tube_x)**2, dim=1))
            WNT_neigh_dists = torch.sqrt(torch.sum((self.wnt_ago_center[None,None].expand(tube_neigh_pos.shape) - tube_neigh_pos)**2, dim=2))
            WNT_x           = self.wnt_ago_conc(distances=WNT_x_dists)
            WNT_neigh       = self.wnt_ago_conc(distances=WNT_neigh_dists)

            WNT_grad_  = (WNT_x[:,None] - WNT_neigh) * tube_z
            
            tot_WNT_grad = torch.sum((WNT_grad_)[:,:,None].expand(WNT_neigh.shape[0], WNT_neigh.shape[1],3) * tube_dx, dim=1)
            tot_WNT_grad[torch.sum(tot_WNT_grad, dim=1) != 0.0] /= torch.sqrt(torch.sum(tot_WNT_grad[torch.sum(tot_WNT_grad, dim=1)  != 0.0] ** 2, dim=1))[:, None]
            tot_WNT_grad = torch.nan_to_num(tot_WNT_grad, nan=0.0, posinf=0.0, neginf=0.0)

        return tot_WNT_grad, WNT_x_dists

    def potential(self, x, p, q, p_mask, idx, phi):

        # Find neighbours
        full_n_list = x[idx]
        dx = x[:, None, :] - full_n_list
        d = torch.sqrt(torch.sum(dx**2, dim=2))
        z_mask = self.find_true_neighbours(d, dx)

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

        #Making interaction masks
        ## STUFF WE MAYBE NEED ###
        if not(self.pre_polar):
            p_mask_true = p_mask.clone()
            p_mask = p_mask.clone()
            p_mask[p_mask == 2] = 3 #treat ves as isoNPC   

        interaction_mask = torch.cat((p_mask[:,None].expand(p_mask.shape[0], idx.shape[1])[:,:,None], p_mask[idx][:,:,None]), dim=2)

        #stroma-stroma mask
        stroma_stroma_mask = torch.sum(interaction_mask == torch.tensor([0,0], device=self.device), dim=2) == 2
        #find all stroma - NPC interactions
        stroma_NPC_mask = torch.any(interaction_mask == 0, dim=2)
        #filter out stroma-stroma
        stroma_NPC_mask = torch.logical_and(stroma_NPC_mask, ~stroma_stroma_mask)

        #preNPC-preNPC mask
        preNPC_preNPC_mask = torch.sum(interaction_mask == torch.tensor([1,1], device=self.device), dim=2) == 2

        #preNPC-restNPC mask
        preNPC_anoNPC_mask = torch.sum(interaction_mask == torch.tensor([1,4], device=self.device), dim=2) == 2
        preNPC_anoNPC_mask = torch.logical_or( torch.sum(interaction_mask == torch.tensor([4,1], device=self.device), dim=2) == 2, preNPC_anoNPC_mask)
        preNPC_isoNPC_mask = torch.sum(interaction_mask == torch.tensor([1,3], device=self.device), dim=2) == 2
        preNPC_isoNPC_mask = torch.logical_or( torch.sum(interaction_mask == torch.tensor([3,1], device=self.device), dim=2) == 2, preNPC_isoNPC_mask)
        preNPC_restNPC_mask = torch.logical_or(preNPC_anoNPC_mask, preNPC_isoNPC_mask)

        #anoNPC-anoNPC mask
        anoNPC_anoNPC_mask = torch.sum(interaction_mask == torch.tensor([4,4], device=self.device), dim=2) == 2

        #anoNPC-isoNPC mask
        anoNPC_isoNPC_mask = torch.sum(interaction_mask == torch.tensor([4,3], device=self.device), dim=2) == 2
        anoNPC_isoNPC_mask = torch.logical_or( torch.sum(interaction_mask == torch.tensor([3,4], device=self.device), dim=2) == 2, anoNPC_isoNPC_mask)

        isoNPC_isoNPC_mask = torch.sum(interaction_mask == torch.tensor([3,3], device=self.device), dim=2) == 2

        #vesNPC-iso/anoNPC mask
        if not(self.pre_polar):
            true_interaction_mask = torch.cat((p_mask_true[:,None].expand(p_mask_true.shape[0], idx.shape[1])[:,:,None], p_mask_true[idx][:,:,None]), dim=2)
            vesNPC_iso_ano_mask = torch.sum(true_interaction_mask == torch.tensor([2,3], device=self.device), dim=2) == 2
            vesNPC_iso_ano_mask = torch.logical_or( torch.sum(true_interaction_mask == torch.tensor([3,2], device=self.device), dim=2) == 2, vesNPC_iso_ano_mask)
            vesNPC_iso_ano_mask = torch.logical_or( vesNPC_iso_ano_mask, torch.sum(true_interaction_mask == torch.tensor([2,4], device=self.device), dim=2) == 2)
            vesNPC_iso_ano_mask = torch.logical_or( vesNPC_iso_ano_mask, torch.sum(true_interaction_mask == torch.tensor([4,2], device=self.device), dim=2) == 2)


        polar_mask      = ~torch.any(interaction_mask == 0, dim=2)
        polar_mask      = torch.logical_or(polar_mask, ~torch.any(interaction_mask == 1, dim=2))
    
        wall_mask = None

        # Calculate S
        if self.pre_polar:
            stroma_stroma_l, stroma_preNPC_l, preNPC_preNPC_l = self.pre_lambdas

            lam = torch.zeros(size=(interaction_mask.shape[0], interaction_mask.shape[1]),
                    device=self.device)
            
            lam[stroma_stroma_mask] = torch.tensor(stroma_stroma_l, device=self.device, dtype=self.dtype) # Setting lambdas for pre non-polar interaction
            lam[preNPC_preNPC_mask] = torch.tensor(preNPC_preNPC_l, device=self.device, dtype=self.dtype) # Setting lambdas for pre polar-nonpolar interaction
            lam[stroma_NPC_mask] = torch.tensor(stroma_preNPC_l, device=self.device, dtype=self.dtype)    # Setting lambdas for pre pure polar interaction

            S = lam

        else:
            # Setting the lambdas
            lam = torch.zeros(size=(interaction_mask.shape[0], interaction_mask.shape[1], 4),
                            device=self.device, dtype=self.dtype)
            
            bouncy_mask_lst = []

            if torch.any(self.lambdas[0]):
                lam[stroma_stroma_mask] = self.lambdas[0]
            else:
                bouncy_mask_lst.append(stroma_stroma_mask)
            
            if torch.any(self.lambdas[1]):
                lam[stroma_NPC_mask] = self.lambdas[1]
            else:
                bouncy_mask_lst.append(stroma_NPC_mask)
            
            if torch.any(self.lambdas[2]):
                lam[preNPC_preNPC_mask] = self.lambdas[2]
            else:
                bouncy_mask_lst.append(preNPC_preNPC_mask)
            
            if torch.any(self.lambdas[3]):
                lam[preNPC_restNPC_mask] = self.lambdas[3]
            else:
                bouncy_mask_lst.append(preNPC_restNPC_mask)
            
            if torch.any(self.lambdas[4]):
                lam[isoNPC_isoNPC_mask] = self.lambdas[4]
            else:
                bouncy_mask_lst.append(isoNPC_isoNPC_mask)
            
            if torch.any(self.lambdas[5]):
                lam[anoNPC_anoNPC_mask] = self.lambdas[5]
            else:
                bouncy_mask_lst.append(anoNPC_anoNPC_mask)
            
            if torch.any(self.lambdas[6]):
                lam[anoNPC_isoNPC_mask] = self.lambdas[6]
            else:
                bouncy_mask_lst.append(anoNPC_isoNPC_mask)
            
            if torch.any(self.lambdas[7]):
                lam[vesNPC_iso_ano_mask] = self.lambdas[7]
            else:
                bouncy_mask_lst.append(vesNPC_iso_ano_mask)

            # Expanding ABP and PCP for easy cross products
            pi = p[:, None, :].expand(p.shape[0], idx.shape[1], 3)
            pj = p[idx]
            qi = q[:, None, :].expand(q.shape[0], idx.shape[1], 3)
            qj = q[idx]

            # Setting the alphas
            alphas = torch.zeros(size=(interaction_mask.shape[0], interaction_mask.shape[1]), device=self.device, dtype=self.dtype)
            alphas[isoNPC_isoNPC_mask]  = self.vesicle_alpha
            alphas[anoNPC_anoNPC_mask]  = self.tube_alpha
            alphas = alphas[:,:,None].expand(alphas.shape[0], alphas.shape[1], 3)
        
            # Calculating the anisotropic angle_dx
            ts = (qi * dx).sum(axis = 2)
            angled_dx = qi * ts[:,:,None]
            Z = phi * dx + (1-phi) * angled_dx


            
            ### OI DO THIS ###
            # if self.pcp_order_thresh:
            #     #First we find all anoNPC neighbors that are also anoNPCs
            #     ano_idx = torch.where(p_mask == 4)[0]
            #     ano_neighbors_idx = idx * anoNPC_anoNPC_mask
                
            #     #delete all zero of only zeros
            #     ano_neighbors_idx = ano_neighbors_idx[torch.any(ano_neighbors_idx, dim=1)]

            #     #sanity check
            #     assert ano_neighbors_idx.shape[0] == ano_idx.shape[0], "Shapes do not match!, something is awry"

            #     iso_ano_idx = {}

            #     for i in range(len(ano_idx)):
            #         neighbor_idxs = ano_neighbors_idx[i]
            #         neighbor_idxs = neighbor_idxs[neighbor_idxs != 0]   #remove zeros
                    
            #         #If an anonpc has 4 or less neighbors its probably an 'end' cell and will constrict isotropically 
            #         if len(neighbor_idxs) <= 4:
            #             iso_ano_idx[ano_idx[i]] = neighbor_idxs
            #             continue
                    
            #         all_idx = torch.cat((neighbor_idxs, ano_idx))

            #         Qi = 1/len(all_idx) * torch.sum(torch.outer(q[all_idx], q[all_idx]) - 1/3 * torch.eye(3, device=self.device, dtype=self.dtype), dim=0)
            #         eigvals, _ = torch.linalg.eig(Qi)
            #         max_eigval_idx = torch.argmax(eigvals.real)
            #         pcp_order = eigvals.real[max_eigval_idx]

            #         if pcp_order < self.pcp_order_thresh:   #threshold for nematic order
            #             iso_ano_idx[ano_idx[i]] = neighbor_idxs
                    
            #     #Now we set the angle_dx for those cells to be isotropic
            #     #But we need to make sure the interactions are symmetric
            #     iso_ano_ano_mask = torch.zeros_like(anoNPC_anoNPC_mask, dtype=torch.bool, device=self.device)

            #     for key in iso_ano_idx.keys():
            #         for neighbor in iso_ano_idx[key]:
            #             iso_ano_ano_mask[key, neighbor] = True
            #             iso_ano_ano_mask[neighbor, key] = True

            #     angle_dx[iso_ano_ano_mask] = dx[iso_ano_ano_mask]                   

            # Calculating the isotropic angle
            Z[~anoNPC_anoNPC_mask] = dx[~anoNPC_anoNPC_mask]

            # Addition if we use seethru
            if self.seethru != 0:
                raise NotImplementedError("Seethru interaction not meant to be used with wedging at the moment.")

            # Permute the ABPs such that we get wedging
            pi_tilde = pi - alphas * Z
            pj_tilde = pj + alphas * Z

            # The permuted ABPs are normalized
            wedged_interactions = torch.logical_or(isoNPC_isoNPC_mask, anoNPC_anoNPC_mask)
            pi_tilde[wedged_interactions] = pi_tilde[wedged_interactions]/torch.sqrt(torch.sum(pi_tilde[wedged_interactions] ** 2, dim=1))[:, None] 
            pj_tilde[wedged_interactions] = pj_tilde[wedged_interactions]/torch.sqrt(torch.sum(pj_tilde[wedged_interactions] ** 2, dim=1))[:, None]

            # All the S-terms are calculated
            S1 = torch.sum(torch.cross(pj_tilde, dx, dim=2) * torch.cross(pi_tilde, dx, dim=2), dim=2)            # Calculating S1 (The ABP-position part of S). Scalar for each particle-interaction. Meaning we get array of size (n, m) , m being the max number of nearest neighbors for a particle
            S2 = torch.sum(torch.cross(pi_tilde, qi, dim=2) * torch.cross(pj_tilde, qj, dim=2), dim=2)            # Calculating S2 (The ABP-PCP part of S).
            S3 = torch.sum(torch.cross(qi, dx, dim=2) * torch.cross(qj, dx, dim=2), dim=2)                        # Calculating S3 (The PCP-position part of S)

            # Nematic PCP?
            if self.abs_s2s3:
                S2 = torch.abs(S2)
                S3 = torch.abs(S3)

            S1 = self.rescale_s(S1)
            S2 = self.rescale_s(S2) 
            S3 = self.rescale_s(S3)

        # Inducing non-polar interaction between cell walls
        if not(self.pre_polar):
            if (self.tube_wall_str != None) and not(self.bouncy_wall):
                with torch.no_grad():
                    wall_mask = (torch.sum(pi * pj , dim = 2) < 0.0) * (torch.sum(-dx * pj , dim = 2) < 0.0)
                    wall_mask = torch.logical_and(wall_mask , polar_mask)
                    lam[wall_mask] = torch.tensor([self.tube_wall_str, 0.0, 0.0, 0.0], device=self.device, dtype=self.dtype)
            # Calculating S
            S = lam[:,:,0] + lam[:,:,1] * S1 + lam[:,:,2] * S2 + lam[:,:,3] * S3

        if not torch.any(self.offsets):
            Vij = z_mask.float() * S * (torch.exp(-d) - torch.exp(-d/5))
        else:
            offset_tensor = torch.zeros_like(z_mask, dtype=self.dtype, device=self.device)

            offset_tensor[stroma_stroma_mask]   = self.offsets[0]
            offset_tensor[stroma_NPC_mask]      = self.offsets[1]
            offset_tensor[preNPC_preNPC_mask]   = self.offsets[2]
            offset_tensor[preNPC_restNPC_mask]  = self.offsets[3]
            offset_tensor[isoNPC_isoNPC_mask]   = self.offsets[4]
            offset_tensor[anoNPC_anoNPC_mask]   = self.offsets[5]
            offset_tensor[anoNPC_isoNPC_mask]   = self.offsets[6]
            offset_tensor[vesNPC_iso_ano_mask]  = self.offsets[7]

            Vij = z_mask.float() * S * (torch.exp(-(d - offset_tensor)) - torch.exp(-(d - offset_tensor)/5))
        
        if not(self.pre_polar):
            if (self.tube_wall_str != None) and (self.bouncy_wall):
                with torch.no_grad():
                    #cells that are on opposite sides of the basal lamina
                    wall_mask = (torch.sum(pi * pj , dim = 2) < 0.0) * (torch.sum(-dx * pj , dim = 2) < 0.0)
                    wall_mask = torch.logical_and(wall_mask , polar_mask)
                
                #finding the masked interactions for which dists < eq_dist
                dist_mask = d < self.r0
                too_close_mask = wall_mask * dist_mask
                not_close_enough_mask = wall_mask * (~dist_mask)
                Vij[not_close_enough_mask] = 0.0    
                Vij[too_close_mask] = (torch.exp(-d[too_close_mask]) - torch.exp(-d[too_close_mask]/5)) - self.r0_val
        
            #if bouncy_mask_lst not empty, let those interactions be bouncy
            if len(bouncy_mask_lst) > 0:
                for bouncy_mask in bouncy_mask_lst:
                    # find the masked interactions for which dists < eq_dist
                    dist_mask = d < self.r0
                    too_close_mask = bouncy_mask * dist_mask
                    if torch.any(self.offsets):
                        Vij[too_close_mask] = (torch.exp(-(d[too_close_mask] - offset_tensor[too_close_mask])) - torch.exp(-(d[too_close_mask] - offset_tensor[too_close_mask])/5)) - self.r0_val
                    else:
                        Vij[too_close_mask] = (torch.exp(-d[too_close_mask]) - torch.exp(-d[too_close_mask]/5)) - self.r0_val 

        Vij_sum = torch.sum(Vij)

        if  not(self.pre_polar):
            tube_mask = lam[:,-1] > 0.0
            tube_idx = torch.argwhere(tube_mask)[:,0]
            if self.WNT_str > 0.0 and len(tube_idx) > 0:
                WNT_grad_, WNT_x_dists   = self.WNT_grad(x=x, dx=dx, idx=idx, z_mask=z_mask, tube_idx=tube_idx)
                S4          = (1.0 - torch.sum(q[tube_idx] * WNT_grad_, dim=1)**2)
                cells_affected = WNT_x_dists < self.WNT_c
                
                Vij_sum    -= self.WNT_str * torch.sum(cells_affected * S4)
        
        # Gravity potential
        if self.grav_str != 0:
            Vij_sum += self.gravity(x)

        # Utilize spherical boundary conditions?
        if self.bound_str != 0:
            # bc = self.dish_bound(x)
            bc = self.cylinder_bound(x)
        else:
            bc = 0.
        # Direct ABPs away from center of mass?
        if not(self.pre_polar) and self.gamma:
            gauss_grad = self.gauss_grad(d, dx, interaction_mask)
            Vi     = torch.sum(self.gamma * p * gauss_grad, dim=1)
            return Vij_sum - torch.sum(Vi) + bc , int(m), z_mask, idx, wall_mask
        else:
            return Vij_sum + bc, int(m), z_mask, idx, wall_mask

    def init_simulation(self, x, p, q, p_mask):
        assert len(x) == len(p)
        assert len(q) == len(x)
        assert len(p_mask) == len(x)

        x = torch.tensor(x, requires_grad=True, dtype=self.dtype, device=self.device)
        p = torch.tensor(p, requires_grad=True, dtype=self.dtype, device=self.device)
        q = torch.tensor(q, requires_grad=True, dtype=self.dtype, device=self.device)
        phi = torch.zeros_like(p_mask, dtype=torch.int, device=self.device)
        p_mask = torch.tensor(p_mask, dtype=torch.int, device=self.device)
        self.beta   = torch.zeros_like(p_mask, dtype=self.dtype, device=self.device)

        return x, p, q, p_mask, phi
    
    def update_k(self, true_neighbour_max):
        k = self.k
        fraction = true_neighbour_max / k   # Fraction between the maximimal number of nearest neighbors and the initial nunber of nearest neighbors we look for.
        if fraction < 0.25:                 # If fraction is small our k is too large and we make k smaller
            k = int(0.75 * k)
        elif fraction > 0.75:               # Vice versa
            k = int(1.5 * k)
        self.k = k                          # We update k
        return k # let's try limiting and see what happens #CHANGE THIS BACK TO K
    
    def update_neighbors_bool(self, division):
        if ((division == True) or (self.tstep % self.flip_time == 0)) or self.idx is None:
            return True
        elif self.tstep < (self.pre_polar_dur):
            alt_tstep = self.tstep
        else:
            alt_tstep = self.tstep - self.pre_polar_dur

        n_update = 1 if alt_tstep < 1_000 else 20

        return (self.tstep % n_update == 0)

    def time_step(self, x, p, q, p_mask, phi):
        if self.tstep < (self.pre_polar_dur):
            self.pre_polar  = True
        elif self.tstep >= (self.pre_polar_dur) and not(self.vesicle_fin):
            self.vesicle_formation = True
            self.pre_polar = False
            self.seethru = 0  
            if torch.sum(p_mask == 1) == 0:
                self.vesicle_fin = True
                self.vesicle_formation = False
        elif self.vesicle_fin and not(self.tube_fin):
            self.tube_formation     = True
            if torch.sum(p_mask == 2) == 0:
                self.tube_formation  = False
                self.tube_fin = True

                if self.prolif_start == 'tube' and not self.proliffing_cells:
                    if torch.any(self.prolif_rates > 0):
                        self.beta[p_mask == 0] = self.prolif_rates[0]
                        self.beta[p_mask == 1] = self.prolif_rates[1]
                        self.beta[p_mask == 2] = self.prolif_rates[2]
                        self.beta[p_mask == 3] = self.prolif_rates[3]
                        self.beta[p_mask == 4] = self.prolif_rates[4]
                        self.proliffing_cells = True
        else:
            pass

        # Start with cell division
        division, x, p, q, p_mask, self.beta, phi = self.cell_division(x, p, q, p_mask, phi)
        
        # Idea: only update _potential_ neighbours every x steps late in simulation
        # For now we do this on CPU, so transfer will be expensive
        k = self.update_k(self.true_neighbour_max)
        k = min(k, len(x) - 1)

        if self.update_neighbors_bool(division):
            self.d, self.idx = self.find_potential_neighbours(x, k=k)
        idx = self.idx

        # Normalizing p and q
        if torch.any(p_mask == 2) or torch.any(p_mask == 3) or torch.any(p_mask == 4):
            with torch.no_grad():
                non_polar_mask = torch.logical_or((p_mask == 0), (p_mask == 1))

                p[~non_polar_mask] /= torch.sqrt(torch.sum(p[~non_polar_mask] ** 2, dim=1))[:, None]          # Normalizing p. Only the non-zero polarities are considered.
                q[~non_polar_mask] /= torch.sqrt(torch.sum(q[~non_polar_mask] ** 2, dim=1))[:, None]          # Normalizing q. Only the non-zero polarities are considered.

                p[non_polar_mask] = torch.tensor([0,0,0], device=self.device, dtype=self.dtype)
                q[non_polar_mask] = torch.tensor([0,0,0], device=self.device, dtype=self.dtype)

                # we want to restrict phi to be between 0 and 1
                phi[non_polar_mask] = torch.tensor(0, device=self.device, dtype=torch.int)
                phi[~non_polar_mask] = torch.clamp(phi[~non_polar_mask], 0, 1)

        # Calculate potential
        V, self.true_neighbour_max, z_mask, idx, wall_mask = self.potential(x, p, q, p_mask, idx, phi)

        # Backpropagation
        V.backward()

        # Time-step
        with torch.no_grad():
            stroma_mask = (p_mask == 0)
            preNPC_mask  = (p_mask == 1)
            polarNPC_mask = torch.logical_or((p_mask == 2), torch.logical_or((p_mask == 3), (p_mask == 4)))

            x[stroma_mask]      += -x.grad[stroma_mask] * self.dt + self.stroma_eta * torch.empty(*x[stroma_mask].shape, dtype=self.dtype, device=self.device).normal_() * self.sqrt_dt
            x[preNPC_mask]      += -x.grad[preNPC_mask] * self.dt + self.nonpolar_NPC_eta * torch.empty(*x[preNPC_mask].shape, dtype=self.dtype, device=self.device).normal_() * self.sqrt_dt
            x[polarNPC_mask]    += -x.grad[polarNPC_mask] * self.dt + self.NPC_eta * torch.empty(*x[polarNPC_mask].shape, dtype=self.dtype, device=self.device).normal_() * self.sqrt_dt

            x.grad.zero_()

            if not(self.pre_polar):
                p += -p.grad * self.dt + self.NPC_eta * torch.empty(*x.shape, dtype=self.dtype, device=self.device).normal_() * self.sqrt_dt
                q += -q.grad * self.dt + self.NPC_eta * torch.empty(*x.shape, dtype=self.dtype, device=self.device).normal_() * self.sqrt_dt
                phi += -phi.grad * self.dt + self.phi_eta * torch.empty(x.shape[0], dtype=self.dtype, device=self.device).normal_() * self.sqrt_dt

                p.grad.zero_()
                q.grad.zero_()
                phi.grad.zero_()
                self.wnt_ago_center = torch.mean(x[stroma_mask], dim=0)
        
        # Vesicle initiation
        if self.vesicle_formation:
            if self.tstep % self.flip_time == 0:
                # pick a random NPC to flip to vesicle generation
                npc_indices = torch.where((p_mask == 1))[0]
                tube_ves_indices = torch.where((p_mask == 2) | (p_mask == 3) | (p_mask == 4))[0]

                # narrow the npc_indices to only those that are not already close to a vesicle or tube forming cell
                # otherwise we get clusters of vesicle forming cells
                if len(tube_ves_indices) > 0:
                    npc_x = x[npc_indices]
                    tube_ves_x = x[tube_ves_indices]
                    distances_to_tube_ves = torch.sqrt(torch.sum((npc_x[:, None, :] - tube_ves_x[None, :, :]) ** 2, dim=2))
                    min_distances, _ = torch.min(distances_to_tube_ves, dim=1)
                    chose_from_npc_indices = npc_indices[min_distances >= (self.flip_radius + 1.0)]
                else:
                    chose_from_npc_indices = npc_indices

                if len(chose_from_npc_indices) == 0:
                    pass
                else:
                    chosen_npc = chose_from_npc_indices[torch.randint(0, len(chose_from_npc_indices), (1,)).item()]

                    #finding all NPCs within flip_radius of chosen_npc    
                    distances = torch.sqrt(torch.sum((x[npc_indices] - x[chosen_npc]) ** 2, dim=1))
                    close_npcs = npc_indices[distances < self.flip_radius]
                    #combining the chosen npc and the close npcs
                    flipped_npcs_idx = torch.cat((chosen_npc[None], close_npcs), dim=0)
                    p_mask[flipped_npcs_idx] = 2  # Set to vesicle forming
                    for npc in flipped_npcs_idx.tolist():
                        self.ves_timing[npc] = self.tstep

                    #polar initialization code (disabled for now and needs to be altered)
                    # At vesicle initialization we make the APB point out
                    if self.polar_initialization:
                        with torch.no_grad():
                            polar_center = torch.mean(x[flipped_npcs_idx], dim=0)
                            p[flipped_npcs_idx] = x[flipped_npcs_idx] - polar_center
                            p[flipped_npcs_idx] /= torch.sqrt(torch.sum(p[flipped_npcs_idx] ** 2, dim=1))[:, None]
                            # p[:,:] = torch.nan_to_num(p) # maybe comment back if infs arise
                    
                    # If any prolif-rates (tensor) are > 0, set them here       
                    if self.prolif_start == 'vesicle' and not self.proliffing_cells:
                        if torch.any(self.prolif_rates > 0):
                            self.beta[p_mask == 0] = self.prolif_rates[0]
                            self.beta[p_mask == 1] = self.prolif_rates[1]
                            self.beta[p_mask == 2] = self.prolif_rates[2]
                            self.beta[p_mask == 3] = self.prolif_rates[3]
                            self.beta[p_mask == 4] = self.prolif_rates[4]
                            self.proliffing_cells = True
        
        # Only do this every N timesteps
        # Only do if vesicle_formation == True
        if not self.pre_polar:
            # What we are trying to do here:
            # when cells in a vesicle forming state have been in that state for long enough, we want to
            # reclassify them as isotropic tube cells (p_mask = 3) or anisotropic tube cells (p_mask = 4)
            # based on the proximity to the wnt inhibitor (stroma cells) - Cells closer to the stroma become isotropic tube cells
            # while cells further away become anisotropic tube cells. 

            # finding the vesicle cells that have been in that state for long enough
            tube_idx = [idx for idx, t0 in self.ves_timing.items() if self.tstep - t0 >= self.min_ves_time]
            if len(tube_idx) > 0:
                tube_idx_tens = torch.tensor(tube_idx, device=self.device)
                tube_x = x[tube_idx_tens]

                #if the average distance from tube_x to the nearest stroma cell is
                #less than self.wnt_ago_ring, we classify as purely anisotropic tube cell
                aggregate_mean = torch.mean(tube_x, dim=0)
                stroma_x = x[p_mask == 0]
                distances_to_stroma = torch.sqrt(torch.sum((stroma_x - aggregate_mean) ** 2, dim=1))
                min_dist_to_stroma, _ = torch.min(distances_to_stroma, dim=0)

                if min_dist_to_stroma > self.wnt_ago_ring:
                    p_mask[tube_idx_tens] = 4  #anisotropic tube cell
                else:
                    #calculating distances to stroma center
                    # For just this one rune ### CHANGE BACK LATER ###
                    # Let's polarize in a random direction
                    random_dir = torch.empty(3, dtype=self.dtype, device=self.device).normal_()         #Delete later
                    random_dir /= torch.sqrt(torch.sum(random_dir ** 2))                                #Delete later
                    random_dir = torch.mean(tube_x, dim=0) + (random_dir * 10.0)       #Delete later   #this should be 

                    distances_to_stroma_center = torch.sqrt(torch.sum((tube_x - random_dir) ** 2, dim=1))      #this should be self.wnt_ago_center
                    median_distance = torch.median(distances_to_stroma_center)
                    close_to_stroma = distances_to_stroma_center < median_distance
                    p_mask[tube_idx_tens[close_to_stroma]] =  3  #isotropic tube cell (should be 3, but we are trying some stuff here)
                    p_mask[tube_idx_tens[~close_to_stroma]] = 4  #anisotropic tube cell
                    with torch.no_grad():
                        phi[tube_idx_tens[close_to_stroma]] = 0.0
                        phi[tube_idx_tens[~close_to_stroma]] = 1.0
                #removing reclassified cells from ves_timing dict
                for idx in tube_idx:
                    del self.ves_timing[idx]

        return x, p, q, p_mask, phi

    def simulation(self, x, p, q, p_mask):
        
        x, p, q, p_mask, phi = self.init_simulation(x, p, q, p_mask)

        self.tstep = 0
        while True:
            x, p, q, p_mask, phi  = self.time_step(x, p, q, p_mask, phi)
            self.tstep += 1

            if self.tstep % self.yield_every == 0 or len(x) > self.max_cells:
                xx = x.detach().to("cpu").numpy().copy()
                pp = p.detach().to("cpu").numpy().copy()
                qq = q.detach().to("cpu").numpy().copy()
                pp_mask = p_mask.detach().to("cpu").numpy().copy()
                yield xx, pp, qq, pp_mask
    
    def get_prolif_positions(self, p, q, p_mask, mask_ind):
        """
        Gives moves orthogonal to ABP of the dividing cells
        """
        with torch.no_grad():
            # We find all the polar particles
            p_polar = p[p_mask == mask_ind].clone().squeeze()
            q_polar = q[p_mask == mask_ind].clone().squeeze()

            # Special case if we only have 1 polar particle proliferating
            if torch.numel(p_polar) == 3:
                # Gram-Schmidt orthanormalization
                p_polar /= torch.sqrt(torch.sum(p_polar ** 2))
                q_polar -= torch.sum(q_polar * p_polar) * p_polar
                q_polar /= torch.sqrt(torch.sum(q_polar ** 2))

                # Matrix for linear transformation
                lin_trans = torch.zeros((3, 3), device=self.device, dtype=self.dtype)
                lin_trans[:,0] = p_polar
                lin_trans[:,1] = q_polar
                lin_trans[:,2] = torch.cross(p_polar,q_polar)

                # Find move in transformed space and transform back
                new_pos =  torch.zeros_like(p_polar, device=self.device).squeeze()
                new_pos[1:] = torch.normal(mean = 0.0, std = 1.0, size=(1,2)).squeeze()
                new_pos = (lin_trans @ new_pos).squeeze()
            else:
                # Gram-Schmidt orthanormalization
                p_polar /= torch.sqrt(torch.sum(p_polar ** 2, dim=1))[:, None]
                q_polar -= torch.sum(q_polar * p_polar, dim=1)[:,None] * p_polar
                q_polar /= torch.sqrt(torch.sum(q_polar ** 2, dim=1))[:, None]

                # Matrix for linear transformation
                lin_trans = torch.zeros((len(p_polar), 3, 3), device=self.device, dtype=self.dtype)
                lin_trans[:,:,0] = p_polar
                lin_trans[:,:,1] = q_polar
                lin_trans[:,:,2] = torch.cross(p_polar,q_polar, dim=1)

                # Find move in transformed space and transform back
                new_pos =  torch.zeros_like(p_polar)
                new_pos[:,1:] = torch.normal(mean = 0.0, std = 1.0, size=(p_polar.shape[0],2)).squeeze()
                new_pos = (lin_trans @ new_pos[:,:,None]).squeeze()

            return new_pos
    
    def cell_division(self, x, p, q, p_mask, m):   

        beta = self.beta
        dt = self.dt

        if torch.sum(beta) < 1e-8:
            return False, x, p, q, p_mask, beta, m

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

                # make a random vector and normalize to get a random direction
                move = torch.empty_like(x0).normal_()
                polar_pmask_vals = [2, 3, 4]
                for val in polar_pmask_vals:
                    move[p_mask0 == val] = self.get_prolif_positions(p0, q0, p_mask0, mask_ind=val)
                move /= torch.sqrt(torch.sum(move ** 2, dim=1))[:, None]
                
                # place new cells
                x0 = x0 + move

                # If proliferation is started at vesicle stage, we want the daughter cells
                # of vesicle forming cells to also be vesicle forming and have the same timing
                # as their parent cell.
                if torch.any(p_mask0 == 2):
                    for i, parent_idx in enumerate(idx):
                        if parent_idx.item() in self.ves_timing:
                            self.ves_timing[len(x) + i] = self.ves_timing[parent_idx.item()]

                # append new cell data to the system state
                x = torch.cat((x, x0))
                p = torch.cat((p, p0))
                q = torch.cat((q, q0))
                p_mask = torch.cat((p_mask, p_mask0))
                beta = torch.cat((beta, b0))

        x.requires_grad = True
        p.requires_grad = True
        q.requires_grad = True

        return division, x, p, q, p_mask, beta    

def save(data_tuple, name, output_folder):
    with open(f'{output_folder}/{name}.npy', 'wb') as f:
        pickle.dump(data_tuple, f)

def run_simulation(sim_dict):
    # Make the simulation runner object:
    data_tuple = sim_dict.pop('data')
    verbose    = sim_dict.pop('verbose')
    yield_steps, yield_every = sim_dict['yield_steps'], sim_dict['yield_every']

    assert len(data_tuple) == 4 or len(data_tuple) == 2, 'data must be tuple of either len 2 (for data generation) or 4 (for data input)'
    

    np.random.seed(sim_dict['random_seed'])
    if len(data_tuple) == 4:
        print('Using input data')
        p_mask, x, p, q = data_tuple
    else:
        data_gen = data_tuple[0]
        p_mask, x, p, q = data_gen(*data_tuple[1])
        
    sim = Simulation(sim_dict)
    runner = sim.simulation(x, p, q, p_mask)

    output_folder = sim_dict['output_folder']

    try: 
        os.mkdir(output_folder)
    except:
        pass

    x_lst = [x]
    p_lst = [p]
    q_lst = [q]
    p_mask_lst = [p_mask]

    with open(output_folder + '/sim_dict.json', 'w') as f:
        sim_dict['dtype'] = str(sim_dict['dtype'])
        json.dump(sim_dict, f, indent = 2)

    save((p_mask_lst, x_lst, p_lst,  q_lst), name='data', output_folder=output_folder)


    notes = sim_dict['notes'] if sim_dict['notes'] else sim_dict['output_folder']

    if verbose:
        print('Starting simulation with notes:')
        print(notes)

    i = 0
    t1 = time()

    for xx, pp, qq, pp_mask in itertools.islice(runner, yield_steps):
        i += 1
        if verbose:
            print(f'Running {i} of {yield_steps}   ({yield_every * i} of {yield_every * yield_steps})   ({len(xx)} cells)', end='\r')

        x_lst.append(xx)
        p_lst.append(pp)
        q_lst.append(qq)
        p_mask_lst.append(pp_mask)
        
        if len(xx) > sim_dict['max_cells']:
            break

        if i % 50 == 0:
            save((p_mask_lst, x_lst, p_lst,  q_lst), name='data', output_folder=output_folder)
    
    if verbose:
        print(f'Simulation done, saved {i} datapoints')
        print('Took', time() - t1, 'seconds')

    save((p_mask_lst, x_lst, p_lst,  q_lst), name='data', output_folder=output_folder)

def make_random_sphere(N, non_polar_frac , radius=35, center=np.array([0,0,0])):
    x = np.random.randn(N, 3)
    r = radius * np.random.rand(N)**(1/3.)
    x /= np.sqrt(np.sum(x**2, axis=1))[:, None]
    x *= r[:, None]
    x += center

    p = np.random.randn(N, 3)
    p /= np.sqrt(np.sum(p**2, axis=1))[:,None]
    q = np.random.randn(N, 3)
    q /= np.sqrt(np.sum(p**2, axis=1))[:,None]

    mask = np.random.choice([0,1], p=[non_polar_frac, 1-non_polar_frac], size=N)        #Mask detailing which particles are non polar
    p[mask == 0] = np.array([0,0,0])                                                    #Setting the polarities of the non-polarized particles to 0
    q[mask == 0] = np.array([0,0,0])

    sphere_data = (mask, x, p, q)
    return sphere_data

# We want to make nested spheres where the inner sphere is polar and the outer sphere is non-polar
def make_2_spheres(N, non_polar_frac, radius1, radius2,
                    center1=np.array([0,0,0]), center2=np.array([0,0,0])):
    
    inner_sphere = make_random_sphere(int(N * (1-non_polar_frac)), 0, radius=radius1, center=center1)
    outer_sphere = make_random_sphere(int(N * non_polar_frac), 1, radius=radius2, center=center2)

    mask = np.concatenate((inner_sphere[0], outer_sphere[0]), axis=0)
    x = np.concatenate((inner_sphere[1], outer_sphere[1]), axis=0)
    p = np.concatenate((inner_sphere[2], outer_sphere[2]), axis=0)
    q = np.concatenate((inner_sphere[3], outer_sphere[3]), axis=0)

    nested_spheres_data = (mask, x, p, q)
    return nested_spheres_data

# lets make the torus 'square' and the sphere a cylinder in the middle instead
# height should be controllable, radius of the cylinder should be the inner radius of the torus
# and no particles should be placed under z=0

def make_square_torus(N, non_polar_frac, R, r, height):
    # Generate positions within a square torus
    # R is the outer radius of the torus, r is the inner radius
    # Thickness is R - r
    u = np.random.uniform(0, 2 * np.pi, N)
    v = np.random.uniform(0, height, N)
    w = np.random.uniform(0, 1, N)
    thickness = R - r
    c = thickness * np.sqrt(w)
    x = (r + c) * np.cos(u)
    y = (r + c) * np.sin(u)
    z = v
    pos = np.vstack((x, y, z)).T
    p = np.random.randn(N, 3)
    p /= np.sqrt(np.sum(p**2, axis=1))[:,None]
    q = np.random.randn(N, 3)
    q /= np.sqrt(np.sum(q**2, axis=1))[:,None]
    mask = np.random.choice([0,1], p=[non_polar_frac, 1-non_polar_frac], size=N)        #Mask detailing which particles are non polar
    p[mask == 0] = np.array([0,0,0])
    q[mask == 0] = np.array([0,0,0])
    torus_data = (mask, pos, p, q)
    return torus_data

def make_cylinder(N, non_polar_frac, R, height):
    # Generate positions within a cylinder
    # R is the radius of the cylinder
    theta = np.random.uniform(0, 2 * np.pi, N)
    rad = R * np.sqrt(np.random.uniform(0, 1, N))
    z = np.random.uniform(0, height, N)
    x = rad * np.cos(theta)
    y = rad * np.sin(theta)
    pos = np.vstack((x, y, z)).T
    p = np.random.randn(N, 3)
    p /= np.sqrt(np.sum(p**2, axis=1))[:,None]
    q = np.random.randn(N, 3)
    q /= np.sqrt(np.sum(q**2, axis=1))[:,None]
    mask = np.random.choice([0,1], p=[non_polar_frac, 1-non_polar_frac], size=N)        #Mask detailing which particles are non polar
    p[mask == 0] = np.array([0,0,0])
    q[mask == 0] = np.array([0,0,0])
    cylinder_data = (mask, pos, p, q)
    return cylinder_data

def make_encircling_square_torus(N, non_polar_frac, R, r, height):
    # R is the outer radius of the torus
    # r is the inner radius of the torus (where cylinder meets torus)
    # Cylinder has radius r
    torus_data = make_square_torus(int(N*non_polar_frac), 0, R, r, height)
    cylinder_data = make_cylinder(int(N*(1-non_polar_frac)), 1, r, height)

    mask = np.concatenate((torus_data[0], cylinder_data[0]))
    pos = np.vstack((torus_data[1], cylinder_data[1]))
    p = np.vstack((torus_data[2], cylinder_data[2]))
    q = np.vstack((torus_data[3], cylinder_data[3]))    

    encircling_square_torus_data = (mask, pos, p, q)
    return encircling_square_torus_data
