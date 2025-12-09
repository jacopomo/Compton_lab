# mc/core/photon.py

import numpy as np

from mc.physics.compton import compton
from mc.physics.kn_sampler import sample_kn, PDF, CDF, THETA_GRID, E_GRID

class Photons:
    def __init__(self, N):
        self.pos = np.zeros((N, 3), dtype=np.float64)       # (N,3) x, y, z
        self.direc = np.zeros((N, 3), dtype=np.float64)     # (N,3) dx, dy, dz, unit vectors
        self.energy = np.zeros(N, dtype=np.float64)         # (N,)
        self.weight = np.ones(N, dtype=np.float64)          # (N,)
        self.alive = np.ones(N, dtype=bool)                 # (N,)

    def append(self, pos, direc, energy, weight=None, alive=None):
        """Adds a new batch of photons to the photon pool, useful for splitting

        Args:
            pos (nparray): (M,3) or (3,) photon positions
            direc (nparray): (M,3) or (3,) photon directions
            energy (nparray): (M,) photon energies
            weight (nparray, optional): photon weights. Defaults to None.
            alive (nparray, optional): photon state. Defaults to None.
        """
        # pos: (M,3) or (3,), dir similar, energy: (M,)
        pos = np.atleast_2d(pos)
        direc = np.atleast_2d(direc)
        energy = np.atleast_1d(energy)
        M = pos.shape[0]
        if weight is None:
            weight = np.ones(M, dtype=self.energy.dtype)
        if alive is None:
            alive = np.ones(M, dtype=bool)
        self.pos = np.vstack([self.pos, pos])
        self.direc = np.vstack([self.direc, direc])
        self.energy = np.concatenate([self.energy, energy])
        self.weight = np.concatenate([self.weight, weight])
        self.alive = np.concatenate([self.alive, alive])

    def compact(self):
        """Removes dead photons to keep arrays small
        """
        keep = self.alive
        self.pos = self.pos[keep]
        self.direc = self.direc[keep]
        self.energy = self.energy[keep]
        self.weight = self.weight[keep]
        self.alive = np.ones(len(self.energy), dtype=bool)

    def move_to_int(self, volume, mask=None, idx=None):
        """Moves photons to their intersections with a volume.
        Only one of a mask or indexes must be passed, or neither to operate on all

        Args:
            volume (Cylinder or RectPrism): volume to intersect
            mask (nparray, optional): mask of photons to operate on. Defaults to None.
            idx (nparray, optional): indexes of photons to operate on. Defaults to None.

        Raises:
            ValueError: provided both mask and index

        Returns:
            nparray: (N,) array of bools, True = exited base
        """
        if (mask is not None) and (idx is not None):
            raise ValueError("Provide only one of mask or idx")
        
        if mask is not None:
            idx = np.where(mask)[0]
        elif idx is None:
            idx = np.where(self.alive)[0]
        
        if idx.size == 0:
            return
        
        distances, exit_base = volume.exit_distance(self.pos[idx], self.direc[idx])
        self.pos[idx] = self.pos[idx] + 0.999*distances[:, np.newaxis]*self.direc[idx]
        assert np.all(volume.contains(self.pos[idx])), "Some photons moved outside volume after intersection!"
        return exit_base
    
    def scatter_update_dirs(self, scatter_angles, mask=None, idx=None):
        """Updates photon directions by sampling a random azimuth around
        original direction and shifting by an angle in that direction
        
        Args:
            scatter_angles (nparray): (N,) scattering angles 
            mask (nparray, optional): mask for which photons to operate on. Defaults to None.
            idx (nparray, optional): indexes of which photons to operate on. Defaults to None.
        """
        if (mask is not None) and (idx is not None):
            raise ValueError("Provide only one of mask or idx")
        
        if mask is not None:
            idx = np.where(mask)[0]
        elif idx is None:
            idx = np.where(self.alive)[0]
        
        if idx.size == 0:
            return

        d = self.direc[idx]
        x = np.array([1., 0., 0.])
        mask = np.abs(d[:,0]) > 0.99
        a = np.where(mask[:,None], np.array([0.,0.,1.]), x)
        u = np.cross(a, d)
        u /= np.linalg.norm(u, axis=1)[:,None]
        
        v = np.cross(d, u)

        delta = np.random.uniform(-np.pi, np.pi, len(scatter_angles))
        new = (u * (np.sin(scatter_angles) * np.cos(delta))[:,None] +
            v * (np.sin(scatter_angles) * np.sin(delta))[:,None] +
            d * np.cos(scatter_angles)[:,None])
        self.direc[idx] = new

    def force_one_scatter_moveto(self, volume, mask=None, idx=None):

        if (mask is not None) and (idx is not None):
            raise ValueError("Provide only one of mask or idx")

        if mask is not None:
            idx = np.where(mask)[0]
        elif idx is None:
            idx = np.where(self.alive)[0]

        if idx.size == 0:
            return

        mat = volume.material
        if mat is None:
            raise AssertionError("Volume must have a 'material' attribute")

        # Energies and directions for selected photons (local arrays)
        E_sel = self.energy[idx].astype(float)        # (M,)
        pos_sel = self.pos[idx]                       # (M,3)
        dir_sel = self.direc[idx]                     # (M,3)

        L_sel, _ = volume.exit_distance(pos_sel, dir_sel)

        # macroscopic total (1/cm) at energies E_sel
        Sigma_tot = 1 / mat.mfp_compton(E_sel)    # vectorized

        # probability to interact at least once inside L
        P_int = 1.0 - np.exp(-Sigma_tot * L_sel)   # (M,)

        # skip photons with essentially zero interaction probability
        valid_mask = P_int > 0.0
        if not np.all(valid_mask):
            keep_idx = idx[valid_mask]
            self.alive[~keep_idx] = False
        else:
            keep_idx = idx

        if keep_idx.size == 0:
            return

        # Redefine arrays limited to valid indices
        rel = np.nonzero(valid_mask)[0]  # positions inside E_sel arrays
        E_v = E_sel[rel].copy()
        pos_v = pos_sel[rel].copy()
        dir_v = dir_sel[rel].copy()
        L_v = L_sel[rel].copy()
        Sigma_v = Sigma_tot[rel].copy()
        P_v = P_int[rel].copy()
        global_idx_v = idx[rel]   # global indices in the pool

        M = len(global_idx_v)

        # --- sample collision distance s from truncated exponential (vectorized) ---
        # s = -1/Sigma * ln(1 - u*(1 - exp(-Sigma*L)))
        u = np.random.random(M)
        exp_term = np.exp(-Sigma_v * L_v)
        # guard against numerical issues when Sigma_v is very small
        # if Sigma_v==0 would have been filtered by valid_mask
        s = -np.log(1.0 - u * (1.0 - exp_term)) / Sigma_v  # (M,)

        # move photons to the sampled collision points (in place)
        self.pos[global_idx_v] = pos_v + (0.999 * s[:, None] * dir_v)

        # multiply weights by P_int (likelihood ratio) to correct for forced collision
        self.weight[global_idx_v] *= P_v

        angles, _ = sample_kn(self.energy[global_idx_v], E_GRID, THETA_GRID, CDF)
        self.energy[global_idx_v] = compton(self.energy[global_idx_v], angles)
        self.scatter_update_dirs(angles, idx=global_idx_v)
        exit_base = self.move_to_int(volume, idx=global_idx_v)
        return exit_base 