# mc/core/photon.py

import numpy as np

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
        """Moves photons to their intersections with a volume

        Args:
            Volume (Cylinder or RectPrism): volume to intersect

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
        self.pos[idx] = self.pos[idx] + distances[:, np.newaxis]*self.direc[idx]
        assert np.all(volume.contains(self.pos[idx])), "Some photons moved outside collimator after intersection!"
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