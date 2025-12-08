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

    def move_to_int(self, volume):
        """Moves photons to their intersections with a volume

        Args:
            volume (Cylinder or RectPrism): volume to intersect
        """
        distances, _ = volume.exit_distance(self.pos, self.direc)
        self.pos = self.pos + distances[:, np.newaxis]*self.direc