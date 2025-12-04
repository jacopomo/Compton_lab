import numpy as np

class Photons:
    def __init__(self, N):
        self.pos = np.zeros((N, 3), dtype=np.float64)      # x, y, z
        self.direc = np.zeros((N, 3), dtype=np.float64)      # dx, dy, dz
        self.energy = np.zeros(N, dtype=np.float64)
        self.weight = np.ones(N, dtype=np.float64)
        self.alive = np.ones(N, dtype=bool)