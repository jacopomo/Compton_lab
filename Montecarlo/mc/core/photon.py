# mc/core/photon.py

import numpy as np

from mc.physics.compton import compton, theta_min_threshold
from mc.physics.kn_sampler import sample_kn, PDF, CDF, THETA_GRID, E_GRID
from mc.utils.math3d import rotate_by_phi, unpack_stacked

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

    def split(self, n):
        assert type(n) == int, "n must be an int"
        if n < 1:
            raise ValueError("n must be >= 1")
        if n == 1:
            return

        # --- divide weights of existing photons ---
        self.weight /= n

        # --- duplicate photons n-1 times ---
        pos_new   = np.repeat(self.pos,   n - 1, axis=0)
        direc_new = np.repeat(self.direc, n - 1, axis=0)
        energy_new = np.repeat(self.energy, n - 1)
        weight_new = np.repeat(self.weight, n - 1)
        alive_new  = np.repeat(self.alive,  n - 1)

        # --- append ---
        self.append(
            pos=pos_new,
            direc=direc_new,
            energy=energy_new,
            weight=weight_new,
            alive=alive_new
        )

    def _resolve_idx(self, mask, idx):
        if (mask is not None) and (idx is not None):
            raise ValueError("Provide only one of mask or idx")
        if mask is not None:
            return np.where(mask)[0]
        if idx is not None:
            return idx
        return np.where(self.alive)[0]

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
        idx = self._resolve_idx(mask, idx)
        if idx.size == 0:
            return
        
        distances, exit_base = volume.exit_distance(self.pos[idx], self.direc[idx])
        self.pos[idx] = self.pos[idx] + 0.999*distances[:, np.newaxis]*self.direc[idx]
        inside = volume.contains(self.pos[idx])

        bad = ~inside
        if np.any(bad):
            self.alive[idx[bad]] = False        
        return exit_base
    
    def scatter_update_dirs(self, scatter_angles, mask=None, idx=None):
        """Updates photon directions by sampling a random azimuth around
        original direction and shifting by an angle in that direction
        
        Args:
            scatter_angles (nparray): (N,) scattering angles 
            mask (nparray, optional): mask for which photons to operate on. Defaults to None.
            idx (nparray, optional): indexes of which photons to operate on. Defaults to None.
        """
        idx = self._resolve_idx(mask, idx)

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

    def _force_first_compton(self, volume, E_th, idx):
        mat = volume.material

        E = self.energy[idx]
        pos = self.pos[idx]
        direc = self.direc[idx]
        w = self.weight[idx]

        L_exit, _ = volume.exit_distance(pos, direc)
        mfp = mat.mfp_compton(E)

        P_int = 1.0 - np.exp(-L_exit / mfp)

        u = np.random.random(len(E))
        s = -np.log(1.0 - u * P_int) * mfp

        pos += 0.999 * s[:, None] * direc
        w *= P_int

        print(f"minimum scattering angle inside {volume.material.name} to have a signal: {round(np.degrees(theta_min_threshold(E, E_th).min()),1)} deg\n")
        angles, w_kn = sample_kn(E, E_GRID, THETA_GRID, CDF, theta_low=theta_min_threshold(E, E_th), theta_high=np.pi)
        w *= w_kn
        E[:] = compton(E, angles)

        self.scatter_update_dirs(angles, idx=idx)

        return E, pos, direc, w

    def _transport_until_exit_or_absorb(
        self, volume, idx, E, pos, direc, max_steps
    ):
        mat = volume.material

        alive = np.ones(len(E), dtype=bool)
        absorbed = np.zeros(len(E), dtype=bool)

        for _ in range(max_steps):
            if not np.any(alive):
                break

            ia = np.where(alive)[0]

            pos_a = pos[ia]
            dir_a = direc[ia]
            E_a = E[ia]

            L_exit, _ = volume.exit_distance(pos_a, dir_a)

            mfp_c = mat.mfp_compton(E_a)
            mfp_pe = mat.mfp_pe(E_a)

            Sigma_c = 1.0 / mfp_c
            Sigma_pe = 1.0 / mfp_pe
            Sigma_t = Sigma_c + Sigma_pe

            s = -np.log(np.random.random(len(ia))) / Sigma_t
            exits = s >= L_exit

            alive[ia[exits]] = False

            ii = ia[~exits]
            if ii.size == 0:
                continue

            pos[ii] += 0.999 * s[~exits][:, None] * direc[ii]

            u = np.random.random(len(ii))
            is_c = u < (Sigma_c[~exits] / Sigma_t[~exits])

            pe = ii[~is_c]
            absorbed[pe] = True
            alive[pe] = False
            E[pe] = 0.0

            co = ii[is_c]
            if co.size:
                angles, _ = sample_kn(E[co], E_GRID, THETA_GRID, CDF)
                E[co] = compton(E[co], angles)
                self.scatter_update_dirs(angles, idx=co)

        return E, alive & (~absorbed)

    def force_one_scatter_moveto(self, volume, E_th, mask=None, idx=None):
        idx = self._resolve_idx(mask, idx)
        if idx.size == 0:
            return

        E, pos, direc, w = self._force_first_compton(volume, E_th, idx)

        self.energy[idx] = E
        self.pos[idx] = pos
        self.weight[idx] = w

        return self.move_to_int(volume, idx=idx)
    
    def force_first_then_transport(self, volume, E_th, mask=None, idx=None, max_steps=100):
        idx = self._resolve_idx(mask, idx)
        if idx.size == 0:
            return np.array([])

        E, pos, direc, w = self._force_first_compton(volume, E_th, idx)

        E, alive = self._transport_until_exit_or_absorb(
            volume, idx, E, pos, direc, max_steps
        )

        self.energy[idx] = E
        self.pos[idx] = pos
        self.weight[idx] = w
        self.alive[idx] = alive

        return E
    
    def moveto_int_disk(self, disk, mask=None, idx=None):

        if (mask is not None) and (idx is not None):
            raise ValueError("Provide only one of mask or idx")

        if mask is not None:
            idx = np.where(mask)[0]
        elif idx is None:
            idx = np.where(self.alive)[0]

        N = len(self.energy)
        hit_mask = np.zeros(N, dtype=bool)

        if idx.size == 0:
            return hit_mask

        pos = rotate_by_phi(self.pos[idx] - disk.center, -disk.angle) 
        direc = rotate_by_phi(self.direc[idx], -disk.angle)

        px, py, pz = unpack_stacked(pos) 
        dx, dy, dz = unpack_stacked(direc)

        def safe_div(numerator, denominator): # avoid division by zero (helper)
            with np.errstate(divide='ignore'):
                div = np.where(np.abs(denominator) > 1e-9, numerator / denominator, np.nan)
            return div 

        t = safe_div(-pz,dz)

        forward = (~np.isnan(t)) & (t >= 0.0)
        if not np.any(forward):
            return hit_mask

        t = t[forward]
        idx_fwd = idx[forward]

        pts = pos[forward] + (t[:, None] * direc[forward])

        d2 = pts[:,0]**2 + pts[:,1]**2
        inside = d2 <= disk.radius**2
        if not np.any(inside):
            return hit_mask

        idx_hit = idx_fwd[inside]
        pts = rotate_by_phi(pts, disk.angle) + disk.center # bring back to original coords
        self.pos[idx_hit] = pts[inside]

        hit_mask[idx_hit] = True

        return hit_mask

    def moveto_int_rect(self, rect, mask=None, idx=None):

        if (mask is not None) and (idx is not None):
            raise ValueError("Provide only one of mask or idx")

        if mask is not None:
            idx = np.where(mask)[0]
        elif idx is None:
            idx = np.where(self.alive)[0]

        N = len(self.energy)
        hit_mask = np.zeros(N, dtype=bool)

        if idx.size == 0:
            return hit_mask

        pos = rotate_by_phi(self.pos[idx] - rect.center, -rect.angle) 
        direc = rotate_by_phi(self.direc[idx], -rect.angle)

        px, py, pz = unpack_stacked(pos) 
        dx, dy, dz = unpack_stacked(direc)

        def safe_div(numerator, denominator): # avoid division by zero (helper)
            with np.errstate(divide='ignore'):
                div = np.where(np.abs(denominator) > 1e-9, numerator / denominator, np.nan)
            return div 

        t = safe_div(-pz,dz)

        forward = (~np.isnan(t)) & (t >= 0.0)
        if not np.any(forward):
            return hit_mask

        t = t[forward]
        idx_fwd = idx[forward]

        pts = pos[forward] + (t[:, None] * direc[forward])

        x, y, _ = unpack_stacked(pts)
        inside = (np.abs(x) < rect.height) & (np.abs(y) < rect.width)

        if not np.any(inside):
            return hit_mask

        idx_hit = idx_fwd[inside]
        pts = rotate_by_phi(pts, rect.angle) + rect.center # bring back to original coords
        self.pos[idx_hit] = pts[inside]

        hit_mask[idx_hit] = True

        return hit_mask
