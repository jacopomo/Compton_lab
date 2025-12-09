# mc/geometry/volume.py

import numpy as np

from mc.utils.math3d import unpack_stacked, rotate_by_phi

class Cylinder:
    def __init__(self, disk, length):
        """Class that handles the cylindrical geometry

        Args:
            disk (Disk): "near" face of the cylinder (low z)
            height (float): extension along +z [cm]
        """
        self.center = disk.center   # nparray (3,) center of the "near" face
        self.radius = disk.radius   # instance of the disk class
        self.length = length        # extension along z
        self.angle = disk.angle     # angle about the vertical x axis (0 = head on)
    
    def contains(self, points):
        """Which points are contained in the cylinder

        Args:
            points (nparray): (N,3))

        Returns:
            nparray: (N,) array of bools
        """
        points = points - self.center 
        points = rotate_by_phi(points, -self.angle) # reference frame of the cylinder
        x, y, z = unpack_stacked(points)

        r = np.sqrt(x**2+y**2)
        mask = (r <= self.radius + 1e-9) & (z <= self.length + 1e-9) & (z > -1e-9)
        if mask.shape == ():
            return bool(mask)
        return mask

    def exit_distance(self, points, directions):
        """Method that computes the distance between each point and the cylinder volume edge

        Args:
            positions (nparray): (N,3) (x,y,z) positions of points within the volume
            directions (nparray): (N,3) unit vector directions associated to the positions

        Returns:
            nparray, nparray: (N,), (N,) distances until the volume is exited [cm], array of bools True = exited base
        """
        points = np.asarray(points, dtype=float)
        directions = np.asarray(directions, dtype=float)
        N = points.shape[0]

        # Basic sanity checks
        if points.shape[1] != 3 or directions.shape[1] != 3:
            raise ValueError("points and directions must be shaped (N,3)")
    
        assert np.all(self.contains(points)), "All initial points must be inside the volume."

        points = points - self.center 
        points = rotate_by_phi(points, -self.angle) # reference frame of the cylinder
        directions = rotate_by_phi(directions, -self.angle)

        px, py, pz = unpack_stacked(points) 
        pz = pz + np.full(len(pz), 1e-9) # shift to avoid self-intersection 
        dx, dy, dz = unpack_stacked(directions)

        def safe_div(numerator, denominator): # avoid division by zero (helper)
            with np.errstate(divide='ignore'):
                div = np.where(np.abs(denominator) > 1e-9, numerator / denominator, np.inf)
            return div 
               
        # Intersection with the z=0, z=L base, asserting dz not 0
        dist_0 = safe_div(-pz,dz)
        dist_L = safe_div(self.length - pz, dz)

        # Intersection with curved sides
        # This involves solving a quadratic equation for t: (px+dx*t)^2 + (py+dy*t)^2 = r^2
        # a*t^2 + b*t + c = 0
        a = dx**2 + dy**2
        b = 2 * ((px * dx) + (py * dy))
        c = px**2 + py**2 - self.radius**2

        a = np.where(np.abs(a) > 1e-9, a, 1e-9)

        delta = b**2 - (4 * a * c)
        delta_mask = delta >= 0

        # Initialize intersection distances to infinity 
        dist_wall_1 = np.full(points.shape[0], np.inf)
        dist_wall_2 = np.full(points.shape[0], np.inf)

        dist_wall_1[delta_mask] = (-b[delta_mask] + np.sqrt(delta[delta_mask])) / (2 * a[delta_mask])
        dist_wall_2[delta_mask] = (-b[delta_mask] - np.sqrt(delta[delta_mask])) / (2 * a[delta_mask])

        all_ints = np.stack([dist_0, dist_L, dist_wall_1, dist_wall_2], axis=1)

        ints = np.where(all_ints > 1e-9, all_ints, np.inf) # only keep positives (forward intersections)
        int_distance = np.min(ints, axis=1)
        int_dist_idx = np.argmin(ints, axis=1)
        exited_base = (int_dist_idx == 0)|(int_dist_idx == 1)

        if np.any(np.isinf(int_distance)):
            print("Warning: Some rays never exit the volume based on parameters.")

        return int_distance, exited_base

class RectPrism:
    def __init__(self, rectangle, length):
        """Class that handles rectangular prism geometry

        Args:
            rectangle (Rectangle): "near" face of the rectangular prism (low z)
            height (float): extension along +z [cm]
        """
        self.center = rectangle.center  # center of the "near" face
        self.width = rectangle.width    # semi-extension along y
        self.height = rectangle.height  # semi-extension along x
        self.length = length            # extension along z
        self.angle = rectangle.angle    # angle about the vertical x axis (0 = head on)

    def contains(self, points):
        """Which points are contained in the rectangular prism

        Args:
            points (nparray): (N,3))

        Returns:
            nparray: (N,) array of bools
        """
        points = points - self.center 
        points = rotate_by_phi(points, - self.angle) # reference frame of the prism
        x, y, z = unpack_stacked(points)

        mask = (np.abs(x) < self.height) & (np.abs(y) < self.width) & (z < self.length) & (z > -1e-9)
        if mask.shape == ():
            return bool(mask)
        return mask
    
    def exit_distance(self, points, directions):
        """Method that computes the distance between each point and the prism volume edge

        Args:
            positions (nparray): (N,3) (x,y,z) positions of points within the volume
            directions (nparray): (N,3) unit vector directions associated to the positions

        Returns:
            nparray, nparray: (N,), (N,) distances until the volume is exited [cm], array of bools True = exited base
        """
        points = np.asarray(points, dtype=float)
        directions = np.asarray(directions, dtype=float)
        N = points.shape[0]

        # Basic sanity checks
        if points.shape[1] != 3 or directions.shape[1] != 3:
            raise ValueError("points and directions must be shaped (N,3)")

        H, W, L = self.height, self.width, self.length

        assert np.all(self.contains(points)), "All initial points must be inside the volume."

        points = points - self.center 
        points = rotate_by_phi(points, -self.angle) # reference frame of the prism
        directions = rotate_by_phi(directions, -self.angle)

        px, py, pz = unpack_stacked(points)
        pz = pz + np.full(len(pz), 1e-9) # shift to avoid self-intersection with z=0 plane
        dx, dy, dz = unpack_stacked(directions)

        def safe_div(numerator, denominator): # avoid division by zero (helper)
            with np.errstate(divide='ignore'):
                div = np.where(np.abs(denominator) > 1e-9, numerator / denominator, np.inf)
            return div

        dist_xmin = safe_div(-H - px, dx)
        dist_xmax = safe_div(H - px, dx)
        dist_ymin = safe_div(-W - py, dy)
        dist_ymax = safe_div(W - py, dy)
        dist_zmin = safe_div(0 - pz, dz)
        dist_zmax = safe_div(L - pz, dz)

        all_ints = np.stack([dist_xmin, dist_xmax, dist_ymin, dist_ymax, dist_zmin, dist_zmax], axis=1)
        
        ints = np.where(all_ints > 1e-9, all_ints, np.inf) # only keep positives (forward intersections)
        int_distance = np.min(ints, axis=1)
        int_dist_idx = np.argmin(ints, axis=1)
        exited_base = (int_dist_idx == 4)|(int_dist_idx == 5)

        if np.any(np.isinf(int_distance)):
            print("Warning: Some rays never exit the volume based on parameters.")

        return int_distance, exited_base
