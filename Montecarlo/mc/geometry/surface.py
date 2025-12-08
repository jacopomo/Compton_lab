# mc/geometry/surface.py

import numpy as np

from mc.utils.math3d import rotate_by_phi

class Disk:
    def __init__(self, center, radius, angle):
        """Class that handles circular surfaces

        Args:
            center (nparray): (3,) x,y,z center coordinates [cm]
            radius (float): radius of the disk [cm]
            angle (float): angle about the veritcal x axis (0 = head on) [radians]
        """
        self.center = center # x,y,z numpy array
        self.radius = radius # float
        self.angle = angle   # angle about the vertical x axis (0 = head on)

    def sample_unif(self, n):
        """Generates n points uniformly on the surface of the disk

        Args:
            n (int): number of points to generate

        Returns:
            numpy.ndarray: (n,3) numpy array of x,y,z positions
        """
        thetas = np.random.uniform(0,2*np.pi, n)
        rs = self.radius * np.sqrt(np.random.uniform(0,1,n))
        px = rs*np.sin(thetas)
        py = rs*np.cos(thetas)
        pz = np.zeros(len(thetas))

        points = rotate_by_phi(np.stack((px,py,pz), axis=-1), self.angle)

        return points + self.center
    
class Rectangle:
    def __init__(self, center, width, height, angle):
        """Class that handles rectangular surfaces

        Args:
            center (nparray): (3,) x,y,z center coordinates [cm]
            width (float): semi-extension along y [cm]
            height (float): semi-extension along x [cm]
            angle (float): angle about the veritcal x axis (0 = head on) [radians]
        """
        self.center = center    # x,y,z numpy array
        self.width = width      # semi-extension along y
        self.height = height    # semi-extension along x
        self.angle = angle      # angle about the vertical x axis (0 = head on)
    
    def sample_unif(self, n):
        """Generates n points uniformly on the rectangular surface

        Args:
            n (int): number of points to generate

        Returns:
            numpy.ndarray: (n,3) numpy array of x,y,z positions
        """
        widths  = self.width * np.random.uniform(-1,1,n)
        heights = self.height * np.random.uniform(-1,1,n)
        px = heights
        py = widths
        pz = np.zeros(len(widths))

        points = rotate_by_phi(np.stack((px,py,pz), axis=-1), self.angle)

        return points + self.center


# intersect(ray_origin, ray_direction)
# returns the coordinates of the intersection with the surface or None None None if it does not