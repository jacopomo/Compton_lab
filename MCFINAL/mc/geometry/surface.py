import numpy as np
class Disk:
    def __init__(self, center, radius, angle):
        self.center = center # x,y,z numpy array
        self.radius = radius # float
        self.angle = angle   # angle about the vertical x axis (0 = head on)

    def sample_unif(self, n):
        """_summary_

        Args:
            n (int): number of points to generate

        Returns:
            numpy.ndarray: nx3 numpy array of x,y,z positions
        """
        thetas = np.random.uniform(0,2*np.pi, n)
        rs = self.radius * np.sqrt(np.random.uniform(0,1,n))
        px = self.center[0] + (rs*np.sin(thetas))
        py = self.center[1] + (rs*np.cos(thetas)*np.cos(self.angle))
        pz = self.center[2] + (rs*np.cos(thetas)*np.sin(self.angle)) 
        return np.stack((px,py,pz), axis=-1)
    
class Rectangle:
    def __init__(self, center, length, width, angle):
        self.center = center # x,y,z numpy array
        self.length = length # extension along y
        self.width = width   # extension along x
        self.angle = angle   # angle about the vertical x axis (0 = head on)
    
    def sample_unif(self, n):
        """_summary_

        Args:
            n (int): number of points to generate

        Returns:
            numpy.ndarray: nx3 numpy array of x,y,z positions
        """
        lens  = self.length * np.random.uniform(-0.5,0.5,n)
        widths = self.width * np.random.uniform(-0.5,0.5,n)
        px = self.center[0] + widths
        py = self.center[1] + lens*np.cos(self.angle)
        pz = self.center[2] + lens*np.sin(self.angle)
        return np.stack((px,py,pz), axis=-1)
