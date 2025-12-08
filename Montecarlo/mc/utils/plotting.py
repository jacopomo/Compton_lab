import numpy as np
import matplotlib.pyplot as plt

def plot_photon_positions(x,y,z):
    """
    Plots and shows 3-D scatterplot of photon positions, rotating the axes
    to reflect the laboratory setting
    
    :param x: Numpy array of length N, photon x-positions
    :param y: Numpy array of length N, photon y-positions
    :param z: Numpy array of length N, photon z-positions
    """

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x,y,z) 
    ax.view_init(elev=285, azim=0)     # Orients the axes correctly
    ax.set_box_aspect([1,1,1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    plt.show()