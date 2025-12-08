# mc/utils/plotting.py

import numpy as np
import matplotlib.pyplot as plt

from mc.utils.math3d import unpack_stacked

def plot_photon_positions(positions):
    """Plots and shows 3-D scatterplot of photon positions, rotating the axes
    to reflect the laboratory setting

    Args:
        positions (nparray): (N,3) numpy array of x,y,z photon positions
    """
    x, y, z = unpack_stacked(positions)
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x,y,z) 
    ax.view_init(elev=285, azim=0)     # Orients the axes correctly

    # --- FIX FOR EQUAL AXIS EXTENSIONS ---
    # 1. Calculate the range of data in each dimension
    max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() / 2.0

    # 2. Find the center of the data in each dimension
    mid_x = (x.max() + x.min()) * 0.5
    mid_y = (y.max() + y.min()) * 0.5
    mid_z = (z.max() + z.min()) * 0.5

    # 3. Set the limits for all axes using the same maximum range
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    # -------------------------------------
    
    ax.set_box_aspect([1,1,1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    plt.show()