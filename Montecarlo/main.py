import numpy as np
import matplotlib.pyplot as plt

from mc.utils.plotting import plot_photon_positions
from mc.utils.math3d import unpack_stacked
from mc.geometry.surface import Rectangle, Disk
from mc.core.photon import Photons
from mc.physics.kn_sampler import kn, build_kn_lut, sample_kn
from mc.config import RE

sorgente = Rectangle(np.array([0,0,-10]), 4.0, 4.0, np.radians(0))
pos = sorgente.sample_unif(1000)
