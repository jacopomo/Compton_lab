import numpy as np
import matplotlib.pyplot as plt

import utils as u

conf = np.loadtxt(r"Analisi/config.txt", dtype=str, unpack=True)


#Cerco i file che mi servono nella directory indicata nel file config.txt
files = u.searchfiles(conf, "dat")
print(files)