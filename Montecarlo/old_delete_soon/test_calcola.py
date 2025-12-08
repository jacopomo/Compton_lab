import sys
sys.path.insert(0, r'c:\Users\jacop\University\Compton_lab\Montecarlo')
import compton_MC as m
import numpy as np
surf = m.Superficie(5, (0,0,10), 0)
N = 1000
E = np.full(N, 1000.0)
px = np.zeros(N); py = np.zeros(N); pz = np.full(N, -1.0)
phi = np.zeros(N); psi = np.zeros(N)
f = m.Fotone(E, px, py, pz, phi, psi)
res = f.calcola_int(surf, debug_graph=False)
print('Returned types:')
for i, arr in enumerate(res):
    if arr is None:
        print(i, 'None')
    else:
        try:
            print(i, 'shape', np.array(arr).shape)
        except Exception as e:
            print(i, 'error', e)
