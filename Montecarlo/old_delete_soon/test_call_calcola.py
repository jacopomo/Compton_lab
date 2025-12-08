import sys
sys.path.insert(0, r'c:\Users\jacop\University\Compton_lab\Montecarlo')
import compton_MC as m
import numpy as np
# create fake exits
N=5
x = np.linspace(-1,1,N)
y = np.zeros(N)
z = np.full(N, 10.0)
phi = np.zeros(N); psi = np.zeros(N)
E = np.full(N, 1000.)
f_exit = m.Fotone(E, x, y, z, phi, psi)
res = f_exit.calcola_int(m.Superficie(5,(0,0,20),0), debug_graph=False)
print('len(res)=', len(res))
for i,r in enumerate(res):
    print(i, type(r), getattr(r,'shape', None))
