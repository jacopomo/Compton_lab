import sys
sys.path.insert(0, r'c:\Users\jacop\University\Compton_lab\Montecarlo')
import compton_MC as m
m.N_MC = 1000
print('Running mc with N_MC=1000')
try:
    energie, scatter_angles = m.mc(m.E1, phi_cristallo=20)
    print('mc finished: energie shape', getattr(energie, 'shape', None))
except Exception as e:
    import traceback
    traceback.print_exc()
    print('mc raised')
