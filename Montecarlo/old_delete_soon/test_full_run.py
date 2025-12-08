#!/usr/bin/env python
import sys
sys.path.insert(0, r'c:\Users\jacop\University\Compton_lab\Montecarlo')
import compton_MC as m
m.N_MC = 100000  # reduce for speed
print('Starting plot_compton...')
try:
    result = m.plot_compton(phi_cristallo=20, plot_scatter_angles=False, all_peaks=True)
    print('SUCCESS: plot_compton completed')
    print('Result shape:', result.shape if hasattr(result, 'shape') else len(result))
except Exception as e:
    import traceback
    traceback.print_exc()
