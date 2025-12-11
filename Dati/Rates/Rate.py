import numpy as np
import matplotlib.pylab as plt

pmt1    = np.array([1481714, 3168607, 2452305, 1573744, 816803, 3034113, 3590344, 2274516, 985582, 1294211, 1736401, 874731])
pmt2    = np.array([1862076, 254243,  116768,  61858,   29099,  1786804, 207934,  92165,   35962,  39237,   50533,   29117])
both    = np.array([461,     201,     233,     400,     401,    469,     250,     454,     404,    550,     263,     430])
delta_t = np.array([599900,  1287160, 999982,  642493,  334747, 1238941, 1464710, 928385,  402655, 528956,  710797,  357859])
deg     = np.array([0,       5,       10,      15,      20,     2.5,     7.5,     12.5,    17.5,   30,      40,      25])

rate1 = pmt1/delta_t
rate2 = pmt2/delta_t
rate_both = both/delta_t

fig, ax1 = plt.subplots()

ax1.errorbar(deg, rate_both*1e3, yerr=np.sqrt(both)/delta_t*1e3, fmt='.', label='Rate coincidenze')
ax1.set_ylabel('Rate delle coincidenze [#/s]')
ax1.set_xlabel('Angolo [deg]')

ax2 = ax1.twinx()
ax2.errorbar(deg, rate1, yerr=np.sqrt(pmt1)/delta_t, fmt='.', label='Rate PMT1', color='r')
ax2.errorbar(deg, rate2, yerr=np.sqrt(pmt2)/delta_t, fmt='.', label='Rate PMT2', color='black')
ax2.set_ylabel('Rate PMT2 [#/ms]')

ax1.grid()
ax2.grid(linestyle='--')
plt.show()