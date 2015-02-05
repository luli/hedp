#!/usr/bin/python
# -*- coding: utf-8 -*-

from hedp.matdb import Storage
from hedp.lib.multigroup import *
import numpy as np
#import cProfile, pstats, StringIO
import matplotlib.pyplot as plt


Nr, Nt, Np, Ng = 110, 120, 1000, 32

nu = np.logspace(-1, np.log10(50e3), Np+1)
dens = np.logspace(-8, 2, Nr)
temp = np.logspace(0.01, 5, Nt)
groups_idx = np.array(np.linspace(0, Np, Ng+1), dtype='int')

tab = Storage(groups=nu, dens=dens, temp=temp,
            opp_mg=np.random.rand(Nr, Nt, Np),
            opr_mg=np.random.rand(Nr, Nt, Np),
            emp_mg=np.random.rand(Nr, Nt, Np))
#pr = cProfile.Profile()
#pr.enable()

op_mg = avg_mg_table(tab, groups_idx, eps=False, emp=False)


ax = plt.subplot(111)

j = 100
for j in [ 20]:
    temp = tab.temp[j]
    nu = tab.groups[:-1]
    u = tab.groups[:-1] / temp
    Bnu = op_mg['Bnu_p'][j]*np.exp(-u)/np.diff(tab.groups)
    Bnu2 = u**3*np.exp(-u)/(1 - np.exp(-u))
    plt.plot(nu, Bnu2/Bnu2.max(), '--')
    plt.step(nu, Bnu/Bnu.max(),
                    label='{0:.0f} eV'.format(temp))
    plt.plot(nu, op_mg['Bnu'][j], ':')

ax.set_xlim(0, 50)


plt.legend(loc='best')
plt.savefig('test_mg.png', bbox_inches='tight')

#s = StringIO.StringIO()
#sortby = 'cumulative'
#ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
#ps.print_stats()
#print s.getvalue()
