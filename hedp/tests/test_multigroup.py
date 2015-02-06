#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
from hedp.matdb import Storage
from unittest.case import SkipTest
import numpy as np
from numpy.testing import assert_allclose
from opacplot2.opg_hdf5  import OpgHdf5
from hedp.opacity.AGS import project_on_grid
try:
    from hedp.lib.multigroup import avg_mg_table
except ImportError:
    raise SkipTest
else:
    raise


def dsds_multigroup1():
    """Use the fact that given u=nu/temp 
        if κ = u⁻³,   κ_p^g = [ln(1 - e^(-u)]_g^{g+1}
    """
    Nr, Nt, Np, Ng = 30, 80, 1000, 100

    nu = np.logspace(-1, np.log10(50e3), Np+1)
    dens = np.logspace(-8, 2, Nr)
    temp = np.logspace(0.01, 5, Nt)
    groups_idx = np.array(np.linspace(0, Np, Ng+1), dtype='int')
    groups_new = nu[groups_idx]

    tab_init = Storage(groups=nu, dens=dens, temp=temp,
                opp_mg=np.random.rand(Nr, Nt, Np),
                opr_mg=np.random.rand(Nr, Nt, Np),
                emp_mg=np.random.rand(Nr, Nt, Np))
    tab = Storage(tab_init.copy())
    U = np.empty((Nr, Nt, Np))
    Ug = np.empty((Nr, Nt, Ng+1))
    for j in range(Nt):
        u = tab.groups[:-1] / tab.temp[j]
        ug = groups_new/tab.temp[j]
        for i in range(Nr):
            U[i,j] = u
            Ug[i,j] = ug

    tab['opp_mg'] = U**(-3)
    tab['opr_mg'] = U**(-4)
    tab['emp_mg'] = U**(-3)
    tab['opp_new'] = Ug[:,:,:-1]**(-3)
    tab['opr_new'] = Ug[:,:,:-1]**(-4)
    tab['emp_new'] = Ug[:,:,:-1]**(-3)
    op_mg = avg_mg_table(tab, groups_idx, ['opp_mg', 'opr_mg', 'emp_mg', 'Bg_p', 'Bg_r'])
    Bg_p, Bg_r = op_mg['Bg_p'], op_mg['Bg_r']
    yield  assert_allclose, np.isnan(Bg_p).any(), False, 1e-07, 0, 'Found nan in Planck weights'
    yield  assert_allclose, np.isnan(Bg_p).any(), False, 1e-07, 0, 'Found nan in Rosseland  weights'
    #
    # analytical expression doesn't work so well cause exp overflows.
    # checked on a plot that it works.
    # kpg_exp = np.log((1-np.exp(-Ug[:,:,1:]))/(1-np.exp(-Ug[:,:,:-1])))/op_mg['Bg_p']
    # krg_exp =  (1./(1-np.exp(-Ug[:,:,:-1])) - 1./(1-np.exp(-Ug[:,:,1:])))**(-1)
    for key in ['opp', 'opr']:
        yield assert_allclose, tab[key+'_new'], op_mg[key+'_mg'], 0.4, 0, 'Fast checking failed for '+key

def test_multigroup2():
    """Check that implemented averaging gives the same results as SNOP"""
    base_path = os.path.dirname(__file__)
    op0 = OpgHdf5.open_file(os.path.join(base_path, 'data/Al_snp_10kgr.h5'))
    op1 = OpgHdf5.open_file(os.path.join(base_path, 'data/Al_snp_40gr.h5'))
    group_idx, groups_new =  project_on_grid(op1['groups'][:], op0['groups'][:])
    res = avg_mg_table(op0, group_idx, fields=['opp_mg', 'opr_mg', 'emp_mg'],
            weight_pars={'opp_mg': 'planck', 'opr_mg': 'rosseland', 'emp_mg': "planck"}) 
    for key, rtol in {'opp_mg':1e-3, 'opr_mg':1e-3, 'emp_mg':0.7}.items():
        yield assert_allclose, op1[key][:], res[key], rtol, 0, 'Not the same '+key



