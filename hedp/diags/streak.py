#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright CNRS 2012
# Roman Yurchak (LULI)
# This software is governed by the CeCILL-B license under French law and
# abiding by the rules of distribution of free software.

import numpy as np
from scipy.interpolate import interp1d

_sens_data = {'hamamatsu': {
    'S20_2' : np.array([[198.7,9.71],[207.5,14.4],[221.2,19.8],[238.3,27.3],
                        [259.5,34.9],[295.8,37.6],[336.1,40.9],[370.7,46.7],
                        [402.8,52.1],[425.9,53.4],[451.5,50.2],[474.7,46.7],
                        [501.2,42.8],[599.7,31.4],[637.1,26.5],[700.8,21.7],
                        [754.6,17.7],[795.3,14.6],[835.1,11.2],[861.0,8.04],
                        [877.2,4.79],[915.6,0.459],[948.8,0.0624],[988.4,0.0103]])
        }
    }

_qe_data = {'hamamatsu': {
    "S20_2" : np.array([[199.8,5.89],[224.0,11.2],[247.7,15.0],[269.0,16.4],
                        [341.7,14.9],[377.9,15.5],[406.0,15.9],[448.1,14.0],
                        [668.4,4.52],[799.3,2.21],[852.5,1.36],[876.1,0.710],
                        [946.3,0.0101]])   }
    }

def streak_sens(lmbda, manufacturer, model):
    """
    Return the Radiant Sensitivity and the Quantum efficiency
    of LULI streaked cameras.

    Parameters:
    -----------
       - lmbda [ndarray]: wavelenght [nm]
       - manufacturer [str]: curently in ['hamamatsu']
       - model [str]: streak id. Currently in ['S20_2']

    Returns:
    --------
       - rad_sens: Radiant Sensitivity [mA/W]
       - quant_eff: Quantum efficiency [in %]
    """
    d_radeff  = _sens_data[manufacturer][model]
    d_qe  = _qe_data[manufacturer][model]
    # interpolate in ylog scale
    i_radeff = interp1d(d_radeff[:,0], np.log10(d_radeff[:,1]), kind='linear')
    i_qe = interp1d(d_qe[:,0], np.log10(d_qe[:,1]),  kind='linear')
    return 10**i_radeff(lmbda), 10**i_qe(lmbda)

#import matplotlib.pyplot as plt 
#lmbda = np.linspace(250, 940, 1000)
#rad_sens, quant_eff = streak_sens(lmbda, 'hamamatsu', 'S20_2')
#plt.semilogy(lmbda, rad_sens, 'r')
#plt.semilogy(lmbda, quant_eff, 'k--')
#plt.grid()
#plt.show()

#rad_sens, quant_eff = streak_sens(np.array([420, 532]), 'hamamatsu', 'S20_2')

#print 'rad_sens', rad_sens[0]/rad_sens[1]
#print 'quant_eff', quant_eff[0]/quant_eff[1]


