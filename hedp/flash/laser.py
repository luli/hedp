#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright CNRS 2012
# Roman Yurchak (LULI)
# This software is governed by the CeCILL-B license under French law and
# abiding by the rules of distribution of free software.
import numpy as np
from ..math.integrals import Int_super_gaussian


class LaserPulse(object):

    def __init__(self, pulse_profile, I0, **args):
        """
        Initialize the laser pulse
        """
        self.I0 = I0
        self.dt = args['dt']
        self.profile = pulse_profile
        if pulse_profile == 'square':
            self._instance_call= lambda x: self.square_pulse(x, I0,  **args)
        else:
            raise NotImplementedError


    def square_pulse(self, P_time, I0, dt=1.0e-9, t0=0.2e-9, rise_time=30e-12):
        P_pattern = np.ones(P_time.shape)
        mask = P_time < t0
        P_pattern[mask] = np.exp(-(P_time[mask] - t0)**2/(2*(rise_time)**2))
        mask = P_time > dt + t0
        P_pattern[mask] = 0
        return P_pattern*I0


    def __call__(self, x):
        return self._instance_call(x)


class LaserBeams(object):
    def __init__(self, pars, N=500, t_end=1.7e-9):
        """
        Initalize the Laser Beam

        Warning: This is a partial implementation that assumes that
        a laser pulse is associated to a single laser beam.
        
        Parameters:
        -----------
          - pars: a MegredDict object with the parameters of the simulation
          - N :  number of points in the laser pulse profile
          - t_end: duration of the ray-tracing
        """
        self.p = pars
        self.t_end = t_end
        self.P_time = np.linspace(0, t_end, N)
        self._get_beams_surface()

    def set_pulses(self, *pulses):
        self.Intensity = [el.I0 for el in pulses]
        self.dt = [el.dt for el in pulses]
        self.pulse_profile = [el.profile for el in pulses]
        self.P_pattern = [el(self.P_time) for el in pulses]
        self.P_power = [S0*P_pattern for P_pattern, S0 in zip(self.P_pattern, self.beam_surface)]
        self.Energy = [ np.trapz(P_power,  self.P_time) for P_power in self.P_power]


    def _get_beams_surface(self):
        self.beam_surface = []
        self.num_bream = len(self.p['ed_crossSectionFunctionType'])
        for idx, cross_section in enumerate(self.p['ed_crossSectionFunctionType']):
            if cross_section == 'gaussian2D':
                S0 = Int_super_gaussian(self.p['ed_gaussianRadiusMajor'][idx], self.p['ed_gaussianExponent'][idx])
            else:
                raise NotImplementedError
            self.beam_surface.append(S0)


    def get_pars(self):
        out = {'ed_power': self.P_power,
               'ed_time': [self.P_time]*self.num_bream,
               'ed_numberOfSections': [len(self.P_time)]*self.num_bream,
               'ed_pulseNumber': range(self.num_bream),
               'ed_numberOfBeams': self.num_bream,
               'ed_numberOfPulses': self.num_bream}

        return out

    def __repr__(self):
        """
        Pretty print the laser parameters
        """

        labels  = ['Intensity [W/cm2]',
                   'Energy [J]',
                   'Wavelenght [um]',
                   'Pulse profile',
                   'Duration [ns]',
                   'Cross section',
                   'FWHM [um]',
                   'SG gamma']

        dataset = [np.asarray(self.Intensity),
                   self.Energy,
                   self.p['ed_wavelength'],
                   self.pulse_profile,
                   np.asarray(self.dt)*1e9,
                   self.p['ed_crossSectionFunctionType'],
                   np.asarray(self.p['ed_gaussianRadiusMajor'])*1e4,
                   self.p['ed_gaussianExponent'],
                   ]


        entry_format = ['{:>10.2e}',
                        '{:>10.2f}',
                        '{:>10.3f}',
                        '{:>10}',
                        '{:>10.3f}',
                        '{:>10}',
                        '{:>10.1f}',
                        '{:>10.2f}',
                        ]


        out = ['', '='*80, ' '*26 + 'Laser parameters', '='*80]

        row_format_labels ="{:18}" + "{:>10}" * self.num_bream

        beam_labels = ['Beam {}'.format(idx) for idx in range(self.num_bream)]
        out.append(row_format_labels.format('', *beam_labels))

        for label, value, fmt in zip(labels, dataset, entry_format):
            row_format ="{:18}" + fmt*self.num_bream
            out.append(
                   row_format.format(label, *value))


        out += ['='*80, '']

        return '\n'.join(out)




