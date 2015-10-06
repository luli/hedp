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
        elif pulse_profile == 'gaussian':
            self._instance_call= lambda x: self.gaussian_pulse(x, I0,  **args)
        else:
            raise NotImplementedError


    def square_pulse(self, P_time, I0, dt=1.0e-9, t0=0.2e-9, rise_time=30e-12):
        P_pattern = np.ones(P_time.shape)
        mask = P_time < t0
        P_pattern[mask] = np.exp(-(P_time[mask] - t0)**2/(2*(rise_time)**2))
        mask = P_time > dt + t0
        P_pattern[mask] = 0
        return P_pattern*I0


    def gaussian_pulse(self, P_time, I0, dt=1.0e-9, t0=1.0e-9):
        P_pattern = np.exp(-(P_time - t0)**2/(2*(dt)**2))
        return P_pattern*I0


    def __call__(self, x):
        return self._instance_call(x)


class LaserBeams(object):
    def __init__(self, pars, N=500, t_end=1.7e-9, gaussian_target_ratio=2.0):
        """
        Initalize the Laser Beam

        Warning: This is a partial implementation that assumes that
        a laser pulse is associated to a single laser beam.
        
        Parameters:
        -----------
          - pars: a MegredDict object with the parameters of the simulation
          - N :  number of points in the laser pulse profile
          - t_end: duration of the ray-tracing
          - gaussian_target_ratio: the size of the domain to map for the laser ray tracing.
                     By default, use 2x the size of ed_gaussianRadiusMajor
        """
        self.p = pars
        self.t_end = t_end
        self.gaussian_target_ratio = gaussian_target_ratio
        self.P_time = np.linspace(0, t_end, N)
        self._get_beams_surface()
        self.gridnRadialTics = []
        self.numberOfRays = []


    def set_pulses(self, pulses):
        self.Intensity = [el.I0 for el in pulses]
        self.dt = [el.dt for el in pulses]
        self.pulse_profile = [el.profile for el in pulses]
        self.P_pattern = [el(self.P_time) for el in pulses]
        if self.p['NDIM'] == 1:
            self.P_power = [P_pattern for  P_pattern in self.P_pattern]
        elif self.p['NDIM'] == 2:
            self.P_power = [S0*P_pattern for P_pattern, S0 in zip(self.P_pattern, self.beam_surface)]
        else:
            raise ValueError
        self.Energy = [ np.trapz(P_power,  self.P_time) for P_power in self.P_power]


    def _get_beams_surface(self):
        self.beam_surface = []
        self.num_bream = len(self.p['ed_crossSectionFunctionType'])
        if self.p['NDIM'] == 1:
            return
        self.targetSemiAxis = []
        for idx, cross_section in enumerate(self.p['ed_crossSectionFunctionType']):
            if cross_section == 'gaussian2D':
                S0 = Int_super_gaussian(self.p['ed_gaussianRadiusMajor'][idx], self.p['ed_gaussianExponent'][idx])
                self.targetSemiAxis.append(self.p['ed_gaussianRadiusMajor'][idx]*self.gaussian_target_ratio)
            else:
                raise NotImplementedError
            self.beam_surface.append(S0)


    def get_pars(self):
        if self.p['NDIM'] == 1:
            out = {'ed_power': self.P_power,
                   'ed_time': [self.P_time]*self.num_bream,
                   'ed_numberOfSections': [len(self.P_time)]*self.num_bream,
                   'ed_pulseNumber': range(self.num_bream),
                   'ed_numberOfBeams': self.num_bream,
                   'ed_numberOfPulses': self.num_bream,
                   'ed_numberOfRays': [1]*self.num_bream,
                   'ed_pulseNumber': range(1, self.num_bream+1) # this is very restrictive and needs to be extended
                  } 
        elif self.p['NDIM'] == 2:
            out = {'ed_power': self.P_power,
                   'ed_time': [self.P_time]*self.num_bream,
                   'ed_numberOfSections': [len(self.P_time)]*self.num_bream,
                   'ed_pulseNumber': range(self.num_bream),
                   'ed_numberOfBeams': self.num_bream,
                   'ed_numberOfPulses': self.num_bream,
                   'ed_targetSemiAxisMajor': self.targetSemiAxis,
                   'ed_targetSemiAxisMinor': self.targetSemiAxis,
                   'ed_gridnRadialTics': self.gridnRadialTics,
                   'ed_numberOfRays': self.numberOfRays,
                   'ed_pulseNumber': range(1, self.num_bream+1) # this is very restrictive and needs to be extended
                  } 
        else:
            raise NotImplementedError
        return out


    def adapt_ray_tracing(self, dx, rays_per_cell=4, radial_ticks_to_rays_factor=8):
        """
        This assumes 3D in 2D ray tracing 
        """
        if self.p['NDIM'] == 1:
            self.numberOfRays = [1]*self.num_bream
        elif self.p['NDIM'] == 2:

            for idx, cross_section in enumerate(self.p['ed_crossSectionFunctionType']):
                if cross_section == 'gaussian2D':
                    self.gridnRadialTics = np.asarray([ rays_per_cell*beam_size/dx for beam_size in self.targetSemiAxis], dtype=np.int)
                    self.numberOfRays = self.gridnRadialTics*int(radial_ticks_to_rays_factor)
                else:
                    raise NotImplementedError


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
                   "numberOfRays"
                   ]

        dataset = [np.asarray(self.Intensity),
                   self.Energy,
                   self.p['ed_wavelength'],
                   self.pulse_profile,
                   np.asarray(self.dt)*1e9,
                   self.p['ed_crossSectionFunctionType'],
                   self.numberOfRays
                   ]


        entry_format = ['{:>10.2e}',
                        '{:>10.2f}',
                        '{:>10.3f}',
                        '{:>10}',
                        '{:>10.3f}',
                        '{:>10}',
                        '{:>10.0f}',
                        ]
        if self.p['NDIM'] == 2:
            dataset.append( np.asarray(self.p['ed_gaussianRadiusMajor'])*1e4,)
            dataset.append( self.p['ed_gaussianExponent'],)
            dataset.append(       self.gridnRadialTics,)
            labels.append( 'FWHM [um]')
            labels.append('SG gamma',)
            labels.append('nRadialTicks')
            entry_format.append( '{:>10.1f}',)
            entry_format.append('{:>10.2f}',)
            entry_format.append('{:>10.0f}')


        out = ['', '='*80, ' '*26 + 'Laser parameters', '='*80]

        row_format_labels ="{:18}" + "{:>10}" * self.num_bream

        beam_labels = ['Beam {}'.format(idx) for idx in range(self.num_bream)]
        out.append(row_format_labels.format('', *beam_labels))

        for label, value, fmt in zip(labels, dataset, entry_format):
            row_format ="{:18}" + fmt*self.num_bream
            #if not isinstance(value, (int, long, float)):
            #    value = [value]
            try:
                out.append( row_format.format(label, *value))
            except:
                #out.append( row_format.format(label, value))

                out.append(('Formatting error: {} {} {}'.format(label, value, fmt)))


        out += ['='*80, '']

        return '\n'.join(out)




