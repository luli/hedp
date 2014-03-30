#!/usr/bin/python
# -*- coding: utf-8 -*-
import tables
import numpy as np
from scipy.spatial import distance_matrix
import re

class BaseProjGrid:

    def set_pars(self, method, **args):
        """
        Set paramaters
        """
        if method not in self.method_list:
            raise ValueError('Method {0}: not implemented! Should be one of {1}.'.format(
                                                                method, str(self.method_list)))
        self.method = method
        self.func = self.method_list[method]
        self.args = args
        self._compute_interpolator()


    @staticmethod
    def _method_df1(nu, op, alpha=1.0, max_val=1.0):
        """
        Cost function: Use normalized first order derivative

        Parameters:
        -----------
            - nu [ndarray]: photon group boundaries
            - op [ndarray]: spectra
            - alpha [float]: in [0.0, 2.0], default 1.0
                        alpha < 1: low sensitivity to gradients in the spectra
                        alpha > 1: high sensitivity to gradients in the spectra
        """
        err = np.abs(np.gradient(op)/op)**alpha
        if max_val < 1.0:
            err_max = np.percentile(err, max_val*100)
            err[err>err_max] = err_max
        return err/err.sum()

    @staticmethod
    def _method_df2(nu, op, alpha=1.0, beta=0.5, nmin_filter=10, log=True):
        """
        Cost function: Use normalized first order derivative

        Parameters:
        -----------
            - nu [ndarray]: photon group boundaries
            - op [ndarray]: spectra
            - alpha [float]: in [0.0, 2.0], default 1.0
                        alpha < 1: low sensitivity to gradients in the spectra
                        alpha > 1: high sensitivity to gradients in the spectra
        """
        from scipy.ndimage.filters import minimum_filter1d
        if log:
            err = np.abs(np.gradient(np.gradient(np.log10(op))))
        else:
            err = np.abs(np.gradient(np.gradient(op)))
        err /= err.max()  # normalising to 1 so pow gives predictive results
        err = err**alpha
        if not log:
            err /= minimum_filter1d(op, nmin_filter)
        err /= err.sum()
        err_f = (err*(1-beta)+beta/len(op))
        err_f /= err_f.sum()
        return err_f

    @staticmethod
    def repr_groups(groups, nlines=10):
        X = np.empty((4, len(groups)-1))
        dgroup_lin = np.diff(groups)
        dgroup_log = np.power(10, np.diff(np.log10(groups)))
        sorted_idx = np.argsort(dgroup_log)
        out = []
        for idx, sidx in enumerate(sorted_idx):
            out.append("{0: 3}. Group {1: 3} @ {2:5.0f} eV with width {3:8.2f} eV  [{4:5.4f}]".format(
                    idx, sidx, groups[sidx], dgroup_lin[sidx], dgroup_log[sidx]))
            if idx>nlines:
                break
        return '\n'.join(out)


    def project(self, ngroups):
        if ngroups<1:
            raise ValueError('Minimum acceptable number of groups is 1!')
        x = np.linspace(0,1,ngroups+1)
        idx_proj = self.itp(x)
        idx_proj = np.unique(np.rint(idx_proj).astype(dtype='int'))
        if  ngroups!= len(idx_proj)-1:
            print 'Warning: projection returned {0} groups while {1} groups were requested!'.format(
                    len(idx_proj)-1, ngroups)
        return idx_proj, self.groups[1:][idx_proj]

class SpectraProjGrid(BaseProjGrid):
    def __init__(self, nu, op):
        """
        Compute an optimal group boundaries based for one opacity spectra

        Parameters:
        -----------
         - nu [ndarray]: photon group boundaries
         - op [ndarray]: spectra
        """
        self.method = None
        self.func = None
        self.args = None
        self.nu = nu
        self.op = op
        self.method_list = {'df1': self._method_df1, 'df2': self._method_df2,}

    def _compute_cost(self):
        if self.func is None:
            raise ValueError("MGrigProjection.set_pars method has to be called once for initialization!")
        err_cum =  np.cumsum(self.func(self.nu, self.op, **self.args))
        err_cum[0] = 0
        err_cum[-1] = 1.0
        return err_cum

    def _compute_interpolator(self):
        from scipy.interpolate import interp1d
        err_cum = self._compute_cost()
        self.itp = interp1d(err_cum, self.nu)


def project_on_grid(points, grid):
    """
    Project points on a grid

    Parameters:
    -----------
    points : ndarray (N,)
    grid :  ndarray (M,)

    Returns:
    -------
    idx: ndarray (N,)
        grid indices closest to given points
    grid_val: ndarray (N,)
        grid values closest to given points
    """
    d = distance_matrix(np.array([points]).T,np.array([grid]).T)
    idx = np.argmin(d,axis=1)
    return idx, grid[idx]

class SelectPoints:
    def __init__(self, rho0, temp0, weight, regexp):
        """
        Get closest points on the density/temperature grid from an opacity table.

        Parameters:
        -----------
          - rho0 : ndarray or float
                   density g/cm³
          - temp : ndarray or float
                   temperature eV
          - weight : ndarray or float
                   relative weight
          - regexp : str
                   regular expression that applyies to the list of keys
        """
        if type(rho0) is float:
            rho0 = [rho0]
        if type(temp0) is float:
            temp0 = [temp0]

        self.rho0 = rho0 = np.array(rho0)
        self.temp0 = temp0 = np.array(temp0)
        if len(rho0) != len(temp0):
            raise ValueError('rho0 and temp0 should have the same lenght!')

        self.regexp = regexp
        self.weight = weight

    def __call__(self, rho, temp):
        ridx, _ = project_on_grid(self.rho0, rho)
        tidx, _ = project_on_grid(self.temp0, temp)
        N = len(ridx)
        return ridx, tidx, np.ones(N)*self.weight/N

class SelectRect:
    def __init__(self, rho_lim, temp_lim, weight, regexp, Nrho=5, Ntemp=5):
        """
        Select a rectable on the density/temperature grid from an opacity table.

        Parameters:
        -----------
          - rho_lim : tuple
                   density upper and lower limits g/cm³
          - temp_lim :  tuple
                   density upper and lower limits eV
          - weight : ndarray or float
                   relative weight
          - regexp : str
                   regular expression that applyies to the list of keys
          - Nrho : int
                   number of points in density
          - Ntemp: int
                   number of points in 
        """
        if not ( len(rho_lim) == len(temp_lim) == 2):
            raise TypeError('rho_lim and temp_lim should be tuples of len 2!')

        self.rho_lim = rho_lim
        self.temp_lim = temp_lim

        self.Nrho = Nrho - 1
        self.Ntemp = Ntemp - 1

        self.regexp = regexp
        self.weight = weight

    def __call__(self, rho, temp):
        ridx_lim, _ = project_on_grid(np.array(self.rho_lim), rho)
        tidx_lim, _ = project_on_grid(np.array(self.temp_lim), temp)
        Nrho_grid = ridx_lim[1] - ridx_lim[0]
        Ntemp_grid = tidx_lim[1] - tidx_lim[0]
        stride_rho =  min(max(int(Nrho_grid/self.Nrho), 1),  Nrho_grid)
        stride_temp =  min(max(int(Ntemp_grid/self.Ntemp), 1),  Ntemp_grid)
        ridx_1d = np.arange(ridx_lim[0], ridx_lim[1], stride_rho)
        tidx_1d = np.arange(tidx_lim[0], tidx_lim[1], stride_temp)

        Ridx_m, Tidx_m = np.meshgrid(ridx_1d, tidx_1d)
        ridx = Ridx_m.flatten()
        tidx = Tidx_m.flatten()

        return ridx, tidx, np.ones(tidx.size)*self.weight/tidx.size



class TableProjGrid(BaseProjGrid):
    def __init__(self, tables, selectors, groups_lim):
        """
        Compute AMR projection for some opacity tables

        Parameters:
        -----------
        tables: dict
                a dictionaray that has for keys the names of the table
                and that has for values OpgHdf5 objetcs
        selectors: list 
                   containing SelectPoints, SelectRect objects
        groups_lim: typle
                       (min, max) first and last group boundaries in eV
        """

        self.method = None
        self.func = None
        self.method_list = {'df1': self._method_df1, 'df2': self._method_df2,}
        groups_dict = {key: op['groups'][:] for key, op in tables.iteritems()}
        groups_list = groups_dict.values()
        self.groups = groups_ref = groups_list[0]
        if len(groups_list)>1:
            for gidx, group in enumerate(groups_list[1:]):
                if not np.array_equal(groups_ref, group):
                    raise ValueError("Opactity tables {0} and {1} don't have the same photon grid!".format(tables.keys()[0], tables.keys()[gidx+1]))
        if len(groups_lim)!=2:
            raise ValueError


        groups_lim_idx, val = project_on_grid(np.array(groups_lim), groups_ref)
        self.groups_slice = slice(groups_lim_idx[0], groups_lim_idx[1])
        self.Ng_ini = len(groups_ref) - 1
        self.Ng = groups_lim_idx[1] - groups_lim_idx[0]

        self.t_op = tables
        self.selectors = selectors
        self.t_selectors = t_selectors = {key: [] for key in tables}
        self.t_mask = t_mask = {}
        self.t_weight = t_weight = {}
        self.t_cost = {key: np.zeros((op.Nr, op.Nt), dtype='object') for key, op in tables.iteritems()}
        self.t_cost_wsum = {key: np.zeros(self.Ng) for key, op in tables.iteritems()}
        self.cost_fn = np.zeros(self.Ng)


        # setting appropriate selectors for every table
        for key, op in tables.iteritems():
            for sel in selectors:
                if re.match(sel.regexp, key):
                    t_selectors[key].append(sel)
        for key in tables:
            if not len(t_selectors[key]):
                raise ValueError('No selectors were applied to the table {0}! \n\
            Please change the selectors argument of TableProjGrid.'.format(key))

        # getting a mask for every point that should be taken
        for key, op in tables.iteritems():
            Nr, Nt = len(op['dens']), len(op['temp'])
            t_mask[key] = np.zeros((Nr,Nt), dtype='bool')
            t_weight[key] =  np.zeros((Nr, Nt))
            for sel in t_selectors[key]:
                ridx, tidx, weights = sel(op['dens'], op['temp'])
                t_mask[key][ridx, tidx] = True
                t_weight[key][ridx, tidx] += weights

    def _compute_cost(self, groups, op):
        """ Compute cost """
        err_cum =  np.cumsum(self.func(groups[self.groups_slice], op[self.groups_slice], **self.args))
        err_cum[0] = 0
        err_cum[-1] = 1.0
        return err_cum

    def _compute_interpolator(self):
        from scipy.interpolate import interp1d
        if self.func is None:
            raise ValueError("TableProjGrid.set_pars method has to be called once for initialization!")
        norm_weight = 0
        for key, op in self.t_op.iteritems():
            ridx_arr, tidx_arr =  np.nonzero(self.t_mask[key])
            norm_weight += self.t_weight[key].sum()
            for idx in range(len(ridx_arr)):
                ridx, tidx = ridx_arr[idx], tidx_arr[idx]
                self.t_cost[key][ridx, tidx] = self._compute_cost(self.groups, op['opp_mg'][ridx, tidx])
                self.t_cost_wsum[key] += self.t_cost[key][ridx, tidx]*self.t_weight[key][ridx, tidx]
            self.cost_fn += self.t_cost_wsum[key]
        self.cost_fn /= norm_weight
        self.itp = interp1d(self.cost_fn, np.arange(self.groups_slice.start, self.groups_slice.stop))


    def plot_selection(self, ax):
        from itertools import cycle 
        colors_list =  cycle(['r', 'b', 'k', 'g', 'orange', 'navy', 'gray', 'brown'])
        for key in self.t_op:
            ridx_arr, tidx_arr =  np.nonzero(self.t_mask[key])
            print self.t_weight[key][ridx_arr, tidx_arr]/self.t_weight[key].max()
            ax.scatter(self.t_op[key]['dens'][ridx_arr], self.t_op[key]['temp'][tidx_arr],
                    alpha=0.7, label=key, c=colors_list.next(),
                    s=200*self.t_weight[key][ridx_arr, tidx_arr]/self.t_weight[key].max(),
                    linewidths=0)
        ax.legend(loc='best')

        ax.set_xscale('log')
        ax.set_yscale('log')


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from opacplot2.opg_hdf5 import OpgHdf5
    filename_Ar = '/home/luli/EoS_tables/Ar/Ar_prp_prp/Ar_prp_10kgr.h5'
    filename_Al = '/home/luli/EoS_tables/Al/Al_prp_prp/Al_prp_10kgr.h5'
    op_Al = OpgHdf5.open_file(filename_Al)
    op_Ar = OpgHdf5.open_file(filename_Ar)


    tg = TableProjGrid( tables={'Al': op_Al, 'Ar': op_Ar },
            selectors=[#SelectPoints([1e-2], [1], weight=0.5, regexp='Al'),
                       SelectPoints([1e-2], [100], weight=0.5, regexp='.*'),
                       SelectPoints([2.7], [0.025], weight=0.1, regexp='Al'),
                       SelectRect((1e-4, 1e-2), (1, 30), weight=0.5, regexp='.*')],
            groups_lim=(1, 20e3))
    tg.set_pars('df2', beta=0.3, nmin_filter=10, alpha=1, log=True)


    idx = 0



    fig = plt.figure()
    ax = [plt.subplot(211), plt.subplot(212)]

    for key in tg.t_op:
        ridx_arr, tidx_arr =  np.nonzero(tg.t_mask[key])
        ridx, tidx = ridx_arr[idx], tidx_arr[idx]
        op = tg.t_op[key]
        
        
        ax[0].loglog(tg.groups[tg.groups_slice], op['opp_mg'][ridx, tidx][tg.groups_slice])
    #print op['opp_mg'][9,:,-1]
    ax[0].set_title('Opacity @ {0:.2e} g/cc, {1:.3f} eV'.format(op['dens'][ridx], op['temp'][tidx]))


    ax[1].semilogx(tg.groups[tg.groups_slice],tg.cost_fn)  
                                              #tg.t_cost_wsum[key])
    ax[0].vlines(tg.project(100)[1], 1, 1e7, alpha=0.4)
    #print mgp.repr_groups(mgp.project(32))
    #ax[1].set_xlabel('Photon energy [eV]')
    #ax[1].set_ylim(0,1)
    for axi in ax:
        axi.set_xlim(1,20e3)
    fig.savefig('test_optimal_grid.pdf', bbox_inches='tight')

    fig =  plt.figure(figsize=(6,4))
    ax = plt.subplot(111)
    tg.plot_selection(ax)
    fig.savefig('test_optimal_grid_selection.pdf', bbox_inches='tight')
