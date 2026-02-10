'''
Class and functions for the equations of the GUTS reduced model.

This implementation is heavily based on the BYOM and openGUTS 
code written by Dr. Tjalling Jager. The original code is written in 
Matlab and can be found here: https://www.debtox.info/byom.html

The original MATLAB files are located in the engine folder of openGUTS 
or in the engine and engine_par folder of the BYOM release.

The rewriting of this software made use of the GitHub Copilot, which is based
on AI technology.

=======================================================================
Copyright (c) 2018-2024, Tjalling Jager (tjalling@debtox.nl)
Copyright (c) 2025-2026, Carlo Romoli - ibacon GmbH (carlo.romoli AT ibacon.com)
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; If not, see <https://www.gnu.org/licenses/>

3rd Party Addendum
=========================================================================
This program uses the python packages numpy, scipy
which are available under the modified 3 clause BSD license

Copyright (c) 2005-2024, NumPy Developers.
Copyright (c) 2001, 2002 Enthought, Inc.
Copyright (c) 2003-2019 SciPy Developers.


All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following
       disclaimer in the documentation and/or other materials provided
       with the distribution.

    * Neither the name of the NumPy Developers nor the names of any
       contributors may be used to endorse or promote products derived
       from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
=======================================================================
'''

import numpy as np
from numba import jit
import scipy.integrate as spi

@jit(nopython=True)
def damage_calculation(y,t,C,timextr,kd):
    # this could be made similar to what BYOM does without 
    # having to solve the ODE
    Cval = np.interp(t, timextr,C)
    dydt = 0
    dydt = kd*(Cval-y[0])
    return(dydt)

@jit(nopython=True)
def damage_linear_calc(kd, timeext, timeconc, conc, slopes):
    '''
    Calculate the damage variable using a linear approximation of 
    the concentration. This is the same as the openGUTS code.
    The function is precompiled using numba.

    Arguments:
    kd -- damage repair/elimination rate
    timeext -- extended time vector
    timeconc -- time vector of the concentration data
    conc -- vector of concentration values
    slopes -- vector of slopes of the concentration data
    '''
    damage = np.zeros_like(timeext)
    Dw0 =0
    damage[0] = Dw0
    # run through all events (not last one, which marks the end of the scenario, to which we extrapolate with the slope)
    for i in range(len(timeconc)-1):
        ind_t   = (timeext>timeconc[i]) & (timeext<=timeconc[i+1]) # find the logical indices for the period between two events
        te      = timeext[ind_t] - timeconc[i]            # find the new part of the time vector, and make it start at zero again
        c_start = conc[i]                       # take as initial concentration the new one in this period
        slope   = slopes[i]                       # take as slope the new one in this period
        # See openGUTS code for explanation
        # original openGUTS comment:     % Thanks to Bob Kooi for using Maple to obtain the solution for me.   
        damage[ind_t] = slope * te + c_start - slope / kd + np.exp(-kd * te) * (Dw0 - c_start + slope / kd)
        t_end = timeconc[i+1] - timeconc[i] # last time point in this interval is start time for new period
        Dw0   = slope * t_end + c_start - slope / kd + np.exp(-kd * t_end) * (Dw0 - c_start + slope / kd)
    return(damage)

@jit(nopython=True)
def calc_damage_const(t,C,kd):
    # analytical solution for constant concentrations
    damage = C*(1-np.exp(-kd*t))
    return(damage)

@jit(nopython=True)
def calc_survival_SD(y,t,D,pars):
    # integrand for the differential equation
    # for the SD variant of the GUTS model
    bw,zw,hb=pars
    dydt = 0
    dydt = -(bw*max(D-zw,0)+hb)*y
    return(dydt) 

#@jit(nopython=1)
def calc_surv_sd_trapz(tvals, Dvals,pars):
    bw,zw,hb=pars
    hz = bw*(np.maximum(Dvals-zw,0))
    hzcum = spi.cumulative_trapezoid(hz, tvals, initial=0)
    Sc = np.minimum(1,np.exp(-hzcum))
    Sc = Sc * np.exp(-hb*tvals)
    return(Sc)

@jit(nopython=True)
def calc_surv_sd_const(tvals,Cvals,pars):
    # analytical solution for constant concentrations
    kd,bw,zw,hb=pars
    minH = np.zeros_like(tvals)
    Sc=0
    # finsish here
    if (Cvals>zw):
        t0   = -np.log(1-zw/Cvals)/kd
        minH = (bw/kd)*np.maximum(0,np.exp(-kd*t0)-np.exp(-kd*tvals))*Cvals - bw*(np.maximum(0,Cvals-zw))*np.maximum(0,tvals-t0); # replace <minH> by the calculated effects over time (vector)
    Sc = np.minimum(1,np.exp(minH))
    Sc = Sc * np.exp(-hb*tvals)
    return(Sc)

@jit(nopython=True)
def guts_sdmodel(y,t,C,timextr,pars):
    Cval = np.interp(t, timextr,C)
    kd,bw,zw,hb=pars
    dydt = np.zeros(2)
    dydt[0] = kd*(Cval-y[0])
    dydt[1] = -(bw*max(y[0]-zw,0)+hb)*y[1]
    return(dydt)
    
@jit(nopython=True)
def guts_itmodel(tvals, Dvals,pars):
    Fs,mw,hb=pars
    beta = np.log(39)/np.log(Fs)
    maxDw = np.copy(Dvals)
    diffs = maxDw[1:]-maxDw[:-1]
    temparr =np.append(0,diffs)
    ind = np.argwhere(temparr<0)
    while ind.size>0:
        ind = ind[0][0]
        maxDw[ind:] = np.maximum(maxDw[ind:],maxDw[ind-1])
        newdiffs = maxDw[1:] - maxDw[:-1]
        temparr =np.append(0,newdiffs)
        ind = np.argwhere(temparr<0)        
    Sc = 1. / (1.+(maxDw/mw)**beta)
    Sc = Sc * np.exp(-hb*tvals)
    return(Sc)

@jit(nopython=True)
def hb_fit_ll(hb, tvals, deathcontroldata):
    survmodelcontrol = np.exp(-hb*tvals)
    pdeath = np.append(-np.diff(survmodelcontrol),survmodelcontrol[-1])
    pdeath = np.maximum(pdeath,1e-50)
    llik=np.dot(deathcontroldata,np.log(pdeath))
    return(-llik)

@jit(nopython=True)
def loglikelihood(modelvector, commontime, deathvector):
    surviv_selected = modelvector[commontime]
    #print(surviv_selected)
    pdeath = np.append(-np.diff(surviv_selected),surviv_selected[-1])
    pdeath = np.maximum(pdeath,1e-50)
    #print(deathvector)
    llik=np.dot(deathvector,np.log(pdeath))
    return(-llik)


class GUTSmodels:
    '''
    Class that contains the functions that are used to calculate the likelihood
    of the GUTS model.

    Attributes:
    - variant: string that specifies the variant of the GUTS model (SD or IT)
    - ndatasets: number of datasets
    - concstruct: list of concentration data structures
    - datastruct: list of survival data structures
    - nbinsperday: number of bins per day
    - timeext: extended time vector
    - index_commontime: indices of the common time points between the concentration and survival data
    - parnames: list of parameter names
    - parvals: list of parameter values
    - islog: list of booleans that specify if the parameter is log-transformed
    - isfree: list of booleans that specify if the parameter is free
    - posfree: indices of the free parameters
    - parbound_lower: list of lower bounds for the parameters
    - parbound_upper: list of upper bounds for the parameters

    Methods:
    - calc_ext_time: calculate the extended time vector and the indices of the common time points with the 
                     original survival data	
    - calc_damage: calculate the damage variable
    - calc_survival: calculate the survival probability
    - log_likelihood: calculate the log-likelihood of the GUTS model
    '''
    def __init__(self, survstruct, concstruct, variant,
                 parnames,
                 parvals,islog, isfree, 
                 parbound_lower, parbound_upper,
                 nbinsperday=96):
        '''
        Constructor for the GUTSmodels class. This class contains the
        functions that are used to calculate the likelihood of the GUTS
        model. The class is initialized with the following arguments:

        Arguments:
          - survstruct: list of survival data structures (length depends on the number of datasets)
          - concstruct: list of concentration data structures (length depends on the number of datasets)
          - variant: string that specifies the variant of the GUTS model (SD or IT)
            - parnames: list of parameter names
            - parvals: list of parameter values
            - islog: list of booleans that specify if the parameter is log-transformed
            - isfree: list of booleans that specify if the parameter is free
            - parbound_lower: list of lower bounds for the parameters
            - parbound_upper: list of upper bounds for the parameters
            - nbinsperday: number of bins per day (default is 96 as in openGUTS)
        '''
        self.variant = variant
        self.ndatasets = len(survstruct)
        self.concstruct = concstruct
        self.datastruct = survstruct
        self.nbinsperday = nbinsperday
        self.timeext = []
        self.index_commontime = [] 
        for i in range(self.ndatasets):
            timeext, indexcommon = self.calc_ext_time(self.datastruct[i])
            self.timeext.append(timeext)
            self.index_commontime.append(indexcommon)
        # attributes that deal with the model parameters
        self.parnames = np.array(parnames,dtype=object)   # make sure these are numpy arrays
        self.parvals = np.array(parvals)
        self.islog = np.array(islog)                   # make sure these are numpy arrays	
        self.isfree = np.array(isfree)                 # make sure these are numpy arrays
        self.posfree = np.argwhere(self.isfree == 1).flatten()  # positions of the free parameters in the parameter vector
        self.parbound_lower = np.array(parbound_lower) # make sure these are numpy arrays
        self.parbound_upper = np.array(parbound_upper) # make sure these are numpy arrays

    def calc_ext_time(self, datastruct):
        '''
        Calculate the extended time vector and the indices of the common time points with the
        original survival data.

        Argument:
        - datastruct : datastruct object
            survival data structure
        '''
        timeexttmp = []
        index_commontime = []
        for i in range(datastruct.ntreats):
            timeexttmptreat = np.linspace(0, datastruct.timetreat[i][-1],
                                 self.nbinsperday*int(datastruct.timetreat[i][-1]))
            timeexttmptreat = np.append(datastruct.timetreat[i],timeexttmptreat)  # to make sure we are not skipping datapoints            
            timeexttmptreat = np.unique(timeexttmptreat)
            index_commontimetreat = np.intersect1d(timeexttmptreat,datastruct.timetreat[i],
                                          return_indices=True,assume_unique=True)[1]
            timeexttmp.append(timeexttmptreat)
            index_commontime.append(index_commontimetreat)
        return timeexttmp, index_commontime
        

    def calc_damage(self, kd, timeext, conctime, concdata, concslopes, constc):
        '''
        Calculate the damage variable using the concentration data. The function
        can use a linear approximation of the concentration data or the analytical
        solution for constant concentrations.

        Arguments:
        - kd: damage repair/elimination rate
        - timeext: extended time vector
        - conctime: time vector of the concentration data
        - concdata: vector of concentration values
        - concslopes: vector of slopes of the concentration data
        - constc: boolean that specifies if the concentration is constant
        '''
        if constc:
            damage = calc_damage_const(timeext, concdata[0], kd)
        else:
            damage = damage_linear_calc(kd, timeext, conctime,
                                        concdata, concslopes)
        return(damage)

    def calc_survival(self, timeext, concdata, damage, pars, consc):
        '''
        Calculate the survival probability for SD or IT variant (defined internally in the class).
        The function can use the analytical solution for constant concentrations or the numerical
        solution (for the SD case, the numerical solution uses the trapezium integration rule
        valid here because the time step is small compared to the dynamic of the concentration).

        Arguments:
        - timeext: extended time vector
        - concdata: vector of concentration values
        - damage: vector of damage values
        - pars: vector of parameter values
        - consc: boolean that specifies if the concentration is constant
        '''
        if self.variant == 'SD':
            if consc:
                # if constant concentration, use analytical solution
                survival = calc_surv_sd_const(timeext,
                                              concdata[0],
                                              pars)
            else:
                # if not constant concentration, use numerical solution with trapezium rule
                survival = calc_surv_sd_trapz(timeext, damage, pars[1:])
        else:
            survival = guts_itmodel(timeext, damage, pars[1:])
        return(survival)

    def log_likelihood(self, theta, allpars, posfree):
        '''
        Calculate the log-likelihood of the GUTS model.

        Arguments:
        - theta: vector of free parameter values
        - allpars: vector of all parameter values
        - posfree: indices of the free parameters in the parameter vector
        '''
        allpars[posfree] = theta
        # TODO: make sure that for each dataset the respective hb value is correctly passed
        modelpars = 10**allpars*self.islog + allpars*(1-self.islog)
        llik = 0
        nd=0
        while nd < self.ndatasets: # this could be run in parallel (or directly back in the parspace explorer)
            # the first three parameters are the same for all datasets, the fourth is the hb value, potentially different for each dataset
            modelpars_nd = np.concatenate((modelpars[:3],[modelpars[3+nd]])) 
            i =0
            while i < self.concstruct[nd].ntreats:
                # for each treatment, calculate the damage and survival probability
                # the class method already accounts for the model variant
                # and whether the concentration is constant or not
                damage = self.calc_damage(modelpars_nd[0],self.timeext[nd][i], self.concstruct[nd].time, 
                                          self.concstruct[nd].concarray[i], self.concstruct[nd].concslopes[i],
                                          self.concstruct[nd].concconst[i])
                surviv = self.calc_survival(self.timeext[nd][i], self.concstruct[nd].concarray[i],
                                            damage, modelpars_nd,
                                            self.concstruct[nd].concconst[i])         
                llik += loglikelihood(surviv,self.index_commontime[nd][i],
                                      self.datastruct[nd].deatharraytreat[i])
                i+=1
            nd+=1
        return(llik)