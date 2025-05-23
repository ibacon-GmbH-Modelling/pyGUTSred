import time
import pandas as pd
import numpy as np
import scipy as sp
import models
import parspace
import matplotlib.pyplot as plt

from copy import deepcopy
import multiprocessing as mp

# return the number of available physical cores
import psutil
n_cores = psutil.cpu_count(logical=False)

def readfile(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    startp =0
    stopp =len(lines)
    startc=len(lines)
    concunits = ""
    survivald = list()
    concd = list()
    for i in range(0,len(lines)):
        if "Survival time" in lines[i]:
            startp=i
        if "Concentration unit" in lines[i]:
            concunits = lines[i].split(":",1)[1].replace('\n','').strip()
            stopp = i
        if "Concentration time" in lines[i]:
            startc = i
        if (i>startp and i<stopp):
            survivald.append(lines[i].split())
        if i>startc and i<len(lines):
            concd.append(lines[i].split())
    
    survdata=pd.DataFrame(survivald)
    concdata=pd.DataFrame(concd)
    print("survival data:")
    print(survdata)
    print("exposure data:")
    print(concdata)
    
    survdata = survdata.apply(pd.to_numeric, errors='coerce')
    concdata = concdata.apply(pd.to_numeric, errors='coerce')
    return((concclass(np.array(concdata),filepath,concunits), dataclass(np.array(survdata))))

def readprofile(filepath, units=''):
    # this function assumes that the file is
    # simply 2 columns, one with the time
    # and the other with the concentration
    # the user can provide optionally the
    # units
    # TODO: make sure that the profile has the same units as the
    #       model calibration
    table = pd.read_csv(filepath,  sep='\s+', header =None)
    table = table.apply(pd.to_numeric, errors='coerce')
    return(concclass(np.array(table.astype(float)), filepath, units))

def lcx_calculation(model, timepoints=[2,4,10,21], levels=[0.1,0.2,0.5], propagationset=None, plot=False, concunits="", savefig=False, figname='', extension='.png'):
    # the calculation of LCx values assumes always that the 
    # exposure is constant
    def survfrac(conc,timevector,modelpars,level):
        return(models.calc_surv_sd_const(timevector,conc,modelpars)[-1] - (1-level))

    modelpars = np.copy(10**model.parvals*model.islog + model.parvals*(1-model.islog))
    modelpars[-model.ndatasets:] = 0 # for the LCx values, bkg mortality is 0
    # print(modelpars)
    LCx = np.zeros((len(timepoints),len(levels)))
    LCxlo = np.zeros((len(timepoints),len(levels)))
    LCxup = np.zeros((len(timepoints),len(levels)))
    if (propagationset is not None):
        #pars95[model.posfree] = propagationset[j]
        par95 = np.copy(model.parvals)
        par95 = np.expand_dims(par95, axis = 0)
        par95 = np.repeat(par95, len(propagationset), axis=0)
        par95[:,model.posfree] = propagationset
        par95 = 10**par95*model.islog + par95*(1-model.islog)
        par95[:,-model.ndatasets:] = 0 # remove the background mortality
        par95 = par95[:,:4]
    for i in range(len(timepoints)):
        timevectors = np.linspace(0,timepoints[i],model.nbinsperday)
        for j in range(len(levels)):
            lcxmin = np.inf
            lcxmax = 0
            if model.variant == 'IT':
                beta = np.log(39)/np.log(modelpars[1])
                LCx[i,j]=(modelpars[2]/(1.-np.exp(-modelpars[0]*timevectors[-1]))) * (levels[j]/(1.-levels[j]))**(1./beta)
                if (propagationset is not None):
                    for k in par95:
                        beta = np.log(39)/np.log(k[1])
                        lcx = (k[2]/(1-np.exp(-k[0]*timevectors[-1]))) * (levels[j]/(1-levels[j]))**(1/beta)
                        if lcx <= lcxmin:
                           lcxmin = lcx
                        if lcx >= lcxmax:
                            lcxmax = lcx
                    LCxlo[i,j] = lcxmin
                    LCxup[i,j] = lcxmax
            else:
                conclims = np.array([modelpars[2]/10, modelpars[2]])
                crit = 1
                while crit>0:
                    conclims[1] = conclims[1] * 10 # shift lower and upper range by factor of 10
                    crit   = survfrac(conclims[1],timevectors,modelpars,levels[j]) # calculate criterion from upper range  
                LCx[i,j] = sp.optimize.brenth(survfrac,conclims[0],conclims[1],args=(timevectors,modelpars,levels[j]))
                if (propagationset is not None):
                    for k in par95:
                        conclims = np.array([k[2]/10, k[2]])
                        crit = 1
                        while crit>0:
                            conclims[1] = conclims[1] * 10
                            crit   = survfrac(conclims[1],timevectors,k,levels[j]) # calculate criterion from upper range
                        lcx = sp.optimize.brenth(survfrac,conclims[0],conclims[1],args=(timevectors,k,levels[j]))
                        if lcx <= lcxmin:
                            lcxmin = lcx
                        if lcx >= lcxmax:
                            lcxmax = lcx
                    LCxlo[i,j] = lcxmin
                    LCxup[i,j] = lcxmax
    if plot:
        plt.figure()
        for i in range(len(levels)):
            plt.plot(timepoints, LCx[:,i],'o-', label='LC%d'%(round(levels[i]*100)))
            if (propagationset is not None):
                plt.fill_between(timepoints,LCxlo[:,i],LCxup[:,i],alpha = 0.2, zorder=0)
        plt.xlabel("Time [d]", fontsize=12)
        plt.ylabel("Concentration "+concunits, fontsize=12)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()
        if savefig:
            plt.savefig(figname+"_"+model.variant+"_LCx"+extension)
    # printing the results
    print("----------------------------------------------------------------")
    print("LCx values:")
    titlestring = "{:<10}".format("Time [d]")
    for i in range(len(levels)):
        titlestring = titlestring + "LC{:<32}".format(round(levels[i]*100))
    print(titlestring)
    for i in range(len(timepoints)):
        values = "{:<10s}".format(str(timepoints[i]))
        for j in range(len(levels)):
            values = values + "{:<7.3g} ({:<7.3g} - {:<7.3g})       ".format(LCx[i,j], LCxlo[i,j], LCxup[i,j])
        print(values)
    return([LCx, LCxlo, LCxup])

def plot_data_model(fit, datastruct, concstruct, model, propagationset, modellabel='model', add_obspred=True, savefig=False, figname='', extension='.png'):
    """
    Function to plot experimental data, model predictions, and optionally confidence intervals.

    Parameters:
    -----------
    fit : int
        Specifies the type of plot:
        - 0: Plot data only.
        - 1: Plot data and model predictions.
        - 2: Plot data, model predictions, and 95% confidence intervals.
    datastruct : list
        List of data structures, where each element contains survival data and related information for a dataset.
    concstruct : list
        List of concentration structures, where each element contains concentration data and related information for a dataset.
    model : object
        Model object containing methods for calculating damage and survival, as well as model parameters.
    propagationset : list or None
        List of parameter sets for propagating uncertainty to calculate confidence intervals. If None, confidence intervals are not plotted.
    modellabel : str, optional
        Label for the model predictions in the plot legend. Default is 'model'.
    add_obspred : bool, optional
        If True, generates additional plots comparing observed and predicted survival probabilities and deaths per interval. Default is True.
    savefig : bool, optional
        If True, saves the generated plots to files. Default is False.
    figname : str, optional
        Base filename for saving plots. Default is an empty string.
    extension : str, optional
        File extension for saving plots (e.g., '.png', '.jpg'). Default is '.png'.

    Notes:
    ------
    - The function generates subplots for each treatment in the dataset, showing concentration over time and survival probabilities.
    - If `fit > 0`, model predictions are plotted alongside the data.
    - If `fit > 1` and `propagationset` is provided, 95% confidence intervals are calculated and plotted.
    - Additional plots are generated if `add_obspred` is True, comparing observed and predicted values.
    - The function supports saving plots to files if `savefig` is True.
    """
    if fit in [0,1,2]:
        for nd in range(len(datastruct)):
            dataset = datastruct[nd]
            concset = concstruct[nd]
            fig = plt.figure()
            ax = fig.subplots(2,dataset.ntreats)
            cmax = np.max(concset.concmax)
            #nmax = np.max(dataset.survdata[:,1:])
            nmax = 1
            for i in range(dataset.ntreats):
                ax[0,i].fill_between(concset.timetr,concset.concarraytr[i], label='Concentration', color='blue', alpha=0.2)
                ax[0,i].set_ylim([0, cmax*1.1])
                #yvals = dataset.survdata[:,i+1]/dataset.survdata[0,i+1]
                yvals = dataset.survprobstreat[i]
                deltalow = np.maximum(yvals-dataset.lowlimtreat[i],0)
                deltaup = np.maximum(dataset.upplimtreat[i]-yvals,0)
                ax[1,i].errorbar(dataset.timetreat[i],yvals, 
                                 yerr=[deltalow,deltaup], fmt='o',label='Survival')
                ax[1,i].set_xlabel("Time [d]")
                ax[1,i].set_ylim([0, nmax*1.1])
                ax[0,i].set_xticklabels([]) # remove x tick labels from the first row
                if i>0:
                    ax[0,i].set_yticklabels([])
                    ax[1,i].set_yticklabels([])
            ax[0,0].set_ylabel("Concentration [%s]"%concset.concunits)
            ax[1,0].set_ylabel("Survival")
            plt.tight_layout()
            if fit>0:
                modelpars = np.copy(10**model.parvals*model.islog + model.parvals*(1-model.islog))
                modelpars = modelpars[[0,1,2]+[3+nd]]
                #survmodelprob = np.zeros_like(dataset.survprobs)
                survmodelprob = []
                # FIX THIS!!
                if add_obspred:
                    fig2 = plt.figure()
                    ax2 = fig2.subplots(1,2)                        
                    ax2[0].plot([0,1],[0,1], 'k--',lw=0.5,label='')
                    ax2[1].plot([0,1],[0,1], 'k--',lw=0.5,label='')
                    ax2[0].set_xlabel("Observerd survival probability")
                    ax2[0].set_ylabel("Predicted survival probability")
                    ax2[1].set_xlabel("Observed deaths per interval")
                    ax2[1].set_ylabel("Predicted deaths per interval")
                for i in range(dataset.ntreats):
                    nmax = 1#dataset.survdata[0,i+1]
                    damage = model.calc_damage(modelpars[0],dataset.timeext[i], concset.time, 
                                                    concset.concarray[i], concset.concslopes[i],
                                                    concset.concconst[i])
                    survival = model.calc_survival(dataset.timeext[i], concset.concarray[i],
                                                        damage, modelpars,
                                                        concset.concconst[i])
                    survmodelprob.append(survival[dataset.index_commontime[i]])
                    ax[0,i].plot(dataset.timeext[i], damage, label=modellabel,color='k', linestyle='--')
                    ax[1,i].plot(dataset.timeext[i], nmax*survival, label=modellabel)
                    # here needs to be modified with actual names of the treatments
                    ax2[0].plot(dataset.survprobstreat[i],survmodelprob[i], 'o', label = "Treatment %i"%(i)) 
                    ax2[1].plot(dataset.deatharraytreat[i],
                                dataset.survarrtreat[i][0]*np.append(-np.diff(survmodelprob[i]),
                                                                [survmodelprob[i][-1]]),
                                'o',label = '')
                maxdeaths = max(ax2[1].get_xlim()[1],ax2[1].get_ylim()[1])
                ax2[1].plot([0,maxdeaths],[0,maxdeaths], 'k--',lw=0.5,label='')
                ax2[0].legend(loc='lower right')
                if (fit>1) & (propagationset is not None):
                    for i in range(dataset.ntreats):
                        damlines = np.zeros((len(propagationset),len(dataset.timeext[i])))
                        surlines = np.zeros((len(propagationset),len(dataset.timeext[i])))
                        pars95 = np.copy(model.parvals)
                        for j in range(len(propagationset)):
                            pars95[model.posfree] = propagationset[j]
                            pars95 = 10**pars95*model.islog + pars95*(1-model.islog)
                            pars95_nd = pars95[[0,1,2]+[3+nd]]
                            damlines[j,:] = model.calc_damage(pars95_nd[0], dataset.timeext[i], concset.time, 
                                                    concset.concarray[i], concset.concslopes[i],
                                                    concset.concconst[i])
                            surlines[j,:] = model.calc_survival(dataset.timeext[i], concset.concarray[i],
                                                                     damlines[j,:], pars95_nd,
                                                                     concset.concconst[i])
                        damlineup   = damlines.max(axis=0)
                        damlinedown = damlines.min(axis=0)
                        surlineup   = surlines.max(axis=0)
                        surlinedown = surlines.min(axis=0)
                        ax[0,i].fill_between(dataset.timeext[i],damlinedown,damlineup, color='gray', alpha=0.5, label='95% CI')
                        ax[1,i].fill_between(dataset.timeext[i],surlinedown,surlineup, color='gray', alpha=0.5, label='95% CI')
                        ax2[0].errorbar(dataset.survprobstreat[i],
                                        survmodelprob[i],
                                        yerr=[survmodelprob[i]-surlinedown[dataset.index_commontime[i]],
                                              surlineup[dataset.index_commontime[i]]-survmodelprob[i]], fmt='none',
                                              ecolor='k', zorder = 0)
                fig2.suptitle("Dataset %d"%(nd+1))
                fig2.tight_layout()
            fig.suptitle("Dataset %d"%(nd+1))
            fig.tight_layout()
            plt.show()
            if savefig:
                fig.savefig(figname+"_"+model.variant+"_dataset%d"%(nd+1)+extension)
                plt.close()
                if add_obspred:
                    fig2.savefig(figname+"_"+model.variant+"_dataset%d_obs_pred"%(nd+1)+extension)
                    plt.close()
    else:
        print("fit can be only 0 (data only), 1 (data and best fit), or 2 (data, best fit, and confidence interval)")

def EFSA_quality_criteria(datastruct, concstruct, model):    
    """
    Evaluate the quality criteria defined by the EFSA (European Food Safety Authority)
    Scientific Opinion on TKTD modelling (EFSA, 2018) 
    based on the provided data structure, concentration structure, and model.
    This function calculates various statistical metrics such as R², NRMSE, 
    and SPPE to assess the performance of a survival model. It also prints 
    detailed results for each dataset and treatment.
    Parameters:
    -----------
    datastruct : list
        A list of dataset objects, where each dataset contains survival 
        probabilities, survival arrays, and other related data for different 
        treatments.
    concstruct : list
        A list of concentration structure objects, where each object contains 
        concentration arrays, slopes, constants, and other related data for 
        different treatments.
    model : object
        A GUTSmodels object that contains parameters and methods for calculating 
        damage and survival probabilities.
    Returns:
    --------
    results : dict
        A dictionary containing the following keys:
        - 'R2': Coefficient of determination (R²) for the model fit.
        - 'NRMSE': Normalized Root Mean Square Error (NRMSE) for the model fit.
        - 'R2_0': R² including the t=0 point.
        - 'NRMSE_0': NRMSE including the t=0 point.
        - 'SPPE': Survival Probability Prediction Error (SPPE) for each treatment.
    Notes:
    ------
    - The function assumes that the model parameters are either in log scale 
      or linear scale, as indicated by the `islog` attribute of the model.
    - The function prints detailed results for each dataset, including R², 
      NRMSE, and SPPE values for each treatment.
    - SPPE is calculated as the percentage difference between the observed 
      and predicted survival probabilities at the last time point.
    Example:
    --------
    >>> results = EFSA_quality_criteria(datastruct, concstruct, model)
    >>> print(results['R2'])
    >>> print(results['SPPE'])
    """
    for nd in range(len(datastruct)):
        dataset = datastruct[nd]
        concset = concstruct[nd]
        ssq_fit = 0
        ssq_fit0 = 0
        ssq_fitnum = 0
        ssq_fitnum0 = 0
        ssq_tot = 0
        ssq_tot0 = 0
        sppe = np.zeros(concset.ntreats)
        modelpars = np.copy(10**model.parvals*model.islog + model.parvals*(1-model.islog))
        modelpars = modelpars[[0,1,2]+[3+nd]]
        results = dict()
        for i in range(concset.ntreats):
            nmax = dataset.survarray[i,0]
            damage = model.calc_damage(modelpars[0],dataset.timeext[i], concset.time, 
                                            concset.concarray[i], concset.concslopes[i],
                                            concset.concconst[i])
            survival = model.calc_survival(dataset.timeext[i], concset.concarray[i],
                                                damage, modelpars,
                                                concset.concconst[i])
            ssq_fitnum += np.sum((dataset.survarrtreat[i][1:]-nmax*survival[dataset.index_commontime[i][1:]])**2) 
            ssq_fitnum0 += np.sum((dataset.survarrtreat[i]-nmax*survival[dataset.index_commontime[i]])**2) 
            ssq_fit += np.sum((dataset.survprobstreat[i][1:]-survival[dataset.index_commontime[i][1:]])**2)
            ssq_fit0 += np.sum((dataset.survprobstreat[i]-survival[dataset.index_commontime[i]])**2)
            sppe[i] = 100 * (dataset.survprobstreat[i][-1] - survival[dataset.index_commontime[i][-1]])
        flattensurvprob0  = np.concatenate(dataset.survprobstreat).ravel()
        flattensurvprob   = np.concatenate([x[1:] for x in dataset.survprobstreat]).ravel()
        flattensurvarray0 = np.concatenate(dataset.survarrtreat).ravel()
        flattensurvarray  = np.concatenate([x[1:] for x in dataset.survarrtreat]).ravel()
        ssq_tot = np.sum((flattensurvprob-np.mean(flattensurvprob))**2)
        ssq_tot0 = np.sum((flattensurvprob0-np.mean(flattensurvprob0))**2)
        nrmse   = 100 * np.sqrt(ssq_fitnum/(len(flattensurvprob))) / np.mean(flattensurvarray)
        nrmse0   = 100 * np.sqrt(ssq_fitnum0/(len(flattensurvprob0))) / np.mean(flattensurvarray0)
        print("-- Dataset %d ---------------------------------"%(nd+1))
        print("R2: %.4f"%(1-ssq_fit/ssq_tot))
        print("NRMSE(%%): %.4f"%nrmse)
        print("----------------------------------------------")
        print("R2 with t=0 point: %.4f"%(1-ssq_fit0/ssq_tot0))
        print("NRMSE(%%) with t=0 point: %.4f"%nrmse0)
        print("----------------------------------------------")
        print("Survival probability prediction error (SPPE)")
        print("{:<12} {:<10}".format("Treatment", "value"))
        for i in range(concset.ntreats):
            print("{:<12.0f} {:#.3g} %".format(i, sppe[i]))
        print("----------------------------------------------")
        results['R2'] = 1-ssq_fit/ssq_tot
        results['NRMSE'] = nrmse
        results['R2_0'] = 1-ssq_fit0/ssq_tot0
        results['NRMSE_0'] = nrmse0
        results['SPPE'] = sppe
    return(results)

def validate(validationfile, fitmodel, propagationset, hbfix = True, plot = True, savefig=False, figname='', extension='.png'):
        tmp = readfile(validationfile)
        valconc = np.array([])
        valdata = np.array([])
        valconc = np.append(valconc,tmp[0])
        valdata = np.append(valdata,tmp[1])        
        model = deepcopy(fitmodel)
        valdata[0].timeext, valdata[0].index_commontime = model.calc_ext_time(valdata[0])
        model.parvals = model.parvals[:4]
        model.islog = model.islog[:4]
        model.islog[-1] = 0 # always force the 
        model.parbound_lower = model.parbound_lower[:4]
        model.parbound_upper = model.parbound_upper[:4]
        if hbfix:
            res = sp.optimize.minimize(models.hb_fit_ll, 
                                   x0=model.parvals[-1], 
                                   args=(valdata[0].timetreat[0],valdata[0].deatharraytreat[0]),
                                   method='Nelder-Mead',
                                   bounds=[(model.parbound_lower[-1], model.parbound_upper[-1])])
            model.parvals[-1] = res.x
            print("hb fitted to control data: %.4f"%(model.parvals[-1]))
        else:
            model.parvals[-1] = 0
            print("hb fixed to 0. For a fit of the background mortality to the control data use hbfix=True")
        print("Validation of model with %s variant"%model.variant)
        valres = EFSA_quality_criteria(np.array(valdata), np.array(valconc), model)
        if plot:
            if propagationset is None:
                plot_data_model(fit =1,datastruct=valdata,concstruct=valconc,model=model,propagationset=None, savefig=savefig, figname=figname, extension=extension)
            else:
                # This will need to change if I want to validate multiple datasets at the same time
                fillhb = np.zeros((len(propagationset),1))
                fillhb[:] = model.parvals[-1]
                par95 = np.hstack((propagationset[:,model.posfree<3], fillhb))
                plot_data_model(fit=2,datastruct=valdata,concstruct=valconc,model=model,propagationset=par95, savefig=savefig, figname=figname, extension=extension)
        return(valres)

def _find_mfrange(timevec, damage, survtest, parsset):
    # auxiliary function to calculate the range of multiplication
    # factors
    Send1 = models.calc_surv_sd_trapz(timevec, damage, parsset[1:])[-1]
    MFlow = 1
    MFhigh = 1
    Send = Send1
    while (Send > survtest):
        MFhigh = MFhigh * 10.
        Send = models.calc_surv_sd_trapz(timevec, MFhigh*damage, parsset[1:])[-1]
    Send = Send1
    while (Send < survtest):
        MFlow = MFlow / 10.
        Send = models.calc_surv_sd_trapz(timevec, MFlow*damage, parsset[1:])[-1]
    return((MFlow, MFhigh))

def _calculate_damage(args):
    par95_k, tlong, profile_time, profile_concarray, profile_concslopes = args
    return models.damage_linear_calc(par95_k, tlong, profile_time, profile_concarray, profile_concslopes)

def lpx_calculation(profile, fitmodel, propagationset = None, subset=0, lpxvals = [0.1,0.5], srange = [0.05, 0.999], len_srange = 50, plot = True, batch=False, savefig=False, figname="", extension='.png'):
    """
    Calculate LPx values and optionally generate plots of the survival probability
    at the end of the profile as a function of the multiplication factor, and the
    plot of the concentration profile and survival probability along the profile
    for the calculated LPx values.
    Parameters:
    -----------
    profile : conclass object
        An object containing the time, concentration, and other profile-related 
        data required for the calculation.
    fitmodel : object
        A GUTSmodels object containing parameters, bounds, and methods for calculating 
        damage and survival.
    propagationset : ndarray, optional
        A set of parameter propagations for calculation of 95% CI. Default is None.
    subset : int, optional
        Number of random element of the propagationset to use for the calculation.
        Default is 0, which means all elements are used. Useful for quick checks.
    lpxvals : list of float, optional
        List of LPx values (e.g., [0.1, 0.5]) to calculate. Default is [0.1, 0.5] 
        corresponding to LP10 and LP50.
    srange : list of float, optional
        Range of survival probabilities for the calculation (e.g., [0.05, 0.999]). 
        Default is [0.05, 0.999].
    len_srange : int, optional
        Number of points to sample within the survival probability range. 
        Default is 50.
    plot : bool, optional
        Whether to generate plots for the results. Default is True.
    batch : bool, optional
        Whether the function is running in batch mode (it closes the plots
        right after creation). 
        Default is False.
    savefig : bool, optional
        Whether to save the generated plots to files. Default is False.
    figname : str, optional
        Base name for saving the figures if `savefig` is True. Default is an empty string.
    extension : str, optional
        File extension for saving the figures (e.g., '.png', '.pdf'). Default is '.png'.
    Returns:
    --------
    ndarray
        A 2D array of LPx values with shape (len(lpxvals), 3), where each row contains 
        the LPx value, lower limit, and upper limit. If `len(lpxvals) == 1`, a flattened 
        array is returned.
    Notes:
    ------
    - The function supports two model variants: "IT" (Individual Tolerance) and "SD" 
      (Stochastic Death). The calculations differ based on the model variant.
    - If `propagationset` is provided, uncertainty bounds are calculated for LPx values.
    - The function generates two types of plots:
        1. Survival probability vs. multiplication factor.
        2. Concentration profiles, damage, and survival probabilities for the required
           LPx values.
    - If `batch` is True, plots are closed after saving (if `savefig` is True).
    - The function assumes there is only one treatment in the original profile file.
    """
    def survfrac(MF, tvals, Dvals, pars, level):
        return(models.calc_surv_sd_trapz(tvals, MF*Dvals,pars)[-1] - (1-level))
    # impose 0 background mortality
    print("Calculation of LPx values.")
    print("Fixing background mortality to 0.")
    model = deepcopy(fitmodel)
    model.parvals = model.parvals[:4]
    model.islog = model.islog[:4]
    model.islog[-1] = 0 
    model.parbound_lower = model.parbound_lower[:4]
    model.parbound_upper = model.parbound_upper[:4]
    model.parvals[-1] = 0
    modelpars = np.copy(10**model.parvals*model.islog + model.parvals*(1-model.islog))

    tlong = np.linspace(profile.time[0], profile.time[-1],int(profile.time[-1] * model.nbinsperday+1))
    tlong = np.append(profile.time,tlong)  # to make sure we are not skipping datapoints            
    tlong = np.unique(tlong)
    
    # create variables to store the results
    srangevec = np.linspace(srange[1],srange[0],len_srange)
    # these are needed in the IT case
    LPxpl = np.zeros(len_srange)
    LPxpllo = np.zeros(len_srange)
    LPxplup = np.zeros(len_srange)
    # these are needed in the SD case
    Survpl = np.zeros(len_srange)
    Survpllo = np.zeros(len_srange)
    Survplup = np.zeros(len_srange)
    LPx = np.zeros((len(lpxvals),3)) # len(lpxvals) rows and 3 columns (val, lowlim, uplim)
    # here we operate under the assumption that there is only one treatment in the profile.
    # so the array indices are set 0, not to change the concstruct class
    # calculating damage without any multiplication factor
    # calculations for MF = 1
    damage1 = model.calc_damage(modelpars[0],tlong, profile.time, 
                                profile.concarray[0], profile.concslopes[0],
                                profile.concconst[0])
    survival1 = model.calc_survival(tlong, profile.concarray[0],
                                    damage1, modelpars,
                                    profile.concconst[0])
    if propagationset is not None:
        if subset > 0:
            if subset > len(propagationset):
                print("Subset is larger than the propagationset. Using the whole set.")
                subset = len(propagationset)
            propagationset = np.copy(propagationset)
            np.random.shuffle(propagationset)
            propagationset = propagationset[:subset,:]
        par95 = np.copy(model.parvals)
        par95 = np.expand_dims(par95, axis = 0)
        par95 = np.repeat(par95, len(propagationset), axis=0)
        par95[:,model.posfree] = propagationset
        par95 = 10**par95*model.islog + par95*(1-model.islog)
        par95[:,-model.ndatasets:] = 0 # remove the background mortality
        par95 = par95[:,:4] # keep only 4 parameters
        Mdamk = np.zeros(len(propagationset))
        betak = np.log(39)/np.log(par95[:,1])
        damk = np.zeros((len(propagationset),len(tlong))) # to store precomuped damage
        survk= np.zeros((len(propagationset),len(tlong))) # temp vector for survival
        print("Precomputing damage vector for all the propagation sets.")
        print("Depending on how finely sampled the profile is, this could take a while.")
        with mp.Pool(n_cores) as pool:
            damk = pool.map(_calculate_damage, [(par95[k][0], tlong, profile.time, profile.concarray[0], profile.concslopes[0]) for k in range(len(par95))])
        damk = np.array(damk)
        Mdamk = np.max(damk, axis=1)
    if model.variant == "IT":
        maxDw = max(damage1)
        beta = np.log(39)/np.log(modelpars[1])
        for i in range(len_srange):
            Feff = 1-srangevec[i]
            LPxpl[i] = (modelpars[2]/maxDw) * (Feff/(1-Feff))**(1/beta)
            if propagationset is not None:
                lpxmin = np.inf
                lpxmax = 0
                for k in range(len(par95)):
                    lpx = (par95[k][2]/Mdamk[k]) * (Feff/(1-Feff))**(1/betak[k])
                    if lpx <= lpxmin:
                        lpxmin = lpx
                    if lpx >= lpxmax:
                        lpxmax = lpx
                LPxpllo[i] = lpxmin
                LPxplup[i] = lpxmax
        for i in range(len(lpxvals)):
            Feff = lpxvals[i]
            LPx[i,0] = (modelpars[2]/maxDw) * (Feff/(1-Feff))**(1/beta)
            if propagationset is not None:
                lpxmin = np.inf
                lpxmax = 0
                for k in range(len(par95)):
                    lpx = (par95[k][2]/Mdamk[k]) * (Feff/(1-Feff))**(1/betak[k])
                    if lpx <= lpxmin:
                        lpxmin = lpx
                    if lpx >= lpxmax:
                        lpxmax = lpx
                LPx[i,1] = lpxmin
                LPx[i,2] = lpxmax
    else: # SD variant, no need to specify            
        # implementation with auxiliary function
        mf1, mf2 = _find_mfrange(tlong, damage1, srange[0], modelpars)
        rootsmall = sp.optimize.brenth(survfrac,mf1,mf2,args=(tlong, damage1, modelpars[1:], 1-srange[0]))
        mf1, mf2 = _find_mfrange(tlong, damage1, srange[1], modelpars)
        rootlarge = sp.optimize.brenth(survfrac,mf1, mf2,args=(tlong, damage1, modelpars[1:], 1-srange[1]))
        # mfvec = np.logspace(np.log10(rootlarge),np.log10(rootsmall), len_srange)
        mfvec = np.logspace(np.log10(rootlarge*0.8),np.log10(rootsmall*1.2), len_srange) # extend a little the range in case the drop in survival is too sharp
        ## calculating S vs MF 
        for i in range(len_srange):
            Survpl[i] = models.calc_surv_sd_trapz(tlong, mfvec[i]*damage1, modelpars[1:])[-1]
            if propagationset is not None:
                survmin = 1.
                survmax = 0.
                for k in range(len(par95)):
                    surv = models.calc_surv_sd_trapz(tlong, mfvec[i]*damk[k,:], par95[k][1:])[-1]
                    if surv<=survmin:
                        survmin = surv
                    if surv>=survmax:
                        survmax = surv
                Survpllo[i] = survmin
                Survplup[i] = survmax
        ## done calculating S vs MF
        for j in range(len(lpxvals)):
            ind_lo = np.argwhere(Survpl < (1-lpxvals[j])).flatten().max()
            ind_hi = np.argwhere(Survpl > (1-lpxvals[j])).flatten().min()
            LPx[j,0] = sp.optimize.brenth(survfrac,mfvec[ind_lo],mfvec[ind_hi],args=(tlong, damage1, modelpars[1:], lpxvals[j]))
            if propagationset is not None:
                lpxmin = np.inf
                lpxmax = 0
                for k in range(len(par95)):
                    mflow,mfhigh = _find_mfrange(tlong, damk[k,:], 1-lpxvals[j], par95[k])                
                    lpxk = sp.optimize.brenth(survfrac,mflow,mfhigh,args=(tlong, damk[k,:], par95[k][1:], lpxvals[j]))
                    if lpxk<=lpxmin:
                        lpxmin = lpxk
                    if lpxk>=lpxmax:
                        lpxmax = lpxk
                LPx[j,1] = lpxmin
                LPx[j,2] = lpxmax
    # printing the results
    print("----------------------------------------------------------------")
    print("LPx values:")
    for j in range(len(lpxvals)):
        values = "LP{:<3}:  {:<7.3g} ({:<7.3g} - {:<7.3g})       ".format(round(lpxvals[j]*100),LPx[j,0], LPx[j,1], LPx[j,2])
        print(values)
    if plot:
        fig=plt.figure()
        ax = fig.subplots(1,1)
        if model.variant == 'IT':
            plt.plot(LPxpl,srangevec,'k-')
            plt.fill_betweenx(srangevec, LPxpllo, LPxplup, alpha=0.2)
        else:
            plt.plot(mfvec,Survpl,'k-')
            plt.fill_between(mfvec, Survpllo, Survplup, alpha=0.2)
        plt.hlines(1-np.array(lpxvals), xmin=ax.get_xlim()[0],xmax=ax.get_xlim()[1],
                   colors='grey', linestyles='--', linewidth=0.5)
        plt.xlabel("Multiplication factor")
        plt.ylabel("Survival probability")
        # Add new plot with profile, profile multiplied, damage, and surv. prop.
        fig2 = plt.figure()
        ax2 = fig2.subplots(2,len(lpxvals)+1) # +1 becuase I am also plotting MF=1
        ax2[0,0].fill_between(profile.timetr, profile.concarraytr[0], label = "Conc",color='blue', alpha=0.2)
        ax2[0,0].plot(tlong, damage1, 'k--')
        ax2[0,0].set_ylim([0,max(profile.concarraytr[0])*max(LPx[:,0])*1.1])
        ax2[0,0].set_title("MF = 1")
        ax2[1,0].plot(tlong, survival1,'k-')
        ax2[1,0].set_ylim([0,1.1])
        ax2[0,0].set_ylabel("Concentration ["+profile.concunits+"]")
        ax2[1,0].set_ylabel("Survival probability")
        ax2[1,0].set_xlabel("Time [d]")
        if propagationset is not None:
            damup = np.max(damk, axis=0)
            damlo = np.min(damk, axis=0)
            ax2[0,0].fill_between(tlong, damlo,damup, color = 'k',alpha= 0.2)
        for i in range(len(lpxvals)):
            ax2[0,i+1].set_ylim([0,max(profile.concarraytr[0])*max(LPx[:,0])*1.1])
            ax2[0,i+1].fill_between(profile.timetr, LPx[i,0]*profile.concarraytr[0], label = "Conc",color='blue', alpha=0.2)
            ax2[0,i+1].plot(tlong, LPx[i,0]*damage1, 'k--')
            ax2[0,i+1].set_title("MF = %.2f"%LPx[i,0])
            ax2[0,i+1].set_yticklabels([])
            ax2[0,i+1].set_xticklabels([])
            surv = model.calc_survival(tlong, 0, LPx[i,0]*damage1, modelpars, 0)
            ax2[1,i+1].plot(tlong, surv, 'k-')
            ax2[1,i+1].set_yticklabels([])
            ax2[1,i+1].set_ylim([0,1.1])
            ax2[1,i+1].set_xlabel("Time [d]")
            if propagationset is not None:
                ax2[0,i+1].fill_between(tlong, LPx[i,0]*damlo,LPx[i,0]*damup, color = 'k',alpha= 0.2)
                for k in range(len(par95)):
                    survk[k,:] = model.calc_survival(tlong, 0, LPx[i,0]*damk[k,:], par95[k], 0)
                survmin = np.min(survk, axis=0)
                survmax = np.max(survk, axis=0)
                ax2[1,i+1].fill_between(tlong,survmin,survmax, color = 'k',alpha= 0.2)
        plt.tight_layout()
    if batch:
        plt.close(fig)
        plt.close(fig2)
        if savefig:
            basename = figname
            nametosave1 = basename +'_mf_surv'+extension
            nametosave2 = basename +'_lpx_full'+extension
            fig.savefig(nametosave1)
            fig2.savefig(nametosave2)
    else:
        plt.show()
        if savefig:
            basename = figname
            nametosave1 = basename +'_mf_surv'+extension
            nametosave2 = basename +'_lpx_full'+extension
            fig.savefig(nametosave1)
            fig2.savefig(nametosave2)
    if len(lpxvals)==1:
        return(LPx.flatten())
    else:
        return(LPx)


class concclass:
    """
    A class to handle concentration data, pre-calculate quantities needed for 
    the GUTS model fits, and plotting of exposure data.
    Attributes:
        name (str): The name or origin of the data.
        concdata (numpy.ndarray): The input concentration data array. The first column represents 
            time, and the subsequent columns represent concentration values for different treatments.
        ntreats (int): The number of treatments.
        timetr (numpy.ndarray): Unmodified time vector extracted from the input data, used for plotting.
        time (numpy.ndarray): The unique time values extracted from the input data.
        concarraytr (numpy.ndarray): Array of concentrations at different time points for every treatment.
        concslopestr (numpy.ndarray): Array to store the slopes of concentration changes over time 
            for each treatment.
        conctwa (numpy.ndarray): Array to store the time-weighted average concentration for each treatment.
        concconst (numpy.ndarray): Array to indicate if a treatment has constant concentration (1) or not (0).
        concmax (numpy.ndarray): Array to store the maximum concentration for each treatment.
        concunits (str): The units of the concentration data.
        concslopes (numpy.ndarray): Array of slopes of concentration changes over time, reshaped 
            to match the number of treatments and unique time points.
        concarray (numpy.ndarray): Array of concentration data, reshaped to match the number of 
            treatments and unique time points.
    Methods:
        __init__(concdata, name, concunits):
            Initializes the concclass object, processes the input data, interpolates missing values, 
            calculates slopes, time-weighted averages, and other attributes.
        plot_exposure(savefig=False, figname='', extension='.png'):
            Plots the exposure data (concentration vs. time) for each treatment. Optionally saves 
            the plot to a file.
            Args:
                savefig (bool): Whether to save the figure to a file. Default is False.
                figname (str): The base name of the file to save the figure. Default is an empty string.
                extension (str): The file extension for the saved figure. Default is '.png'.
    """
    def __init__(self,concdata,name,concunits):
        self.name = name # to store the origin of the data
        self.concdata = concdata
        self.ntreats = concdata.shape[1] - 1
        self.timetr = concdata[:,0] # needed only for plotting
        self.time = concdata[:,0]
        self.concarraytr = np.transpose(concdata[:,1:])
        self.concslopestr = np.zeros_like(self.concarraytr)
        self.conctwa = np.zeros(self.ntreats)
        # array to store if a treatment has constant concentration or not
        self.concconst = np.zeros(self.ntreats) 
        self.concmax = np.zeros(self.ntreats)
        self.concunits = concunits
        # all the following is to account for all the cases in which the data is not complete
        # in presence of NaNs the values areinterpolated beteween the
        # closest non-NaN values. If only one values is given, then the concentration is
        # assumed constant
        for i in range(self.ntreats):
            nans, x = np.isnan(self.concarraytr[i]), lambda z: z.nonzero()[0]
            if np.sum(~nans) == 1:
                self.concarraytr[i][nans] = self.concarraytr[i][~nans][0]
            else:
                for nan_idx in np.where(nans)[0]:
                    if nan_idx == 0:
                        self.concarraytr[i][nan_idx] = np.nan
                    elif nan_idx == len(self.time) - 1:
                        self.concarraytr[i][nan_idx] = self.concarraytr[i][nan_idx - 1]
                    else:
                        prev_idx = nan_idx - 1
                        next_idx = nan_idx + 1
                        while next_idx < len(self.time) and np.isnan(self.concarraytr[i][next_idx]):
                            next_idx += 1
                        if next_idx < len(self.time):
                            self.concarraytr[i][nan_idx] = np.interp(self.time[nan_idx], [self.time[prev_idx], self.time[next_idx]], [self.concarraytr[i][prev_idx], self.concarraytr[i][next_idx]])
                        else:
                            self.concarraytr[i][nan_idx] = np.nan
            self.concslopestr[i,:-1] = np.diff(self.concarraytr[i])/np.diff(self.time)   
            self.conctwa[i] = np.trapz(self.concarraytr[i],self.time)/self.time[-1]
            self.concmax[i] = np.max(self.concarraytr[i])
            if (np.all(self.concslopestr[i])==0) & (len(np.unique(self.concarraytr[i]))<2):
                self.concconst[i] = 1
        self.time = np.unique(self.time)
        tmpslopes = self.concslopestr[np.isfinite(self.concslopestr)]
        tmparray = self.concarraytr[np.isfinite(self.concslopestr)]
        self.concslopes = tmpslopes.reshape((self.ntreats,len(self.time)))
        self.concarray = tmparray.reshape((self.ntreats,len(self.time)))

    def plot_exposure(self, savefig=False, figname='', extension='.png'):
        fig = plt.figure()
        ax = fig.subplots(1,self.ntreats)
        cmax = np.max(self.concmax)
        if self.ntreats==1:
            ax.fill_between(self.timetr,self.concarraytr[0], label='Concentration', color='blue', alpha=0.2)
            ax.set_ylim([0, cmax*1.1])
            ax.set_ylabel("Concentration [%s]"%self.concunits)
            ax.set_xlabel("Time [d]")
        else:
            for i in range(self.ntreats):
                ax[i].fill_between(self.timetr,self.concarraytr[i], label='Concentration', color='blue', alpha=0.2)
                ax[i].set_ylim([0, cmax*1.1])
                ax[i].set_xlabel("Time [d]")
            ax[0].set_ylabel("Concentration [%s]"%self.concunits)
        plt.tight_layout()
        plt.show()
        if savefig:
            fig.savefig(figname+"_"+self.name+"_conc"+extension)


class dataclass:
    """
    A class to process and store survival data for multiple treatments.
    Attributes:
        survdata (numpy.ndarray): Input survival data matrix where the first column 
            represents time and subsequent columns represent survival counts for 
            different treatments.
        ntreats (int): Number of treatments (columns in survdata excluding the time column).
        time (numpy.ndarray): Array of time points extracted from the first column of survdata.
        survarray (numpy.ndarray): Array of survival data (one row for each treatment).
        survprobs (numpy.ndarray): Array to store survival probabilities for each treatment.
        timetreat (list): List of time arrays for each treatment, excluding missing data.
        survarrtreat (list): List of survival arrays for each treatment, excluding missing data.
        deatharraytreat (list): List of death counts for each treatment.
        survprobstreat (list): List of survival probabilities for each treatment.
        lowlimtreat (list): List of lower confidence limits for each treatment.
        upplimtreat (list): List of upper confidence limits for each treatment.
    Methods:
        __init__(survdata):
            Initializes the dataclass object, processes the input survival data, 
            and calculates survival probabilities and confidence intervals for 
            each treatment. Handles missing data in the survival matrix.
    """
    def __init__(self,survdata):
        self.survdata = survdata
        self.ntreats = survdata.shape[1] - 1
        self.time = survdata[:,0]
        self.survarray = np.transpose(survdata[:,1:])
        self.survprobs = np.zeros((self.ntreats,len(self.time)))
        # this is needed to handle possibility of missing data in the
        # survival matrix
        self.timetreat = []
        self.survarrtreat = []
        self.deatharraytreat = []
        self.survprobstreat = []
        self.lowlimtreat = []
        self.upplimtreat = []
        z= 1.96
        for i in range(self.ntreats):
            tmpsurv = self.survarray[i, np.isnan(self.survarray[i])==False]
            tmptime = self.time[np.isnan(self.survarray[i])==False]
            self.survarrtreat.append(tmpsurv)
            self.timetreat.append(tmptime)
            self.deatharraytreat.append(np.append( -(np.diff(tmpsurv[:]).astype('float')), tmpsurv[-1]) )
            ninit = survdata[0,i+1] # time 0 in principle should never have a nan value
            tmpprob = tmpsurv/ninit
            self.survprobstreat.append(tmpprob)
            # Wilson score interval on data probabilities. From openGUTS code
            # https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Wilson_score_interval
            a = (tmpprob + z**2/(2*ninit))/(1+z**2/ninit)
            b = z/(1+z**2/ninit) * np.sqrt(tmpprob*(1-tmpprob)/ninit + z**2/(4*ninit**2))
            a[0]=1
            b[0]=0
            tmplowlim = np.maximum(0,a-b)
            tmpupplim = np.minimum(1,a+b)
            self.lowlimtreat.append(tmplowlim)
            self.upplimtreat.append(tmpupplim)
            
# this class inherits from PyParspace and extends the functionalities
class pyGUTSred(parspace.PyParspace):
    def __init__(self,
                 datafile,
                 variant,
                 hbfree = True,
                 preset=True,parvalues=None,lbound=None,ubound=None,islog=None,isfree=None,
                 profile=True,
                 rough=False):
        self.variant = variant
        self.hbfree = hbfree
        self.calibpath = datafile
        self.ndatasets = len(datafile)
        self.fullset = []
        self.bestaic = np.inf
        print("number of datasets")
        print(self.ndatasets)
        # initialize empty arrays to store the data
        self.concstruct = np.array([])
        self.datastruct = np.array([])
        if type(datafile) != list:
            raise ValueError("Please, provide the calibration file as a list")
        for i in range(self.ndatasets):
            tmp = readfile(datafile[i])
            #self.concstruct.append(tmp[0])
            #self.datastruct.append(tmp[1]) 
            self.concstruct = np.append(self.concstruct, tmp[0])
            self.datastruct = np.append(self.datastruct, tmp[1])
            if self.concstruct[i].time[-1] < self.datastruct[i].time[-1]:
                self.concstruct[i].time = np.append(self.concstruct[i].time, self.datastruct[i].time[-1])
                self.concstruct[i].concarray = np.append(self.concstruct[i].concarray,
                                                        np.transpose([self.concstruct[i].concarray[:,-1]+self.concstruct[i].concslopes[:,-1]*(self.datastruct[i].time[-1]-self.concstruct[i].time[-2])]),
                                                        axis=1)
        unitslist = [x.concunits for x in self.concstruct]
        if len(np.unique(unitslist))>1:
            print("Warining, the concentration units in the datafiles are different. Hope you know what you are doing.")
            print("The code will proceed with the units reported in the first file")
        self.concunits = self.concstruct[0].concunits   # add an error or a warning in case the units are different
        if variant == 'SD':
            self.parnames = ["kd","bs","zs"]
        else:
            self.parnames = ["kd","Fs","zs"]
        for i in range(self.ndatasets):
            self.parnames = self.parnames+["hb%d"%(i+1)]
        if preset:
            self._preset_pars()
        else:
            # in this case the user needs to insert the values
            self.parvals = parvalues
            self.lbound = lbound
            self.ubound = ubound
            self.islog = islog
            self.isfree = isfree
            if len(parvalues) != 4 or len(lbound) != 4 or len(ubound) != 4 or len(islog) != 4 or len(isfree) != 4:
                raise ValueError("need to set all the various parameters")
        if self.hbfree == False:
            print("fit hb to control data")
            self.fit_hb()    
        self.model = models.GUTSmodels(self.datastruct,self.concstruct,self.variant,
                                       self.parnames,self.parvals,self.islog,self.isfree,
                                       self.lbound,self.ubound)
        # make sure that the time vectors will belong to the data structure as well
        for i in range(self.ndatasets):
            self.datastruct[i].timeext = self.model.timeext[i]
            self.datastruct[i].index_commontime = self.model.index_commontime[i]
        print("precompile the functions")
        self.model.log_likelihood(self.parvals[self.model.posfree],self.parvals,self.model.posfree)
        print("setup the parameter space explorer")
        self.parspacesetup = parspace.SettingParspace(rough=rough,profile=profile)
        super().__init__(self.parspacesetup,self.model)
        self.plot_data_model(fit=0)

    def _preset_pars(self):
        self.isfree = np.concatenate(([1,1,1],[0]*self.ndatasets)) # last positions are for the hb values
        self.islog = np.concatenate(([1,1,0],[0]*self.ndatasets))
        self.lbound = np.zeros(3+self.ndatasets)
        self.ubound = np.zeros(3+self.ndatasets)
        if self.hbfree:
            self.isfree[-self.ndatasets:] = 1
        self.lbound[-self.ndatasets:] = 1e-6
        self.ubound[-self.ndatasets:] = 0.07

        tmptime = []
        tmpconcm = []
        tmpconcM = []
        for i in range(self.ndatasets):
            tmptime.append(self.datastruct[i].time[self.datastruct[i].time>0])
            tmpconcm.append(self.concstruct[i].conctwa[self.concstruct[i].conctwa>0])
            tmpconcM.append(self.concstruct[i].concarray[self.concstruct[i].concarray>0])
        tmptime = np.concatenate(tmptime).ravel()
        tmpconcm = np.concatenate(tmpconcm).ravel()
        tmpconcM = np.concatenate(tmpconcM).ravel()
        tmax = np.max(tmptime) #np.max(self.datastruct[:].time)
        tmin = np.min(tmptime)
        cmax = np.max(tmpconcM)
        cmin = np.min(tmpconcm)
        # limits for kd
        self.lbound[0] = min([np.log(20)/(5*365),-np.log(1-0.05)/tmax])
        self.ubound[0] = max([np.log(20)/(0.5/24) , -np.log(1-0.99)/(0.1*tmin)])
        self.lbound[2] = cmin*(1-np.exp(-self.lbound[0]*(4./24.)))
        if self.variant == 'SD':
            # limits for bs
            self.lbound[1] = -np.log(0.9)/(cmax*tmax)
            self.ubound[1] = (24**2*0.95)/(self.lbound[0]*cmax*np.exp(-self.lbound[0]*tmax*0.5))
             # upper limits for zs
            self.ubound[2] = 0.99*cmax
        else:
            # limits for Fs
            self.lbound[1] = 1.05
            self.ubound[1] = 20
             # upper limits for zs
            self.ubound[2] = 2*cmax
        
        self.lbound = np.log10(self.lbound)*self.islog+self.lbound*(1-self.islog)
        self.ubound = np.log10(self.ubound)*self.islog+self.ubound*(1-self.islog)    
        self.parvals = (self.lbound+self.ubound)/2
        print("Parameter settings:")
        print("parnames: ",self.parnames)
        print("parameters lower bounds: ",self.lbound)
        print("parameters upper bounds: ",self.ubound)
        print("parameters are log-transformed: ",self.islog)
        print("parameters are free: ",self.isfree)

    def fit_hb(self):
        for i in range(self.ndatasets):
            res = sp.optimize.minimize(models.hb_fit_ll, 
                                   x0=self.parvals[2+(i+1)], 
                                   args=(self.datastruct[i].timetreat[0],self.datastruct[i].deatharraytreat[0]),
                                   method='Nelder-Mead',
                                   bounds=[(self.lbound[2+(i+1)], self.ubound[2+(i+1)])])
            self.parvals[2+(i+1)] = res.x
            print("hb fitted to control data for dataset %d: %.4f"%(i+1,self.parvals[2+(i+1)]))

    def run_and_time_parspace(self):
        # wrapper around the parameter space explorer so that
        # we can include meausures in case there is slow kinetic
        start = time.time()
        out = self.run_parspace()
        if out[0]==-1:
            print("slow kinetic was detected")
            print("threshold parameter will be explored in logarithmic scale")
            self.model.islog[2] = 1
            self.islog[2] = 1 # 
            self.parlabels[2] = "log10(%s)"%self.parlabels[2]
            # update boundary for threshold
            # self.model.parbound_upper[2] = min(np.log10(self.model.parbound_upper[2]),np.log10(out[2][self.model.posfree==2]*self.opts.slowkin_f_mw))
            self.model.parbound_upper[2] = np.log10(self.model.parbound_upper[2])
            self.model.parbound_lower[2] = np.log10(self.model.parbound_lower[2])
            # update boundary for kd
            # self.model.parbound_upper[0] = min(self.model.parbound_upper[0],out[2][self.model.posfree==0]*self.opts.slowkin_f_kd)
            out = self.run_parspace()
        stop = time.time()
        print("Elapsed time for the parameter space exploration: %.4f"%(stop-start))

    def plot_data_model(self,fit,modellabel='model', add_obspred=True, savefig=False, figname='', extension='.png'):
        plot_data_model(fit=fit, datastruct=self.datastruct, concstruct=self.concstruct, model=self.model,
                        propagationset = self.propagationset, modellabel=modellabel, add_obspred=add_obspred,
                        savefig=savefig, figname=figname, extension=extension)	

    def EFSA_quality_criteria(self):
        efsares = EFSA_quality_criteria(self.datastruct, self.concstruct, self.model)
        return(efsares)

    def lcx_calculation(self, timepoints=[2,4,10,21],levels=[0.1,0.2,0.5], plot=False, propagationset=None, savefig=False, figname='', extension='.png'):
        lcxvals = lcx_calculation(self.model, timepoints=timepoints, levels=levels, propagationset=propagationset, plot=plot, concunits=self.concunits,
                        savefig=savefig, figname=figname, extension=extension)
        return(lcxvals)

