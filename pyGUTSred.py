import time
import pandas as pd
import numpy as np
import scipy as sp
import models
import parspace
import matplotlib.pyplot as plt

class concclass:
    def __init__(self,concdata):
        self.concdata = concdata
        self.ntreats = concdata.shape[1] - 1
        self.time = concdata[:,0]
        self.concarray = np.transpose(concdata[:,1:])
        self.concslopes = np.zeros_like(self.concarray)
        self.conctwa = np.zeros(self.ntreats)
        # array to store if a treatment has constant concentration or not
        self.concconst = np.zeros(self.ntreats) 
        self.concmax = np.zeros(self.ntreats)
        for i in range(self.ntreats):
            self.concslopes[i,:-1] = np.diff(self.concarray[i])/np.diff(self.time)
            self.conctwa[i] = np.trapz(self.concarray[i],self.time)/self.time[-1]
            self.concmax[i] = np.max(self.concarray[i])
            if (np.all(self.concslopes[i])==0) & (len(np.unique(self.concarray[i]))<2):
                self.concconst[i] = 1


class dataclass:
    def __init__(self,survdata):
        self.survdata = survdata
        self.ntreats = survdata.shape[1] - 1
        self.time = survdata[:,0]
        self.survarray = np.transpose(survdata[:,1:])
        self.deatharray = np.zeros((self.ntreats,len(self.time)))
        self.survprobs = np.zeros((self.ntreats,len(self.time)))
        self.lowlim = np.zeros((self.ntreats,len(self.time)))
        self.upplim = np.zeros((self.ntreats,len(self.time)))
        z= 1.96
        for i in range(self.ntreats):
            self.deatharray[i,:len(self.time)-1] = np.array(-np.diff(survdata[:,i+1]))
            self.deatharray[i,-1] = survdata[-1,i+1]
            ninit = survdata[0,i+1]
            # Wilson score interval on data probabilities.
            # https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Wilson_score_interval
            survprop = survdata[:,i+1]/ninit
            self.survprobs[i,:] = survprop
            a = (survprop + z**2/(2*ninit))/(1+z**2/ninit)
            b = z/(1+z**2/ninit) * np.sqrt(survprop*(1-survprop)/ninit + z**2/(4*ninit**2))
            a[0]=1
            b[0]=0
            self.lowlim[i,:] = np.maximum(0,a-b)
            self.upplim[i,:] = np.minimum(1,a+b)
            
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
        print("number of datasets")
        print(self.ndatasets)
        # initialize empty arrays to store the data
        self.concstruct = np.array([])
        self.datastruct = np.array([])
        if type(datafile) != list:
            raise ValueError("Please, provide the calibration file as a list")
        for i in range(self.ndatasets):
            tmp = self._readfile(datafile[i])
            #self.concstruct.append(tmp[0])
            #self.datastruct.append(tmp[1]) 
            self.concstruct = np.append(self.concstruct, tmp[0])
            self.datastruct = np.append(self.datastruct, tmp[1])
            if self.concstruct[i].time[-1] < self.datastruct[i].time[-1]:
                self.concstruct[i].time = np.append(self.concstruct[i].time, self.datastruct[i].time[-1])
                self.concstruct[i].concarray = np.append(self.concstruct[i].concarray,
                                                        np.transpose([self.concstruct[i].concarray[:,-1]+self.concstruct[i].concslopes[:,-1]*(self.datastruct[i].time[-1]-self.concstruct[i].time[-2])]),
                                                        axis=1)
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
            self.parvalues = parvalues
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
                                       self.parnames,self.parvalues,self.islog,self.isfree,
                                       self.lbound,self.ubound)
        # make sure that the time vectors will belong to the data structure as well
        for i in range(self.ndatasets):
            self.datastruct[i].timeext = self.model.timeext[i]
            self.datastruct[i].index_commontime = self.model.index_commontime[i]
        print("precompile the functions")
        self.model.log_likelihood(self.parvalues[self.model.posfree],self.parvalues,self.model.posfree)
        print("setup the parameter space explorer")
        self.parspacesetup = parspace.SettingParspace(rough=rough,profile=profile)
        super().__init__(self.parspacesetup,self.model)
        self.plot_data_model(fit=0)

    def _readfile(self,filepath):
        with open(filepath, 'r') as f:
            lines = f.readlines()
        startp =0
        stopp =len(lines)
        startc=len(lines)
        self.concunits = ""
        survivald = list()
        concd = list()
        for i in range(0,len(lines)):
            if "Survival time" in lines[i]:
                startp=i
            if "Concentration unit" in lines[i]:
                self.conctunits = lines[i].split(":",1)[1].replace('\n','').strip()
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
        
        survdata=survdata.apply(pd.to_numeric)
        concdata=concdata.apply(pd.to_numeric)
        #self.concstruct=concclass(np.array(concdata))
        #self.datastruct=dataclass(np.array(survdata))
        return((concclass(np.array(concdata)), dataclass(np.array(survdata))))

    def _preset_pars(self):
        #FIX this
        # need to set up the conditions with multiple
        # values for the background mortality depending
        # on the number of datasets
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
        self.parvalues = (self.lbound+self.ubound)/2
        #self.parvalues = np.log10(self.parvalues)*self.islog+self.parvalues*(1-self.islog)
        print("Parameter settings:")
        print("parnames: ",self.parnames)
        print("parameters lower bounds: ",self.lbound)
        print("parameters upper bounds: ",self.ubound)
        print("parameters are log-transformed: ",self.islog)
        print("parameters are free: ",self.isfree)

    # TODO: modify for multiple datasets
    def fit_hb(self):
        for i in range(self.ndatasets):
            res = sp.optimize.minimize(models.hb_fit_ll, 
                                   x0=self.parvalues[2+(i+1)], 
                                   args=(self.datastruct[i].time,self.datastruct[i].deatharray[0]),
                                   method='Nelder-Mead',
                                   bounds=[(self.lbound[2+(i+1)], self.ubound[2+(i+1)])])
            self.parvalues[2+(i+1)] = res.x
            print("hb fitted to control data for dataset %d: %.4f"%(i+1,self.parvalues[2+(i+1)]))

    def run_and_time_parspace(self):
        start = time.time()
        self.run_parspace()
        stop = time.time()
        print("Elapsed time for the parameter space exploration: %.4f"%(stop-start))

    def plot_data_model(self, fit=0, datastruct=None, concstruct=None, modellabel='model', add_obspred=True):
        '''
        Function to plot data and/or model

        Arguments:
        ----------
            fit = int
                0 for data only, 1 for data and model, 2 for data, model,
                and 95% confidence interval
            modellabel : string
                Customize model fit label (defualt "model")
        '''
        if fit in [0,1,2]:
            if ((datastruct == None) | (concstruct == None)):
                datastruct = self.datastruct
                concstruct = self.concstruct
            for nd in range(len(datastruct)):
                dataset = datastruct[nd]
                concset = concstruct[nd]
                fig = plt.figure()
                ax = fig.subplots(2,dataset.ntreats)
                cmax = np.max(concset.concmax)
                #nmax = np.max(dataset.survdata[:,1:])
                nmax = 1
                for i in range(dataset.ntreats):
                    ax[0,i].fill_between(concset.time,concset.concarray[i], label='Concentration', color='blue', alpha=0.2)
                    ax[0,i].set_ylim([0, cmax*1.1])
                    yvals = dataset.survdata[:,i+1]/dataset.survdata[0,i+1]
                    deltalow = np.maximum(yvals-dataset.lowlim[i],0)
                    deltaup = np.maximum(dataset.upplim[i]-yvals,0)
                    ax[1,i].errorbar(dataset.time,yvals, 
                                     yerr=[deltalow,deltaup], fmt='o',label='Survival')
                    ax[1,i].set_xlabel("Time [d]")
                    ax[1,i].set_ylim([0, nmax*1.1])
                ax[0,0].set_ylabel("Concentration [%s]"%self.conctunits)
                ax[1,0].set_ylabel("Survival")
                plt.tight_layout()
                if fit>0:
                    modelpars = np.copy(10**self.fullset*self.islog + self.fullset*(1-self.islog))
                    modelpars = modelpars[[0,1,2]+[3+nd]]
                    survmodelprob = np.zeros_like(dataset.survprobs)
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
                        damage = self.model.calc_damage(modelpars[0],dataset.timeext, concset.time, 
                                                        concset.concarray[i], concset.concslopes[i],
                                                        concset.concconst[i])
                        survival = self.model.calc_survival(dataset.timeext, concset.concarray[i],
                                                            damage, modelpars,
                                                            concset.concconst[i])
                        survmodelprob[i] = survival[dataset.index_commontime]
                        ax[0,i].plot(dataset.timeext, damage, label=modellabel,color='k', linestyle='--')
                        ax[1,i].plot(dataset.timeext, nmax*survival, label=modellabel)
                        # here needs to be modified with actual names of the treatments
                        ax2[0].plot(dataset.survprobs[i],survmodelprob[i], 'o', label = "Treatment %i"%(i)) 
                        ax2[1].plot(dataset.deatharray[i],
                                    dataset.survarray[i,0]*np.append(-np.diff(survmodelprob[i]),
                                                                    [survmodelprob[i,-1]]),
                                    'o',label = '')
                    maxdeaths = max(ax2[1].get_xlim()[1],ax2[1].get_ylim()[1])
                    ax2[1].plot([0,maxdeaths],[0,maxdeaths], 'k--',lw=0.5,label='')
                    ax2[0].legend(loc='lower right')
                    if fit>1:
                        for i in range(dataset.ntreats):
                            damlines = np.zeros((len(self.propagationset),len(dataset.timeext)))
                            surlines = np.zeros((len(self.propagationset),len(dataset.timeext)))
                            pars95 = np.copy(self.fullset)
                            for j in range(len(self.propagationset)):
                                pars95[self.posfree] = self.propagationset[j]
                                pars95 = 10**pars95*self.islog + pars95*(1-self.islog)
                                pars95_nd = pars95[[0,1,2]+[3+nd]]
                                damlines[j,:] = self.model.calc_damage(pars95_nd[0], dataset.timeext, concset.time, 
                                                        concset.concarray[i], concset.concslopes[i],
                                                        concset.concconst[i])
                                surlines[j,:] = self.model.calc_survival(dataset.timeext, concset.concarray[i],
                                                                         damlines[j,:], pars95_nd,
                                                                         concset.concconst[i])
                            damlineup   = damlines.max(axis=0)
                            damlinedown = damlines.min(axis=0)
                            surlineup   = surlines.max(axis=0)
                            surlinedown = surlines.min(axis=0)
                            ax[0,i].fill_between(dataset.timeext,damlinedown,damlineup, color='gray', alpha=0.5, label='95% CI')
                            ax[1,i].fill_between(dataset.timeext,surlinedown,surlineup, color='gray', alpha=0.5, label='95% CI')
                            ax2[0].errorbar(dataset.survprobs[i],
                                            survmodelprob[i],
                                            yerr=[survmodelprob[i]-surlinedown[dataset.index_commontime],
                                                  surlineup[dataset.index_commontime]-survmodelprob[i]], fmt='none',
                                                  ecolor='k', zorder = 0)
                    fig2.suptitle("Dataset %d"%(nd+1))
                    fig2.tight_layout()
                fig.suptitle("Dataset %d"%(nd+1))
                fig.tight_layout()
                plt.show()
        else:
            print("fit can be only 0 (data only), 1 (data and best fit), or 2 (data, best fit, and confidence interval)")


    def EFSA_quality_criteria(self, datastruct = None, concstruct = None):
        if ((dataset == None) | (concset == None)):
            datastruct = self.datastruct
            concstruct = self.concstruct
        for nd in range(self.ndatasets):
            dataset = datastruct[nd]
            concset = concstruct[nd]
            ssq_fit = 0
            ssq_fit0 = 0
            ssq_fitnum = 0
            ssq_fitnum0 = 0
            ssq_tot = 0
            ssq_tot0 = 0
            sppe = np.zeros(concset.ntreats)
            modelpars = 10**self.fullset*self.islog + self.fullset*(1-self.islog)
            for i in range(concset.ntreats):
                nmax = dataset.survarray[i,0]
                damage = self.model.calc_damage(modelpars[0],dataset.timeext, concset.time, 
                                                concset.concarray[i], concset.concslopes[i],
                                                concset.concconst[i])
                survival = self.model.calc_survival(dataset.timeext, concset.concarray[i],
                                                    damage, modelpars,
                                                    concset.concconst[i])
                ssq_fitnum += np.sum((dataset.survarray[i,1:]-nmax*survival[dataset.index_commontime[1:]])**2) 
                ssq_fitnum0 += np.sum((dataset.survarray[i]-nmax*survival[dataset.index_commontime])**2) 
                ssq_fit += np.sum((dataset.survprobs[i,1:]-survival[dataset.index_commontime[1:]])**2)
                ssq_fit0 += np.sum((dataset.survprobs[i]-survival[dataset.index_commontime])**2)
                sppe[i] = 100 * (dataset.survprobs[i,-1] - survival[dataset.index_commontime[-1]])
            ssq_tot = np.sum((dataset.survprobs.flatten()-np.mean(dataset.survprobs.flatten()))**2)
            ssq_tot0 = np.sum((dataset.survprobs[:,1:].flatten()-np.mean(dataset.survprobs[:,1:].flatten()))**2)
            nrmse   = 100 * np.sqrt(ssq_fitnum/(concset.ntreats*len(dataset.survprobs[0,1:]))) / np.mean(dataset.survarray[:,1:].flatten())
            nrmse0   = 100 * np.sqrt(ssq_fitnum0/(concset.ntreats*len(dataset.survprobs[0]))) / np.mean(dataset.survarray.flatten())
            print("Dataset %d----------------------------------------"%(nd+1))
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


    def lcx_calculation(self, timepoints=[2,4,10,21], levels=[0.1,0.2,0.5], pars = None, plot=False):
        # the calculation of LCx values assumes always that the 
        # exposure is constant
        if pars == None:
            modelpars = 10**self.fullset*self.islog + self.fullset*(1-self.islog)
            modelpars[-1] = 0 # for the LCx values, bkg mortality is 0
        else:
            modelpars = pars
        LCx = np.zeros((len(timepoints),len(levels)))
        LCxlo = np.zeros((len(timepoints),len(levels)))
        LCxup = np.zeros((len(timepoints),len(levels)))
        par95 = 10**self.propagationset*self.islog + self.propagationset*(1-self.islog)
        par95[:,-1] = 0 # remove the background mortality
        for i in range(len(timepoints)):
            timevectors = np.linspace(0,timepoints[i],self.model.nbinsperday)
            for j in range(len(levels)):
                lcxmin = np.inf
                lcxmax = 0
                if self.variant == 'IT':
                    # modelpars[1] = np.log(39)/np.log(modelpars[1])
                    # par95[:,1] = np.log(39)/np.log(par95[:,1])
                    beta = np.log(39)/np.log(modelpars[1])
                    LCx[i,j]=(modelpars[2]/(1.-np.exp(-modelpars[0]*timevectors[-1]))) * (levels[j]/(1.-levels[j]))**(1./beta)
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
                        conclims = conclims * 10 # shift lower and upper range by factor of 10
                        crit   = self.survfrac(conclims[1],timevectors,modelpars,levels[j]) - (1-levels[j]) # calculate criterion from upper range     
                    LCx[i,j] = sp.optimize.brenth(self.survfrac,conclims[0],conclims[1],args=(timevectors,modelpars,levels[j]))
                    for k in par95:
                        conclims = np.array([k[2]/10, k[2]])
                        crit = 1
                        while crit>0:
                            conclims = conclims * 10
                            crit   = self.survfrac(conclims[1],timevectors,k,levels[j]) - (1-levels[j]) # calculate criterion from upper range
                        lcx = sp.optimize.brenth(self.survfrac,conclims[0],conclims[1],args=(timevectors,k,levels[j]))
                        if lcx <= lcxmin:
                            lcxmin = lcx
                        if lcx >= lcxmax:
                            lcxmax = lcx
                    LCxlo[i,j] = lcxmin
                    LCxup[i,j] = lcxmax
        if plot:
            plt.figure()
            for i in range(len(levels)):
                plt.plot(timepoints, LCx[:,i],'o-', label='LD%d'%(round(levels[i]*100)))
                plt.fill_between(timepoints,LCxlo[:,i],LCxup[:,i],alpha = 0.2, zorder=0)
            plt.xlabel("Time [d]", fontsize=12)
            plt.ylabel("Concentration "+self.conctunits, fontsize=12)
            plt.legend(fontsize=12)
            plt.tight_layout()
            plt.show()
        # printing the results
        print("----------------------------------------------------------------")
        print("LCx values:")
        titlestring = "{:<10}".format("Time [d]")
        for i in range(len(levels)):
            titlestring = titlestring + "LD{:<32}".format(round(levels[i]*100))
        print(titlestring)
        for i in range(len(timepoints)):
            values = "{:<10d}".format(timepoints[i])
            for j in range(len(levels)):
                values = values + "{:<7.3g} ({:<7.3g} - {:<7.3g})       ".format(LCx[i,j], LCxlo[i,j], LCxup[i,j])
            print(values)

    def survfrac(self,conc,timevector,modelpars, level):
        return(self.model.calc_surv_sd_const(timevector,conc,modelpars)[-1] - (1-level))

    def validate(self, validationfile, hbfix = 0):
        pass

    def lpx_calculation(self):
        pass
