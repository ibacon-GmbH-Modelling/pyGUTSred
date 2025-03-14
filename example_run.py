# testing script to run the GUTS model
import pyGUTSred as pg
import numpy as np

if __name__ == '__main__':

    # constant exposure test
    #SDfit = pg.pyGUTSred(["datasets/ringtest_A_SD.txt"], "SD", hbfree=True)
    # ITfit = pg.pyGUTSred("datasets/ringtest_A_IT.txt", "IT", hbfree=True)
    # SDfit.run_and_time_parspace()
    SDfit = pg.pyGUTSred.load_class("test_SDa.pkl")
    profile = pg.readprofile("./profiles/efsa_so_scenario/cereal_D5_pond.txt")
    profile.plot_exposure()
    # pg.lpx_calculation(profile, SDfit.model, 
    #                    propagationset = SDfit.propagationset, 
    #                    lpxvals = [0.1,0.5], 
    #                    srange = [0.05, 0.999], 
    #                    len_srange = 50)
    lp = pg.lpx_calculation(profile, SDfit.model, 
                       propagationset = SDfit.propagationset, 
                       lpxvals = [0.1], 
                       srange = [0.05, 0.999], 
                       len_srange = 50)

    # # pulsed exposure
    # SDpuls = pg.pyGUTSred("datasets/ringtest_B_pulsed.txt",'SD')
    # ITpuls = pg.pyGUTSred("datasets/ringtest_B_pulsed.txt",'IT',preset=True)

    # to run uncomment the following lines
    #SDfit.run_parspace()
    #SDfit.plot_data_model(fit=2)
    #SDfit.save_sample("test2.pkl")
    #SDfit.plot_data_model(fit=2)
    #SDfit=pg.pyGUTSred.load_class("test2.pkl")

    #ITfit = pg.pyGUTSred.load_class("test_ITsample.pkl")
    #ITfit = pg.pyGUTSred("datasets/ringtest_A_IT.txt", "IT", hbfree=True)
    #ITfit.run_and_time_parspace()
    # SDfit2 = pg.pyGUTSred(["datasets/ringtest_B_constant.txt"], "SD", hbfree=True)
    # SDfit2=pg.pyGUTSred.load_class("ttest.pkl")
    # SDfit2.run_and_time_parspace()
    # SDfit2.plot_data_model(fit=2)
    # # SDfit2.EFSA_quality_criteria()

    # newdataset = SDfit2._readfile("datasets/ringtest_A_SD.txt")
    # datastr = np.array([newdataset[1]])
    # concstr = np.array([newdataset[0]])
    # datastr[0].timeext = SDfit2.model.calc_ext_time(datastr[0])[0]
    # datastr[0].index_commontime = SDfit2.model.calc_ext_time(datastr[0])[1]
    # SDfit = pg.pyGUTSred(["datasets/fluorophenyl_minnows.txt"], "SD", hbfree=True)
    # ITfit = pg.pyGUTSred(["datasets/setsfromopenguts/diazinon_gammarus.txt"], "IT", hbfree=True)
    # ITfit.run_and_time_parspace()

    # SDfit = pg.pyGUTSred(["datasets/setsfromopenguts/dieldrin_guppy.txt"], "SD", hbfree=True)
    # SDfit.run_and_time_parspace()
    # ITfit = pg.pyGUTSred(["datasets/setsfromopenguts/dieldrin_guppy.txt"], "IT", hbfree=True)
    # ITfit.run_and_time_parspace()

    # SDfit = pg.pyGUTSred(["datasets/setsfromopenguts/fluorophenyl_minnows.txt"], "SD", hbfree=True)
    # SDfit.run_and_time_parspace()
    # SDfit.plot_data_model(fit=2)
    # SDfit.lcx_calculation(plot=True)
    # ITfit = pg.pyGUTSred(["datasets/setsfromopenguts/fluorophenyl_minnows.txt"], "IT", hbfree=True)
    # ITfit.run_and_time_parspace()
    # ITfit = pg.pyGUTSred.load_class("test_ITsample2.pkl")
    # ITfit.lcx_calculation(plot=True)
    # profile = pg.readprofile("./profiles/test1.txt")
    # profile.plot_exposure()
    # pg.lpx_calculation(profile, ITfit.model, 
    #                    propagationset = ITfit.propagationset, 
    #                    lpxvals = [0.1,0.5], 
    #                    srange = [0.05, 0.999], 
    #                    len_srange = 50)

    # SDfit = pg.pyGUTSred(["datasets/setsfromopenguts/methomyl_minnows.txt"], "SD", hbfree=True)
    # SDfit.run_and_time_parspace()
    # ITfit = pg.pyGUTSred(["datasets/setsfromopenguts/methomyl_minnows.txt"], "IT", hbfree=True)
    # ITfit.run_and_time_parspace()

    # SDfit = pg.pyGUTSred(["datasets/setsfromopenguts/propiconazole_weird.txt"], "SD", hbfree=True)
    # SDfit.run_and_time_parspace()
    # ITfit = pg.pyGUTSred(["datasets/setsfromopenguts/propiconazole_weird.txt"], "IT", hbfree=True)
    # ITfit.run_and_time_parspace()
    pass