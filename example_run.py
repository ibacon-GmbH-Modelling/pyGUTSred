# testing script to run the GUTS model
import pyGUTSred as pg
import numpy as np

if __name__ == '__main__':

    # constant exposure test
    # SDfit = pg.pyGUTSred("datasets/ringtest_A_SD.txt", "SD", hbfree=True)
    # ITfit = pg.pyGUTSred("datasets/ringtest_A_IT.txt", "IT", hbfree=True)

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
    ITfit = pg.pyGUTSred(["datasets/setsfromopenguts/diazinon_gammarus.txt"], "IT", hbfree=True)
    # ITfit.run_and_time_parspace()