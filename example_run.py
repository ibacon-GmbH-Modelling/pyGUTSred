# testing script to run the GUTS model
import pyGUTSred as pg
import numpy as np

if __name__ == '__main__':

    # constant exposure test
    #SDfit = pg.pyGUTSred("datasets/ringtest_A_SD.txt", "SD", hbfree=True)
    # ITfit = pg.pyGUTSred("datasets/ringtest_A_IT.txt", "IT", hbfree=True)

    # # pulsed exposure
    # SDpuls = pg.pyGUTSred("datasets/ringtest_B_pulsed.txt",'SD')
    #ITpuls2 = pg.pyGUTSred("datasets/ringtest_B_pulsed.txt",'IT',
    #                      preset=True)

    # to run uncomment the following lines
    #SDfit.run_parspace()
    #SDfit.plot_data_model(fit=2)
    #SDfit.save_sample("test2.pkl")
    #SDfit.plot_data_model(fit=2)
    #SDfit=pg.pyGUTSred.load_class("test2.pkl")

    #ITfit = pg.pyGUTSred.load_class("test_ITsample.pkl")
    #ITfit = pg.pyGUTSred("datasets/ringtest_A_IT.txt", "IT", hbfree=True)
    #ITfit.run_and_time_parspace()
    SDfit2 = pg.pyGUTSred(["datasets/ringtest_B_constant.txt","datasets/ringtest_B_pulsed.txt"], "SD", hbfree=True)
    SDfit2.run_and_time_parspace()
    SDfit2.plot_data_model(fit=2)