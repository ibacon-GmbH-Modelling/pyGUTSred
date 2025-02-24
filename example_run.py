# testing script to run the GUTS model
import pyGUTSred as pg
import numpy as np

if __name__ == '__main__':

    # constant exposure test
    # SDfit = pg.pyGUTSred("datasets/ringtest_A_SD.txt", "SD", hbfree=True)
    # ITfit = pg.pyGUTSred("datasets/ringtest_A_IT.txt", "IT", hbfree=True)

    # # pulsed exposure
    # SDpuls = pg.pyGUTSred("datasets/ringtest_B_pulsed.txt",'SD')
    ITpuls2 = pg.pyGUTSred("datasets/ringtest_B_pulsed.txt",'IT',
                          preset=True)

    # to run uncomment the following lines
    #SDpuls2.run_parspace()
    #SDpuls2.plot_data_model(fit=2)
