# testing script to run the GUTS model
import pyGUTSred as pg


if __name__ == '__main__':

    SDfit = pg.pyGUTSred(["datasets/ringtest_A_SD.txt"], "SD", hbfree=True)
    SDfit.run_and_time_parspace()
    SDfit.plot_data_model(fit=2)
