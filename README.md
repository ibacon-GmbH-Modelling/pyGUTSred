# pyGUTSred

Python implementation of the reduced GUTS model.

This code is a python translation of the openGUTS and BYOM code developed originally 
for Matlab by Tjalling Jager to analyse survival data with a reduced GUTS model.
In particular the credits for the parameter space explorer algorithm go to Tjalling Jager.

For reference, the original openGUTS code can be found [here](https://openguts.info/), while the
complete BYOM code is [here](https://www.debtox.info/byom.html).

The current python implementation covers most of the functionalities available in the Matlab code:
* read-in of datasets
* model calibration
* model validation
* prediction of survival for FOCUS profiles

The aim of the code is to provide a modern object-oriented tool to perform analysis of
survival data for environmental risk assessment using the reduced GUTS model.

## Funding declaration

This project was funded by Syngenta AG.

## License

The software is distributed with GPLv3 license, with addenda due to the use of
third party python packages.

See details in the source code.

## Prerequisites
In order to use the code, the following addittional packages are needed:
* matplotlib (>= 3.7.1)
* numpy (>= 1.25.2)
* scipy (>= 1.11.1)
* pandas (>= 2.0.3)
* numba (>= 0.58.1)
* psutil (>= 5.9.0)

The code runs with python3. Good results in terms of interactivity are reached when used in a `ipython` terminal.

## Usage
The code relies on the following files:
* models.py
* parspace.py
* pyGUTSred.py

The file `models.py` contains all the specific functions related to the GUTS model and the likelihood calculations
given a dataset and model parameters

The file `parspace.py` contains the parameter space explorer algorithm adapted for the GUTS model

The file `pyGUTSred.py` contains the all the general framework and additional functions to run a complete GUTS modelling
analysis.

Place the three files either in your working directory or in a different folder that is added to your `PYTHONPATH` as in the following

```python
import sys
sys.path.append('/path/to/your/folder/with/the/pygutsred/scripts')
```

For a basic usage see the script `example_run.py`, while for a more detailed view of all the main functionalities, look at the jupyter notebook `use_example.ipynb`

(**NOTE 1**: the example script runs under the assumption that they are placed in the same directory as the 3 source files)

(**NOTE 2**: In a future iteration I will structure the scripts in the form of a python package
for an easier installation and usage)