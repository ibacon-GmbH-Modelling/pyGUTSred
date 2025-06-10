# pyGUTSred
python implementation of the reduced GUTS model

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

Place them either in your working directory or in a different folder that is added to your `PYTHONPATH` as in the following

```python
import sys
sys.path.append('/path/to/your/folder/with/the/pygutsred/scripts')
```

For a basic usage see the script `example_run.py`, while for a more detailed view of all the main functionalities, look at the jupyter notebook `use_example.ipynb`

(**NOTE**: the example script run under the assumption that they are placed in the same directory as the 3 source files)