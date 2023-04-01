# Reproduction of Results from Paper
The directory `demo` provides Python scripts that reproduce the figures and the tables shown in the [paper published in the GMD journal](https://gmd.copernicus.org/articles/16/1697/2023/gmd-16-1697-2023.pdf).
In order to execute them with Python >= 3.8, you best proceed as follows:
1) Download the zip from the `fast-barnes-py` repository provided here on GitHub and extract its contents to an empty directory. Change into this directory.


2) Optional step: create a virtual Python environment inside this directory and activate it.


3) Install fast-barnes-py by executing
   ```
   pip install .
   ```
   In doing so also numpy, scipy and numba will be installed.


4) The visualization of the figures requires the installation of further packages (matplotlib, basemap and Pillow) listed in requirements.txt. Execute therefore also
   ```
   pip install -r requirements.txt
   ```

5) The preparation of your Python environment is now complete. Change into the `demo` directory and execute there for instance the `figure1.py` script

   ```
   cd demo
   
   python figure1.py
   ```
   The call of this script will create 6 plots with different levels of approximation to a Gaussian, similar to figure 1 of the paper.
   Likewise, you can execute the other `figure*.py` or `timing*.py` scripts provided in this directory.

### Potentially Long Execution Times
Due to the increased algorithmic complexity or repeated time measurements, most of the scripts suffer from annoyingly long execution times, which can exceed 5 hours and more.
The concerned scripts indicate countermeasures in their header comment on how these execution times can be reduced to an acceptable degree at the expense of test significance.
Scripts with relatively fast execution times of around 1 minute and below are given by `figure1.py`, `figure3b.py` and `figure11.py` through `figure14d.py`.

### Potentially Deviating Plot Diagrams
Depending on the used CPUs and the possibly slightly modified test setups described above, the time measurements might deviate from the results in the paper.
This can also be reflected in differing plot diagrams, in particular in axes with unsuitable axis ranges.

