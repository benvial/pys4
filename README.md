

# pys4


This is an attempt to simplify the installation and use of `S4` as a Python package. 
`pys4` is a thin layer on top of the `S4` library.

## S4

S4 (Stanford Stratified Structure Solver) is a program for computing electromagnetic fields in periodic, layered structures, developed by Victor Liu (victorliu@alumni.stanford.edu) of the Fan group in the Stanford Electrical Engineering Department.


See information about the original package in `src`, on the [webpage](http://fan.group.stanford.edu/S4/) or the [github repository](http://github.com/victorliu/S4)


## Installation

The easiest way to get all the dependencies is to create a `conda` environment.

```bash
mamba create --name pys4
mamba activate pys4
mamba install python pip numpy suitesparse openblas mpich boost fftw cxx-compiler
```
or alternatively using a file:

```bash
mamba env create -f environment.yml
conda activate pys4
```

and then pip install `pys4`:

```bash
pip install .
```

Optionally, check the installation:


```bash
make check
```

and run the tests:

```bash
make test
```


## Usage

```python
from pys4 import S4
```