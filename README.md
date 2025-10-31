


![Codecov (with branch)](https://img.shields.io/codecov/c/github/benvial/pys4/main?style=for-the-badge&logo=codecov&logoColor=white)


# pys4

`pys4` is a thin layer on top of the `S4` library as an attempt to simplify the installation and use of `S4` as a Python package. 

## Original S4 code

S4 (Stanford Stratified Structure Solver) is a program for computing electromagnetic fields in periodic, layered structures, developed by Victor Liu (victorliu@alumni.stanford.edu) of the Fan group in the Stanford Electrical Engineering Department.


See information about the original package in `src`, on the [webpage](http://fan.group.stanford.edu/S4/) or the [github repository](http://github.com/victorliu/S4)


## Installation

### From source

Clone this repository

```bash
git clone git@github.com:benvial/pys4.git
cd pys4
```


#### Using linux libraries

First install the dependencies

```bash
sudo apt-get update && sudo apt-get install -y --no-install-recommends  \
    gcc g++ pkg-config cmake libsuitesparse-dev liblapack-dev libopenblas-dev \
    libfftw3-dev libfabric-dev mpich libboost-serialization-dev \
    libboost-mpi-dev imagemagick povray ghostscript
```

and then pip install `pys4`:

```bash
pip install . 
```

or in editable mode

```bash
pip install -e . --no-build-isolation
```


#### Using conda

Alternatively, create a `conda` environment.

```bash
mamba env create -f environment.yml
conda activate pys4
```

and then pip install `pys4`:

```bash
pip install . -Csetup-args="-Dconda=true"
```

or in editable mode

```bash
pip install -e . -Csetup-args="-Dconda=true" --no-build-isolation
```

