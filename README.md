

# pys4


This is an attempt to simplify the installation and use of `S4` as a Python package. 
`pys4` is a thin layer on top of the `S4` library.

## S4

S4 (Stanford Stratified Structure Solver) is a program for computing electromagnetic fields in periodic, layered structures, developed by Victor Liu (victorliu@alumni.stanford.edu) of the Fan group in the Stanford Electrical Engineering Department.


See information about the original package in `src`, on the [webpage](http://fan.group.stanford.edu/S4/) or the [github repository](http://github.com/victorliu/S4)


## Installation

The easiest way to get all the dependencies is to create a `conda` environment.

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

