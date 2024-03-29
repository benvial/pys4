
Examples
========

This page contains a set of examples that solve simple problems
or approximately reproduce simulation results in existing publications.
For a quick set of examples and instructions on how to run the them,
refer to the directories 1d/ for a simple non-MPI script, and
MPI_example/ for a simple MPI script.

Generally, the examples are run by:

.. code::

    /path/to/python example-file.py



.. Simple examples
.. ---------------
.. patterns    - Shows how to use various layer patterning methods
.. fabry_perot - Demonstrates common computations performed on the
..               simplest of structures.
.. 1d          - Shows how to specify 1D periodicity and the issues
..               to be aware of.


.. Published result examples
.. -------------------------
.. Fan_PRB_65_2002             - Simple example of transmission
..                               spectrum through photonic crystal
..                               slabs.
.. Antonoyiannakis_PRB_60_1999 - Demonstration of various force
..                               effects.
.. Suh_APL_82_1999_2003        - High Q Fano resonances.
.. Liu_OE_17_21897_2009        - Resonance enhancement of forces.
.. Li_JOSA_14_2758_1997        - Convergence test with original FMM
..                               reformulation examples, including
..                               a non-orthogonal lattice.
.. Tikhodeev_PRB_66_45102_2002 - Example of transmission spectrum
..                               through a photonic crystal slab.
.. Christ_PRB_70_125113_2004   - Metal grating transmission spectrum.


.. Other examples
.. --------------
.. spectrum_sampler - Demonstrates how to use the SpectrumSampler
..                    object on any function.
.. interpolator - Simple example of how to use the interpolator
..                object.
.. polarization_basis - Shows the vector fields generated for
..                      various lattices.
.. threading - Shows how to parallelize computations when threading
..             support is enabled.
.. magneto - Test case for tensor dieletric compared against analytic
..           theory.
