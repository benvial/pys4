#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

"""
Checkerboard grating
--------------------

Example 1A in
 Lifeng Li,
 "New formulation of the Fourier modal method for crossed surface-relief gratings"
 Journal of the Optical Society of America A, Vol. 14, No. 10, p. 2758 (1997)
This is Fig. 6

"""


import matplotlib.pyplot as plt

from pys4 import Simulation


def simulation(ng, formulation):
    simu = Simulation(((2.5, 0), (0, 2.5)), ng)
    dielectric = simu.Material(name="Dielectric", epsilon=2.25)
    vacuum = simu.Material(name="Vacuum", epsilon=1)
    above = simu.Layer(name="StuffAbove", thickness=0, material=dielectric)
    slab = simu.Layer(name="Slab", thickness=1, material=vacuum)
    below = simu.Layer(name="AirBelow", thickness=0, material=vacuum)
    slab.add_rectangle(
        material=dielectric, widths=(1.25, 1.25), center=(1.25 / 2, 1.25 / 2)
    )
    slab.add_rectangle(
        material=dielectric, widths=(1.25, 1.25), center=(-1.25 / 2, -1.25 / 2)
    )
    # polar angle in [0,180)  # azimuthal angle in [0,360)
    simu.PlaneWave(frequency=1, angles=(0, 0), sp_amplitudes=(1, 0))

    if formulation == "new":
        simu.polarization_decomposition = True
        simu.polarization_basis = "default"
    elif formulation == "normal":
        simu.polarization_decomposition = True
        simu.polarization_basis = "normal"
        simu.discretization_resolution = 8
    return simu


##########################################################
# Convergence for the different methods

ngs = range(41, 401, 40)
formulations = ["original", "new", "normal"]
results = dict()
truncation = dict()

for f in formulations:
    print("---------------")
    print(f)
    print("---------------")
    trans = []
    actual_ngs = []

    for ng in ngs:
        simu = simulation(ng, f)
        power_inc = simu.layers["StuffAbove"].get_power_flux(0)[0].real
        P = simu.layers["AirBelow"].get_power_flux(0, order=True)
        actualg = simu.num_basis_actual
        T = P[5][0].real / power_inc
        print(actualg, T)
        trans.append(T)
        actual_ngs.append(actualg)
    results[f] = trans
    truncation[f] = actual_ngs


marks = dict(original="o", new="s", normal="^")
plt.figure()
for f in formulations:
    plt.plot(truncation[f], results[f], marker=marks[f], label=f)
plt.xlabel("Truncation Order $n_g$")
plt.ylabel("Diffraction Efficiency $T_{0,-1}$")
plt.legend()
plt.tight_layout()
