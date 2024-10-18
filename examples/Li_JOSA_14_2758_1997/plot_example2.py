#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


"""
Circular pillar grating
-----------------------

Example 1A in
 Lifeng Li,
 "New formulation of the Fourier modal method for crossed surface-relief gratings"
 Journal of the Optical Society of America A, Vol. 14, No. 10, p. 2758 (1997)
This is Fig. 7

"""

import matplotlib.pyplot as plt

from pys4 import Simulation

plt.ion()


def simulation(ng, formulation):

    simu = Simulation(((1, 0), (1 / 2, 3**0.5 / 2)), ng)

    dielectric = simu.Material(name="Dielectric", epsilon=2.56)
    vacuum = simu.Material(name="Vacuum", epsilon=1)
    superstrate = simu.Layer(name="Superstrate", thickness=0, material=vacuum)
    slab = simu.Layer(name="Slab", thickness=0.5, material=vacuum)
    slab.add_circle(material=dielectric, radius=0.25)
    substrate = simu.Layer(name="Substrate", thickness=0, material=dielectric)
    simu.PlaneWave(frequency=2.0000001, angles=(30, 30), sp_amplitudes=(0, 1))
    simu.polarization_decomposition = True
    if formulation == "new":
        simu.polarization_basis = "default"
    elif formulation == "normal":
        simu.polarization_basis = "normal"
    elif formulation == "jones":
        simu.polarization_basis = "jones"
    else:
        simu.polarization_decomposition = False
    return simu


##########################################################
# Convergence for the different methods

ngs = range(41, 401, 40)
formulations = ["original", "new", "normal", "jones"]
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
        power_inc = simu.layers["Superstrate"].get_power_flux(0)[0].real
        G = simu.get_basis_set()
        for i, g in enumerate(G):
            if -1 == g[0] and -2 == g[1]:
                Gi = i
                break
        Pt = simu.layers["Substrate"].get_power_flux(0, order=True)
        T = (Pt[Gi][0] / power_inc).real
        actualg = simu.num_basis_actual
        print(f"nh = {actualg}, T = {T}")
        trans.append(T)
        actual_ngs.append(actualg)
    results[f] = trans
    truncation[f] = actual_ngs

marks = dict(original="o", new="s", normal="^", jones="x")
plt.figure()
for f in formulations:
    plt.plot(truncation[f], results[f], marker=marks[f], label=f)
plt.xlabel("Truncation Order $n_g$")
plt.ylabel("Diffraction Efficiency $T_{-1,-2}$")
plt.legend()
plt.tight_layout()
