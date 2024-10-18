#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


"""
Metallic grating
----------------

Example 1A in
 Lifeng Li,
 "New formulation of the Fourier modal method for crossed surface-relief gratings"
 Journal of the Optical Society of America A, Vol. 14, No. 10, p. 2758 (1997)
This is Fig. 10

"""


import matplotlib.pyplot as plt

from pys4 import Simulation


def simulation(ng, formulation, truncation):

    simu = Simulation(((2.0, 0), (1, 3**0.5)), ng)

    dielectric = simu.Material(name="Dielectric", epsilon=2.25)
    metal = simu.Material(name="Metal", epsilon=1 + 5.0j)
    vacuum = simu.Material(name="Vacuum", epsilon=1)
    superstrate = simu.Layer(name="Superstrate", thickness=0, material=vacuum)
    slab = simu.Layer(name="Slab", thickness=1, material=metal)
    vertices = (
        (-0.75, -0.25 * 3**0.5),
        (0.25, -0.25 * 3**0.5),
        (0.75, 0.25 * 3**0.5),
        (-0.25, 0.25 * 3**0.5),
    )
    slab.add_polygon(material=vacuum, vertices=vertices)
    substrate = simu.Layer(name="Substrate", thickness=0, material=dielectric)
    simu.PlaneWave(frequency=1.0000001, angles=(30, 30), sp_amplitudes=(0, 1))
    simu.lattice_truncation = truncation
    simu.polarization_decomposition = True if formulation == "new" else False
    return simu


##########################################################
# Convergence for the different methods

ngs = range(11, 401, 40)
formulations = ["original", "new"]
truncations = ["Circular", "Parallelogramic"]
resultsR = dict()
resultsT = dict()
truncation = dict()

for t in truncations:
    for f in formulations:
        key = f + " " + t.lower()
        print("---------------")
        print(key)
        print("---------------")
        trans = []
        refl = []
        actual_ngs = []

        for ng in ngs:
            simu = simulation(ng, f, t)
            power_inc = simu.layers["Superstrate"].get_power_flux(0)
            G = simu.get_basis_set()
            order = (0, -1)
            for i, g in enumerate(G):
                if order[0] == g[0] and order[1] == g[1]:
                    Gi = i
                    break
            Pt = simu.layers["Substrate"].get_power_flux(0, order=True)
            T = Pt[Gi][0].real / power_inc[0].real

            order = (0, 0)
            for i, g in enumerate(G):
                if order[0] == g[0] and order[1] == g[1]:
                    Gi = i
                    break
            Pr = simu.layers["Superstrate"].get_power_flux(0, order=True)
            R = Pr[Gi][1].real / power_inc[1].real
            actualg = simu.num_basis_actual
            print(f"nh = {actualg}, T{0,-1} = {T}, R{0,0} = {R}")
            trans.append(T)
            refl.append(R)
            actual_ngs.append(actualg)

        resultsR[key] = refl
        resultsT[key] = trans
        truncation[key] = actual_ngs


##########################################################
# Transmission


marks = {
    "original circular": "o",
    "new circular": "s",
    "original parallelogramic": "^",
    "new parallelogramic": "x",
}
plt.figure()
for t in truncations:
    for f in formulations:
        key = f + " " + t.lower()
        plt.plot(truncation[key], resultsT[key], marker=marks[key], label=key)
plt.xlabel("Truncation Order $n_g$")
plt.ylabel("Diffraction Efficiency $T_{0,-1}$")
plt.legend()
plt.tight_layout()


##########################################################
# Reflection


plt.figure()
for t in truncations:
    for f in formulations:
        key = f + " " + t.lower()
        plt.plot(truncation[key], resultsR[key], marker=marks[key], label=key)
plt.xlabel("Truncation Order $n_g$")
plt.ylabel("Diffraction Efficiency $R_{0,0}$")
plt.legend()
plt.tight_layout()
