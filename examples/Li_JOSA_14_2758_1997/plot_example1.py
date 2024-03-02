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

import S4


def simulation(ng, formulation):

    S = S4.New(((2.5, 0), (0, 2.5)), ng)
    S.SetMaterial(Name="Dielectric", Epsilon=2.25 + 0.0j)
    S.SetMaterial(Name="Vacuum", Epsilon=1 + 0.0j)
    S.AddLayer(Name="StuffAbove", Thickness=0, Material="Dielectric")
    S.AddLayer(Name="Slab", Thickness=1, Material="Vacuum")
    S.SetRegionRectangle(
        Layer="Slab",
        Material="Dielectric",
        Center=(1.25 / 2, 1.25 / 2),
        Angle=0,
        Halfwidths=(1.25 / 2, 1.25 / 2),
    )
    S.SetRegionRectangle(
        Layer="Slab",
        Material="Dielectric",
        Center=(-1.25 / 2, -1.25 / 2),
        Angle=0,
        Halfwidths=(1.25 / 2, 1.25 / 2),
    )
    S.SetExcitationPlanewave(
        IncidenceAngles=(0, 0),  # polar angle in [0,180)  # azimuthal angle in [0,360)
        sAmplitude=1,
        pAmplitude=0,
        Order=0,
    )
    S.AddLayer(Name="AirBelow", Thickness=0, Material="Vacuum")
    S.SetFrequency(1)

    if formulation == "new":

        S.SetOptions(
            PolarizationDecomposition=True,
            PolarizationBasis="Default",
        )
    elif formulation == "normal":

        S.SetOptions(
            PolarizationDecomposition=True,
            PolarizationBasis="Normal",
            DiscretizationResolution=8,
        )

    return S


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
        S = simulation(ng, f)
        power_inc = S.GetPoyntingFlux("StuffAbove", 0)[0].real
        G = S.GetBasisSet()
        P = S.GetPoyntingFluxByOrder("AirBelow", 0)
        actualg = len(G)
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
