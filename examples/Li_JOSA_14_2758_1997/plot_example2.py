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

import S4

plt.ion()


def simulation(ng, formulation):

    S = S4.New(((1, 0), (1 / 2, 3**0.5 / 2)), ng)

    S.SetMaterial(Name="Dielectric", Epsilon=2.56 + 0.0j)
    S.SetMaterial(Name="Vacuum", Epsilon=1 + 0.0j)
    S.AddLayer(Name="StuffAbove", Thickness=0, Material="Vacuum")
    S.AddLayer(Name="Slab", Thickness=0.5, Material="Vacuum")
    S.SetRegionCircle(Layer="Slab", Material="Dielectric", Center=(0, 0), Radius=0.25)
    S.AddLayer(Name="StuffBelow", Thickness=0, Material="Dielectric")
    S.SetExcitationPlanewave(
        IncidenceAngles=(
            30,
            30,
        ),  # polar angle in [0,180)  # azimuthal angle in [0,360)
        sAmplitude=0,
        pAmplitude=1,
        Order=0,
    )

    S.SetOptions(
        Verbosity=0,
        LatticeTruncation="Circular",
        # LatticeTruncation="Parallelogramic",
        # DiscretizedEpsilon=True,
        DiscretizationResolution=8,
        PolarizationDecomposition=False,
        PolarizationBasis="Default",
        LanczosSmoothing=False,
        SubpixelSmoothing=False,
        ConserveMemory=False,
    )

    S.SetFrequency(2.0000001)

    if formulation == "new":

        S.SetOptions(
            PolarizationDecomposition=True,
            PolarizationBasis="Default",
        )
    elif formulation == "normal":

        S.SetOptions(
            PolarizationDecomposition=True,
            PolarizationBasis="Normal",
        )
    elif formulation == "jones":
        S.SetOptions(
            PolarizationDecomposition=True,
            PolarizationBasis="Jones",
        )
    return S


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
        S = simulation(ng, f)
        power_inc = S.GetPoyntingFlux("StuffAbove", 0)[0].real
        G = S.GetBasisSet()
        for i, g in enumerate(G):
            if -1 == g[0] and -2 == g[1]:
                Gi = i
                break
        Pt = S.GetPoyntingFluxByOrder("StuffBelow", 0)
        T = (Pt[Gi][0] / power_inc).real
        actualg = len(G)
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
