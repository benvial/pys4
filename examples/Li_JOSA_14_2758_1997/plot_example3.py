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

import S4

plt.ion()


def simulation(ng, formulation, truncation):

    S = S4.New(((2, 0), (1, 3**0.5)), ng)

    S.SetMaterial(Name="Dielectric", Epsilon=2.25 + 0.0j)
    S.SetMaterial(Name="Metal", Epsilon=1 + 5.0j)
    S.SetMaterial(Name="Vacuum", Epsilon=1 + 0.0j)
    S.AddLayer(Name="StuffAbove", Thickness=0, Material="Vacuum")
    S.AddLayer(Name="Slab", Thickness=1, Material="Metal")
    S.SetRegionPolygon(
        "Slab",
        "Vacuum",
        [0, 0],
        0,
        (
            (-0.75, -0.25 * 3**0.5),
            (0.25, -0.25 * 3**0.5),
            (0.75, 0.25 * 3**0.5),
            (-0.25, 0.25 * 3**0.5),
        ),
    )
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
        LatticeTruncation=truncation,
        DiscretizationResolution=8,
        PolarizationDecomposition=False,
        PolarizationBasis="Default",
        LanczosSmoothing=False,
        SubpixelSmoothing=False,
        ConserveMemory=False,
    )

    S.SetFrequency(1.0000001)

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
            S = simulation(ng, f, t)
            power_inc = S.GetPoyntingFlux("StuffAbove", 0)
            G = S.GetBasisSet()
            order = (0, -1)
            for i, g in enumerate(G):
                if order[0] == g[0] and order[1] == g[1]:
                    Gi = i
                    break
            Pt = S.GetPoyntingFluxByOrder("StuffBelow", 0)
            T = Pt[Gi][0].real / power_inc[0].real

            order = (0, 0)
            for i, g in enumerate(G):
                if order[0] == g[0] and order[1] == g[1]:
                    Gi = i
                    break
            Pr = S.GetPoyntingFluxByOrder("StuffAbove", 0)
            R = Pr[Gi][1].real / power_inc[1].real
            actualg = len(G)
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
