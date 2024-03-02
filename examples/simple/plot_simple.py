"""
Simple example
--------------

Almost the simplest example. a simple planewave travelling through vacuum
"""

import sys

import numpy as np

import S4

S = S4.New(((1, 0), (0, 1)), 1)  # 1D lattice

S.AddMaterial("Vacuum", 1)

S.AddLayer("Front", 0, "Vacuum")  # name  # thickness  # background material
S.AddLayerCopy("Back", 0, "Front")  # new layer name  # thickness  # layer to copy

S.SetExcitationPlanewave(
    (
        0,
        0,
    ),  # incidence angles (spherical coordinates. phi in [0,180], theta in [0,360])
    1,  # s-polarization amplitude and phase (in degrees)
    0,
)  # p-polarization amplitude and phase

if len(sys.argv) <= 1:
    S.SetFrequency(0.5)
else:
    S.SetFrequency(sys.argv[1])

for z in np.arange(-2, 2, 0.1):
    print(S.GetFields(0, 0, z))
    print(S.GetPoyntingFlux("Front", z))
    print(S.GetPoyntingFlux("Back", z))
