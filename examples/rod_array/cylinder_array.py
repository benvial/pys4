import matplotlib.pyplot as plt
import numpy as np
import sys

from pys4 import Simulation


period = 0.65
diameter = 0.503
h = 0.8

n_glass = 1.45
n_cylinder = 3.361

wl = 1.1


formulations = ["original", "new", "normal"]

formulation = formulations[1]
print(formulation)

nh=100

Rs = []


wls = np.linspace(1.4, 1.6, 1)

for wl in wls:

    simu = Simulation(((period, 0), (0, period)), nh)

    simu.discretized_epsilon = False
    if formulation == "new":
        simu.polarization_decomposition = True
        simu.polarization_basis = "default"
    elif formulation == "normal":
        simu.polarization_decomposition = True
        simu.polarization_basis = "normal"
        simu.discretization_resolution = 8

    subpixel_smoothing = True

    glass = simu.Material(name="glass", epsilon=n_glass**2)
    diel = simu.Material(name="diel", epsilon=n_cylinder**2)
    vacuum = simu.Material(name="vacuum", epsilon=1)

    superstrate = simu.Layer(name="Superstrate", material=vacuum)
    slab = simu.Layer(name="Slab", thickness=h, material=vacuum)
    substrate = simu.Layer(name="Substrate", material=glass)
    slab.add_circle(material=diel, radius=diameter / 2, center=(0 / 2, 0 / 2))
    pw = simu.PlaneWave(frequency=1 / wl, angles=(0, 0), sp_amplitudes=(1, 0))

    power_inc = simu.layers["Superstrate"].get_power_flux(0)[0].real
    Pr = simu.layers["Superstrate"].get_power_flux(order=False)
    R = -Pr[1].real / power_inc
    Pt = simu.layers["Substrate"].get_power_flux(order=False)
    T = Pt[0].real / power_inc
    print(nh, simu.num_basis_actual, R, T, R + T)

    Rs.append(R)


# plt.clf()

plt.plot(wls, Rs)
