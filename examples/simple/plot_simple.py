"""
Simple example
--------------

Almost the simplest example: a simple plane wave travelling through vacuum
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
from pys4 import Simulation

# 1D lattice
simu = Simulation(((1, 0), (0, 1)), 1)
vacuum = simu.Material("Vacuum", 1)

front = simu.Layer("Front", vacuum, 0)  # name  # background material # thickness  
back = front.copy("Back", 0)  # new layer name  # thickness

if len(sys.argv) <= 1:
    f = 0.5
else:
    f = float(sys.argv[1])
pw = simu.PlaneWave(f, (0, 0), (1, 0))

wave = []
zs = np.arange(-2, 2, 0.1)
for z in zs:
    wave.append(simu.get_fields(0, 0, z))
    print(front.get_power_flux(z))
    print(back.get_power_flux(z))

wave = np.array(wave)

plt.plot(zs, wave[:, 0, 0].real, label="Ex")
plt.plot(zs, wave[:, 0, 1].real, label="Ey")
plt.plot(zs, wave[:, 0, 2].real, label="Ez")
plt.plot(zs, wave[:, 1, 0].real, label="Hx")
plt.plot(zs, wave[:, 1, 1].real, label="Hy")
plt.plot(zs, wave[:, 1, 2].real, label="Hz")
plt.xlabel("z")

plt.legend()
