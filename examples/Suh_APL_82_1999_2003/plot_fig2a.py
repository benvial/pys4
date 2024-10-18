
"""
PhC slab
---------

Fig. 2a in
Wonjoo Suh, M. F. Yanik, Olav Solgaard, and Shanhui Fan,
"Displacement-sensitive photonic crystal structures based on guided resonance in photonic crystal slabs",
Appl. Phys. Letters, Vol. 82, No. 13, 2003
"""

import matplotlib.pyplot as plt
import numpy as np
from pys4 import Simulation

simu = Simulation(((1, 0), (0, 1)), 40, polarization_decomposition=True)
silicon = simu.Material(name="Silicon", epsilon=12)
vacuum = simu.Material(name="Vacuum", epsilon=1)


superstrate = simu.Layer(name="Superstrate", material=vacuum)
slab = simu.Layer(name="Slab", thickness=0.55, material=silicon)
substrate = superstrate.copy(name="Substrate")
slab.add_circle(material=vacuum, radius=0.4)
pw = simu.PlaneWave(frequency=1, angles=(0, 0), sp_amplitudes=(1, 0))


freqs = np.arange(0.49,0.6,0.003)
transmission = []
for freq in freqs:
    pw.frequency = freq
    power_inc = superstrate.get_power_flux(0)
    Pt = substrate.get_power_flux(0, order=True)
    T = Pt[0][0].real / power_inc[0].real
    transmission.append(T)

##################################################################################
# Plot transmission

plt.figure()
plt.plot(freqs, transmission, c="#cf2f26")
plt.ylim(0., 1)
plt.xlabel("frequency (c/a)")
plt.ylabel("Transmission")
plt.tight_layout()
