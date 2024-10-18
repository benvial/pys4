"""
PhC slab
---------

Fig. 2b in
Wonjoo Suh, M. F. Yanik, Olav Solgaard, and Shanhui Fan,
"Displacement-sensitive photonic crystal structures based on guided resonance in photonic crystal slabs",
Appl. Phys. Letters, Vol. 82, No. 13, 2003
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pys4 import Simulation

simu = Simulation(((1, 0), (0, 1)), 40, polarization_decomposition=True)
silicon = simu.Material(name="Silicon", epsilon=12)
vacuum = simu.Material(name="Vacuum", epsilon=1)


superstrate = simu.Layer(name="Superstrate", material=vacuum)
slab1 = simu.Layer(name="Slab 1", thickness=0.55, material=silicon)
spacer = simu.Layer(name="Spacer", thickness=0.55, material=vacuum)
slab2 = slab1.copy(name="Slab 2")
substrate = superstrate.copy(name="Substrate")
slab1.add_circle(material=vacuum, radius=0.4)
pw = simu.PlaneWave(frequency=1, angles=(0, 0), sp_amplitudes=(1, 0))


freqs = np.linspace(0.49, 0.6, 300)
transmission_vs_thickness = []
thicknesses = [1.35, 1.1, 0.95, 0.85, 0.75, 0.65, 0.55]
for h in thicknesses:
    transmission = []
    for freq in freqs:
        pw.frequency = freq
        spacer.thickness = h
        power_inc = superstrate.get_power_flux(0)
        Pt = substrate.get_power_flux(0, order=True)
        T = Pt[0][0].real / power_inc[0].real
        transmission.append(T)
    transmission_vs_thickness.append(transmission)

transmission_vs_thickness = np.array(transmission_vs_thickness)

##################################################################################
# Plot transmission
colors = plt.cm.turbo(np.linspace(0, 1, len(thicknesses)))
plt.figure(figsize=(5, 4))
for i, transmission in enumerate(transmission_vs_thickness):
    plt.plot(freqs, transmission, c=colors[i], label=rf"$d = {thicknesses[i]}a$")
plt.xlim(0.49, 0.6)
plt.ylim(0, 1)
plt.xlabel("frequency (c/a)")
plt.ylabel("Transmission")
plt.legend(loc=(1.05, 0.3))
plt.tight_layout()
