"""
Dielectric patch array
-----------------------

Fig. 4 in
Quasi-guided modes and optical properties of photonic crystal slabs
S.G. Tikhodeev, A.L. Yablonskii, E.A. Muljarov, N.A. Gippius, and T. Ishihara
Phys. Rev. B 66, 45102 (2002)
"""

import numpy as np

import matplotlib.pyplot as plt
from pys4 import Simulation, SpectrumSampler

period = 0.68
simu = Simulation(((period, 0), (0, period)), 100, polarization_decomposition=True)
active = simu.Material(name="Active", epsilon=3.97)
quartz = simu.Material(name="Quartz", epsilon=2.132)
vacuum = simu.Material(name="Vacuum", epsilon=1)

superstrate = simu.Layer(name="Superstrate", material=vacuum)
slab = simu.Layer(name="Slab", thickness=0.12, material=quartz)
substrate = simu.Layer(name="Substrate", material=quartz)
slab.add_rectangle(material=active, widths=(0.8 * period, 0.8 * period))
pw = simu.PlaneWave(frequency=1, angles=(0, 0), sp_amplitudes=(1, 0))

ev2freq = 0.8065548889615557
sampler = SpectrumSampler(1, 2.6, initial_num_points=33)


def compute_transmission(ev):
    f = ev2freq * ev
    pw.frequency = f
    power_inc = simu.layers["Superstrate"].get_power_flux(0)
    Pt = simu.layers["Substrate"].get_power_flux(0, order=True)
    T = Pt[0][0].real / power_inc[0].real
    return T


fev = []
transmission = []
while not sampler.is_done():
    ev = sampler.get_frequency()
    T = compute_transmission(ev)
    sampler.submit_result(T)
    fev.append(ev)
    transmission.append(T)


srt = np.argsort(fev)
fev = np.array(fev)[srt]
transmission = np.array(transmission)[srt]

##################################################################################
# Plot transmission

plt.figure()
plt.plot(fev * 1000, transmission, c="#be4c83")
plt.ylim(0.4, 1)
plt.xlabel("frequency (meV)")
plt.ylabel("Transmissivity")
plt.tight_layout()
