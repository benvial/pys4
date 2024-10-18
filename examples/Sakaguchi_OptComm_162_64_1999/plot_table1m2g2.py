"""
Magneto-optical and dielectric multilayer
------------------------------------------------------

Duplicates Table 1, M2G2 structure, in
"Multilayer films composed of periodic magneto-optical and dielectric layers for use as Faraday rotators"
S. Sakaguchi and N. Sugimoto
Optics Communications 162, p. 64-70 (1999)
"""

from pys4 import Simulation
import numpy as np

print("n      rotation      transmittance")
print("----------------------------------")

g = 1.3784

for n in range(5, 11):
	simu = Simulation(((1, 0), (0, 1)), 1)

	# Material definition
	eps_mo = np.eye(3, dtype=complex) * 4.75
	eps_mo[0, 1] = 0.00269j
	eps_mo[1, 0] = -0.00269j
	eps_mo = tuple(map(tuple, eps_mo))
	mo = simu.Material(name="MO", epsilon=eps_mo)
	dielectric = simu.Material(name="Dielectric", epsilon=2.5)
	vacuum = simu.Material(name="Vacuum", epsilon=1)
	superstrate = simu.Layer(name="Superstrate", material=vacuum)

	# M2
	mlayer1 = simu.Layer("mlayer1", mo, 1)
	dlayer1 = simu.Layer("dlayer1", dielectric, g)
	for i in range(2, n+1):
		mlayer1.copy("mlayer" + str(i))
		dlayer1.copy("dlayer" + str(i))
	mlayer1.copy("m2", 2)
	for i in range(1, 2*n+1):
		dlayer1.copy("dlayer" + str(i + n))
		mlayer1.copy("mlayer" + str(i + n))
	dlayer1.copy("d2", 2 * g)
	for i in range(1, n+1 ):
		mlayer1.copy("mlayer" + str(i + 3 * n))
		dlayer1.copy("dlayer" + str(i + 3 * n))
	substrate = superstrate.copy(name="Substrate")
	pw = simu.PlaneWave(
		frequency=0.25 / np.sqrt(4.75), angles=(0, 0), sp_amplitudes=(1, 0)
	)
	t = substrate.get_power_flux(0)[0].real
	E, H = simu.get_fields(0, 0, 100)
	Hx = np.abs(H[0])
	Hy = np.abs(H[1])
	rot = np.rad2deg(np.arctan2(Hy,Hx))
	print(f"{n}        {rot:.3f}           {t:.3f}")
