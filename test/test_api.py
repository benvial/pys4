#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: GPLv3


from pys4 import Simulation, SpectrumSampler


def test_api():
    import matplotlib.pyplot as plt
    import numpy as np

    ###### Simulation ######
    plt.close("all")

    # ------- no output needed --------------

    simu = Simulation(((1, 0), (0, 1)), 51, verbosity=0)

    # simu.verbosity = 2
    # simu._S4_simu.SetOptions(Verbosity=9)
    # simu._S4_simu.SetOptions(ConserveMemory=True)

    mat = simu.Material("air", 1.0)
    mat1 = simu.Material("whatever", 13.0)
    si = simu.Material("silicon", 7.0 - 0.1j)

    sub = simu.Layer("substrate", mat1, 0.1)

    lay0 = simu.Layer("Layer0", mat, 1)
    lay0.add_circle(si, 0.4)
    lay1 = simu.Layer("Layer1", mat, 1)
    lay1.add_square(si, 0.33)

    x, y = (np.linspace(-0.5, 0.5, 100), np.linspace(-0.5, 0.5, 100))

    for z in [0.5, 1.5]:
        eps_map = simu.get_epsilon_slice(x, y, z)
        plt.figure()
        plt.imshow(eps_map.real)
        plt.colorbar()
        plt.title(rf"$z=${z}")

    lay1_copy = lay0.copy("Layer1copy", 2)
    lay2 = simu.Layer("Layer2", si, 1)

    lay2.thickness = 2

    lay2.add_circle(si, 0.4)

    # plt.figure()
    # lay2.show()

    lay2.clean()
    lay2.add_rectangle(si, (0.5, 0.1), center=(-0.2, 0.2), angle=45)
    lay2.add_square(si, 0.22, angle=45)

    vertices = ((0, 0), (0.2, 0), (0.2, 0.2), (0.1, 0.2), (0.1, 0.1), (0, 0.1))
    lay2.add_polygon(si, vertices, center=(0.1, -0.3))

    lay2.get_propagation_constants()

    pw = simu.PlaneWave(0.6, (10, 30), sp_amplitudes=(1, 0), order=0)

    rl = simu.reciprocal_lattice()
    eps = simu.get_epsilon(0.0, 0.0, 0.2)

    # table = (( 1, 'x', ( 1, 0.1)),)
    # out = simu._S4_simu.SetExcitationExterior(table)

    plt.figure()
    lay2.show()

    import tempfile, os

    tmpdir = tempfile.mkdtemp()
    filename = os.path.join(tmpdir, "tmp.pov")
    simu.save_povray(filename)

    # -------  output needed --------------

    A = lay2.get_amplitudes(0.2)
    lay2.get_power_flux(0.2, order=True)
    lay2.get_power_flux(0.2, order=False)
    lay2.get_stress_tensor_integral(z)
    lay2.get_layer_volume_integral("U")
    lay2.get_layer_z_integral(0, 0)

    simu.get_basis_set()
    assert simu.get_num_basis() == simu.num_basis_actual

    simu.get_fields(0, 0, 1)

    z = 1
    E, H = simu.get_fields_on_grid(z, (100, 100))

    plt.figure()
    plt.imshow(E[:, :, 0].real)
    plt.title(rf"$E_x, z=${z}")

    det = simu.get_s_matrix_determinant()

    simu.save_solution("test.sln")
    simu.load_solution("test.sln")

    simu_copy = simu.copy()

    def f(x):
        return np.sin(1 / (x * x + 0.05))

    sampler = SpectrumSampler(1, 2, parallelize=False)

    while not sampler.is_done():
        x = sampler.get_frequency()
        y = f(x)
        sampler.submit_result(y)

    sampler = SpectrumSampler(1, 2, parallelize=True)

    spec = sampler.get_spectrum()

    while not sampler.is_done():
        x = sampler.get_frequencies()
        y = []
        for _x in x:
            y.append(f(_x))
        y = tuple(y)
        sampler.submit_results(y)

    spec = sampler.get_spectrum()
