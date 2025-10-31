#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: GPLv3

import os
import subprocess
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pys4 import Simulation, SpectrumSampler


def test_api(monkeypatch):
    """
    Test the pys4 API functionality including simulation setup,
    material and layer creation, geometric shapes, plane waves,
    and spectrum sampling.
    """
    plt.close("all")

    # Simulation setup
    sim = Simulation(((1, 0), (0, 1)), 51, verbosity=0)

    air = sim.Material("air", 1.0)
    substrate_material = sim.Material("whatever", 13.0)
    silicon = sim.Material("silicon", 7.0 - 0.1j)

    print(air)

    substrate = sim.Layer("substrate", substrate_material, 0.1)
    layer0 = sim.Layer("Layer0", air, 1)

    with pytest.raises(ValueError):
        layer0.add_circle(silicon, -0.4)
    layer0.add_circle(silicon, 0.4)

    layer1 = sim.Layer("Layer1", air, 1)
    layer1.add_square(silicon, 0.33)
    assert layer0.thickness == 1

    print(layer0)

    with pytest.raises(ValueError):
        layer0.add_ellipse(silicon, 0.4)
    layer0.add_ellipse(silicon, (0.1, 0.2))

    with pytest.raises(ValueError):
        layer0.add_rectangle(silicon, 0.4)

    with pytest.raises(ValueError):
        layer0_error = sim.Layer("Layer0", air, 1)

    with pytest.raises(ValueError):
        layer0_error = layer0.copy("Layer0")

    x, y = (np.linspace(-0.5, 0.5, 100), np.linspace(-0.5, 0.5, 100))

    for z in [0.5, 1.5]:
        eps_map = sim.get_epsilon_slice(x, y, z)
        plt.figure()
        plt.imshow(eps_map.real)
        plt.colorbar()
        plt.title(rf"$z=${z}")

    layer1_copy = layer0.copy("Layer1copy", 2)
    layer2 = sim.Layer("Layer2", silicon, 1)
    layer2.thickness = 2
    layer2.add_circle(silicon, 0.4)
    layer2.clean()
    layer2.add_rectangle(silicon, (0.5, 0.1), center=(-0.2, 0.2), angle=45)
    layer2.add_square(silicon, 0.22, angle=45)

    with pytest.raises(ValueError):
        layer2.add_square(silicon, -0.22, angle=45)

    vertices = ((0, 0), (0.2, 0), (0.2, 0.2), (0.1, 0.2), (0.1, 0.1), (0, 0.1))
    layer2.add_polygon(silicon, vertices, center=(0.1, -0.3))

    with pytest.raises(ValueError):
        layer2.add_polygon(silicon, vertices[:2])

    with pytest.raises(ValueError):
        layer2.add_polygon(silicon, 2)

    layer2.get_propagation_constants()

    plane_wave = sim.PlaneWave(0.6, (10, 30), sp_amplitudes=(1, 0), order=0)
    print(plane_wave)
    print(plane_wave.frequency)
    plane_wave.frequency = 0.61

    reciprocal_lattice = sim.reciprocal_lattice()
    epsilon = sim.get_epsilon(0.0, 0.0, 0.2)

    plt.figure()
    layer2.show()

    tmpdir = tempfile.mkdtemp()
    filename = os.path.join(tmpdir, "tmp.pov")
    sim.save_povray(filename)

    # Output computations
    amplitudes = layer2.get_amplitudes(0.2)
    layer2.get_power_flux(0.2, order=True)
    layer2.get_power_flux(0.2, order=False)
    layer2.get_stress_tensor_integral(z)
    layer2.get_layer_volume_integral("U")
    layer2.get_layer_z_integral(0, 0)

    sim.get_basis_set()
    assert sim.get_num_basis() == sim.num_basis_actual

    sim.get_fields(0, 0, 1)

    z = 1
    E, H = sim.get_fields_on_grid(z, (100, 100))

    plt.figure()
    plt.imshow(E[:, :, 0].real)
    plt.title(rf"$E_x, z=${z}")

    det = sim.get_s_matrix_determinant()

    sim.save_solution("test.sln")
    sim.load_solution("test.sln")

    sim_copy = sim.copy()

    def test_function(x):
        return np.sin(1 / (x * x + 0.05))

    sampler = SpectrumSampler(1, 2, parallelize=False)
    print(sampler.is_parallelized())

    with pytest.raises(ValueError):
        sampler = SpectrumSampler(3, 2, parallelize=False)

    while not sampler.is_done():
        freq = sampler.get_frequency()
        result = test_function(freq)
        sampler.submit_result(result)

    sampler = SpectrumSampler(1, 2, parallelize=True)
    spectrum = sampler.get_spectrum()

    while not sampler.is_done():
        frequencies = sampler.get_frequencies()
        results = []
        for freq in frequencies:
            results.append(test_function(freq))
        results = tuple(results)
        sampler.submit_results(results)

    spectrum = sampler.get_spectrum()

    # Patch subprocess.check_call to always raise FileNotFoundError
    def fake_check_call(*args, **kwargs):
        raise FileNotFoundError()

    monkeypatch.setattr(subprocess, "check_call", fake_check_call)

    with pytest.raises(RuntimeError, match="Neither 'magick' nor 'convert' commands found"):
        layer0.show()
