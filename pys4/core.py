#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: GPLv3

__all__ = ["Simulation", "SpectrumSampler"]


from . import S4 as _S4
import os
import subprocess
import tempfile
import numpy as np
import matplotlib.pyplot as plt


class _Material:
    def __init__(self, name, epsilon, _S4_simu):
        self.name = name
        self.epsilon = epsilon
        _S4_simu.SetMaterial(name, epsilon)

    def __repr__(self):
        return f"Material {self.name}(eps={self.epsilon})"


class _Layer:
    def __init__(self, name, material,thickness, _S4_simu, layers_dict, init=True):
        self.name = name
        self._thickness = thickness
        self.material = material
        self._S4_simu = _S4_simu
        self.layers_dict = layers_dict
        self.layers_dict[name] = self
        if init:
            _S4_simu.AddLayer(name, thickness, material.name)

    def copy(self, name, thickness=None):
        thickness = thickness if thickness is not None else self.thickness
        self._S4_simu.AddLayerCopy(name, thickness, self.name)
        return _Layer(
            name, self.material, thickness, self._S4_simu, self.layers_dict, init=False
        )

    @property
    def thickness(self):
        return self._thickness

    @thickness.setter
    def thickness(self, thickness):
        self._S4_simu.SetLayerThickness(self.name, thickness)
        self._thickness = thickness

    def clean(self):
        self._S4_simu.RemoveLayerRegions(self.name)

    def __repr__(self):
        return f"Layer {self.name}(h={self.thickness}, mat={self.material.name})"

    def add_circle(self, material, radius, center=(0, 0)):
        self._S4_simu.SetRegionCircle(self.name, material.name, center, radius)

    def add_ellipse(self, material, half_widths, angle=0):
        self._S4_simu.SetRegionEllipse(
            self.name, material.name, center, angle, half_widths
        )

    def add_rectangle(self, material, widths, center=(0, 0), angle=0):
        self._S4_simu.SetRegionRectangle(
            self.name, material.name, center, angle, (widths[0] / 2, widths[1] / 2)
        )

    def add_square(self, material, width, center=(0, 0), angle=0):
        self.add_rectangle(material, (width, width), center, angle)

    def add_polygon(self, material, vertices, center=(0, 0), angle=0):
        self._S4_simu.SetRegionPolygon(
            self.name, material.name, center, angle, vertices
        )

    def save_ps(self, filename):
        self._S4_simu.OutputLayerPatternPostscript(Layer=self.name, Filename=filename)

    def show(self):
        tmpdir = tempfile.mkdtemp()
        filename = os.path.join(tmpdir, "tmpimg.ps")
        self.save_ps(filename)
        filename_png = os.path.join(tmpdir, "tmpimg.png")
        args = ["magick", filename, filename_png]
        try:
            subprocess.check_call(args)
        except FileNotFoundError:
            args = ["convert", filename, filename_png]
        subprocess.call(args)
        img = plt.imread(filename_png)
        plt.imshow(img)
        plt.title(self.name)

    def get_amplitudes(self, z):
        return self._S4_simu.GetAmplitudes(Layer=self.name, zOffset=z)

    def get_power_flux(self, z, order=False):
        if order:
            return self._S4_simu.GetPowerFluxByOrder(Layer=self.name, zOffset=z)
        return self._S4_simu.GetPowerFlux(Layer=self.name, zOffset=z)

    def get_stress_tensor_integral(self, z):
        return self._S4_simu.GetStressTensorIntegral(Layer=self.name, zOffset=z)

    def get_layer_volume_integral(self, quantity):
        return self._S4_simu.GetLayerVolumeIntegral(Layer=self.name, Quantity=quantity)

    def get_layer_z_integral(self, x, y):
        return self._S4_simu.GetLayerZIntegral(Layer=self.name, xy=(x, y))

    def get_propagation_constants(self):
        return self._S4_simu.GetPropagationConstants(self.name)


class _PlaneWave:
    def __init__(self, frequency, angles, sp_amplitudes, order, _S4_simu):
        self._S4_simu = _S4_simu
        self.angles = angles
        self.sp_amplitudes = sp_amplitudes
        self.order = order
        self._frequency = frequency
        self.frequency = frequency
        s, p = sp_amplitudes
        _S4_simu.SetExcitationPlanewave(angles, sAmplitude=s, pAmplitude=p, Order=order)

    @property
    def frequency(self):
        return self._frequency

    @frequency.setter
    def frequency(self, frequency):
        self._S4_simu.SetFrequency(frequency)
        self._frequency = frequency

    def __repr__(self):
        return f"PlaneWave {self.name}(f={self.frequency},angles={self.angles}, sp ={self.sp_amplitudes}, order={self.order})"


class Simulation:
    def __init__(
        self,
        lattice,
        num_basis,
        verbosity=0,
        lattice_truncation="circular",
        discretized_epsilon=False,
        discretization_resolution=8,
        polarization_decomposition=False,
        polarization_basis="default",
        lanczos_smoothing=False,
        subpixel_smoothing=False,
        conserve_memory=False,
    ):
        self._S4_simu = _S4.New(lattice, num_basis)
        self.layers = {}
        self.lattice = lattice
        self.num_basis = num_basis

        self._verbosity = verbosity
        self._lattice_truncation = lattice_truncation
        self._polarization_decomposition = polarization_decomposition
        self._discretized_epsilon = discretized_epsilon
        self._discretization_resolution = discretization_resolution
        self._polarization_basis = polarization_basis
        self._lanczos_smoothing = lanczos_smoothing
        self._subpixel_smoothing = subpixel_smoothing
        self._conserve_memory = conserve_memory

        self.verbosity = verbosity
        self.lattice_truncation = lattice_truncation
        self.polarization_decomposition = polarization_decomposition
        self.discretized_epsilon = discretized_epsilon
        self.discretization_resolution = discretization_resolution
        self.polarization_basis = polarization_basis
        self.lanczos_smoothing = lanczos_smoothing
        self.subpixel_smoothing = subpixel_smoothing
        self.conserve_memory = conserve_memory

    @property
    def verbosity(self):
        return self._verbosity

    @verbosity.setter
    def verbosity(self, verbosity):
        self._S4_simu.SetOptions(Verbosity=verbosity)
        self._verbosity = verbosity

    @property
    def lattice_truncation(self):
        return self._lattice_truncation

    @lattice_truncation.setter
    def lattice_truncation(self, lattice_truncation):
        lattice_truncation = lattice_truncation.lower().capitalize()
        self._S4_simu.SetOptions(LatticeTruncation=lattice_truncation)
        self._lattice_truncation = lattice_truncation

    @property
    def discretized_epsilon(self):
        return self._discretized_epsilon

    @discretized_epsilon.setter
    def discretized_epsilon(self, discretized_epsilon):
        self._S4_simu.SetOptions(DiscretizedEpsilon=discretized_epsilon)
        self._discretized_epsilon = discretized_epsilon

    @property
    def discretization_resolution(self):
        return self._discretization_resolution

    @discretization_resolution.setter
    def discretization_resolution(self, discretization_resolution):
        self._S4_simu.SetOptions(DiscretizationResolution=discretization_resolution)
        self._discretization_resolution = discretization_resolution

    @property
    def polarization_decomposition(self):
        return self._polarization_decomposition

    @polarization_decomposition.setter
    def polarization_decomposition(self, polarization_decomposition):
        self._S4_simu.SetOptions(PolarizationDecomposition=polarization_decomposition)
        self._polarization_decomposition = polarization_decomposition

    @property
    def polarization_basis(self):
        return self._polarization_basis

    @polarization_basis.setter
    def polarization_basis(self, polarization_basis):
        polarization_basis = polarization_basis.lower().capitalize()
        self._S4_simu.SetOptions(PolarizationBasis=polarization_basis)
        self._polarization_basis = polarization_basis

    @property
    def lanczos_smoothing(self):
        return self._lanczos_smoothing

    @lanczos_smoothing.setter
    def lanczos_smoothing(self, lanczos_smoothing):
        self._S4_simu.SetOptions(LanczosSmoothing=lanczos_smoothing)
        self._lanczos_smoothing = lanczos_smoothing

    @property
    def subpixel_smoothing(self):
        return self._subpixel_smoothing

    @subpixel_smoothing.setter
    def subpixel_smoothing(self, subpixel_smoothing):
        self._S4_simu.SetOptions(SubpixelSmoothing=subpixel_smoothing)
        self._subpixel_smoothing = subpixel_smoothing

    @property
    def conserve_memory(self):
        return self._conserve_memory

    @conserve_memory.setter
    def conserve_memory(self, conserve_memory):
        self._S4_simu.SetOptions(ConserveMemory=conserve_memory)
        self._conserve_memory = conserve_memory

    def Material(self, name, epsilon):
        return _Material(name, epsilon, self._S4_simu)

    def Layer(self, name, material, thickness=0):
        return _Layer(name, material, thickness, self._S4_simu, self.layers)

    def PlaneWave(self, frequency, angles, sp_amplitudes=(1, 0), order=0):
        return _PlaneWave(frequency, angles, sp_amplitudes, order, self._S4_simu)

    def reciprocal_lattice(self):
        return self._S4_simu.GetReciprocalLattice()

    def get_epsilon(self, x, y, z):
        return self._S4_simu.GetEpsilon(x, y, z)

    def get_epsilon_slice(self, x, y, z=0):
        eps = np.zeros(x.shape + y.shape, dtype=complex)
        for i, _x in enumerate(x):
            for j, _y in enumerate(y):
                eps[i, j] = self.get_epsilon(_x, _y, z)
        return eps

    def save_povray(self, filename):
        self._S4_simu.OutputStructurePOVRay(filename)

    def get_basis_set(self):
        return self._S4_simu.GetBasisSet()

    def get_num_basis(self):
        return len(self.get_basis_set())

    @property
    def num_basis_actual(self):
        return self.get_num_basis()

    def get_fields(self, x, y, z):
        return self._S4_simu.GetFields(x, y, z)

    def get_fields_on_grid(self, z, num_samples):
        E, H = self._S4_simu.GetFieldsOnGrid(z, num_samples, Format="Array")
        return np.array(E), np.array(H)

    def get_s_matrix_determinant(self):
        return self._S4_simu.GetSMatrixDeterminant()

    def copy(self):
        sim_copy = Simulation(
            self.lattice,
            self.num_basis,
            self.verbosity,
            self.lattice_truncation,
            self.discretized_epsilon,
            self.discretization_resolution,
            self.polarization_decomposition,
            self.polarization_basis,
            self.lanczos_smoothing,
            self.subpixel_smoothing,
            self.conserve_memory,
        )
        sim_copy._S4_simu = self._S4_simu.Clone()
        return sim_copy

    def save_solution(self, filename):
        self._S4_simu.SaveSolution(filename)

    def load_solution(self, filename):
        self._S4_simu.LoadSolution(filename)


class SpectrumSampler:
    def __init__(
        self,
        start,
        stop,
        initial_num_points=33,
        range_threshold=0.001,
        max_bend=10,
        minimum_spacing=1e-6,
        parallelize=False,
    ):

        self.initial_num_points = initial_num_points
        self.range_threshold = range_threshold
        self.max_bend = max_bend
        self.minimum_spacing = minimum_spacing
        self.parallelize = parallelize

        options = dict(
            InitialNumPoints=initial_num_points,
            RangeThreshold=range_threshold,
            MaxBend=np.cos(np.rad2deg(max_bend)),
            MinimumSpacing=minimum_spacing,
            Parallelize=parallelize,
        )
        self.start = start
        self.stop = stop
        self._sampler = _S4.NewSpectrumSampler(start, stop, **options)

    def get_frequency(self):
        return self._sampler.GetFrequency()

    def get_frequencies(self):
        return self._sampler.GetFrequencies()

    def is_done(self):
        return self._sampler.IsDone()

    def is_parallelized(self):
        return self._sampler.IsParallelized()

    def submit_result(self, y):
        self._sampler.SubmitResult(y)

    def submit_results(self, y):
        self._sampler.SubmitResults(y)

    def get_spectrum(self):
        return self._sampler.GetSpectrum()

    def get_frequency(self):
        return self._sampler.GetFrequency()

    def get_frequencies(self):
        return self._sampler.GetFrequencies()

    def get_spectrum(self):
        return self._sampler.GetSpectrum()
