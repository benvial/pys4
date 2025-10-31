#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: GPLv3

__all__ = ["Simulation", "SpectrumSampler"]


from . import _S4
import os
import subprocess
import tempfile
import numpy as np
import matplotlib.pyplot as plt


class _Material:
    """Material class for defining optical materials."""
    def __init__(self, name, epsilon, _S4_simu):
        """Initialize a material.
        
        Args:
            name (str): Name of the material.
            epsilon (complex or float): Permittivity of the material.
            _S4_simu: S4 simulation object.
        """
        self.name = name
        self.epsilon = epsilon
        _S4_simu.SetMaterial(name, epsilon)

    def __repr__(self):
        return f"Material {self.name}(eps={self.epsilon})"


class _Layer:
    """Layer class for defining simulation layers."""
    def __init__(self, name, material, thickness, _S4_simu, layers_dict, init=True):
        """Initialize a layer.
        
        Args:
            name (str): Name of the layer.
            material (_Material): Material of the layer.
            thickness (float): Thickness of the layer.
            _S4_simu: S4 simulation object.
            layers_dict (dict): Dictionary of layers.
            init (bool): Whether to initialize the layer in S4.
            
        Raises:
            ValueError: If a layer with the same name already exists and init is True.
        """
        self.name = name
        self._thickness = thickness
        self.material = material
        self._S4_simu = _S4_simu
        self.layers_dict = layers_dict
        
        # Check if layer already exists
        if init and name in self.layers_dict:
            raise ValueError(f"Layer with name '{name}' already exists.")
            
        self.layers_dict[name] = self
        if init:
            _S4_simu.AddLayer(name, thickness, material.name)

    def copy(self, name, thickness=None):
        """Create a copy of the layer.
        
        Args:
            name (str): Name of the new layer.
            thickness (float, optional): Thickness of the new layer. If None, uses the current layer's thickness.
            
        Returns:
            _Layer: The copied layer.
            
        Raises:
            ValueError: If a layer with the same name already exists.
        """
        if name in self.layers_dict:
            raise ValueError(f"Layer with name '{name}' already exists.")
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
        """Remove all regions from the layer."""
        self._S4_simu.RemoveLayerRegions(self.name)

    def __repr__(self):
        return f"Layer {self.name}(h={self.thickness}, mat={self.material.name})"

    def add_circle(self, material, radius, center=(0, 0)):
        """Add a circular region to the layer.
        
        Args:
            material (_Material): Material of the region.
            radius (float): Radius of the circle.
            center (tuple): Center coordinates of the circle (x, y).
        """
        # Validate radius parameter
        if radius <= 0:
            raise ValueError("Radius must be positive.")
        self._S4_simu.SetRegionCircle(self.name, material.name, center, radius)

    def add_ellipse(self, material, radii, center=(0, 0), angle=0):
        """Add an elliptical region to the layer.
        
        Args:
            material (_Material): Material of the region.
            radii (tuple): Radii of the ellipse (rx, ry).
            center (tuple): Center coordinates of the ellipse (x, y).
            angle (float): Rotation angle of the ellipse in degrees.
        """
        # Validate radii parameter
        if not isinstance(radii, (tuple, list)) or len(radii) != 2:
            raise ValueError("Radii must be a tuple or list of length 2.")
        self._S4_simu.SetRegionEllipse(
            self.name, material.name, center, angle, radii
        )

    def add_rectangle(self, material, widths, center=(0, 0), angle=0):
        """Add a rectangular region to the layer.
        
        Args:
            material (_Material): Material of the region.
            widths (tuple): Widths of the rectangle (wx, wy).
            center (tuple): Center coordinates of the rectangle (x, y).
            angle (float): Rotation angle of the rectangle in degrees.
        """
        # Validate widths parameter
        if not isinstance(widths, (tuple, list)) or len(widths) != 2:
            raise ValueError("Widths must be a tuple or list of length 2.")
        self._S4_simu.SetRegionRectangle(
            self.name, material.name, center, angle, (widths[0] / 2, widths[1] / 2)
        )

    def add_square(self, material, width, center=(0, 0), angle=0):
        """Add a square region to the layer.
        
        Args:
            material (_Material): Material of the region.
            width (float): Width of the square.
            center (tuple): Center coordinates of the square (x, y).
            angle (float): Rotation angle of the square in degrees.
        """
        # Validate width parameter
        if width <= 0:
            raise ValueError("Width must be positive.")
        self.add_rectangle(material, (width, width), center, angle)

    def add_polygon(self, material, vertices, center=(0, 0), angle=0):
        """Add a polygonal region to the layer.
        
        Args:
            material (_Material): Material of the region.
            vertices (list): List of vertices defining the polygon.
            center (tuple): Center coordinates of the polygon (x, y).
            angle (float): Rotation angle of the polygon in degrees.
        """
        # Validate vertices parameter
        if not isinstance(vertices, (list, tuple)):
            raise ValueError("Vertices must be a list or tuple.")
        if len(vertices) < 3:
            raise ValueError("A polygon must have at least 3 vertices.")
        self._S4_simu.SetRegionPolygon(
            self.name, material.name, center, angle, vertices
        )

    def save_ps(self, filename):
        """Save the layer pattern as a PostScript file.
        
        Args:
            filename (str): Name of the file to save.
        """
        self._S4_simu.OutputLayerPatternPostscript(Layer=self.name, Filename=filename)

    def show(self):
        """Display the layer pattern."""
        tmpdir = tempfile.mkdtemp()
        filename = os.path.join(tmpdir, "tmpimg.ps")
        self.save_ps(filename)
        filename_png = os.path.join(tmpdir, "tmpimg.png")
        
        # Try to convert PS to PNG using ImageMagick
        conversion_successful = False
        for command in ["magick", "convert"]:
            try:
                subprocess.check_call([command, filename, filename_png])
                conversion_successful = True
                break
            except FileNotFoundError:
                continue
        
        # If conversion failed, raise an error
        if not conversion_successful:
            raise RuntimeError("Neither 'magick' nor 'convert' commands found. Please install ImageMagick.")
        
        # Display the image
        img = plt.imread(filename_png)
        plt.imshow(img)
        plt.title(self.name)
        plt.axis("equal")
        # plt.show()

    def get_amplitudes(self, z=0):
        """Get the amplitudes of the electromagnetic field.
        
        Args:
            z (float): Z-coordinate offset.
            
        Returns:
            tuple: Amplitudes of the electromagnetic field.
        """
        return self._S4_simu.GetAmplitudes(Layer=self.name, zOffset=z)

    def get_power_flux(self, z=0, order=False):
        """Get the power flux.
        
        Args:
            z (float): Z-coordinate offset.
            order (bool): Whether to get power flux by order.
            
        Returns:
            tuple: Power flux values.
        """
        if order:
            return self._S4_simu.GetPowerFluxByOrder(Layer=self.name, zOffset=z)
        return self._S4_simu.GetPowerFlux(Layer=self.name, zOffset=z)

    def get_stress_tensor_integral(self, z=0):
        """Get the stress tensor integral.
        
        Args:
            z (float): Z-coordinate offset.
            
        Returns:
            tuple: Stress tensor integral values.
        """
        return self._S4_simu.GetStressTensorIntegral(Layer=self.name, zOffset=z)

    def get_layer_volume_integral(self, quantity):
        """Get the layer volume integral.
        
        Args:
            quantity (str): Quantity to integrate.
            
        Returns:
            float: Layer volume integral value.
        """
        return self._S4_simu.GetLayerVolumeIntegral(Layer=self.name, Quantity=quantity)

    def get_layer_z_integral(self, x, y):
        """Get the layer Z integral.
        
        Args:
            x (float): X coordinate.
            y (float): Y coordinate.
            
        Returns:
            float: Layer Z integral value.
        """
        return self._S4_simu.GetLayerZIntegral(Layer=self.name, xy=(x, y))

    def get_propagation_constants(self):
        """Get the propagation constants.
        
        Returns:
            tuple: Propagation constants.
        """
        return self._S4_simu.GetPropagationConstants(self.name)


class _PlaneWave:
    """Plane wave excitation class."""
    def __init__(self, frequency, angles, sp_amplitudes, order, _S4_simu):
        """Initialize a plane wave excitation.
        
        Args:
            frequency (float): Frequency of the plane wave.
            angles (tuple): Angles of the plane wave (theta, phi).
            sp_amplitudes (tuple): S and P polarization amplitudes.
            order (int): Order of the plane wave.
            _S4_simu: S4 simulation object.
        """
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
        return f"PlaneWave(f={self.frequency},angles={self.angles}, sp={self.sp_amplitudes}, order={self.order})"


class Simulation:
    """Simulation class for S4 simulations."""
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
        """Initialize a simulation.
        
        Args:
            lattice: Lattice parameters.
            num_basis (int): Number of basis functions.
            verbosity (int): Verbosity level.
            lattice_truncation (str): Lattice truncation method.
            discretized_epsilon (bool): Whether to use discretized epsilon.
            discretization_resolution (int): Discretization resolution.
            polarization_decomposition (bool): Whether to use polarization decomposition.
            polarization_basis (str): Polarization basis.
            lanczos_smoothing (bool): Whether to use Lanczos smoothing.
            subpixel_smoothing (bool): Whether to use subpixel smoothing.
            conserve_memory (bool): Whether to conserve memory.
        """
        # Initialize the S4 simulation
        self._S4_simu = _S4.New(lattice, num_basis)
        self.layers = {}
        self.lattice = lattice
        self.num_basis = num_basis

        # Store option values
        self._verbosity = verbosity
        self._lattice_truncation = lattice_truncation
        self._polarization_decomposition = polarization_decomposition
        self._discretized_epsilon = discretized_epsilon
        self._discretization_resolution = discretization_resolution
        self._polarization_basis = polarization_basis
        self._lanczos_smoothing = lanczos_smoothing
        self._subpixel_smoothing = subpixel_smoothing
        self._conserve_memory = conserve_memory

        # Apply options through properties to ensure proper validation
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
        """Create a material.
        
        Args:
            name (str): Name of the material.
            epsilon (complex or float): Permittivity of the material.
            
        Returns:
            _Material: The created material.
        """
        return _Material(name, epsilon, self._S4_simu)

    def Layer(self, name, material, thickness=0):
        """Create a layer.
        
        Args:
            name (str): Name of the layer.
            material (_Material): Material of the layer.
            thickness (float): Thickness of the layer.
            
        Returns:
            _Layer: The created layer.
        """
        return _Layer(name, material, thickness, self._S4_simu, self.layers)

    def PlaneWave(self, frequency, angles, sp_amplitudes=(1, 0), order=0):
        """Create a plane wave excitation.
        
        Args:
            frequency (float): Frequency of the plane wave.
            angles (tuple): Angles of the plane wave (theta, phi).
            sp_amplitudes (tuple): S and P polarization amplitudes.
            order (int): Order of the plane wave.
            
        Returns:
            _PlaneWave: The created plane wave.
        """
        return _PlaneWave(frequency, angles, sp_amplitudes, order, self._S4_simu)

    def reciprocal_lattice(self):
        """Get the reciprocal lattice.
        
        Returns:
            tuple: Reciprocal lattice vectors.
        """
        return np.asarray(self._S4_simu.GetReciprocalLattice())

    def get_epsilon(self, x, y, z):
        """Get the permittivity at a point.
        
        Args:
            x (float): X coordinate.
            y (float): Y coordinate.
            z (float): Z coordinate.
            
        Returns:
            complex: Permittivity at the point.
        """
        return np.asarray(self._S4_simu.GetEpsilon(x, y, z))

    def get_epsilon_slice(self, x, y, z=0):
        """Get a slice of the permittivity.
        
        Args:
            x (array): X coordinates.
            y (array): Y coordinates.
            z (float): Z coordinate.
            
        Returns:
            array: Permittivity values on the grid.
        """
        # Create a meshgrid for vectorized computation
        X, Y = np.meshgrid(x, y, indexing='ij')
        # Vectorize the get_epsilon function
        get_epsilon_vectorized = np.vectorize(self.get_epsilon)
        # Compute epsilon values on the grid
        eps = get_epsilon_vectorized(X, Y, z)
        return eps

    def save_povray(self, filename: str) -> None:
        """Save the structure as a POV-Ray file.
        
        Args:
            filename (str): Name of the file to save.
        """
        self._S4_simu.OutputStructurePOVRay(filename)

    def get_basis_set(self) -> list:
        """Get the basis set.
        
        Returns:
            list: Basis set vectors.
        """
        return np.asarray(self._S4_simu.GetBasisSet())

    def get_num_basis(self) -> int:
        """Get the number of basis functions.
        
        Returns:
            int: Number of basis functions.
        """
        return len(self.get_basis_set())

    @property
    def num_basis_actual(self) -> int:
        """Get the actual number of basis functions.
        
        Returns:
            int: Actual number of basis functions.
        """
        return self.get_num_basis()

    def get_fields(self, x: float, y: float, z: float) -> tuple:
        """Get the electromagnetic fields at a point.
        
        Args:
            x (float): X coordinate.
            y (float): Y coordinate.
            z (float): Z coordinate.
            
        Returns:
            tuple: Electric and magnetic field components.
        """
        E,H = self._S4_simu.GetFields(x, y, z)
        E = np.asarray(E)
        H = np.asarray(H)
        return E, H

    def get_fields_on_grid(self, z: float, num_samples: int) -> tuple:
        """Get the electromagnetic fields on a grid.
        
        Args:
            z (float): Z coordinate.
            num_samples (int): Number of samples in each direction.
            
        Returns:
            tuple: Electric and magnetic field arrays.
        """
        E, H = self._S4_simu.GetFieldsOnGrid(z, num_samples, Format="Array")
        # Convert to numpy arrays if they aren't already
        E = np.asarray(E)
        H = np.asarray(H)
        return E, H

    def get_s_matrix_determinant(self) -> complex:
        """Get the S-matrix determinant.
        
        Returns:
            complex: S-matrix determinant.
        """
        return self._S4_simu.GetSMatrixDeterminant()

    def copy(self) -> "Simulation":
        """Create a copy of the simulation.
        
        Returns:
            Simulation: A copy of the simulation.
        """
        # Create a new simulation with the same parameters
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
        # Clone the underlying S4 simulation
        sim_copy._S4_simu = self._S4_simu.Clone()
        # Copy the layers dictionary
        sim_copy.layers = self.layers.copy()
        return sim_copy

    def save_solution(self, filename: str) -> None:
        """Save the solution to a file.
        
        Args:
            filename (str): Name of the file to save.
        """
        self._S4_simu.SaveSolution(filename)

    def load_solution(self, filename: str) -> None:
        """Load a solution from a file.
        
        Args:
            filename (str): Name of the file to load.
        """
        self._S4_simu.LoadSolution(filename)


class SpectrumSampler:
    """Spectrum sampler class for adaptive sampling of spectra."""
    def __init__(
        self,
        start: float,
        stop: float,
        initial_num_points: int = 33,
        range_threshold: float = 0.001,
        max_bend: float = 10,
        minimum_spacing: float = 1e-6,
        parallelize: bool = False,
    ):
        """Initialize a spectrum sampler.
        
        Args:
            start (float): Start frequency.
            stop (float): Stop frequency.
            initial_num_points (int): Initial number of points.
            range_threshold (float): Range threshold.
            max_bend (float): Maximum bend angle in degrees.
            minimum_spacing (float): Minimum spacing between points.
            parallelize (bool): Whether to parallelize computation.
            
        Raises:
            ValueError: If start >= stop.
        """
        if start >= stop:
            raise ValueError("Start frequency must be less than stop frequency.")
            
        self.initial_num_points = initial_num_points
        self.range_threshold = range_threshold
        self.max_bend = max_bend
        self.minimum_spacing = minimum_spacing
        self.parallelize = parallelize

        options = dict(
            InitialNumPoints=initial_num_points,
            RangeThreshold=range_threshold,
            MaxBend=np.cos(np.deg2rad(max_bend)),
            MinimumSpacing=minimum_spacing,
            Parallelize=parallelize,
        )
        self.start = start
        self.stop = stop
        self._sampler = _S4.NewSpectrumSampler(start, stop, **options)

    def get_frequency(self) -> float:
        """Get the current frequency.
        
        Returns:
            float: Current frequency.
        """
        return self._sampler.GetFrequency()

    def get_frequencies(self) -> list:
        """Get all frequencies.
        
        Returns:
            list: List of frequencies.
        """
        return self._sampler.GetFrequencies()

    def is_done(self) -> bool:
        """Check if sampling is done.
        
        Returns:
            bool: True if sampling is done, False otherwise.
        """
        return self._sampler.IsDone()

    def is_parallelized(self) -> bool:
        """Check if sampling is parallelized.
        
        Returns:
            bool: True if sampling is parallelized, False otherwise.
        """
        return self._sampler.IsParallelized()

    def submit_result(self, y: float) -> None:
        """Submit a result for the current frequency.
        
        Args:
            y (float): Result value.
        """
        self._sampler.SubmitResult(y)

    def submit_results(self, y: list) -> None:
        """Submit results for all frequencies.
        
        Args:
            y (list): List of result values.
        """
        self._sampler.SubmitResults(y)

    def get_spectrum(self) -> tuple:
        """Get the complete spectrum.
        
        Returns:
            tuple: Frequencies and corresponding spectrum values.
        """
        return self._sampler.GetSpectrum()
