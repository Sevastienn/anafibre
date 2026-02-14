"""
Basic test suite for Anafibre package.

This test suite provides basic coverage of core functionality including:
1. Creating a StepIndexFibre object with typical parameters
2. Computing a fundamental mode (HE11) for a fiber at a standard wavelength (1550 nm)
3. Asserting that the mode object has expected attributes (effective index, mode label, field calculation methods)
4. Checking that mode field calculations (E and H) run without exceptions for sample input arrays

Author: Test suite for Anafibre
"""

import pytest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from anafibre import StepIndexFibre, GuidedMode


class TestStepIndexFibre:
    """Test the StepIndexFibre class basic functionality."""
    
    def test_fiber_creation_with_refractive_indices(self):
        """Test creating a StepIndexFibre with typical refractive index parameters."""
        # Typical single-mode fiber parameters
        core_radius = 250e-9  # 250 nm
        n_core = 2.00  # Core refractive index
        n_clad = 1.33  # Cladding refractive index
        
        fiber = StepIndexFibre(core_radius=core_radius, n_core=n_core, n_clad=n_clad)
        
        # Verify basic attributes
        assert fiber.core_radius == core_radius
        assert hasattr(fiber, 'n_core')
        assert hasattr(fiber, 'n_clad')
        assert hasattr(fiber, 'V')
        
        # Test V-parameter calculation at 500 nm
        wavelength = 500e-9
        V = fiber.V(wavelength)
        assert isinstance(V, (float, np.floating))
        assert V > 0  # V-parameter should be positive
        
    def test_fiber_creation_with_permittivity(self):
        """Test creating a StepIndexFibre with direct permittivity parameters."""
        core_radius = 250e-9
        eps_core = 2.00**2  # ε = n²
        eps_clad = 1.33**2
        
        fiber = StepIndexFibre(core_radius=core_radius, eps_core=eps_core, eps_clad=eps_clad)
        
        assert fiber.core_radius == core_radius
        assert hasattr(fiber, 'eps_core')
        assert hasattr(fiber, 'eps_clad')
        
        # Test that refractive index methods work
        wavelength = 500e-9
        n_core = fiber.n_core(wavelength)
        n_clad = fiber.n_clad(wavelength)
        
        assert np.isclose(n_core, 2.00, rtol=1e-6)
        assert np.isclose(n_clad, 1.33, rtol=1e-6)


class TestGuidedMode:
    """Test the GuidedMode class and mode computation."""
    
    @pytest.fixture
    def typical_fiber(self):
        """Create a typical single-mode step-index fiber for testing."""
        return StepIndexFibre(core_radius=250e-9, n_core=2.00, n_clad=1.33)
    
    def test_fundamental_mode_creation(self, typical_fiber):
        """Test computing a fundamental mode (HE11) for a fiber at 500 nm."""
        wavelength = 500e-9  # 500 nm (visible wavelength)
        
        # Create HE11 mode (fundamental mode: ell=1, n=1)
        mode = typical_fiber.HE(ell=1, n=1, wl=wavelength)
        
        # Verify mode was created successfully
        assert isinstance(mode, GuidedMode)
        assert mode.ell == 1
        assert mode.m == 1
        assert mode.wavelength == wavelength
        
    def test_mode_attributes(self, typical_fiber):
        """Test that the mode object has expected attributes."""
        wavelength = 500e-9
        mode = typical_fiber.HE(ell=1, n=1, wl=wavelength)
        
        # Check required attributes exist
        assert hasattr(mode, 'neff'), "Mode should have effective index (neff)"
        assert hasattr(mode, 'mode_label'), "Mode should have mode_label method"
        assert hasattr(mode, 'E'), "Mode should have E field method"
        assert hasattr(mode, 'H'), "Mode should have H field method"
        assert hasattr(mode, 'V'), "Mode should have V parameter"
        assert hasattr(mode, 'b'), "Mode should have normalized propagation constant b"
        
        # Check attribute types and values
        assert isinstance(mode.neff, (float, np.floating, complex)), "neff should be numeric"
        assert isinstance(mode.V, (float, np.floating)), "V should be numeric"
        assert isinstance(mode.b, (float, np.floating)), "b should be numeric"
        
        # Physical constraints
        assert 0 < mode.b < 1, "Normalized propagation constant b should be between 0 and 1"
        assert mode.neff > typical_fiber.n_clad(wavelength), "Effective index should be > cladding index"
        assert mode.neff < typical_fiber.n_core(wavelength), "Effective index should be < core index"
        
    def test_mode_label(self, typical_fiber):
        """Test mode labeling functionality."""
        wavelength = 500e-9
        mode = typical_fiber.HE(ell=1, n=1, wl=wavelength)
        
        label = mode.mode_label()
        assert isinstance(label, str), "Mode label should be a string"
        assert "HE" in label, "HE11 mode label should contain 'HE'"
        assert "1" in label, "HE11 mode label should contain '1'"
        
    def test_field_calculations_run_without_exceptions(self, typical_fiber):
        """Test that mode field calculations (E and H) run without exceptions for sample input arrays."""
        wavelength = 500e-9
        mode = typical_fiber.HE(ell=1, n=1, wl=wavelength)
        
        # Create sample input arrays - various radial and azimuthal positions
        rho_test = np.array([1e-6, 2e-6, 4e-6, 6e-6])  # Mix of core and cladding positions
        phi_test = np.array([0, np.pi/4, np.pi/2, np.pi])  # Various azimuthal angles
        z_test = 0  # Single z position
        
        # Test E field calculation
        try:
            E_field = mode.E(rho=rho_test, phi=phi_test, z=z_test)
            assert E_field is not None, "E field should not be None"
            assert isinstance(E_field, np.ndarray), "E field should be numpy array"
            assert E_field.shape[-1] == 3, "E field should have 3 components (x, y, z)"
        except Exception as e:
            pytest.fail(f"E field calculation failed: {e}")
        
        # Test H field calculation  
        try:
            H_field = mode.H(rho=rho_test, phi=phi_test, z=z_test)
            assert H_field is not None, "H field should not be None"
            assert isinstance(H_field, np.ndarray), "H field should be numpy array"
            assert H_field.shape[-1] == 3, "H field should have 3 components (x, y, z)"
        except Exception as e:
            pytest.fail(f"H field calculation failed: {e}")
            
        # Test with Cartesian coordinates as well
        try:
            x_test = np.array([1e-6, -2e-6])
            y_test = np.array([2e-6, 1e-6])
            
            E_field_cart = mode.E(x=x_test, y=y_test, z=z_test)
            H_field_cart = mode.H(x=x_test, y=y_test, z=z_test)
            
            assert E_field_cart is not None
            assert H_field_cart is not None
            assert isinstance(E_field_cart, np.ndarray)
            assert isinstance(H_field_cart, np.ndarray)
        except Exception as e:
            pytest.fail(f"Cartesian field calculation failed: {e}")
    
    def test_different_mode_types(self, typical_fiber):
        """Test that different mode types can be created."""
        wavelength = 500e-9
        
        # Test TE mode creation
        try:
            te_mode = typical_fiber.TE(n=1, wl=wavelength)
            assert isinstance(te_mode, GuidedMode)
            assert te_mode.ell == 0
            assert "TE" in te_mode.mode_label()
        except Exception as e:
            pytest.fail(f"Cartesian field calculation failed: {e}")
            
        # Test TM mode creation
        try:
            tm_mode = typical_fiber.TM(n=1, wl=wavelength)
            assert isinstance(tm_mode, GuidedMode)
            assert tm_mode.ell == 0
            assert "TM" in tm_mode.mode_label()
        except Exception as e:
            pytest.fail(f"Cartesian field calculation failed: {e}")
    
    def test_field_normalization_option(self, typical_fiber):
        """Test that field calculations work with normalization option."""
        wavelength = 500e-9
        mode = typical_fiber.HE(ell=1, n=1, wl=wavelength)
        try:
            P=mode.Power()  
            assert 0.99 < P < 1.00 # Power should be normalized to 1W
        except Exception as e:
            pytest.fail(f"Field normalization test failed: {e}")


class TestIntegrationBasics:
    """Integration tests for basic workflow."""
    
    def test_basic_workflow(self):
        """Test the basic workflow: create fiber -> create mode -> calculate fields."""
        # Step 1: Create fiber
        fiber = StepIndexFibre(core_radius=250e-9, n_core=2.00, n_clad=1.33)
        
        # Step 2: Create mode  
        wavelength = 500e-9
        mode = fiber.HE(ell=1, n=1, wl=wavelength)
        
        # Step 3: Calculate some properties
        V = fiber.V(wavelength)
        neff = mode.neff
        label = mode.mode_label()
        
        # Step 4: Calculate fields at a few points
        rho = np.array([1e-6, 3e-6, 5e-6])
        phi = np.array([0, np.pi/4, np.pi/2])
        
        E = mode.E(rho=rho, phi=phi)
        H = mode.H(rho=rho, phi=phi)
        
        # Basic sanity checks
        assert V > 0
        assert 1.33 < neff < 2.00  # Should be between cladding and core index
        assert isinstance(label, str)
        assert E.shape == H.shape
        assert E.shape[-1] == 3  # 3 field components
        
        print(f"Workflow test passed: V={V:.3f}, neff={neff:.6f}, label={label}")


if __name__ == "__main__":
    # Allow running tests directly with python
    pytest.main([__file__])