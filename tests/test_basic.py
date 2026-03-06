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
from pathlib import Path
from scipy.special import jv

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from anafibre import StepIndexFibre, GuidedMode, RefractiveIndexMaterial
from anafibre.plotting import plot_complex_field, plot_complex_field_polar


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
        wl = 500e-9
        V = fiber.V(wl)
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
        wl = 500e-9
        n_core = fiber.n_core(wl)
        n_clad = fiber.n_clad(wl)
        
        assert np.isclose(n_core, 2.00, rtol=1e-6)
        assert np.isclose(n_clad, 1.33, rtol=1e-6)

    def test_phi_alpha_matches_dispersion_relation(self):
        """Phi^eps and Phi^mu should reproduce the dispersion residual form."""
        fiber = StepIndexFibre(core_radius=250e-9, n_core=2.00, n_clad=1.33)
        ell = 1
        wl = 500e-9
        V = fiber.V(wl)
        b = 0.4

        eps1 = fiber._eval(fiber.eps_core, wl)
        eps2 = fiber._eval(fiber.eps_clad, wl)
        mu1 = fiber._eval(fiber.mu_core, wl)
        mu2 = fiber._eval(fiber.mu_clad, wl)
        phi_eps = fiber.Phi(ell=ell, b=b, V=V, alpha=[eps1, eps2])
        phi_mu = fiber.Phi(ell=ell, b=b, V=V, alpha=[mu1, mu2])
        ne = np.sqrt(b * fiber.n_core(wl) ** 2 + (1 - b) * fiber.n_clad(wl) ** 2)
        u = V * np.sqrt(1 - b)
        J = jv(ell, u)
        F_expected = (J ** 2) * (phi_eps * phi_mu - (ell * ne) ** 2) * (b * (1 - b)) / (ell ** 2)
        F_val = fiber.F(ell=ell, b=b, V=V)

        assert np.isfinite(phi_eps)
        assert np.isfinite(phi_mu)
        assert np.isclose(F_val, F_expected, rtol=1e-10, atol=1e-12)

    def test_phi_alpha_rejects_invalid_selector(self):
        """Phi helper should validate alpha selector."""
        fiber = StepIndexFibre(core_radius=250e-9, n_core=2.00, n_clad=1.33)
        with pytest.raises(ValueError, match="alpha must be a length-2 sequence"):
            fiber.Phi(ell=1, b=0.3, wl=500e-9, alpha="bad")

    def test_phi_alpha_accepts_custom_pair(self):
        """Custom [alpha1, alpha2] should be accepted and match eps-case if equal."""
        fiber = StepIndexFibre(core_radius=250e-9, n_core=2.00, n_clad=1.33)
        ell = 1
        wl = 500e-9
        b = 0.3
        eps1 = fiber._eval(fiber.eps_core, wl)
        eps2 = fiber._eval(fiber.eps_clad, wl)
        phi_custom = fiber.Phi(ell=ell, b=b, wl=wl, alpha=[eps1, eps2])
        phi_eps = fiber.Phi(ell=ell, b=b, wl=wl, alpha=[eps1, eps2])
        assert np.isclose(phi_custom, phi_eps, rtol=1e-12, atol=0.0)

    def test_phi_alpha_accepts_callable_pair(self):
        """Callable alpha entries should be evaluated as f(wl)."""
        fiber = StepIndexFibre(core_radius=250e-9, n_core=2.00, n_clad=1.33)
        ell = 1
        wl = 500e-9
        b = 0.3
        eps1 = fiber._eval(fiber.eps_core, wl)
        eps2 = fiber._eval(fiber.eps_clad, wl)
        phi_val = fiber.Phi(ell=ell, b=b, wl=wl, alpha=[lambda _wl: eps1, lambda _wl: eps2])
        phi_ref = fiber.Phi(ell=ell, b=b, wl=wl, alpha=[eps1, eps2])
        assert np.isclose(phi_val, phi_ref, rtol=1e-12, atol=0.0)


class TestGuidedMode:
    """Test the GuidedMode class and mode computation."""
    
    @pytest.fixture
    def typical_fiber(self):
        """Create a typical single-mode step-index fiber for testing."""
        return StepIndexFibre(core_radius=250e-9, n_core=2.00, n_clad=1.33)
    
    def test_fundamental_mode_creation(self, typical_fiber):
        """Test computing a fundamental mode (HE11) for a fiber at 500 nm."""
        wl = 500e-9  # 500 nm (visible wavelength)
        
        # Create HE11 mode (fundamental mode: ell=1, n=1)
        mode = typical_fiber.HE(ell=1, n=1, wl=wl)
        
        # Verify mode was created successfully
        assert isinstance(mode, GuidedMode)
        assert mode.ell == 1
        assert mode.m == 1
        assert mode.wl == wl
        
    def test_mode_attributes(self, typical_fiber):
        """Test that the mode object has expected attributes."""
        wl = 500e-9
        mode = typical_fiber.HE(ell=1, n=1, wl=wl)
        
        # Check required attributes exist
        assert hasattr(mode, 'neff'), "Mode should have effective index (neff)"
        assert hasattr(mode, 'mode_label'), "Mode should have mode_label attribute"
        assert hasattr(mode, 'E'), "Mode should have E field method"
        assert hasattr(mode, 'H'), "Mode should have H field method"
        assert hasattr(mode, 'V'), "Mode should have V parameter"
        assert hasattr(mode, 'b'), "Mode should have normalised propagation constant b"
        
        # Check attribute types and values
        assert isinstance(mode.neff, (float, np.floating, complex)), "neff should be numeric"
        assert isinstance(mode.V, (float, np.floating)), "V should be numeric"
        assert isinstance(mode.b, (float, np.floating)), "b should be numeric"
        
        # Physical constraints
        assert 0 < mode.b < 1, "normalised propagation constant b should be between 0 and 1"
        assert mode.neff > typical_fiber.n_clad(wl), "Effective index should be > cladding index"
        assert mode.neff < typical_fiber.n_core(wl), "Effective index should be < core index"
        
    def test_mode_label(self, typical_fiber):
        """Test mode labeling functionality."""
        wl = 500e-9
        mode = typical_fiber.HE(ell=1, n=1, wl=wl)
        
        label = mode.mode_label
        assert isinstance(label, str), "Mode label should be a string"
        assert "HE" in label, "HE11 mode label should contain 'HE'"
        assert "1" in label, "HE11 mode label should contain '1'"
        
    def test_field_calculations_run_without_exceptions(self, typical_fiber):
        """Test that mode field calculations (E and H) run without exceptions for sample input arrays."""
        wl = 500e-9
        mode = typical_fiber.HE(ell=1, n=1, wl=wl)
        
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
        wl = 500e-9
        
        # Test TE mode creation
        try:
            te_mode = typical_fiber.TE(n=1, wl=wl)
            assert isinstance(te_mode, GuidedMode)
            assert te_mode.ell == 0
            assert "TE" in te_mode.mode_label
        except Exception as e:
            pytest.fail(f"Cartesian field calculation failed: {e}")
            
        # Test TM mode creation
        try:
            tm_mode = typical_fiber.TM(n=1, wl=wl)
            assert isinstance(tm_mode, GuidedMode)
            assert tm_mode.ell == 0
            assert "TM" in tm_mode.mode_label
        except Exception as e:
            pytest.fail(f"Cartesian field calculation failed: {e}")
    
    def test_field_normalization_option(self, typical_fiber):
        """Test that field calculations work with normalization option."""
        wl = 500e-9
        mode = typical_fiber.HE(ell=1, n=1, wl=wl)
        try:
            P=mode.Power()  
            assert 0.99 < P < 1.00 # Power should be normalised to 1W
        except Exception as e:
            pytest.fail(f"Field normalization test failed: {e}")

    def test_stokes_parameters_are_exposed_and_consistent(self, typical_fiber):
        """GuidedMode should expose unnormalised S0..S3 and normalised stokes()."""
        wl = 500e-9
        a_plus = 1.0 + 0.0j
        a_minus = 0.3 + 0.4j
        mode = typical_fiber.HE(ell=1, n=1, wl=wl, a_plus=a_plus, a_minus=a_minus)

        S0_ref = np.abs(a_plus) ** 2 + np.abs(a_minus) ** 2
        S1_ref = 2 * np.real(a_plus * np.conj(a_minus))
        S2_ref = 2 * np.imag(a_plus * np.conj(a_minus))
        S3_ref = np.abs(a_plus) ** 2 - np.abs(a_minus) ** 2

        assert np.isclose(mode.S0, S0_ref)
        assert np.isclose(mode.S1, S1_ref)
        assert np.isclose(mode.S2, S2_ref)
        assert np.isclose(mode.S3, S3_ref)

        S0n, s1, s2, s3 = mode.stokes(normalised=True)
        assert np.isclose(S0n, S0_ref)
        assert np.isclose(s1, S1_ref / S0_ref)
        assert np.isclose(s2, S2_ref / S0_ref)
        assert np.isclose(s3, S3_ref / S0_ref)

    def test_stokes_parameters_account_for_te_tm_modes(self, typical_fiber):
        """TE/TM modes should map to fixed Stokes orientation conventions."""
        wl = 500e-9
        te = typical_fiber.TE(n=1, wl=wl)
        tm = typical_fiber.TM(n=1, wl=wl)

        S0_te, s1_te, s2_te, s3_te = te.stokes(normalised=True)
        S0_tm, s1_tm, s2_tm, s3_tm = tm.stokes(normalised=True)
        assert np.isclose(s1_te, -1.0) and np.isclose(s2_te, 0.0) and np.isclose(s3_te, 0.0)
        assert np.isclose(s1_tm, 1.0) and np.isclose(s2_tm, 0.0) and np.isclose(s3_tm, 0.0)

        # Unnormalised values must equal S0 * normalised components
        assert np.isclose(te.S1, -S0_te) and np.isclose(te.S2, 0.0) and np.isclose(te.S3, 0.0)
        assert np.isclose(tm.S1, S0_tm) and np.isclose(tm.S2, 0.0) and np.isclose(tm.S3, 0.0)


class TestIntegrationBasics:
    """Integration tests for basic workflow."""
    
    def test_basic_workflow(self):
        """Test the basic workflow: create fiber -> create mode -> calculate fields."""
        # Step 1: Create fiber
        fiber = StepIndexFibre(core_radius=250e-9, n_core=2.00, n_clad=1.33)
        
        # Step 2: Create mode  
        wl = 500e-9
        mode = fiber.HE(ell=1, n=1, wl=wl)
        
        # Step 3: Calculate some properties
        V = fiber.V(wl)
        neff = mode.neff
        label = mode.mode_label
        
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


class TestOptionalExtras:
    """Tests for optional extras: units and refractiveindex integrations."""

    def test_plotting_coord_unit_defaults_are_optional_safe(self):
        """Plotting APIs should not require astropy units at import time."""
        assert plot_complex_field.__kwdefaults__["coord_unit"] is None
        assert plot_complex_field_polar.__kwdefaults__["coord_unit"] is None

    def test_units_quantity_inputs_match_plain_float(self):
        """If astropy is installed, Quantity and float inputs should agree."""
        units = pytest.importorskip("astropy.units")

        fiber = StepIndexFibre(core_radius=250e-9, n_core=2.00, n_clad=1.33)
        wl_float = 500e-9
        wl_qty = 500 * units.nm

        assert np.isclose(fiber.V(wl_qty), fiber.V(wl_float), rtol=1e-12)
        assert np.isclose(fiber.n_core(wl_qty), fiber.n_core(wl_float), rtol=1e-12)
        assert np.isclose(fiber.n_clad(wl_qty), fiber.n_clad(wl_float), rtol=1e-12)

    def test_upstream_refractiveindex_material_is_wrapped(self):
        """If refractiveindex + local DB are available, real upstream objects are wrapped safely."""
        ri = pytest.importorskip("refractiveindex.refractiveindex")
        db_path = Path.home() / ".refractiveindex.info-database"
        if not db_path.exists():
            pytest.skip("Local refractiveindex database not found; skipping real upstream material test.")

        upstream = ri.RefractiveIndexMaterial(
            "main",
            "Si3N4",
            "Luke",
            db_path=db_path,
            auto_download=False,
        )

        with pytest.warns(UserWarning, match="auto-wrapping"):
            fiber = StepIndexFibre(core_radius=250e-9, core=upstream, clad=upstream)

        assert isinstance(fiber.core, RefractiveIndexMaterial)
        assert isinstance(fiber.clad, RefractiveIndexMaterial)
        assert np.isfinite(fiber.n_core(500e-9))
        assert np.isfinite(fiber.n_clad(500e-9))

    def test_upstream_and_anafibre_material_give_same_fiber_indices(self):
        """Real upstream and anafibre wrappers should produce the same fibre index values."""
        ri = pytest.importorskip("refractiveindex.refractiveindex")
        db_path = Path.home() / ".refractiveindex.info-database"
        if not db_path.exists():
            pytest.skip("Local refractiveindex database not found; skipping material equivalence test.")

        upstream = ri.RefractiveIndexMaterial(
            "main",
            "Si3N4",
            "Luke",
            db_path=db_path,
            auto_download=False,
        )
        local = RefractiveIndexMaterial(
            "main",
            "Si3N4",
            "Luke",
            db_path=db_path,
            auto_download=False,
        )

        with pytest.warns(UserWarning, match="auto-wrapping"):
            fiber_upstream = StepIndexFibre(core_radius=250e-9, core=upstream, clad=upstream)
        fiber_local = StepIndexFibre(core_radius=250e-9, core=local, clad=local)

        wavelengths = np.array([450e-9, 500e-9, 650e-9, 1.0e-6])
        n_up = fiber_upstream.n_core(wavelengths)
        n_local = fiber_local.n_core(wavelengths)
        assert np.allclose(n_up, n_local, rtol=1e-12, atol=0.0)

    def test_material_wrapper_uses_meter_api_for_all_methods(self):
        """Wrapper methods should accept meters and convert to nm only internally."""
        class FakeUpstreamMaterial:
            def __init__(self):
                self.n_calls = []
                self.k_calls = []
                self.eps_calls = []

            @staticmethod
            def _as_like_input(wl_m, val):
                arr = np.asarray(wl_m)
                out = np.full(arr.shape, val, dtype=np.result_type(val, float))
                return out.item() if out.shape == () else out

            def get_refractive_index(self, wl, unit="m"):
                assert unit == "m"
                self.n_calls.append(np.array(wl, copy=True))
                return self._as_like_input(wl, 1.5)

            def get_extinction_coefficient(self, wl, unit="m"):
                assert unit == "m"
                self.k_calls.append(np.array(wl, copy=True))
                return self._as_like_input(wl, 0.02)

            def get_epsilon(self, wl, unit="m", exp_type="exp_minus_i_omega_t"):
                assert unit == "m"
                self.eps_calls.append(np.array(wl, copy=True))
                return self._as_like_input(wl, (1.5 + 1j * 0.02) ** 2)

            def get_wl_range(self, unit="m"):
                if unit == "m":
                    return (0.4e-6, 1.6e-6)
                if unit == "um":
                    return (0.4, 1.6)
                raise ValueError("Unsupported unit")

        fake = FakeUpstreamMaterial()
        mat = RefractiveIndexMaterial.from_upstream(fake)

        assert np.isclose(mat.rangeMin, 0.4e-6)
        assert np.isclose(mat.rangeMax, 1.6e-6)

        n = mat.get_refractive_index(500e-9)
        k = mat.get_extinction_coefficient(500e-9)
        eps = mat.get_eps(500e-9, real=False)

        assert np.isclose(n, 1.5)
        assert np.isclose(k, 0.02)
        assert np.isclose(eps, (1.5 + 1j * 0.02) ** 2)
        assert np.isclose(fake.n_calls[-1], 500e-9)
        assert np.isclose(fake.k_calls[0], 500e-9)

    def test_material_wrapper_accepts_astropy_units_in_all_methods(self):
        """If astropy is installed, Quantity inputs should work for n/k/eps APIs."""
        units = pytest.importorskip("astropy.units")

        class FakeUpstreamMaterial:
            def __init__(self):
                self.n_calls = []
                self.k_calls = []
                self.eps_calls = []

            @staticmethod
            def _as_like_input(wl_m, val):
                arr = np.asarray(wl_m)
                out = np.full(arr.shape, val, dtype=np.result_type(val, float))
                return out.item() if out.shape == () else out

            def get_refractive_index(self, wl, unit="m"):
                assert unit == "m"
                self.n_calls.append(np.array(wl, copy=True))
                return self._as_like_input(wl, 1.7)

            def get_extinction_coefficient(self, wl, unit="m"):
                assert unit == "m"
                self.k_calls.append(np.array(wl, copy=True))
                return self._as_like_input(wl, 0.01)

            def get_epsilon(self, wl, unit="m", exp_type="exp_minus_i_omega_t"):
                assert unit == "m"
                self.eps_calls.append(np.array(wl, copy=True))
                return self._as_like_input(wl, (1.7 + 1j * 0.01) ** 2)

            def get_wl_range(self, unit="m"):
                if unit == "m":
                    return (0.4e-6, 1.6e-6)
                if unit == "um":
                    return (0.4, 1.6)
                raise ValueError("Unsupported unit")

        fake = FakeUpstreamMaterial()
        mat = RefractiveIndexMaterial.from_upstream(fake)
        wl = np.array([500, 650]) * units.nm

        n = mat.get_refractive_index(wl)
        k = mat.get_extinction_coefficient(wl)
        eps = mat.get_eps(wl, real=False)

        assert np.allclose(n, [1.7, 1.7])
        assert np.allclose(k, [0.01, 0.01])
        assert np.allclose(eps, (1.7 + 1j * 0.01) ** 2)
        assert np.allclose(fake.n_calls[-1], [500e-9, 650e-9])
        assert np.allclose(fake.k_calls[0], [500e-9, 650e-9])


if __name__ == "__main__":
    # Allow running tests directly with python
    pytest.main([__file__])
