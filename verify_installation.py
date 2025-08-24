#!/usr/bin/env python3
"""
Verification script for Anafibre installation.
Run this to check if the package was installed correctly.
"""

def test_imports():
    """Test that all main components can be imported."""
    print("Testing imports...")
    
    try:
        import anafibre
        print("✓ anafibre imported successfully")
        print(f"  Version: {anafibre.__version__}")
        print(f"  Author: {anafibre.__author__}")
    except ImportError as e:
        print(f"✗ Failed to import anafibre: {e}")
        return False
    
    try:
        from anafibre import StepIndexFibre, GuidedMode
        print("✓ Core classes imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import core classes: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality."""
    print("\nTesting basic functionality...")
    
    try:
        import anafibre
        
        # Create a simple step-index fiber
        fiber = anafibre.StepIndexFibre(
            core_radius=4.1e-6,  # 4.1 μm
            n_core=1.46,         # Core index
            n_clad=1.45          # Cladding index
        )
        print("✓ StepIndexFibre created successfully")
        print(f"  Core radius: {fiber.core_radius:.1e} m")
        print(f"  Core index: {fiber.n_core(1550e-9)}")
        print(f"  Clad index: {fiber.n_clad(1550e-9)}")
        
        # Calculate V parameter
        wavelength = 1550e-9  # 1550 nm
        V = fiber.V(wavelength)
        print(f"  V parameter at 1550 nm: {V:.3f}")
        
        # Test fundamental mode creation
        mode = fiber.HE(ell=1, n=1, wl=wavelength)
        print(f"  Fundamental mode (HE11): neff={mode.neff:.6f}")
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False
    
    return True

def test_optional_dependencies():
    """Test optional dependencies."""
    print("\nTesting optional dependencies...")
    
    try:
        import astropy.units as u
        print("✓ astropy available (units support enabled)")
    except ImportError:
        print("○ astropy not available (units support disabled)")
    
    try:
        import refractiveindex
        print("✓ refractiveindex available (refractive index database enabled)")
    except ImportError:
        print("○ refractiveindex not available (refractive index database disabled)")

def main():
    """Run all tests."""
    print("Anafibre Installation Verification")
    print("=" * 40)
    
    success = True
    success &= test_imports()
    success &= test_basic_functionality()
    test_optional_dependencies()
    
    print("\n" + "=" * 40)
    if success:
        print("✓ All tests passed! Anafibre is ready to use.")
    else:
        print("✗ Some tests failed. Please check your installation.")
    
    return success

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)