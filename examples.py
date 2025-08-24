#!/usr/bin/env python3
"""
Basic example demonstrating Anafibre usage.

This script shows how to:
1. Create a step-index fiber
2. Calculate fiber parameters  
3. Find guided modes
4. Display mode information
"""

import numpy as np

def basic_fiber_example():
    """Basic example of fiber analysis."""
    print("Anafibre Basic Example")
    print("=" * 40)
    
    # Import anafibre
    import anafibre as af
    
    # Define fiber parameters
    print("1. Creating step-index fiber...")
    core_radius = 4.1e-6    # 4.1 μm core radius
    core_index = 1.46       # Core refractive index
    clad_index = 1.45       # Cladding refractive index
    wavelength = 1550e-9    # 1550 nm wavelength
    
    # Create fiber
    fiber = af.StepIndexFibre(
        core_radius=core_radius,
        n_core=core_index,
        n_clad=clad_index
    )
    
    print(f"   Core radius: {core_radius*1e6:.1f} μm")
    print(f"   Core index: {core_index}")
    print(f"   Cladding index: {clad_index}")
    print(f"   Wavelength: {wavelength*1e9:.0f} nm")
    
    # Calculate basic parameters
    print("\n2. Calculating fiber parameters...")
    V = fiber.V(wavelength)
    
    # Calculate numerical aperture manually
    n_core = fiber.n_core(wavelength)
    n_clad = fiber.n_clad(wavelength)
    numerical_aperture = np.sqrt(n_core**2 - n_clad**2)
    
    print(f"   V parameter: {V:.3f}")
    print(f"   Numerical aperture: {numerical_aperture:.3f}")
    
    # Estimate number of modes
    num_modes_estimate = V**2 / 2 if V > 2.405 else 1
    print(f"   Estimated number of modes: ~{num_modes_estimate:.0f}")
    
    # Mode classification
    if V < 2.405:
        print("   Fiber type: Single-mode")
    else:
        print("   Fiber type: Multi-mode")
    
    # Find fundamental mode
    print("\n3. Finding fundamental mode...")
    try:
        # For single mode fiber, fundamental mode is HE11
        fundamental_mode = fiber.HE(ell=1, n=1, wl=wavelength)
        print(f"   Effective index: {fundamental_mode.neff:.6f}")
        print(f"   Propagation constant: {fundamental_mode.kz:.2e} rad/m")
        print(f"   Mode label: {fundamental_mode.mode_label()}")
        
        # Calculate field at fiber center
        E_center = fundamental_mode.E(rho=0, phi=0)
        print(f"   |E| at center: {np.abs(E_center).max():.3e}")
        
    except Exception as e:
        print(f"   Could not find fundamental mode: {e}")
    
    # Try to find multiple modes if multimode
    if V > 2.405:
        print("\n4. Finding multiple modes...")
        try:
            modes = []
            # Try to find first few modes
            try:
                modes.append(fiber.HE(ell=1, n=1, wl=wavelength))  # HE11
            except:
                pass
            try:
                modes.append(fiber.TE(n=1, wl=wavelength))  # TE01
            except:
                pass
            try:
                modes.append(fiber.TM(n=1, wl=wavelength))  # TM01
            except:
                pass
            try:
                modes.append(fiber.HE(ell=2, n=1, wl=wavelength))  # HE21
            except:
                pass
            try:
                modes.append(fiber.HE(ell=1, n=2, wl=wavelength))  # HE12
            except:
                pass
                
            print(f"   Found {len(modes)} modes:")
            for i, mode in enumerate(modes):
                print(f"     Mode {i+1}: {mode.mode_label()}, neff={mode.neff:.6f}")
        except Exception as e:
            print(f"   Could not find multiple modes: {e}")
    
    print("\n" + "=" * 40)
    print("Example completed!")

def field_calculation_example():
    """Example showing field calculations."""
    print("\nAdvanced Example: Field Calculations")
    print("=" * 40)
    
    import anafibre as af
    import matplotlib.pyplot as plt
    
    # Create a single-mode fiber
    fiber = af.StepIndexFibre(
        core_radius=4.1e-6,
        n_core=1.46,
        n_clad=1.45
    )
    
    wavelength = 1550e-9
    mode = fiber.HE(ell=1, n=1, wl=wavelength)  # Fundamental HE11 mode
    
    print("Calculating mode fields...")
    
    # Create coordinate grids
    r_max = 3 * fiber.core_radius
    r = np.linspace(0, r_max, 50)
    phi = np.linspace(0, 2*np.pi, 50)
    R, Phi = np.meshgrid(r, phi)
    
    # Calculate fields
    E_field = mode.E(rho=R, phi=Phi)
    H_field = mode.H(rho=R, phi=Phi)
    
    print(f"   Field array shape: {E_field.shape}")
    print(f"   Max |E|: {np.abs(E_field).max():.3e}")
    print(f"   Max |H|: {np.abs(H_field).max():.3e}")
    
    # Power calculation
    power = mode.Power(rho=R, phi=Phi)
    print(f"   Power distribution calculated, max: {power.max():.3e}")

if __name__ == "__main__":
    try:
        basic_fiber_example()
        
        # Only run advanced example if matplotlib is available
        try:
            import matplotlib.pyplot as plt
            field_calculation_example()
        except ImportError:
            print("\nNote: matplotlib not available, skipping field calculation example")
            
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()