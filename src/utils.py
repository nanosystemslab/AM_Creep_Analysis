"""
Utility Functions for Nanoindentation Creep Analysis
=====================================================

This module consolidates all utility functions for creep compliance calculations,
probe geometry conversions, Poisson ratio calculations, and area functions.

Key Features:
    - Compliance calculators for different probe geometries
    - Probe information registry
    - Flat punch and conical area calculations
    - Poisson ratio calculations from bulk compression tests
    - Material property conversions

References:
    - Harding & Sneddon (1945): Flat punch elastic solution
    - Oliver & Pharr (1992): Berkovich/pyramidal indentation
    - Hertz (1881): Spherical contact
    - Thapa & Cheng (2024): Flat punch viscoelastic solution
    - Peng et al. (2015): Conical indenter compliance, Polymer Testing 43:38-43

Author: Creep Analysis Framework
Date: 2024-2025
"""

import numpy as np
from typing import Tuple, Optional, Dict


# =============================================================================
# PROBE GEOMETRY CONSTANTS
# =============================================================================

# Standard indenter half-angles (degrees)
BERKOVICH_HALF_ANGLE = 70.3  # Berkovich equivalent cone half-angle
CONICAL_60_HALF_ANGLE = 30.0  # 60° cone angle = 30° half-angle
CONICAL_90_HALF_ANGLE = 45.0  # 90° included angle = 45° half-angle

# Berkovich geometry factor for area calculation
# A = 24.5 * h^2 (for perfect Berkovich)
BERKOVICH_AREA_COEFF = 24.5


# =============================================================================
# FLAT PUNCH COMPLIANCE CALCULATIONS
# =============================================================================

def calculate_flat_punch_compliance(
    displacement_nm: np.ndarray,
    load_uN: np.ndarray,
    area_m2: float,
    poisson_ratio: float = 0.43
) -> np.ndarray:
    """
    Calculate shear creep compliance for flat punch indentation (constant area).

    From Harding & Sneddon (1945) elastic solution for a flat cylindrical punch:
        F = 2R × E/(1-ν²) × h

    Via the elastic-viscoelastic correspondence principle, converting the
    tensile compliance D(t)=1/E to shear compliance J_s(t)=1/G=2(1+ν)/E:
        J_s(t) = 4R/(1-ν) × h(t)/P(t)

    This is consistent with the conical/Berkovich formula (Peng et al. 2015),
    which also returns shear creep compliance.

    For circular flat punch: R = sqrt(A/π)

    Args:
        displacement_nm: Displacement array (nm)
        load_uN: Load array (µN)
        area_m2: Flat punch contact area (m²)
        poisson_ratio: Poisson's ratio

    Returns:
        compliance_gpa_inv: Shear creep compliance J_s(t) in 1/GPa

    Example:
        >>> h = np.array([100, 110, 120])  # nm
        >>> P = np.array([1000, 1000, 1000])  # µN
        >>> A = 3.62e-10  # m²
        >>> J = calculate_flat_punch_compliance(h, P, A, nu=0.43)
    """
    # Calculate effective radius from area (for circular punch)
    R_m = np.sqrt(area_m2 / np.pi)  # m

    # Convert units
    displacement_m = displacement_nm * 1e-9  # nm to m
    load_N = load_uN * 1e-6  # µN to N

    # Calculate shear creep compliance using flat punch equation
    # J_s(t) = 4R/(1 - ν) × h(t)/P(t)
    # Result in 1/Pa
    compliance_Pa_inv = 4 * R_m / (1 - poisson_ratio) * displacement_m / load_N

    # Convert to 1/GPa
    compliance_GPa_inv = compliance_Pa_inv * 1e9  # 1/Pa to 1/GPa (1 Pa^-1 = 1e9 GPa^-1)

    return compliance_GPa_inv


def calculate_flat_punch_area(radius_m: float) -> float:
    """
    Calculate area of circular flat punch.

    Args:
        radius_m: Punch radius (m)

    Returns:
        area_m2: Contact area (m²)
    """
    return np.pi * radius_m**2


def radius_from_area(area_m2: float) -> float:
    """
    Calculate radius from circular area.

    Args:
        area_m2: Area (m²)

    Returns:
        radius_m: Radius (m)
    """
    return np.sqrt(area_m2 / np.pi)


# =============================================================================
# CONICAL/BERKOVICH COMPLIANCE CALCULATIONS
# =============================================================================

def calculate_conical_compliance(
    displacement_nm: np.ndarray,
    load_uN: np.ndarray,
    half_angle_deg: float = 70.3,
    poisson_ratio: float = 0.43
) -> np.ndarray:
    """
    Calculate creep compliance for conical/Berkovich indentation.

    From Peng et al. (2015) Polymer Testing 43:38-43, equation 2:
        J(t) = [4 tan α] / [π(1 - ν)F₀] × h²(t)

    Where:
        α = included half-angle of conical indenter
        ν = Poisson's ratio
        F₀ = applied load (constant during creep hold)
        h(t) = indentation depth during creep

    Standard half-angles:
        - Berkovich: 70.3° (equivalent cone angle)
        - Conical 60°: 30.0° half-angle (60° cone angle)
        - Conical 90°: 45.0° half-angle

    Args:
        displacement_nm: Displacement array h(t) (nm)
        load_uN: Load array F(t) (µN) - should be constant during creep hold
        half_angle_deg: Included half-angle α of cone (degrees)
        poisson_ratio: Poisson's ratio ν

    Returns:
        compliance_gpa_inv: Compliance J(t) in 1/GPa

    Reference:
        Peng et al. (2015), "Nanoindentation creep of nonlinear viscoelastic
        polypropylene," Polymer Testing 43:38-43, Eq. 2
    """
    # Convert units
    displacement_m = displacement_nm * 1e-9  # nm to m
    load_N = load_uN * 1e-6  # µN to N
    alpha_rad = np.radians(half_angle_deg)

    # Use mean load during hold (should be constant)
    F0_N = np.mean(load_N)

    # Calculate compliance using Peng equation 2
    # J(t) = [4 tan α] / [π(1 - ν)F₀] × h²(t)
    # Result in 1/Pa
    prefactor = (4 * np.tan(alpha_rad)) / (np.pi * (1 - poisson_ratio) * F0_N)
    compliance_Pa_inv = prefactor * displacement_m**2

    # Convert to 1/GPa
    compliance_GPa_inv = compliance_Pa_inv * 1e9  # 1/Pa to 1/GPa (1 Pa^-1 = 1e9 GPa^-1)

    return compliance_GPa_inv


def calculate_berkovich_compliance(
    displacement_nm: np.ndarray,
    load_uN: np.ndarray,
    poisson_ratio: float = 0.43
) -> np.ndarray:
    """
    Calculate compliance for Berkovich indenter (α = 70.3°).

    This is a convenience wrapper around calculate_conical_compliance()
    with the standard Berkovich equivalent cone half-angle.

    Args:
        displacement_nm: Displacement array (nm)
        load_uN: Load array (µN)
        poisson_ratio: Poisson's ratio

    Returns:
        compliance_gpa_inv: Compliance J(t) in 1/GPa
    """
    return calculate_conical_compliance(
        displacement_nm, load_uN, half_angle_deg=BERKOVICH_HALF_ANGLE, poisson_ratio=poisson_ratio
    )


def calculate_conical_area(depth_nm: float, half_angle_deg: float) -> float:
    """
    Calculate projected contact area for conical indenter.

    For a conical indenter:
        A = π * h² * tan²(α)

    Where:
        h = indentation depth
        α = half-angle of cone

    Args:
        depth_nm: Indentation depth (nm)
        half_angle_deg: Cone half-angle (degrees)

    Returns:
        area_nm2: Projected contact area (nm²)
    """
    alpha_rad = np.radians(half_angle_deg)
    return np.pi * depth_nm**2 * np.tan(alpha_rad)**2


def calculate_berkovich_area(depth_nm: float) -> float:
    """
    Calculate projected contact area for Berkovich indenter.

    For a perfect Berkovich indenter:
        A = 24.5 * h²

    Args:
        depth_nm: Indentation depth (nm)

    Returns:
        area_nm2: Projected contact area (nm²)
    """
    return BERKOVICH_AREA_COEFF * depth_nm**2


# =============================================================================
# COMPLIANCE DISPATCHER
# =============================================================================

def calculate_compliance(
    probe_type: str,
    displacement_nm: np.ndarray,
    load_uN: np.ndarray,
    poisson_ratio: float = 0.43,
    area_m2: Optional[float] = None,
    half_angle_deg: Optional[float] = None,
    tip_radius_nm: Optional[float] = None
) -> np.ndarray:
    """
    Dispatcher function to calculate compliance based on probe type.

    Args:
        probe_type: Probe geometry ('flat_punch', 'berkovich', 'conical', 'conical_60', 'conical_90')
        displacement_nm: Displacement array (nm)
        load_uN: Load array (µN)
        poisson_ratio: Poisson's ratio
        area_m2: Flat punch area (m²) - required for flat_punch
        half_angle_deg: Cone half-angle (°) - optional for conical (uses probe default if not specified)

    Returns:
        compliance_gpa_inv: Compliance J(t) in 1/GPa

    Raises:
        ValueError: If required parameters for probe type are missing

    Example:
        >>> # Flat punch
        >>> J = calculate_compliance('flat_punch', h, P, area_m2=3.62e-10)
        >>>
        >>> # Berkovich (70.3° default)
        >>> J = calculate_compliance('berkovich', h, P)
        >>>
        >>> # 60° conical
        >>> J = calculate_compliance('conical_60', h, P)
        >>> # or equivalently
        >>> J = calculate_compliance('conical', h, P, half_angle_deg=30.0)
        >>>
    """
    probe_type = probe_type.lower()

    if probe_type == 'flat_punch':
        if area_m2 is None:
            raise ValueError("flat_punch requires area_m2 parameter")
        return calculate_flat_punch_compliance(
            displacement_nm, load_uN, area_m2, poisson_ratio
        )

    elif probe_type == 'berkovich':
        return calculate_berkovich_compliance(
            displacement_nm, load_uN, poisson_ratio
        )

    elif probe_type in ['conical', 'conical_60', 'conical_90']:
        # Determine half-angle
        if half_angle_deg is not None:
            angle = half_angle_deg
        elif probe_type == 'conical_60':
            angle = CONICAL_60_HALF_ANGLE
        elif probe_type == 'conical_90':
            angle = CONICAL_90_HALF_ANGLE
        else:
            raise ValueError(
                "conical probe requires half_angle_deg parameter or use "
                "specific probe type like 'conical_60' or 'conical_90'"
            )

        return calculate_conical_compliance(
            displacement_nm, load_uN, angle, poisson_ratio
        )

    else:
        raise ValueError(
            f"Unknown probe type: {probe_type}. "
            f"Supported: 'flat_punch', 'berkovich', 'conical', 'conical_60', 'conical_90'"
        )


# =============================================================================
# PROBE REGISTRY
# =============================================================================

PROBE_TYPES = {
    'flat_punch': {
        'name': 'Flat Punch',
        'description': 'Cylindrical flat punch with constant contact area',
        'required_params': ['area_m2'],
        'formula': 'J_s(t) = 4R/(1-ν) × h(t)/P(t)',
        'reference': 'Harding & Sneddon (1945), Thapa & Cheng (2024)',
        'half_angle': None
    },
    'berkovich': {
        'name': 'Berkovich',
        'description': 'Triangular pyramidal indenter (α=70.3°)',
        'required_params': [],
        'formula': 'J(t) = [4 tan α]/(π(1-ν)F₀) × h²(t)',
        'reference': 'Peng et al. (2015), Eq. 2',
        'half_angle': BERKOVICH_HALF_ANGLE
    },
    'conical_60': {
        'name': 'Conical 60°',
        'description': 'Conical indenter with 60° cone angle (30° half-angle)',
        'required_params': [],
        'formula': 'J(t) = [4 tan α]/(π(1-ν)F₀) × h²(t)',
        'reference': 'Peng et al. (2015), Eq. 2',
        'half_angle': CONICAL_60_HALF_ANGLE
    },
    'conical_90': {
        'name': 'Conical 90°',
        'description': 'Conical indenter with 90° included angle (45° half-angle)',
        'required_params': [],
        'formula': 'J(t) = [4 tan α]/(π(1-ν)F₀) × h²(t)',
        'reference': 'Peng et al. (2015), Eq. 2',
        'half_angle': CONICAL_90_HALF_ANGLE
    }
}


def get_probe_info(probe_type: str) -> Dict:
    """
    Get information about a probe type.

    Args:
        probe_type: Probe type name

    Returns:
        dict with probe information

    Raises:
        ValueError: If probe type is unknown
    """
    probe_type = probe_type.lower()
    if probe_type not in PROBE_TYPES:
        raise ValueError(f"Unknown probe type: {probe_type}. Available: {list_probe_types()}")
    return PROBE_TYPES[probe_type]


def list_probe_types() -> list:
    """List all available probe types."""
    return list(PROBE_TYPES.keys())


def print_probe_info():
    """Print detailed information about all probe types."""
    print("\n" + "="*80)
    print("AVAILABLE PROBE TYPES")
    print("="*80)

    for probe_key, info in PROBE_TYPES.items():
        print(f"\n{probe_key.upper()}")
        print(f"  Name: {info['name']}")
        print(f"  Description: {info['description']}")
        if info['half_angle'] is not None:
            print(f"  Half-angle: {info['half_angle']:.1f}°")
        print(f"  Formula: {info['formula']}")
        print(f"  Reference: {info['reference']}")
        if info['required_params']:
            print(f"  Required parameters: {', '.join(info['required_params'])}")

    print("\n" + "="*80)


# =============================================================================
# MATERIAL PROPERTY CONVERSIONS
# =============================================================================

def modulus_to_compliance(E: float, nu: float = 0.43) -> float:
    """
    Convert elastic modulus to shear compliance.

    J = 2(1+ν)/E

    Args:
        E: Elastic modulus (GPa)
        nu: Poisson's ratio

    Returns:
        J: Compliance (1/GPa)
    """
    return 2 * (1 + nu) / E


def compliance_to_modulus(J: float, nu: float = 0.43) -> float:
    """
    Convert compliance to elastic modulus.

    E = 2(1+ν)/J

    Args:
        J: Compliance (1/GPa)
        nu: Poisson's ratio

    Returns:
        E: Elastic modulus (GPa)
    """
    return 2 * (1 + nu) / J


def youngs_to_shear_modulus(E: float, nu: float = 0.43) -> float:
    """
    Convert Young's modulus to shear modulus.

    G = E / [2(1+ν)]

    Args:
        E: Young's modulus (GPa)
        nu: Poisson's ratio

    Returns:
        G: Shear modulus (GPa)
    """
    return E / (2 * (1 + nu))


def shear_to_youngs_modulus(G: float, nu: float = 0.43) -> float:
    """
    Convert shear modulus to Young's modulus.

    E = 2G(1+ν)

    Args:
        G: Shear modulus (GPa)
        nu: Poisson's ratio

    Returns:
        E: Young's modulus (GPa)
    """
    return 2 * G * (1 + nu)


def bulk_modulus(E: float, nu: float = 0.43) -> float:
    """
    Calculate bulk modulus from Young's modulus and Poisson's ratio.

    K = E / [3(1 - 2ν)]

    Args:
        E: Young's modulus (GPa)
        nu: Poisson's ratio

    Returns:
        K: Bulk modulus (GPa)
    """
    if abs(nu - 0.5) < 1e-6:
        raise ValueError("Incompressible material (ν = 0.5): bulk modulus is infinite")
    return E / (3 * (1 - 2*nu))


# =============================================================================
# STRESS CALCULATIONS
# =============================================================================

def calculate_stress_from_load_area(load_uN: float, area_nm2: float) -> float:
    """
    Calculate stress from load and contact area.

    σ = F / A

    Args:
        load_uN: Applied load (µN)
        area_nm2: Contact area (nm²)

    Returns:
        stress_GPa: Stress (GPa)

    Note:
        Conversion: µN/nm² = 1e3 GPa
        Therefore: σ_GPa = (F_µN / A_nm²) × 1e-3
    """
    return (load_uN / area_nm2) * 1e-3  # Convert µN/nm² to GPa


def calculate_stress_from_indentation(
    load_uN: float,
    depth_nm: float,
    probe_type: str = 'berkovich',
    half_angle_deg: Optional[float] = None
) -> float:
    """
    Calculate stress from indentation depth using probe geometry.

    Args:
        load_uN: Applied load (µN)
        depth_nm: Indentation depth (nm)
        probe_type: Probe type ('berkovich', 'conical_60', etc.)
        half_angle_deg: Cone half-angle if probe_type='conical'

    Returns:
        stress_GPa: Stress (GPa)
    """
    if probe_type == 'berkovich':
        area = calculate_berkovich_area(depth_nm)
    elif probe_type.startswith('conical'):
        if half_angle_deg is None:
            if probe_type == 'conical_60':
                half_angle_deg = CONICAL_60_HALF_ANGLE
            elif probe_type == 'conical_90':
                half_angle_deg = CONICAL_90_HALF_ANGLE
            else:
                raise ValueError("half_angle_deg required for general conical probe")
        area = calculate_conical_area(depth_nm, half_angle_deg)
    else:
        raise ValueError(f"Unsupported probe_type: {probe_type}")

    return calculate_stress_from_load_area(load_uN, area)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def validate_positive(value: float, name: str = "value") -> None:
    """
    Validate that a value is positive.

    Args:
        value: Value to check
        name: Name of the parameter (for error message)

    Raises:
        ValueError: If value is not positive
    """
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def check_indentation_validity(depth_nm: float, load_uN: float) -> bool:
    """
    Check if indentation data point is valid.

    Args:
        depth_nm: Indentation depth (nm)
        load_uN: Applied load (µN)

    Returns:
        bool: True if valid, False otherwise
    """
    return depth_nm > 0 and load_uN > 0


if __name__ == "__main__":
    # Print available probe types
    print_probe_info()

    # Example calculation
    print("\n" + "="*80)
    print("EXAMPLE CALCULATION: Berkovich Indenter")
    print("="*80)

    h = np.array([100, 110, 120, 130, 140])  # nm
    P = np.array([1000, 1000, 1000, 1000, 1000])  # µN (constant hold)
    nu = 0.43

    J = calculate_berkovich_compliance(h, P, nu)

    print(f"\nDisplacement: {h} nm")
    print(f"Load: {P[0]:.0f} µN (constant)")
    print(f"Poisson's ratio: {nu}")
    print(f"\nCompliance J(t): {J} (1/GPa)")
    print(f"Compliance range: {J.min():.2e} to {J.max():.2e} (1/GPa)")

    # Corresponding stress
    for i, (hi, Pi) in enumerate(zip(h, P)):
        area = calculate_berkovich_area(hi)
        stress = calculate_stress_from_load_area(Pi, area)
        print(f"  t={i}: h={hi} nm, A={area:.1f} nm², σ={stress*1000:.2f} MPa")

    print("\n" + "="*80)
