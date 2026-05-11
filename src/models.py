"""
Comprehensive Creep Model Library for Nanoindentation and Bulk Testing
=======================================================================

This module consolidates all viscoelastic, viscoelastic-plastic, and
viscoelastic-viscoplastic models for creep compliance and stress relaxation analysis.

Model Categories:
    1. Empirical Models: Logarithmic, Power-Law
    2. Kelvin-Voigt Chain Models: Generalized Kelvin, Burgers, Prony Series
    3. Stress-Lock Models: Peng et al. nonlinear viscoelasticity
    4. Maxwell Models: Generalized Maxwell (for stress relaxation)

References:
    - Liu et al.: Logarithmic empirical model
    - Thapa & Cheng: 3-parameter Kelvin-Voigt
    - Hou & Jennett: N-element Kelvin-Voigt, Standard Linear Solid
    - Christöfl et al.: Generalized Maxwell
    - Barick: VEVP and VEP models
    - Peng et al. (2015): Stress-lock model, Polymer Testing 43:38-43

Author: Creep Analysis Framework
Date: 2024-2025
"""

import numpy as np
from lmfit import Model, Parameters
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Optional
import logging


# =============================================================================
# ABSTRACT BASE CLASS FOR ALL MODELS
# =============================================================================

class CreepModel(ABC):
    """
    Abstract base class for all creep models.

    All models inherit from this class and implement:
        - model_func: The mathematical model
        - initial_guess: Smart parameter initialization
        - parameter_bounds: Physical constraints on parameters
    """

    def __init__(self, name, n_params):
        self.name = name
        self.n_params = n_params
        self.fit_result = None

    @abstractmethod
    def model_func(self, t, **params):
        """
        Model function that calculates response at time t.

        Args:
            t: Time array (seconds)
            **params: Model parameters (unpacked from lmfit)

        Returns:
            Calculated response (displacement, compliance, etc.)
        """
        pass

    @abstractmethod
    def initial_guess(self, time_data, response_data, **kwargs):
        """
        Generate smart initial parameter guesses based on data.

        Args:
            time_data: Time array
            response_data: Measured response
            **kwargs: Additional arguments specific to model

        Returns:
            lmfit.Parameters object with initial guesses
        """
        pass

    def fit(self, time_data, response_data, **kwargs):
        """
        Fit model to data using nonlinear least squares.

        Args:
            time_data: Time array
            response_data: Measured response
            **kwargs: Additional arguments (weights, etc.)

        Returns:
            lmfit.ModelResult object
        """
        # Get initial parameter guesses
        params = self.initial_guess(time_data, response_data, **kwargs)

        # Create lmfit Model
        model = Model(self.model_func, independent_vars=['t'])

        # Perform fit
        weights = kwargs.get('weights', None)
        self.fit_result = model.fit(response_data, params, t=time_data, weights=weights)

        # Log fit quality
        logging.info(f"[{self.name}] Fit complete:")
        logging.info(f"  R² = {self.fit_result.rsquared:.6f}")
        logging.info(f"  Reduced χ² = {self.fit_result.redchi:.6e}")

        return self.fit_result

    def predict(self, time_array):
        """
        Predict response at new time points using fitted parameters.

        Args:
            time_array: Time array for prediction

        Returns:
            Predicted response array
        """
        if self.fit_result is None:
            raise ValueError("Model must be fitted before prediction")

        return self.fit_result.eval(t=time_array)

    def get_parameters(self):
        """
        Extract fitted parameters and uncertainties.

        Returns:
            dict with parameter values and standard errors
        """
        if self.fit_result is None:
            raise ValueError("Model must be fitted first")

        param_dict = {}
        for name, param in self.fit_result.params.items():
            param_dict[name] = {
                'value': param.value,
                'stderr': param.stderr if param.stderr else 0.0,
                'units': self._get_param_units(name)
            }

        return param_dict

    def _get_param_units(self, param_name):
        """Get physical units for parameter (override in subclass if needed)."""
        return ''

    def get_statistics(self):
        """
        Get fit statistics.

        Returns:
            dict with R², RMSE, AIC, BIC
        """
        if self.fit_result is None:
            raise ValueError("Model must be fitted first")

        residuals = self.fit_result.residual
        n = len(residuals)
        k = len(self.fit_result.params)

        rmse = np.sqrt(np.mean(residuals**2))

        stats = {
            'R_squared': self.fit_result.rsquared,
            'Reduced_ChiSq': self.fit_result.redchi,
            'RMSE': rmse,
            'AIC': self.fit_result.aic,
            'BIC': self.fit_result.bic,
            'n_data': n,
            'n_params': k
        }

        return stats


# =============================================================================
# EMPIRICAL MODELS
# =============================================================================

class LogarithmicModel(CreepModel):
    """
    Logarithmic empirical model (Liu et al.)

    Model:
        h(t) = a * ln(t + b) + c

    Where:
        a: Creep magnitude parameter (larger a = more creep)
        b: Time shift parameter (s)
        c: Baseline displacement (nm)

    Use cases:
        - Simple empirical fitting
        - Quick creep characterization
        - Comparison baseline
    """

    def __init__(self):
        super().__init__("Logarithmic (Liu et al.)", n_params=3)

    def model_func(self, t, a, b, c):
        """h(t) = a * ln(t + b) + c"""
        return a * np.log(t + b) + c

    def initial_guess(self, time_data, response_data, **kwargs):
        """
        Smart initialization:
            - c: Initial response value
            - a: Approximate creep rate
            - b: Small time shift
        """
        params = Parameters()

        # c: baseline (first data point)
        c_guess = response_data[0]

        # a: estimate from slope of response vs log(time)
        if len(time_data) > 10:
            log_time = np.log(time_data[1:])  # Skip t=0
            slope, _ = np.polyfit(log_time, response_data[1:], 1)
            a_guess = slope
        else:
            a_guess = (response_data[-1] - response_data[0]) / np.log(time_data[-1] + 1)

        # b: small time shift
        b_guess = 0.1

        params.add('a', value=a_guess, min=-1e6, max=1e6)
        params.add('b', value=b_guess, min=1e-6, max=time_data[-1])
        params.add('c', value=c_guess, min=-1e6, max=1e6)

        return params

    def _get_param_units(self, param_name):
        if param_name == 'a':
            return 'nm'
        elif param_name == 'b':
            return 's'
        elif param_name == 'c':
            return 'nm'
        return ''


class PowerLawModel(CreepModel):
    """
    Power-law creep model (empirical)

    Model:
        ε(t) = ε_0 * (t/t_0)^n

    Or for compliance:
        J(t) = J_0 * (t/t_0)^n

    Where:
        J_0 (or ε_0): Initial compliance (strain)
        t_0: Reference time (s)
        n: Power-law exponent (dimensionless)

    Common in:
        - Primary creep (n < 1)
        - Secondary creep (n ≈ 1)
        - Tertiary creep (n > 1)
    """

    def __init__(self):
        super().__init__("Power-Law Creep", n_params=3)

    def model_func(self, t, J0, t0, n):
        """J(t) = J_0 * (t/t_0)^n"""
        return J0 * (t / t0)**n

    def initial_guess(self, time_data, compliance_data, **kwargs):
        """Smart initialization using log-log slope."""
        params = Parameters()

        # Filter positive values for log-log fit
        valid_mask = (time_data > 0) & (compliance_data > 0)
        t_valid = time_data[valid_mask]
        J_valid = compliance_data[valid_mask]

        if len(t_valid) > 2:
            # Log-log linear fit to get n
            log_t = np.log(t_valid)
            log_J = np.log(J_valid)
            n_guess, log_J0_at_t0 = np.polyfit(log_t, log_J, 1)

            # Reference time (middle of range)
            t0_guess = np.median(t_valid)

            # J0 at reference time
            J0_guess = np.exp(log_J0_at_t0 + n_guess * np.log(t0_guess))
        else:
            # Fallback
            J0_guess = compliance_data[0]
            t0_guess = 1.0
            n_guess = 0.3

        params.add('J0', value=J0_guess, min=J0_guess * 0.1, max=J0_guess * 10)
        params.add('t0', value=t0_guess, min=1e-3, max=time_data[-1])
        params.add('n', value=n_guess, min=0, max=2)

        return params

    def _get_param_units(self, param_name):
        if param_name == 'J0':
            return '1/GPa'
        elif param_name == 't0':
            return 's'
        elif param_name == 'n':
            return 'dimensionless'
        return ''


# =============================================================================
# KELVIN-VOIGT CHAIN MODELS
# =============================================================================

class GeneralizedKelvinModel(CreepModel):
    """
    N-Element Generalized Kelvin Model (Kelvin-Voigt chain).

    Model (creep compliance):
        J(t) = J_0 + Σ[J_i * (1 - exp(-t/τ_i))] + t/η_n  for i=1 to N

    This represents the creep response of a free spring (J_0) in series with N
    Kelvin-Voigt elements, plus a viscous dashpot for permanent flow.

    Where:
        J_0: Instantaneous compliance (from the free spring)
        J_i: Compliance of element i
        τ_i: Retardation time of element i (s)
        η_n: Permanent viscosity (GPa·s)

    Special cases:
        - N=1: Burgers model (4-parameter)
        - N>1: Generalized Kelvin Model
    """

    def __init__(self, n_elements=3, mode='compliance'):
        """
        Args:
            n_elements: Number of Kelvin-Voigt elements (default: 3)
            mode: 'compliance' or 'displacement' (default: compliance)
        """
        self.n_elements = n_elements
        self.mode = mode
        name = f"{n_elements}-Element Gen. Kelvin w/ Flow"
        if n_elements == 1:
            name = "Burgers (4-parameter)"
        super().__init__(name, n_params=2 + 2*n_elements)

    def model_func(self, t, J0, eta_n, **J_tau_dict):
        """
        J(t) = J_0 + Σ[J_i * (1 - exp(-t/τ_i))] + t/η_n

        Parameters come as J1, tau1, J2, tau2, etc.
        """
        # Instantaneous compliance
        compliance = J0 * np.ones_like(t)

        # Add each Kelvin-Voigt element
        for i in range(1, self.n_elements + 1):
            J_i = J_tau_dict[f'J{i}']
            tau_i = J_tau_dict[f'tau{i}']
            compliance += J_i * (1 - np.exp(-t / tau_i))

        # Add permanent viscous flow
        compliance += t / eta_n

        return compliance

    def initial_guess(self, time_data, compliance_data, **kwargs):
        """
        Smart initialization using time-domain spacing.

        Strategy:
            - J0: Compliance at first time point
            - η_n: From long-time slope of compliance data
            - J_i: Distributed based on remaining recoverable compliance
            - τ_i: Logarithmically spaced across time range
        """
        params = Parameters()

        # J0: Instantaneous compliance (use first few points)
        J0_guess = np.min(compliance_data[:5]) if len(compliance_data) > 5 else compliance_data[0]
        J0_guess = max(J0_guess, compliance_data[0] * 0.5)  # Ensure reasonable value

        params.add('J0', value=J0_guess, min=J0_guess * 0.1, max=J0_guess * 10)

        # η_n: From long-time slope
        if len(time_data) > 10:
            # Linear fit to last 30% of data
            idx_start = int(0.7 * len(time_data))
            slope, _ = np.polyfit(time_data[idx_start:], compliance_data[idx_start:], 1)
            eta_n_guess = 1 / slope if slope > 1e-12 else 1e12
        else:
            eta_n_guess = 1e9

        params.add('eta_n', value=eta_n_guess, min=1.0)

        # Remaining compliance to distribute among elements
        viscous_contrib = time_data[-1] / eta_n_guess
        J_total_recoverable = compliance_data[-1] - J0_guess - viscous_contrib
        J_per_element = J_total_recoverable / self.n_elements if J_total_recoverable > 0 else 0

        # Logarithmically space retardation times across test duration
        time_range = time_data[-1] - time_data[0]
        tau_guesses = np.logspace(
            np.log10(max(time_range / 1000, 0.01)),  # Minimum tau
            np.log10(time_range * 0.5),  # Maximum tau
            self.n_elements
        )

        # Add parameters for each element
        for i in range(1, self.n_elements + 1):
            # J_i: compliance for this element
            params.add(f'J{i}',
                      value=J_per_element,
                      min=0,
                      max=J_total_recoverable * 2 if J_total_recoverable > 0 else 1e-9)

            # tau_i: retardation time
            params.add(f'tau{i}',
                      value=tau_guesses[i-1],
                      min=time_range / 10000,
                      max=time_range * 10)

        return params

    def _get_param_units(self, param_name):
        if param_name.startswith('J'):
            return '1/GPa'
        elif param_name.startswith('tau'):
            return 's'
        elif param_name == 'eta_n':
            return 'GPa·s'
        return ''


class BurgersModel(CreepModel):
    """
    4-Parameter Burgers model (special case of Generalized Kelvin with n=1)

    Model (creep compliance):
        J(t) = J_g + J_k*(1 - exp(-t/τ_k)) + t/η_n

    Where:
        J_g: Instantaneous glassy compliance (1/GPa)
        J_k: Delayed elastic compliance (1/GPa)
        τ_k: Retardation time (s)
        η_n: Newtonian viscosity (GPa·s)

    Physical interpretation:
        - J_g: Immediate elastic response
        - J_k, τ_k: Delayed elastic (recoverable) creep
        - η_n: Permanent (viscous) flow

    This is Maxwell + Kelvin-Voigt in series
    """

    def __init__(self):
        super().__init__("Burgers (4-parameter)", n_params=4)

    def model_func(self, t, Jg, Jk, tau_k, eta_n):
        """J(t) = J_g + J_k*(1 - exp(-t/τ_k)) + t/η_n"""
        return Jg + Jk * (1 - np.exp(-t / tau_k)) + t / eta_n

    def initial_guess(self, time_data, compliance_data, **kwargs):
        """Smart initialization for Burgers model."""
        params = Parameters()

        # J_g: Initial compliance
        Jg_guess = compliance_data[0]

        # η_n: From long-time slope
        if len(time_data) > 10:
            # Linear fit to last 30% of data
            idx_start = int(0.7 * len(time_data))
            slope, _ = np.polyfit(time_data[idx_start:], compliance_data[idx_start:], 1)
            eta_n_guess = 1 / slope if slope > 0 else 1e6
        else:
            eta_n_guess = 1e6

        # J_k: Intermediate compliance
        J_total = compliance_data[-1] - Jg_guess
        viscous_contrib = time_data[-1] / eta_n_guess
        Jk_guess = max(0, J_total - viscous_contrib)

        # tau_k: Characteristic time (estimate from data)
        tau_k_guess = time_data[len(time_data) // 3]  # 1/3 through test

        params.add('Jg', value=Jg_guess, min=Jg_guess * 0.5, max=Jg_guess * 2)
        params.add('Jk', value=Jk_guess, min=0, max=J_total * 2)
        params.add('tau_k', value=tau_k_guess, min=time_data[1], max=time_data[-1] * 2)
        params.add('eta_n', value=eta_n_guess, min=1e3, max=1e9)

        return params

    def _get_param_units(self, param_name):
        if param_name in ['Jg', 'Jk']:
            return '1/GPa'
        elif param_name == 'tau_k':
            return 's'
        elif param_name == 'eta_n':
            return 'GPa·s'
        return ''


class PronySeriesModel(CreepModel):
    """
    Prony series model (alternative parameterization of Kelvin-Voigt)

    Model (creep compliance):
        J(t) = J_g + J_∞ * [1 - Σ(ρ_i * exp(-t/τ_i))]

    Where:
        J_g: Glassy (instantaneous) compliance
        J_∞: Equilibrium (long-term) compliance
        ρ_i: Weight factors (Σρ_i = 1)
        τ_i: Retardation times

    Commonly used in:
        - FEM software (ANSYS, ABAQUS)
        - Polymer viscoelasticity
    """

    def __init__(self, n_terms=3):
        """
        Args:
            n_terms: Number of Prony terms
        """
        self.n_terms = n_terms
        super().__init__(f"Prony Series ({n_terms} terms)", n_params=2 + 2*n_terms)

    def model_func(self, t, Jg, Jinf, **rho_tau_dict):
        """J(t) = J_g + J_∞ * [1 - Σ(ρ_i * exp(-t/τ_i))]"""
        J = Jg + Jinf * np.ones_like(t)

        for i in range(1, self.n_terms + 1):
            rho_i = rho_tau_dict[f'rho{i}']
            tau_i = rho_tau_dict[f'tau{i}']
            J -= Jinf * rho_i * np.exp(-t / tau_i)

        return J

    def initial_guess(self, time_data, compliance_data, **kwargs):
        """Smart initialization with normalized weights."""
        params = Parameters()

        # J_g and J_inf
        Jg_guess = compliance_data[0]
        Jinf_guess = compliance_data[-1]

        params.add('Jg', value=Jg_guess, min=Jg_guess * 0.5, max=Jg_guess * 2)
        params.add('Jinf', value=Jinf_guess, min=Jg_guess, max=Jinf_guess * 2)

        # Equal weight distribution
        rho_guess = 1.0 / self.n_terms

        # Logarithmically spaced times
        time_range = time_data[-1] - time_data[0]
        tau_guesses = np.logspace(
            np.log10(time_range / 100),
            np.log10(time_range),
            self.n_terms
        )

        # Add terms with constraint that Σρ_i = 1
        for i in range(1, self.n_terms + 1):
            if i < self.n_terms:
                params.add(f'rho{i}', value=rho_guess, min=0, max=1)
            else:
                # Last rho is constrained: rho_n = 1 - Σrho_i
                expr = '1 - (' + ' + '.join([f'rho{j}' for j in range(1, self.n_terms)]) + ')'
                params.add(f'rho{self.n_terms}', expr=expr)

            params.add(f'tau{i}', value=tau_guesses[i-1], min=time_range/1000, max=time_range*10)

        return params

    def _get_param_units(self, param_name):
        if param_name.startswith('J'):
            return '1/GPa'
        elif param_name.startswith('tau'):
            return 's'
        elif param_name.startswith('rho'):
            return 'dimensionless'
        return ''


# =============================================================================
# MAXWELL MODELS (FOR STRESS RELAXATION)
# =============================================================================

class GeneralizedMaxwellModel(CreepModel):
    """
    Generalized Maxwell model (Christöfl et al.)

    **WARNING: FOR STRESS RELAXATION DATA ONLY**
    This model describes modulus relaxation E(t) and should NOT be fitted
    directly to creep compliance J(t) data.

    Model (stress relaxation):
        σ(t) = σ_∞ + Σ[σ_i * exp(-t/τ_i)]

    Or for modulus relaxation:
        E(t) = E_∞ + Σ[E_i * exp(-t/τ_i)]

    Where:
        E_∞: Long-term equilibrium modulus
        E_i: Modulus of Maxwell element i
        τ_i: Relaxation time of element i (s)

    This is the "dual" of Kelvin-Voigt (for relaxation vs creep)
    """

    def __init__(self, n_elements=2):
        """
        Args:
            n_elements: Number of Maxwell elements (default: 2)
        """
        self.n_elements = n_elements
        name = f"{n_elements}-Element Generalized Maxwell (Christöfl)"
        super().__init__(name, n_params=1 + 2*n_elements)

    def fit(self, time_data, response_data, **kwargs):
        """Override to prevent misuse on creep data."""
        raise NotImplementedError(
            f"'{self.name}' is a stress-relaxation model and cannot be directly fitted to creep data. "
            "This feature is disabled to prevent incorrect physical analysis."
        )

    def model_func(self, t, Einf, **E_tau_dict):
        """
        E(t) = E_∞ + Σ[E_i * exp(-t/τ_i)]
        """
        # Long-term modulus
        modulus = Einf * np.ones_like(t)

        # Add each Maxwell element
        for i in range(1, self.n_elements + 1):
            E_i = E_tau_dict[f'E{i}']
            tau_i = E_tau_dict[f'tau{i}']
            modulus += E_i * np.exp(-t / tau_i)

        return modulus

    def initial_guess(self, time_data, modulus_data, **kwargs):
        """Smart initialization for relaxation data."""
        params = Parameters()

        # E_inf: Long-term modulus (last few points)
        Einf_guess = np.mean(modulus_data[-5:]) if len(modulus_data) > 5 else modulus_data[-1]
        params.add('Einf', value=Einf_guess, min=0, max=Einf_guess * 2)

        # Initial modulus
        E0 = modulus_data[0]
        E_relaxed = E0 - Einf_guess
        E_per_element = E_relaxed / self.n_elements

        # Logarithmically space relaxation times
        time_range = time_data[-1] - time_data[0]
        tau_guesses = np.logspace(
            np.log10(max(time_range / 1000, 0.01)),
            np.log10(time_range * 0.5),
            self.n_elements
        )

        # Add parameters for each element
        for i in range(1, self.n_elements + 1):
            params.add(f'E{i}', value=E_per_element, min=0, max=E_relaxed * 2)
            params.add(f'tau{i}', value=tau_guesses[i-1], min=time_range/10000, max=time_range*10)

        return params

    def _get_param_units(self, param_name):
        if param_name.startswith('E'):
            return 'GPa'
        elif param_name.startswith('tau'):
            return 's'
        return ''


# =============================================================================
# STRESS-LOCK MODELS (PENG ET AL. 2015)
# =============================================================================

# Indenter geometry constants
BERKOVICH_HALF_ANGLE = 70.3  # degrees


def moduli_to_compliance(E, nu=0.43):
    """
    Convert elastic modulus to compliance.

    J = 2(1+ν)/E

    Args:
        E: Elastic modulus (GPa)
        nu: Poisson's ratio (dimensionless)

    Returns:
        Compliance J (1/GPa)
    """
    return 2 * (1 + nu) / E


def compliance_to_modulus(J, nu=0.43):
    """
    Convert compliance to elastic modulus.

    E = 2(1+ν)/J

    Args:
        J: Compliance (1/GPa)
        nu: Poisson's ratio (dimensionless)

    Returns:
        Elastic modulus E (GPa)
    """
    return 2 * (1 + nu) / J


class PengStressLockModel(CreepModel):
    """
    Peng et al. (2015) Stress-Lock Model for Nonlinear Viscoelasticity

    Reference: Peng, G., et al. (2015). "Nanoindentation creep of nonlinear
    viscoelastic polypropylene." Polymer Testing 43, 38-43.

    Model:
        J(t, σ) = J₀ + Σᵢ Jᵢ[1 - exp(-t/τᵢ)] + t/η₀

    Key Innovation: "Stress-Lock" Elements
        - Each element has a critical stress σ*ᵢ
        - Element only contributes if σ > σ*ᵢ
        - Creates stress-dependent compliance

    Architecture:
        - Maxwell unit (E₀, η₀): Elastic + viscous flow
        - Standard Voigt-Kelvin: Always active
        - Locked VK units: Activate above threshold stress
        - Optional viscoplastic: Activates above yield stress

    Use Cases:
        - Nonlinear viscoelastic polymers
        - Stress-dependent creep behavior
        - Multi-stress-level characterization

    Example:
        For PP with 2 stress-locks (σ*₁=14.9 MPa, σ*₂=19.9 MPa):
        - σ < 14.9 MPa  → Only base elements (Burgers model)
        - 14.9 < σ < 19.9 → Base + locked element 1
        - σ > 19.9 MPa  → Base + both locked elements
    """

    def __init__(self, n_locked_elements=2, include_viscoplastic=False, applied_stress=None):
        """
        Args:
            n_locked_elements: Number of stress-locked Voigt-Kelvin units (default: 2)
            include_viscoplastic: Include viscoplastic element (default: False)
            applied_stress: Applied stress in GPa (required for determining active elements)
        """
        self.n_locked_elements = n_locked_elements
        self.include_viscoplastic = include_viscoplastic
        self.applied_stress = applied_stress  # GPa

        # Parameters: J0, J1, tau1 (base), + (J_i, tau_i, sigma_i*) for each locked element,
        # + eta0 (Maxwell), + (eta_vp, sigma_y) if viscoplastic
        n_params = 3  # J0, J1, tau1 (base element, always active)
        n_params += 3 * n_locked_elements  # J_i, tau_i, sigma*_i for each locked element
        n_params += 1  # eta0 (Maxwell dashpot)
        if include_viscoplastic:
            n_params += 2  # eta_vp, sigma_y

        n_vk = n_locked_elements + 1  # base + locked elements
        name = f"Peng ({n_vk} VK elements, {1 + int(include_viscoplastic)} dashpots"
        if include_viscoplastic:
            name += ", viscoplastic"
        name += ")"

        super().__init__(name, n_params=n_params)

    def model_func(self, t, J0, J1, tau1, eta0, **locked_params):
        """
        J(t, σ) = J₀ + J₁[1 - exp(-t/τ₁)] + Σᵢ Jᵢ[1 - exp(-t/τᵢ)]×H(σ-σ*ᵢ) + t/η₀ + t/ηᵥₚ×H(σ-σᵧ)

        Where H(x) is Heaviside function (1 if x>0, else 0)
        """
        # Initialize with instantaneous compliance
        compliance = J0 * np.ones_like(t)

        # Base Voigt-Kelvin element (always active)
        compliance += J1 * (1 - np.exp(-t / tau1))

        # Add locked Voigt-Kelvin elements (stress-dependent)
        # When applied_stress is None (fitting mode), all elements are active
        for i in range(2, self.n_locked_elements + 2):  # Elements 2, 3, ...
            J_i = locked_params[f'J{i}']
            tau_i = locked_params[f'tau{i}']
            sigma_star_i = locked_params[f'sigma_star_{i}']

            if self.applied_stress is None or self.applied_stress > sigma_star_i:
                compliance += J_i * (1 - np.exp(-t / tau_i))

        # Maxwell dashpot (viscous flow)
        compliance += t / eta0

        # Viscoplastic element (if included; gated by stress when set)
        if self.include_viscoplastic:
            eta_vp = locked_params['eta_vp']
            sigma_y = locked_params['sigma_y']

            if self.applied_stress is None or self.applied_stress > sigma_y:
                compliance += t / eta_vp

        return compliance

    def initial_guess(self, time_data, compliance_data, **kwargs):
        """
        Smart initialization for stress-lock model.

        Strategy:
            - J0: Minimum compliance (instantaneous)
            - J1, τ1: Base element from early time behavior
            - J_i: Distributed across remaining compliance
            - σ*ᵢ: Logarithmically spaced stress thresholds
            - η₀: Long-term viscous flow slope

        Kwargs:
            stress_thresholds: List of stress thresholds (GPa) for locked elements
            yield_stress: Yield stress (GPa) for viscoplastic element
        """
        params = Parameters()

        # J0: Instantaneous compliance (minimum value)
        J0_guess = np.min(compliance_data[:5]) if len(compliance_data) > 5 else compliance_data[0]
        J0_guess = max(J0_guess, compliance_data[0] * 0.5)
        params.add('J0', value=J0_guess, min=J0_guess * 0.1, max=J0_guess * 10)

        # Remaining compliance to distribute
        J_total = compliance_data[-1] - J0_guess
        J_per_element = J_total / (self.n_locked_elements + 1)

        # Time range for τ estimation
        time_range = time_data[-1] - time_data[0]

        # Base Voigt-Kelvin element (always active)
        params.add('J1', value=J_per_element, min=0, max=J_total * 2)
        params.add('tau1', value=time_range * 0.1, min=time_range / 1000, max=time_range * 10)

        # Locked Voigt-Kelvin elements
        tau_guesses = np.logspace(
            np.log10(time_range / 100),
            np.log10(time_range),
            self.n_locked_elements
        )

        # Get stress thresholds from kwargs or create defaults
        if 'stress_thresholds' in kwargs and kwargs['stress_thresholds'] is not None:
            stress_thresholds = kwargs['stress_thresholds']
        else:
            # Default: logarithmically spaced from 0.01 to 0.03 GPa (10-30 MPa)
            stress_thresholds = np.logspace(
                np.log10(0.01),  # 10 MPa
                np.log10(0.03),  # 30 MPa
                self.n_locked_elements
            )

        for i in range(2, self.n_locked_elements + 2):
            idx = i - 2  # Index into arrays

            params.add(f'J{i}', value=J_per_element, min=0, max=J_total * 2)
            params.add(f'tau{i}', value=tau_guesses[idx], min=time_range / 1000, max=time_range * 10)

            # Stress threshold (GPa) — freeze during fitting when no stress is set,
            # since all elements are active anyway and these params would be inert
            sigma_guess = stress_thresholds[idx] if idx < len(stress_thresholds) else 0.02
            params.add(f'sigma_star_{i}',
                      value=sigma_guess,
                      min=0.001,  # 1 MPa minimum
                      max=0.1,    # 100 MPa maximum
                      vary=(self.applied_stress is not None))

        # Maxwell dashpot viscosity (estimate from long-term slope)
        if len(compliance_data) > 10:
            # Linear fit to last portion of curve
            fit_fraction = 0.3
            n_fit = max(10, int(len(time_data) * fit_fraction))
            slope = (compliance_data[-1] - compliance_data[-n_fit]) / (time_data[-1] - time_data[-n_fit])
            eta0_guess = 1.0 / max(slope, 1e-10)
        else:
            eta0_guess = 1000.0  # GPa·s

        params.add('eta0', value=eta0_guess, min=1.0, max=1e6)

        # Viscoplastic element (if included)
        if self.include_viscoplastic:
            if 'yield_stress' in kwargs and kwargs['yield_stress'] is not None:
                sigma_y_guess = kwargs['yield_stress']
            else:
                sigma_y_guess = 0.035  # 35 MPa default

            params.add('eta_vp', value=eta0_guess * 10, min=1.0, max=1e6)
            params.add('sigma_y', value=sigma_y_guess, min=0.02, max=0.2,
                       vary=(self.applied_stress is not None))

        return params

    def set_stress(self, stress_gpa):
        """
        Set the applied stress for this model.

        Args:
            stress_gpa: Applied stress in GPa
        """
        self.applied_stress = stress_gpa

    def _get_param_units(self, param_name):
        """Get units for parameters."""
        if param_name.startswith('J'):
            return '1/GPa'
        elif param_name.startswith('tau'):
            return 's'
        elif param_name.startswith('eta'):
            return 'GPa·s'
        elif param_name.startswith('sigma'):
            return 'GPa'
        return ''


# =============================================================================
# SIMPLIFIED FUNCTIONAL MODELS (FOR COMPATIBILITY)
# =============================================================================

def prony_3param(t, J0, J1, tau1):
    """3-parameter Prony series: J(t) = J₀ + J₁[1 - exp(-t/τ₁)]"""
    return J0 + J1 * (1 - np.exp(-t / tau1))


def prony_5param(t, J0, J1, tau1, J2, tau2):
    """5-parameter Prony series: J(t) = J₀ + J₁[1 - exp(-t/τ₁)] + J₂[1 - exp(-t/τ₂)]"""
    return (J0 +
            J1 * (1 - np.exp(-t / tau1)) +
            J2 * (1 - np.exp(-t / tau2)))


def prony_7param(t, J0, J1, tau1, J2, tau2, J3, tau3):
    """7-parameter Prony series: J(t) = J₀ + Σᵢ₌₁³ Jᵢ[1 - exp(-t/τᵢ)]"""
    return (J0 +
            J1 * (1 - np.exp(-t / tau1)) +
            J2 * (1 - np.exp(-t / tau2)) +
            J3 * (1 - np.exp(-t / tau3)))


def burgers_model(t, J0, J1, tau1, eta0, eta_y=None):
    """
    Burgers model: J(t) = J₀ + J₁[1 - exp(-t/τ₁)] + t/η₀
    With optional viscoplastic term: + t/ηᵧ (if eta_y is provided)
    """
    J = J0 + J1 * (1 - np.exp(-t / tau1)) + t / eta0
    if eta_y is not None:
        J += t / eta_y
    return J


def m_plus_2vk_model(t, J0, J1, tau1, J2, tau2, eta0, eta_y=None):
    """M+2VK model: J(t) = J₀ + J₁[1-exp(-t/τ₁)] + J₂[1-exp(-t/τ₂)] + t/η₀"""
    J = (J0 +
         J1 * (1 - np.exp(-t / tau1)) +
         J2 * (1 - np.exp(-t / tau2)) +
         t / eta0)
    if eta_y is not None:
        J += t / eta_y
    return J


def m_plus_3vk_model(t, J0, J1, tau1, J2, tau2, J3, tau3, eta0, eta_y=None):
    """M+3VK model: J(t) = J₀ + Σᵢ₌₁³ Jᵢ[1-exp(-t/τᵢ)] + t/η₀"""
    J = (J0 +
         J1 * (1 - np.exp(-t / tau1)) +
         J2 * (1 - np.exp(-t / tau2)) +
         J3 * (1 - np.exp(-t / tau3)) +
         t / eta0)
    if eta_y is not None:
        J += t / eta_y
    return J


def m_plus_4vk_model(t, J0, J1, tau1, J2, tau2, J3, tau3, J4, tau4, eta0, eta_y=None):
    """
    M+4VK model (10-element): 1 Maxwell + 4 Voigt-Kelvin units.

    J(t) = J₀ + Σᵢ₌₁⁴ Jᵢ[1-e^(-t/τᵢ)] + t/η₀

    Args:
        t: Time (s)
        J0: Instantaneous compliance (1/GPa)
        J1-J4: Retarded compliances (1/GPa)
        tau1-tau4: Retardation times (s)
        eta0: Maxwell dashpot viscosity (GPa·s)
        eta_y: Viscoplastic dashpot viscosity (GPa·s), None if below yield

    Returns:
        Compliance J(t) (1/GPa)
    """
    J = (J0 +
         J1 * (1 - np.exp(-t / tau1)) +
         J2 * (1 - np.exp(-t / tau2)) +
         J3 * (1 - np.exp(-t / tau3)) +
         J4 * (1 - np.exp(-t / tau4)) +
         t / eta0)
    if eta_y is not None:
        J += t / eta_y
    return J


def m_plus_5vk_model(t, J0, J1, tau1, J2, tau2, J3, tau3, J4, tau4, J5, tau5, eta0, eta_y=None):
    """
    M+5VK model (12-element): 1 Maxwell + 5 Voigt-Kelvin units.

    J(t) = J₀ + Σᵢ₌₁⁵ Jᵢ[1-e^(-t/τᵢ)] + t/η₀

    Args:
        t: Time (s)
        J0: Instantaneous compliance (1/GPa)
        J1-J5: Retarded compliances (1/GPa)
        tau1-tau5: Retardation times (s)
        eta0: Maxwell dashpot viscosity (GPa·s)
        eta_y: Viscoplastic dashpot viscosity (GPa·s), None if below yield

    Returns:
        Compliance J(t) (1/GPa)
    """
    J = (J0 +
         J1 * (1 - np.exp(-t / tau1)) +
         J2 * (1 - np.exp(-t / tau2)) +
         J3 * (1 - np.exp(-t / tau3)) +
         J4 * (1 - np.exp(-t / tau4)) +
         J5 * (1 - np.exp(-t / tau5)) +
         t / eta0)
    if eta_y is not None:
        J += t / eta_y
    return J


def m_plus_6vk_model(t, J0, J1, tau1, J2, tau2, J3, tau3, J4, tau4, J5, tau5, J6, tau6, eta0, eta_y=None):
    """
    M+6VK model (14-element): 1 Maxwell + 6 Voigt-Kelvin units.

    J(t) = J₀ + Σᵢ₌₁⁶ Jᵢ[1-e^(-t/τᵢ)] + t/η₀

    Args:
        t: Time (s)
        J0: Instantaneous compliance (1/GPa)
        J1-J6: Retarded compliances (1/GPa)
        tau1-tau6: Retardation times (s)
        eta0: Maxwell dashpot viscosity (GPa·s)
        eta_y: Viscoplastic dashpot viscosity (GPa·s), None if below yield

    Returns:
        Compliance J(t) (1/GPa)
    """
    J = (J0 +
         J1 * (1 - np.exp(-t / tau1)) +
         J2 * (1 - np.exp(-t / tau2)) +
         J3 * (1 - np.exp(-t / tau3)) +
         J4 * (1 - np.exp(-t / tau4)) +
         J5 * (1 - np.exp(-t / tau5)) +
         J6 * (1 - np.exp(-t / tau6)) +
         t / eta0)
    if eta_y is not None:
        J += t / eta_y
    return J


def generalized_kv_model(t, J0, J_list, tau_list, eta0, eta_y=None):
    """
    Generalized Kelvin-Voigt (Prony series) model with arbitrary number of VK elements.

    J(t) = J₀ + Σᵢ Jᵢ[1-e^(-t/τᵢ)] + t/η₀

    Args:
        t: Time (s) - can be scalar or array
        J0: Instantaneous compliance (1/GPa)
        J_list: List of retarded compliances [J1, J2, ..., Jn] (1/GPa)
        tau_list: List of retardation times [τ1, τ2, ..., τn] (s)
        eta0: Maxwell dashpot viscosity (GPa·s)
        eta_y: Viscoplastic dashpot viscosity (GPa·s), None if below yield

    Returns:
        Compliance J(t) (1/GPa)
    """
    J = J0 + t / eta0

    # Add contribution from each VK element
    for Ji, taui in zip(J_list, tau_list):
        J += Ji * (1 - np.exp(-t / taui))

    # Add viscoplastic term if present
    if eta_y is not None:
        J += t / eta_y

    return J


# =============================================================================
# MODEL REGISTRY AND FACTORY FUNCTIONS
# =============================================================================

AVAILABLE_MODELS = {
    'logarithmic': LogarithmicModel,
    'gen-kelvin-1': lambda: GeneralizedKelvinModel(n_elements=1),
    'gen-kelvin-2': lambda: GeneralizedKelvinModel(n_elements=2),
    'gen-kelvin-3': lambda: GeneralizedKelvinModel(n_elements=3),
    'gen-kelvin-4': lambda: GeneralizedKelvinModel(n_elements=4),
    'gen-kelvin-5': lambda: GeneralizedKelvinModel(n_elements=5),
    'gen-kelvin-7': lambda: GeneralizedKelvinModel(n_elements=7),
    'gen-kelvin-9': lambda: GeneralizedKelvinModel(n_elements=9),
    'sls-creep': lambda: GeneralizedKelvinModel(n_elements=1),  # Creep response of a Standard Linear Solid
    'maxwell-2': lambda: GeneralizedMaxwellModel(n_elements=2),
    'maxwell-3': lambda: GeneralizedMaxwellModel(n_elements=3),
    'power-law': PowerLawModel,
    'burgers': BurgersModel,
    'prony-3': lambda: PronySeriesModel(n_terms=3),
    'prony-5': lambda: PronySeriesModel(n_terms=5),
    'peng-2': lambda: PengStressLockModel(n_locked_elements=2),
    'peng-3': lambda: PengStressLockModel(n_locked_elements=3),
    'peng-vp': lambda: PengStressLockModel(n_locked_elements=2, include_viscoplastic=True),
    'peng-3vp': lambda: PengStressLockModel(n_locked_elements=3, include_viscoplastic=True),
}


def get_model(model_name, **kwargs):
    """
    Factory function to get model instance.

    Args:
        model_name: Name of model from AVAILABLE_MODELS
        **kwargs: Additional arguments (e.g., n_elements for KV)

    Returns:
        CreepModel instance

    Example:
        >>> model = get_model('gen-kelvin-5')
        >>> model = get_model('burgers')
    """
    if model_name not in AVAILABLE_MODELS:
        available = ', '.join(AVAILABLE_MODELS.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")

    model_class = AVAILABLE_MODELS[model_name]

    # Handle lambda factories vs direct classes
    if callable(model_class):
        if isinstance(model_class, type):
            # Direct class (e.g., BurgersModel)
            return model_class()
        else:
            # Lambda factory (e.g., KelvinVoigtModel with fixed n_elements)
            return model_class()

    return model_class(**kwargs)


def list_available_models():
    """Print all available models with descriptions."""
    print("\n" + "="*70)
    print("AVAILABLE CREEP MODELS")
    print("="*70)

    for name in sorted(AVAILABLE_MODELS.keys()):
        model = get_model(name)
        print(f"\n{name:20s} - {model.name}")
        print(f"{'':20s}   {model.n_params} parameters")

    print("\n" + "="*70)


if __name__ == "__main__":
    # Test model creation
    list_available_models()
