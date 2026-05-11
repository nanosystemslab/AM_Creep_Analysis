"""
Analysis functions for nanoindentation creep data processing.
Contains all calculation and data processing functions.
"""

import logging
import numpy as np
import pandas as pd
from scipy.stats import linregress

from utils import (
    calculate_berkovich_compliance,
    calculate_conical_compliance,
    calculate_flat_punch_compliance,
    BERKOVICH_HALF_ANGLE,
    CONICAL_60_HALF_ANGLE,
    CONICAL_90_HALF_ANGLE,
)


def calculate_creep_compliance(df, hold_info, load_column=None,
                               probe_type='berkovich', poisson_ratio=0.43,
                               flat_punch_area_m2=None):
    """
    Calculate shear creep compliance J_s(t) using contact-mechanics formulas
    from utils.py, applied to data during the hold period.

    Formulas (all return shear creep compliance 1/G):
        Conical/Berkovich (Peng et al. 2015):
            J_s(t) = [4 tan α] / [π(1-ν)F₀] × h²(t)
        Flat punch (Harding & Sneddon 1945):
            J_s(t) = 4R/(1-ν) × h(t)/P(t)

    Args:
        df: DataFrame with test data.
        hold_info: Dict with 't_start', 't_end', 'success' keys.
        load_column: Name of the load column (auto-detects if None).
        probe_type: 'berkovich', 'conical_60', 'conical_90', or 'flat_punch'.
        poisson_ratio: Poisson's ratio (default 0.43).
        flat_punch_area_m2: Contact area in m² (required for flat_punch).

    Returns:
        DataFrame with 'Shear Creep Compliance (1/GPa)' column added.
    """
    # Auto-detect load column if not specified
    if load_column is None:
        if "Indent Load (µN)" in df.columns:
            load_column = "Indent Load (µN)"
        else:
            logging.warning("[WARN] No load column found for compliance calculation")
            return df

    # Check for required columns
    required_cols = ["Indent Disp. (nm)", load_column, "Test Time (s)"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logging.warning(f"[WARN] Missing columns for compliance calculation: {missing_cols}")
        return df

    if not hold_info or not hold_info.get('success'):
        logging.warning("[WARN] Valid hold period required for creep compliance calculation.")
        return df

    # Validate flat punch area
    if probe_type == 'flat_punch' and flat_punch_area_m2 is None:
        logging.warning("[WARN] flat_punch probe requires flat_punch_area_m2 parameter")
        return df

    # Extract hold period data
    t_start_hold = hold_info['t_start']
    t_end_hold = hold_info['t_end']
    hold_mask = (df["Test Time (s)"] >= t_start_hold) & (df["Test Time (s)"] <= t_end_hold)

    if not hold_mask.any():
        logging.warning("[WARN] No data within the detected hold period.")
        return df

    displacement_hold = df.loc[hold_mask, "Indent Disp. (nm)"].values
    load_hold = df.loc[hold_mask, load_column].values
    time_hold = df.loc[hold_mask, "Test Time (s)"].values

    # Filter out invalid points instead of aborting entirely
    valid_pts = (displacement_hold > 0) & (load_hold > 0)
    n_invalid = np.sum(~valid_pts)
    if n_invalid > 0:
        logging.warning(f"[WARN] {n_invalid}/{len(valid_pts)} invalid (≤0) points in hold period, filtering them out")
    if not np.any(valid_pts):
        logging.warning("[WARN] No valid displacement/load values in hold period.")
        return df

    # Trim loading ramp and unloading artifact using 95% of stable hold load
    # Estimate true hold load from the middle 50% of data (avoids both ramp and unload)
    n_pts = np.sum(valid_pts)
    load_for_stable = load_hold[valid_pts]
    q1 = n_pts // 4
    q3 = 3 * n_pts // 4
    if q3 > q1:
        hold_load_ref = np.mean(load_for_stable[q1:q3])
    else:
        hold_load_ref = np.mean(load_for_stable)
    load_threshold = 0.95 * hold_load_ref

    # Find points below threshold (loading ramp at start, unloading at end)
    below_threshold = load_hold < load_threshold

    # Trim loading ramp: everything up to the last sub-threshold point at the start
    ramp_end_idx = 0
    for i in range(len(below_threshold)):
        if below_threshold[i]:
            ramp_end_idx = i + 1
        else:
            break

    # Trim unloading: everything from the first sub-threshold point at the end
    unload_start_idx = len(below_threshold)
    for i in range(len(below_threshold) - 1, -1, -1):
        if below_threshold[i]:
            unload_start_idx = i
        else:
            break

    n_ramp = ramp_end_idx
    n_unload = len(below_threshold) - unload_start_idx

    if n_ramp > 0 or n_unload > 0:
        if n_ramp > 0:
            logging.info(f"[INFO] Trimming {n_ramp} loading ramp points "
                         f"(load < {load_threshold:.1f} = 95% of {hold_load_ref:.1f}, "
                         f"ramp ends at t={time_hold[min(ramp_end_idx, len(time_hold)-1)]:.2f}s)")
        if n_unload > 0:
            logging.info(f"[INFO] Trimming {n_unload} unloading points "
                         f"(load drops below {load_threshold:.1f} at "
                         f"t={time_hold[max(0, unload_start_idx-1)]:.2f}s)")

        if ramp_end_idx < unload_start_idx:
            displacement_hold = displacement_hold[ramp_end_idx:unload_start_idx]
            load_hold = load_hold[ramp_end_idx:unload_start_idx]
            time_hold = time_hold[ramp_end_idx:unload_start_idx]

            valid_pts = (displacement_hold > 0) & (load_hold > 0)
            if not np.any(valid_pts):
                logging.warning("[WARN] No valid data after trimming ramp/unload.")
                return df

            # Rebuild hold_mask to match trimmed data
            hold_mask = (df["Test Time (s)"] >= time_hold[0]) & (df["Test Time (s)"] <= time_hold[-1])
        else:
            logging.warning("[WARN] Ramp and unload trimming overlap — no valid hold data remains.")
            return df

    disp_valid = displacement_hold[valid_pts]
    load_valid = load_hold[valid_pts]

    # Calculate shear creep compliance using the appropriate utils.py formula
    probe_type_lower = probe_type.lower()
    if probe_type_lower == 'berkovich':
        compliance_valid = calculate_berkovich_compliance(
            disp_valid, load_valid, poisson_ratio
        )
        formula_desc = f"Berkovich (α={BERKOVICH_HALF_ANGLE}°)"
    elif probe_type_lower == 'conical_60':
        compliance_valid = calculate_conical_compliance(
            disp_valid, load_valid, CONICAL_60_HALF_ANGLE, poisson_ratio
        )
        formula_desc = f"Conical (α={CONICAL_60_HALF_ANGLE}°)"
    elif probe_type_lower == 'conical_90':
        compliance_valid = calculate_conical_compliance(
            disp_valid, load_valid, CONICAL_90_HALF_ANGLE, poisson_ratio
        )
        formula_desc = f"Conical (α={CONICAL_90_HALF_ANGLE}°)"
    elif probe_type_lower == 'flat_punch':
        compliance_valid = calculate_flat_punch_compliance(
            disp_valid, load_valid, flat_punch_area_m2, poisson_ratio
        )
        formula_desc = f"Flat punch (A={flat_punch_area_m2:.2e} m²)"
    else:
        logging.warning(f"[WARN] Unknown probe type '{probe_type}', defaulting to berkovich")
        compliance_valid = calculate_berkovich_compliance(
            disp_valid, load_valid, poisson_ratio
        )
        formula_desc = f"Berkovich (α={BERKOVICH_HALF_ANGLE}°, fallback)"

    # Build full-length compliance column (NaN outside hold period and for invalid points)
    compliance_full = np.full(len(df), np.nan)
    compliance_hold = np.full(len(displacement_hold), np.nan)
    compliance_hold[valid_pts] = compliance_valid
    compliance_full[hold_mask.values] = compliance_hold

    df["Shear Creep Compliance (1/GPa)"] = compliance_full

    # Build creep time column: re-zero to first valid compliance point
    creep_time_full = np.full(len(df), np.nan)
    t0_creep = time_hold[0] if len(time_hold) > 0 else t_start_hold
    creep_time_hold = time_hold - t0_creep
    creep_time_full[hold_mask.values] = np.where(
        valid_pts, creep_time_hold, np.nan
    ) if len(creep_time_hold) == len(valid_pts) else creep_time_hold
    df["Creep Time (s)"] = creep_time_full

    valid_count = np.sum(np.isfinite(compliance_hold))
    valid_ct = creep_time_full[np.isfinite(creep_time_full)]
    if valid_count > 0:
        logging.info(f"[INFO] Shear creep compliance calculated: {formula_desc}, ν={poisson_ratio}")
        logging.info(f"  {valid_count} valid points during hold (creep t=0 at test t={t0_creep:.1f}s, "
                     f"ends {t_end_hold:.1f}s)")
        logging.info(f"  Creep time range: {valid_ct.min():.4f} to {valid_ct.max():.2f}s ({len(valid_ct)} pts)")
        logging.info(f"  Compliance range: {np.nanmin(compliance_hold):.4e} to {np.nanmax(compliance_hold):.4e} (1/GPa)")
    else:
        logging.warning("[WARN] No valid compliance points calculated.")

    return df


def calculate_stress(df):
    """
    Calculate stress σ = Load/Contact_Area during the test.
    Returns stress in GPa units.
    """
    required_cols = ["Indent Load (µN)", "Contact Area (nm^2)"]

    # Check if required columns exist
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logging.warning(f"[WARN] Missing columns for stress calculation: {missing_cols}")
        return df

    # Get data arrays
    load = df["Indent Load (µN)"].values  # µN
    contact_area = df["Contact Area (nm^2)"].values  # nm^2
    
    # Create mask for valid data (positive values)
    valid_load_mask = load > 0
    valid_area_mask = contact_area > 0
    valid_mask = valid_load_mask & valid_area_mask
    
    if not np.any(valid_mask):
        logging.warning("[WARN] No valid data points for stress calculation")
        return df
    
    # Initialize stress array
    stress = np.full_like(load, np.nan, dtype=float)
    
    # Calculate stress for valid points
    # Convert: µN/nm² to GPa
    # 1 µN = 1e-6 N, 1 nm² = 1e-18 m²
    # So µN/nm² = 1e12 Pa = 1e3 GPa
    # Therefore: stress_GPa = (load_µN / area_nm²) * 1e-3
    
    stress[valid_mask] = (load[valid_mask] / contact_area[valid_mask]) * 1e-3  # GPa
    
    # Add to dataframe
    df["Stress (GPa)"] = stress
    
    valid_count = np.sum(valid_mask)
    if valid_count > 0:
        stress_range = (np.nanmin(stress), np.nanmax(stress))
        logging.info(f"[INFO] Stress calculated: {valid_count} valid points, "
                    f"range: {stress_range[0]:.3f} to {stress_range[1]:.3f} GPa")
    
    return df


def calculate_creep_strain(df, creep_hold_start=None, creep_hold_end=None):
    """
    Calculate creep strain with improved hold period detection.

    Args:
        df: DataFrame with nanoindentation data
        creep_hold_start: Time (s) when hold period starts
        creep_hold_end: Time (s) when hold period ends (optional)

    Returns:
        DataFrame with 'Creep Strain' column added
    """
    if "Indent Disp. (nm)" not in df.columns or "Test Time (s)" not in df.columns:
        logging.warning("[WARN] Required columns not found for creep strain calculation")
        return df

    # Method 1: If creep_hold_start is specified (recommended for CMX)
    if creep_hold_start is not None:
        # Find the displacement at the start of the hold period
        time_mask = df["Test Time (s)"] >= creep_hold_start
        valid_disp_mask = df["Indent Disp. (nm)"].notna() & (df["Indent Disp. (nm)"] != 0)
        hold_start_mask = time_mask & valid_disp_mask

        if hold_start_mask.any():
            # Get the first valid displacement at or after hold start time
            hold_start_idx = hold_start_mask.idxmax()
            h0_hold = df.loc[hold_start_idx, "Indent Disp. (nm)"]

            # Calculate creep strain relative to hold period start
            df["Creep Strain"] = (df["Indent Disp. (nm)"] - h0_hold) / abs(h0_hold)

            # Set creep strain to NaN before hold period starts
            df.loc[df["Test Time (s)"] < creep_hold_start, "Creep Strain"] = np.nan

            # Set creep strain to NaN after hold period ends (if specified)
            if creep_hold_end is not None:
                df.loc[df["Test Time (s)"] > creep_hold_end, "Creep Strain"] = np.nan
                logging.info(f"[INFO] Creep Strain calculated (hold: {creep_hold_start}s to {creep_hold_end}s, h0: {h0_hold:.2f} nm)")
            else:
                logging.info(f"[INFO] Creep Strain calculated (hold start: {creep_hold_start}s, h0: {h0_hold:.2f} nm)")

            # Set creep strain to NaN for invalid displacement values
            df.loc[~valid_disp_mask, "Creep Strain"] = np.nan
        else:
            df["Creep Strain"] = np.nan
            logging.warning(f"[WARN] No valid data at hold start time {creep_hold_start}s")
    
    # Method 2: Auto-detect hold period (fallback)
    else:
        # Try to automatically detect hold period by finding plateau in displacement
        valid_disp_mask = df["Indent Disp. (nm)"].notna() & (df["Indent Disp. (nm)"] != 0)
        
        if valid_disp_mask.any():
            # Calculate gradient to find where displacement rate becomes small
            disp_vals = df.loc[valid_disp_mask, "Indent Disp. (nm)"].values
            time_vals = df.loc[valid_disp_mask, "Test Time (s)"].values
            
            if len(disp_vals) > 10:  # Need enough points for gradient
                # Smooth gradient calculation
                window_size = min(20, len(disp_vals) // 10)
                gradient = np.gradient(disp_vals, time_vals)
                
                # Find where gradient becomes small (< 5% of max gradient)
                max_gradient = np.max(np.abs(gradient))
                threshold = 0.05 * max_gradient
                
                # Find first point where gradient stays below threshold
                low_gradient_mask = np.abs(gradient) < threshold
                if np.any(low_gradient_mask):
                    # Find first sustained low gradient region
                    low_indices = np.where(low_gradient_mask)[0]
                    hold_start_data_idx = low_indices[0]
                    hold_start_original_idx = df.loc[valid_disp_mask].iloc[hold_start_data_idx].name
                    h0_auto = df.loc[hold_start_original_idx, "Indent Disp. (nm)"]
                    
                    # Calculate creep strain from auto-detected start
                    df["Creep Strain"] = (df["Indent Disp. (nm)"] - h0_auto) / abs(h0_auto)
                    
                    # Set creep strain to NaN before auto-detected hold start
                    hold_start_time = df.loc[hold_start_original_idx, "Test Time (s)"]
                    df.loc[df["Test Time (s)"] < hold_start_time, "Creep Strain"] = np.nan
                    df.loc[~valid_disp_mask, "Creep Strain"] = np.nan
                    
                    logging.info(f"[INFO] Auto-detected creep strain (hold start: {hold_start_time:.1f}s, h0: {h0_auto:.2f} nm)")
                else:
                    # Fallback to original method
                    first_valid_idx = valid_disp_mask.idxmax()
                    initial_disp = df.loc[first_valid_idx, "Indent Disp. (nm)"]
                    df["Creep Strain"] = (df["Indent Disp. (nm)"] - initial_disp) / abs(initial_disp)
                    df.loc[~valid_disp_mask, "Creep Strain"] = np.nan
                    logging.info(f"[INFO] Fallback creep strain (initial disp: {initial_disp:.2f} nm)")
            else:
                # Too few points - use original method
                first_valid_idx = valid_disp_mask.idxmax()
                initial_disp = df.loc[first_valid_idx, "Indent Disp. (nm)"]
                df["Creep Strain"] = (df["Indent Disp. (nm)"] - initial_disp) / abs(initial_disp)
                df.loc[~valid_disp_mask, "Creep Strain"] = np.nan
                logging.info(f"[INFO] Basic creep strain (initial disp: {initial_disp:.2f} nm)")
        else:
            df["Creep Strain"] = np.nan
            logging.warning("[WARN] No valid displacement data")
    
    return df


def calculate_creep_rate(df, compute_creep):
    """
    Calculate creep rate in specified time window.
    """
    if not compute_creep or "Indent Disp. (nm)" not in df.columns or "Test Time (s)" not in df.columns:
        return df, None
    
    t_start, t_end = compute_creep
    mask = (df["Test Time (s)"] >= t_start) & (df["Test Time (s)"] <= t_end)
    
    if not mask.any():
        return df, None
    
    # Calculate creep rate in specified window
    time_data = df.loc[mask].copy()
    time_vals = time_data["Test Time (s)"].values
    disp_vals = time_data["Indent Disp. (nm)"].values
    
    if len(time_vals) > 1:
        disp_gradient = np.gradient(disp_vals, time_vals)
        
        # Calculate normalized creep rate (1/h * dh/dt)
        # Avoid division by zero
        valid_disp = disp_vals != 0
        norm_creep_rate = np.full_like(disp_vals, np.nan)
        norm_creep_rate[valid_disp] = disp_gradient[valid_disp] / disp_vals[valid_disp]
        
        # Store in dataframe
        df.loc[mask, "Creep Rate (1/s)"] = norm_creep_rate
        df.loc[mask, "Creep Rate (nm/s)"] = disp_gradient
        
        # Calculate average for reporting
        valid_norm_rates = norm_creep_rate[~np.isnan(norm_creep_rate)]
        if len(valid_norm_rates) > 0:
            avg_norm_creep_rate = valid_norm_rates.mean()
            logging.info(f"[INFO] Creep Rate calculated: {avg_norm_creep_rate:.6f} (1/s)")
            return df, avg_norm_creep_rate
    
    return df, None


def calculate_strain_rate(df, compute_strain_rate):
    """
    Calculate strain rate in specified time window.
    """
    if not compute_strain_rate or "Creep Strain" not in df.columns or "Test Time (s)" not in df.columns:
        return df, None
    
    t_start, t_end = compute_strain_rate
    mask = (df["Test Time (s)"] >= t_start) & (df["Test Time (s)"] <= t_end)
    
    if not mask.any():
        return df, None
    
    # Calculate strain rate in specified window
    time_data = df.loc[mask].copy()
    time_vals = time_data["Test Time (s)"].values
    strain_vals = time_data["Creep Strain"].values
    
    # Remove NaN values
    valid_mask = ~np.isnan(strain_vals) & ~np.isnan(time_vals)
    if np.sum(valid_mask) > 1:
        time_vals_clean = time_vals[valid_mask]
        strain_vals_clean = strain_vals[valid_mask]
        
        # Calculate strain rate (dε/dt)
        strain_gradient = np.gradient(strain_vals_clean, time_vals_clean)
        
        # Store in dataframe - map back to original indices
        valid_indices = np.where(mask)[0][valid_mask]
        df.loc[df.index[valid_indices], "Strain Rate (1/s)"] = strain_gradient
        
        # Calculate average for reporting
        if len(strain_gradient) > 0:
            avg_strain_rate = np.mean(strain_gradient)
            logging.info(f"[INFO] Strain Rate calculated: {avg_strain_rate:.6e} (1/s)")
            return df, avg_strain_rate
    else:
        logging.warning("[WARN] Not enough valid strain data for strain rate calculation")
    
    return df, None


def calculate_elastic_modulus(df):
    """
    Calculate effective elastic modulus.
    """
    if "Storage Mod. (GPa)" in df.columns and "Loss Mod. (GPa)" in df.columns:
        df["Effective Mod. (GPa)"] = np.sqrt(df["Storage Mod. (GPa)"]**2 + df["Loss Mod. (GPa)"]**2)
        logging.info("[INFO] Elastic Modulus calculated")
    return df


def apply_drift_correction(df, correct_drift):
    """
    Apply drift correction to displacement data.
    """
    if correct_drift and "Indent Disp. (nm)" in df.columns and "Test Time (s)" in df.columns:
        drift_nm = correct_drift * df["Test Time (s)"]
        df["Indent Disp. (nm)"] -= drift_nm
        logging.info(f"[INFO] Drift corrected by {correct_drift} nm/s")
    return df


def compute_strain_recovery(df, time_col, disp_col, t_start, t_end):
    """
    Compute strain recovery during specified time window.
    """
    hold = df[(df[time_col] >= t_start) & (df[time_col] <= t_end)].copy()
    if hold.empty:
        logging.warning("[WARN] No data found in the specified time window for recovery.")
        return None

    # Find peak displacement
    disp_max = hold[disp_col].max()
    peak_time = hold[hold[disp_col] == disp_max][time_col].values[0]

    # Only consider data after the peak
    after_peak = hold[hold[time_col] >= peak_time]
    disp_min = after_peak[disp_col].min()
    final_time = after_peak[after_peak[disp_col] == disp_min][time_col].values[0]

    recovery_percent = (disp_max - disp_min) / disp_max * 100
    return recovery_percent, disp_max, disp_min, peak_time


def fit_linear_sections(x, y, sections=None, log_x=True, log_y=False):
    """
    Fit linear sections to data, handling log transformations.
    """
    results = []

    # Apply log transform if needed
    if log_x:
        positive_mask = x > 0
        if not np.all(positive_mask):
            print(f"[WARN] Removing {np.sum(~positive_mask)} non-positive x values for log transform")
            x = x[positive_mask]
            y = y[positive_mask]
        x_data = np.log10(x)
    else:
        x_data = x.copy()

    if log_y:
        positive_mask = y > 0
        if not np.all(positive_mask):
            print(f"[WARN] Removing {np.sum(~positive_mask)} non-positive y values for log transform")
            if log_x:
                x_data = x_data[positive_mask]
            else:
                x = x[positive_mask]
            y = y[positive_mask]
        y_data = np.log10(y)
    else:
        y_data = y.copy()

    # Process specified fit sections
    if sections:
        for xmin, xmax in sections:
            # Convert to log space if needed
            if log_x:
                if xmin <= 0:
                    min_positive_x = np.min(x[x > 0])
                    print(f"[WARN] Replacing invalid log value {xmin} with {min_positive_x}")
                    xmin = min_positive_x
                if xmax <= 0:
                    xmax = xmin * 1.1

                xmin_log = np.log10(xmin)
                xmax_log = np.log10(xmax)

                # Find points in range
                mask = (x_data >= xmin_log) & (x_data <= xmax_log)
            else:
                mask = (x_data >= xmin) & (x_data <= xmax)

            # Check if we have enough points
            if np.sum(mask) >= 2:
                # Perform linear regression
                slope, intercept, r, p, stderr = linregress(x_data[mask], y_data[mask])

                # Store results
                if log_x:
                    x_orig_range = (10**xmin_log, 10**xmax_log)
                else:
                    x_orig_range = (xmin, xmax)

                results.append({
                    "slope": slope,
                    "intercept": intercept,
                    "r_squared": r**2,
                    "x_range": (xmin_log if log_x else xmin, xmax_log if log_x else xmax),
                    "x_orig_range": x_orig_range,
                    "points_used": np.sum(mask)
                })
                print(f"[INFO] Fit: slope={slope:.4f}, intercept={intercept:.4f}, R²={r**2:.4f}")
            else:
                print(f"[WARN] Not enough points in range {xmin}-{xmax}")

    return results


def detect_hold_period(df, control_column, time_column='Test Time (s)',
                       threshold_percentile=5, min_duration=1.0, smoothing_window=10):
    """
    Detect hold period by finding where control variable plateaus.

    Works for both load-controlled and displacement-controlled tests:
    - LC tests: control_column = 'Indent Load (µN)' -> detects creep hold
    - DC tests: control_column = 'Indent Disp. (nm)' -> detects relaxation hold

    Args:
        df: DataFrame with test data
        control_column: Column to monitor for plateau
        time_column: Time column name (default: 'Test Time (s)')
        threshold_percentile: Gradient threshold as % of max gradient (default: 5%)
        min_duration: Minimum hold duration in seconds (default: 1.0s)
        smoothing_window: Window size for gradient smoothing (default: 10 points)

    Returns:
        dict with keys:
            - 't_start': Hold period start time (s)
            - 't_end': Hold period end time (s)
            - 'control_value': Mean control variable value during hold
            - 'duration': Hold period duration (s)
            - 'success': Boolean indicating if hold was detected
            - 'method': Detection method used ('auto' or 'failed')
        Or None if detection fails
    """

    # Validate inputs
    if time_column not in df.columns:
        logging.error(f"[ERROR] Time column '{time_column}' not found")
        return None

    if control_column not in df.columns:
        logging.error(f"[ERROR] Control column '{control_column}' not found")
        return None

    # Get data arrays
    time = df[time_column].values
    control = df[control_column].values

    # Remove NaN values
    valid_mask = ~np.isnan(time) & ~np.isnan(control)
    if not np.any(valid_mask):
        logging.error("[ERROR] No valid data for hold detection")
        return None

    time = time[valid_mask]
    control = control[valid_mask]

    if len(time) < 20:
        logging.error(f"[ERROR] Insufficient data points ({len(time)}) for hold detection")
        return None

    # Calculate gradient (rate of change)
    gradient = np.gradient(control, time)

    # Smooth gradient to reduce noise
    if smoothing_window > 1:
        window = np.ones(smoothing_window) / smoothing_window
        gradient_smooth = np.convolve(gradient, window, mode='same')
    else:
        gradient_smooth = gradient

    # Find maximum gradient magnitude
    max_gradient = np.max(np.abs(gradient_smooth))

    if max_gradient == 0:
        logging.error("[ERROR] Zero gradient detected - no change in control variable")
        return None

    # Calculate threshold for "low gradient" (plateau region)
    threshold = (threshold_percentile / 100.0) * max_gradient

    # Find regions where gradient is below threshold
    low_gradient_mask = np.abs(gradient_smooth) < threshold

    # Find ALL plateaus that meet min_duration requirement, then select the longest
    plateaus = []
    i = 0
    while i < len(low_gradient_mask):
        if low_gradient_mask[i]:
            # Start of potential plateau
            plateau_start_idx = i
            plateau_start_time = time[i]

            # Find end of plateau
            j = i
            while j < len(low_gradient_mask) and low_gradient_mask[j]:
                j += 1

            plateau_end_idx = j - 1 if j > i else i
            plateau_end_time = time[plateau_end_idx]
            plateau_duration = plateau_end_time - plateau_start_time

            # Check if plateau meets minimum duration
            if plateau_duration >= min_duration:
                # Store this plateau for comparison
                plateaus.append({
                    'start_idx': plateau_start_idx,
                    'end_idx': plateau_end_idx,
                    'start_time': plateau_start_time,
                    'end_time': plateau_end_time,
                    'duration': plateau_duration
                })

            # Move to next potential plateau
            i = j
        else:
            i += 1

    # Select the longest plateau
    if plateaus:
        # Sort by duration and select longest
        longest_plateau = max(plateaus, key=lambda p: p['duration'])

        plateau_start_idx = longest_plateau['start_idx']
        plateau_end_idx = longest_plateau['end_idx']
        plateau_start_time = longest_plateau['start_time']
        plateau_end_time = longest_plateau['end_time']
        plateau_duration = longest_plateau['duration']

        control_mean = np.mean(control[plateau_start_idx:plateau_end_idx+1])
        control_std = np.std(control[plateau_start_idx:plateau_end_idx+1])

        logging.info(f"[INFO] Hold period detected (longest of {len(plateaus)} plateaus):")
        logging.info(f"  Time range: {plateau_start_time:.2f}s to {plateau_end_time:.2f}s")
        logging.info(f"  Duration: {plateau_duration:.2f}s")
        logging.info(f"  Control variable: {control_mean:.2f} ± {control_std:.2f}")
        logging.info(f"  Points in hold: {plateau_end_idx - plateau_start_idx + 1}")

        return {
            't_start': plateau_start_time,
            't_end': plateau_end_time,
            'control_value': control_mean,
            'control_std': control_std,
            'duration': plateau_duration,
            'n_points': plateau_end_idx - plateau_start_idx + 1,
            'success': True,
            'method': 'auto'
        }

    # No suitable plateau found
    logging.warning(f"[WARN] No hold period detected with threshold {threshold_percentile}% over {min_duration}s")
    logging.warning(f"  Max gradient: {max_gradient:.6f}, Threshold: {threshold:.6f}")
    logging.warning(f"  Try adjusting threshold_percentile or min_duration")

    return {
        'success': False,
        'method': 'failed',
        'max_gradient': max_gradient,
        'threshold_used': threshold
    }


def validate_hold_period(df, hold_info, control_column, response_column,
                        time_column='Test Time (s)', plot=False):
    """
    Validate detected hold period by checking control stability and response change.

    Args:
        df: DataFrame with test data
        hold_info: Output from detect_hold_period()
        control_column: Column that should be constant (load or displacement)
        response_column: Column that should change (displacement or load)
        time_column: Time column name
        plot: If True, create validation plot

    Returns:
        dict with validation metrics
    """
    if not hold_info or not hold_info.get('success'):
        logging.error("[ERROR] Cannot validate failed hold detection")
        return None

    # Extract hold period data
    t_start = hold_info['t_start']
    t_end = hold_info['t_end']

    # Get data in hold period
    mask = (df[time_column] >= t_start) & (df[time_column] <= t_end)
    hold_data = df[mask]

    if len(hold_data) < 2:
        logging.error("[ERROR] Insufficient data in hold period")
        return None

    # Calculate control variable stability (should be low)
    control_values = hold_data[control_column].values
    control_cv = np.std(control_values) / np.abs(np.mean(control_values)) * 100  # Coefficient of variation (%)

    # Calculate response variable change (should be significant)
    response_values = hold_data[response_column].values
    response_change = (response_values[-1] - response_values[0]) / response_values[0] * 100  # Percent change

    # Validation criteria
    control_stable = control_cv < 5.0  # Control should vary < 5%
    response_changing = abs(response_change) > 0.1  # Response should change > 0.1%

    validation = {
        'control_cv_percent': control_cv,
        'response_change_percent': response_change,
        'control_stable': control_stable,
        'response_changing': response_changing,
        'valid': control_stable and response_changing
    }

    if validation['valid']:
        logging.info("[INFO] Hold period validation PASSED")
    else:
        logging.warning("[WARN] Hold period validation FAILED")
        if not control_stable:
            logging.warning(f"  Control variable too variable: {control_cv:.2f}% CV")
        if not response_changing:
            logging.warning(f"  Response variable not changing enough: {response_change:.2f}%")

    return validation