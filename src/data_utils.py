"""
Data loading and utility functions for nanoindentation creep analysis.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import hashlib
import json
from analysis_functions import (
    calculate_creep_compliance, calculate_stress as calc_stress, calculate_creep_strain,
    calculate_creep_rate, calculate_strain_rate, calculate_elastic_modulus,
    apply_drift_correction, detect_hold_period
)


# Cache for method configuration
_method_config_cache = None


def load_method_config(config_path=None):
    """
    Load method configuration from JSON file.

    Args:
        config_path: Path to config file. If None, searches in standard locations.

    Returns:
        dict with method configurations
    """
    global _method_config_cache

    # Return cached config if available
    if _method_config_cache is not None:
        return _method_config_cache

    # Search for config file
    if config_path is None:
        # Look in standard locations
        search_paths = [
            Path.cwd() / 'method_config.json',
            Path(__file__).parent.parent / 'method_config.json',
            Path('/Users/ethan/Desktop/AM_Creep_Analysis/method_config.json'),
        ]

        for path in search_paths:
            if path.exists():
                config_path = path
                break

    if config_path is None or not Path(config_path).exists():
        logging.warning("[WARN] Method config file not found. Using default behavior.")
        return {"methods": {}, "default": {"hold_start": None, "hold_end": None}}

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        logging.info(f"[INFO] Loaded method config from: {config_path}")
        _method_config_cache = config
        return config
    except Exception as e:
        logging.error(f"[ERROR] Failed to load method config: {e}")
        return {"methods": {}, "default": {"hold_start": None, "hold_end": None}}


def extract_method_from_path(file_path, config=None):
    """
    Extract method identifier from folder path.
    Tries to match the most specific method in the config first.

    Args:
        file_path: Path object or string path to data file
        config: Optional method config dict to check for matches

    Returns:
        str: Method identifier (e.g., "19", "44", "41-1", "41-8.33"), or None if not found

    Examples:
        "/path/to/19_description/file.txt" -> "19"
        "/path/to/44_test_data/file.txt" -> "44"
        "/path/to/41-1/file.txt" -> "41-1" (if in config)
        "/path/to/41-8.33/file.txt" -> "41-8.33" (if in config)
        "/path/to/50slow_data/file.txt" -> "50slow"
    """
    file_path = Path(file_path)
    folder_name = file_path.parent.name

    # Get the base method name (before any underscore suffix)
    if '_' in folder_name:
        base_name = folder_name.split('_')[0]
    else:
        base_name = folder_name

    # If we have a config, try to find the most specific match
    if config and 'methods' in config:
        methods = config['methods']

        # First, try the exact base name (e.g., "41-1", "41-8.33")
        if base_name in methods:
            return base_name

        # Try progressively shorter prefixes by removing from the end
        # This handles cases like "41-833" -> check "41-83" -> check "41-8" -> check "41"
        if '-' in base_name:
            parts = base_name.split('-')
            # Try combining parts: "41-1", then just "41"
            for i in range(len(parts), 0, -1):
                candidate = '-'.join(parts[:i])
                if candidate in methods:
                    return candidate

            # Try skipping non-numeric middle parts (material names)
            # e.g., "41-pc-1" -> "41-1", "41-t1500-83.33" -> "41-83.33"
            if len(parts) >= 3:
                numeric_parts = [p for p in parts if p and p[0].isdigit()]
                if len(numeric_parts) >= 2:
                    candidate = f"{numeric_parts[0]}-{numeric_parts[-1]}"
                    if candidate in methods:
                        return candidate

    # Fallback: split by underscore or hyphen and take first part
    if '_' in folder_name:
        parts = folder_name.split('_')
    elif '-' in folder_name:
        parts = folder_name.split('-')
    else:
        parts = [folder_name]

    if parts and len(parts[0]) > 0:
        # Check if it starts with a digit (handles "19", "50slow", "103", "44", etc.)
        if parts[0][0].isdigit():
            return parts[0]

    logging.warning(f"[WARN] Could not extract method identifier from folder: {folder_name}")
    return None


def get_method_hold_period(file_path, config=None):
    """
    Get hold period configuration for a given file based on its method.

    Args:
        file_path: Path to data file
        config: Method configuration dict (optional, will load if not provided)

    Returns:
        dict with keys: 'hold_start', 'hold_end', 'method'
        Returns default values if method not found in config
    """
    if config is None:
        config = load_method_config()

    method = extract_method_from_path(file_path, config)

    if method and method in config.get("methods", {}):
        method_config = config["methods"][method]
        result = {
            'hold_start': method_config.get('hold_start'),
            'hold_end': method_config.get('hold_end'),
            'method': method,
            'description': method_config.get('description', f'Method {method}'),
            'probe': method_config.get('probe', 'berkovich')
        }
        logging.info(f"[INFO] Method {method} detected: hold period {result['hold_start']}s to {result['hold_end']}s, probe={result['probe']}")
        return result
    else:
        default_config = config.get("default", {})
        result = {
            'hold_start': default_config.get('hold_start'),
            'hold_end': default_config.get('hold_end'),
            'method': method if method else 'unknown',
            'description': 'Default/Auto-detect',
            'probe': default_config.get('probe', 'berkovich')
        }
        if method:
            logging.warning(f"[WARN] Method {method} not in config, using default settings")
        return result


def get_cache_path(files, processing_params):
    """
    Generate a cache file path based on input files and processing parameters.
    
    Args:
        files: List of file paths
        processing_params: Dict of processing parameters that affect output
    
    Returns:
        Path to cache file
    """
    # Create cache directory
    cache_dir = Path('.cache')
    cache_dir.mkdir(exist_ok=True)
    
    # Create a hash of file paths and modification times
    file_info = []
    for f in sorted(files):
        try:
            mtime = f.stat().st_mtime
            file_info.append(f"{f.name}:{mtime}")
        except:
            file_info.append(f.name)
    
    # Add processing parameters to hash
    param_str = "_".join([f"{k}={v}" for k, v in sorted(processing_params.items())])
    
    # Create hash
    hash_input = "|".join(file_info) + "|" + param_str
    cache_hash = hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    # Create readable cache name
    if files:
        folder_name = files[0].parent.name
    else:
        folder_name = "unknown"
    
    cache_file = cache_dir / f"{folder_name}_{cache_hash}.pkl"
    return cache_file


def load_from_cache(cache_path):
    """Load processed data from cache if it exists."""
    if cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            print(f"[INFO] Loaded data from cache: {cache_path.name}")
            return data
        except Exception as e:
            logging.warning(f"[WARN] Failed to load cache: {e}")
            return None
    return None


def save_to_cache(cache_path, data):
    """Save processed data to cache."""
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"[INFO] Saved data to cache: {cache_path.name}")
    except Exception as e:
        logging.warning(f"[WARN] Failed to save cache: {e}")


def clear_old_caches(keep_recent=10):
    """
    Clear old cache files, keeping only the most recent ones.
    
    Args:
        keep_recent: Number of most recent cache files to keep
    """
    cache_dir = Path('.cache')
    if not cache_dir.exists():
        return
    
    cache_files = sorted(cache_dir.glob('*.pkl'), key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Delete old cache files
    for cache_file in cache_files[keep_recent:]:
        try:
            cache_file.unlink()
            print(f"[INFO] Removed old cache: {cache_file.name}")
        except Exception as e:
            logging.warning(f"[WARN] Failed to remove cache {cache_file.name}: {e}")


def resolve_columns(columns, df_columns):
    """
    Resolve column references (convert indices and shortcuts to names).
    """
    col_names = []
    for col in columns:
        if col.lower() == "cr":
            col_names.append("Creep Rate (nm/s)")
        elif col.lower() == "crs":
            col_names.append("Creep Strain")
        elif col.lower() == "crn":
            col_names.append("Creep Rate (1/s)")
        elif col.lower() == "sr":
            col_names.append("Strain Rate (1/s)")
        elif col.lower() == "cc" or col.lower() == "compliance":
            col_names.append("Shear Creep Compliance (1/GPa)")
        elif col.lower() == "ct":
            col_names.append("Creep Time (s)")
        elif col.lower() == "stress" or col.lower() == "s":
            col_names.append("Stress (GPa)")
        elif col.isdigit():
            idx = int(col)
            if 0 <= idx < len(df_columns):
                col_names.append(df_columns[idx].strip())
            else:
                print(f"[WARN] Index {col} out of range. Available indices: 0-{len(df_columns)-1}")
                print(f"[INFO] Available columns: {list(enumerate(df_columns))}")
                # Don't append anything for out-of-range indices
        else:
            col_names.append(col.strip())
    return col_names


def group_files_by_max_load(files, load_tolerance=100):
    """
    Group files by their maximum load value.
    
    Args:
        files: List of file paths
        load_tolerance: Maximum difference in µN to consider loads as "same group" (default: 100)
    
    Returns:
        dict: {max_load: [file_list]}
    """
    import logging
    
    load_groups = {}
    
    for file in files:
        try:
            # Read file to find max load
            with open(file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            header_index = next(i for i, line in enumerate(lines) if "Test Time" in line or "Time" in line)
            df = pd.read_csv(file, sep='\t', skiprows=header_index, engine='python')
            df.columns = [col.strip() for col in df.columns]
            
            # Get max load
            if "Indent Load (µN)" in df.columns:
                df["Indent Load (µN)"] = pd.to_numeric(df["Indent Load (µN)"], errors='coerce')
                max_load = df["Indent Load (µN)"].max()
                
                if pd.isna(max_load):
                    logging.warning(f"[WARN] Could not determine max load for {file.name}")
                    continue
                
                # Round to nearest 10 µN for grouping
                max_load = round(max_load / 10) * 10
                
                # Find existing group within tolerance
                assigned = False
                for group_load in list(load_groups.keys()):
                    if abs(max_load - group_load) <= load_tolerance:
                        load_groups[group_load].append(file)
                        assigned = True
                        break
                
                if not assigned:
                    load_groups[max_load] = [file]
                    
            else:
                logging.warning(f"[WARN] No load column found in {file.name}")
                
        except Exception as e:
            logging.warning(f"[WARN] Could not read {file.name} for load grouping: {e}")
    
    # Sort groups by load
    sorted_groups = dict(sorted(load_groups.items()))
    
    # Print summary
    print("\n[INFO] Load-based grouping summary:")
    for load, file_list in sorted_groups.items():
        print(f"  {load:.0f} µN: {len(file_list)} files")
    print()
    
    return sorted_groups


def find_files(folder_paths, file_type):
    """
    Find files matching the specified pattern in given folders.
    """
    files = []
    
    # Handle both load-controlled and displacement-controlled file types
    if file_type == "DC":
        pattern = f"* DC.txt"  # Displacement controlled files
    else:
        pattern = f"*_{file_type}.txt"  # Load controlled files (DYN, AVG, QS)
    
    for folder in folder_paths:
        files.extend(Path(folder).rglob(pattern))
    return files


def load_and_clean_data(files, correct_drift=None, calculate_elastic=False, compute_creep=None,
                       compute_strain_rate=None, calculate_compliance=False, calculate_stress=False,
                       calculate_creep_strain=False, creep_hold_start=None, creep_hold_end=None,
                       use_method_config=True, use_cache=True,
                       probe_type=None, poisson_ratio=0.43,
                       flat_punch_area_m2=None):
    """
    Load and process data files with various analysis options.
    Uses caching to speed up repeated processing of the same files.

    Args:
        ...
        use_cache: If True, use cached results when available.
    """
    # If caching is disabled by the user, bypass all cache operations
    if not use_cache:
        logging.info("[INFO] Cache disabled by user command.")
    
    # Load method config if using auto-detection
    method_config = load_method_config() if use_method_config else None

    # Create cache key from processing parameters
    # Note: We don't include method-specific hold periods in cache key because
    # they vary per file. Instead, we include the use_method_config flag.
    processing_params = {
        'correct_drift': correct_drift,
        'calculate_elastic': calculate_elastic,
        'compute_creep': str(compute_creep),
        'compute_strain_rate': str(compute_strain_rate),
        'calculate_compliance': calculate_compliance,
        'calculate_stress': calculate_stress,
        'calculate_creep_strain': calculate_creep_strain,
        'creep_hold_start': creep_hold_start,
        'creep_hold_end': creep_hold_end,
        'use_method_config': use_method_config,
        'probe_type': probe_type,
        'poisson_ratio': poisson_ratio,
        'flat_punch_area_m2': flat_punch_area_m2,
    }
    
    # Try to load from cache
    if use_cache:
        cache_path = get_cache_path(files, processing_params)
        cached_data = load_from_cache(cache_path)
        if cached_data is not None:
            # Restore the avg rates that are stored as function attributes
            load_and_clean_data.avg_creep_rates = cached_data.get('avg_creep_rates', [])
            load_and_clean_data.avg_strain_rates = cached_data.get('avg_strain_rates', [])
            return cached_data['dataframes']
    dataframes = []
    first_headers_printed = False
    avg_creep_rates = []  # Store avg creep rates
    avg_strain_rates = []  # Store avg strain rates
    
    for file in files:
        try:
            # Determine hold period for this file
            if use_method_config and (creep_hold_start is None or creep_hold_end is None):
                # Auto-detect method and get hold periods from config
                hold_config = get_method_hold_period(file, method_config)
                file_hold_start = creep_hold_start if creep_hold_start is not None else hold_config['hold_start']
                file_hold_end = creep_hold_end if creep_hold_end is not None else hold_config['hold_end']
            else:
                # Use manually specified values
                hold_config = {}
                file_hold_start = creep_hold_start
                file_hold_end = creep_hold_end

            # Detect file format: DC files vs. standard nanoDMA files
            is_dc_file = " DC.txt" in str(file)

            # Try UTF-8 first, fallback to latin-1 for special characters (µ symbol)
            try:
                encoding = 'utf-8'
                with open(file, 'r', encoding=encoding) as f:
                    lines = f.readlines()
            except UnicodeDecodeError:
                encoding = 'latin-1'
                with open(file, 'r', encoding=encoding) as f:
                    lines = f.readlines()

            if is_dc_file:
                # DC file format:
                # Line 0: Date
                # Line 1: Empty line
                # Line 2: "Number of Points = X"
                # Line 3: Header with SPACE separators "Depth (nm) Load (µN) Time (s) Depth (V) Load (V)"
                # Line 4+: Data with TAB separators
                # Skip first 3 lines (0, 1, 2), then line 3 becomes header
                df = pd.read_csv(file, sep=r'\s+', skiprows=[0, 1, 2], engine='python', encoding=encoding)

                # Standardize column names to match nanoDMA format
                column_mapping = {
                    'Depth': 'Indent Disp. (nm)',
                    'Load': 'Indent Load (µN)',
                    'Time': 'Test Time (s)'
                }
                df.rename(columns=column_mapping, inplace=True)
            else:
                # Standard nanoDMA file format
                header_index = next(i for i, line in enumerate(lines) if "Test Time" in line or "Time" in line)
                df = pd.read_csv(file, sep='\t', skiprows=header_index, engine='python', encoding=encoding)

            df.columns = [col.strip() for col in df.columns]

            # Calculate creep strain (only if requested)
            if calculate_creep_strain:
                df = calculate_creep_strain(df, file_hold_start, file_hold_end)

            # Calculate creep rate
            if compute_creep:
                df, avg_rate = calculate_creep_rate(df, compute_creep)
                if avg_rate is not None:
                    avg_creep_rates.append(avg_rate)

            # Calculate strain rate
            if compute_strain_rate:
                df, avg_rate = calculate_strain_rate(df, compute_strain_rate)
                if avg_rate is not None:
                    avg_strain_rates.append(avg_rate)

            # Calculate stress
            if calculate_stress:
                df = calc_stress(df)

            # Calculate creep compliance
            if calculate_compliance:
                # Use Indent Load column (not Indent Act. Load, which is actuation load)
                if "Indent Load (µN)" in df.columns:
                    load_col = "Indent Load (µN)"
                else:
                    load_col = None

                # Build hold_info dict for compliance calculation
                if file_hold_start is not None and file_hold_end is not None:
                    # Use hold times from config or manual specification
                    hold_info = {
                        't_start': file_hold_start,
                        't_end': file_hold_end,
                        'success': True,
                        'method': 'config'
                    }
                else:
                    # Auto-detect hold period from load plateau
                    if load_col:
                        hold_info = detect_hold_period(
                            df,
                            control_column=load_col,
                            time_column="Test Time (s)"
                        )
                    else:
                        logging.warning("[WARN] Cannot auto-detect hold period - no load column found")
                        hold_info = None

                # Determine probe type: CLI override > method config > default
                file_probe = probe_type if probe_type else hold_config.get('probe', 'berkovich')

                if hold_info and hold_info.get('success'):
                    df = calculate_creep_compliance(
                        df, hold_info,
                        load_column=load_col,
                        probe_type=file_probe,
                        poisson_ratio=poisson_ratio,
                        flat_punch_area_m2=flat_punch_area_m2
                    )
                else:
                    logging.warning("[WARN] Skipping compliance calculation - no valid hold period")

            # Show column headers for the first file
            if not first_headers_printed:
                print("Available columns:")
                for i, col in enumerate(df.columns):
                    print(f"  [{i}] {col}")
                first_headers_printed = True

            # Convert to numeric and clean - but preserve our calculated columns
            # Only convert original columns to numeric, keep our new columns as is
            calculated_cols = ["Creep Strain", "Creep Rate (1/s)", "Creep Rate (nm/s)", 
                             "Strain Rate (1/s)", "Shear Creep Compliance (1/GPa)", "Stress (GPa)"]
            original_cols = [col for col in df.columns if col not in calculated_cols]
            df[original_cols] = df[original_cols].apply(pd.to_numeric, errors='coerce')
            
            # Remove completely empty columns
            df.dropna(axis=1, how='all', inplace=True)

            # Apply drift correction
            if correct_drift:
                df = apply_drift_correction(df, correct_drift)

                # Recalculate creep strain after drift correction
                if calculate_creep_strain and file_hold_start is not None:
                    df = calculate_creep_strain(df, file_hold_start, file_hold_end)

            # Calculate elastic modulus
            if calculate_elastic:
                df = calculate_elastic_modulus(df)
                
            dataframes.append(df)

        except Exception as e:
            logging.warning(f"[WARN] Failed to read {file}: {e}")

    # Store average rates for later use
    load_and_clean_data.avg_creep_rates = avg_creep_rates
    load_and_clean_data.avg_strain_rates = avg_strain_rates
    
    # Save to cache
    if use_cache and dataframes:
        cache_data = {
            'dataframes': dataframes,
            'avg_creep_rates': avg_creep_rates,
            'avg_strain_rates': avg_strain_rates
        }
        save_to_cache(cache_path, cache_data)
        
        # Clean up old caches periodically
        clear_old_caches(keep_recent=20)
    
    return dataframes


def merge_dual_type_data(df_y, df_r, x_col, y_cols, r_cols):
    """
    Merge data from two different file types based on a common X column.
    
    Args:
        df_y: DataFrame with Y-axis columns
        df_r: DataFrame with R-axis columns  
        x_col: Common column to merge on (e.g., "Test Time (s)")
        y_cols: Columns to extract from df_y
        r_cols: Columns to extract from df_r
    
    Returns:
        Merged DataFrame with all columns
    """
    # Keep only necessary columns
    y_data = df_y[[x_col] + y_cols].copy()
    r_data = df_r[[x_col] + r_cols].copy()
    
    # Merge on common X column
    # Use outer join to keep all data points, then interpolate if needed
    merged = pd.merge(y_data, r_data, on=x_col, how='outer', suffixes=('', '_r'))
    
    # Sort by X column
    merged = merged.sort_values(x_col)
    
    # Interpolate missing values (in case file types have different sampling rates)
    for col in y_cols + r_cols:
        if merged[col].isna().any():
            merged[col] = merged[col].interpolate(method='linear', limit_direction='both')
    
    return merged


def average_data_no_bins(dataframes, x_col, y_cols, r_cols=None, plot_range=None):
    """
    Average data without binning by interpolating each run onto a reference x grid.

    Args:
        dataframes: DataFrame or list of DataFrames
        x_col: X-axis column name
        y_cols: Y-axis column names
        r_cols: Right Y-axis column names
        plot_range: Optional (t_start, t_end) to filter data
    """
    if r_cols is None:
        r_cols = []

    if isinstance(dataframes, pd.DataFrame):
        data_list = [dataframes]
    else:
        data_list = list(dataframes) if dataframes is not None else []

    if not data_list:
        return pd.DataFrame()

    filtered_list = []
    for df in data_list:
        if df is None or df.empty:
            continue
        df_local = df.copy()
        if plot_range:
            t_start, t_end = plot_range
            print(f"[INFO] Filtering data to time range {t_start}s to {t_end}s")
            df_local = df_local[(df_local[x_col] >= t_start) & (df_local[x_col] <= t_end)]
            if df_local.empty:
                print(f"[WARN] No data found in the specified time range {t_start}s to {t_end}s")
                continue
            print(f"[INFO] Data filtered to {len(df_local)} points")
        filtered_list.append(df_local)

    if not filtered_list:
        return pd.DataFrame()

    ref_df = next((df for df in filtered_list if df is not None and not df.empty), None)
    if ref_df is None:
        return pd.DataFrame()

    ref_df = ref_df[[x_col]].dropna().sort_values(by=x_col)
    ref_x = np.unique(ref_df[x_col].to_numpy())
    if ref_x.size == 0:
        print(f"[WARN] No valid {x_col} values for no-bin averaging")
        return pd.DataFrame()

    # Exclude truncated/outlier runs and clip to robust overlap range
    x_ranges = []  # (min, max, span, index)
    for idx, df in enumerate(filtered_list):
        if df is None or df.empty:
            continue
        vals = df[x_col].dropna().to_numpy()
        if vals.size > 0:
            x_ranges.append((vals.min(), vals.max(), vals.max() - vals.min(), idx))

    if not x_ranges:
        print(f"[WARN] No valid {x_col} data across runs")
        return pd.DataFrame()

    # Step 1: Exclude runs whose span is <50% of the median span (truncated files)
    spans = np.array([r[2] for r in x_ranges])
    median_span = np.median(spans)
    if median_span > 0:
        keep_mask = spans >= 0.5 * median_span
        excluded = [x_ranges[i] for i in range(len(x_ranges)) if not keep_mask[i]]
        if excluded:
            excluded_indices = {r[3] for r in excluded}
            for r in excluded:
                print(f"[WARN] Excluding truncated run (span {r[2]:.1f} vs median {median_span:.1f}, "
                      f"range [{r[0]:.3f}, {r[1]:.3f}])")
            filtered_list = [df for idx, df in enumerate(filtered_list) if idx not in excluded_indices]
            x_ranges = [r for r in x_ranges if r[3] not in excluded_indices]

    if not filtered_list or not x_ranges:
        print(f"[WARN] No runs remaining after excluding truncated data")
        return pd.DataFrame()

    # Pick a new ref_df from the remaining filtered_list (longest run)
    longest_idx = max(range(len(filtered_list)), key=lambda i: len(filtered_list[i]))
    ref_df = filtered_list[longest_idx][[x_col]].dropna().sort_values(by=x_col)
    ref_x = np.unique(ref_df[x_col].to_numpy())
    if ref_x.size == 0:
        print(f"[WARN] No valid {x_col} values for no-bin averaging")
        return pd.DataFrame()

    # Step 2: Use 10th/90th percentile of min/max instead of strict min/max
    # This prevents a single slightly-short run from clipping everything
    x_mins = np.array([r[0] for r in x_ranges])
    x_maxes = np.array([r[1] for r in x_ranges])
    if len(x_maxes) >= 5:
        common_max = np.percentile(x_maxes, 10)
        common_min = np.percentile(x_mins, 90)
    else:
        # With few runs, use strict overlap (outliers already excluded above)
        common_max = min(x_maxes)
        common_min = max(x_mins)

    n_before = len(ref_x)
    ref_x = ref_x[(ref_x >= common_min) & (ref_x <= common_max)]
    n_trimmed = n_before - len(ref_x)
    if n_trimmed > 0:
        print(f"[INFO] Trimmed {n_trimmed} edge points to overlap range "
              f"[{common_min:.3f}, {common_max:.3f}] ({len(x_ranges)} runs)")

    if ref_x.size == 0:
        print(f"[WARN] No overlapping {x_col} range across runs")
        return pd.DataFrame()

    print(f"[INFO] Using no-bin averaging with reference grid ({len(ref_x)} points, "
          f"{len(filtered_list)} runs)")
    result = {x_col: ref_x}

    for col in y_cols + r_cols:
        run_values = []
        for df in filtered_list:
            if df is None or df.empty or col not in df.columns:
                run_values.append(np.full(ref_x.shape, np.nan, dtype=float))
                continue

            df_local = df[[x_col, col]].dropna(subset=[x_col, col])
            if df_local.empty:
                run_values.append(np.full(ref_x.shape, np.nan, dtype=float))
                continue

            df_local = df_local.sort_values(by=x_col)
            df_local = df_local.groupby(x_col, as_index=False)[col].mean()
            x_vals = df_local[x_col].to_numpy()
            y_vals = df_local[col].to_numpy()

            if x_vals.size < 2:
                values = np.full(ref_x.shape, np.nan, dtype=float)
                if x_vals.size == 1:
                    values[ref_x == x_vals[0]] = y_vals[0]
                run_values.append(values)
                continue

            interp_vals = np.interp(ref_x, x_vals, y_vals, left=np.nan, right=np.nan)
            run_values.append(interp_vals)

        if not run_values:
            continue

        run_values = np.vstack(run_values)
        mean = np.nanmean(run_values, axis=0)
        std = np.nanstd(run_values, axis=0, ddof=0)
        n = np.sum(~np.isnan(run_values), axis=0)
        result[col + '_mean'] = mean
        result[col + '_std'] = std
        result[col + '_n'] = n

    return pd.DataFrame(result)


def average_data_with_bins(dataframes, x_col, y_cols, r_cols=None, bins=20, plot_range=None, adaptive=False, sem_by_run=False, log_x=False):
    """
    Average data with binning, optionally filtering by time range.

    Args:
        dataframes: DataFrame or list of DataFrames
        x_col: X-axis column name
        y_cols: Y-axis column names
        r_cols: Right Y-axis column names
        bins: Number of bins
        plot_range: Optional (t_start, t_end) to filter data
        adaptive: If True, use equal-point binning; if False, use equal-interval binning
        sem_by_run: If True, compute stats across per-run binned means
        log_x: If True, use log-spaced bins instead of linear
    """
    if r_cols is None:
        r_cols = []

    if isinstance(dataframes, pd.DataFrame):
        data_list = [dataframes]
    else:
        data_list = list(dataframes) if dataframes is not None else []

    if not data_list:
        return pd.DataFrame()

    filtered_list = []
    for df in data_list:
        if df is None or df.empty:
            continue
        df_local = df.copy()
        if plot_range:
            t_start, t_end = plot_range
            print(f"[INFO] Filtering data to time range {t_start}s to {t_end}s")
            df_local = df_local[(df_local[x_col] >= t_start) & (df_local[x_col] <= t_end)]
            if df_local.empty:
                print(f"[WARN] No data found in the specified time range {t_start}s to {t_end}s")
                continue
            print(f"[INFO] Data filtered to {len(df_local)} points")
        filtered_list.append(df_local)

    if not filtered_list:
        return pd.DataFrame()

    if sem_by_run:
        pooled = pd.concat(filtered_list, ignore_index=True)
        pooled = pooled.sort_values(by=x_col)

        if adaptive:
            all_x = pooled[x_col].dropna().values
            if len(all_x) == 0:
                print(f"[ERROR] No valid {x_col} values for adaptive binning")
                return pd.DataFrame()
            quantiles = np.linspace(0, 1, bins + 1)
            bin_edges = np.quantile(all_x, quantiles)
            bin_edges = np.unique(bin_edges)
            if len(bin_edges) < 2:
                print(f"[ERROR] Adaptive binning failed due to duplicate edges")
                return pd.DataFrame()
            if len(bin_edges) - 1 != bins:
                print(f"[WARN] Adaptive binning collapsed to {len(bin_edges) - 1} bins")
        else:
            x_min_val = pooled[x_col].min()
            x_max_val = pooled[x_col].max()
            if log_x and x_max_val > 0:
                # Log-spaced bins: use small positive floor for x_min
                x_min_log = max(x_min_val, x_max_val / 1e4, 1e-3)
                bin_edges = np.geomspace(x_min_log, x_max_val, bins + 1)
                print(f"[INFO] Using log-spaced bins: {bins} bins from {x_min_log:.3f} to {x_max_val:.1f}")
            else:
                bin_edges = np.linspace(x_min_val, x_max_val, bins + 1)

        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        result = {x_col: bin_centers}

        for col in y_cols + r_cols:
            if col not in pooled.columns:
                print(f"[WARN] Column '{col}' not found - skipping")
                continue

            run_means = []
            for df in filtered_list:
                if col not in df.columns:
                    run_means.append(np.full(len(bin_centers), np.nan))
                    continue
                binned = pd.cut(df[x_col], bins=bin_edges, include_lowest=True)
                grouped = df.groupby(binned, observed=False)[col]
                means = grouped.mean().values
                run_means.append(means)

            if not run_means:
                continue

            run_means = np.vstack(run_means)
            mean = np.nanmean(run_means, axis=0)
            std = np.nanstd(run_means, axis=0, ddof=0)
            n = np.sum(~np.isnan(run_means), axis=0)
            result[col + '_mean'] = mean
            result[col + '_std'] = std
            result[col + '_n'] = n

        return pd.DataFrame(result)

    all_data = pd.concat(filtered_list, ignore_index=True)

    # Sort data
    all_data = all_data.sort_values(by=x_col)

    if adaptive:
        # Adaptive binning: equal number of points per bin
        print(f"[INFO] Using adaptive binning ({bins} bins with ~{len(all_data)//bins} points each)")

        # Calculate points per bin
        points_per_bin = len(all_data) // bins
        if points_per_bin < 1:
            points_per_bin = 1
            bins = len(all_data)

        # Create bin assignments based on row position
        all_data['bin_index'] = np.arange(len(all_data)) // points_per_bin
        # Ensure we don't exceed the desired number of bins
        all_data['bin_index'] = np.minimum(all_data['bin_index'], bins - 1)

        # Calculate bin centers as mean x value in each bin
        bin_centers = all_data.groupby('bin_index')[x_col].mean().values
        result = {x_col: bin_centers}

        # Process each column
        for col in y_cols + r_cols:
            if col not in all_data.columns:
                print(f"[WARN] Column '{col}' not found - skipping")
                continue

            # Skip empty columns
            valid_data = all_data[col].dropna()
            if valid_data.empty:
                print(f"[WARN] No valid data in column '{col}' - skipping")
                continue

            # Calculate stats for each bin
            try:
                grouped = all_data.groupby('bin_index')[col]
                mean = grouped.mean().values
                std = grouped.std(ddof=0).values
                count = grouped.count().values
                result[col + '_mean'] = mean
                result[col + '_std'] = std
                result[col + '_n'] = count
                print(f"[INFO] Successfully binned column: {col} (adaptive)")
            except Exception as e:
                print(f"[WARN] Error binning column '{col}': {e}")

        # Clean up temporary column
        all_data.drop('bin_index', axis=1, inplace=True)

    else:
        # Original equal-interval binning
        x_min_val = all_data[x_col].min()
        x_max_val = all_data[x_col].max()
        if log_x and x_max_val > 0:
            x_min_log = max(x_min_val, x_max_val / 1e4, 1e-3)
            bin_edges = np.geomspace(x_min_log, x_max_val, bins + 1)
            print(f"[INFO] Using log-spaced binning ({bins} bins from {x_min_log:.3f} to {x_max_val:.1f})")
        else:
            print(f"[INFO] Using equal-interval binning ({bins} bins)")
            bin_edges = np.linspace(x_min_val, x_max_val, bins + 1)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        result = {x_col: bin_centers}

        # Process each column
        for col in y_cols + r_cols:
            if col not in all_data.columns:
                print(f"[WARN] Column '{col}' not found - skipping")
                continue

            # Skip empty columns
            valid_data = all_data[col].dropna()
            if valid_data.empty:
                print(f"[WARN] No valid data in column '{col}' - skipping")
                continue

            # Calculate stats for each bin
            try:
                binned = pd.cut(all_data[x_col], bins=bin_edges, include_lowest=True)
                grouped = all_data.groupby(binned, observed=False)[col]
                mean = grouped.mean().values
                std = grouped.std(ddof=0).values
                count = grouped.count().values
                result[col + '_mean'] = mean
                result[col + '_std'] = std
                result[col + '_n'] = count
                print(f"[INFO] Successfully binned column: {col}")
            except Exception as e:
                print(f"[WARN] Error binning column '{col}': {e}")

    return pd.DataFrame(result)
