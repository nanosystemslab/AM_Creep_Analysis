"""
Plotting functions for nanoindentation creep data visualization.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import textwrap
from datetime import datetime
from pathlib import Path
from analysis_functions import fit_linear_sections, compute_strain_recovery
from data_utils import load_and_clean_data
from plot_style import apply_journal_style, get_folder_style, get_style_by_index
from _figure_style import save_figure


def _calc_markevery(x, n_markers, log_scale=False):
    """Return a list of indices that space n_markers evenly across x.

    For linear axes the indices are evenly spaced by count.
    For log axes the indices are evenly spaced in log(x), so markers
    appear visually equidistant on a logarithmic axis.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n <= n_markers:
        return list(range(n))

    if log_scale:
        pos = x.copy()
        pos[pos <= 0] = np.nan
        if np.all(np.isnan(pos)):
            # Fallback to linear spacing
            return list(np.round(np.linspace(0, n - 1, n_markers)).astype(int))
        log_x = np.log10(pos)
        log_min = np.nanmin(log_x)
        log_max = np.nanmax(log_x)
        targets = np.linspace(log_min, log_max, n_markers)
        indices = []
        for t in targets:
            diffs = np.abs(log_x - t)
            idx = int(np.nanargmin(diffs))
            if idx not in indices:
                indices.append(idx)
        return sorted(indices)
    else:
        return list(np.round(np.linspace(0, n - 1, n_markers)).astype(int))


def _escape_latex(text):
    """Escape LaTeX special characters in a string."""
    replacements = [
        ('&', r'\&'),
        ('%', r'\%'),
        ('$', r'\$'),
        ('#', r'\#'),
        ('_', r'\_'),
        ('{', r'\{'),
        ('}', r'\}'),
        ('~', r'\textasciitilde{}'),
        ('^', r'\textasciicircum{}'),
    ]
    for char, escaped in replacements:
        text = text.replace(char, escaped)
    return text


def _wrap_title(text, fig, ax, fontsize=None):
    """Wrap title text so it never exceeds the axes width.

    Returns a string where each wrapped line is its own \\textbf{} block
    joined by newlines, so LaTeX brace matching stays valid.
    """
    if fontsize is None:
        fontsize = plt.rcParams.get('axes.titlesize', 12)
        if isinstance(fontsize, str):
            fontsize = 12  # fallback for named sizes like 'large'
    # Approximate: 1 character ≈ 0.6 × fontsize in points, 72 pt/inch
    char_width_in = 0.6 * fontsize / 72.0
    bbox = ax.get_position()
    axes_width_in = fig.get_size_inches()[0] * bbox.width
    max_chars = max(20, int(axes_width_in / char_width_in))
    lines = textwrap.wrap(text, width=max_chars)
    # Each line gets its own \textbf so LaTeX braces stay balanced
    return "\n".join(r"\textbf{" + line + "}" for line in lines)


def _set_plain_log_ticks(ax, which='y'):
    """Format log-scale ticks as plain numbers (e.g. '3') instead of
    scientific notation (e.g. '3×10⁰').

    Uses ScalarFormatter so tick *positions* are chosen for the actual
    data range (not just powers of 10), and labels are plain decimals.
    """
    fmt = mticker.ScalarFormatter()
    fmt.set_scientific(False)
    fmt.set_useOffset(False)
    if which in ('y', 'both'):
        ax.yaxis.set_major_locator(mticker.AutoLocator())
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    if which in ('x', 'both'):
        ax.xaxis.set_major_locator(mticker.AutoLocator())
        ax.xaxis.set_major_formatter(fmt)
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())


def _save_fit_stats_csv(fit_curves, args):
    """Save fit statistics to a CSV file alongside the plot output."""
    import csv
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "fit_stats.csv"

    rows = []
    for fc in fit_curves:
        model_name = fc.get('model_name', '')
        r2 = fc.get('R2', 0)
        rmse = fc.get('RMSE', 0)
        label = fc.get('label', '')
        params = fc.get('params', {})
        # One row per parameter
        for pname, pinfo in params.items():
            rows.append({
                'folder': label.split(' fit')[0] if ' fit' in label else label,
                'model': model_name,
                'R2': f"{r2:.6f}",
                'RMSE': f"{rmse:.4e}",
                'parameter': pname,
                'value': f"{pinfo['value']:.6e}",
                'stderr': f"{pinfo['stderr']:.2e}" if pinfo.get('stderr') else '',
                'units': pinfo.get('units', ''),
            })

    if rows:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['folder', 'model', 'R2', 'RMSE',
                                                    'parameter', 'value', 'stderr', 'units'])
            writer.writeheader()
            writer.writerows(rows)
        print(f"[INFO] Fit statistics saved to: {csv_path}")


def get_column_style(column_name):
    """
    Return consistent color, marker, and linestyle for a given column name.
    This ensures the same column always looks the same across different plots.
    """
    # Define consistent styling for common columns
    column_styles = {
        'Indent Disp. (nm)': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-'},      # Blue, circle, solid
        'Hardness (GPa)': {'color': '#ff7f0e', 'marker': 's', 'linestyle': '-'},         # Orange, square, solid
        'Storage Mod. (GPa)': {'color': '#2ca02c', 'marker': 'D', 'linestyle': '-'},     # Green, diamond, solid
        'Loss Mod. (GPa)': {'color': '#d62728', 'marker': '^', 'linestyle': '--'},       # Red, triangle, dashed
        'Complex Mod. (GPa)': {'color': '#9467bd', 'marker': 'v', 'linestyle': '-.'},    # Purple, triangle down, dashdot
        'Indent Load (µN)': {'color': '#8c564b', 'marker': 'x', 'linestyle': '-'},  # Brown, x, solid
        'Contact Area (nm^2)': {'color': '#e377c2', 'marker': '+', 'linestyle': '--'},   # Pink, plus, dashed
        'Contact Depth (nm)': {'color': '#7f7f7f', 'marker': '*', 'linestyle': '-.'},    # Gray, star, dashdot
        'Tan-Delta': {'color': '#bcbd22', 'marker': 'p', 'linestyle': ':'},              # Olive, pentagon, dotted
        'Creep Strain': {'color': '#17becf', 'marker': 'h', 'linestyle': '-'},           # Cyan, hexagon, solid
        'Stress (GPa)': {'color': '#d62728', 'marker': 'D', 'linestyle': '--'},          # Red, diamond, dashed
        'Shear Creep Compliance (1/GPa)': {'color': '#9467bd', 'marker': 's', 'linestyle': ':'}, # Purple, square, dotted
    }
    
    # Return style if column is in the map, otherwise use defaults
    if column_name in column_styles:
        return column_styles[column_name]
    else:
        # Default styling for unmapped columns
        return {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-'}


def _std_to_sem(std_values, n_values):
    if n_values is None:
        return np.asarray(std_values, dtype=float)
    std = np.asarray(std_values, dtype=float)
    n = np.asarray(n_values, dtype=float)
    sem = std / np.sqrt(n)
    sem[n < 2] = np.nan
    return sem


def _add_cli_watermark(fig, args, wrap_width=120):
    if getattr(args, "no_watermark", False):
        return False
    cli_command = getattr(args, "cli_command", None)
    if not cli_command:
        return False
    text = textwrap.fill(f"CLI: {cli_command}", width=wrap_width)
    fig.text(0.01, 0.01, text, ha="left", va="bottom", fontsize=7, color="0.5", alpha=0.7)
    return True


def draw_callout(ax, x_pos, y_pos, lines, label=None, color='red', position='inside'):
    """Add a callout box with text and arrow."""
    box_text = (f"{label}\n" if label else "") + "\n".join(lines)
    
    # Get axis limits for positioning
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    
    # Calculate position based on axis scales
    x_scale = ax.get_xscale()
    y_scale = ax.get_yscale()
    
    # Find a good position in the whitespace
    if position == 'inside':
        # Use the top right quadrant by default (often has more whitespace)
        quad_pos = (0.75, 0.75)
        
        # Convert relative positions to actual values
        if x_scale == 'log':
            x_ratio = x_max / x_min
            x_text = x_min * (x_ratio ** quad_pos[0])
        else:
            x_text = x_min + (x_max - x_min) * quad_pos[0]
            
        if y_scale == 'log':
            y_ratio = y_max / y_min
            y_text = y_min * (y_ratio ** quad_pos[1])
        else:
            y_text = y_min + (y_max - y_min) * quad_pos[1]
    else:
        # Position relative to target but with better offset
        x_offset = (x_max - x_min) * 0.2
        y_offset = (y_max - y_min) * 0.2
        
        if x_scale == 'log':
            x_text = x_pos * (x_max / x_min) ** 0.2
        else:
            x_text = x_pos + x_offset
            
        if y_scale == 'log':
            y_text = y_pos * (y_max / y_min) ** 0.2
        else:
            y_text = y_pos + y_offset
    
    # Create annotation
    ax.annotate(
        box_text,
        xy=(x_pos, y_pos),
        xytext=(x_text, y_text),
        textcoords='data',
        fontsize=10,
        color=color,
        arrowprops=dict(
            arrowstyle="->",
            color=color,
            lw=1.0,
            connectionstyle="arc3,rad=0.3",
            relpos=(1.0, 0.0),
            shrinkA=0,
            shrinkB=1,
        ),
        bbox=dict(
            boxstyle="round,pad=0.3",
            fc="white",
            ec=color,
            lw=0.8,
            alpha=0.9
        ),
        ha='center',
        va='center',
        zorder=30
    )


def plot_data_with_stdev(datasets, x_col, y_cols, r_cols, args, folder_labels,
                        y_individual_data=None, r_individual_data=None,
                        y_individual_labels=None, r_individual_labels=None,
                        folder_keys=None):
    """
    Plot data with SEM bands and optional fits.

    Args:
        datasets: List of averaged DataFrames (with _mean/_std/_n columns)
        x_col: X-axis column name
        y_cols: Left Y-axis column names
        r_cols: Right Y-axis column names
        args: Command line arguments
        folder_labels: Display labels for legend
        y_individual_data: Optional list of raw DataFrames for Y-axis (plots as individual lines)
        r_individual_data: Optional list of raw DataFrames for R-axis (plots as individual lines)
        y_individual_labels: Optional labels for Y-axis individual data
        r_individual_labels: Optional labels for R-axis individual data
        folder_keys: Raw folder names for style lookup (defaults to folder_labels)
    """
    apply_journal_style()

    # Use raw folder names for style lookup, display labels for legend
    if folder_keys is None:
        folder_keys = folder_labels

    # Create figure — wide enough so the plot area spans the 4-col legend
    fig, ax1 = plt.subplots(figsize=(18, 10))
    ax2 = ax1.twinx() if r_cols else None

    # Build the title
    title_parts = []
    if len(datasets) > 1:
        title_parts.append(f"{', '.join(y_cols + r_cols)} vs. {x_col} - Multiple Folders")
    else:
        title_parts.append(f"{', '.join(y_cols + r_cols)} vs. {x_col}")

    # Add time range if specified
    if args.plot_range:
        t_start, t_end = args.plot_range
        title_parts.append(f"t={t_start}s to {t_end}s")

    plot_title = " - ".join(title_parts)
    custom_title = getattr(args, 'title', None)
    title_pad = getattr(args, 'title_pad', None) or 25
    if custom_title:
        ax1.set_title(_wrap_title(_escape_latex(custom_title), fig, ax1), pad=title_pad)
    else:
        ax1.set_title(_wrap_title(_escape_latex(plot_title), fig, ax1), pad=title_pad)

    # Set axis scales
    if args.log_x:
        ax1.set_xscale('log')
    if args.log_y:
        ax1.set_yscale('log')
    if args.log_r and ax2:
        ax2.set_yscale('log')

    use_sem = not getattr(args, "nosem", False)
    
    # Plot individual Y-axis data if provided
    indiv_total_cols = len(y_cols) + len(r_cols)
    if y_individual_data is not None and y_individual_labels is not None:
        print(f"[INFO] Plotting {len(y_individual_data)} individual traces on Y-axis")
        for idx, (df, label) in enumerate(zip(y_individual_data, y_individual_labels)):
            for col_idx, y_col in enumerate(y_cols):
                if y_col not in df.columns:
                    continue

                x = df[x_col]
                y = df[y_col]

                # Unique style per (trace, column) pair
                style_index = idx * indiv_total_cols + col_idx
                color, marker = get_style_by_index(style_index)
                linestyle = '-'

                # Create label
                if len(y_cols) > 1:
                    plot_label = f"{label} - {y_col}"
                else:
                    plot_label = label

                # Plot individual trace
                me = _calc_markevery(x, 10, log_scale=args.log_x)
                ax1.plot(x, y, label=plot_label, color=color,
                        marker=marker, linestyle=linestyle,
                        markevery=me, markersize=12, linewidth=1, alpha=0.7)

    # Plot individual R-axis data if provided
    if r_individual_data is not None and r_individual_labels is not None and ax2:
        print(f"[INFO] Plotting {len(r_individual_data)} individual traces on R-axis")
        for idx, (df, label) in enumerate(zip(r_individual_data, r_individual_labels)):
            for col_idx, r_col in enumerate(r_cols):
                if r_col not in df.columns:
                    continue

                x = df[x_col]
                y = df[r_col]

                # Unique style per (trace, column) pair
                style_index = idx * indiv_total_cols + len(y_cols) + col_idx
                color, marker = get_style_by_index(style_index)
                linestyle = '-'

                # Create label
                if len(r_cols) > 1:
                    plot_label = f"{label} - {r_col}"
                else:
                    plot_label = label

                # Plot individual trace
                me = _calc_markevery(x, 10, log_scale=args.log_x)
                ax2.plot(x, y, label=plot_label, color=color,
                        marker=marker, linestyle=linestyle,
                        markevery=me, markersize=12, linewidth=1, alpha=0.7)
    
    # Plot each dataset (folder) separately - only if individual data not provided
    total_cols = len(y_cols) + len(r_cols)

    for folder_index, (df, folder_label, folder_key) in enumerate(
            zip(datasets, folder_labels, folder_keys)):

        # Plot left axis columns (only if not in individual mode for Y-axis)
        if y_individual_data is None:
            for col_index, y_col in enumerate(y_cols):
                if y_col + '_mean' not in df.columns:
                    print(f"[WARN] Skipping {y_col} for {folder_label} - {y_col + '_mean'} not found")
                    continue

                x = df[x_col]
                y_mean = df[y_col + '_mean']
                y_std = df[y_col + '_std']
                y_n = df[y_col + '_n'] if (y_col + '_n') in df.columns else None
                y_sem = _std_to_sem(y_std, y_n)
                y_err = y_sem if use_sem else np.asarray(y_std, dtype=float)

                # Unique style per (folder, column) pair
                style_index = folder_index * total_cols + col_index
                color, marker = get_style_by_index(style_index)
                linestyle = '-'

                # Create label
                if len(y_cols + r_cols) > 1:
                    label = f"{folder_label} - {y_col}"
                else:
                    label = folder_label

                # Plot based on mode
                if hasattr(args, 'fs') and args.fs or hasattr(args, 'fs_lines') and args.fs_lines:
                    # Check if lines should be drawn
                    if hasattr(args, 'fs_lines') and args.fs_lines:
                        ax1.plot(x, y_mean, marker=marker, label=label,
                                color=color, markersize=8, linewidth=1.5, linestyle='-')
                    else:
                        ax1.plot(x, y_mean, marker=marker, label=label,
                                color=color, markersize=8, linewidth=0, linestyle='none')
                    # Scale error bars if specified
                    error_scale = args.error_scale if hasattr(args, 'error_scale') else 1.0
                    ax1.errorbar(x, y_mean, yerr=y_err * error_scale, fmt='none', color=color,
                                alpha=0.3, capsize=3, capthick=1)
                else:
                    # Normal mode: lines with markers
                    error_scale = args.error_scale if hasattr(args, 'error_scale') else 1.0
                    me = _calc_markevery(x, 5, log_scale=args.log_x)
                    ax1.plot(x, y_mean, label=label, color=color,
                            marker=marker, linestyle=linestyle,
                            markevery=me, markersize=12, linewidth=2)
                    ax1.fill_between(x, y_mean - y_err * error_scale, y_mean + y_err * error_scale,
                                    alpha=0.15, color=color, rasterized=True)

        # Plot right axis columns (only if not in individual mode for R-axis)
        if r_cols and ax2 and r_individual_data is None:
            for col_index, r_col in enumerate(r_cols):
                if r_col + '_mean' not in df.columns:
                    print(f"[WARN] Skipping {r_col} for {folder_label} - {r_col + '_mean'} not found")
                    continue

                x = df[x_col]
                y_mean = df[r_col + '_mean']
                y_std = df[r_col + '_std']
                y_n = df[r_col + '_n'] if (r_col + '_n') in df.columns else None
                y_sem = _std_to_sem(y_std, y_n)
                y_err = y_sem if use_sem else np.asarray(y_std, dtype=float)

                # Unique style per (folder, column) pair
                style_index = folder_index * total_cols + len(y_cols) + col_index
                color, marker = get_style_by_index(style_index)
                linestyle = '-'

                # Create label
                if len(y_cols + r_cols) > 1:
                    label = f"{folder_label} - {r_col}"
                else:
                    label = folder_label

                # Plot based on mode
                if hasattr(args, 'fs') and args.fs or hasattr(args, 'fs_lines') and args.fs_lines:
                    # Check if lines should be drawn
                    if hasattr(args, 'fs_lines') and args.fs_lines:
                        ax2.plot(x, y_mean, marker=marker, label=label,
                                color=color, markersize=12, linewidth=1.5, linestyle='-')
                    else:
                        ax2.plot(x, y_mean, marker=marker, label=label,
                                color=color, markersize=12, linewidth=0, linestyle='none')
                    # Scale error bars if specified
                    error_scale = args.error_scale if hasattr(args, 'error_scale') else 1.0
                    ax2.errorbar(x, y_mean, yerr=y_err * error_scale, fmt='none', color=color,
                                alpha=0.3, capsize=3, capthick=1)
                else:
                    # Normal mode: lines with markers
                    error_scale = args.error_scale if hasattr(args, 'error_scale') else 1.0
                    me = _calc_markevery(x, 5, log_scale=args.log_x)
                    ax2.plot(x, y_mean, label=label, color=color,
                            marker=marker, linestyle=linestyle,
                            markevery=me, markersize=12, linewidth=2)
                    ax2.fill_between(x, y_mean - y_err * error_scale, y_mean + y_err * error_scale,
                                    alpha=0.15, color=color, rasterized=True)

    # Plot model fit curves if available
    fit_curves = getattr(args, '_fit_curves', None)
    if fit_curves:
        for fc in fit_curves:
            ax1.plot(fc['t'], fc['J'], '--', color=fc['color'], lw=2.5,
                     label=fc['label'], alpha=0.9, zorder=10)

        # Save fit statistics as CSV if --fit_stats is set
        if getattr(args, 'fit_stats', False):
            _save_fit_stats_csv(fit_curves, args)

        # Add model name as subtitle if all fits use the same model
        custom_title = getattr(args, 'title', None)
        if custom_title:
            title_pad = getattr(args, 'title_pad', None) or 25
            ax1.set_title(_wrap_title(_escape_latex(custom_title), fig, ax1), pad=title_pad)
        else:
            model_names = set(fc['model_name'] for fc in fit_curves)
            if len(model_names) == 1:
                model_name = model_names.pop()
                title_pad = getattr(args, 'title_pad', None) or 25
                ax1.set_title(_wrap_title("Model: " + _escape_latex(model_name), fig, ax1), pad=title_pad)

    # Add axis labels and legend
    x_label = x_col
    y_label = ", ".join(y_cols) if y_cols else ""
    r_label = ", ".join(r_cols) if r_cols else ""

    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    if ax2:
        ax2.set_ylabel(r_label)

    # Combine legends — interleave fit lines next to their data lines
    lines, labels = ax1.get_legend_handles_labels()
    if ax2:
        rlines, rlabels = ax2.get_legend_handles_labels()
        lines += rlines
        labels += rlabels

    # Reorder: interleave each data line with its fit line
    if fit_curves:
        data_lines = [(h, l) for h, l in zip(lines, labels) if '$R^2$' not in l]
        fit_lines = [(h, l) for h, l in zip(lines, labels) if '$R^2$' in l]
        ordered = []
        for i, (dh, dl) in enumerate(data_lines):
            ordered.append((dh, dl))
            if i < len(fit_lines):
                ordered.append(fit_lines[i])
        lines = [h for h, _ in ordered]
        labels = [l for _, l in ordered]

    # Legend below plot
    ncol = 3
    legend_y = getattr(args, 'legend_y', None) or -0.15
    ax1.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, legend_y),
               borderaxespad=0, ncol=ncol)

    # Fix log scale axis limits to prevent tiny data display
    if args.log_y or args.log_x:
        # Collect all data values to compute proper limits
        all_x_vals = []
        all_y_vals = []

        for df in datasets:
            for y_col in y_cols:
                if y_col + '_mean' in df.columns:
                    x_data = df[x_col].values
                    y_data = df[y_col + '_mean'].values
                    # Only include positive finite values
                    valid_mask = np.isfinite(x_data) & np.isfinite(y_data) & (x_data > 0) & (y_data > 0)
                    all_x_vals.extend(x_data[valid_mask])
                    all_y_vals.extend(y_data[valid_mask])

        if args.log_x and len(all_x_vals) > 0:
            x_min, x_max = np.min(all_x_vals), np.max(all_x_vals)
            # Floor x_min to 1s (10^0) for cleaner log plots
            x_min = max(x_min, 1.0)
            # Add 10% padding in log space
            x_range = np.log10(x_max) - np.log10(x_min)
            ax1.set_xlim(10**(np.log10(x_min) - 0.1*x_range), 10**(np.log10(x_max) + 0.1*x_range))

        if args.log_y and len(all_y_vals) > 0:
            y_min, y_max = np.min(all_y_vals), np.max(all_y_vals)
            # Add 10% padding in log space
            y_range = np.log10(y_max) - np.log10(y_min)
            ax1.set_ylim(10**(np.log10(y_min) - 0.1*y_range), 10**(np.log10(y_max) + 0.1*y_range))

    if args.log_r and ax2 and r_cols:
        all_r_vals = []
        for df in datasets:
            for r_col in r_cols:
                if r_col + '_mean' in df.columns:
                    r_data = df[r_col + '_mean'].values
                    valid_mask = np.isfinite(r_data) & (r_data > 0)
                    all_r_vals.extend(r_data[valid_mask])

        if len(all_r_vals) > 0:
            r_min, r_max = np.min(all_r_vals), np.max(all_r_vals)
            r_range = np.log10(r_max) - np.log10(r_min)
            ax2.set_ylim(10**(np.log10(r_min) - 0.1*r_range), 10**(np.log10(r_max) + 0.1*r_range))

    # Reference lines
    _needs_legend_refresh = False
    if getattr(args, 'datasheet', None) is not None:
        ax1.axhline(args.datasheet, color='red', ls='--', lw=1.5, zorder=0,
                     label=f'Datasheet ({args.datasheet})')
        _needs_legend_refresh = True
    if getattr(args, 'shallow', None) is not None:
        ax1.axvline(args.shallow, color='blue', ls='--', lw=1.5, zorder=0,
                     label=f'Shallow cutoff ({args.shallow:.0f} nm)')
        _needs_legend_refresh = True
    if _needs_legend_refresh:
        lines, labels = ax1.get_legend_handles_labels()
        if ax2:
            rlines, rlabels = ax2.get_legend_handles_labels()
            lines += rlines
            labels += rlabels
        legend_y = getattr(args, 'legend_y', None) or -0.15
        ax1.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, legend_y),
                   borderaxespad=0, ncol=3)

    # Save figure
    custom_filename = getattr(args, 'filename', None)
    if custom_filename:
        filename = custom_filename
    else:
        name = "_".join([c.strip().replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")
                        for c in y_cols + r_cols])
        x_clean = x_col.strip().replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")

        # Add folder count to filename
        if args.plot_range:
            t_start, t_end = args.plot_range
            filename = f"{args.type}_{x_clean}_vs_{name}_{len(datasets)}folders_{t_start}s-{t_end}s"
        else:
            filename = f"{args.type}_{x_clean}_vs_{name}_{len(datasets)}folders"

    output_path = Path(args.output) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Apply layout and save
    has_watermark = _add_cli_watermark(fig, args)
    fig.subplots_adjust(bottom=0.30)
    formats = ["pdf", "png"] if getattr(args, "png", False) else ["pdf"]
    save_figure(fig, str(output_path), formats=formats, dpi=600, close=not args.plot_show)
    print(f"[INFO] Plot saved to: {output_path}.pdf")

    if args.plot_show:
        plt.show()
        plt.close()


def plot_individual_files(datasets, x_col, y_cols, r_cols, args, file_labels):
    """Plot individual test files without averaging - each file gets its own line."""
    apply_journal_style()

    # Create figure — wide enough so the plot area spans the 4-col legend
    fig, ax1 = plt.subplots(figsize=(18, 10))
    ax2 = ax1.twinx() if r_cols else None

    # Build the title
    title_parts = []
    title_parts.append(f"{', '.join(y_cols + r_cols)} vs. {x_col} - Individual Tests")

    if args.plot_range:
        t_start, t_end = args.plot_range
        title_parts.append(f"t={t_start}s to {t_end}s")

    plot_title = " - ".join(title_parts)
    custom_title = getattr(args, 'title', None)
    title_pad = getattr(args, 'title_pad', None) or 25
    if custom_title:
        ax1.set_title(_wrap_title(_escape_latex(custom_title), fig, ax1), pad=title_pad)
    else:
        ax1.set_title(_wrap_title(_escape_latex(plot_title), fig, ax1), pad=title_pad)

    # Set axis scales
    if args.log_x:
        ax1.set_xscale('log')
    if args.log_y:
        ax1.set_yscale('log')
    if args.log_r and ax2:
        ax2.set_yscale('log')

    # Linestyle cycle for distinguishing files within same folder
    linestyle_cycle = ['-', '--', '-.', ':']

    # Plot each dataset (individual file)
    indiv_total_cols = len(y_cols) + len(r_cols)
    for idx, (df, file_label) in enumerate(zip(datasets, file_labels)):

        # Plot left axis columns
        for col_idx, y_col in enumerate(y_cols):
            if y_col not in df.columns:
                print(f"[WARN] Column '{y_col}' not found in {file_label}")
                continue

            style_index = idx * indiv_total_cols + col_idx
            color, marker = get_style_by_index(style_index)

            x = df[x_col]
            y = df[y_col]

            # Remove NaN values
            valid_mask = ~np.isnan(x) & ~np.isnan(y)
            if not valid_mask.any():
                print(f"[WARN] No valid data for {y_col} in {file_label}")
                continue

            x_clean = x[valid_mask]
            y_clean = y[valid_mask]

            if len(y_cols) > 1:
                label = f"{file_label} - {y_col}"
            else:
                label = file_label

            me = _calc_markevery(x_clean, 20, log_scale=args.log_x)
            ax1.plot(x_clean, y_clean, linestyle='-', label=label,
                   color=color, marker=marker, markevery=me,
                   markersize=12, linewidth=1.5, alpha=0.8)

        # Plot right axis columns
        if r_cols and ax2:
            for col_idx, r_col in enumerate(r_cols):
                if r_col not in df.columns:
                    print(f"[WARN] Column '{r_col}' not found in {file_label}")
                    continue

                style_index = idx * indiv_total_cols + len(y_cols) + col_idx
                color, marker = get_style_by_index(style_index)

                x = df[x_col]
                y = df[r_col]

                valid_mask = ~np.isnan(x) & ~np.isnan(y)
                if not valid_mask.any():
                    continue

                x_clean = x[valid_mask]
                y_clean = y[valid_mask]
                me = _calc_markevery(x_clean, 20, log_scale=args.log_x)

                if len(r_cols) > 1 or len(y_cols) > 0:
                    label = f"{file_label} - {r_col}"
                else:
                    label = file_label

                ax2.plot(x_clean, y_clean, linestyle='-', label=label,
                       color=color, marker=marker, markevery=me,
                       markersize=12, linewidth=1.5, alpha=0.8)

    # Add axis labels
    x_label = x_col
    y_label = ", ".join(y_cols) if y_cols else ""
    r_label = ", ".join(r_cols) if r_cols else ""

    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    if ax2:
        ax2.set_ylabel(r_label)

    # Handle legend
    lines, labels = ax1.get_legend_handles_labels()
    if ax2:
        rlines, rlabels = ax2.get_legend_handles_labels()
        lines += rlines
        labels += rlabels

    # Legend below plot
    ncol = 3
    legend_y = getattr(args, 'legend_y', None) or -0.15
    ax1.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, legend_y),
               borderaxespad=0, ncol=ncol)

    # Reference lines
    _needs_legend_refresh = False
    if getattr(args, 'datasheet', None) is not None:
        ax1.axhline(args.datasheet, color='red', ls='--', lw=1.5, zorder=0,
                     label=f'Datasheet ({args.datasheet})')
        _needs_legend_refresh = True
    if getattr(args, 'shallow', None) is not None:
        ax1.axvline(args.shallow, color='blue', ls='--', lw=1.5, zorder=0,
                     label=f'Shallow cutoff ({args.shallow:.0f} nm)')
        _needs_legend_refresh = True
    if _needs_legend_refresh:
        lines, labels = ax1.get_legend_handles_labels()
        if ax2:
            rlines, rlabels = ax2.get_legend_handles_labels()
            lines += rlines
            labels += rlabels
        legend_y = getattr(args, 'legend_y', None) or -0.15
        ax1.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, legend_y),
                   borderaxespad=0, ncol=3)

    # Save figure
    custom_filename = getattr(args, 'filename', None)
    if custom_filename:
        filename = custom_filename
    else:
        name = "_".join([c.strip().replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")
                        for c in y_cols + r_cols])
        x_clean = x_col.strip().replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")

        if args.plot_range:
            t_start, t_end = args.plot_range
            filename = f"{args.type}_{x_clean}_vs_{name}_individual_{len(datasets)}files_{t_start}s-{t_end}s"
        else:
            filename = f"{args.type}_{x_clean}_vs_{name}_individual_{len(datasets)}files"

    output_path = Path(args.output) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    has_watermark = _add_cli_watermark(fig, args)
    if has_watermark:
        fig.tight_layout(rect=[0, 0.06, 1, 1])
    else:
        fig.tight_layout()
    formats = ["pdf", "png"] if getattr(args, "png", False) else ["pdf"]
    save_figure(fig, str(output_path), formats=formats, dpi=600, close=not args.plot_show)
    print(f"[INFO] Individual plot saved to: {output_path}.pdf")

    if args.plot_show:
        plt.show()
        plt.close()
