import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import logging
import itertools

__version__ = "0.0.1"

def setup_logging(verbosity):
    log_fmt = ("%(levelname)s - %(module)s - "
               "%(funcName)s @%(lineno)d: %(message)s")
    logging.basicConfig(format=log_fmt, level=logging.getLevelName(verbosity))

def parse_command_line():
    parser = argparse.ArgumentParser(description="Analyse sensor data")
    parser.add_argument("-V", "--version", action="version", version=__version__)
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        dest="verbosity", help="verbose output")
    parser.add_argument("-d", "--dir", dest="dirs", nargs='+', required=True,
                        help="directories with data files")
    parser.add_argument("-i", "--in", dest="input", nargs='+', required=True,
                        help="path to output directory")
    parser.add_argument("-x", "--x_axis", required=True,
                        help="x-axis column name or number")
    parser.add_argument("-y", "--y_axis", required=True,
                        help="Comma-separated y-axis column names or numbers")
    parser.add_argument("-r", "--right_y_axis", default=None,
                        help="Optional right y-axis column name or number")
    parser.add_argument("--bins", type=int, default=None,
                        help="Number of bins to average data across")
    parser.add_argument("--plot_show", action="store_true",
                        help="Show plot after saving")
    parser.add_argument("--log_fit", action="store_true",
                        help="Perform log fit")
    parser.add_argument("--no_std", action="store_true",
                        help="Suppress standard deviation shading")
    parser.add_argument("--log_x", action="store_true",
                        help="Set x-axis to log scale")
    parser.add_argument("--log_y", action="store_true",
                        help="Set y-axis to log scale")
    return vars(parser.parse_args())

args = parse_command_line()
args["verbosity"] = max(0, 30 - 10 * args["verbosity"])
output_dir = args["input"][0]
os.makedirs(output_dir, exist_ok=True)

def clean_column_names(df):
    df.columns = [re.sub(r',\s+', ', ', col.strip()) for col in df.columns]
    return df

def load_first_columns(directory):
    folder_name = os.path.basename(os.path.normpath(directory))
    pattern = "AVG.txt" if folder_name.startswith("FS") else "DYN.txt"

    for fname in os.listdir(directory):
        if fname.endswith(pattern):
            path = os.path.join(directory, fname)
            df = pd.read_csv(path, sep="\t", skiprows=1, engine="python")
            df = clean_column_names(df)
            print(f"\n📄 Columns in {fname} from {folder_name}:")
            for i, col in enumerate(df.columns, 1):
                print(f"  {i:2d}. {col}")
            return df.columns
    return []

def resolve_column(col_input, all_columns):
    try:
        idx = int(col_input) - 1
        if 0 <= idx < len(all_columns):
            return all_columns[idx]
        else:
            raise ValueError(f"Column index {col_input} out of range.")
    except ValueError:
        if col_input in all_columns:
            return col_input
        else:
            raise ValueError(f"Column '{col_input}' not found.")

def resolve_columns(col_inputs, all_columns):
    if isinstance(col_inputs, str):
        col_inputs = [c.strip() for c in col_inputs.split(",")]
    return [resolve_column(col, all_columns) for col in col_inputs]

def load_dataframes(directory):
    data = []
    folder_name = os.path.basename(os.path.normpath(directory))
    pattern = "AVG.txt" if folder_name.startswith("FS") else "DYN.txt"

    for f in os.listdir(directory):
        if f.endswith(pattern):
            try:
                path = os.path.join(directory, f)
                df = pd.read_csv(path, sep="\t", skiprows=1, engine="python")
                df = clean_column_names(df)
                data.append(df)
            except Exception as e:
                print(f"❌ Failed to read {f}: {e}")
    return data

def compute_average(data_frames, x_col, y_cols, r_col=None, bins=None):
    if not data_frames:
        return pd.DataFrame()
    df = pd.concat(data_frames, axis=0, ignore_index=True)
    all_needed = [x_col] + y_cols + ([r_col] if r_col else [])
    if any(col not in df.columns for col in all_needed):
        print(f"Error: Required column(s) missing.\nAvailable: {df.columns.tolist()}")
        return pd.DataFrame()

    if bins:
        df = df.sort_values(by=x_col)
        df["bin"] = pd.cut(df[x_col], bins=bins)
        grouped = df.groupby("bin", observed=False)
        result = grouped[y_cols + ([r_col] if r_col else [])].agg(["mean", "std"])
        result[x_col] = grouped[x_col].apply(lambda s: s.name.mid)
        result.columns = ['_'.join(filter(None, col)).strip() for col in result.columns]
        return result.reset_index(drop=True)
    else:
        return df[all_needed].groupby(x_col, as_index=False).mean()

def plot_data(dfs, x_col, y_cols, r_col, labels):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx() if r_col else None
    styles = itertools.cycle(['-', '--', '-.', ':'])
    markers = itertools.cycle(['o', 's', '^', 'D', 'v', '*'])

    for df, label in zip(dfs, labels):
        x = df[x_col]
        style = next(styles)
        marker = next(markers)

        for y in y_cols:
            col_mean = f"{y}_mean"
            col_std = f"{y}_std"
            if col_mean in df:
                y_val = df[col_mean]
                y_std = df.get(col_std, None)
            else:
                y_val = df[y]
                y_std = None

            ax1.plot(x, y_val, linestyle=style, marker=marker, label=f"{y} ({label})")
            if y_std is not None and not args["no_std"]:
                ax1.fill_between(x, y_val - y_std, y_val + y_std, alpha=0.3)

            if args["log_fit"] and "Time" in x_col and "Disp" in y:
                try:
                    valid = x > 0
                    log_x = np.log(x[valid])
                    y_fit = y_val[valid]
                    coeffs = np.polyfit(log_x, y_fit, 1)
                    fit_line = coeffs[0] * np.log(x[valid]) + coeffs[1]
                    ax1.plot(x[valid], fit_line, '--', label=f"Log Fit ({label})")
                except Exception as e:
                    print(f"Log fit failed for {label}: {e}")

        if r_col:
            col_mean = f"{r_col}_mean"
            col_std = f"{r_col}_std"
            if col_mean in df:
                y_val = df[col_mean]
                y_std = df.get(col_std, None)
            else:
                y_val = df[r_col]
                y_std = None

            ax2.plot(x, y_val, linestyle=style, marker=marker, label=f"{r_col} ({label})", color='red')
            if y_std is not None and not args["no_std"]:
                ax2.fill_between(x, y_val - y_std, y_val + y_std, color='red', alpha=0.2)

    ax1.set_xlabel(x_col)
    ax1.set_ylabel(", ".join(y_cols))
    if r_col:
        ax2.set_ylabel(r_col)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    else:
        ax1.legend(loc="best")

    if args["log_x"]:
        ax1.set_xscale("log")
        if ax2: ax2.set_xscale("log")
    if args["log_y"]:
        ax1.set_yscale("log")
        if ax2: ax2.set_yscale("log")

    ax1.grid(True)
    safe_x = x_col.replace(" ", "_").replace("(", "").replace(")", "")
    safe_y = "_".join(y.replace(" ", "_") for y in y_cols)
    safe_r = f"_and_{r_col.replace(' ', '_')}" if r_col else ""
    outname = f"{'-'.join(labels)}_avg_{safe_x}_vs_{safe_y}{safe_r}.png"
    outpath = os.path.join(output_dir, outname)
    plt.savefig(outpath)
    if args["plot_show"]:
        plt.show()
    print(f"✅ Plot saved to: {outpath}")

def main():
    setup_logging(args["verbosity"])
    first_cols = load_first_columns(args["dirs"][0])
    try:
        x_col = resolve_column(args["x_axis"], first_cols)
        y_cols = resolve_columns(args["y_axis"], first_cols)
        r_col = resolve_column(args["right_y_axis"], first_cols) if args["right_y_axis"] else None
    except Exception as e:
        print(f"❌ {e}")
        return

    all_dfs, labels = [], []
    for d in args["dirs"]:
        frames = load_dataframes(d)
        avg_df = compute_average(frames, x_col, y_cols, r_col, args.get("bins"))
        if not avg_df.empty:
            all_dfs.append(avg_df)
            labels.append(os.path.basename(os.path.normpath(d)))

    if all_dfs:
        plot_data(all_dfs, x_col, y_cols, r_col, labels)
    else:
        print("❌ No data to plot.")

if __name__ == "__main__":
    main()
