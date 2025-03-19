import argparse
import logging
import sys
import types
from pathlib import Path
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__version__ = "0.0.1"


def setup_logging(verbosity):
    """ Configure logging settings. """
    log_fmt = ("%(levelname)s - %(module)s - "
               "%(funcName)s @%(lineno)d: %(message)s")
    logging.basicConfig(filename=None,
                        format=log_fmt,
                        level=logging.getLevelName(verbosity))
    return


def parse_command_line():
    """ Parse command-line arguments. """
    parser = argparse.ArgumentParser(description="Analyse sensor data from DYN files")
    parser.add_argument("-V", "--version", action="version",
                        version="%(prog)s {}".format(__version__))
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        dest="verbosity", help="Verbose output")
    parser.add_argument("-d", "--directory", dest="directory",
                        required=True, help="Path to directory containing *_DYN.txt files")
    parser.add_argument("-x", "--x_axis", dest="x_axis",
                        type=int, required=False, help="X-axis column index")
    parser.add_argument("-y", "--y_axis", dest="y_axis",
                        type=int, required=False, help="Y-axis column index")
    parser.add_argument("--bins", dest="bins", type=int, default=100,
                        help="Number of bins for averaging (default: 100)")
    parser.add_argument("--plot_show", dest="plot_show", action="store_true", 
                        help="Show plot")

    ret = vars(parser.parse_args())
    ret["verbosity"] = max(0, 30 - 10 * ret["verbosity"])
    return ret


def find_dyn_files(directory):
    """ Search for all files ending with '_DYN.txt' in the given directory. """
    dyn_files = glob.glob(f"{directory}/*_DYN.txt")
    
    if not dyn_files:
        print("No *_DYN.txt files found in the specified directory.")
        sys.exit(1)

    print(f"Found {len(dyn_files)} DYN files:")
    for file in dyn_files:
        print(f" - {file}")

    return dyn_files


def load_file(filepath):
    """ Load a single file into a DataFrame. """
    df = pd.read_csv(filepath, encoding='unicode_escape', sep='\t', skiprows=(0, 1))
    
    # Clean column names
    df.columns = [col.encode("latin1").decode("utf-8").strip() for col in df.columns]

    df.meta = types.SimpleNamespace()
    df.meta.filepath = filepath
    df.meta.filename = Path(filepath).stem

    return df


def plot_averaged(files, x_axis=None, y_axis=None, bins=100, plot_show=False):
    """ Load multiple DYN files, average data, compute standard deviation, and plot a single line. """
    dataframes = [load_file(file) for file in files]

    # Merge all files
    merged_df = pd.concat(dataframes, axis=0, ignore_index=True)

    # Display available columns if X and Y are not provided
    col = merged_df.columns.tolist()

    if x_axis is None or y_axis is None:
        print("\nAvailable columns:")
        for idx, column in enumerate(col):
            print(f"{idx}: {column}")

        while x_axis is None or x_axis < 0 or x_axis >= len(col):
            try:
                x_axis = int(input("\nEnter the number for the X-axis column: "))
                if x_axis < 0 or x_axis >= len(col):
                    print("Invalid index. Please select a valid column number.")
                    x_axis = None
            except ValueError:
                print("Invalid input. Please enter a number.")

        while y_axis is None or y_axis < 0 or y_axis >= len(col):
            try:
                y_axis = int(input("Enter the number for the Y-axis column: "))
                if y_axis < 0 or y_axis >= len(col):
                    print("Invalid index. Please select a valid column number.")
                    y_axis = None
            except ValueError:
                print("Invalid input. Please enter a number.")

    # Assign selected columns
    x_col = col[x_axis]
    y_col = col[y_axis]

    # Convert selected columns to numeric
    merged_df[x_col] = pd.to_numeric(merged_df[x_col], errors='coerce')
    merged_df[y_col] = pd.to_numeric(merged_df[y_col], errors='coerce')

    # Ensure there are valid numerical values before binning
    if merged_df[x_col].isnull().all():
        print(f"Error: Column '{x_col}' contains only NaN values. Please check the data.")
        sys.exit(1)

    # **BINNING TO AVOID TOO MANY UNIQUE X-VALUES**
    merged_df[x_col] = pd.cut(merged_df[x_col], bins=bins).apply(lambda x: x.mid)

    # Compute mean and standard deviation
    grouped = merged_df.groupby(x_col)[y_col].agg(['mean', 'std']).reset_index()
    grouped = grouped.sort_values(by=x_col)  # Sort for smooth plotting

    # Plot averaged data
    plt.figure(figsize=(8, 6))
    plt.plot(grouped[x_col], grouped["mean"], linestyle='-', color='b', label="Averaged Data")  # Smooth line
    plt.fill_between(grouped[x_col], grouped["mean"] - grouped["std"], grouped["mean"] + grouped["std"], 
                     color='blue', alpha=0.2, label="Standard Deviation")

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"Averaged {x_col} vs {y_col} (Bins: {bins})")
    plt.legend()
    plt.grid(True)

    # Save plot
    x_col_clean = x_col.replace(" ", "_")
    y_col_clean = y_col.replace(" ", "_")
    output_file = f"out/plot_averaged_{x_col_clean}_vs_{y_col_clean}.png"
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved as: {output_file}")

    if plot_show:
        plt.show()


def main():
    cmd_args = parse_command_line()
    setup_logging(cmd_args['verbosity'])

    # Search for *_DYN.txt files in the given directory
    dyn_files = find_dyn_files(cmd_args["directory"])

    # Process and plot data with dynamic binning
    plot_averaged(dyn_files, x_axis=cmd_args["x_axis"], y_axis=cmd_args["y_axis"], bins=cmd_args["bins"], plot_show=cmd_args['plot_show'])


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Exited.")

