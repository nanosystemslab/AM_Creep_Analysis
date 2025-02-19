import argparse
import glob
import logging
import os
import sys
import shutil
import types
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas.plotting

__version__ = "0.0.1"


def setup_logging(verbosity):
    log_fmt = ("%(levelname)s - %(module)s - "
               "%(funcName)s @%(lineno)d: %(message)s")
    logging.basicConfig(filename=None,
                        format=log_fmt,
                        level=logging.getLevelName(verbosity))

    return


def parse_command_line():
    parser = argparse.ArgumentParser(description="Analyse sensor data")
    parser.add_argument("-V", "--version", "--VERSION", action="version",
                        version="%(prog)s {}".format(__version__))
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        dest="verbosity", help="verbose output")
    parser.add_argument("-d", "--dir", dest="dirs",
                        nargs='+',
                        default=None, required=False,
                        help="directories with data files")
    parser.add_argument("-i", "--in", dest="input",
                        nargs='+',
                        default=None, required=False, help="path to input")
    parser.add_argument("-x", "--x_axis", dest="x_axis",
                        nargs='+', type=int,
                        default=None, required=False, help="x axis parameter")
    parser.add_argument("-y", "--y_axis", dest="y_axis",
                        nargs='+', type=int,
                        default=None, required=False, help="y axis parameter")
    parser.add_argument("--plot_show", dest="plot_show", action="store_true", 
                        help="plot show")
    ret = vars(parser.parse_args())
    ret["verbosity"] = max(0, 30 - 10 * ret["verbosity"])
    return ret


def load_file(filepath):
    df = pd.read_csv(filepath, encoding='unicode_escape',
                     sep='\t', skiprows=(0, 1))
    logging.info(df)
    col = df.columns.tolist()
    new_col = [i.encode("latin1").decode("utf-8").lstrip() for i in col]
    df.columns = new_col
    df.meta = types.SimpleNamespace()
    df.meta.filepath = filepath
    df.meta.filepath = Path(filepath).stem
    return df

def plot_single(file, x_axis=None, y_axis=None, plot_show=False):
    df = load_file(file)  # Load file into DataFrame
    filename = df.meta.filepath 
    
    col = df.columns.tolist()  # Get list of column names
    
    print("\nAvailable columns:")
    for idx, column in enumerate(col):
        print(f"{idx}: {column}")  # Display column index and name
    if x_axis != None: 
        x_idx = x_axis
        x_col = col[x_idx]
    else:
        # Prompt user to select X-axis column
        while True:
            try:
                x_idx = int(input("\nEnter the number for the X-axis column: "))
                x_col = col[x_idx]
                break
            except (IndexError, ValueError):
                print("Invalid selection. Please enter a valid column number.")
    if y_axis != None: 
        y_idx = y_axis
        y_col = col[y_idx]
    else: 
        # Prompt user to select Y-axis column
        while True:
            try:
                y_idx = int(input("Enter the number for the Y-axis column: "))
                y_col = col[y_idx]
                break
            except (IndexError, ValueError):
                print("Invalid selection. Please enter a valid column number.")
    
    # Plot the selected columns
    plt.figure(figsize=(8, 6))
    plt.plot(df[x_col], df[y_col], marker='o', linestyle='-')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"Plot of {y_col} vs {x_col}\n {filename}")
    plt.grid(True)
    
    # Replace whitespace in x_col and y_col
    x_col_clean = x_col.replace(" ", "_")
    y_col_clean = y_col.replace(" ", "_")
    
    # Define output file name with sanitized column names
    output_file = f"out/plot_{filename}--{x_col_clean}-vs-{y_col_clean}.png"
    
    # Save the plot
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved as: {output_file}")
    
    if plot_show: 
        plt.show()



def main():
    cmd_args = parse_command_line()
    setup_logging(cmd_args['verbosity'])
    files_passed = len(cmd_args['input'])
    if files_passed == 1: 
        plot_single(cmd_args['input'][0], x_axis=cmd_args["x_axis"], y_axis=cmd_args["y_axis"], plot_show = cmd_args['plot_show'])
    else: 
        print("more files then I can do right now")


if "__main__" == __name__:
    try:
        sys.exit(main())
    except KeyboardInterrupt:
       print("exited")


