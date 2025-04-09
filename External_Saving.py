import numpy as np
import csv
import os


def save_edges_to_csv(edges, name):
    """
        Saves the updated edge list to the CSV file.

        :param edges: Edge indices.
        :param name: Name of the compound.
    """

    CSV_FILE = f"Edge Indices/edge_indices_{name}.csv"  # CSV file storing connections
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(edges)


def load_existing_edges(name):
    """ Load existing edges from the CSV file. """
    CSV_FILE = f"Edge Indices/edge_indices_{name}.csv"  # CSV file storing connections
    if not os.path.exists(CSV_FILE):
        print("Does not exist")
        return []

    with open(CSV_FILE, "r") as f:
        reader = csv.reader(f)
        return [list(map(int, row)) for row in reader]


def save_xyz(atom_symbols, coords, filepath):
    if not filepath.endswith('.xyz'):
        filepath += '.xyz'
    with open(filepath, 'w') as f:
        f.write(f"{len(atom_symbols)}\n")
        f.write("0 1\n")
        for symbol, (x, y, z) in zip(atom_symbols, coords):
            f.write(f"{symbol}  {x:.6f}  {y:.6f}  {z:.6f}\n")
    print(f"\nSaved optimised geometry to: {filepath}")


def save_optimised_coords(coords, filename):
    """
        Save a list of 3D coordinates to a CSV file.

        :param coords: List of lists or NumPy array of shape (N, 3), where N is the number of points.
        :param filename: Name of the CSV file to save the coordinates.
    """
    np.savetxt(filename, coords, delimiter=",", fmt="%.8f")


def load_optimised_coords(filename):
    """
    Load 3D coordinates from a CSV file into a NumPy array.

    :param filename: Name of the CSV file to load the coordinates from.
    :return: NumPy array of shape (N, 3).
    """
    return np.loadtxt(filename, delimiter=",")