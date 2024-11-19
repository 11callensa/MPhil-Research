import os
import numpy as np

from mp_api.client import MPRester
from ase.io import read
from dotenv import load_dotenv
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from itertools import combinations

load_dotenv()

key = os.getenv("MATERIALS_KEY")


def extract_hydrogen(file_path):

    structure = read(file_path)
    atoms = structure.get_chemical_symbols()
    positions = structure.get_positions()

    if atoms.count('H') == 2:
        pos1 = positions[0]
        pos2 = positions[1]
        bond_length = np.linalg.norm(pos1 - pos2)                                                                       # Calculate distance between two H atoms

        xyz_format = []

        for symbol, pos in zip(atoms, positions):
            xyz_format.append(f"{symbol} {pos[0]:.10f} {pos[1]:.10f} {pos[2]:.10f}")

        return bond_length, xyz_format
    else:
        print("File does not contain exactly two hydrogen atoms.")
        return None, None


def extract_compound(material_id):

    with MPRester(key) as m:

        structure = m.get_structure_by_material_id(material_id)

        sga = SpacegroupAnalyzer(structure)
        conventional_structure = sga.get_conventional_standard_structure()

    bond_lengths = {}

    for i, site in enumerate(structure):                                                                                # Iterate through all sites in the structure
        distances = []
        for j in range(len(structure)):
            if i != j:                                                                                                  # Avoid self-distance
                distance = structure.get_distance(i, j)                                                                 # Get distance with minimum image convention
                distances.append(distance)

        bond_lengths = distances                                                                                        # Store the bond lengths in the dictionary

    atoms = [site.species_string for site in conventional_structure]
    positions = [site.coords for site in conventional_structure]

    xyz_format = []

    for symbol, pos in zip(atoms, positions):
        xyz_format.append(f"{symbol} {pos[0]:.10f} {pos[1]:.10f} {pos[2]:.10f}")

    return bond_lengths, xyz_format
