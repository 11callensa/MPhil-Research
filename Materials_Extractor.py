import os
import numpy as np

from mp_api.client import MPRester
from ase.io import read
from dotenv import load_dotenv
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

load_dotenv()

key = os.getenv("MATERIALS_KEY")


def extract_hydrogen(file_path):
    """
        Extracts information about H2 and calculates bond length, necessary for molecule placement.

        :param file_path: Contains the name of the poscar file for H2.
        :return: Hydrogen bond lengths.
    """

    structure = read(file_path)
    atoms = structure.get_chemical_symbols()
    positions = structure.get_positions()

    if atoms.count('H') == 2:
        pos1 = positions[0]
        pos2 = positions[1]
        bond_length = np.linalg.norm(pos1 - pos2)                                                                       # Calculate distance between two H atoms

        return bond_length
    else:
        print("File does not contain exactly two hydrogen atoms.")                                                      # Process only works for H2, not other forms of hydrogen
        return None, None


def extract_compound(material_id):
    """
        Takes in the Material Project ID and extracts xyz coordinates of the compound.

        :param material_id: Materials Project ID.
        :return: Compound coordinates.
    """

    with MPRester(key) as m:                                                                                            # Access API using key

        structure = m.get_structure_by_material_id(material_id)

        sga = SpacegroupAnalyzer(structure)
        conventional_structure = sga.get_conventional_standard_structure()

    atoms = [site.species_string for site in conventional_structure]                                                    # Extract atoms
    positions = [site.coords for site in conventional_structure]                                                        # Extract positions

    xyz_format = []

    for symbol, pos in zip(atoms, positions):                                                                           # Format in xyz format
        xyz_format.append(f"{symbol} {pos[0]:.10f} {pos[1]:.10f} {pos[2]:.10f}")

    return xyz_format
