import os
import numpy as np

from ase.io import read
from mp_api.client import MPRester
from dotenv import load_dotenv



load_dotenv()

key = os.getenv("MATERIALS_KEY")


def extract_material_info(material_ID):

    with MPRester(key) as mpr:
        structure = mpr.get_structure_by_material_id(material_ID)
        composition = structure.composition

        expanded_formula, factor = structure.composition.get_integer_formula_and_factor()

        atom_counts = {str(atom): count for atom, count in composition.items()}

        print("Reduced Formula:", structure.composition.reduced_formula)
        print("Expanded Formula:", expanded_formula)
        print("Scaling Factor:", factor)

        print(atom_counts)

        element_smiles = ''.join([f'[{element}]' for element, count in atom_counts.items() for _ in range(int(count))])

        print(element_smiles)

        bond_lengths = {}

        for i, site in enumerate(structure):                                                                            # Iterate through all sites in the structure
            distances = []
            for j in range(len(structure)):
                if i != j:                                                                                              # Avoid self-distance
                    distance = structure.get_distance(i, j)                                                             # Get distance with minimum image convention
                    distances.append(distance)

            bond_lengths = distances                                                                                    # Store the bond lengths in the dictionary

        return element_smiles, bond_lengths


def extract_hydrogen(file_path):

    structure = read(file_path)
    atoms = structure.get_chemical_symbols()
    positions = structure.get_positions()

    element_smiles = ''.join(f'[{atom}]' for atom in atoms)

    if atoms.count('H') == 2:
        pos1 = positions[0]
        pos2 = positions[1]
        bond_length = np.linalg.norm(pos1 - pos2)  # Calculate distance between two H atoms
    else:
        print("File does not contain exactly two hydrogen atoms.")

    return element_smiles, [bond_length]
