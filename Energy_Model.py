import time

import torch
from ase.io import read
from mace.calculators import MACECalculator
import numpy as np


def remove_hydrogens_from_xyz(input_filename, output_filename):
    with open(input_filename, 'r') as f:
        lines = f.readlines()

    # First line: number of atoms
    num_atoms = int(lines[0].strip())

    # Second line: comment (can keep as is)
    comment = lines[1]

    # Atom lines start from line 2 to line 2+num_atoms
    atom_lines = lines[2:2 + num_atoms]

    # Filter out lines where the atom is Hydrogen (starts with 'H' or 'H ')
    filtered_atoms = [line for line in atom_lines if not line.strip().startswith('H')]

    # New number of atoms after filtering
    new_num_atoms = len(filtered_atoms)

    # Write to output file
    with open(output_filename, 'w') as f:
        f.write(f"{new_num_atoms}\n")
        f.write(comment)
        for line in filtered_atoms:
            f.write(line)


def keep_only_hydrogens_from_xyz(input_filename, output_filename):
    with open(input_filename, 'r') as f:
        lines = f.readlines()

    # First line: number of atoms
    num_atoms = int(lines[0].strip())

    # Second line: comment (keep as is)
    comment = lines[1]

    # Atom lines start from line 2 to line 2+num_atoms
    atom_lines = lines[2:2 + num_atoms]

    # Keep only lines where the atom is Hydrogen (starts with 'H' or 'H ')
    hydrogen_atoms = [line for line in atom_lines if line.strip().startswith('H')]

    # New number of atoms after filtering
    new_num_atoms = len(hydrogen_atoms)

    # Write to output file
    with open(output_filename, 'w') as f:
        f.write(f"{new_num_atoms}\n")
        f.write(comment)
        for line in hydrogen_atoms:
            f.write(line)


def run_testing(name):

    calc = MACECalculator(model_paths="MACE Models/mace_agnesi_small.model", device="cpu")  # DECENT - NiO, Si, FeTi

    filename = f"{name}_predicted.xyz"

    atoms_comb = read(f"Predicted Coords Testing/{filename}")

    keep_only_hydrogens_from_xyz(f"Predicted Coords Testing/{filename}", f"MACE Files/H-({name})_alone.xyz")
    atoms_H = read(f"MACE Files/H-({name})_alone.xyz")

    remove_hydrogens_from_xyz(f"Predicted Coords Testing/{filename}", f"MACE Files/{name}_alone.xyz")
    atoms_comp = read(f'MACE Files/{name}_alone.xyz')

    atoms_comb.calc = calc
    atoms_comp.calc = calc
    atoms_H.calc = calc

    start_time = time.time()

    energy_comb = np.float32(atoms_comb.get_total_energy())
    energy_comp = np.float32(atoms_comp.get_total_energy())
    energy_H = np.float32(atoms_H.get_total_energy())

    end_time = time.time()
    print("Time spent predicting: ", end_time-start_time)

    num_h_atoms = len(atoms_H)
    num_h2_molecules = num_h_atoms // 2  # Assumes all hydrogens are in H2 molecules

    adsorption_energy_per_h2 = (energy_comb - energy_comp - energy_H) / num_h2_molecules

    return adsorption_energy_per_h2
