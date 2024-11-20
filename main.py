import pandas as pd

from Genetic_Algorithm import set_materials, set_PT, calculate_spacing
from Materials_Extractor import extract_hydrogen, extract_compound
from DFT import calculate_energy
from Mol_Geometry import plot, surface_finder, reorient_coordinates
from Combine_Matrices import combine_matrices, place_hydrogen_molecules

import test_bench


choice = input("Press 1. for testing mode, 2. for full mode: ")

if choice == '2':

    hydrogen, compound_name, compound_ID = set_materials()
    temperature, pressure = set_PT()
    hydrogen_spacing = calculate_spacing(temperature, pressure)

    hydrogen_bond, hydrogen_xyz = extract_hydrogen(f"{hydrogen}.poscar")
    compound_bond, compound_xyz = extract_compound(compound_ID)

    # Compute the total energy of the compound
    compound_energy, compound_symbols = calculate_energy(compound_xyz)
    print("Compound Energy: ", compound_energy)

    # Conduct geometry transformations on compound
    plot(compound_xyz)
    surface_points = surface_finder(compound_xyz)

    plot(reorient_coordinates(compound_xyz, surface_points))
    compound_reo_xyz = reorient_coordinates(compound_xyz, surface_points)

    # Compute total energy of random permutations of H2 on surface
    loops = 0
    ads_energies = []

    while loops < 10:

        print("Molecule placement: ", loops)
        hydrogen_placed, mol_count = place_hydrogen_molecules(compound_bond, compound_reo_xyz, hydrogen_bond, hydrogen_spacing)
        print("Hydrogen successfully placed")

        hydrogen_energy, hydrogen_symbols = calculate_energy(hydrogen_placed)
        print("Hydrogen Energy: ", hydrogen_energy)

        matrix_combined = combine_matrices(compound_reo_xyz, hydrogen_placed)
        energy_combined, symbol_count = calculate_energy(matrix_combined)
        print("Combined Energy: ", energy_combined)

        adsorption_energy = energy_combined - (compound_energy + hydrogen_energy)
        ads_energies.append(adsorption_energy)

        loops += 1

    # Calculate average adsorption
    average_adsorption = sum(ads_energies) / len(ads_energies)
    print(f"Average Adsorption Energy: {average_adsorption} eV")

    if average_adsorption <= -0.2:
        print("Chemisorption will mostly likely occur")

    elif -0.2 < average_adsorption < 0:
        print("Physisorption will most likely occur")

elif choice == '1':

    hydrogen, compound_name, compound_ID = set_materials()
    temperature, pressure = set_PT()
    hydrogen_spacing = calculate_spacing(temperature, pressure)

    hydrogen_bond, hydrogen_xyz = extract_hydrogen(f"{hydrogen}.poscar")
    compound_bond, compound_xyz = extract_compound(compound_ID)

    surface_points = surface_finder(compound_xyz)
    compound_reo_xyz = reorient_coordinates(compound_xyz, surface_points)
    plot(reorient_coordinates(compound_xyz, surface_points))

    # Compute total energy of random permutations of H2 on surface
    loops = 0
    ads_energies = []

    while loops < 10:
        print("Molecule placement No.: ", loops)
        hydrogen_placed, mol_count = place_hydrogen_molecules(compound_bond, compound_reo_xyz, hydrogen_bond,
                                                              hydrogen_spacing)
        print("Hydrogen successfully placed")

        hydrogen_energy, hydrogen_symbols = calculate_energy(hydrogen_placed)
        print("Hydrogen Energy: ", hydrogen_energy)

        matrix_combined = combine_matrices(compound_reo_xyz, hydrogen_placed)
        energy_combined, symbol_count = calculate_energy(matrix_combined)
        print("Combined Energy: ", energy_combined)

        loops += 1
