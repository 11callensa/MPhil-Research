import pandas as pd

from Genetic_Algorithm import set_materials, set_PT, calculate_spacing
from Materials_Extractor import extract_hydrogen, extract_compound
from DFT import calculate_energy
from Mol_Geometry import plot, surface_finder, reorient_coordinates
from Binding_Sites import mesh, compute_energy_with_hydrogen, find_local_minima, plot_local_minima
from Combine_Matrices import combine_matrices, place_hydrogen_molecules


choice = input("Press 1. for testing mode, 2. for full mode: ")

if choice == '2':

    hydrogen, compound_name, compound_ID = set_materials()
    temperature, pressure = set_PT()
    hydrogen_spacing = calculate_spacing(temperature, pressure)

    hydrogen_bond, hydrogen_xyz = extract_hydrogen(f"{hydrogen}.poscar")
    compound_bond, compound_xyz = extract_compound(compound_ID)

    # Compute the total energy of the compound
    compound_energy, compound_symbols = calculate_energy(compound_xyz)

    # Conduct geometry transformations on compound
    plot(compound_xyz)
    surface_points = surface_finder(compound_xyz)

    compound_reo_xyz = reorient_coordinates(compound_xyz, surface_points)
    plot(compound_reo_xyz)

    # Find binding sites

    mesh_points = mesh(compound_reo_xyz)
    binding_sites, total_energies = compute_energy_with_hydrogen(compound_reo_xyz, mesh_points)
    local_minima_positions = find_local_minima(binding_sites, total_energies)

    print("Local minima coordinates:")
    print(local_minima_positions)

    # Compute total energy of H2 on surface


elif choice == '1':

    hydrogen, compound_name, compound_ID = set_materials()
    temperature, pressure = set_PT()
    hydrogen_spacing = calculate_spacing(temperature, pressure)

    hydrogen_bond, hydrogen_xyz = extract_hydrogen(f"{hydrogen}.poscar")
    compound_bond, compound_xyz = extract_compound(compound_ID)

    surface_points = surface_finder(compound_xyz)
    compound_reo_xyz = reorient_coordinates(compound_xyz, surface_points)

    plot(compound_reo_xyz)

    # Find binding sites

    mesh_points, surface_points = mesh(compound_reo_xyz)
    binding_sites, total_energies = compute_energy_with_hydrogen(compound_reo_xyz, mesh_points)
    local_minima_positions = find_local_minima(binding_sites, total_energies)

    plot_local_minima(binding_sites, total_energies, local_minima_positions, mesh_points, surface_points)

    # Compute total energy of H2 on surface


