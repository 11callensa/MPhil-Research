from Genetic_Algorithm import set_materials
from Materials_Extractor import extract_hydrogen, extract_compound
from Mol_Geometry import plot, surface_finder, reorient_coordinates
from Binding_Sites import mesh, energy_profile, find_local_minima, plot_local_minima
from Combine_Matrices import combine_matrices, place_hydrogen_molecules
from DFT import calculate_energy

choice = input("Press 1. for testing mode, 2. for full mode: ")

if choice == '2':

    # Extract hydrogen and compound bond lengths and coordinates
    hydrogen, compound_ID = set_materials()
    hydrogen_bond = extract_hydrogen(f"{hydrogen}.poscar")
    compound_xyz = extract_compound(compound_ID)

    # Conduct geometry transformations on compound
    surface_points = surface_finder(compound_xyz)
    plot(compound_xyz)
    compound_reo_xyz, surface_reo_points = reorient_coordinates(compound_xyz, surface_points)

    plot(compound_reo_xyz)

    # Find binding sites
    mesh_points = mesh(compound_reo_xyz, surface_reo_points)
    binding_sites, total_energies = energy_profile(compound_reo_xyz, mesh_points)
    local_minima_positions = find_local_minima(binding_sites, total_energies)

    plot_local_minima(binding_sites, total_energies, local_minima_positions, mesh_points)

    # Place hydrogen molecules on binding sites
    hydrogen_placed = place_hydrogen_molecules(local_minima_positions, surface_reo_points, hydrogen_bond)

    # Combine matrices
    combined_xyz = combine_matrices(compound_reo_xyz, hydrogen_placed)

    # Calculate individual and combined energies
    compound_energy = calculate_energy(compound_reo_xyz)                                                                # Compute the total energy of the compound
    hydrogen_energy = calculate_energy(hydrogen_placed)                                                                 # Compute the total energy of the hydrogen in-situ
    combined_energy = calculate_energy(combined_xyz)                                                                    # Compute the total energy of the combined compound and hydrogen

    # Calculate adsorption energy
    adsorption_energy = combined_energy - (hydrogen_energy + compound_energy)
    print(f"Adsorption Energy: {adsorption_energy} eV")

elif choice == '1':

    hydrogen, compound_ID = set_materials()

    hydrogen_bond = extract_hydrogen(f"{hydrogen}.poscar")
    compound_xyz = extract_compound(compound_ID)

    surface_points = surface_finder(compound_xyz)
    compound_reo_xyz, surface_reo_points = reorient_coordinates(compound_xyz, surface_points)

    plot(compound_reo_xyz)

    # Find binding sites
    mesh_points = mesh(compound_reo_xyz, surface_reo_points)
    binding_sites, total_energies = energy_profile(compound_reo_xyz, mesh_points)
    local_minima_positions = find_local_minima(binding_sites, total_energies)

    plot_local_minima(binding_sites, total_energies, local_minima_positions, mesh_points)

    hydrogen_placed = place_hydrogen_molecules(local_minima_positions, surface_reo_points, hydrogen_bond)

    combined_matrix = combine_matrices(compound_reo_xyz, hydrogen_placed)

