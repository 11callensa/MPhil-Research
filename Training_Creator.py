from Materials_Extractor import extract_hydrogen, extract_compound
from Mol_Geometry import plot, surface_finder, reorient_coordinates
from Binding_Sites import mesh, energy_profile, placement_xy, plot_local_minima
from Place_Molecules import combine_matrices, placement_z
from DFT import calculate_energy


def training_data_creator(hydrogen, compound_ID):

    # Extract hydrogen and compound bond lengths and coordinates
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
    local_minima_positions = placement_xy(binding_sites, total_energies)

    plot_local_minima(binding_sites, total_energies, local_minima_positions, mesh_points)

    print("Local minima: ", repr(local_minima_positions))
    print(type(local_minima_positions))

    print("Binding sites: ", repr(binding_sites))
    print(type(binding_sites))

    print("Total energies: ", total_energies)
    print(type(total_energies))

    print("Compound XYZ", compound_reo_xyz)
    print(type(compound_reo_xyz))

    print("Surface points: ", surface_reo_points)
    print(type(surface_reo_points))

    print("Hydrogen bond: ", hydrogen_bond)
    print(type(hydrogen_bond))

    # Place hydrogen molecules on binding sites
    hydrogen_placed = placement_z(local_minima_positions, binding_sites, total_energies, compound_reo_xyz,
                                  surface_reo_points, hydrogen_bond)

    # Combine matrices
    combined_xyz = combine_matrices(compound_reo_xyz, hydrogen_placed)

    # Calculate individual and combined energies
    compound_energy = calculate_energy(compound_reo_xyz)                                                                # Compute the total energy of the compound
    hydrogen_energy = calculate_energy(hydrogen_placed)                                                                 # Compute the total energy of the hydrogen in-situ
    combined_energy = calculate_energy(combined_xyz)                                                                    # Compute the total energy of the combined compound and hydrogen

    # # Calculate adsorption energy
    adsorption_energy = combined_energy - (hydrogen_energy + compound_energy)
    print(f"Adsorption Energy: {adsorption_energy} eV")

    return adsorption_energy
