from collections import Counter

from Materials_Extractor import extract_hydrogen, extract_compound
from Mol_Geometry import plot, surface_finder, reorient_coordinates, centre_coords, unorient_minima, unorient_mol
from Binding_Sites import mesh, energy_profile, placement_xy, plot_local_minima
from Place_Molecules import combine_matrices, placement_z
from DFT import calculate_energy, get_spin
from Compound_Properties import node_edge_features, system_feature_compiler, mass_and_charge


def training_data_creator(hydrogen, compound_ID, edge_indices):

    # Extract hydrogen and compound bond lengths and coordinates
    hydrogen_bond = extract_hydrogen(f"{hydrogen}.poscar")
    compound_xyz, oxidation_states = extract_compound(compound_ID)

    print("Oxidation States: ", oxidation_states)

    centered_xyz, center = centre_coords(compound_xyz)
    print("Compound xyz: ", compound_xyz)
    print("Center: ", center)
    plot(compound_xyz)
    print("Centered xyz: ", centered_xyz)
    plot(centered_xyz)

    node_features, edge_features = node_edge_features(centered_xyz, edge_indices, oxidation_states)

    tot_mass, tot_charge = mass_and_charge(centered_xyz, oxidation_states)

    print("Total mass: ", tot_mass)
    print("Total charge: ", tot_charge)
    print("Node Features: ", node_features)
    print("Edge Features: ", edge_features)

    # Conduct geometry transformations on compound
    surface_points = surface_finder(compound_xyz)
    compound_reo_xyz, surface_reo_points, rotation_matrix, translation = reorient_coordinates(compound_xyz,
                                                                                              surface_points)

    plot(compound_reo_xyz)

    # Find binding sites
    mesh_points = mesh(compound_reo_xyz, surface_reo_points)
    binding_sites, total_energies = energy_profile(compound_reo_xyz, mesh_points)
    local_minima_positions = placement_xy(binding_sites, total_energies)

    plot_local_minima(binding_sites, total_energies, local_minima_positions, mesh_points)

    local_minima_unreo = unorient_minima(local_minima_positions, rotation_matrix, translation, center)

    print("Compound Reo XYZ", compound_reo_xyz)
    print(type(compound_reo_xyz))

    print("Local minima: ", repr(local_minima_positions))
    print(type(local_minima_positions))

    print("Un-oriented Local Minima: ", repr(local_minima_unreo))
    print(type(local_minima_unreo))

    print("Binding sites: ", repr(binding_sites))
    print(type(binding_sites))

    print("Total energies: ", total_energies)
    print(type(total_energies))

    print("Surface points: ", surface_reo_points)
    print(type(surface_reo_points))

    print("Hydrogen bond: ", hydrogen_bond)
    print(type(hydrogen_bond))

    # Place hydrogen molecules on binding sites
    hydrogen_placed, num_molecules, mol_pair_xyz = placement_z(local_minima_positions, binding_sites, total_energies,
                                                               compound_reo_xyz, surface_reo_points, hydrogen_bond)

    print("Number of molecules: ", num_molecules)

    mol_pair_xyz_unreo = unorient_mol(mol_pair_xyz, rotation_matrix, translation, center)

    print("Mol Pair XYZ: ", mol_pair_xyz)

    print("Un-oriented Mol Pair XYZ: ", mol_pair_xyz_unreo)

    # Combine matrices
    combined_xyz = combine_matrices(compound_reo_xyz, hydrogen_placed)

    print("Combined XYZ: ", combined_xyz)

    # Calculate individual and combined energies
    compound_energy = calculate_energy(compound_reo_xyz)  # Compute the total energy of the compound
    hydrogen_energy = calculate_energy(hydrogen_placed)  # Compute the total energy of the hydrogen in-situ
    combined_energy = calculate_energy(combined_xyz)  # Compute the total energy of the combined compound and hydrogen

    print("Compound Energy: ", compound_energy)
    print("Hydrogen Energy: ", hydrogen_energy)
    print("Combined Energy: ", combined_energy)

    # Calculate adsorption energy
    adsorption_energy = combined_energy - (hydrogen_energy + compound_energy)
    print(f"Adsorption Energy: {adsorption_energy} eV")

    adsorption_per_mol = adsorption_energy / num_molecules
    print(f"Adsorption Energy per molecule: {adsorption_per_mol} eV")

    chemicals = [line.split()[0] for line in centered_xyz]
    element_count = [(atom, str(count)) for atom, count in Counter(chemicals).items()]
    net_spin = get_spin(element_count)

    system_features, masks = system_feature_compiler(local_minima_unreo, mol_pair_xyz_unreo, num_molecules, net_spin,
                                                     tot_mass,
                                                     tot_charge, adsorption_energy, adsorption_per_mol)

    print("System Features: ", system_features)
    print("Masks: ", masks)

    return node_features, edge_features, system_features, masks

