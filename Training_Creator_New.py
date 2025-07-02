from collections import Counter

from Materials_Extractor import extract_hydrogen, extract_compound, compute_bonds, parsing
from Plotting import plot_crystal, plot_adsorption_sites, plot_adsorbed_atoms
from Mol_Geometry import surface_finder, compute_volume, centre_coords, layer1_extractor, layer2_extractor, rotater, tiler, site_finder, coverage_atomsnm2_to_ML, place_hydrogen

from Build_Connections import build_connections
from External_Saving import save_edges_to_csv, save_original_xyz
from DFT_New import calculate_energy, optimise_geometry
from Compound_Properties import get_spin, node_edge_features, mass_and_charge

import networkx as nx


def data_creator(hydrogen, compound_ID, name, test_train):

    hydrogen_bond = extract_hydrogen(f"{hydrogen}.poscar")                                                              # Extract hydrogen and compound bond lengths and coordinates.
    print("Hydrogen Bond: ", hydrogen_bond)

    compound_xyz, oxidation_states, extracted_input_features, extracted_output_features, uncertain_features = \
        (extract_compound(compound_ID, name))                                                                           # Extract coordinates, oxidation states, mechanical features, temperatures and features that could be used.

    print("Uncertain features: ", uncertain_features)
    print("Compound XYZ: ", compound_xyz)
    print('Num atoms in original cell: ', len(compound_xyz))

    ################## Setup crystal surface + underlayer ##################

    surface_points = surface_finder(compound_xyz)

    miller_input = input('Input the miller index (In the form x, y, z) of the cleavage plane, to expose the surface of the compound: ')
    miller_index = [int(x) for x in miller_input.replace(',', ' ').split()]

    distance = float(input('Input the distance limit between the 1st and 2nd layer of the compound: '))

    layer1 = layer1_extractor(compound_xyz, surface_points, miller_index)
    layer2 = layer2_extractor(compound_xyz, miller_index, layer1, distance)

    layered_xyz = list(dict.fromkeys(layer1 + layer2))
    layered_xyz.sort(key=lambda x: x.split()[0], reverse=True)
    print('Layered XYZ: ', layered_xyz)

    rotated_xyz = rotater(layered_xyz)
    print('Rotated XYZ: ', rotated_xyz)

    # edge_indices_trial = compute_bonds(rotated_xyz)
    # plot_crystal(rotated_xyz, edge_indices_trial)

    # tiled_raw_xyz = tiler(rotated_xyz)

    tiled_raw_xyz = rotated_xyz
    tiled_xyz = parsing(tiled_raw_xyz)
    tiled_xyz.sort(key=lambda x: x.split()[0], reverse=True)
    print('Tiled XYZ: ', tiled_xyz)

    # edge_indices_crystal = compute_bonds(rotated_xyz)

    edge_indices_crystal = compute_bonds(tiled_xyz)                                                                  # Computer the crystal edge indices.

    save = input("Save crystal edge indices to csv file? y/n: ")

    if save == 'y':
        save_edges_to_csv(edge_indices_crystal, name)                                                                   # Save the crystal edge indices.
    else:
        print("Crystal edge indices not saved")

    print("Edge Indices of the initial crystal alone: ", edge_indices_crystal)

    G = nx.Graph()
    G.add_edges_from([tuple(edge) for edge in edge_indices_crystal])

    # Step 2: Identify connected components
    connected_components = list(nx.connected_components(G))

    # Step 3: Keep only the largest component
    largest_component = max(connected_components, key=len)

    # Step 4: Filter atoms and edges
    largest_component_indices = sorted(largest_component)

    # Create a mapping from old index to new index
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(largest_component_indices)}

    # Filter atom list
    tiled_xyz_filtered = [tiled_xyz[i] for i in largest_component_indices]

    # Filter and remap edge indices
    edge_indices_filtered = [
        [index_map[i], index_map[j]]
        for i, j in edge_indices_crystal
        if i in largest_component and j in largest_component
    ]

    # Use filtered data from now on
    tiled_xyz = tiled_xyz_filtered
    edge_indices_crystal = edge_indices_filtered

    plot_crystal(tiled_xyz, edge_indices_crystal)
    # plot_crystal(rotated_xyz, edge_indices_crystal) # Plot the crystal and its bonds alone.

    num_atoms = len(tiled_xyz)
    # Find the number of atoms in the crystal.
    # num_atoms = len(rotated_xyz)
    print(f"The first {num_atoms} atoms are fixed.")

    ################## Crystal initial alone code ##################

    centered_xyz, center = centre_coords(tiled_xyz, num_atoms)
    print('Len of centred xyz: ', len(centered_xyz))
    # centered_xyz, center = centre_coords(rotated_xyz, num_atoms) # Centre all atoms around the crystal's centroid.
    print("Crystal center: ", center)
    print("Centered xyz: ", centered_xyz)

    node_features_crystal, edge_features_crystal =\
        node_edge_features(centered_xyz, edge_indices_crystal, oxidation_states, num_atoms, 0)                     # Extract node and edge features from the crystal alone.

    tot_mass_crystal, tot_charge_crystal = mass_and_charge(centered_xyz, oxidation_states, num_atoms, 0)           # Extract the total mass and charge of the crystal.

    print("Total mass of the crystal alone: ", tot_mass_crystal)
    print("Total charge of the crystal alone: ", tot_charge_crystal)
    print("Node Features of the crystal alone: ", node_features_crystal)
    print("Edge Features of the crystal alone: ", edge_features_crystal)

    chemicals = [line.split()[0] for line in centered_xyz]
    element_count = [(atom, str(count)) for atom, count in Counter(chemicals).items()]                                  # Extract each type of element and how many there are.
    net_spin_crystal = get_spin(element_count)                                                                          # Calculate the net spin of the crystal alone.

    print("Net Spin of the crystal alone: ", net_spin_crystal)

    volume_crystal = compute_volume(centered_xyz)                                                                       # Calculate the volume encapsulated by the crystal alone.
    print('Crystal volume: ', volume_crystal)

    adsorption_sites = site_finder(centered_xyz, min_bond=0, max_bond=2.36)
    print('Adsorption sites: ', adsorption_sites)

    plot_adsorption_sites(centered_xyz, edge_indices_crystal, adsorption_sites)

    while True:
        coverage_select = input(
            "Select coverage input type:\n"
            "1 - atoms per nm²\n"
            "2 - monolayers (ML)\n"
            "Enter 1 or 2: "
        ).strip()

        if coverage_select == '1':
            atoms_nm2 = float(input("Enter coverage in atoms per nm²: "))
            coverage = coverage_atomsnm2_to_ML(atoms_nm2, centered_xyz, adsorption_sites)
            break
        elif coverage_select == '2':
            coverage = float(input("Enter coverage in ML: "))
            break
        else:
            print("Invalid input. Please enter 1 or 2.")

    combined_xyz = place_hydrogen(centered_xyz, adsorption_sites, coverage, hydrogen_bond)
    num_H = len(combined_xyz) - num_atoms

    print("Combined XYZ: ", combined_xyz)
    print("Number of placed hydrogen atoms: ", num_H)

    plot_adsorbed_atoms(combined_xyz, title=f"H₂ Adsorbed Surface ({coverage} ML)")

    save_original_xyz(combined_xyz, 'Optimised Coordinates', f"{name}_original_coords.xyz")
    print('Combined XYZ saved!')

    #################### Combined initial Code #####################
    edge_indices_comb = build_connections(combined_xyz, num_atoms, name)
    print("Edge indices combined: ", edge_indices_comb)

    plot_crystal(combined_xyz, edge_indices_comb)                                                                       # Plot the combined system.

    node_features_init_comb, edge_features_init_comb = node_edge_features(combined_xyz, edge_indices_comb,
                                                                   oxidation_states, num_atoms, 0)                 # Extract node and edge features of the combined system.

    print("Node features initial combined: ", node_features_init_comb)
    print("Edge features initial combined: ", edge_features_init_comb)

    tot_mass_comb, tot_charge_comb = mass_and_charge(combined_xyz, oxidation_states, num_atoms, 0)                 # Extract total mass and total charge of the combined system.

    print("Total mass combined: ", tot_mass_comb)
    print("Total charge combined: ", tot_charge_comb)

    ##################### H initial alone Code #####################

    H_init_xyz = combined_xyz[num_atoms:]                                                                               # Extract the placed hydrogen atoms form the combined system.

    print("H init XYZ: ", H_init_xyz)

    filtered_indices = [pair for pair in edge_indices_comb if pair[0] >= num_atoms and pair[1] >= num_atoms]            # Extract the edge indices of the plaed hydrogen atoms.
    print("Filtered Indices: ", filtered_indices)
    edge_indices_H = [[a - num_atoms, b - num_atoms] for a, b in filtered_indices]                                      # Subtract the edge indices of the compound from the raw edge indices of the placed hydrogen atoms.
    print("Updated edge indices for initial H atoms:", edge_indices_H)

    plot_crystal(H_init_xyz, edge_indices_H)                                                                            # Plot the placed hydrogen alone.

    node_features_init_H, edge_features_init_H = node_edge_features(H_init_xyz, edge_indices_H,
                                                                   oxidation_states, num_atoms, 1)                 # Extract the node and edge features of the hydrogen alone.

    print("Node features initial H alone: ", node_features_init_H)
    print("Edge features initial H alone: ", edge_features_init_H)

    tot_mass_H, tot_charge_H = mass_and_charge(H_init_xyz, oxidation_states, num_atoms, 1)                         # Extract the total mass and charge of the hydrogen alone.

    print("Total mass H alone: ", tot_mass_H)
    print("Total charge H alone: ", tot_charge_H)

    ####################### Set TESTING system input features #######################

    node_features_test = [node_features_init_comb, node_features_crystal, node_features_init_H]                         # Collect the node features of the crystal alone, combined system and hydrogen alone.
    edge_features_test = [edge_features_init_comb, edge_features_crystal, edge_features_init_H]                         # Collect the edge features of the crystal alone, combined system and hydrogen alone.
    edge_indices = [edge_indices_comb, edge_indices_crystal, edge_indices_H]                                            # Collect the edge indices of the crystal alone, combined system and hydrogen alone.

    print("TEST Node features: ", node_features_test)
    print("TEST Edge features: ", edge_features_test)
    print("TEST Edge indices: ", edge_indices)

    diffusion_inputs = [tot_mass_crystal, extracted_input_features[0]]                                                  # Stores diffusion model system features (Mass and energy abpve hull).
    diffusion_input_xyz = combined_xyz                                                                                  # Stores the combined initial (Non optimised) coordinates.

    print("TEST Diffusion inputs: ", diffusion_inputs)
    print("Diffusion input xyz: ", diffusion_input_xyz)

    energy_inputs_test = [net_spin_crystal, tot_mass_crystal, volume_crystal, tot_charge_crystal,
                          extracted_input_features[0]]                                                                  # Stores energy model system features.

    print("TEST Energy inputs: ", energy_inputs_test)

    if test_train == '1':

        # optimised_xyz_raw = (optimise_geometry(combined_xyz, num_atoms, name))
        optimised_xyz_raw = ['O -5.585093 -5.803385 0.648110', 'O -3.368041 -3.753687 0.648110', 'O 3.938545 3.227738 0.403422', 'O 6.155597 5.277436 0.403422', 'O -5.340491 3.472425 0.403422', 'O -3.123438 5.522123 0.403422', 'O 1.708155 -4.280650 0.602900', 'O 1.327943 -1.734725 0.591624', 'O 4.318834 0.684741 0.525766', 'O -0.508820 -6.327421 0.713968', 'O -0.264217 2.948390 0.469281', 'O 1.952758 4.995160 0.358212', 'O 6.424779 -1.478955 0.525766', 'O -2.854256 -1.234267 0.525766', 'O -4.960201 0.929428 0.525766', 'O -0.778080 0.426042 0.480556', 'O -3.044148 -6.327910 0.268396', 'O 6.479490 2.703213 0.023708', 'O -2.799545 2.947901 0.023708', 'O 3.859721 -1.868834 0.146052', 'O 1.786434 0.795281 0.077265', 'O -0.833335 -3.776766 0.199609', 'O -0.588732 5.499044 -0.045079', 'O -5.419314 -1.624147 0.146052', 'Cr -4.259096 -5.013764 -0.623229', 'Cr 5.264542 4.017359 -0.867916', 'Cr -4.014493 4.262047 -0.867916', 'Cr 2.400008 -2.868799 -0.684401', 'Cr 3.244956 1.750051 -0.806745', 'Cr 0.380422 -5.136107 -0.623229', 'Cr 0.625024 4.139703 -0.867916', 'Cr 5.142241 -0.620546 -0.745573', 'Cr -4.136795 -0.375858 -0.745573', 'Cr 0.502723 -0.498202 -0.745573', 'Cr -2.239510 -2.746455 -0.684401', 'Cr -1.394562 1.872394 -0.806745', 'H 1.227979 -3.026039 2.180860', 'H -0.967164 -3.785065 4.124145']
        print('Raw optimised xyz: ', optimised_xyz_raw)

        optimised_xyz, optimised_centre = centre_coords(optimised_xyz_raw, num_atoms)                                     # Centre all atoms around the crystal's centroid.

        print("Centered optimised XYZ: ", optimised_xyz)
        print("Optimised centre: ", optimised_centre)

        plot_crystal(optimised_xyz, edge_indices_comb)                                                                  # Plot optimised system.

        # energy_comb = calculate_energy(optimised_xyz)
        energy_comb = -441836.170474524
        print('Combined energy: ', energy_comb)

        #################### Optimised combined Code #################

        node_features_opt_comb, edge_features_opt_comb = (
            node_edge_features(optimised_xyz, edge_indices_comb, oxidation_states, num_atoms, 1))                  # Extract node and edge features of the optimised system.

        print("Node features optimised combined: ", node_features_opt_comb)
        print("Edge features optimised combined: ", edge_features_opt_comb)

        volume_opt_comb = compute_volume(optimised_xyz)                                                                 # Compute the volume encapsulated by the optimised system.
        print("Volume optimised combined: ", volume_opt_comb)

        #################### Optimised H alone Code ##################

        H_opt_xyz = optimised_xyz[num_atoms:]                                                                            # Extract the optimised hydrogen positions.

        print("H Opt XYZ: ", H_opt_xyz)

        node_features_opt_H, edge_features_opt_H = node_edge_features(H_opt_xyz, edge_indices_H,
                                                                        oxidation_states, num_atoms, 1)            # Extract the node and edge features of the optimised hydrogen.

        print("Node features optimised H alone: ", node_features_opt_H)
        print("Edge features optimised H alone: ", edge_features_opt_H)

        # volume_opt_H = compute_volume(H_opt_xyz)
        volume_opt_H = 0
        print("Volume optimised H alone: ", volume_opt_H)

        # energy_crystal = calculate_energy(centered_xyz)
        energy_crystal = -440707.436409493
        print('Crystal alone energy: ', energy_crystal)

        energy_H = calculate_energy(H_opt_xyz)
        print('Energy H alone: ', energy_H)

        adsorption_energy = energy_comb - (energy_H + energy_crystal)                                                   # Calculate adsorption energy.
        print("Calculated adsorption energy: ", adsorption_energy)

        ########## Assemble TRAINING Node & Edge features + Indices ###########

        node_features = [node_features_init_comb, node_features_opt_comb, node_features_crystal,
                         node_features_init_H, node_features_opt_H]                                                     # Collect all node features of initial and optimised combined and hydrogen alone systems and crystal alone system.
        edge_features = [edge_features_init_comb, edge_features_opt_comb, edge_features_crystal,
                         edge_features_init_H, edge_features_opt_H]                                                     # Collect all edge features of initial and optimised combined and hydrogen alone systems and crystal alone system.

        print("TRAIN Node features: ", node_features)
        print("TRAIN Edge features: ", edge_features)
        print("TRAIN Edge indices: ", edge_indices)

        ############### Set TRAINING system INPUT features ############

        energy_opt_input_feats = [net_spin_crystal, tot_mass_comb, volume_opt_comb, tot_charge_comb,
                                  extracted_input_features[0]]                                                          # Energy model system features of the optimised combined system.
        energy_H_input_feats = [net_spin_crystal, tot_mass_H, volume_opt_H, tot_charge_H,
                                extracted_input_features[0]]                                                            # Energy model system features of the hydrogen alone.
        energy_inputs_train = [energy_opt_input_feats, energy_inputs_test, energy_H_input_feats]                        # Collect all energy model system features.

        temperature_inputs = energy_opt_input_feats                                                                     # Temperature model system inputs are the same as the energy model's.

        print("Energy TOTAL inputs: ", energy_inputs_train)
        print("Temperature TOTAL inputs: ", temperature_inputs)

        ############### Set TRAINING system OUTPUT features ##############

        diffusion_output_xyz = optimised_xyz                                                                            # Set the diffusion model target as the optimised coordinates.
        energy_outputs = [energy_comb, energy_crystal, energy_H]                                                        # Set the energy model targets as the energies of the crystal, combined and hydrogen systems.
        temperature_outputs = extracted_output_features                                                                 # Set the temperature model targets as the user-input adsorption and desorption temperatures.

        uncertain_features.append(H_opt_xyz)                                                                            # Add optimised hydrogen coordinates to uncertain features (These may be used later).
        uncertain_features.append(adsorption_energy)                                                                    # Add adsorption energy to uncertain features (These may be used later).

        print("Diffusion output xyz: ", diffusion_output_xyz)
        print("Energy TOTAL outputs: ", energy_outputs)
        print("Temperature TOTAL outputs: ", temperature_outputs)
        print("Uncertain TOTAL features: ", uncertain_features)

        return [node_features, edge_features, edge_indices, diffusion_inputs, diffusion_input_xyz, diffusion_output_xyz,
                energy_inputs_train, energy_outputs, temperature_inputs, temperature_outputs, uncertain_features,
                num_atoms, num_H]

    elif test_train == '2':

        return [node_features_test, edge_features_test, edge_indices, diffusion_inputs, diffusion_input_xyz,
                energy_inputs_test, uncertain_features, num_atoms, num_H, oxidation_states]

    else:
        return None