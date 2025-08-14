from collections import Counter

from Materials_Extractor import extract_hydrogen, extract_compound, compute_bonds, reassociate_coordinates
from Plotting import plot_crystal, plot_external_surfaces
from Mol_Geometry import surface_finder, compute_volume, centre_coords, place_hydrogen
from External_Saving import save_edges_to_csv, load_existing_edges, save_original_xyz
from DFT import calculate_energy, optimise_geometry
from Compound_Properties import get_spin, node_edge_features, mass_and_charge


def data_creator(hydrogen, compound_ID, name, test_train):

    hydrogen_bond = extract_hydrogen(f"{hydrogen}.poscar")                                                              # Extract hydrogen and compound bond lengths and coordinates.
    print("Hydrogen Bond: ", hydrogen_bond)

    compound_xyz, oxidation_states, extracted_input_features, extracted_output_features, uncertain_features = (
        extract_compound(compound_ID, name))                                                                            # Extract coordinates, oxidation states, mechanical features, temperatures and features that could be used.

    print("Uncertain TEST features: ", uncertain_features)

    ################## Crystal initial alone code ##################

    edge_indices_crystal = compute_bonds(compound_xyz)                                                                  # Computer the crystal edge indices.

    save = input("Save crystal edge indices to csv file? y/n: ")

    if save == 'y':
        save_edges_to_csv(edge_indices_crystal, name)                                                                   # Save the crystal edge indices.
    else:
        print("Crystal edge indices not saved")

    print("Compound XYZ: ", compound_xyz)
    print("Edge Indices of the initial crystal alone: ", edge_indices_crystal)

    plot_crystal(compound_xyz, edge_indices_crystal)                                                                    # Plot the crystal and its bonds alone.

    num_atoms = len(compound_xyz)                                                                                       # Find the number of atoms in the crystal.
    print(f"The first {num_atoms} atoms are fixed.")

    centered_xyz, center = centre_coords(compound_xyz, num_atoms)                                                       # Centre all atoms around the crystal's centroid.
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

    surface_points = surface_finder(centered_xyz)                                                                       # Extract external faces of the crystal alone.
    print("Surface Points: ", surface_points)
    plot_external_surfaces(surface_points)                                                                              # Plot all the external faces of the compound alone.

    my_list_str = str(surface_points)
    count = my_list_str.count('array')
    print("Number of unique Surfaces: ", count)

    combined_xyz = place_hydrogen(centered_xyz, surface_points, hydrogen_bond, 10, 8)                      # Place hydrogen molecules at offset positions around the surfaces of the crystal.
    num_H = len(centered_xyz) - num_atoms                                                                               # Calculate the number of placed hydrogen atoms.

    print("Combined XYZ: ", combined_xyz)
    print("Number of placed hydrogen atoms: ", num_H)

    save_original_xyz(combined_xyz, 'Optimised Coordinates', f"{name}_original_coords.xyz")
    print('Combined XYZ saved!')

    #################### Combined initial Code #####################

    print("Open the Building_Connections python script, and call the build_connections function using "
          "Combined XYZ from above as the argument.\n\n"
          "Run and re-run the script until all new connections have been made.")

    while True:
        connection_complete = input("Type the word 'done' when complete: ")

        if connection_complete == 'done':
            break
        else:
            print("Keep adding connections.")

    edge_indices_comb = load_existing_edges(name)                                                                       # Load the combined system edge indices.
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

        energy_crystal = calculate_energy(centered_xyz)
        print('Crystal alone energy: ', energy_crystal)

        # optimised_xyz_raw = optimise_geometry(combined_xyz, num_atoms, name)

        optimised_xyz_raw = ['Ti 1.066663 0.706563 -0.316090', 'Ti 1.654235 -0.932366 2.199730', 'Ti 1.286793 2.822634 1.886613', 'Ti -1.415317 -1.791616 -4.538165', 'Ti 0.204650 -0.850600 4.891627', 'Ti -1.785263 1.959889 -4.853389', 'Ti -0.165142 2.905426 4.576369', 'Ti 2.294464 -1.485390 -5.208907', 'Ti 3.915401 -0.540440 4.223156', 'Ti 1.927378 2.267050 -5.519114', 'Ti 3.547340 3.208957 3.909372', 'Ti -1.195976 0.318541 -2.338017', 'Ti 2.515458 0.626614 -3.006263', 'O -0.860688 0.513936 -0.385142', 'O 2.851279 0.820588 -1.053048', 'O 1.401539 0.902481 1.638787', 'O 1.318308 -1.118358 0.246236', 'O 0.951993 2.622343 -0.067202', 'O -0.129346 -1.045092 2.937018', 'O -0.499448 2.709453 2.625163', 'O 3.577734 -0.737073 2.269462', 'O 3.211871 3.015649 1.957178', 'O 1.989606 -0.736198 4.153234', 'O 1.621932 3.017275 3.839335', 'O -1.080264 -1.597144 -2.585347', 'O -1.448552 2.154927 -2.898400', 'O 2.629348 -1.290068 -3.253476', 'O 2.262842 2.462957 -3.567049', 'O -1.531069 0.123482 -4.291365', 'O 2.180341 0.431308 -4.959968', 'O 0.730295 0.512577 -2.268447', 'O 1.768262 -2.848678 1.950497', 'O 1.033908 4.662513 1.328169', 'O -3.340707 -1.986257 -4.609772', 'O -1.163242 -3.628042 -3.980330', 'O -1.751521 -1.986380 -6.493745', 'O 0.369725 -1.678461 -5.277990', 'O -1.720862 -1.042786 4.820711', 'O 0.456027 -2.683583 5.450626', 'O 0.539181 -0.653388 6.843962', 'O 0.089157 1.067827 5.138666', 'O -3.708161 1.765309 -4.922185', 'O -2.119430 1.764020 -6.805678', 'O -1.897671 3.875367 -4.603549', 'O 0.002000 2.072867 -5.590689', 'O -2.087683 2.709258 4.507792', 'O 0.172249 3.098261 6.532093', 'O -0.277775 4.817957 4.828247', 'O 2.547356 -3.320950 -4.648160', 'O 1.959334 -1.679782 -7.161579', 'O 4.079920 -1.370526 -5.945851', 'O 4.168068 -2.375937 4.782292', 'O 4.249405 -0.345583 6.175894', 'O 3.799186 1.374895 4.470379', 'O 5.700548 -0.426874 3.484346', 'O 1.590514 2.071863 -7.473929', 'O 1.813381 4.183182 -5.271949', 'O 3.712382 2.380906 -6.259142', 'O 3.882429 3.405534 5.864066', 'O 3.433554 5.125795 4.159341', 'O 5.332956 3.322438 3.172810', 'O -2.980372 0.205106 -1.599415', 'O 4.440447 0.820533 -2.936018', 'H 4.507543 4.185502 7.037716', 'H 5.865098 4.414096 7.425192', 'H -4.429507 6.250309 -7.438794', 'H -5.062725 5.597427 -8.265052', 'H -5.570688 -4.712083 7.290730', 'H -4.709005 -3.846338 7.000075', 'H -5.351273 -5.782764 -7.779804', 'H -4.189465 -6.091884 -7.974843', 'H -4.694023 4.507373 7.524543', 'H -5.672278 4.113412 6.924116', 'H 4.415844 6.110100 -7.372132', 'H 5.113053 5.699831 -8.308763', 'H 4.430129 -5.507336 -7.506592', 'H 3.958387 -4.577619 -6.621464', 'H 5.855229 -5.427091 7.851208', 'H 6.032976 -4.221755 7.938679', 'H -4.388738 1.001256 -9.351852', 'H -5.415380 0.852603 -8.729091', 'H -5.952635 -0.833786 -9.440679', 'H -7.168918 -1.006362 -9.365212', 'H -3.988941 5.426417 -1.469406', 'H -5.074817 4.931276 -1.310224', 'H -5.693746 6.008687 1.308967', 'H -6.672535 5.312299 1.517095', 'H -0.033032 5.989750 -0.818134', 'H -0.891961 6.718324 -1.248875', 'H 1.162325 8.298052 2.051957', 'H 0.044676 8.019725 1.684204', 'H 4.848535 5.419714 -1.244781', 'H 3.832505 4.480951 -1.476762', 'H 5.962663 6.221776 1.275218', 'H 6.476903 5.127123 1.420563', 'H -7.192582 -0.010927 -1.712346', 'H -7.104255 -1.168679 -2.045948', 'H -8.568835 0.938947 1.285471', 'H -7.543669 0.265079 1.026722', 'H -4.587112 -4.636073 -1.462410', 'H -5.798889 -4.789852 -1.716559', 'H -5.220899 -6.692873 1.799940', 'H -5.968795 -5.779982 1.511613', 'H -1.004329 -6.507553 1.939618', 'H -0.023703 -7.164333 1.724964', 'H 0.327981 -7.360827 -1.032227', 'H 0.900072 -8.459948 -1.248445', 'H -6.419216 0.823178 7.196441', 'H -5.310778 0.578553 6.395712', 'H -6.921723 -0.498942 8.768896', 'H -6.070002 -1.081412 9.408511', 'H -0.480600 6.252393 8.545814', 'H -1.261752 5.407720 8.130107', 'H 0.678992 5.411513 10.641600', 'H 1.065799 6.074850 9.659829', 'H 1.476970 3.161054 -5.571194', 'H 0.340795 2.532143 -6.551209', 'H -0.654436 5.791105 -8.861472', 'H -0.811609 7.037348 -9.008932', 'H 6.246805 0.021175 0.849246', 'H 6.777930 1.011327 1.218100', 'H 8.138104 -1.168275 -2.094253', 'H 8.262382 -0.013335 -1.756617', 'H 4.946186 -5.541495 1.638693', 'H 4.063779 -4.590034 1.474371', 'H 5.727083 -5.963829 -1.492255', 'H 6.653207 -5.242497 -1.794580', 'H 5.199058 -0.631603 -8.489482', 'H 6.306637 -1.045915 -8.153909', 'H 5.188878 0.744844 -10.313117', 'H 6.304589 0.994644 -9.815931', 'H -0.576896 -1.371957 -10.333750', 'H -0.979475 -0.237377 -10.466150', 'H 0.272623 0.998601 -11.460316', 'H 1.326415 0.643167 -11.967260', 'H 0.815899 -6.306746 -7.006629', 'H 0.557001 -4.867392 -6.122547', 'H -0.462399 -6.814233 -8.642100', 'H -1.046707 -6.002962 -9.326721', 'H 4.833776 -0.537767 4.690905', 'H 4.681810 -0.084753 5.113321', 'H 6.957253 0.814544 8.652561', 'H 6.119703 0.655988 9.551687', 'H -0.551691 -0.556531 10.901155', 'H -1.146651 -1.102494 10.054845', 'H 1.127758 1.080277 12.253483', 'H 0.615221 0.580910 11.290583', 'H 0.434896 -6.178804 8.321833', 'H 1.332561 -5.388476 8.148125', 'H -1.041924 -5.490572 10.675220', 'H -0.720280 -5.972574 9.610902']

        print('Raw optimised xyz: ', optimised_xyz_raw)

        optimised_xyz, optimised_centre = centre_coords(optimised_xyz_raw, num_atoms)                                     # Centre all atoms around the crystal's centroid.

        print("Centered optimised XYZ: ", optimised_xyz)
        print("Optimised centre: ", optimised_centre)

        plot_crystal(optimised_xyz, edge_indices_comb)                                                                  # Plot optimised system.

        #################### Optimised combined Code #################

        node_features_opt_comb, edge_features_opt_comb = (
            node_edge_features(optimised_xyz, edge_indices_comb, oxidation_states, num_atoms, 1))                  # Extract node and edge features of the optimised system.

        print("Node features optimised combined: ", node_features_opt_comb)
        print("Edge features optimised combined: ", edge_features_opt_comb)

        volume_opt_comb = compute_volume(optimised_xyz)                                                                 # Compute the volume encapsulated by the optimised system.
        print("Volume optimised combined: ", volume_opt_comb)

        energy_comb = calculate_energy(optimised_xyz)
        print('Combined energy: ', energy_comb)

        #################### Optimised H alone Code ##################

        H_opt_xyz = combined_xyz[num_atoms:]                                                                            # Extract the optimised hydrogen positions.

        print("H Opt XYZ: ", H_opt_xyz)

        node_features_opt_H, edge_features_opt_H = node_edge_features(H_opt_xyz, edge_indices_H,
                                                                        oxidation_states, num_atoms, 1)            # Extract the node and edge features of the optimised hydrogen.

        print("Node features optimised H alone: ", node_features_opt_H)
        print("Edge features optimised H alone: ", edge_features_opt_H)

        volume_opt_H = compute_volume(H_opt_xyz)                                                                        # Compute the volume encapsulated by the optimised hydrogen.
        print("Volume optimised H alone: ", volume_opt_H)

        energy_H = calculate_energy(H_opt_xyz)

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

    elif test_train == '0':

        return [node_features_test, edge_features_test, edge_indices, diffusion_inputs, diffusion_input_xyz,
                energy_inputs_test, uncertain_features, num_atoms, num_H]
