from collections import Counter

from Materials_Extractor import extract_hydrogen, extract_compound, compute_bonds, reassociate_coordinates
from Plotting import plot_crystal, plot_external_surfaces
from Mol_Geometry import surface_finder, compute_volume, centre_coords, place_hydrogen
from Build_Connections import save_edges_to_csv, load_existing_edges
from DFT import setup_compound, optimiser, calculate_energy
from Compound_Properties import get_spin, node_edge_features, mass_and_charge


def data_creator(hydrogen, compound_ID, name, test_train):

    hydrogen_bond = extract_hydrogen(f"{hydrogen}.poscar")                                                              # Extract hydrogen and compound bond lengths and coordinates
    print("Hydrogen Bond: ", hydrogen_bond)

    compound_xyz, oxidation_states, extracted_input_features, extracted_output_features, uncertain_features = extract_compound(compound_ID, name)

    compound_xyz = ['Li 0.0000000000 0.0000000000 0.0000000000', 'Li 0.0000000000 -2.0086193919 -2.0086193919',
                    'Li 2.0086193919 -2.0086193919 0.0000000000', 'Li 2.0086193919 0.0000000000 -2.0086193919',
                    'H 0.0000000000 -2.0086193919 0.0000000000', 'H 0.0000000000 0.0000000000 -2.0086193919',
                    'H 2.0086193919 0.0000000000 0.0000000000', 'H 2.0086193919 -2.0086193919 -2.0086193919']

    print("Uncertain TEST features: ", uncertain_features)

    ################## Crystal initial alone code ##################

    edge_indices_crystal = compute_bonds(compound_xyz)

    save_edges_to_csv(edge_indices_crystal)

    print("Compound XYZ: ", compound_xyz)
    print("Edge Indices of the initial crystal alone: ", edge_indices_crystal)

    plot_crystal(compound_xyz, edge_indices_crystal)

    num_atoms = len(compound_xyz)
    print(f"The first {num_atoms} atoms are fixed.")

    centered_xyz, center = centre_coords(compound_xyz, num_atoms)
    print("Crystal center: ", center)
    print("Centered xyz: ", centered_xyz)

    node_features_crystal, edge_features_crystal = node_edge_features(centered_xyz, edge_indices_crystal, oxidation_states, num_atoms, 0)
    tot_mass_crystal, tot_charge_crystal = mass_and_charge(centered_xyz, oxidation_states, num_atoms, 0)

    print("Total mass of the crystal alone: ", tot_mass_crystal)
    print("Total charge of the crystal alone: ", tot_charge_crystal)
    print("Node Features of the crystal alone: ", node_features_crystal)
    print("Edge Features of the crystal alone: ", edge_features_crystal)

    chemicals = [line.split()[0] for line in centered_xyz]
    element_count = [(atom, str(count)) for atom, count in Counter(chemicals).items()]
    net_spin_crystal = get_spin(element_count)

    print("Net Spin of the crystal alone: ", net_spin_crystal)

    volume_crystal = compute_volume(centered_xyz)

    surface_points = surface_finder(centered_xyz)
    print("Surface Points: ", surface_points)
    plot_external_surfaces(surface_points)

    my_list_str = str(surface_points)
    count = my_list_str.count('array')
    print("Number of unique Surfaces: ", count)

    energy_crystal, _ = calculate_energy(centered_xyz)
    print("Crystal alone energy: ", energy_crystal)

    combined_xyz = place_hydrogen(centered_xyz, surface_points, hydrogen_bond, 2.75, 1.5)
    num_H = len(centered_xyz) - num_atoms

    print("Combined XYZ: ", combined_xyz)
    print("Number of placed hydrogen atoms: ", num_H)

    #################### Combined initial Code #####################

    print("Open the Building_Connections python script, and call the build_connections function using"
          "Combined XYZ from above as the argument.\n\n"
          "Run and re-run the script until all new connections have been made.")

    while True:
        connection_complete = input("Type the word 'done' when complete: ")

        if connection_complete == 'done':
            break
        else:
            print("Keep adding connections.")

    edge_indices_comb = load_existing_edges()
    print("Edge indices combined: ", edge_indices_comb)

    plot_crystal(combined_xyz, edge_indices_comb)

    node_features_init_comb, edge_features_init_comb = node_edge_features(combined_xyz, edge_indices_comb,
                                                                   oxidation_states, num_atoms, 0)

    print("Node features initial combined: ", node_features_init_comb)
    print("Edge features initial combined: ", edge_features_init_comb)

    tot_mass_comb, tot_charge_comb = mass_and_charge(combined_xyz, oxidation_states, num_atoms, 0)

    print("Total mass combined: ", tot_mass_comb)
    print("Total charge combined: ", tot_charge_comb)

    ##################### H initial alone Code #####################

    H_init_xyz = combined_xyz[num_atoms:]

    print("H init XYZ: ", H_init_xyz)

    filtered_indices = [pair for pair in edge_indices_comb if pair[0] >= num_atoms and pair[1] >= num_atoms]
    print("Filtered Indices: ", filtered_indices)
    edge_indices_H = [[a - num_atoms, b - num_atoms] for a, b in filtered_indices]
    print("Updated edge indices for initial H atoms:", edge_indices_H)

    plot_crystal(H_init_xyz, edge_indices_H)

    node_features_init_H, edge_features_init_H = node_edge_features(H_init_xyz, edge_indices_H,
                                                                   oxidation_states, num_atoms, 1)

    print("Node features initial H alone: ", node_features_init_H)
    print("Edge features initial H alone: ", edge_features_init_H)

    tot_mass_H, tot_charge_H = mass_and_charge(H_init_xyz, oxidation_states, num_atoms, 1)

    print("Total mass H alone: ", tot_mass_H)
    print("Total charge H alone: ", tot_charge_H)

    ##################### Set TESTING system input features #######################

    node_features_test = [node_features_init_comb, node_features_crystal, node_features_init_H]
    edge_features_test = [edge_features_init_comb, edge_features_crystal, edge_features_init_H]
    edge_indices = [edge_indices_comb, edge_indices_crystal, edge_indices_H]

    print("TEST Node features: ", node_features_test)
    print("TEST Edge features: ", edge_features_test)
    print("TEST Edge indices: ", edge_indices)

    diffusion_inputs = [tot_mass_crystal, extracted_input_features[0]]
    diffusion_input_xyz = combined_xyz

    print("TEST Diffusion inputs: ", diffusion_inputs)
    print("Diffusion input xyz: ", diffusion_input_xyz)

    energy_inputs_test = [net_spin_crystal, tot_mass_crystal, volume_crystal, tot_charge_crystal, extracted_input_features[0]]

    print("TEST Energy inputs: ", energy_inputs_test)

    if test_train == '1':

        ###################### Optimisation ######################

        # mol, mf_grad_scan, initial_coordinates = setup_compound(combined_xyz)
        # raw_optimised_xyz = optimiser(mol, mf_grad_scan, initial_coordinates, num_atoms)

        raw_optimised_xyz = [[-1.05769119,  0.88377819,  1.06264705],
                            [-1.04843132, -1.02335637, -1.08641292],
                            [ 0.96998176, -1.16082231,  0.9254738 ],
                            [ 0.93416292,  0.89730277, -0.9453166 ],
                            [-0.99042321, -0.91866055,  0.90859759],
                            [-0.87803216,  0.81048418, -0.90940785],
                            [ 0.87659563,  0.73869425,  0.85793386],
                            [ 0.85446915, -0.95907353, -0.93180427],
                            [-1.00970367, -3.28547723, -1.0586689 ],
                            [-0.58452318, -2.81170912, -0.63354481],
                            [ 0.74810459, -3.22450592,  0.73296617],
                            [ 1.25913186, -3.45033845,  1.24874249],
                            [ 0.44638802, -0.67925204,  2.61205223],
                            [ 0.85063651, -1.08650749,  3.12395833],
                            [-1.38339672,  1.01656587,  3.49198715],
                            [-0.86354154,  0.53230196,  3.21887648],
                            [ 3.2122623,  -1.05792248,  0.87902753],
                            [ 2.72327126, -0.62635142,  0.47691763],
                            [ 3.02545024,  0.75158712, -0.77765075],
                            [ 3.32710172,  1.27399649, -1.24138633],
                            [-3.28920869, -0.92775707, -0.95494387],
                            [-2.7879272,  -0.54100774, -0.52238075],
                            [-3.15825745,  0.64161726,  0.91741076],
                            [-3.43190945,  1.09394603,  1.46315057],
                            [-0.95036279,  3.15603602,  1.03893172],
                            [-0.54582345,  2.66429332,  0.61271235],
                            [ 0.76925715,  3.02640577, -0.74124431],
                            [ 1.25959762,  3.33100477, -1.23708364],
                            [-0.49861112, -0.45448672, -2.78476902],
                            [-0.90644196, -0.83927756, -3.30807953],
                            [ 1.31978262,  1.37821083, -3.31440276],
                            [ 0.80809173,  0.85028117, -3.12428943]]

        print("Uncentered optimised XYZ: ", raw_optimised_xyz)

        reassociate_xyz = reassociate_coordinates(raw_optimised_xyz, combined_xyz)

        optimised_xyz, optimised_centre = centre_coords(reassociate_xyz, num_atoms)

        print("Centered optimised XYZ: ", optimised_xyz)
        print("Optimised centre: ", optimised_centre)

        plot_crystal(optimised_xyz, edge_indices_comb)

        #################### Optimised combined Code #################

        node_features_opt_comb, edge_features_opt_comb = node_edge_features(optimised_xyz, edge_indices_comb, oxidation_states, num_atoms, 1)

        print("Node features optimised combined: ", node_features_opt_comb)
        print("Edge features optimised combined: ", edge_features_opt_comb)

        volume_opt_comb = compute_volume(optimised_xyz)

        print("Volume optimised combined: ", volume_opt_comb)

        energy_comb, _ = calculate_energy(optimised_xyz)
        print("Optimised combined energy: ", energy_comb)

        #################### Optimised H alone Code ##################

        H_opt_xyz = combined_xyz[num_atoms:]

        print("H Opt XYZ: ", H_opt_xyz)

        node_features_opt_H, edge_features_opt_H = node_edge_features(H_opt_xyz, edge_indices_H,
                                                                        oxidation_states, num_atoms, 1)

        print("Node features optimised H alone: ", node_features_opt_H)
        print("Edge features optimised H alone: ", edge_features_opt_H)

        volume_opt_H = compute_volume(H_opt_xyz)

        print("Volume optimised H alone: ", volume_opt_H)

        energy_H, _ = calculate_energy(H_opt_xyz)
        print("Optimised H alone energy: ", energy_H)

        adsorption_energy = energy_comb - (energy_H + energy_crystal)

        print("Calculated adsorption energy: ", adsorption_energy)

        ########## Assemble TRAINING Node & Edge features + Indices ###########

        node_features = [node_features_init_comb, node_features_opt_comb, node_features_crystal, node_features_init_H, node_features_opt_H]
        edge_features = [edge_features_init_comb, edge_features_opt_comb, edge_features_crystal, edge_features_init_H, edge_features_opt_H]

        print("TRAIN Node features: ", node_features)
        print("TRAIN Edge features: ", edge_features)
        print("TRAIN Edge indices: ", edge_indices)

        ############### Set TRAINING system INPUT features ############

        energy_opt_input_feats = [net_spin_crystal, tot_mass_comb, volume_opt_comb, tot_charge_comb, extracted_input_features[0]]
        energy_H_input_feats = [net_spin_crystal, tot_mass_H, volume_opt_H, tot_charge_H, extracted_input_features[0]]
        energy_inputs_train = [energy_opt_input_feats, energy_inputs_test, energy_H_input_feats]

        temperature_inputs = energy_opt_input_feats

        print("Energy TOTAL inputs: ", energy_inputs_train)
        print("Temperature TOTAL inputs: ", temperature_inputs)

        ############### Set TRAINING system OUTPUT features ##############

        diffusion_output_xyz = optimised_xyz
        energy_outputs = [energy_comb, energy_crystal, energy_H]
        temperature_outputs = extracted_output_features

        uncertain_features.append(H_opt_xyz)
        uncertain_features.append(adsorption_energy)

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
