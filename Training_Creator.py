from collections import Counter

from Materials_Extractor import extract_hydrogen, extract_compound, compute_bonds, reassociate_coordinates
from Plotting import plot_crystal, plot_external_surfaces
from Mol_Geometry import surface_finder, compute_volume, centre_coords, place_hydrogen
from External_Saving import save_edges_to_csv, load_existing_edges, load_optimised_coords
# from DFT import setup_compound, optimiser, calculate_energy
from Compound_Properties import get_spin, node_edge_features, mass_and_charge


def data_creator(hydrogen, compound_ID, name, test_train):

    hydrogen_bond = extract_hydrogen(f"{hydrogen}.poscar")                                                              # Extract hydrogen and compound bond lengths and coordinates.
    print("Hydrogen Bond: ", hydrogen_bond)

    compound_xyz, oxidation_states, extracted_input_features, extracted_output_features, uncertain_features = (
        extract_compound(compound_ID, name))                                                                            # Extract coordinates, oxidation states, mechanical features, temperatures and features that could be used.

    # compound_xyz = ['Li 0.0000000000 0.0000000000 0.0000000000', 'Li 0.0000000000 -2.0086193919 -2.0086193919',
    #                 'Li 2.0086193919 -2.0086193919 0.0000000000', 'Li 2.0086193919 0.0000000000 -2.0086193919',
    #                 'H 0.0000000000 -2.0086193919 0.0000000000', 'H 0.0000000000 0.0000000000 -2.0086193919',
    #                 'H 2.0086193919 0.0000000000 0.0000000000', 'H 2.0086193919 -2.0086193919 -2.0086193919']

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

    combined_xyz = place_hydrogen(centered_xyz, surface_points, hydrogen_bond, 6, 5)                      # Place hydrogen molecules at offset positions around the surfaces of the crystal.
    num_H = len(centered_xyz) - num_atoms                                                                               # Calculate the number of placed hydrogen atoms.

    print("Combined XYZ: ", combined_xyz)
    print("Number of placed hydrogen atoms: ", num_H)

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

        while True:
            crystal_energy_input = input("Run 'calculate energy' in the DFT script as follows: "
                                         "'energy_crystal, _ = calculate_energy(centered_xyz)', "
                                         "where 'centered_xyz' is printed at the start of data_creator. "
                                         "When this is done, type 'done': ")

            if crystal_energy_input.lower() == 'done':
                try:
                    energy_crystal = float(input("Input the energy of the crystal alone: "))
                    break
                except ValueError:
                    print("Invalid input! Please enter a numeric value for the energy.")
            else:
                print("Invalid response. Please type 'done' when ready.")

        ###################### Optimisation ######################

        while True:
            opt_input = input("Run the optimiser (Second block of commented code) in the DFT script as follows"
                              "using the combined_xyz printed above. When this is done, type 'done': ")

            if opt_input.lower() == 'done':
                file = f'Optimised Coordinates/{name}_optimised_coords'
                raw_optimised_xyz = load_optimised_coords(file)
                break
            else:
                print("Invalid response. Please type 'done' when ready.")

        print("Uncentered optimised XYZ: ", raw_optimised_xyz)

        reassociate_xyz = reassociate_coordinates(raw_optimised_xyz, combined_xyz)                                      # Re-associate the optimisated cooridnates with the atoms types.

        optimised_xyz, optimised_centre = centre_coords(reassociate_xyz, num_atoms)                                     # Centre all atoms around the crystal's centroid.

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

        while True:
            combined_energy_input = input("Run 'calculate energy' in the DFT script as follows: "
                                         "'energy_comb, _ = calculate_energy(optimised_xyz)', "
                                         "where 'optimised_xyz' is printed at the start of data_creator. "
                                         "When this is done, type 'done': ")

            if combined_energy_input.lower() == 'done':  # Case insensitive check
                try:
                    energy_comb = float(input("Input the energy of the combined optimimised system: "))
                    break  # Exit loop after getting a valid energy input
                except ValueError:
                    print("Invalid input! Please enter a numeric value for the energy.")
            else:
                print("Invalid response. Please type 'done' when ready.")

        #################### Optimised H alone Code ##################

        H_opt_xyz = combined_xyz[num_atoms:]                                                                            # Extract the optimised hydrogen positions.

        print("H Opt XYZ: ", H_opt_xyz)

        node_features_opt_H, edge_features_opt_H = node_edge_features(H_opt_xyz, edge_indices_H,
                                                                        oxidation_states, num_atoms, 1)            # Extract the node and edge features of the optimised hydrogen.

        print("Node features optimised H alone: ", node_features_opt_H)
        print("Edge features optimised H alone: ", edge_features_opt_H)

        volume_opt_H = compute_volume(H_opt_xyz)                                                                        # Compute the volume encapsulated by the optimised hydrogen.

        print("Volume optimised H alone: ", volume_opt_H)

        while True:
            H_energy_input = input("Run 'calculate energy' in the DFT script as follows: "
                                         "'energy_H, _ = calculate_energy(H_opt_xyz)', "
                                         "where 'H_opt_xyz' is printed just above. "
                                         "When this is done, type 'done': ")

            if H_energy_input.lower() == 'done':  # Case insensitive check
                try:
                    energy_H = float(input("Input the energy of the optimised H alone: "))
                    break  # Exit loop after getting a valid energy input
                except ValueError:
                    print("Invalid input! Please enter a numeric value for the energy.")
            else:
                print("Invalid response. Please type 'done' when ready.")

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
