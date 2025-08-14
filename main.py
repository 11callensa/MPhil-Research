import csv

from Compound_Database import set_train_materials, set_test_materials
from Training_Testing_Data_Creator import data_creator

import Displacement_Model
import Adsorption_Temperature_Model
import Desorption_Temperature_Model

from Displacement_Model import PreProcess, MinMaxNormalizer
from Adsorption_Temperature_Model import Temp_PreProcess

run_choice = input("Enter 1) Feature extraction mode, 2) Fully test a material mode, 3) Train a SINGLE model mode, "
                   "4) Test a SINGLE preloaded model mode: ")

if run_choice == '1':

    test_train_choice = input("Do you want to extract features for 1) Training or 2) Testing?: ")                       # Training data creation or testing data creation.

    if test_train_choice == '1':

        hydrogen, compound_ID_set = set_train_materials()  # Extract hydrogen and compounds.
        print("Materials set.")

        diffusion_filename = 'diffusion_training.csv'                                                                   # Define the training data file names (For 3 types of model).
        energy_filename = 'energy_training.csv'
        temperature_filename = 'temperature_training.csv'

        file_data = [
            (diffusion_filename, ['Compound', 'Node Features Initial Combined', 'Edge Features Initial Combined',
                                  'Node Features Optimised Combined', 'Edge features Optimised Combined',
                                  'Edge Indices Combined', 'Diffusion Input Features', 'Diffusion Initial Coords',
                                  'Diffusion Output Coords', 'Uncertain Features', 'Num. Fixed Atoms',
                                  'Num. Placed H Atoms']),
            (energy_filename, ['Compounds', 'Node Features (Triple)', 'Edge Features (Triple)',
                               'Edge Indices (Triple)', 'Energy Input Features (Triple)',
                               'Energy Output Features (Triple)', 'Uncertain Features', 'Num. Fixed Atoms',
                               'Num. Placed H Atoms']),
            (temperature_filename, ['Compound', 'Node Features Optimised Combined', 'Edge Features Optimised Combined',
                                    'Edge Indices Combined', 'Temp. Input Features', 'Temp. Output Features',
                                    'Uncertain Features', 'Num. Fixed Atoms', 'Num. Placed H Atoms'])
        ]                                                                                                               # Define the column headings for each file.

        for filename, header in file_data:
            try:
                with open(filename, mode='x', newline='') as file:                                                      # Try creating the file in exclusive ('x') mode
                    writer = csv.writer(file)
                    writer.writerow(header)                                                                             # Write the header if the file is new
                    print(f"{filename} setup complete (new file created).")

            except FileExistsError:
                print(f"File '{filename}' already exists. Appended new row.")

        for compound in compound_ID_set:                                                                                # Loop through each of the compounds to extract data.
            compound_ID = compound_ID_set[compound]

            try:
                print(f"Processing Compound: {compound}")

                training_features = data_creator(hydrogen, compound_ID, compound, test_train_choice)                    # Run data creator in train mode.

                (node_features, edge_features, edge_indices, diffusion_input_features, diffusion_init_coords,
                 diffusion_output_features, energy_input_features, energy_output_features,
                 temp_input_features, temp_output_features, uncertain_features, num_fixed, num_H) = training_features   # Extract every aspect of training data that will be saved.

                with open(diffusion_filename, mode='a', newline='') as file:                                            # Append mode.
                    writer = csv.writer(file)
                    writer.writerow([f'{compound}', str(node_features[0]),
                                     str(edge_features[0]), str(node_features[1]), str(edge_features[1]),
                                     str(edge_indices[0]), str(diffusion_input_features), str(diffusion_init_coords),
                                     str(diffusion_output_features), str(uncertain_features), str(num_fixed), str(num_H)])

                print(f"Saved diffusion training data for {compound} to {diffusion_filename} CSV.")

                with open(energy_filename, mode='a', newline='') as file1:                                              # Append mode.
                    writer = csv.writer(file1)
                    writer.writerow([f'[{compound}-H, {compound}, H-({compound})]', str([node_features[1], node_features[2], node_features[4]]),
                                     str([edge_features[1], edge_features[2], edge_features[4]]),
                                     str(edge_indices), str(energy_input_features), str(energy_output_features),
                                     str(uncertain_features), str(num_fixed), str(num_H)])

                print(f"Saved energy training data for {compound} to {energy_filename} CSV.")

                with open(temperature_filename, mode='a', newline='') as file2:                                         # Append mode.
                    writer = csv.writer(file2)
                    writer.writerow([f'{compound}', str(node_features[1]), str(edge_features[1]), str(edge_indices[0]),
                                     str(temp_input_features), str(temp_output_features), str(uncertain_features),
                                     str(num_fixed), str(num_H)])

                print(f"Saved temperature training data for {compound} to {temperature_filename}.")

            except Exception as e:
                print(f"Error processing {compound}: {e}")

    elif test_train_choice == '2':

        hydrogen, compound_ID_set = set_test_materials()                                                                # Extract hydrogen and compounds.
        print("Materials set.")

        for compound in compound_ID_set:                                                                                # Loop through each of the compounds to extract data.
            compound_ID = compound_ID_set[compound]

            diffusion_test_filename = f'{compound}_diffusion_testing.csv'
            energy_test_filename = f'{compound}_energy_testing.csv'
            temperature_test_filename = f'{compound}_temperature_testing.csv'

            file_data = [('Diffusion Testing Data',
                          diffusion_test_filename, ['Compound', 'Node Features Initial Combined',
                                                    'Edge Features Initial Combined', 'Edge Indices Combined',
                                                    'Diffusion Input Features', 'Diffusion Initial Coords',
                                                    'Uncertain Features', 'Num. Fixed Atoms', 'Num. Placed Atoms',
                                                    'Oxidation States']),
                         ('Energy Testing Data',
                          energy_test_filename, ['Compounds', 'Node Features (Triple)', 'Edge Features (Triple)',
                                                 'Edge Indices (Triple)', 'Energy Input Features',
                                                 'Uncertain Features', 'Num. Fixed Atoms', 'Num. Placed H Atoms',
                                                 'Oxidation States']),
                         ('Temperature Testing Data',
                          temperature_test_filename, ['Compound', 'Node Features Optimised Combined',
                                                      'Edge Features Optimised Combined', 'Edge Indices Combined',
                                                      'Temperature Input Features', 'Uncertain Features',
                                                      'Num. Fixed Atoms', 'Placed H Atoms'])]

            for folder, filename, header in file_data:
                try:                                                                                                    # Try creating the file in exclusive ('x') mode.
                    with open(f'{folder}/{filename}', mode='x', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(header)                                                                         # Write the header if the file is new.
                        print(f"{filename} setup complete (new file created).")

                except FileExistsError:
                    print(f"File '{filename}' already exists. Appended new row.")

            try:
                print(f"Processing Compound: {compound}")

                testing_features = data_creator(hydrogen, compound_ID, compound, test_train_choice)                     # Run data creator in test mode.

                (node_features, edge_features, edge_indices, diffusion_input_features, diffusion_init_coords,
                 energy_input_features, temperature_input_features, uncertain_features, num_fixed, num_H,
                 oxidation_states) = testing_features      # Extract every aspect of testing data that will be saved.

                with open(f'Diffusion Testing Data/{diffusion_test_filename}', mode='a', newline='') as file:           # Append mode.
                    writer = csv.writer(file)
                    writer.writerow(
                        [f'{compound}', str(node_features[0]), str(edge_features[0]), str(edge_indices[0]),
                         str(diffusion_input_features), str(diffusion_init_coords), str(uncertain_features),
                         str(num_fixed), str(num_H), str(oxidation_states)])

                print(f"Saved diffusion testing data for {compound} to CSV.")

                with open(f'Energy Testing Data/{energy_test_filename}', mode='a', newline='') as file:                 # Append mode.
                    writer = csv.writer(file)
                    writer.writerow(
                        [f'[{compound}-H, {compound}, H-({compound})]', str([0, node_features[1], 0]),
                         str([0, edge_features[1], 0]), str([edge_indices[0], edge_indices[1], edge_indices[2]]),
                         str(energy_input_features), str(uncertain_features), str(num_fixed), str(num_H),
                         str(oxidation_states)])

                print(f"Saved energy testing data for {compound} CRYSTAL ALONE to CSV.")

                with open(f'Temperature Testing Data/{temperature_test_filename}', mode='a', newline='') as file:       # Append mode.
                    writer = csv.writer(file)
                    writer.writerow(
                        [f'{compound}', str([0]), str([0]), str(edge_indices[0]), str(temperature_input_features),
                         str(uncertain_features), str(num_fixed), str(num_H)])

                print(f"Saved temperature data for {compound} to CSV.")

            except Exception as e:
                print(f"Error processing {compound}: {e}")

elif run_choice == '2':

    name = input('Type in the name of the adsorbent you want to test: ')

    Displacement_Model.run_testing(name)

elif run_choice == '3':

    model_choice = input("Which model do you want to train? 1) Adsorbed State Model, 2) Adsorption Temperature Model"
                         "or 3) Desorption Temperature Model: ")

    if model_choice == '1':
        Displacement_Model.run_training()
    elif model_choice == '2':
        Adsorption_Temperature_Model.run_training()
    elif model_choice == '3':
        Desorption_Temperature_Model.run_training()
    else:
        pass

elif run_choice == '4':

    name = input('Type in the name of the adsorbent you want to test: ')

    model_choice = input("Which type of model do you want to perform an individual test with? "
                         "1) Adsorbed State Model, 2) Adsorption Temperature Model or 3) Desorption Temperature"
                         "Model: ")

    if model_choice == '1':
        Displacement_Model.run_testing(name)
    elif model_choice == '2':
        Adsorption_Temperature_Model.run_testing(name)
    elif model_choice == '3':
        Desorption_Temperature_Model.run_testing(name)
    else:
        pass
