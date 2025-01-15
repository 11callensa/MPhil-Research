import csv

from Compound_Database import set_materials
from Training_Creator import training_data_creator
import keras
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam


# run_choice = input("Enter 1) Training data creation mode, 2) Training mode or 3) Testing on a preloaded model mode?: ")

run_choice = '1'

if run_choice == '1':

    hydrogen, compound_ID_set = set_materials()
    print("Materials set")

    csv_filename = 'compound_properties.csv'
    try:

        with open(csv_filename, mode='x', newline='') as file:

            print("File opened")

            writer = csv.writer(file)
            writer.writerow(['Compound', 'Node Features', 'Edge Features', 'Edge Indices', 'System Features', 'Masks'])

    except FileExistsError:
        print(f"File '{csv_filename}' already exists. Appending new rows.")

    for compound in compound_ID_set:

        compound_ID = compound_ID_set[compound][0]
        edge_indices = compound_ID_set[compound][1]

        try:
            print(f"Processing Compound: {compound}")

            node_features, edge_features, system_features, masks = training_data_creator(hydrogen, compound_ID, edge_indices)

            with open(csv_filename, mode='a', newline='') as file:  # Append mode
                writer = csv.writer(file)
                writer.writerow([f'{compound}', str(node_features), str(edge_features), str(edge_indices), str(system_features), str(masks)])

            print(f"Saved data for {compound} to CSV.")

        except Exception as e:
            print(f"Error processing {compound}: {e}")


elif run_choice == '2':

    print("Select the .csv file containing the training data")


elif run_choice == '3':

    model_name = input("Input the name of the pre-trained model you want to use: ")

    try:
        model = load_model(f"{model_name}.h5", compile=False)
        model.compile(loss="logcosh", optimizer=Adam(0.001), metrics=["mean_absolute_error"])

    except Exception as e:
        print(f"Error finding the saved model {model_name}: {e}")


