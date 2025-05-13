import ast
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, simpledialog


def parse_column(df, col_name, dtype=torch.float32):  # Function to safely parse string representations of nested lists.
    value = df[col_name].iloc[0]  # Get first row.

    try:
        parsed_value = ast.literal_eval(value)  # Convert string to Python list.

        if col_name in ['Diffusion Initial Coords', 'Diffusion Output Coords']:  # Extract only the x, y, z coordinates.
            parsed_value = [[float(coord) for coord in line.split()[1:]] for line in parsed_value]
            return torch.tensor(parsed_value, dtype=dtype)  # Convert to tensor.

        if col_name == 'Edge Indices Combined':
            parsed_value = torch.tensor(list(zip(*ast.literal_eval(value))),
                                        dtype=torch.int64)  # Convert to tensor and convert all values to 64 bit.
            return parsed_value

        return torch.tensor(parsed_value, dtype=dtype)  # Convert other columns normally.

    except (ValueError, SyntaxError) as e:
        print(f"Error parsing {col_name}: {e}")
        return None


def load_training_data(csv_path):
    """
        Extracts the node and edge features, edge indices and target output coordinates. All features are normalised.

        :param csv_path: Path of the training data file.
        :return: All features stored in one variable.
    """

    df = pd.read_csv(csv_path)                                                                                          # Read the CSV file.

    node_features = parse_column(df, 'Node Features Initial Combined')                                         # Extract each set of features.
    edge_features = parse_column(df, 'Edge Features Initial Combined')
    edge_indices = parse_column(df, 'Edge Indices Combined').to(torch.int64)
    system_features = parse_column(df, 'Diffusion Input Features')
    initial_coords = parse_column(df, 'Diffusion Initial Coords')
    output_coords = parse_column(df, 'Diffusion Output Coords')

    try:                                                                                                                # Convert num_fixed_atoms to int.
        num_fixed_atoms = int(df['Num. Fixed Atoms'].iloc[0])
    except ValueError:
        print("Error converting 'Num Fixed' to int")
        num_fixed_atoms = None

    return {
        "node_features": node_features,
        "edge_features": edge_features,
        "edge_indices": edge_indices,
        "system_features": system_features,
        "initial_coords": initial_coords,
        "output_coords": output_coords,
        "num_fixed_atoms": num_fixed_atoms
    }


def load_testing_data(csv_path):
    """
        Extracts the node and edge features, edge indices. All features are normalised.

        :param csv_path: Path of the testing data file.
        :return: All features stored in one variable.
    """

    df = pd.read_csv(csv_path)                                                                                          # Read the CSV file.

    node_features = parse_column(df, 'Node Features Initial Combined')                                         # Extract each set of features.
    edge_features = parse_column(df, 'Edge Features Initial Combined')
    edge_indices = parse_column(df, 'Edge Indices Combined').to(torch.int64)
    system_features = parse_column(df, 'Diffusion Input Features')
    initial_coords = parse_column(df, 'Diffusion Initial Coords')

    try:
        num_fixed_atoms = int(df['Num. Fixed Atoms'].iloc[0])
    except ValueError:
        print("Error converting 'Num Fixed' to int")
        num_fixed_atoms = None

    return {
        "node_features": node_features,
        "edge_features": edge_features,
        "edge_indices": edge_indices,
        "system_features": system_features,
        "initial_coords": initial_coords,
        "num_fixed_atoms": num_fixed_atoms
    }


class Normaliser():
    def __init__(self):

        self.input='hello'


class FCNN_FeatureCombiner(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim):
        super(FCNN_FeatureCombiner, self).__init__()

        self.input = nn.Linear(input_dim, hidden_size)
        self.fc1 = nn.Linear(hidden_size, output_dim)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.input(x)
        x = self.relu(x)
        x = self.fc1(x)

        return x


class GNN(nn.Module):
    def __init__(self, node_dim, hidden_node_dim, output_node_dim,
                 edge_dim, hidden_edge_dim, output_edge_dim,
                 hidden_gnn_dim1, hidden_gnn_dim2, output_gnn_dim=3):
        super(GNN, self).__init__()

        # Node feature combiner
        self.node_feature_combiner = FCNN_FeatureCombiner(input_dim=node_dim,
                                                          hidden_size=hidden_node_dim,
                                                          output_dim=output_node_dim)

        # Edge feature combiner (to scalar weight)
        self.edge_feature_combiner = FCNN_FeatureCombiner(input_dim=edge_dim,
                                                          hidden_size=hidden_edge_dim,
                                                          output_dim=1)  # Output size = 1 (weight per edge)

        # GCN Layer (uses edge weights)
        self.gcn1 = GCNConv(output_node_dim, hidden_gnn_dim1)
        self.gcn2 = GCNConv(hidden_gnn_dim1, hidden_gnn_dim2)

        # Final output layer
        self.fc_out = nn.Linear(hidden_gnn_dim2, output_gnn_dim)

    def forward(self, node_features, edge_index, edge_features):
        """
        Forward pass through the GNN.

        node_features: Tensor of shape (num_nodes, feature_dim).
        edge_index: Tensor of shape (2, num_edges).
        edge_features: Tensor of shape (num_edges, edge_feature_dim).

        Returns:
        output: Predicted output features for each node.
        """

        # Process node and edge features
        node_features_combined = self.node_feature_combiner(node_features)
        edge_weights = self.edge_feature_combiner(edge_features).squeeze(-1)  # Ensure shape is (num_edges,)

        # Convert edge indices to int64
        edge_index = edge_index.to(torch.long)

        # Pass through GCN layers
        x = self.gcn1(node_features_combined, edge_index, edge_weights)
        x = F.relu(x)
        x = self.gcn2(x, edge_index, edge_weights)
        x = F.relu(x)

        # Final output layer
        output = self.fc_out(x)

        return output


def run_training():
    """
        Opens the training file, extracts and normalises data and runs training on the GNN.
        The GNN full model parameters are then saved for use in testing later.
    """

    # root = tk.Tk()
    # root.withdraw()                                                                                                     # Hide the root window.

    # file_path = filedialog.askopenfilename(title="Select CSV File", filetypes=[("CSV Files", "*.csv")])                 # Open file dialog to select training CSV file.
    # if not file_path:
    #     print("No file selected. Exiting...")
    #     return

    file_path = "diffusion_training.csv"

    data = load_training_data(file_path)                                                                                # Load the diffusion data from the selected file.
    data["node_features"] = data["node_features"][:, 0:6]                                                               # Extract the first 7 features.

    epochs = 2000

    node_size = 6
    node_hidden_size = 2
    node_output_size = 1

    edge_size = 2
    edge_hidden_size = 4
    edge_output_size = 1

    hidden_size1 = 8
    hidden_size2 = 8
    gnn_output_size = 3

    model = GNN(node_size, node_hidden_size, node_output_size,
                edge_size, edge_hidden_size, edge_output_size,
                hidden_size1, hidden_size2, gnn_output_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_train_list = []

    loss_func = nn.MSELoss()

    for epoch in range(epochs):
        model.train()

        train_loss = 0

        node_features = data["node_features"]                                                                           # Extract node and edge features, edge indices and output coordinates.
        edge_index = data["edge_indices"]
        edge_attr = data["edge_features"]
        output_coords = data["output_coords"]

        predicted_coords = model(node_features, edge_index, edge_attr)

        loss = loss_func(predicted_coords, output_coords)                                                               # Compute loss.

        optimizer.zero_grad()                                                                                           # Clear gradients.
        loss.backward()                                                                                                 # Backpropagate.
        optimizer.step()                                                                                                # Step the optimiser.

        train_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")

        loss_train_list.append(train_loss)

    plt.figure()
    plt.plot(loss_train_list, label="Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f'Train Loss')
    plt.show()

    hyperparameters = {
        "node_dim": node_size,
        "hidden_node_dim": node_hidden_size,
        "output_node_dim": node_output_size,
        "edge_dim": edge_size,
        "hidden_edge_dim": edge_hidden_size,
        "output_edge_dim": edge_output_size,
        "hidden_gnn_dim1": hidden_size1,
        "hidden_gnn_dim2": hidden_size2,
        "output_gnn_dim": gnn_output_size
    }                                                                                                                   # Store the hyperparameters in a dictionary for saving.

    root = tk.Tk()                                                                                                      # Request user input for model name.
    root.withdraw()                                                                                                     # Hide the root window.

    model_name = simpledialog.askstring("Input", "Enter the model name:")
    if not model_name:
        print("No model name entered. Exiting...")
        exit()

    model_dir = "Coordinate GNN Models"                                                                                 # Save the trained model in a folder along with its hyperparameters.
    os.makedirs(model_dir, exist_ok=True)
    model_save_path = os.path.join(model_dir, f"coordinate_model_{model_name}.pt")

    model_data = {
        "model_state_dict": model.state_dict(),
        "hyperparameters": hyperparameters
    }                                                                                                                   # Create a dictionary to store both the state_dict and hyperparameters.

    torch.save(model_data, model_save_path)                                                                             # Save the model data.

    print(f"Model and hyperparameters saved at: {model_save_path}")


def run_testing():
    """
        Opens the testing file, extracts and normalises data and runs training on the GNN.
    """

    root = tk.Tk()                                                                                                      # Initialize Tkinter root for file selection.
    root.withdraw()                                                                                                     # Hide the root window.

    testing_file_path = filedialog.askopenfilename(title="Select Testing CSV File", filetypes=[("CSV Files", "*.csv")]) # Select the testing CSV file.
    if not testing_file_path:
        print("No file selected. Exiting...")
        return

    print(f"Selected testing file: {testing_file_path}")                                                                # Load the testing data.
    data = load_testing_data(testing_file_path)

    data["node_features"] = data["node_features"][:, 0:6]                                                               # Extract the first 7 node features per node.

    model_dir = "Coordinate GNN Models"                                                                                 # Select the model to test.
    model_file_path = filedialog.askopenfilename(title="Select Model to Test", initialdir=model_dir,
                                                 filetypes=[("PyTorch Models", "*.pt")])

    if not model_file_path:
        print("No model selected. Exiting...")
        return

    print(f"Selected model: {model_file_path}")

    model_data = torch.load(model_file_path)                                                                            # Load the model and hyperparameters.

    hyperparameters = model_data["hyperparameters"]                                                                     # Retrieve the hyperparameters and use them to initialize the model.

    model = GNN(
        node_dim=hyperparameters["node_dim"],
        hidden_node_dim=hyperparameters["hidden_node_dim"],
        output_node_dim=hyperparameters["output_node_dim"],
        edge_dim=hyperparameters["edge_dim"],
        hidden_edge_dim=hyperparameters["hidden_edge_dim"],
        output_edge_dim=hyperparameters["output_edge_dim"],
        hidden_gnn_dim1=hyperparameters["hidden_gnn_dim1"],
        hidden_gnn_dim2=hyperparameters["hidden_gnn_dim2"],
        output_gnn_dim=hyperparameters["output_gnn_dim"]
    )

    model.load_state_dict(model_data["model_state_dict"])                                                               # Load the state_dict (weights).
    model.eval()                                                                                                        # Set the model to evaluation mode.

    with torch.no_grad():
        node_features = data["node_features"]
        edge_index = data["edge_indices"]
        edge_attr = data["edge_features"]

        predicted_coords = model(node_features, edge_index, edge_attr)

        print("Predicted Optimised Coordinates: ", predicted_coords)

    print("Testing completed.")


# run_training()
# run_testing()
