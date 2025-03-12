import pandas as pd
import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, simpledialog
import os


def load_training_data(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Function to safely parse string representations of nested lists
    def parse_column(col_name, dtype=torch.float32):
        value = df[col_name].iloc[0]  # Get first row

        try:
            parsed_value = ast.literal_eval(value)  # Convert string to Python list

            # Special case for coordinate parsing
            if col_name in ['Diffusion Initial Coords', 'Diffusion Output Coords']:
                # Extract only the x, y, z coordinates
                parsed_value = [[float(coord) for coord in line.split()[1:]] for line in parsed_value]
                return torch.tensor(parsed_value, dtype=dtype)  # Convert to tensor

            if col_name == 'Edge Indices Combined':
                parsed_value = torch.tensor(list(zip(*ast.literal_eval(value))), dtype=torch.int64)
                return parsed_value

            return torch.tensor(parsed_value, dtype=dtype)  # Convert other columns normally

        except (ValueError, SyntaxError) as e:
            print(f"Error parsing {col_name}: {e}")
            return None  # Return None if there's an issue

    # Convert each column
    node_features = parse_column('Node Features Initial Combined')
    edge_features = parse_column('Edge Features Initial Combined')
    edge_indices = parse_column('Edge Indices Combined').to(torch.int64)
    system_features = parse_column('Diffusion Input Features')
    initial_coords = parse_column('Diffusion Initial Coords')
    output_coords = parse_column('Diffusion Output Coords')

    # Convert num_fixed_atoms to int
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
        "output_coords": output_coords,
        "num_fixed_atoms": num_fixed_atoms
    }


def preprocess_node_features(node_features):
    """
    Extracts mass, proton number, and electron number from each node's feature list.

    Args:
        node_features (torch.Tensor): Tensor of shape (num_atoms, feature_dim)

    Returns:
        torch.Tensor: Processed node features of shape (num_atoms, 3)
    """
    return node_features[:, 0:6]  # Extract columns 3, 4, and 5


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
    # Initialize Tkinter root for file selection
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Open file dialog to select training CSV file
    file_path = filedialog.askopenfilename(title="Select CSV File", filetypes=[("CSV Files", "*.csv")])
    if not file_path:
        print("No file selected. Exiting...")
        return

    # Load the diffusion data from the selected file
    data = load_training_data(file_path)
    data["node_features"] = preprocess_node_features(data["node_features"])

    epochs = 100

    node_size = 6
    node_hidden_size = 16
    node_output_size = 1

    edge_size = 2
    edge_hidden_size = 8
    edge_output_size = 1

    hidden_size1 = 32
    hidden_size2 = 64
    gnn_output_size = 3

    model = GNN(node_size, node_hidden_size, node_output_size,
                edge_size, edge_hidden_size, edge_output_size,
                hidden_size1, hidden_size2, gnn_output_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_train_list = []

    for epoch in range(epochs):
        model.train()

        train_loss = 0

        # Extract data
        node_features = data["node_features"]
        edge_index = data["edge_indices"]
        edge_attr = data["edge_features"]
        output_coords = data["output_coords"]

        predicted_coords = model(node_features, edge_index, edge_attr)

        # Compute loss (using MSE loss)
        loss = F.mse_loss(predicted_coords, output_coords)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")

        loss_train_list.append(train_loss)

    plt.figure()
    plt.plot(loss_train_list, label="Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f'Train and Test Losses')
    plt.show()

    # Define the hyperparameters dictionary
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
    }

    # Request user input for model name
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    model_name = simpledialog.askstring("Input", "Enter the model name:")
    if not model_name:
        print("No model name entered. Exiting...")
        exit()

    # Save the trained model in a folder along with its hyperparameters
    model_dir = "Coordinate GNN Models"
    os.makedirs(model_dir, exist_ok=True)

    model_save_path = os.path.join(model_dir, f"coordinate_model_{model_name}.pt")

    # Create a dictionary to store both the state_dict and hyperparameters
    model_data = {
        "model_state_dict": model.state_dict(),
        "hyperparameters": hyperparameters
    }

    # Save the model data
    torch.save(model_data, model_save_path)

    print(f"Model and hyperparameters saved at: {model_save_path}")


def load_testing_data(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Function to safely parse string representations of nested lists
    def parse_column(col_name, dtype=torch.float32):
        value = df[col_name].iloc[0]  # Get first row

        try:
            parsed_value = ast.literal_eval(value)  # Convert string to Python list

            # Special case for coordinate parsing
            if col_name == 'Diffusion Initial Coords':
                # Extract only the x, y, z coordinates
                parsed_value = [[float(coord) for coord in line.split()[1:]] for line in parsed_value]
                return torch.tensor(parsed_value, dtype=dtype)  # Convert to tensor

            if col_name == 'Edge Indices Combined':
                parsed_value = torch.tensor(list(zip(*ast.literal_eval(value))), dtype=torch.int64)
                return parsed_value

            return torch.tensor(parsed_value, dtype=dtype)  # Convert other columns normally

        except (ValueError, SyntaxError) as e:
            print(f"Error parsing {col_name}: {e}")
            return None  # Return None if there's an issue

    # Convert each column
    node_features = parse_column('Node Features Initial Combined')
    edge_features = parse_column('Edge Features Initial Combined')
    edge_indices = parse_column('Edge Indices Combined').to(torch.int64)
    system_features = parse_column('Diffusion Input Features')
    initial_coords = parse_column('Diffusion Initial Coords')

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


def run_testing():
    # Initialize Tkinter root for file selection
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Step 1: Select the testing CSV file
    testing_file_path = filedialog.askopenfilename(title="Select Testing CSV File", filetypes=[("CSV Files", "*.csv")])
    if not testing_file_path:
        print("No file selected. Exiting...")
        return

    # Load the testing data (you can implement your own loading function similar to load_diffusion_data)
    print(f"Selected testing file: {testing_file_path}")
    data = load_testing_data(testing_file_path)

    data["node_features"] = preprocess_node_features(data["node_features"])

    # Step 2: Select the model to test
    model_dir = "Coordinate GNN Models"
    model_file_path = filedialog.askopenfilename(title="Select Model to Test", initialdir=model_dir,
                                                 filetypes=[("PyTorch Models", "*.pt")])

    if not model_file_path:
        print("No model selected. Exiting...")
        return

    print(f"Selected model: {model_file_path}")

    # Load the model and hyperparameters
    model_data = torch.load(model_file_path)

    # Retrieve the hyperparameters and use them to initialize the model
    hyperparameters = model_data["hyperparameters"]

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

    # Load the state_dict (weights)
    model.load_state_dict(model_data["model_state_dict"])
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        node_features = data["node_features"]
        edge_index = data["edge_indices"]
        edge_attr = data["edge_features"]

        predicted_coords = model(node_features, edge_index, edge_attr)

        print("Predicted Optimised Coordinates: ", predicted_coords)

    print("Testing completed.")


# run_training()
# run_testing()
