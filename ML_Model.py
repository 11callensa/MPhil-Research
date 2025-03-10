import ast
import csv
import torch

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader


class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(4, 4)  # First layer: 4 input features, 4 output features
        self.conv2 = GCNConv(4, 2)  # Second layer: 4 input features, 2 output features
        self.pool = global_mean_pool  # Use global mean pooling
        self.fc1 = torch.nn.Linear(2 + 4, 1)  # Concatenate graph-level features to node output

    def forward(self, data):
        x, edge_index, edge_attr, batch, graph_features = data.x, data.edge_index, data.edge_attr, data.batch, data.graph_features
        x = self.conv1(x, edge_index, edge_weight=edge_attr[:, 0])  # Pass the first edge feature as edge weights
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_weight=edge_attr[:, 0])  # Same for the second layer
        x = torch.relu(x)
        x = self.pool(x, batch)  # Aggregate to graph-level output

        # Concatenate graph-level features with the pooled result
        x = torch.cat([x, graph_features], dim=1)

        # Fully connected layer to reduce output to scalar
        x = self.fc1(x)  # Final prediction
        return x


def minmax_scaler(x, xmin, xmax, flag):

    if flag == 1:

        x_scaled = (x-xmin)/(xmax-xmin)

    if flag == 2:

        x_scaled = 2*((x-xmin)/(xmax-xmin)) - 1

    return x_scaled


feature = minmax_scaler(8, 1, 10, flag=1)
print(feature)


def read_training_file(filename, start_row=None, end_row=None):
    """
    Reads a CSV file and converts specific string columns to numeric numpy arrays.
    Skips the first row containing column headers.
    Allows selection of rows by specifying `start_row` and `end_row`.

    Parameters:
        filename (str): Name of the CSV file to read.
        start_row (int): The first row index to include (0-based).
        end_row (int): The last row index to include (exclusive).

    Returns:
        Tuple: Compounds and feature lists.
    """
    compounds = []
    node_features_list = []
    edge_features_list = []
    edge_indices_list = []
    system_input_list = []
    system_output_list = []

    graphs = []

    with open(filename, mode='r', newline='') as file:
        reader = csv.reader(file)

        # Skip the header row
        next(reader)  # Skips the first header row

        # Iterate through the CSV rows with index
        for row_index, row in enumerate(reader):
            if start_row is not None and row_index < start_row:
                continue
            if end_row is not None and row_index >= end_row:
                break

            # Extract columns into variables
            compound = row[0]
            node_features = row[1]  # String representation of a list
            edge_features = row[2]  # String
            edge_indices_raw = row[3]  # String
            edge_indices = torch.tensor(list(zip(*ast.literal_eval(edge_indices_raw))), dtype=torch.long)

            system_input_features = row[4]  # String
            system_output_features = row[5]  # String

            # Add to corresponding lists
            compounds.append(compound)

            node_features_list.append(ast.literal_eval(node_features))
            edge_features_list.append(ast.literal_eval(edge_features))
            edge_indices_list.append(edge_indices)
            system_input_list.append(ast.literal_eval(system_input_features))
            system_output_list.append(ast.literal_eval(system_output_features))

            graphs.append([compound, ast.literal_eval(node_features), ast.literal_eval(edge_features), edge_indices,
                           ast.literal_eval(system_input_features), ast.literal_eval(system_output_features)])

    return compounds, node_features_list, edge_features_list, edge_indices_list, system_input_list, system_output_list, graphs


def GNN_training(train_file):

    # Example usage
    compounds, node_features_list, edge_features_list, edge_indices_list, system_input_list, system_output_list, graphs = read_training_file(train_file, start_row=0, end_row=3)

    print("Compounds being trained on: ", compounds)

    dataset = []

    for index, graph in enumerate(graphs):

        print("Model training on: ", compounds)

        node_feats = node_features_list[index]
        node_feats_select = [node[3:7] for node in node_feats]
        node_feats = torch.tensor(node_feats_select, dtype=torch.float)

        edge_feats = edge_features_list[index]
        edge_feats = torch.tensor(edge_feats, dtype=torch.float)

        edge_inds = edge_indices_list[index]

        system_input_feats = system_input_list[index][0:6]
        ads_energy = system_output_list[index][2]

        y = torch.tensor([ads_energy], dtype=torch.float)
        graph_feats = torch.tensor(system_input_feats, dtype=torch.float).unsqueeze(0)  # Add graph-level features

        compile_graph = Data(x=node_feats, edge_index=edge_inds, edge_attr=edge_feats, y=y, graph_features=graph_feats)

        dataset.append(compile_graph)

    # Use a DataLoader to handle batching
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Instantiate the model
    model = GCN()

    # Define optimizer and loss function
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    # Training loop
    model.train()
    total_epochs = 100

    for epoch in range(total_epochs):
        total_loss = 0
        for batch in loader:
            optimiser.zero_grad()
            out = model(batch)  # Graph-level prediction
            loss = criterion(out.view(-1), batch.y)  # Match shapes
            loss.backward()
            optimiser.step()

            # Adjust the learning rate based on epochs
            # if epoch < total_epochs * 0.25:
            #     lr = 0.01  # Large learning rate during first 50%
            # elif epoch < total_epochs * 0.75:
            #     lr = 0.005  # Decrease learning rate at 75%
            # else:
            #     lr = 0.001  # Further decrease learning rate at 90%

            lr = 0.01

            # Apply the new learning rate
            for param_group in optimiser.param_groups:
                param_group['lr'] = lr

            total_loss += loss.item()

        print(f'Epoch {epoch + 1}/{total_epochs}, Loss: {total_loss:.4f}, Learning Rate: {lr:.4e}')

    def save_model(model, path):
        torch.save(model.state_dict(), path)
        print(f"Model saved to {path}")

    # Define the model name based on your choice

    model_num = input("Input the GNN model you want to assign this model: ", )

    model_filename = f"GNN Models/GNN_model_{model_num}.pth"  # You can modify this as needed
    save_model(model, model_filename)


def read_testing_file(filename, start_row=None, end_row=None):
    """
        Reads a CSV file and converts specific string columns to numeric numpy arrays.
        Skips the first row containing column headers.
        Allows selection of rows by specifying `start_row` and `end_row`.

        Parameters:
            filename (str): Name of the CSV file to read.
            start_row (int): The first row index to include (0-based).
            end_row (int): The last row index to include (exclusive).

        Returns:
            Tuple: Compounds and feature lists.
        """
    compounds = []
    node_features_list = []
    edge_features_list = []
    edge_indices_list = []
    system_input_list = []

    graphs = []

    with open(filename, mode='r', newline='') as file:
        reader = csv.reader(file)

        # Skip the header row
        next(reader)  # Skips the first header row

        # Iterate through the CSV rows with index
        for row_index, row in enumerate(reader):
            if start_row is not None and row_index < start_row:
                continue
            if end_row is not None and row_index >= end_row:
                break

            # Extract columns into variables
            compound = row[0]
            node_features = row[1]  # String representation of a list
            edge_features = row[2]  # String
            edge_indices_raw = row[3]  # String
            edge_indices = torch.tensor(list(zip(*ast.literal_eval(edge_indices_raw))), dtype=torch.long)

            system_input_features = row[4]  # String

            # Add to corresponding lists
            compounds.append(compound)

            node_features_list.append(ast.literal_eval(node_features))
            edge_features_list.append(ast.literal_eval(edge_features))
            edge_indices_list.append(edge_indices)
            system_input_list.append(ast.literal_eval(system_input_features))

            graphs.append([compound, ast.literal_eval(node_features), ast.literal_eval(edge_features), edge_indices,
                           ast.literal_eval(system_input_features)])

    return compounds, node_features_list, edge_features_list, edge_indices_list, system_input_list, graphs


def GNN_testing(model_name, test_name):

    compounds, node_features_list, edge_features_list, edge_indices_list, system_input_list, graphs = read_testing_file(test_name)

    model = GCN()

    # Load the trained weights
    model.load_state_dict(torch.load(model_name))
    model.eval()  # Set to evaluation mode

    for index, graph in enumerate(graphs):
        print("Model predicting on: ", compounds[index])

        node_feats_select_test = [node[3:7] for node in node_features_list[index]]
        node_feats_test = torch.tensor(node_feats_select_test, dtype=torch.float)

        edge_feats_test = edge_features_list[index]
        edge_feats_test = torch.tensor(edge_feats_test, dtype=torch.float)

        system_input_test = system_input_list[index][0:6]

        edge_inds_test = edge_indices_list[index]

        graph_test_features = torch.tensor(system_input_test, dtype=torch.float).unsqueeze(
            0)  # Add graph-level features
        graph_test = Data(x=node_feats_test, edge_index=edge_inds_test, edge_attr=edge_feats_test,
                          graph_features=graph_test_features)

        # Make a prediction using the trained model
        with torch.no_grad():
            prediction = model(graph_test)

        print(f"Predicted adsorption energy for {compounds[index]}: {prediction.item():.4f} eV")

    return True
