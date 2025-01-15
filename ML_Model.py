import tensorflow as tf
from spektral.layers import GraphSageConv, GlobalSumPool
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Layer, Reshape


# Custom layer for creating Sparse Tensor (Adjacency matrix)
class SparseAdjacencyMatrix(Layer):
    def __init__(self, **kwargs):
        super(SparseAdjacencyMatrix, self).__init__(**kwargs)

    def call(self, edge_indices):
        # Reshape to get shape (num_edges, 2)
        edge_indices = tf.reshape(edge_indices, [-1, 2])

        # Cast to int64 for SparseTensor compatibility
        edge_indices = tf.cast(edge_indices, dtype=tf.int64)

        # Number of nodes can be inferred by looking at the max index in the edge indices
        n_nodes = tf.reduce_max(edge_indices) + 1  # Adding 1 because index starts from 0

        # Edge values (typically ones, or edge-specific features)
        edge_values = tf.ones_like(edge_indices[:, 0])

        return tf.sparse.SparseTensor(indices=edge_indices, values=edge_values, dense_shape=(n_nodes, n_nodes))


# Define inputs
node_input = Input(shape=(6,), name='node_features')                                                                    # Defines an input layer with node features: shape (None, 6)
edge_input = Input(shape=(1,), name='edge_features')                                                                    # Defines an input layer with edge features (e.g., length): shape (None, 1)
edge_indices_input = Input(shape=(None, 2), dtype=tf.int32, name='edge_indices')  # Edge index tensor (shape (None, 2)) # Defines an input layer for edge indices

# Reshape edge_indices_input tensor to fit the correct shape
# For a batch of graphs, reshaping in the custom layer should be sufficient

# Use the custom SparseAdjacencyMatrix layer to generate the adjacency matrix
adjacency_matrix = SparseAdjacencyMatrix()(edge_indices_input)

# Build the GNN model using GraphSageConv layers
x = GraphSageConv(64, activation='relu')([node_input, adjacency_matrix, edge_input])
x = GraphSageConv(64, activation='relu')([x, adjacency_matrix, edge_input])

# Global pooling to get the graph-level representation
x = GlobalSumPool()(x)

# Fully connected layers for the final prediction of adsorption energy
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
output = Dense(1, activation='linear', name='output')(x)  # Output a single scalar value: adsorption energy

# Define the model
model = Model(inputs=[node_input, edge_input, edge_indices_input], outputs=output)

# Compile the model with an optimizer and loss function
model.compile(optimizer='adam', loss='mean_squared_error')

# Display model summary
model.summary()
