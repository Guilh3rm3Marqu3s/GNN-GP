import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Dropout, ReLU
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv

class VanillaGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout_rate=0.5):
        super().__init__()
        self.dropout_rate = dropout_rate
        
        self.layers = ModuleList()
        
        if num_layers == 1:
            self.layers.append(GCNConv(in_channels, out_channels))
        else:
            #Layer 1
            self.layers.append(GCNConv(in_channels, hidden_channels))
            
            #Hidden layers
            for _ in range (num_layers-2):
                self.layers.append(GCNConv(hidden_channels, hidden_channels))
                
            # Last layer
            self.layers.append(GCNConv(hidden_channels, out_channels))
            
        self.relu = ReLU()
        self.dropout = Dropout(p=self.dropout_rate)
        
    def forward(self, data):
        
        x, edge_index = data.x, data.edge_index
        
        for layer in self.layers[:-1]:
            x = layer(x, edge_index)
            x = self.relu(x)
            x = self.dropout(x)
            
        x = self.layers[-1](x, edge_index)
        
        return F.log_softmax(x, dim=1)
    
    
    
class VanillaSAGE(torch.nn.Module):
    """
    Standard GraphSAGE implementation using PyTorch Geometric's SAGEConv.
    It uses the default aggregation ('mean') and concatenates/sums 
    with the node's own features.
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout_rate=0.5, **kwargs):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.layers = torch.nn.ModuleList()
        
        # Input Layer
        self.layers.append(SAGEConv(in_channels, hidden_channels))
        
        # Hidden Layers
        for _ in range(num_layers - 2):
            self.layers.append(SAGEConv(hidden_channels, hidden_channels))
            
        # Output Layer
        self.layers.append(SAGEConv(hidden_channels, out_channels))
        
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # iterate over all layers except the last one
        for layer in self.layers[:-1]:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            
        # last layer (no ReLU/Dropout before softmax)
        x = self.layers[-1](x, edge_index)
        
        return F.log_softmax(x, dim=1)
        
        
class VanillaGAT(torch.nn.Module):
    """
    Standard GAT implementation.
    Uses multi-head attention mechanisms to weigh neighbors.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, heads=1, dropout_rate=0.5, **kwargs):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.layers = torch.nn.ModuleList()
        
        # Input Layer
        # concat=True means output dimension will be heads * hidden_channels
        self.layers.append(GATConv(in_channels, hidden_channels, heads=heads, concat=True))
        
        # We need to adjust dimensions because of concatenation
        hidden_dim_concat = hidden_channels * heads
        
        # Hidden Layers
        for _ in range(num_layers - 2):
            self.layers.append(GATConv(hidden_dim_concat, hidden_channels, heads=heads, concat=True))
            
        # Output Layer
        # Usually concat=False for the last layer to average heads and match out_channels
        self.layers.append(GATConv(hidden_dim_concat, out_channels, heads=1, concat=False))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        for layer in self.layers[:-1]:
            x = layer(x, edge_index)
            x = F.elu(x) # GAT papers typically use ELU instead of ReLU
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            
        x = self.layers[-1](x, edge_index)
        return F.log_softmax(x, dim=1)


class VanillaGIN(torch.nn.Module):
    """
    Standard GIN implementation.
    Uses an MLP after aggregation and a learnable epsilon parameter.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout_rate=0.5, **kwargs):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.layers = torch.nn.ModuleList()
        
        # Helper to create the MLP (Linear -> ReLU -> Linear) expected by GIN
        def make_mlp(input_dim, output_dim):
            return torch.nn.Sequential(
                torch.nn.Linear(input_dim, output_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(output_dim, output_dim)
            )
            
        # Input Layer
        self.layers.append(GINConv(make_mlp(in_channels, hidden_channels), train_eps=True))
        
        # Hidden Layers
        for _ in range(num_layers - 2):
            self.layers.append(GINConv(make_mlp(hidden_channels, hidden_channels), train_eps=True))
            
        # Output Layer
        # Note: GIN usually ends with an MLP mapping to classes
        self.layers.append(GINConv(make_mlp(hidden_channels, out_channels), train_eps=True))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        for layer in self.layers[:-1]:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            
        x = self.layers[-1](x, edge_index)
        return F.log_softmax(x, dim=1)