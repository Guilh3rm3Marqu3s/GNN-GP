import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax


class GCNConv(MessagePassing):
    """
        Standard GCN convolutional layer (Kipf & Welling, 2017).
        Aggregation: normalized sum of neighbor features.
    """

    def __init__(self, in_channels: int, out_channels: int):

        super().__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # add self-loops
        x = self.lin(x)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        #return self.lin(aggr_out) + self.bias
        return aggr_out + self.bias

class GCN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int, dropout_rate: float = 0.5, **kwargs):
        super().__init__()

        assert num_layers >= 2, "num_layers must be at least 2"

        self.dropout_rate = dropout_rate
        self.convs = nn.ModuleList()

        # first layer: in_channels -> hidden_channels
        self.convs.append(GCNConv(in_channels, hidden_channels))

        # intermediate layers: hidden_channels -> hidden_channels
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        # last layer: hidden_channels -> out_channels
        self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

            x = conv(x, edge_index)

            if i < len(self.convs) - 1:
                # relu after every layer except the last
                x = F.relu(x)

        return F.log_softmax(x, dim=1)
    
    
class GINConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, train_eps: bool = True):
        # GIN must use 'add' aggregation to be injective
        super().__init__(aggr='add')
        
        # The MLP is critical for GIN's discriminative ability
        self.nn = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        
        self.initial_eps = 0.0
        if train_eps:
            self.eps = nn.Parameter(torch.Tensor([self.initial_eps]))
        else:
            self.register_buffer('eps', torch.Tensor([self.initial_eps])) 
            
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Standard neighbor aggregation (sum)
        out = self.propagate(edge_index, x=x)
        
        # Combine central node with neighbors and pass through MLP
        return self.nn((1 + self.eps) * x + out)
    
    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        return x_j
    
    
class GIN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int, dropout_rate: float = 0.5, **kwargs):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.convs = nn.ModuleList()
        
        # Build multi-layer GIN
        self.convs.append(GINConv(in_channels, hidden_channels))
        
        for _ in range(num_layers - 1):
            self.convs.append(GINConv(hidden_channels, hidden_channels))
            
        self.lin = nn.Linear(hidden_channels, out_channels)
        
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
                
        return self.lin(x)
    
        
class GATv2Conv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, heads: int = 1, concat: bool = True, dropout: float = 0.0):
        super().__init__(aggr='add', node_dim=0)
        
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        
        self.lin_l = nn.Linear(in_channels, heads * out_channels, bias=True)
        self.lin_r = nn.Linear(in_channels, heads * out_channels, bias=False)
        
        self.att = nn.Parameter(torch.Tensor(1, heads, out_channels))
        nn.init.xavier_uniform_(self.att)
        
        if concat:
            self.bias = nn.Parameter(torch.zeros(heads * out_channels))
        else:
            self.bias = nn.Parameter(torch.zeros(out_channels))
            
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        H, C = self.heads, self.out_channels
        
        x_l = self.lin_l(x).view(-1, H, C)
        x_r = self.lin_r(x).view(-1, H, C)
        
        out = self.propagate(edge_index, x=(x_l, x_r))
        
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            # Last layer -> MEAN
            # shape: [N, out_channels]
            out = out.mean(dim=1)
            
        out = out + self.bias
        
        return out
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, index: torch.Tensor, ptr, size_i) -> torch.Tensor:
         # x_i and x_j have [E, heads, out_chanels] shape
         
         # Add before activation
         x_combined = x_i + x_j
         x_combined = F.leaky_relu(x_combined, negative_slope=0.2)
         
         alpha = (x_combined * self.att).sum(dim=-1) # alpha shape = (E, heads)
         
         alpha = softmax(alpha, index, ptr, size_i)
         
         alpha = F.dropout(alpha, p=self.dropout, training=self.training)
         
         return x_j * alpha.unsqueeze(-1)
     
     
class GATv2(nn.Module):
     def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int, heads: int = 4, dropout_rate: float = 0.5, **kwargs):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.convs = nn.ModuleList()
        
        # First Layer
        self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=heads, concat=True, dropout=dropout_rate))
        
        # Intermediate Layers
        
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, concat=True, dropout=dropout_rate))
            
        # Last Layer
        
        self.convs.append(GATv2Conv(hidden_channels * heads, out_channels, heads=heads, concat=False, dropout=dropout_rate))
        
        
     def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
         for i, conv in enumerate(self.convs):
             x = F.dropout(x, p=self.dropout_rate, training=self.training)
             x = conv(x, edge_index)
             
             if i < len(self.convs) - 1:
                 x = F.elu(x)
                 
         return F.log_softmax(x, dim=1) 
             
        