import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax


class GCNGPConv(MessagePassing):
    """ 
        GCN-style convolutional layer
    """
    
    def __init__(self, in_channels: int, out_channels: int, aggr_func):
        
        super().__init__(aggr=None)
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.aggr_func = aggr_func
        
        
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
    
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        return norm.view(-1,1) * x_j, x_i
        
        #return x_j, x_i
    
    def aggregate(self, message_data, index: torch.Tensor, dim_size: int = None) -> torch.Tensor:
        
        
        neigh_x, target_x = message_data
        return self.aggr_func(neigh_x, target_x, index, dim_size)
                
        
    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        return aggr_out + self.bias
    
    
    
class GCN_GP(nn.Module):
        def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int, aggr_func, dropout_rate: float = 0.5, **kwargs):
            super().__init__()
            
            assert num_layers >= 2, "num_layers must be at least 2"
            
            
            self.dropout_rate = dropout_rate
            self.convs = nn.ModuleList()
            
            # first layer: in_channels -> hidden_channels
            self.convs.append(GCNGPConv(in_channels, hidden_channels, aggr_func))
            
            # intermediate layers: hidden_channels -> hidden_channels
            for _ in range(num_layers - 2):
                self.convs.append(GCNGPConv(hidden_channels, hidden_channels, aggr_func))
                
                
            # last layer: hidden_channels -> out_channels
            self.convs.append(GCNGPConv(hidden_channels, out_channels, aggr_func))
            
            
        def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            for i, conv in enumerate(self.convs):
                 x = F.dropout(x, p=self.dropout_rate, training=self.training)
                 
                 x = conv(x, edge_index)
                 
                 if i<len(self.convs) - 1:
                     #relu after every layer execept the last
                     x = F.relu(x)
                     
                     
            return F.log_softmax(x, dim=1)
                     
                     
                
# --------------------------------------------------------------------------------------------------

# ------------------------------------------ / / / / / / /------------------------------------------

class GINGPConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, aggr_func):
       # Disable internal aggregation
       super().__init__(aggr=None)
       self.aggr_func = aggr_func
       
       # MLP processed after GP aggregation
       self.nn = nn.Sequential(
           nn.Linear(in_channels, out_channels),
           nn.BatchNorm1d(out_channels),
           nn.ReLU(),
           nn.Linear(out_channels, out_channels)
       ) 
       
       self.eps = nn.Parameter(torch.Tensor([0.0]))
       
    
    def forward(self, x, edge_index):
        # Propagate calls aggregate() where GP function is applied
        out = self.propagate(edge_index, x=x)
        return self.nn((1 + self.eps) * x + out)
    
    def message(self, x_i, x_j):
        # Return neighbor and target data for the NeighborTensor
        return x_j, x_i
    
    def aggregate(self, message_data, index, dim_size = None):
        x_j, x_i = message_data
        # Invoke the compiled GP expression
        return self.aggr_func(x_j, x_i, index, dim_size)
    
class GIN_GP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, aggr_func, dropout_rate=0.5, **kwargs):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.convs = nn.ModuleList()
        
        # Inject the same aggr_func into all layers
        for i in range(num_layers):
            in_c = in_channels if i == 0 else hidden_channels
            self.convs.append(GINGPConv(in_c, hidden_channels, aggr_func))
            
        self.final_lin = nn.Linear(hidden_channels, out_channels)
        
        
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        return self.final_lin(x)
    
    
class GATv2GPConv(MessagePassing):
    """
    GATv2-style layer where the attention scoring mechanism is evolved via GP.
    """
    def __init__(self, in_channels: int, out_channels: int, aggr_func):
        super().__init__(aggr='add')
        
        self.score_fn = aggr_func 
        
        # Linear transformation for the actual features being passed as messages
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Pre-transform the features that will be weighted and summed
        x_proj = self.lin(x)
        
        # Pass both original features (for scoring) and projected features (for the message)
        return self.propagate(edge_index, x=x, x_proj=x_proj)

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, x_proj_j: torch.Tensor, index: torch.Tensor, ptr, size_i) -> torch.Tensor:
       
        # The GP function takes original features [E, in_channels] and returns a score [E, 1]
        e = self.score_fn(x_i, x_j) 
        
       
        # Important: e must be shape [E] or [E, 1]
        alpha = softmax(e, index, ptr, size_i)
        
        # Weight the transformed neighbor features
        return x_proj_j * alpha.view(-1, 1)

class GATv2_GP(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int, aggr_func, dropout_rate: float = 0.5, **kwargs):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.convs = nn.ModuleList()

        self.convs.append(GATv2GPConv(in_channels, hidden_channels, aggr_func))
        for _ in range(num_layers - 2):
            self.convs.append(GATv2GPConv(hidden_channels, hidden_channels, aggr_func))
        self.convs.append(GATv2GPConv(hidden_channels, out_channels, aggr_func))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                
        return F.log_softmax(x, dim=1)