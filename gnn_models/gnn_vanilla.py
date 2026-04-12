import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


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
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int, dropout_rate: float = 0.5):
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