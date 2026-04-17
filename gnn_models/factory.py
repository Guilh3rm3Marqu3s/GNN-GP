from gnn_models.gnn_gp import GCN_GP, GIN_GP, GATv2_GP
from gnn_models.gnn_vanilla import GCN, GIN, GATv2
def get_model_class(model_name):
    """
        Input:
            model_name: str = which GNN model you want to use
        Output:
            (ModelClass, is_gp_model)
    """
    name = model_name.lower()
    
    mapping = {
        # GP models (evolution required)
        'gcn_gp': (GCN_GP, True),
        'gin_gp': (GIN_GP, True),
        'gatv2_gp': (GATv2_GP, True),
        
        # Vanilla models (Only training)
        'gcn_vanilla': (GCN, False),
        'gin_vanilla': (GIN, False),
        'gatv2_vanilla': (GATv2, False),
    }
    
    if name in mapping:
        return mapping[name]
    else:
        raise ValueError(f"Model '{model_name}' not founded.")