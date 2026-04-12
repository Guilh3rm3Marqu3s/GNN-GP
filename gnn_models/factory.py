from gnn_models.gnn_gp import GCN_GP
from gnn_models.gnn_vanilla import GCN
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
        
        # Vanilla models (Only training)
        'gcn_vanilla': (GCN, False),
    }
    
    if name in mapping:
        return mapping[name]
    else:
        raise ValueError(f"Model '{model_name}' not founded.")