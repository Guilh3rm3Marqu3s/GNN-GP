from gnn_models.gnn_gp import GCN_GP, SAGE_GP, GIN_GP, GAT_GP
from gnn_models.gnn_vanilla import VanillaGCN, VanillaSAGE, VanillaGAT, VanillaGIN

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
        'sage_gp': (SAGE_GP, True),
        'gin_gp' : (GIN_GP, True),
        'gat_gp': (GAT_GP, True),
        
        # Vanilla models (Only training)
        'gcn_vanilla': (VanillaGCN, False),
        'sage_vanilla': (VanillaSAGE, False),
        'gin_vanilla': (VanillaGIN, False),
        'gat_vanilla': (VanillaGAT, False),
    }
    
    if name in mapping:
        return mapping[name]
    else:
        raise ValueError(f"Model '{model_name}' not founded.")