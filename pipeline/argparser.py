import argparse
import sys

def parse_arguments() -> None:
    
    #Initialize the parser
    
    parser = argparse.ArgumentParser(
        description='',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # --- Argument Definitions ---
    
    # --- GNN (Graph Neural Network) Hyperparameters ---
    gnn_group = parser.add_argument_group('GNN Hyperparameters')
    gnn_group.add_argument(
        '--gnn_model',
        type=str,
        default='gcn_gp',
        choices=['gcn_gp', 'gcn_vanilla', 'sage_gp', 'sage_vanilla', 'gin_gp', 'gin_vanilla', 'gat_gp', 'gat_vanilla'],
        help='Type of GNN model architecture to use.'
    )
    
    
    gnn_group.add_argument(
        '--gnn_layers',
        type=int,
        default=2,
        help='The number of GNN layers'
    )
    
    gnn_group.add_argument(
        '--gnn_hidden_dim',
        type=int,
        default=64,
        help='Dimension of hidden GNN layers.'

    )
    
    
    gnn_group.add_argument(
        '--gnn_lr',
        type=float,
        default=0.01,
        help='Learning rate for the GNN optimizer.'
    )
    
    gnn_group.add_argument(
        '--gnn_epochs',
        type=int,
        default=200,
        help='Number of training epochs for GNN'
    )
    
    gnn_group.add_argument(
        '--gnn_dropout',
        type=float,
        default=0.5,
        help='Dropout rate for GNN layers.'
    )
    
    gnn_group.add_argument(
        '--gnn_weight_decay',
        type=float,
        default=5e-4,
        help='L2 regularization for the GNN optimizer'
    )
    
    gnn_group.add_argument(
        '--gnn_heads',
        type=int,
        default=8,
        help='Number of attention heads (specific for GAT model.)'
    )
    
    gnn_group.add_argument(
    '--dataset',
    type=str,
    default='cora',
    help='The problem dataset'    
    )
    gnn_group.add_argument(
        '--patience',
        type=int,
        default=5,
        help='GNN-EPOCHS to Early Stopping '
    )
    
    # ------- GP (DEAP) Hyperparameters
    
    gp_group = parser.add_argument_group("GP Hyperparameters")
    
    gp_group.add_argument(
        '--gp_pop_size',
        type=int,
        default=100,
        help='Population size for Genetic Programming'
    )
    
    gp_group.add_argument(
        '--gp_generations',
        type=int,
        default=50,
        help='Number of generations for GP evolution'
    )
    
    gp_group.add_argument(
        '--gp_gnn_epochs',
        type=int,
        default=60,
        help='Number of GNNs epochs on GP evolution'
    )
    
    gp_group.add_argument(
        '--gp_cx_prob', 
        type=float, 
        default=0.7,
        help="Crossover probability (CXPB) for GP."
    )
    
    gp_group.add_argument(
        '--gp_mut_prob', 
        type=float, 
        default=0.2,
        help="Mutation probability (MUTPB) for GP."
    )
    
    gp_group.add_argument(
        '--gp_max_depth', 
        type=int, 
        default=8,
        help="Maximum tree depth for GP individuals during initialization and mutation."
    )
    
    gp_group.add_argument(
        '--gp_tourn_size', 
        type=int, 
        default=3,
        help="Tournament size for GP selection (e.g., selTournament)."
    )
    
    gp_group.add_argument(
        '--gp_hof_size', 
        type=int, 
        default=1,
        help="Size of the Hall of Fame (number of best individuals to track)."
    )
    
    # ---- General Experimental Parameters ---
    
    exp_group = parser.add_argument_group('Experiment Parameters')
    
    exp_group.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help="Random seed for reproducibility across all libraries."
    )
    exp_group.add_argument(
        '--batch_size', 
        type=int, 
        default=32,
        help="Batch size (can be used for GNN training or GP evaluation)."
    )
        
    return parser.parse_args()




# -- parser test ----

if __name__ == '__main__':
    print('Testing Parser')
    
    args = parse_arguments()
    print('Parsed arguments successfully:')
    
    for key, value in vars(args).items():
        print(f'{key:<20}: {value}')