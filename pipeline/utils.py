import torch
import numpy as np
import random
import os
from torch_geometric import seed_everything
def set_seed(seed):
    """
    Set the seed for all random number generators (Python, NumPy, PyTorch)
    to ensure the experiment is strictly reproducible.
    
    Args:
        seed (int): The seed value to be used.
    """
    # Set seed for Python's built-in random module
    random.seed(seed)
    
    # set seed for NumPy
    np.random.seed(seed)
    
    # set seed for PyTorch (CPU)
    torch.manual_seed(seed)
    
    # set seed for PyTorch (GPU)
    # manual_seed_all sets the seed for all available GPUs
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # ensure deterministic behavior in CuDNN (Backend)
    # this might slightly reduce performance but is required for reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # set environment variable for Python hash seed
    # crucial for operations that rely on hash randomization 
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    seed_everything(seed=seed)
    
    
    print(f"[Utils] Global seed set to: {seed}")


def save_checkpoint(model, best_individual_str, args, test_acc, 
                    evolution_time=0.0, training_time=0.0, filename="best_gnn_model.pth"):
    """
    Saves the complete model checkpoint, including weights, architecture,
    hyperparameters, and timing metrics.

    Args:
        model (torch.nn.Module): The trained model.
        best_individual_str (str): The GP formula.
        args (Namespace): Hyperparameters.
        test_acc (float): Final accuracy.
        evolution_time (float): Total time spent in Phase 1 (GP), in seconds.
        training_time (float): Total time spent in Phase 2 (Training), in seconds.
        filename (str): Save path.
    """
    checkpoint = {
        'state_dict': model.state_dict(),
        'gp_formula': str(best_individual_str),
        'args': vars(args),
        'test_acc': test_acc,
        'evolution_time': evolution_time,  
        'training_time': training_time 
    }
    
    torch.save(checkpoint, os.path.join('outputs/',filename))
    print(f"[Utils] Model saved to {filename} | Acc: {test_acc:.4f} | GP Time: {evolution_time:.2f}s | Train Time: {training_time:.2f}s")