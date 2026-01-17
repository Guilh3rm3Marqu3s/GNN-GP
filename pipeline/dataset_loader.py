import os
from torch_geometric.datasets import (
    Planetoid,
    Coauthor,
    Amazon,
    WikipediaNetwork,
    WebKB,
    Actor
    )
import torch_geometric.transforms as T

#transforms applies features normalization

def load_dataset(ds:str='cora'):
    print("Loading dataset...")
    ds_name = ds.lower()
    root = '../data/'
    path = os.path.join(root,ds_name)
    
    transforms_list = [T.NormalizeFeatures()]
    
    if ds_name not in ['cora', 'citeseer', 'pubmed']:
        transforms_list.append(T.RandomNodeSplit(
            split='train_rest',
            num_val=0.2,
            num_test=0.2
        ))
        
    transform = T.Compose(transforms_list)
    
    
    try:
        dataset = None
        # --- Citation Networks (Planetoid) ---
        if ds_name in ['cora', 'citeseer', 'pubmed']:
            dataset = Planetoid(root=path,
                                name=ds_name.capitalize(),
                                transform=transform)
        
        # --- Coauthor Networks ---
        elif ds_name == 'cs':
            dataset = Coauthor(root=path, name='CS', transform=transform)
        elif ds_name == 'physics':
            dataset = Coauthor(root=path, name='Physics', transform=transform)
            
        # --- Amazon Networks ---
        elif ds_name == 'computers':
            dataset = Amazon(root=path, name='Computers', transform=transform)
        elif ds_name == 'photo':
            dataset = Amazon(root=path, name='Photo', transform=transform)
        
        # --- Wikipedia Networks (Heterophilous) ---
        elif ds_name in ['chameleon', 'squirrel']:
            dataset = WikipediaNetwork(root=path, name= ds_name.capitalize(), transform=transform)
            
        # --- WebKB Networks ---
        elif ds_name in ['cornell', 'texas', 'wisconsin']:
            dataset = WebKB(root=path, name=ds_name.capitalize(), transform=transform)
            
        # --- Actor (Film) Network ---
        elif ds_name == 'actor':
            dataset = Actor(root=path, transform=transform)
            
        else:
            raise ValueError(f"Invalid dataset: '{ds}'")
        
        data = dataset[0]
        print("Dataset was succesfully loaded...")
        return dataset, data
    
    except Exception as ex:
        print(f'Error: {ex}')
        return None, None
    