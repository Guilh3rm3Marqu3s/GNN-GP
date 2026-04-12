import os
import urllib.request
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.datasets import (
    Planetoid,
    Coauthor,
    Amazon,
    WikipediaNetwork,
    WebKB,
    Actor,
)
import torch_geometric.transforms as T

# ---------------------------------------------------------------------------
# DATASET REGISTRY
# ---------------------------------------------------------------------------
# FOUR protocols, matched to the original papers:
#
#   'planetoid'  — Yang et al. (2016) single public split.
#                  20 nodes/class train | 500 val | 1000 test.
#                  Refs: Kipf & Welling (2017), Velicković et al. (2018).

#   'shchur'     — Shchur et al. (2018) "Pitfalls of GNN Evaluation".
#                  20 nodes/class train | 30 nodes/class val | rest test.
#                  split_idx is used as the RNG seed, giving independent
#                  and reproducible splits (use 0–9 for 10 splits).
#
#
#   'geom_gcn'   — Pei et al. (2020) Geom-GCN.
#                  10 pre-generated splits, 60% train / 20% val / 20% test,
#                  stratified by class. split_idx selects one of the 10 splits.
#                  Refs: H2GCN (2020), GPR-GNN (2021), and virtually all
#                  heterophily papers since.
#
#   'geom_gcn_filtered' — Same Geom-GCN protocol but using the duplicate-free
#                  versions of Chameleon and Squirrel released by
#                  Platonov et al. (2023) "A Critical Look at the Evaluation
#                  of GNNs under Heterophily" (ICLR 2023).
#                  Data: github.com/yandex-research/heterophilous-graphs
#                  The filtered graphs ship with their own 10 splits.
# ---------------------------------------------------------------------------

DATASET_REGISTRY = {
    #  Planetoid 
    'cora':  {'class': Planetoid, 'name': 'cora', 'protocol': 'planetoid'},
    'citeseer': {'class': Planetoid, 'name': 'citeseer', 'protocol': 'planetoid'},
    'pubmed': {'class': Planetoid, 'name': 'pubmed', 'protocol': 'planetoid'},
    
    # Coauthor / Amazon (Shchur et. al)
    
    'cs': {'class': Coauthor, 'name': 'cs', 'protocol': 'shchur'},
    'physics': {'class': Coauthor, 'name': 'physics', 'protocol': 'shchur'},
    'computers': {'class': Amazon, 'name': 'computers', 'protocol': 'shchur'},
    'photo': {'class': Amazon, 'name': 'photo', 'protocol': 'shchur'},

    #  Geom-GCN heterophilous (original Pei et al. 2020 splits) 
    'chameleon': {'class': WikipediaNetwork, 'name': 'chameleon', 'protocol': 'geom_gcn'},
    'squirrel':  {'class': WikipediaNetwork, 'name': 'squirrel', 'protocol': 'geom_gcn'},
    'cornell':   {'class': WebKB, 'name': 'cornell', 'protocol': 'geom_gcn'},
    'texas':     {'class': WebKB, 'name': 'texas', 'protocol': 'geom_gcn'},
    'wisconsin': {'class': WebKB, 'name': 'wisconsin', 'protocol': 'geom_gcn'},
    'actor':     {'class': Actor, 'name': None, 'protocol': 'geom_gcn'},

    # Filtered variants (Platonov et al. 2023, duplicate-free) 
    # Use these instead of 'chameleon'/'squirrel' for leakage-free evaluation.
    'chameleon-filtered': {'name': 'chameleon_filtered', 'protocol': 'geom_gcn_filtered'},
    'squirrel-filtered':  {'name': 'squirrel_filtered',  'protocol': 'geom_gcn_filtered'},
}

GEOM_GCN_NUM_SPLITS = 10

# Shchur et al. (2018) split builder
def _shchur_split(data, num_train_per_class=20, num_val_per_class=30, seed=0):
    """
        Build train/val/test masks following Shchur et al. (2018).
        20 nodes/class train | 30 nodes/class val | rest test.
        Use split_idx as seed for reproducible independent splits (0-9).
    """
    rng  = np.random.default_rng(seed)
    labels = data.y.numpy()
    classes = np.unique(labels)
    N = data.num_nodes
 
    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask  = torch.zeros(N, dtype=torch.bool)
    test_mask = torch.zeros(N, dtype=torch.bool)
 
    for cls in classes:
        cls_idx  = np.where(labels == cls)[0]
        required = num_train_per_class + num_val_per_class
        if len(cls_idx) < required:
            raise ValueError(
                f"Class {cls} has only {len(cls_idx)} nodes but "
                f"{required} are needed for train+val."
            )
        perm = rng.permutation(cls_idx)
        train_mask[perm[:num_train_per_class]] = True
        val_mask[perm[num_train_per_class:required]] = True
 
    test_mask[~(train_mask | val_mask)] = True
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data
# Filtered dataset loader  (Platonov et al. 2023)

# The Yandex Research repo ships .npz files that contain:
#   node_features, node_labels, edges, train_masks, val_masks, test_masks
# Each mask array has shape (N, num_splits).
# We download once and cache locally.

_FILTERED_BASE_URL = (
    "https://raw.githubusercontent.com/"
    "yandex-research/heterophilous-graphs/main/data/"
)

_FILTERED_FILES = {
    'chameleon_filtered': 'chameleon_filtered_directed.npz',
    'squirrel_filtered':  'squirrel_filtered_directed.npz',
}


def _download_filtered(name: str, root: str) -> str:
    """Download the filtered .npz once and return its local path."""
    os.makedirs(root, exist_ok=True)
    filename = _FILTERED_FILES[name]
    local_path = os.path.join(root, filename)
    if not os.path.exists(local_path):
        url = _FILTERED_BASE_URL + filename
        print(f"Downloading {filename} ...")
        urllib.request.urlretrieve(url, local_path)
        print("Download complete.")
    return local_path


def _load_filtered_dataset(name: str, root: str, split_idx: int, transform):
    """
    Load a Platonov et al. (2023) filtered Wikipedia dataset.

    Returns a (dataset_wrapper, data) tuple that behaves identically to
    what PyG returns, so the rest of the pipeline works without changes.
    """
    path = _download_filtered(name, root)
    raw  = np.load(path)

    x  = torch.tensor(raw['node_features'], dtype=torch.float)
    y  = torch.tensor(raw['node_labels'],  dtype=torch.long)
    edges = torch.tensor(raw['edges'],  dtype=torch.long).t().contiguous()

    # The .npz stores each undirected edge once — mirror to get both directions.
    edges = torch.cat([edges, edges.flip(0)], dim=1)

    num_splits = raw['train_masks'].shape[1]
    idx = split_idx % num_splits
    if split_idx >= num_splits:
        print(
            f"Warning: split_idx={split_idx} >= {num_splits} available splits "
            f"in {name}. Using split {idx} (modulo)."
        )

    train_mask = torch.tensor(raw['train_masks'][:, idx], dtype=torch.bool)
    val_mask  = torch.tensor(raw['val_masks'][:, idx],  dtype=torch.bool)
    test_mask  = torch.tensor(raw['test_masks'][:, idx], dtype=torch.bool)

    data = Data(x=x, edge_index=edges, y=y,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    if transform is not None:
        data = transform(data)

    # Lightweight wrapper so callers can read .num_node_features / .num_classes.
    class _Wrapper:
        def __init__(self, d):
            self._data            = d
            self.num_node_features = d.num_node_features
            self.num_classes       = int(d.y.max().item()) + 1
        def __getitem__(self, _):
            return self._data

    return _Wrapper(data), data



# Main loader

def load_dataset(ds: str = 'cora', split_idx: int = 0):
    """
    Load a benchmark GNN dataset following its canonical literature protocol.

    Parameters
    ----------
    ds : str
        Dataset name (case-insensitive). Supported:
          Planetoid — cora, citeseer, pubmed
          Geom-GCN — chameleon, squirrel, cornell, texas, wisconsin, actor
          Filtered — chameleon-filtered, squirrel-filtered

    split_idx : int
        - Planetoid : ignored (single fixed public split).
        - geom_gcn : index of the pre-generated Geom-GCN split (0–9).
        - geom_gcn_filtered : index of the Platonov et al. split (0–9).

    Returns
    -------
    dataset : PyG Dataset or lightweight wrapper
    data : torch_geometric.data.Data  with train/val/test masks applied
    """
    print(f"Loading dataset: {ds}...")
    ds_name = ds.lower()

    if ds_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Invalid dataset: '{ds}'. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )

    config = DATASET_REGISTRY[ds_name]
    protocol = config['protocol']
    name_arg = config.get('name')

    root = '../data/'
    path = os.path.join(root, ds_name)
    transform = T.NormalizeFeatures()

    try:
        
        # Filtered variants — Platonov et al. (2023)
       
        if protocol == 'geom_gcn_filtered':
            dataset, data = _load_filtered_dataset(
                name=name_arg,
                root=path,
                split_idx=split_idx,
                transform=transform,
            )

        # Planetoid — single canonical public split
        
        elif protocol == 'planetoid':
            dataset = config['class'](
                root=path, name=name_arg,
                transform=transform, split='public',
            )
            data = dataset[0]

       
        # Geom-GCN — 10 pre-generated splits (60 / 20 / 20 %)
        elif protocol == 'geom_gcn':
            DatasetClass = config['class']
            if name_arg:
                dataset = DatasetClass(root=path, name=name_arg, transform=transform)
            else:
                dataset = DatasetClass(root=path, transform=transform)

            data = dataset[0]

            # PyG stores the 10 Geom-GCN splits as columns of 2-D tensors.
            if data.train_mask.dim() > 1:
                num_splits = data.train_mask.size(1)
                idx = split_idx % num_splits
                if split_idx >= num_splits:
                    print(
                        f"Warning: split_idx={split_idx} >= {num_splits} "
                        f"available splits. Using split {idx} (modulo)."
                    )
                data.train_mask = data.train_mask[:, idx]
                data.val_mask   = data.val_mask[:, idx]
                data.test_mask  = data.test_mask[:, idx]
            else:
                # Older PyG versions return 1-D masks for split 0 only.
                if split_idx != 0:
                    print(
                        "Warning: 1-D masks found; only split 0 is available. "
                        "Ignoring split_idx."
                    )
        elif protocol == 'shchur':
            dataset = config['class'](
                root=path, name=name_arg, transform=transform,
            )
            data = dataset[0]
            data = _shchur_split(data, seed=split_idx)

        else:
            raise ValueError(f"Unknown protocol: '{protocol}'")

       
        # Summary
        
        train_size = int(data.train_mask.sum())
        val_size  = int(data.val_mask.sum())
        test_size  = int(data.test_mask.sum())

        _desc = {
            'planetoid': "public split — Yang et al. (2016)",
            'geom_gcn': f"Geom-GCN split {split_idx % GEOM_GCN_NUM_SPLITS} — Pei et al. (2020) [60/20/20%]",
            'geom_gcn_filtered':  f"filtered split {split_idx % GEOM_GCN_NUM_SPLITS} — Platonov et al. (2023) [60/20/20%]",
            'shchur':  f"Shchur et al. (2018) — seed {split_idx} [20 train/class, 30 val/class]",
        }[protocol]

        display_name = name_arg if name_arg else ds_name.capitalize()
        print(f"Successfully loaded {display_name}")
        print(f"Protocol : {_desc}")
        print(f"Nodes: {data.num_nodes} | Edges: {data.num_edges}")
        print(f"Train: {train_size} | Val: {val_size} | Test: {test_size}")

        return dataset, data

    except Exception as ex:
        print(f"Error loading dataset: {ex}")
        raise