# GNN-GP: Evolutionary Graph Neural Networks

This repository implements a framework for automatically discovering optimal **Aggregation Functions** for Graph Neural Networks (GNNs) using **Genetic Programming (GP)**.

Unlike traditional GNNs (GCN, GraphSAGE, GIN, GAT) that rely on fixed aggregation operators (mean, sum, max), this project uses evolutionary algorithms to evolve mathematical formulas that combine messages from neighbors to maximize classification accuracy.

## 🚀 Key Features

* **Hybrid Architecture:** Combines **PyTorch Geometric** (for GNN training) with **DEAP** (for Evolutionary search).
* **Reproducibility:** Full seeding support to ensure fair comparison against Vanilla baselines.
* **Interpretable Output:** Generates clean visualizations of the discovered mathematical formulas.

## 🛠️ Installation

Ensure you have Python 3.8+ installed.

```bash
# 1. Install PyTorch (check [https://pytorch.org/](https://pytorch.org/) for your specific CUDA version)
pip install torch

# 2. Install PyTorch Geometric
pip install torch_geometric
pip install torch_scatter torch_sparse

# 3. Install DEAP
pip install deap
```


## 💻 Usage

The main entry point is `main.py`. You can choose between the Evolutionary model (e.g., `gcn_gp`) or standard benchmarks (e.g., `gcn_vanilla`).

1. Running the Evolutionary Search

To evolve a new aggregation function for the Cora dataset:

```bash
python3 main.py --gnn_model gcn_gp --dataset cora --seed 42 --gnn_epochs 200 --gp_generations 50
```

2. Running Vanilla Baselines

To compare the results against a standard GCN with the same settings:

```bash
python3 main.py --gnn_model gcn_vanilla --dataset cora --seed 42 --gnn_epochs 200
```