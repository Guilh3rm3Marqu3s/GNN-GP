from __future__ import annotations

from deap import gp, creator, base, tools
import torch
import random


class EdgePairTensor:
    """Holds (central node x_i, neighbor x_j) on edge space."""
    def __init__(self, x_i: torch.Tensor, x_j: torch.Tensor):
        self.x_i = x_i
        self.x_j = x_j

    @classmethod
    def from_nodes(cls, x_i, x_j) -> EdgePairTensor:
        return cls(x_i.clone(), x_j.clone())


# ── Primitives ────────────────────────────────────────────────────────────────

# EdgePairTensor -> EdgePairTensor  (transformations)
def edge_add(ep: EdgePairTensor) -> EdgePairTensor:
    return EdgePairTensor(ep.x_i, ep.x_i + ep.x_j)

def edge_sub(ep: EdgePairTensor) -> EdgePairTensor:
    return EdgePairTensor(ep.x_i, ep.x_i - ep.x_j)

def edge_mul(ep: EdgePairTensor) -> EdgePairTensor:
    return EdgePairTensor(ep.x_i, ep.x_i * ep.x_j)

def edge_abs_diff(ep: EdgePairTensor) -> EdgePairTensor:
    return EdgePairTensor(ep.x_i, torch.abs(ep.x_i - ep.x_j))

def edge_gate(ep: EdgePairTensor) -> EdgePairTensor:
    return EdgePairTensor(ep.x_i, torch.sigmoid(ep.x_i) * ep.x_j)

def edge_leaky_relu(ep: EdgePairTensor) -> EdgePairTensor:
    return EdgePairTensor(ep.x_i, torch.nn.functional.leaky_relu(ep.x_j))

def edge_tanh(ep: EdgePairTensor) -> EdgePairTensor:
    return EdgePairTensor(ep.x_i, torch.tanh(ep.x_j))

# EdgePairTensor -> torch.Tensor  (reductions — root-level only)
def to_dot_product(ep: EdgePairTensor) -> torch.Tensor:
    return torch.sum(ep.x_i * ep.x_j, dim=-1, keepdim=True)        # [E, 1]

def to_l1_dist(ep: EdgePairTensor) -> torch.Tensor:
    return torch.norm(ep.x_i - ep.x_j, p=1, dim=-1, keepdim=True)  # [E, 1]

def to_l2_dist(ep: EdgePairTensor) -> torch.Tensor:
    return torch.norm(ep.x_i - ep.x_j, p=2, dim=-1, keepdim=True)  # [E, 1]

def to_cosine_sim(ep: EdgePairTensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(
        ep.x_i, ep.x_j, dim=-1
    ).unsqueeze(-1)                                                   # [E, 1]

def to_dot_leaky(ep: EdgePairTensor) -> torch.Tensor:
    return torch.nn.functional.leaky_relu(
        torch.sum(ep.x_i * ep.x_j, dim=-1, keepdim=True)
    )

def to_neg_l2(ep: EdgePairTensor) -> torch.Tensor:
    return -torch.norm(ep.x_i - ep.x_j, p=2, dim=-1, keepdim=True)


# ── PrimitiveSet ──────────────────────────────────────────────────────────────

def make_pset_gat() -> gp.PrimitiveSetTyped:

    pset = gp.PrimitiveSetTyped(
        "MAIN",
        [EdgePairTensor],  # "edge_pair" is the sole terminal — covers every leaf
        torch.Tensor
    )
    pset.renameArguments(ARG0="edge_pair")

    # EdgePairTensor -> EdgePairTensor
    pset.addPrimitive(edge_add, [EdgePairTensor], EdgePairTensor, name="AddPair")
    pset.addPrimitive(edge_sub, [EdgePairTensor], EdgePairTensor, name="SubPair")
    pset.addPrimitive(edge_mul, [EdgePairTensor], EdgePairTensor, name="MulPair")
    pset.addPrimitive(edge_abs_diff, [EdgePairTensor], EdgePairTensor, name="AbsDiff")
    pset.addPrimitive(edge_gate, [EdgePairTensor], EdgePairTensor, name="Gate")
    pset.addPrimitive(edge_leaky_relu,[EdgePairTensor], EdgePairTensor, name="LeakyReLUPair")
    pset.addPrimitive(edge_tanh, [EdgePairTensor], EdgePairTensor, name="TanhPair")

    # EdgePairTensor -> torch.Tensor  (reductions, always at the root)
    pset.addPrimitive(to_dot_product, [EdgePairTensor], torch.Tensor, name="DotProduct")
    pset.addPrimitive(to_l1_dist, [EdgePairTensor], torch.Tensor, name="L1Dist")
    pset.addPrimitive(to_l2_dist, [EdgePairTensor], torch.Tensor, name="L2Dist")
    pset.addPrimitive(to_cosine_sim, [EdgePairTensor], torch.Tensor, name="CosineSim")
    pset.addPrimitive(to_dot_leaky, [EdgePairTensor], torch.Tensor, name="DotLeaky")
    pset.addPrimitive(to_neg_l2, [EdgePairTensor], torch.Tensor, name="NegL2")

    return pset


# Mutation 

def mutate_scalars_gat(individual: gp.PrimitiveTree):
    """No scalar ephemerals in this pset — kept for API compatibility."""
    return (individual,)

def custom_mutate_gat(individual, toolbox, struct_prob: float = 0.5):
    if random.random() < struct_prob:
        return toolbox.mutate_structure(individual)
    return toolbox.mutate_scalars(individual)


# Setup

def setup_deap_gat(args):
    pset = make_pset_gat()

    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=args.gp_max_depth)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("select", tools.selTournament, tournsize=args.gp_tourn_size)
    toolbox.register("mate", gp.cxOnePointLeafBiased, termpb=0.1)
    toolbox.register("mutate_structure", gp.mutUniform, expr=toolbox.expr, pset=pset)
    toolbox.register("mutate_scalars", mutate_scalars_gat)

    toolbox.decorate("mate", gp.staticLimit(key=lambda i: i.height, max_value=args.gp_max_depth))
    toolbox.decorate("mutate_structure", gp.staticLimit(key=lambda i: i.height, max_value=args.gp_max_depth))

    return toolbox, pset, pset.context


# Score function factory 

def make_gat_score_fn(individual, toolbox):
    compiled = toolbox.compile(expr=individual)

    def score_fn(x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        ep = EdgePairTensor.from_nodes(x_i, x_j)
        out = compiled(ep)
        if out.dim() == 1:
            out = out.unsqueeze(-1)  # [E] -> [E, 1]
        return out

    return score_fn