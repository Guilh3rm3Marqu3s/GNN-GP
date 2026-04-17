from deap import gp, creator, base, tools
import torch
import random
import copy
from torch_geometric.utils import scatter


class NeighborTensor:
    # Pre-reduction space [N, D].
    def __init__(self, x_j: torch.Tensor, x_i: torch.Tensor, index: torch.Tensor, dim_size: int):
        self.x_j = x_j
        self.x_i = x_i
        self.index = index
        self.dim_size = dim_size

    @classmethod
    def from_message(cls, x_j, x_i, index, dim_size) -> "NeighborTensor":
        return cls(x_j.clone(), x_i.clone(), index, dim_size)


class AggTensor:
    # Post-reduction space [B, D].
    def __init__(self, data: torch.Tensor):
        self.data = data

    def apply(self, fn) -> "AggTensor":
        return AggTensor(fn(self.data))

    def binary(self, other: "AggTensor", fn) -> "AggTensor":
        return AggTensor(fn(self.data, other.data))


class ScalarValue(float):
    pass


# Ephemeral samplers

def sample_scalar(lo: float = 0.5, hi: float = 4.0) -> ScalarValue:
    return ScalarValue(random.uniform(lo, hi))


# NeighborTensor transforms (stay in [N, D])

def _nb_transform(nb: NeighborTensor, fn) -> NeighborTensor:
    return NeighborTensor(fn(nb.x_i, nb.x_j), nb.x_i, nb.index, nb.dim_size)

def nb_contrast(nb: NeighborTensor) -> NeighborTensor:
    return _nb_transform(nb, lambda xi, xj: xi - xj)

def nb_similarity(nb: NeighborTensor) -> NeighborTensor:
    return _nb_transform(nb, lambda xi, xj: xi * xj)

def nb_gate(nb: NeighborTensor) -> NeighborTensor:
    return _nb_transform(nb, lambda xi, xj: torch.sigmoid(xi) * xj)

def nb_abs_diff(nb: NeighborTensor) -> NeighborTensor:
    return _nb_transform(nb, lambda xi, xj: torch.abs(xi - xj))

def nb_relu_diff(nb: NeighborTensor) -> NeighborTensor:
    return _nb_transform(nb, lambda xi, xj: torch.relu(xj - xi))


# Reductions: NeighborTensor -> AggTensor 

def _scatter(nb: NeighborTensor, reduce: str) -> AggTensor:
    return AggTensor(scatter(nb.x_j, nb.index, dim=0, dim_size=nb.dim_size, reduce=reduce))

def reduce_mean(nb: NeighborTensor) -> AggTensor: return _scatter(nb, "mean")
def reduce_sum(nb: NeighborTensor) -> AggTensor: return _scatter(nb, "sum")
def reduce_max(nb: NeighborTensor) -> AggTensor: return _scatter(nb, "max")

def reduce_std(nb: NeighborTensor) -> AggTensor:
    mean_sq = scatter(nb.x_j * nb.x_j, nb.index, dim=0, dim_size=nb.dim_size, reduce="mean")
    sq_mean = scatter(nb.x_j, nb.index, dim=0, dim_size=nb.dim_size, reduce="mean").pow(2)
    return AggTensor(torch.sqrt(torch.relu(mean_sq - sq_mean)))

def reduce_softmax(nb: NeighborTensor) -> AggTensor:
    # Weighted sum: weights = softmax over neighbor L1 norms.
    norms = nb.x_j.abs().sum(dim=-1, keepdim=True)
    exp_n = torch.exp(norms - norms.detach().max())
    denom = scatter(exp_n, nb.index, dim=0, dim_size=nb.dim_size, reduce="sum")
    w = exp_n / (denom[nb.index] + 1e-8)
    return AggTensor(scatter(w * nb.x_j, nb.index, dim=0, dim_size=nb.dim_size, reduce="sum"))


# AggTensor ops 

def agg_add(x: AggTensor, y: AggTensor) -> AggTensor: return x.binary(y, torch.add)
def agg_mul(x: AggTensor, y: AggTensor) -> AggTensor: return x.binary(y, torch.mul)
def agg_sin(x: AggTensor) -> AggTensor: return x.apply(torch.sin)
def agg_cos(x: AggTensor) -> AggTensor: return x.apply(torch.cos)
def agg_tanh(x: AggTensor) -> AggTensor: return x.apply(torch.tanh)
def agg_relu(x: AggTensor) -> AggTensor: return x.apply(torch.relu)
def agg_norm(x: AggTensor) -> AggTensor: return x.apply(lambda t: torch.nn.functional.normalize(t, dim=-1))
def agg_scale(x: AggTensor, s: ScalarValue) -> AggTensor: return x.apply(lambda t: t * float(s))
def pass_through_scalar(s: ScalarValue) -> ScalarValue: return s
def root_unwrap(x: AggTensor) -> torch.Tensor: return x.data


# PrimitiveSet

def make_pset() -> gp.PrimitiveSetTyped:
    pset = gp.PrimitiveSetTyped("AGG", [NeighborTensor, AggTensor], torch.Tensor)
    pset.renameArguments(ARG0="nb", ARG1="default_agg")

    pset.addEphemeralConstant("scalar", sample_scalar, ret_type=ScalarValue)

    # Transforms
    pset.addPrimitive(nb_contrast, [NeighborTensor], NeighborTensor, name="Contrast")
    pset.addPrimitive(nb_similarity, [NeighborTensor], NeighborTensor, name="Sim")
    pset.addPrimitive(nb_gate, [NeighborTensor], NeighborTensor, name="Gate")
    pset.addPrimitive(nb_abs_diff, [NeighborTensor], NeighborTensor, name="AbsDiff")
    pset.addPrimitive(nb_relu_diff, [NeighborTensor], NeighborTensor, name="ReluDiff")

    # Reductions
    pset.addPrimitive(reduce_mean, [NeighborTensor], AggTensor, name="Mean")
    pset.addPrimitive(reduce_sum, [NeighborTensor], AggTensor, name="Sum")
    pset.addPrimitive(reduce_max, [NeighborTensor], AggTensor, name="Max")
    pset.addPrimitive(reduce_std, [NeighborTensor], AggTensor, name="Std")
    pset.addPrimitive(reduce_softmax, [NeighborTensor], AggTensor, name="SoftmaxSum")

    # Post-reduction ops
    pset.addPrimitive(agg_add, [AggTensor, AggTensor], AggTensor, name="Add")
    pset.addPrimitive(agg_mul, [AggTensor, AggTensor], AggTensor, name="Mul")
    pset.addPrimitive(agg_sin, [AggTensor], AggTensor, name="Sin")
    pset.addPrimitive(agg_cos, [AggTensor], AggTensor, name="Cos")
    pset.addPrimitive(agg_tanh, [AggTensor], AggTensor, name="Tanh")
    pset.addPrimitive(agg_relu, [AggTensor], AggTensor, name="ReLU")
    pset.addPrimitive(agg_norm, [AggTensor], AggTensor, name="Norm")
    pset.addPrimitive(agg_scale, [AggTensor, ScalarValue], AggTensor, name="Scale")
    pset.addPrimitive(pass_through_scalar, [ScalarValue], ScalarValue, name="Id_scalar")

    # Root
    pset.addPrimitive(root_unwrap, [AggTensor], torch.Tensor, name="Out")

    return pset


#  Mutation 

def mutate_scalars(individual: gp.PrimitiveTree):
    for i, node in enumerate(individual):
        if isinstance(node, gp.Terminal) and isinstance(node.value, ScalarValue):
                individual[i] = copy.deepcopy(node)
                individual[i].value = sample_scalar()
    return (individual,)


# Setup

def setup_deap(args):
    pset = make_pset()

    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("expr",  gp.genHalfAndHalf, pset=pset, min_=2, max_=args.gp_max_depth)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("select", tools.selTournament, tournsize=args.gp_tourn_size)
    toolbox.register("mate", gp.cxOnePointLeafBiased, termpb=0.1)
    toolbox.register("mutate_structure", gp.mutUniform, expr=toolbox.expr, pset=pset)
    toolbox.register("mutate_scalars", mutate_scalars)

    toolbox.decorate("mate", gp.staticLimit(key=lambda i: i.height, max_value=args.gp_max_depth))
    toolbox.decorate("mutate_structure", gp.staticLimit(key=lambda i: i.height, max_value=args.gp_max_depth))

    return toolbox, pset, pset.context


# Compile individual to aggregation function 

def make_aggr_fn(individual, toolbox):
    # aggr_fn(x_j, x_i, index, dim_size) -> torch.Tensor [B, D]
    compiled = toolbox.compile(expr=individual)

    def aggr_fn(x_j: torch.Tensor, x_i: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
        nb = NeighborTensor.from_message(x_j, x_i, index, dim_size)
        default_agg = reduce_mean(nb)
        
        return compiled(nb, default_agg)

    return aggr_fn


def custom_mutate(individual, toolbox, struct_prob: float = 0.5):
    if random.random() < struct_prob:
        return toolbox.mutate_structure(individual)
    return toolbox.mutate_scalars(individual)