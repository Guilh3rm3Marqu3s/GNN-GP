import operator
import random
import torch
from torch_scatter import scatter
from torch_geometric.utils import degree
from deap import base, creator, tools, gp

# ==========================================
#          TYPE DEFINITIONS
# ==========================================

class EdgeTensor:
    """Features on the edges (Messages). Shape: [E, F]"""
    pass

class NodeTensor:
    """Aggregated features on the nodes. Shape: [N, F]"""
    pass

class IndexTensor:
    """Connectivity (Edge Indices). Shape: [2, E]"""
    pass

# ==========================================
#          PRIMITIVE FUNCTIONS
# ==========================================

# --- aggregators  ---
# These functions must accept 4 arguments to match the pset inputs.

def aggr_add(inputs, index, dim_size, zeros):
    """Sum aggregation (essential for GIN, GAT)."""
    if not isinstance(index, torch.Tensor): pass
    if index.dtype != torch.long: index = index.long()
    return scatter(inputs, index, dim=0, dim_size=dim_size, reduce='add')

def aggr_mean(inputs, index, dim_size, zeros):
    """Mean aggregation (essential for GCN)."""
    if index.dtype != torch.long: index = index.long()
    return scatter(inputs, index, dim=0, dim_size=dim_size, reduce='mean')

def aggr_max(inputs, index, dim_size, zeros):
    """Max aggregation (essential for SAGE-Pool)."""
    if index.dtype != torch.long: index = index.long()
    # clone to avoid inplace modification errors during autograd
    inputs = inputs.clone()
    inputs[inputs == 0] = -1e9 
    return scatter(inputs, index, dim=0, dim_size=dim_size, reduce='max')

# --- topology & Projection  ---

def calc_degree(index, dim_size, inputs_ref):
    """
    Computes the degree of each node.
    """
    # index is already the list of target nodes (cols)
    
    deg = degree(index, dim_size, dtype=torch.float)
    
    # Avoid division by zero
    deg[deg == 0] = 1.0
    
    # Reshape to [N, 1] to allow broadcasting
    return deg.view(-1, 1).to(inputs_ref.device)


def gen_linear_weights(inputs_ref, dim_size):
    """
    Generates a Linear Projection Matrix [N, F].
    This represents a fixed Basis Transformation or 
    a Random Projection layer that allows the model to map features 
    to a new latent space without backpropagation on this specific component.
    """
    num_features = inputs_ref.size(1)
    # generate orthogonal-like distribution or standard normal
    return torch.randn((dim_size, num_features), device=inputs_ref.device)

def broadcast_scalar(inputs, value, dim_size):
    """
    Creates a NodeTensor filled with a specific constant value.
    Allows global bias/thresholding.
    """
    num_features = inputs.size(1)
    return inputs.new_full((dim_size, num_features), value)

# --- node Operations (Element-wise) ---

def elt_add(a, b): return torch.add(a, b)
def elt_sub(a, b): return torch.sub(a, b)
def elt_mul(a, b): return torch.mul(a, b) # Powerful when combined with LinearW
def unary_relu(x): return torch.relu(x)
def unary_sigmoid(x): return torch.sigmoid(x)
def unary_neg(x): return -x

# --- scalar Operations (Learned Weights) ---

def node_mul_float(tensor, scalar): return torch.mul(tensor, scalar)
def edge_mul_float(tensor, scalar): return torch.mul(tensor, scalar)

# float arithmetic to allow combining constants 
def float_add(a, b): return a + b
def float_sub(a, b): return a - b
def float_mul(a, b): return a * b
def identity_float(a): return a

# --- identities & Helpers ---

def identity_index(x): return x
def identity_int(n): return n
def identity_edge(x): return x
def unary_edge_relu(x): return torch.relu(x)
def unary_edge_neg(x): return -x

def generate_random_float():
    """Generates a random float for Ephemeral Constants."""
    return random.uniform(-1, 1)

# ==========================================
#          DEAP CONFIGURATION
# ==========================================

def setup_deap():
    
    # Define inputs for the GP Tree:
    # ARG0: EdgeTensor (inputs/messages)
    # ARG1: IndexTensor (edge_index)
    # ARG2: int (num_nodes)
    # ARG3: NodeTensor (zeros - required terminal for type safety)
    
    pset = gp.PrimitiveSetTyped("MAIN", 
                                [EdgeTensor, IndexTensor, int, NodeTensor], 
                                NodeTensor)
    
    pset.renameArguments(ARG0='inputs')
    pset.renameArguments(ARG1='index')
    pset.renameArguments(ARG2='dim_size')
    pset.renameArguments(ARG3='zeros') 
    
    # ---------------------------------------------------------
    # REGISTER PRIMITIVES
    # ---------------------------------------------------------
    
    # aggregators
    pset.addPrimitive(aggr_add, [EdgeTensor, IndexTensor, int, NodeTensor], NodeTensor, name="AggrAdd")
    pset.addPrimitive(aggr_mean, [EdgeTensor, IndexTensor, int, NodeTensor], NodeTensor, name="AggrMean")
    pset.addPrimitive(aggr_max, [EdgeTensor, IndexTensor, int, NodeTensor], NodeTensor, name="AggrMax")
    
   
    
    # Degree: Allows structural normalization (Scientific Term: Topological Bias)
    pset.addPrimitive(calc_degree, [IndexTensor, int, EdgeTensor], NodeTensor, name="Degree")
    
    # LinearW: Allows feature projection 
    pset.addPrimitive(gen_linear_weights, [EdgeTensor, int], NodeTensor, name="LinearW")
    
    # ConstTensor: Allows global constants (Scientific Term: Global Bias)
    pset.addPrimitive(broadcast_scalar, [EdgeTensor, float, int], NodeTensor, name="ConstTensor")
    
    # operations
    pset.addPrimitive(elt_add, [NodeTensor, NodeTensor], NodeTensor, name="Add")
    pset.addPrimitive(elt_sub, [NodeTensor, NodeTensor], NodeTensor, name="Sub")
    pset.addPrimitive(elt_mul, [NodeTensor, NodeTensor], NodeTensor, name="Mul")
    pset.addPrimitive(unary_relu, [NodeTensor], NodeTensor, name="Relu")
    pset.addPrimitive(unary_sigmoid, [NodeTensor], NodeTensor, name="Sigmoid")
    pset.addPrimitive(unary_neg, [NodeTensor], NodeTensor, name="Neg")
    
    # hybrid Ops (Tensor * Scalar)
    pset.addPrimitive(node_mul_float, [NodeTensor, float], NodeTensor, name="MulScalar")
    pset.addPrimitive(edge_mul_float, [EdgeTensor, float], EdgeTensor, name="EdgeMulScalar")
    
    # edge Ops
    pset.addPrimitive(identity_edge, [EdgeTensor], EdgeTensor, name="IdEdge")
    pset.addPrimitive(unary_edge_relu, [EdgeTensor], EdgeTensor, name="EdgeRelu")
    pset.addPrimitive(unary_edge_neg, [EdgeTensor], EdgeTensor, name="EdgeNeg")
    
    # float Ops 
    pset.addPrimitive(float_add, [float, float], float, name="FAdd")
    pset.addPrimitive(float_sub, [float, float], float, name="FSub")
    pset.addPrimitive(float_mul, [float, float], float, name="FMul")
    pset.addPrimitive(identity_float, [float], float, name="IdFloat")
    
    # Ephemeral Constant Generator
    pset.addEphemeralConstant("RandFloat", generate_random_float, float)

    # identities
    pset.addPrimitive(identity_index, [IndexTensor], IndexTensor, name="IdIndex")
    pset.addPrimitive(identity_int, [int], int, name="IdInt")
    
    # --- DEAP Creator & Toolbox ---
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    
    
    toolbox.register("expr", gp.genGrow, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    # bloat control (limit tree depth)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=7))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=7))

    return toolbox, pset
