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

# --- Aggregators ---

def aggr_add(inputs, index, dim_size, zeros):
    """Sum aggregation (essential for GIN, GAT)."""
    if index.dtype != torch.long: index = index.long()
    return scatter(inputs, index, dim=0, dim_size=dim_size, reduce='add')

def aggr_mean(inputs, index, dim_size, zeros):
    """Mean aggregation (essential for GCN)."""
    if index.dtype != torch.long: index = index.long()
    return scatter(inputs, index, dim=0, dim_size=dim_size, reduce='mean')

def aggr_max(inputs, index, dim_size, zeros):
    """Max aggregation (essential for SAGE-Pool)."""
    if index.dtype != torch.long: index = index.long()
    # Usando o menor valor possível para o tipo de dado para evitar viés de zeros
    fill_value = torch.finfo(inputs.dtype).min
    return scatter(inputs, index, dim=0, dim_size=dim_size, reduce='max', fill_value=fill_value)

# --- Topology & Projection ---

def calc_degree(index, dim_size, inputs_ref):
    """Computes the degree of each node for structural normalization."""
    deg = degree(index, dim_size, dtype=torch.float)
    deg[deg == 0] = 1.0  # Avoid division by zero
    return deg.view(-1, 1).to(inputs_ref.device)

def gen_linear_weights(inputs_ref, seed_float, dim_size):
    """
    Gera pesos aleatórios baseados em uma semente vinda da GP.
    Isso garante que os pesos mudem na mutação (se a semente mudar),
    mas permaneçam idênticos durante todo o treino do indivíduo.
    """
    num_features = inputs_ref.size(1)
    # Converte o float da GP em uma semente inteira determinística
    seed = int(abs(seed_float) * 1_000_000)
    g = torch.Generator(device=inputs_ref.device).manual_seed(seed)
    
    return torch.randn((dim_size, num_features), device=inputs_ref.device, generator=g)

def broadcast_scalar(inputs, value, dim_size):
    """Creates a NodeTensor filled with a specific constant value (Global Bias)."""
    num_features = inputs.size(1)
    return inputs.new_full((dim_size, num_features), value)

# --- Node Operations (element-wise) ---

def elt_add(a, b): return torch.add(a, b)
def elt_sub(a, b): return torch.sub(a, b)
def elt_mul(a, b): return torch.mul(a, b)
def unary_relu(x): return torch.relu(x)
def unary_sigmoid(x): return torch.sigmoid(x)
def unary_neg(x): return -x

# --- Scalar Operations ---

def node_mul_float(tensor, scalar): return torch.mul(tensor, scalar)
def edge_mul_float(tensor, scalar): return torch.mul(tensor, scalar)

def float_add(a, b): return a + b
def float_sub(a, b): return a - b
def float_mul(a, b): return a * b
def identity_float(a): return a

# --- Identities and Helpers ---

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
    # ARG0: EdgeTensor, ARG1: IndexTensor, ARG2: int (num_nodes), ARG3: NodeTensor (zeros)
    pset = gp.PrimitiveSetTyped("MAIN", 
                                [EdgeTensor, IndexTensor, int, NodeTensor], 
                                NodeTensor)
    
    pset.renameArguments(ARG0='inputs')
    pset.renameArguments(ARG1='index')
    pset.renameArguments(ARG2='dim_size')
    pset.renameArguments(ARG3='zeros') 
    
    # --- Register Primitives ---
    
    # Aggregators
    pset.addPrimitive(aggr_add, [EdgeTensor, IndexTensor, int, NodeTensor], NodeTensor, name="AggrAdd")
    pset.addPrimitive(aggr_mean, [EdgeTensor, IndexTensor, int, NodeTensor], NodeTensor, name="AggrMean")
    pset.addPrimitive(aggr_max, [EdgeTensor, IndexTensor, int, NodeTensor], NodeTensor, name="AggrMax")
    
    # Structural & Projections
    pset.addPrimitive(calc_degree, [IndexTensor, int, EdgeTensor], NodeTensor, name="Degree")
    # LinearW agora recebe um float (RandFloat) para garantir estabilidade dos pesos
    pset.addPrimitive(gen_linear_weights, [EdgeTensor, float, int], NodeTensor, name="LinearW")
    pset.addPrimitive(broadcast_scalar, [EdgeTensor, float, int], NodeTensor, name="ConstTensor")
    
    # Node Operations
    pset.addPrimitive(elt_add, [NodeTensor, NodeTensor], NodeTensor, name="Add")
    pset.addPrimitive(elt_sub, [NodeTensor, NodeTensor], NodeTensor, name="Sub")
    pset.addPrimitive(elt_mul, [NodeTensor, NodeTensor], NodeTensor, name="Mul")
    pset.addPrimitive(unary_relu, [NodeTensor], NodeTensor, name="Relu")
    pset.addPrimitive(unary_sigmoid, [NodeTensor], NodeTensor, name="Sigmoid")
    pset.addPrimitive(unary_neg, [NodeTensor], NodeTensor, name="Neg")
    
    # Hybrid/Scalar Ops
    pset.addPrimitive(node_mul_float, [NodeTensor, float], NodeTensor, name="MulScalar")
    pset.addPrimitive(edge_mul_float, [EdgeTensor, float], EdgeTensor, name="EdgeMulScalar")
    
    # Edge Ops
    pset.addPrimitive(identity_edge, [EdgeTensor], EdgeTensor, name="IdEdge")
    pset.addPrimitive(unary_edge_relu, [EdgeTensor], EdgeTensor, name="EdgeRelu")
    pset.addPrimitive(unary_edge_neg, [EdgeTensor], EdgeTensor, name="EdgeNeg")
    
    # Float Arithmetic
    pset.addPrimitive(float_add, [float, float], float, name="FAdd")
    pset.addPrimitive(float_sub, [float, float], float, name="FSub")
    pset.addPrimitive(float_mul, [float, float], float, name="FMul")
    pset.addPrimitive(identity_float, [float], float, name="IdFloat")
    
    # Ephemeral Constants
    pset.addEphemeralConstant("RandFloat", generate_random_float, float)

    # Type Connectors (Identities)
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

    # Bloat control
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=7))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=7))

    return toolbox, pset