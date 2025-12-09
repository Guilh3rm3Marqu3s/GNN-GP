import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

import numpy as np
from deap import tools, algorithms, gp

# --- pipeline
from pipeline.argparser import parse_arguments
from pipeline.dataset_loader import load_dataset
from pipeline.train import train_one_epoch, evaluate
from pipeline.utils import set_seed, save_checkpoint

# --- model and GP ---
from gnn_models.factory import get_model_class
from pipeline.deap_config import setup_deap 


def save_tree_plot(individual, filename='best_gnn_structure.png'):
    """
    Visualizes the GP tree and saves it as a PNG image.
    """
    # 1 - extract raw structure from DEAP
    nodes, edges, labels = gp.graph(individual)
    
    # 2 - create a Directed Graph (DiGraph) using NetworkX for easier manipulation
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    
    # 3 - define definitions for "trash" (nodes to delete) and "bridges" (nodes to contract)
    
    # trash: technical terminals that don't aid logical interpretation of the formula
    trash_labels = ['zeros', 'dummy_float', 'dim_size', 'index'] 
    
    # bridges: identity functions that just pass data through (e.g., IdIndex, IdInt)
    bridge_labels = ['IdIndex', 'IdInt', 'IdEdge', 'IdFloat']

    # --- removing trash nodes ---
   
    for node in list(g.nodes()):
        # check if node has a label
        if node in labels:
            label = str(labels[node])
            
            # if it's a technical node, remove it
            if label in trash_labels:
                g.remove_node(node)
            
    # --- contract bridges ---
    # Logic: Parent -> IdNode -> Child   ==becomes==>   Parent -> Child
    # we repeat this loop a few times to ensure chains of identities (Id -> Id -> Id) are resolved
    for _ in range(3): 
        for node in list(g.nodes()):
            if node not in g: continue # skip if already deleted
            
            if node in labels:
                label = str(labels[node])
                
                # check if it is an Identity node
                if any(bridge in label for bridge in bridge_labels):
                    preds = list(g.predecessors(node)) # parents
                    succs = list(g.successors(node))   # children
                    
                    # if it has both parent and child, bridge them directly
                    if preds and succs:
                        for p in preds:
                            for s in succs:
                                g.add_edge(p, s)
                    
                    # remove the identity node itself
                    g.remove_node(node)

    # --- visual styling & layout ---
    pos = None
    if graphviz_layout:
        try:
            pos = graphviz_layout(g, prog='dot')
        except:
            pos = nx.spring_layout(g)
    else:
        pos = nx.spring_layout(g)

    plt.figure(figsize=(12, 8))
    
    color_map = []
    final_labels = {}
    
    for node in g.nodes():
        lbl = str(labels.get(node, '?'))
        final_labels[node] = lbl
        
        # color Logic
        if lbl.startswith('Aggr'):
            color_map.append('#ffcccb') # light Red (Aggregators - The Core)
        elif lbl in ['inputs']:
            color_map.append('#90ee90') # light Green (Data Input)
        elif 'Const' in lbl or any(c.isdigit() for c in lbl) or 'Rand' in lbl:
            color_map.append('#add8e6') # light Blue (Learned Constants/Numbers)
        elif lbl in ['MulScalar', 'Add', 'Sub', 'Mul', 'Relu', 'Sigmoid', 'Neg', 'AddScalar']:
             color_map.append('#ffe4b5') # light Orange (Math Operations)
        else:
            color_map.append('#d3d3d3') # gray (Others)

    # Draw the Graph
    nx.draw(g, pos, 
            labels=final_labels, 
            node_color=color_map, 
            with_labels=True, 
            node_size=2500, 
            font_size=11, 
            font_weight='bold', 
            edge_color='gray', 
            width=1.5,
            arrows=True,
            arrowstyle='-|>',
            arrowsize=20)
            
    plt.title("Evolved GNN Aggregation Formula", fontsize=16)
    plt.axis('off')
    output_file = os.path.join('outputs/images/',filename)
    # save to File
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Viz] Tree image saved to: {output_file}")
    
    
def eval_wrapper(individual, toolbox, dataset, data, args, device):
    """
    Evaluation Function (Fitness).
    """
    # 1. Compile individuals
    try:
        func = toolbox.compile(expr=individual)
    except Exception as e:
        
        return (0.0,)

    # 2. Dinamicaly instantiate the model
    try:
       
        ModelClass, is_gp = get_model_class(args.gnn_model)
        
        
        if not is_gp:
            print(f"Error: Trying to evolve a vanilla model with GP ({args.gnn_model}).")
            return (0.0,)

        model = ModelClass(
            in_channels=dataset.num_node_features,
            hidden_channels=args.gnn_hidden_dim, 
            out_channels=dataset.num_classes,
            num_layers=args.gnn_layers,
            aggr_func=func,    #
            dropout_rate=args.gnn_dropout
        ).to(device)
        
    except Exception as e:
        
        print(f"Model Initialization error: {e}") 
        return (0.0,)

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.gnn_lr,
        weight_decay=args.gnn_weight_decay
    )
    criterion = F.nll_loss
    
    # train
    eval_epochs = 20 
    best_val_acc = 0.0
    
    try:
        for _ in range(eval_epochs):
            train_one_epoch(model, optimizer, data, criterion)
            val_acc = evaluate(model, data)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
    except Exception as e:
        
        print(f"Execution error (train): {e}")
        return (0.0,)
        
    return (best_val_acc,)


def main():
    args = parse_arguments()
    
    evolution_time = 0. #for .pth file
    
    # --- Reproducibility setup ---
    set_seed(args.seed)
    
    # --- initial config ---
    print("--- Configuration ---")
    for key, value in vars(args).items():
        print(f"{key:<20}: {value}")
    print("---------------------------------")
    
    ModelClass, is_gp_model = get_model_class(args.gnn_model)
    print(f"Selected model: {args.gnn_model} (GP Mode: {is_gp_model})")
    
    best_func = None
    best_ind_str = 'Vanilla_Architeture' # default for vanilla
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # --- 1. load data ---
    dataset, data = load_dataset(ds=args.dataset) 
    data = data.to(device)
    
    if is_gp_model:
        # --- 2. Setup GP (DEAP) ---
        print("\n[Step 1] Configuring evolutionary algorithm...")
        
        
        # return the configured toolbox and the primitive set
        toolbox, pset = setup_deap()
        
        
        # register the custom evaluation function defined above
        
        toolbox.register("evaluate", eval_wrapper, 
                        toolbox=toolbox, 
                        dataset=dataset, 
                        data=data, 
                        args=args, 
                        device=device)

        # GP hyperparameters
        POP_SIZE = args.gp_pop_size
        N_GEN = args.gp_generations  
        CX_PB = args.gp_cx_prob # crossover probability
        MUT_PB = args.gp_mut_prob # mutation probability
        
        # population initialization
        pop = toolbox.population(n=POP_SIZE)
        hof = tools.HallOfFame(1) # store best tree
        
        # Stats
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # --- Evaluation ---
        print(f"Starting evaluation: {N_GEN} generations...")
        start_time_gp = time.time()
        
        
        algorithms.eaSimple(pop, toolbox, cxpb=CX_PB, mutpb=MUT_PB, ngen=N_GEN, 
                            stats=stats, halloffame=hof, verbose=True)
        
        end_time_gp = time.time()
        evolution_time=end_time_gp - start_time_gp
        
        print(f"Evaluation ended. It took {evolution_time:.2f}s")
        
        # --- retrieves the best candidate ---
        best_ind = hof[0]
        best_ind_str = str(best_ind)
        print(f"\nBest solution: {best_ind}")
        print(f"Fitness: {best_ind.fitness.values[0]:.4f}")

        # --- 3. Final train ---
        print("\n[Step 2] Training the best solution again...")
        
        
        best_func = toolbox.compile(expr=best_ind)
        
        # visualize tree
        try:
            save_tree_plot(best_ind, filename=f'tree_{args.gnn_model}_{args.dataset}_{args.seed}.png')
        except Exception as e:
            print(f"Could not save tree image: {e}")
        
    else:
        print("\n[Step 1] Skipped (Vanilla Model selected).")
        best_func = None
    
    
    
    if is_gp_model:
    # Instantiate the final model with the discovered aggregation function
        model = ModelClass(
            in_channels=dataset.num_node_features,
            hidden_channels=args.gnn_hidden_dim,
            out_channels=dataset.num_classes,
            num_layers=args.gnn_layers,
            aggr_func=best_func,
            dropout_rate=args.gnn_dropout
        ).to(device)
        
    else:
        # without 'aggr_func' (Vanilla Model)
        model = ModelClass(
            in_channels=dataset.num_node_features,
            hidden_channels=args.gnn_hidden_dim,
            out_channels=dataset.num_classes,
            num_layers=args.gnn_layers,
            dropout_rate=args.gnn_dropout
        ).to(device)
        
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.gnn_lr,
        weight_decay=args.gnn_weight_decay
    )
    criterion = F.nll_loss
    
    print('Starting complete training...')
    start_time = time.time()
    best_acc = 0.0
    
    # complete training loop
    for epoch in range(1, args.gnn_epochs + 1):
        loss = train_one_epoch(model, optimizer, data, criterion)
        
        if epoch % 10 == 0:
            test_acc = evaluate(model, data)
            if test_acc > best_acc:
                best_acc = test_acc
                
                save_checkpoint(
                    model=model,
                    best_individual_str=best_ind_str,
                    args=args,
                    evolution_time=evolution_time,
                    training_time=time.time()-start_time,
                    filename=f"checkpoint_{args.gnn_model}_{args.dataset}_{epoch}_{args.seed}.pth",
                    test_acc=test_acc
                    
                )
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Test Acc: {test_acc:.4f} | Best: {best_acc:.4f}")
            
    end_time = time.time()
    print("Training ended.")
    print(f'Total training time: {end_time-start_time:.2f}s')
    print(f'Best final accuracy: {best_acc:.4f}')

if __name__ == '__main__':
    main()