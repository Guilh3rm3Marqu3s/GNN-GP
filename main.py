import os
import copy
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np
from deap import tools
from typing import  Dict

from pipeline.argparser import parse_arguments
from pipeline.dataset_loader import load_dataset, DATASET_REGISTRY
from pipeline.train import train_one_epoch, evaluate
from pipeline.utils import set_seed
from gnn_models.factory import get_model_class
from pipeline.deap_config import setup_deap, make_aggr_fn, custom_mutate
from pipeline.deap_config_gat import setup_deap_gat, make_gat_score_fn, custom_mutate_gat

import gc
import json

def eval_wrapper(individual, toolbox, ctx, dataset, data, args, device, chosen_aggr_fn):
    try:
        aggr_func = chosen_aggr_fn(individual, toolbox)
    except Exception as e:
        print(f"Error compiling individual: {e}") 
        return (0.0,)

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
            aggr_func=aggr_func,    
            dropout_rate=args.gnn_dropout,
            heads=args.gnn_heads
        ).to(device)
    except Exception as e:
        print(f"Model Initialization error: {e}") 
        return (0.0,)

    optimizer = optim.Adam(model.parameters(), lr=args.gnn_lr, weight_decay=args.gnn_weight_decay)
    criterion = F.nll_loss
    
    best_val_acc = 0.0
    try:
        for _ in range(args.gp_gnn_epochs):
            train_one_epoch(model, optimizer, data, criterion, set='train')
            val_acc = evaluate(model, data, mask=data.val_mask)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
               
    except Exception as e:
        print(f"Execution error (train): {e}")
        return (0.0,)
    finally:
        del model, optimizer
        torch.cuda.empty_cache()
        gc.collect()
        
    return (best_val_acc,)

def run_evolution(pop, toolbox, ctx, n_gen, cx_pb, mut_pb, hof, stats, hof_size, chosen_mutation):
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals", "avg", "std", "min", "max"]
    
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    for ind in invalid_ind:
        ind.fitness.values = toolbox.evaluate(ind)
    hof.update(pop)
    
    record = stats.compile(pop)
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    print(logbook.stream)
    
    for gen in range(1, n_gen + 1):
        offspring = toolbox.select(pop, len(pop))
        offspring = [copy.deepcopy(ind) for ind in offspring]
        
        for i in range(0, len(offspring) - 1, 2):
            if random.random() < cx_pb:
                offspring[i], offspring[i + 1] = toolbox.mate(offspring[i], offspring[i + 1])
                del offspring[i].fitness.values, offspring[i + 1].fitness.values
                
        for ind in offspring:
            if random.random() < mut_pb:
                ind, = chosen_mutation(ind, toolbox)
                del ind.fitness.values
                
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_ind:
            ind.fitness.values = toolbox.evaluate(ind)
            
        pop[:] = offspring
        for i, elite in enumerate(hof[:hof_size]):
            pop[i] = copy.deepcopy(elite)
            
        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        print(logbook.stream)
        
    return pop, logbook

def run_split(split_idx, args, device, chosen_setup, chosen_mutation, chosen_aggr_func):
    print(f"\n{'='*50}\nStarting Split {split_idx}\n{'='*50}")
    
    set_seed(args.seed)
    
    dataset, data = load_dataset(ds=args.dataset, split_idx=split_idx)
    if dataset is None:
        return 0.0
    data = data.to(device)
    
    ModelClass, is_gp_model = get_model_class(args.gnn_model)
    evo_best_acc = "Vanilla_Architecture"
    best_ind_str = 'Vanilla_Architecture'
    best_func = None

    evolution_time = None
    
    if is_gp_model:
        print(f"\n[Split {split_idx} - Step 1] Running GP Evolution...")
        toolbox, pset, ctx = chosen_setup(args)
        toolbox.register("evaluate", eval_wrapper, toolbox=toolbox, ctx=ctx,
                         dataset=dataset, data=data, args=args, device=device, chosen_aggr_fn=chosen_aggr_func)

        pop = toolbox.population(n=args.gp_pop_size)
        hof = tools.HallOfFame(args.gp_hof_size)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        start_time_gp = time.time()
        pop, logbook = run_evolution(pop, toolbox, ctx, n_gen=args.gp_generations, 
                                     cx_pb=args.gp_cx_prob, mut_pb=args.gp_mut_prob, 
                                     hof=hof, stats=stats, hof_size=args.gp_hof_size, chosen_mutation=chosen_mutation)
        evolution_time = time.time() - start_time_gp
        
        log_filename = f"log_model_{args.gnn_model}_dataset_{args.dataset}_split_{split_idx}_{args.seed}.csv"
        os.makedirs('outputs/logs', exist_ok=True)
        with open(os.path.join('outputs/logs', log_filename), 'w') as f:
           f.write(str(logbook))
           
        best_ind = hof[0]
        best_ind_str = str(best_ind)
        print(f"Best solution for Split {split_idx}: {best_ind} (Fitness: {best_ind.fitness.values[0]:.4f})")
        evo_best_acc = best_ind.fitness.values[0]
        best_func = chosen_aggr_func(best_ind, toolbox)
    else:
        print(f"\n[Split {split_idx} - Step 1] Skipped (Vanilla Model).")

    print(f"\n[Split {split_idx} - Step 2] Training final model...")
    if is_gp_model:
        model = ModelClass(in_channels=dataset.num_node_features, hidden_channels=args.gnn_hidden_dim,
                           out_channels=dataset.num_classes, num_layers=args.gnn_layers,
                           aggr_func=best_func, dropout_rate=args.gnn_dropout, heads=args.gnn_heads).to(device)
    else:
        model = ModelClass(in_channels=dataset.num_node_features, hidden_channels=args.gnn_hidden_dim,
                           out_channels=dataset.num_classes, num_layers=args.gnn_layers,
                           dropout_rate=args.gnn_dropout, heads=args.gnn_heads).to(device)
        
    optimizer = optim.Adam(model.parameters(), lr=args.gnn_lr, weight_decay=args.gnn_weight_decay)
    criterion = F.nll_loss
    
    best_val_acc = 0.0
    best_test_acc_at_val = 0.0
    counter = 0
    infos: Dict = {
        "seed": args.seed,
        "split_idx": split_idx,
        "evolution_time": evolution_time,
        "evaluation_time": None,
        "best_ind": best_ind_str,
        "evolution_best_acc": evo_best_acc,
        "acc": {
            "epoch": [],
            "value": []
        }
    }
    
    start_time = time.time()
    
    for epoch in range(1, args.gnn_epochs + 1):
        train_one_epoch(model, optimizer, data, criterion, set='train')
        
        val_acc = evaluate(model, data, mask=data.val_mask)
        
        if val_acc > best_val_acc:
            test_acc = evaluate(model, data, mask=data.test_mask)
            infos["acc"]["epoch"].append(epoch)
            infos["acc"]["value"].append(test_acc)
            best_val_acc = val_acc
            best_test_acc_at_val = test_acc
            counter = 0
        else:
            counter += 1
            if counter >= args.patience:
                break
            
    evaluation_time = time.time() - start_time
    infos["evaluation_time"] = evaluation_time
    
    print(f"Split {split_idx} Training ended. Best Test Acc (at optimal validation): {best_test_acc_at_val:.4f}")
    
    os.makedirs("./outputs/evaluation", exist_ok=True)
    with open(f"./outputs/evaluation/model_{args.gnn_model}_dataset_{args.dataset}_{args.seed}.txt","a") as file:
        json.dump(infos, file)
        file.write("\n")   
    
    return best_test_acc_at_val

def main():
    args = parse_arguments()
    set_seed(args.seed)
    
    if args.gnn_model == 'gatv2_gp':
        chosen_mutation = custom_mutate_gat
        chosen_setup = setup_deap_gat
        chosen_aggr_func = make_gat_score_fn
    else:
        chosen_mutation = custom_mutate
        chosen_setup = setup_deap
        chosen_aggr_func = make_aggr_fn
    
    print("--- Configuration ---")
    for key, value in vars(args).items(): print(f"{key:<20}: {value}")
    
    ModelClass, is_gp_model = get_model_class(args.gnn_model)
    print(f"Selected model: {args.gnn_model} (GP Mode: {is_gp_model})")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    ds_name = args.dataset.lower()
    if ds_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset {args.dataset}")
        
    config = DATASET_REGISTRY[ds_name]
    protocol = config.get('protocol', '')
    num_splits = 1 if protocol == 'planetoid' else 10
    
    print(f"Running evaluation over {num_splits} split(s)...")
    
    split_accuracies = []
    for split_idx in range(num_splits):
        acc = run_split(split_idx, args, device, chosen_setup, chosen_mutation, chosen_aggr_func)
        split_accuracies.append(acc)
        
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    for i, acc in enumerate(split_accuracies):
        print(f"Split {i}: {acc:.4f}")
        
    mean_acc = np.mean(split_accuracies)
    std_acc = np.std(split_accuracies)
    print(f"\nFinal Performance: {mean_acc:.4f} ± {std_acc:.4f}")

if __name__ == '__main__':
    main()