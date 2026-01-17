import os
import re
import torch
import numpy as np

NOME_DATASET = "citeseer"  
NOME_MODELO  = "gcn_vanilla"   
DIRETORIO    = "../outputs_experimentos/resultados_vanilla_gp_gcn_completo"

def realizar_analise_estatistica():
    
    arquivos = [f for f in os.listdir(DIRETORIO) if f.endswith('.pth')]
    
   
    seeds_dict = {}

    for f in arquivos:
        #
        padrao = f"checkpoint_{NOME_MODELO}_{NOME_DATASET}_(\d+)_(\d+)\.pth"
        match = re.search(padrao, f)
        
        if match:
            epoch = int(match.group(1))
            seed = int(match.group(2))
            
            
            if seed not in seeds_dict or epoch > seeds_dict[seed][0]:
                seeds_dict[seed] = (epoch, f)

    if not seeds_dict:
        print(f"Nenhum checkpoint encontrado para: {NOME_MODELO} no dataset {NOME_DATASET}")
        return

  
    acuracias = []
    print(f"\n--- Resultados Individuais ({NOME_DATASET} | {NOME_MODELO}) ---")
    print(f"{'Seed':<10} | {'Época':<10} | {'Acurácia (%)'}")
    print("-" * 40)

    for seed in sorted(seeds_dict.keys()):
        epoca, filename = seeds_dict[seed]
        path = os.path.join(DIRETORIO, filename)
        
        checkpoint = torch.load(path, map_location='cpu')
        acc = checkpoint.get('test_acc', 0.0)
        
       
        acc_percent = acc * 100 
        acuracias.append(acc_percent)
        
        print(f"{seed:<10} | {epoca:<10} | {acc_percent:.2f}%")

   
    media = np.mean(acuracias)
    desvio_padrao = np.std(acuracias) 

    print("-" * 40)
    print(f"MÉDIA FINAL:      {media:.2f}%")
    print(f"DESVIO PADRÃO:   ±{desvio_padrao:.2f}%")
    print(f"Nº DE EXPERIMENTOS (SEEDS): {len(acuracias)}")
    print("-" * 40)

if __name__ == "__main__":
    realizar_analise_estatistica()