'''
Este módulo contém funções de pré-processamento dos dados
- Divisão da base em treino, validação e teste
- Normalização (min-max)
'''

import numpy as np


def normalizar(entrada: np.ndarray, saida: np.ndarray, 
               tr: float = 0.6, val: float = 0.2, tst: float = 0.2, 
               inf: float = 0, sup: float = 1) -> tuple:
    
    # Verificar se a soma das percentagens é igual a 1
    assert tr + val + tst == 1, "A soma das percentagens tem que ser igual a 1"
    
    # Combina entrada e saída em um único dataset
    dataset = np.concatenate((entrada, saida), axis=1)
    np.random.shuffle(dataset)
    
    # Define os cortes para treino, validação e teste
    recorte_tr = int(len(dataset) * tr)
    recorte_val = int(len(dataset) * val) + recorte_tr
    
    # Separa os dados em conjuntos de treino, validação e teste
    x_treino, y_treino = dataset[:recorte_tr, :-1], dataset[:recorte_tr, -1:]
    x_validacao, y_validacao = dataset[recorte_tr:recorte_val, :-1], dataset[recorte_tr:recorte_val, -1:]
    x_teste, y_teste = dataset[recorte_val:, :-1], dataset[recorte_val:, -1:]
    
    # Obter o mínimo e o máximo dos dados de treino para cada atributo
    x_min = np.min(x_treino, axis=0)
    x_max = np.max(x_treino, axis=0)
    
    # Normaliza os dados de treino, validação e teste
    def reescala(X, min_v, max_v, inf_lim, sup_lim):
        return inf_lim + ((X - min_v) / (max_v - min_v)) * (sup_lim - inf_lim)
    
    x_treino = reescala(x_treino, x_min, x_max, inf, sup)
    x_validacao = reescala(x_validacao, x_min, x_max, inf, sup)
    x_teste = reescala(x_teste, x_min, x_max, inf, sup)
    
    return (x_treino, y_treino), (x_validacao, y_validacao), (x_teste, y_teste)