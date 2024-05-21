'''
Este módulo contém funções de ativação da saída do neurónio e suas derivadas
- ReLu
- Tangente Hiperbólico
- Sigmoide/Logística
'''

import numpy as np


# FUNCAO ReLu
# Função
def ativacao_relu(entrada: np.ndarray) -> np.ndarray:
    return np.maximum(0, entrada)



# Derivada
def deriv_relu(entrada: np.ndarray) -> np.ndarray:
    return np.where(entrada > 0, 1, 0)



# FUNCAO TANGENTE HIPERBOLICO
# # Função
def ativacao_tanh (entrada: np.ndarray) -> np.ndarray: 
    return (np.exp(2*entrada)-1)/(np.exp(2*entrada)+1)



# Derivada
def deriv_tanh(entrada: np.ndarray) -> np.ndarray:
    return 1 - np.tanh(entrada)**2



# FUNCAO SIGMOIDE
# Função
def ativacao_sigmoide(entrada: np.ndarray) -> np.ndarray:
    return np.exp(entrada) / (np.exp(entrada)+1)



# Derivada
def deriv_sigmoide(entrada: np.ndarray) -> np.ndarray:
    sig = ativacao_sigmoide(entrada)
    return sig * (1 - sig)



# Funcao de ativacao, aplica a funcao selecionada a um conjunto de dados num vetor
def ativacao(entrada: np.ndarray, func_ativacao: str) -> np.ndarray:
    funcoes_ativacao = {
        'relu': ativacao_relu,
        'tanh': ativacao_tanh,
        'sigmoide': ativacao_sigmoide
    }
    if func_ativacao in funcoes_ativacao:
        return funcoes_ativacao[func_ativacao](entrada)
    else:
        raise ValueError("Função de ativação '{}' não encontrada".format(func_ativacao))
    


# FUNCAO DE CUSTO
# Erro Médio Quadrático - MSE
def custo_emq(saida_atual: np.ndarray, saida_real: np.ndarray) -> np.ndarray:
    return np.mean((saida_atual - saida_real)**2)



# Derivada da função de custo - MSE
def deriv_custo_emq(saida_atual: np.ndarray, saida_real: np.ndarray) -> np.ndarray:
    return saida_atual - saida_real