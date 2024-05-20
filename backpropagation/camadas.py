'''
Este módulo contém funções de operações entre as camadas da rede:
- Inicialização de pesos e bias
- Propagação direta (forward)
- Avaliação do erro na saída (Erro Médio Quadrático)
- Retropropagação (backward)
'''

import numpy as np
from funcoes import deriv_relu, deriv_sigmoide, deriv_tanh, deriv_custo_emq

# INICIALIZACAO DOS PESOS
# n_entradas: número de neurônios na entrada
# n_saidas: número de neurônios na saída
def pesos (n_entradas:int, n_saidas:int, low=-1, high=1) -> np.ndarray:
    return np.random.uniform(low, high, (n_entradas, n_saidas))*0.01



# INICIALIZACAO DOS BIASES
def biases (n_entradas:int) -> np.ndarray:
    return np.zeros((n_entradas, 1))
 


# PROPAGACAO DIRETA
# Faz o feedforward entre as camadas
# Retorna uma matriz com a soma ponderada entre pesos e biases
def propagacao_direta (entrada: np.ndarray, pesos: np.ndarray, bias:np.ndarray) -> np.ndarray: 
    return np.dot(pesos.T, entrada) + bias



# Delta na camada de saída (d1)
def delta_saida(saida_atual: np.ndarray, saida_real: np.ndarray) -> np.ndarray:
    return deriv_custo_emq(saida_atual, saida_real) * deriv_sigmoide(saida_atual)



# Delta na camada escondida (d2)
def delta_oculta(d_saida: np.ndarray, pesos: np.ndarray, saida_atual: np.ndarray, f_ativacao: str) -> np.ndarray:
    der_ativacao = {
        'relu': deriv_relu,
        'tanh': deriv_tanh,
        'sigmoide': deriv_sigmoide
    }
    if f_ativacao in der_ativacao:
        return np.dot(pesos, d_saida) * der_ativacao[f_ativacao](saida_atual)
    else:
        raise ValueError("Função de ativação '{}' não encontrada".format(f_ativacao))



# Gradiente
def gradiente(delta: np.ndarray, entrada: np.ndarray) -> np.ndarray:
    grad = np.dot(delta, entrada.T)
    return grad.T
