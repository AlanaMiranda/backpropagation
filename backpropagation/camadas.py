'''
Este módulo contém funções de operações entre as camadas da rede:
- Inicialização de pesos e bias
- Propagação direta (forward)
- Avaliação do erro na saída (Erro Médio Quadrático)
- Retropropagação (backward)
'''

import numpy as np


# INICIALIZACAO DOS PESOS
# n_entradas: número de neurônios na entrada
# n_saidas: número de neurônios na saída
def pesos (n_entradas, n_saidas, low=-1, high=1):
    return np.random.uniform(low, high, (n_entradas, n_saidas))



# INICIALIZACAO DOS BIASES
def biases (n_entradas):
    return np.zeros((n_entrada, 1))
 



# PROPAGACAO DIRETA
# Faz o feedforward entre as camadas
# Retorna uma matriz com a soma ponderada entre pesos e biases
def propagacao_direta (entrada: np.ndarray, pesos: np.ndarray, bias:np.ndarray) -> np.ndarray:
    
    return np.dot(pesos.T, entrada) + bias



# FUNCAO DE CUSTO
# Função de custo Erro Médio Quadrático
def custo_emq (saida_atual:np.ndarray, saida_real:np.ndarray) -> np.ndarray:
    error = np.abs(saida_atual-saida_real)**2
    return np.sum(error/2, axis=0)



# RETROPROPAGACAO
# Faz backward entre as camadas
def retropropagacao():
    '''CODE HERE'''
    pass
