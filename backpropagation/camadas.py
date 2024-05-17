'''
Este módulo contém funções de operações entre as camadas da rede:
- Inicialização de pesos e bias
- Propagação direta (forward)
- Avaliação do erro na saída (Erro Médio Quadrático)
- Retropropagação (backward)
'''

import numpy as np


# INICIALIZACAO DOS PESOS
def pesos ():
    '''CODE HERE'''
    pass



# INICIALIZACAO DOS BIASES
def biases ():
    '''CODE HERE'''
    pass
 



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