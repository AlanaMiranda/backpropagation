import numpy as np


# INICIALIZACAO DOS PESOS
# Recebe o numero de neuronios na  entrada e numero de neuronios na saida
# Gera uma matriz de pesos com numeros pequenos aleatorios de -9 a 9
# Divide por 1000 para obetr nrs mais pequenos ainda\
def pesos (nr_neurons_entrada: int, nr_neurons_saida:int) -> np.ndarray:
    pesos = np.random.randint(-9,9, size=(nr_neurons_entrada, nr_neurons_saida))/1000

    return pesos



# INICIALIZACAO DOS BIASES
# Recebe numero de neuronios da camada seguinte
# Gera um vetor coluna de zeros
def biases (nr_neurons_saida:int) -> np.ndarray:
    bias = np.zeros((nr_neurons_saida,1))

    return bias



# PROPAGACAO DIRETA
# Faz o feedforward entre as camadas
# Retorna uma matriz com a soma ponderada entre pesos e biases
def propagacao_direta (entrada: np.ndarray, pesos: np.ndarray, bias:np.ndarray) -> np.ndarray:
    
    return np.dot(pesos.T, entrada) + bias



# FUNCAO DE CUSTO
# Função de custo Erro Médio Quadrático
def custo_emq (saida_atual:np.ndarray, saida_real:np.ndarray) -> np.ndarray:
    m = len(saida_atual)
    error = np.abs(saida_atual-saida_real)**2
    return np.sum(error, axis=0)/(2*m)



# RETROPROPAGACAO
# Faz backward entre as camadas
def retropropagacao():
    # CODE HERE
    pass