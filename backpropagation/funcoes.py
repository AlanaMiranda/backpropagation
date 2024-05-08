'''
Este módulo contém funções de ativação da saída do neurónio e suas derivas
- ReLu
- Tangente Hiperbólico
- Sigmoide/Logística
'''
import numpy as np


# Função ReLu
def ativacao_relu(entrada: np.ndarray) -> np.ndarray:

    rl = [max(0, x) for x in entrada]

    return np.array(rl)



# Função Tangente Hiperbólico
def ativacao_tanh (entrada: np.ndarray) -> np.ndarray: 

    tanh = [((np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))) for x in entrada ] 

    return np.array(tanh)



# Função Sigmóide
def ativacao_sigmoide(entrada: np.ndarray) -> np.ndarray:
    
    sig = [1/(1+np.exp(-x)) for x in entrada]
    
    return np.array(sig)


# Funcao de ativacao, aplica a funcao selecionada a um conjunto de dados
def ativacao(entrada: np.ndarray, func_ativacao: str) -> np.ndarray:
    funcoes_ativacao = {
        'relu': ativacao_relu,
        'tanh': ativacao_tanh,
        'sigmoide': ativacao_sigmoide
    }

    if func_ativacao in funcoes_ativacao:
        return funcoes_ativacao[func_ativacao](entrada)
    else:
        return 'Função de ativação não encontrada'
    


# Derivada ReLu
def derivada_relu(entrada):
    # CODE HERE
    pass



# Função Tangente Hiperbólico
def derivada_tan_h (entrada):
    # CODE HERE
    pass



# Função Sigmóide
def derivada_sigmoide(entrada: np.ndarray) -> np.ndarray: 
    ds =[ np.exp(-x)*(1+np.exp(-x))**2 for x in entrada ]

    return np.array(ds)