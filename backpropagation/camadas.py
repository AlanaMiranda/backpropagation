'''
Este módulo contém funções de operações entre as camadas da rede:
- Inicialização de pesos e bias
- Propagação direta (forward)
- Avaliação do erro na saída (Erro Médio Quadrático)
- Retropropagação (backward)
'''

import numpy as np
from .funcoes import deriv_relu, deriv_sigmoide, deriv_tanh, deriv_custo_emq, ativacao, custo_emq

# INICIALIZACAO DOS PESOS
# n_entradas: número de neurônios na entrada
# n_saidas: número de neurônios na saída
def pesos (n_entradas:int, n_saidas:int, low=-1, high=1) -> np.ndarray:
    return np.random.uniform(low, high, (n_entradas, n_saidas))
#gera uma matriz de pesos com valores aleatórios dentro do intervalo especificado por low e high


# INICIALIZACAO DOS BIASES
def biases (n_entradas:int) -> np.ndarray:
    return np.zeros((n_entradas, 1))
 #cria uma matriz de zeros com dimensões n_entradas x 1, onde cada entrada (ou neurônio) recebe um bias inicial de zero.


# Retorna uma matriz com a soma ponderada entre pesos e biases
def soma_ponderada (entrada: np.ndarray, pesos: np.ndarray, bias:np.ndarray) -> np.ndarray: 
    return np.dot(pesos.T, entrada) + bias



# PROPAGACAO DIRETA

# Faz o feedforward entre as camadas
def propagacao_direta(lote_x_treino: np.ndarray, lote_y_treino:np.ndarray,
                      W2:np.ndarray, B2:np.ndarray, f_ativacao:str,
                      W1:np.ndarray, B1:np.ndarray
                      )->tuple[np.ndarray, np.ndarray, np.ndarray]:
 
    # Propagação direta p/camada oculta
    # Calcula a soma ponderada das entradas lote_x_treino usando os pesos W2 e bias B2.
    # Aplica a função de ativação especificada (f_ativacao) na soma ponderada para obter a ativação da camada escondida.
    soma_oculta = soma_ponderada(lote_x_treino, W2, B2)
    ativacao_oculta = ativacao(soma_oculta, f_ativacao)  # Ativação na camada escondida

    # Propagação direta p/camada saida 
    # Calcula a soma ponderada das ativações ocultas usando os pesos W1 e bias B1.
    # Aplica a função de ativação sigmoide na soma ponderada para obter a ativação da camada de saída.
    soma_saida = soma_ponderada(ativacao_oculta, W1, B1)
    ativacao_saida = ativacao(soma_saida, 'sigmoide')  # Ativação na camada de saída
    
    # Cálculo de erro na saída
    # Utiliza a função de custo do erro quadrático médio (custo_emq), 
    # comparando a ativação da camada de saída com as saídas esperadas lote_y_treino.
    erro_atual = custo_emq(ativacao_saida,lote_y_treino)
    
    return ativacao_oculta, ativacao_saida, erro_atual



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


# Calculo de erro nos dados de validacao
def calc_erro_validacao(x_val: np.ndarray, y_val: np.ndarray,
                      pesos2: np.ndarray, bias2:np.ndarray, f_ativacao:str, 
                      pesos1: np.ndarray, bias1:np.ndarray, tamanho_do_lote:int=1):
 
    erro_val =np.array([[]])
    item = 0
    while item < x_val.shape[1]:

        _, _, err = propagacao_direta(
                                    x_val[:, item: item+tamanho_do_lote],
                                    y_val[:, item: item+tamanho_do_lote],
                                    pesos2, bias2, f_ativacao, pesos1, bias1)

        erro_val = np.concatenate((erro_val,err), axis=1) # Atualiza a lista de erro Medio
        item += tamanho_do_lote

    erro_val = 0.5*np.mean(erro_val)

    return erro_val
