'''
Módulo principal do algoritmo.
Contém a função principal backpropagation
'''

from backpropagation.avaliacoes import *
from backpropagation.funcoes import *
from backpropagation.camadas import *
from backpropagation.preprocessamento import *


# Função principal do algoritmo de backpropagation
def backpropagation(x_treino: np.ndarray, y_treino: np.ndarray, neuronios_camada_escondida: int, f_ativacao: str, taxa_de_aprendizagem=0.01, epocas=5):
    items, atributos = x_treino.shape

    # Inicialização dos pesos e biases
    W1 = pesos(atributos, neuronios_camada_escondida)  # Pesos para a camada oculta
    B1 = biases(neuronios_camada_escondida)  # Biases para a camada oculta
    W2 = pesos(neuronios_camada_escondida, 1)  # Pesos para a camada de saída
    B2 = biases(1)  # Biases para a camada de saída

    erros = []
    for e in range(epocas):
        for item in range (items):
            # Propagação direta
            S2 = propagacao_direta(x_treino[:,item].reshape(-1,1), W1, B1)
            Z2 = ativacao(S2, f_ativacao)  # Ativação na camada escondida
            S1 = propagacao_direta(Z2, W2, B2)
            Z1 = ativacao(S1, 'sigmoide')  # Ativação na camada de saída

            # Cálculo de erro na saída
            erro_atual = custo_emq(Z1, y_treino[item])
            erros.append(erro_atual)  # Atualiza a lista de erros da saída

            # Retropropagação na camada de saída
            d1 = delta_saida(Z1, y_treino[item])

            # Atualizar W2 e B2
            W2 -= taxa_de_aprendizagem * gradiente(d1,Z2)
        
            B2 -= taxa_de_aprendizagem * np.sum(d1, axis=1, keepdims=True)#verificar bem isso

            # Retropropagação na camada oculta
            d2 = delta_oculta(d1, W2, S2, f_ativacao)

            # Atualizar W1 e B1
            W1 -= taxa_de_aprendizagem * gradiente(d2, x_treino[item].reshape(-1,1))
            B1 -= taxa_de_aprendizagem * np.sum(d2, axis=1, keepdims=True)#verificar bem isso

            # Mostrar progresso do treinamento
        if e % (epocas // 10) == 0 or e == epocas - 1:
            print(f'Época {e+1}/{epocas}, Erro: {erro_atual}')

    return erros, W1, B1, W2, B2