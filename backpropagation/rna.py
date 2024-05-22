'''
Módulo principal do algoritmo.
Contém a função principal backpropagation
'''

from .avaliacoes import *
from .funcoes import *
from .camadas import *
from .preprocessamento import *


# Função principal do algoritmo de backpropagation
def backpropagation(x_treino: np.ndarray, y_treino: np.ndarray,
                    x_validacao: np.ndarray, y_validacao: np.ndarray,
                    neuronios_camada_escondida: int, f_ativacao: str,
                    taxa_de_aprendizagem=0.001, epocas=1000):
    
    atributos,items = x_treino.shape

    # Inicialização dos pesos e biases
    W1 = pesos(atributos, neuronios_camada_escondida)  # Pesos para a camada oculta
    B1 = biases(neuronios_camada_escondida)  # Biases para a camada oculta
    W2 = pesos(neuronios_camada_escondida, 1)  # Pesos para a camada de saída
    B2 = biases(1)  # Biases para a camada de saída

    erro_medio_treino = []
    erro_medio_validacao = []
    for epoca in range(epocas):
        erros =[]
        for item in range (items):
            # Propagação direta
            S2 = propagacao_direta(x_treino[:,item].reshape(-1,1), W1, B1)
            Z2 = ativacao(S2, f_ativacao)  # Ativação na camada escondida
            S1 = propagacao_direta(Z2, W2, B2)
            Z1 = ativacao(S1, 'sigmoide')  # Ativação na camada de saída

            # Cálculo de erro na saída
            erro_atual = custo_emq(Z1, y_treino[item])
            erros.append(erro_atual)  # Atualiza a lista de erro Medio

            # Retropropagação na camada de saída
            d2 = delta_saida(Z1, y_treino[item])

            # Atualizar W2 e B2
            W2 -= taxa_de_aprendizagem * gradiente(d2,Z2)
            B2 -= taxa_de_aprendizagem * d2

            # Retropropagação na camada oculta
            d1 = delta_oculta(d2, W2, S2, f_ativacao)

            # Atualizar W1 e B1
            W1 -= taxa_de_aprendizagem * gradiente(d1, x_treino[:,item].reshape(-1,1))
            B1 -= taxa_de_aprendizagem * d1

        # Calcula o MSE para o lote do treino
        emq_treino = np.mean(erros)
        erro_medio_treino.append(emq_treino)  # Atualiza a lista de erros da saída

        # Calcular o erro nos dados de validacao
        emq_val = calc_erro_validacao(x_validacao, y_validacao, W1,B1, f_ativacao, W2, B2)
        erro_medio_validacao.append(emq_val)

        # Mostrar progresso do treinamento
        if epoca % (epocas // 10) == 0 or epoca == epocas - 1:        
            print(f'Época {epoca+1}/{epocas}, \t | Erro_tr: {emq_treino}, \t | Erro_val: {emq_val}')

        # Critério de parada: Validação cruzada
        if (epoca>1) and (erro_medio_validacao[-1] >= erro_medio_validacao[-2]):
            print('Treinamento finalizado por validação cruzada após {} épocas.'.format(epoca))
            break

    return erro_medio_treino, erro_medio_validacao, W1, B1, f_ativacao, W2, B2