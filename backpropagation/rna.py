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
                    tamanho_do_lote = 100, taxa_de_aprendizagem = 0.01, epocas = 2000):
    
    assert epocas >= 10, 'O número de épocas não pode ser inferior a 10'
    assert tamanho_do_lote > 0, 'O tamanho do lote não pode ser zero ou negativo'

    atributos,items = x_treino.shape

    W2 = pesos(atributos, neuronios_camada_escondida)  # Pesos para a camada oculta
    B2 = biases(neuronios_camada_escondida)  # Biases para a camada oculta
    W1 = pesos(neuronios_camada_escondida, 1)  # Pesos para a camada de saída
    B1 = biases(1)  # Biases para a camada de saída

    erro_medio_treino = []
    erro_medio_validacao = []
    print(f"{'Época':^15} {'Erro_treino':<23} {'Erro_validacao':<23}")
    print('-' * 60)

    for epoca in range(epocas):
        erros =np.array([[]])
        item =0
        while item < items:
            # Propagação direta
            Z2, Z1, err = propagacao_direta(x_treino[:, item: item+tamanho_do_lote],
                                            y_treino[:, item: item+tamanho_do_lote],
                                            W2, B2, f_ativacao, W1, B1)
            erros = np.concatenate((erros,err),axis=1) # Atualiza a lista de erro Medio

            # RETROPROPAGAÇÃO DO ERRO
            # Retropropagação na camada de saída
            d1 = delta_saida(Z1, y_treino[:, item: item+tamanho_do_lote])
            W1 -= taxa_de_aprendizagem * gradiente(d1,Z1)
            B1 -= taxa_de_aprendizagem * np.sum(d1, axis=1, keepdims=True)

            # Retropropagação na camada oculta
            d2 = delta_oculta(d1, W1, Z2, f_ativacao)
            W2 -= taxa_de_aprendizagem * gradiente(d2, x_treino[:, item: item+tamanho_do_lote])
            B2 -= taxa_de_aprendizagem * np.sum(d2, axis=1, keepdims=True)

            item += tamanho_do_lote
 
        # Calcula o MSE para o lote do treino
        emq_treino = 0.5*np.mean(erros)
        erro_medio_treino.append(emq_treino)  # Atualiza a lista de erros da saída

        # Calcular o erro nos dados de validacao
        emq_val = calc_erro_validacao(x_validacao, y_validacao,
                                      W2, B2, f_ativacao, W1, B1,
                                      tamanho_do_lote)

        erro_medio_validacao.append(emq_val) # Atualiza a lista de erros de validacao

        # Mostrar progresso do treinamento
        if epoca % (epocas // 10) == 0 or epoca == epocas - 1:    
            print(f"{epoca:>6}/{epocas:<8} {emq_treino:<23} {emq_val:<23}")

        # Critério de parada: Validação cruzada
        if (epoca>1) and (erro_medio_validacao[-1] >= erro_medio_validacao[-2]):
            print('Treinamento finalizado por validação cruzada após {} épocas.'.format(epoca))
            break

    return erro_medio_treino, erro_medio_validacao, W2, B2, f_ativacao, W1, B1
#  