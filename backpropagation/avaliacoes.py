'''
Este módulo contém funções de avaliação do modelo
- Matriz de confusão
- Acurácia
- Gráfico do erro médio quadrático ao longo do treinamento
'''

import matplotlib.pyplot as plt
import numpy as np
from .camadas import propagacao_direta



# Matriz de confusao
def matriz_confusao(y_real, y_prev):
    assert y_real.shape == y_prev.shape, \
        'Verifique as dimensões dos dados de saída real e saída prevista'

    # Inicializa a matriz de confusão
    c_matrix = np.zeros((2, 2), dtype=np.int32)

    # Incrementa a matriz de confusão com base nos valores reais e previstos
    for real, prev in zip(y_real[0], y_prev[0]):
        c_matrix[real, prev] += 1

    # Configuração do gráfico
    plt.figure(figsize=(6, 6))
    plt.imshow(c_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Matriz de Confusão")
 
    # Definição de rótulos e ticks
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, [0, 1])
    plt.yticks(tick_marks, [0, 1])
    
    # Adição dos valores nas células da matriz
    thresh = c_matrix.max() / 2.
    for i, j in np.ndindex(c_matrix.shape):
        plt.text(j, i, format(c_matrix[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if c_matrix[i, j] > thresh else "black")
    
    # Labels dos eixos
    plt.ylabel('Valores Verdadeiros')
    plt.xlabel('Previsão do Modelo')
    plt.tight_layout()
    plt.show()



# Acuracia
def acuracia(y_real, y_prev):
    assert y_real.shape == y_prev.shape, \
        'Verifique as dimensões dos dados de saída real e saída prevista'
    
    # Calcula a acurácia como a razão de previsões corretas
    acc = np.sum(y_real == y_prev) / len(y_real[0])
    return acc



# Plotar o gráfico
def grafico (modelo_treinado, cores:list=['green','orange'])->plt.figure:

    curva_tr = modelo_treinado[0]
    curva_val = modelo_treinado[1]

    plt.plot(np.arange(0,len(curva_tr)), curva_val, color=cores[1], label='Validação')
    plt.plot(np.arange(0,len(curva_tr)), curva_tr, color=cores[0], label='Treino')
    plt.title ('Desempenho do modelo ao longo do treinamento')
    plt.ylabel("Erros na saída")
    plt.xlabel("Épocas")
    plt.legend()
    plt.show()



# Fazer previsoes
def previsao(modelo_treinado:tuple, x_test:np.ndarray,
             tamanho_do_lote:int=1)-> np.ndarray:

    pesos2, bias2  = modelo_treinado[2], modelo_treinado[3]
    f_ativ = modelo_treinado[4]
    pesos1, bias1  = modelo_treinado[5], modelo_treinado[6]

    y_prev =np.array([[]])
    item = 0
    while item < x_test.shape[1]:
        _, ativacao_saida, _ = propagacao_direta(
                                    x_test[:, item: item+tamanho_do_lote],
                                    x_test[:, item: item+tamanho_do_lote],
                                    pesos2, bias2, f_ativ, pesos1, bias1)

        y_prev = np.concatenate((y_prev,ativacao_saida),
                                axis=1) # Atualiza a lista de previsoes
        item += tamanho_do_lote
    
    y_prev = np.where(y_prev>0.5,1,0)

    return y_prev