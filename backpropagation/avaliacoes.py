'''
Este módulo contém funções de avaliação do modelo:
- Matriz de confusão;
- Acurácia
- Gráfico do erro médio quadrático ao longo do treinamento
'''

import numpy as np
import matplotlib.pyplot as plt


# Matriz de confusao
def matriz_confusao(y_real, y_prev):
    assert len(y_real)==len(y_prev), \
        'Verifique as dimensões dos dados de saída real e saída prevista'
    c_matrix = np.zeros((2,2), dtype=np.int32)

    count=0
    for _, _ in zip(y_real, y_prev):
        real = y_real[count]
        prev = y_prev[count]
        c_matrix[real, prev]+=1
        count+=1
    return c_matrix



# Acuracia
def acuracia(y_real, y_prev):
    assert len(y_real)==len(y_prev), \
        'Verifique as dimensões dos dados de saída real e saída prevista'
    acc = 0

    count=0
    for _, _ in zip(y_real, y_prev):
        real = y_real[count]
        prev = y_prev[count]
        if real == prev:
            acc+=1
        count+=1
    
    acc = acc/len(y_real)
    return acc



# Grafico
def grafico ():
    # CODE HERE
    pass
