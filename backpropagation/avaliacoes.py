'''
Este módulo contém funções de avaliação do modelo
- Matriz de confusão
- Acurácia
- Gráfico do erro médio quadrático ao longo do treinamento
'''

import matplotlib.pyplot as plt
import numpy as np
from .camadas import propagacao_direta, ativacao



# Matriz de confusao
def matriz_confusao(y_real, y_prev):
    assert len(y_real)==len(y_prev), \
        'Verifique as dimensões dos dados de saída real e saída prevista'
    c_matrix = np.zeros((2,2), dtype=np.int32)
    labels =[0,1]
    count=0
    for _, _ in zip(y_real, y_prev):
        real = y_real[count]
        prev = y_prev[count]
        c_matrix[real, prev]+=1
        count+=1

    for i in range(c_matrix.shape[0]):
        for j in range(c_matrix.shape[1]):
            text = plt.text(j,i, c_matrix[i,j], color="red")

    x_ticks = labels
    y_ticks = labels

    plt.imshow(c_matrix)
    plt.xticks(ticks=x_ticks,labels=[0,1], rotation=0)
    plt.yticks(ticks=y_ticks,labels=[0,1])
    
    plt.title("Matriz de confusão")
    plt.xlabel("Previsao do modelo")
    plt.ylabel("Valores verdadeiros")
    plt.show()  



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


# fazer previsoes
def previsao(modelo_treinado, x_test:np.ndarray):
    y_prev = []
    val_items = x_test.shape[1]

    pesos1, bias1  = modelo_treinado[2], modelo_treinado[3]
    f_ativ = modelo_treinado[4]
    pesos2, bias2  = modelo_treinado[5], modelo_treinado[6]

    for item in range(val_items):
    
        S1_val = propagacao_direta(x_test[:,item].reshape(-1,1), pesos1, bias1)
        Z1_val = ativacao(S1_val, f_ativ)
        S2_val = propagacao_direta(Z1_val, pesos2, bias2)
        Z2_val = ativacao(S2_val, 'sigmoide')
        y_prev.append(1) if Z2_val>0.5 else y_prev.append(0)

    return np.array(y_prev)