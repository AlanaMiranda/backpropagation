'''
Módulo principal do algoritimo.
Contém a função principal backpropagation
'''

from backpropagation.avaliacoes import *
from backpropagation.funcoes import *
from backpropagation.camadas import *
from backpropagation.preprocessamento import *


# Funcao principal do algoritmo
def backpropagation (entrada_X:np.ndarray, saida_Y:np.ndarray,
                     neuronios_camada_escondida:int,ativ:str,
                     taxa_de_aprendizagem=0.01, epocas=5) -> np.ndarray:
    
   
    # Divisao dos dados em 60-20-20
    '''CODE HERE'''
    
    # Normalizacao dos dados de entrada
    x_train, y_train = min_max_normalizacao(entrada_X)
    x_validacao, y_validacao = min_max_normalizacao(entrada_X)
    x_teste, y_teste = min_max_normalizacao(entrada_X)
    
    # Inicializacao arbitraria de pesos e biases
    W1 = pesos(x_train.shape[0], neuronios_camada_escondida) # Pesos na camada 1
    B1 = biases(neuronios_camada_escondida) # Biases na camada escondida
    W2 = pesos(neuronios_camada_escondida,1) # Pesos na camada 2
    B2 = biases(1) # Biases na camada de saida
    erro_validacao = [np.array([1])]
    for _ in range(epocas):
        for i,j in zip(x_train, y_train):
            
            # Fazer propagacao direta
            S1 = propagacao_direta(x_train[i], W1, B1)
            Z1 = ativacao(S1, ativ) # Ativacao na camada escondida
            S2 = propagacao_direta(Z1, W2, B2)
            Z2 = ativacao(S2, 'sigmoide') # Ativacao na camada de saida

            # Calculo de erro na saida
            erro_atual = custo_emq(y_train[j], Z2)
        
            # Comparacao do erro para validacao cruzada
            if erro_atual > erro_validacao[-1]:
                break
            else:
                erro_validacao.append(erro_atual) # Atualiza a lista de erros da saida
                # Fazer o retropropagacao
                # Retropropagacao na segunda camada
                derivada_Erro_B2 = retropropagacao(S2)
                derivada_Erro_W2 = retropropagacao(W2)

                # Atualizar B2 e W2
                W2 = W2 - taxa_de_aprendizagem * derivada_Erro_W2
                B2 = B2 - taxa_de_aprendizagem * derivada_Erro_B2

                # Retropropagacao na primeira camada
                derivada_Erro_B1 = retropropagacao(S1)
                derivada_Erro_W1 = retropropagacao(W1)
                
                # Atualizar B1 e W1
                W1 = W1 - taxa_de_aprendizagem * derivada_Erro_W1
                B1 = B1 - taxa_de_aprendizagem * derivada_Erro_B1

       
    return Z2, np.array(erro_validacao), (W1, B1, W2, B2)