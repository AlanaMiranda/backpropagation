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
                    tamanho_do_lote = 100, taxa_de_aprendizagem = 0.01,
                    epocas = 2000, parada_antecipada=5) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    assert epocas >= 10, 'O número de épocas não pode ser inferior a 10'
    assert tamanho_do_lote > 0, 'O tamanho do lote não pode ser zero ou negativo'

    atributos,items = x_treino.shape

    # INICIALIZACAO ARBITRARIA DOS PESOS E BIAS 
    W2 = pesos(atributos, neuronios_camada_escondida)  # Pesos para a camada oculta
    B2 = biases(neuronios_camada_escondida)  # Biases para a camada oculta
    W1 = pesos(neuronios_camada_escondida, 1)  # Pesos para a camada de saída
    B1 = biases(1)  # Biases para a camada de saída


    # SALVAR OS PESOS INICIAIS 
    pesos_iniciais = (W2.copy(), B2.copy(), W1.copy(), B1.copy())
    

    # INCIALIZACAO DE VARIVEIS
    erro_medio_treino = [] # Lista de erros no treinamento
    erro_medio_validacao = [] # Lista de erros na validacao
    epocas_sem_alteracao = 0 # numero de epocas sem altercao no erro
    menor_erro = float('inf') # Inicializacao do erro com o maior valor possivel


    # MOSTRA O CABECALHO DO PROGRESSO DO TREINAMENTO
    print(f"{'Época':^15} {'Erro_treino':<23} {'Erro_validacao':<23}")
    print('-' * 60)


    # TREINAMENTO POR EPOCA 
    for epoca in range(epocas):

        erros = np.array([[]]) # Inicializamos a lista de erros 
        item = 0 # Contador de iteracoes para amostras de treinamento

        # TREINAMENTO POR LOTE
        while item < items:

            # PROPAGACAO DIRETA
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

            item += tamanho_do_lote # pula para lote seguinte

        
        # CALCULAR O ERRO
        # Calcula o MSE para o lote do treino
        emq_treino = 0.5*np.mean(erros)
        erro_medio_treino.append(emq_treino)  # Atualiza a lista de erros da saída

        # Calcular o erro nos dados de validacao
        emq_val = calc_erro_validacao(x_validacao, y_validacao,
                                      W2, B2, f_ativacao, W1, B1,
                                      tamanho_do_lote)

        erro_medio_validacao.append(emq_val) # Atualiza a lista de erros de validacao


        # MOSTRAR PROGRESSO DO TREINAMENTO
        if epoca % (epocas // 10) == 0 or epoca == epocas - 1:    
            print(f"{epoca:>6}/{epocas:<8} {emq_treino:<23} {emq_val:<23}")


        # CRITÉRIO DE PARADA: Validação cruzada
        # Verifica se o erro medio quadratico atual é menor que o menor erro inicializado
        if emq_val < menor_erro:
            
            # Atualiza menor erro 
            menor_erro = emq_val

            # Reseta o numero de epocas sem alteracao no erro
            epocas_sem_alteracao = 0

            # Guarda os melhores parametros para essa epoca
            pesos_finais = (W2.copy(), B2.copy(), f_ativacao, W1.copy(), B1.copy())

        else:
            # Incrementa o numero de epocas sem alteracao no erro
            epocas_sem_alteracao += 1

            if epocas_sem_alteracao == parada_antecipada:
                print(f'Treinamento finalizado por validação cruzada após {epoca} épocas.')
                break


    return erro_medio_treino, erro_medio_validacao, pesos_iniciais, pesos_finais