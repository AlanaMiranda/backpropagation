# backpropagation
Multilayer Perceptron and the Backpropagation algorithm - Artificial Neural Networks discipline

1. Objetivos:

▪ Implementar algoritmo Backpropagation.

▪ Aplicar algoritmo em um problema de classificação.

2. Metodologia:

  2.1. Criar um pseudocódigo para o algoritmo Backpropagation.
  
  2.2. Implementar o algoritmo Backpropagation padrão em qualquer linguagem de programação.
  
  2.3. Critério de parada: validação cruzada/parada antecipada (60% dos dados para treino, 20%
dos dados para validação e 20% dos dados para teste). Parada antecipada após verificação
de erro médio quadrático na base de validação aumentar após 5 épocas consecutivas.

  2.4. Apresentar gráfico de evolução do erro médio quadrático ao longo das épocas (para dados
de treino e dados de validação no mesmo gráfico).

  2.5. Algoritmo deve possibilitar variar número de neurônios na camada escondida e funções de
ativação para os neurônios (usar apenas uma camada escondida).

  2.6. Normalização dos dados de entrada.
  
  2.7. Possibilidade de verificar valores dos pesos sinápticos antes e após finalização do
treinamento.


# Instruções de uso
1. Importe os dados com _m_ items e _n_ atributos.
2. Coloque os dados no formato matricial.
3. Normalize os dados e divida em treino, validação e teste usando a função 'normalizar'.
4. Treine o modelo usando a função 'backpropagation'.
5. Visualize os erro médio ao longo do treinamento usando a função 'grafico'.
6. Visualize a acurácia usando a função 'acuracia'.


## Pre processamento (normalização)

Importe os dados e transforme para uma forma matricial. Separe as colunas da entrada e saída.

```Python
import pandas as pd

# Importar e processar os dados
df = pd.read_excel('https://raw.githubusercontent.com/AlanaMiranda/backpropagation/main/dadosmamografia.xlsx', header=None)

Xt=df.values[:,:5] # separar dados de entrada

yt=df.values[:,-1] # separar dados de saida
yt=yt.reshape(-1,1)
```

Use a função 'normalizar' para dividir em treino, validação, teste e para normalizar os dados.\
Por padrão, a função divide na proporção 60% - 20% -20%. Essa proporção pode ser alterada.\
A função normaliza, por padrão, no intervalo $[0;1]$. Esse intervalo pode ser alterado.

#### Sintaxe:
```Python

normalizar(entrada, saida, tr, val, tst, inf, sup)

```

#### Argumentos:
* **entrada** - Tipo _array_. Dados de entrada no formato $(m,n)$.
* **saida** - Tipo _array_. Dados de saida no formato $(m,1)$.
* **tr** - (Opcional). Tipo _float_. Padrão = 0.6. Proporção para os dados de treino.
* **val** - (Opcional). Tipo _float_. Padrão = 0.2. Proporção para os dados de validação.
* **tst** - (Opcional). Tipo _float_. Padrão = 0.2. Proporção para os dados de teste.
* **inf** - (Opcional). Tipo _float_. Padrão = 0. Limite inferior da normalização.
* **sup** - (Opcional). Tipo _float_. Padrão = 1. Limite superior da normalização.

#### Retorna:
Uma tupla contendo três tuplas de treino validação e teste:
- (x_treino, y_treino),
- (x_validacao, y_validacao),
- (x_teste, y_teste)

#### Exemplo:
```Python
from backpropagation.preprocessamento import normalizar

(x_treino, y_treino), (x_validacao, y_validacao), (x_teste, y_teste) = normalizar(Xt, yt)

# Fazer transposta nos dados de entrada e saida
x_treino = x_treino.T
x_validacao = x_validacao.T
x_teste = x_teste.T

y_treino = y_treino.T
y_validacao = y_validacao.T
y_teste = y_teste.T
```

## Treinamento do modelo
Use a função 'backpropagation' para treinar o modelo.

#### Sintaxe:
```Python
backpropagation(x_treino, y_treino, x_validacao, y_validacao, neuronios_camada_escondida,
                f_ativacao, tamanho_do_lote, taxa_de_aprendizagem, epocas)
```

#### Argumentos:
* **x_treino** - Tipo _array_, dimensões _(n,m)_. Dados de treino para entrada da rede neural.
* **y_treino** - Tipo _array_, dimensões _(m,1)_. Dados de treino para saida da rede neural.
* **x_validacao** - Tipo _array_, dimensões _(n,m)_. Dados de treino para entrada da rede neural.
* **y_validacao** - Tipo _array_, dimensões _(m,1)_. Dados de treino para saida da rede neural.
* **neuronios_camada_escondida** - Tipo _int_. Número de neurônios na camada escondida.
* **f_ativacao** - Tipo _str_. Funções de ativação na camada escondida. As funções podem ser 'relu', 'tanh' ou 'sigmoide'.
* **tamanho_do_lote** (Opcional)- Tipo _int_. Padrão = 100.  Número de amostras por treinamento.
* **taxa_de_aprendizagem** (Opcional)- Tipo _bool_. Padrão = 0.0001.  Taxa de aprendizagem para atualização dos parâmetros.
* **epocas** (Opcional) - Tipo _int_. Padrão = 1000.  Número de épocas de treinamento. O número de épocas não pode ser inferior a 10.

#### Retorna:
* **erro_medio_treino** - Tipo _list_. Lista de erros do treinamento.
* **erro_medio_validacao** - Tipo _list_. Lista de erros da validação.
* **W1** - Tipo _array_. Pesos da primeira camada.
* **B1** - Tipo _array_. Biases da primeira camada.
* **f_ativacao** - Tipo _str_. A função de ativação da camada oculta.
* **W2** - Tipo _array_. Pesos da segunda camada.
* **B2** - Tipo _array_. Biases da segunda camada.

#### Exemplo:
Criar uma variável e chamar a função.
```Python
from backpropagation.rna import backpropagation

# Executar a função
treinar = backpropagation(x_treino, y_treino, 
                          x_validacao, y_validacao,
                          15, 'sigmoide')

```



## Plotar gráfico
Para plotar gráficos importamos a função gráfico do módulo avaliações e passamos como parâmetro o modelo treinado:

#### Sintaxe:
```Python
grafico(modelo_treinado, cor)
```

#### Argumentos:
* **modelo_treinado** - Tipo _tuple_. Variável que é retornada pela função backpropagation.
* **cor** - (Opcional). Tipo _list_. Padrão=['green', 'orange']. Lista de cores  para os gráficos.

#### Retorna:
* Gráfico do erro médio quadrático para os dados de treino e da validação cruzada. 

#### Exemplo:
```Python
from backpropagation.avaliacoes import grafico

# Plotar o gráfico
grafico(treinar, ['orange', 'magenta'])

```

## Fazer previsões
Podemos usar o modelo treinado para fazer previsões no conjunto de teste.

#### Sintaxe:
```Python
previsao(modelo_treinado, x_teste)
```

#### Argumentos:
* **modelo_treinado** - Tipo _tuple_. Variável que é retornada pela função backpropagation.
* **x_teste** - Tipo _array_. Dados do conjunto de teste na dimensão $(n,m)$.

#### Retorna:
* **y_prev** - Tipo _array_. Lista com saída das previsões do modelo.

#### Exemplo:
```Python
from backpropagation.avaliacoes import previsao

# Fazer previsões
y_prev = previsao(treinar, x_teste)
```

## Matriz de confusão
A matriz de confusão mostra as previsões do modelo para cada classe.

#### Sintaxe:
```Python
matriz_confusao(y_real, y_prev)

```

#### Argumentos:
* **y_real** - Tipo _array_. Lista com saída do grupo de teste.
* **y_prev** - Tipo _array_. Lista de saída das previsões do modelo.

#### Retorna:
* Matriz de confusão.

#### Exemplo:
```Python
from backpropagation.avaliacoes import matriz_confusao

# Matriz de confusão
matriz_confusao(y_teste, y_prev)
```



## Acurácia
A acurácia mostra a percentagem de acertos do modelo.

#### Sintaxe:
```Python
acuracia(y_real, y_prev)
```

#### Argumentos:
* **y_real** - Tipo _array_. Lista com saída do grupo de teste.
* **y_prev** - Tipo _List_. Lista de saída das previsões do modelo.

#### Retorna:
* Acurácia - Tipo _float_.

#### Exemplo:
```Python
from backpropagation.avaliacoes import acuracia

# Acurácia
acuracia(y_teste, y_prev)
```

