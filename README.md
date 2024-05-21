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


## Pre processamento

Importe os dados e transforme para uma forma matricial. Separe as colunas da entrada e saída.

```Python
# Importar e pre-processar os dados
import pandas as pd

df = pd.read_excel('https://raw.githubusercontent.com/AlanaMiranda/backpropagation/main/dadosmamografia.xlsx', header=None)

Xt=df.values[:,:5] # separar dados de entrada

yt=df.values[:,-1] # separar dados de saida
yt=yt.reshape(-1,1)
```


Use a função 'normalizar' para dividir em treino, validação, teste e para normalizar os dados.

Por padrão, a função divide na proporção 60% - 20% -20%. Essa proporção pode ser alterada. 

A função normaliza, por padrão, no intervalo $[0;1]$. Esse intervalo pode ser alterado.



#### Sintaxe:
```Python

normalizar(entrada, saida, tr, val, tst, inf, sup)

```

#### Parâmetros:
* **entrada** - Tipo _array_. Dados de entrada no formato $(m,n)$.
* **saida** - Tipo _array_. Dados de saida no formato $(m,n)$.
* **tr** - (Opcional). Tipo _float_. Padrão = 0.6. Proporção para os dados de treino.
* **val** - (Opcional). Tipo _float_. Padrão = 0.2. Proporção para os dados de validação.
* **tst** - (Opcional). Tipo _float_. Padrão = 0.2. Proporção para os dados de teste.
* **inf** - (Opcional). Tipo _float_. Padrão = 0. Limite inferior da normalização.
* **sup** - (Opcional). Tipo _float_. Padrão = 1. Limite superior da normalização.


#### Retorna:
Uma tupla contendo três tuplas:
- (x_treino, y_treino),
- (x_validacao, y_validacao),
- (x_teste, y_teste)


#### Exemplo:
```Python
from backpropagation.preprocessamento import normalizar

(x_treino, y_treino), (x_validacao, y_validacao), (x_teste, y_teste) = normalizar(Xt, yt)

# Fazer transposta da matriz de entrada
x_treino = x_treino.T
```

## Treinamento do modelo
Use a função 'backpropagation' para treinar o modelo.

#### Sintaxe:

```Python
backpropagation(x_treino, y_treino, neuronios_camada_escondida,
                f_ativacao, taxa_de_aprendizagem, epocas)
```

#### Parâmetros:
* **x_treino** - Tipo _array_, dimensões _(n,m)_. Dados de treino para entrada da rede neural.
* **y_treino** - Tipo _array_, dimensões _(m,1)_. Dados de treino para saida da rede neural.
* **x_validacao _(por implementar)_** - Tipo _array_, dimensões _(n,m)_. Dados de treino para entrada da rede neural.
* **y_validacao _(por implementar)_** - Tipo _array_, dimensões _(m,1)_. Dados de treino para saida da rede neural.
* **neuronios_camada_escondida** - Tipo _int_. Número de neurônios na camada escondida.
* **f_ativacao** - Tipo _str_. Funções de ativação na camada escondida. As funções podem ser 'relu', 'tanh' ou 'sigmoide'.
* **taxa_de_aprendizagem** (Opcional)- Tipo _bool_. Padrão = 0.0001.  Taxa de aprendizagem para atualização dos parâmetros.
* **epocas** (Opcional) - Tipo _int_. Padrão = 1000.  Número de épocas de treinamento.


#### Retorna:
* **erros** - Tipo _list_. Lista de erros ao longo do treinamento.
* **W1** - Tipo _array_. Pesos da primeira camada.
* **B1** - Tipo _array_. Biases da primeira camada.
* **W2** - Tipo _array_. Pesos da segunda camada.
* **B2** - Tipo _array_. Biases da segunda camada.



#### Exemplo:
Criar uma variável e chamar a função.
```Python
from backpropagation.rna import backpropagation

# Executar a função
treinar = backpropagation(x_treino, y_treino, 6, 'sigmoide', 0.00001, 1500)

```


## Plotar gráfico

Para plotar gráficos importamos a função gráfico do módulo avaliações e passamos como parâmetro o modelo treinado:
#### Sintaxe
```Python
grafico(modelo_treinado, cor)
```

#### Parâmetros:
* **modelo_treinado** - Tipo _tuple_. Variável que é retornada pela função backpropagation.
* **cor** - Tipo _str_. Cor do gráfico.


#### Exemplo:
```Python
# Importar a função
from backpropagation.avaliacoes import grafico

# Executar a função
grafico(treinar, 'orange')

```





