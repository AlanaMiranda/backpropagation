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

2. Coloque os dados no formato matricial e divida em treino, validação e teste.

3. Importe a função backpropagation.


#### Sintaxe:

```Pyhton
from backpropagation.rna import backpropagation

backpropagation(x_treino, y_treino, neuronios_camada_escondida,
                f_ativacao, taxa_de_aprendizagem=0.0001, epocas=1500)
```

#### Parâmetros:
* **x_treino** - tipo array, dimensoes _(n,m)_. Dados de treino para entrada da rede neural.
* **y_treino** - tipo array, dimensoes _(m,1)_. Dados de treino para saida da rede neural.
* **x_validacao _(por implementar)_** - tipo array, dimensoes _(n,m)_. Dados de treino para entrada da rede neural.
* **y_validacao _(por implementar)_** - tipo array, dimensoes _(m,1)_. Dados de treino para saida da rede neural.
* **neuronios_camada_escondida** - tipo int,. Número de neurônios na camada escondia.
* **f_ativacao** - tipo str. Funções de ativação na camada escondida. As funções podem ser 'relu', 'tanh' ou 'sigmoide'.
* **taxa_de_aprendizagem** - tipo bool, padrão 0.0001. (Opcional) Taxa de aprendizagem para atualização dos parâmetros.
* **epocas** - tipo int, padrão 1500. (Opcional) Números de épocas de treinamento.


#### Retorna:
* **erros** - tipo list. Lista de erros ao longo do treinamento.
* **W1** - tipo array. Pesos da primeira camada.
* **B1** - tipo array. Biases da primeira camada.
* **W2** - tipo array. Pesos da segunda camada.
* **B2** - tipo array. Biases da segunda camada.


#### Exemplo:
```Pyhton
# Importar e pre-processar os dados
df = pd.read_excel('https://raw.githubusercontent.com/AlanaMiranda/backpropagation/main/dadosmamografia.xlsx', header=None)
Xt=df.values[:,:5]
Xt=Xt.T # Tem que fazer a transposta da matriz de entrada
yt=df.values[:,-1]
yt=yt.reshape(-1,1) 

# Executar a função
backpropagation(Xt, yt, 8, 'sigmoide', 0.00015, 1000)
```


