'''
Este módulo contém funções de pré-processamento do dos dados
- Divisão da base em treino, validação e teste
- Normalização (min-max)
'''

import numpy as np
import pandas as pd


# Divisao dos dados em 60-20-20
'''CODE HERE'''



# Normalizacao dos dados de entrada
#Normalização entre 0 e 1
def min_max_normalizacao(df):
    df_normalizado = df.copy()
    for i in df_normalizado.columns:
        min_val = df_normalizado[i].min()
        max_val = df_normalizado[i].max()
        df_normalizado[i] = (df_normalizado[i] - min_val) / (max_val - min_val)
    return df_normalizado