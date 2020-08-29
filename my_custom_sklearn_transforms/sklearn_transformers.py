from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        data.drop(labels=self.columns, axis='columns', inplace=True)
        # Remover 4000 registros aleatórios de perfil DIFICULDADE
        drop_indices_dificuldade = np.random.choice(data[data['PERFIL'] == 'DIFICULDADE'].index, 4000, replace=False)
        data.drop(drop_indices_dificuldade, inplace=True)
        return data
