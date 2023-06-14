from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.impute import SimpleImputer


# Заполнение пропусков
class CustomImputer(TransformerMixin, BaseEstimator):

    def __init__(self, strategy='most_frequent', fill_value=None):
        self.__imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
        self.strategy = strategy
        self.fill_value = fill_value

    def fit(self, X, y=None):
        self.__imputer.fit(X)
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X[X.columns] = self.__imputer.transform(X[X.columns])

        return X
