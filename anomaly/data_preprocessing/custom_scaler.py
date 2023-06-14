from sklearn.base import TransformerMixin, BaseEstimator
from sklearn import preprocessing


# Масштабирование числовых признаков
class CustomScaler(TransformerMixin, BaseEstimator):

    def __init__(self, cols, scaler=None):
        self.cols = cols
        self.scaler = scaler or preprocessing.MinMaxScaler()

    def fit(self, X, y=None):
        self.cols = list(set(self.cols).intersection(set(X.columns)))
        num_cols = X[self.cols]
        self.scaler.fit(num_cols)
        return self

    def transform(self, X, y=None):
        X_res = X.copy()
        X_res[self.cols] = self.scaler.transform(X_res[self.cols])
        return X_res
