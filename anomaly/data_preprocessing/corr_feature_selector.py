from sklearn.base import TransformerMixin, BaseEstimator


# CorrFeatureSelector удаляет колонки, у которых корреляция больше max_corr
# (в нашем случае зададим max_corr = 0.9)
class CorrFeatureSelector(TransformerMixin, BaseEstimator):

    def __init__(self, max_corr):
        self.cols_to_remove = set()
        self.max_corr = max_corr

    def fit(self, X, y=None):
        self.cols_to_remove = set()
        corrs = X.corr()
        i = 0
        for col_name, col_corrs in corrs.iterrows():
            if col_name not in {'TransactionID', 'isFraud', 'TransactionDT'} and col_name not in self.cols_to_remove:
                big_corr_columns_names = set(map(
                    lambda item: item[0],
                    list(filter(
                        lambda item: (abs(item[1]) > 0.9),
                        col_corrs[i + 1:].items()))))
                self.cols_to_remove.update(big_corr_columns_names)
            i += 1
        return self

    def transform(self, X):
        return X.drop(columns=self.cols_to_remove)
