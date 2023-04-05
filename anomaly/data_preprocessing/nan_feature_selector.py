import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator


# NanFeatureSelector удаляет колонки, в которых доля пропусков больше max_nan_rate
# (в нашем случае зададим max_nan_rate = 85%)
class NanFeatureSelector(TransformerMixin, BaseEstimator):

    def __init__(self, max_nan_rate):
        self.cols_to_remove = []
        self.max_nan_rate = max_nan_rate

    def fit(self, X, y=None):
        nan_stat = self.get_share_of_NaN(X)
        for i, column in nan_stat.iterrows():
            if column['Share_of_NaN'] > self.max_nan_rate:
                self.cols_to_remove.append(column['Name'])
        return self

    def transform(self, X):
        return X.drop(columns=self.cols_to_remove)

    def get_share_of_NaN(self, df):
        '''
        Рассчитывает количество пропусков в каждой колонке, а также долю пропусков в каждой колонке
        Параметры:
        df - датафрейм
        Возвращает датафрейм, содержащий информацию о пропусках в каждой колонке датафрейма df
        '''
        result = pd.DataFrame(columns=['Name', 'Number_of_NaN', 'Share_of_NaN'])
        colcount = df.count()
        length = len(df)
        for col_name in colcount.keys():
            result.loc[len(result)] = [col_name, length - colcount[col_name], (length - colcount[col_name]) / length]
        return result
