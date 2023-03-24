import pickle

import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.impute import SimpleImputer
from category_encoders.target_encoder import TargetEncoder
from sklearn import preprocessing
from sklearn.pipeline import Pipeline


# NanFeatureSelector удаляет колонки, в которых доля пропусков больше max_nan_rate
# (в нашем случае зададим max_nan_rate = 85%)
class NanFeatureSelector(TransformerMixin, BaseEstimator):

    def __init__(self, max_nan_rate):
        self.cols_to_remove = []
        self.max_nan_rate = max_nan_rate

    def fit(self, X, y=None):
        nan_stat = self.get_share_of_NaN(X)
        for i in range(len(nan_stat)):
            column = nan_stat.loc[i]
            if (column['Share_of_NaN'] > self.max_nan_rate):
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


# CorrFeatureSelector удаляет колонки, у которых корреляция больше max_corr
# (в нашем случае зададим max_corr = 0.9)
class CorrFeatureSelector(TransformerMixin, BaseEstimator):

    def __init__(self, max_corr):
        self.cols_to_remove = set()
        self.max_corr = max_corr

    def fit(self, X, y=None):
        corrs = X.corr()
        cols = corrs.columns
        for i in range(len(cols)):
            col_name_1 = cols[i]
            if col_name_1 in {'TransactionID', 'isFraud', 'TransactionDT'} or col_name_1 in self.cols_to_remove:
                continue
            for j in range(i + 1, len(cols)):
                col_name_2 = cols[j]
                if abs(corrs[col_name_1][col_name_2]) > self.max_corr:
                    self.cols_to_remove.add(col_name_2)
        return self

    def transform(self, X):
        return X.drop(columns=self.cols_to_remove)


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


# Кодирование категориальных признаков
class ObjectEncoder:
    def __init__(self, ohe_limit, obj_cols):
        self.__ohe_limit = ohe_limit
        self.__obj_cols = obj_cols
        self.__encoded_columns = None
        self.__ohe_cols = []
        self.__targ_enc_cols = []
        self.__targ_enc = None

    def divide_columns(self, df):
        for col in self.__obj_cols:
            if col in df.columns:
                if col in ['P_emaildomain', 'R_emaildomain'] or df[col].unique().shape[0] <= self.__ohe_limit:
                    self.__ohe_cols.append(col)
                else:
                    self.__targ_enc_cols.append(col)

    def encode_ohe_cols(self, df):
        # use OHE for columns from self.__ohe_cols only
        if self.__encoded_columns is not None:
            df = pd.get_dummies(df, drop_first=False, columns=self.__ohe_cols)

            removed_cols = []
            for col in df.columns:
                if col not in self.__encoded_columns:
                    removed_cols.append(col)
            df.drop(columns=removed_cols, inplace=True)

            for i in range(len(self.__encoded_columns)):
                col = self.__encoded_columns[i]
                if col not in df:
                    df.insert(i, col, 0)

        else:
            df = pd.get_dummies(df, drop_first=True, columns=self.__ohe_cols)
            self.__encoded_columns = df.columns

        return df

    def fit(self, X, y):
        self.divide_columns(X)
        self.__targ_enc = TargetEncoder(cols=self.__targ_enc_cols)
        self.__targ_enc.fit(X, y)

    def transform(self, X):
        X = self.__targ_enc.transform(X)
        return self.encode_ohe_cols(X)

    def fit_transform(self, X, y):
        self.divide_columns(X)
        self.__targ_enc = TargetEncoder(cols=self.__targ_enc_cols)
        X = self.__targ_enc.fit_transform(X, y)
        return self.encode_ohe_cols(X)


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


def get_real_cols(data, cat_cols):
    real_cols = list(data.columns)

    for col in cat_cols:
        if col in real_cols:
            real_cols.remove(col)
    return real_cols


def get_cat_cols():
    with open('cat_feat.yaml', 'r', encoding='utf-8') as fs:
        line = fs.readline()
    return line.split(',')


def get_coding_pipeline(data):
    cat_cols = get_cat_cols()
    real_cols = get_real_cols(data, cat_cols)

    prep_data_pipe = Pipeline([
        ('nan_feature_selector_', NanFeatureSelector(0.85)),
        ('corr_feature_selector_', CorrFeatureSelector(0.9)),
        ('imputer_', CustomImputer(strategy='constant', fill_value=-999)),
        ('encoder_', ObjectEncoder(10, cat_cols)),
        ('scaler_', CustomScaler(real_cols, preprocessing.MinMaxScaler()))
    ])

    return prep_data_pipe


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            module = "anomaly.feature_selection"
        return super().find_class(module, name)


def get_prep_data_pipe_from_pkl():
    pipeline = None
    with open('anomaly/models_pkl/PREP_DATA_PIPE.pkl', 'rb') as f:
        unpickler = CustomUnpickler(f)
        pipeline = unpickler.load()
    return pipeline
