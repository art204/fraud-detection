import pandas as pd
from collections import OrderedDict
from sklearn.preprocessing import OneHotEncoder
from category_encoders.target_encoder import TargetEncoder


# Кодирование категориальных признаков
class ObjectEncoder:
    def __init__(self, ohe_limit, obj_cols):
        self.__ohe_limit = ohe_limit
        self.__obj_cols = obj_cols
        self.__ohe_cols = []
        self.__targ_enc_cols = []
        self.__ohe_enc = None
        self.__targ_enc = None

    def split_columns(self, df):
        for col in self.__obj_cols:
            if col in df.columns:
                if col in ['P_emaildomain', 'R_emaildomain'] or df[col].unique().shape[0] <= self.__ohe_limit:
                    self.__ohe_cols.append(col)
                else:
                    self.__targ_enc_cols.append(col)

    def add_ohe_cols_in_df(self, df, cat_cols):
        transformed = self.__ohe_enc.transform(df[cat_cols].astype(str)).toarray()
        ordered_dict = OrderedDict()
        transformed_start = 0
        transformed_end = 0
        for i in range(len(cat_cols)):
            col_name = cat_cols[i]
            categories = col_name + '_' + self.__ohe_enc.categories_[i]
            transformed_end += len(categories)
            for j in range(1, len(categories)):
                ordered_dict[categories[j]] = transformed[:, transformed_start + j]
            transformed_start += len(categories)
        df = pd.concat([df, pd.DataFrame(ordered_dict)], axis=1)
        df.drop(columns=cat_cols, inplace=True)
        return df

    def fit(self, X, y):
        self.split_columns(X)
        self.__targ_enc = TargetEncoder(cols=self.__targ_enc_cols)
        self.__targ_enc.fit(X, y)
        self.__ohe_enc = OneHotEncoder()
        self.__ohe_enc.fit(X[self.__ohe_cols].astype(str))

    def transform(self, X):
        X = self.__targ_enc.transform(X)
        return self.add_ohe_cols_in_df(X, self.__ohe_cols)

    def fit_transform(self, X, y):
        self.split_columns(X)
        self.__targ_enc = TargetEncoder(cols=self.__targ_enc_cols)
        X = self.__targ_enc.fit_transform(X, y)
        self.__ohe_enc = OneHotEncoder()
        self.__ohe_enc.fit(X[self.__ohe_cols].astype(str))
        return self.add_ohe_cols_in_df(X, self.__ohe_cols)
