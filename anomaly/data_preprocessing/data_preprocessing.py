import pickle
from sklearn import preprocessing
from sklearn.pipeline import Pipeline

from anomaly.data_preprocessing.corr_feature_selector import CorrFeatureSelector
from anomaly.data_preprocessing.custom_imputer import CustomImputer
from anomaly.data_preprocessing.custom_scaler import CustomScaler
from anomaly.data_preprocessing.nan_feature_selector import NanFeatureSelector
from anomaly.data_preprocessing.object_encoder import ObjectEncoder


def get_real_cols(data, cat_cols):
    real_cols = list(data.columns)

    for col in cat_cols:
        if col in real_cols:
            real_cols.remove(col)
    return real_cols


def get_cat_cols():
    with open('../cat_feat.yaml', 'r', encoding='utf-8') as fs:
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
            if name == 'NanFeatureSelector':
                module = "anomaly.data_preprocessing.nan_feature_selector"
            elif name == 'CorrFeatureSelector':
                module = "anomaly.data_preprocessing.corr_feature_selector"
            elif name == 'CustomImputer':
                module = "anomaly.data_preprocessing.custom_imputer"
            elif name == 'ObjectEncoder':
                module = "anomaly.data_preprocessing.object_encoder"
            elif name == 'CustomScaler':
                module = "anomaly.data_preprocessing.custom_scaler"
        return super().find_class(module, name)


def get_prep_data_pipe_from_pkl():
    pipeline = None
    with open('anomaly/models_pkl/PREP_DATA_PIPE.pkl', 'rb') as f:
        unpickler = CustomUnpickler(f)
        pipeline = unpickler.load()
    return pipeline
