
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import json
import pickle


def get_cat_feat(data):
    with open('cat_feat.yaml', 'r', encoding='utf-8') as fs:
        line = fs.readline()
    return line.split(',')

def get_model(model_name):
    dictionary = {'logreg' : 'LR_CLF.pkl',
                  'svc_lin' : 'SVC_LIN_CLF.pkl',
                  'svc_rbf': 'SVC_RBF_CLF.pkl',
                  'random_forest' : 'RF_CLF.pkl',
                  'catboost' : 'CB_CLF.pkl',
                  'xgboost' : 'XGB_CLF.pkl',
                  'lightgbm' : 'LGBM_CLF.pkl',
                  'staking' : 'STACK_CLF.pkl'}
    return  pickle.load(open('models_pkl/' + dictionary.get(model_name), 'rb'))


def get_parms(model_name):
    dictionary = {'logreg': 'logreg.json',
                  'svc_lin': 'svc_lin.json',
                  'svc_rbf': 'svc_rbf.json',
                  'catboost': 'catboost.json',
                  'xgboost': 'xgboost.json',
                  'lightgbm': 'lightgbm.json',
                  'lgmb': 'Slgbm.json',
                  'cb': 'Scb.json',
                  'xgb': 'Sxgb.json'}
    with open('models_parms/' + dictionary.get(model_name), 'r', encoding='utf-8') as fh:
        data = json.load(fh)
    return data


def get_stacking_estimators():
    lgbm_clf = LGBMClassifier(**get_parms('lgbm'))
    cb_clf = CatBoostClassifier(**get_parms('cb'))
    xgb_clf = XGBClassifier(**get_parms('xgb'))
    estimators = [('lgbm', lgbm_clf), ('cb', cb_clf), ('xgb', xgb_clf)]
    return estimators

def model_choice(model_name, data):
    if model_name == 'staking':
        return StackingClassifier(estimators=get_stacking_estimators())

    parms = get_parms(model_name)
    dictionary = {'logreg' : LogisticRegression(**parms),
                  'svc_lin' : SVC(**parms),
                  'svc_rbf': SVC(**parms),
                  'catboost' : CatBoostClassifier(**parms, cat_features= get_cat_feat(data)),
                  'xgboost' : XGBClassifier(**parms),
                  'lightgbm' : LGBMClassifier(**parms),
                  'staking' : 6}

    return dictionary.get(model_name)