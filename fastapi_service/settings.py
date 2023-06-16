class Settings:
    '''
    Available models:
    Fully connected neural network: 'fcnn'
    Logistic regression: 'logreg'
    Boosting: 'catboost', 'xgboost'
    Stacking classifier with 3 estimators (CatBoost, XGB, LGBM): 'staking'
    '''

    __model_name = 'fcnn'

    def get_model_name(self):
        return self.__model_name
