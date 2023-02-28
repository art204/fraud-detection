import sys
import pandas as pd
from sklearn.metrics import roc_auc_score
import edu
from models import model_choice, get_model
from feature_selection import get_train_test



if __name__ == '__main__':
    mode = int(sys.argv[1])
    model_name = sys.argv[2]
    #чтение данных
    raw_data = pd.merge(pd.read_csv('data/train_transaction.csv'), pd.read_csv('data/train_identity.csv'),
                        left_on='TransactionID', right_on='TransactionID', how='left')
    pd.set_option('display.max_columns', None)

    # отбор признаков и разбиение на train, test
    X_train, X_test, y_train, y_test = get_train_test(raw_data)

    # загрузка модели для обучения
    if mode == 1:
        model = model_choice('staking', raw_data)
        model.fit(X_train, y_train)

    #загрузка модели pkl
    elif mode == 2:
        model = get_model('staking')

    y_pred = model.predict(X_test)
    score = roc_auc_score(y_test, y_pred)

    print(score)

