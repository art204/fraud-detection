from argparse import ArgumentParser

from sklearn.metrics import accuracy_score, roc_auc_score

import pandas as pd
import json

from anomaly.data_preprocessing.data_preprocessing import get_coding_pipeline, get_prep_data_pipe_from_pkl
from models import get_model, model_choice


def parse_args():
    parser = ArgumentParser('Trainer')
    parser.add_argument('--train', required=False, default='anomaly/data/train.csv', help='Path to train dataset')
    parser.add_argument('--test', required=False, default='anomaly/data/test.csv', help='Path to test dataset')
    parser.add_argument('--fit_prep_data_pipe', required=False, type=bool, default=False,
                        help='Fit pipeline for data preprocessing: True/False')
    parser.add_argument('--clf_model', required=False, default='staking', help='Classifier model')
    parser.add_argument('--fit_clf', required=False, type=bool, default=False, help='Fit classifier: True/False')

    return parser.parse_args()


def extract_labels(df: pd.DataFrame):
    X = df.drop(['isFraud'], axis=1)
    y = df['isFraud']
    return X, y


def main(args):
    X_train = pd.read_csv(args.train)
    X_test = pd.read_csv(args.test)
    X_train, y_train = extract_labels(X_train)
    X_test, y_test = extract_labels(X_test)

    prep_data_pipe = None
    # fit prep_data_pipe
    if args.fit_prep_data_pipe:
        prep_data_pipe = get_coding_pipeline(X_train)
        prep_data_pipe.fit(X_train, y_train)
    # load prep_data_pipe from pkl file
    else:
        prep_data_pipe = get_prep_data_pipe_from_pkl()

    model = None
    # fit model
    if args.fit_clf:
        model = model_choice(args.clf_model, X_train)
        model.fit(X_train, y_train)
    # load model from pkl file
    else:
        model = get_model(args.clf_model)

    X_train = prep_data_pipe.transform(X_train)
    X_test = prep_data_pipe.transform(X_test)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    y_train_pred_proba = model.predict_proba(X_train)[:, 1]
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    train_roc_auc = roc_auc_score(y_train, y_train_pred_proba)
    test_roc_auc = roc_auc_score(y_test, y_test_pred_proba)

    with open('anomaly/metrics.json', 'w') as fp:
        json.dump({
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_roc_auc': train_roc_auc,
            'test_roc_auc': test_roc_auc
        }, fp)


if __name__ == '__main__':
    args = parse_args()
    main(args)
