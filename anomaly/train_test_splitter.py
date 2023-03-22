import pandas as pd
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split

import yaml


def parse_args():
    parser = ArgumentParser('Train val split')
    parser.add_argument('--input_transaction', type=str, required=True, help='Path to input transaction dataset')
    parser.add_argument('--input_identity', type=str, required=True, help='Path to input identity dataset')
    parser.add_argument('--output_train', type=str, required=True, help='Path to train')
    parser.add_argument('--output_test', type=str, required=True, help='Path to test')
    parser.add_argument('--params', type=str, required=True, help='Path to params file')
    return parser.parse_args()


def main(args):
    with open(args.params, 'r') as fp:
        params = yaml.safe_load(fp)['split']

    raw_data = pd.merge(pd.read_csv(args.input_transaction), pd.read_csv(args.input_identity),
                        left_on='TransactionID', right_on='TransactionID', how='left')

    raw_data = raw_data.drop(['TransactionID'], axis=1)

    df_train, df_test = train_test_split(raw_data, **params)

    df_train.to_csv(args.output_train, index=None)
    df_test.to_csv(args.output_test, index=None)


if __name__ == '__main__':
    args = parse_args()
    main(args)
