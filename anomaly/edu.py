import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt

def sigma3(data):
    quants = data.quantile(.996)
    for col in quants.index:
        data_temp = data[data[col] <= quants[col]]
        data_temp2 = data[data[col] > quants[col]]

    return data_temp2['isFraud'].sum() / data_temp2.shape[0]

def print_corr_matrix_top30(data):
    matrix = np.triu(data.iloc[:, 1:26].corr())
    sns.set(font_scale=1.8)
    sns.set(rc={'figure.figsize': (30, 30)})
    sns.heatmap(data.iloc[:, 1:26].corr(), cmap=sns.cubehelix_palette(as_cmap=False), annot=True, mask=matrix)
    plt.show()

def print_corr_matrix(data):
    matrix = np.triu(data.iloc[:, 27:].corr())
    sns.set(font_scale=1.8)
    sns.set(rc={'figure.figsize': (30, 30)})
    sns.heatmap(data.iloc[:, 27:].corr(), cmap=sns.cubehelix_palette(as_cmap=False), mask=matrix)
    plt.show()

def print_hist_card(data):
    for col in ['card1', 'card2', 'card3', 'card4', 'card5']:
        plt.figure(figsize=(15, 10))
        sns.histplot(data=data, x=col, hue="isFraud", multiple="dodge", shrink=.8)
        plt.show()

def plot_fraud_fraction(data):
    fraud_percent = data.query('isFraud == 1').shape[0] / data.shape[0] * 100
    print(f'Доля мошеннических транзакций в датасете составляет {fraud_percent:.1f}%')
    plt.figure(figsize=(7, 7))
    sns.countplot(data=data, x='isFraud');
    plt.show()

