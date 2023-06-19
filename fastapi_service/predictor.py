import pandas as pd

from typing import List
from typing import Dict
from collections import defaultdict

from anomaly.nn_utils import predict as nn_predict
from anomaly.nn_utils import train as nn_train
from anomaly.data_preprocessing.data_preprocessing import get_prep_data_pipe_from_pkl
from anomaly.models import get_model, get_fcnn_model
from fastapi_service.transaction import Transaction


class Predictor:

    def __init__(self, clf_name='logreg'):
        self.__prep_data_pipe = get_prep_data_pipe_from_pkl()
        self.__clf_name = clf_name
        self.__optimizer = None
        self.__scheduler = None
        if clf_name == 'fcnn':
            self.__clf, self.__optimizer, self.__scheduler = get_fcnn_model()
        else:
            self.__clf = get_model(clf_name)
        self.__raw_columns = ['TransactionDT', 'TransactionAmt', 'ProductCD',
                              'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
                              'addr1', 'addr2', 'dist1', 'dist2', 'P_emaildomain', 'R_emaildomain',
                              'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14',
                              'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14',
                              'D15',
                              'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
                              'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14',
                              'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27',
                              'V28', 'V29', 'V30', 'V31', 'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V39', 'V40',
                              'V41', 'V42', 'V43', 'V44', 'V45', 'V46', 'V47', 'V48', 'V49', 'V50', 'V51', 'V52', 'V53',
                              'V54', 'V55', 'V56', 'V57', 'V58', 'V59', 'V60', 'V61', 'V62', 'V63', 'V64', 'V65', 'V66',
                              'V67', 'V68', 'V69', 'V70', 'V71', 'V72', 'V73', 'V74', 'V75', 'V76', 'V77', 'V78', 'V79',
                              'V80', 'V81', 'V82', 'V83', 'V84', 'V85', 'V86', 'V87', 'V88', 'V89', 'V90', 'V91', 'V92',
                              'V93', 'V94', 'V95', 'V96', 'V97', 'V98', 'V99', 'V100', 'V101', 'V102', 'V103', 'V104',
                              'V105', 'V106', 'V107', 'V108', 'V109', 'V110', 'V111', 'V112', 'V113', 'V114', 'V115',
                              'V116', 'V117', 'V118', 'V119', 'V120', 'V121', 'V122', 'V123', 'V124', 'V125', 'V126',
                              'V127', 'V128', 'V129', 'V130', 'V131', 'V132', 'V133', 'V134', 'V135', 'V136', 'V137',
                              'V138', 'V139', 'V140', 'V141', 'V142', 'V143', 'V144', 'V145', 'V146', 'V147', 'V148',
                              'V149', 'V150', 'V151', 'V152', 'V153', 'V154', 'V155', 'V156', 'V157', 'V158', 'V159',
                              'V160', 'V161', 'V162', 'V163', 'V164', 'V165', 'V166', 'V167', 'V168', 'V169', 'V170',
                              'V171', 'V172', 'V173', 'V174', 'V175', 'V176', 'V177', 'V178', 'V179', 'V180', 'V181',
                              'V182', 'V183', 'V184', 'V185', 'V186', 'V187', 'V188', 'V189', 'V190', 'V191', 'V192',
                              'V193', 'V194', 'V195', 'V196', 'V197', 'V198', 'V199', 'V200', 'V201', 'V202', 'V203',
                              'V204', 'V205', 'V206', 'V207', 'V208', 'V209', 'V210', 'V211', 'V212', 'V213', 'V214',
                              'V215', 'V216', 'V217', 'V218', 'V219', 'V220', 'V221', 'V222', 'V223', 'V224', 'V225',
                              'V226', 'V227', 'V228', 'V229', 'V230', 'V231', 'V232', 'V233', 'V234', 'V235', 'V236',
                              'V237', 'V238', 'V239', 'V240', 'V241', 'V242', 'V243', 'V244', 'V245', 'V246', 'V247',
                              'V248', 'V249', 'V250', 'V251', 'V252', 'V253', 'V254', 'V255', 'V256', 'V257', 'V258',
                              'V259', 'V260', 'V261', 'V262', 'V263', 'V264', 'V265', 'V266', 'V267', 'V268', 'V269',
                              'V270', 'V271', 'V272', 'V273', 'V274', 'V275', 'V276', 'V277', 'V278', 'V279', 'V280',
                              'V281', 'V282', 'V283', 'V284', 'V285', 'V286', 'V287', 'V288', 'V289', 'V290', 'V291',
                              'V292', 'V293', 'V294', 'V295', 'V296', 'V297', 'V298', 'V299', 'V300', 'V301', 'V302',
                              'V303', 'V304', 'V305', 'V306', 'V307', 'V308', 'V309', 'V310', 'V311', 'V312', 'V313',
                              'V314', 'V315', 'V316', 'V317', 'V318', 'V319', 'V320', 'V321', 'V322', 'V323', 'V324',
                              'V325', 'V326', 'V327', 'V328', 'V329', 'V330', 'V331', 'V332', 'V333', 'V334', 'V335',
                              'V336', 'V337', 'V338', 'V339',
                              'id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06', 'id_07', 'id_08', 'id_09', 'id_10',
                              'id_11', 'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20',
                              'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29', 'id_30',
                              'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38',
                              'DeviceType', 'DeviceInfo']

    # Подготовка данных
    def prep_data(self, df):
        return self.__prep_data_pipe.transform(df)

    # Создание DataFrame из исходных данных
    def _get_dataframe(self, obj):
        if isinstance(obj, pd.DataFrame):
            return obj.copy()
        if isinstance(obj, Transaction):
            dict_for_df = defaultdict(list)
            for k, v in obj.dict().items():
                dict_for_df[k].append(v)
            return pd.DataFrame(dict_for_df)
        if isinstance(obj, List):
            dict_for_df = defaultdict(list)
            for i in range(len(obj)):
                for k, v in obj[i].dict().items():
                    dict_for_df[k].append(v)
            return pd.DataFrame(dict_for_df)
        if isinstance(obj, Dict):
            return pd.DataFrame(obj)

    def predict(self, data):
        df = self._get_dataframe(data)
        if 'TransactionID' in df.columns:
            df.drop(columns=['TransactionID'], inplace=True)
        if 'isFraud' in df.columns:
            df.drop(columns=['isFraud'], inplace=True)
        df = self.prep_data(df)
        if self.__clf_name == 'fcnn':
            result = nn_predict(self.__clf, df)
        else:
            result = self.__clf.predict(df)
        return result

    def train(self, x_train):
        x_train = self._get_dataframe(x_train)
        y_train = x_train['isFraud']
        if 'TransactionID' in x_train.columns:
            x_train.drop(columns=['TransactionID'], inplace=True)
        x_train.drop(columns=['isFraud'], inplace=True)
        x_train = self.prep_data(x_train)
        if self.__clf_name == 'fcnn':
            nn_train(self.__clf, self.__optimizer, self.__scheduler, x_train, y_train)
        else:
            self.__clf.fit(x_train, y_train)
        return True
