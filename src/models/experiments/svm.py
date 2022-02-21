from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd


def read_data(path_input):
    """
    :param path_input: путь до файла
    :return: датафрейм
    """
    df = pd.read_csv(path_input)
    return df


def split_data(df):
    """
    Делит данные на выборки.
    :param df: датафрейм
    :return: словарь с train, val, test выборками
    """
    X = df[['ScaledFare', 'ScaledAge', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Female', 'Male']]
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)
    df_dict = {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }
    return df_dict


def make_model(df_dict):
    """
    Обучить модель и получить предсказание.
    :param df_dict: словарь с train, val, test выборками
    :return: массив предсказанных значений для валидационной выборки
    """
    X_train, y_train = df_dict['train']
    X_val, y_val = df_dict['val']

    clf = LinearSVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    return y_pred


def evaluate_model(df_dict, y_pred):
    """
    :param df_dict: словарь с train, val, test выборками
    :param y_pred: массив предсказанных значений для валидационной выборки
    :return: значение roc-auc метрики
    """
    X_val, y_val = df_dict['val']
    roc_auc = roc_auc_score(y_val, y_pred)
    return roc_auc


path_input = r'C:\Users\pavlu\PycharmProjects\A-Statistical-Analysis-ML-workflow-of-Titanic\data\prepared_train.csv'
df = read_data(path_input)
df_dict = split_data(df)
y_pred = make_model(df_dict)
roc_auc = evaluate_model(df_dict, y_pred)
