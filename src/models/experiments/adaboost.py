import os
import statistics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd
from matplotlib import pyplot as plt


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
    X = df[['Fare', 'Age', 'Pclass', 'Female', 'Male']]
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)
    df_dict = {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }
    return df_dict


def get_best_n_estimators_value(df_dict):
    """
    Выбрать оптимальное число n_estimators.
    :param df_dict: словарь с train, val, test выборками
    :return: максимальное значение метрики и параметр n_estimators, который дал это значение метрики
    """
    X_train, y_train = df_dict['train']
    X_val, y_val = df_dict['val']

    roc_auc_list = []
    ix_list = []
    for n in range(25, 300, 25):
        clf = AdaBoostClassifier(n_estimators=n)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        roc_auc = roc_auc_score(y_val, y_pred)
        roc_auc_list.append(roc_auc)
        ix_list.append(n)
    max_roc_auc = max(roc_auc_list)
    best_n_estimators_value = ix_list[roc_auc_list.index(max_roc_auc)]
    return max_roc_auc, best_n_estimators_value


def choose_best_model_params(df, n=20, build_hist=False, build_plot=False):
    """
    Проводит n экспериментов и выбирает лучший параметр для n_estimators
    :param df: датафрейм
    :param n: число экспериментов
    :param build_hist: если True, строит гистограмму
    :param build_plot: если True, строит график roc_auc
    :return: лучший параметр для n_estimators
    """
    best_n_estimators_values = []
    max_roc_auc_list = []
    for i in range(n):
        df_dict = split_data(df)
        max_roc_auc, best_n_estimators_value = get_best_n_estimators_value(df_dict)
        max_roc_auc_list.append(max_roc_auc)
        best_n_estimators_values.append(best_n_estimators_value)
    if build_hist:
        plt.figure(figsize=(10, 6))
        plt.hist(best_n_estimators_values)
        plt.show()
    if build_plot:
        x = [i for i in range(1, n+1)]
        plt.figure(figsize=(10, 6))
        plt.plot(x, max_roc_auc_list)
        plt.xlabel(f"Effective n_of_estimators")
        plt.ylabel("roc_auc")
        plt.show()
    return statistics.mode(best_n_estimators_values)


dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, r'..\..\..\data\prepared_train.csv')
df = read_data(filename)
best_n_estimators_value = choose_best_model_params(df, n=40, build_hist=True, build_plot=True)
