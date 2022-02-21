import statistics
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
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


def choose_best_k(df, scoring, build_graph=False):
    """
    :param df: датафрейм
    :param scoring: метрика
    :param build_graph: логическое значение: строить ли график
    :return: оптимальное k и график
    """
    X = df[['Fare', 'Age', 'Pclass', 'Female', 'Male']]
    y = df['Survived']
    mean_cross_val = []
    k = [j for j in range(1, 15)]
    for n in range(1, 15):
        clf = KNeighborsClassifier(n_neighbors=n, algorithm='kd_tree', weights='distance')
        cross_val_score_list = cross_val_score(clf, X, y, cv=5, scoring=scoring)
        mean_cross_val.append(statistics.mean(cross_val_score_list))
    max_acc_ix = mean_cross_val.index(max(mean_cross_val))
    optimal_k = k[max_acc_ix]

    if build_graph:
        plt.figure(figsize=(10, 6))
        plt.grid()
        plt.plot(k, mean_cross_val)
        plt.xlabel("k")
        plt.ylabel("roc_auc scores")
        plt.show()
    return optimal_k


path_input = r'C:\Users\pavlu\PycharmProjects\A-Statistical-Analysis-ML-workflow-of-Titanic\data\prepared_train.csv'
df = read_data(path_input)
choose_best_k(df, 'roc_auc', build_graph=True)

