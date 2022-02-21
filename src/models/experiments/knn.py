import statistics
from matplotlib import pyplot as plt
from sklearn.metrics import make_scorer, roc_auc_score
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


def choose_best_k(df_dict, n, metric, build_graph=False):
    """
    :param df_dict: словарь с train, val, test выборками
    :param n: максимальное количество возможных соседей
    :param metric: метрика
    :param build_graph: логическое значение: строить ли график
    :return: оптимальное k и график
    """
    X_train, y_train = df_dict['train']
    X_val, y_val = df_dict['val']
    X = pd.concat([X_train, X_val], ignore_index=True)
    y = pd.concat([y_train, y_val], ignore_index=True)

    mean_cross_val = []
    k_list = [j for j in range(1, n)]
    for k in range(1, n):
        clf = KNeighborsClassifier(n_neighbors=k, algorithm='kd_tree', weights='distance')
        cross_val_score_list = cross_val_score(clf, X, y, cv=5, scoring=make_scorer(metric))
        mean_cross_val.append(statistics.mean(cross_val_score_list))
    max_metric_ix = mean_cross_val.index(max(mean_cross_val))
    optimal_k = k_list[max_metric_ix]

    if build_graph:
        plt.figure(figsize=(10, 6))
        plt.grid()
        plt.plot(k_list, mean_cross_val)
        plt.xlabel("Effective k for KNN")
        plt.ylabel(f"{metric.__name__}")
        plt.show()

    return optimal_k


path_input = r'C:\Users\pavlu\PycharmProjects\A-Statistical-Analysis-ML-workflow-of-Titanic\data\prepared_train.csv'
df = read_data(path_input)
df_dict = split_data(df)
choose_best_k(df_dict, n=30, metric=roc_auc_score, build_graph=False)

