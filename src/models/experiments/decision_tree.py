from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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


def tree_get_params(df_dict, build_graph=False):
    """
    Термины:
        pruning - обрезать лишние ветки с итогового дерева, чтобы убрать возможное переобучение
        alpha - штраф за каждый добавленный лист
        Error = [error] + alpha*[tree_size]
    Ищет оптимальное значение alpha, при котором увеличилось accuracy
    :param build_graph: логическое значение: строить ли график
    :param df_dict: словарь с train, val, test выборками
    :return: ccp_alpha параметр для построения дерева
    """
    X_train, y_train = df_dict['train']
    X_val, y_val = df_dict['val']

    clf = DecisionTreeClassifier()
    path = clf.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        clf.fit(X_train, y_train)
        clfs.append(clf)

    acc_scores = [accuracy_score(y_val, clf.predict(X_val)) for clf in clfs]
    max_acc_ix = acc_scores.index(max(acc_scores))
    optimal_alpha = ccp_alphas[max_acc_ix]

    if build_graph:
        plt.figure(figsize=(10, 6))
        plt.grid()
        plt.plot(ccp_alphas[:-1], acc_scores[:-1])
        plt.xlabel("Effective alpha")
        plt.ylabel("Accuracy scores")
        plt.show()

    return optimal_alpha


def make_model(df_dict, optimal_alpha):
    """
    Обучить модель и получить предсказание.
    :param df_dict: словарь с train, val, test выборками
    :param optimal_alpha: оптимальное альфа, вычисленное ранее
    :return: массив предсказанных значений для валидационной выборки
    """
    X_train, y_train = df_dict['train']
    X_val, y_val = df_dict['val']

    clf = DecisionTreeClassifier(criterion='gini', ccp_alpha=optimal_alpha, max_depth=10)
    clf = clf.fit(X_train, y_train)
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


def compare_cpp_alphas(df, n):
    """
    Сравнивает альфа, полученную в результате исследования выше, и дефолтную нулевую.
    :param df: dataframe from file
    :param n: number of experiments
    :return: доля экспериментов, в которых альфа из эксперимента лучше, чем дефолтная, количество таких случаев,
            лучшее значение метрики roc_auc за все эксперименты в каждом случае
    """
    res = []
    roc_auc_exp_list = []
    roc_auc_default_list = []
    for i in range(n):
        df_dict = split_data(df)
        optimal_alpha = tree_get_params(df_dict, build_graph=False)

        y_pred_exp = make_model(df_dict, optimal_alpha)
        y_pred_default = make_model(df_dict, optimal_alpha=0)

        roc_auc_exp = evaluate_model(df_dict, y_pred_exp)
        roc_auc_default = evaluate_model(df_dict, y_pred_default)
        roc_auc_default_list.append(roc_auc_default)
        roc_auc_exp_list.append(roc_auc_exp)

        res.append(1 if roc_auc_exp > roc_auc_default else 0)

    max_roc_auc_exp = max(roc_auc_exp_list)
    max_roc_auc_def = max(roc_auc_default_list)

    return sum(res)/n, sum(res), max_roc_auc_exp, max_roc_auc_def


path_input = r'C:\Users\pavlu\PycharmProjects\A-Statistical-Analysis-ML-workflow-of-Titanic\data\prepared_train.csv'
df = read_data(path_input)
fraction, n_of_exp_wins, max_roc_auc_exp, max_roc_auc_def = compare_cpp_alphas(df, n=15)
