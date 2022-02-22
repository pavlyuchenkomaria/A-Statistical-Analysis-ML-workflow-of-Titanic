import os
import statistics
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, make_scorer, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


def read_data(path_input):
    """
    :param path_input: путь до файла
    :return: датафрейм
    """
    df = pd.read_csv(path_input)
    return df


def collect_data(df_train):
    """
    Делит данные на выборки.
    :param df: датафрейм
    :return: словарь с train, val, test выборками
    """
    X = df_train[['ScaledFare', 'ScaledAge', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Female', 'Male']]
    y = df_train['Survived']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    df_dict = {
        'train': (X_train, y_train),
        'val': (X_val, y_val)
    }
    return df_dict


def tree_get_params(df_dict, metric, build_graph=False):
    """
    Термины:
        pruning - обрезать лишние ветки с итогового дерева, чтобы убрать возможное переобучение
        alpha - штраф за каждый добавленный лист
        Error = [error] + alpha*[tree_size]
    Ищет оптимальное значение alpha, при котором увеличилось accuracy

    :param df_dict: словарь с train, val, test выборками
    :param metric: метрика
    :param build_graph: логическое значение: строить ли график
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

    scores = [metric(y_val, clf.predict(X_val)) for clf in clfs]
    max_score_ix = scores.index(max(scores))
    optimal_alpha = ccp_alphas[max_score_ix]

    if build_graph:
        plt.figure(figsize=(10, 6))
        plt.grid()
        plt.plot(ccp_alphas[:-1], scores[:-1])
        plt.xlabel("Effective alpha for DecisionTree")
        plt.ylabel(f"{metric.__name__}")
        plt.show()

    return optimal_alpha


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


def get_best_n_estimators_value(df_dict, metric, classifier, build_graph=False):
    """
    Выбрать оптимальное число n_estimators.
    :param classifier: классификатор
    :param metric: метрика
    :param df_dict: словарь с train, val, test выборками
    :param build_graph: логическое значение: строить ли график
    :return: параметр n_estimators, который дал лучшее значение метрики
    """
    X_train, y_train = df_dict['train']
    X_val, y_val = df_dict['val']

    metric_list = []
    n_of_estimators_list = []
    for n in range(25, 300, 25):
        clf = classifier(n_estimators=n)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        metric_value = metric(y_val, y_pred)
        metric_list.append(metric_value)
        n_of_estimators_list.append(n)
    max_metric_value = max(metric_list)
    best_n_estimators_value = n_of_estimators_list[metric_list.index(max_metric_value)]

    if build_graph:
        plt.figure(figsize=(10, 6))
        plt.grid()
        plt.plot(n_of_estimators_list, metric_list)
        plt.xlabel(f"Effective n_of_estimators for {classifier.__name__}")
        plt.ylabel(f"{metric.__name__}")
        plt.show()

    return best_n_estimators_value


def make_prediction(df_dict, df_test, metric=roc_auc_score, max_n=30):
    """
    Сделать предсказание на тесте.
    :param df_dict: словарь с train, val выборками
    :param df_test: тестовая выборка без целевого столбца
    :param metric: метрика
    :param max_n: максимальное количество возможных соседей для Knn
    :return: датафрейм с предсказанием
    """
    X_train, y_train = df_dict['train']

    optimal_alpha = tree_get_params(df_dict, metric, build_graph=False)
    clf_DS = DecisionTreeClassifier(criterion='gini', ccp_alpha=optimal_alpha, max_depth=10)
    clf_DS.fit(X_train, y_train)

    optimal_k = choose_best_k(df_dict, max_n, metric, build_graph=False)
    clf_KNN = KNeighborsClassifier(n_neighbors=optimal_k, algorithm='kd_tree', weights='distance')
    clf_KNN.fit(X_train, y_train)

    clf_SVC = LinearSVC()
    clf_SVC.fit(X_train, y_train)

    n_estimators_adaboost = get_best_n_estimators_value(df_dict, metric, AdaBoostClassifier, build_graph=False)
    clf_ADABOOST = AdaBoostClassifier(n_estimators=n_estimators_adaboost)
    clf_ADABOOST.fit(X_train, y_train)

    n_estimators_gradientboost = get_best_n_estimators_value(df_dict, metric, GradientBoostingClassifier,
                                                            build_graph=False)
    clf_GRADIENTBOOST = GradientBoostingClassifier(n_estimators=n_estimators_gradientboost)
    clf_GRADIENTBOOST.fit(X_train, y_train)

    ensemble_all_hard = VotingClassifier(
        estimators=[('dtree', clf_DS), ('knn', clf_KNN), ('svc', clf_SVC), ('adaboost', clf_ADABOOST),
                    ('gradientboost', clf_GRADIENTBOOST)],
        voting='hard')
    ensemble_all_hard.fit(X_train, y_train)

    X_test = df_test[['ScaledFare', 'ScaledAge', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Female', 'Male']]
    y_pred_ensemble_all_hard = ensemble_all_hard.predict(X_test)
    df_test["Survived"] = y_pred_ensemble_all_hard

    return df_test[["PassengerId", "Survived"]]


dirname = os.path.dirname(__file__)
filename_train = os.path.join(dirname, r'..\..\data\prepared_train.csv')
filename_test = os.path.join(dirname, r'..\..\data\prepared_test.csv')
filename_prediction = os.path.join(dirname, r'..\..\data\prediction.csv')

df_train = read_data(filename_train)
df_test = read_data(filename_test)[['PassengerId', 'ScaledFare', 'ScaledAge', 'Pclass_1', 'Pclass_2', 'Pclass_3',
                                    'Female', 'Male']]
df_dict = collect_data(df_train)
result_df = make_prediction(df_dict, df_test, metric=roc_auc_score, max_n=30)
result_df.to_csv(path_or_buf=filename_prediction, index=False)
