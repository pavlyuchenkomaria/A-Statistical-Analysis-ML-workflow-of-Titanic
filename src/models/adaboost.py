from sklearn.ensemble import AdaBoostClassifier
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


def choose_best_model_params(df_dict):
    """
    Обучить модель и получить предсказание.
    :param df_dict: словарь с train, val, test выборками
    :param optimal_alpha: оптимальное альфа, вычисленное ранее
    :return: массив предсказанных значений для валидационной выборки
    """
    X_train, y_train = df_dict['train']
    X_val, y_val = df_dict['val']

    roc_auc_list = []
    ix_list = []
    for i in range(50, 300, 25):
        clf = AdaBoostClassifier(n_estimators=100, random_state=0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        roc_auc = roc_auc_score(y_val, y_pred)
        roc_auc_list.append(roc_auc)
        ix_list.append(i)
    max_roc_auc = max(roc_auc_list)
    best_n_estimators_value = ix_list[roc_auc_list.index(max_roc_auc)]
    return max_roc_auc, best_n_estimators_value


path_input = r'C:\Users\pavlu\PycharmProjects\A-Statistical-Analysis-ML-workflow-of-Titanic\data\prepared_train.csv'
df = read_data(path_input)
df_dict = split_data(df)
max_roc_auc, best_n_estimators_value = choose_best_model_params(df_dict)
