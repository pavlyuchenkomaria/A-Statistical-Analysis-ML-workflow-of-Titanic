import pandas as pd
from sklearn import preprocessing


def replace_nans(df):
    """
    Change nans in "age", "fare to mean statistics of feature.
    :param df: dataframe
    :return: transformed dataframe without nan values
    """
    df.Age.fillna(df.Age.mean(), inplace=True)
    df.Fare.fillna(df.Fare.mean(), inplace=True)
    return df


def delete_extra_features(df):
    """
    Delete columns which have no correlation with target column.
    :param df: dataframe
    :return: transformed dataframe with columns "survived", "sex", "pclass", "age", "fare"
    """
    return df.drop(columns=["SibSp", "Parch", "Ticket", "Cabin", "Embarked"])


def scale_data(df):
    """
    Use MinMax Scaler.
    :param df: dataframe
    :return: transformed dataframe with scaled "age" and "fare".
    """
    scaler = preprocessing.MinMaxScaler()
    minmax_df = scaler.fit_transform(df[["Fare", "Age"]])
    df[["ScaledFare", "ScaledAge"]] = minmax_df
    return df


def one_hot_encode_data(df):
    enc = preprocessing.OneHotEncoder()
    enc_df_pclass = pd.DataFrame(enc.fit_transform(df[['Pclass']]).toarray(),
                                 columns=['Pclass_1', 'Pclass_2', 'Pclass_3'])
    df = df.join(enc_df_pclass, how='left')
    enc_df_sex = pd.DataFrame(enc.fit_transform(df[['Sex']]).toarray(), columns=['Female', 'Male'])
    df = df.join(enc_df_sex, how='left')
    return df


def prepare_data(path_read, path_write):
    df = pd.read_csv(filepath_or_buffer=path_read)
    df = delete_extra_features(df)
    df = replace_nans(df)
    df = scale_data(df)
    df = one_hot_encode_data(df)
    df.to_csv(path_or_buf=path_write)


prepare_data(
    path_read=r'C:\Users\pavlu\PycharmProjects\A-Statistical-Analysis-ML-workflow-of-Titanic\data\test.csv',
    path_write=r'C:\Users\pavlu\PycharmProjects\A-Statistical-Analysis-ML-workflow-of-Titanic\data\prepared_test.csv'
)
