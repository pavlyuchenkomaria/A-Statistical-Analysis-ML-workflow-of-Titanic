import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
import statistics


def replace_nans(df):
    """
    Change nans in "age" to mean statistics of feature, check for nans in test.
    :param df: dataframe
    :return: transformed dataframe without nan values
    """
    df.Age.fillna(df.Age.mean(), inplace=True)
    if df.Fare.isna().sum():
        df.Fare.fillna(df.Fare.mean(), inplace=True)
    return df


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
    """
    Use OneHotEncoder for Pclass and Sex features.
    :param df: dataframe
    :return: transformed dataframe with OneHotEncoded "Pclass" and "Sex"
    """
    enc = preprocessing.OneHotEncoder()
    enc_df_pclass = pd.DataFrame(enc.fit_transform(df[['Pclass']]).toarray(),
                                 columns=['Pclass_1', 'Pclass_2', 'Pclass_3'])
    df = df.join(enc_df_pclass, how='left')
    enc_df_sex = pd.DataFrame(enc.fit_transform(df[['Sex']]).toarray(), columns=['Female', 'Male'])
    df = df.join(enc_df_sex, how='left')
    return df


def change_outliers_to_median(df):
    """
    In "fare" column change outliers to median of its group.
    :param df: dataframe
    :return: transformed dataframe without outliers in "fare" column.
    """
    fare_greater_200_ix = np.where((df.Fare > 200))
    big_fare_median = statistics.median(df.Fare.iloc[fare_greater_200_ix])
    df.Fare.iloc[fare_greater_200_ix] = big_fare_median
    return df


def convert_number_to_cat(df):
    """
    Convert numerical features to categories.
    :param df: dataframe
    :return: df with new columns for each converted feature
    """
    # fare
    df.loc[df.Fare <= 15, 'Fare_cat'] = 0
    df.loc[(15 < df.Fare) & (df.Fare <= 35), 'Fare_cat'] = 1
    df.loc[(35 < df.Fare) & (df.Fare <= 100), 'Fare_cat'] = 2
    df.loc[df.Fare > 100, 'Fare_cat'] = 3

    # age
    df.loc[df.Age < 25, 'Age_cat'] = 0
    df.loc[(25 <= df.Age) & (df.Age <= 50), 'Age_cat'] = 1
    df.loc[df.Age > 50, 'Age_cat'] = 2

    # parch
    df.loc[df.Parch == 0, 'Parch_cat'] = 0
    df.loc[(df.Parch == 1) | (df.Parch == 2), 'Parch_cat'] = 1
    df.loc[df.Parch >= 3, 'Parch_cat'] = 2

    # SibSp
    df.loc[df.SibSp == 0, 'SibSp_cat'] = 0
    df.loc[df.SibSp == 1, 'SibSp_cat'] = 1
    df.loc[df.SibSp >= 2, 'SibSp_cat'] = 2

    return df[['Fare_cat', 'Age_cat', 'Parch_cat', 'SibSp_cat']]


def get_features_for_train(df):
    """
    :param df: dataframe
    :return: transformed dataframe with features
    """
    return df[['Survived', 'Fare_cat', 'Age_cat', 'Parch_cat', 'SibSp_cat', 'Female', 'Male', 'Pclass_1', 'Pclass_2',
               'Pclass_3', 'ScaledFare', 'ScaledAge']]


def get_features_for_test(df):
    """
    :param df: dataframe
    :return: transformed dataframe with features
    """
    return df[['PassengerId', 'Fare_cat', 'Age_cat', 'Parch_cat', 'SibSp_cat', 'Female', 'Male', 'Pclass_1', 'Pclass_2', 'Pclass_3',
               'ScaledFare', 'ScaledAge']]


def prepare_data_for_train(path_read, path_write):
    df = pd.read_csv(filepath_or_buffer=path_read)
    replace_nans(df)
    change_outliers_to_median(df)
    convert_number_to_cat(df)
    scale_data(df)
    df = one_hot_encode_data(df)
    get_features_for_train(df).to_csv(path_or_buf=path_write, index=False)


def prepare_data_for_test(path_read, path_write):
    df = pd.read_csv(filepath_or_buffer=path_read)
    replace_nans(df)
    change_outliers_to_median(df)
    convert_number_to_cat(df)
    scale_data(df)
    df = one_hot_encode_data(df)
    get_features_for_test(df).to_csv(path_or_buf=path_write, index=False)


dirname = os.path.dirname(__file__)
filename_train = os.path.join(dirname, r'..\data\train.csv')
filename_prepared_train = os.path.join(dirname, r'..\data\prepared_train.csv')
filename_test = os.path.join(dirname, r'..\data\test.csv')
filename_prepared_test = os.path.join(dirname, r'..\data\prepared_test.csv')

prepare_data_for_train(
   path_read=filename_train,
   path_write=filename_prepared_train
)
prepare_data_for_test(
   path_read=filename_test,
   path_write=filename_prepared_test
)
