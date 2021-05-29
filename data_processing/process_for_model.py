from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from keras.utils import np_utils


def merge_with_bounds(data):
    data_bound = pd.read_csv("Data/bounds.csv", index_col=0)
    data_bound.index = pd.to_datetime(data_bound.index)

    # filling nan with -1
    # data_bound = data_bound.fillna(int(-1))

    # data_bound.to_csv("data_bound.csv")#TODO
    df = pd.concat([data, data_bound], axis=1)
    # df.to_csv("df.csv") #TODO
    # df = df.fillna(int(-1))

    # in case, drop rows with missing values of original dataset
    print("bounds")
    print("data.shape", data.shape)
    print("df.shape", df.shape)
    df.dropna(subset=["Open"], inplace=True)
    # df = df.drop(df[df.Open == -1].index)
    print("df.shape", df.shape)

    df.fillna(df.mean(), inplace=True)

    return df


def merge_with_FRD_Kibot(data):
    data_FRD_K = pd.read_csv("Data/FRD+Kibot.csv", index_col=0)
    data_FRD_K.index = pd.to_datetime(data_FRD_K.index)

    # filling nan with -1
    # data_FRD_K = data_FRD_K.fillna(int(-1))

    df = pd.concat([data, data_FRD_K], axis=1)

    # df = df.fillna(int(-1))

    # in case, drop rows with missing values of original dataset
    print("FRD_Kibot")
    print("data_FRD_K.shape", data_FRD_K.shape)
    print("data.shape", data.shape)
    print("df.shape", df.shape)
    df.dropna(subset=["Open"], inplace=True)
    # df = df.drop(df[df.Open == -1].index)
    print("df.shape", df.shape)

    df.fillna(df.mean(), inplace=True)

    return df


def data_preprocess(data, model=None, merge_bounds=True, merge_FRD_Kibot=True):
    print("checking if any null values are present\n", data.isna().sum())
    data = data.drop(['Range', 'RangeP', 'zscore'], axis=1)
    label = data[['z1cat']]
    label = label['z1cat'] - 1
    training_set = data.drop(['z1cat'], axis=1)

    if model == "lstm":
        label = np_utils.to_categorical(label)

    if merge_bounds:
        training_set = merge_with_bounds(training_set)

    if merge_FRD_Kibot:
        training_set = merge_with_FRD_Kibot(training_set)

    # TODO
    # sc = MinMaxScaler(feature_range=(0, 100))
    # training_set = sc.fit_transform(training_set)
    training_set.to_csv("ts.csv")
    return training_set, label
