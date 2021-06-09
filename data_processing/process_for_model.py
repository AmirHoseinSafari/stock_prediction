import datetime

from matplotlib.ticker import StrMethodFormatter
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np


def merge_with_bounds(data):
    data_bound = pd.read_csv("Data/bounds.csv", index_col=0)
    data_bound.index = pd.to_datetime(data_bound.index)

    data_bound = data_bound.replace(-1, np.nan)

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

    # df.fillna(df.mean(), inplace=True)
    df.fillna(0, inplace=True)

    return df


def merge_with_FRD_Kibot(data):
    data_FRD_K = pd.read_csv("Data/FRD+Kibot.csv", index_col=0)
    data_FRD_K.index = pd.to_datetime(data_FRD_K.index)

    data_FRD_K = data_FRD_K.replace(-1, np.nan)

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

    # df.fillna(df.mean(), inplace=True)
    df.fillna(0, inplace=True)

    return df


def visualize_data(data):
    hist = data.plot(kind='hist', color='#A0E8AF')
    plt.show()


def drop_columns(data):
    Kibot = ["JY", "US", "NG", "NE", "M6E", "NIY", "QM", "PX", "HO", "NKD", "MGC", "M6A", "M6B", "JE", "MJY", "RP",
             "RY", "MCD", "RF", "BR", "RU", "RA", "XAE", "XAF", "XAU", "XAV", "XAY", "XAK", "XAI", "MSF", "XAP", "XAB",
             "SEK", "AC", "NOK", "AJY", "PJY", "EAD", "ECD",
             "MES", "LBS", "UB", "BTC"]
    names = []

    for i in range(0, len(Kibot)):
        names.append(Kibot[i] + "_hour_prior_volume")
        names.append(Kibot[i] + "_hour_prior_range")
        names.append(Kibot[i] + "_overnight_range")
        names.append(Kibot[i] + "_overnight_volume")

    for name in names:
        data.drop(name, axis=1, inplace=True)


def data_preprocess(data, model=None, merge_bounds=True, merge_FRD_Kibot=True):
    print("checking if any null values are present\n", data.isna().sum())

    for index, row in data.iterrows():
        if index.year < 2006:
            data.drop(index, inplace=True)
        if index.year == 2021 and index.month == 4:
            data.drop(index, inplace=True)


    data = data.drop(['Range', 'RangeP', 'zscore'], axis=1)
    label = data[['z1cat']]
    label = label['z1cat'] - 1

    visualize_data(label)
    visualize_data(label[int(len(label)*0.8):])

    # shift the labels since we want to predict the next day
    # label = label.append(label.iloc[0:1])
    # label = label.iloc[1:]

    training_set = data.drop(['z1cat'], axis=1)

    if model == "lstm":
        label = np_utils.to_categorical(label)

    if merge_bounds:
        training_set = merge_with_bounds(training_set)

    if merge_FRD_Kibot:
        training_set = merge_with_FRD_Kibot(training_set)

    print("drop columns")
    drop_columns(training_set)

    training_set.to_csv("training_set.csv")

    # TODO
    sc = MinMaxScaler(feature_range=(1, 10))
    training_set = sc.fit_transform(training_set)

    training_set = pd.DataFrame(training_set)
    # training_set.to_csv("training_set.csv")

    return training_set, label
