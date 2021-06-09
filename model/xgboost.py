from sklearn.model_selection import train_test_split
import xgboost.sklearn as xgb
import numpy as np
from xgboost import XGBClassifier
import pandas as pd
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

from data_processing.process_for_model import data_preprocess
from evaluations import ROC_PR

test_size = 0.3


def run_model(data):
    train, label = data_preprocess(data)
    print("data shape:", train.shape)
    print("label shape:", label.shape)
    train.to_csv("train.csv")
    label.to_csv("label.csv")
    X_train, X_test, y_train, y_test = train_test_split(train, label, test_size=test_size, shuffle=False)

    # fit model no training data
    model = XGBClassifier(max_depth=50, objective='multi:softmax', n_estimators=50,
                            num_classes=3)
    model.fit(X_train, y_train)

    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    # make predictions for test data
    y_pred = model.predict(X_train)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_train, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    # TODO multi:softprob
    # gbt_model = xgb.XGBClassifier(max_depth=50, objective='multi:softmax', n_estimators=50,
    #                         num_classes=3)
    #
    # gbt_model.fit(X_train, y_train)
    # ______
    # predictions_train = gbt_model.predict_proba(X_train)
    # print(predictions_train)
    # predictions_train = [round(value) for value in predictions_train]
    # print(predictions_train)
    # accuracy_train = accuracy_score(y_train, predictions_train)
    # print("Accuracy train: %.2f%%" % (accuracy_train * 100.0))
    #
    # predictions_test = gbt_model.predict_proba(X_test)
    # accuracy_test = accuracy_score(y_test, predictions_test)
    # print("Accuracy test: %.2f%%" % (accuracy_test * 100.0))

    # gbt_model = xgb.XGBModel(objective='multi:softmax', n_estimators=50, min_samples_split=30,
    #                          random_state=10, max_depth=1000, num_classes=3).fit(np.array(X_train), np.array(y_train),
    #                          eval_set=[(np.array(X_train), np.array(y_train)),
    #                         (np.array(X_test), np.array(y_test))], verbose=True)

    # score_test, score_sr, score_pr = ROC_PR.ROC_ML(gbt_model, np.array(X_test), np.array(y_test), "GBT", 0,
    #                                                xgb=True)
    #
    # print(score_test)
    # print(score_sr)
    # print(score_pr)

