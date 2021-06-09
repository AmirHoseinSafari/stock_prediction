from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from data_processing.process_for_model import data_preprocess
from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier


test_size = 0.3


def run_model(data):
    train, label = data_preprocess(data)
    # label = label.sample(frac=1)
    # train = train.sample(frac=1)
    print("data shape:", train.shape)
    print("label shape:", label.shape)
    train.to_csv("train.csv")
    label.to_csv("label.csv")
    X_train, X_test, y_train, y_test = train_test_split(train, label, test_size=test_size, shuffle=False)

    # fit model no training data
    model = RandomForestClassifier(n_estimators=200, max_depth=500, random_state=0)

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    n_scores = cross_val_score(model, train, label, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    # report performance
    print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

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


