from keras import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras import optimizers
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split


batch_size = 128
time_steps = 20
feautures = 5
test_size = 0.2

def data_preprocess(data):
    print("checking if any null values are present\n", data.isna().sum())
    data = data.drop(['Range', 'RangeP', 'zscore'], axis=1)
    label = data[['z1cat']]
    label = label['z1cat'] - 1
    training_set = data.drop(['z1cat'], axis=1)

    sc = MinMaxScaler(feature_range=(0, 100))
    training_set_scaled = sc.fit_transform(training_set)
    label = np_utils.to_categorical(label)
    return training_set_scaled, label
    # return training_set.to_numpy(), label.to_numpy()


def make_time_series(train, label):
    train_sequential = []
    label_sequential = []

    print(len(train))

    # if len(train) < ((len(train) // batch_size + 1) * batch_size):
    #     train = train[0:((len(train) // batch_size) * batch_size)]
    #     label = label[0:((len(train) // batch_size) * batch_size)]
        # tmp = np.mean(train, axis=0)

        # for j in range(0, (((len(train) // batch_size + 1) * batch_size) - len(train))):
            # train = np.append(train, [tmp], axis=0)
            # label = np.append(label, 3, axis=0)

    i = time_steps
    while i < len(train) - batch_size:
        tmp_train = []
        tmp_test = []
        for j in range(0, batch_size):
            tmp_train.append(train[i + j - time_steps:i+j])
            tmp_test.append(label[i+j])
        i = i + batch_size
        train_sequential.extend(tmp_train)
        label_sequential.extend(tmp_test)

    return np.array(train_sequential), np.array(label_sequential)


def create_model():
    lstm_model = Sequential()
    # (batch_size, timesteps, data_dim)
    lstm_model.add(LSTM(100, batch_input_shape=(batch_size, time_steps, feautures),
                        dropout=0.0, recurrent_dropout=0.0, stateful=True, return_sequences=True,
                        kernel_initializer='random_uniform'))
    lstm_model.add(Dropout(0.1))
    lstm_model.add(LSTM(60, dropout=0.0))
    lstm_model.add(Dropout(0.1))
    lstm_model.add(Dense(20, activation='relu'))
    lstm_model.add(Dense(3, activation='softmax'))
    # optimizer = optimizers.RMSprop(learning_rate=0.1)
    # optimizer = optimizers.SGD(lr=0.000001, decay=1e-6, momentum=0.9, nesterov=True)
    lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return lstm_model


def run_model(data):
    train, label = data_preprocess(data)
    train_sequential, label_sequential = make_time_series(train, label)
    model = create_model()
    X_train, X_test, y_train, y_test = train_test_split(train_sequential, label_sequential, test_size=test_size, shuffle=False)

    if len(X_train) < ((len(X_train) // batch_size + 1) * batch_size):
        X_train = X_train[0:((len(X_train) // batch_size) * batch_size)]
        y_train = y_train[0:((len(y_train) // batch_size) * batch_size)]

    model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=30, validation_data=(X_test, y_test), verbose=1)
