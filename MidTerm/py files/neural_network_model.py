import nltk
import string
import numpy as np
import pandas as pd
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.preprocessing.text import Tokenizer


def label_data(path):
    df = pd.read_csv(path)
    reviews = df.text

    food_tags = np.where(df.text.str.contains(
        'food|vegetables|veggie|veggies|meat|chicken|pho|soup|lunch|dinner|menu|bland|flavor'), 'food', None)
    cleanliness_tags = np.where(df.text.str.contains('clean|dirty|hygiene'), 'cleanliness', None)
    service_tags = np.where(df.text.str.contains(
        'service|waitress|hostess|waiter|worker|staff|'), 'service', None)
    ambience_tags = np.where(df.text.str.contains('ambience|place'), 'ambience', None)

    tags = zip(food_tags, cleanliness_tags, service_tags, ambience_tags)
    tag_list = list(tags)
    tag_list = [list(tags) for tags in tag_list]
    valid_tag_list = []
    for tags in tag_list:
        for tag in tags:
            valid_list = [t for t in tags if t is not None]
        valid_tag_list.append(valid_list)

    df['tags'] = valid_tag_list
    return df


def build_model_ann(tf_idf, labels):
    binarizer = MultiLabelBinarizer()
    labels = binarizer.fit_transform(labels)

    X = tf_idf
    y = labels
    X_train, X_test = X[:5000], X[5000:]
    y_train, y_test = y[:5000], y[5000:]

    model = Sequential()
    model.add(Dense(1000, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dropout(0.1))
    model.add(Dense(y_train.shape[1], activation='sigmoid'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd)

    model.fit(X_train, y_train, epochs=5, batch_size=2000)

    preds = model.predict(X_test)
    preds[preds >= 0.5] = 1
    preds[preds < 0.5] = 0
    label = ['food', 'cleanliness', 'ambience', 'service']
    return classification_report(y_test, preds, target_names=label)
