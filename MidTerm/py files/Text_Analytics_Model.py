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


def tokenize_Lemmatize_POS(data):
    WNlemma = nltk.WordNetLemmatizer()

    def helper(data):
        ans = []
        for val in sent_tokenize(data):
            val = val.strip(string.punctuation)
            ans.extend([WNlemma.lemmatize(w.strip(string.punctuation))
                        for w in word_tokenize(val) if w.strip(string.punctuation) != ""])
        return pos_tag(ans)

    return list(map(helper, data))


def construct_tf_idf(reviews):
    t = Tokenizer()
    t.fit_on_texts(reviews)
    tf_idf = t.texts_to_matrix(reviews, mode='count')
    return tf_idf
