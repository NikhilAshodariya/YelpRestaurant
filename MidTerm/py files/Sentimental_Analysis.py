import numpy as np
import pandas as pd
import nltk
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag
import string
import seaborn as sns
import matplotlib.pyplot as plt


def sentimental_Analysis(FILE_PATH):
    def naive_sentiment(x):
        res = []
        ans = []
        positive_tokens = []
        negative_tokens = []
        negations = ['not', 'too', 'n\'t', 'no', 'cannot', 'neither', 'nor']
        final_review = []
        data = x[1]
        for val in sent_tokenize(data):
            val = val.strip(string.punctuation)

            for token in word_tokenize(val):

                if token in positive_words:
                    positive_tokens.append(token)
                else:
                    if token in negative_words:
                        negative_tokens.append(token)

        if len(positive_tokens) > len(negative_tokens):
            final_review.append("positive")
        else:
            if len(negative_tokens) > len(positive_tokens):
                final_review.append("negative")
            else:
                if len(positive_tokens) == len(negative_tokens):
                    final_review.append("neutral")
        return final_review

    POSITIVE_FILE_PATH = "C:\\Users\\Nikhil\\Documents\\Python Notebooks\\BIA_660_B(Web_Mining_Notebook)\\project\\submit\\Sentimental_Analysis_Text_File\\positive-words.txt"
    NEGATIVE_FILE_PATH = "C:\\Users\\Nikhil\\Documents\\Python Notebooks\\BIA_660_B(Web_Mining_Notebook)\\project\\submit\\Sentimental_Analysis_Text_File\\negative-words.txt"

    with open(POSITIVE_FILE_PATH, 'r') as f:
        positive_words = [line.strip() for line in f]
    with open(NEGATIVE_FILE_PATH, 'r') as f:
        negative_words = [line.strip() for line in f]

    df = pd.read_csv(FILE_PATH)

    df.drop(["cool", "date", "funny", "useful"], axis=1, inplace=True)

    # Pre-processing Reviews data
    processedData = list(df[["business_id", "text", "user_id"]].as_matrix())

    WNlemma = nltk.WordNetLemmatizer()

    processed_doc = list(map(naive_sentiment, processedData))

    review_sentiment = []
    for x in processed_doc:
        review_sentiment += x

    return review_sentiment
