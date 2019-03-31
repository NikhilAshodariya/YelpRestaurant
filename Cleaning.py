import numpy as np
import pandas as pd
import nltk
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag
import string


df = pd.read_csv("data/il_reviews.csv")

df.drop(["cool","date","funny","useful"],axis = 1,inplace=True)


processedData = df.text.tolist()

WNlemma = nltk.WordNetLemmatizer()

def tokenize_Lemmatize_POS(data):
    ans = []
    for val in sent_tokenize(data):
        val = val.strip(string.punctuation)
        ans.extend([WNlemma.lemmatize(w.strip(string.punctuation)) for w in word_tokenize(val) if w.strip(string.punctuation)!=""])
    return pos_tag(ans)

print(map(tokenize_Lemmatize_POS,processedData))


tagged_POS = list(map(tokenize_Lemmatize_POS,processedData))
tagged_POS_POS[:5]
