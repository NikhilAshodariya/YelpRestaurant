import numpy as np
import pandas as pd
import nltk
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag
import string


df = pd.read_csv("data/il_reviews.csv")

df.drop(["cool","date","funny","useful"],axis = 1,inplace=True)


processedData = list(df[["business_id","text"]].as_matrix())

WNlemma = nltk.WordNetLemmatizer()

def funcTemp(x):
    res = []
    ans = []
    ans.append(x[0])
    data = x[1]
    
    for val in sent_tokenize(data):
        val = val.strip(string.punctuation)
        res.append(pos_tag([WNlemma.lemmatize(w.strip(string.punctuation)) for w in word_tokenize(val) if w.strip(string.punctuation)!=""]))
    
	ans.append(res)
    return ans

print(map(funcTemp,processedData))


print(list(map(funcTemp,processedData))[0][1])

