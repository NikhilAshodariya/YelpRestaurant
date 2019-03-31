#!/usr/bin/env python
# coding: utf-8

# In[172]:


import numpy as np
import pandas as pd
import nltk
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag
import string
import seaborn as sns
import matplotlib.pyplot as plt

with open("positive-words.txt",'r') as f:
    positive_words=[line.strip() for line in f]
with open("negative-words.txt",'r') as f:
    negative_words=[line.strip() for line in f]
        
df = pd.read_csv("data/il_reviews.csv")

df.drop(["cool","date","funny","useful"],axis = 1,inplace=True)


processedData = list(df[["business_id","text"]].as_matrix())

WNlemma = nltk.WordNetLemmatizer()

def funcTemp(x):
    res = []
    ans = []
    positive_tokens=[]
    negative_tokens=[]
    negations=['not', 'too', 'n\'t', 'no', 'cannot', 'neither','nor']
    final_review=[]
    ans.append(x[0])
    data = x[1]
      
    for val in sent_tokenize(data):
        val = val.strip(string.punctuation)
        res.append([WNlemma.lemmatize(w.strip(string.punctuation)) for w in word_tokenize(val) if w.strip(string.punctuation)!=""])
#         removed pos
        for token in word_tokenize(val):
            if token in positive_words:
                    positive_tokens.append(token)
            else:
                  if token in negative_words:
                        negative_tokens.append(token)
                    
    if len(positive_tokens)>len(negative_tokens):
        final_review.append("positive")
#         positive_tokens.append([token for token in word_tokenize(val) \
#                  if token in positive_words ])
#         negative_tokens.append([token for token in word_tokenize(val) \
#                  if token in negative_words])    
    else:
        if len(negative_tokens)>len(positive_tokens):
            final_review.append("negative")
        else:
                if len(positive_tokens)==len(negative_tokens):
                    final_review.append("neutral")  
    ans.append(res)
    print(final_review)
#     stores final review sentiment
#     d['ans']=ans
#     d['final_review']=final_review
    df.join(final_review)
    return ans


# In[ ]:





# In[173]:



print(map(funcTemp,processedData))


list(map(funcTemp,processedData))


# In[ ]:

# plotting the sentiment count
# ax = df.final_review.value_counts().plot.bar(figsize=(6,4), title="Model count by sentiment");  
# ax.set(xlabel = "sentiment", ylabel = "model count");
# plt.show()

# In[ ]:




