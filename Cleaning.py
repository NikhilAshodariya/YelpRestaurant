
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv("data/il_reviews.csv")


# In[3]:


df.head(2)


# In[4]:


df.drop(["cool","date","funny","useful"],axis = 1,inplace=True)


# In[5]:


df.head(2)


# In[6]:


df[["business_id","text"]].as_matrix()[0]


# In[7]:


processedData = list(df[["business_id","text"]].as_matrix())


# In[8]:


processedData[0]


# # Convert to Tokens

# In[9]:


import nltk


# In[34]:


from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag
import string


# In[24]:


WNlemma = nltk.WordNetLemmatizer()


# In[47]:


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


# In[48]:


map(funcTemp,processedData[:30])


# In[49]:


list(map(funcTemp,processedData[:30]))[0][1]

