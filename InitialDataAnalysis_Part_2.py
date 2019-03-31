#!/usr/bin/env python
# coding: utf-8

# ## Initial Data Analysis - Part 2

# In[17]:


import numpy as np
import pandas as pd
import nltk
import string
import collections
import seaborn as sns
import matplotlib.pyplot as plt

from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words+=["restaurant", "restaurants", "food", "would", "u", "n't", "ve"]
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# In[18]:


df = pd.read_csv("data/il_reviews.csv")


# In[19]:


df.head(2)


# In[20]:


df.drop(["cool","date","funny","useful"],axis = 1,inplace=True)


# In[21]:


df.sample()


# In[22]:


df[["business_id","text"]].as_matrix()[0]


# In[23]:


processedData = list(df[["business_id","text"]].as_matrix())


# In[24]:


processedData[6702]


# ## Tokenization

# In[26]:


WNlemma = nltk.WordNetLemmatizer()


# In[27]:


def tokenize(x):
    res = []
    ans = []
    ans.append(x[0])
    data = x[1]
    
    for val in sent_tokenize(data):
        val = val.strip(string.punctuation).lower()
        filtered_text = [w for w in word_tokenize(val) if not w in stop_words]
 
        lemmatized_tokens = [WNlemma.lemmatize(w.strip(string.punctuation)) for w in filtered_text if w.strip(string.punctuation)!=""]
        
        res = res+lemmatized_tokens
    
    ans.append(res)
    
    return ans


# In[28]:


map(tokenize,processedData)


# In[14]:


clean_reviews = list(map(tokenize,processedData))
clean_text = []
for x in clean_reviews:
    clean_text.append(x[1])
# print (clean_text)
token_list = []
for x in clean_text:
    token_list = token_list + x
# print(token_list)
# print (stop_words)


# - Word cloud with stop words and without lemmatiztion to analyze the top words used in the reviews.

# In[29]:


text = ""
for x in token_list:
    text = text + " " + x
# text = str(token_list)  # raw reviews

# Create stopword list:
stop_words = set(STOPWORDS)
stop_words.update(["restaurant","restaurants", "food"])

# Generate a word cloud image
wordcloud = WordCloud(stopwords = stop_words, min_font_size = 10, background_color = "white").generate(text)

# Display the generated image:
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis("off")
plt.show()

# Save the image in the img folder:
# wordcloud.to_file("Images/after_lemmatization.png")


# In[31]:


cv = CountVectorizer()
bow = cv.fit_transform(token_list)
# print (cv.get_feature_names())
word_freq = dict(zip(cv.get_feature_names(), np.asarray(bow.sum(axis=0)).ravel()))
word_counter = collections.Counter(word_freq)
word_counter_df = pd.DataFrame(word_counter.most_common(20), columns = ['word', 'freq'])
fig, ax = plt.subplots(figsize=(12, 10))
sns.barplot(x="word", y="freq", data=word_counter_df, palette="PuBuGn_d", ax=ax)
plt.show()


# In[ ]:




