#!/usr/bin/env python
# coding: utf-8

# # <center>Exploratory Data Analysis (EDA) </center>

# ## Initial Data Analysis

# In[1]:


import numpy as np
import pandas as pd
import nltk
import seaborn as sns
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words+=["restaurant", "restaurants", "food", "would", "place", "eat", "menu", "u", "n't", "ve"]
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# %matplotlib inline


# In[2]:


reviews_data = pd.read_csv("data/il_reviews.csv", header = 0)
reviews_data.head()  # Snapshot of reviews data


# In[3]:


reviews_data.info() # summary of the reviews dataset


# In[4]:


reviews_data.describe()


# In[5]:


reviews_data.corr()   # corerraltion between numerical data


# Cool, funny and useful describes the number of cool, funny and useful votes given to the review.
# From the above dataset description, it is observed that less than 25% of the reviewers has voted the reviews as cool, funny or useful.
# 
# So, the cool, funny and useful votes data is not useful here.  

# In[6]:


reviews_data.drop(["cool","date","funny","useful"],axis = 1,inplace=True) # dropping cool, date, funny and useful columns
reviews_data.sample()


# ## Data Visualization

# In[7]:


reviews_data.stars.value_counts() # counts the stars rating in the reviews data


# In[8]:


# plotting the stars rating againt count
ax = reviews_data.stars.value_counts().plot.bar(figsize=(6,4), title="Model count by stars for Reviews");  
ax.set(xlabel = "stars", ylabel = "model count");


# From the above chart, we can analyze that the majority of reviews have received 5 start ratings... 
# 
# ### Generating a word cloud for more visualization
# -  Word cloud with stop words and without lemmatiztion to analyze the top words used in the reviews.

# In[9]:


text = " ".join(x for x in reviews_data.text)  # raw reviews

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
# wordcloud.to_file("Images/before_lemmatization.png")


# In[10]:


business_data = pd.read_csv("data/IL_BusinessData.csv", header = 0)
business_data.head()


# In[11]:


business_data.drop(["attributes"], axis = 1, inplace = True)  # dropping attributes column


# In[12]:


business_data.sample() # sample business dataset


# In[ ]:


business_data.info() # summary of the business dataset


# In[13]:


business_data.describe()


# In[14]:


business_data.stars.value_counts()  # counts the stars rating in the business data


# In[16]:


business_data.corr()   # correlation between numeric data in the business dataset


# In[17]:


# plotting the stars rating againt count for business data

ax = business_data.stars.value_counts().plot.bar(figsize=(6,4), title="Model count by stars for Businesses");
ax.set(xlabel = "stars", ylabel = "model count");
plt.show()

# In[ ]:




