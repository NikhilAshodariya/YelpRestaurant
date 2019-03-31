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

df = pd.read_csv("data/il_reviews.csv")

df.head(2)
df.drop(["cool","date","funny","useful"],axis = 1,inplace=True)
df.sample()
df[["business_id","text"]].as_matrix()[0]

processedData = list(df[["business_id","text"]].as_matrix())

# ## Tokenization
WNlemma = nltk.WordNetLemmatizer()

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

# Generating a word cloud for more visualization
# - Word cloud with stop words and with lemmatiztion to analyze the top words used in the reviews.

text = ""
for x in token_list:
    text = text + " " + x

# Generate a word cloud image
wordcloud = WordCloud(stopwords = stop_words, min_font_size = 10, background_color = "white").generate(text)

# Display the generated image:
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis("off")
plt.show()

# Save the image in the img folder:
# wordcloud.to_file("Images/after_lemmatization.png")

# Plotting top 20 words used in the reviews
cv = CountVectorizer()
bow = cv.fit_transform(token_list)
# print (cv.get_feature_names())
word_freq = dict(zip(cv.get_feature_names(), np.asarray(bow.sum(axis=0)).ravel()))
word_counter = collections.Counter(word_freq)
word_counter_df = pd.DataFrame(word_counter.most_common(20), columns = ['word', 'freq'])
fig, ax = plt.subplots(figsize=(12, 10))
sns.barplot(x="word", y="freq", data=word_counter_df, palette="PuBuGn_d", ax=ax)
plt.show()
