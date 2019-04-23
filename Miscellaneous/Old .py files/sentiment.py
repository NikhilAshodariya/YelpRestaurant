import numpy as np
import pandas as pd
import nltk
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag
import string
import seaborn as sns
import matplotlib.pyplot as plt

# Reading positive and negative words text file
with open("positive-words.txt",'r') as f:
    positive_words=[line.strip() for line in f]
with open("negative-words.txt",'r') as f:
    negative_words=[line.strip() for line in f]
        
df = pd.read_csv("data/il_reviews.csv")

df.drop(["cool","date","funny","useful"],axis = 1,inplace=True)

# Pre-processing Reviews data
processedData = list(df[["business_id","text","user_id"]].as_matrix())

WNlemma = nltk.WordNetLemmatizer()

# Performs naive Sentiment Analysis 
def naive_sentiment(x):
    res = []
    ans = []
    positive_tokens=[]
    negative_tokens=[]
    negations=['not', 'too', 'n\'t', 'no', 'cannot', 'neither','nor']
    final_review=[]
#     final_review.append(x[0])
    data = x[1]
#     final_review.append(x[2])  
    for val in sent_tokenize(data):
        val = val.strip(string.punctuation)
        
        for token in word_tokenize(val):
            
            if token in positive_words:   
                  positive_tokens.append(token)
            else:
                  if token in negative_words:
                        negative_tokens.append(token)
                    
    if len(positive_tokens)>len(negative_tokens):
        final_review.append("positive")  
    else:
        if len(negative_tokens)>len(positive_tokens):
            final_review.append("negative")
        else:
            if len(positive_tokens)==len(negative_tokens):
                final_review.append("neutral")  

#     print(final_review)
    
    return final_review

if __name__ == "__main__":

    processed_doc = list(map(naive_sentiment,processedData))
    # print (processed_doc)

    review_sentiment = []
    for x in processed_doc:
        review_sentiment+=x
    # print (review_sentiment)

    # Appending the sentiment analysis data to the Review dataset
    df['sentiment'] = review_sentiment
    # print (df)

    # # plotting sentiment against it's count
    ax = df.sentiment.value_counts().plot.bar(figsize=(6,4), title="Model count by sentiment");  
    ax.set(xlabel = "sentiment", ylabel = "model count");
    plt.show()
