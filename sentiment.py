import numpy as np
import pandas as pd
import nltk
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag
import string

with open("positive-words.txt",'r') as f:
    positive_words=[line.strip() for line in f]
with open("negative-words.txt",'r') as f:
    negative_words=[line.strip() for line in f]
        
df = pd.read_csv("data/il_reviews.csv")

df.drop(["cool","date","funny","useful"],axis = 1,inplace=True)


processedData = list(df[["business_id","text","user_id"]].as_matrix())

WNlemma = nltk.WordNetLemmatizer()

def naive_sentiment(x):
    res = []
    ans = []
    positive_tokens=[]
    negative_tokens=[]
    negations=['not', 'too', 'n\'t', 'no', 'cannot', 'neither','nor']
    final_review=[]
    final_review.append(x[0])
    data = x[1]
    final_review.append(x[2])  
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
    print(final_review)                
    return final_review




print(map(naive_sentiment,processedData))


print(list(map(naive_sentiment,processedData))[0][1])
