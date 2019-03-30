import numpy as np
import pandas as pd
import nltk
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag
import string

def filter_restaurant_businesses(file):
    f = pd.read_json(file, lines=True)
    filtered_df = df[(df.categories.str.contains('Restaurants')==True) & (df.state.str.contains('IL')==True)]
    il_business = filtered_df.business_id.to_string(index=False)
    with open('./il_business_ids.txt', 'w') as outfile:
        outfile.write(az_business)

def filter_il_reviews():
    df = pd.read_json('reviews_1000000.json', lines=True)
    business_ids = df.business_id.to_list()
    df.set_index('business_id', inplace=True)
    reviews_df = pd.DataFrame()

    with open('./il_business_ids.txt', 'r') as infile:
        for b_id in infile:
            b_id = b_id.strip()
            if b_id in business_ids:
                data = df.loc[[b_id]]
                reviews_df = reviews_df.append(data)
    with open('./il_reviews_test.csv', 'a') as outfile:
        reviews_df.to_csv(outfile, header=False)
        
def funcTemp(x):
    WNlemma = nltk.WordNetLemmatizer()
    res = []
    ans = []
    ans.append(x[0])
    data = x[1]

    for val in sent_tokenize(data):
        val = val.strip(string.punctuation)
        res.append(pos_tag([WNlemma.lemmatize(w.strip(string.punctuation)) for w in word_tokenize(val) if w.strip(string.punctuation)!=""]))

    ans.append(res)
    return ans

def label_data(path):
    df = pd.read_csv(path)
    reviews = df.text
    
    food_tags = np.where(df.text.str.contains('food'), 'food', None)
    cleanliness_tags = np.where(df.text.str.contains('clean|dirty'), 'cleanliness', None)
    service_tags = np.where(df.text.str.contains('service|waitress|hostess|waiter|worker'), 'service', None)
    ambience_tags = np.where(df.text.str.contains('ambience|place'), 'ambience', None)

    tags = zip(food_tags, cleanliness_tags, service_tags, ambience_tags)
    tag_list = list(tags)
    tag_list = [list(tags) for tags in tag_list]
    valid_tag_list = []
    for tags in tag_list:
        for tag in tags:
            valid_list = [t for t in tags if t is not None]
        valid_tag_list.append(valid_list)

    df['tags'] = valid_tag_list
    return df
if __name__ == "__main__":

    #The dataset has different business entities such as grocery stores, furnitures, plumbing services, etc., 
    #apart from restaurants. This method filters businesses that belong to restaurants.
    business_ids = './business.json'
    filter_restaurant_businesses(file) 
    
    #Filter reviews that belong to the state 'IL'
    filter_il_reviews()
    
    #Label each review with any or all of the following tags: food, cleanliness, service, ambience
    path = "./il_reviews.csv"
    df = label_data(path)
    
    #Pre-process text
    df.drop(["cool","date","funny","useful"],axis = 1,inplace=True)
    processedData = list(df[["business_id","text"]].as_matrix())
    processed_doc = list(map(funcTemp, processedData))
    
    