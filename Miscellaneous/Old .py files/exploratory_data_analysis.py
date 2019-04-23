import numpy as np
import pandas as pd
import nltk
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag
import string
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.preprocessing.text import Tokenizer



def filter_restaurant_businesses(file):
    df = pd.read_json(file, lines=True)
    filtered_df = df[(df.categories.str.contains('Restaurants')==True) & (df.state.str.contains('IL')==True)]
    il_business = filtered_df.business_id.to_string(index=False)
    with open('./il_business_ids.txt', 'w') as outfile:
        outfile.write(il_business)

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
        
def tokenize_Lemmatize_POS(data):
    ans = []
    for val in sent_tokenize(data):
        val = val.strip(string.punctuation)
        ans.extend([WNlemma.lemmatize(w.strip(string.punctuation)) for w in word_tokenize(val) if w.strip(string.punctuation)!=""])
    return pos_tag(ans)


def label_data(path):
    df = pd.read_csv(path)
    reviews = df.text
    
    food_tags = np.where(df.text.str.contains('food|vegetables|veggie|veggies|meat|chicken|pho|soup|lunch|dinner|menu|bland|flavor'), 'food', None)
    cleanliness_tags = np.where(df.text.str.contains('clean|dirty|hygiene'), 'cleanliness', None)
    service_tags = np.where(df.text.str.contains('service|waitress|hostess|waiter|worker|staff|'), 'service', None)
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

def construct_tf_idf(reviews):
    t = Tokenizer()
    t.fit_on_texts(reviews)
    tf_idf = t.texts_to_matrix(reviews, mode='count')
    return tf_idf


def build_model_ann(tf_idf, labels):
    binarizer = MultiLabelBinarizer()
    labels = binarizer.fit_transform(labels)

    X = tf_idf
    y = labels
    X_train, X_test = X[:5000], X[5000:]
    y_train, y_test = y[:5000], y[5000:]
    
    model = Sequential()
    model.add(Dense(1000, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dropout(0.1))
    model.add(Dense(y_train.shape[1], activation='sigmoid'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd)

    model.fit(X_train, y_train, epochs=5, batch_size=2000)

    preds = model.predict(X_test)
    preds[preds>=0.5] = 1
    preds[preds<0.5] = 0
    label = ['food', 'cleanliness', 'ambience', 'service']
    print(classification_report\
      (y_test, preds, target_names=label))
    
if __name__ == "__main__":

    #The dataset has different business entities such as grocery stores, furnitures, plumbing services, etc., 
    #apart from restaurants. This method filters businesses that belong to restaurants.
    business_ids = './business.json'
    filter_restaurant_businesses(file) 
    
    #Filter reviews that belong to the state 'IL'
    filter_il_reviews()
    
    #Label each review with any or all of the following tags: food, cleanliness, service, ambience
    path = "/Users/revathyramasundaram/revathy/Stevens/BIA-660-Web-mining/yelp_dataset/il_reviews_test.csv"
    labelled_df = label_data()
    reviews = labelled_df.text
    labels = labelled_df.tags
    
    #Pre-process text
    df.drop(["cool","date","funny","useful"],axis = 1,inplace=True)
    processedData = df.text.tolist()
    processed_doc = list(map(tokenize_Lemmatize_POS, processedData))
    
    #Construct tf-idf matrix and send it to the Machine Learning model
    tf_idf = construct_tf_idf(reviews)
    build_model_ann(tf_idf, labels)
    
