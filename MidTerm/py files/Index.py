import Text_Analytics_Model
import data_PreProcessing
import Sentimental_Analysis
import numpy as np
import pandas as pd
import ANN

if __name__ == '__main__':
    print("program started")
    FILE_PATH = "C:\\Users\\Nikhil\\Documents\\Python Notebooks\\BIA_660_B(Web_Mining_Notebook)\\project\\submit\\data\\il_reviews.csv"

# Data Preprocessing
    # This will give all the business_ids that belongs to Illinos
    # data_PreProcessing.filter_restaurant_businesses("../../Yelp_dataset/business.json")

    # This will give us data regarding Illinos in a CSV File
    # data_PreProcessing.filter_il_reviews("../../Yelp_dataset/reviews_1000000.json")

# Text Analytical model"
    df = pd.read_csv(FILE_PATH)
    textData = df["text"]
    processedData = df.text.tolist()
    tagged_POS = Text_Analytics_Model.tokenize_Lemmatize_POS(processedData)
    # print(tagged_POS.encode("utf-8"))

    tf_idf = Text_Analytics_Model.construct_tf_idf(processedData)
    print(tf_idf)

# Sentimental Analysis
    ans = Sentimental_Analysis.sentimental_Analysis(FILE_PATH)
    print(ans)

# ANN
    df_Labeled = ANN.label_data(FILE_PATH)
    ans = ANN.build_model_ann(tf_idf, df_Labeled.tags)
    print(ans)

# filter_restaurant_businesses("../../Yelp_dataset/business.json")
# filter_il_reviews("../../Yelp_dataset/reviews_1000000.json")
