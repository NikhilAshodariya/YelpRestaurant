import numpy as np
import pandas as pd
import json
import csv


df = pd.read_json("yelp_dataset/checkin.json", lines=True)
df.head()


# def process(element):
#     tempList = element.split(",")
#     inn = []
#     for v in tempList:
#         temp = v.strip().split(" ")
#         inn.append((temp[0], temp[1]))
#     return inn

# df["date_time"] = df.apply(lambda x:process(x[1]), axis = 1)
# df["business_id"] = df.apply(lambda x:x[0].replace("-","").replace("=",""),axis = 1)
# df = df.drop(["date"],axis = 1)

# df.head(6)


df.to_csv("./yelp_dataset/newCheckin.csv", encoding='utf-8')
