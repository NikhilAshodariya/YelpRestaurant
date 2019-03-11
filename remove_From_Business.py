import numpy as np
import pandas as pd
import json
import csv

from pymongo import MongoClient

DB_HOST = "mongodb://127.0.0.1:27017/"
DB_NAME = "yelp"
DB_COLLECTION = "business"


client = MongoClient(DB_HOST)
db = client[DB_NAME]
collection = db[DB_COLLECTION]

cursor = collection.find({})

print("Code started successfully")
count = 0
for doc in cursor:
    print(count)
    count = count + 1

    if doc["categories"] == None:
        continue

    if "Food" in doc["categories"] or "Restaurants" in doc["categories"] or "Restaurant" in doc["categories"]:
        continue
    else:
        collection.delete_one(doc)

print("Code successfully exited")
