from pymongo import MongoClient

DB_HOST = "mongodb://127.0.0.1:27017/"
DB_NAME = "yelp"
DB_COLLECTION = "review"


client = MongoClient(DB_HOST)
db = client[DB_NAME]
review_Collection = db[DB_COLLECTION]

review_Cursor = review_Collection.find({})


business_Collection = db["business"]

print("Code started successfully")
count = 0

for doc in review_Cursor:
    count = count + 1
    print(count)

    tempCursor = business_Collection.find({
        "business_id": doc["business_id"]
    })

    if tempCursor.count() == 0:
        review_Collection.delete_one(doc)

print("Code successfully exited")
