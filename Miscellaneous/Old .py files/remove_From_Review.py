from pymongo import MongoClient

DB_HOST = "mongodb://127.0.0.1:27017/"
DB_NAME = "yelp"
DB_COLLECTION = "review"


client = MongoClient(DB_HOST)
db = client[DB_NAME]
review_Collection = db[DB_COLLECTION]

review_Cursor = review_Collection.find(no_cursor_timeout=False)


business_Collection = db["business"]

print("Code started successfully")

skip = 0
total_Count = review_Collection.find({}).count()
limit = 10
count = 0


while count < total_Count:
    # print(f"count of while = {count}")
    review_Cursor = review_Collection.aggregate([
                                                    {
                                                        "$skip":skip
                                                    },
                                                    {
                                                        "$limit":limit
                                                    }
                                                ])

    for doc in review_Cursor:
        count = count + 1
        print(f"count of for = {count}")
        tempCursor = business_Collection.find({
            "business_id":doc["business_id"]
        })

        if tempCursor.count() == 0:
            review_Collection.delete_one(doc)


    skip = skip + limit
print("Code successfully exited")
