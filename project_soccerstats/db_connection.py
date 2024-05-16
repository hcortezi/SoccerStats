import pymongo
from pymongo import MongoClient

client = MongoClient("mongodb+srv://hcortezi10:9voCKDy8spievy8R@soccerstats.savault.mongodb.net/?retryWrites=true&w=majority&appName=soccerstats") #connection string
db = client['soccerstats']