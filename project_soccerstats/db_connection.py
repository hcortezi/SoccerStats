import pymongo
from pymongo import MongoClient

client = MongoClient("") #connection string
db = client['jogadores']