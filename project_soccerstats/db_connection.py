import pymongo
from pymongo import MongoClient

client = MongoClient("mongodb+srv://robsonbsfilho:<password>@jogadoresteste.cogvr0m.mongodb.net/?retryWrites=true&w=majority&appName=JogadoresTeste")
db = client['jogadores']