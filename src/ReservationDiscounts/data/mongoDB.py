



from pymongo import MongoClient  
from dotenv import load_dotenv
from src.ReservationDiscounts.logger import logger

load_dotenv()

class MongoDBConnection:
    """Handles MongoDB connections synchronously."""
    def __init__(self, uri, db_name, collection_name):
        self.uri = uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.client = None
        self.db = None
        self.collection = None

    def __enter__(self):
        self.client = MongoClient(self.uri)
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]
        logger.info("Connected to MongoDB Database")
        return self.collection
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed.")

