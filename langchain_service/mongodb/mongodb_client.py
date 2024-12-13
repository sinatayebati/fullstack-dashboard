import pymongo
from pymongo import MongoClient
from config.config import Config, MONGODB_URI


def connect_to_mongodb(config: Config):
    """Enhanced connection function that ensures collection exists"""
    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=config.MONGODB_TIMEOUT_MS)
        client.server_info()
        db = client[config.DB_NAME]
        
        # Explicitly create collection if it doesn't exist
        if config.COLLECTION_VECTOR not in db.list_collection_names():
            db.create_collection(config.COLLECTION_VECTOR)
            print(f"Created collection: {config.COLLECTION_VECTOR}")
        
        collection = db[config.COLLECTION_VECTOR]
        return client, db, collection
        
    except pymongo.errors.ServerSelectionTimeoutError:
        print("Failed to connect to MongoDB server")
        raise
    except pymongo.errors.OperationFailure as e:
        print(f"Authentication failed: {str(e)}")
        raise