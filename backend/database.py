from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "cvd_diagnostic")

class Database:
    client: AsyncIOMotorClient = None
    
    @classmethod
    async def connect_db(cls):
        cls.client = AsyncIOMotorClient(MONGODB_URL)
        print(f"✅ Connected to MongoDB at {MONGODB_URL}")
    
    @classmethod
    async def close_db(cls):
        if cls.client:
            cls.client.close()
            print("❌ MongoDB connection closed")
    
    @classmethod
    def get_database(cls):
        return cls.client[DATABASE_NAME]
    
    # ---------------- DIAGNOSIS ----------------
    @classmethod
    async def save_diagnosis(cls, patient_data):
        db = cls.get_database()
        collection = db.diagnoses
        
        document = {
            **patient_data,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        result = await collection.insert_one(document)
        return str(result.inserted_id)
    
    @classmethod
    async def get_diagnosis_by_id(cls, diagnosis_id):
        from bson import ObjectId
        db = cls.get_database()
        collection = db.diagnoses
        
        return await collection.find_one({"_id": ObjectId(diagnosis_id)})
    
    @classmethod
    async def get_patient_history(cls, patient_id):
        db = cls.get_database()
        collection = db.diagnoses
        
        cursor = collection.find({"patient_id": patient_id}).sort("created_at", -1)
        return await cursor.to_list(length=100)

    # ---------------- FEEDBACK ----------------
    @classmethod
    async def save_feedback(cls, feedback_data):
        """
        Stores:
        - diagnosis_id
        - prediction
        - user_feedback (Correct / Wrong)
        - correct_label (optional)
        - remarks (optional)
        """
        db = cls.get_database()
        collection = db.feedback
        
        document = {
            **feedback_data,
            "created_at": datetime.utcnow()
        }
        
        await collection.insert_one(document)
        return True
