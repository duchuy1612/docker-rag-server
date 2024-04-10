import os

MONGO_URI = os.environ.get("AI_BOT_MONGO_URI", "mongodb://0.0.0.0:27017")
API_TIMEOUT = 10