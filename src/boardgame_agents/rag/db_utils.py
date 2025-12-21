import os
from dotenv import load_dotenv

load_dotenv()
PG_DSN = os.getenv("DB_DSN", "")
