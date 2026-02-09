import os
from dotenv import load_dotenv

load_dotenv()

BRAPI_KEY = os.getenv("API_KEY_BRAPI")

if not BRAPI_KEY:
    raise RuntimeError("API_KEY_BRAPI não encontrada. Configure no arquivo .env")
