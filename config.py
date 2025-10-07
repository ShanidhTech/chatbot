from decouple import config
from functools import lru_cache


class Settings:
    """
    Application settings loaded from environment variables.
    Uses python-decouple to manage environment variables.
    """
    GOOGLE_API_KEY: str = config('GOOGLE_API_KEY')
    GEMINI_API_KEY: str = config('GEMINI_API_KEY')
    VECTORSTORE_DIR: str = config('VECTORSTORE_DIR')
    COLLECTION_NAME: str = config('COLLECTION_NAME')
    UPLOAD_DIR: str = config('UPLOAD_DIR')

    DATABASE_URL: str = config('DATABASE_URL')

    def __init__(self):
        pass


@lru_cache
def get_settings():
    """
    Get the application settings.
    This function caches the settings to avoid reloading them multiple times.
    """
    return Settings() 
  

settings = get_settings()