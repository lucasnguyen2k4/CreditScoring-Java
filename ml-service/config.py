"""
ML Service Configuration
Loads environment variables and provides config for the ML service.
"""

import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings loaded from environment variables"""

    # Server
    HOST: str = os.getenv("ML_SERVICE_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("ML_SERVICE_PORT", "8000"))
    DEBUG: bool = os.getenv("ML_DEBUG", "true").lower() == "true"

    # CORS - Allow Spring Boot backend
    CORS_ORIGINS: list = os.getenv(
        "CORS_ORIGINS", "http://localhost:8080,http://localhost:5173"
    ).split(",")

    # LLM Configuration
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")
    GOOGLE_MODEL: str = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "google")
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "8000"))
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))

    # File Upload
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./uploads")

    @classmethod
    def is_llm_configured(cls) -> bool:
        return cls.GOOGLE_API_KEY is not None


settings = Settings()
