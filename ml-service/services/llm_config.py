"""
=============================================================================
LLM INTEGRATION CONFIG - CẤU HÌNH TÍCH HỢP AI
=============================================================================
Mô tả:
    Module cấu hình cho các dịch vụ LLM (Large Language Model).
    Hỗ trợ nhiều provider: OpenAI, Anthropic Claude, Google Gemini.

Các biến môi trường cần thiết (trong file .env):
    - GOOGLE_API_KEY: API key cho Google Gemini (khuyến nghị)
    - OPENAI_API_KEY: API key cho OpenAI GPT (tùy chọn)
    - ANTHROPIC_API_KEY: API key cho Anthropic Claude (tùy chọn)
    - GOOGLE_MODEL: Model Gemini (mặc định: gemini-2.5-flash)
    - LLM_PROVIDER: Provider mặc định (mặc định: google)
    - LLM_MAX_TOKENS: Số token tối đa (mặc định: 8000)
    - LLM_TEMPERATURE: Temperature (mặc định: 0.7)

Cách sử dụng:
    from backend.llm_integration.config import LLMConfig
    
    # Kiểm tra đã cấu hình chưa
    if LLMConfig.is_configured():
        api_key = LLMConfig.get_api_key()
        model = LLMConfig.get_model()
=============================================================================
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load biến môi trường từ file .env
load_dotenv()


# =============================================================================
# CLASS CẤU HÌNH LLM
# =============================================================================


class LLMConfig:
    """Configuration for LLM services"""
    
    # OpenAI
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4")
    
    # Anthropic Claude
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229")
    
    # Google Gemini
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")
    GOOGLE_MODEL: str = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
    
    # General settings
    DEFAULT_PROVIDER: str = os.getenv("LLM_PROVIDER", "google")  # openai, anthropic, google
    MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "8000"))
    TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    
    @classmethod
    def get_api_key(cls, provider: Optional[str] = None) -> Optional[str]:
        """Get API key for specified provider"""
        provider = provider or cls.DEFAULT_PROVIDER
        
        if provider == "openai":
            return cls.OPENAI_API_KEY
        elif provider == "anthropic":
            return cls.ANTHROPIC_API_KEY
        elif provider == "google":
            return cls.GOOGLE_API_KEY
        else:
            return None
    
    @classmethod
    def get_model(cls, provider: Optional[str] = None) -> str:
        """Get model name for specified provider"""
        provider = provider or cls.DEFAULT_PROVIDER
        
        if provider == "openai":
            return cls.OPENAI_MODEL
        elif provider == "anthropic":
            return cls.ANTHROPIC_MODEL
        elif provider == "google":
            return cls.GOOGLE_MODEL
        else:
            return "unknown"
    
    @classmethod
    def is_configured(cls, provider: Optional[str] = None) -> bool:
        """Check if LLM is properly configured"""
        return cls.get_api_key(provider) is not None
