import logging
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr, field_validator
import os


class Settings(BaseSettings):

    model_config = SettingsConfigDict(env_file=".env")
    anthropic_api_key : SecretStr = "ANTHROPIC_API_KEY"
    embedded_model : str = "all-MiniLM-L6-v2"
    knowledge_dir : str = "./knowledge-base"
    chunk_size : int = 1000
    chunk_overlap : int = 200
    chroma_persist_dir : str = "./chroma_db"
    message : str = "Help me, what is the cost of AK-47 in CS?"


    @field_validator("chunk_overlap")
    @classmethod
    def chunk_validation(cls, v, info):
        chunk_size = info.data.get("chunk_size")
        if v >= chunk_size:
            raise ValueError("Error, too big chunk_overlap")
        return v
    

    @property
    def knowledge_path(self):
        knowledge_path = os.path.abspath("./knowledge_base")
        return knowledge_path
    
    @property
    def chroma_path(self):
        chroma_path = os.path.abspath("./chroma_db")
        return chroma_path
    
settings = Settings()
    

