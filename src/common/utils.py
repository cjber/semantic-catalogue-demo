import tomllib
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv()


with open("./config/config.toml", "rb") as f:
    Config = tomllib.load(f)


class CDRCSettings(BaseSettings):
    api_url: str
    login_url: str


class DataStoreSettings(BaseSettings):
    index_name: str = Field(min_length=1)
    embed_dim: int = Field(gt=0, le=10_000)
    chunk_size: int = Field(gt=0, le=10_000)
    chunk_overlap: int = Field(ge=0, le=10_000)
    overwrite: bool


class ModelSettings(BaseSettings):
    top_k: int = Field(gt=0, le=100)
    vector_store_query_mode: str = Field(pattern="default|sparse|hybrid")
    alpha: float = Field(gt=0, le=1)
    prompt: str = Field(min_length=1)
    response_mode: str = Field(min_length=1)


class Settings(BaseSettings):
    model: ModelSettings = ModelSettings.model_validate(Config["model"])
    datastore: DataStoreSettings = DataStoreSettings.model_validate(Config["datastore"])
    cdrc: CDRCSettings = CDRCSettings.model_validate(Config["cdrc-api"])


class Paths:
    DATA_DIR: Path = Path("data")
    CDRC_PROFILES_DIR: Path = DATA_DIR / "cdrc" / "profiles"
    ADR_DIR: Path = DATA_DIR / "adr" / "descriptions"
    PIPELINE_STORAGE: Path = Path("./pipeline_storage")


class Consts:
    OPENAI_MODEL = "gpt-4o"
    OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
