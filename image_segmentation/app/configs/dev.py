"""DEV Environment"""
# mypy: ignore-errors
from .base import Settings


class SettingsDev(Settings):
    DEBUG = True
    # Use SQLite by default in development
    DATABASE_TYPE: str = "sqlite"
    SQLITE_DATABASE_PATH: str = "data/dev.db"
