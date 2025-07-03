"""Base settings class contains only important fields."""
# mypy: ignore-errors
import ast
import os
import secrets
from typing import List, Union, Dict, Optional, Any
from pydantic import BaseModel, AnyHttpUrl, BaseSettings, validator, PostgresDsn
from ..utils.logging import StandardFormatter, ColorFormatter


class LoggingConfig(BaseModel):
    version: int
    disable_existing_loggers: bool = False
    formatters: Dict
    handlers: Dict
    loggers: Dict


class Settings(BaseSettings):
    PROJECT_NAME: str = 'Image Segmentation'
    PROJECT_SLUG: str = 'image_segmentation'

    DEBUG: bool = True
    API_STR: str = "/api/v1"

    # ##################### Access Token Configuration #########################
    # TODO: Please note that, the secret key will be different for each running
    # instance or each time restart the service, if you prefer a stable one,
    # please use an environment variable.
    SECRET_KEY: str = secrets.token_urlsafe(32)
    # 60 minutes * 24 hours * 8 days = 8 days
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8
    # 60 minutes * 24 hours * 30 * 6  months = 6 months
    ACCESS_TOKEN_EXPIRE_MINUTES_ADMIN: int = 60 * 24 * 30 * 6
    JWT_ENCODE_ALGORITHM: str = "HS256"

    # ########################### CORS Configuration ###########################
    """CORS_ORIGINS is a JSON-formatted list of origins
    e.g: ["http://localhost", "http://localhost:4200", "http://localhost:3000",
    "http://localhost:8080"]"""
    CORS_ORIGINS: Union[List[AnyHttpUrl], str] = []
    """A regex string to match against origins that should be permitted to make
    cross-origin requests.
    For example, your domain is example, then the regex should be something like
        ```r"https:\/\/.*\.example\.?"```
    """
    CORS_ORIGIN_REGEX: str = None
    """A list of HTTP methods that should be allowed for cross-origin requests.
    Defaults to ['*']. You can use ['GET'] to allow standard GET method."""
    CORS_METHODS: List[str] = ['GET']
    """A list of HTTP request headers that should be supported for cross-origin
    requests. Defaults to ['*'] to allow all headers. """
    CORS_HEADERS: List[str] = []
    """ Indicate that cookies should be supported for cross-origin requests.
    Defaults to True."""
    CORS_CREDENTIALS: bool = True

    # noinspection PyMethodParameters
    @validator("CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        """Validate the value of BACKEND_CORS_ORIGINS.

        Args:
            v (Union[str, List[str]): the value of BACKEND_CORS_ORIGINS.

        Returns:
            A list of urls, if v is a list of str in string format.
            The given value v, if v is a list or string.

        Raises
            ValueError, if v is in other format.
        """
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, str) and v.startswith("[") and v.endswith("]"):
            return ast.literal_eval(v)
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    # ########################### DB Configuration #############################
    # Database type: 'postgresql' or 'sqlite'
    DATABASE_TYPE: str = os.getenv("DATABASE_TYPE", "postgresql")
    
    # PostgreSQL Configuration
    POSTGRES_SERVER: str = os.getenv("POSTGRES_SERVER")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB")
    
    # SQLite Configuration
    SQLITE_DATABASE_PATH: str = os.getenv("SQLITE_DATABASE_PATH", "data/app.db")
    
    # set the default value to None, such that the assemble_db_connection can
    # build the URI for us and do checks.
    SQLALCHEMY_DATABASE_URI: Optional[str] = None

    # noinspection PyMethodParameters
    @validator("SQLALCHEMY_DATABASE_URI", pre=True)
    def assemble_db_connection(cls, v: Optional[str], values: Dict[str, Any]) -> str:
        """Assemble the database URI based on the DATABASE_TYPE setting.
        
        For PostgreSQL: builds the postgres DB URI with the provided POSTGRES_SERVER,
        POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB.
        
        For SQLite: builds the sqlite DB URI with the provided SQLITE_DATABASE_PATH.

        Args:
            v (Optional[str]): the value of defined SQLALCHEMY_DATABASE_URI.
            values (Dict[str, Any]): a dictionary contains the requisite values.

        Returns:
            str: the database URI.
        """
        if isinstance(v, str):
            return v
            
        database_type = values.get("DATABASE_TYPE", "postgresql")
        
        if database_type == "sqlite":
            sqlite_path = values.get("SQLITE_DATABASE_PATH", "data/app.db")
            # Create the directory if it doesn't exist
            import os
            if "/" in sqlite_path:
                os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
            return f"sqlite:///{sqlite_path}"
        else:
            # Default to PostgreSQL - validate required fields
            postgres_server = values.get("POSTGRES_SERVER")
            postgres_user = values.get("POSTGRES_USER")
            postgres_password = values.get("POSTGRES_PASSWORD")
            postgres_db = values.get("POSTGRES_DB")
            
            if not all([postgres_server, postgres_user, postgres_password, postgres_db]):
                raise ValueError(
                    "PostgreSQL configuration requires POSTGRES_SERVER, POSTGRES_USER, "
                    "POSTGRES_PASSWORD, and POSTGRES_DB to be set"
                )
            
            return PostgresDsn.build(
                scheme="postgresql",
                user=postgres_user,
                password=postgres_password,
                host=postgres_server,
                path=f"/{postgres_db}",
            )

    # ######################## Logging Configuration ###########################
    # logging configuration for the project logger, uvicorn loggers
    LOGGING_CONFIG: LoggingConfig = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            'colorFormatter': {'()': ColorFormatter},
            'standardFormatter': {'()': StandardFormatter},
        },
        "handlers": {
            'consoleHandler': {
                'class': 'logging.StreamHandler',
                'level': "DEBUG",
                'formatter': 'standardFormatter',
                'stream': 'ext://sys.stdout',
            },
        },
        "loggers": {
            "image_segmentation": {
                'handlers': ['consoleHandler'],
                'level': "DEBUG",
            },
            "uvicorn": {
                'handlers': ['consoleHandler']
            },
            "uvicorn.access": {
                # Use the project logger to replace uvicorn.access logger
                'handlers': []
            }
        }
    }

    class Config:
        case_sensitive = True
