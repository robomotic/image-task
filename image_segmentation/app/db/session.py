"""Define a session instance for doing all database related operations inside
the app."""
# mypy: ignore-errors
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
from ..configs import get_settings

settings = get_settings()

# Configure engine based on database type
if settings.DATABASE_TYPE == "sqlite":
    # SQLite specific configurations
    engine = create_engine(
        settings.SQLALCHEMY_DATABASE_URI,
        pool_pre_ping=True,
        connect_args={"check_same_thread": False}  # SQLite specific
    )
else:
    # PostgreSQL and other databases
    engine = create_engine(settings.SQLALCHEMY_DATABASE_URI, pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@contextmanager
def session_scope():
    """Provide a transactional scope around a series of operations.

    Yields:
        sqlalchemy.orm.Session: A local SQLAlchemy session.

    Examples:

        >>> with session_scope() as session:
        ...    session.add(Table1(url="https://www.example.com"))
        ...    session.commit()
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()
