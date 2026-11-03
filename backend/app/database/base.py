"""SQLAlchemy 2.0 Declarative Base configuration and naming conventions."""

from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase

# Standardized constraint naming convention across SQLite and PostgreSQL
convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata_obj = MetaData(naming_convention=convention)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy 2.0 ORM models in OpenSourceLens."""

    metadata = metadata_obj

    def __repr__(self) -> str:
        """Informative representation with class name and primary key."""
        cols = [
            f"{col.name}={getattr(self, col.name)!r}"
            for col in self.__table__.primary_key.columns
        ]
        return f"<{self.__class__.__name__}({', '.join(cols)})>"
