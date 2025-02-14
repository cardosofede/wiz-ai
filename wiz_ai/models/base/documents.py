from abc import ABC

from pydantic import UUID4, Field

from models.base.nosql_base import NoSQLBaseDocument


class RawDocument(NoSQLBaseDocument, ABC):
    content: dict
    platform: str
    author_id: UUID4 = Field(alias="author_id")
    author_full_name: str = Field(alias="author_full_name")