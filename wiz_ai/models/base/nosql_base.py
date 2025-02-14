import uuid
from abc import ABC
from typing import Generic, Type, TypeVar

import logfire
from pydantic import UUID4, BaseModel, Field
from pymongo import errors

from wiz_ai.connectors.mongo import connection
from wiz_ai.settings import settings

_database = connection.get_database(settings.DATABASE_NAME)


T = TypeVar("T", bound="NoSQLBaseDocument")


class NoSQLBaseDocument(BaseModel, Generic[T], ABC):
    id: UUID4 = Field(default_factory=uuid.uuid4)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, self.__class__):
            return False

        return self.id == value.id

    def __hash__(self) -> int:
        return hash(self.id)

    @classmethod
    def from_mongo(cls: Type[T], data: dict) -> T:
        """Convert "_id" (str object) into "id" (UUID object)."""

        if not data:
            raise ValueError("Data is empty.")

        id = data.pop("_id")

        return cls(**dict(data, id=id))

    def to_mongo(self: T, **kwargs) -> dict:
        """Convert "id" (UUID object) into "_id" (str object)."""
        exclude_unset = kwargs.pop("exclude_unset", False)
        by_alias = kwargs.pop("by_alias", True)

        parsed = self.model_dump(exclude_unset=exclude_unset, by_alias=by_alias, **kwargs)

        if "_id" not in parsed and "id" in parsed:
            parsed["_id"] = str(parsed.pop("id"))

        for key, value in parsed.items():
            if isinstance(value, uuid.UUID):
                parsed[key] = str(value)

        return parsed

    def model_dump(self: T, **kwargs) -> dict:
        dict_ = super().model_dump(**kwargs)

        for key, value in dict_.items():
            if isinstance(value, uuid.UUID):
                dict_[key] = str(value)

        return dict_

    async def save(self: T, **kwargs) -> T | None:
        collection = _database[self.get_collection_name()]
        try:
            await collection.insert_one(self.to_mongo(**kwargs))
            return self
        except errors.WriteError:
            logfire.exception("Failed to insert document.")
            return None

    @classmethod
    async def get_or_create(cls: Type[T], **filter_options) -> T:
        collection = _database[cls.get_collection_name()]
        try:
            instance = await collection.find_one(filter_options)
            if instance:
                return cls.from_mongo(instance)

            new_instance = cls(**filter_options)
            new_instance = await new_instance.save()

            return new_instance
        except errors.OperationFailure:
            logfire.exception(f"Failed to retrieve document with filter options: {filter_options}")
            raise

    @classmethod
    async def bulk_insert(cls: Type[T], documents: list[T], **kwargs) -> bool:
        collection = _database[cls.get_collection_name()]
        try:
            await collection.insert_many([doc.to_mongo(**kwargs) for doc in documents])
            return True
        except (errors.WriteError, errors.BulkWriteError):
            logfire.error(f"Failed to insert documents of type {cls.__name__}")
            return False

    @classmethod
    async def find(cls: Type[T], **filter_options) -> T | None:
        collection = _database[cls.get_collection_name()]
        try:
            instance = await collection.find_one(filter_options)
            if instance:
                return cls.from_mongo(instance)
            return None
        except errors.OperationFailure:
            logfire.error("Failed to retrieve document")
            return None

    @classmethod
    async def bulk_find(cls: Type[T], **filter_options) -> list[T]:
        collection = _database[cls.get_collection_name()]
        try:
            cursor = collection.find(filter_options)
            documents = []
            async for instance in cursor:
                if document := cls.from_mongo(instance):
                    documents.append(document)
            return documents
        except errors.OperationFailure:
            logfire.error("Failed to retrieve documents")
            return []

    @classmethod
    def get_collection_name(cls):
        return cls.__name__
