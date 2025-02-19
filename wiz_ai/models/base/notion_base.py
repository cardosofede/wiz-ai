import uuid
from abc import ABC
from datetime import datetime
from typing import Any, ClassVar, Dict, Generic, Optional, Type, TypeVar

import logfire
from pydantic import UUID4, BaseModel, Field

from wiz_ai.connectors.notion import notion

T = TypeVar("T", bound="NotionBaseDocument")


class NotionBaseDocument(BaseModel, Generic[T], ABC):
    notion_id: Optional[str] = None
    id: UUID4 = Field(default_factory=uuid.uuid4)
    database_id: str
    last_edited_time: Optional[datetime] = None
    created_time: Optional[datetime] = None

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, self.__class__):
            return False
        return self.id == value.id

    def __hash__(self) -> int:
        return hash(self.id)

    @classmethod
    def from_notion(cls: Type[T], data: dict) -> T:
        """Convert Notion page data into a model instance."""
        if not data:
            raise ValueError("Data is empty.")

        properties = data.get("properties", {})
        model_data = {
            "notion_id": data.get("id"),
            "last_edited_time": datetime.fromisoformat(data.get("last_edited_time", "").replace("Z", "+00:00")),
            "created_time": datetime.fromisoformat(data.get("created_time", "").replace("Z", "+00:00")),
        }

        # Each subclass should implement this method to handle specific property mappings
        model_data.update(cls._map_notion_properties(properties))
        
        return cls(**model_data)

    def to_notion(self) -> dict:
        """Convert model instance into Notion page properties."""
        # Each subclass should implement this method to handle specific property mappings
        properties = self._map_model_properties()
        
        return {
            "parent": {"database_id": self.database_id},
            "properties": properties
        }

    @classmethod
    def _map_notion_properties(cls, properties: Dict[str, Any]) -> dict:
        """Map Notion properties to model fields. Should be implemented by subclasses."""
        raise NotImplementedError

    def _map_model_properties(self) -> dict:
        """Map model fields to Notion properties. Should be implemented by subclasses."""
        raise NotImplementedError

    @classmethod
    async def find(cls: Type[T], **filter_options) -> Optional[T]:
        """Find a single page in the Notion database."""
        try:
            response = await notion.databases.query(
                database_id=cls.database_id,
                filter=cls._build_notion_filter(filter_options),
                page_size=1
            )
            if response["results"]:
                return cls.from_notion(response["results"][0])
            return None
        except Exception as e:
            logfire.error(f"Failed to retrieve Notion page: {e}")
            return None

    @classmethod
    async def bulk_find(cls: Type[T], **filter_options) -> list[T]:
        """Find multiple pages in the Notion database."""
        try:
            response = await notion.databases.query(
                database_id=cls.database_id,
                filter=cls._build_notion_filter(filter_options)
            )
            return [cls.from_notion(page) for page in response["results"]]
        except Exception as e:
            logfire.error(f"Failed to retrieve Notion pages: {e}")
            return []

    async def save(self) -> Optional[T]:
        """Create or update a page in the Notion database."""
        try:
            if self.notion_id:
                # Update existing page
                response = await notion.pages.update(
                    page_id=self.notion_id,
                    properties=self.to_notion()["properties"]
                )
            else:
                # Create new page
                response = await notion.pages.create(**self.to_notion())
            
            return self.__class__.from_notion(response)
        except Exception as e:
            logfire.error(f"Failed to save Notion page: {e}")
            return None

    @staticmethod
    def _build_notion_filter(filter_options: dict) -> dict:
        """Convert filter options to Notion filter format."""
        # This is a basic implementation. Subclasses might want to override this
        # to provide more sophisticated filtering
        notion_filter = {"and": []}
        for key, value in filter_options.items():
            notion_filter["and"].append({
                "property": key,
                "rich_text": {
                    "equals": str(value)
                }
            })
        return notion_filter if filter_options else {} 