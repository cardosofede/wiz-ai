import uuid
from abc import ABC
from typing import Any, Dict, Generic, Type, TypeVar, Optional, List


from datetime import datetime
from enum import Enum

import logfire
from pydantic import BaseModel, Field, UUID4

from wiz_ai.connectors.notion import notion

T = TypeVar("T", bound="NotionBaseDocument")


class NotionBaseDocument(BaseModel, Generic[T], ABC):
    notion_id: Optional[str] = None
    id: UUID4 = Field(default_factory=uuid.uuid4)
    database_id: str
    last_edited_time: Optional[datetime] = None
    created_time: Optional[datetime] = None
    content: Optional[str] = None

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
        # Query content

        
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
    async def bulk_find(cls: Type[T], database_id:str, **filter_options) -> list[T]:
        """Find multiple pages in the Notion database."""
        try:
            response = await notion.databases.query(
                database_id=database_id,
                filter=cls._build_notion_filter(filter_options)
            )
            return [cls.from_notion(page) for page in response["results"]]
        except Exception as e:
            logfire.error(f"Failed to retrieve Notion pages: {e}")
            return []

    async def update_content(self) -> Optional[str]:
        """Retrieve the content of a Notion page and transform it into markdown."""
        if not self.notion_id:
            return None
            
        try:
            response = await notion.blocks.children.list(block_id=self.notion_id)
            content = []
            
            for block in response["results"]:
                block_type = block.get("type")
                block_content = block.get(block_type, {})
                
                if block_type == "paragraph":
                    text = self._get_rich_text(block_content.get("rich_text", []))
                    if text:
                        content.append(text + "\n\n")
                
                elif block_type in ["heading_1", "heading_2", "heading_3"]:
                    level = int(block_type[-1])
                    text = self._get_rich_text(block_content.get("rich_text", []))
                    if text:
                        content.append(f"{'#' * level} {text}\n\n")
                
                elif block_type == "bulleted_list_item":
                    text = self._get_rich_text(block_content.get("rich_text", []))
                    if text:
                        content.append(f"* {text}\n")
                
                elif block_type == "numbered_list_item":
                    text = self._get_rich_text(block_content.get("rich_text", []))
                    if text:
                        content.append(f"1. {text}\n")
                
                elif block_type == "quote":
                    text = self._get_rich_text(block_content.get("rich_text", []))
                    if text:
                        content.append(f"> {text}\n\n")
                
                elif block_type == "to_do":
                    text = self._get_rich_text(block_content.get("rich_text", []))
                    checked = block_content.get("checked", False)
                    if text:
                        content.append(f"- [{'x' if checked else ' '}] {text}\n")
                
                elif block_type == "divider":
                    content.append("---\n\n")
                
            self.content = "".join(content)
            
        except Exception as e:
            logfire.error(f"Failed to retrieve Notion page content: {e}")

    def _get_rich_text(self, rich_text: list) -> str:
        """Extract text content from Notion's rich text array."""
        if not rich_text:
            return ""
            
        text_parts = []
        for text in rich_text:
            content = text.get("text", {}).get("content", "")
            annotations = text.get("annotations", {})
            
            if annotations.get("bold"):
                content = f"**{content}**"
            if annotations.get("italic"):
                content = f"_{content}_"
            if annotations.get("strikethrough"):
                content = f"~~{content}~~"
            if annotations.get("code"):
                content = f"`{content}`"
                
            text_parts.append(content)
            
        return "".join(text_parts)

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



class NotionUser(BaseModel):
    id: str
    object: str = "user"


class ExternalIcon(BaseModel):
    url: str


class Icon(BaseModel):
    type: str
    external: ExternalIcon


class PropertyType(str, Enum):
    TITLE = "title"
    RICH_TEXT = "rich_text"
    MULTI_SELECT = "multi_select"
    STATUS = "status"
    CHECKBOX = "checkbox"
    DATE = "date"
    PEOPLE = "people"


class TextContent(BaseModel):
    content: str
    link: Optional[dict] = None


class TextAnnotations(BaseModel):
    bold: bool = False
    code: bool = False
    color: str = "default"
    italic: bool = False
    strikethrough: bool = False
    underline: bool = False


class TitleText(BaseModel):
    type: str = "text"
    text: TextContent
    annotations: TextAnnotations
    href: Optional[str] = None
    plain_text: str


class DatabaseProperty(BaseModel):
    id: str
    name: str
    type: PropertyType


class DatabaseProperties(BaseModel):
    title: DatabaseProperty = Field(alias="Title")
    summary: DatabaseProperty = Field(alias="Summary")
    category: DatabaseProperty = Field(alias="Category")
    status: DatabaseProperty = Field(alias="Status")
    deprecated: DatabaseProperty = Field(alias="Deprecated")
    author: DatabaseProperty = Field(alias="Author")
    created: DatabaseProperty = Field(alias="Created")
    last_updated: DatabaseProperty = Field(alias="Last Updated")
    related_documents: DatabaseProperty = Field(alias="Related Documents")


class DatabaseParent(BaseModel):
    type: str = "page_id"
    page_id: str


class NotionDatabase(BaseModel):
    id: str
    object: str = "database"
    title: List[TitleText]
    description: List[dict] = Field(default_factory=list)
    properties: DatabaseProperties
    parent: DatabaseParent
    url: str
    public_url: Optional[str] = None
    archived: bool = False
    icon: Optional[Icon] = None
    cover: Optional[dict] = None
    is_inline: bool = True
    created_time: datetime
    created_by: NotionUser
    last_edited_time: datetime
    last_edited_by: NotionUser
    in_trash: bool = False

    @property
    def name(self) -> str:
        """Get the database name from the title."""
        if not self.title:
            return "Untitled"
        return self.title[0].plain_text

    @classmethod
    async def list_databases(cls) -> List["NotionDatabase"]:
        """List all available Notion databases the integration has access to."""
        try:
            response = await notion.search(
                filter={"property": "object", "value": "database"}
            )
            return [NotionDatabase.model_validate(db) for db in response["results"]]
        except Exception as e:
            logfire.error(f"Failed to list databases: {e}")
            return []

    @classmethod
    async def get_database(cls, database_id: str):
        """Get a specific database by ID."""
        try:
            response = await notion.databases.retrieve(database_id=database_id)
            return NotionDatabase.model_validate(response)
        except Exception as e:
            logfire.error(f"Failed to get database {database_id}: {e}")
            return None