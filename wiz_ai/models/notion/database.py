from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field


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
    title: DatabaseProperty
    summary: DatabaseProperty
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