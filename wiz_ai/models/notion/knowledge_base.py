from datetime import datetime
from enum import Enum
from typing import List, Optional, Set

from pydantic import Field

from settings import settings
from wiz_ai.models.base.notion_base import NotionBaseDocument


class DocumentCategory(str, Enum):
    INSTALLATION = "Installation"
    DEPLOY = "Deploy"
    CONFIGS = "Configs"
    BOT_ERRORS = "Bot Errors"


class DocumentStatus(str, Enum):
    PUBLISHED = "Published"
    DRAFT = "Draft"


class KnowledgeBaseDocument(NotionBaseDocument):
    database_id: str = settings.KNOWLEDGE_BASE_DB_ID  # Knowledge Base database ID
    
    title: str
    summary: str
    categories: Set[DocumentCategory]
    status: DocumentStatus = DocumentStatus.DRAFT
    deprecated: bool = False
    author: Optional[str] = None
    related_documents: List[str] = Field(default_factory=list)  # List of related document IDs

    @classmethod
    def _map_notion_properties(cls, properties: dict) -> dict:
        return {
            "title": cls._get_title_content(properties.get("Title", {}).get("title", [])),
            "summary": cls._get_rich_text_content(properties.get("Summary", {}).get("rich_text", [])),
            "categories": cls._get_multiselect_values(properties.get("Category", {}).get("multi_select", [])),
            "deprecated": cls._get_checkbox_value(properties.get("Deprecated", {}).get("checkbox")),
            "status": cls._get_status_value(properties.get("Status", {}).get("status")),
            "author": cls._get_rich_text_content(properties.get("Author", {}).get("rich_text", [])),
            "related_documents": cls._get_relation_values(properties.get("Related Documents", {}).get("relation", [])),
        }

    def _map_model_properties(self) -> dict:
        return {
            "Title": {
                "title": [{"text": {"content": self.title}}]
            },
            "Summary": {
                "rich_text": [{"text": {"content": self.summary}}]
            },
            "Category": {
                "multi_select": [{"name": category.value} for category in self.categories]
            },
            "Deprecated": {
                "checkbox": self.deprecated
            },
            "Status": {
                "status": {"name": self.status.value}
            },
            "Author": {
                "rich_text": [{"text": {"content": self.author}}] if self.author else []
            },
            "Related Documents": {
                "relation": [{"id": doc_id} for doc_id in self.related_documents]
            }
        }

    @staticmethod
    def _get_title_content(title_list: list) -> str:
        if not title_list:
            return ""
        return "".join(text.get("plain_text", "") for text in title_list)

    @staticmethod
    def _get_rich_text_content(rich_text_list: list) -> str:
        if not rich_text_list:
            return ""
        return "".join(text.get("plain_text", "") for text in rich_text_list)

    @staticmethod
    def _get_multiselect_values(multiselect_list: list) -> Set[DocumentCategory]:
        if not multiselect_list:
            return set()
        return {DocumentCategory(item.get("name")) for item in multiselect_list if item.get("name")}

    @staticmethod
    def _get_checkbox_value(checkbox_value: Optional[bool]) -> bool:
        return bool(checkbox_value)

    @staticmethod
    def _get_status_value(status_obj: Optional[dict]) -> DocumentStatus:
        if not status_obj or "name" not in status_obj:
            return DocumentStatus.DRAFT
        return DocumentStatus(status_obj["name"])

    @staticmethod
    def _get_relation_values(relation_list: list) -> List[str]:
        if not relation_list:
            return []
        return [item.get("id") for item in relation_list if item.get("id")]

    @classmethod
    def _build_notion_filter(cls, filter_options: dict) -> dict:
        """Customized filter builder for knowledge base documents."""
        notion_filter = {"and": []}
        
        for key, value in filter_options.items():
            if key == "categories" and isinstance(value, (list, set)):
                # Handle multiple categories as OR condition
                category_conditions = []
                for category in value:
                    category_conditions.append({
                        "property": "Category",
                        "multi_select": {
                            "contains": category.value if isinstance(category, DocumentCategory) else str(category)
                        }
                    })
                if category_conditions:
                    notion_filter["and"].append({"or": category_conditions})
            
            elif key == "deprecated":
                notion_filter["and"].append({
                    "property": "Deprecated",
                    "checkbox": {
                        "equals": bool(value)
                    }
                })
            
            elif key == "status":
                notion_filter["and"].append({
                    "property": "Status",
                    "status": {
                        "equals": value.value if isinstance(value, DocumentStatus) else str(value)
                    }
                })
            
            elif key == "title":
                notion_filter["and"].append({
                    "property": "Title",
                    "title": {
                        "contains": str(value)
                    }
                })
            
            elif key == "summary":
                notion_filter["and"].append({
                    "property": "Summary",
                    "rich_text": {
                        "contains": str(value)
                    }
                })

        return notion_filter if notion_filter["and"] else {} 