import logfire
from enum import Enum
from typing import List, Optional, Set
import hashlib
import uuid
from uuid import UUID

from pydantic import Field, UUID4
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

from models.base.vector_base import VectorBaseDocument
from settings import settings
from wiz_ai.models.base.notion_base import NotionBaseDocument
from wiz_ai.models.base.documents import ChunkingMixin, EmbeddedChunk
from wiz_ai.networks import EmbeddingModelSingleton
from wiz_ai.connectors.qdrant import qdrant_connection

embedding_model = EmbeddingModelSingleton()

class DocumentCategory(str, Enum):
    INSTALLATION = "Installation"
    DEPLOY = "Deploy"
    CONFIGS = "Configs"
    BOT_ERRORS = "Bot Errors"


class DocumentStatus(str, Enum):
    PUBLISHED = "Published"
    DRAFT = "Draft"


class NotionKnowledgeVectorDocument(VectorBaseDocument):
    """A vector document ready for embedding and storage in vector database."""
    content: str
    metadata: dict
    document_id: UUID4
    platform: str = "notion"
    author: str
    embedding: List[float]

    def __init__(self, **data):
        # Generate deterministic ID based on document_id and content
        if 'document_id' in data and 'content' in data:
            # Create a namespace UUID from the document_id
            namespace = uuid.UUID(str(data['document_id']))
            # Create a deterministic name by hashing the content
            content_hash = hashlib.md5(data['content'].encode()).hexdigest()
            # Create a deterministic UUID5 using namespace and content hash
            data['id'] = uuid.uuid5(namespace, content_hash)
        
        # Get embedding before initializing
        if 'content' in data and 'embedding' not in data:
            data['embedding'] = embedding_model([data['content']], to_list=True)[0]
        
        super().__init__(**data)

    @classmethod
    def from_record(cls, point):
        """Convert a Qdrant record back to a NotionKnowledgeVectorDocument."""
        payload = point.payload or {}
        
        # Extract metadata from payload
        metadata = payload.pop('metadata', {})
        
        # Create document with all necessary fields
        return cls(
            id=UUID(point.id),
            content=payload.get('content', ''),
            document_id=UUID(payload.get('document_id')),
            platform=payload.get('platform', 'notion'),
            author=payload.get('author', 'Unknown'),
            metadata=metadata,
            embedding=point.vector if point.vector else []
        )

    @classmethod
    def get_collection_name(cls):
        return "knowledge_base"

    @classmethod
    def delete_by_notion_id(cls, notion_id: str) -> None:
        """Delete all vectors belonging to a specific Notion document."""
        filter_condition = Filter(
            must=[
                FieldCondition(
                    key="metadata.notion_id",
                    match=MatchValue(value=notion_id)
                )
            ]
        )
        
        # First find all matching documents to verify they exist
        matching_docs = cls.search(
            query_vector=[0] * embedding_model.embedding_size,  # Dummy vector for search
            query_filter=filter_condition,
            limit=1000  # High limit to get all chunks
        )
        
        if matching_docs:
            qdrant_connection.delete(
                collection_name=cls.get_collection_name(),
                points_selector=filter_condition
            )
            logfire.info(f"Deleted {len(matching_docs)} vectors for Notion document {notion_id}")
        else:
            logfire.info(f"No existing vectors found for Notion document {notion_id}")

class KnowledgeBaseDocument(NotionBaseDocument, ChunkingMixin):
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

    def to_vector_documents(self, chunk_size: int = 1000) -> List[NotionKnowledgeVectorDocument]:
        """Convert this document into a list of vector documents ready for embedding."""
        chunks = []
        current_chunk = []
        current_size = 0
        
        # Start with title and summary
        header = f"# {self.title}\n\n{self.summary}\n\n"
        current_chunk.append(header)
        current_size = len(header)
        
        if not self.content:
            # If no content, just return the header as one chunk
            return [self._create_vector_doc("".join(current_chunk))]
            
        # Split content by headers and dividers
        lines = self.content.split('\n')
        
        for line in lines:
            # Check if line is a header or divider
            is_separator = line.startswith('#') or line.strip() == '---'
            line_with_newline = line + '\n'
            line_size = len(line_with_newline)
            
            if is_separator and current_chunk and (current_size + line_size > chunk_size):
                # If we hit a separator and current chunk would be too big, start new chunk
                chunks.append(self._create_vector_doc("".join(current_chunk)))
                current_chunk = []
                current_size = 0
            
            current_chunk.append(line_with_newline)
            current_size += line_size
            
            # If chunk size exceeded, create new chunk
            if current_size >= chunk_size:
                chunks.append(self._create_vector_doc("".join(current_chunk)))
                current_chunk = []
                current_size = 0
        
        # Add remaining content as final chunk
        if current_chunk:
            chunks.append(self._create_vector_doc("".join(current_chunk)))
        
        return chunks

    def _create_vector_doc(self, content: str) -> NotionKnowledgeVectorDocument:
        """Helper method to create a vector document with metadata."""
        return NotionKnowledgeVectorDocument(
            content=content.strip(),
            document_id=self.id,
            author=self.author if self.author else "Unknown",
            metadata={
                "notion_id": self.notion_id,
                "title": self.title,
                "categories": [cat.value for cat in self.categories],
                "status": self.status.value,
                "deprecated": self.deprecated,
            }
        )



