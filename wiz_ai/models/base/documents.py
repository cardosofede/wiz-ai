import re
from abc import abstractmethod
from typing import cast

from langchain_text_splitters import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from pydantic import UUID4, Field, BaseModel

from wiz_ai.models.base.nosql_base import NoSQLBaseDocument
from wiz_ai.models.base.vector_base import VectorBaseDocument
from wiz_ai.networks import EmbeddingModelSingleton


class BaseDocument(BaseModel):
    platform: str
    author_id: UUID4 = Field(alias="author_id")
    author_full_name: str = Field(alias="author_full_name")

class RawDocument(BaseDocument, NoSQLBaseDocument):
    content: dict

    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r"[^\w\s.,!?]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def clean(self) -> "CleanedDocument":
        return CleanedDocument(
            platform=self.platform,
            author_id=self.author_id,
            author_full_name=self.author_full_name,
            content=self.clean_text(" #### ".join([content for content in self.content.values() if content is not None])),
        )

class ChunkingMixin:
    """Mixin class that provides chunking and embedding functionality for documents."""
    embedding_model = EmbeddingModelSingleton()
    
    @property
    def metadata(self) -> dict:
        return {
            "chunk_size": 500,
            "chunk_overlap": 50,
        }

    def chunk_text(self, text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
        # First split by markdown section separators
        character_splitter = RecursiveCharacterTextSplitter(
            separators=["---\n", "---\r\n", "\n---\n", "\r\n---\r\n"],  # Various forms of markdown section separators
            chunk_size=chunk_size * 10,  # Larger chunk size to keep sections together
            chunk_overlap=0  # No overlap for sections
        )
        text_split_by_sections = character_splitter.split_text(text)

        # Then use token splitter only if sections are too large
        token_splitter = SentenceTransformersTokenTextSplitter(
            chunk_overlap=chunk_overlap,
            tokens_per_chunk=self.embedding_model.max_input_length,
            model_name=self.embedding_model.model_id,
        )
        
        chunks_by_tokens = []
        for section in text_split_by_sections:
            # Only split section if it exceeds the token limit
            if len(section.split()) > self.embedding_model.max_input_length:
                chunks_by_tokens.extend(token_splitter.split_text(section))
            else:
                chunks_by_tokens.append(section)

        return chunks_by_tokens

    def chunk(self) -> list["Chunk"]:
        data_models_list = []

        cleaned_content = self.content
        chunks = self.chunk_text(
            cleaned_content, chunk_size=self.metadata["chunk_size"], chunk_overlap=self.metadata["chunk_overlap"]
        )

        for chunk in chunks:
            model = Chunk(
                content=chunk,
                document_id=self.id,
                metadata=self.metadata,
                platform=self.platform,
                author_id=self.author_id,
                author_full_name=self.author_full_name,
            )
            data_models_list.append(model)

        return data_models_list

    def embed_batch(self, data_model: list["Chunk"]) -> list["EmbeddedChunk"]:
        embedding_model_input = [data_model.content for data_model in data_model]
        embeddings = self.embedding_model(embedding_model_input, to_list=True)

        embedded_chunk = [
            self.map_model(data_model, cast(list[float], embedding))
            for data_model, embedding in zip(data_model, embeddings, strict=False)
        ]

        return embedded_chunk

    @abstractmethod
    def map_model(self, data_model: "Chunk", embedding: list[float]) -> "EmbeddedChunk":
        ...

    def chunk_and_embed(self) -> list["EmbeddedChunk"]:
        chunks = self.chunk()
        return self.embed_batch(chunks)

class CleanedDocument(BaseDocument, VectorBaseDocument, ChunkingMixin):
    content: str

    def map_model(self, data_model: "Chunk", embedding: list[float]) -> "EmbeddedChunk":
        return EmbeddedChunk(
            id=data_model.id,
            content=data_model.content,
            embedding=embedding,
            platform=data_model.platform,
            document_id=data_model.document_id,
            author_id=data_model.author_id,
            author_full_name=data_model.author_full_name,
            metadata={
                "embedding_model_id": self.embedding_model.model_id,
                "embedding_size": self.embedding_model.embedding_size,
                "max_input_length": self.embedding_model.max_input_length,
            },
        )


class Chunk(BaseDocument, VectorBaseDocument):
    content: str
    document_id: UUID4 = Field(alias="document_id")
    metadata: dict = Field(default_factory=dict)


class EmbeddedChunk(BaseDocument, VectorBaseDocument):
    content: str
    document_id: UUID4 = Field(alias="document_id")
    embedding: list[float] | None
    metadata: dict = Field(default_factory=dict)

    @classmethod
    def to_context(cls, chunks: list["EmbeddedChunk"]) -> str:
        context = ""
        for i, chunk in enumerate(chunks):
            context += f"""
                Chunk {i + 1}:
                Type: {chunk.__class__.__name__}
                Platform: {chunk.platform}
                Author: {chunk.author_full_name}
                Content: {chunk.content}\n
                """

        return context
