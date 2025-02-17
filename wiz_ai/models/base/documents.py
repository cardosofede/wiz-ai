import re
from abc import ABC

from langchain_text_splitters import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from pydantic import UUID4, Field, BaseModel

from models.base.nosql_base import NoSQLBaseDocument
from models.base.vector_base import VectorBaseDocument
from networks import EmbeddingModelSingleton

embedding_model = EmbeddingModelSingleton()


class BaseDocument(ABC, BaseModel):
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
            content=self.clean_text(" #### ".join(self.content.values())),
        )

class CleanedDocument(BaseDocument, VectorBaseDocument):
    content: str

    @property
    def metadata(self) -> dict:
        return {
            "chunk_size": 500,
            "chunk_overlap": 50,
        }

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
        character_splitter = RecursiveCharacterTextSplitter(separators=["\n\n"], chunk_size=chunk_size, chunk_overlap=0)
        text_split_by_characters = character_splitter.split_text(text)

        token_splitter = SentenceTransformersTokenTextSplitter(
            chunk_overlap=chunk_overlap,
            tokens_per_chunk=embedding_model.max_input_length,
            model_name=embedding_model.model_id,
        )
        chunks_by_tokens = []
        for section in text_split_by_characters:
            chunks_by_tokens.extend(token_splitter.split_text(section))

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

class Chunk(BaseDocument, VectorBaseDocument):
    content: str
    document_id: UUID4 = Field(alias="document_id")
    metadata: dict = Field(default_factory=dict)


class EmbeddedChunk(BaseDocument, VectorBaseDocument):
    content: str
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
