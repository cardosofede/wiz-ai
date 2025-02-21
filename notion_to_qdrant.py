import asyncio
from typing import Optional, Set

import logfire

logfire.configure()

from wiz_ai.settings import settings
from wiz_ai.models.notion.knowledge_base import KnowledgeBaseDocument, DocumentCategory, NotionKnowledgeVectorDocument
from wiz_ai.networks import EmbeddingModelSingleton


embedding_model = EmbeddingModelSingleton()

async def process_knowledge_base_docs(categories: Optional[Set[DocumentCategory]] = None, include_deprecated: bool = False):
    """Process and upload knowledge base documents to vector store."""
    # Ensure collection exists before processing
    NotionKnowledgeVectorDocument.get_or_create_collection()
    
    filter_options = {}
    if categories:
        filter_options["categories"] = categories
    if not include_deprecated:
        filter_options["deprecated"] = False
        
    docs = await KnowledgeBaseDocument.bulk_find(database_id=settings.KNOWLEDGE_BASE_DB_ID, **filter_options)
    
    for doc in docs:
        try:
            # Skip if no Notion ID (shouldn't happen but better be safe)
            if not doc.notion_id:
                logfire.warning(f"Skipping document without Notion ID: {doc.title}")
                continue

            # Fetch content from Notion
            await doc.update_content()
            
            # Delete existing vectors for this document using Notion ID
            NotionKnowledgeVectorDocument.delete_by_notion_id(doc.notion_id)
            
            # Convert to vector documents (embeddings are created automatically)
            vector_docs = doc.to_vector_documents()
            
            # Upload new vectors to Qdrant
            NotionKnowledgeVectorDocument.bulk_insert(vector_docs)
            
            logfire.info(f"Successfully processed document: {doc.title}")
            
        except Exception as e:
            logfire.error(f"Error processing document {doc.title}: {e}")

async def main():
    # Process all non-deprecated installation documents
    print("\nProcessing Installation Documents:")
    await process_knowledge_base_docs(
        categories={DocumentCategory.INSTALLATION}
    )


if __name__ == "__main__":
    asyncio.run(main())