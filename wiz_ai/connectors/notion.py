import logfire
from notion_client import AsyncClient
from notion_client.errors import APIResponseError
from typing import List

from wiz_ai.settings import settings
from wiz_ai.models.notion.database import NotionDatabase


class NotionConnector:
    _instance: AsyncClient | None = None

    def __new__(cls, *args, **kwargs) -> AsyncClient:
        if cls._instance is None:
            try:
                cls._instance = AsyncClient(auth=settings.NOTION_TOKEN)
            except APIResponseError as e:
                logfire.error(f"Couldn't connect to Notion API: {e!s}")
                raise

        logfire.info("Connection to Notion API successful")
        return cls._instance

    @classmethod
    async def list_databases(cls) -> List[NotionDatabase]:
        """List all available Notion databases the integration has access to."""
        try:
            response = await cls._instance.search(
                filter={"property": "object", "value": "database"}
            )
            return [NotionDatabase.model_validate(db) for db in response["results"]]
        except Exception as e:
            logfire.error(f"Failed to list databases: {e}")
            return []

    @classmethod
    async def get_database(cls, database_id: str) -> NotionDatabase | None:
        """Get a specific database by ID."""
        try:
            response = await cls._instance.databases.retrieve(database_id=database_id)
            return NotionDatabase.model_validate(response)
        except Exception as e:
            logfire.error(f"Failed to get database {database_id}: {e}")
            return None


notion = NotionConnector()
