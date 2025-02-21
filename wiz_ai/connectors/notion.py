import logfire
from notion_client import AsyncClient
from notion_client.errors import APIResponseError

from wiz_ai.settings import settings


class NotionConnector:
    _instance: AsyncClient | None = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            try:
                cls._instance = AsyncClient(auth=settings.NOTION_TOKEN)
            except APIResponseError as e:
                logfire.error(f"Couldn't connect to Notion API: {e!s}")
                raise
        logfire.info("Connection to Notion API successful")
        return cls._instance

notion = NotionConnector()
