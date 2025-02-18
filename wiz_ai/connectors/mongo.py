import logfire
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure


from wiz_ai.settings import settings


class MongoDatabaseConnector:
    _instance: AsyncIOMotorClient | None = None

    def __new__(cls, *args, **kwargs) -> AsyncIOMotorClient:
        if cls._instance is None:
            try:
                cls._instance = AsyncIOMotorClient(settings.DATABASE_HOST)
            except ConnectionFailure as e:
                logfire.error(f"Couldn't connect to the database: {e!s}")

                raise

        logfire.info(f"Connection to MongoDB with URI successful: {settings.DATABASE_HOST}")

        return cls._instance


connection = MongoDatabaseConnector()
