import asyncio

import logfire

from wiz_ai.models.nosql_base import NoSQLBaseDocument


class UserDocument(NoSQLBaseDocument):
    first_name: str
    last_name: str

    class Settings:
        name = "users"

    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}"

async def main():
    user = await UserDocument.get_or_create(first_name="Fede", last_name="Cardoso")
    print(user.full_name)

if __name__ == "__main__":
    asyncio.run(main())
