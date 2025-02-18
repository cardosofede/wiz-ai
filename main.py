import asyncio

import logfire

logfire.configure()

from models.generic_article import GenericArticle
from crawlers.dispatcher import CrawlerDispatcher
from models.user import UserDocument


async def test_user_creation():
    user = await UserDocument.get_or_create(first_name="Fede", last_name="Cardoso")
    print(user.full_name)

async def test_scrape_custom_article():
    link = "https://hummingbot.org/installation/docker/"
    crawler = CrawlerDispatcher().get_crawler(link)
    doc: GenericArticle = await crawler.extract(link=link)
    cleaned_doc = doc.clean()
    chunks = cleaned_doc.chunk()
    embeddings = cleaned_doc.chunk_and_embed()
    a = 2


if __name__ == "__main__":
    # asyncio.run(test_user_creation())
    asyncio.run(test_scrape_custom_article())