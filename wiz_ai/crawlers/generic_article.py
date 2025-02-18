from urllib.parse import urlparse
import logfire

from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers.html2text import Html2TextTransformer

from models.generic_article import GenericArticle
from models.user import UserDocument
from wiz_ai.crawlers.base import BaseCrawler


class GenericArticleCrawler(BaseCrawler):
    base_url = "https://"
    model = GenericArticle

    def __init__(self) -> None:
        super().__init__()

    async def extract(self, link: str, **kwargs) -> GenericArticle:
        old_model = await self.model.find(link=link)
        if old_model is not None:
            logfire.info(f"Article already exists in the database: {link}")
            return old_model

        logfire.info(f"Starting scrapping article: {link}")

        loader = AsyncHtmlLoader([link])
        docs = loader.load()

        html2text = Html2TextTransformer()
        docs_transformed = html2text.transform_documents(docs)
        doc_transformed = docs_transformed[0]

        content = {
            "Title": doc_transformed.metadata.get("title", ""),
            "Subtitle": doc_transformed.metadata.get("description", ""),
            "Content": doc_transformed.page_content,
            "language": doc_transformed.metadata.get("language", ""),
        }

        parsed_url = urlparse(link)
        platform = parsed_url.netloc

        user = kwargs.get("user", None)
        if user is None:
            logfire.warning("No user provided for the article, defaulting to anonymous.")
            user = await UserDocument.get_or_create(first_name="Anonymous", last_name="User")
        instance = self.model(
            content=content,
            link=link,
            platform=platform,
            author_id=user.id,
            author_full_name=user.full_name,
        )
        await instance.save()
        logfire.info(f"Finished scrapping custom article: {link}")
        return instance
