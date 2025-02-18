import re
from urllib.parse import urlparse

import logfire

from wiz_ai.crawlers.base import BaseCrawler
from wiz_ai.crawlers.generic_article import GenericArticleCrawler
from wiz_ai.crawlers.github import GithubCrawler


class CrawlerDispatcher:
    crawlers = [GithubCrawler]

    def __init__(self) -> None:
        self._crawlers = {}
        for crawler in self.crawlers:
            self.register(crawler.base_url, crawler)

    def register(self, domain: str, crawler: type[BaseCrawler]) -> None:
        parsed_domain = urlparse(domain)
        domain = parsed_domain.netloc

        self._crawlers[r"https://(www\.)?{}/*".format(re.escape(domain))] = crawler

    def get_crawler(self, url: str) -> BaseCrawler:
        for pattern, crawler in self._crawlers.items():
            if re.match(pattern, url):
                return crawler()
        else:
            logfire.warning(f"No crawler found for {url}. Defaulting to CustomArticleCrawler.")
            return GenericArticleCrawler()
