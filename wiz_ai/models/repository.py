from models.base.documents import RawDocument


class RepositoryDocument(RawDocument):
    name: str
    link: str
