from models.base.nosql_base import NoSQLBaseDocument


class UserDocument(NoSQLBaseDocument):
    first_name: str
    last_name: str

    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}"
