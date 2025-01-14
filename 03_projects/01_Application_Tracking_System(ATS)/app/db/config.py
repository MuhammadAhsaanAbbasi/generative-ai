from sqlmodel import SQLModel, create_engine, Field
from starlette.datastructures import Secret
import os

# DATABASe URL Connection String SEtup
DATABASE_URL = os.getenv("DATABASE_URL")
connection_string = str(DATABASE_URL).replace("postgresql", "postgresql+psycopg")
engine = create_engine(connection_string, connect_args={"sslmode": "require"}, pool_recycle=6000)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

class ChatSession(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    chat_messages: str
    vector_index: str


if __name__ == "__main__":
    create_db_and_tables()