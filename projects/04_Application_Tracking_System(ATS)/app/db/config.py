from sqlmodel import SQLModel, create_engine, Field, Session
from starlette.datastructures import Secret
from typing import Annotated
from fastapi import Depends
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


def get_session():    
    with Session(engine) as session:
        yield session

DB_SESSION = Annotated[Session, Depends(get_session)]

if __name__ == "__main__":
    create_db_and_tables()

