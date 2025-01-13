from sqlmodel import SQLModel, create_engine
from starlette.datastructures import Secret
import os

# DATABASe URL Connection String SEtup
DATABASE_URL = os.getenv("DATABASE_URL")
connection_string = str(DATABASE_URL).replace("postgresql", "postgresql+psycopg")
engine = create_engine(connection_string, connect_args={"sslmode": "require"}, pool_recycle=6000)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)
