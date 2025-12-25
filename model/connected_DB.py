import mysql.connector
from dotenv import load_dotenv
import os

load_dotenv()


def connected_DB():
    return mysql.connector.connect(
        host=os.getenv("MYSQL_HOST"),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DB"),
        auth_plugin='mysql_native_password'
    )
