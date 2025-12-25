# import mysql.connector
# from dotenv import load_dotenv
# import os

# load_dotenv()


# def connected_DB():
#     return mysql.connector.connect(
#         host=os.getenv("MYSQL_HOST"),
#         user=os.getenv("MYSQL_USER"),
#         password=os.getenv("MYSQL_PASSWORD"),
#         database=os.getenv("MYSQL_DB"),
#         auth_plugin='mysql_native_password'
#     )
# model/Connected_DB.py
import mysql.connector

def connected_DB():
    """Kết nối MySQL"""
    return mysql.connector.connect(
        host="127.0.0.1",
        port=3306,
        user="root",
        password="123456",
        database="motordatabase"
    )
