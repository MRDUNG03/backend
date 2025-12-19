import mysql.connector

def connected_DB():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="123456",
        database="motordatabase",
        auth_plugin='mysql_native_password'
    )
