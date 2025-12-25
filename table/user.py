from model.Connected_DB import connected_DB
def create_table_user():
    conn = connected_DB()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(50) UNIQUE,
            password VARCHAR(100)
        )
    """)
    conn.commit()
    cursor.close()
    conn.close()